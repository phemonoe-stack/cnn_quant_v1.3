"""
quant_cnn_v2.py  —  Classroom Edition (Balanced)
==================================================
Zero heavy dependencies. Runs on Google Colab with ONE install line.

In Colab, run this first:
    !pip install -q yfinance

Everything else (numpy, pandas, sklearn, tensorflow, matplotlib)
is pre-installed in Colab and most university Python environments.

All technical indicators are computed from scratch using only
numpy + pandas — no pandas-ta, no ta-lib, no extra packages.

Fixes applied vs original:
  FIX 1 — Wider label thresholds (±1.5%) to surface real Sell signals
  FIX 2 — Class weights passed to model.fit() so Sell misses are penalised
  FIX 3 — Oversampling of minority classes before training
"""

# ── Only ONE extra install needed ──────────────────────────────
# !pip install -q yfinance
# ───────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight          # FIX 2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# ─────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────
def load_market_data(ticker="SPY", period="10y", interval="1d"):
    print(f"Fetching {ticker} data from Yahoo Finance...")
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df.dropna(inplace=True)
    # yfinance ≥0.2 returns MultiIndex columns — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"  Loaded {len(df)} rows.")
    return df


# ─────────────────────────────────────────────
# 2. INDICATORS — pure numpy/pandas, no extra packages
# ─────────────────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def bollinger_width(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (2 * std) / sma          # normalized width

def atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_features(df):
    df = df.copy()

    # Returns
    df["Return"]     = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Momentum
    df["RSI"]  = rsi(df["Close"])
    df["MACD"], df["MACD_Signal"] = macd(df["Close"])

    # Trend
    df["EMA_20"]    = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"]    = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_ratio"] = df["EMA_20"] / df["EMA_50"]

    # Volatility
    df["BB_width"] = bollinger_width(df["Close"])
    df["ATR"]      = atr(df["High"], df["Low"], df["Close"])

    # Volume
    df["Volume_MA"]    = df["Volume"].rolling(20).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_MA"]

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3. LABELS  (Buy=2 / Hold=1 / Sell=0)
#    FIX 1: thresholds raised from ±1% → ±1.5%
#    This shrinks the Hold majority and forces the model
#    to see enough Sell examples to learn the downward loop.
# ─────────────────────────────────────────────
def generate_labels(df, forward_days=5,
                    buy_thresh=0.015, sell_thresh=-0.015):   # FIX 1
    future_return = df["Close"].shift(-forward_days) / df["Close"] - 1
    labels = np.where(future_return >  buy_thresh,  2,
             np.where(future_return <  sell_thresh, 0, 1))
    return labels


# ─────────────────────────────────────────────
# 4. SLIDING WINDOW  → (samples, time, features)
# ─────────────────────────────────────────────
def create_windows(features, labels, window_size=30):
    X, y = [], []
    for i in range(window_size, len(features) - 1):
        X.append(features[i - window_size:i])
        y.append(labels[i])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 4b. CLASS WEIGHTS  (FIX 2)
#     Tells the loss function: a missed Sell costs more
#     than a missed Hold.  Scales automatically to your data.
# ─────────────────────────────────────────────
def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced",
                                   classes=classes,
                                   y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"  Class weights: {class_weight_dict}")
    return class_weight_dict


# ─────────────────────────────────────────────
# 4c. OVERSAMPLE MINORITY CLASSES  (FIX 3)
#     Duplicate Sell (and Buy if needed) samples until
#     every class has the same count as the largest class.
#     ONLY call this on training data — never on test data.
# ─────────────────────────────────────────────
def balance_training_data(X_train, y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()

    X_parts, y_parts = [], []
    for cls in classes:
        idx = np.where(y_train == cls)[0]
        idx_resampled = np.random.choice(idx, size=max_count, replace=True)
        X_parts.append(X_train[idx_resampled])
        y_parts.append(y_train[idx_resampled])

    X_balanced = np.concatenate(X_parts)
    y_balanced = np.concatenate(y_parts)

    # Shuffle so classes aren't batched together
    shuffle_idx = np.random.permutation(len(X_balanced))
    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]


# ─────────────────────────────────────────────
# 5. 1D CNN MODEL
# ─────────────────────────────────────────────
def build_cnn(input_shape, num_classes=3):
    model = Sequential([
        Conv1D(64,  kernel_size=3, activation="relu",
               padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),

        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),

        Conv1D(64,  kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64,  activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ─────────────────────────────────────────────
# 6. PLOTS — matplotlib only, no seaborn
# ─────────────────────────────────────────────
def plot_results(history, y_true, y_pred,
                 class_names=["Sell", "Hold", "Buy"]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Quant CNN Results (Balanced)", fontsize=14, fontweight="bold")

    # Accuracy curve
    axes[0].plot(history.history["accuracy"],     label="Train", color="#7B68EE")
    axes[0].plot(history.history["val_accuracy"],  label="Val",   color="#FF6347")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss curve
    axes[1].plot(history.history["loss"],     label="Train", color="#7B68EE")
    axes[1].plot(history.history["val_loss"],  label="Val",   color="#FF6347")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend()

    # Confusion matrix — pure matplotlib, no seaborn
    cm = confusion_matrix(y_true, y_pred)
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names)
    axes[2].set_yticks(range(len(class_names)))
    axes[2].set_yticklabels(class_names)
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(j, i, str(cm[i, j]),
                         ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig("quant_cnn_results.png", dpi=150)
    plt.show()
    print("Plot saved: quant_cnn_results.png")


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── Data ──────────────────────────────────
    df_raw = load_market_data(ticker="SPY", period="10y")
    df     = add_features(df_raw)

    # FIX 1: wider thresholds expose Sell signals
    labels_raw = generate_labels(df, forward_days=5,
                                  buy_thresh=0.015, sell_thresh=-0.015)

    FEATURE_COLS = [
        "Return", "Log_Return", "RSI", "MACD", "MACD_Signal",
        "EMA_ratio", "BB_width", "ATR", "Volume_ratio"
    ]
    feature_matrix = df[FEATURE_COLS].values

    scaler         = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    WINDOW = 30
    X, y   = create_windows(feature_matrix, labels_raw, window_size=WINDOW)

    print(f"\nRaw dataset:  X={X.shape}  y={y.shape}")
    print(f"Classes:  Sell={np.sum(y==0)}  Hold={np.sum(y==1)}  Buy={np.sum(y==2)}")

    # ── Temporal split — NEVER shuffle time-series data ───────
    split           = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # FIX 3: oversample minority classes in training set only
    X_train, y_train = balance_training_data(X_train, y_train)
    print(f"\nAfter balancing:")
    print(f"  Sell={np.sum(y_train==0)}  Hold={np.sum(y_train==1)}  Buy={np.sum(y_train==2)}")

    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat  = to_categorical(y_test,  num_classes=3)

    # FIX 2: class weights (extra penalty for ignoring Sell)
    print("\nComputing class weights...")
    class_weights = get_class_weights(y_train)

    # ── Build & train ──────────────────────────────────────────
    model = build_cnn(input_shape=(WINDOW, X.shape[2]))
    model.summary()

    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weights,          # FIX 2 — key line
        callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                  restore_best_weights=True)],
        verbose=1
    )

    # ── Evaluate ───────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")
    print("NOTE: overall accuracy will be lower than the unbalanced model —")
    print("      that is expected and means the model is actually learning Sell.")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Sell", "Hold", "Buy"]))

    plot_results(history, y_test, y_pred)

    model.save("quant_cnn_model.keras")
    print("Model saved: quant_cnn_model.keras")
