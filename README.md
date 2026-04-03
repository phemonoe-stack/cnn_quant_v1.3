# CNN Quant Demo v1.3
A Space Monkey & Claude Production w/ Gemini

<img width="915" height="357" alt="modelerrors" src="https://github.com/user-attachments/assets/e502c13b-fa1a-4fe0-bdea-e0bb5eb39833" />

That's a really insightful diagnosis from Gemini! The perma-bull problem is a classic **class imbalance issue** in financial ML. In most market data, "Hold" days vastly outnumber "Buy" and "Sell" days, so the model learns to just predict the majority class and still gets decent accuracy — but it's essentially useless for real trading.

Let me look at your code and fix this properly.Here's the core problem and the fix.

The issue is **three-fold**:

1. **Class imbalance** — "Hold" days dominate the dataset (often 60-70%), so the model ignores rare "Sell" signals entirely
2. **Asymmetric thresholds** — `buy_thresh=0.01` and `sell_thresh=-0.01` over 5 days is a tiny bar, meaning many days technically qualify as Buy/Hold but few as Sell
3. **No class weighting** — the loss function treats all three classes equally, so the model is never penalized for missing Sell signals

Here's what the fix looks like conceptually before I show you the code:Here's the updated code with all three fixes applied — drop-in replacements for the relevant functions:

```python
# ── FIX 1: Wider thresholds so Sell signals actually appear ──────────────────
def generate_labels(df, forward_days=5, buy_thresh=0.015, sell_thresh=-0.015):
    """
    Raised thresholds from ±1% to ±1.5%.
    This reduces the Hold class and forces more meaningful Buy/Sell signals.
    """
    future_return = df["Close"].shift(-forward_days) / df["Close"] - 1
    labels = np.where(future_return >  buy_thresh,  2,
             np.where(future_return <  sell_thresh, 0, 1))
    return labels


# ── FIX 2: Compute class weights to penalize ignoring rare classes ────────────
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(y_train):
    """
    Tells the model: 'A missed Sell is 3x worse than a missed Hold.'
    Automatically scales to whatever imbalance exists in YOUR data.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced",
                                   classes=classes,
                                   y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict


# ── FIX 3 (optional but powerful): Oversample the minority class ──────────────
def balance_training_data(X_train, y_train):
    """
    For each class, randomly duplicate samples until all classes are equal.
    This is 'oversampling' — no new data invented, just rebalanced representation.
    Only apply to TRAINING data, never to test data.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()

    X_parts, y_parts = [], []
    for cls in classes:
        idx = np.where(y_train == cls)[0]
        # Randomly repeat samples from underrepresented classes
        idx_resampled = np.random.choice(idx, size=max_count, replace=True)
        X_parts.append(X_train[idx_resampled])
        y_parts.append(y_train[idx_resampled])

    X_balanced = np.concatenate(X_parts)
    y_balanced = np.concatenate(y_parts)

    # Shuffle so classes aren't in blocks
    shuffle_idx = np.random.permutation(len(X_balanced))
    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]
```

And the updated `__main__` block where you wire it all together:

```python
if __name__ == "__main__":

    df_raw = load_market_data(ticker="SPY", period="10y")
    df     = add_features(df_raw)

    # FIX 1: wider thresholds
    labels_raw = generate_labels(df, forward_days=5,
                                  buy_thresh=0.015, sell_thresh=-0.015)

    FEATURE_COLS = [
        "Return", "Log_Return", "RSI", "MACD", "MACD_Signal",
        "EMA_ratio", "BB_width", "ATR", "Volume_ratio"
    ]
    feature_matrix = df[FEATURE_COLS].values

    scaler         = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    WINDOW   = 30
    X, y     = create_windows(feature_matrix, labels_raw, window_size=WINDOW)

    print(f"\nDataset:  X={X.shape}  y={y.shape}")
    print(f"Classes:  Sell={np.sum(y==0)}  Hold={np.sum(y==1)}  Buy={np.sum(y==2)}")

    split               = int(len(X) * 0.8)
    X_train, X_test     = X[:split], X[split:]
    y_train, y_test     = y[:split], y[split:]

    # FIX 3: balance training data via oversampling
    X_train, y_train = balance_training_data(X_train, y_train)
    print(f"\nAfter balancing — Classes: Sell={np.sum(y_train==0)}  Hold={np.sum(y_train==1)}  Buy={np.sum(y_train==2)}")

    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat  = to_categorical(y_test,  num_classes=3)

    # FIX 2: compute class weights
    class_weights = get_class_weights(y_train)

    model = build_cnn(input_shape=(WINDOW, X.shape[2]))
    model.summary()

    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weights,   # ← THE KEY LINE
        callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                  restore_best_weights=True)],
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Sell", "Hold", "Buy"]))

    plot_results(history, y_test, y_pred)
```

**What to watch for after the fix:** Your overall accuracy number will likely *drop* (from maybe 55% to 45%), but don't panic — that's the model actually working. Before it was just guessing "Hold" all day and getting cheap accuracy. Now check the classification report: you want `Sell` and `Buy` to both have non-zero precision/recall. A balanced 40% across three classes beats a fake 60% that never sells.

One caveat worth naming: even a fixed, balanced model here is a research/learning tool. SPY on daily bars is a nearly efficient market, so don't expect the model to be reliably profitable — but it will at least be *honest* about uncertainty now.

Got it! Here's the complete, unified `quant_cnn_v3.py` with all three fixes merged in — one clean file you can drop straight into Colab:All three fixes are woven into the single file. The only structural changes from the original:

- `compute_class_weight` is imported at the top with the other sklearn imports
- `get_class_weights()` and `balance_training_data()` are new functions in section 4b/4c, slotted between windowing and the CNN so the flow reads naturally top to bottom
- `generate_labels()` defaults are now `±0.015` instead of `±0.01`
- The `__main__` block calls them in the right order — oversample first, then compute weights from the balanced set, then pass `class_weight` into `model.fit()`

- The comment in the evaluate section is worth paying attention to when you run it: if you see accuracy drop from ~55% to ~42%, that's the puzzle clicking into place, not breaking.


------------------------------------------------------------------------------
Test Accuracy: 0.3793  |  Test Loss: 2.3274
NOTE: overall accuracy will be lower than the unbalanced model —
      that is expected and means the model is actually learning Sell.
16/16 ━━━━━━━━━━━━━━━━━━━━ 1s 22ms/step

Classification Report:
              precision    recall  f1-score   support

        Sell       0.22      0.46      0.30        80
        Hold       0.63      0.42      0.50       300
         Buy       0.19      0.20      0.19       113

    accuracy                           0.38       493
   macro avg       0.34      0.36      0.33       493
weighted avg       0.46      0.38      0.40       493
<img width="1766" height="495" alt="download" src="https://github.com/user-attachments/assets/52ad3b22-7406-4c5d-8985-5d880e6f38e0" />

