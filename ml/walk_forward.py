def walk_forward_split(
    candles,
    train_window=2000,
    test_window=300,
    step=300,
):
    splits = []

    start = train_window
    while start + test_window < len(candles):
        train = candles[start - train_window : start]
        test = candles[start : start + test_window]

        splits.append((train, test))
        start += step

    return splits
