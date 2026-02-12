def mse(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
