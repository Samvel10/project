import numpy as np


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def expectancy(returns):
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(losses) == 0:
        return np.mean(wins)

    return (
        (len(wins) / len(returns)) * np.mean(wins)
        - (len(losses) / len(returns)) * abs(np.mean(losses))
    )


def sharpe(returns, rf=0):
    if returns.std() == 0:
        return 0
    return (returns.mean() - rf) / returns.std()
