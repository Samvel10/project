current_drawdown = 0

def allow_trade():
    return current_drawdown < 0.15
