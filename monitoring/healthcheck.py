def system_health(portfolio, data_ok=True, model_loaded=True):
    if portfolio.equity <= 0:
        return False, "Equity depleted"

    if not data_ok:
        return False, "Market data unavailable"

    if not model_loaded:
        return False, "ML model not loaded"

    return True, "OK"
