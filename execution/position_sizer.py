def calculate_position_size(
    capital,
    risk_per_trade,
    entry_price,
    stop_price,
    leverage,
):
    risk_amount = capital * risk_per_trade
    stop_distance = abs(entry_price - stop_price)

    if stop_distance <= 0:
        return 0

    qty = (risk_amount / stop_distance) * leverage
    return round(qty, 3)
