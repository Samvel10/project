class PositionSizer:
    """
    Hashvum e position size-y` hashvi arnelov
    capital, risk %, entry, stop-loss
    """

    def __init__(self, capital: float, risk_per_trade: float):
        self.capital = capital
        self.risk_per_trade = risk_per_trade  # օրինակ՝ 0.01 = 1%

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Position size = (capital * risk%) / |entry - stop|
        """

        risk_amount = self.capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss_price)

        if stop_distance == 0:
            raise ValueError("Stop-loss-ը չի կարող հավասար լինել entry-ին")

        position_size = risk_amount / stop_distance
        return round(position_size, 6)
