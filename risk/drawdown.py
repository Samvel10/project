class DrawdownTracker:
    """
    Հետևում է max drawdown-ին
    """

    def __init__(self):
        self.peak_equity = 0.0
        self.max_drawdown = 0.0

    def update(self, equity: float) -> float:
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)

        return drawdown

    def get_max_drawdown(self) -> float:
        return round(self.max_drawdown * 100, 2)
