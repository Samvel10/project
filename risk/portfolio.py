from .drawdown import DrawdownTracker
from .exposure import ExposureController


class Portfolio:
    """
    Portfolio class to manage multiple positions, capital, exposure, and drawdown.
    """

    def __init__(self, capital: float, max_exposure: float = 1.0):
        self.initial_capital = capital
        self.capital = capital
        self.positions = {}  # {symbol: {"qty": float, "entry": float, "value": float}}
        self.exposure_controller = ExposureController(max_exposure)
        self.drawdown_tracker = DrawdownTracker()

    def add_position(self, symbol: str, qty: float, entry_price: float):
        """
        Ավելացնում է նոր դիրք
        """
        position_value = qty * entry_price
        if not self.exposure_controller.can_open_position(position_value, self.capital):
            raise ValueError(f"Cannot open position for {symbol}: exposure limit exceeded")

        self.positions[symbol] = {
            "qty": qty,
            "entry": entry_price,
            "value": position_value
        }
        self.exposure_controller.add_position(position_value, self.capital)

    def close_position(self, symbol: str, exit_price: float):
        """
        Փակում է դիրքը և հաշվարկում եկամուտը/կորուստը
        """
        if symbol not in self.positions:
            raise KeyError(f"No position for {symbol} to close")

        pos = self.positions.pop(symbol)
        pnl = (exit_price - pos["entry"]) * pos["qty"]
        self.capital += pnl
        self.exposure_controller.remove_position(pos["value"], self.capital)
        return pnl

    def update_drawdown(self):
        """
        Հաշվարկում է ներկա drawdown-ը
        """
        return self.drawdown_tracker.update(self.capital)

    def get_max_drawdown(self):
        return self.drawdown_tracker.get_max_drawdown()

    def get_total_exposure(self):
        return round(self.exposure_controller.current_exposure * 100, 2)

    def get_position(self, symbol: str):
        return self.positions.get(symbol, None)

    def get_portfolio_summary(self):
        summary = {
            "capital": round(self.capital, 2),
            "max_drawdown": self.get_max_drawdown(),
            "total_exposure_percent": self.get_total_exposure(),
            "positions": {k: v for k, v in self.positions.items()}
        }
        return summary
