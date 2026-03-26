from .position_sizer import PositionSizer
from .drawdown import DrawdownTracker
from .exposure import ExposureController


class RiskManager:
    """
    Central risk manager
    """

    def __init__(
        self,
        capital: float,
        risk_per_trade: float,
        max_drawdown: float,
        max_exposure: float
    ):
        self.capital = capital
        self.max_drawdown_limit = max_drawdown

        self.position_sizer = PositionSizer(capital, risk_per_trade)
        self.drawdown_tracker = DrawdownTracker()
        self.exposure_controller = ExposureController(max_exposure)

    def check_drawdown(self, equity: float) -> bool:
        """
        Վերադարձնում է False, եթե drawdown-ը գերազանցել է limit-ը
        """
        dd = self.drawdown_tracker.update(equity)
        return dd <= self.max_drawdown_limit

    def calculate_position(
        self,
        entry: float,
        stop: float
    ) -> float:
        return self.position_sizer.calculate_position_size(entry, stop)

    def can_open_trade(
        self,
        position_value: float,
        capital: float
    ) -> bool:
        return self.exposure_controller.can_open_position(position_value, capital)
