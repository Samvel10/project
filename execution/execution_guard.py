from execution.exceptions import RiskViolation


class ExecutionGuard:
    def __init__(self, risk_config):
        self.max_drawdown = risk_config["risk"]["max_drawdown"]
        self.max_daily_loss = risk_config["risk"]["max_daily_loss"]

    def check(self, portfolio):
        if portfolio.drawdown > self.max_drawdown:
            raise RiskViolation("Max drawdown exceeded")

        if portfolio.daily_loss > self.max_daily_loss:
            raise RiskViolation("Daily loss exceeded")

        return True
