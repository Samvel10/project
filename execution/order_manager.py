from execution.position_sizer import calculate_position_size
from execution.sl_tp import compute_sl_tp
from execution.execution_guard import ExecutionGuard


class OrderManager:
    def __init__(self, client, config, portfolio):
        self.client = client
        self.config = config
        self.guard = ExecutionGuard(config.risk)
        self.portfolio = portfolio

    def execute(self, symbol, signal, entry_price, atr):
        self.guard.check(self.portfolio)

        direction = "BUY" if signal == "LONG" else "SELL"
        stop, tps = compute_sl_tp(entry_price, atr, signal)

        qty = calculate_position_size(
            capital=self.portfolio.equity,
            risk_per_trade=self.config.risk["risk"]["per_trade"],
            entry_price=entry_price,
            stop_price=stop,
            leverage=self.config.trading["leverage"]["default"],
        )

        if qty <= 0:
            return None

        order = self.client.place_order(
            symbol=symbol,
            side=direction,
            quantity=qty,
        )

        return {
            "order": order,
            "stop": stop,
            "tps": tps,
        }
