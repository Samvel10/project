import time


class Telemetry:
    def __init__(self):
        self.start_time = time.time()
        self.trades = 0
        self.errors = 0

    def record_trade(self):
        self.trades += 1

    def record_error(self):
        self.errors += 1

    def uptime(self):
        return int(time.time() - self.start_time)

    def snapshot(self):
        return {
            "uptime_sec": self.uptime(),
            "trades": self.trades,
            "errors": self.errors,
        }
