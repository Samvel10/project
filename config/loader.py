import yaml
from pathlib import Path


class Config:
    def __init__(self, base_path="config"):
        self.base_path = Path(base_path)

        self.trading = self._load("trading.yaml")
        self.risk = self._load("risk.yaml")
        self.execution = self._load("execution.yaml")
        self.ml = self._load("ml.yaml")
        self.backtest = self._load("backtest.yaml")
        self.symbols = self._load("symbols.yaml")

    def _load(self, filename):
        path = self.base_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing config file: {filename}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def summary(self):
        return {
            "capital": self.trading["capital"]["initial"],
            "risk_per_trade": self.risk["risk"]["per_trade"],
            "model": self.ml["model"]["type"],
            "universe": self.symbols["universe"]
        }
