class ConfidenceScaler:
    """
    Adjust confidence based on market volatility or other metrics.
    """

    def __init__(self, max_confidence=0.99, min_confidence=0.1):
        self.max_conf = max_confidence
        self.min_conf = min_confidence

    def scale(self, raw_conf, volatility=0.0):
        """
        Scale confidence inversely with volatility
        """
        factor = max(0.0, 1.0 - volatility)
        conf = raw_conf * factor
        return min(max(conf, self.min_conf), self.max_conf)
