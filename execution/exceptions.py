class ExecutionError(Exception):
    pass


class ExchangeUnavailable(ExecutionError):
    pass


class InvalidOrder(ExecutionError):
    pass


class RiskViolation(ExecutionError):
    pass
