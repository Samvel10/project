class ExposureController:
    """
    Kontroli e anum total exposure-ը շուկայում
    """

    def __init__(self, max_exposure: float):
        """
        max_exposure օրինակ՝ 1.0 = 100% capital
        """
        self.max_exposure = max_exposure
        self.current_exposure = 0.0

    def can_open_position(self, position_value: float, capital: float) -> bool:
        exposure_after = self.current_exposure + (position_value / capital)
        return exposure_after <= self.max_exposure

    def add_position(self, position_value: float, capital: float):
        self.current_exposure += position_value / capital

    def remove_position(self, position_value: float, capital: float):
        self.current_exposure -= position_value / capital
        self.current_exposure = max(0, self.current_exposure)
