class UnknownOrderStatusException(Exception):
    """Raised when Fyers returns a status code not in _ORDER_STATUS_MAP."""

    def __init__(self, code):
        self.code = code
        super().__init__(f"Unrecognized Fyers order status code: {code}")
