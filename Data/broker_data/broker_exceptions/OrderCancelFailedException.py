class OrderCancelFailedException(Exception):
    """Raised when order cancelation throws an Exception."""

    def __init__(self, code, msg: str):
        self.code = code
        super().__init__(
            f"Exception for order with order id {code} cancelaltion returned msg: {msg}"
        )
