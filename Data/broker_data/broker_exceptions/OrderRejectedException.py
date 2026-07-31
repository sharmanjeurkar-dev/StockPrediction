class OrderRejectedException(Exception):
    """Raised when the order transaction fails"""

    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(self.msg)
