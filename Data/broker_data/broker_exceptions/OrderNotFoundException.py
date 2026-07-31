class OrderNotFoundException(Exception):
    """Raised when Fyers has no record of the given order_id at all."""

    def __init__(self, order_id: str):
        self.order_id = order_id
        super().__init__(f"No order found with id: {order_id}")
