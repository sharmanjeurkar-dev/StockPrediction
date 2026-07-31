"""
Broker-neutral vocabulary shared by BrockerInterface and every concrete
broker implementation (FyersBroker, MockBroker, future brokers). No
broker-specific codes belong in this file -- those live in per-broker
mapping tables (e.g. fyers_broker.py's _SIDE_MAP).
"""

from enum import Enum


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = (
        "STOP_LIMIT"  # confirm Fyers actually supports this before relying on it
    )


class ProductType(Enum):
    DELIVERY = "DELIVERY"  # CNC-equivalent -- your primary case given multi-day holds
    INTRADAY = "INTRADAY"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
