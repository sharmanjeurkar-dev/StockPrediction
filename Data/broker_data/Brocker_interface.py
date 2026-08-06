import os
from abc import ABC, abstractmethod

import pandas as pd
from dotenv import load_dotenv

from Data.broker_data.enums import OrderStatus

base_dir = os.path.dirname(os.path.abspath(__file__))
token_path = os.path.join(base_dir, "access_token.env")
load_dotenv(dotenv_path=token_path)


class BrockerInterface(ABC):
    def __init__(
        self,
    ):
        self.CLIENT_ID = os.getenv("CLIENT_ID")
        self.SECRET_ID = os.getenv("SECRET_ID")
        self.REDIRECT_URI = os.getenv("REDIRECT_URI")

    @abstractmethod
    def validate_token(
        self,
    ):  # should return authtoken
        pass

    @abstractmethod
    def get_token(self, client_id, secret_id) -> str:  # should return authtoken
        pass

    @abstractmethod
    def scrape_data(
        self,
        symbol,
        resolution,
        DAYS,
    ) -> pd.DataFrame | None:  # be default resolution set to daily. DAYS set max value
        pass

    @abstractmethod
    def get_live_data(self, symbols) -> pd.DataFrame | None:
        pass

    @abstractmethod
    def placeOrder(
        self, symbol, quantity, side, order_type, limitPrice, stopPrice, product_type
    ) -> str:
        """
        symbol (str) — which instrument
        quantity (int) — how many shares/lots
        side — buy or sell (use an enum, not a raw string like "BUY"/"buy", to avoid silent typos causing wrong-direction trades)
        order_type — market, limit, stop-loss (again, enum)
        price (optional, only required for limit/SL orders — think about whether this should default to None for market orders)
        product_type — intraday vs delivery/carry-forward (this matters a lot given your daily-bar multi-day holding period — you almost certainly want delivery/CNC-equivalent,
        not intraday, and getting this wrong means your broker auto-squares-off your position same day without telling your bot)
        Return order id for get_order_status()
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id) -> OrderStatus:
        """
        order_id (str) — that's essentially it, maybe plus symbol if the broker's API requires it for lookup
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id):
        """
        order_id (str)
        Return: a status enum (pending/filled/partially-filled/rejected/cancelled) — standardized across brokers, not whatever raw string Fyers returns
        """
        pass

    @abstractmethod
    def get_positions(
        self,
    ) -> list[dict]:
        """
        returns all current open positions as a standard structure (symbol, quantity, avg price, current P&L(realized and unrealized),sell postion or buy position)
        """

    @abstractmethod
    def get_funds(
        self,
    ) -> float:
        """
        returns available capital as a float/standard structure
        """

    @abstractmethod
    def is_market_open(
        self,
    ) -> int | None:
        """
        returns int if market
        close 0
        open 1
        postclose_open 2
        postclose_closed 3
        preopen 4
        preopen_closed 5
        error 6
        """
        pass

    @abstractmethod
    def get_unrealized_pnl(
        self,
    ) -> list[dict]:
        """
        returns a list of dictiories where symbol, date, unrealized pnl is given
        """
        pass

    @abstractmethod
    def get_realized_pnl(
        self,
    ) -> list[dict]:
        """
        returns a list of dictiories where symbol, realized pnl is given
        """
        pass

    @abstractmethod
    def get_order_book(
        self,
    ) -> list[dict]:
        """
        Order Book: This shows all pending, canceled, modified, or completed requests.
        It provides an active list of what you want to trade.
        returns order_book as a standard structure (symbol, quantity, avg price, status)
        """
        pass

    @abstractmethod
    def get_trade_book(
        self,
    ) -> list[dict]:
        """
        Trade_book:This acts as your transaction history. It logs only the orders that have been successfully executed or filled.
        The trade book records the exact price, quantity, and timestamp for your completed deals.
        returns order_book as a standard structure (symbol, quantity, avg price, time_stamp)
        """
        pass

    @abstractmethod
    def get_holdings(
        self,
    ) -> list[dict]:
        """
        similar functionality like that of get_positon and get_tradebook
        """
        pass
