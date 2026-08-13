import math
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from fyers_apiv3.fyersModel import FyersModel

from Data.broker_data.Brocker_interface import BrockerInterface
from Data.broker_data.broker_exceptions.OrderCancelFailedException import (
    OrderCancelFailedException,
)
from Data.broker_data.broker_exceptions.OrderNotFoundException import (
    OrderNotFoundException,
)
from Data.broker_data.broker_exceptions.OrderRejectedException import (
    OrderRejectedException,
)
from Data.broker_data.broker_exceptions.UnknownOrderException import (
    UnknownOrderStatusException,
)
from Data.broker_data.enums import OrderStatus, OrderType, ProductType, Side

_SIDE_MAP = {
    Side.BUY: 1,
    Side.SELL: -1,
}

_ORDER_TYPE_MAP = {
    OrderType.LIMIT: 1,
    OrderType.MARKET: 2,
    OrderType.STOP_LOSS: 3,
    OrderType.STOP_LIMIT: 4,
}

_PRODUCT_TYPE_MAP = {
    ProductType.DELIVERY: "CNC",
    ProductType.INTRADAY: "INTRADAY",
}

_ORDER_STATUS_MAP = {
    1: OrderStatus.CANCELLED,
    2: OrderStatus.FILLED,
    4: OrderStatus.PENDING,  # "Transit" -- sent to exchange, awaiting confirmation
    5: OrderStatus.REJECTED,
    6: OrderStatus.PENDING,
    7: OrderStatus.CANCELLED,  # "Expired" -- treat as a form of cancelled; reconsider if you want a distinct EXPIRED member
    # 3 ("For Future Use") intentionally omitted -- not a real status, should never appear
}

_SIDE_MAP_REVERSE = {v: k for k, v in _SIDE_MAP.items()}
_ORDER_STATUS_MAP_REVERSE = {v: k for k, v in _ORDER_STATUS_MAP.items()}
_PRODUCT_TYPE_MAP_REVERSE = {v: k for k, v in _PRODUCT_TYPE_MAP.items()}
_ORDER_TYPE_MAP_REVERSE = {v: k for k, v in _ORDER_TYPE_MAP.items()}


class FyersBrocker(BrockerInterface):
    def __init__(
        self,
    ):
        super().__init__()

        self.client_id = self.CLIENT_ID
        self.secret_id = self.SECRET_ID
        self.redirect_uri = self.REDIRECT_URI

        self.token = self.get_token(self.client_id, self.secret_id)
        self.validate_token()

        self.fyers = FyersModel(
            client_id=self.client_id, token=self.token, is_async=False, log_path=""
        )

    def is_market_open(self):
        try:
            response = self.fyers.market_status()
        except Exception as e:
            print(f"Error occurred while retrieving market status: {e}")
            raise

        if not response or response.get("s") != "ok":
            print("Error interpreting the status message for the market")
            raise ValueError("Market status unavailable")

        market_data = response["marketStatus"][0]["status"]
        return market_data == "OPEN"

    # validating token and loggin in
    def get_token(self, client_id, secret_id) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        token_path = os.path.join(base_dir, "access_token.env")

        if not os.path.exists(token_path):
            raise FileNotFoundError(
                f"❌ Authentication failed: Could not find the file at {token_path}"
            )
        load_dotenv(dotenv_path=token_path)
        token = os.getenv("FYERS_ACCESS_TOKEN")
        if not token:
            raise ValueError(
                "❌ Authentication failed: 'FYERS_ACCESS_TOKEN' was not found inside the .env file. "
                "Did you run the generation script today?"
            )

        return token

    # Sanity check for login
    def validate_token(self):
        if not self.token:
            raise ValueError(
                "FyersBroker cannot be constructed without a valid access token.login again to get the valid access_tocken"
            )

    # scrape historical data
    def scrape_data(
        self, symbol, resolution, DAYS, end_date: datetime | None, total_chunks=1
    ) -> pd.DataFrame | None:
        all_data = []
        failed_chunks = []
        MAX_RETRIES = 2
        RETRY_DELAY_SECOND = 2

        if end_date is None:
            end_date = datetime.now()
        for i in range(total_chunks):
            # Calculate the chunk's start and end dates
            chunk_end = end_date - timedelta(days=(i * DAYS))
            chunk_start = chunk_end - timedelta(days=DAYS)

            print(
                f"Fetching chunk for symbol: {symbol} {i + 1}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
            )

            data = {
                "symbol": symbol,
                "resolution": str(resolution),  # 1D timeframe
                "date_format": "1",
                "range_from": chunk_start.strftime("%Y-%m-%d"),
                "range_to": chunk_end.strftime("%Y-%m-%d"),
                "cont_flag": "1",
            }

            # Call the FYERS API
            for j in range(MAX_RETRIES):
                try:
                    response = self.fyers.history(data=data)
                except Exception as e:
                    print(f"Failed fetching of chunk due to error: {e}")
                    continue

                if "candles" in response and response["candles"]:
                    # Create a dataframe for this chunk
                    df_chunk = pd.DataFrame(
                        response["candles"],
                        columns=["Datetime", "Open", "High", "Low", "Close", "Volume"],
                    )
                    all_data.append(df_chunk)
                    time.sleep(1)
                    break
                else:
                    time.sleep(RETRY_DELAY_SECOND**j)
                    print(
                        f"Couldn't fetch all the candles at attempt {j} trying again after {RETRY_DELAY_SECOND**j} "
                    )
                    continue

            # Failed attempts
            else:
                chunk = (chunk_start, chunk_end)
                failed_chunks.append(chunk)
                print(
                    f"Chunk {chunk_start} to {chunk_end} failed after {MAX_RETRIES} attempts"
                )

        failure = len(failed_chunks)
        successful_chunks = total_chunks - failure
        completeness_ratio = successful_chunks / total_chunks

        if completeness_ratio < 0.80:
            print("Failed to load greater than 80% of ideal range")
            return None
        else:
            final_df = pd.concat(all_data, ignore_index=True)
            print(f"✅ Success! Downloaded {len(final_df)} rows of data.")
            # Combine all the chunks together
            final_df["symbol"] = symbol
            # Convert Unix timestamps to human-readable datetime
            final_df["Datetime"] = pd.to_datetime(final_df["Datetime"], unit="s")

            # check and drop the duplicates
            final_df = final_df.drop_duplicates(subset="Datetime")

            # Sort from oldest to newest
            final_df = final_df.sort_values(by=["symbol", "Datetime"])

            # Set Datetime as index
            final_df.set_index("Datetime", inplace=True)
            final_df.attrs["missing_ranges"] = [
                (str(start), str(end)) for start, end in failed_chunks
            ]
            return final_df

    # live data pooling
    def get_live_data(self, symbols):
        live_symbol_name = []
        live_lp = []
        live_open = []
        live_high = []
        live_low = []
        live_close = []
        live_volume = []
        live_timestamp = []

        MAX_RETRIES = 5
        RETRY_DELAY_SECOND = 2
        CHUNK_SIZE = 50

        failed_chunks = []  # (chunk_index, symbols_in_chunk) that failed entirely
        failed_symbols = []  # individual symbols missing from an otherwise-successful chunk

        total_chunks = math.ceil(len(symbols) / CHUNK_SIZE)

        for i in range(total_chunks):
            symbols_chunk = symbols[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
            data = {"symbols": symbols_chunk}

            for attempt in range(MAX_RETRIES):
                try:
                    response = self.fyers.quotes(data)
                except Exception as e:
                    print(f"Chunk {i} attempt {attempt} raised an error: {e}")
                    time.sleep(RETRY_DELAY_SECOND)
                    continue

                if response and response.get("d"):
                    # one API call already returned the whole chunk -- just parse it
                    returned_symbols = {entry["n"] for entry in response["d"]}
                    missing_in_chunk = set(symbols_chunk) - returned_symbols
                    if missing_in_chunk:
                        failed_symbols.extend(missing_in_chunk)

                    for entry in response["d"]:
                        v = entry["v"]
                        live_symbol_name.append(entry["n"])
                        live_lp.append(v["lp"])
                        live_open.append(v["open_price"])
                        live_close.append(v["prev_close_price"])
                        live_high.append(v["high_price"])
                        live_low.append(v["low_price"])
                        live_volume.append(v["volume"])
                        live_timestamp.append(v["tt"])
                    break
                else:
                    print(
                        f"Chunk {i} attempt {attempt} failed, retrying in {RETRY_DELAY_SECOND}s"
                    )
                    time.sleep(RETRY_DELAY_SECOND)
                    continue
            else:
                # only runs if every retry in this chunk failed without ever breaking
                failed_chunks.append((i, symbols_chunk))
                print(f"Chunk {i} failed completely after {MAX_RETRIES} attempts")

        live_data = {
            "Time_stamp": live_timestamp,
            "symbols": live_symbol_name,
            "last_traded_price": live_lp,
            "high": live_high,
            "low": live_low,
            "open": live_open,
            "close": live_close,
            "volume": live_volume,
        }

        if live_symbol_name:
            result = pd.DataFrame(live_data)
            result.attrs["failed_chunks"] = failed_chunks
            result.attrs["failed_symbols"] = failed_symbols
            return result
        else:
            return None

    def placeOrder(
        self, symbol, quantity, side, order_type, limitPrice, stopPrice, product_type
    ) -> str:

        fyers_side = _SIDE_MAP.get(side)
        if fyers_side is None:
            raise ValueError(f"Side {side} is not supported by FyersBroker")
        fyers_order_type = _ORDER_TYPE_MAP.get(order_type)
        if fyers_order_type is None:
            raise ValueError(f"Order type {order_type} is not supported by FyersBroker")
        fyers_product_type = _PRODUCT_TYPE_MAP.get(product_type)
        if fyers_product_type is None:
            raise ValueError(
                f"Product type {product_type} is not supported by FyersBroker"
            )

        data = {
            "symbol": symbol,
            "qty": quantity,
            "type": fyers_order_type,
            "side": fyers_side,
            "productType": fyers_product_type,
            "limitPrice": limitPrice,  # O is for no limit price and any vfalue givin will act as the limit
            "stopPrice": stopPrice,  # O is for no stop price and any vfalue givin will act as the stop
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "orderTag": "tag1",
            "stopLoss": 0,
            "takeProfit": 0,
            "isSliceOrder": False,
        }
        try:
            response = self.fyers.place_order(data=data)
        except Exception as e:
            print(f"place_order call to Fyers raised an exception: {e}")
            raise

        if response and response.get("s") == "ok":
            return response["id"]

        error_message = (
            response.get("message", "Unknown rejection reason")
            if response
            else "Empty response from Fyers"
        )
        raise OrderRejectedException(msg=error_message)

    def cancel_order(self, order_id) -> OrderStatus:
        if not order_id:
            raise ValueError("order_id is required")

        data = {"id": order_id}

        try:
            response = self.fyers.cancel_order(data=data)
        except Exception as e:
            print(f"Cancel order call to Fyers raised an exception: {e}")
            raise

        if not response or response.get("s") != "ok":
            error_message = (
                response.get("message", "Unknown cancellation failure")
                if response
                else "Empty response from Fyers"
            )
            raise OrderCancelFailedException(code=order_id, msg=error_message)

        # Fyers confirmed the cancel request went through -- verify the
        # actual resulting state via get_order_status rather than assuming
        return self.get_order_status(order_id)

    def get_order_status(self, order_id: str) -> OrderStatus:
        if not order_id:
            raise ValueError("order_id is required")

        data = {"id": order_id}

        try:
            response = self.fyers.orderbook(data=data)
        except Exception as e:
            print(f"get_order_status call to Fyers raised an exception: {e}")
            raise

        if not response or response.get("s") != "ok":
            raise OrderNotFoundException(order_id)

        orders = response.get("orderBook", [])
        if not orders:
            raise OrderNotFoundException(order_id)

        order = orders[0]
        fyers_status_code = order.get("status")

        # detect partial fill before trusting a pure status-code lookup
        filled_qty = order.get("filledQty", 0)
        total_qty = order.get("qty", 0)
        if fyers_status_code == 2 and 0 < filled_qty < total_qty:
            return OrderStatus.PARTIALLY_FILLED

        mapped_status = _ORDER_STATUS_MAP.get(fyers_status_code)
        if mapped_status is None:
            raise UnknownOrderStatusException(fyers_status_code)

        return mapped_status

    def get_positions(self) -> list[dict]:
        try:
            response = self.fyers.positions()
        except Exception as e:
            print(f"Error in fetching current positions: {e}")
            raise

        if not response or response.get("s") != "ok":
            raise ValueError("Positions data unavailable")

        raw_positions = response.get("netPositions", [])

        position_data = []
        for p in raw_positions:
            fyers_side = p.get("side")
            mapped_side = _SIDE_MAP_REVERSE.get(fyers_side)
            if mapped_side is None:
                raise ValueError(
                    f"Unrecognized side code in position data: {fyers_side}"
                )

            position_data.append(
                {
                    "symbol": p.get("symbol"),
                    "quantity": p.get("netQty"),
                    "average_price": p.get("netAvg"),
                    "current_price": p.get("ltp"),
                    "side": mapped_side,
                    "product_type": p.get("productType"),
                    "unrealized_pnl": p.get("unrealized_profit"),
                    "realized_pnl": p.get("realized_profit"),
                    "total_pnl": p.get("pl"),
                }
            )

        return position_data

    def get_holdings(self) -> list[dict]:
        try:
            response = self.fyers.holdings()
        except Exception as e:
            print(f"Error in fetching current Holdings: {e}")
            raise

        if not response or response.get("s") != "ok":
            raise ValueError("Positions data unavailable")

        raw_holdings = response.get("holdings", [])

        holdings_data = []
        for h in raw_holdings:
            holdings_data.append(
                {
                    "symbol": h.get("symbol"),
                    "quantity": h.get("quantity"),
                    "average_price": h.get("costPrice"),
                    "current_price": h.get("ltp"),
                    "current_value": h.get("marketVal"),
                    "unrealized_pnl": h.get("pl"),
                }
            )

        return holdings_data

    def get_funds(self) -> float:
        try:
            totalFunds = self.fyers.funds()
        except Exception as e:
            print(f"Error in return ing available Funds: {e}")
            raise

        if not totalFunds and totalFunds.get("s") != "ok":
            raise ValueError("Funds Data unavailable")

        fund_rows = totalFunds.get("fund_limit", [])

        available_funds: float | None = None
        for row in fund_rows:
            if row.get("title") == "Available Balance":
                available_funds = row.get("equityAmount")
                break
        if available_funds is None:
            raise ValueError("Avaialble Funds could not be extracted")

        return available_funds

    def get_unrealized_pnl(
        self,
    ) -> list[dict]:
        pnl_data = []

        holdings = self.get_holdings()
        for h in holdings:
            pnl_data.append(
                {
                    "symbol": h["symbol"],
                    "unrealized_pnl": h["unrealized_pnl"],  # from holdings' 'pl' field
                    "source": "holdings",
                }
            )

        positions = self.get_positions()
        for p in positions:
            pnl_data.append(
                {
                    "symbol": p["symbol"],
                    "unrealized_pnl": p["unrealized_pnl"],
                    "source": "positions",
                }
            )

        return pnl_data

    def get_realized_pnl(self) -> list[dict]:
        realized_pnl = []

        posiions = self.get_positions()

        for p in posiions:
            realized_pnl.append(
                {
                    "symbol": p["symbol"],
                    "realized_pnl": p["realized_pnl"],
                }
            )

        return realized_pnl

    def get_order_book(self) -> list[dict]:
        try:
            response = self.fyers.orderbook()
        except Exception as e:
            print(f"Failed to retrive orderbook: {e}")
            raise

        if not response or response.get("s") != "ok":
            raise ValueError("Failed to retrive orderbook")

        orders = response.get("orderBook", [])
        order_book = []

        for o in orders:
            order_id = o.get("id")
            mapping_warnings = []

            fyers_side = o.get("side")
            mapped_side = _SIDE_MAP_REVERSE.get(fyers_side)
            if mapped_side is None:
                mapping_warnings.append(f"unrecognized side code: {fyers_side}")

            fyers_status = o.get("status")
            mapped_status = _ORDER_STATUS_MAP_REVERSE.get(fyers_status)
            if mapped_status is None:
                mapping_warnings.append(f"unrecognized order status: {fyers_status}")

            fyers_product_type = o.get("productType")
            mapped_product = _PRODUCT_TYPE_MAP_REVERSE.get(fyers_product_type)
            if mapped_product is None:
                mapping_warnings.append(
                    f"unrecognized product type: {fyers_product_type}"
                )

            fyers_type = o.get("type")
            mapped_type = _ORDER_TYPE_MAP_REVERSE.get(fyers_type)
            if mapped_type is None:
                mapping_warnings.append(f"unrecognized order type: {fyers_type}")

            if mapping_warnings:
                print(f"Order {order_id} has mapping issues: {mapping_warnings}")

            order_book.append(
                {
                    "id": order_id,
                    "symbol": o.get("symbol"),
                    "side": mapped_side,
                    "type": mapped_type,
                    "status": mapped_status,
                    "qty": o.get("qty"),
                    "filledQty": o.get("filledQty"),
                    "limitPrice": o.get("limitPrice"),
                    "stopPrice": o.get("stopPrice"),
                    "tradedPrice": o.get("tradedPrice"),
                    "orderDateTime": o.get("orderDateTime"),
                    "productType": mapped_product,
                    "mapping_warnings": mapping_warnings,
                }
            )

        return order_book

    def get_trade_book(self) -> list[dict]:
        try:
            response = self.fyers.tradebook()
        except Exception as e:
            print(f"Failed to retrive tradebook: {e}")
            raise

        if not response or response.get("s") != "ok":
            raise ValueError("Failed to retrive tradebook")

        tradebooks = response.get("tradeBook", [])
        tradebook = []

        for t in tradebooks:
            order_id = t.get("id")
            mapping_warnings = []

            fyers_side = t.get("side")
            mapped_side = _SIDE_MAP_REVERSE.get(fyers_side)
            if mapped_side is None:
                mapping_warnings.append(f"unrecognized side code: {fyers_side}")

            fyers_status = t.get("status")
            mapped_status = _ORDER_STATUS_MAP_REVERSE.get(fyers_status)
            if mapped_status is None:
                mapping_warnings.append(f"unrecognized order status: {fyers_status}")

            fyers_product_type = t.get("productType")
            mapped_product = _PRODUCT_TYPE_MAP_REVERSE.get(fyers_product_type)
            if mapped_product is None:
                mapping_warnings.append(
                    f"unrecognized product type: {fyers_product_type}"
                )

            fyers_type = t.get("type")
            mapped_type = _ORDER_TYPE_MAP_REVERSE.get(fyers_type)
            if mapped_type is None:
                mapping_warnings.append(f"unrecognized order type: {fyers_type}")

            if mapping_warnings:
                print(f"Order {order_id} has mapping issues: {mapping_warnings}")

            tradebook.append(
                {
                    "id": order_id,
                    "symbol": t.get("symbol"),
                    "side": mapped_side,
                    "type": mapped_type,
                    "status": mapped_status,
                    "qty": t.get("qty"),
                    "filledQty": t.get("filledQty"),
                    "limitPrice": t.get("limitPrice"),
                    "stopPrice": t.get("stopPrice"),
                    "tradedPrice": t.get("tradedPrice"),
                    "orderDateTime": t.get("orderDateTime"),
                    "productType": mapped_product,
                    "mapping_warnings": mapping_warnings,
                }
            )

        return tradebook

    def get_recent_history(
        self,
        symbol: str,
        as_of_date,
        lookback_days: int = 90,
    ) -> pd.DataFrame | None:
        if as_of_date is not None:
            end_date = datetime.strptime(as_of_date, "%Y-%m-%d")

        try:
            data = self.scrape_data(
                symbol=symbol, resolution="1D", DAYS=lookback_days, end_date=end_date
            )
            return data
        except Exception as e:
            print(
                f"Failed to load the data for the given symbol and time period with error:\n {e}"
            )
            return None
