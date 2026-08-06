import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from Data.broker_data.broker_exceptions import (
    SymbolMasterUnavailableException,
)

SYMBOL_MASTER_URL = "https://public.fyers.in/sym_details/NSE_CM.csv"
SYMBOL_MASTER_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "cache",
    "symbol_master_nse_cm.csv",
)
CACHE_MAX_AGE = timedelta(hours=4)  # matches Fyers' own cache-control max-age

SYMBOL_MASTER_COLUMNS = [
    "fytoken",
    "name",
    "exchange_instrument_type",
    "lot_size",
    "tick_size",
    "isin",
    "trading_session",
    "last_update_date",
    "expiry_date",
    "symbol",
    "exchange",
    "segment",
    "scrip_code",
    "underlying_symbol",
    "underlying_scrip_code",
    "strike_price",
    "option_type",
    "underlying_fytoken",
    "reserved_1",
    "reserved_2",
    "reserved_3",
]

EQUITY_INSTRUMENT_TYPE_CODE = 0
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 2


def get_symbols() -> list[dict]:
    os.makedirs(os.path.dirname(SYMBOL_MASTER_CACHE_PATH), exist_ok=True)

    if _cache_is_fresh():
        df = pd.read_csv(
            SYMBOL_MASTER_CACHE_PATH, header=None, names=SYMBOL_MASTER_COLUMNS
        )
        return _filter_and_format(df)

    try:
        df = _fetch_symbol_master()
    except Exception as e:
        print(f"Failed to fetch fresh symbol master: {e}")
        if os.path.exists(SYMBOL_MASTER_CACHE_PATH):
            print("Falling back to stale cached symbol master.")
            df = pd.read_csv(
                SYMBOL_MASTER_CACHE_PATH, header=None, names=SYMBOL_MASTER_COLUMNS
            )
            return _filter_and_format(df)
        raise SymbolMasterUnavailableException(
            "Could not fetch symbol master and no cached copy exists."
        )

    df.to_csv(SYMBOL_MASTER_CACHE_PATH, header=False, index=False)
    return _filter_and_format(df)


def _cache_is_fresh() -> bool:
    if not os.path.exists(SYMBOL_MASTER_CACHE_PATH):
        return False
    modified_time = datetime.fromtimestamp(os.path.getmtime(SYMBOL_MASTER_CACHE_PATH))
    return datetime.now() - modified_time < CACHE_MAX_AGE


def _fetch_symbol_master() -> pd.DataFrame:
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(SYMBOL_MASTER_URL, timeout=30)
            response.raise_for_status()
            break
        except Exception as e:
            print(f"Symbol master fetch attempt {attempt} failed: {e}")
            time.sleep(RETRY_DELAY_SECONDS**attempt)
            continue
    else:
        raise ConnectionError(
            f"Failed to fetch symbol master after {MAX_RETRIES} attempts"
        )

    from io import StringIO

    df = pd.read_csv(StringIO(response.text), header=None, names=SYMBOL_MASTER_COLUMNS)
    return df


def _filter_and_format(df: pd.DataFrame) -> list[dict]:
    equities = df[df["exchange_instrument_type"] == EQUITY_INSTRUMENT_TYPE_CODE]

    result = []
    for _, row in equities.iterrows():
        result.append(
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "isin": row["isin"],
                "lot_size": row["lot_size"],
                "tick_size": row["tick_size"],
            }
        )
    return result


final_df = get_symbols()
