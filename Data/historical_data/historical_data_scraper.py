import os
from datetime import datetime

import pandas as pd

from Data.broker_data import download_symbols
from Data.broker_data.Fyersbrocker import FyersBrocker
from Data.liquidity_screener import stocks_screener

broker = FyersBrocker()
HISTORICAL_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _safe_filename(symbol: str) -> str:
    return symbol.replace(":", "_") + ".parquet"


def save_histortrical_data(
    resolution: str, DAYS: int, total_chunks: int, end_date: datetime | None = None
):
    failed_symbols = []
    os.makedirs(HISTORICAL_DATA_PATH, exist_ok=True)
    today = datetime.strftime(datetime.now(), "%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now()

    symbol_records = download_symbols.get_symbols()
    if not symbol_records:
        print("No symbols found")
        return
    symbol_list = [s["symbol"] for s in symbol_records]

    snapshot, _ = stocks_screener.screener(
        symbols=symbol_list,
        as_of_date=today,
        lookback_days=90,
        min_avg_daily_value=20000000,
    )

    valid_symbols = snapshot["symbols"]

    for vs in valid_symbols:
        last_date = check_existing_data(vs)
        if last_date is not None:
            if last_date.date() == end_date.date():
                print(f"{vs} already up to date")
                continue
            else:
                filepath = os.path.join(HISTORICAL_DATA_PATH, _safe_filename(vs))
                existing_df = pd.read_parquet(filepath)
                success = increment_data(vs, last_date, end_date, existing_df)
                if not success:
                    failed_symbols.append(vs)
        else:
            success = bootstarp_symbol_history(
                vs,
                resolution=resolution,
                DAYS=DAYS,
                total_chunks=total_chunks,
                end_date=end_date,
            )
            if not success:
                failed_symbols.append(vs)

    if failed_symbols:
        print(f"{len(failed_symbols)} symbols failed: {failed_symbols}")
    return failed_symbols


def check_existing_data(symbol: str):
    filepath = os.path.join(HISTORICAL_DATA_PATH, _safe_filename(symbol))

    if not os.path.exists(filepath):
        return None

    df = pd.read_parquet(filepath, columns=[])
    if df.empty:
        return None

    most_recent_date = df.index.max()
    return most_recent_date


def bootstarp_symbol_history(
    symbol: str,
    resolution: str,
    DAYS: int,
    total_chunks: int,
    end_date: datetime | None = None,
):
    os.makedirs(HISTORICAL_DATA_PATH, exist_ok=True)
    filepath = os.path.join(HISTORICAL_DATA_PATH, _safe_filename(symbol))

    if end_date is None:
        end_date = datetime.now()

    try:
        df = broker.scrape_data(
            symbol=symbol,
            resolution=resolution,
            end_date=end_date,
            DAYS=DAYS,
            total_chunks=total_chunks,
        )
    except Exception as e:
        print(f"Bootstrap failed for {symbol}: {e}")
        return False

    if df is None:
        print(f"Bootstrap returned no data for {symbol}")
        return False

    df.to_parquet(filepath, index=True)
    print(f"Bootstrapped {symbol}: {len(df)} rows saved to {filepath}")
    return True


def increment_data(symbol: str, last_date, end_date: datetime, df: pd.DataFrame):
    os.makedirs(HISTORICAL_DATA_PATH, exist_ok=True)
    filepath = os.path.join(HISTORICAL_DATA_PATH, _safe_filename(symbol))

    diff = (end_date - last_date).days
    new_df = broker.get_recent_history(
        symbol=symbol,
        as_of_date=end_date.strftime("%Y-%m-%d"),
        lookback_days=diff,
    )

    if new_df is not None:
        final_df = pd.concat((df, new_df), axis=0)
        final_df = final_df[~final_df.index.duplicated(keep="last")]
        final_df = final_df.sort_index()
        final_df.to_parquet(filepath)
        return True
    else:
        return False


BENCHMARK_SYMBOL = "NSE:NIFTY50-INDEX"


def save_benchmark_data(
    resolution: str, DAYS: int, total_chunks: int, end_date: datetime = None
):
    """
    Fetches/updates the market benchmark (Nifty 50) using the exact same
    bootstrap-or-increment logic as every regular stock, so it stays in
    sync with the rest of the historical data store. Called separately
    from save_historical_data since the benchmark isn't part of the
    screened universe and shouldn't go through liquidity screening.
    """
    if end_date is None:
        end_date = datetime.now()

    last_date = check_existing_data(BENCHMARK_SYMBOL)

    if last_date is not None:
        if last_date.date() == end_date.date():
            print(f"{BENCHMARK_SYMBOL} already up to date")
            return True
        filepath = os.path.join(HISTORICAL_DATA_PATH, _safe_filename(BENCHMARK_SYMBOL))
        existing_df = pd.read_parquet(filepath)
        return increment_data(BENCHMARK_SYMBOL, last_date, end_date, existing_df)
    else:
        return bootstarp_symbol_history(
            BENCHMARK_SYMBOL,
            resolution=resolution,
            DAYS=DAYS,
            total_chunks=total_chunks,
            end_date=end_date,
        )


if __name__ == "__main__":
    result = save_benchmark_data(resolution="1D", DAYS=365, total_chunks=15)
    print(result)
    df = pd.read_parquet("Data/historical_data/data/NSE_NIFTY50-INDEX.parquet")
    print(df.attrs.get("missing_ranges"))
