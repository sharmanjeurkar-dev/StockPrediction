import os
import traceback

import numpy as np
import pandas as pd

from Data.historical_data.historical_data_scraper import (
    BENCHMARK_SYMBOL,
    HISTORICAL_DATA_PATH,
    _safe_filename,
)
from Data.label import apply_triple_barrier_labels
from Data.liquidity_screener.snapshot_store import load_snapshot

benchmark_filename = _safe_filename(BENCHMARK_SYMBOL)
BENCHMARK_DATA_PATH = os.path.join(HISTORICAL_DATA_PATH, benchmark_filename)


def feature_enginiering(df: pd.DataFrame, symbol) -> pd.DataFrame:
    df["symbol"] = symbol
    df.sort_index(inplace=True)
    df = df.loc[~df.index.duplicated(keep="first")].copy()

    # Calculating the %returns on close price if the share was brought
    df["Returns"] = df["Close"].pct_change()

    # RSI score
    df["Change"] = df["Close"].diff()
    df["Gain"] = np.where(df["Change"] > 0, df["Change"], 0)
    df["Loss"] = np.where(df["Change"] < 0, abs(df["Change"]), 0)

    df["Avg_Gain"] = df["Gain"].ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["Avg_Loss"] = df["Loss"].ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    df["Rs"] = df["Avg_Gain"] / df["Avg_Loss"]
    df["RSI-close-score"] = 100 - 100 / (1 + df["Rs"])

    df["Intraday_Spread"] = (df["High"] - df["Low"]) / df["Close"]

    # ATR Compressoin Ratio
    df["high_low"] = df["High"] - df["Low"]
    df["high_prev_close"] = (df["High"] - df["Close"].shift(1)).abs()
    df["low_prev_close"] = (df["Low"] - df["Close"].shift(1)).abs()

    df["true_range"] = df[["high_low", "high_prev_close", "low_prev_close"]].max(axis=1)
    df["atr-s"] = df["true_range"].rolling(window=14).mean()
    df["atr-l"] = df["true_range"].rolling(window=50).mean()

    df["ATR-Ratio"] = df["atr-s"] / df["atr-l"]
    df.drop(columns=["high_low", "high_prev_close", "low_prev_close"], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df["Daily_Vol"] = df["Close"].pct_change().rolling(window=25).std()

    # VWAP
    rolling_vwap = (df["Close"] * df["Volume"]).rolling(window=20).sum() / df[
        "Volume"
    ].rolling(window=20).sum()
    df["VWAP-20D-Dist"] = (df["Close"] - rolling_vwap) / rolling_vwap

    # Gap open
    df["Overnight_Gap"] = df["Open"] - df["Close"].shift(1)

    # Gap Fill ratio
    df["Fill"] = df["Close"] - df["Open"]
    df["Gap_Fill_Ratio"] = df["Fill"] / df["Overnight_Gap"]
    df["Gap_Fill_Ratio"] = df["Gap_Fill_Ratio"].replace([np.inf, -np.inf], np.nan)

    # High low disatnce through out rolling year
    df["Dist_52W_High"] = df["Close"] - df["Close"].rolling(window=252).max()
    df["Dist_52W_Low"] = df["Close"] - df["Close"].rolling(window=252).min()

    # Intraday spread zscore
    df["Intraday_Spread_MA20"] = df["Intraday_Spread"].rolling(window=20).mean()
    df["Intraday_Spread_STD20"] = df["Intraday_Spread"].rolling(window=20).std()
    df["Intraday_Spread_Zscore"] = (
        df["Intraday_Spread"] - df["Intraday_Spread_MA20"]
    ) / df["Intraday_Spread_STD20"]
    df["Intraday_Spread_Zscore"] = df["Intraday_Spread_Zscore"].replace(
        [np.inf, -np.inf], np.nan
    )
    df["Amihud_Illiquidity"] = df["Returns"] / (df["Close"] * df["Volume"])
    df.dropna(inplace=True)
    return df


def add_cross_sectional_vol_rank(
    df: pd.DataFrame, vol_col: str = "Daily_Vol", date_col: str = "Datetime"
) -> pd.DataFrame:
    df["Vol_Percentile_Rank"] = df.groupby(date_col)[vol_col].rank(pct=True)
    return df


def add_rolling_beta(
    df: pd.DataFrame,
    relative_index_df: pd.DataFrame,
    window: int = 60,
    return_col: str = "Returns",
) -> pd.DataFrame:
    relative_index_df["Returns"] = relative_index_df["Close"].pct_change()
    benchmark_returns = relative_index_df[return_col]
    df["nifty-50-returns"] = df.index.map(benchmark_returns)

    rolling_cov = df[return_col].rolling(window).cov(df["nifty-50-returns"])
    rolling_var = df["nifty-50-returns"].rolling(window).var()

    df["Beta_60D"] = rolling_cov / rolling_var
    df.drop(columns=["nifty-50-returns"], inplace=True)
    return df


# relative strength with se: nifty-50 index
WINDOW = [60, 120]


def _returns_ND(window: int, price_close: str, df: pd.DataFrame) -> pd.Series:
    """Trailing N-day return: today's price vs. price `window` days ago."""
    return (df[price_close] / df[price_close].shift(window)) - 1


def add_relative_strength(
    df: pd.DataFrame, relative_index_df: pd.DataFrame, window: list[int] = WINDOW
) -> pd.DataFrame:
    """
    Adds a relative_strength column to df (a single stock's feature
    dataframe, indexed by Datetime). relative_index_df is the benchmark's
    (Nifty 50) dataframe, loaded ONCE by the caller and passed in here --
    this function must never fetch the benchmark itself.
    """
    for w in window:
        stock_returns_ND = _returns_ND(window=w, price_close="Close", df=df)
        benchmark_close = relative_index_df["Close"]
        df["nifty-50-close"] = df.index.map(benchmark_close)

        benchmark_returns_ND = _returns_ND(
            window=w, price_close="nifty-50-close", df=df
        )

        df[f"relative_strength_{w}d"] = stock_returns_ND - benchmark_returns_ND
    # +ve: stock outperformed the market over the window
    # -ve: stock underperformed the market over the window

    df.drop(columns=["nifty-50-close"], inplace=True)
    df.dropna(inplace=True)

    return df


def build_training_set(
    snapshot_date: str,
    labeling_kwargs: dict | None = None,
) -> pd.DataFrame:
    universe_df, resolved_date = load_snapshot(snapshot_date)
    symbols = universe_df["symbol"].tolist()

    benchmark_df = pd.read_parquet(BENCHMARK_DATA_PATH)
    processed_dataframes = []
    failed_symbols = []
    failure_reason = []

    for symbol in symbols:
        # Name changing accrding to file name format
        file_name = _safe_filename(symbol)
        file = os.path.join(HISTORICAL_DATA_PATH, file_name)

        if not os.path.isfile(file):
            failed_symbols.append(symbol)
            failure_reason.append(
                f"File doesn't exist for symbol:{symbol} at location: {file}"
            )
            continue

        df = pd.read_parquet(file)
        if df.empty:
            failed_symbols.append(symbol)
            failure_reason.append(f"Dataframe for symbol: {symbol} found but is empty")
            continue
        try:
            df = feature_enginiering(df, symbol)
            df = add_relative_strength(df, benchmark_df, window=WINDOW)
            df = add_rolling_beta(df, benchmark_df, window=60)
            df = apply_triple_barrier_labels(df)
            df["symbol"] = symbol
            processed_dataframes.append(df)

        except Exception as e:
            failed_symbols.append(symbol)
            failure_reason.append("Failure in preprocessing of data")
            print(f"Exception occured \n {traceback.extract_tb(e.__traceback__)}")

    if not processed_dataframes:
        raise ValueError("No symbols were successfully processed")

    final_df = pd.concat(processed_dataframes, axis=0)
    final_df = add_cross_sectional_vol_rank(final_df)

    if failed_symbols:
        print(f"{len(failed_symbols)} symbols failed: {failed_symbols}")

    failure_df = pd.DataFrame(
        {"Failed symbols": failed_symbols, "Failure Reason": failure_reason}
    )
    return final_df, failure_df


def walk_forward_out_of_sample_dataframe_slices(
    df: pd.DataFrame, startDate=None, endDate=None, jump: int = 3, max_days: int = 8
) -> list[list[pd.DataFrame]]:
    if startDate is None:
        startDate = df.index.min()
    if endDate is None:
        endDate = df.index.max()

    total_months = (endDate.year - startDate.year) * 12 + (
        endDate.month - startDate.month
    )

    dfcollection = []
    for i in range(0, total_months, jump):
        nextSetEndDate = startDate + pd.DateOffset(months=jump + i)
        nextSetEndDateWithoutEmbargo = nextSetEndDate - pd.DateOffset(days=max_days)
        testNextSetEndDate = nextSetEndDate + pd.DateOffset(months=jump)

        train_mask = (df.index >= startDate) & (
            df.index <= nextSetEndDateWithoutEmbargo
        )
        test_mask = (df.index >= nextSetEndDate) & (df.index <= testNextSetEndDate)

        if testNextSetEndDate <= endDate:
            dfcollection.append([df[train_mask], df[test_mask]])
        else:
            test_mask_final = df.index >= nextSetEndDate
            dfcollection.append([df[train_mask], df[test_mask_final]])
            print("All sets before the end date covered")
            break

    return dfcollection
