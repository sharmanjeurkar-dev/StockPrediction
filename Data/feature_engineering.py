import numpy as np
import pandas as pd

from Data.historical_data import historical_data_scraper


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

    # On-Balance Volume (OBV)
    condition = [(df["Change"] > 0), (df["Change"] < 0)]
    choices = [1, -1]
    df["obv-dir"] = np.select(condition, choices, default=0)
    df["OBV"] = (df["Volume"] * df["obv-dir"]).cumsum()
    df["OBV-ROC"] = df["OBV"].pct_change(10).replace([np.inf, -np.inf], 0).fillna(0)
    df["Volume-Rate-of-Change"] = (df["Volume"].pct_change(periods=10)).replace(
        [np.inf, -np.inf], 0
    ).fillna(0) * 100

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

    df["MA-20"] = df["Close"].rolling(window=20).mean()
    df["MA-50"] = df["Close"].rolling(window=50).mean()
    df["MA-200"] = df["Close"].rolling(window=200).mean()
    df["MA-Cross"] = (df["MA-20"] - df["MA-50"]) / df["MA-50"]

    # VWAP
    rolling_vwap = (df["Close"] * df["Volume"]).rolling(window=20).sum() / df[
        "Volume"
    ].rolling(window=20).sum()
    df["VWAP-20D-Dist"] = (df["Close"] - rolling_vwap) / rolling_vwap

    df.drop(columns=["OBV"], inplace=True)

    df.dropna(inplace=True)
    return df


# relative strength with se: nifty-50 index
WINDOW = 20


def _returns_ND(window: int, price_close: str, df: pd.DataFrame) -> pd.Series:
    """Trailing N-day return: today's price vs. price `window` days ago."""
    return (df[price_close] / df[price_close].shift(window)) - 1


def add_relative_strength(
    df: pd.DataFrame, relative_index_df: pd.DataFrame, window: int = WINDOW
) -> pd.DataFrame:
    """
    Adds a relative_strength column to df (a single stock's feature
    dataframe, indexed by Datetime). relative_index_df is the benchmark's
    (Nifty 50) dataframe, loaded ONCE by the caller and passed in here --
    this function must never fetch the benchmark itself.
    """
    relative_index_df.set_index("Datetime", inplace=True)
    stock_returns_ND = _returns_ND(window=window, price_close="Close", df=df)
    benchmark_close = relative_index_df["Close"]
    df["nifty-50-close"] = df.index.map(benchmark_close)

    benchmark_returns_ND = _returns_ND(
        window=window, price_close="nifty-50-close", df=df
    )

    df["relative_strength"] = stock_returns_ND - benchmark_returns_ND
    # +ve: stock outperformed the market over the window
    # -ve: stock underperformed the market over the window

    df.drop(columns=["nifty-50-close"], inplace=True)
    df.dropna(inplace=True)

    return df


def concat_df(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    processed = []

    for df in dfs:
        symbol = df["symbol"].iloc[0]
        df_processed = feature_enginiering(df, symbol)
        df_processed["symbol"] = symbol
        processed.append(df_processed)

    final_df = pd.concat(processed, ignore_index=False)
    return final_df


def walk_forward_slices(
    df: pd.DataFrame,
    date_col: str = "Datetime",
    symbol_col: str = "symbol",
    embargo_candles: int = 8,
) -> list:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    years = sorted(df[date_col].dt.year.unique())
    append_df = []

    for i in range(1, len(years)):
        train_years = years[:i]
        train_df = df[df[date_col].dt.year.isin(train_years)].copy()

        if embargo_candles > 0:
            train_df = train_df.sort_values([symbol_col, date_col])
            row_rank_desc = train_df.groupby(symbol_col).cumcount(ascending=False)
            train_df = train_df[row_rank_desc >= embargo_candles]

        test_year = years[i]
        test_df = df[df[date_col].dt.year == test_year].copy()
        append_df.append((train_df, test_df))

    return append_df
