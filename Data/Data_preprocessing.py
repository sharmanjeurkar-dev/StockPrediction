import numpy as np
import pandas as pd


def feature_enginiering(df: pd.DataFrame):

    # Calculating the %returns on close price if the share was brought
    df["Returns"] = df["Close"].pct_change()

    # z -score
    df["Z-score-close"] = (df["Close"] - df["Close"].rolling(window=20).mean()) / df[
        "Close"
    ].rolling(window=20).std()

    # RSI score
    df["Change"] = df["Close"].diff()
    df["Gain"] = np.where(df["Change"] > 0, df["Change"], 0)
    df["Loss"] = np.where(df["Change"] < 0, abs(df["Change"]), 0)

    df["Avg_Gain"] = df["Gain"].rolling(window=14).mean()
    df["Avg_Loss"] = df["Loss"].rolling(window=14).mean()

    df["Rs"] = df["Avg_Gain"] / df["Avg_Loss"]
    df["RSI-close-score"] = 100 - 100 / (1 + df["Rs"])

    # EMA Sscores for different windows
    df["EMA-Today-close-26D"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["EMA-Today-close-12D"] = df["Close"].ewm(span=12, adjust=False).mean()

    # MACD Scores
    df["MACD-Line"] = df["EMA-Today-close-12D"] - df["EMA-Today-close-26D"]
    df["Single-Line"] = df["MACD-Line"].ewm(span=9, adjust=False).mean()
    df["MACD-Histogram"] = df["MACD-Line"] - df["Single-Line"]

    # Bollinger Bandwidth
    df["Middle-Band"] = df["Close"].rolling(window=20).mean()
    df["Upper-Band"] = df["Middle-Band"] + 2 * df["Close"].rolling(window=20).std()
    df["Lower-Band"] = df["Middle-Band"] - 2 * df["Close"].rolling(window=20).std()

    df["Bollinger-Bandwidth"] = (df["Upper-Band"] - df["Lower-Band"]) / df[
        "Middle-Band"
    ]
    df["%-Band"] = (df["Close"] - df["Lower-Band"]) / (
        df["Upper-Band"] - df["Lower-Band"]
    )

    # On-Balance Volume (OBV)
    condition = [(df["Change"] > 0), (df["Change"] < 0)]
    choices = [1, -1]
    df["obv-dir"] = np.select(condition, choices, default=0)
    df["OBV"] = (df["Volume"] * df["obv-dir"]).cumsum()
    df["OBV-ROC"] = df["OBV"].pct_change(10)
    # Volume Rate of Change (VROC)
    df["Volume-Rate-of-Change"] = (df["Volume"].pct_change(periods=10)) * 100

    # ATR Compressoin Ratio
    df["high_low"] = df["High"] - df["Low"]
    df["high_prev_close"] = (df["High"] - df["Close"].shift(1)).abs()
    df["low_prev_close"] = (df["Low"] - df["Close"].shift(1)).abs()

    df["true_range"] = df[["high_low", "high_prev_close", "low_prev_close"]].max(axis=1)
    df["atr-s"] = df["true_range"].rolling(window=14).mean()
    df["atr-l"] = df["true_range"].rolling(window=50).mean()

    df["ATR-Ratio"] = df["atr-s"] / df["atr-l"]

    df["Target"] = np.where(df["Returns"].shift(-1) > 0, 1, 0).astype(int)
    df["Month"] = df.index.to_series().dt.to_period("M")
    df = df.loc[~df.index.duplicated(keep="first")].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def walk_forward_slices(df: pd.DataFrame):
    months = sorted(df["Month"].unique())
    append_df = []
    for i in range(1, len(months)):
        train_month = months[:i]
        train_df = df[df["Month"].isin(train_month)]

        test_month = months[i]
        test_df = df[df["Month"] == test_month]

        append_df.append((train_df, test_df))

    return append_df
