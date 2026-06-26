import numpy as np
import pandas as pd


def feature_enginiering(df: pd.DataFrame, symbol):
    df["symbol"] = symbol
    df.sort_index(inplace=True)
    df = df.loc[~df.index.duplicated(keep="first")].copy()
    # Calculating the %returns on close price if the share was brought
    df["Returns"] = df["Close"].pct_change()

    # # z -score
    # df["Z-score-close"] = (df["Close"] - df["Close"].rolling(window=20).mean()) / df[
    #     "Close"
    # ].rolling(window=20).std()

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
    df.drop(columns=["EMA-Today-close-26D", "EMA-Today-close-12D"], inplace=True)

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
    df["Intraday_Spread"] = (df["High"] - df["Low"]) / df["Close"]
    # df["Volume_Price_Velocity"] = df["Returns"] * (
    #     df["Volume"] / df["Volume"].rolling(20).mean()
    # )

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

    minutes_elapsed = (df.index.hour * 60 + df.index.minute) - 555
    time_fraction = np.clip(minutes_elapsed / 375.0, 0.0, 1.0)
    df["Time_Sin"] = np.sin(time_fraction * 2 * np.pi)
    df["Time_Cos"] = np.cos(time_fraction * 2 * np.pi)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df["Daily_Vol"] = df["Close"].pct_change().rolling(window=25).std()
    df = apply_triple_barrier_labels(df, vol_col="Daily_Vol", max_candles=8)

    df["MA-20"] = df["Close"].rolling(window=20).mean()
    df["MA-50"] = df["Close"].rolling(window=50).mean()
    df["MA-Cross"] = (df["MA-20"] - df["MA-50"]) / df["MA-50"]

    df.dropna(inplace=True)

    return df


def apply_triple_barrier_labels(df: pd.DataFrame, vol_col="Daily_Vol", max_candles=8):

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    vol = df[vol_col].values

    returns = np.zeros(len(df))

    for i in range(len(df) - max_candles):
        entry_price = close[i]
        current_vol = vol[i]

        # Handle early rows where rolling volatility is still NaN or Zero
        if np.isnan(current_vol) or current_vol == 0:
            returns[i] = -1  # inmvalid data for traing and validations
            continue

        # Dynamically scale barriers: 2.0x vol for Profit, 2.0x vol for Stop Loss
        upper_barrier = entry_price * (1 + (1.5 * current_vol))
        lower_barrier = entry_price * (1 - (1.0 * current_vol))

        for j in range(1, max_candles + 1):
            future_idx = i + j

            #  Check Upper Barrier (Profit Target hit first)
            if high[future_idx] >= upper_barrier:
                returns[i] = (
                    high[future_idx] - entry_price
                ) / entry_price  # Bullish Breakout - positive
                break

            # Check Lower Barrier (Stop Loss hit first)
            elif low[future_idx] <= lower_barrier:
                returns[i] = (
                    high[future_idx] - entry_price
                ) / entry_price  # Bearish Breakdown
                break

            #  Vertical Time Barrier: Sideways Chop
            if j == max_candles:
                returns[i] = 0.0  # Mark sideways noise

    df["Target"] = returns
    return df


def concat_df(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    processed = []

    for df in dfs:
        symbol = df["symbol"].iloc[0]
        df_processed = feature_enginiering(df, symbol)
        processed.append(df_processed)

    final_df = pd.concat(processed, ignore_index=False)
    return final_df


def walk_forward_slices(df: pd.DataFrame):
    year = sorted(df.index.year)
    append_df = []
    for i in range(1, len(year)):
        train_year = year[:i]
        train_df = df[df["Year"].isin(train_year)].copy()

        test_year = year[i]
        test_df = df[df["Year"] == test_year]

        append_df.append((train_df, test_df))

    return append_df
