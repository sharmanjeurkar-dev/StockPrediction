import numpy as np
import pandas as pd


def apply_triple_barrier_labels(
    df: pd.DataFrame,
    vol_col="Daily_Vol",
    max_days: int = 8,
    profit_mult: float = 1.5,
    stop_mult: float = 1.0,
    price_close: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
) -> pd.DataFrame:

    close = df[price_close].values
    high = df[high_col].values
    low = df[low_col].values
    vol = df[vol_col].values

    returns = np.zeros(len(df))

    for i in range(len(df) - max_days):
        entry_price = close[i]
        current_vol = vol[i]

        # Handle early rows where rolling volatility is still NaN or Zero
        if np.isnan(current_vol) or current_vol == 0:
            returns[i] = np.nan  # inmvalid data for traing and validations
            continue

        # Dynamically scale barriers: 2.0x vol for Profit, 2.0x vol for Stop Loss
        upper_barrier = entry_price * (1 + (profit_mult * current_vol))
        lower_barrier = entry_price * (1 - (stop_mult * current_vol))

        for j in range(1, max_days + 1):
            future_idx = i + j

            # check if both the barriers hit
            if high[future_idx] >= upper_barrier and low[future_idx] <= lower_barrier:
                returns[i] = (
                    low[future_idx] - entry_price
                ) / entry_price  # Bearish Breakdown overirght returns
                break

            #  Check Upper Barrier (Profit Target hit first)
            elif high[future_idx] >= upper_barrier:
                returns[i] = (
                    high[future_idx] - entry_price
                ) / entry_price  # Bullish Breakout - positive
                break

            # Check Lower Barrier (Stop Loss hit first)
            elif low[future_idx] <= lower_barrier:
                returns[i] = (
                    low[future_idx] - entry_price
                ) / entry_price  # Bearish Breakdown
                break

            #  Vertical Time Barrier: Sideways Chop
            if j == max_days:
                returns[i] = close[i + max_days] - entry_price  # Mark sideways noise

    df["Target"] = returns
    return df


def symbolwiseLabeling(
    df: pd.DataFrame, symbol_col: str = "symbol", date_col: str = "Datetime"
) -> pd.DataFrame:
    labled_frames = []

    for _, group in df.groupby(symbol_col, sort=False):
        symbol_df = group.sort_values(date_col).reset_index(drop=True)
        target_df = apply_triple_barrier_labels(symbol_df)
        labled_frames.append(target_df)

    result = pd.concat(labled_frames, axis=0, ignore_index=False)

    return result
