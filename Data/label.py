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
    n = len(df)
    close = df[price_close].values
    high = df[high_col].values
    low = df[low_col].values
    vol = df[vol_col].values

    upper_barrier = close * (1 + (profit_mult * vol))
    lower_barrier = close * (1 - (stop_mult * vol))

    upper_touch_day, lower_touch_day = compute_touch_days(
        high, low, upper_barrier, lower_barrier, max_days
    )

    full_window_mask = (np.arange(n) + max_days) < n

    upper_hit = ~np.isnan(upper_touch_day) & full_window_mask
    lower_hit = ~np.isnan(lower_touch_day) & full_window_mask

    collision_mask = upper_hit & lower_hit & (upper_touch_day == lower_touch_day)
    upper_wins_mask = (
        upper_hit & (~lower_hit | (upper_touch_day < lower_touch_day)) & ~collision_mask
    )
    lower_wins_mask = (
        lower_hit & (~upper_hit | (lower_touch_day < upper_touch_day)) & ~collision_mask
    )
    neither_hit_mask = ~upper_hit & ~lower_hit & full_window_mask

    returns = np.full(n, np.nan)

    def _price_on_touch_day(price_array, touch_day):
        result = np.full(n, np.nan)
        valid_rows = ~np.isnan(touch_day)
        idx = np.arange(n)[valid_rows]
        offsets = touch_day[valid_rows].astype(int)
        result[valid_rows] = price_array[idx + offsets]
        return result

    high_on_upper_touch = _price_on_touch_day(high, upper_touch_day)
    low_on_lower_touch = _price_on_touch_day(low, lower_touch_day)

    returns[collision_mask] = (
        low_on_lower_touch[collision_mask] - close[collision_mask]
    ) / close[collision_mask]

    returns[upper_wins_mask] = (
        high_on_upper_touch[upper_wins_mask] - close[upper_wins_mask]
    ) / close[upper_wins_mask]

    returns[lower_wins_mask] = (
        low_on_lower_touch[lower_wins_mask] - close[lower_wins_mask]
    ) / close[lower_wins_mask]

    vertical_idx = np.arange(n)
    vertical_valid = neither_hit_mask & (vertical_idx + max_days < n)
    returns[vertical_valid] = (
        close[vertical_idx[vertical_valid] + max_days] - close[vertical_valid]
    ) / close[vertical_valid]  # type: ignore

    invalid_vol_mask = np.isnan(vol) | (vol <= 0)  # type: ignore
    returns[invalid_vol_mask] = np.nan

    out = df.copy()
    out["Target"] = returns
    return out


def _first_touch_day(condition_matrix: np.ndarray) -> np.ndarray:
    """
    condition_matrix: shape (n_rows, max_days), boolean.
    condition_matrix[i, j] = True if the barrier was touched on day (j+1)
    looking forward from row i.

    Returns: array of shape (n_rows,) with the 1-indexed day of first touch,
    or np.nan if never touched within the window.
    """
    touched_at_all = condition_matrix.any(axis=1)
    # argmax finds the position of the first True (True=1 beats False=0);
    # if a row is all False, argmax returns 0, which is meaningless --
    # that's exactly why we mask it with touched_at_all right after.
    first_touch_idx = condition_matrix.argmax(axis=1)
    first_touch_day = first_touch_idx + 1  # convert to 1-indexed day offset

    result = np.where(touched_at_all, first_touch_day, np.nan)
    return result


def compute_touch_days(
    high: np.ndarray,
    low: np.ndarray,
    upper_barrier: np.ndarray,
    lower_barrier: np.ndarray,
    max_days: int,
):
    n = len(high)

    # Build the (n_rows, max_days) shifted-comparison matrices.
    # Column j (0-indexed) = "touched on day j+1 from now".
    upper_matrix = np.full((n, max_days), False)
    lower_matrix = np.full((n, max_days), False)

    for j in range(1, max_days + 1):
        shifted_high = np.roll(high, -j)
        shifted_low = np.roll(low, -j)

        # rows near the end don't have j days of real future data --
        # np.roll wraps around, which would be wrong, so mask those out
        valid = np.arange(n) < (n - j)

        upper_matrix[:, j - 1] = valid & (shifted_high >= upper_barrier)
        lower_matrix[:, j - 1] = valid & (shifted_low <= lower_barrier)

    upper_touch_day = _first_touch_day(upper_matrix)
    lower_touch_day = _first_touch_day(lower_matrix)

    return upper_touch_day, lower_touch_day


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
