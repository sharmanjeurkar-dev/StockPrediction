"""
Standalone diagnostic: does ANY individual feature actually correlate with
Target? Checks three things, cheapest/most-informative first:

1. Global correlation (Pearson + Spearman) across all rows pooled together
   -- quick sanity check, but can be misleading (mixes cross-sectional and
   time-series effects).

2. Cross-sectional IC per date: for each day, rank stocks by the feature,
   rank stocks by Target, spearman-correlate those ranks. This is the SAME
   thing your compute_fold_metrics() does for model predictions -- doing it
   for raw features tells you the ceiling a single feature alone could hit.
   Averaged over all dates -> "feature IC", directly comparable to your
   model's IC numbers.

3. Per-feature summary sorted by |mean IC| descending, so you can see which
   (if any) features carry real cross-sectional signal.

Does not touch label.py, model_execute.py, or the running training job.
Read-only, CPU-only, no GPU contention.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from Data.feature_engineering import build_training_set

FEATURES = [
    "Intraday_Spread",
    "ATR-Ratio",
    "Intraday_Spread_Zscore",
    "RSI-close-score",
    "relative_strength_60d",
    "relative_strength_120d",
    "Vol_Percentile_Rank",
    "Beta_60D",
]


TARGET = "Target"
DATE_COL = "Datetime"
GROUP_COL = "symbol"


def global_correlation(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    valid = df.dropna(subset=[TARGET])
    for feat in features:
        if feat not in valid.columns:
            continue
        sub = valid.dropna(subset=[feat])
        if len(sub) < 30:
            continue
        pear_r, pear_p = pearsonr(sub[feat], sub[TARGET])
        spear_r, spear_p = spearmanr(sub[feat], sub[TARGET])
        rows.append(
            {
                "feature": feat,
                "n_rows": len(sub),
                "pearson_r": pear_r,
                "pearson_p": pear_p,
                "spearman_r": spear_r,
                "spearman_p": spear_p,
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_r", key=abs, ascending=False)


def cross_sectional_ic(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    For each feature, compute daily cross-sectional Spearman IC (feature vs
    Target, ranked across symbols on that date), then average across all
    dates. This is the apples-to-apples comparison to your model's IC.
    """
    valid = df.dropna(subset=[TARGET])
    results = {feat: [] for feat in features if feat in valid.columns}

    for date, group in valid.groupby(DATE_COL):
        if len(group) < 10:
            continue  # too few symbols that day for a meaningful rank IC
        for feat in results.keys():
            sub = group.dropna(subset=[feat])
            if len(sub) < 10:
                continue
            if sub[feat].nunique() < 2 or sub[TARGET].nunique() < 2:
                continue
            ic, _ = spearmanr(sub[feat], sub[TARGET])
            if not np.isnan(ic):
                results[feat].append(ic)

    rows = []
    for feat, ic_list in results.items():
        if not ic_list:
            continue
        ic_arr = np.array(ic_list)
        rows.append(
            {
                "feature": feat,
                "n_dates": len(ic_arr),
                "mean_ic": ic_arr.mean(),
                "std_ic": ic_arr.std(),
                "ic_ir": ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else np.nan,
                "pct_positive_days": 100 * (ic_arr > 0).mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_ic", key=abs, ascending=False)


if __name__ == "__main__":
    today = datetime.now()
    df, failure_df = build_training_set(
        snapshot_date=datetime.strftime(today, "%Y-%m-%d")
    )

    if len(failure_df):
        print(f"Note: {len(failure_df)} symbols failed during build_training_set")

    print("=" * 70)
    print("GLOBAL CORRELATION (all rows pooled -- quick sanity check)")
    print("=" * 70)
    global_df = global_correlation(df, FEATURES)
    print(global_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("CROSS-SECTIONAL IC (per-date rank correlation, avg across dates)")
    print("This is directly comparable to your model's fold IC numbers.")
    print("=" * 70)
    cs_df = cross_sectional_ic(df, FEATURES)
    print(cs_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("mean_ic near 0 with high variance across features -> feature set")
    print("likely lacks standalone predictive power for this Target design.")
    print("mean_ic clearly nonzero (e.g. |IC| > 0.02-0.03) for any feature ->")
    print("real signal exists; TFT should be able to pick it up if trained")
    print("correctly -- worth revisiting model/training setup instead.")
