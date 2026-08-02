import os
from datetime import date, datetime, timedelta

import pandas as pd


def _parse_date_from_filename(f: str):
    return datetime.strptime(f.replace(".parquet", ""), "%Y-%m-%d").date()


path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


def load_snapshot(target_date_str: str) -> tuple[pd.DataFrame, str]:
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    list_of_snapshots_files = []
    list_of_snapshots_files = [
        f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    ]

    sorted_snapshot_files = []
    diff_dict = {}
    diff_list = []
    sorted_snapshot_files = sorted(list_of_snapshots_files)

    for f in sorted_snapshot_files:
        if f == f"{target_date_str}.parquet":
            df = pd.read_parquet(os.path.join(path, f))
            return df, target_date_str
        file_date = _parse_date_from_filename(f)
        diff = (target_date - file_date).days
        diff_list.append((file_date, diff, f))

    prior_candidates = [
        (file_date, diff, f) for (file_date, diff, f) in diff_list if diff >= 0
    ]

    if not prior_candidates:
        raise ValueError(f"No snapshot found on or before {target_date}")

    closest = min(prior_candidates, key=lambda x: x[1])
    resolved_date, _, matched_filename = closest
    df = pd.read_parquet(os.path.join(path, matched_filename))
    return df, resolved_date.strftime("%Y-%m-%d")


def save_snapshot(snapshot_date_str: str, universal_df: pd.DataFrame, overwrite=False):
    target_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    file = os.path.join(path, f"{snapshot_date_str}.parquet")
    os.makedirs(path, exist_ok=True)

    if os.path.exists(file) and overwrite == False:
        raise FileExistsError("Snapshot for the date already exists")

    if "symbol" not in universal_df.columns:
        raise ValueError("Symbols don't exist in the dataframe provided")

    universal_df["snapshot_date"] = snapshot_date_str
    universal_df["generated_at"] = datetime.now()

    universal_df.to_parquet(path=file, index=False)

    return file
