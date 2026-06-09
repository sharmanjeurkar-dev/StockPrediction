import os
import sys

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Pulling data from Data preprocessing module which alos donwload the data for me
df = pd.read_csv(
    "NIFTY_with_stage1_confidence.csv", index_col="Datetime", parse_dates=True
)
df["Target-Returns"] = df["Returns"].shift(-1)
temp = df
temp["Stage-1-confidence"] = 1 / (1 + np.exp(-temp["Stage-1-confidence"]))
max = temp["Stage-1-confidence"].max()
min = temp["Stage-1-confidence"].min()
mean = temp["Stage-1-confidence"].mean()

print(max, min, mean)
up = np.arange(mean, max, 0.001)
down = np.arange(min, mean, 0.001)

best_profit = -np.inf
best_up = 0.0
best_down = 0.0


for up_check in up:
    for down_check in down:
        temp["Position"] = np.where(temp["Stage-1-confidence"] > up_check, 1.0, 0.0)
        temp["Position"] = np.where(temp["Stage-1-confidence"] < down_check, -1, 0)

        temp["Trade_Taken"] = (temp["Position"] != temp["Position"].shift(1)).astype(
            int
        )
        temp["Strategy_Returns"] = temp["Position"].shift(1) * temp["Returns"]
        temp["Net_Returns"] = temp["Strategy_Returns"] - (temp["Trade_Taken"] * 0.0004)

        final_equity = (1 + temp["Net_Returns"]).cumprod().iloc[-1]

        if final_equity > best_profit:
            best_profit = final_equity

            best_up = up_check
            best_down = down_check

net_pct = (best_profit - 1) * 100
print(f"✅ Optimal Up Threshold: {best_up:.8f}")
print(f"✅ Optimal Down Threshold: {best_down:.8f}")
print(f"🚀 Maximum Net Profit: {net_pct:.8f}%")
