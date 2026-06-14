import os
import sys

import numpy as np
import pandas as pd
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Data import Data_prep
from Data.Slider import Slider
from models.LSTM_Market_Direction import LSTM_Market_Direction
from models.LSTM_Position_Detector import LSTM_Position_Detector

df = pd.read_csv(
    "NIFTY_with_stage1_confidence.csv", index_col="Datetime", parse_dates=True
)


def scale_sequences_locally(X_sequences):
    scaled_X = np.zeros_like(X_sequences)
    for i in range(X_sequences.shape[0]):
        seq = X_sequences[i]
        mean = np.mean(seq, axis=0)
        std = np.std(seq, axis=0) + 1e-8
        scaled_X[i] = (seq - mean) / std
    return scaled_X


dt = pd.to_datetime(df.index, unit="s")
df["year-month"] = dt.to_period("M")
train_df = df.loc[df["year-month"] < "2026-01"]
test_df = df.loc[df["year-month"] > "2026-01"]


best_profit = -np.inf
best_up = 0.0
best_down = 0.0

device = "mps" if torch.backends.mps.is_available() else "cpu"

INPUT_SIZE1 = 13
HIDDEN_UNITS1 = 128
OUTPUT_SIZE1 = 1
stage_1 = LSTM_Market_Direction(
    in_size=INPUT_SIZE1, hidden_units=HIDDEN_UNITS1, out_feautures=OUTPUT_SIZE1
).to(device=device)
stage_1.load_state_dict(
    torch.load(
        "/Users/sharmanjeurkar/Projects/StockPrediction/models/saved/Stage1.pt",
        map_location=device,
        weights_only=True,
    )
)
INPUT_SIZE2 = 14
HIDDEN_UNITS2 = 64
OUTPUT_SIZE2 = 1
stage_2 = LSTM_Position_Detector(
    in_size=INPUT_SIZE2, hidden_units=HIDDEN_UNITS2, out_features=OUTPUT_SIZE2
).to(device=device)
stage_2.load_state_dict(
    torch.load(
        "/Users/sharmanjeurkar/Projects/StockPrediction/models/saved/Stage2.pt",
        map_location=device,
        weights_only=True,
    )
)
stage_1.eval()
stage_2.eval()

feat_cols1 = [
    "Time_Cos",
    "Time_Sin",
    "Volume_Price_Velocity",
    "Intraday_Spread",
    "Returns",
    "Z-score-close",
    "RSI-close-score",
    "MACD-Histogram",
    "Bollinger-Bandwidth",
    "%-Band",
    "Volume-Rate-of-Change",
    "ATR-Ratio",
    "OBV-ROC",
]
feat_cols2 = [
    "Time_Cos",
    "Time_Sin",
    "Volume_Price_Velocity",
    "Intraday_Spread",
    "Returns",
    "Z-score-close",
    "RSI-close-score",
    "MACD-Histogram",
    "Bollinger-Bandwidth",
    "%-Band",
    "Volume-Rate-of-Change",
    "ATR-Ratio",
    "OBV-ROC",
    "Stage-1-confidence",
]

train_df_stage2 = train_df[60:].copy()
X_raw_1 = train_df[feat_cols1].values
y_raw1 = np.zeros(len(X_raw_1))
slider1 = Slider(X_raw_1, y_raw1, length=60)
x_window_1, _ = slider1.slider()
x_scaled_1 = scale_sequences_locally(x_window_1)
X_clean1, _ = Data_prep.convertNumpyToTensors(x_scaled_1, y_raw1)
X_clean1 = X_clean1.to(device=device)


with torch.inference_mode():
    stage_1_output = stage_1(X_clean1)
    stage_1_prob = torch.sigmoid(stage_1_output).cpu().detach().numpy().flatten()
    train_df_stage2["Stage-1-confidence"] = stage_1_prob

    train_df_final = train_df_stage2[10:].copy()
    X_raw_2 = train_df_stage2[feat_cols2].values
    y_raw = np.zeros(len(X_raw_2))
    slider2 = Slider(X_raw_2, y_raw, length=10)
    X_window2, _ = slider2.slider()
    x_prescaled = X_window2[:, :, :13]
    x_prob_2 = X_window2[:, :, 13:]
    X_scaled_unstitched = scale_sequences_locally(x_prescaled)
    X_scaled_stitched = np.concatenate((X_scaled_unstitched, x_prob_2), axis=2)
    X_clean2, _ = Data_prep.convertNumpyToTensors(X_scaled_stitched, y_raw)
    X_clean2 = X_clean2.to(device=device)
    stage1_prob_aligned = stage_1_prob[10:]

    stage_2_output = stage_2(X_clean2)
    stage_2_prob = torch.sigmoid(stage_2_output).cpu().detach().numpy().flatten()
    stage_2_signal = np.round(stage_2_prob).astype(int)

    max_val = train_df_final["Stage-1-confidence"].max()
    min_val = train_df_final["Stage-1-confidence"].min()
    mean_val = train_df_final["Stage-1-confidence"].mean()

    print(max_val, min_val, mean_val)
    up = np.arange(mean_val, max_val, 0.001)
    down = np.arange(min_val, mean_val, 0.001)
    for up_check in up:
        for down_check in down:
            train_df_final["Position"] = np.where(
                (stage1_prob_aligned > up_check) & (stage_2_signal == 1),
                1,
                np.where(
                    (stage1_prob_aligned < down_check) & (stage_2_signal == 0), -1, 0
                ),
            )

            train_df_final["Trade_Taken"] = (
                train_df_final["Position"] != train_df_final["Position"].shift(1)
            ).astype(int)
            train_df_final["Strategy_Returns"] = (
                train_df_final["Position"].shift(1) * train_df_final["Returns"]
            )
            train_df_final["Net_Returns"] = train_df_final["Strategy_Returns"] - (
                train_df_final["Trade_Taken"] * 0.0004
            )

            final_equity = (1 + train_df_final["Net_Returns"]).cumprod().iloc[-1]

            if final_equity > best_profit:
                best_profit = final_equity

                best_up = up_check
                best_down = down_check

net_pct = (best_profit - 1) * 100
print(f"✅ Optimal Up Threshold: {best_up:.8f}")
print(f"✅ Optimal Down Threshold: {best_down:.8f}")
print(f"🚀 Maximum Net Profit: {net_pct:.8f}%")

# OutofSample -- Unseen data -- test the grid search values

test_df_stage2 = test_df[60:].copy()
X_raw_1 = test_df[feat_cols1].values
y_raw = np.zeros(len(X_raw_1))
slider1 = Slider(X_raw_1, y_raw, length=60)
x_window_1, _ = slider1.slider()
x_scaled_1 = scale_sequences_locally(x_window_1)
X_clean1, _ = Data_prep.convertNumpyToTensors(x_scaled_1, y_raw)
X_clean1 = X_clean1.to(device=device)

with torch.inference_mode():
    stage_1_output_test = stage_1(X_clean1)
    stage_1_prob_test = (
        torch.sigmoid(stage_1_output_test).cpu().detach().numpy().flatten()
    )
    test_df_stage2["Stage-1-confidence"] = stage_1_prob_test
    test_df_stage2 = test_df[60:]

    X_raw_test2 = test_df_stage2[feat_cols2].values
    y_raw = np.zeros(len(X_raw_test2))
    slider2 = Slider(X_raw_test2, y_raw, length=10)
    x_window_test2, _ = slider2.slider()
    x_prescaled = x_window_test2[:, :, :13]
    x_prob_2 = x_window_test2[:, :, 13:]
    X_scaled_unstitched = scale_sequences_locally(x_prescaled)
    X_scaled_stitched = np.concatenate((X_scaled_unstitched, x_prob_2), axis=2)
    X_clean2_test, _ = Data_prep.convertNumpyToTensors(X_scaled_stitched, y_raw)
    X_clean2_test = X_clean2_test.to(device=device)
    stage1_prob_aligned_test = stage_1_prob_test[10:]

    stage_2_output_test = stage_2(X_clean2_test)
    stage_2_prob_test = (
        torch.sigmoid(stage_2_output_test).cpu().detach().numpy().flatten()
    )
    stage_2_signal_test = np.round(stage_2_prob_test).astype(int)

    test_df_final = test_df_stage2[10:]
    stage_1_prob_aligned_test = stage_1_prob_test[10:]
    up_check = 0.45314366
    down_check = 0.42459068
    test_df_final["Position"] = np.where(
        (stage1_prob_aligned_test >= up_check) & (stage_2_signal_test == 1),
        1,
        np.where(
            (stage_1_prob_aligned_test <= down_check) & (stage_2_signal_test == 0),
            -1,
            0,
        ),
    )

    test_df_final["Trade_Taken"] = (
        test_df_final["Position"] != test_df_final["Position"].shift(1)
    ).astype(int)
    test_df_final["Strategy_Returns"] = (
        test_df_final["Position"].shift(1) * test_df_final["Returns"]
    )
    test_df_final["Net_Returns"] = test_df_final["Strategy_Returns"] - (
        test_df_final["Trade_Taken"] * 0.0004
    )

    final_equity = (1 + test_df_final["Net_Returns"]).cumprod().iloc[-1]
    net_pct = (final_equity - 1) * 100
    print("-" * 50)
    print(f"Final profit test {net_pct}")
    print("-" * 50)
