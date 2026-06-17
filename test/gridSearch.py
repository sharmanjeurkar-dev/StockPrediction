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

# ==========================================
# 1. DATA LOADING & SCALING
# ==========================================
df = pd.read_csv(
    "indbank_with_stage1_confidence.csv", index_col="Datetime", parse_dates=True
)
df.dropna(inplace=True)


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

# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================
device = "mps" if torch.backends.mps.is_available() else "cpu"

INPUT_SIZE1 = 13
HIDDEN_UNITS1 = 128
OUTPUT_SIZE1 = 1
stage_1 = LSTM_Market_Direction(
    in_size=INPUT_SIZE1, hidden_units=HIDDEN_UNITS1, out_feautures=OUTPUT_SIZE1
).to(device=device)
stage_1.load_state_dict(
    torch.load(
        "/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/INDBANK1.pt",
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
        "/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/INDBANK2.pt",
        map_location=device,
        weights_only=True,
    )
)

stage_1.eval()
stage_2.eval()

feat_cols1 = [
    "Time_Sin",
    "Time_Cos",
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
feat_cols2 = feat_cols1 + ["Stage-1-confidence"]


# ==========================================
# 3. THE WALK-FORWARD FUNCTION
# ==========================================
def run_optimization_for_window(
    train_start_date, train_end_date, test_start_date, test_end_date
):
    """
    Trains/Optimizes thresholds on data prior to train_end_date.
    Validates on data after test_start_date.
    """
    print(
        f"\n🚀 Running Optimization: Train < {train_end_date} | Test >= {test_start_date}"
    )
    print("-" * 60)

    train_df = df.loc[
        (df["year-month"] >= train_start_date) & (df["year-month"] < train_end_date)
    ]
    test_df = df.loc[
        (df["year-month"] >= test_start_date) & (df["year-month"] < test_end_date)
    ]

    if len(train_df) < 100 or len(test_df) < 60:
        print(f"⚠️ Not enough data for Test Month {test_start_date}. Skipping.")
        return

    best_profit = -np.inf
    best_up = 1.0
    best_down = 0.0

    train_df_stage2 = train_df.copy()

    with torch.inference_mode():
        # --- IN-SAMPLE TRAINING ---
        train_df_final = train_df_stage2[9:].copy()
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

        stage1_prob_aligned = train_df_final["Stage-1-confidence"].values
        stage_2_output = stage_2(X_clean2)
        stage_2_raw = stage_2_output.cpu().detach().numpy().flatten()

        # 0.05 Logit Barrier
        stage_2_signal = np.where(
            stage_2_raw > 0.05, 1, np.where(stage_2_raw < -0.05, -1, 0)
        )

        mean_val = train_df_final["Stage-1-confidence"].mean()

        # The Dead Zone is enforced here!
        up = np.arange(0.54, 0.90, 0.01)
        down = np.arange(0.10, 0.35, 0.01)

        for up_check in up:
            for down_check in down:
                long_entry = (stage1_prob_aligned > up_check) & (stage_2_signal == 1)
                short_entry = (stage1_prob_aligned < down_check) & (
                    stage_2_signal == -1
                )
                neutral_exit = (stage1_prob_aligned <= (mean_val + 0.05)) & (
                    stage1_prob_aligned >= (mean_val - 0.05)
                )

                signals = np.where(
                    long_entry,
                    1,
                    np.where(short_entry, -1, np.where(neutral_exit, 0, np.nan)),
                )

                train_df_final["Position"] = pd.Series(signals).ffill().fillna(0).values
                train_df_final["Trade_Taken"] = (
                    train_df_final["Position"] != train_df_final["Position"].shift(1)
                ).astype(int)
                train_df_final["Strategy_Returns"] = (
                    train_df_final["Position"].shift(1) * train_df_final["Returns"]
                )
                train_df_final["Net_Returns"] = train_df_final["Strategy_Returns"] - (
                    train_df_final["Trade_Taken"] * 0.0003
                )

                final_equity = (1 + train_df_final["Net_Returns"]).cumprod().iloc[-1]
                trade_count = train_df_final["Trade_Taken"].sum()

                if final_equity > best_profit:
                    best_profit = final_equity
                    best_up = up_check
                    best_down = down_check

        net_pct = (best_profit - 1) * 100
        print(f"✅ In-Sample Optimal Up Threshold: {best_up:.4f}")
        print(f"✅ In-Sample Optimal Down Threshold: {best_down:.4f}")
        print(f"🚀 In-Sample Maximum Net Profit: {net_pct:.2f}%")

        # --- OUT-OF-SAMPLE TESTING ---
        test_df_stage2 = test_df[59:].copy()
        X_raw_1 = test_df[feat_cols1].values
        y_raw = np.zeros(len(X_raw_1))
        slider1 = Slider(X_raw_1, y_raw, length=60)
        x_window_1, _ = slider1.slider()
        x_scaled_1 = scale_sequences_locally(x_window_1)
        X_clean1, _ = Data_prep.convertNumpyToTensors(x_scaled_1, y_raw)
        X_clean1 = X_clean1.to(device=device)

        stage_1_output_test = stage_1(X_clean1)
        stage_1_prob_test = (
            torch.sigmoid(stage_1_output_test).cpu().detach().numpy().flatten()
        )
        test_df_stage2["Stage-1-confidence"] = stage_1_prob_test

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

        stage1_prob_aligned_test = stage_1_prob_test[9:]
        stage_2_output_test = stage_2(X_clean2_test)
        stage_2_raw_test = stage_2_output_test.cpu().detach().numpy().flatten()
        stage_2_signal_test = np.where(
            stage_2_raw_test > 0.05, 1, np.where(stage_2_raw_test < -0.05, -1, 0)
        )

        test_df_final = test_df_stage2[9:]

        long_entry_test = (stage1_prob_aligned_test >= best_up) & (
            stage_2_signal_test == 1
        )
        short_entry_test = (stage1_prob_aligned_test <= best_down) & (
            stage_2_signal_test == -1
        )
        neutral_exit_test = (stage1_prob_aligned_test <= (mean_val + 0.05)) & (
            stage1_prob_aligned_test >= (mean_val - 0.05)
        )

        signals_test = np.where(
            long_entry_test,
            1,
            np.where(short_entry_test, -1, np.where(neutral_exit_test, 0, np.nan)),
        )

        test_df_final["Position"] = pd.Series(signals_test).ffill().fillna(0).values
        test_df_final["Trade_Taken"] = (
            test_df_final["Position"] != test_df_final["Position"].shift(1)
        ).astype(int)
        test_df_final["Strategy_Returns"] = (
            test_df_final["Position"].shift(1) * test_df_final["Returns"]
        )
        test_df_final["Net_Returns"] = test_df_final["Strategy_Returns"] - (
            test_df_final["Trade_Taken"] * 0.0003
        )

        final_equity_test = (1 + test_df_final["Net_Returns"]).cumprod().iloc[-1]
        net_pct_test = (final_equity_test - 1) * 100
        test_trade_count = test_df_final["Trade_Taken"].sum()

        print("-" * 60)
        print(f"📊 Out-of-Sample Final Profit: {net_pct_test:.2f}%")
        print(f"🛠️ Out-of-Sample Trades Taken: {test_trade_count}")
        print(f"📈 Test Oracle Max Prob: {stage1_prob_aligned_test.max():.4f}")
        print(f"📉 Test Oracle Min Prob: {stage1_prob_aligned_test.min():.4f}")
        print("-" * 60)


# ==========================================
# 4. WALK-FORWARD MASTER CONTROLLER
# ==========================================
if __name__ == "__main__":
    print("Initializing Walk-Forward Optimization Engine...")
    print("Strategy: 12-Month Rolling Train -> 1-Month Blind Test")

    # Define your testing horizon (Adjust these based on what CSV data you have)
    # We will start testing in Jan 2025, which means it will train on Jan-Dec 2024.
    start_test_month = pd.Period("2026-01", freq="M")
    end_test_month = pd.Period("2026-06", freq="M")  # The final month you want to test

    TRAIN_WINDOW_SIZE = 12  # Months of memory

    current_test = start_test_month

    # Slide the window forward one month at a time
    while current_test <= end_test_month:
        # Calculate the exact boundaries for this step
        train_start = current_test - TRAIN_WINDOW_SIZE
        train_end = current_test  # Exclusive upper bound
        test_start = current_test
        test_end = current_test + 1  # Test for exactly 1 month

        print("\n" + "=" * 70)
        print(
            f"🔄 WFO STEP: Train [{train_start} to {train_end}) | Test [{test_start}]"
        )
        print("=" * 70)

        # Execute the optimization and validation for this specific slice
        run_optimization_for_window(
            train_start_date=str(train_start),
            train_end_date=str(train_end),
            test_start_date=str(test_start),
            test_end_date=str(test_end),
        )

        # Step time forward by 1 month and repeat
        current_test += 1
