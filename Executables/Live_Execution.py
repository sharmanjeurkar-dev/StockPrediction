import csv
import os
import sys
import time
from collections import deque
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.LSTM_Market_Direction import LSTM_Market_Direction
from models.LSTM_Position_Detector import LSTM_Position_Detector

# ==========================================
# 1. AUTHENTICATION & SETUP
# ==========================================
env_path = "/Users/sharmanjeurkar/Projects/StockPrediction/Data/access_token.env"

if not os.path.exists(env_path):
    print(f"🚨 FILE NOT FOUND: The path {env_path} does not exist!")
else:
    print(f"📁 Found env file at: {env_path}")

load_dotenv(dotenv_path=env_path)
CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")

if CLIENT_ID is None or ACCESS_TOKEN is None:
    print("🚨 ERROR: Credentials not found in .env file.")
    sys.exit()

fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID, is_async=False, token=ACCESS_TOKEN, log_path=""
)


# ==========================================
# 2. LIVE FEATURE ENGINEERING
# ==========================================
def live_feature_engineering(df: pd.DataFrame):
    """Calculates 13 features for live execution."""
    df.sort_index(inplace=True)
    df = df.loc[~df.index.duplicated(keep="first")].copy()

    df["Returns"] = df["Close"].pct_change()

    df["Intraday_Spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["Volume_Price_Velocity"] = df["Returns"] * (
        df["Volume"] / df["Volume"].rolling(20).mean()
    )

    df["Z-score-close"] = (df["Close"] - df["Close"].rolling(window=20).mean()) / df[
        "Close"
    ].rolling(window=20).std()

    df["Change"] = df["Close"].diff()
    df["Gain"] = np.where(df["Change"] > 0, df["Change"], 0)
    df["Loss"] = np.where(df["Change"] < 0, abs(df["Change"]), 0)

    df["Avg_Gain"] = df["Gain"].rolling(window=14).mean()
    df["Avg_Loss"] = df["Loss"].rolling(window=14).mean()
    df["Rs"] = df["Avg_Gain"] / df["Avg_Loss"]
    df["RSI-close-score"] = 100 - 100 / (1 + df["Rs"])

    df["EMA-Today-close-26D"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["EMA-Today-close-12D"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["MACD-Line"] = df["EMA-Today-close-12D"] - df["EMA-Today-close-26D"]
    df["Single-Line"] = df["MACD-Line"].ewm(span=9, adjust=False).mean()
    df["MACD-Histogram"] = df["MACD-Line"] - df["Single-Line"]

    df["Middle-Band"] = df["Close"].rolling(window=20).mean()
    df["Upper-Band"] = df["Middle-Band"] + 2 * df["Close"].rolling(window=20).std()
    df["Lower-Band"] = df["Middle-Band"] - 2 * df["Close"].rolling(window=20).std()
    df["Bollinger-Bandwidth"] = (df["Upper-Band"] - df["Lower-Band"]) / df[
        "Middle-Band"
    ]
    df["%-Band"] = (df["Close"] - df["Lower-Band"]) / (
        df["Upper-Band"] - df["Lower-Band"]
    )

    condition = [(df["Change"] > 0), (df["Change"] < 0)]
    choices = [1, -1]
    df["obv-dir"] = np.select(condition, choices, default=0)
    df["OBV"] = (df["Volume"] * df["obv-dir"]).cumsum()
    df["OBV-ROC"] = df["OBV"].pct_change(10).replace([np.inf, -np.inf], 0).fillna(0)
    df["Volume-Rate-of-Change"] = (df["Volume"].pct_change(periods=10)).replace(
        [np.inf, -np.inf], 0
    ).fillna(0) * 100

    df["high_low"] = df["High"] - df["Low"]
    df["high_prev_close"] = (df["High"] - df["Close"].shift(1)).abs()
    df["low_prev_close"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["true_range"] = df[["high_low", "high_prev_close", "low_prev_close"]].max(axis=1)
    df["atr-s"] = df["true_range"].rolling(window=14).mean()
    df["atr-l"] = df["true_range"].rolling(window=50).mean()
    df["ATR-Ratio"] = df["atr-s"] / df["atr-l"]

    dt_index = pd.to_datetime(df.index)
    minutes_elapsed = (dt_index.hour * 60 + dt_index.minute) - 555
    time_fraction = np.clip(minutes_elapsed / 375.0, 0.0, 1.0)
    df["Time_Sin"] = np.sin(time_fraction * 2 * np.pi)
    df["Time_Cos"] = np.cos(time_fraction * 2 * np.pi)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.bfill().ffill().fillna(0)

    return df


# ==========================================
# 3. LOCAL SEQUENCE SCALER
# ==========================================
def scale_sequences_locally(X_sequences):
    """Standardizes a 3D sequence window individually based on its own local momentum."""
    scaled_X = np.zeros_like(X_sequences)
    for i in range(X_sequences.shape[0]):
        seq = X_sequences[i]
        mean = np.mean(seq, axis=0)
        std = np.std(seq, axis=0) + 1e-8
        scaled_X[i] = (seq - mean) / std
    return scaled_X


# ==========================================
# 4. API DATA FETCHING
# ==========================================
def fyers_load_data_live(symbol="NSE:RELIANCE-EQ", resolution="15", daysback=10):
    today = date.today()
    start_date = today - timedelta(days=daysback)
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start_date.strftime("%Y-%m-%d"),
        "range_to": today.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }
    response = fyers.history(data=data)
    if response.get("s") == "ok":
        candles = response["candles"]
        df = pd.DataFrame(
            candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
        )
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
        df["Datetime"] = (
            df["Datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
        )
        df["Datetime"] = df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.set_index("Datetime", inplace=True)
        return df
    else:
        print(f"⚠️ Error fetching data: {response}")
        return None


# ==========================================
# 5. LIVE EXECUTION ENGINE
# ==========================================
def start_trading_bot():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Starting Live Trading Engine on {device}...")

    INPUT_SIZE_1 = 13
    HIDDEN_UNITS_1 = 128
    stage1_model = LSTM_Market_Direction(
        in_size=INPUT_SIZE_1, hidden_units=HIDDEN_UNITS_1, out_feautures=1
    ).to(device)
    stage1_model.load_state_dict(
        torch.load(
            "/Users/sharmanjeurkar/Projects/StockPrediction/models/saved/Stage1.pt",
            map_location=device,
            weights_only=True,
        )
    )
    stage1_model.eval()

    INPUT_SIZE_2 = 14
    HIDDEN_UNITS_2 = 64
    stage2_model = LSTM_Position_Detector(
        in_size=INPUT_SIZE_2, hidden_units=HIDDEN_UNITS_2, out_features=1
    ).to(device)
    stage2_model.load_state_dict(
        torch.load(
            "/Users/sharmanjeurkar/Projects/StockPrediction/models/saved/Stage2.pt",
            map_location=device,
            weights_only=True,
        )
    )
    stage2_model.eval()

    feat_cols = [
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

    # Grid Search Optimized Thresholds
    UP_THRESH = 0.4531
    DOWN_THRESH = 0.4245

    # ==========================================
    # 🔥 THE MEMORY PRIMER (PRE-LOAD BUFFER) 🔥
    # ==========================================
    prob_buffer = deque(maxlen=10)

    print("🔄 Running Primer Sequence to pre-fill the memory buffer...")
    df_hist = fyers_load_data_live(symbol="NSE:RELIANCE-EQ", daysback=10)
    df_hist = live_feature_engineering(df_hist)
    X_hist_raw = df_hist[feat_cols].values

    if len(X_hist_raw) < 70:
        print(
            "🚨 CRITICAL: Not enough historical data to prime the buffer! Need at least 70 candles."
        )
        sys.exit()

    with torch.inference_mode():
        # Sweep over the most recent 10 historical candles to populate the deque
        for i in range(len(X_hist_raw) - 10, len(X_hist_raw)):
            window = X_hist_raw[i - 59 : i + 1]  # Exact 60-candle window
            window_3d = window.reshape(1, 60, INPUT_SIZE_1)
            window_scaled = scale_sequences_locally(window_3d)
            input_tensor = torch.tensor(window_scaled, dtype=torch.float32).to(device)
            prob = torch.sigmoid(stage1_model(input_tensor)).item()
            prob_buffer.append(prob)

    print(
        f"✅ Primer Complete. Trajectory Loaded: {[round(p, 4) for p in prob_buffer]}"
    )
    print("⏳ Waiting for next 15-minute interval...")

    CSV_FILE = "/Users/sharmanjeurkar/Projects/StockPrediction/Executables/Live_Paper_Trades.csv"
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Timestamp", "Close", "Stage1_Prob", "Risk_Manager_Signal"]
            )

    last_processed_time = None

    while True:
        now = datetime.now()

        if now.minute % 15 == 0 and now.second < 10:
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")

            if last_processed_time != current_time:
                last_processed_time = current_time
                df_live = fyers_load_data_live(symbol="NSE:RELIANCE-EQ")

                if df_live is None or df_live.empty:
                    continue

                current_close = df_live["Close"].iloc[-1]
                df_live = live_feature_engineering(df_live)
                X_live_raw = df_live[feat_cols].values

                with torch.inference_mode():
                    # ---------------------------------------------
                    # STAGE 1: MACRO ORACLE
                    # ---------------------------------------------
                    stage1_input_raw = X_live_raw[-60:]
                    stage1_input_3d = stage1_input_raw.reshape(1, 60, INPUT_SIZE_1)
                    stage1_input_scaled = scale_sequences_locally(stage1_input_3d)

                    input_ten_1 = torch.tensor(
                        stage1_input_scaled, dtype=torch.float32
                    ).to(device)
                    stage_1_output = stage1_model(input_ten_1)
                    stage_1_prob = torch.sigmoid(stage_1_output).item()

                    # Update Memory Buffer with the new live probability
                    prob_buffer.append(stage_1_prob)

                    # ---------------------------------------------
                    # STAGE 2: MICRO SNIPER & DUAL-KEY LOGIC
                    # ---------------------------------------------
                    final_signal = 0

                    if len(prob_buffer) == 10 and (
                        stage_1_prob >= UP_THRESH or stage_1_prob <= DOWN_THRESH
                    ):
                        stage2_input_raw = X_live_raw[-10:]
                        stage2_input_3d = stage2_input_raw.reshape(1, 10, INPUT_SIZE_1)
                        stage2_tech_scaled = scale_sequences_locally(stage2_input_3d)[0]

                        # Format buffer into a 10x1 vector and stitch it to the 10x13 features
                        prob_col = np.array(prob_buffer).reshape(10, 1)
                        stage2_final_input = np.hstack((stage2_tech_scaled, prob_col))

                        input_ten_2 = (
                            torch.tensor(stage2_final_input, dtype=torch.float32)
                            .unsqueeze(0)
                            .to(device)
                        )
                        stage_2_output = stage2_model(input_ten_2).item()

                        stage_2_prob = 1 / (1 + np.exp(-stage_2_output))
                        stage_2_raw_pred = int(round(stage_2_prob))

                        if stage_1_prob >= UP_THRESH and stage_2_raw_pred == 1:
                            final_signal = 1
                        elif stage_1_prob <= DOWN_THRESH and stage_2_raw_pred == 0:
                            final_signal = -1

                with open(CSV_FILE, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            current_time,
                            current_close,
                            round(stage_1_prob, 4),
                            final_signal,
                        ]
                    )

                print("-" * 50)
                print(f"⏰ Time: {current_time} | Close: ₹{current_close:.2f}")
                print(
                    f"🔮 Oracle Probability: {stage_1_prob:.4f} | Risk Signal: {final_signal}"
                )
                print("-" * 50)

            time.sleep(60)
        else:
            time.sleep(1)


if __name__ == "__main__":
    print("Testing API Connection...")
    test_df = fyers_load_data_live(symbol="NSE:RELIANCE-EQ")
    if test_df is not None:
        print(
            f"✅ Connection Successful! Downloaded {len(test_df)} historical candles."
        )
        start_trading_bot()
    else:
        print("🚨 Fix your API Credentials before starting.")
