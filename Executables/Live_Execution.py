import csv
import os
import random  # Added for API rate-limit staggering
import sys
import threading
import time
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
env_path = "/Users/sharmanjeurkar/Projects/SequenceAlpha/Data/access_token.env"

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
        "resolution": str(resolution),
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
        print(f"⚠️ Error fetching data for {symbol}: {response}")
        return None


# ==========================================
# 5. LIVE EXECUTION ENGINE
# ==========================================
def start_trading_bot(
    symbol, stage1_path, stage2_path, csv_file, UPTHRESHOLD, DOWNTHRESHOLD
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Starting Live Trading Engine for {symbol} on {device}...")

    # Dynamic Weights Initialization
    INPUT_SIZE_1 = 13
    HIDDEN_UNITS_1 = 128
    stage1_model = LSTM_Market_Direction(
        in_size=INPUT_SIZE_1, hidden_units=HIDDEN_UNITS_1, out_feautures=1
    ).to(device)
    stage1_model.load_state_dict(
        torch.load(stage1_path, map_location=device, weights_only=True)
    )
    stage1_model.eval()

    INPUT_SIZE_2 = 14
    HIDDEN_UNITS_2 = 64
    stage2_model = LSTM_Position_Detector(
        in_size=INPUT_SIZE_2, hidden_units=HIDDEN_UNITS_2, out_features=1
    ).to(device)
    stage2_model.load_state_dict(
        torch.load(stage2_path, map_location=device, weights_only=True)
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

    CSV_FILE = os.path.join(
        "/Users/sharmanjeurkar/Projects/SequenceAlpha/Executables", csv_file
    )
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Close",
                    "Stage1_Prob",
                    "Stage2_Raw_Logit",
                    "Risk_Manager_Signal",
                ]
            )

    last_processed_time = None

    while True:
        now = datetime.now()

        # 1. Market Hours Check (9:15 AM to 3:30 PM IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if market_open <= now <= market_close:
            # 2. 15-minute sync boundary anchor
            if now.minute % 15 == 0 and now.second < 45:
                # 3. API Rate-Limit Staggering
                time.sleep(random.uniform(0.1, 2.5))

                current_candle_minute = now.strftime("%Y-%m-%d %H:%M")
                try:
                    if last_processed_time != current_candle_minute:
                        last_processed_time = current_candle_minute
                        df_live = fyers_load_data_live(symbol=symbol, daysback=7)

                        if df_live is None or df_live.empty:
                            print(
                                f"⚠️ [{symbol}] API returned empty data at {current_candle_minute}. Retrying in 2 seconds..."
                            )
                            time.sleep(2)
                            continue
                        last_processed_time = current_candle_minute
                        current_close = df_live["Close"].iloc[-1]
                        df_live = live_feature_engineering(df_live)
                        X_live_raw = df_live[feat_cols].values

                        # Ensure enough historical data is present for a full 60-candle window + 10-candle recent history
                        if len(X_live_raw) < 70:
                            print(
                                f"⚠️ [{symbol}] Waiting for more historical data rows (Found {len(X_live_raw)}). Needs 70+."
                            )
                            continue

                        with torch.inference_mode():
                            # ---------------------------------------------
                            # STAGE 1: STATELESS MACRO REGIME ORACLE
                            # ---------------------------------------------
                            recent_probs = []
                            for i in range(len(X_live_raw) - 10, len(X_live_raw)):
                                window = X_live_raw[i - 59 : i + 1]
                                window_3d = window.reshape(1, 60, INPUT_SIZE_1)
                                window_scaled = scale_sequences_locally(window_3d)
                                input_ten_1 = torch.tensor(
                                    window_scaled, dtype=torch.float32
                                ).to(device)
                                prob = torch.sigmoid(stage1_model(input_ten_1)).item()
                                recent_probs.append(prob)

                            # The current moment's probability is the last one calculated
                            stage_1_prob = recent_probs[-1]

                            # ---------------------------------------------
                            # STAGE 2: MICRO SNIPER TRIGGER
                            # ---------------------------------------------
                            final_signal = 0
                            stage_2_logit = 0.0

                            if (
                                stage_1_prob >= UPTHRESHOLD
                                or stage_1_prob <= DOWNTHRESHOLD
                            ):
                                stage2_input_raw = X_live_raw[-10:]
                                stage2_input_3d = stage2_input_raw.reshape(
                                    1, 10, INPUT_SIZE_1
                                )
                                stage2_tech_scaled = scale_sequences_locally(
                                    stage2_input_3d
                                )[0]

                                prob_col = np.array(recent_probs).reshape(10, 1)
                                stage2_final_input = np.hstack(
                                    (stage2_tech_scaled, prob_col)
                                )

                                input_ten_2 = (
                                    torch.tensor(
                                        stage2_final_input, dtype=torch.float32
                                    )
                                    .unsqueeze(0)
                                    .to(device)
                                )
                                stage_2_logit = stage2_model(input_ten_2).item()

                                # 4. Upgraded Logit confirmation barrier (> 0.5)
                                if stage_1_prob >= UPTHRESHOLD and stage_2_logit > 0.5:
                                    final_signal = 1
                                elif (
                                    stage_1_prob <= DOWNTHRESHOLD
                                    and stage_2_logit < -0.5
                                ):
                                    final_signal = -1

                        with open(CSV_FILE, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [
                                    current_candle_minute,
                                    current_close,
                                    round(stage_1_prob, 4),
                                    round(stage_2_logit, 4),
                                    final_signal,
                                ]
                            )

                        print(
                            f"⚡ [{symbol}] Time: {current_candle_minute} | Close: ₹{current_close:.2f}"
                        )
                        print(
                            f"🔮 Oracle Prob: {stage_1_prob:.4f} | Sniper Logit: {stage_2_logit:.4f} | Execution Signal: {final_signal}"
                        )
                        print("-" * 65)
                except Exception as e:
                    print(f"🚨 FATAL ERROR in {symbol} Thread: {str(e)}")
                    # Reset the lock so it tries again next candle instead of permanently dying
                    last_processed_time = None
            time.sleep(60)
        else:
            time.sleep(1)


# ==========================================
# 6. ENGINE MULTI-THREAD CONTROLLER
# ==========================================
def run_all_bots():
    print("Initializing SequenceAlpha Master Controller Thread Pool...")

    # Complete configuration mapping with specific target thresholds matching structural behaviors
    bots_config = [
        {
            "symbol": "NSE:ADANIENT-EQ",
            "stage1_path": "/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/ADANI1.pt",
            "stage2_path": "/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/ADANI2.pt",
            "csv_file": "ADANI_ENTERPRIZES_trades.csv",
            "UPTHRESHOLD": 0.7000,
            "DOWNTHRESHOLD": 0.1600,
        },
        {
            "symbol": "NSE:TRENT-EQ",
            "stage1_path": "/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/TRENT1.pt",
            "stage2_path": "/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/TRENT2.pt",
            "csv_file": "TRENT_trades.csv",
            "UPTHRESHOLD": 0.5750,
            "DOWNTHRESHOLD": 0.1650,
        },
        {
            "symbol": "NSE:INDUSINDBK-EQ",
            "stage1_path": "/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/INDBANK1.pt",
            "stage2_path": "/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/INDBANK2.pt",
            "csv_file": "INDBANK_trades_30m.csv",
            "UPTHRESHOLD": 0.6200,
            "DOWNTHRESHOLD": 0.1200,
        },
    ]

    threads = []

    for config in bots_config:
        # Verify the specific models actually exist before launching the thread
        if not os.path.exists(config["stage1_path"]) or not os.path.exists(
            config["stage2_path"]
        ):
            print(
                f"🚨 SKIP WARNING: Missing .pt files for {config['symbol']}. Ensure Specialist models are trained."
            )
            continue

        t = threading.Thread(
            target=start_trading_bot,
            args=(
                config["symbol"],
                config["stage1_path"],
                config["stage2_path"],
                config["csv_file"],
                config["UPTHRESHOLD"],
                config["DOWNTHRESHOLD"],
            ),
        )
        t.start()
        threads.append(t)
        time.sleep(2)  # Non-throttling API cadence lock

    for t in threads:
        t.join()


if __name__ == "__main__":
    print("Testing Master API Infrastructure Connection...")
    test_df = fyers_load_data_live(symbol="NSE:RELIANCE-EQ", daysback=2)
    if test_df is not None and not test_df.empty:
        print(f"✅ Connection Handshake Succeeded. System Hot.")
        run_all_bots()
    else:
        print("🚨 Fix your API access_token variables before structural boot.")
