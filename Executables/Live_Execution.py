import csv
import os
import sys
import time
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from Data.Data_prep import convertNumpyToTensors
from Data.Data_preprocessing import feature_enginiering
from Data.Slider import Slider
from models.LSTM_Market_Direction import LSTM_Market_Direction
from models.LSTM_Position_Detector import LSTM_Position_Detector

env_path = "/Users/sharmanjeurkar/Projects/StockPrediction/Data/access_token.env"

if not os.path.exists(env_path):
    print(f"🚨 FILE NOT FOUND: The path {env_path} does not exist!")
else:
    print(f"📁 Found env file at: {env_path}")


load_dotenv(dotenv_path=env_path)
CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")

print(f"🔍 Diagnostic - CLIENT_ID loaded: {CLIENT_ID}")
print(
    f"🔍 Diagnostic - ACCESS_TOKEN loaded: {ACCESS_TOKEN[:15] if ACCESS_TOKEN else None}..."
)

if CLIENT_ID is None or ACCESS_TOKEN is None:
    print("🚨 STOPPING: Variables failed to load from the env file. Fix paths or keys.")
    sys.exit()


fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False)


def fyers_load_data_live(symbol="NSE:NIFTY50-INDEX", resolution="15", daysback=10):
    endDate = date.today()
    startDate = endDate - timedelta(days=daysback)

    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": startDate.strftime("%Y-%m-%d"),
        "range_to": endDate.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }
    try:
        response = fyers.history(data=data)
        if response["s"] != "ok":
            print(f"❌ API Error: {response['message']}")
            return None

        raw_candles = response["candles"]
        df = pd.DataFrame(
            raw_candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
        )

        # Convert Unix epoch time to readable Datetime
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
        df["Datetime"] = (
            df["Datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
        )
        df.set_index("Datetime", inplace=True)

        return df

    except Exception as e:
        print(f"Exception occured,{e}")
        return None


def start_trading_bot():

    print("🧠 Loading Neural Networks into memory...")
    print("🤖 Live Execution Engine Started. Waiting for the next 15-minute candle...")

    stage1_model = LSTM_Market_Direction(in_size=9, hidden_units=32, out_feautures=1)
    stage2_model = LSTM_Position_Detector(in_size=9, hidden_units=32, out_features=1)

    stage1_model.load_state_dict(
        torch.load(os.path.join(project_root, "models/saved/", "Stage1.pt"))
    )
    stage2_model.load_state_dict(
        torch.load(os.path.join(project_root, "models/saved/", "Stage2.pt"))
    )

    scaler_stage1 = joblib.load(
        os.path.join(project_root, "models", "saved", "scaler_stage1.pkl")
    )
    scaler_stage2 = joblib.load(
        os.path.join(project_root, "models", "saved", "scaler_stage2.pkl")
    )

    stage1_model.eval()
    stage2_model.eval()
    CSV_FILE = os.path.join(script_dir, "Live_Paper_Trades.csv")
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Close", "Stage1_Prob", "Signal"])
    while True:
        now = datetime.now()

        if now.minute % 15 == 0 and now.second == 2:
            print(
                f"\n⏱️ [{now.strftime('%H:%M:%S')}] Candle Closed. Processing snapshot..."
            )
            live_df = fyers_load_data_live()
            if live_df is not None:
                live_df = feature_enginiering(live_df)

                last_row = live_df.iloc[-1]
                current_close = last_row["Close"]
                current_time = last_row.name

                feat_cols1 = [
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

                X = live_df[feat_cols1].values

                X_pross = scaler_stage1.transform(X)

                dummy_labels = np.zeros(len(X_pross))
                liveslider = Slider(feature=X_pross, labels=dummy_labels, length=10)
                x_windows, y = liveslider.slider()
                current_window = x_windows[-1]

                current_close = live_df["Close"].iloc[-1]
                current_time = live_df.index[-1]

                X_input, _ = convertNumpyToTensors(current_window, y)

                with torch.no_grad():
                    raw_logit = stage1_model(X_input).item()
                    stage_1_prob = 1 / (1 + np.exp(-raw_logit))

                    feat_cols2 = [
                        "Returns",
                        "Z-score-close",
                        "RSI-close-score",
                        "MACD-Histogram",
                        "Bollinger-Bandwidth",
                        "%-Band",
                        "Volume-Rate-of-Change",
                        "ATR-Ratio",
                    ]
                    last_row = live_df.iloc[-1]
                    input_raw = last_row[feat_cols2].values
                    input = np.insert(input_raw, 0, stage_1_prob)
                    input_process = scaler_stage2.transform(input.reshape(1, -1))
                    dummy_labels = np.zeros(len(input_process))
                    input_ten, _ = convertNumpyToTensors(input_process, dummy_labels)
                    input_ten = input_ten.unsqueeze(0)
                    output = stage2_model(input_ten).item()
                    final_signal = int(round(output))

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
                print(f"⏰ Time: {current_time} | Close: {current_close:.2f}")
                print(
                    f"🔮 Oracle Probability: {stage_1_prob:.4f} | Final Signal: {final_signal}"
                )
                print("-" * 50)

            time.sleep(60)
        else:
            time.sleep(2)


if __name__ == "__main__":
    # Test a single fetch immediately before starting the infinite loop
    print("Testing API Connection...")
    test_df = fyers_load_data_live()
    if test_df is not None:
        print(
            f"✅ Connection Successful! Downloaded {len(test_df)} historical candles."
        )
        start_trading_bot()
    else:
        print("🚨 Fix your API Credentials before starting.")
