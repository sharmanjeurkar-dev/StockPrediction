import os
import sys

import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DualLogger:
    def __init__(self, filepath, stream):
        self.terminal = stream
        self.log = open(filepath, "a")  # "a" ensures it appends instead of overwriting

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Route both standard prints AND tqdm progress bars to the file
sys.stdout = DualLogger("stage1_training_logs.txt", sys.stdout)
sys.stderr = DualLogger("stage1_training_logs.txt", sys.stderr)

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

from Data import Data_prep, Data_preprocessing, data_scraper
from Data.Slider import Slider
from models.LSTM_Market_Direction import LSTM_Market_Direction

data = "NSE:RELIANCE-EQ"

df_raw = data_scraper.scrape_data(symbol=data, DAYS=100, resolution="15")

df = Data_preprocessing.feature_enginiering(df=df_raw)

dataframe_collection = Data_preprocessing.walk_forward_slices(df=df)

feat_cols = [
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

all_predictions = []

for train_df, test_df in dataframe_collection:
    # extract features and labels
    X_train_raw = train_df[feat_cols].values
    Y_train = train_df["Target"].values

    X_test_raw = test_df[feat_cols].values
    Y_test = test_df["Target"].values

    def scale_sequences_locally(X_sequences):

        scaled_X = np.zeros_like(X_sequences)
        for i in range(X_sequences.shape[0]):
            seq = X_sequences[i]
            # Calculate mean and std strictly within this specific 60-candle window
            mean = np.mean(seq, axis=0)
            std = np.std(seq, axis=0) + 1e-8  # Prevent division by zero
            scaled_X[i] = (seq - mean) / std
        return scaled_X

    slider_tr = Slider(feature=X_train_raw, labels=Y_train, length=60)
    slider_te = Slider(feature=X_test_raw, labels=Y_test, length=60)

    x_train_slide, y_train_slide = slider_tr.slider()
    x_test_slide, y_test_slide = slider_te.slider()

    valid_train_idx = np.where(y_train_slide != -1)[0]
    valid_test_idx = np.where(y_test_slide != -1)[0]

    x_train_clean = x_train_slide[valid_train_idx]
    y_train_clean = y_train_slide[valid_train_idx]

    x_test_clean = x_test_slide[valid_test_idx]
    y_test_clean = y_test_slide[valid_test_idx]

    x_train_clean_scaled = scale_sequences_locally(x_train_clean)
    x_test_clean_scaled = scale_sequences_locally(x_test_clean)

    # convert numpy data to tensors
    x_train_convert, y_train_convert = Data_prep.convertNumpyToTensors(
        x_train_clean, y_train_clean
    )
    x_test_convert, y_test_convert = Data_prep.convertNumpyToTensors(
        x_test_clean, y_test_clean
    )

    # Add data to a dataset
    train_dataset = Data_prep.createTensorDataset(x_train_convert, y_train_convert)
    test_dataset = Data_prep.createTensorDataset(x_test_convert, y_test_convert)

    # load dataset
    train_data_load = Data_prep.loadData(dataset=train_dataset, batch=128, num_worker=0)
    test_data_load = Data_prep.loadData(
        dataset=test_dataset, batch=128, num_worker=0, shuffle=False
    )

    # device setup
    device = "mps" if (torch.backends.mps.is_available()) else "cpu"

    # model
    INPUT_SIZE = 13
    HIDDEN_UNITS = 128
    OUT_FEATURES = 1

    model = LSTM_Market_Direction(
        in_size=INPUT_SIZE, hidden_units=HIDDEN_UNITS, out_feautures=OUT_FEATURES
    ).to(device=device)

    # loss and optimizer
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    EPOCHS = 30
    for epoch in range(EPOCHS):
        model.train()
        train_running_loss = 0.0
        test_running_loss = 0.0

        loop = iter(tqdm(train_data_load, desc=f"Epoch: {epoch + 1}/{EPOCHS}"))

        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            prediction = model(x)

            train_loss = loss_fn(prediction, y)

            train_running_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        total_train_loss = train_running_loss / len(train_data_load)
        if epoch % 25 == 0:
            print(f"|Train Loss: {total_train_loss: 0.8f} |")
    model.eval()
    test_pred = []
    with torch.inference_mode():
        for x, y in test_data_load:
            x = x.to(device)
            y = y.to(device)
            probabilities = model(x)

            test_loss = loss_fn(probabilities, y)
            test_running_loss += test_loss.item()
            test_pred.append(probabilities.to("cpu").numpy().flatten())

        total_test_loss = test_running_loss / len(test_data_load)
        test_pred = np.concat(test_pred)

        look_back = 60

        # 1. Get the original full timeline of sliding windows
        base_test_index = test_df.index[look_back:]

        # 2. Filter the timeline using the EXACT SAME indices that survived the filter earlier
        align_test_index = base_test_index[valid_test_idx]

        # 3. Now both the predictions and the index will be exactly 3972!
        test_pred_series = pd.Series(test_pred, index=align_test_index)
        all_predictions.append(test_pred_series)

        print(
            f"Finished Month Step. Generated {len(test_pred_series)} clean out-of-sample predictions."
        )


final_confidence = pd.concat(all_predictions)

df["Stage-1-confidence"] = final_confidence
df["Stage-1-confidence"].dropna()

df.to_csv("NIFTY_with_stage1_confidence.csv", index=True)
print("Stage 1 Pipeline Complete! Dataset saved for Stage 2.")


def plot_metrics(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader):
    model.to("cpu")
    model.eval()

    all_preds = []
    all_targets = []

    print("Running diagnostics on Stage 1 Test Data...")

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to("cpu").to(torch.float32)
            y = y.to("cpu").to(torch.float32)

            output = torch.sigmoid(model(x))

            all_preds.append(output.numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Confidence Distribution Histogram
    axes[0].hist(preds, bins=50, color="purple", alpha=0.7, edgecolor="black")
    axes[0].axvline(
        0.5, color="red", linestyle="dashed", linewidth=1.5, label="Neutral (0.5)"
    )
    axes[0].set_title("Model Confidence Distribution")
    axes[0].set_xlabel("Predicted Probability (0 = Down, 1 = Up)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # 2. Precision vs. Threshold Curve
    precision, recall, thresholds = precision_recall_curve(targets, preds)
    # Thresholds array is 1 element shorter than precision/recall arrays
    axes[1].plot(thresholds, precision[:-1], "b-", label="Precision", linewidth=2)
    axes[1].set_title("Precision vs. Confidence Threshold")
    axes[1].set_xlabel("Confidence Threshold")
    axes[1].set_ylabel("Precision (Win Rate %)")
    axes[1].axhline(0.5, color="gray", linestyle="dotted", label="50% Baseline")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. ROC Curve & AUC
    fpr, tpr, _ = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)
    axes[2].plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    axes[2].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[2].set_title("Receiver Operating Characteristic (ROC)")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_metrics(model=model, test_loader=test_data_load)
torch.save(
    model.state_dict(),
    "/Users/sharmanjeurkar/Projects/StockPrediction/models/saved/Stage1.pt",
)

print("✅ Stage 1 Model Saved!")
