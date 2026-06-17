import os
import sys

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
sys.stdout = DualLogger("stage2_training_logs.txt", sys.stdout)
sys.stderr = DualLogger("stage2_training_logs.txt", sys.stderr)

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

from Data import Data_prep
from Data.Slider import Slider
from models.LSTM_Position_Detector import LSTM_Position_Detector
from utility import CalculatePNL
from utility.DirectionalPeneltyLoss import SharpeRatio

# Pulling data from Data preprocessing module which alos donwload the data for me
df = pd.read_csv(
    "indbank_with_stage1_confidence.csv", index_col="Datetime", parse_dates=True
)

df.dropna(inplace=True)
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
    "Stage-1-confidence",
]

features = df[feat_cols]
features = features.values
labels = np.where((df["Returns"].shift(-1) + df["Returns"].shift(-2)) > 0, 1.0, 0.0)
train_size = int(0.8 * len(features))

X_train_raw = features[:train_size]
X_test_raw = features[train_size:]
Y_train = labels[:train_size]
Y_test = labels[train_size:]


def scale_sequences_locally(X_sequences):
    """
    Standardizes each 10-candle sequence window individually.
    """
    scaled_X = np.zeros_like(X_sequences)
    for i in range(X_sequences.shape[0]):
        seq = X_sequences[i]
        mean = np.mean(seq, axis=0)
        std = np.std(seq, axis=0) + 1e-8  # Prevent division by zero
        scaled_X[i] = (seq - mean) / std
    return scaled_X


slider_tr = Slider(feature=X_train_raw, labels=Y_train, length=10)
slider_te = Slider(feature=X_test_raw, labels=Y_test, length=10)

x_train_slide, y_train_slide = slider_tr.slider()
x_test_slide, y_test_slide = slider_te.slider()

# train data prep and load
train_tech = x_train_slide[:, :, :-1]  # First 13 columns
train_prob = x_train_slide[:, :, -1:]  # The 14th column (Probability)

test_tech = x_test_slide[:, :, :-1]
test_prob = x_test_slide[:, :, -1:]

# B. Scale ONLY the technical features locally
train_tech_scaled = scale_sequences_locally(train_tech)
test_tech_scaled = scale_sequences_locally(test_tech)

# C. Stitch the untouched probability column back onto the scaled features
x_train_final = np.concatenate((train_tech_scaled, train_prob), axis=2)
x_test_final = np.concatenate((test_tech_scaled, test_prob), axis=2)

x_t, y_t = Data_prep.convertNumpyToTensors(x_train_final, y_train_slide)

# test data pred and load
x_te, y_te = Data_prep.convertNumpyToTensors(x_test_final, y_test_slide)
test_dataset = Data_prep.createTensorDataset(x_te, y_te)
test_data_load = Data_prep.loadData(
    dataset=test_dataset, batch=128, num_worker=0, shuffle=False
)
train_dataset = Data_prep.createTensorDataset(x_t, y_t)
train_data_load = Data_prep.loadData(
    dataset=train_dataset, batch=128, num_worker=0, shuffle=False
)
device = "mps" if (torch.backends.mps.is_available()) else "cpu"

# model
INPUT_SIZE = 14
HIDDEN_UNITS = 64
OUT_FEATURES = 1
model = LSTM_Position_Detector(
    in_size=INPUT_SIZE, hidden_units=HIDDEN_UNITS, out_features=OUT_FEATURES
).to(device)


# loss funtiona and Optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=1e-3)


# training and testing loop
def traintest(
    model: nn.Module,
    device: torch.device,
    Epoch: int,
    optimizer: torch.optim.Adam,
    traindataloader: torch.utils.data.DataLoader,
    testdataloader: torch.utils.data.DataLoader,
    interval: int = 10,
):

    for epoch in range(Epoch):
        model.train()
        train_running_loss = 0.0
        test_running_loss = 0.0

        loop = iter(tqdm(traindataloader, desc=f"Epoch: {epoch + 1}/{Epoch}"))

        for x, y in loop:
            x = x.to(device)
            y = y.to(device).view(-1, 1).float()
            train_pred = model(x)

            train_loss = loss_fn(train_pred, y)
            train_running_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        total_train_loss = train_running_loss / len(traindataloader)

        model.eval()
        with torch.inference_mode():
            for x, y in testdataloader:
                x = x.to(device)
                y = y.to(device)
                test_pred = model(x)

                test_loss = loss_fn(test_pred, y)
                test_running_loss += test_loss.item()

            total_test_loss = test_running_loss / len(testdataloader)

        if epoch % interval == 0:
            print(
                f"|Train Loss: {total_train_loss: 0.8f} | Test Loss: {total_test_loss: 0.8f} |"
            )


# training and testing
EPOCH = 50
INTERVAL = 10
traintest(
    model=model,
    device=torch.device(device),
    Epoch=EPOCH,
    optimizer=optimizer,
    traindataloader=train_data_load,
    testdataloader=test_data_load,
    interval=INTERVAL,
)


def plot_predictions(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader):
    """
    Plots the model's predictions against actual values using stacked vertical subplots.
    """

    model.to("cpu")
    model.eval()

    # 1. Store Predictions and Actuals
    all_preds = []
    all_targets = []

    print("Running inference on test data...")

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to("cpu")
            y = y.to("cpu")

            output = model(x)

            all_preds.append(output.numpy())
            all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    preds_flat = all_preds[:, 0].reshape(-1, 1)
    targets_flat = all_targets[:, 0].reshape(-1, 1)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True)

    ax1.plot(targets_flat, color="blue", label="Actual Percentage Change")
    ax1.set_title("Algorithmic Agent: Position Sizing vs Market Returns")
    ax1.set_ylabel("Market Returns", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        preds_flat,
        color="red",
        label="Predicted Position Size",
        alpha=0.8,
        linestyle="--",
        linewidth=1.5,
    )
    ax2.set_ylim(-1.1, 1.1)  # Lock the agent limits
    ax2.set_ylabel("Agent Position (-1.0 to 1.0)", color="red")
    ax2.set_xlabel("Time (Test Data Points)")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# plot_predictions(model=model, test_loader=test_data_load)

from sklearn.metrics import accuracy_score, f1_score


def evaluate_binary_classifier(
    model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: str
):
    print("\nRunning Classification Evaluation on Unseen Data...")
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).view(-1, 1).float()

            # 1. Get raw model outputs (logits)
            logits = model(x)

            # 2. Squash logits to probabilities (0.0 to 1.0)
            probs = torch.sigmoid(logits)

            # 3. Round to hard binary predictions (0 or 1)
            preds = torch.round(probs)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    # Calculate how often the model predicts "1" (Long) vs "0" (Short)
    buy_ratio = np.mean(all_preds) * 100

    print("-" * 50)
    print("🚀 Sniper Classification Metrics")
    print("-" * 50)
    print(f"✅ Accuracy:       {acc * 100:.2f}%")
    print(f"✅ F1 Score:       {f1:.4f}")
    print(f"📊 Buy Propensity: {buy_ratio:.2f}% (How often it triggers)")
    print("-" * 50)


# Run the evaluation
evaluate_binary_classifier(model, test_data_load, device)

torch.save(
    model.state_dict(),
    "/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/INDBANK2.pt",
)
print("✅ Binary Stage 2 Model Saved!")
