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
    "NIFTY_with_stage1_confidence.csv", index_col="Datetime", parse_dates=True
)
df["Target-Returns"] = df["Returns"].shift(-1)
df.dropna(inplace=True)
feat_cols = [
    "Stage-1-confidence",
    "Returns",
    "Z-score-close",
    "RSI-close-score",
    "MACD-Histogram",
    "Bollinger-Bandwidth",
    "%-Band",
    "Volume-Rate-of-Change",
    "ATR-Ratio",
]

features = df[feat_cols]
features = features.values
labels = df["Target-Returns"]
labels = labels.to_numpy(dtype=float)
train_size = int(0.8 * len(features))

X_train_raw = features[:train_size]
X_test_raw = features[train_size:]
Y_train = labels[:train_size]
Y_test = labels[train_size:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform((X_test_raw))

slidertr = Slider(feature=X_train, labels=Y_train, length=10)

sliderts = Slider(feature=X_test, labels=Y_test, length=10)

x_trainf, y_trainf = slidertr.slider()
x_testf, y_testf = sliderts.slider()

# train data prep and load
x_t, y_t = Data_prep.convertNumpyToTensors(x_trainf, y_trainf)
train_dataset = Data_prep.createTensorDataset(x_t, y_t)
train_data_load = Data_prep.loadData(
    dataset=train_dataset, batch=64, num_worker=0, shuffle=False
)
# test data pred and load
x_te, y_te = Data_prep.convertNumpyToTensors(x_testf, y_testf)
test_dataset = Data_prep.createTensorDataset(x_te, y_te)
test_data_load = Data_prep.loadData(
    dataset=test_dataset, batch=64, num_worker=0, shuffle=False
)

device = "mps" if (torch.backends.mps.is_available()) else "cpu"

# model
INPUT_SIZE = 9
HIDDEN_UNITS = 32
OUT_FEATURES = 1
model = LSTM_Position_Detector(
    in_size=INPUT_SIZE, hidden_units=HIDDEN_UNITS, out_features=OUT_FEATURES
).to(device)


# loss funtiona and Optimizer
loss_fn = SharpeRatio(transaction_cost=0.0002, holding_cost=0.0)
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
            y = y.to(device)
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
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    save_path = project_root / "Output2.png"
    plt.savefig(save_path)


plot_predictions(model=model, test_loader=test_data_load)
CalculatePNL.calculate_pnl(
    model=model, test_loader=test_data_load, transaction_cost=0.0002
)
