import os
import pickle
import sys

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from scipy.stats import spearmanr
from torch.utils.data import WeightedRandomSampler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Data.Data_preprocessing import concat_df
from Data.data_scraper import scrape_data

df1 = scrape_data("NSE:PAYTM-EQ")  # paytm
df2 = scrape_data("NSE:TCS-EQ")  # tcs
df3 = scrape_data("NSE:HINDUNILVR-EQ")  # Hindustan Unilever

df_raw = [df1, df2, df3]
df = concat_df(df_raw)

df = df.sort_values(by=["symbol", "Datetime"])
df["timeidx"] = df.groupby("symbol").cumcount()
df = df.reset_index()

float_cols = [
    "MA-Cross",
    "Bollinger-Bandwidth",
    "ATR-Ratio",
    "OBV-ROC",
    "%-Band",
    "Volume-Rate-of-Change",
    "RSI-close-score",
    "Intraday_Spread",
    "Time_Sin",
    "Time_Cos",
    "Target",
]
df[float_cols] = df[float_cols].astype(np.float32)

train_data_cutoff = int(df["timeidx"].max() * 0.80)

train_df = df[df["timeidx"] <= train_data_cutoff].copy()
train_df = train_df.reset_index(drop=True)

pos_count = (train_df["Target"] > 0).sum()
neg_count = (train_df["Target"] < 0).sum()
zero_count = (train_df["Target"] == 0).sum()

weights_full = np.where(
    train_df["Target"] > 0,
    1.0 / pos_count,
    np.where(train_df["Target"] < 0, 1.0 / neg_count, 1.0 / zero_count),
)

# timeseries dataset
training = TimeSeriesDataSet(
    data=df[df["timeidx"] <= train_data_cutoff],
    time_idx="timeidx",
    target="Target",
    group_ids=["symbol"],
    max_encoder_length=32,
    max_prediction_length=1,
    time_varying_known_reals=["Time_Sin", "Time_Cos"],
    time_varying_unknown_reals=[
        "MA-Cross",
        "Bollinger-Bandwidth",
        "ATR-Ratio",
        "OBV-ROC",
        "%-Band",
        "Volume-Rate-of-Change",
        "RSI-close-score",
        "Intraday_Spread",
    ],
    static_categoricals=["symbol"],
    target_normalizer=TorchNormalizer(method="robust"),
    allow_missing_timesteps=True,
)

dataset_indices = training.index["index_start"].values
weights_aligned = weights_full[dataset_indices]

sampler = WeightedRandomSampler(
    weights=weights_aligned, num_samples=len(training), replacement=True
)

validation = training.from_dataset(
    training,
    df[df["timeidx"] > train_data_cutoff],
    stop_randomization=True,
)

train_dataloader = training.to_dataloader(
    train=False, batch_size=128, num_workers=0, sampler=sampler
)
validation_dataloader = validation.to_dataloader(
    train=False, batch_size=128, num_workers=0
)

model = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    learning_rate=3e-4,
    optimizer="adam",
    reduce_on_plateau_patience=5,
)

# training and validation
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="mps",
    gradient_clip_val=0.1,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
)
trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=validation_dataloader,
)
predictions = model.predict(validation_dataloader, return_y=True)
pred_values = predictions.output.cpu().numpy().flatten()
actual_values = predictions.y[0].cpu().numpy().flatten()

ic, pvalue = spearmanr(pred_values, actual_values)
print(f"TFT Validation IC: {ic:.4f}")
print(f"P-value: {pvalue:.4f}")
print("LightGBM baseline IC: 0.1651")
print(f"Improvement: {ic - 0.1651:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(pred_values, actual_values, alpha=0.1, s=5, color="steelblue")
plt.xlabel("Predicted Return")
plt.ylabel("Actual Return")
plt.title("TFT: Predicted vs Actual Return")
plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
plt.show()

raw_predictions = model.predict(validation_dataloader, mode="raw", return_x=True)

interpretation = model.interpret_output(raw_predictions.output, reduction="sum")
model.plot_interpretation(interpretation)
plt.tight_layout()
plt.savefig("tft_interpretation.png", dpi=150)
plt.show()

raw_preds = raw_predictions.output.prediction.cpu().numpy()
p10 = raw_preds[:, 0, 0]
p50 = raw_preds[:, 0, 1]
p90 = raw_preds[:, 0, 2]

within_band = np.mean((actual_values >= p10) & (actual_values <= p90))
print(f"Quantile coverage (should be ~80%): {within_band * 100:.1f}%")

plt.figure(figsize=(12, 5))
sample_idx = np.arange(min(200, len(actual_values)))
plt.fill_between(
    sample_idx, p10[:200], p90[:200], alpha=0.3, color="steelblue", label="p10-p90 band"
)
plt.plot(sample_idx, p50[:200], color="steelblue", linewidth=1, label="p50 prediction")
plt.plot(sample_idx, actual_values[:200], color="coral", linewidth=1, label="actual")
plt.xlabel("Sample index")
plt.ylabel("Return")
plt.title("TFT: Quantile predictions vs actual (first 200 samples)")
plt.legend()
plt.tight_layout()
plt.savefig("quantile_coverage.png", dpi=150)
plt.show()

trainer.save_checkpoint("tft_model.ckpt")
pickle.dump(training, open("training_dataset.pkl", "wb"))
print("Model and dataset saved successfully")
