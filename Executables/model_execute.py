import os
import pickle
from datetime import datetime, timedelta

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from scipy.stats import spearmanr
from torch.utils.data import WeightedRandomSampler

from Data.feature_engineering import (
    HISTORICAL_DATA_PATH,
    build_training_set,
    walk_forward_out_of_sample_dataframe_slices,
)
from Data.historical_data.historical_data_scraper import (
    save_benchmark_data,
    save_histortrical_data,
)

TARGET = "Target"
GROUP = "symbol"
TIMEIDX = "timeidx"
DATE_COL = "date"
MAX_ENCODER_LENGTH = 32
MAX_PRED_LENGTH = 1
EMBARGO_HOURS = 2
RESOLUTION = "1D"
DAYS = 365
TOTAL_CHUNKS = 5
END_DATE = TODAY = datetime.now()

# Scan the univrse, screen the right symbols, download the historical data for timedelta greater than one day
all_items = os.listdir(HISTORICAL_DATA_PATH)
if all_items:
    first_item_path = os.path.join(HISTORICAL_DATA_PATH, all_items[0])
    if os.path.isfile(first_item_path):
        print("Historical data files found found")
        raw_time = os.path.getctime(first_item_path)
        file_creation_date = datetime.fromtimestamp(raw_time).date()

        target_date = TODAY.date()

        time_difference = target_date - file_creation_date
        print(f"Time difference in creation and today:{time_difference}")
        if time_difference > timedelta(2):
            print("Data stale....\nDownloading fresh data......")
            failed_symbols = save_histortrical_data(
                resolution=RESOLUTION,
                DAYS=DAYS,
                total_chunks=TOTAL_CHUNKS,
                end_date=END_DATE,
            )

            if failed_symbols is not None:
                print(len(failed_symbols))
                for fs in failed_symbols:
                    print(fs, "\t")

            # downloading/Updating the benchmark index historical data
            _ = save_benchmark_data(
                resolution=RESOLUTION,
                DAYS=DAYS,
                total_chunks=TOTAL_CHUNKS,
                end_date=END_DATE,
            )

# accessing the benchmark_data file
benchmark_df = pd.read_parquet("Data/historical_data/data/NSE_NIFTY50-INDEX.parquet")

# build the training dataset -> combine the index data as well all the symbol data, feuture engineer, labeling and target formation
df, _ = build_training_set(snapshot_date=datetime.strftime(TODAY, "%Y-%m-%d"))
print(f"\t \t {len(df)} \t \t")
df = df.sort_values(by=["symbol", "Datetime"])

features = [
    "MA-Cross",
    "ATR-Ratio",       
    "OBV-ROC",
    "Volume-Rate-of-Change",
    "RSI-close-score",
    "Intraday_Spread",
    "Target",
    "MA-200",
    "VWAP-20D-Dist",
    "relative_strength",
]

df[features] = df[features].astype(np.float32)


def rebuild_time_idx(df, symbol_col="symbol"):
    df = df.sort_values([symbol_col, "Datetime"])
    df["timeidx"] = df.groupby(symbol_col).cumcount()
    return df


def build_dataset(train_df, test_df):
    training = TimeSeriesDataSet(
        train_df,
        time_idx="timeidx",
        target="Target",
        group_ids=["symbol"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PRED_LENGTH,
        time_varying_known_reals=[],
        time_varying_unknown_reals=[
            "MA-Cross",
            "ATR-Ratio",
            "OBV-ROC",
            "Volume-Rate-of-Change",
            "RSI-close-score",
            "Intraday_Spread",
        ],
        target_normalizer=TorchNormalizer(method="robust"),
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
    )

    validation = training.from_dataset(
        training,
        test_df,
        stop_randomization=True,
    )

    dataset_indices = training.index["index_start"].values
    target_tensor = training.data["target"][0]
    target_values = target_tensor[dataset_indices].numpy().reshape(-1)

    labels = (target_values > 0).astype(int)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    weights_aligned = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights_aligned),
        num_samples=len(training),
        replacement=True,
    )

    train_dataloader = training.to_dataloader(
        train=False, batch_size=256, num_workers=0, sampler=sampler
    )
    validation_dataloader = validation.to_dataloader(
        train=False, batch_size=256, num_workers=0
    )
    print("class_counts:", class_counts)
    return training, validation, train_dataloader, validation_dataloader


def model_and_trainer_setup(training: TimeSeriesDataSet):
    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        learning_rate=3e-4,
        optimizer="adam",
        reduce_on_plateau_patience=5,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=15, mode="min")
    # training and validation
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu",
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    return model, trainer


def compute_fold_metrics(model, val_dl, validation_dataset, test_df, fold_id):
    raw_preds = model.predict(val_dl, mode="quantiles", return_index=True)
    preds = raw_preds.output
    index_df = raw_preds.index
    quantiles = model.loss.quantiles
    p10_idx = quantiles.index(min(quantiles, key=lambda q: abs(q - 0.1)))
    p50_idx = quantiles.index(min(quantiles, key=lambda q: abs(q - 0.5)))
    p90_idx = quantiles.index(min(quantiles, key=lambda q: abs(q - 0.9)))

    preds = preds.squeeze(1).numpy()
    merged = index_df.copy()
    merged["pred_p10"] = preds[:, p10_idx]
    merged["pred_p50"] = preds[:, p50_idx]
    merged["pred_p90"] = preds[:, p90_idx]

    actual_lookup = test_df.set_index([GROUP, TIMEIDX])[TARGET]
    merged["actual"] = merged.apply(
        lambda r: actual_lookup.get((r[GROUP], r[TIMEIDX] + MAX_PRED_LENGTH), np.nan),
        axis=1,
    )
    merged = merged.dropna(subset=["actual"])

    ic, pval = spearmanr(merged["pred_p50"], merged["actual"])
    coverage = (
        (merged["actual"] >= merged["pred_p10"])
        & (merged["actual"] <= merged["pred_p90"])
    ).mean()

    per_symbol_ic = {}
    for sym, g in merged.groupby(GROUP):
        if len(g) > 5:
            sym_ic, _ = spearmanr(g["pred_p50"], g["actual"])
            per_symbol_ic[sym] = sym_ic

    return {
        "fold": fold_id,
        "ic": ic,
        "pvalue": pval,
        "coverage": coverage,
        "n_test_rows": len(merged),
        **{f"ic_{k}": v for k, v in per_symbol_ic.items()},
    }


def plot_model_output(model, validation_dataloader):
    predictions = model.predict(validation_dataloader, return_y=True)
    pred_values = predictions.output.cpu().numpy().flatten()
    actual_values = predictions.y[0].cpu().numpy().flatten()

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
        sample_idx,
        p10[:200],
        p90[:200],
        alpha=0.3,
        color="steelblue",
        label="p10-p90 band",
    )
    plt.plot(
        sample_idx, p50[:200], color="steelblue", linewidth=1, label="p50 prediction"
    )
    plt.plot(
        sample_idx, actual_values[:200], color="coral", linewidth=1, label="actual"
    )
    plt.xlabel("Sample index")
    plt.ylabel("Return")
    plt.title("TFT: Quantile predictions vs actual (first 200 samples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("quantile_coverage.png", dpi=150)
    plt.show()


# WalkForward Slicing for training and Out of sample Validation
df_acc_folds = walk_forward_out_of_sample_dataframe_slices(df=df)

results = []
for i, (train_df, test_df) in enumerate(df_acc_folds):
    train_df = rebuild_time_idx(train_df, symbol_col="symbol")
    train_df = train_df.reset_index(drop=True)
    test_df = rebuild_time_idx(test_df, symbol_col="symbol")
    test_df = test_df.reset_index(drop=True)

    training, validation, train_dataloader, validation_dataloader = build_dataset(
        train_df=train_df, test_df=test_df
    )
    model, trainer = model_and_trainer_setup(training=training)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )
    metrics = compute_fold_metrics(
        model=model,
        val_dl=validation_dataloader,
        validation_dataset=validation,
        test_df=test_df,
        fold_id=i,
    )
    results.append(metrics)
    pd.DataFrame(results).to_csv("walk_forward_out_of_sample_results.csv", index=False)
    trainer.save_checkpoint(
        f"/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/model_fold_{i}"
    )
    print("\n\n\n")
    print("*" * 50)
    print(metrics)
    print("*" * 50)
    print("\n\n\n")

results_df = pd.DataFrame(results)

print("-" * 50)
print("\n\n")
print(results_df)
print("Mean IC:", results_df["ic"].mean(), "Std IC:", results_df["ic"].std())
print("\n\n")
print("-" * 50)
trainer.save_checkpoint(
    "/Users/sharmanjeurkar/Projects/SequenceAlpha/models/saved/tft_model.ckpt"
)

plot_model_output(model=model, validation_dataloader=validation_dataloader)
pickle.dump(training, open("training_dataset.pkl", "wb"))
print("Model and dataset saved successfully")
