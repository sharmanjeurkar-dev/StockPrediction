from datetime import datetime

import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from Data.feature_engineering import (
    build_training_set,
    walk_forward_out_of_sample_dataframe_slices,
)
from Data.historical_data.historical_data_scraper import (
    save_benchmark_data,
    save_histortrical_data,
)

fold_accuracies = []
fold_ics = []

RESOLUTION = "1D"
DAYS = 365
TOTAL_CHUNKS = 5
END_DATE = TODAY = datetime.now()

failed_symbols = save_histortrical_data(
    resolution=RESOLUTION, DAYS=DAYS, total_chunks=TOTAL_CHUNKS, end_date=END_DATE
)

if failed_symbols is not None:
    for fs in failed_symbols:
        print(fs)

_ = save_benchmark_data(
    resolution=RESOLUTION, DAYS=DAYS, total_chunks=TOTAL_CHUNKS, end_date=END_DATE
)
benchmark_df = pd.read_parquet("Data/historical_data/data/NSE_NIFTY50-INDEX.parquet")

universal_df, failure_df = build_training_set(
    snapshot_date=datetime.strftime(TODAY, "%Y-%m-%d")
)

universal_df["Target_binary"] = (universal_df["Target"] > 0).astype(int)

df_list = walk_forward_out_of_sample_dataframe_slices(
    df=universal_df, jump=3, max_days=8
)

feature_columns = [
    "RSI-close-score",
    "MA-Cross",
    "MA-200",
    "relative_strength",
    "OBV-ROC",
    "VWAP-20D-Dist",
    "ATR-Ratio",
    "Intraday_Spread",
    "Volume-Rate-of-Change",
]


model = LogisticRegression(max_iter=1000)

predictions = []
y_tests = []


for fold_num, (train_df, test_df) in enumerate(df_list):
    X_train = train_df[feature_columns]
    y_train = train_df["Target_binary"]
    X_test = test_df[feature_columns]
    y_test = test_df["Target_binary"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predicted_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1
    predicted_class = model.predict(X_test)

    acc = accuracy_score(y_test, predicted_class)
    fold_accuracies.append(acc)

    # IC: rank correlation between predicted probability and the ACTUAL
    # continuous return (Target), not the binary label -- this tells you
    # whether higher predicted confidence actually corresponds to larger
    # real forward returns, which accuracy alone can't show.
    ic, p_value = spearmanr(predicted_proba, test_df["Target"])
    fold_ics.append(ic)

    print(f"Fold {fold_num}: accuracy={acc:.3f}, IC={ic:.4f} (p={p_value:.3f})")

print(
    f"\nMean accuracy across folds: {sum(fold_accuracies) / len(fold_accuracies):.3f}"
)
print(f"Mean IC across folds: {sum(fold_ics) / len(fold_ics):.4f}")
print(f"IC std across folds: {pd.Series(fold_ics).std():.4f}")

"""
Result of baseline:
ll sets before the end date covered
/Users/sharmanjeurkar/Projects/SequenceAlpha/venv/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:599: ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=1000).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Fold 0: accuracy=0.478, IC=0.0255 (p=0.000)
/Users/sharmanjeurkar/Projects/SequenceAlpha/venv/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:599: ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=1000).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Fold 1: accuracy=0.634, IC=-0.0037 (p=0.482)
Fold 2: accuracy=0.448, IC=-0.0319 (p=0.000)
Fold 3: accuracy=0.524, IC=0.0037 (p=0.474)
Fold 4: accuracy=0.500, IC=-0.0101 (p=0.052)
Fold 5: accuracy=0.580, IC=-0.0168 (p=0.001)
Fold 6: accuracy=0.525, IC=0.0197 (p=0.000)
Fold 7: accuracy=0.577, IC=-0.0204 (p=0.000)
Fold 8: accuracy=0.618, IC=-0.0469 (p=0.000)
Fold 9: accuracy=0.650, IC=-0.1380 (p=0.000)
Fold 10: accuracy=0.531, IC=-0.0251 (p=0.000)
Fold 11: accuracy=0.602, IC=-0.0730 (p=0.000)
Fold 12: accuracy=0.623, IC=-0.0359 (p=0.000)
Fold 13: accuracy=0.679, IC=0.0003 (p=0.947)
Fold 14: accuracy=0.506, IC=-0.0727 (p=0.000)
Fold 15: accuracy=0.683, IC=nan (p=nan)

Mean accuracy across folds: 0.572
Mean IC across folds: nan
IC std across folds: 0.0419
"""
