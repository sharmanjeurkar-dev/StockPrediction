import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Data import data_scraper


def feature_enginiering():

    df = data_scraper.scrape_data("NSE:NIFTY50-INDEX")

    # Calculating the %returns on close price if the share was brought
    df["Returns"] = df["Close"].pct_change()

    # z -score
    df["Z-score-close"] = (df["Close"] - df["Close"].rolling(window=20).mean()) / df[
        "Close"
    ].rolling(window=20).std()

    # RSI score
    df["Change"] = df["Close"].diff()
    df["Gain"] = np.where(df["Change"] > 0, df["Change"], 0)
    df["Loss"] = np.where(df["Change"] < 0, abs(df["Change"]), 0)

    df["Avg_Gain"] = df["Gain"].rolling(window=14).mean()
    df["Avg_Loss"] = df["Loss"].rolling(window=14).mean()

    df["Rs"] = df["Avg_Gain"] / df["Avg_Loss"]
    df["RSI-close-score"] = 100 - 100 / (1 + df["Rs"])

    # EMA Sscores for different windows
    df["EMA-Today-close-26D"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["EMA-Today-close-12D"] = df["Close"].ewm(span=12, adjust=False).mean()

    # MACD Scores
    df["MACD-Line"] = df["EMA-Today-close-12D"] - df["EMA-Today-close-26D"]
    df["Single-Line"] = df["MACD-Line"].ewm(span=9, adjust=False).mean()
    df["MACD-Histogram"] = df["MACD-Line"] - df["Single-Line"]

    df["Target"] = df["Returns"].shift(-1)
    df.dropna(inplace=True)

    feat_cols = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Returns",
        "Z-score-close",
        "RSI-close-score",
        "MACD-Line",
        "Single-Line",
        "MACD-Histogram",
    ]
    feartures = df[feat_cols]
    feartures = feartures.values

    label_col = ["Target"]
    label = df[label_col]
    label = label.values
    print(df)

    train_size = int(len(feartures) * 0.8)
    X_train = feartures[:train_size]
    X_test = feartures[train_size:]
    Y_train = label[:train_size]
    Y_test = label[train_size:]
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    process_feat = MinMaxScaler(feature_range=(0, 1))
    process_targ = MinMaxScaler(feature_range=(0, 1))

    X_train = process_feat.fit_transform(X_train)
    X_test = process_feat.transform(X_test)

    Y_train = process_targ.fit_transform(Y_train)
    Y_test = process_targ.transform(Y_test)

    print(X_train[0], X_test[0], Y_train[0], Y_test[0])

    return X_train, X_test, Y_train, Y_test, process_targ


feature_enginiering()
