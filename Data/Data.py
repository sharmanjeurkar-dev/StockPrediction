import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scrape_data():

    TICKER = ['^NSEI']
    START = '2007-09-17'
    END = '2026-02-02'

    df = yf.download(tickers = TICKER,
                    start = START,
                    end = END)

    

    feat_cols = ['Open','High','Low','Volume','Close']
    feartures = df[feat_cols]
    feartures = feartures.values

    label_col = ['Close']
    label = df[label_col]
    label = label.values
    print(df)


    train_size = int(len(feartures)*0.8)
    X_train = feartures[:train_size]
    X_test = feartures[train_size:]
    Y_train = label[:train_size]
    Y_test = label[train_size:]
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)


    process_feat = MinMaxScaler(feature_range=(0,1))
    process_targ = MinMaxScaler(feature_range=(0,1))


    X_train = process_feat.fit_transform(X_train)
    X_test = process_feat.fit_transform(X_test)

    Y_train = process_targ.fit_transform(Y_train)
    Y_test = process_targ.fit_transform(Y_test)

    print(X_train[0],X_test[0],Y_train[0],Y_test[0])
   

    return df


scrape_data()

