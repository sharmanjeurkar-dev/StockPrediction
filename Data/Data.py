import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


TICKER = 'Goldbees.ns'
START = '2009-01-02'
END = '2026-02-02'

data = yf.download(tickers = TICKER,
                   start = START,
                   end = END)

print(data)

df = pd.DataFrame(data)
df.set_index

feat_cols = ['Open','High','Low','Volume','Close']
feartures = df[feat_cols]
feartures = feartures.values

label_col = ['Close']
label = df[label_col]
label = label.values


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