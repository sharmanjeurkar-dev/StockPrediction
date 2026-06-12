from torch import nn


class LSTM_Market_Direction(nn.Module):
    def __init__(self, in_size, hidden_units, out_feautures):
        super().__init__()
        self.lstm_layer = nn.LSTM(
            input_size=in_size,
            num_layers=2,
            dropout=0.2,
            hidden_size=hidden_units,
            batch_first=True,
        )
        self.Dropout = nn.Dropout(p=0.2)
        self.linear_layer = nn.Linear(
            in_features=hidden_units, out_features=out_feautures
        )

    def forward(self, x):
        output, _ = self.lstm_layer(x)
        x = output[:, -1, :]
        x = self.Dropout(x)
        x = self.linear_layer(x)

        return x
