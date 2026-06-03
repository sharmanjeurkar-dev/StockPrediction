from torch import nn


class LSTM(nn.Module):
    def __init__(self, in_size, hidden_units, out_features):
        super().__init__()
        self.lstm_layer = nn.LSTM(
            input_size=in_size, hidden_size=hidden_units, batch_first=True
        )

        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=hidden_units, out_features=hidden_units // 2)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_units // 2, out_features=out_features)

        # adding the squashing layer which will gently squash my output in between -1.0 to 1.0 so that gradients dont die
        self.position_activation = nn.Softsign()

    def forward(self, x):
        output, _ = self.lstm_layer(x)
        x = output[:, -1, :]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        position = self.position_activation(x)

        return position
