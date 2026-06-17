import torch
import torch.nn.functional as F
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
        self.attention = nn.Linear(hidden_units, 1)

        self.fc1 = nn.Linear(hidden_units, 32)
        self.relu = nn.ReLU()
        self.out = nn.Linear(32, out_feautures)

    def forward(self, x):
        output, _ = self.lstm_layer(x)
        attn_weights_raw = self.attention(output)
        attn_weights = F.softmax(attn_weights_raw, dim=1)

        context_weight = torch.sum(attn_weights * output, dim=1)

        x = self.fc1(context_weight)
        x = self.relu(x)
        x = self.out(x)

        return x
