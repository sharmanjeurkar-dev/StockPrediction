from torch import nn

class LSTM(nn.Module):
    
    def __init__(self,in_size,hidden_units,out_features):
        super().__init__()
        self.lstm_layer = nn.LSTM(input_size=in_size,
                                  hidden_size=hidden_units,
                                  batch_first=True)
        self.linear = nn.Linear(in_features=hidden_units,
                                out_features=out_features)
        
    def forward(self,x):
        _,(hn,cn) = self.lstm_layer(x)
        x = hn[0]
        x = self.linear(x)

        return x