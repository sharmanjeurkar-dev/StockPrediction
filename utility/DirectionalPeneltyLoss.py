import torch
from torch import nn


class DirectionaMagnitudelLoss(nn.Module):
    def __init__(self, dir_weights=5.0, mag_scaler=1000.0):
        super().__init__()
        self.dir_weights = dir_weights
        self.mag_scaler = mag_scaler

    def forward(self, predictions, targets):
        a = torch.sign(predictions)
        b = torch.sign(targets)
        basic_loss = torch.subtract(predictions, targets)
        basic_loss = torch.abs(basic_loss)

        directional_penelty = torch.where(a == b, 1.0, self.dir_weights)
        magnitude_penelty = 1.0 + (torch.abs(targets) * self.mag_scaler)
        final_loss = torch.mean(directional_penelty * magnitude_penelty * basic_loss)

        return final_loss
