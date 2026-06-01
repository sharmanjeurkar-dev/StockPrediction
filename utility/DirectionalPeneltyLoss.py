import torch
from torch import nn


class DirectionalLoss(nn.Module):
    def __init__(self, weights=5):
        super().__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        a = torch.sign(predictions)
        b = torch.sign(targets)
        loss = torch.subtract(predictions, targets)
        loss = torch.abs(loss)
        weighted_loss = torch.where(a == b, loss * 1, loss * 5)
        final_loss = torch.mean(weighted_loss)

        return final_loss
