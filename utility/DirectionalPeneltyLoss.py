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


class SharpeRatio(nn.Module):
    def __init__(
        self, risk_free_rate=0.0, holding_cost=0.0005, transaction_cost=0.00002
    ):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.holding_cost = holding_cost
        self.transaction_cost = transaction_cost

    def forward(self, predicted_position, returns):
        pnl = predicted_position * returns
        position_change = torch.diff(
            predicted_position,
            dim=0,
            prepend=torch.zeros(1, 1, device=predicted_position.device),
        )
        transaction_fee = torch.abs(position_change) * self.transaction_cost
        holding_fee = torch.abs(predicted_position) * self.holding_cost
        net_pnl = pnl - holding_fee - transaction_fee

        mean_net_pnl = torch.mean(net_pnl)
        std_net_pnl = torch.std(net_pnl)
        epsilon = 1e-6

        annualized_mean = 252 * mean_net_pnl
        annualized_std = std_net_pnl * 252**0.5

        sharpe_ratio = (annualized_mean - self.risk_free_rate) / (
            annualized_std + epsilon
        )

        return -sharpe_ratio
