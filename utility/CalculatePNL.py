import numpy as np
import torch


def calculate_pnl(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    initial_capital=100000,
    transaction_cost=0.0002,
):

    model.to("cpu")
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to("cpu").to(torch.float32)
            y = y.to("cpu").to(torch.float32)

            output = model(x)

            all_preds.append(output.numpy())
            all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    gross_returns = all_preds * all_targets

    position_changes = np.diff(all_preds, prepend=0.0)
    transaction_fees = np.abs(position_changes) * transaction_cost

    net_returns = gross_returns - transaction_fees

    gross_pnl_rupees = np.sum(gross_returns) * initial_capital
    total_fees_rupees = np.sum(transaction_fees) * initial_capital
    net_pnl_rupees = gross_pnl_rupees - total_fees_rupees

    total_turnover = np.sum(np.abs(position_changes))
    approx_round_trip_trades = (
        total_turnover / 2.0
    )  # 1 buy + 1 sell = 1 round trip trade

    active_minutes = np.sum(np.abs(all_preds) > 0.05)
    winning_minutes = np.sum(net_returns > 0)
    win_rate = (winning_minutes / active_minutes * 100) if active_minutes > 0 else 0

    print("=" * 45)
    print("🚀 QUANTITATIVE BACKTEST REPORT")
    print("=" * 45)
    print(f"Initial Capital:      ₹{initial_capital:,.2f}")
    print(f"Gross Profit:         ₹{gross_pnl_rupees:,.2f}")
    print(f"Brokerage & Fees:   - ₹{total_fees_rupees:,.2f}")
    print("-" * 45)
    print(f"Net Profit (Real):    ₹{net_pnl_rupees:,.2f}")
    print(f"Final Account Value:  ₹{(initial_capital + net_pnl_rupees):,.2f}")
    print(f"Total Net Return:     {((net_pnl_rupees) / initial_capital) * 100:.2f}%")
    print("-" * 45)
    print(f"Approximate Trades:   {approx_round_trip_trades:,.1f}")
    print(f"Win Rate (Minutes):   {win_rate:.2f}%")
    print("=" * 45)

    return net_returns
