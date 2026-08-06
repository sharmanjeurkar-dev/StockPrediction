import pandas as pd

from Data.broker_data.Fyersbrocker import FyersBrocker


def screener(
    symbols: list[str],
    as_of_date: str,
    lookback_days: int,
    min_avg_daily_value: float,
    min_trading_days: int = 50,
):
    broker = FyersBrocker()
    failed_symbols = []
    failure_reason = []
    successfull_symbols = []
    avg_traded_value_list = []
    actual_trading_days_list = []
    lookback_days_list = []

    for symbol in symbols:
        df = broker.get_recent_history(
            symbol=symbol, as_of_date=as_of_date, lookback_days=lookback_days
        )

        if df is None:
            failed_symbols.append(symbol)
            failure_reason.append("Couldn't load data")
            continue

        avg_traded_value = (df["Close"] * df["Volume"]).mean()
        actual_trading_days = len(df)

        if actual_trading_days < min_trading_days:
            failed_symbols.append(symbol)
            failure_reason.append(
                f"Trading days: {actual_trading_days} less than the minimum threshold by: {min_trading_days - actual_trading_days}"
            )
            continue

        if avg_traded_value < min_avg_daily_value:
            failed_symbols.append(symbol)
            failure_reason.append(
                f"Avg daily value: {avg_traded_value} less than the threshold by: {min_avg_daily_value - avg_traded_value}"
            )
            continue

    successfull_symbols.append(symbol)
    avg_traded_value_list.append(avg_traded_value)
    actual_trading_days_list.append(actual_trading_days)
    lookback_days_list.append(lookback_days)

    snapshot = pd.DataFrame(
        {
            "symbols": successfull_symbols,
            "avg_traded_value": avg_traded_value_list,
            "actual_trading_days": actual_trading_days_list,
            "look_back_window": lookback_days_list,
        }
    )
    return snapshot, pd.DataFrame({"symbol": failed_symbols, "reason": failure_reason})
