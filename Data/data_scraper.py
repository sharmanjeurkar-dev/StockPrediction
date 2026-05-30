import os
from datetime import date, timedelta

import pandas as pd
from fyers_apiv3 import fyersModel

CLIENT_ID = "CGVLNFTR73-100"


def get_token():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.join(base_dir, "access_token.env")

    try:
        with open(token_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise Exception(
            f"Authentication failed: Could not find the file at {token_path}"
        )


fyres = fyersModel.FyersModel(
    client_id=CLIENT_ID, token=get_token(), is_async=False, log_path=""
)


def scrape_data(
    symbol: str,
    DAYS=100,
    resolution: str = "2",
):
    today = date.today()

    START = today - timedelta(days=DAYS)
    END = today

    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": START.strftime("%Y-%m-%d"),
        "range_to": END.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }

    try:
        response = fyres.history(data=data)
        if response.get("s") != "ok":
            raise Exception(
                f"Fyers API Error for {symbol}: {response.get('message', 'Unknown Error')}"
            )

        columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
        df = pd.DataFrame(response["candles"], columns=columns)

        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
        df["Datetime"] = (
            df["Datetime"]
            .dt.tz_localize("UTC")
            .dt.tz_convert("Asia/Kolkata")
            .dt.tz_localize(None)
        )
        df.set_index("Datetime", inplace=True)

        return df

    except Exception as e:
        print("Error fetching data from Fyers API:", e)
        return pd.DataFrame()
        # Return an empty DataFrame on error
