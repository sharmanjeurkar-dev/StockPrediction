import os
import time
from datetime import datetime, timedelta

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


fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID, token=get_token(), is_async=False, log_path=""
)


def scrape_data(
    symbol: str,
    DAYS=100,
    resolution: str = "15",
):
    all_data = []

    # End date is today
    end_date = datetime.now()

    # We will do two chunks of 90 days (Total = 180 days / 6 months)
    for i in range(2):
        # Calculate the chunk's start and end dates
        chunk_end = end_date - timedelta(days=(i * DAYS))
        chunk_start = chunk_end - timedelta(days=DAYS)

        print(
            f"Fetching chunk {i + 1}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
        )

        data = {
            "symbol": symbol,
            "resolution": "15",  # 15-minute timeframe
            "date_format": "1",
            "range_from": chunk_start.strftime("%Y-%m-%d"),
            "range_to": chunk_end.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }

        # Call the FYERS API
        response = fyers.history(data=data)

        if "candles" in response and response["candles"]:
            # Create a dataframe for this chunk
            df_chunk = pd.DataFrame(
                response["candles"],
                columns=["Datetime", "Open", "High", "Low", "Close", "Volume"],
            )
            all_data.append(df_chunk)
        else:
            print("Error or no data in this chunk:", response)

        # Sleep for 1 second to avoid API rate limits
        time.sleep(1)

    # Combine all the chunks together
    final_df = pd.concat(all_data, ignore_index=True)

    # Convert Unix timestamps to human-readable datetime
    final_df["Datetime"] = pd.to_datetime(final_df["Datetime"], unit="s")

    # Sort from oldest to newest
    final_df = final_df.sort_values(by="Datetime")

    # Set Datetime as index
    final_df.set_index("Datetime", inplace=True)

    print(f"✅ Success! Downloaded {len(final_df)} rows of data.")
    return final_df
