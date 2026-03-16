import yfinance as yf
from datetime import date,timedelta

def scrape_data():

    today = date.today()

    TICKER = ['^NSEI']
    START = str(today - timedelta(days=59))
    END = str(today)
    INTERVAL = '2m'

    df = yf.download(tickers = TICKER,
                    start = START,
                    end = END,
                    interval=INTERVAL)


    return df