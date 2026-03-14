import yfinance as yf

def scrape_data():

    TICKER = ['^NSEI']
    START = '2007-09-17'
    END = '2026-02-02'

    df = yf.download(tickers = TICKER,
                    start = START,
                    end = END)


    return df