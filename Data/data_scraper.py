import os
import pandas as pd
from datetime import date,timedelta
from fyers_apiv3 import fyersModel


CLIENT_ID = "CGVLNFTR73-100"


def get_tocken():
    tocken_path = os.path.join('/home/sharmanjeurkar/PycharmProjects/StockPrediction/auth','access_token.txt')

    try:
        with open(tocken_path,'r') as f:
            return f.read().strip()
    except:
        raise Exception("Could not open token file")
        

def scrape_data(SYMBOL = 'NSE:NIFTY50-INDEX', DAYS = 100,INTERVAL = '2'):

    access_tocken = get_tocken()
    fyres = fyersModel.FyersModel(is_async = False,log_path = "",client_id=CLIENT_ID,token = access_tocken)

    today = date.today()

    START = today - timedelta(days=DAYS)
    END = today

    DATA_PAYLOAD = {
        "symbol":SYMBOL,
        "resolution": INTERVAL,
        "date_format":"1",
        "range_from":START.strftime('%Y-%m-%d'),
        "range_to":END.strftime('%Y-%m-%d'),
        "cont_flag":"1"
    }

    print(f'Fetching data for every {INTERVAL}m from {START} to {END} \n')
    response = fyres.history(data=DATA_PAYLOAD)

    if(response['s']!='ok'):
       raise Exception(f"Fyers API Error: {response.get('message', 'Unknown Error')}")
    
    coloumns = ['Datetime','Open','High','Low','Close','Volume']
    df = pd.DataFrame(response['candles'],columns=coloumns)
    df['Datetime'] = pd.to_datetime(df['Datetime'],unit='s')
    df.set_index('Datetime',inplace = True)
    print(df.head(10))
    print(df.shape)

    return df