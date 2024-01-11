
import yfinance as yf

def GatherData(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.head()
    stock_data.reset_index(drop=False,inplace=True)
    return stock_data
    