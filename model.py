import yfinance as yf
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fetch_stock_data(stock_symbol, period='1y'):
    """
    Fetch historical stock data for the given stock symbol.
    
    """
    stock_data = yf.download(stock_symbol, period=period)
    stock_data.reset_index(inplace=True)
    return stock_data

def forecast_stock_prices(stock_data, forecast_days):
    """
    Forecast future stock prices using the Holt-Winters Exponential Smoothing model.
    
    """
    model = ExponentialSmoothing(stock_data['Close'], trend='add', seasonal=None)
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=stock_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    
    return forecast, forecast_index