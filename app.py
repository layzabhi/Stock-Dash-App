import dash
from dash import dcc, html, Input, Output
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Layout of the app
app.layout = html.Div([
    html.H1("Stock Visualization and Forecasting"),
    
    dcc.Input(id='stock-input', type='text', placeholder='Enter Stock Symbol (e.g., AAPL)'),
    html.Button('Get Stock Data', id='get-data-button', n_clicks=0),
    
    dcc.Graph(id='stock-graph'),
    
    dcc.Input(id='forecast-days', type='number', placeholder='Number of Days to Forecast'),
    html.Button('Forecast', id='forecast-button', n_clicks=0),
    
    dcc.Graph(id='forecast-graph')
])

# Callback to update stock graph
@app.callback(
    Output('stock-graph', 'figure'),
    Input('get-data-button', 'n_clicks'),
    Input('stock-input', 'value')
)
def update_stock_graph(n_clicks, stock_symbol):
    if n_clicks > 0 and stock_symbol:
        # Fetch stock data
        stock_data = yf.download(stock_symbol, period='1y')
        stock_data.reset_index(inplace=True)

        # Create the stock price graph
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Close Price'))
        figure.update_layout(title=f'{stock_symbol} Stock Price', xaxis_title='Date', yaxis_title='Price (USD)')
        return figure
    return go.Figure()

# Callback to forecast stock prices
@app.callback(
    Output('forecast-graph', 'figure'),
    Input('forecast-button', 'n_clicks'),
    Input('stock-input', 'value'),
    Input('forecast-days', 'value')
)
def update_forecast_graph(n_clicks, stock_symbol, forecast_days):
    if n_clicks > 0 and stock_symbol and forecast_days:
        # Fetch stock data
        stock_data = yf.download(stock_symbol, period='1y')
        stock_data.reset_index(inplace=True)

        # Fit the model
        model = ExponentialSmoothing(stock_data['Close'], trend='add', seasonal=None)
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=forecast_days)
        forecast_index = pd.date_range(start=stock_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)

        # Create the forecast graph
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Historical Close Price'))
        figure.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecasted Price', line=dict(dash='dash')))
        figure.update_layout(title=f'{stock_symbol} Stock Price Forecast', xaxis_title='Date', yaxis_title='Price (USD)')
        return figure
    return go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)