# import libraries 
import joblib
from streamlit_chat import message
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import time 
from pathlib import Path
from prophet import Prophet
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si


investment_portfolio = []
portfolio_weights = []
investment_allocation = []

def user_age_verification(age):
    # Ensure that user_age is a string and check if it's a digit
    if not isinstance(age, str) or not age.isdigit():
        return "I'm sorry, but it looks like you entered an invalid number. Please enter a valid whole number!"

    # Safe to convert to int now that we've confirmed it's all digits
    age = int(age)
    
    if 18 <= age <= 110:
        return "You are over 18 years old! Enjoy the use of our investment portfolio generator!"
    elif age > 110:
        return "I'm sorry, but it looks like you are too old to use this application. Please enter an age less than 110!"
    else:
        return "This application requires you to be at least 18 years old!"


##Forecasting Functions##

def calculate_technical_indicators(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    data['20_SMA'] = data['Close'].rolling(window=20).mean()
    data['20_std'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['20_SMA'] + (data['20_std'] * 2)
    data['Lower_Band'] = data['20_SMA'] - (data['20_std'] * 2)
    
    data['Close_1'] = data['Close'].shift(1)
    data['Close_2'] = data['Close'].shift(2)
    
    data.dropna(inplace=True)
    return data

# Function to fetch historical data for a stock based on today's date
def fetch_historical_data(ticker, start_date):
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        data = si.get_data(ticker, start_date=start_date, end_date=end_date)
        data = data.reset_index()

        if 'close' not in data.columns:
            raise ValueError("'Close' column not found in the data")

        data = data[['index', 'close', 'open', 'high', 'low', 'volume']]  # Ensure only needed columns are included
        data.rename(columns={'index': 'Date', 'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
        
        data = calculate_technical_indicators(data)
        return data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


# def display_forecasts_from_data(df, ticker, periods=30):
#     if df.empty:
#         print(f"Error: Empty DataFrame for {ticker}")
#         return


#     # Load the saved model
#     model = joblib.load('Machine Learning\stock_prophet_model.pkl')

#     df.rename(columns={'Date': 'ds', 'Close Shift': 'y'}, inplace=True)
#     df.dropna(inplace=True)
#     print(df.columns)
#     # model = Prophet()
#     # model.fit(df)

#     last_date = df['ds'].iloc[-1]
#     future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
#     future = pd.DataFrame({'ds': future_dates})

#     forecast = model.predict(future)
#     last_known_close = df['Close'].iloc[-1]  # Ensure this column exists before renaming
#     forecast['Forecasted Close'] = last_known_close + forecast['yhat'].cumsum()

#     print(f"Forecasted Close Prices for {ticker}:")
#     print(forecast[['ds', 'Forecasted Close']])

#     fig = px.line(forecast, x='ds', y='Forecasted Close', labels={'Forecasted Close': 'Forecast', 'ds':'Date'}, title=f'{ticker} Forecast')
#     return fig


def display_forecasts_from_data(df, ticker, periods=30):
    if df.empty:
        print(f"Error: Empty DataFrame for {ticker}")
        return

    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    df.dropna(inplace=True)

    # Load the saved model
    model_filename = f"Machine Learning\Models\{ticker}_prophet_model.pkl"
    try:
        model = joblib.load(model_filename)
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        return

    last_date = df['ds'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    future = pd.DataFrame({'ds': future_dates})
    
    # Add regressors to future dataframe
    future['SMA_10'] = df['SMA_10'].iloc[-1]
    future['RSI_14'] = df['RSI_14'].iloc[-1]
    future['Upper_Band'] = df['Upper_Band'].iloc[-1]
    future['Lower_Band'] = df['Lower_Band'].iloc[-1]
    future['Close_1'] = df['y'].iloc[-1]
    future['Close_2'] = df['y'].iloc[-2]

    forecast = model.predict(future)

    #print(forecast[['ds', 'yhat']])
    forecast['Forecasted Close'] = df['y'].iloc[-1] + (forecast['yhat'] - forecast['yhat'].iloc[0])
    forecast['Forecasted High'] = df['y'].iloc[-1] + (forecast['yhat_upper'] - forecast['yhat_upper'].iloc[0])
    forecast['Forecasted Low'] = df['y'].iloc[-1] + (forecast['yhat_lower'] - forecast['yhat_lower'].iloc[0])
    
    # Generate candlestick data
    forecast['Open'] = forecast['Forecasted Close'].shift(1)
    # forecast['High'] = forecast[['Forecasted Close', 'Open']].max(axis=1)
    # forecast['Low'] = forecast[['Forecasted Close', 'Open']].min(axis=1)
    forecast['High'] = forecast['Forecasted High']
    forecast['Low'] = forecast['Forecasted Low']
    forecast['Close'] = forecast['Forecasted Close']
    forecast.dropna(inplace=True)

    fig = go.Figure(data=[go.Candlestick(x=forecast['ds'],
                                         open=forecast['Open'],
                                         high=forecast['High'],
                                         low=forecast['Low'],
                                         close=forecast['Close'],
                                         increasing_line_color='green', decreasing_line_color='red')])
    

    
    # Adjust y-axis range for a tighter view
    y_min = forecast['Low'].min() * 0.995
    y_max = forecast['High'].max() * 1.005


    fig.update_layout(title=f'{ticker} 30-Day Forecast',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      yaxis_range=[y_min, y_max],
                      xaxis_rangeslider_visible=False)
    
    return fig


def process_multiple_tickers(tickers, start_date, user_confirmation):

    user_confirmation = user_confirmation.strip().lower()

    if user_confirmation == 'yes':
        figures = {}
        with st.spinner("Displaying Forecasts..."):
            time.sleep(0.5)
            for ticker in tickers:
                historical_data = fetch_historical_data(ticker, start_date)
                if not historical_data.empty:
                    fig = display_forecasts_from_data(historical_data, ticker)
                    if fig:
                        figures[ticker] = fig
            time.sleep(0.5) 

        # If figures are created, display them in Streamlit
        if figures:
            selected_ticker = st.selectbox('Select a ticker to display the forecast:', list(figures.keys()))
            st.plotly_chart(figures[selected_ticker])
        else:
            st.error("No data available for the selected tickers.")