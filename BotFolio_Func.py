# import libraries 
import os
import joblib
from streamlit_chat import message
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import time 
from pathlib import Path
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si

from MCPortfolioSimulator_LR import PortfolioSimulator


investment_portfolio = []
portfolio_weights = []
investment_allocation = []

def user_age_verification(age):
    # Ensure that user_age is a string and check if it's a digit
    if not isinstance(age, str) or not age.isdigit():
        return "I'm sorry, but it looks like you entered an invalid number for your age. Please enter a valid whole number! (e.g, 18, 40, 65)"

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


# @st.cache_data
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


# @st.cache_resource
def load_model(model_filename):
    try:
        model = joblib.load(model_filename)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def display_forecasts_from_data(df, ticker, periods=30):
    if df.empty:
        print(f"Error: Empty DataFrame for {ticker}")
        return

    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    df.dropna(inplace=True)

    # model_filename = f"Machine Learning\Models\{ticker}_prophet_model.pkl"
    model_filename = os.path.join("Machine Learning", "Models", f"{ticker}_prophet_model.pkl")
    try:
        model = load_model(model_filename)
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

    forecast['Forecasted Close'] = df['y'].iloc[-1] + (forecast['yhat'] - forecast['yhat'].iloc[0])
    forecast['Forecasted High'] = df['y'].iloc[-1] + (forecast['yhat_upper'] - forecast['yhat_upper'].iloc[0])
    forecast['Forecasted Low'] = df['y'].iloc[-1] + (forecast['yhat_lower'] - forecast['yhat_lower'].iloc[0])
    
    forecast['Open'] = forecast['Forecasted Close'].shift(1)
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
    
    y_min = forecast['Low'].min() * 0.995
    y_max = forecast['High'].max() * 1.005

    fig.update_layout(title=f'{ticker} 30-Day Forecast',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      yaxis_range=[y_min, y_max],
                      xaxis_rangeslider_visible=False)
    
    return fig


def process_multiple_tickers(tickers, start_date):
    figures = {}
    fetched_data = st.session_state.fetched_data  # Use cached data
    with st.spinner("Displaying Forecasts..."):
        time.sleep(0.5)
        for ticker in tickers:
            if ticker in fetched_data:
                print(f"Using cached data for {ticker}")
                historical_data = fetched_data[ticker]
            else:
                print(f"Fetching data for {ticker}")
                historical_data = fetch_historical_data(ticker, start_date)
                if not historical_data.empty:
                    fetched_data[ticker] = historical_data  # Cache the fetched data
                else:
                    print(f"No data fetched for {ticker}")
                    continue
            
            print(f"Data for {ticker}: {historical_data.shape[0]} rows")
            fig = display_forecasts_from_data(historical_data, ticker)
            if fig:
                figures[ticker] = fig
                fetched_data[ticker] = historical_data  # Store fetched data
            else:
                print(f"Error generating forecast for {ticker}")
        time.sleep(0.5)

    if figures:
        st.session_state.fetched_data = fetched_data  # Store all fetched data in session state
        selected_ticker = st.selectbox('Select a ticker to display the forecast:', list(figures.keys()))
        st.plotly_chart(figures[selected_ticker])
    else:
        st.error("No data available for the selected tickers.")


def monte_carlo_simulation(assets, weights, start_date, num_simulations, num_trading_days, progress_bar, status_text):
    end_date = datetime.now()
    all_assets_df = pd.DataFrame()
    for asset in assets:
        df = si.get_data(asset, start_date=start_date, end_date=end_date, index_as_date=True)
        df.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
            'adjclose': 'adj close', 'volume': 'volume'
        }, inplace=True)
        if all_assets_df.empty:
            all_assets_df = df
            all_assets_df.columns = pd.MultiIndex.from_product([[asset], all_assets_df.columns])
        else:
            df.columns = pd.MultiIndex.from_product([[asset], df.columns])
            all_assets_df = pd.concat([all_assets_df, df], axis=1)
    all_assets_df.dropna(inplace=True)
    historical_returns = {}
    for asset in assets:
        close_prices = all_assets_df[asset]['close']
        daily_returns = close_prices.pct_change().dropna()
        avg_daily_return = daily_returns.mean()
        annual_return = ((1 + avg_daily_return) ** 252 - 1) * 100
        historical_returns[asset] = annual_return
    sim_returns = PortfolioSimulator(
        portfolio_data=all_assets_df,
        annual_returns=historical_returns,
        weights=weights,
        num_simulation=num_simulations,
        num_trading_days=num_trading_days,
        progress_bar=progress_bar,
        status_text=status_text
    )
    cumulative_returns = sim_returns.calc_cumulative_return()
    summary = sim_returns.summarize_cumulative_return()
    return summary, sim_returns

# def monte_carlo_simulation(assets, weights, start_date, num_simulations=500, num_trading_days=252*15):
#     end_date = datetime.now()
#     all_assets_df = pd.DataFrame()
#     for asset in assets:
#         df = si.get_data(asset, start_date=start_date, end_date=end_date, index_as_date=True)
#         df.rename(columns={
#             'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
#             'adjclose': 'adj close', 'volume': 'volume'
#         }, inplace=True)
#         if all_assets_df.empty:
#             all_assets_df = df
#             all_assets_df.columns = pd.MultiIndex.from_product([[asset], all_assets_df.columns])
#         else:
#             df.columns = pd.MultiIndex.from_product([[asset], df.columns])
#             all_assets_df = pd.concat([all_assets_df, df], axis=1)
#     all_assets_df.dropna(inplace=True)
#     historical_returns = {}
#     for asset in assets:
#         close_prices = all_assets_df[asset]['close']
#         daily_returns = close_prices.pct_change().dropna()
#         avg_daily_return = daily_returns.mean()
#         annual_return = ((1 + avg_daily_return) ** 252 - 1) * 100
#         historical_returns[asset] = annual_return
#     sim_returns = PortfolioSimulator(
#         portfolio_data=all_assets_df,
#         annual_returns=historical_returns,
#         weights=weights,
#         num_simulation=num_simulations,
#         num_trading_days=num_trading_days
#     )
#     cumulative_returns = sim_returns.calc_cumulative_return()
#     summary = sim_returns.summarize_cumulative_return()
#     return summary, sim_returns