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
    
def determine_weights(age): 

    age = int(age) 

    portfolio_weights.clear()

    user_stock_weights = 110 - age
    user_bonds_weights = (100 - user_stock_weights) - 5
    user_reits_weights = 5

    user_stock_weights = user_stock_weights / 100
    user_bonds_weights = user_bonds_weights / 100
    user_reits_weights = user_reits_weights / 100

    portfolio_weights.append(user_stock_weights)
    portfolio_weights.append(user_bonds_weights)
    portfolio_weights.append(user_reits_weights)

def allocate_portfolio(user_investment_amount): 
    '''
    This function allocates the user's buying power towards each asset class based on the weights calculated above
    
    Parameters: 

    user_investment_amount --> pass in the user_investment_amount from streamlit to calculate how much money the user will invest in each asset class
    '''
    investment_allocation.clear()

    for weight in portfolio_weights: 
        investments_per_asset = float(user_investment_amount) * float(weight)
        investments_per_asset = round(investments_per_asset, 2)
        investments_per_asset = '${:,.2f}'.format(investments_per_asset)
        investment_allocation.append(str(investments_per_asset))
    
    assets = ['Stocks', 'Bonds', 'REITs']
    values = investment_allocation
    #df = pd.DataFrame({'Asset': assets, 'Value': values})
    message(f'I recommend allocating your buying power towards each asset class in the following format: ', seed=21, key=26)
    
    st.table(pd.DataFrame({'Asset': assets, 'Value': values}))



def display_portfolio_allocation(portfolio_type): 
    '''
    This function displays the user's portfolio allocation via a plotly pie chart 
    
    Parameters: 

    portfolio_type --> pass in the portfolio_type from streamlit to check which portfolio the user chose
    '''

    if str(portfolio_type).lower().strip() == 'low risk portfolio':
        
        # Set the labels for the pie chart
        # portfolio_list = ['PEP', 'PG', 'KO', 'JNJ', 'BRK-B', 'MRK', 'PFE', 
        #               'XOM', 'CVX','JPM', 'HD', 'V', '30yr Treasury Yield', 'BTC']

        # Set the labels for the pie chart
        labels = ["Stocks", "Bonds", "REITs"]

        # Set the sizes for each pie slice based on the weights calculated in the determine_weights function
        sizes = portfolio_weights 

        # Set the colors for each pie slice
        #colors = ['#ff9999','#66b3ff','#99ff99']

        # Create a list of lists containing the stock lists for each asset type
        #hover_data = [[', '.join(stocks)], [', '.join(bonds)], [', '.join(crypto)]]

        # Create the pie chart using the go.Pie object
        fig = px.pie(values=sizes, names=labels, title='Asset Allocation', hole=.3)

        # Display the pie chart using st.plotly_chart
        st.plotly_chart(fig)    

    if str(portfolio_type).lower().strip() == 'moderate risk portfolio':
        
        # Set the labels for the pie chart
        # labels = ['UNH', 'MSFT', 'LLY', 'MA', 'GOOG', 'GOOGL', 'ABBV', 
        #           'BAC', 'AAPL','AMZN', 'META', '30yr Treasury Yield', 'BTC']

        labels = ["Stocks", "Bonds", "REITs"]

        # Set the sizes for each pie slice based on the weights calculated in the determine_weights function
        sizes = portfolio_weights 

        # Set the colors for each pie slice
        #colors = ['#ff9999','#66b3ff','#99ff99']

        # Create the pie chart using the go.Pie object
        fig = px.pie(values=sizes, names=labels, title='Asset Allocation', hole=.3)

        # Display the pie chart using st.plotly_chart
        st.plotly_chart(fig)

    if str(portfolio_type).lower().strip() == 'high risk portfolio':
        
        # Set the labels for the pie chart
        #labels = ['TSLA', 'NVDA', '10yr Treasury Yield', 'ETH']

        labels = ["Stocks", "Bonds", "REITs"]

        # Set the sizes for each pie slice based on the weights calculated in the determine_weights function
        sizes = portfolio_weights 

        # Set the colors for each pie slice
        #colors = ['#ff9999','#66b3ff','#99ff99']

        # Create the pie chart using the go.Pie object
        fig = px.pie(values=sizes, names=labels, title='Asset Allocation', hole=.3)

        # Display the pie chart using st.plotly_chart
        st.plotly_chart(fig)


@st.cache_data
# Function to fetch historical data for a stock based on today's date
def fetch_historical_data(ticker, start_date):
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        data = si.get_data(ticker, start_date=start_date, end_date=end_date)
        data = data.reset_index()
        data = data[['index', 'close']]  # Ensure only needed columns are included
        data.rename(columns={'index': 'Date', 'close': 'Close'}, inplace=True)
        data['Close Shift'] = data['Close'].diff()
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data
def display_forecasts_from_data(df, ticker, periods=30):
    if df.empty:
        print(f"Error: Empty DataFrame for {ticker}")
        return


    # Load the saved model
    model = joblib.load('Machine Learning\stock_prophet_model.pkl')

    df.rename(columns={'Date': 'ds', 'Close Shift': 'y'}, inplace=True)
    df.dropna(inplace=True)
    print(df.columns)
    # model = Prophet()
    # model.fit(df)

    last_date = df['ds'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    future = pd.DataFrame({'ds': future_dates})

    forecast = model.predict(future)
    last_known_close = df['Close'].iloc[-1]  # Ensure this column exists before renaming
    forecast['Forecasted Close'] = last_known_close + forecast['yhat'].cumsum()

    print(f"Forecasted Close Prices for {ticker}:")
    print(forecast[['ds', 'Forecasted Close']])

    fig = px.line(forecast, x='ds', y='Forecasted Close', labels={'Forecasted Close': 'Forecast', 'ds':'Date'}, title=f'{ticker} Forecast')
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