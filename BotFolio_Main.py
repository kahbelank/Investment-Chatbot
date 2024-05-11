# import libraries 
from tkinter import Image
import streamlit as st
from streamlit_chat import message
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

from BotFolio_Func import user_age_verification, determine_weights, allocate_portfolio, display_portfolio_allocation,process_multiple_tickers


st.markdown("## BotFolio: Your Personalized Investment Portfolio")

def style_specific_rows(row):
    # Define color for each portfolio type
    color_map = {
        'Aggressive': 'background-color: #800020;',  # Burgundy
        'Moderate': 'background-color: #E08A1E;',    # Mustard
        'Conservative': 'background-color: #013220;' # Dark Green
    }
    # Apply color based on the portfolio type
    return [color_map.get(row['Portfolio Type'], '') for _ in row]

def display_forecast_and_historical_prices(df, historical_prices_df, asset_name):
    # Rename columns to match the required format for plotting
    df.rename(columns={'ds': 'Date', 'yhat': 'Most Likely Case'}, inplace=True)
    historical_prices_df.rename(columns={'Date': 'Date', 'Adj Close': 'Close'}, inplace=True)
    
    # Create line charts using Plotly Express
    trace_forecast = px.line(df, x='Date', y='Most Likely Case', title=f'{asset_name} Forecasts up to the next 30 days')
    trace_historical_prices = px.line(historical_prices_df, x='Date', y='Close', title=f'{asset_name} Historical Prices')
    
    # Display the charts
    st.plotly_chart(trace_forecast, use_container_width=True)
    st.plotly_chart(trace_historical_prices, use_container_width=True)

def botfolio():
    message("Hello, I'm a chatbot that can generate a weighted investment portfolio for you!\n\n Please enter your age to get started!", seed=21, key=10)

    user_age = st.text_input(' ', placeholder='Enter your age').strip()
    if user_age:
        message(user_age, is_user=True, seed=1, key=12)
        
        # Pre-validate before calling the function
        if not user_age.isdigit():
            message("I'm sorry, but it looks like you entered an invalid number. Please enter a valid whole number!", seed=21, key=13)
        else:
            age_verification_message = user_age_verification(user_age)
            if age_verification_message:
                message(age_verification_message, seed=21, key=13)
                
                if "Enjoy the use" in age_verification_message:
                    
                    message("Please enter your desired investment amount in USD", seed=21, key=14)
                    user_investment_amount = st.text_input(' ', placeholder='Enter investment amount in USD').strip()
                    if user_investment_amount:
                        message(user_investment_amount, is_user=True, seed=1, key=6)
                        user_investment_amount = user_investment_amount.strip()
                        if user_investment_amount.isnumeric():
                            data = {
                                "Portfolio Type": ["Aggressive", "Moderate", "Conservative"],
                                "Risk Tolerance": [
                                    "Comfortable with significant fluctuations in portfolio value.",
                                    "Comfortable with moderate fluctuations in portfolio value.",
                                    "Preferring minimal fluctuations in portfolio value."
                                ],
                                "Investment Goals": [
                                    "Maximizing long-term wealth accumulation.",
                                    "Balanced approach to wealth accumulation and capital preservation.",
                                    "Preservation of capital and wealth."
                                ],
                                "Examples": [
                                    "Young professional with stable income, willing to endure short-term losses for long-term growth potential.",
                                    "Mid-career individual with balanced financial situation, seeking a balance between growth and stability.",
                                    "Retiree living off investment income, focused on capital preservation and income generation."
                                ]
                            }
                            df = pd.DataFrame(data)
                            
                            message("Based on the table below, please enter a portfolio type that matches your risk tolerance (Aggressive Portfolio, Moderate Portfolio, Conservative Portfolio)", seed=21, key=16)
                            # Style the DataFrame
                            styled_df = df.style.apply(style_specific_rows, axis=1)
                            
                            # Display the styled DataFrame as HTML
                            st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
                            portfolio_type = st.text_input(' ', placeholder='Enter portfolio type')

                            valid_portfolio_types = ['aggressive portfolio', 'moderate portfolio', 'conservative portfolio']

                            if portfolio_type: 
                                if str(portfolio_type).lower().strip() in valid_portfolio_types: 
                                    if str(portfolio_type).lower().strip() == 'aggressive portfolio': 

                                        df = pd.DataFrame(
                                            {'Stocks': ['NVDA', 'TSLA', 'AMZN', 'META', 'ADBE', 'AVGO', 'LLY'],
                                            'Bonds': ['10yr Treasury Yield', '-', '-', '-', '-', '-','-'], 
                                            'REITs': ['WELL', '-', '-', '-', '-', '-', '-']
                                            })
                                        message('Your Aggressive Portfolio contains the following assets: ', seed=21, key=17)
                                        st.table(df) 

                                        # Define ticker lists based on portfolio type
                                        tickers = {
                                            'Aggressive Portfolio': ['NVDA', 'TSLA', 'AMZN', 'META', 'ADBE', 'AVGO', 'LLY'],
                                            # 'Moderate Portfolio': ['MSFT', 'AAPL', 'GOOGL', 'FB', 'INTC'],
                                            # 'Conservative Portfolio': ['JNJ', 'PG', 'KO', 'PEP', 'UL']
                                        }

                                        start_date = datetime.today() - timedelta(days=3650)

                                        # Fetch the appropriate tickers for the selected portfolio
                                        selected_tickers = tickers.get(portfolio_type, [])

        
                                        # calculate weights for portfolio
                                        determine_weights(user_age)
                                        allocate_portfolio(user_investment_amount)

                                        # display portfolio allocation pie chart 
                                        display_portfolio_allocation(portfolio_type)

                                        # Ask the user for their desired investment amount 
                                        message("Would you like me to display forecasts of each asset in your portfolio?", seed=21, key=35)
                                        user_input = st.text_input(' ', placeholder='Display forecasts? (enter yes/no)')
                                        message(user_input, is_user=True, seed=1, key=36)

                                        # display prophet model forecasts
                                        process_multiple_tickers(selected_tickers, start_date, user_input)

                                        message('If you would wish to view forecasts for other assets not within the portfolio, input their ticker symbols (e.g JNJ, NVDA, TSLA)', seed=21, key=37)
                                        user_input = st.text_input(' ', placeholder='Ticker Symbols (TSLA, NVDA, AAPL, MSFT)')
                                        message(user_input, is_user=True, seed=1, key=38)
                                        
                                        
                                        

         



botfolio()