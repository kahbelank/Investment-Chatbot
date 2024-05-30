from tkinter import Image
import streamlit as st
from streamlit_chat import message
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import yahoo_fin.stock_info as si
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from BotFolio_Func import user_age_verification, process_multiple_tickers
from risk_tolerance_data import RISK_QUESTIONS, OPTIONS, SCORES


# Define the list of stocks globally
stocks = ["AAPL", "ABBV", "ADBE", "AMZN", "AVGO", "BRK-B", "CRM", "COST", "CVX", "HD", 
          "JNJ", "JPM", "LLY", "MA", "META", "MRK", "MSFT", "NVDA", "PG", "TSLA", "UNH", "V", "XOM"]
bonds = ["^TNX", "^TYX"]
reits = ["WELL", "O", "CCI"]


# Define global variable for the optimal portfolio
optimal_portfolio = None

st.markdown("## BotFolio: Your Personalized Investment Portfolio")

st.markdown(
    """
    <style>
    .message-content {
        font-size: 18px;
        line-height: 1.5;
    }
    .stMarkdown p {
        font-size: 18px;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def display_forecast_and_historical_prices(df, historical_prices_df, asset_name):
    df.rename(columns={'ds': 'Date', 'yhat': 'Most Likely Case'}, inplace=True)
    historical_prices_df.rename(columns={'Date': 'Date', 'Adj Close': 'Close'}, inplace=True)
    
    trace_forecast = px.line(df, x='Date', y='Most Likely Case', title=f'{asset_name} Forecasts up to the next 30 days')
    trace_historical_prices = px.line(historical_prices_df, x='Date', y='Close', title=f'{asset_name} Historical Prices')
    
    st.plotly_chart(trace_forecast, use_container_width=True)
    st.plotly_chart(trace_historical_prices, use_container_width=True)

def display_risk_score_chart(risk_score):
    if risk_score >= 33:
        risk_level = "High tolerance for risk"
        threshold_color = "#e51f1f" 
    elif risk_score >= 29:
        risk_level = "Above-average tolerance for risk"
        threshold_color = "#f2a134"  
    elif risk_score >= 23:
        risk_level = "Average/moderate tolerance for risk"
        threshold_color = "#f7e379"  
    elif risk_score >= 19:
        risk_level = "Below-average tolerance for risk"
        threshold_color = "#bbdb44"  
    else:
        risk_level = "Low tolerance for risk"
        threshold_color = "#44ce1b"  

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Tolerance Score: {risk_score} ({risk_level})"},
        gauge={
            'axis': {'range': [0, 47], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': threshold_color},
     
            'steps': [
                {'range': [0, 18], 'color': "rgba(0, 0, 0, 0)"},
                {'range': [19, 22], 'color': "rgba(0, 0, 0, 0)"},
                {'range': [23, 28], 'color': "rgba(0, 0, 0, 0)"},
                {'range': [29, 32], 'color': "rgba(0, 0, 0, 0)"},
                {'range': [33, 47], 'color': "rgba(0, 0, 0, 0)"}
            ],
            'threshold': {
                'line': {'color': threshold_color, 'width': 8},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))

    fig.update_layout(
        margin=dict(t=50, b=0, l=25, r=25),
        height=300
    )

    return fig

def calculate_age_score(age):
    min_age = 18
    max_age = 110
    normalized_age = (max_age - age) / (max_age - min_age)
    return max(0, min(normalized_age, 1))

def calculate_investment_horizon_score(horizon):
    investment_horizon_map = {
        "Short-Term": 0.33,
        "Medium-Term": 0.67,
        "Long-Term": 1.0
    }
    return investment_horizon_map.get(horizon, 0)

def calculate_investment_amount_score(amount):
    try:
        amount = float(amount)
    except ValueError:
        return 0
    max_amount = 1000000
    normalized_amount = min(amount / max_amount, 1)
    return normalized_amount

def calculate_income_score(income):
    income_map = {
        "Less than $30,000": 0.0,
        "$30,000 - $49,999": 0.2,
        "$50,000 - $99,999": 0.4,
        "$100,000 - $149,999": 0.6,
        "$150,000 - $199,999": 0.8,
        "More than $200,000": 1.0
    }
    return income_map.get(income, 0)

def calculate_risk_capacity(age, horizon, amount, income):
    age_score = calculate_age_score(age)
    horizon_score = calculate_investment_horizon_score(horizon)
    amount_score = calculate_investment_amount_score(amount)
    income_score = calculate_income_score(income)

    risk_capacity_score = (age_score + horizon_score + amount_score + income_score) / 4
    return risk_capacity_score

def normalize_risk_tolerance_score(risk_score):
    max_score = 47
    return risk_score / max_score

def calculate_composite_risk_profile(risk_capacity, risk_tolerance_score):
    normalized_risk_tolerance_score = normalize_risk_tolerance_score(risk_tolerance_score)
    composite_risk_profile = (risk_capacity + normalized_risk_tolerance_score) / 2
    return composite_risk_profile

def map_composite_risk_profile_to_target_risk(composite_risk_profile, min_risk_level=0.05, max_risk_level=0.20):
    return min_risk_level + (max_risk_level - min_risk_level) * composite_risk_profile

def portfolio_optimization(user_age, investment_horizon, investment_amount, income, risk_tolerance_score, target_risk):
    global optimal_portfolio  # Declare as global to modify the global variable

    assets = stocks + bonds + reits


    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=2*365)).strftime('%Y-%m-%d')

    data = {}
    failed_assets = []
    for asset in assets:
        try:
            data[asset] = si.get_data(asset, start_date=start_date, end_date=end_date)['close']
        except Exception as e:
            failed_assets.append(asset)

    price_df = pd.DataFrame(data)
    price_df = price_df.ffill().dropna(axis=1)
    valid_assets = price_df.columns.tolist()

    if len(valid_assets) == 0:
        st.write("No valid assets fetched. Please check your asset list and data fetching process.")
        return
    
    def calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_portfolios=50):
        target_risks = np.linspace(0.05, 0.25, num_portfolios)
        portfolios = []

        for target_risk in target_risks:
            weights = get_optimal_weights(mean_returns, cov_matrix, risk_free_rate, target_risk)
            returns, std = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
            portfolios.append((std, returns, weights))

        return portfolios

    def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)
        return returns, std

    def minimize_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    def objective_function(weights, mean_returns, cov_matrix, risk_free_rate, target_risk):
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_volatility = minimize_volatility(weights, cov_matrix)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        penalty = abs(portfolio_volatility - target_risk)
        return -sharpe_ratio + penalty

    def get_optimal_weights(mean_returns, cov_matrix, risk_free_rate, target_risk):
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix, risk_free_rate, target_risk)
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: target_risk - minimize_volatility(x, cov_matrix)}
        )
        bounds = []
        for asset in assets:
            if asset in reits:
                bounds.append((0.01, 0.20))  # Ensure at least 1% allocation to REITs
            else:
                bounds.append((0, 0.20))
        bounds = tuple(bounds)

        result = minimize(objective_function, num_assets * [1./num_assets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    returns_df = price_df.pct_change().dropna()
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    risk_free_rate = si.get_data("^TNX")['close'].iloc[-1] / 100

    efficient_frontier = calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)

    selected_portfolio = min(efficient_frontier, key=lambda x: abs(x[0] - target_risk))
    selected_weights = selected_portfolio[2]

    # Determine the category based on the selected portfolio's volatility
    min_volatility = min(p[0] for p in efficient_frontier)
    max_volatility = max(p[0] for p in efficient_frontier)
    volatility_range = max_volatility - min_volatility

    # Define dynamic thresholds
    conservative_threshold = min_volatility + volatility_range / 3
    moderate_threshold = min_volatility + 2 * volatility_range / 3

    if selected_portfolio[0] <= conservative_threshold:
        portfolio_type = "Conservative"
    elif selected_portfolio[0] <= moderate_threshold:
        portfolio_type = "Balanced"
    else:
        portfolio_type = "Aggressive"

    message(f"Based on the efficient frontier, your portfolio type is: {portfolio_type}", seed=21, key=26)

    non_zero_weights = selected_weights[selected_weights > 0]
    non_zero_assets = np.array(valid_assets)[selected_weights > 0]

    normalized_weights = non_zero_weights / np.sum(non_zero_weights)
    rounded_weights = np.round(normalized_weights, 2)
    diff = 1.0 - np.sum(rounded_weights)
    rounded_weights[np.argmax(rounded_weights)] += diff

    # Calculate weights as percentages
    weights_percent = rounded_weights * 100

    # Filter out assets with 0% weights
    non_zero_weights_percent = weights_percent[weights_percent > 0]
    non_zero_assets_percent = non_zero_assets[weights_percent > 0]

    # Create a DataFrame for the filtered and normalized portfolio
    optimal_portfolio = pd.DataFrame({'Asset': non_zero_assets_percent, 'Weight (%)': non_zero_weights_percent})

    st.write("Optimal Portfolio Allocation:")
    st.table(optimal_portfolio)


    optimal_return, optimal_volatility = portfolio_annualized_performance(rounded_weights, mean_returns.loc[non_zero_assets], cov_matrix.loc[non_zero_assets, non_zero_assets])
    sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility

    portfolio_stocks = [(asset, weight) for asset, weight in zip(non_zero_assets, rounded_weights) if asset in stocks]
    portfolio_bonds = [(asset, weight) for asset, weight in zip(non_zero_assets, rounded_weights) if asset in bonds]
    portfolio_reits = [(asset, weight) for asset, weight in zip(non_zero_assets, rounded_weights) if asset in reits]

    user_portfolio = {
        "Stocks": portfolio_stocks,
        "Bonds": portfolio_bonds,
        "REITs": portfolio_reits
    }

    total_stocks = sum(weight for asset, weight in user_portfolio["Stocks"])
    total_bonds = sum(weight for asset, weight in user_portfolio["Bonds"])
    total_reits = sum(weight for asset, weight in user_portfolio["REITs"])

    amount_stocks = total_stocks * float(investment_amount)
    amount_bonds = total_bonds * float(investment_amount)
    amount_reits = total_reits * float(investment_amount)

    investment_allocation = pd.DataFrame({
        'Asset Class': ['Stocks', 'Bonds', 'REITs'],
        'Allocation ($)': [amount_stocks, amount_bonds, amount_reits]
    })

    st.write("Investment Amount Allocation:")
    st.table(investment_allocation)

    # plt.figure(figsize=(10, 6))
    # plt.plot([p[0] for p in efficient_frontier], [p[1] for p in efficient_frontier], 'k-', label='Efficient Frontier')
    # plt.plot(selected_portfolio[0], selected_portfolio[1], 'ms', markersize=10, label='Selected Portfolio')

    # plt.title('Efficient Frontier with Selected Portfolio')
    # plt.xlabel('Volatility (Standard Deviation)')
    # plt.ylabel('Expected Return')
    # plt.legend()
    # plt.grid(True)
    # st.pyplot(plt)

# Initialize state variables at the beginning
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
    st.session_state.answers = [None] * len(RISK_QUESTIONS)
    st.session_state.risk_score = None
    st.session_state.composite_risk_profile = None
    st.session_state.optimal_portfolio = None
    st.session_state.user_age = ""
    st.session_state.investment_horizon = "Short-Term"
    st.session_state.investment_amount = ""
    st.session_state.income = "Less than $30,000"
    st.session_state.show_forecast_prompt = False  # New state variable for forecast prompt
    st.session_state.show_forecasts = False   # New state variable for forecast prompt


def reset_calculations():
    st.session_state.risk_score = None
    st.session_state.composite_risk_profile = None
    st.session_state.optimal_portfolio = None
    st.session_state.show_forecast_prompt = False  # Reset forecast prompt
    st.session_state.show_forecasts = False  

# Main function
def botfolio():
    # User inputs
    user_age = st.text_input('Age', placeholder='Enter your age', value=st.session_state.user_age).strip()
    investment_horizon = st.selectbox(
        "Select your investment horizon",
        options=["Short-Term", "Medium-Term", "Long-Term"],
        index=["Short-Term", "Medium-Term", "Long-Term"].index(st.session_state.investment_horizon)
    )
    investment_amount = st.text_input('Investment Amount', placeholder='Enter your investment amount in USD', value=st.session_state.investment_amount).strip()
    income = st.selectbox(
        "Select your annual income range:",
        options=[
            "Less than $30,000",
            "$30,000 - $49,999",
            "$50,000 - $99,999",
            "$100,000 - $149,999",
            "$150,000 - $199,999",
            "More than $200,000"
        ],
        index=[
            "Less than $30,000",
            "$30,000 - $49,999",
            "$50,000 - $99,999",
            "$100,000 - $149,999",
            "$150,000 - $199,999",
            "More than $200,000"
        ].index(st.session_state.income)
    )

    # Detect changes in user inputs and reset calculations if they change
    if (
        user_age != st.session_state.user_age or
        investment_horizon != st.session_state.investment_horizon or
        investment_amount != st.session_state.investment_amount or
        income != st.session_state.income
    ):
        st.session_state.user_age = user_age
        st.session_state.investment_horizon = investment_horizon
        st.session_state.investment_amount = investment_amount
        st.session_state.income = income
        reset_calculations()

    if user_age and investment_horizon and investment_amount and income:
        user_details = (
            f"I am {user_age} years old\n"
            f"with an annual income of {income}\n"
            f"and I am willing to invest ${investment_amount}\n"
            f"for an investment horizon of {investment_horizon}"
        )
        message(user_details, is_user=True, seed=90, key=12)

        age_verification_message = user_age_verification(user_age)
        message(age_verification_message, seed=21, key=17)

        if "Enjoy the use" in age_verification_message:
            risk_capacity = calculate_risk_capacity(int(user_age), investment_horizon, float(investment_amount), income)
            if risk_capacity >= 0.75:
                risk_capacity_level = "High"
            elif risk_capacity >= 0.5:
                risk_capacity_level = "Moderate"
            else:
                risk_capacity_level = "Low"
            message(f"Your risk capacity is: {risk_capacity} and {risk_capacity_level}", seed=21, key=18)

            current_question = st.session_state.current_question

            st.write(RISK_QUESTIONS[current_question])

            if st.session_state.answers[current_question] is not None:
                try:
                    answer = st.radio(
                        "Select an option:",
                        OPTIONS[current_question],
                        index=OPTIONS[current_question].index(st.session_state.answers[current_question]),
                        key=f"q{current_question}"
                    )
                except ValueError:
                    answer = st.radio(
                        "Select an option:",
                        OPTIONS[current_question],
                        key=f"q{current_question}"
                    )
            else:
                answer = st.radio(
                    "Select an option:",
                    OPTIONS[current_question],
                    key=f"q{current_question}"
                )

            if answer:
                st.session_state.answers[current_question] = answer

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if current_question > 0:
                    if st.button("Previous"):
                        st.session_state.current_question -= 1
                        st.experimental_rerun()

            with col2:
                if current_question < len(RISK_QUESTIONS) - 1:
                    if st.button("Next"):
                        st.session_state.current_question += 1
                        st.experimental_rerun()

            if current_question == len(RISK_QUESTIONS) - 1:
                if st.button("Submit"):
                    if None not in st.session_state.answers:
                        risk_score = 0
                        for i, answer in enumerate(st.session_state.answers):
                            index = OPTIONS[i].index(answer)
                            risk_score += SCORES[i][index]

                        st.session_state.risk_score = risk_score

                        # Determine the risk level
                        if risk_score >= 33:
                            risk_level = "High tolerance for risk"
                        elif risk_score >= 29:
                            risk_level = "Above-average tolerance for risk"
                        elif risk_score >= 23:
                            risk_level = "Average/moderate tolerance for risk"
                        elif risk_score >= 19:
                            risk_level = "Below-average tolerance for risk"
                        else:
                            risk_level = "Low tolerance for risk"

                        message(f"Your risk tolerance score is {risk_score}, indicating a {risk_level}.", seed=21, key=24)

                        # Display risk score chart
                        fig = display_risk_score_chart(risk_score)
                        st.plotly_chart(fig)

                        # Calculate and store composite risk profile
                        composite_risk_profile = calculate_composite_risk_profile(risk_capacity, risk_score)
                        st.session_state.composite_risk_profile = composite_risk_profile

                        message(f"Your overall risk profile score is: {composite_risk_profile:.2f}", seed=21, key=25)

                        # Calculate target risk
                        target_risk = map_composite_risk_profile_to_target_risk(composite_risk_profile)
                        target_risk_percentage = target_risk * 100
                        st.write(f"Your target risk level is: {target_risk:.2f} ({target_risk_percentage:.1f}%)")

                        # Optimize portfolio and store the result
                        portfolio_optimization(int(user_age), investment_horizon, float(investment_amount), income, risk_score, target_risk)
                        st.session_state.optimal_portfolio = optimal_portfolio

                        # Show forecast prompt after portfolio optimization
                        st.session_state.show_forecast_prompt = True
                    else:
                        message("Please answer all questions before submitting.", seed=21, key=25)

            # Show forecast prompt if ready
            if st.session_state.show_forecast_prompt:
                # st.write(st.session_state.optimal_portfolio)
                message("Would you like me to display forecasts of each asset in your portfolio?", seed=21, key=20)
                user_input = st.text_input(' ', placeholder='Display forecasts? (enter yes/no)').strip().lower()

                if user_input == 'yes' and st.session_state.optimal_portfolio is not None:
                    optimal_stocks = st.session_state.optimal_portfolio[st.session_state.optimal_portfolio['Asset'].isin(stocks)]['Asset'].tolist()
                    start_date = datetime.today() - timedelta(days=3650)
                    process_multiple_tickers(optimal_stocks, start_date, user_input)

botfolio()
# def botfolio():
#     user_age = st.text_input('Age', placeholder='Enter your age').strip()
#     investment_horizon = st.selectbox(
#         "Select your investment horizon",
#         options=[
#             "Short-Term",
#             "Medium-Term",
#             "Long-Term",
#         ]
#     )
#     investment_amount = st.text_input('Investment Amount', placeholder='Enter your investment amount in USD').strip()
#     income = st.selectbox(
#         "Select your annual income range:",
#         options=[
#             "Less than $30,000",
#             "$30,000 - $49,999",
#             "$50,000 - $99,999",
#             "$100,000 - $149,999",
#             "$150,000 - $199,999",
#             "More than $200,000"
#         ]
#     )

#     if "current_question" not in st.session_state:
#         st.session_state.current_question = 0
#         st.session_state.answers = [None] * len(RISK_QUESTIONS)
    
#     if user_age and investment_horizon and investment_amount and income:
#         user_details = (
#                 f"I am {user_age} years old\n"
#                 f"with an annual income of {income}\n"
#                 f"and I am willing to invest ${investment_amount}\n"
#                 f"for an investment horizon of {investment_horizon}"
#         )
#         message(user_details, is_user=True, seed=1, key=12)

#         age_verification_message = user_age_verification(user_age)
#         message(age_verification_message, seed=21, key=17)

#         if "Enjoy the use" in age_verification_message:
#             risk_capacity = calculate_risk_capacity(int(user_age), investment_horizon, float(investment_amount), income)
#             if risk_capacity >= 0.75:
#                 risk_capacity_level = "High"
#             elif risk_capacity >= 0.5:
#                 risk_capacity_level = "Moderate"
#             else:
#                 risk_capacity_level = "Low"
#             message(f"Your risk capacity is: {risk_capacity} and {risk_capacity_level}", seed=21, key=18)

#             current_question = st.session_state.current_question

#             st.write(RISK_QUESTIONS[current_question])

#             if st.session_state.answers[current_question] is not None:
#                 try:
#                     answer = st.radio(
#                         "Select an option:",
#                         OPTIONS[current_question],
#                         index=OPTIONS[current_question].index(st.session_state.answers[current_question]),
#                         key=f"q{current_question}"
#                     )
#                 except ValueError:
#                     answer = st.radio(
#                         "Select an option:",
#                         OPTIONS[current_question],
#                         key=f"q{current_question}"
#                     )
#             else:
#                 answer = st.radio(
#                     "Select an option:",
#                     OPTIONS[current_question],
#                     key=f"q{current_question}"
#                 )

#             if answer:
#                 st.session_state.answers[current_question] = answer

#             col1, col2, col3 = st.columns([1, 1, 2])

#             with col1:
#                 if current_question > 0:
#                     if st.button("Previous"):
#                         st.session_state.current_question -= 1
#                         st.experimental_rerun()

#             with col2:
#                 if current_question < len(RISK_QUESTIONS) - 1:
#                     if st.button("Next"):
#                         st.session_state.current_question += 1
#                         st.experimental_rerun()
        
#             if "risk_score" not in st.session_state:
#                 st.session_state.risk_score = None
#             if "composite_risk_profile" not in st.session_state:
#                 st.session_state.composite_risk_profile = None
#             if "optimal_portfolio" not in st.session_state:
#                 st.session_state.optimal_portfolio = None

#             if current_question == len(RISK_QUESTIONS) - 1:
#                 if st.button("Submit"):
#                     if None not in st.session_state.answers:
#                         risk_score = 0
#                         for i, answer in enumerate(st.session_state.answers):
#                             index = OPTIONS[i].index(answer)
#                             risk_score += SCORES[i][index]

#                         st.session_state.risk_score = risk_score

#                         if risk_score >= 33:
#                             risk_level = "High tolerance for risk"
#                         elif risk_score >= 29:
#                             risk_level = "Above-average tolerance for risk"
#                         elif risk_score >= 23:
#                             risk_level = "Average/moderate tolerance for risk"
#                         elif risk_score >= 19:
#                             risk_level = "Below-average tolerance for risk"
#                         else:
#                             risk_level = "Low tolerance for risk"

#                         message(f"Your risk tolerance score is {risk_score}, indicating a {risk_level}.", seed=21, key=24)
#                         # Display risk score chart
#                         fig = display_risk_score_chart(risk_score)
#                         st.plotly_chart(fig)

#                         # Calculate and store composite risk profile
#                         composite_risk_profile = calculate_composite_risk_profile(risk_capacity, risk_score)
#                         st.session_state.composite_risk_profile = composite_risk_profile

#                         message(f"Your overall risk profile score is: {composite_risk_profile:.2f}", seed=21, key=25)

#                         # Calculate target risk
#                         target_risk = map_composite_risk_profile_to_target_risk(composite_risk_profile)
#                         target_risk_percentage = target_risk * 100
#                         st.write(f"Your target risk level is: {target_risk:.2f} ({target_risk_percentage:.1f}%)")

#                         # Optimize portfolio and store the result
#                         portfolio_optimization(int(user_age), investment_horizon, float(investment_amount), income, risk_score, target_risk)
#                         st.session_state.optimal_portfolio = optimal_portfolio

#                         start_date = datetime.today() - timedelta(days=3650)
#                         message("Would you like me to display forecasts of each asset in your portfolio?", seed=21, key=20)
#                         user_input = st.text_input(' ', placeholder='Display forecasts? (enter yes/no)').strip().lower()
                        
#                         if user_input == 'yes' and st.session_state.optimal_portfolio is not None:
#                             optimal_stocks = st.session_state.optimal_portfolio[st.session_state.optimal_portfolio['Asset'].isin(stocks)]['Asset'].tolist()
#                             process_multiple_tickers(optimal_stocks, start_date, user_input)
#                     else:
#                         message("Please answer all questions before submitting.", seed=21, key=25)


# botfolio()
