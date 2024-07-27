import time
import requests
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
from streamlit_extras.stateful_button import button

from BotFolio_Func import monte_carlo_simulation, user_age_verification, process_multiple_tickers
from risk_tolerance_data import RISK_QUESTIONS, OPTIONS, SCORES

# Sidebar for app information
with st.sidebar:
    st.title("About OptiFi")
    st.markdown("""
        **OptiFi** is your personalized investment advisor.
        This app helps you create an optimized investment portfolio based on your risk profile and investment preferences.

        ### Features:
        - **Risk Assessment:** Determine your risk tolerance and capacity.
        - **Portfolio Optimization:** Get recommendations for an optimal investment portfolio.
        - **Forecasts:** View stock price prediction for stocks in your portfolio.
        - **Monte Carlo Simulation:** See long-term projected returns based on Monte Carlo simulations.

        ### How to Use:
        1. Enter your age, investment horizon, investment amount, and income.
        2. Complete the risk tolerance questionnaire.
        3. Run the portfolio optimization to get your recommended portfolio.
        4. Optionally, view forecasts and long-term projections.

        ### Disclaimer:
        - The Portfolio Creation, Forecasting and Long-term projections are based on historical data of 2 years up till the current date, making it dynamic. Hence, the results may take about 2 to 3 minutes to complete.
        - This tool is for educational purposes only and should not be considered financial advice.
    """)

# Initialize session state for fetched data
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = {}

# Define the list of stocks globally
stocks = ["AAPL", "ABBV", "ADBE", "AMZN", "AVGO", "BRK-B", "CRM", "COST", "CVX", "HD", 
          "JNJ", "JPM", "LLY", "MA", "META", "MRK", "MSFT", "NVDA", "PG", "TSLA", "UNH", "V", "XOM"]
bonds = ["^TNX", "^TYX"]
reits = ["WELL", "O", "CCI"]


# Define global variable for the optimal portfolio
optimal_portfolio = None

st.markdown("## OptiFi: Your Personalized Investment Portfolio")

st.markdown(
    """
    <style>
    .message-content {
        font-size: 18px;
        line-height: 1.5;
    }
    .message-content {
        font-size: 18px;
        line-height: 1.5;
        margin: 0;
    }
    .stDeployButton {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

with open('chat.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

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
        height=200
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

    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
    message(f"Based on the efficient frontier, your portfolio type is: {portfolio_type}", is_user=False, avatar_style="avataaars",seed=avatar_url)

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


    allocation_message = "Your portfolio allocation is as follows:\n\n"
    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
    message(allocation_message, is_user=False, avatar_style="avataaars", seed=avatar_url)

    st.write("Optimal Portfolio Allocation:")
    st.table(optimal_portfolio)

    # Explanation for the split
    explanation_message = ("The portfolio allocation above is designed based on your risk profile to balance risk and return.\n\n"
                       "Higher percentages in stocks indicate a more aggressive approach, while higher percentages in bonds and REITs indicate a more conservative approach.\n\n"
                       "This allocation aims to achieve the optimal balance for your risk profile and investment horizon.")



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

    portfolio_plot_message = ("The donut plot below shows the allocation of your investment amount across different asset classes.\n"
                          "\nThis visual representation helps you understand the distribution of your investment in Stocks, Bonds, and REITs.")
    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
    message(portfolio_plot_message, is_user=False, avatar_style="avataaars", seed=avatar_url)

    # Define the data for the donut plot
    labels = ['Stocks', 'Bonds', 'REITs']
    values = [total_stocks * float(investment_amount), total_bonds * float(investment_amount), total_reits * float(investment_amount)]

    # Create the donut plot using Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

    # Update the layout for better appearance
    fig.update_layout(title_text='Portfolio Allocation', annotations=[dict(text='', x=0.5, y=0.5, font_size=20, showarrow=False)])

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Investment Amount Allocation Table
    investment_table_message = ("The table below details the allocation of your investment amount in dollar terms.\n"
                                "\nIt breaks down your investment into specific amounts allocated to Stocks, Bonds, and REITs based on the calculated percentages.")
    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
    message(investment_table_message, is_user=False, avatar_style="avataaars", seed=avatar_url)

    amount_stocks = total_stocks * float(investment_amount)
    amount_bonds = total_bonds * float(investment_amount)
    amount_reits = total_reits * float(investment_amount)

    investment_allocation = pd.DataFrame({
        'Asset Class': ['Stocks', 'Bonds', 'REITs'],
        'Allocation ($)': [amount_stocks, amount_bonds, amount_reits]
    })

    st.write("Investment Amount Allocation:")
    st.table(investment_allocation)
        # Display the messages
    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
    message(explanation_message, is_user=False, avatar_style="avataaars", seed=avatar_url)


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
    st.session_state.forecasts_processed = False
    st.session_state.forecast_yes_clicked = False  # New state variable for "Yes" button
    st.session_state.forecast_no_clicked = False
    st.session_state.monte_yes_clicked = False  # New state variable for "Yes" button
    st.session_state.monte_no_clicked = False
    st.session_state.optimization_run = False


def reset_calculations():
    st.session_state.risk_score = None
    st.session_state.composite_risk_profile = None
    st.session_state.optimal_portfolio = None
    st.session_state.show_forecast_prompt = False  # Reset forecast prompt
    st.session_state.show_forecasts = False  
    st.session_state.forecasts_processed = False
    st.session_state.forecast_yes_clicked = False  # Reset "Yes" button state
    st.session_state.forecast_no_clicked = False # Reset "No" button state
    st.session_state.monte_yes_clicked = False  # New state variable for "Yes" button
    st.session_state.monte_no_clicked = False
    st.session_state.optimization_run = False



avatar_user = "https://api.dicebear.com/9.x/thumbs/svg?seed=white&backgroundColor=69d2e7&backgroundType=gradientLinear"
avatar_url ="https://api.dicebear.com/9.x/avataaars-neutral/svg?seed=James2&backgroundColor=ffd5dc&backgroundType=gradientLinear&eyes=happy&hairColor=2c1b18&skinColor=d08b5b,edb98a"
tooltips = {
    "age": (
        "The age score is calculated by normalizing the age within a predefined range. "
        "Younger investors typically have a higher capacity for risk due to a longer recovery horizon. "
        "This decision is based on the life-cycle hypothesis which suggests younger individuals are in the wealth accumulation phase."
    ),
    "investment_horizon": (
        "The investment horizon is the time you plan to keep your investments. "
        "Longer horizons allow for higher risk capacity as there's more time to recover from market downturns. "
        "We score this based on whether it's short, medium, or long-term."
    ),
    "investment_amount": (
        "The amount you would like to invest should start from a minimum of USD$100 for the OptiFi chatbot to give more meaningful recommendations."
        "Larger investment amounts suggest a greater ability to handle losses, indicating a higher risk capacity. "
        "We normalize this amount to a scale to make sure very large amounts don't overly influence the score."
    ),
    "income": (
        "The income score evaluates your capacity for financial risk based on your annual earnings. "
        "Higher incomes allow for more risk-taking due to increased financial stability. "
        "Income levels are categorized into ranges, each assigned a normalized score from 0 to 1."
    ),
    "risk_capacity": (
        "Risk capacity combines objective measures like age, investment horizon, investment amount, and income. "
        "These factors are normalized and combined to create an overall risk capacity score."
    ),
    "risk_tolerance": (
        "Risk tolerance assesses your psychological comfort with risk through a standardized questionnaire. "
        "The 13-Item Risk Tolerance Scale, validated by extensive research, is used to gauge your comfort level with market fluctuations and potential losses."
    ),
    "portfolio_optimization": (
        "Portfolio optimization uses Modern Portfolio Theory (MPT) to recommend an optimal portfolio. "
        "Historical returns and the covariance matrix are calculated to construct the Efficient Frontier, identifying the best portfolios for a given risk level. "
        "This involves analyzing historical returns and risks of different assets to find the best mix for your risk level."
    ),
    "short_term_forecasting": (
        "Short-term stock price forecasting employs a machine learning model (Facebook Prophet Model). "
        "Prophet decomposes time series data into trend, seasonality, and holiday effects. "
        "These model analyzes historical data to predict future prices, helping you make informed decisions such as Buy or Sell."
    ),
    "long_term_projection": (
        "Long-term projections use Monte Carlo simulations to estimate future portfolio performance. "
        "Simulations are run using historical data to generate cumulative returns, providing insights into the expected range of returns and associated risks. "
        "The simulations account for random sampling of returns, generating a range of potential outcomes and confidence intervals which shows the predicted amount the portfolio will end within for a specified number of years"
    ),
    "number_years": (
        "To view the long-term projections, select the number of years you would like to forecast. "
        "The number of years determines the duration over which the future portfolio performance is projected, affecting the range and accuracy of the simulated outcomes."
    ),
}

def add_scroll_js(element_id):
    scroll_script = f"""
    <script>
    var element = document.getElementById('{element_id}');
    if (element) {{
        element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}
    </script>
    """
    st.markdown(scroll_script, unsafe_allow_html=True)


def validate_investment_amount(amount):
    if not amount:  # Check if the input is empty
        return None
    try:
        amount = float(amount)
        if amount < 100:
            st.error("A minimum investment amount of USD$100 is required for more meaningful recommendations.")
            return None
        return amount
    except ValueError:
        st.error("Invalid investment amount, please enter a valid number from USD$100 onwards.")
        return None

def botfolio():
# = st.sidebar.checkbox('Forecast')
    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
    message("Welcome to OptiFi! Let's get started by collecting some information about you to determine your risk capacity.", is_user=False, avatar_style="avataaars",seed=avatar_url)
    st.subheader("Risk Capacity Assessment", divider='green', help=tooltips["risk_capacity"])
    user_age = st.text_input('Age', placeholder='Enter your age', help=tooltips["age"]).strip()
    investment_horizon = st.selectbox(
    "Select your investment horizon",
    options=[
    "Short-Term",
    "Medium-Term",
    "Long-Term",
    ],
    help=tooltips["investment_horizon"]
    )
    investment_amount = st.text_input('Investment Amount', placeholder='Enter your investment amount in USD', help=tooltips["investment_amount"]).strip()
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
    help=tooltips["income"]
    )

    # Reset session state if necessary
    reset_calculations()

    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
        st.session_state.answers = [None] * len(RISK_QUESTIONS)

    # Validate investment amount
    valid_investment_amount = validate_investment_amount(investment_amount)

    if user_age and investment_horizon and valid_investment_amount is not None and income:
        if income in [
            "$30,000 - $49,999",
            "$50,000 - $99,999",
            "$100,000 - $149,999",
            "$150,000 - $199,999"
        ]:
            income = "\\" + income
        user_details = (
                f" I am {user_age} years old\n"
                f" with an annual income of {income}\n"
                f" \nand I am willing to invest USD${investment_amount}\n"
                f" for an investment horizon of {investment_horizon}"
        )
        st.markdown("<div style='text-align: right; font-weight: bold; color: cyan; font-size: 17px;'>You</div>", unsafe_allow_html=True)
        message(user_details, is_user=True, avatar_style="thumbs",seed=avatar_user)

        age_verification_message = user_age_verification(user_age)
        st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
        message(age_verification_message, is_user=False, avatar_style="avataaars",seed=avatar_url)

        if "Enjoy the use" in age_verification_message:
            risk_capacity = calculate_risk_capacity(int(user_age), investment_horizon, float(investment_amount), income)
            if risk_capacity >= 0.75:
                risk_capacity_level = "High"
            elif risk_capacity >= 0.5:
                risk_capacity_level = "Moderate"
            else:
                risk_capacity_level = "Low"
            #message(f"Your risk capacity is: {risk_capacity} and {risk_capacity_level}", seed=21, key=18)
             # Add the message with a unique ID
            message_id = "risk_capacity_message"
            st.markdown(f"<div id='{message_id}'></div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
            message(f"Based on your inputs, I have determined that you have a {risk_capacity_level} risk capacity.\n\n Next, please complete the risk tolerance questionnaire below to allow me to determine your risk tolerance:", is_user=False, avatar_style="avataaars",seed=avatar_url)
            add_scroll_js(message_id)           
            st.subheader("Risk Tolerance Questionnaire", divider='blue',help=tooltips["risk_tolerance"])
            if "submit_clicked" not in st.session_state:
                st.session_state.submit_clicked = False
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
                        st.rerun()

            with col2:
                if current_question < len(RISK_QUESTIONS) - 1:
                    if st.button("Next"):
                        st.session_state.current_question += 1
                        st.rerun()
        
            if "risk_score" not in st.session_state:
                st.session_state.risk_score = None
            if "composite_risk_profile" not in st.session_state:
                st.session_state.composite_risk_profile = None
            if "optimal_portfolio" not in st.session_state:
                st.session_state.optimal_portfolio = None

            if current_question == len(RISK_QUESTIONS) - 1:
                 if button("Submit", key="submit"):
                    if None not in st.session_state.answers:
                        risk_score = 0
                        for i, answer in enumerate(st.session_state.answers):
                            index = OPTIONS[i].index(answer)
                            risk_score += SCORES[i][index]

                        st.session_state.risk_score = risk_score

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
                        st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                        message(f"Your risk tolerance score is {risk_score}, indicating a {risk_level}.", is_user=False, avatar_style="avataaars",seed=avatar_url)
                        # Display risk score chart
                        fig = display_risk_score_chart(risk_score)
                        st.plotly_chart(fig)

                        # Calculate and store composite risk profile
                        composite_risk_profile = calculate_composite_risk_profile(risk_capacity, risk_score)
                        st.session_state.composite_risk_profile = composite_risk_profile
                        st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                        message(f"Your overall risk profile score is: {composite_risk_profile:.2f} \n\n Based on your risk profile, I will now proceed to determine your portfolio type and the recommended allocations. \n\nPlease tick the checkbox below to run the portfolio optimization process", is_user=False, avatar_style="avataaars",seed=avatar_url)

                        # Calculate target risk
                        target_risk = map_composite_risk_profile_to_target_risk(composite_risk_profile)
                        target_risk_percentage = target_risk * 100
                        #st.write(f"Your target risk level is: {target_risk:.2f} ({target_risk_percentage:.1f}%)")
                        if st.checkbox("Run Portfolio Optimization",help=tooltips["portfolio_optimization"] ) and not st.session_state.optimization_run:
                            with st.spinner('Creating your portfolio...'):
                                st.toast("Processing your request... Please wait.", icon="⏳")
                                time.sleep(1)
                                st.toast("This may take up to 1-2 minutes...", icon="⏳")
                                # Optimize portfolio and store the result
                                st.subheader("Recommended Portfolio", divider='orange')
                                portfolio_optimization(int(user_age), investment_horizon, float(investment_amount), income, risk_score, target_risk)
                                st.session_state.optimal_portfolio = optimal_portfolio
                                st.session_state.optimization_run = True
                                 

            
                            start_date = datetime.today() - timedelta(days=3650)
                            st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                            message("Would you like me to display the next 30-day forecasts for the stocks in your portfolio shown above?", is_user=False, avatar_style="avataaars", seed=avatar_url)
                        if st.session_state.optimization_run:
                            if not st.session_state.get('forecast_prompt_shown', False):
                                st.session_state.forecast_prompt_shown = True

                            forecast_selection = st.selectbox("Please select an option from the dropdown menu",
                                                            options=["Select an option", "Yes", "No"],
                                                            index=0,
                                                            key="forecast_selection", help=tooltips["short_term_forecasting"])

                            if forecast_selection == "Yes" and not st.session_state.forecast_yes_clicked:
                                st.toast("Running the short term predictions... Please wait.", icon="⏳")
                                time.sleep(1)
                                st.toast("This may take up to 1-2 minutes...", icon="⏳")
                                time.sleep(15)
                                st.session_state.forecast_yes_clicked = True
                                st.session_state.forecast_no_clicked = False
                                # Display user message
                                st.markdown("<div style='text-align: right; font-weight: bold; color: cyan; font-size: 17px;'>You</div>", unsafe_allow_html=True)
                                message("Yes, I would like to view the forecasts", is_user=True, avatar_style="thumbs", seed=avatar_user)
                                # Display chatbot response
                                st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                                message("Okay, I will process the forecasts now.", is_user=False, avatar_style="avataaars", seed=avatar_url)
                                # Pause for 4 seconds
                                time.sleep(4)
                            elif forecast_selection == "No" and not st.session_state.forecast_no_clicked:
                                st.session_state.forecast_yes_clicked = False
                                st.session_state.forecast_no_clicked = True
                                st.markdown("<div style='text-align: right; font-weight: bold; color: cyan; font-size: 17px;'>You</div>", unsafe_allow_html=True)
                                message("No, I would not like to view the forecasts", is_user=True, avatar_style="thumbs", seed=avatar_user)
                                time.sleep(4)

                            if st.session_state.forecast_yes_clicked and not st.session_state.forecasts_processed:
                                st.subheader("Stock Price Forecasts", divider='violet')
                                optimal_stocks = st.session_state.optimal_portfolio[st.session_state.optimal_portfolio['Asset'].isin(stocks)]['Asset'].tolist()
                                process_multiple_tickers(optimal_stocks, start_date)
                                st.session_state.forecasts_processed = True
                                #st.write("Forecasts have been processed.")

                            if st.session_state.forecasts_processed or st.session_state.forecast_no_clicked:
                                #if not st.session_state.get('monte_prompt_shown', False):
                                st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                                message("Would you like to view the long-term projected returns of the recommended portfolio?", is_user=False, avatar_style="avataaars", seed=avatar_url)
                                    

                                monte_selection = st.selectbox("Please select an option from the dropdown menu",
                                                            options=["Select an option", "Yes", "No"],
                                                            index=0,
                                                            key="monte_selection", help=tooltips["long_term_projection"])

                                if monte_selection == "Yes":
                                    st.toast("Running the long term projections... Please wait.", icon="⏳")
                                    time.sleep(1)
                                    st.toast("This may take a 1 to 2 minutes..", icon="⏳")
                                    time.sleep(15)
                                    st.session_state.monte_yes_clicked = True
                                    st.session_state.monte_no_clicked = False
                                    st.markdown("<div style='text-align: right; font-weight: bold; color: cyan; font-size: 17px;'>You</div>", unsafe_allow_html=True)
                                    message("Yes, I would like to view the long-term projected returns", is_user=True, avatar_style="thumbs", seed=avatar_user)
                                    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                                    message("Okay, I will process the long-term projected simulation.\n\n Please enter the number of years you would like to project for, and to begin the processing, tick the checkbox to run the simulation", is_user=False, avatar_style="avataaars", seed=avatar_url)
                                    time.sleep(4)  # Pause for 4 seconds
                                    st.subheader("Long-term Project Returns Simulations", divider='red')
                                    years_to_simulate = st.text_input('Number of years to simulate:', placeholder='Enter a number between 1 and 30', help=tooltips["number_years"])

                                    if st.checkbox("Run Simulation"):
                                        if years_to_simulate:
                                            try:
                                                years = int(years_to_simulate)
                                                if years < 1 or years > 30:
                                                    raise ValueError
                                            except ValueError:
                                                st.error("Please enter a valid number of years between 1 and 30.")
                                            else:
                                                if st.session_state.optimal_portfolio is not None:
                                                    assets = st.session_state.optimal_portfolio['Asset'].tolist()
                                                    weights = st.session_state.optimal_portfolio['Weight (%)'] / 100
                                                    num_trading_days = years * 252
                                                    num_simulations = 500

                                                    # Initialize progress bar and spinner
                                                    progress_bar = st.progress(0)
                                                    with st.spinner("Running Monte Carlo simulations..."):
                                                        # Placeholder for displaying the progress
                                                        status_text = st.empty()

                                                        summary, sim_returns = monte_carlo_simulation(
                                                            assets, weights, start_date, num_simulations, num_trading_days, progress_bar, status_text
                                                        )

                                                    # Set initial investment based on user input
                                                    initial_investment = float(investment_amount)

                                                    # Use the lower and upper 95% confidence intervals to calculate the range of the possible outcomes
                                                    port_high_return = round(initial_investment * summary[9], 2)
                                                    port_low_return = round(initial_investment * summary[8], 2)

                                                    test = f"""
                                                        There is a 95% chance that an initial investment of **\${initial_investment}** in the portfolio
                                                        over the next **{years} years** will end within the range of: **\${port_low_return} to \${port_high_return}**
                                                        """

                                                    st.write(test)
                                                                                                
                    
                                                    lower_annual_return = (port_low_return / initial_investment) ** (1 / years) - 1
                                                    upper_annual_return = (port_high_return / initial_investment) ** (1 / years) - 1

                                                    st.write(f"Lower Implied Annual Return: {lower_annual_return * 100:.2f}%")
                                                    st.write(f"Upper Implied Annual Return: {upper_annual_return * 100:.2f}%")
                                                    
                                                    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                                                    message("Please kindly hold on, while the simulation graph is being generated", is_user=False, avatar_style="avataaars", seed=avatar_url)

                                                    with st.spinner("Plotting simulations..."):
                                                    # Plot the simulation results
                                                        fig = sim_returns.plot_simulation_fig()
                                                        st.plotly_chart(fig)
                                                    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                                                    message("The simulation has been completed. The results are displayed above. When hovering over the plot, the values e.g. (130, 2.4362) represent the number of trading days and the cumulative return of the portfolio respectively.", is_user=False, avatar_style="avataaars", seed=avatar_url)
                                                    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                                                    message("Thank you for using OptiFi! To create a new portfolio, please restart in a new session or amend your current inputs.", is_user=False, avatar_style="avataaars", seed=avatar_url)
                                                else:
                                                    st.error("Optimal portfolio not found. Please complete the previous steps.")
                                elif monte_selection == "No":
                                    st.session_state.monte_yes_clicked = False
                                    st.session_state.monte_no_clicked = True
                                    message("No, I would not like to view the long-term projected returns", is_user=True, avatar_style="thumbs", seed=avatar_user)
                                    time.sleep(4)
                                    st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                                    message("Thank you for using OptiFi! To create a new portfolio, please restart in a new session or amend your current inputs.", is_user=False, avatar_style="avataaars", seed=avatar_url)
                    else:
                        st.markdown("<div style='text-align: left; font-weight: bold; color: pink;font-size: 17px;'>OptiFi</div>", unsafe_allow_html=True)
                        message("Please answer all questions before submitting.", is_user=False, avatar_style="avataaars", seed=avatar_url)

botfolio()
