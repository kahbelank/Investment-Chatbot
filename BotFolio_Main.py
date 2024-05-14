# import libraries 
from tkinter import Image
import streamlit as st
from streamlit_chat import message
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

from BotFolio_Func import user_age_verification, determine_weights, allocate_portfolio, display_portfolio_allocation,process_multiple_tickers
from risk_tolerance_data import RISK_QUESTIONS, OPTIONS, SCORES 

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

def display_risk_score_chart(risk_score):
    # Determine the risk tolerance level and threshold color
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
        height=300  # Adjust the height to make the chart smaller
    )

    return fig


def botfolio():
    user_age = st.text_input('Age', placeholder='Enter your age').strip()
    investment_horizon = st.selectbox(
        "Select your investment horizon",
        options=[
            "Short-Term",
            "Medium-Term",
            "Long-Term",
        ]
    )
    investment_amount = st.text_input('Investment Amount', placeholder='Enter your investment amount in USD').strip()
    income = st.selectbox(
        "Select your annual income range:",
        options=[
            "Less than $30,000",
            "$30,000 - $49,999",
            "$50,000 - $99,999",
            "$100,000 - $149,999",
            "$150,000 - $199,999",
            "More than $200,000"
        ]
    )

    investment_horizon_map ={
        "Short-Term": 1,
        "Medium-Term":2,
        "Long-Term":3,
    }

    # Mapping income to numeric values for validation and processing
    income_map = {
        "Less than $30,000": 1,
        "$30,000 - $49,999": 2,
        "$50,000 - $99,999": 3,
        "$100,000 - $149,999": 4,
        "$150,000 - $199,999": 5,
        "More than $200,000": 6
    }

    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
        st.session_state.answers = [None] * len(RISK_QUESTIONS)
    
    # Validating user inputs
    if user_age and investment_horizon in investment_horizon_map and investment_amount and income in income_map:
        user_details = (
                f"I am {user_age} years old\n"
                f"with an annual income of {income}\n"
                f"and I am willing to invest ${investment_amount}\n"
                f"for an investment horizon of {investment_horizon}"
        )
        message(user_details, is_user=True, seed=1, key=12)

        age_verification_message = user_age_verification(user_age)
        message(age_verification_message, seed=21, key=17)

        # if "Enjoy the use" in age_verification_message:
        #         current_question = st.session_state.current_question
        #         st.write(RISK_QUESTIONS[current_question])

        #         # Display the radio buttons for the current question
        #         answer = st.radio(
        #             "Select an option:",
        #             OPTIONS[current_question],
        #             index=OPTIONS[current_question].index(st.session_state.answers[current_question]) if st.session_state.answers[current_question] is not None else None,
        #             key=f"q{current_question}"
        #         )

        #         if answer:
        #             st.session_state.answers[current_question] = answer

        #         col1, col2, col3 = st.columns([1, 1, 2])

        #         with col1:
        #             if st.button("Previous"):
        #                 if current_question > 0:
        #                     st.session_state.current_question -= 1
        #                     st.experimental_rerun()

        #         with col2:
        #             if st.button("Next"):
        #                 if current_question < len(RISK_QUESTIONS) - 1:
        #                     st.session_state.current_question += 1
        #                     st.experimental_rerun()

        #         with col3:
        #             if st.button("Submit"):
        #                 if None not in st.session_state.answers:
        #                     # Calculate the risk score
        #                     risk_score = 0
        #                     for i, answer in enumerate(st.session_state.answers):
        #                         index = OPTIONS[i].index(answer)
        #                         risk_score += SCORES[i][index]

        #                     # Determine the risk tolerance level
        #                     if risk_score >= 33:
        #                         risk_level = "High tolerance for risk"
        #                     elif risk_score >= 29:
        #                         risk_level = "Above-average tolerance for risk"
        #                     elif risk_score >= 23:
        #                         risk_level = "Average/moderate tolerance for risk"
        #                     elif risk_score >= 19:
        #                         risk_level = "Below-average tolerance for risk"
        #                     else:
        #                         risk_level = "Low tolerance for risk"

        #                     message(f"Your risk tolerance score is {risk_score}, indicating a {risk_level}.", seed=21, key=24)
        #                 else:
        #                     message("Please answer all questions before submitting.", seed=21, key=25)
        if "Enjoy the use" in age_verification_message:
            current_question = st.session_state.current_question

            st.write(RISK_QUESTIONS[current_question])

            # Display the radio buttons for the current question
            if st.session_state.answers[current_question] is not None:
                try:
                    answer = st.radio(
                        "Select an option:",
                        OPTIONS[current_question],
                        index=OPTIONS[current_question].index(st.session_state.answers[current_question]),
                        key=f"q{current_question}"
                    )
                except ValueError:
                    # Handle case where the saved answer is not in the options
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

            # Conditionally display navigation buttons
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

            # Display the Submit button separately outside of columns
            if current_question == len(RISK_QUESTIONS) - 1:
                if st.button("Submit"):
                    if None not in st.session_state.answers:
                        # Calculate the risk score
                        risk_score = 0
                        for i, answer in enumerate(st.session_state.answers):
                            index = OPTIONS[i].index(answer)
                            risk_score += SCORES[i][index]

                        # Determine the risk tolerance level
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
                       # Display the gauge chart
                        fig = display_risk_score_chart(risk_score)
                        st.plotly_chart(fig)
                    else:
                        message("Please answer all questions before submitting.", seed=21, key=25)

                # data = {
                #     "Portfolio Type": ["Aggressive", "Moderate", "Conservative"],
                #     "Risk Tolerance": [
                #         "Comfortable with significant fluctuations in portfolio value.",
                #         "Comfortable with moderate fluctuations in portfolio value.",
                #         "Preferring minimal fluctuations in portfolio value."
                #     ],
                #     "Investment Goals": [
                #         "Maximizing long-term wealth accumulation.",
                #         "Balanced approach to wealth accumulation and capital preservation.",
                #         "Preservation of capital and wealth."
                #     ],
                #     "Examples": [
                #         "Young professional with stable income, willing to endure short-term losses for long-term growth potential.",
                #         "Mid-career individual with balanced financial situation, seeking a balance between growth and stability.",
                #         "Retiree living off investment income, focused on capital preservation and income generation."
                #     ]
                # }
                # df = pd.DataFrame(data)
                # message("Based on the table below, please enter a portfolio type that matches your risk tolerance (Aggressive Portfolio, Moderate Portfolio, Conservative Portfolio)", seed=21, key=18)
                # styled_df = df.style.apply(style_specific_rows, axis=1)
                # st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

                # portfolio_type = st.text_input(' ', placeholder='Enter portfolio type').strip().lower()
                # valid_portfolio_types = ['aggressive portfolio', 'moderate portfolio', 'conservative portfolio']

                # if portfolio_type:
                #     if portfolio_type in valid_portfolio_types:
                #         if portfolio_type == 'aggressive portfolio':
                #             df = pd.DataFrame({
                #                 'Stocks': ['NVDA', 'TSLA', 'AMZN', 'META', 'ADBE', 'AVGO', 'LLY'],
                #                 'Bonds': ['10yr Treasury Yield', '-', '-', '-', '-', '-', '-'],
                #                 'REITs': ['WELL', '-', '-', '-', '-', '-', '-']
                #             })
                #             message('Your Aggressive Portfolio contains the following assets:', seed=21, key=19)
                #             st.table(df)

                #             tickers = ['NVDA', 'TSLA', 'AMZN', 'META', 'ADBE', 'AVGO', 'LLY']
                #             start_date = datetime.today() - timedelta(days=3650)

                #             determine_weights(user_age)
                #             allocate_portfolio(investment_amount)
                #             display_portfolio_allocation(portfolio_type)

                #             message("Would you like me to display forecasts of each asset in your portfolio?", seed=21, key=20)
                #             user_input = st.text_input(' ', placeholder='Display forecasts? (enter yes/no)').strip().lower()

                #             if user_input == 'yes':
                #                 process_multiple_tickers(tickers, start_date, user_input)

                #             message('If you would wish to view forecasts for other assets not within the portfolio, input their ticker symbols (e.g JNJ, NVDA, TSLA)', seed=21, key=21)
                #             user_input = st.text_input(' ', placeholder='Ticker Symbols (TSLA, NVDA, AAPL, MSFT)').strip()
                #             message(user_input, is_user=True, seed=1, key=22)

botfolio()