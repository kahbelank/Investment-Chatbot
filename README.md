# OptiFi Personalized Portfolio Recommendation Chatbot

Welcome to the OptiFi Personalized Portfolio Recommendation Chatbot! This application is designed to provide users with personalized investment recommendations by integrating short-term stock price forecasting and long-term portfolio projections.

---

## Table of Contents

1. [About the Application](#about-the-application)
2. [Features](#features)
3. [Installation Requirements](#installation-requirements)
4. [Setup and Installation](#setup-and-installation)
5. [Running the Application](#running-the-application)
6. [Usage](#usage)
7. [Deployment](#deployment)
8. [Support](#support)
9. [License](#license)

---

## About the Application

OptiFi is a personalized portfolio recommendation chatbot designed to help novice investors make informed investment decisions. The application integrates short-term stock price forecasting with long-term portfolio projections, providing users with a comprehensive strategy that balances immediate market opportunities with long-term financial goals.

## Features

- **User Profile Setup**: Collects user information including investment preferences and risk tolerance.
- **Personalized Recommendations**: Provides tailored investment portfolio recommendations.
- **Stock Price Forecasting**: Forecasts stock prices for the next 30 days using advanced machine learning models.
- **Long-term Portfolio Projections**: Uses Monte Carlo simulations to project long-term returns.
- **Educational Resources**: Offers educational content to help users understand investment strategies.
- **Interactive Chatbot**: User-friendly interface for easy interaction and data input.

## Installation Requirements

- Python 3.7 or higher
- Streamlit
- Ngrok
- Google Colab (for model retraining)
- Required Python libraries: pandas, numpy, scikit-learn, prophet, tensorflow, matplotlib, seaborn

## Setup and Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/kahbelank/Investment-Chatbot.git 
    cd Investment-Chatbot
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

1. **(For Developer to Deploy) Start Ngrok**:
    ```sh
    ngrok http 8501
    ```
   - Note the forwarding URL provided by Ngrok (e.g., `http://1234abcd.ngrok.io`).

2. **Run the Streamlit Application**:
    ```sh
    streamlit run BotFolio_test.py
    ```

3. **Access the Application**:
   - Open a web browser and navigate to the Ngrok URL (e.g., `http://1234abcd.ngrok.io`).
   - Access the URL provided once the app is run (application should open automatically)

## Usage

1. **Initial Profile Setup**:
   - Input your investment preferences and risk tolerance through the chatbot interface.

2. **Get Recommendations**:
   - View your personalized investment portfolio recommendations.

3. **Stock Price Forecasting**:
   - Use the stock price forecasting tool to see projections for the next 30 days.

4. **Long-term Projections**:
   - View long-term projected returns for your recommended portfolio using Monte Carlo simulations.

5. **Educational Resources**:
   - Access educational content to understand various investment strategies.

## Deployment

For deployment, the application is hosted locally on your machine, and remote access is provided via Ngrok. The current setup bypasses the limitations of Streamlit Community Cloud by utilizing local computational resources for better performance.

## Support

For support and queries, please contact kahbelannk@gmail.com or open an issue on the GitHub repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Follow these instructions to set up and run the OptiFi Personalized Portfolio Recommendation Chatbot on your local machine. Enjoy making informed investment decisions with the help of advanced machine learning models and comprehensive financial analysis tools!
