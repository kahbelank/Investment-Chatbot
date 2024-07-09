import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
import pytz
from scipy.stats import t
import plotly.graph_objects as go
import plotly.express as px

class PortfolioSimulator:
    def __init__(self, portfolio_data, annual_returns, weights=None, num_simulation=1000, num_trading_days=252, progress_bar=None, status_text=None):
        if not isinstance(portfolio_data, pd.DataFrame):
            raise TypeError("portfolio_data must be a Pandas DataFrame")
        
        num_assets = len(portfolio_data.columns.get_level_values(0).unique())
        
        if weights is None:
            weights = np.array([1.0 / num_assets for _ in range(num_assets)])
        else:
            weights = np.array(weights)
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError("Sum of portfolio weights must equal one.")

        if 'close' in portfolio_data.columns.get_level_values(1).unique():
            close_prices = portfolio_data.xs('close', level=1, axis=1)
            log_returns = np.log(close_prices / close_prices.shift(1))
            tickers = portfolio_data.columns.get_level_values(0).unique()
            column_names = [(ticker, "log_return") for ticker in tickers]
            log_returns.columns = pd.MultiIndex.from_tuples(column_names)
            portfolio_data = portfolio_data.merge(log_returns, left_index=True, right_index=True).reindex(columns=tickers, level=0) 

        # Convert annual percentage returns to daily log returns
        self.daily_log_returns = np.log(1 + np.array(list(annual_returns.values())) / 100) / 252
        # Handle outliers in log returns
        self.portfolio_data = self.handle_outliers(portfolio_data)
        self.annual_returns = annual_returns
        self.weights = weights
        self.num_simulation = num_simulation
        self.num_trading_days = num_trading_days
        self.simulated_return = pd.DataFrame()
        self.progress_bar = progress_bar
        self.status_text = status_text

    def handle_outliers(self, data, limits=(0.01, 0.99)):
        """ Cap the outliers in log returns within specified quantile limits directly in the DataFrame. """
        log_returns = data.xs('log_return', level=1, axis=1)
        for column in log_returns.columns:
            lower_quantile = log_returns[column].quantile(limits[0])
            upper_quantile = log_returns[column].quantile(limits[1])
            data.loc[:, (column, 'log_return')] = log_returns[column].clip(lower=lower_quantile, upper=upper_quantile)
        return data

    def calc_cumulative_return(self):
        last_prices = self.portfolio_data.xs('close', level=1, axis=1).iloc[-1].values
        log_returns = self.portfolio_data.xs('log_return', level=1, axis=1)
        
        mean_returns = self.daily_log_returns
        std_returns = log_returns.std().values
        all_simulations = []

        for n in range(self.num_simulation):
            if n % 10 == 0:
                print(f"Running Monte Carlo simulation number {n}.")
            simvals = [np.log(last_prices)]
            for _ in range(self.num_trading_days):
                simulated_log_returns = np.random.normal(mean_returns, std_returns)
                simvals.append(simvals[-1] + simulated_log_returns)
            
            simvals = np.exp(simvals)  # Convert log prices back to normal prices
            sim_df = pd.DataFrame(simvals).pct_change().dot(self.weights)
            all_simulations.append((1 + sim_df.fillna(0)).cumprod().to_frame(name=n))


            if self.progress_bar and self.status_text:
                progress = (n + 1) / self.num_simulation
                self.progress_bar.progress(progress)
                self.status_text.text(f"Running Monte Carlo simulations... {n + 1}/{self.num_simulation}")

        portfolio_cumulative_returns = pd.concat(all_simulations, axis=1)
        self.simulated_return = portfolio_cumulative_returns
        self.confidence_interval = portfolio_cumulative_returns.iloc[-1].quantile(q=[0.025, 0.975])
        
        return portfolio_cumulative_returns

    def plot_simulation(self):
        if self.simulated_return.empty:
            self.calc_cumulative_return()
        plot_title = f"{self.num_simulation} Simulations of Cumulative Portfolio Return Trajectories Over the Next {self.num_trading_days} Trading Days."
        return self.simulated_return.plot(legend=None, title=plot_title)
    
    def plot_simulation_fig(self):
        if self.simulated_return.empty:
            self.calc_cumulative_return()

        fig = go.Figure()

        # Use Plotly Express color sequence for better visibility
        color_sequence = px.colors.qualitative.Plotly

        for i in range(len(self.simulated_return.columns)):
            fig.add_trace(go.Scatter(
                x=self.simulated_return.index,
                y=self.simulated_return.iloc[:, i],
                mode='lines',
                line=dict(width=1, color=color_sequence[i % len(color_sequence)]),
                opacity=0.7,  # Slightly increase the opacity for better visibility
                showlegend=False
            ))

        plot_title = f"{self.num_simulation} Simulations of Cumulative Portfolio Return Trajectories Over the Next {self.num_trading_days} Trading Days."
        fig.update_layout(
            title=plot_title,
            xaxis_title='Trading Days',
            yaxis_title='Cumulative Return',
            template='plotly_dark',  # Use a dark template for better contrast
            font=dict(size=12),
            title_font=dict(size=16)
        )

        return fig

    def plot_distribution(self):
        if self.simulated_return.empty:
            self.calc_cumulative_return()
        plot_title = "Distribution of Final Cumulative Returns Across All Simulations"
        plt = self.simulated_return.iloc[-1].plot(kind='hist', bins=10, density=True, title=plot_title)
        plt.axvline(self.confidence_interval.iloc[0], color='r')
        plt.axvline(self.confidence_interval.iloc[1], color='r')
        return plt

    def summarize_cumulative_return(self):
        if self.simulated_return.empty:
            self.calc_cumulative_return()
        metrics = self.simulated_return.iloc[-1].describe()
        ci_series = self.confidence_interval
        ci_series.index = ["95% CI Lower", "95% CI Upper"]
        return metrics.concat(ci_series)

