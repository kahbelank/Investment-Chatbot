import numpy as np
import pandas as pd
import os
import datetime as dt
import pytz

class PortfolioSimulator:
    def __init__(self, portfolio_data, weights="", num_simulation=1000, num_trading_days=252):
        if not isinstance(portfolio_data, pd.DataFrame):
            raise TypeError("portfolio_data must be a Pandas DataFrame")
        
        if weights == "":
            num_stocks = len(portfolio_data.columns.get_level_values(0).unique())
            weights = [1.0 / num_stocks for _ in range(num_stocks)]
        else:
            if round(sum(weights), 2) < .99:
                raise AttributeError("Sum of portfolio weights must equal one.")

        if 'close' in portfolio_data.columns.get_level_values(1).unique():
            close_prices = portfolio_data.xs('close', level=1, axis=1)
            log_returns = np.log(close_prices / close_prices.shift(1))
            tickers = portfolio_data.columns.get_level_values(0).unique()
            column_names = [(ticker, "log_return") for ticker in tickers]
            log_returns.columns = pd.MultiIndex.from_tuples(column_names)
            portfolio_data = portfolio_data.merge(log_returns,left_index=True,right_index=True).reindex(columns=tickers,level=0) 

        # Handle outliers in log returns
        self.portfolio_data = self.handle_outliers(portfolio_data)

        self.weights = weights
        self.nSim = num_simulation
        self.nTrading = num_trading_days
        self.simulated_return = pd.DataFrame()

    def handle_outliers(self, data, limits=(0.01, 0.99)):
        """ Cap the outliers in log returns within specified quantile limits directly in the DataFrame. """
        log_returns = data.xs('log_return', level=1, axis=1)
        # Apply the clipping directly to the DataFrame
        for column in log_returns.columns:
            lower_quantile = log_returns[column].quantile(limits[0])
            upper_quantile = log_returns[column].quantile(limits[1])
            # Use .loc to ensure that the changes are done in place on the original DataFrame
            data.loc[:, (column, 'log_return')] = log_returns[column].clip(lower=lower_quantile, upper=upper_quantile)
        return data

    # def calc_cumulative_return(self):
    #     last_prices = self.portfolio_data.xs('close', level=1, axis=1).iloc[-1].values
    #     log_returns = self.portfolio_data.xs('log_return', level=1, axis=1)
    #     mean_returns = log_returns.mean().values
    #     std_returns = log_returns.std().values
        
    #     portfolio_cumulative_returns = pd.DataFrame()
        
    #     for n in range(self.nSim):
    #         if n % 10 == 0:
    #             print(f"Running Monte Carlo simulation number {n}.")
    #         simvals = [np.log(last_prices)]
    #         for _ in range(self.nTrading):
    #             simulated_log_returns = np.random.normal(mean_returns, std_returns)
    #             simvals.append(simvals[-1] + simulated_log_returns)
            
    #         simvals = np.exp(simvals)  # Convert log prices back to normal prices
    #         sim_df = pd.DataFrame(simvals).pct_change().dot(self.weights)
    #         portfolio_cumulative_returns[n] = (1 + sim_df.fillna(0)).cumprod()

    #     self.simulated_return = portfolio_cumulative_returns
    #     self.confidence_interval = portfolio_cumulative_returns.iloc[-1].quantile(q=[0.025, 0.975])
        
    #     return portfolio_cumulative_returns

    def calc_cumulative_return(self):
        last_prices = self.portfolio_data.xs('close', level=1, axis=1).iloc[-1].values
        log_returns = self.portfolio_data.xs('log_return', level=1, axis=1)
        mean_returns = log_returns.mean().values - 0.5 * log_returns.var().values  # Adjust for long-term drift
        std_returns = log_returns.std().values

        portfolio_cumulative_returns = pd.DataFrame()

        for n in range(self.nSim):
            simvals = [np.log(last_prices)]
            for _ in range(self.nTrading):
                simulated_log_returns = np.random.normal(mean_returns, std_returns)
                simvals.append(simvals[-1] + simulated_log_returns)

            simvals = np.exp(simvals)  # Convert log prices back to normal prices
            sim_df = pd.DataFrame(simvals).pct_change().dot(self.weights)
            portfolio_cumulative_returns[n] = (1 + sim_df.fillna(0)).cumprod()
            
        self.simulated_return = portfolio_cumulative_returns
        self.confidence_interval = portfolio_cumulative_returns.iloc[-1].quantile(q=[0.025, 0.975])

        return portfolio_cumulative_returns


    def plot_simulation(self):
        if self.simulated_return.empty:
            self.calc_cumulative_return()
        plot_title = f"{self.nSim} Simulations of Cumulative Portfolio Return Trajectories Over the Next {self.nTrading} Trading Days."
        return self.simulated_return.plot(legend=None, title=plot_title)

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
        return metrics.append(ci_series)
