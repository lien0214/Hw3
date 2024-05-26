"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]
data = pd.DataFrame()

# Fetch the data for each stock and concatenate it to the `data` DataFrame
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01")
    raw["Symbol"] = asset
    data = pd.concat([data, raw], axis=0)

# Initialize df and df_returns
Bdf = portfolio_data = data.pivot_table(
    index="Date", columns="Symbol", values="Adj Close"
)
df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


import pandas as pd
import numpy as np
import gurobipy as gp

class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=20, gamma=0.01, momentum_lookback=3, target_volatility=0.1):
        # Initialize with price data, asset to exclude, and various parameters for calculations
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma
        self.momentum_lookback = momentum_lookback
        self.target_volatility = target_volatility

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Prepare a DataFrame to store portfolio weights
        self.portfolio_weights = pd.DataFrame(index=self.price.index, columns=self.price.columns)

        # Calculate the portfolio weights for each time period
        for i in range(self.lookback + 1, len(self.price)):
            # Extract the return data for the lookback period
            R_n = self.returns.copy()[assets].iloc[i - self.lookback : i]
            # Calculate the momentum for the current period
            momentum = self.calculate_momentum(i, assets)
            # Perform mean-variance optimization to get weights
            weights = self.mv_opt(R_n, momentum, self.gamma)
            # Adjust weights based on the target volatility
            vol_adjusted_weights = self.volatility_targeting(weights, R_n)
            # Store the calculated weights
            self.portfolio_weights.loc[self.price.index[i], assets] = vol_adjusted_weights

        # Fill forward and handle any remaining NaN values
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_momentum(self, index, assets):
        # Calculate momentum as the average percentage change over the momentum lookback period
        momentum = self.price[assets].iloc[index - self.momentum_lookback:index].pct_change().mean()
        return momentum

    def mv_opt(self, R_n, momentum, gamma):
        # Calculate the covariance matrix and mean returns adjusted for momentum
        Sigma = R_n.cov().values
        mu = R_n.mean().values + momentum.values
        n = len(R_n.columns)

        # Set up the optimization environment and model
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                # Define the optimization variables (portfolio weights)
                w = model.addMVar(n, name="w")
                # Define the portfolio return and variance
                portfolio_return = mu @ w
                portfolio_variance = w @ Sigma @ w
                # Set the objective to maximize return adjusted for risk
                objective = portfolio_return - (gamma / 2) * portfolio_variance
                model.setObjective(objective, gp.GRB.MAXIMIZE)
                # Add constraints: weights sum to 1 and no short selling
                model.addConstr(w.sum() == 1, name="budget")
                model.addConstr(w >= 0, name="no_short")
                
                model.optimize()

                # Check for optimal solution and return normalized weights
                if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                    weights = w.X
                    return weights / weights.sum()  # Normalize to ensure the weights sum to 1
                else:
                    return np.zeros(n)

    def volatility_targeting(self, weights, R_n):
        # Calculate the portfolio volatility
        portfolio_volatility = np.sqrt(weights @ R_n.cov().values @ weights)
        # Scale weights to target the desired volatility
        scaling_factor = self.target_volatility / portfolio_volatility
        scaled_weights = weights * scaling_factor

        # Ensure the scaled weights sum to 1 or less
        if scaled_weights.sum() > 1:
            scaled_weights = scaled_weights / scaled_weights.sum()

        return scaled_weights

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns based on the weights
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Assignment Judge

The following functions will help check your solution.
"""


class AssignmentJudge:
    def __init__(self):
        self.mp = MyPortfolio(df, "SPY").get_results()
        self.Bmp = MyPortfolio(Bdf, "SPY").get_results()

    def plot_performance(self, price, strategy):
        # Plot cumulative returns
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label=f"MyPortfolio")

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()
        return None

    def plot_allocation(self, df_weights):
        df_weights = df_weights.fillna(0).ffill()

        # long only
        df_weights[df_weights < 0] = 0

        # Plotting
        _, ax = plt.subplots()
        df_weights.plot.area(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.set_title("Asset Allocation Over Time")
        plt.show()
        return None

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame()
        returns = price.pct_change().fillna(0)
        df_bl["SPY"] = returns["SPY"]
        df_bl[f"MP"] = pd.to_numeric(strategy[1]["Portfolio"], errors="coerce")

        qs.reports.metrics(df_bl, mode="full", display=show)

        sharpe_ratio = qs.stats.sharpe(df_bl)

        return sharpe_ratio

    def cumulative_product(self, dataframe):
        (1 + dataframe.pct_change().fillna(0)).cumprod().plot()

    def check_sharp_ratio_greater_than_one(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if self.report_metrics(df, self.mp)[1] > 1:
            print("Problem 4.1 Success - Get 10 points")
            return 10
        else:
            print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if (
            self.report_metrics(Bdf, self.Bmp)[1]
            > self.report_metrics(Bdf, self.Bmp)[0]
        ):
            print("Problem 4.2 Success - Get 10 points")
            return 10
        else:
            print("Problem 4.2 Fail")
        return 0

    def check_portfolio_position(self, portfolio_weights):
        if (portfolio_weights.sum(axis=1) <= 1.01).all():
            return True
        print("Portfolio Position Exceeds 1. No Leverage.")
        return False

    def check_all_answer(self):
        score = 0
        score += self.check_sharp_ratio_greater_than_one()
        score += self.check_sharp_ratio_greater_than_spy()
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()

    if args.score:
        if ("one" in args.score) or ("spy" in args.score):
            if "one" in args.score:
                judge.check_sharp_ratio_greater_than_one()
            if "spy" in args.score:
                judge.check_sharp_ratio_greater_than_spy()
        elif "all" in args.score:
            print(f"==> totoal Score = {judge.check_all_answer()} <==")

    if args.allocation:
        if "mp" in args.allocation:
            judge.plot_allocation(judge.mp[0])
        if "bmp" in args.allocation:
            judge.plot_allocation(judge.Bmp[0])

    if args.performance:
        if "mp" in args.performance:
            judge.plot_performance(df, judge.mp)
        if "bmp" in args.performance:
            judge.plot_performance(Bdf, judge.Bmp)

    if args.report:
        if "mp" in args.report:
            judge.report_metrics(df, judge.mp, show=True)
        if "bmp" in args.report:
            judge.report_metrics(Bdf, judge.Bmp, show=True)

    if args.cumulative:
        if "mp" in args.cumulative:
            judge.cumulative_product(df)
        if "bmp" in args.cumulative:
            judge.cumulative_product(Bdf)
