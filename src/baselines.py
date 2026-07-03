"""
Baseline trading strategies.

1. Buy & Hold
2. MACD (using MA1 and MA5 crossover)

These are used for comparison with RL algorithms.
"""

import matplotlib.pyplot as plt


class BuyAndHold:

    def __init__(self, data, initial_cash=100000):

        self.data = data
        self.initial_cash = initial_cash

    def run(self):

        price0 = self.data.iloc[0]["Close"]

        shares = self.initial_cash / price0

        portfolio = []

        for _, row in self.data.iterrows():

            value = shares * row["Close"]

            portfolio.append(value)

        return portfolio


class MACDStrategy:

    """
    Buy whenever MA1 crosses above MA5.

    Sell whenever MA1 falls below MA5.

    Same assumptions as the paper:
        - either fully invested
        - or fully in cash
    """

    def __init__(self, data, initial_cash=100000):

        self.data = data

        self.initial_cash = initial_cash

    def run(self):

        cash = self.initial_cash

        shares = 0

        portfolio = []

        for _, row in self.data.iterrows():

            price = row["Close"]

            ma1 = row["MA1"]

            ma5 = row["MA5"]

            # BUY

            if ma1 > ma5 and shares == 0:

                shares = cash / price

                cash = 0

            # SELL

            elif ma1 < ma5 and shares > 0:

                cash = shares * price

                shares = 0

            value = cash + shares * price

            portfolio.append(value)

        return portfolio


def plot_portfolio(values, title):

    plt.figure(figsize=(10,5))

    plt.plot(values, linewidth=2)

    plt.title(title)

    plt.xlabel("Trading Days")

    plt.ylabel("Portfolio Value ($)")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def compare_strategies(results):

    """
    Parameters
    ----------
    results : dict

    Example

    {
        "Buy & Hold": buy_values,
        "MACD": macd_values
    }
    """

    plt.figure(figsize=(12,6))

    for name, values in results.items():

        plt.plot(values, label=name)

    plt.title("Portfolio Comparison")

    plt.xlabel("Trading Days")

    plt.ylabel("Portfolio Value ($)")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    from data_loader import DataLoader

    from indicators import add_indicators

    loader = DataLoader()

    df = loader.load_data()

    df = add_indicators(df)

    train, test = loader.train_test_split(df)

    buy = BuyAndHold(test)

    buy_values = buy.run()

    macd = MACDStrategy(test)

    macd_values = macd.run()

    compare_strategies({

        "Buy & Hold": buy_values,

        "MACD": macd_values

    })

    print()

    print("Buy & Hold Final Value : ${:,.2f}".format(buy_values[-1]))

    print("MACD Final Value       : ${:,.2f}".format(macd_values[-1]))
