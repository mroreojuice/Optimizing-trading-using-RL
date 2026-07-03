"""
Downloads and prepares SPY data.
"""

import yfinance as yf
import pandas as pd


class DataLoader:

    def __init__(
        self,
        ticker="SPY",
        start="2010-01-01",
        end="2021-01-01"
    ):
        self.ticker = ticker
        self.start = start
        self.end = end

    def load_data(self):

        df = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False
        )

        df = df.reset_index()

        return df

    def train_test_split(self, df):

        train = df[
            (df["Date"] >= "2010-01-04")
            &
            (df["Date"] <= "2015-12-31")
        ].copy()

        test = df[
            (df["Date"] >= "2016-01-04")
            &
            (df["Date"] <= "2020-12-30")
        ].copy()

        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        return train, test


if __name__ == "__main__":

    loader = DataLoader()

    data = loader.load_data()

    train, test = loader.train_test_split(data)

    print(train.head())

    print(test.head())
