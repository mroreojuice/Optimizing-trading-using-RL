"""
Technical indicators used by all algorithms.
"""

import pandas as pd
import numpy as np


def add_indicators(df):

    df = df.copy()

    # 1-day moving average
    df["MA1"] = df["Close"].rolling(1).mean()

    # 5-day moving average
    df["MA5"] = df["Close"].rolling(5).mean()

    # Daily return
    df["Return"] = df["Close"].pct_change()

    # 5-day return
    df["Return5"] = df["Close"].pct_change(5)

    # Momentum used in Q-learning
    df["Momentum"] = np.where(
        df["MA1"] > df["MA5"],
        1,
        0
    )

    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":

    from data_loader import DataLoader

    loader = DataLoader()

    df = loader.load_data()

    df = add_indicators(df)

    print(df.head())
