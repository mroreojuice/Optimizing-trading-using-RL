"""
Custom stock trading environment.

State:
    (Momentum, Holding)

Momentum:
    0 -> MA1 <= MA5
    1 -> MA1 > MA5

Holding:
    0 -> Cash
    1 -> Holding SPY

Actions:
    0 -> Buy
    1 -> Hold
    2 -> Sell
"""

import numpy as np


class TradingEnvironment:

    def __init__(self, data, initial_cash=100000):

        self.data = data.reset_index(drop=True)

        self.initial_cash = initial_cash

        self.reset()

    def reset(self):

        self.current_step = 0

        self.cash = self.initial_cash

        self.shares = 0

        self.done = False

        return self._get_state()

    def _current_price(self):

        return self.data.loc[self.current_step, "Close"]

    def _portfolio_value(self):

        return self.cash + self.shares * self._current_price()

    def _get_state(self):

        momentum = int(self.data.loc[self.current_step, "Momentum"])

        holding = 1 if self.shares > 0 else 0

        return (momentum, holding)

    def step(self, action):

        if self.done:
            raise Exception("Episode already finished.")

        current_value = self._portfolio_value()

        price = self._current_price()

        # BUY
        if action == 0 and self.shares == 0:

            self.shares = self.cash / price

            self.cash = 0

        # SELL
        elif action == 2 and self.shares > 0:

            self.cash = self.shares * price

            self.shares = 0

        self.current_step += 1

        if self.current_step >= len(self.data) - 1:

            self.done = True

        next_value = self._portfolio_value()

        reward = next_value - current_value

        next_state = self._get_state()

        return next_state, reward, self.done

    def portfolio_value(self):

        return self._portfolio_value()
