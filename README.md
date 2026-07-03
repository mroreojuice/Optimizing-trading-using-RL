# Optimizing Stock Trading Strategy With Reinforcement Learning

This repository explores the application of Reinforcement Learning (RL) algorithms to optimize stock trading strategies, specifically focusing on capital allocation for the SPY Exchange-Traded Fund (ETF). By formulating the stochastic nature of the stock market as a Markov Decision Process (MDP), this project evaluates and compares the performance of both value-iteration and policy-search methods.

## Abstract
The goal of this project is to maximize cumulative wealth by finding an optimal trading strategy. We define two variations of an MDP (discrete and continuous state spaces) and implement three distinct RL algorithms:
1. **Discrete Q-Learning** (Value Iteration)
2. **Hill Climbing** (Policy Search)
3. **Deep Q-Learning / DQN** (Function Approximation)

The models are benchmarked against standard trading strategies, including "Buy-and-Hold" and the Moving Average Convergence Divergence (MACD) momentum indicator.

---

## Dataset and Features
The models are trained and tested on historical S&P 500 data (using the SPY index fund) pulled from Yahoo Finance. 
* **Training Set:** January 4, 2010 – December 31, 2015
* **Test Set:** January 4, 2016 – December 30, 2020

The core features extracted for state representation rely on market momentum, specifically the relationship between the **1-day average** (current closing price) and the **5-day average** (geometric mean of immediate ratios of closing prices over the past five days).

---

## Problem Formulation (Markov Decision Process)
The trading environment is modeled as an MDP where the agent starts with an initial wealth of $100,000. The action space $\mathcal{A}$ and reward function $\mathcal{R}$ remain consistent across models:
* **Action ($a$):** Buy, Hold, or Sell. Constraints are enforced (e.g., the agent cannot buy if already fully invested, and cannot sell if holding only cash).
* **Reward ($r$):** The change in cumulative wealth when action $a$ is taken at state $s$, resulting in state $s'$.

The State space $\mathcal{S}$ is defined differently based on the algorithm's constraints:

### 1. Discrete MDP (For Q-Learning & Hill Climbing)
The state is a tuple $s = (u, t)$ resulting in a $2 \times 2$ matrix:
* **Momentum ($u$):** $u=1$ indicates upward momentum (5-day average > 1-day average). $u=0$ indicates downward price pressure.
* **Trading Status ($t$):** $t=1$ indicates the agent holds cash and can buy. $t=0$ indicates the agent is holding the stock.

### 2. Continuous MDP (For Deep Q-Learning)
The state is a tuple $s = (r, t)$:
* **Momentum ($r$):** A continuous variable $r \in [-1.0, 1.0]$ representing the exact return difference between the 5-day average and 1-day average.
* **Trading Status ($t$):** Same binary constraint as above.

---

## Algorithms and Methodology

### 1. Discrete Q-Learning
A classic value-iteration algorithm that attempts to find the optimal action-value function $Q^*(s,a)$. The Q-table is updated iteratively using the Bellman equation:

$$
Q^{new}(s_{t},a_{t})=Q^{old}(s_{t},a_{t})+\alpha\cdot(r_{t}+\gamma\cdot \max_{a} Q(s_{t+1},a)-Q^{old}(s_{t},a_{t}))
$$

* **Hyperparameters:** Learning rate $\alpha=0.05$, discount factor $\gamma=0.15$ (to prevent overly narrow decision margins), and an $\epsilon$-greedy exploration rate decaying from $1.0$ to $0.15$.

### 2. Hill Climbing
A direct policy-search method that bypasses value estimation. It generates a random policy, modifies a single action within that policy, and tests the cumulative wealth across the entire training set. If the portfolio value improves, the new policy is adopted. This process guarantees monotonic improvement at each iteration.

### 3. Deep Q-Learning with Experience Replay
To handle the continuous state space without suffering the curse of dimensionality, DQN uses a neural network to approximate the Q-value function. 
* **Architecture:** A 3-layer feed-forward network ($2 \rightarrow 64 \rightarrow 32 \rightarrow 3$). The first two layers use ReLU activations, while the final layer outputs to a softmax transformation (without ReLU to prevent truncating negative values).
* **Optimization:** The network minimizes the Mean Squared Error (MSE) using the Adam optimizer:

$$
MSE=(Q_{target}-Q_{local})^{2}
$$

To stabilize training and avoid oscillation, the algorithm utilizes an **Experience Replay** buffer to sample randomized sequences of $(s_{t}, a_{t}, r, s_{t+1})$ and maintains separate local and target networks.

---

## Results and Discussion

1. **Hill Climbing (The Winner):** The policy-iteration method significantly outperformed both value-iteration methods and standard benchmarks (Buy-and-Hold, MACD). The algorithm converged on an active strategy that flees the market during downward momentum and aggressively re-enters when in cash. It proved to be the most stable approach across different random seeds.

2. **Discrete Q-Learning:**
   Converged upon a simplistic "Buy-and-Hold" strategy (buying at the first sign of upward momentum and holding into perpetuity). While it outperformed the reactive MACD strategy by not missing the market's best trading days, it failed to capitalize on optimal exit points.

3. **Deep Q-Learning (DQN):**
   Produced a non-trivial policy but underperformed compared to the basic Buy-and-Hold strategy. The continuous state space approximation struggled to establish strict preferences, with the network's action values hovering near a uniform distribution.

### Conclusion
For highly stochastic, simplified financial market dynamics, direct policy-search algorithms (Hill Climbing) produce more reliable and profitable results than value-search approximations (Q-Learning and DQN), avoiding the pitfalls of inaccurate value function estimations.
