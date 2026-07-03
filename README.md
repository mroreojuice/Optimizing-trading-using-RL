# Optimizing Stock Trading Strategies: A Reinforcement Learning Approach

This repository explores the theoretical formulation and practical implementation of Reinforcement Learning (RL) algorithms for stock market trading. By modeling market dynamics as a Markov Decision Process (MDP), this project evaluates the convergence and profitability of both value-iteration and policy-search methods.

## Abstract
The stock market is a highly stochastic environment. This project abstracts the trading of a single asset (SPY ETF) into discrete and continuous state spaces to test three distinct optimization algorithms:
1. **Tabular Q-Learning:** A value-iteration approach utilizing the Bellman Optimality Equation.
2. **Hill Climbing:** A derivative-free, policy-search algorithm.
3. **Deep Q-Networks (DQN):** A continuous-state function approximation method using a neural network and experience replay.

## Mathematical Formulation

The trading environment is defined as an MDP tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. 

The objective of the value-based agents is to find the optimal action-value function $Q^*(s,a)$, which satisfies the Bellman equation:
$$Q^*(s,a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t = s, A_t = a \right]$$

For the continuous state space, the DQN algorithm approximates this function using a neural network parameterized by $\theta$, minimizing the Temporal Difference (TD) error:
$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

## Repository Structure
* `src/environment.py`: The custom Gymnasium environment defining the transition dynamics and reward function.
* `notebooks/`: Interactive Jupyter Notebooks detailing the mathematical proofs, algorithm implementations, and final evaluations.
