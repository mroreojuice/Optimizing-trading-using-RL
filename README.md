# Optimizing Stock Trading Strategy with Reinforcement Learning

A complete implementation of the Stanford CS229 project:

> Optimizing Stock Trading Strategy with Reinforcement Learning

This project reproduces and extends the paper by implementing reinforcement learning algorithms for stock trading from scratch.

---

## Project Objective

The objective is to maximize cumulative portfolio wealth while trading the SPY ETF using reinforcement learning.

We compare:

- Buy & Hold
- MACD Strategy
- Tabular Q-Learning
- Hill Climbing
- Deep Q-Network (DQN)

---

## Dataset

- Source: Yahoo Finance
- Ticker: SPY
- Training Period: January 2010 – December 2015
- Testing Period: January 2016 – December 2020

---

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- PyTorch
- yfinance

No reinforcement learning libraries (Gym, Stable Baselines, RLlib, etc.) are used.

---

## Repository Structure

```
RL-Stock-Trading/

README.md
requirements.txt
.gitignore

notebooks/
    Stock_Trading_RL.ipynb

src/

figures/

data/
```

---

## Algorithms

- Buy & Hold
- MACD
- Tabular Q-Learning
- Hill Climbing
- Deep Q-Network

---

## Results

The notebook generates:

- Stock price visualization
- Technical indicators
- Portfolio value curves
- Reward curves
- DQN loss curve
- Final comparison plots

---

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open:

```
notebooks/Stock_Trading_RL.ipynb
```

Run all cells from top to bottom.

---

## License

MIT License
