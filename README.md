# Optimizing Stock Trading Strategy using Reinforcement Learning

A complete implementation of the Stanford CS229 project:

> Optimizing Stock Trading Strategy With Reinforcement Learning

This repository implements three reinforcement learning algorithms for trading SPY.

## Algorithms

- Buy & Hold
- MACD
- Tabular Q-Learning
- Hill Climbing
- Deep Q Network (DQN)

## Dataset

Yahoo Finance SPY ETF

Training:
2010-01-04 to 2015-12-31

Testing:
2016-01-04 to 2020-12-30

## Libraries

- NumPy
- Pandas
- Matplotlib
- PyTorch
- yfinance

No RL libraries such as Stable Baselines or Gym are used.

## Folder Structure

```
RL-Stock-Trading/

src/
    data_loader.py
    indicators.py
    environment.py
    baselines.py
    qlearning.py
    hill_climbing.py
    replay_buffer.py
    models.py
    dqn.py
    trainer.py
    evaluate.py

figures/

notebooks/

README.md
requirements.txt
```

Run the notebook after installing the dependencies.

```
pip install -r requirements.txt
```
