# Optimizing Stock Trading Strategy With Reinforcement Learning

This repository presents a theoretical investigation into the application of Reinforcement Learning (RL) for optimizing capital allocation in the stock market. Specifically, the project models the trading of a single asset (the SPY Exchange-Traded Fund) as a Markov Decision Process (MDP).

By abstracting financial market momentum into both discrete and continuous state spaces, this research evaluates the convergence properties, stability, and empirical profitability of value-iteration methods (Tabular Q-Learning), policy-search methods (Hill Climbing), and function approximation techniques (Deep Q-Networks).

## Abstract

Given the highly stochastic nature of financial markets, finding an optimal trading policy $\pi^{*}$ requires an agent to navigate sequential decision-making under uncertainty. This project examines three distinct algorithmic approaches to solve this optimization problem:
1. **Tabular Q-Learning:** A classical, model-free value iteration algorithm.
2. **Hill Climbing:** A derivative-free, direct policy-search algorithm.
3. **Deep Q-Networks (DQN):** A continuous-state approximation method utilizing neural networks and experience replay.

The empirical results demonstrate that direct policy-search methods exhibit superior stability and cumulative wealth generation compared to value-estimation approaches when operating in constrained, high-variance financial environments.

---

## 1. Formalizing the Markov Decision Process (MDP)

The trading environment is formulated as a discrete-time MDP defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. The agent initializes with a portfolio value of $100,000. 

### Action Space ($\mathcal{A}$)
At any timestep $t$, the agent selects an action $a_{t} \in \{Buy, Hold, Sell\}$. The action space is strictly bounded by the state constraints: an agent cannot execute a $Buy$ order if already fully allocated to the asset, nor can it execute a $Sell$ order if holding a pure cash position.

### Reward Function ($\mathcal{R}$)
The reward $r_{t}$ is defined as the absolute change in cumulative wealth following the execution of action $a_{t}$ and the subsequent transition from state $s_{t}$ to $s_{t+1}$.

### State Space ($\mathcal{S}$) Formulations
To accommodate the specific requirements of the chosen algorithms, the state space is modeled in two distinct variations based on market momentum indicators (comparing the 1-day average closing price against the 5-day geometric mean).

**A. Discrete State Space (For Q-Learning and Hill Climbing)**
The state is mapped to a finite $2 \times 2$ matrix, $s = (u, t)$:
* **$u \in \{0, 1\}$:** A binary momentum indicator. $u=1$ denotes upward momentum (5-day average $>$ 1-day average), and $u=0$ denotes downward pressure.
* **$t \in \{0, 1\}$:** A binary allocation constraint. $t=1$ indicates liquid cash (ability to buy), and $t=0$ indicates full asset allocation.

**B. Continuous State Space (For Deep Q-Learning)**
To prevent information loss through discretization, the state is mapped as $s = (r, t)$:
* **$r \in [-1.0, 1.0]$:** A bounded continuous variable representing the exact ratio of return difference between the 5-day and 1-day averages.
* **$t \in \{0, 1\}$:** The allocation constraint remains binary.

---

## 2. Algorithmic Implementations

### 2.1 Tabular Q-Learning (Value Iteration)
The objective of Q-learning is to estimate the optimal action-value function $Q^{*}(s,a)$, defined as the maximum expected discounted return. By the Bellman Optimality Equation, the iterative update rule is defined as:

$$
Q_{new}(s_{t},a_{t}) = Q_{old}(s_{t},a_{t}) + \alpha \cdot \left( r_{t} + \gamma \cdot \max_{a} Q(s_{t+1},a) - Q_{old}(s_{t},a_{t}) \right)
$$

**Theoretical Constraints:** * A highly conservative discount factor ($\gamma = 0.15$) was required to achieve convergence; higher values resulted in overly narrow decision margins within the Q-matrix.
* The algorithm utilized a small step size ($\alpha = 0.05$) and an $\epsilon$-greedy exploration rate decaying to an asymptotic lower bound of $0.15$.

### 2.2 Hill Climbing (Policy Search)
Rather than estimating the value of being in a state, this algorithm directly optimizes the policy $\pi_{\theta}(a|s)$ by traversing the policy space. 
* The algorithm initializes a random policy and iteratively perturbs a single action mapping.
* The modified policy is evaluated across the entire training dataset. If the objective function (ending portfolio value) improves, the update is accepted.
* This ensures a monotonically improving policy at each iteration, avoiding the estimation errors inherent in high-variance value functions.

### 2.3 Deep Q-Networks (Function Approximation)
To operate within the continuous state space $\mathcal{S} \subset \mathbb{R}^{2}$, DQN utilizes a non-linear function approximator (a multi-layer perceptron) parameterized by weights $\theta$ to estimate $Q(s, a; \theta) \approx Q^{*}(s,a)$.

**Network Architecture and Optimization:**
* The architecture maps $2 \rightarrow 64 \rightarrow 32 \rightarrow 3$, utilizing ReLU activation functions for hidden layers and a linear-to-softmax output layer.
* The network parameters are optimized via Stochastic Gradient Descent (Adam) to minimize the Temporal Difference (TD) error across a minibatch:

$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \right)^{2} \right]
$$

**Stabilization Techniques:**
To prevent catastrophic divergence caused by highly correlated sequential financial data, the model implements an **Experience Replay** buffer ($\mathcal{D}$) for uniform sampling, and maintains a distinct target network ($\theta^{-}$) that undergoes delayed parameter synchronization.

---

## 3. Performance Analysis and Discussion

The algorithms were trained on historical SPY data (2010–2015) and validated on an unseen test set (2016–2020), benchmarked against a static Buy-and-Hold strategy and the Moving Average Convergence Divergence (MACD) indicator.

### 3.1 The Superiority of Direct Policy Search
**Hill Climbing** produced the highest cumulative wealth and demonstrated the greatest structural stability across varied random seed initializations. By fleeing the market during mathematically defined downward momentum and immediately re-entering upon liquidity, the policy bypassed the delayed reaction times of the MACD benchmark. In a highly stochastic environment, directly solving the linear system of equations for the policy proved significantly more reliable than attempting to approximate the underlying value function.

### 3.2 Information Loss in Discretization
**Tabular Q-Learning** converged to a simplistic, non-active strategy: it mathematically verified that buying on the first indicator of upward momentum and holding into perpetuity was a local optimum. While this outperformed the reactive MACD indicator (as it avoided missing high-yield trading days), the rigid $2 \times 2$ state matrix effectively stripped the environment of the granularity required to identify optimal exit points.

### 3.3 The Failure of the Continuous Approximation
Despite its theoretical capacity to handle continuous variables, the **Deep Q-Network (DQN)** underperformed the basic Buy-and-Hold benchmark. An analysis of the network's output revealed that the predicted action values hovered near a uniform distribution. The continuous state variables alone did not provide a distinct enough gradient signal for the neural network to establish a strong, mathematical preference for specific state-action pairs, leading to ambiguous decision-making during high-volatility periods.

### Conclusion
For single-asset stock trading modeled under constrained state parameters, derivative-free policy iteration (Hill Climbing) guarantees monotonically improving results and greater stability than value-search methods (Q-Learning and DQN), which suffer heavily from discretization limitations and gradient ambiguity.
