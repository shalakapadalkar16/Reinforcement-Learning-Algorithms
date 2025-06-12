# Multi-Armed Bandit Algorithms

A comprehensive implementation and analysis of multi-armed bandit algorithms, exploring different exploration strategies, learning rates, and initialization techniques.

## 🎯 Problem Overview

This project implements a 2-armed bandit problem where an agent must learn to maximize rewards by selecting between two levers with unknown reward distributions:

- **Lever 1**: Gaussian distribution with μ₁ = 6, σ₁² = 15
- **Lever 2**: Binomial Gaussian mixture with μ₂₁ = 11, σ₂₁² = 16, μ₂₂ = 3, σ₂₂² = 8

**True Optimal Values**: Q*(a₁) = 6, Q*(a₂) = 7

## 🧪 Experiments Conducted

### Part A: ε-Greedy Policy Analysis
**Objective**: Compare different learning rates and exploration strategies

**Learning Rates Tested**:
- α = 1 (constant replacement)
- α = 0.9^k (exponential decay)
- α = 1/(1 + ln(1 + k)) (logarithmic decay)
- α = 1/k (inverse decay)

**Exploration Strategies**:
- ε = 0 (pure greedy)
- ε = 0.1 (mostly exploitation)
- ε = 0.2 (balanced)
- ε = 0.5 (random policy)

### Part B: Optimistic Initialization
**Objective**: Analyze the impact of initial Q-values on learning performance

**Initial Values Tested**:
- Q = [0, 0] (pessimistic)
- Q = [6, 7] (true values)
- Q = [15, 15] (optimistic)

### Part C: Gradient Bandit Algorithm
**Objective**: Compare policy gradient approach with ε-greedy methods

## 📊 Key Findings

### Learning Rate Performance
- **Logarithmic decay (α = 1/(1 + ln(1 + k)))** achieved the best balance between stability and adaptability
- **Constant α = 1** worked well for pure greedy but caused instability with exploration
- **Exponential decay (α = 0.9^k)** favored exploration-heavy policies

### Exploration Strategy Results
- **ε = 0.2** with logarithmic learning rate achieved **highest accumulated rewards**
- **ε = 0.5** produced most accurate Q-value estimates but lower rewards due to over-exploration
- **ε = 0** performed well only with constant learning rate

### Optimistic Initialization Impact
- **True value initialization [6, 7]** achieved fastest convergence and highest rewards
- **Optimistic initialization [15, 15]** encouraged exploration but converged slower
- **Pessimistic initialization [0, 0]** led to conservative exploration and lower performance

### Gradient Bandit vs ε-Greedy
- **Gradient bandit** outperformed ε-greedy due to adaptive exploration strategy
- Fixed ε-greedy exploration rate (ε = 0.1) less efficient than dynamic probability updates

## 🔧 Implementation Details

### Q-Learning Update Rule
```
Q(a) = Q(a) + α(r - Q(a))
```

### Gradient Bandit Policy
```
π_t(a) = e^(H_t(a)) / Σ e^(H_t(a'))

H_{t+1}(a_t) = H_t(a_t) + α(R_t - R̄_t)(1 - π_t(a_t))
H_{t+1}(a) = H_t(a) - α(R_t - R̄_t)π_t(a)  [for non-selected actions]
```

### Experimental Setup
- **Episodes**: 1000 steps per run
- **Independent Runs**: 100 (for statistical significance)
- **Metrics**: Average accumulated reward, Q-value estimates

## 📈 Results Summary

### Optimal Configuration
- **Best Performance**: α = 1/(1 + ln(1 + k)), ε = 0.2
- **Final Q-values**: Q(a₁) = 5.34, Q(a₂) = 6.48 (close to true values)
- **Convergence**: ~200 episodes for stable policy

### Detailed Experimental Results

#### Part A: Learning Rate and Exploration Analysis

**α = 1 (Constant Replacement)**
| ε | Q(a₁) | Q(a₂) | Performance |
|---|-------|-------|-------------|
| 0.0 (greedy) | -1.89 | 2.61 | Highest reward but poor estimates |
| 0.1 | 2.92 | 2.82 | Moderate performance |
| 0.2 | 3.66 | 3.81 | Better estimates |
| 0.5 (random) | 5.06 | 5.31 | Best estimates, lower rewards |

**Key Finding**: Greedy policy performs well with α=1 due to no exploration interference, but produces inaccurate Q-values.

**α = 0.9^k (Exponential Decay)**
| ε | Q(a₁) | Q(a₂) | Performance |
|---|-------|-------|-------------|
| 0.0 (greedy) | 5.28 | 0.74 | Lowest reward |
| 0.1 | 5.10 | 2.44 | Moderate |
| 0.2 | 4.92 | 4.28 | Good balance |
| 0.5 (random) | 5.37 | 6.17 | **Highest reward** |

**Key Finding**: Random policy (ε=0.5) achieves highest reward as exponential decay allows continued exploration.

**α = 1/(1 + ln(1 + k)) (Logarithmic Decay)**
| ε | Q(a₁) | Q(a₂) | Performance |
|---|-------|-------|-------------|
| 0.0 (greedy) | 5.59 | 0.48 | Poor due to lack of exploration |
| 0.1 | 5.21 | 5.87 | Good |
| 0.2 | 5.34 | 6.48 | **Optimal performance** |
| 0.5 (random) | 5.83 | 6.74 | Best estimates but lower rewards |

**Key Finding**: ε=0.2 achieves optimal balance - maximizes rewards while maintaining good Q-value estimates.

**α = 1/k (Inverse Decay)**
| ε | Q(a₁) | Q(a₂) | Performance |
|---|-------|-------|-------------|
| 0.0 (greedy) | 5.18 | 0.84 | Lowest reward |
| 0.1 | 5.15 | 2.62 | Moderate |
| 0.2 | 5.44 | 3.79 | Good |
| 0.5 (random) | 5.79 | 6.16 | **Highest reward** |

**Key Finding**: Slow learning rate benefits from extensive exploration (ε=0.5).

#### Part B: Optimistic Initialization Results

**Fixed Parameters**: α = 0.1, ε = 0.1

| Initial Q Values | Final Q(a₁) | Final Q(a₂) | Performance |
|------------------|-------------|-------------|-------------|
| [0, 0] (pessimistic) | 0.88 | 0.99 | Lower rewards |
| [6, 7] (true values) | 0.83 | 1.06 | **Highest rewards** |
| [15, 15] (optimistic) | 0.94 | 0.90 | Aggressive exploration, slower convergence |

**Key Finding**: Initialization close to true values provides fastest convergence and best performance.

#### Part C: Gradient Bandit vs ε-Greedy

**Comparison**: Gradient Bandit (α=0.1) vs ε-Greedy (α=0.1, ε=0.1)

- **Gradient Bandit**: Achieved higher accumulated rewards
- **ε-Greedy**: Fixed exploration rate limiting performance
- **Advantage**: Adaptive exploration strategy in gradient bandit outperforms fixed ε

### Overall Performance Ranking

| Rank | Configuration | Learning Rate | ε | Final Performance |
|------|---------------|---------------|---|-------------------|
| 1 | **ε-greedy** | logarithmic | 0.2 | **Optimal** |
| 2 | ε-greedy | exponential | 0.5 | High |
| 3 | ε-greedy | inverse | 0.5 | High |
| 4 | Gradient Bandit | 0.1 | - | Better than comparable ε-greedy |
| 5 | ε-greedy | constant | 0.0 | High but unstable |

## 🚀 Key Insights

### 1. Exploration-Exploitation Trade-off
- **Moderate exploration (ε = 0.2)** with logarithmic learning rate achieved optimal performance
- **Pure greedy (ε = 0)** works only with constant learning rates but produces poor Q-estimates
- **Random policy (ε = 0.5)** excels with slower decaying learning rates due to continued exploration benefits

### 2. Learning Rate Impact
- **Logarithmic decay α = 1/(1 + ln(1 + k))** provides best balance between stability and adaptability
- **Exponential decay α = 0.9^k** favors exploration-heavy policies
- **Constant α = 1** enables greedy exploitation but causes instability with exploration
- **Inverse decay α = 1/k** requires extensive exploration for good performance

### 3. Initialization Strategy
- **True value initialization [6, 7]** provides fastest convergence and highest accumulated rewards
- **Optimistic initialization [15, 15]** encourages exploration but converges slower
- **Pessimistic initialization [0, 0]** leads to conservative exploration and suboptimal performance

### 4. Policy Gradient vs Value-Based Methods
- **Gradient bandit** outperforms ε-greedy with comparable parameters
- **Adaptive exploration** (gradient bandit) more effective than fixed exploration rates
- **Dynamic probability updates** better than static ε-greedy exploration

### 5. Algorithm-Specific Observations
- **α = 1**: Greedy policy dominates due to complete replacement of Q-values
- **Exponential/Inverse decay**: Random policy (ε = 0.5) performs best due to maintained exploration
- **Logarithmic decay**: Balanced policy (ε = 0.2) achieves optimal performance

## 📁 File Structure
```
01-multi-armed-bandits/
├── main.ipynb
├── README.md
└── results/
    └── results.docx
```