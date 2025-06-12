# Temporal Difference Learning Algorithms

A comprehensive implementation and comparative analysis of temporal difference (TD) learning methods, featuring both maze navigation and biological gene network control applications with detailed performance evaluation across multiple algorithms.

## 🎯 Overview

This project implements and compares four key temporal difference learning algorithms:
- **Q-Learning** (Off-policy)
- **SARSA** (On-policy) 
- **SARSA(λ)** (On-policy with eligibility traces)
- **Tabular Actor-Critic** (Policy gradient method)

## 🧪 Problem Domains

### Problem 1: Maze Navigation
**Environment**: Same 18×18 gridworld from Project 2
- **Parameters**: p=0.025, γ=0.96, α=0.25, ε=0.1
- **Training**: 1000 episodes, max 1000 steps per episode
- **Evaluation**: 10 independent runs per algorithm

### Problem 2: p53-MDM2 Gene Network Control
**Environment**: 4-gene biological network [ATM, p53, Wip1, MDM2]
- **Parameters**: p=0.1, γ=0.9, α=0.25, ε=0.15, λ=0.95 (SARSA-λ), β=0.05 (Actor-Critic)
- **Training**: 1000 episodes, max 100 steps per episode
- **Evaluation**: 10 independent runs per algorithm

## 📊 Detailed Results

### Problem 1: Maze Navigation Performance

#### Success Rate Analysis
| Algorithm | Success Rate | Performance Characteristics |
|-----------|--------------|---------------------------|
| **Q-Learning** | **10/10 (100%)** | Optimal off-policy learning |
| **SARSA** | **9/10 (90%)** | Reliable on-policy learning |
| **Actor-Critic** | **2/10 (20%)** | High parameter sensitivity |

#### Learning Dynamics

**Q-Learning Performance**:
- **Convergence**: Episodes 180-200
- **Early Phase (0-150)**: Large negative spikes from exploration
- **Learning Phase (150-200)**: Rapid improvement in average rewards
- **Stable Phase (200+)**: Consistent high positive rewards
- **Key Strength**: Off-policy nature enables optimal path discovery

**SARSA Performance**:
- **Convergence**: Similar to Q-Learning (episodes 180-200)
- **Learning Pattern**: Comparable initial negative spikes, gradual improvement
- **Policy Quality**: Nearly identical to Q-Learning optimal policy
- **Key Characteristic**: On-policy learning with slight conservatism

**Actor-Critic Performance**:
- **Convergence**: Highly unstable, frequent fluctuations
- **Reward Pattern**: Predominantly negative rewards throughout training
- **Parameter Sensitivity**: Extremely sensitive to α and β values
- **Challenge**: Noisy gradient updates causing policy instability

### Problem 2: Gene Network Control Results

#### Optimal Policy Analysis (16-State Network)

**Policy Convergence Across Algorithms**:
- **Q-Learning, SARSA, Actor-Critic**: Identical optimal policies
- **SARSA(λ)**: Requires parameter tuning for optimal performance
- **Dominant Actions**: a₂ and a₃ (gene activation) heavily favored
- **Strategy**: Prioritize gene activation (+5 reward each) over action costs (-1)

#### Performance Ranking

| Rank | Algorithm | Performance Characteristics |
|------|-----------|---------------------------|
| 1 | **Actor-Critic** | Best performance in smaller state space |
| 2 | **Q-Learning** | Strong off-policy learning, close second |
| 3 | **SARSA** | Conservative on-policy approach, slower convergence |
| 4 | **SARSA(λ)** | Underperforming, needs parameter optimization |

#### Key Insights from Gene Network

**Actor-Critic Advantage**:
- **State Space Sensitivity**: Excels in smaller state spaces (16 vs 248)
- **Policy Gradient Benefits**: Direct policy optimization effective for biological control
- **Domain Suitability**: Well-suited for continuous control problems

**SARSA(λ) Challenges**:
- **Eligibility Traces**: Parameter λ=0.95 may be suboptimal
- **Tuning Requirements**: Needs careful hyperparameter adjustment
- **Complexity Trade-off**: Added complexity not justified by performance

## 🔍 Algorithm-Specific Analysis

### Q-Learning (Off-Policy)
**Strengths**:
- **Optimal Policy Discovery**: Learns true optimal policy regardless of behavior policy
- **Exploration Efficiency**: Can learn from any exploratory policy
- **Maze Performance**: 100% success rate with efficient convergence

**Update Rule**: `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`

### SARSA (On-Policy)
**Strengths**:
- **Policy Consistency**: Learns value of policy being followed
- **Stable Learning**: More conservative, less prone to overestimation
- **Practical Performance**: 90% success rate, nearly matching Q-Learning

**Update Rule**: `Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]`

### SARSA(λ) (Eligibility Traces)
**Intended Benefits**:
- **Credit Assignment**: Better temporal credit assignment through traces
- **Faster Learning**: Should accelerate learning through trace decay

**Observed Challenges**:
- **Parameter Sensitivity**: λ=0.95 may be too high for this domain
- **Complexity vs Benefit**: Additional complexity not yielding better performance

**Update Rule**: Multi-step updates with `e(s,a) ← γλe(s,a) + 1`

### Tabular Actor-Critic (Policy Gradient)
**Domain-Dependent Performance**:
- **Large State Space (Maze)**: Poor performance, high instability
- **Small State Space (Gene Network)**: Best performance among all methods

**Key Characteristics**:
- **Direct Policy Learning**: Updates policy parameters directly
- **Parameter Sensitivity**: Requires careful tuning of α and β
- **Softmax Policy**: `π(a|s) = exp(H(s,a))/Σ exp(H(s,a'))`

## 🚀 Key Insights

### Algorithm Selection Guidelines

1. **For Large State Spaces**:
   - **Q-Learning**: Best overall performance and reliability
   - **SARSA**: Good alternative with slightly more conservative exploration
   - **Avoid Actor-Critic**: High instability in complex environments

2. **For Small State Spaces**:
   - **Actor-Critic**: Can outperform value-based methods
   - **Q-Learning**: Reliable fallback option
   - **SARSA(λ)**: Potential with proper parameter tuning

3. **Parameter Sensitivity**:
   - **Q-Learning/SARSA**: Relatively robust to parameter choices
   - **Actor-Critic**: Requires extensive hyperparameter tuning
   - **SARSA(λ)**: Sensitive to λ value selection

### Convergence Characteristics

**Fast Convergence (Episodes 180-200)**:
- Q-Learning and SARSA show similar convergence patterns
- Initial exploration phase with negative rewards
- Rapid improvement once good states discovered
- Stable performance after convergence

**Learning Curve Phases**:
1. **Exploration Phase (0-150)**: High variance, many failures
2. **Learning Phase (150-200)**: Rapid improvement, policy refinement
3. **Exploitation Phase (200+)**: Stable high performance

### Domain-Specific Findings

**Maze Navigation Insights**:
- Off-policy methods (Q-Learning) excel in deterministic optimal path problems
- On-policy methods (SARSA) nearly match performance with more stable learning
- Actor-Critic struggles with large action spaces and sparse rewards

**Gene Network Control Insights**:
- Policy gradient methods effective for biological control systems
- Smaller state spaces favor direct policy optimization
- Action costs vs. gene activation rewards create clear optimization objectives

## 📁 File Structure
```
03-temporal-difference/
├── README.md
├── maze_navigation.ipynb
├── p53-MDM2_gene_network_control.ipynb
└── results/
    └── results.docx
```