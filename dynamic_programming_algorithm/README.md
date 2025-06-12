# Dynamic Programming Algorithms: Policy & Value Iteration

A comprehensive implementation of dynamic programming methods for solving Markov Decision Processes (MDPs), featuring both gridworld navigation and biological network control applications.

## 🎯 Problems Overview

### Problem 1: Maze Navigation
**Environment**: 18×18 gridworld with walls, bumps, oil spills, and goal
- **State Space**: 248 navigable cells (324 total - 76 walls)
- **Action Space**: {Up, Down, Left, Right}
- **Stochasticity**: Slip probability `p` causing perpendicular movement
- **Rewards**: Goal (+300), Regular (-1), Oil (-5), Bump (-10), Wall (-0.8)

### Problem 2: p53-MDM2 Gene Network Control
**Environment**: 4-gene biological network [ATM, p53, Wip1, MDM2]
- **State Space**: 16 binary states (2^4 combinations)
- **Action Space**: 5 control actions for gene activation
- **Objective**: Maximize simultaneous gene activation
- **Rewards**: +5 per active gene, -1 per control action

## 🧪 Experimental Design

### Problem 1: Parameter Sensitivity Analysis

#### Scenarios Tested:
1. **Base Scenario**: p=0.025, γ=0.99, θ=0.01
2. **High Stochasticity**: p=0.4, γ=0.99, θ=0.01  
3. **Low Discount**: p=0.025, γ=0.5, θ=0.01

#### Algorithms Compared:
- **Policy Iteration**: Vector-form implementation
- **Value Iteration**: Vector-form implementation

### Problem 2: Noise Impact Analysis

#### Noise Levels Tested:
- **Low Noise**: p=0.04
- **Medium Noise**: p=0.15
- **High Noise**: p=0.48
- **Comparison**: p=0.05 (Policy vs Value Iteration)

## 📊 Detailed Results

### Problem 1: Maze Navigation Results

#### Convergence Analysis
| Scenario | Policy Iteration | Value Iteration | Performance Gap |
|----------|------------------|-----------------|-----------------|
| Base (p=0.025, γ=0.99) | **17 iterations** | 52 iterations | 3.1× faster |
| High Stochasticity (p=0.4, γ=0.99) | **14 iterations** | 110 iterations | 7.9× faster |
| Low Discount (p=0.025, γ=0.5) | **6 iterations** | 15 iterations | 2.5× faster |

#### Impact of Parameters

**Base Scenario (p=0.025, γ=0.99)**:
- **Behavior**: Clear gradient from goal to start
- **Policy**: Direct paths avoiding hazards
- **Optimal Path**: Efficient navigation around obstacles
- **Value Range**: High positive values near goal, decreasing with distance

**High Stochasticity (p=0.4, γ=0.99)**:
- **Behavior**: Lower overall state values due to risk
- **Policy**: More conservative, routes through hazards when necessary
- **Key Finding**: 60% intended movement, 20% slip in each perpendicular direction
- **Effect**: Agent accounts for increased risk of unintended hazard encounters

**Low Discount (p=0.025, γ=0.5)**:
- **Behavior**: Myopic decision-making
- **Policy**: Random-appearing directions, early termination
- **Critical Finding**: Future rewards discounted too heavily
- **Result**: Path terminates at (3,12) rather than reaching goal

### Problem 2: Gene Network Control Results

#### Performance by Noise Level

**Low Noise (p=0.04)**:
- **Optimal Policy**: Primarily action a₃ (p53 activation) with occasional a₄, a₅
- **Average Activation**: 2.87 (vs <1 without control)
- **Convergence**: 142 iterations
- **Control Effectiveness**: High - reliable gene activation

**Medium Noise (p=0.15)**:
- **Optimal Policy**: Similar to low noise, mostly a₃ with some a₄, a₅
- **Average Activation**: 2.53 (slight decrease from low noise)
- **Convergence**: 139 iterations
- **Control Effectiveness**: Good - moderate impact from noise

**High Noise (p=0.48)**:
- **Optimal Policy**: All a₁ (no control)
- **Average Activation**: ~2 (same as uncontrolled)
- **Convergence**: 136 iterations
- **Control Effectiveness**: Minimal - noise overwhelms control

**Policy vs Value Iteration Comparison (p=0.05)**:
- **Policy Iteration**: 3 iterations
- **Value Iteration**: 142 iterations
- **Result**: Identical optimal policies
- **Advantage**: Policy iteration 47× faster convergence

## 🔍 Key Insights

### Algorithm Performance
1. **Policy Iteration Superiority**: Consistently faster convergence across all scenarios
2. **Convergence Speed**: 2.5× to 7.9× faster than Value Iteration
3. **Scenario Sensitivity**: Larger gaps in complex environments (high stochasticity)

### Parameter Impact Analysis
1. **Discount Factor (γ)**:
   - **High (0.99)**: Long-term planning, reaches goal despite costs
   - **Low (0.5)**: Myopic behavior, fails to reach distant goals
   - **Critical Threshold**: Below ~0.7 may cause goal abandonment

2. **Stochasticity (p)**:
   - **Low (0.025)**: Deterministic-like behavior, direct paths
   - **High (0.4)**: Risk-aware policies, lower state values
   - **Effect**: Higher uncertainty reduces overall value function

3. **Noise in Control Systems**:
   - **Low-Medium Noise**: Control remains effective
   - **High Noise (≥0.48)**: Control becomes ineffective
   - **Threshold Effect**: Sharp transition around p=0.3-0.4

### Biological Network Insights
1. **Gene Control Strategy**: p53 activation (a₃) most effective
2. **Noise Robustness**: Control effective up to moderate noise levels
3. **System Behavior**: Without control, system defaults to inactive state
4. **Control Cost**: Action cost (-1) worthwhile for gene activation (+5 each)

## 🚀 Implementation Highlights

### Vector-Form Dynamic Programming
```python
# Policy Evaluation (Matrix Form)
V = (I - γ * P_π)^(-1) * R_π

# Policy Improvement  
π'(s) = argmax_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]

# Value Iteration Update
V(s) ← max_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]
```

### Key Features
- **Matrix Operations**: Efficient vector-form implementations
- **Convergence Tracking**: Iteration counts and threshold monitoring
- **Visualization**: State values, policies, and optimal paths
- **Performance Metrics**: Average gene activation, path optimality

## 📈 Practical Applications

### Maze Navigation Applications
- **Robotics**: Autonomous navigation with sensor uncertainty
- **Game AI**: Pathfinding in stochastic environments
- **Risk Assessment**: Decision-making under uncertainty

### Gene Network Applications
- **Drug Design**: Therapeutic intervention strategies
- **Systems Biology**: Understanding cellular control mechanisms
- **Synthetic Biology**: Designing controlled biological circuits

## 📁 File Structure
```
02-dynamic-programming/
├── README.md
├── maze_navigation.ipynb
├── p53-MDM2_gene_network_control.ipynb
└── results/
    └── results.docx
```