# Deep Q-Networks: From Standard to Advanced Architectures

A comprehensive implementation and comparative analysis of Deep Q-Network variants, showcasing the evolution from basic DQN to sophisticated architectures with detailed performance evaluation and architectural improvements.

## 🎯 Project Overview

This project implements and compares three fundamental Deep Q-Network architectures:
- **Standard DQN** - Baseline deep reinforcement learning
- **Double DQN** - Addressing overestimation bias
- **Dueling DQN** - Separate value and advantage estimation

## 🗺️ Environment: Maze Navigation

**Maze Configuration**:
- **Size**: 8×8 gridworld with complex layout
- **State Representation**: (x,y) coordinates as continuous inputs
- **Action Space**: {Up, Down, Left, Right}
- **Stochasticity**: p=0.025 slip probability
- **Starting**: Random initial positions (not fixed start state)

**Reward Structure**:
- Goal state: +100
- Regular movement: -1 (delay penalty)
- Yellow states: -5 (moderate penalty)
- Red states: -10 (high penalty) 
- Wall collision: -0.8

## 🧠 Network Architectures

### Standard DQN Architecture
```python
class DQN(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear(2, 128)    # State input: (x,y)
        self.layer2 = nn.Linear(128, 128)  # Hidden layer
        self.layer3 = nn.Linear(128, 128)  # Hidden layer  
        self.layer4 = nn.Linear(128, 4)    # Output: 4 Q-values
        # Added LayerNorm for stability
```

### Double DQN Modification
**Key Change**: Decoupled action selection and evaluation
```python
# Standard DQN target
target = reward + γ * max(Q_target(next_state, a))

# Double DQN target  
action = argmax(Q_online(next_state, a))
target = reward + γ * Q_target(next_state, action)
```

### Dueling DQN Architecture
**Dual-Stream Design**:
```python
# Feature extraction
features = self.feature_layers(state)

# Value stream: V(s)
value = self.value_stream(features)

# Advantage stream: A(s,a)  
advantage = self.advantage_stream(features)

# Combine streams
Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

## 📊 Detailed Results & Analysis

### Standard DQN Performance

**Hyperparameters**:
- Replay Buffer: 5,000 transitions
- Episode Length: 50 steps max
- Batch Size: 128
- Learning Rate: 1e-4
- Discount Factor: 0.99
- Soft Update: η = 5e-2
- Episodes: 2,000

**Learning Dynamics**:
- **Convergence**: Episode ~100 (rapid learning)
- **Early Phase**: Steep reward increase from exploration
- **Stable Phase**: Consistent high performance post-convergence
- **Loss Pattern**: Initial spikes, then stable decrease

**Policy Quality**: Optimal navigation avoiding penalties, direct goal-seeking behavior

### Double DQN Performance

**Key Improvements**:
- **Training Efficiency**: Converged in 1,200 episodes (vs 2,000 for standard)
- **Stability**: More reliable learning with reduced overestimation
- **Convergence**: Episodes 90-95 (faster than standard DQN)
- **Policy Quality**: Near-optimal with minor suboptimalities near walls

**Hyperparameter Changes**:
- Batch Size: 64 (reduced from 128)
- Episodes: 1,200 (reduced from 2,000)

### Dueling DQN Performance

**Architecture Benefits**:
- **Fastest Convergence**: Stable performance by episode 100
- **Training Efficiency**: Only 500 episodes needed
- **Value Estimation**: More effective state value learning
- **Learning Rate**: 1e-3 (higher than others)
- **Soft Update**: η = 1e-2 (more aggressive updates)

## 🔍 Comparative Analysis

### Convergence Speed Comparison
| Architecture | Episodes to Convergence | Training Efficiency |
|--------------|------------------------|---------------------|
| **Dueling DQN** | **~100 (500 total)** | **Fastest** |
| **Double DQN** | **~95 (1,200 total)** | **Fast** |
| **Standard DQN** | **~100 (2,000 total)** | **Baseline** |

### Key Performance Insights

**Standard DQN**:
- ✅ Reliable baseline performance
- ✅ Clear learning progression
- ❌ Requires most training episodes
- ❌ Potential overestimation bias

**Double DQN**:
- ✅ Reduced overestimation bias
- ✅ More stable learning
- ✅ 40% fewer episodes required
- ✅ Better generalization
- ❌ Minor suboptimalities near boundaries

**Dueling DQN**:
- ✅ Fastest convergence (75% fewer episodes)
- ✅ Superior state value estimation
- ✅ Most sample-efficient learning
- ✅ Robust policy extraction
- ⚠️ Requires careful hyperparameter tuning

### Architecture-Specific Advantages

**Overestimation Bias Reduction (Double DQN)**:
- Separates action selection from value estimation
- More conservative Q-value updates
- Better performance in stochastic environments

**Value-Advantage Decomposition (Dueling DQN)**:
- Learns state values independent of actions
- Better handles states where action choice matters less
- More efficient learning in sparse reward environments

## 🛠️ Implementation Highlights

### Training Process Innovations

**Experience Replay**:
```python
# Store transitions: (s, a, r, s', done)
replay_buffer.store(state, action, reward, next_state, done)

# Sample minibatch for training
batch = replay_buffer.sample(batch_size)
```

**Epsilon Decay Strategy**:
```python
# Exponential decay with minimum threshold
epsilon = max(0.1, (0.995)**episode)
```

**Soft Target Updates**:
```python
# Gradual target network updates
target_params = η * online_params + (1-η) * target_params
```

### Network Architecture Enhancements

**Layer Normalization** (Standard DQN):
- Added after each hidden layer
- Improved training stability
- Better convergence properties

**Dual-Stream Architecture** (Dueling DQN):
- Shared feature extraction
- Separate value and advantage branches
- Mathematically sound recombination

## 📈 Learning Curve Analysis

### Reward Progression Patterns

**All Architectures Show**:
1. **Exploration Phase** (0-50 episodes): High variance, negative rewards
2. **Learning Phase** (50-100 episodes): Rapid improvement
3. **Exploitation Phase** (100+ episodes): Stable high performance

**Distinguishing Characteristics**:
- **Standard DQN**: Gradual, steady improvement
- **Double DQN**: Faster stabilization, less variance
- **Dueling DQN**: Rapid early improvement, quick convergence

### Loss Function Behavior

**Common Pattern**:
- Initial high variance (unstable Q-estimates)
- Rapid decrease during learning phase
- Low, stable values post-convergence

**Architecture Differences**:
- **Dueling DQN**: Lowest final loss values
- **Double DQN**: Most stable loss progression
- **Standard DQN**: Higher variance throughout

## 🎯 Practical Applications

### When to Use Each Architecture

**Standard DQN**:
- Baseline implementation
- Simple environments
- Educational purposes
- Proof of concept projects

**Double DQN**:
- Stochastic environments
- When overestimation is problematic
- Continuous action spaces
- Real-world applications requiring stability

**Dueling DQN**:
- Sparse reward environments
- When state values vary more than action advantages
- Sample-efficient learning requirements
- Time-critical applications

## 📁 File Structure
```
04-deep-q-network/
├── README.md
├── deep_q_network.ipynb
└── results/
    └── results.docx
```