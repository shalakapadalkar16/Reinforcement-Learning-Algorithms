# Reinforcement Learning Algorithms

A comprehensive collection of reinforcement learning implementations, spanning from foundational multi-armed bandits to cutting-edge deep Q-networks. This repository demonstrates the evolution of RL algorithms through rigorous experimentation and comparative analysis across diverse problem domains.

## 🎯 Repository Overview

This repository contains **four major projects** that systematically explore reinforcement learning from basic exploration-exploitation trade-offs to sophisticated neural network approximations. Each project includes complete implementations, detailed experimental analysis, and practical insights for real-world applications.

## 🗂️ Project Structure

### 01. [Multi-Armed Bandit Algorithms](./multi-arm_bandit_algorithm/)
**Foundation**: Exploration vs Exploitation Trade-offs

**Algorithms Implemented**:
- ε-Greedy with multiple learning rates
- Optimistic initialization strategies  
- Gradient bandit algorithms

**Key Features**:
- **4 learning rate schedules**: Constant, exponential decay, logarithmic decay, inverse decay
- **4 exploration strategies**: Pure greedy, balanced, random policies
- **Statistical rigor**: 100 independent runs, 1000 steps each
- **Optimal configuration discovered**: α = 1/(1 + ln(1 + k)), ε = 0.2

**Applications**: A/B testing, clinical trials, online advertising, recommendation systems

---

### 02. [Dynamic Programming Algorithms](./dynamic_programming_algorithm/)
**Foundation**: Exact Solutions for Known MDPs

**Algorithms Implemented**:
- Policy Iteration (vector-form)
- Value Iteration (vector-form)

**Domains Explored**:
- **18×18 Maze Navigation**: Complex gridworld with stochastic transitions
- **p53-MDM2 Gene Network**: 4-gene biological control system

**Key Insights**:
- **Policy Iteration superiority**: 2.5× to 7.9× faster convergence
- **Parameter sensitivity**: Critical discount factor thresholds discovered
- **Cross-domain applicability**: From robotics to systems biology

**Applications**: Robotics navigation, game AI, biological system control, resource allocation

---

### 03. [Temporal Difference Learning](./temporal_diff_learning_algorithm/)
**Foundation**: Learning from Experience Without Models

**Algorithms Implemented**:
- Q-Learning (off-policy)
- SARSA (on-policy)
- SARSA(λ) with eligibility traces
- Tabular Actor-Critic (policy gradient)

**Experimental Design**:
- **Success rate analysis**: Q-Learning (100%), SARSA (90%), Actor-Critic (20%)
- **Domain-specific performance**: Algorithm suitability varies by state space size
- **10 independent runs** per algorithm for statistical validity

**Key Findings**:
- **Q-Learning**: Optimal for large state spaces, deterministic environments
- **Actor-Critic**: Excels in small state spaces, biological control systems
- **Parameter sensitivity**: Actor-Critic requires extensive hyperparameter tuning

**Applications**: Game playing, autonomous navigation, control systems, financial trading

---

### 04. [Deep Q-Networks](./deep_q_network/)
**Foundation**: Neural Network Function Approximation

**Algorithms Implemented**:
- Standard DQN with experience replay
- Double DQN (overestimation bias reduction)
- Dueling DQN (value-advantage decomposition)

**Technical Achievements**:
- **Sample efficiency**: 75% reduction in training episodes (Dueling vs Standard)
- **Architecture innovations**: LayerNorm integration, dual-stream design
- **Convergence analysis**: Detailed learning phase characterization

**Performance Comparison**:
| Architecture | Episodes Required | Key Advantage |
|--------------|-------------------|---------------|
| **Dueling DQN** | **500** | Fastest convergence, superior state value estimation |
| **Double DQN** | **1,200** | Bias reduction, stable learning |
| **Standard DQN** | **2,000** | Reliable baseline, straightforward implementation |

**Applications**: Atari games, robotics control, autonomous vehicles, resource management

## 📊 Key Results Summary

### Multi-Armed Bandits
- **Optimal policy discovered**: Logarithmic learning rate with ε = 0.2
- **Gradient bandit superiority**: Adaptive exploration outperforms fixed rates
- **Parameter interaction effects**: Learning rate and exploration strategy dependencies

### Dynamic Programming  
- **Algorithm efficiency**: Policy iteration consistently faster than value iteration
- **Parameter impact**: Discount factor critical for goal-reaching behavior
- **Domain adaptability**: Effective across navigation and biological control

### Temporal Difference Learning
- **Algorithm specialization**: Performance varies significantly by domain characteristics
- **State space sensitivity**: Large spaces favor value-based methods, small spaces favor policy gradient
- **Robustness analysis**: Q-Learning most reliable across different environments

### Deep Q-Networks
- **Architecture impact**: Network design critically affects learning efficiency  
- **Sample efficiency**: Advanced architectures achieve 75% training reduction
- **Bias mitigation**: Double DQN successfully addresses overestimation problems

### Environment Compatibility
- **Python 3.8+** with standard scientific computing stack
- **PyTorch/TensorFlow** for deep learning implementations
- **NumPy/SciPy** for numerical computations
- **Matplotlib/Seaborn** for visualization

## 📈 Performance Benchmarks

### Training Efficiency Improvements
- **Multi-Armed Bandits**: 16 parameter combinations systematically evaluated
- **Dynamic Programming**: Up to 7.9× faster convergence with policy iteration
- **Temporal Difference**: Domain-specific algorithm selection guidelines
- **Deep Q-Networks**: 75% sample efficiency improvement with advanced architectures

### Real-World Impact
- **Biological Control**: Successful gene network optimization
- **Navigation**: Optimal path planning in stochastic environments  
- **Sample Efficiency**: Dramatic reduction in training requirements
- **Robustness**: Algorithms tested across multiple problem domains

This repository demonstrates:
- **Systematic algorithm comparison** across RL paradigms
- **Cross-domain validation** of algorithmic approaches
- **Performance optimization** through architectural innovation
- **Practical implementation guidance** for real-world applications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
