# **Reinforcement Learning and Loss of Plasticity Phenomenon in Coverage Path Planning Environment**

## Introduction

This repository implements a single-agent Coverage Path Planning (CPP) environment in OpenAI Gym, designed to investigate the phenomenon of loss of plasticity in deep reinforcement learning. In CPP, an agent must systematically cover every accessible tile in a map without unnecessary overlap. We compare popular RL algorithms (DQN, PPO) under a curriculum-learning protocol, and benchmark their performance against an A\* planner as a gold-standard baseline.

## Repository Structure

```
projeto-intermediario-p-j-cpp-main/
├── assets/                  # Sample figures and plots
├── coverage_env.py          # Gym environment for CPP tasks
├── pygame_renderer.py       # Pygame-based visualization of the agent
├── logs/                    # TensorBoard logs for training runs
├── requirements.txt         # Python dependencies
├── a_star_test.ipynb        # Notebook: A* baseline evaluation
├── dqn_mlp_test.ipynb       # Notebook: Training and evaluation of DQN agent
├── ppo_mlp_test.ipynb       # Notebook: Training and evaluation of PPO agent
├── ppo_mlp_l2_test.ipynb    # Notebook: PPO agent with L2 regularization
├── model_per_level.ipynb    # Notebook: Curriculum-learning across levels
├── conclusion.ipynb         # Report on results and analysis
└── README.md                # Project overview and usage instructions
```

## Prerequisites

* Python 3.8 or higher
* Git (to clone this repository)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/insper-classroom/projeto-intermediario-p-j-cpp.git
   cd projeto-intermediario-p-j-cpp-main
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Running Notebooks

All experiments and visualizations are organized as Jupyter notebooks. Open the notebook of interest:

* **A* Baseline*\*: `a_star_test.ipynb`
* **DQN Agent**: `dqn_mlp_test.ipynb`
* **PPO Agent**: `ppo_mlp_test.ipynb`
* **PPO with L2**: `ppo_mlp_l2_test.ipynb`
* **Individual Level Learning (no curriculum)**: `model_per_level.ipynb`
* **Results Summary**: `conclusion.ipynb`

Each notebook contains step-by-step code cells to train agents, log metrics, and plot performance.

### 2. TensorBoard Visualization

Training logs are saved under the `logs/` directory. To inspect learning curves:

```bash
tensorboard --logdir logs
```

Navigate to [http://localhost:6006](http://localhost:6006) in your browser.

### 3. Custom Experiments

You can adapt hyperparameters or extend the environment by editing:

* **`coverage_env.py`**: modify the grid layouts, reward function or action space.
* **`pygame_renderer.py`**: adjust visualization settings.

Import `CoverageEnv` in your own scripts or notebooks:

```python
from coverage_env import CoverageEnv
```

### 4. Analysis of our Work

The analysis of our work is presented in the `conclusion.ipynb` notebook. It includes:

* A summary of the results obtained from the experiments.
* A discussion of the implications of the findings.
* Suggestions for future work and improvements.