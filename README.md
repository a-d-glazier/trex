# Trajectory-ranked Reward EXtrapolation (T-REX)

This repository contains implementations of various Inverse Reinforcement Learning (IRL) and Imitation Learning algorithms applied to CartPole and GridWorld environments.

## Environments

### CartPole Environment

The CartPole environment is a classic control problem where a pole is attached to a cart that moves along a frictionless track. The goal is to prevent the pole from falling over by applying forces to the cart. The implementations include:

- `learn_malik.py`: Implementation of the Malik algorithm for inverse reinforcement learning in the CartPole environment
- `learn_max_ent_cartpole.py`: Maximum Entropy IRL implementation for CartPole
- `learn_max_margin_cartpole.py`: Maximum Margin IRL algorithm applied to CartPole
- `learn_gail_cartpole.py`: Generative Adversarial Imitation Learning (GAIL) for CartPole

### GridWorld Environment

The GridWorld environment is a simple 2D grid-based environment used for testing and developing reinforcement learning algorithms. The implementations include:

- `learn_malik.py`: Malik algorithm implementation for GridWorld
- `learn_max_ent.py`: Maximum Entropy IRL for GridWorld
- `learn_max_margin.py`: Maximum Margin IRL for GridWorld
- `learn_gail.py`: GAIL implementation for GridWorld
- `run_icrl_grid.py`: Implementation of Inverse Constraint Reinforcement Learning for GridWorld
- `customgrid.py`: Custom GridWorld environment implementation

## Algorithms

### Maximum Entropy IRL
Implements the Maximum Entropy Inverse Reinforcement Learning algorithm, which finds a reward function that maximizes the entropy of the demonstrated trajectories while matching feature expectations.

### Maximum Margin IRL
Implements the Maximum Margin approach to IRL, which finds a reward function that maximizes the margin between the expert's policy and other policies.

### GAIL (Generative Adversarial Imitation Learning)
Implements GAIL, which uses adversarial training to learn a policy that matches the expert demonstrations without explicitly recovering the reward function.

### Malik Algorithm
Implements a variant of IRL that focuses on efficient reward function recovery from demonstrations.

### ICRL (Inverse Constraint Reinforcement Learning)
Implements ICRL for the GridWorld environment, focusing on learning constraints from demonstrations.

## Usage

Each implementation file contains its own training loop and can be run independently. For example:

```bash
# Run Maximum Entropy IRL on CartPole
python cartpole/learn_max_ent_cartpole.py

# Run GAIL on GridWorld
python gridworld/learn_gail.py
```

## Requirements

- Python 3.7+
- PyTorch
- Gym/Gymnasium
- NumPy
- Stable-Baselines3 (for some implementations)

## References

The implementations are based on the following papers:
- "Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations" (Brown et al., 2019)
- "Generative Adversarial Imitation Learning" (Ho & Ermon, 2016)
- "Maximum Entropy Inverse Reinforcement Learning" (Ziebart et al., 2008)
- "Apprenticeship Learning via Inverse Reinforcement Learning" (Abbeel & Ng, 2004)
- "Inverse Constrained Reinforcement Learning" (Malik et al., ICML 2021)