# PPO Reinforcement Learning CartPole Agent

## Objective
Train a reinforcement learning agent using PPO to solve the CartPole-v1 control environment.

## Tools Used
- Python
- Gymnasium
- Stable-Baselines3
- PyTorch
- TensorBoard
- NumPy

## Key Skills Demonstrated
- Reinforcement learning environment setup
- PPO agent training
- Policy-based learning
- Model evaluation
- AI model saving and reuse
- TensorBoard logging

## Project Outcome
A PPO agent was trained on the CartPole-v1 environment and saved as a reusable model artifact.

## Troubleshooting Log
- Replaced deprecated `gym` with `gymnasium`.
- Avoided outdated `gym[all]` installation because it commonly breaks due to optional Atari/MuJoCo dependencies.
- Used modern Gymnasium-compatible Stable-Baselines3 workflow.
