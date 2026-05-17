import math
import random
import pickle
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")

buckets = (1, 1, 6, 12)
q_table = np.zeros(buckets + (env.action_space.n,))

def discretize(obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    return tuple(int(round((buckets[i] - 1) * min(1, max(0, ratios[i])))) for i in range(len(obs)))

episodes = 300
alpha = 0.1
gamma = 0.99
epsilon = 1.0

for episode in range(episodes):
    obs, info = env.reset()
    state = discretize(obs)
    total_reward = 0

    done = False
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)

        q_table[state + (action,)] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state + (action,)]
        )

        state = next_state
        total_reward += reward

    epsilon = max(0.01, epsilon * 0.98)

print("Training complete")
print("Final episode reward:", total_reward)

with open("qlearning_cartpole_model.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Model saved as qlearning_cartpole_model.pkl")
env.close()
