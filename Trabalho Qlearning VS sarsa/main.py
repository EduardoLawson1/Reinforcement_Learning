#Importando as bibliotecas necessárias
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
from time import sleep
from matplotlib import animation

#funções que estão nos outros módulos
from train import *
from animation import *

#iniciando e reiniciando ambiente
env = gym.make("Taxi-v3", render_mode="rgb_array").env
state, _ = env.reset()

# Print dimensions of state and action space
print("State space: {}".format(env.observation_space))
print("Action space: {}".format(env.action_space))

# Sample random action
action = env.action_space.sample(env.action_mask(state))
next_state, reward, done, _, _ = env.step(action)

# Print output
print("State: {}".format(state))
print("Action: {}".format(action))
print("Action mask: {}".format(env.action_mask(state)))
print("Reward: {}".format(reward))

# Render and plot an environment frame
frame = env.render()
plt.imshow(frame)
plt.axis("off")
plt.show()


def test_policy(env, q_table, num_episodes=1, store_gif=True, filename='test_policy.gif'):
    num_epochs = 0
    total_failed_deliveries = 0
    experience_buffer = []

    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        done = False
        cum_reward = 0
        epoch = 0

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, _, _ = env.step(action)
            cum_reward += reward

            if reward == -10:
                total_failed_deliveries += 1

            experience_buffer.append({
                'frame': env.render(),
                'episode': episode,
                'epoch': epoch,
                'state': state,
                'action': action,
                'reward': cum_reward
            })

            epoch += 1

        num_epochs += epoch

    if store_gif:
        store_episode_as_gif(experience_buffer, filename=filename)

    print(f"Test results after {num_episodes} episodes:")
    print(f"Mean # epochs per episode: {num_epochs / num_episodes}")
    print(f"Mean # failed drop-offs per episode: {total_failed_deliveries / num_episodes}")


"""TREINOS E TESTES"""
# Hyperparameters
alpha = 0.1
gamma = 1.0
epsilon = 0.1
num_episodes = 10000

# Training with Q-Learning
print("Training with Q-Learning...")
q_table_q_learning, cum_rewards_q_learning, total_epochs_q_learning = train_q_learning(env, alpha, gamma, epsilon, num_episodes)

# Training with SARSA
print("Training with SARSA...")
q_table_sarsa, cum_rewards_sarsa, total_epochs_sarsa = train_sarsa(env, alpha, gamma, epsilon, num_episodes)

# Plot reward convergence for Q-Learning
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Q-Learning: Cumulative reward per episode")
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.plot(cum_rewards_q_learning)

plt.subplot(1, 2, 2)
plt.title("Q-Learning: # epochs per episode")
plt.xlabel("Episode")
plt.ylabel("# epochs")
plt.plot(total_epochs_q_learning)
plt.show()

# Plot reward convergence for SARSA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("SARSA: Cumulative reward per episode")
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.plot(cum_rewards_sarsa)

plt.subplot(1, 2, 2)
plt.title("SARSA: # epochs per episode")
plt.xlabel("Episode")
plt.ylabel("# epochs")
plt.plot(total_epochs_sarsa)
plt.show()

# Test policies
print("Testing Q-Learning policy...")
test_policy(env, q_table_q_learning, filename='q_learning_policy.gif')

print("Testing SARSA policy...")
test_policy(env, q_table_sarsa, filename='sarsa_policy.gif')




