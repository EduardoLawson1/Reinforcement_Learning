#Importando as bibliotecas necessárias
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
from time import sleep
from matplotlib import animation

"""Função de treinamento para Q learning"""
def train_q_learning(env, alpha, gamma, epsilon, num_episodes):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    cum_rewards = np.zeros([num_episodes])
    total_epochs = np.zeros([num_episodes])

    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        done = False
        cum_reward = 0
        epoch = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _, _ = env.step(action)
            cum_reward += reward

            old_q_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_q_value

            state = next_state
            epoch += 1

        cum_rewards[episode-1] = cum_reward
        total_epochs[episode-1] = epoch

        if episode % 100 == 0:
            clear_output(wait=True)
            print(f"Episode #: {episode}")

    return q_table, cum_rewards, total_epochs



"""Função de treinamento para Sarsa"""
def train_sarsa(env, alpha, gamma, epsilon, num_episodes):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    cum_rewards = np.zeros([num_episodes])
    total_epochs = np.zeros([num_episodes])

    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        done = False
        cum_reward = 0
        epoch = 0

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            cum_reward += reward

            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])

            old_q_value = q_table[state, action]
            next_q_value = q_table[next_state, next_action]
            new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_q_value)
            q_table[state, action] = new_q_value

            state = next_state
            action = next_action
            epoch += 1

        cum_rewards[episode-1] = cum_reward
        total_epochs[episode-1] = epoch

        if episode % 100 == 0:
            clear_output(wait=True)
            print(f"Episode #: {episode}")

    return q_table, cum_rewards, total_epochs
