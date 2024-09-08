import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Advantage Actor Critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='NU',
                    help='num_episodes (default: 1000)')
parser.add_argument('--seed', type=int, default=679, metavar='N',
                    help='random seed (default: 679)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

env = gym.make('MountainCar-v0', render_mode='human' if args.render else None)

num_inputs = env.observation_space.shape[0]
epsilon = 0.99
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def epsilon_value(epsilon):
    eps = 0.99 * epsilon
    return eps

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.Linear1 = nn.Linear(num_inputs, 64)
        nn.init.xavier_uniform_(self.Linear1.weight)
        self.Linear2 = nn.Linear(64, 128)
        nn.init.xavier_uniform_(self.Linear2.weight)
        self.Linear3 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.Linear3.weight)
        num_actions = env.action_space.n

        self.actor_head = nn.Linear(64, num_actions)
        self.critic_head = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.critic_head.weight)
        self.action_history = []
        self.rewards_achieved = []

    def forward(self, state_inputs):
        x = F.relu(self.Linear1(state_inputs))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        return self.critic_head(x), x

    def act(self, state_inputs, eps):
        value, x = self(state_inputs)
        x = F.softmax(self.actor_head(x), dim=-1)
        m = Categorical(x)
        e_greedy = random.random()
        if e_greedy > eps:
            action = m.sample()
        else:
            action = m.sample_n(3)
            pick = random.randint(-1, 2)
            action = action[pick]
        return value, action, m.log_prob(action)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.002)

def perform_updates():
    r = 0
    saved_actions = model.action_history
    returns = []
    rewards = model.rewards_achieved
    policy_losses = []
    critic_losses = []

    for i in rewards[::-1]:
        r = args.gamma * r + i
        returns.insert(0, r)
    returns = torch.tensor(returns)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculating policy loss
        policy_losses.append(-log_prob * advantage)

        # calculating value loss
        critic_losses.append(F.mse_loss(value, torch.tensor([R])))
    optimizer.zero_grad()

    # Finding cumulative loss
    loss = torch.stack(policy_losses).sum() + torch.stack(critic_losses).sum()
    loss.backward()
    optimizer.step()
    # Action history and rewards cleared for next episode
    del model.rewards_achieved[:]
    del model.action_history[:]
    return loss.item()

def main():
    eps = epsilon_value(epsilon)
    losses = []
    counters = []
    plot_rewards = []

    for i_episode in range(0, args.num_episodes):
        counter = 0
        state, _ = env.reset()
        ep_reward = 0
        done = False
        max_steps = 1000  # Maximum steps per episode to prevent infinite loops

        print(f"Starting episode {i_episode}")

        while not done and counter < max_steps:
            if args.render:
                env.render()

            state = torch.tensor(state, dtype=torch.float32)
            value, action, ac_log_prob = model.act(state, eps)
            model.action_history.append(SavedAction(ac_log_prob, value))
            
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            model.rewards_achieved.append(reward)
            ep_reward += reward
            counter += 1

            if counter % 100 == 0:  # Print debug info every 100 steps
                print(f"Episode {i_episode}, Step {counter}, Reward: {ep_reward:.2f}")

            if counter % 5 == 0:
                loss = perform_updates()
            eps = epsilon_value(eps)

        loss = perform_updates()  # Perform one last update at the end of the episode
        losses.append(loss)
        counters.append(counter)
        plot_rewards.append(ep_reward)

        print(f'Episode {i_episode}\tLoss: {loss:.2f}\tSteps: {counter}\tReward: {ep_reward:.2f}')

        if counter >= max_steps:
            print(f"Episode {i_episode} reached max steps!")

    env.close()

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(losses)
    plt.title('Loss over episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.subplot(132)
    plt.plot(counters)
    plt.title('Steps per episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.subplot(133)
    plt.plot(plot_rewards)
    plt.title('Reward per episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.tight_layout()
    plt.savefig('a2c_mountaincar_results.png')
    plt.show()

if __name__ == '__main__':
    main()