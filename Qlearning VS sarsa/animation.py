#Importando as bibliotecas necessárias
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
from time import sleep
from matplotlib import animation
from train import *

#Funções gerais
def run_animation(experience_buffer):
    time_lag = 0.05  # Delay (in s) between frames
    for experience in experience_buffer:
        plt.imshow(experience['frame'])
        plt.axis('off')
        plt.show(block=False)
        plt.pause(time_lag)
        plt.close()

        print(f"Episode: {experience['episode']}/{experience_buffer[-1]['episode']}")
        print(f"Epoch: {experience['epoch']}/{experience_buffer[-1]['epoch']}")
        print(f"State: {experience['state']}")
        print(f"Action: {experience['action']}")
        print(f"Reward: {experience['reward']}")

def store_episode_as_gif(experience_buffer, path='./', filename='animation.gif'):
    fps = 5   # Set frames per second
    dpi = 30  # Set dots per inch
    interval = 50  # Interval between frames (in ms)

    frames = [experience['frame'] for experience in experience_buffer]

    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval)
    anim.save(path + filename, writer='imagemagick', fps=fps)
