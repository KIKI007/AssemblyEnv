import os
import numpy as np
from AssemblyEnv.reinforce.playground import AssemblyPlayground
from multiprocessing import Process, Queue
from stable_baselines3 import PPO
from AssemblyEnv.geometry import Assembly2D
import time

def

def train(queue):
    parts = queue.get()

    assembly = Assembly2D(parts)
    env = AssemblyPlayground(assembly)
    TIMESTEP = 1000
    env.send_time_delay = 0.1
    env.render = False
    env.reset()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    for epoch in range(100):
        model.learn(total_timesteps=TIMESTEP, tb_log_name="PPO", reset_num_timesteps= (epoch == 0))
        model.save(f"models/PPO/{epoch}")