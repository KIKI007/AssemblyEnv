import os
import numpy as np
from AssemblyEnv.reinforce.playground import AssemblyPlayground
from multiprocessing import Process, Queue
from stable_baselines3 import PPO
import time
from AssemblyEnv.geometry import Assembly2D
def test(queue):
    parts = queue.get()
    assembly = Assembly2D(parts)
    env = AssemblyPlayground(assembly)
    env.render = True
    env.send_time_delay = 1
    model = PPO.load(f"models/PPO/{25000}.zip", env)

    env.reset()
    start_part_status = np.array([0, 1, 1, 0, 1, 1, 1, 1])
    env._part_status = np.copy(start_part_status)
    obs = env._part_status
    env.send()
    for it in range(100):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            #env.reset()
            env._part_status = np.copy(start_part_status)
            obs = env._part_status
            env.send()