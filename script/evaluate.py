from multiprocessing import Process, Queue
import os
import numpy as np
from AssemblyEnv.reinforce.playground import AssemblyPlayground
from multiprocessing import Process, Queue
from stable_baselines3 import PPO
import time
from AssemblyEnv.geometry import Assembly2D
import itertools
def evaluate(queue):
    parts = queue.get()
    assembly = Assembly2D(parts)
    env = AssemblyPlayground(assembly)
    env.render = False
    env.send_time_delay = 1
    model = PPO.load(f"models/PPO_2/{21}", env)

    num_total_states = 0
    num_sucess_states = 0
    for part_state in itertools.product([0, 1], repeat=assembly.n_part() - 1):
        env._part_status = np.copy(part_state)
        if assembly.check_stability(env.part_status()) != None and env._part_status.sum() < assembly.n_part() - 1:
            num_total_states = num_total_states + 1
            obs = np.copy(env._part_status)
            while True:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    if reward > 0:
                        num_sucess_states += 1
                    else:
                        print(part_state)
                    break
                    #obs, info = env.reset_random()

    print(f"{num_sucess_states}/{num_total_states}")

if __name__ == "__main__":
    queue = Queue()
    # parts = [[[4.0, 3.0], [5.0, 3.0], [5.0, 5.0], [4.0, 5.0]],
    #          [[0.0, 3.0], [1.0, 3.0], [1.0, 5.0], [0.0, 5.0]],
    #          [[2.0, 3.0], [3.0, 3.0], [3.0, 5.0], [2.0, 5.0]],
    #          [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]],
    #          [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
    #          [[4.0, 0.0], [5.0, 0.0], [5.0, 2.0], [4.0, 2.0]],
    #          [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]],
    #          [[0.0, 5.0], [5.0, 5.0], [5.0, 6.0], [0.0, 6.0]]]
    parts = [[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]], [[5.0, 0.0], [6.0, 0.0], [6.0, 3.0], [5.0, 3.0]], [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]], [[5.0, 4.0], [6.0, 4.0], [6.0, 6.0], [5.0, 6.0]], [[3.0, 3.0], [6.0, 3.0], [6.0, 4.0], [3.0, 4.0]], [[2.0, 3.0], [3.0, 3.0], [3.0, 6.0], [2.0, 6.0]], [[2.0, 6.0], [6.0, 6.0], [6.0, 7.0], [2.0, 7.0]]]
    queue.put(parts)
    evaluate(queue)