import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import polyscope as ps
from geometry import Assembly2D, Assembly2D_GUI
from assembly_env import AssemblyEnv
from multiprocessing import Process, Queue
from compas_eve import Message
from compas_eve import Subscriber
from compas_eve import Publisher
from compas_eve import Topic
from compas_eve.mqtt import MqttTransport
from stable_baselines3 import PPO
import time
import itertools
import pickle
def binatodeci(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))
def plot_policy():
    parts = [[[4.0, 3.0], [5.0, 3.0], [5.0, 5.0], [4.0, 5.0]],
             [[0.0, 3.0], [1.0, 3.0], [1.0, 5.0], [0.0, 5.0]],
             [[2.0, 3.0], [3.0, 3.0], [3.0, 5.0], [2.0, 5.0]],
             [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]],
             [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
             [[4.0, 0.0], [5.0, 0.0], [5.0, 2.0], [4.0, 2.0]],
             [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]],
             [[0.0, 5.0], [5.0, 5.0], [5.0, 6.0], [0.0, 6.0]]]
    assembly = Assembly2D(parts)
    env = AssemblyEnv(assembly)
    env.render = True
    env.send_time_delay = 1
    model = PPO.load(f"models/PPO/{25000}.zip", env)

    G = nx.Graph()
    edge_prob = []
    edge_list = []
    for status in itertools.product([0, 1], repeat=8):
        env._part_status = np.array(status)
        if assembly.check_stability(env.part_status()) == None:
            continue
        obs_tensor, vectorized_env = model.policy.obs_to_tensor(status)
        distrib = model.policy.get_distribution(obs_tensor)
        prob = distrib.distribution.probs.cpu().detach().numpy()

        for part_i in range(assembly.n_part() - 1):
            if status[part_i] == 1:
                continue
            status_i = np.copy(status)
            status_i[part_i] = 1
            env._part_status = np.copy(status_i)
            if assembly.check_stability(env.part_status()) == None:
                continue
            G.add_edge(binatodeci(status), binatodeci(status_i))
            edge_prob.append(prob[0, part_i])
            edge_list.append((binatodeci(status), binatodeci(status_i)))
    pos = nx.bfs_layout(G, start=0)
    options = {
        "edgelist": edge_list,
        "node_color": "#A0CBE2",
        "edge_color": edge_prob,
        "width": 3,
        "edge_cmap": plt.cm.Blues,
        "with_labels": True,
        "node_size" : 1000,
        "font_size" : 10
    }
    # Set margins for the axes so that nodes aren't clipped
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, **options)
    plt.savefig('PPO_25000.png')

if __name__ == '__main__':
    plot_policy()