import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import polyscope as ps
from geometry import Assembly2D, Assembly2D_GUI
from AssemblyEnv.reinforce.env import RobotPlayground
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
    parts =   [[[7.5760441129501697, 2.757454550601556], [8.0622577482985509, 0.0], [10.0, 0.0], [9.3969262078590852, 3.420201433256687], [7.5760441129501697, 2.757454550601556]], [[6.1760477470770274, 5.182319386705605], [7.5760441129501697, 2.757454550601556], [9.3969262078590852, 3.420201433256687], [7.6604444311897817, 6.4278760968653907], [6.1760477470770274, 5.182319386705605]], [[4.0311288741492755, 6.9821200218844721], [6.1760477470770274, 5.182319386705605], [7.6604444311897817, 6.4278760968653907], [5.0000000000000027, 8.6602540378443873], [4.0311288741492755, 6.9821200218844721]], [[1.3999963658739316, 7.9397739373070175], [4.0311288741492755, 6.9821200218844721], [5.0000000000000027, 8.6602540378443873], [1.7364817766693057, 9.8480775301220831], [1.3999963658739316, 7.9397739373070175]], [[-1.3999963658787249, 7.9397739373061738], [1.3999963658739316, 7.9397739373070175], [1.7364817766693057, 9.8480775301220831], [-1.7364817766693017, 9.8480775301220831], [-1.3999963658787249, 7.9397739373061738]], [[-4.0311288741492737, 6.9821200218844721], [-1.3999963658787249, 7.9397739373061738], [-1.7364817766693017, 9.8480775301220831], [-4.9999999999999991, 8.6602540378443891], [-4.0311288741492737, 6.9821200218844721]], [[-7.6604444311897808, 6.427876096865397], [-6.1760477470770265, 5.1823193867056068], [-4.0311288741492737, 6.9821200218844721], [-4.9999999999999991, 8.6602540378443891], [-7.6604444311897808, 6.427876096865397]], [[-9.3969262078590852, 3.4202014332566906], [-7.5760441129501599, 2.7574545506015555], [-6.1760477470770265, 5.1823193867056068], [-7.6604444311897808, 6.427876096865397], [-9.3969262078590852, 3.4202014332566906]], [[-10.000000000000004, 3.1086244689504383e-15], [-8.0622577482985527, 2.5062531711346134e-15], [-7.5760441129501599, 2.7574545506015555], [-9.3969262078590852, 3.4202014332566906], [-10.000000000000004, 3.1086244689504383e-15]]]

    assembly = Assembly2D(parts)
    env = RobotPlayground(assembly)
    env.render = False
    env.send_time_delay = 1
    #model = PPO.load(f"models/PPO/{25000}.zip", env)
    model = PPO.load(f"models/PPO3/{18}.zip", env)

    G = nx.Graph()
    edge_prob = []
    edge_list = []
    for status in [[1, 1, 1, 0, 1, 1, 1, 1]]:
        print(status)
        env._part_status = np.array(status)
        if assembly.check_stability(env.part_status()) == None:
            continue
        obs_tensor, vectorized_env = model.policy.obs_to_tensor(status)
        distrib = model.policy.get_distribution(obs_tensor)
        prob = distrib.distribution.probs.cpu().detach().numpy()

        for part_i in range(assembly.n_part() - 1):
            # if status[part_i] == 1:
            #     continue
            status_i = np.copy(status)
            status_i[part_i] = 1
            print(status_i, binatodeci(status_i))
            env._part_status = np.copy(status_i)
            if assembly.check_stability(env.part_status()) == None:
                continue
            G.add_edge(binatodeci(status), binatodeci(status_i))
            edge_prob.append(prob[0, part_i])
            edge_list.append((binatodeci(status), binatodeci(status_i)))
    pos = nx.bfs_layout(G, start=239)
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
    plt.show()
    #plt.savefig('PPO_25000.png')

if __name__ == '__main__':
    plot_policy()