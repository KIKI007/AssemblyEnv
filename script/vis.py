import os
import numpy as np
import polyscope as ps
from AssemblyEnv.geometry import Assembly2D, Assembly2D_GUI
from AssemblyEnv.reinforce.playground import AssemblyPlayground
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

def test(queue):
    parts = queue.get()
    assembly = Assembly2D(parts)
    env = AssemblyPlayground(assembly)
    env.render = True
    env.send_time_delay = 1
    model = PPO.load(f"models/PPO_3/{22}", env)

    obs, info = env.reset_random()
    env.send()
    for it in range(100):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            return
            #obs, info = env.reset_random()


def update_viewer(msg):
    global viewer
    viewer.update_status(msg["state"])

def gui(queue):
    global viewer
    parts = queue.get()
    viewer = Assembly2D_GUI(parts)

    ps.init()
    ps.set_navigation_style("turntable")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.look_at((0., -5., 3.5), (0., 0., 0.))

    tx = MqttTransport("localhost")
    topic = Topic("/rl/sequence/", Message)
    subscriber = Subscriber(topic, callback=update_viewer, transport=tx)
    subscriber.subscribe()

    viewer.render()
    ps.set_user_callback(viewer.interface)

    ps.show()

if __name__ == "__main__":
    # gui()
    queue1 = Queue()
    queue2 = Queue()
    # parts = [[[4.0, 3.0], [5.0, 3.0], [5.0, 5.0], [4.0, 5.0]],
    #          [[0.0, 3.0], [1.0, 3.0], [1.0, 5.0], [0.0, 5.0]],
    #          [[2.0, 3.0], [3.0, 3.0], [3.0, 5.0], [2.0, 5.0]],
    #          [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]],
    #          [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
    #          [[4.0, 0.0], [5.0, 0.0], [5.0, 2.0], [4.0, 2.0]],
    #          [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]],
    #          [[0.0, 5.0], [5.0, 5.0], [5.0, 6.0], [0.0, 6.0]]]
    #parts = [[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]], [[5.0, 0.0], [6.0, 0.0], [6.0, 3.0], [5.0, 3.0]], [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]], [[5.0, 4.0], [6.0, 4.0], [6.0, 6.0], [5.0, 6.0]], [[3.0, 3.0], [6.0, 3.0], [6.0, 4.0], [3.0, 4.0]], [[2.0, 3.0], [3.0, 3.0], [3.0, 6.0], [2.0, 6.0]], [[2.0, 6.0], [6.0, 6.0], [6.0, 7.0], [2.0, 7.0]]]
    parts =   [[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]], [[5.0, 0.0], [6.0, 0.0], [6.0, 3.0], [5.0, 3.0]], [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]], [[5.0, 4.0], [6.0, 4.0], [6.0, 6.0], [5.0, 6.0]], [[2.0, 3.0], [3.0, 3.0], [3.0, 6.0], [2.0, 6.0]], [[2.0, 6.0], [6.0, 6.0], [6.0, 7.0], [2.0, 7.0]], [[7.0, 0.0], [8.0, 0.0], [8.0, 3.0], [7.0, 3.0]], [[10.0, 0.0], [11.0, 0.0], [11.0, 7.0], [10.0, 7.0]], [[2.0, 7.0], [11.0, 7.0], [11.0, 8.0], [2.0, 8.0]], [[3.0, 3.0], [10.0, 3.0], [10.0, 4.0], [3.0, 4.0]]]
    p1 = Process(target=gui, args=(queue1, ))
    p2 = Process(target=test, args=(queue2, ))

    #
    p1.start()
    p2.start()
    queue1.put(parts)
    queue2.put(parts)
    p1.join()
    p2.join()