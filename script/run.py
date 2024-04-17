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
from AssemblyEnv.reinforce.test import test
from AssemblyEnv.reinforce.train import train

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
    parts = [[[4.0, 3.0], [5.0, 3.0], [5.0, 5.0], [4.0, 5.0]],
             [[0.0, 3.0], [1.0, 3.0], [1.0, 5.0], [0.0, 5.0]],
             [[2.0, 3.0], [3.0, 3.0], [3.0, 5.0], [2.0, 5.0]],
             [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]],
             [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
             [[4.0, 0.0], [5.0, 0.0], [5.0, 2.0], [4.0, 2.0]],
             [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]],
             [[0.0, 5.0], [5.0, 5.0], [5.0, 6.0], [0.0, 6.0]]]
    queue1.put(parts)
    train(queue1)
    # p1 = Process(target=gui, args=(queue1, ))
    # p2 = Process(target=train, args=(queue2, ))
    #
    # #
    # p1.start()
    # p2.start()
    # queue1.put(parts)
    # queue2.put(parts)
    # p1.join()
    # p2.join()