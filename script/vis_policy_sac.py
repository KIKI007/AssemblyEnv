import polyscope as ps
from AssemblyEnv.geometry import AssemblyGUI, AssemblyCheckerMosek
from AssemblyEnv.env import AssemblyPlayground
from multiprocessing import Process, Queue
from compas_eve import Message
from compas_eve import Subscriber
from compas_eve import Topic
from compas_eve.mqtt import MqttTransport
from AssemblyEnv.sac.model import CateoricalPolicy
import time
from AssemblyEnv import DATA_DIR
import torch

def test(queue):
    parts = queue.get()
    assembly = AssemblyCheckerMosek(parts)
    assembly.load_from_file(DATA_DIR + "/block/dome.obj")

    env = AssemblyPlayground(assembly)
    #env = RobotPlayground(assembly)
    env.render = True
    env.send_time_delay = 0.2
    model = CateoricalPolicy(72, 72)
    model.load(f"{DATA_DIR}/../script/logs/dome/model/final/policy.pth")

    obs, info = env.reset()
    env.send()
    while True:
        state = torch.tensor(obs, device="cpu", dtype=torch.float)
        state = state.reshape(-1, 72)
        #action, action_probs, log_action_probs = model.sample(state)
        action = model.act(state)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if reward < 1:
                time.sleep(5)
            obs, info = env.reset()
            env.send()


def update_viewer(msg):
    global viewer
    viewer.update_status(msg["state"])

def gui(queue):
    global viewer
    parts = queue.get()
    viewer = AssemblyGUI(parts)
    viewer.load_from_file(DATA_DIR + "/block/dome.obj")

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

    torch.multiprocessing.set_start_method('spawn')
    queue1 = Queue()
    queue2 = Queue()
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