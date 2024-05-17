import polyscope as ps
from AssemblyEnv.geometry import AssemblyGUI, AssemblyCheckerMosek
from AssemblyEnv.env import RobotPlayground
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts = queue.get()
    assembly = AssemblyCheckerMosek(parts)
    assembly.load_from_file(DATA_DIR + "/block/dome.obj")

    env = RobotPlayground(assembly)

    env.render = True
    env.send_time_delay = 0.2

    model = CateoricalPolicy()
    model.load(f"{DATA_DIR}/../logs/dome_gnn/model/best/policy.pth", device)
    n_part = env.assembly.n_part()
    edge_index, batch_edge_index, edge_attr, batch_edge_attr = env.compute_batch_graph(1, device)
    model.set_graph(edge_index, batch_edge_index, edge_attr, batch_edge_attr, n_part)
    model = model.to(device)

    obs, info = env.reset()
    env.send()
    while True:
        state = torch.tensor(obs, device=device, dtype=torch.float)
        state = state.reshape(-1, n_part * 2)
        action, action_probs, log_action_probs = model.sample(state)
        #action = model.act(state)
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

    parts =   [[[7.5760441129501697, 2.757454550601556], [8.0622577482985509, 0.0], [10.0, 0.0], [9.3969262078590852, 3.420201433256687], [7.5760441129501697, 2.757454550601556]], [[6.1760477470770274, 5.182319386705605], [7.5760441129501697, 2.757454550601556], [9.3969262078590852, 3.420201433256687], [7.6604444311897817, 6.4278760968653907], [6.1760477470770274, 5.182319386705605]], [[4.0311288741492755, 6.9821200218844721], [6.1760477470770274, 5.182319386705605], [7.6604444311897817, 6.4278760968653907], [5.0000000000000027, 8.6602540378443873], [4.0311288741492755, 6.9821200218844721]], [[1.3999963658739316, 7.9397739373070175], [4.0311288741492755, 6.9821200218844721], [5.0000000000000027, 8.6602540378443873], [1.7364817766693057, 9.8480775301220831], [1.3999963658739316, 7.9397739373070175]], [[-1.3999963658787249, 7.9397739373061738], [1.3999963658739316, 7.9397739373070175], [1.7364817766693057, 9.8480775301220831], [-1.7364817766693017, 9.8480775301220831], [-1.3999963658787249, 7.9397739373061738]], [[-4.0311288741492737, 6.9821200218844721], [-1.3999963658787249, 7.9397739373061738], [-1.7364817766693017, 9.8480775301220831], [-4.9999999999999991, 8.6602540378443891], [-4.0311288741492737, 6.9821200218844721]], [[-7.6604444311897808, 6.427876096865397], [-6.1760477470770265, 5.1823193867056068], [-4.0311288741492737, 6.9821200218844721], [-4.9999999999999991, 8.6602540378443891], [-7.6604444311897808, 6.427876096865397]], [[-9.3969262078590852, 3.4202014332566906], [-7.5760441129501599, 2.7574545506015555], [-6.1760477470770265, 5.1823193867056068], [-7.6604444311897808, 6.427876096865397], [-9.3969262078590852, 3.4202014332566906]], [[-10.000000000000004, 3.1086244689504383e-15], [-8.0622577482985527, 2.5062531711346134e-15], [-7.5760441129501599, 2.7574545506015555], [-9.3969262078590852, 3.4202014332566906], [-10.000000000000004, 3.1086244689504383e-15]]]

    # parts = [[[4.0, 3.0], [5.0, 3.0], [5.0, 5.0], [4.0, 5.0]],
    #          [[0.0, 3.0], [1.0, 3.0], [1.0, 5.0], [0.0, 5.0]],
    #          [[2.0, 3.0], [3.0, 3.0], [3.0, 5.0], [2.0, 5.0]],
    #          [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]],
    #          [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
    #          [[4.0, 0.0], [5.0, 0.0], [5.0, 2.0], [4.0, 2.0]],
    #          [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]],
    #          [[0.0, 5.0], [5.0, 5.0], [5.0, 6.0], [0.0, 6.0]]]

    p1 = Process(target=gui, args=(queue1, ))
    p2 = Process(target=test, args=(queue2, ))

    #
    p1.start()
    p2.start()
    queue1.put(parts)
    queue2.put(parts)
    p1.join()
    p2.join()