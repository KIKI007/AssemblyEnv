from AssemblyEnv.reinforce.policy import AssemblyACPolicy, RobotACPolicy
from AssemblyEnv.geometry import AssemblyChecker, AssemblyCheckerMosek
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from AssemblyEnv.reinforce.env import AssemblyPlayground, RobotPlayground
from multiprocessing import Process, Queue
from stable_baselines3 import PPO
from AssemblyEnv import DATA_DIR
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
import gymnasium as gym
def train_2robot(assembly):
    env = RobotPlayground(assembly)
    env.render = False
    env.reset()
    model = PPO(RobotACPolicy, env, verbose=1, learning_rate=0.00003, tensorboard_log="./logs/robot2", policy_kwargs=(dict(n_robot = env.n_robot(), n_part = env.n_part())))
    for epoch in range(0, 50):
        model.learn(total_timesteps=10000, reset_num_timesteps=(epoch == 0))
        model.save(f"models/PPO_mask/{epoch}")

def make_env() -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        assembly = AssemblyCheckerMosek()
        filename = DATA_DIR + "/block/dome.obj"
        assembly.load_from_file(filename)
        env = AssemblyPlayground(assembly)
        env.render = False
        env.reset()
        return env
    return _init

def train_1robot(assembly, name = ""):
    num_cpu = 16
    vec_env = make_vec_env(make_env(), n_envs = num_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO(AssemblyACPolicy, vec_env, verbose=1, tensorboard_log=f"./logs/{name}")

    for epoch in range(0, 1000):
        model.learn(total_timesteps=100000, reset_num_timesteps= (epoch == 0))
        model.save(f"models/{name}/PPO/{epoch}")

if __name__ == "__main__":
    # gui()
    queue = Queue()
    # parts = [[[4.0, 3.0], [5.0, 3.0], [5.0, 5.0], [4.0, 5.0]],
    #          [[0.0, 3.0], [1.0, 3.0], [1.0, 5.0], [0.0, 5.0]],
    #          [[2.0, 3.0], [3.0, 3.0], [3.0, 5.0], [2.0, 5.0]],
    #          [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]],
    #          [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
    #          [[4.0, 0.0], [5.0, 0.0], [5.0, 2.0], [4.0, 2.0]],
    #          [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]],
    #          [[0.0, 5.0], [5.0, 5.0], [5.0, 6.0], [0.0, 6.0]]]
    #parts =   [[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]], [[5.0, 0.0], [6.0, 0.0], [6.0, 3.0], [5.0, 3.0]], [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]], [[5.0, 4.0], [6.0, 4.0], [6.0, 6.0], [5.0, 6.0]], [[2.0, 3.0], [3.0, 3.0], [3.0, 6.0], [2.0, 6.0]], [[2.0, 6.0], [6.0, 6.0], [6.0, 7.0], [2.0, 7.0]], [[7.0, 0.0], [8.0, 0.0], [8.0, 3.0], [7.0, 3.0]], [[10.0, 0.0], [11.0, 0.0], [11.0, 7.0], [10.0, 7.0]], [[2.0, 7.0], [11.0, 7.0], [11.0, 8.0], [2.0, 8.0]], [[3.0, 3.0], [10.0, 3.0], [10.0, 4.0], [3.0, 4.0]]]
    #parts =   [[[7.5760441129501697, 2.757454550601556], [8.0622577482985509, 0.0], [10.0, 0.0], [9.3969262078590852, 3.420201433256687], [7.5760441129501697, 2.757454550601556]], [[6.1760477470770274, 5.182319386705605], [7.5760441129501697, 2.757454550601556], [9.3969262078590852, 3.420201433256687], [7.6604444311897817, 6.4278760968653907], [6.1760477470770274, 5.182319386705605]], [[4.0311288741492755, 6.9821200218844721], [6.1760477470770274, 5.182319386705605], [7.6604444311897817, 6.4278760968653907], [5.0000000000000027, 8.6602540378443873], [4.0311288741492755, 6.9821200218844721]], [[1.3999963658739316, 7.9397739373070175], [4.0311288741492755, 6.9821200218844721], [5.0000000000000027, 8.6602540378443873], [1.7364817766693057, 9.8480775301220831], [1.3999963658739316, 7.9397739373070175]], [[-1.3999963658787249, 7.9397739373061738], [1.3999963658739316, 7.9397739373070175], [1.7364817766693057, 9.8480775301220831], [-1.7364817766693017, 9.8480775301220831], [-1.3999963658787249, 7.9397739373061738]], [[-4.0311288741492737, 6.9821200218844721], [-1.3999963658787249, 7.9397739373061738], [-1.7364817766693017, 9.8480775301220831], [-4.9999999999999991, 8.6602540378443891], [-4.0311288741492737, 6.9821200218844721]], [[-7.6604444311897808, 6.427876096865397], [-6.1760477470770265, 5.1823193867056068], [-4.0311288741492737, 6.9821200218844721], [-4.9999999999999991, 8.6602540378443891], [-7.6604444311897808, 6.427876096865397]], [[-9.3969262078590852, 3.4202014332566906], [-7.5760441129501599, 2.7574545506015555], [-6.1760477470770265, 5.1823193867056068], [-7.6604444311897808, 6.427876096865397], [-9.3969262078590852, 3.4202014332566906]], [[-10.000000000000004, 3.1086244689504383e-15], [-8.0622577482985527, 2.5062531711346134e-15], [-7.5760441129501599, 2.7574545506015555], [-9.3969262078590852, 3.4202014332566906], [-10.000000000000004, 3.1086244689504383e-15]]]
    #assembly = Assembly2D(parts)
    assembly = AssemblyCheckerMosek()
    assembly.load_from_file(DATA_DIR + "/block/dome.obj")
    train_1robot(assembly, "dome")