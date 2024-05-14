from AssemblyEnv.reinforce.policy import AssemblyACPolicy, AssemblyACExtractor
from AssemblyEnv.geometry import AssemblyChecker, AssemblyCheckerMosek
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from AssemblyEnv.reinforce.env import AssemblyPlayground
from multiprocessing import Process, Queue
from stable_baselines3 import PPO
from AssemblyEnv import DATA_DIR
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
import gymnasium as gym

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
    env = AssemblyPlayground(assembly)
    policy_kwargs = dict(
        features_extractor_class=AssemblyACExtractor,
        features_extractor_kwargs=dict(features_dim=env.observation_space['obs'].shape[0]),
    )
    model = PPO(AssemblyACPolicy, env, verbose=1, tensorboard_log=f"./logs/{name}", policy_kwargs=policy_kwargs)

    for epoch in range(0, 1000):
        model.learn(total_timesteps=100000, reset_num_timesteps= (epoch == 0))
        model.save(f"models/{name}/PPO/{epoch}")

if __name__ == "__main__":
    # gui()
    queue = Queue()
    assembly = AssemblyCheckerMosek()
    assembly.load_from_file(DATA_DIR + "/block/dome.obj")
    train_1robot(assembly, "dome")