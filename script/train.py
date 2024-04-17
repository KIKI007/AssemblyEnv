from AssemblyEnv.reinforce.train import SequenceACPolicy
from AssemblyEnv.geometry import Assembly2D
from AssemblyEnv.reinforce.playground import AssemblyPlayground
from multiprocessing import Process, Queue
from stable_baselines3 import PPO
def train(queue):
    parts = queue.get()

    assembly = Assembly2D(parts)
    env = AssemblyPlayground(assembly)
    env.send_time_delay = 0.1
    env.render = False
    env.reset()
    model = PPO(SequenceACPolicy, env, verbose=1, tensorboard_log="./logs/")
    for epoch in range(100):
        model.learn(total_timesteps=1000, tb_log_name="PPO", reset_num_timesteps= (epoch == 0))
        model.save(f"models/PPO_3/{epoch}")

if __name__ == "__main__":
    # gui()
    queue = Queue()
    parts =   [[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]], [[5.0, 0.0], [6.0, 0.0], [6.0, 3.0], [5.0, 3.0]], [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]], [[5.0, 4.0], [6.0, 4.0], [6.0, 6.0], [5.0, 6.0]], [[2.0, 3.0], [3.0, 3.0], [3.0, 6.0], [2.0, 6.0]], [[2.0, 6.0], [6.0, 6.0], [6.0, 7.0], [2.0, 7.0]], [[7.0, 0.0], [8.0, 0.0], [8.0, 3.0], [7.0, 3.0]], [[10.0, 0.0], [11.0, 0.0], [11.0, 7.0], [10.0, 7.0]], [[2.0, 7.0], [11.0, 7.0], [11.0, 8.0], [2.0, 8.0]], [[3.0, 3.0], [10.0, 3.0], [10.0, 4.0], [3.0, 4.0]]]
    queue.put(parts)
    train(queue)