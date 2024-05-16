from AssemblyEnv.geometry import AssemblyChecker, AssemblyCheckerMosek
from AssemblyEnv.env import AssemblyPlayground
from AssemblyEnv import DATA_DIR
from AssemblyEnv.sac.agent.sacd import SacdAgent

def train_1robot(assembly, name = ""):
    env = AssemblyPlayground(assembly)
    env.render = False
    env.reset()
    agent = SacdAgent(env, env, f"./logs/{name}", num_steps=1E7, batch_size=128, start_steps=1000, lr =1E-4, use_per=False, target_entropy_ratio = 0.4)
    agent.run()

if __name__ == "__main__":
    parts = [[[4.0, 3.0], [5.0, 3.0], [5.0, 5.0], [4.0, 5.0]],
             [[0.0, 3.0], [1.0, 3.0], [1.0, 5.0], [0.0, 5.0]],
             [[2.0, 3.0], [3.0, 3.0], [3.0, 5.0], [2.0, 5.0]],
             [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]],
             [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
             [[4.0, 0.0], [5.0, 0.0], [5.0, 2.0], [4.0, 2.0]],
             [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]],
             [[0.0, 5.0], [5.0, 5.0], [5.0, 6.0], [0.0, 6.0]]]
    assembly = AssemblyCheckerMosek(parts)
    train_1robot(assembly, "stack")