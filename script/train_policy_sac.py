from AssemblyEnv.geometry import AssemblyChecker, AssemblyCheckerMosek
from AssemblyEnv.reinforce.env import AssemblyPlayground, RobotPlayground
from AssemblyEnv import DATA_DIR
from AssemblyEnv.sac.agent.sacd import SacdAgent

def train_1robot(assembly, name = ""):
    env = AssemblyPlayground(assembly)
    env.render = False
    env.reset()
    agent = SacdAgent(env, env, f"./logs/{name}", num_steps=1E7, batch_size=128, start_steps=1000, lr =1E-4, use_per=False, target_entropy_ratio = 0.4)
    agent.run()

if __name__ == "__main__":
    assembly = AssemblyCheckerMosek()
    assembly.load_from_file(DATA_DIR + "/block/dome.obj")
    train_1robot(assembly, "dome")