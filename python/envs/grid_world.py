import time
from multiprocessing import Process, Queue, Pipe
import polyscope as ps
import numpy as np
import pygame
import py_rigidblock as pyrb

import multiprocessing as mp

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, dimensions=5):
        self.dimensions = dimensions  # The size of the square grid
        self.assembly = pyrb.Assembly()
        self._block_size = np.array([0.5, 1, 0.5])
        self._refresh = False

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                # node features: (select, center_x, center_y, center_z)
                # edge features: (center_x, center_y, center_z, normal_x, normal_y, normal_z)
                "graph": spaces.Graph(node_space=spaces.Box(low = -dimensions / 2, high = dimensions / 2, shape=(4, ), dtype = float),
                                      edge_space=spaces.Box(low = -dimensions / 2, high = dimensions / 2, shape=(6, ), dtype = float)),

                # target_features: (center_x, center_y, center_z)
                "target": spaces.Box(low = -dimensions / 2, high = dimensions / 2, shape=(3,), dtype=float),
            }
        )

        # We have 1 actions, place cuboid in angle of [0, 2 * pi) direction with respect to the select block
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=float)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {"agent": self._agent_location,
                "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.assembly = pyrb.Assembly()

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.random(3, )
        self._agent_location[0] = 0
        self._agent_location[2] = self._block_size[2] / 2

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.linalg.norm(self._agent_location -  self._target_location) < 0.5:
            self._target_location = self.np_random.random(3, )
            self._target_location[0] = 0
            self._target_location[2] = self._block_size[2] / 2

        self.add_part()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def add_part(self):
        part = pyrb.Part.cuboid(self._agent_location, self._block_size)
        self.assembly.add_part(part)
        plane = self.assembly.ground()
        self.assembly.set_boundary(plane, "fix")

    def step(self, action):

        self._agent_location[1] += (action[1] - 0.5) * 2 * self._block_size[1]

        if action[0] < 0.5:
            self._agent_location[2] -= self._block_size[2]
        else:
            self._agent_location[2] += self._block_size[2]
            
        self.add_part()

        terminated = False

        if self.assembly.self_collision():
            print("collison")
            terminated = True

        if np.linalg.norm(self._agent_location - self._target_location) <= 0.1:
            terminated = True
            print("success")


        if self._agent_location[2] < 0:
            terminated = True
            print("below ground")

        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        if self.render_mode == "human":
            self._refresh = True

    def close(self):
        pass

def simulation_process(conn):
    env = GridWorldEnv(render_mode="human")
    observation, info = env.reset()
    for it in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        conn.send(env.assembly)

        if terminated or truncated:
            observation, info = env.reset()


        time.sleep(0.5)

def user_callback():
    global env
    print(parent_conn.recv())
    if env._refresh:
        print(env._refresh)

        ps.remove_all_structures()
        ps.remove_all_groups()
        assembly_group = ps.create_group("assembly")
        print(env.assembly.n_part())
        for part_id in range(env.assembly.n_part()):
            part = env.assembly.part(part_id)
            if part.fixed == True:
                color = (0, 0, 0)
            else:
                color = (1, 1, 1)
            obj = ps.register_surface_mesh("part" + str(part_id), part.V, part.F, color=color)
            obj.set_edge_width(1)
            obj.add_to_group(assembly_group)
        assembly_group.set_hide_descendants_from_structure_lists(True)
        env._refresh = False

def gui_process(conn):
    ps.init()
    ps.set_navigation_style("turntable")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.look_at((0., -5., 3.5), (0., 0., 0.))
    ps.set_user_callback(user_callback)
    ps.show()

if __name__ == "__main__":
    parent_conn, child_conn = Pipe()
    p1 = Process(target=gui_process, args=(parent_conn,))
    p2 = Process(target=simulation_process, args=(child_conn, ))
    p1.start()
    p2.start()
    p2.join()
    p1.join()







