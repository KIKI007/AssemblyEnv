import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete, Discrete, MultiBinary, Tuple

from compas_eve import Message
from compas_eve import Subscriber
from compas_eve import Publisher
from compas_eve import Topic
from compas_eve.mqtt import MqttTransport
import time
class AssemblyPlayground(gym.Env):
	def __init__(self, assembly):
		# last part in the assembly is always the ground part
		self.assembly = assembly

		self.observation_space = spaces.Dict(
		{
			"obs" :	 spaces.MultiBinary(self.n_part()),
			"mask": spaces.MultiBinary(self.n_part()),
		})

		self.action_space = Discrete(self.n_part())
		self._state = np.zeros(self.n_part(), dtype=int)
		self._mask = np.zeros(self.n_part(), dtype=int)

		self.send_time_delay = 1
		self.render = False

	def terminate_label(self):
		label = np.ones(self.n_part())
		label = np.append(label, [2])
		return label

	def current_label(self):
		label = np.copy(self._state)
		label = np.append(label, [2])
		return label

	def _get_obs(self):
		return {"obs": np.copy(self._state), "mask": np.copy(self._mask)}

	def _get_info(self):
		return {
			"finished": (self.current_label() == self.terminate_label()).all()
		}

	def n_part(self):
		return self.assembly.n_part() - 1

	def n_install(self):
		return np.sum(self._state)

	def seed(self, seed):
		super().reset(seed=seed)

	def reset_random(self, seed=None, options=None):
		super().reset(seed=seed)

		while True:
			self._state = np.random.randint(2, size = self.n_part())
			self._mask = np.copy(self._state)
			if (self.n_install() < self.n_part()
					and self.assembly.check_stability(self.current_label()) != None):
				break

		observation = self._get_obs()
		info = self._get_info()
		self.send()
		return observation, info

	def reset(self, seed=None, options=None):
		# We need the following line to seed self.np_random
		super().reset(seed=seed)
		self._state = np.zeros(self.n_part(), dtype=int)
		self._mask = np.zeros(self.n_part(), dtype=int)
		self.compute_mask()
		self.assembly.reset()

		observation = self._get_obs()
		info = self._get_info()
		return observation, info

	def step(self, action):

		terminated = False
		reward = 0.0

		if (self.terminate_label() == self.current_label()).all():
			terminated = True
			reward = 1
		else:
			self._state[action] = 1
			self.compute_mask()
			if np.sum(self._mask) == self.n_part():
				terminated = True
				reward = -1

		if reward != 0:
			print(*self._state, sep=', ')

		observation = self._get_obs()
		info = self._get_info()

		self.send()

		return observation, reward, terminated, False, info

	def compute_mask(self):
		self._mask = np.copy(self._state)
		copy_state = np.copy(self._state)
		for i in range(self.n_part()):
			if self._mask[i] == 0:
				self._state = np.copy(copy_state)
				self._state[i] = 1
				status = self.assembly.check_stability(self.current_label())
				if status == None:
					self._mask[i] = 1
		self._state = np.copy(copy_state)

	def send(self):
		if self.render:
			data = {"state": self.current_label()}
			topic = Topic("/rl/sequence/", Message)
			tx = MqttTransport(host="localhost")
			publisher = Publisher(topic, transport=tx)
			msg = Message(data)
			publisher.publish(msg)
			time.sleep(self.send_time_delay)

	def close(self):
		pass