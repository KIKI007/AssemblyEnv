import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete, Discrete, MultiBinary

from compas_eve import Message
from compas_eve import Subscriber
from compas_eve import Publisher
from compas_eve import Topic
from compas_eve.mqtt import MqttTransport
import time

class AssemblyEnv(gym.Env):
	metadata = {}

	def __init__(self, assembly):
		self.assembly = assembly
		self.observation_space = MultiBinary(self.assembly.n_part() - 1)
		self.action_space = Discrete(self.assembly.n_part() - 1)
		self._part_status = np.zeros(self.assembly.n_part() - 1, dtype=int)
		self.iter = 0
		self.send_time_delay = 1
		self.render = False


	def _get_obs(self):
		return self._part_status

	def _get_info(self):
		return {
			"num_installed_parts": np.sum(self._part_status)
		}

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		while True:
			self._part_status = np.random.randint(2, size=self.assembly.n_part() - 1)
			if np.sum(self._part_status) < self.assembly.n_part() - 1 and self.assembly.check_stability(self.part_status()) != None:
				break
		# self._part_status = np.array([0, 0, 0, 1, 1, 0, 1, 0])
		# self._part_status = np.zeros(self.assembly.n_part() - 1, dtype=int)
		# self._part_status[5] = 1
		# self._part_status[3] = 1
		# self._part_status[6] = 1
		#print(f"{self.part_status()}:Init")
		observation = self._get_obs()
		info = self._get_info()
		self.send()

		return observation, info

	# def reset(self, seed=None, options=None):
	# 	# We need the following line to seed self.np_random
	# 	super().reset(seed=seed)
	#
	# 	self._part_status = np.zeros(self.assembly.n_part() - 1, dtype=int)
	#
	# 	observation = self._get_obs()
	# 	info = self._get_info()
	#
	# 	return observation, info

	def part_status(self):
		status = np.copy(self._part_status)
		status = np.append(status, [2])
		return status

	def step(self, action):

		terminated = False
		reward = 0

		if self._part_status[action] != 0:
			terminated = True
			reward = -1
			if self.render:
				print(f"{self.part_status()}:Failed")
		else:
			self._part_status[action] = 1
			if self.assembly.check_stability(self.part_status()) == None:
				terminated = True
				reward = -1
				if self.render:
					print(f"{self.part_status()}:Failed")
			elif np.sum(self._part_status) == self.assembly.n_part() - 1:
				terminated = True
				reward = 1
				if self.render:
					print(f"{self.part_status()}:Sucess")

		observation = self._get_obs()
		info = self._get_info()

		self.send()
		self.iter = self.iter + 1

		return observation, reward, terminated, False, info

	def send(self):
		if self.render:
			data = {"state": self.part_status()}
			topic = Topic("/rl/sequence/", Message)
			tx = MqttTransport(host="localhost")
			publisher = Publisher(topic, transport=tx)
			msg = Message(data)
			publisher.publish(msg)
			time.sleep(self.send_time_delay)

	def close(self):
		pass
