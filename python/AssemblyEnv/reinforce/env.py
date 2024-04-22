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

		self.observation_space = MultiBinary(self.n_state())
		self.action_space = Discrete(self.n_state())
		self._state = np.zeros(self.n_state(), dtype=int)

		self.send_time_delay = 1
		self.render = False

	def terminate_status(self):
		status = np.ones(self.n_state())
		status = np.append(status, [2])
		return status

	def part_status(self):
		status = np.copy(self._state)
		status = np.append(status, [2])
		return status

	def _get_obs(self):
		return self._state

	def _get_info(self):
		return {
			"finished": (self.part_status() == self.terminate_status()).all()
		}

	def n_state(self):
		return self.assembly.n_part() - 1

	def n_installed(self):
		return np.sum(self._state)

	def reset_random(self, seed=None, options=None):
		super().reset(seed=seed)

		while True:
			self._state = np.random.randint(2, size = self.n_state())
			if (self.n_installed() < self.n_state()
					and self.assembly.check_stability(self.part_status()) != None):
				break

		observation = self._get_obs()
		info = self._get_info()
		self.send()
		return observation, info

	def reset(self, seed=None, options=None):
		# We need the following line to seed self.np_random
		super().reset(seed=seed)
		self._state = np.zeros(self.assembly.n_part() - 1, dtype=int)

		observation = self._get_obs()
		info = self._get_info()
		return observation, info

	def step(self, action):

		terminated = False
		reward = 0

		# cannot take duplicate action
		if self._state[action] != 0:
			terminated = True
			reward = -1
		else:
			self._state[action] = 1
			if self.assembly.check_stability(self.part_status()) == None:
				terminated = True
				reward = -1
			elif (self.terminate_status() == self.part_status()).all():
				terminated = True
				reward = 1

		observation = self._get_obs()
		info = self._get_info()

		self.send()

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


class RobotPlayground(gym.Env):
	def __init__(self, assembly):
		# last part in the assembly is always the ground part
		self.assembly = assembly

		self.observation_space = MultiBinary(self.n_state())

		self.action_space = Discrete(2 * self.n_part())

		self._state = np.zeros(self.n_state(), dtype=int)

		self.send_time_delay = 1
		self.render = False

		# different between label and state
		# label is the full descirption of the assembly's status
	 	# state only covers the status of avaiable parts
	def terminate_label(self):
		label = np.ones(self.n_part())
		label = np.append(label, [2])
		return label
	def current_label(self):
		label = np.copy(self.install_state()) +  np.copy(self.fixed_state())
		label = np.append(label, [2])

		return label

	def install_state(self):
		return self._state[: self.n_part()]

	def fixed_state(self):
		return self._state[self.n_part() :]

	def _get_obs(self):
		return self._state

	def _get_info(self):
		return {"sucess" :
					(self.current_label() == self.terminate_label()).all()
				}

	def n_action(self):
		return 2 * self.n_part()

	def n_state(self):
		return 2 * self.n_part()

	def n_part(self):
		return self.assembly.n_part() - 1

	def n_robot(self):
		return 2

	def n_installed(self):
		state = self.install_state()
		return np.sum(state)

	def clear_state(self):
		self._state = np.zeros(self.n_state(), dtype=int)

	def set_install_state(self, state):
		self._state[:self.n_part()] = np.copy(state)

	def set_fixed_state(self, state):
		self._state[self.n_part(): ] = np.copy(state)

	def reset_random(self, seed=None, options=None):
		super().reset(seed=seed)

		while True:
			rnd_state = np.random.randint(2, size = self.n_part())
			self.clear_state()
			self.set_install_state(rnd_state)
			if (self.n_installed() < self.n_part()
					and self.assembly.check_stability(self.current_label()) != None):
				break
		observation = self._get_obs()
		info = self._get_info()
		self.send()
		return observation, info

	def reset(self, seed=None, options=None):
		# We need the following line to seed self.np_random
		super().reset(seed=seed)
		self._state = np.zeros(self.n_state(), dtype=int)
		observation = self._get_obs()
		info = self._get_info()
		return observation, info

	def check_step(self, action):

		#print(self.current_label(), end=", ")

		action_type = "install"
		part_id = action
		if action >= self.n_part():
			action_type = "release"
			part_id -= self.n_part()

		install_state = np.copy(self.install_state())
		fixed_state = np.copy(self.fixed_state())

		# pick an already installed part id
		if action_type == "install" and install_state[part_id] == 1:
			#print("pick an already installed part id")
			return False

		# more than available robot
		if action_type == "install" and np.sum(fixed_state) >= self.n_robot():
			#print("more than available robot")
			return False

		# release part id that are not fixed
		if action_type == "release" and fixed_state[part_id] == 0:
			#print("release part id that are not fixed")
			return False

		if action_type == "install":
			install_state[part_id] = 1
			fixed_state[part_id] = 1
			self.set_install_state(install_state)
			self.set_fixed_state(fixed_state)
			return True

		if action_type == "release":
			fixed_state[part_id] = 0
			self.set_install_state(install_state)
			self.set_fixed_state(fixed_state)
			if self.assembly.check_stability(self.current_label()) == None:
				#print("not stable")
				return False
			else:
				return True

		return False

	def step(self, action):
		terminated = False
		reward = 0

		if not self.check_step(action):
			reward = -1
			terminated = True
		else:
			if (self.terminate_label() == self.current_label()).all():
				reward = 1
				terminated = terminated or True

		observation = self._get_obs()
		info = self._get_info()
		self.send()
		return observation, reward, terminated, False, info

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