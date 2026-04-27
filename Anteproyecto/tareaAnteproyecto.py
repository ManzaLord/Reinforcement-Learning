import gymnasium as gym
import numpy as np

class Pesca1D(gym.Env):
	def __init__(self, r, sigma = 0.05, init_B = 0.5, T = 200) -> None:
		#
		self.r = r # tasa de crecimiento exponencial
		self.sigma = sigma # fuerza de la estocasticidad
		self.init_B = init_B
		self.T = T # duracion de episodio
		#
		self.observation_space = gym.spaces.Box(
			np.array([-1], dtype=np.float32),
      np.array([1], dtype=np.float32),
      dtype=np.float32, # mejor para correr cálculos en GPU
		) # observaciones que recibe la función de decisión
		self.action_space = gym.spaces.Box(
			np.array([-1], dtype=np.float32),
      np.array([1], dtype=np.float32),
      dtype=np.float32, 
		) # acciones que toma la función de decisión
		#
	def reset(self, *, seed=None, options=None):
		""" 
		Función que resetea un ambiente cuando se comienza un episodio 
		(cuando se comienza en el tiempo 0 una simulacion) 
		"""
		self.B = self.init_B + self.sigma * np.random.normal()
		self.t = 0
		observation = self.normalize(self.B)
		info = {}
		#
		return observation, {}

	def step(self, action):
		# action en [-1, 1]
		harvest = self.B * self.unnormalize(action)
		self.B = (
			self.B
			+ self.r * self.B * (1 - self.B) 
			- harvest
		)
		#
		reward = harvest
		observation = self.normalize(B)
		#
		terminated = self.t > self.T
		truncated = False
		info={}
		#
		self.t += 1
		#
		return observation, reward, terminated, truncated, info

	def normalize(self, qty):
		# [0,1] -> [-1,1]
		return -1 + 2 * qty

	def unnormalize(self, qty):
		# [-1, 1] -> [0,1]
		return (qty + 1) / 2
