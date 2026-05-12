import gymnasium as gym
import numpy as np

class Pesca1D(gym.Env):
    def __init__(self, r, sigma=0.05, init_B=0.5, T=200):
        self.r = r
        self.sigma = sigma
        self.init_B = init_B
        self.T = T

        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.B = self.init_B + self.sigma * self.np_random.normal()
        self.B = np.clip(self.B, 0.0, 1.0)

        self.t = 0

        return np.array([self.normalize(self.B)], dtype=np.float32), {}

    def step(self, action):
        action = float(action[0])

        harvest_fraction = self.unnormalize(action)
        harvest = self.B * harvest_fraction

        # dinámica
        self.B = (
            self.B
            + self.r * self.B * (1 - self.B)
            - harvest
        )

        self.B += self.sigma * self.np_random.normal()

        self.B = np.clip(self.B, 0.0, 1.0)

        reward = harvest

        observation = np.array([self.normalize(self.B)], dtype=np.float32)

        self.t += 1

        terminated = self.t >= self.T
        truncated = False

        return observation, reward, terminated, truncated, {}

    def normalize(self, qty):
        return -1 + 2 * qty

    def unnormalize(self, qty):
        return (qty + 1) / 2