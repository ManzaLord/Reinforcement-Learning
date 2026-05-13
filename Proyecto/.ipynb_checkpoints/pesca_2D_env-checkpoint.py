import gymnasium as gym
import numpy as np

class Pesca2D(gym.Env):
    """
    params = 
    {
    r, sigma=0.05, init_B=0.5, T=200, mortalidad, C, epsilon
    }"""
    def __init__(self, params):
        self.r = params["r_crecimiento"]
        self.sigma = params["sigma"]
        self.init_B = params["init_B"]
        self.T = params["T"]
        self.mort = params["mortalidad"]
        self.C = params["C"]
        self.eps = params["epsilon"]

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
        self.B[0] = self.B[0] + self.r[0] * self.B[0] (1 - self.B[0]) - self.B[0] * self.B[1] * self.mort[0]

        self.B[1] = self.B[1] + self.r[1] * self.B[1] * (1 - self.B[1] / (self.B[0] * self.C[1] + self.eps))
        
        #self.B += self.sigma * self.np_random.normal()

        self.B = np.clip(self.B, 0.0, 1.0)

        reward = 0.1 * self.B[1]

        observation = np.array([self.normalize(self.B[0])], dtype=np.float32)

        self.t += 1

        terminated = self.t >= self.T
        truncated = False

        return observation, reward, terminated, truncated, {}

    def normalize(self, qty):
        return -1 + 2 * qty

    def unnormalize(self, qty):
        return (qty + 1) / 2