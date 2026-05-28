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


class Pesca2D(gym.Env):
    def __init__(self, params):
        self.r = params["r_crecimiento"]
        self.sigma = params["sigma"]
        self.init_B = params["init_B"]
        self.T = params["T"]
        self.mort = params["mortalidad"]
        self.C = params["C"]
        self.eps = params["epsilon"]

        self.observation_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2,),
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

        observation = np.array([self.normalize(self.B[0]), self.normalize(self.B[1])], dtype=np.float32)

        return observation, {}

    def step(self, action):
        action = float(action[0])

        harvest_fraction = self.unnormalize(action)
        harvest = self.B[0] * harvest_fraction

        # dinámica
        B_temp = np.copy(self.B[0])
        
        self.B[0] = self.B[0] + (self.r[0] * self.B[0] * (1 - self.B[0]) - self.B[0] * self.B[1] * self.mort[0]) * np.random.normal(1,0.4) - harvest

        self.B[1] = self.B[1] + self.r[1] * self.B[1] * (1 - self.B[1] / (B_temp * self.C + self.eps)) * np.random.normal(1,0.4)
        
        #self.B += self.sigma * self.np_random.normal()

        self.B = np.clip(self.B, 0.0, 1.0)

        reward = 0.1 * self.B[1] +  harvest

        observation = np.array([self.normalize(self.B[0]), self.normalize(self.B[1])], dtype=np.float32)

        self.t += 1

        terminated = self.t >= self.T
        truncated = False

        return observation, reward, terminated, truncated, {}

    def normalize(self, qty):
        return -1 + 2 * qty

    def unnormalize(self, qty):
        return (qty + 1) / 2


    import gymnasium as gym
import numpy as np

#Dos depredadores y dos presas
class Pesca4D(gym.Env):
    def __init__(self, params):
        self.r = params["r_crecimiento"]
        self.sigma = params["sigma"]
        self.init_B = params["init_B"]
        self.T = params["T"]
        self.mort = params["mortalidad"]
        self.C = params["C"]
        self.eps = params["epsilon"]

        self.observation_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4,),
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

        observation = np.array([self.normalize(self.B[0])], dtype=np.float32)

        return observation, {}

    def step(self, action):
        #Define la pesca solo para B0
        action = float(action[0])
        harvest_fraction = self.unnormalize(action)
        harvest = self.B[0] * harvest_fraction

        #Obtiene los valores actuales de las poblaciones
        B_init = np.copy(self.B)

        #Evolucion de B0
        self.B[0] = self.r[0] * B_init[0] * (1 - B_init[0] / (1 - """"a""" * B_init[1])) - sel.mort[0] * B_init[0] * B_init[3]

        #Evolucion de B1
        self.B[1] = self.r[1] * B_init[1] * (1 - B_init[1] / (1 - """"a""" * B_init[0])) - self.mort[1] * B_init[1] * B_init[2]

        #Evolucion de B2
        self.B[2] = self.r[2] * B_init[2] * B_init[1] (1 - B_init[2] / (1 - """"b""" * B_init[3]))

        #Evolucion de B3
        self.B[3] = self.r[3] * B_init[3] * B_init[0] (1 - B_init[3] / (1 - """"b""" * B_init[2]))

        #Normaliza las evoluciones
        self.B = np.clip(self.B, 0.0, 1.0)

        #Funcion  reward
        reward = harvest + 0.2 * self.B[3]

        #Define la observacion que devuelve
        observation = np.array([self.normalize(self.B[0])], dtype=np.float32)

        #Evolucion temporal
        self.t += 1
        
        terminated = self.t >= self.T
        truncated = False

        return observation, reward, terminated, truncated, {}

    def normalize(self, qty):
        return -1 + 2 * qty

    def unnormalize(self, qty):
        return (qty + 1) / 2