import gymnasium as gym
import numpy as np

#Dos depredadores y dos presas [Presa1, Presa2, Depredador1, Depredador2]
class Pesca4D_4obs(gym.Env):
    def __init__(self, params):
        self.r = params["r_crecimiento"]
        self.sigma = params["sigma"]
        self.init_B = params["init_B"]
        self.T = params["T"]
        self.mort = params["mortalidad"]
        self.C = params["C"]
        self.eps = params["epsilon"]
        self.arrastre = params["arrastre"]
        self.alpha = 0.05

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

        self.B = self.init_B + self.sigma * self.np_random.normal(1,0.4)
        self.B = np.clip(self.B, 0.0, 1.0)

        self.t = 0

        observation = self.normalize(self.B).astype(np.float32)

        return observation, {}

    def step(self, action):
        #Define la pesca solo para B0
        action = float(action[0])
        harvest_fraction = self.unnormalize(action)
        harvest = self.B[0] * harvest_fraction

        #Obtiene los valores actuales de las poblaciones
        B_init = np.copy(self.B)

        #Evolucion de B0
        self.B[0] = B_init[0] + (self.r[0] * B_init[0] * (1 - B_init[0] / (1 - self.arrastre[0] * B_init[1])) - self.mort[0] * B_init[0] * B_init[3]) + np.random.normal(0, self.alpha) - harvest

        #Evolucion de B1
        self.B[1] = B_init[1] + (self.r[1] * B_init[1] * (1 - B_init[1] / (1 - self.arrastre[0] * B_init[0])) - self.mort[1] * B_init[1] * B_init[2]) + np.random.normal(0, self.alpha)

        #Evolucion de B2
        self.B[2] = B_init[2] + (self.r[2] * B_init[2] * B_init[1] * (1 - B_init[2] / (1 - self.arrastre[1] * B_init[3]))) + np.random.normal(0, self.alpha)

        #Evolucion de B3
        self.B[3] = B_init[3] + (self.r[3] * B_init[3] * B_init[0] * (1 - B_init[3] / (1 - self.arrastre[1] * B_init[2]))) + np.random.normal(0, self.alpha)

        #Normaliza las evoluciones
        self.B = np.clip(self.B, 0.0, 1.0)

        #Funcion  reward
        reward = harvest + 0.4 * self.B[3]

        #Define la observacion que devuelve
        observation = self.normalize(self.B).astype(np.float32)

        #Evolucion temporal
        self.t += 1
        
        terminated = self.t >= self.T
        truncated = False

        return observation, reward, terminated, truncated, {}

    def normalize(self, qty):
        return -1 + 2 * qty

    def unnormalize(self, qty):
        return (qty + 1) / 2
