import gymnasium as gym
import numpy as np

# ============================================================
# Entorno con una sola población
# ============================================================
class Pesca1D(gym.Env):

    def __init__(self, r, sigma=0.05, init_B=0.5, T=200):
        # Parámetros del modelo
        self.r = r                    # Tasa de crecimiento
        self.sigma = sigma            # Intensidad del ruido
        self.init_B = init_B          # Población inicial
        self.T = T                    # Horizonte temporal

        # Espacio de observación: población normalizada en [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Espacio de acciones: fracción de pesca normalizada
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        # Reinicia el entorno
        super().reset(seed=seed)

        # Inicializa la población con ruido aleatorio
        self.B = self.init_B + self.sigma * self.np_random.normal()
        self.B = np.clip(self.B, 0.0, 1.0)

        # Reinicia el tiempo
        self.t = 0

        # Devuelve la observación inicial
        return np.array([self.normalize(self.B)], dtype=np.float32), {}

    def step(self, action):
        # Convierte la acción a una fracción de pesca
        action = float(action[0])

        harvest_fraction = self.unnormalize(action)

        # Biomasa extraída
        harvest = self.B * harvest_fraction

        # Evolución logística de la población
        self.B = (
            self.B
            + self.r * self.B * (1 - self.B)
            - harvest
        )

        # Agrega ruido ambiental
        self.B += self.sigma * self.np_random.normal()

        # Restringe la población al intervalo permitido
        self.B = np.clip(self.B, 0.0, 1.0)

        # La recompensa corresponde únicamente a la pesca realizada
        reward = harvest

        # Observación normalizada
        observation = np.array([self.normalize(self.B)], dtype=np.float32)

        # Avanza el tiempo
        self.t += 1

        # Finaliza el episodio al alcanzar T pasos
        terminated = self.t >= self.T
        truncated = False

        return observation, reward, terminated, truncated, {}

    def normalize(self, qty):
        # Convierte [0,1] → [-1,1]
        return -1 + 2 * qty

    def unnormalize(self, qty):
        # Convierte [-1,1] → [0,1]
        return (qty + 1) / 2


# ============================================================
# Entorno con una presa y un depredador
# ============================================================
class Pesca2D(gym.Env):

    def __init__(self, params):
        # Parámetros del modelo
        self.r = params["r_crecimiento"]
        self.sigma = params["sigma"]
        self.init_B = params["init_B"]
        self.T = params["T"]
        self.mort = params["mortalidad"]
        self.C = params["C"]
        self.eps = params["epsilon"]

        # Espacio de observaciones: presa y depredador
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Espacio de acciones
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        # Reinicia el entorno
        super().reset(seed=seed)

        # Mensajes de depuración
        print(self.init_B)
        print(self.sigma)
        print(self.np_random.normal())

        # Inicializa ambas poblaciones
        self.B = self.init_B + self.sigma * self.np_random.normal()
        self.B = np.clip(self.B, 0.0, 1.0)

        # Reinicia el tiempo
        self.t = 0

        # Devuelve la observación inicial
        observation = np.array([
            self.normalize(self.B[0]),
            self.normalize(self.B[1])
        ], dtype=np.float32)

        return observation, {}

    def step(self, action):
        # Convierte la acción a una fracción de pesca
        action = float(action[0])

        harvest_fraction = self.unnormalize(action)

        # Biomasa extraída de la presa
        harvest = self.B[0] * harvest_fraction

        # Guarda la población anterior de la presa
        B_temp = np.copy(self.B[0])

        # Evolución de la presa
        self.B[0] = (
            self.B[0]
            + (
                self.r[0] * self.B[0] * (1 - self.B[0])
                - self.B[0] * self.B[1] * self.mort[0]
            ) * np.random.normal(1, 0.4)
            - harvest
        )

        # Evolución del depredador
        self.B[1] = (
            self.B[1]
            + self.r[1] * self.B[1]
            * (1 - self.B[1] / (B_temp * self.C + self.eps))
            * np.random.normal(1, 0.4)
        )

        # Limita ambas poblaciones
        self.B = np.clip(self.B, 0.0, 1.0)

        # Recompensa: pesca más incentivo por conservar el depredador
        reward = 0.1 * self.B[1] + harvest

        # Observación normalizada
        observation = np.array([
            self.normalize(self.B[0]),
            self.normalize(self.B[1])
        ], dtype=np.float32)

        # Avanza el tiempo
        self.t += 1

        terminated = self.t >= self.T
        truncated = False

        return observation, reward, terminated, truncated, {}

    def normalize(self, qty):
        # Convierte [0,1] → [-1,1]
        return -1 + 2 * qty

    def unnormalize(self, qty):
        # Convierte [-1,1] → [0,1]
        return (qty + 1) / 2


# ============================================================
# Entorno con dos presas y dos depredadores.
# El agente observa únicamente la primera presa.
# Orden:
# [Presa1, Presa2, Depredador1, Depredador2]
# ============================================================
class Pesca4D_1obs(gym.Env):

    def __init__(self, params):
        # Parámetros del modelo
        self.r = params["r_crecimiento"]
        self.sigma = params["sigma"]
        self.init_B = params["init_B"]
        self.T = params["T"]
        self.mort = params["mortalidad"]
        self.C = params["C"]
        self.eps = params["epsilon"]
        self.arrastre = params["arrastre"]
        self.alpha = 0.05

        # El agente sólo observa la primera población
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Acción: intensidad de pesca sobre la presa 1
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        # Reinicia el entorno
        super().reset(seed=seed)

        # Inicializa las poblaciones con ruido
        self.B = self.init_B + self.sigma * self.np_random.normal(1, 0.4)
        self.B = np.clip(self.B, 0.0, 1.0)

        # Reinicia el tiempo
        self.t = 0

        # Devuelve únicamente la observación de la primera presa
        observation = np.array(
            [self.normalize(self.B[0])],
            dtype=np.float32
        )

        return observation, {}

    def step(self, action):
        # Convierte la acción en una fracción de pesca
        action = float(action[0])
        harvest_fraction = self.unnormalize(action)

        # Biomasa extraída de la presa 1
        harvest = self.B[0] * harvest_fraction

        # Guarda el estado anterior
        B_init = np.copy(self.B)

        # Evolución de la presa 1
        self.B[0] = (
            B_init[0]
            + (
                self.r[0] * B_init[0]
                * (1 - B_init[0] / (1 - self.arrastre[0] * B_init[1]))
                - self.mort[0] * B_init[0] * B_init[3]
            )
            + np.random.normal(0, self.alpha)
            - harvest
        )

        # Evolución de la presa 2
        self.B[1] = (
            B_init[1]
            + (
                self.r[1] * B_init[1]
                * (1 - B_init[1] / (1 - self.arrastre[0] * B_init[0]))
                - self.mort[1] * B_init[1] * B_init[2]
            )
            + np.random.normal(0, self.alpha)
        )

        # Evolución del depredador 1
        self.B[2] = (
            B_init[2]
            + (
                self.r[2] * B_init[2] * B_init[1]
                * (1 - B_init[2] / (1 - self.arrastre[1] * B_init[3]))
            )
            + np.random.normal(0, self.alpha)
        )

        # Evolución del depredador 2
        self.B[3] = (
            B_init[3]
            + (
                self.r[3] * B_init[3] * B_init[0]
                * (1 - B_init[3] / (1 - self.arrastre[1] * B_init[2]))
            )
            + np.random.normal(0, self.alpha)
        )

        # Limita las poblaciones al intervalo permitido
        self.B = np.clip(self.B, 0.0, 1.0)

        # Recompensa
        reward = harvest + 0.4 * self.B[3]

        # El agente sólo observa la presa 1
        observation = np.array(
            [self.normalize(self.B[0])],
            dtype=np.float32
        )

        # Avanza el tiempo
        self.t += 1

        terminated = self.t >= self.T
        truncated = False

        return observation, reward, terminated, truncated, {}

    def normalize(self, qty):
        # Convierte [0,1] → [-1,1]
        return -1 + 2 * qty

    def unnormalize(self, qty):
        # Convierte [-1,1] → [0,1]
        return (qty + 1) / 2