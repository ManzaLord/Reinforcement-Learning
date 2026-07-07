import gymnasium as gym
import numpy as np

# Entorno con dos presas y dos depredadores.
# Orden de las poblaciones:
# [Presa1, Presa2, Depredador1, Depredador2]
class Pesca4D_4obs(gym.Env):

    def __init__(self, params):
        # Parámetros del modelo
        self.r = params["r_crecimiento"]      # Tasas de crecimiento
        self.sigma = params["sigma"]          # Intensidad del ruido inicial
        self.init_B = params["init_B"]        # Poblaciones iniciales
        self.T = params["T"]                  # Eje temporal
        self.mort = params["mortalidad"]      # Tasas de mortalidad
        self.C = params["C"]                  # Capacidad máxima del depredador
        self.eps = params["epsilon"]          # Constante pequeña
        self.arrastre = params["arrastre"]    # Coeficientes de interacción
        self.alpha = 0.05                     # Ruido dinámico agregado en cada paso

        # Espacio de observaciones: las cuatro poblaciones normalizadas en [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Espacio de acciones: fracción de pesca sobre la presa 1
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        # Reinicia el entorno
        super().reset(seed=seed)

        # Inicializa las poblaciones agregando ruido aleatorio
        self.B = self.init_B + self.sigma * self.np_random.normal(1, 0.4)

        # Limita las poblaciones al intervalo [0, 1]
        self.B = np.clip(self.B, 0.0, 1.0)

        # Reinicia el contador temporal
        self.t = 0

        # Devuelve la observación inicial normalizada
        observation = self.normalize(self.B).astype(np.float32)

        return observation, {}

    def step(self, action):
        # Convierte la acción del agente a una fracción de pesca
        action = float(action[0])
        harvest_fraction = self.unnormalize(action)

        # Cantidad extraída de la presa 1
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

        # Restringe las poblaciones al intervalo permitido
        self.B = np.clip(self.B, 0.0, 1.0)

        # Calcula la recompensa:
        # beneficio por pesca más un incentivo para conservar el depredador 2
        reward = harvest + 0.4 * self.B[3]

        # Genera la observación normalizada
        observation = self.normalize(self.B).astype(np.float32)

        # Avanza un paso temporal
        self.t += 1

        # Finaliza el episodio al alcanzar el horizonte temporal
        terminated = self.t >= self.T
        truncated = False

        return observation, reward, terminated, truncated, {}

    def normalize(self, qty):
        # Convierte valores de [0, 1] al intervalo [-1, 1]
        return -1 + 2 * qty

    def unnormalize(self, qty):
        # Convierte valores de [-1, 1] al intervalo [0, 1]
        return (qty + 1) / 2