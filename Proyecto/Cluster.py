from pesca_env import Pesca4D
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from huggingface_hub import HfApi, login

"""
Orden de los datos
[Presa1, Presa2, Depredador1, Depredador2]
"""

#Factor de crecimiento de cada poblacion
r = np.array([0.9, 0.7, 0.3, 0.2])

#Factor de ruido agregado al sistema
sigma = 0.05

#Poblacion inicial de cada especie
init_B = np.array([0.4, 0.7, 0.1, 0.2])

#Factor terminated
T = 200

#Factor de mortalidad de cada especie
mort = [0.1, 0.2, 0.3, 0.02]

#Factor de arrastre para presas y cazadores
arrastre = [0.1, 0.2]

#Factor que limita el valor maximo de la poblacion depredadora
C = 1

observables = np.array([0])

#Diccionario con los parametros del env
params = { "r_crecimiento": r,
           "sigma": sigma,
           "init_B": init_B,
           "T": T,
           "mortalidad": mort,
           "C": C,
           "epsilon": 1e-5,
           "arrastre": arrastre,
           "observables": observables
}

def init_agent(env, model, training=False, name="pesca_4D"):
    if training == True:
        #Inicializa el modelo
        if model == "RecurrentPPO":
            agent = RecurrentPPO("MlpLstmPolicy", env, verbose=1, device="cuda")
        
            #Entrena el modelo
            agent.learn(total_timesteps=5000)
        
            #Guarda el modelo
            agent.save("RecurrentPPO_" + name)

        elif model == "PPO":
            #Inicializa el modelo
            agent = PPO("MlpPolicy", env, verbose=1, device="cuda")
            
            #Entrena el modelo
            agent.learn(total_timesteps=25000)
            
            #Guarda el modelo
            agent.save("PPO_" + name)  

    else:
        if model == "RecurrentPPO":
            #Carga el modelo
            agent = RecurrentPPO.load("RecurrentPPO_" + name, device="cuda")
            
        elif model == "PPO":
            #Carga el modelo
            agent = PPO.load("PPO_" + name, device="cuda")
        
    return agent

def subir_a_hub(local_file, repo_id, commit_message="Subiendo modelo PPO"):
    api = HfApi()
    
    # Crear repo si no existe
    api.create_repo(repo_id=repo_id, exist_ok=True)
    
    # Subir archivo
    api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=local_file,
        repo_id=repo_id,
        commit_message=commit_message
    )
    print(f"Modelo subido exitosamente a {repo_id}")

#Inicializa el ambiente
env = Pesca4D(params)

#Inicializa el agente
agentPPO = init_agent(env,"PPO", training=False)
agentRecPPO = init_agent(env, "RecurrentPPO", training=False)
subir_a_hub("RecurrentPPO_pesca_4D", "Esporrasm/PPO_y_RecPPOrepo_id")
