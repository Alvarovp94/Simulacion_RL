# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:24:44 2024

@author: avasquez
"""

#python -m tensorboard.main --logdir=logs_sac_1 --host localhost --port 8088


import gymnasium as gym
from stable_baselines3 import SAC
import os
import re
import numpy as np
import torch

models_dir = "models_SAC/model_test_1"
logdir = "logs_sac_1"

# Crear directorios si no existen
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Buscar el último modelo guardado
model_files = [f for f in os.listdir(models_dir) if re.match(r'\d+', f)]
if model_files:
    latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
    print(f"Cargando el modelo desde {latest_model}")
    model_path = os.path.join(models_dir, latest_model)
else:
    latest_model = None

# Configurar la semilla para reproducibilidad
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

env = gym.make("CartPole-30", render_mode="human", max_episode_steps=300)
obs, info = env.reset(seed=SEED)


# Definir los hiperparámetros
hyperparameters = {
    'learning_rate': 5e-4,  # Incrementar ligeramente la tasa de aprendizaje
    'buffer_size': 2000000,  # Mantener el tamaño del buffer
    'learning_starts': 2000,  # Mantener el inicio del aprendizaje
    'batch_size': 256,  # Incrementar el tamaño de lote
    'tau': 0.005,  # Mantener tau
    'gamma': 0.99,
    'train_freq': 8,
    'gradient_steps': 8,
    'ent_coef': 0.01,  # Mantener el coeficiente de entropía fijo
}

# Cargar el modelo si existe, de lo contrario crear uno nuevo
if latest_model:
    model = SAC.load(model_path, env=env, tensorboard_log=logdir, seed=SEED, **hyperparameters)
else:
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir, seed=SEED, **hyperparameters)

# TIMESTEPS = x15
TIMESTEPS = 33600
i = 0

# Continuar el entrenamiento
for i in range(4):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC_test_5")
    model.save(f"{models_dir}/{TIMESTEPS*(i+1)}_2")


