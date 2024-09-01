# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:24:44 2024

@author: avasquez
"""

#python -m tensorboard.main --logdir=logs_ppo_4 --host localhost --port 8088

import gymnasium as gym
from stable_baselines3 import PPO
import os
import re
import numpy as np
import torch

# Establecer la semilla para reproducibilidad
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

models_dir = "models_PPO/model_test_5"
logdir = "logs_ppo_4"

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

# Configurar el entorno
env = gym.make("CartPole-30", render_mode="human", max_episode_steps=300)
obs, info = env.reset(seed=SEED)

# Definir los hiperparámetros
hyperparameters = {
    'n_steps': 2400,  # Horizon (T)
    'learning_rate': 3.75e-4,  # Adam stepsize
    'clip_range': 0.25,  # Clipping parameter (ε)
    'ent_coef': 0.001,  # Entropy coefficient (c2)
    'vf_coef': 0.5,  # Value function coefficient (c1)
    'n_epochs': 35,  # Number of epochs (K)
    'batch_size': 150,  # Minibatch size (M)
    'gamma': 0.4  # Discount factor
}

# Cargar el modelo si existe, de lo contrario crear uno nuevo
if latest_model:
    model = PPO.load(model_path, env=env, tensorboard_log=logdir)
else:
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, seed=SEED, **hyperparameters)

# TIMESTEPS = x15
TIMESTEPS = 48000

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_test_1")
model.save(f"{models_dir}/{TIMESTEPS*1}")

