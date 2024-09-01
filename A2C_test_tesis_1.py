# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:24:44 2024

@author: avasquez
"""

#python -m tensorboard.main --logdir=logs_a2c_1 --host localhost --port 8088

import gymnasium as gym
from stable_baselines3 import A2C
import os
import re

models_dir = "models_A2C/model_test_2"
logdir = "logs_a2c_2"

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
obs, info = env.reset(seed=0)

# Definir los hiperparámetros
hyperparameters = {
    'n_steps': 1200,  # Number of steps to run for each environment per update
    'learning_rate': 2e-4,  # Learning rate for the optimizer
    'gamma': 0.99,  # Discount factor
    'vf_coef': 0.25,  # Value function coefficient in the loss function
    'ent_coef': 0.01,  # Entropy coefficient in the loss function
    'max_grad_norm': 0.5,  # The maximum value for the gradient clipping
    'rms_prop_eps': 1e-5,  # RMSProp epsilon
    'use_rms_prop': True,  # Whether to use RMSProp or Adam as optimizer
    'normalize_advantage': False  # Whether to normalize advantage
}

# Cargar el modelo si existe, de lo contrario crear uno nuevo
if latest_model:
    model = A2C.load(model_path, env=env, tensorboard_log=logdir, **hyperparameters)
else:
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir, **hyperparameters)

#TIMESTEPS = x15
TIMESTEPS = 33600
i = 0

# Continuar el entrenamiento
for i in range(4):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C_test_1")
    model.save(f"{models_dir}/{TIMESTEPS*(i+1)}")

#################### cambio modelo ##############################

hyperparameters = {
    'n_steps': 30,  # Number of steps to run for each environment per update
    'learning_rate': 1.5e-4,  # Learning rate for the optimizer
    'gamma': 0.99,  # Discount factor
    'vf_coef': 0.25,  # Value function coefficient in the loss function
    'ent_coef': 0.015,  # Entropy coefficient in the loss function
    'max_grad_norm': 0.5,  # The maximum value for the gradient clipping
    'rms_prop_eps': 1e-5,  # RMSProp epsilon
    'use_rms_prop': True,  # Whether to use RMSProp or Adam as optimizer
    'normalize_advantage': False  # Whether to normalize advantage
}

#TIMESTEPS = x15
TIMESTEPS = 33600
i = 0

# Continuar el entrenamiento
for i in range(8):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C_test_2")
    model.save(f"{models_dir}/{TIMESTEPS*(i+1)}_2")
