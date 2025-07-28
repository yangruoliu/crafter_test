import gym
import crafter
import env_wrapper
from model_with_attn import CustomResNet, CustomACPolicy, CustomPPO, TQDMProgressBar
import torch.nn as nn
import torch
from stable_baselines3 import PPO
from gym.wrappers import FrameStack
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
import time
timestamp = time.strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":

    config = {
        "total_timesteps": 2000000,
        "save_dir": f"./stone_with_direction_{timestamp}",
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "target_obj_id": 7,  # Stone object ID
        "target_obj_name": "stone",
        "direction_weight": 0.3  # Weight for direction prediction loss
    }

    env = gym.make("MyCrafter-v0") 

    # Apply wrappers in the correct order
    env = env_wrapper.MineStoneWrapper(env)
    env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=config["init_num"])
    
    # Add direction label wrapper - this should be the last wrapper to ensure
    # it gets the correct info from the environment
    env = env_wrapper.DirectionLabelWrapper(
        env, 
        target_obj_id=config["target_obj_id"], 
        target_obj_name=config["target_obj_name"]
    )

    policy_kwargs = {
        "features_extractor_class": CustomResNet,
        "features_extractor_kwargs": {"features_dim": 1024},
        "activation_fn": nn.ReLU,
        "net_arch": [],
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"eps": 1e-5},
        "num_aux_classes": 9  # 9 direction classes
    }

    model = CustomPPO(
        CustomACPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=3,
        gamma=0.95,
        gae_lambda=0.65,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        normalize_advantage=False,
        direction_weight=config["direction_weight"],  # Weight for direction prediction auxiliary loss
        loss_normalization=True,  # Enable loss normalization
        norm_decay=0.99  # Decay factor for moving averages
    )

    total_timesteps = config["total_timesteps"]

    print(f"Starting training with direction prediction auxiliary task...")
    print(f"Target object: {config['target_obj_name']} (ID: {config['target_obj_id']})")
    print(f"Direction weight: {config['direction_weight']}")
    print(f"Total timesteps: {total_timesteps}")

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(config["save_dir"])
    print(f"Model saved to {config['save_dir']}")

    env.close() 