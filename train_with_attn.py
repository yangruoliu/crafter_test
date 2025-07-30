import gym
import crafter
import env_wrapper
from model.model_with_attn import CustomResNet, CustomACPolicy, CustomPPO, TQDMProgressBar
import torch.nn as nn
import torch
from stable_baselines3 import PPO
from gym.wrappers import FrameStack
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from my_label_oracle import get_label


if __name__ == "__main__":

    config = {
        "total_timesteps": 1,
        "save_dir": "./wood",
        "init_items": ["wood", "stone"],
        "init_num": [1, 1]
    }

    env = gym.make("MyCrafter-v0") 

    env = env_wrapper.LabelGeneratingWrapper(env, get_label_func=get_label, target_obj="tree", num_aux_classes=2)

    env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=config["init_num"])

    env = env_wrapper.WoodPickaxeWrapper(env)

    policy_kwargs = {
        "features_extractor_class": CustomResNet,
        "features_extractor_kwargs": {"features_dim": 1024},
        "activation_fn": nn.ReLU,
        "net_arch": [],
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"eps": 1e-5},
        "num_aux_classes": 2
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
        aux_weight=0.2
    )

    total_timesteps = config["total_timesteps"]

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(config["save_dir"])

    env.close()
