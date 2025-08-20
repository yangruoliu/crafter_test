#!/usr/bin/env python3
"""
修复训练问题的改进配置 - V2
基于当前训练日志的问题分析
"""

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
        "total_timesteps": 1000000,
        "save_dir": f"./stone_with_direction_fixed_{timestamp}",
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "target_obj_id": 7,  # Stone object ID
        "target_obj_name": "stone",
        "direction_weight": 0.4  # 显著增加方向预测权重
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

    # 修复后的训练配置 - 解决权重失衡问题
    model = CustomPPO(
        CustomACPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-4,  # 增加学习率，加快收敛
        n_steps=4096,
        batch_size=512,
        n_epochs=2,          # 减少epochs，防止过拟合
        gamma=0.95,
        gae_lambda=0.65,
        clip_range=0.2,
        ent_coef=0.08,       # 大幅增加entropy权重，改善探索
        vf_coef=0.2,         # 适度减少value权重，防止主导
        max_grad_norm=0.5,
        verbose=1,
        normalize_advantage=False,
        direction_weight=config["direction_weight"],  # 增加方向预测权重
        loss_normalization=True,  # 启用修复后的损失归一化
        norm_decay=0.85      # 降低衰减因子，更快适应变化
    )

    total_timesteps = config["total_timesteps"]

    print(f"🔧 启动修复后的训练配置")
    print(f"Target object: {config['target_obj_name']} (ID: {config['target_obj_id']})")
    print(f"Direction weight: {config['direction_weight']} (大幅增加)")
    print(f"Entropy coefficient: {model.ent_coef} (增强探索)")
    print(f"Value coefficient: {model.vf_coef} (适度减少)")
    print(f"Learning rate: {model.learning_rate} (提高学习速度)")
    print(f"Normalization decay: {model.norm_decay} (更快适应)")
    print(f"N epochs: {model.n_epochs} (减少过拟合)")
    print(f"Total timesteps: {total_timesteps}")
    print("=" * 70)
    
    print("🎯 预期改善:")
    print("  - adaptive_weights/entropy > 0.02 (从0.007)")
    print("  - adaptive_weights/direction > 0.05 (从0.0002)")  
    print("  - ep_rew_mean 逐步上升 (从-5.94)")
    print("  - direction_accuracy > 40% (从33.8%)")
    print("  - 更稳定的训练过程")
    print("=" * 70)

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(config["save_dir"])
    print(f"Model saved to {config['save_dir']}")

    env.close()

# 详细的修复说明
"""
🔧 主要修复措施:

1. 最小权重保护: 0.01 → 0.1 (10倍增加)
   - 确保entropy和direction真正参与训练
   
2. 基础权重调整:
   - ent_coef: 0.02 → 0.08 (4倍增加，强化探索)
   - direction_weight: 0.2 → 0.4 (2倍增加)
   - vf_coef: 0.3 → 0.2 (适度减少)
   
3. 训练参数优化:
   - learning_rate: 3e-4 → 5e-4 (提高收敛速度)
   - n_epochs: 3 → 2 (减少过拟合风险)
   - norm_decay: 0.95 → 0.85 (更快适应)
   
4. 预期效果:
   - 解决exploration不足问题
   - 恢复direction learning
   - 提升整体任务表现
   - 更稳定的训练过程

🚨 注意: 建议重新开始训练，因为当前模型可能已陷入局部最优
"""