#!/usr/bin/env python3
"""
紧急修复配置 - V4
解决V3中Value权重过度主导的问题
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
        "total_timesteps": 100000,
        "save_dir": f"./stone_with_direction_v4_{timestamp}",
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "target_obj_id": 7,  # Stone object ID
        "target_obj_name": "stone",
        "direction_weight": 0.8  # 大幅增加方向预测权重
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

    # V4紧急修复配置 - 解决Value权重过度主导
    model = CustomPPO(
        CustomACPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=2e-3,   # 大幅提高学习率，强制策略更新
        n_steps=4096,
        batch_size=512,
        n_epochs=3,           # 适度减少，防止Value过拟合
        gamma=0.95,
        gae_lambda=0.65,
        clip_range=0.2,
        ent_coef=0.2,         # 大幅增加entropy，强制探索
        vf_coef=0.05,         # 大幅降低value权重，防止主导
        max_grad_norm=0.5,
        verbose=1,
        normalize_advantage=False,
        direction_weight=config["direction_weight"],  # 大幅增加方向权重
        loss_normalization=True,  # 启用带权重上限保护的损失归一化
        norm_decay=0.7        # 更快适应变化
    )

    total_timesteps = config["total_timesteps"]

    print(f"🚨 启动紧急修复配置 V4")
    print(f"Target object: {config['target_obj_name']} (ID: {config['target_obj_id']})")
    print(f"Direction weight: {config['direction_weight']} (大幅增加)")
    print(f"Entropy coefficient: {model.ent_coef} (强制探索)")
    print(f"Value coefficient: {model.vf_coef} (大幅降低)")
    print(f"Learning rate: {model.learning_rate} (强化更新)")
    print(f"N epochs: {model.n_epochs} (防止过拟合)")
    print(f"Normalization decay: {model.norm_decay} (快速适应)")
    print(f"Total timesteps: {total_timesteps}")
    print("=" * 70)
    
    print("🚨 V4紧急修复措施:")
    print("  - 代码修复: 添加权重上限保护 (max 40%)")
    print("  - 代码修复: 增强最小权重保护 (15%)")
    print("  - 代码修复: 权重重新归一化机制")
    print("  - 学习率: 1e-3 → 2e-3 (强制策略更新)")
    print("  - Entropy系数: 0.12 → 0.2 (强制探索)")
    print("  - Value系数: 0.3 → 0.05 (大幅降低)")
    print("  - Direction权重: 0.6 → 0.8 (大幅增加)")
    print("  - Norm decay: 0.8 → 0.7 (更快适应)")
    print("=" * 70)
    
    print("🎯 V4预期修复效果:")
    print("  - adaptive_weights/value < 40% (从28%受限)")
    print("  - adaptive_weights/policy > 15% (从1.14%)")
    print("  - adaptive_weights/entropy > 5% (从0.22%)")
    print("  - adaptive_weights/direction > 10% (从2.18%)")
    print("  - approx_kl > 1e-6 (策略开始更新)")
    print("  - entropy_loss < -0.01 (真实探索)")
    print("  - direction_accuracy > 35% (从29.4%)")
    print("=" * 70)

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(config["save_dir"])
    print(f"Model saved to {config['save_dir']}")

    env.close()

# V4版本紧急修复说明
"""
🚨 V4主要解决的问题:

1. Value权重过度主导 (V3中达到28%):
   - 添加权重上限保护 (max 40%)
   - 大幅降低vf_coef: 0.3 → 0.05
   - 权重重新归一化机制

2. 探索机制完全失效:
   - 大幅增加ent_coef: 0.12 → 0.2
   - 增强最小权重保护: 0.1 → 0.15

3. 策略更新停滞:
   - 大幅提高learning_rate: 1e-3 → 2e-3
   - 适度减少n_epochs: 4 → 3

4. 方向学习不足:
   - 进一步增加direction_weight: 0.6 → 0.8

🎯 V3 → V4 关键改进:
- 解决Value Function Overfitting
- 恢复Policy和Entropy的主导地位
- 平衡四个损失的权重分布
- 重新激活策略更新机制

📊 如果V4仍有问题:
- 考虑进一步降低vf_coef到0.01
- 增加ent_coef到0.3
- 检查环境奖励设计
- 考虑重新初始化模型
"""