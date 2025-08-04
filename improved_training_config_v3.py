#!/usr/bin/env python3
"""
最终修复配置 - V3
基于V2的结果分析，解决剩余的策略更新停滞问题
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
        "total_timesteps": 2000000,
        "save_dir": f"./stone_with_direction_final_{timestamp}",
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "target_obj_id": 7,  # Stone object ID
        "target_obj_name": "stone",
        "direction_weight": 0.6  # 进一步增加方向预测权重
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

    # 最终修复配置 - 解决策略更新停滞问题
    model = CustomPPO(
        CustomACPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,   # 再次提高学习率，强化更新
        n_steps=4096,
        batch_size=512,
        n_epochs=4,           # 增加训练轮数，确保充分更新
        gamma=0.95,
        gae_lambda=0.65,
        clip_range=0.2,
        ent_coef=0.12,        # 进一步增加entropy，强制探索
        vf_coef=0.3,          # 适度提升value权重
        max_grad_norm=0.5,
        verbose=1,
        normalize_advantage=False,
        direction_weight=config["direction_weight"],  # 大幅增加方向权重
        loss_normalization=True,  # 启用修复后的损失归一化
        norm_decay=0.8        # 进一步降低，更快适应变化
    )

    total_timesteps = config["total_timesteps"]

    print(f"🚀 启动最终修复配置 V3")
    print(f"Target object: {config['target_obj_name']} (ID: {config['target_obj_id']})")
    print(f"Direction weight: {config['direction_weight']} (大幅增加)")
    print(f"Entropy coefficient: {model.ent_coef} (强制探索)")
    print(f"Value coefficient: {model.vf_coef} (平衡价值学习)")
    print(f"Learning rate: {model.learning_rate} (提高更新强度)")
    print(f"N epochs: {model.n_epochs} (增加训练充分性)")
    print(f"Normalization decay: {model.norm_decay} (更快适应)")
    print(f"Total timesteps: {total_timesteps}")
    print("=" * 70)
    
    print("🎯 V3版本主要改进:")
    print("  - 学习率: 5e-4 → 1e-3 (解决策略更新停滞)")
    print("  - Entropy系数: 0.08 → 0.12 (强制探索)")
    print("  - Direction权重: 0.4 → 0.6 (强化方向学习)")
    print("  - 训练轮数: 2 → 4 (确保充分更新)")
    print("  - Norm decay: 0.85 → 0.8 (更快适应)")
    print("=" * 70)
    
    print("📈 预期V3改善:")
    print("  - approx_kl > 1e-6 (策略开始更新)")
    print("  - adaptive_weights/direction > 2% (方向学习加强)")  
    print("  - adaptive_weights/policy > 10% (策略权重恢复)")
    print("  - entropy_loss < -0.01 (真实探索)")
    print("  - ep_rew_mean 开始上升趋势")
    print("  - direction_accuracy > 45%")
    print("=" * 70)

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(config["save_dir"])
    print(f"Model saved to {config['save_dir']}")

    env.close()

# V3版本更新说明
"""
🔧 V3主要解决的问题:

1. 策略更新停滞:
   - learning_rate: 5e-4 → 1e-3 (更强梯度更新)
   - n_epochs: 2 → 4 (更充分的训练)
   
2. 探索机制异常:
   - ent_coef: 0.08 → 0.12 (强制产生探索)
   
3. 权重分布优化:
   - direction_weight: 0.4 → 0.6 (进一步加强)
   - vf_coef: 0.2 → 0.3 (平衡价值学习)
   
4. 适应性提升:
   - norm_decay: 0.85 → 0.8 (更快响应变化)

🎯 V2 → V3 预期改善:
- 解决 approx_kl = 0 的问题
- 提升 policy/value 权重比例
- 真正激活探索机制
- 加速任务学习进程

📊 如果V3仍有问题，考虑:
- 检查环境奖励设计
- 进一步增加学习率到2e-3
- 或者尝试不同的优化器设置
"""