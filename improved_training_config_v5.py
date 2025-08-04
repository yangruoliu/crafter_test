#!/usr/bin/env python3
"""
V5优化配置 - 针对奖励优化和效率提升
基于测试结果：完成率100%但奖励为负，需要提升策略效率
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
        "total_timesteps": 1000000,  # 增加训练步数
        "save_dir": f"./stone_with_direction_v5_{timestamp}",
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "target_obj_id": 7,  # Stone object ID
        "target_obj_name": "stone",
        "direction_weight": 1.0  # 最大化方向预测权重
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

    # V5效率优化配置 - 专注于策略效率提升
    model = CustomPPO(
        CustomACPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-3,   # 大幅提高学习率，加速策略改进
        n_steps=4096,
        batch_size=512,
        n_epochs=2,           # 减少epochs，避免过拟合
        gamma=0.98,           # 提高gamma，更重视长期回报
        gae_lambda=0.7,       # 调整GAE，平衡bias-variance
        clip_range=0.15,      # 更保守的裁剪，稳定训练
        ent_coef=0.15,        # 适度降低探索，提高利用效率
        vf_coef=0.03,         # 进一步降低value权重
        max_grad_norm=0.3,    # 降低梯度裁剪，允许更大更新
        verbose=1,
        normalize_advantage=False,
        direction_weight=config["direction_weight"],  # 最大化方向学习
        loss_normalization=True,  # 启用权重上限保护的损失归一化
        norm_decay=0.6        # 更快的适应速度
    )

    total_timesteps = config["total_timesteps"]

    print(f"🎯 启动V5效率优化配置")
    print(f"Target object: {config['target_obj_name']} (ID: {config['target_obj_id']})")
    print(f"Direction weight: {config['direction_weight']} (最大化)")
    print(f"Entropy coefficient: {model.ent_coef} (平衡探索)")
    print(f"Value coefficient: {model.vf_coef} (最小化)")
    print(f"Learning rate: {model.learning_rate} (高速学习)")
    print(f"Gamma: {model.gamma} (重视长期)")
    print(f"Clip range: {model.clip_range} (保守更新)")
    print(f"Normalization decay: {model.norm_decay} (快速适应)")
    print(f"Total timesteps: {total_timesteps}")
    print("=" * 70)
    
    print("🎯 V5核心优化策略:")
    print("  - 基于测试结果：完成率100%，但奖励-6.24")
    print("  - 目标：保持完成率，大幅提升策略效率")
    print("  - 学习率: 2e-3 → 3e-3 (更激进的策略更新)")
    print("  - Gamma: 0.95 → 0.98 (更重视长期效益)")
    print("  - Entropy: 0.2 → 0.15 (减少无效探索)")
    print("  - Value权重: 0.05 → 0.03 (进一步削弱)")
    print("  - Direction权重: 0.8 → 1.0 (完全最大化)")
    print("  - Clip range: 0.2 → 0.15 (更稳定的策略更新)")
    print("  - 训练步数: 2M → 3M (充分训练)")
    print("=" * 70)
    
    print("📈 V5预期改善目标:")
    print("  - 短期(1000步): ep_rew_mean > -4.0")
    print("  - 中期(3000步): ep_rew_mean > -1.0")
    print("  - 长期(5000步): ep_rew_mean > 2.0")
    print("  - 保持: 完成率 = 100%")
    print("  - 改善: 平均步数 < 150")
    print("  - 目标: 模型评级 ≥ B+")
    print("=" * 70)
    
    print("⚠️ V5关键监控指标:")
    print("  - adaptive_weights/value < 25% (防止重新主导)")
    print("  - adaptive_weights/policy > 20% (确保策略学习)")
    print("  - direction_accuracy > 45% (强化方向学习)")
    print("  - entropy_loss < -0.005 (保持适度探索)")
    print("  - approx_kl > 1e-5 (确保策略更新)")
    print("=" * 70)

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(config["save_dir"])
    print(f"Model saved to {config['save_dir']}")

    env.close()

# V5版本核心改进说明
"""
🎯 V5针对性优化分析:

基于测试结果问题诊断:
✅ 已解决: 模型能完成任务 (100%完成率)
✅ 已解决: 训练稳定性良好 (低标准差)
❌ 核心问题: 策略效率低下 (奖励-6.24)

V5核心改进策略:

1. **提升策略学习效率**:
   - learning_rate: 2e-3 → 3e-3 (更快策略改进)
   - gamma: 0.95 → 0.98 (更重视长期回报优化)
   - max_grad_norm: 0.5 → 0.3 (允许更大的策略更新)

2. **优化探索-利用平衡**:
   - ent_coef: 0.2 → 0.15 (减少无效探索)
   - clip_range: 0.2 → 0.15 (更稳定的策略更新)
   - n_epochs: 3 → 2 (避免过拟合)

3. **强化方向学习**:
   - direction_weight: 0.8 → 1.0 (最大化方向预测)
   - 期望提升空间感知和导航效率

4. **进一步削弱Value主导**:
   - vf_coef: 0.05 → 0.03 (最小化value影响)
   - 确保策略学习不被价值估计干扰

5. **加速权重适应**:
   - norm_decay: 0.7 → 0.6 (更快的损失归一化适应)

📊 预期改善路径:
V4测试结果 → V5目标
- 奖励: -6.24 → 2.0+
- 步数: 188 → 150-
- 完成率: 100% → 100% (保持)
- 评级: 🔴 → 🟡/🟢

🔄 如果V5仍未达到预期:
考虑环境级别的优化:
- 调整奖励函数 (增加效率奖励)
- 检查动作空间设计
- 考虑课程学习策略
"""