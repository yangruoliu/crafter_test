#!/usr/bin/env python3
"""
四大损失函数计算演示
"""

import numpy as np
import torch
import torch.nn.functional as F

def demo_policy_loss():
    """演示Policy Loss计算"""
    print("🎯 Policy Loss 计算演示")
    print("-" * 50)
    
    # 模拟数据
    batch_size = 4
    
    # 优势函数 (正数=好动作，负数=坏动作)
    advantages = torch.tensor([2.5, -1.2, 0.8, -0.5])
    
    # 旧策略和新策略的对数概率
    old_log_probs = torch.tensor([-1.6, -0.9, -2.3, -1.1])  # log(概率)
    new_log_probs = torch.tensor([-1.2, -1.5, -1.8, -1.3])
    
    # 计算概率比值
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    print("输入数据:")
    print(f"  优势函数: {advantages.numpy()}")
    print(f"  旧策略log概率: {old_log_probs.numpy()}")
    print(f"  新策略log概率: {new_log_probs.numpy()}")
    print(f"  概率比值: {ratio.numpy()}")
    
    # PPO裁剪
    clip_range = 0.2
    ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    
    # 两种损失计算
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * ratio_clipped
    
    # 取较小值（更保守）
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    
    print(f"\nPPO计算:")
    print(f"  裁剪范围: [{1-clip_range:.1f}, {1+clip_range:.1f}]")
    print(f"  裁剪后比值: {ratio_clipped.numpy()}")
    print(f"  损失1 (原始): {policy_loss_1.numpy()}")
    print(f"  损失2 (裁剪): {policy_loss_2.numpy()}")
    print(f"  最终Policy Loss: {policy_loss.item():.4f}")

def demo_value_loss():
    """演示Value Loss计算"""
    print("\n📊 Value Loss 计算演示")
    print("-" * 50)
    
    # 模拟状态价值预测和实际回报
    predicted_values = torch.tensor([50.2, 78.5, 92.1, 15.3])
    actual_returns = torch.tensor([52.0, 85.2, 88.7, 18.1])
    
    print("输入数据:")
    print(f"  预测价值: {predicted_values.numpy()}")
    print(f"  实际回报: {actual_returns.numpy()}")
    
    # 计算MSE损失
    value_loss = F.mse_loss(predicted_values, actual_returns)
    
    # 详细计算过程
    differences = predicted_values - actual_returns
    squared_errors = differences ** 2
    
    print(f"\n计算过程:")
    print(f"  预测误差: {differences.numpy()}")
    print(f"  平方误差: {squared_errors.numpy()}")
    print(f"  平均平方误差: {squared_errors.mean().item():.4f}")
    print(f"  最终Value Loss: {value_loss.item():.4f}")

def demo_entropy_loss():
    """演示Entropy Loss计算"""
    print("\n🎲 Entropy Loss 计算演示")
    print("-" * 50)
    
    # 模拟不同探索程度的策略
    print("比较两种策略的熵:")
    
    # 高熵策略 (探索性强)
    high_entropy_probs = torch.tensor([[0.25, 0.25, 0.25, 0.25],  # 均匀分布
                                      [0.3, 0.2, 0.3, 0.2]])
    
    # 低熵策略 (确定性强)  
    low_entropy_probs = torch.tensor([[0.9, 0.05, 0.03, 0.02],   # 偏向一个动作
                                     [0.85, 0.1, 0.03, 0.02]])
    
    def calculate_entropy(probs):
        # 熵 = -sum(p * log(p))
        log_probs = torch.log(probs + 1e-8)  # 避免log(0)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy
    
    high_entropy = calculate_entropy(high_entropy_probs)
    low_entropy = calculate_entropy(low_entropy_probs)
    
    print(f"高熵策略概率: {high_entropy_probs.numpy()}")
    print(f"高熵策略熵值: {high_entropy.numpy()}")
    print(f"平均熵: {high_entropy.mean().item():.4f}")
    
    print(f"\n低熵策略概率: {low_entropy_probs.numpy()}")
    print(f"低熵策略熵值: {low_entropy.numpy()}")
    print(f"平均熵: {low_entropy.mean().item():.4f}")
    
    # 熵损失 (负熵，因为要最大化熵)
    high_entropy_loss = -high_entropy.mean()
    low_entropy_loss = -low_entropy.mean()
    
    print(f"\n熵损失:")
    print(f"  高熵策略损失: {high_entropy_loss.item():.4f} (训练早期期望)")
    print(f"  低熵策略损失: {low_entropy_loss.item():.4f} (训练后期)")

def demo_direction_loss():
    """演示Direction Loss计算"""
    print("\n🧭 Direction Loss 计算演示")
    print("-" * 50)
    
    # 9个方向类别
    direction_names = [
        "上左", "上", "上右",
        "左", "中心", "右", 
        "下左", "下", "下右"
    ]
    
    # 模拟预测和真实标签
    batch_size = 3
    num_classes = 9
    
    # 真实方向标签
    true_directions = torch.tensor([2, 5, 7])  # 上右, 右, 下
    
    # 模拟网络输出的logits
    predicted_logits = torch.tensor([
        [-0.5, 0.2, 2.1, -1.0, 0.1, 0.3, -0.8, 0.0, -0.2],  # 预测上右(正确)
        [0.1, -0.3, 0.2, -0.1, 0.5, 1.8, 0.0, -0.5, 0.1],   # 预测右(正确)
        [0.3, 0.8, -0.2, 0.1, -0.1, 0.0, 0.2, -0.1, 0.4]    # 预测上(错误,应该是下)
    ])
    
    print("输入数据:")
    print(f"  真实方向: {[direction_names[i] for i in true_directions]}")
    print(f"  预测logits形状: {predicted_logits.shape}")
    
    # 转换为概率
    predicted_probs = F.softmax(predicted_logits, dim=1)
    
    # 计算交叉熵损失
    direction_loss = F.cross_entropy(predicted_logits, true_directions)
    
    print(f"\n详细分析:")
    for i in range(batch_size):
        true_idx = true_directions[i].item()
        pred_probs = predicted_probs[i]
        pred_idx = torch.argmax(pred_probs).item()
        
        print(f"  样本 {i+1}:")
        print(f"    真实方向: {direction_names[true_idx]}")
        print(f"    预测方向: {direction_names[pred_idx]}")
        print(f"    预测概率: {pred_probs[true_idx].item():.4f}")
        print(f"    单样本损失: {-torch.log(pred_probs[true_idx]).item():.4f}")
    
    print(f"\n最终Direction Loss: {direction_loss.item():.4f}")

def demo_combined_loss():
    """演示四个损失的组合"""
    print("\n🔗 组合损失演示")
    print("-" * 50)
    
    # 模拟四个损失值
    policy_loss = 0.85
    value_loss = 15.2
    entropy_loss = -1.8
    direction_loss = 1.2
    
    print("原始损失值 (不同量级):")
    print(f"  Policy Loss:    {policy_loss:.3f}")
    print(f"  Value Loss:     {value_loss:.3f}  ← 主导训练!")
    print(f"  Entropy Loss:   {entropy_loss:.3f}")
    print(f"  Direction Loss: {direction_loss:.3f}")
    
    # 原始权重系统
    original_weights = [1.0, 0.3, 0.02, 0.2]
    original_total = (policy_loss * original_weights[0] + 
                     value_loss * original_weights[1] + 
                     entropy_loss * original_weights[2] + 
                     direction_loss * original_weights[3])
    
    print(f"\n原始加权 (有问题):")
    print(f"  总损失: {original_total:.3f}")
    print(f"  Value部分贡献: {value_loss * original_weights[1]:.3f} (占主导)")
    
    # 归一化系统
    moving_averages = [1.0, 48.5, -2.47, 1.58]
    normalized_losses = [
        policy_loss / abs(moving_averages[0]),
        value_loss / abs(moving_averages[1]), 
        entropy_loss / abs(moving_averages[2]),
        direction_loss / abs(moving_averages[3])
    ]
    
    adaptive_weights = [0.25, 0.35, 0.02, 0.15]  # 基于变异系数
    
    normalized_total = sum(w * l for w, l in zip(adaptive_weights, normalized_losses))
    
    print(f"\n归一化系统 (修复后):")
    print(f"  归一化损失: {[f'{l:.3f}' for l in normalized_losses]}")
    print(f"  自适应权重: {[f'{w:.3f}' for w in adaptive_weights]}")
    print(f"  总损失: {normalized_total:.3f}")
    print(f"  各部分均衡参与训练 ✓")

if __name__ == "__main__":
    print("🧩 四大损失函数详细演示")
    print("=" * 70)
    
    demo_policy_loss()
    demo_value_loss()
    demo_entropy_loss()
    demo_direction_loss()
    demo_combined_loss()
    
    print("\n🎯 总结")
    print("=" * 70)
    print("1. Policy Loss: 学习更好的决策策略 (PPO核心)")
    print("2. Value Loss: 准确估计状态价值 (支持策略学习)")
    print("3. Entropy Loss: 维持探索与利用平衡 (防止早熟)")
    print("4. Direction Loss: 学习空间感知能力 (辅助任务)")
    print("\n🔧 损失归一化确保四者协调工作，而非某个主导训练！")