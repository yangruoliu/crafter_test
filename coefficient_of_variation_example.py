#!/usr/bin/env python3
"""
变异系数 (Coefficient of Variation) 实例演示
"""

import numpy as np
import matplotlib.pyplot as plt

def demonstrate_coefficient_of_variation():
    """演示变异系数的含义和计算"""
    
    print("🔍 变异系数 (Coefficient of Variation) 详细解释")
    print("=" * 60)
    
    # 示例1: 不同变异程度的数据
    print("\n📊 示例1: 不同变异程度的损失数据")
    
    # 稳定的损失 (低变异)
    stable_losses = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 1.01]
    
    # 不稳定的损失 (高变异)  
    unstable_losses = [1.0, 2.5, 0.3, 1.8, 0.1, 2.2, 0.5, 1.9, 0.2, 2.1]
    
    def calculate_cov(data):
        mean = np.mean(data)
        std = np.std(data)
        cov = std / abs(mean) if mean != 0 else 0
        return mean, std, cov
    
    stable_mean, stable_std, stable_cov = calculate_cov(stable_losses)
    unstable_mean, unstable_std, unstable_cov = calculate_cov(unstable_losses)
    
    print(f"稳定损失:")
    print(f"  数据: {stable_losses}")
    print(f"  平均值: {stable_mean:.3f}")
    print(f"  标准差: {stable_std:.3f}")
    print(f"  变异系数: {stable_cov:.3f} (低变异 - 稳定)")
    
    print(f"\n不稳定损失:")
    print(f"  数据: {unstable_losses}")
    print(f"  平均值: {unstable_mean:.3f}")
    print(f"  标准差: {unstable_std:.3f}")
    print(f"  变异系数: {unstable_cov:.3f} (高变异 - 不稳定)")
    
    # 示例2: 你的训练日志中的实际数据
    print(f"\n📈 示例2: 你的训练日志分析")
    print("根据你提供的训练日志:")
    
    # 从你的日志中提取的数据
    training_data = {
        "policy": {"ma": -4.85, "cov": -0.787},  # 原始有问题的值
        "value": {"ma": 48.5, "cov": 95.1},     # 原始有问题的值  
        "entropy": {"ma": -2.47, "cov": -0.0646}, # 原始有问题的值
        "direction": {"ma": 1.58, "cov": 6.61}
    }
    
    print("原始计算结果 (有问题):")
    for loss_type, data in training_data.items():
        print(f"  {loss_type:9}: MA={data['ma']:6.2f}, CoV={data['cov']:6.3f}")
    
    # 修复后的计算 (使用绝对值)
    print("\n修复后的计算:")
    for loss_type, data in training_data.items():
        ma = data["ma"]
        # 模拟方差计算 (实际中来自移动方差)
        if loss_type == "value":
            var = (95.1 * abs(ma)) ** 2  # 反推方差
        else:
            var = (abs(data["cov"]) * abs(ma)) ** 2  # 反推方差
        
        std = np.sqrt(var)
        fixed_cov = std / abs(ma) if ma != 0 else 0
        
        print(f"  {loss_type:9}: MA={ma:6.2f}, CoV={fixed_cov:6.3f} (修复后)")

def demonstrate_adaptive_weighting():
    """演示自适应权重分配"""
    
    print("\n🎯 自适应权重分配演示")
    print("=" * 60)
    
    # 模拟4种损失的变异系数
    cov_values = {
        "policy": 0.2,      # 低变异 - 稳定
        "value": 1.5,       # 高变异 - 不稳定  
        "entropy": 0.1,     # 很低变异 - 很稳定
        "direction": 0.8    # 中等变异 - 中等稳定
    }
    
    # 基础权重
    base_weights = {
        "policy": 1.0,
        "value": 0.3,
        "entropy": 0.02,
        "direction": 0.2
    }
    
    print("变异系数 (CoV):")
    for loss_type, cov in cov_values.items():
        interpretation = ""
        if cov < 0.3:
            interpretation = "稳定，需要较少关注"
        elif cov < 1.0:
            interpretation = "中等变异，需要适度关注"
        else:
            interpretation = "不稳定，需要更多关注"
        print(f"  {loss_type:9}: {cov:.3f} - {interpretation}")
    
    # 计算自适应权重 (简化版本)
    total_cov = sum(cov_values.values())
    
    print(f"\n自适应权重分配:")
    print(f"总变异系数: {total_cov:.3f}")
    
    for loss_type, cov in cov_values.items():
        base_weight = base_weights[loss_type]
        # 变异系数高的获得更多权重
        adaptive_weight = (cov / total_cov) * base_weight
        
        print(f"  {loss_type:9}: 基础权重={base_weight:.3f}, 自适应权重={adaptive_weight:.4f}")

def demonstrate_loss_normalization():
    """演示损失归一化过程"""
    
    print("\n⚖️ 损失归一化演示")
    print("=" * 60)
    
    # 不同量级的损失
    current_losses = {
        "policy": 1.2,
        "value": 45.8,      # 很大的值
        "entropy": -2.3,    # 负值
        "direction": 1.7
    }
    
    # 移动平均 (历史统计)
    moving_averages = {
        "policy": 1.0,
        "value": 48.5,
        "entropy": -2.47,
        "direction": 1.58
    }
    
    print("原始损失值 (量级差异很大):")
    for loss_type, loss in current_losses.items():
        print(f"  {loss_type:9}: {loss:6.2f}")
    
    print("\n移动平均值:")
    for loss_type, ma in moving_averages.items():
        print(f"  {loss_type:9}: {ma:6.2f}")
    
    print("\n归一化后的损失 (loss / |moving_average|):")
    normalized_losses = {}
    for loss_type, loss in current_losses.items():
        ma = moving_averages[loss_type]
        normalized = loss / abs(ma) if ma != 0 else 0
        normalized_losses[loss_type] = normalized
        print(f"  {loss_type:9}: {loss:6.2f} / |{ma:6.2f}| = {normalized:.3f}")
    
    print("\n📝 归一化的好处:")
    print("  1. 所有损失都在相似的量级上 (接近1.0)")
    print("  2. 权重设置变得有意义")
    print("  3. 避免大值损失主导训练")

if __name__ == "__main__":
    demonstrate_coefficient_of_variation()
    demonstrate_adaptive_weighting()
    demonstrate_loss_normalization()
    
    print("\n🎯 总结")
    print("=" * 60)
    print("1. 变异系数 = 标准差 / |平均值|")
    print("2. 高变异系数 → 损失不稳定 → 需要更多关注")
    print("3. 损失归一化解决不同损失量级差异问题")
    print("4. 自适应权重根据损失稳定性动态调整")
    print("5. 最终目标: 让所有损失都得到合适的关注")