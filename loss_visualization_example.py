#!/usr/bin/env python3
"""
Loss组成可视化示例
"""

def visualize_loss_composition():
    """可视化展示loss的组成和计算过程"""
    
    print("🧩 最终Loss组成分解")
    print("=" * 70)
    
    # 模拟训练过程中的实际数值
    print("\n📊 实际训练数值示例 (基于你的训练日志)")
    
    # 原始损失值 (不同量级)
    original_losses = {
        "policy_loss": 1.2,
        "value_loss": 48.5,      # 很大
        "entropy_loss": -2.47,   # 负值
        "direction_loss": 1.76
    }
    
    # 移动平均值 (用于归一化)
    moving_averages = {
        "policy_loss": 1.0,
        "value_loss": 48.5,
        "entropy_loss": -2.47,
        "direction_loss": 1.58
    }
    
    # 基础权重系数
    base_weights = {
        "policy": 1.0,
        "entropy": 0.02,
        "value": 0.3,
        "direction": 0.2
    }
    
    print("🔸 第一步: 原始损失值")
    for loss_name, value in original_losses.items():
        print(f"  {loss_name:15}: {value:8.3f}")
    
    print("\n🔸 第二步: 归一化处理 (loss / |moving_average|)")
    normalized_losses = {}
    for loss_name, value in original_losses.items():
        ma = moving_averages[loss_name]
        normalized = value / abs(ma) if ma != 0 else 0
        normalized_losses[loss_name] = normalized
        print(f"  {loss_name:15}: {value:6.3f} / |{ma:6.3f}| = {normalized:6.3f}")
    
    print("\n🔸 第三步: 模拟自适应权重 (基于变异系数)")
    # 模拟变异系数
    cov_values = {
        "policy": 0.25,
        "value": 0.80,
        "entropy": 0.15,
        "direction": 0.45
    }
    
    total_cov = sum(cov_values.values())
    adaptive_weights = {}
    
    for loss_type, cov in cov_values.items():
        base_weight = base_weights[loss_type]
        # 平滑处理
        cov_smooth = cov ** 0.5
        adaptive_weight = (cov_smooth / (total_cov ** 0.5)) * base_weight
        # 最小权重保护
        adaptive_weight = max(0.01 * base_weight, adaptive_weight)
        adaptive_weights[loss_type] = adaptive_weight
        
        print(f"  {loss_type:9}: CoV={cov:.3f} → 自适应权重={adaptive_weight:.4f}")
    
    print(f"\n🔸 第四步: 最终Loss计算")
    
    # 计算最终loss
    final_loss = 0.0
    
    loss_mapping = {
        "policy": "policy_loss",
        "value": "value_loss", 
        "entropy": "entropy_loss",
        "direction": "direction_loss"
    }
    
    print("  各部分贡献:")
    for loss_type, weight in adaptive_weights.items():
        normalized_val = normalized_losses[loss_mapping[loss_type]]
        contribution = weight * normalized_val
        final_loss += contribution
        
        print(f"    {loss_type:9}: {weight:.4f} × {normalized_val:.3f} = {contribution:7.4f}")
    
    print(f"\n  🎯 最终Loss = {final_loss:.4f}")
    
    print("\n" + "=" * 70)
    print("📈 对比分析")
    
    # 对比原始权重系统
    print("\n❌ 原始权重系统 (有问题):")
    original_final = 0.0
    contributions_old = {}
    
    weight_mapping = {
        "policy_loss": 1.0,
        "entropy_loss": 0.02,
        "value_loss": 0.3,
        "direction_loss": 0.2
    }
    
    for loss_name, loss_val in original_losses.items():
        weight = weight_mapping[loss_name]
        contribution = weight * loss_val
        original_final += contribution
        contributions_old[loss_name] = contribution
        
        print(f"  {loss_name:15}: {weight:.2f} × {loss_val:6.3f} = {contribution:8.3f}")
    
    print(f"  总计: {original_final:.3f}")
    print(f"  💡 Value loss贡献: {contributions_old['value_loss']:.1f} (主导训练!)")
    
    print("\n✅ 归一化权重系统 (修复后):")
    print(f"  总计: {final_loss:.4f}")
    print(f"  💡 各部分贡献均衡，权重设置有意义")
    
    print("\n🎯 关键改进:")
    print("  1. 所有损失归一化到相似量级 (~1.0)")
    print("  2. 自适应权重防止某个损失主导")
    print("  3. 变异系数高的损失获得更多关注")
    print("  4. 方向预测任务真正参与训练")

def explain_each_loss_component():
    """详细解释每个loss组件"""
    
    print("\n🔍 各Loss组件详细解释")
    print("=" * 70)
    
    components = {
        "Policy Loss": {
            "作用": "学习更好的策略，提升任务表现",
            "计算": "PPO的Clipped Surrogate Objective",
            "期望": "随训练逐渐减小，表示策略改进",
            "典型值": "0.5 - 2.0",
            "监控": "配合episode reward观察"
        },
        
        "Value Loss": {
            "作用": "准确估计状态价值，支持优势计算", 
            "计算": "预测价值与实际回报的MSE",
            "期望": "随训练减小，explained_variance接近1",
            "典型值": "10 - 100 (归一化前)",
            "监控": "配合explained_variance观察"
        },
        
        "Entropy Loss": {
            "作用": "维持探索性，防止策略过早收敛",
            "计算": "策略分布的负熵",
            "期望": "训练初期较小(高熵)，后期增大(低熵)",
            "典型值": "-1 到 -3 (负值正常)",
            "监控": "太小=过度探索，太大=探索不足"
        },
        
        "Direction Loss": {
            "作用": "学习空间感知，预测目标方向",
            "计算": "9分类交叉熵损失", 
            "期望": "随训练减小，accuracy提升",
            "典型值": "0.5 - 2.5",
            "监控": "配合direction_accuracy观察"
        }
    }
    
    for name, info in components.items():
        print(f"\n🎯 {name}")
        for key, value in info.items():
            print(f"  {key:6}: {value}")

if __name__ == "__main__":
    visualize_loss_composition()
    explain_each_loss_component()
    
    print("\n📋 总结")
    print("=" * 70)
    print("最终Loss = 四个归一化损失的加权和")
    print("  ├─ Policy Loss:    学习决策策略")
    print("  ├─ Value Loss:     学习价值估计") 
    print("  ├─ Entropy Loss:   维持探索性")
    print("  └─ Direction Loss: 学习空间感知")
    print("\n🎯 每个组件都很重要，缺一不可！")