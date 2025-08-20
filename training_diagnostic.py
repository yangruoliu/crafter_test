#!/usr/bin/env python3
"""
训练问题快速诊断脚本
"""

def analyze_training_metrics():
    """基于提供的训练日志分析问题"""
    
    print("🔍 训练状态快速诊断")
    print("=" * 60)
    
    # 从用户提供的训练日志
    metrics = {
        "ep_rew_mean": -5.94,
        "direction_accuracy": 0.338,
        "adaptive_weights": {
            "policy": 0.164,
            "value": 0.144, 
            "entropy": 0.00707,
            "direction": 0.000242
        },
        "losses": {
            "policy": 0.0893,
            "value": 0.00233,
            "entropy": -0.000512,
            "direction": 1.75
        },
        "cov": {
            "policy": 10.7,
            "value": 31.3,
            "entropy": 23.1,
            "direction": 0.0788
        },
        "explained_variance": -3.09,
        "approx_kl": 3.472087e-08,
        "clip_fraction": 0
    }
    
    print("📊 关键指标分析:")
    
    # 1. 任务表现分析
    print(f"\n🎮 任务表现:")
    if metrics["ep_rew_mean"] < 0:
        print(f"  ❌ 平均奖励: {metrics['ep_rew_mean']:.2f} (负值，任务失败)")
        severity = "严重"
    elif metrics["ep_rew_mean"] < 10:
        print(f"  ⚠️ 平均奖励: {metrics['ep_rew_mean']:.2f} (较低)")
        severity = "中等"
    else:
        print(f"  ✅ 平均奖励: {metrics['ep_rew_mean']:.2f} (良好)")
        severity = "轻微"
    
    # 2. 权重分布分析
    print(f"\n⚖️ 权重分布分析:")
    weights = metrics["adaptive_weights"]
    total_weight = sum(weights.values())
    
    for name, weight in weights.items():
        percentage = weight / total_weight * 100
        if name == "entropy" and percentage < 2:
            status = "❌ 过低(探索不足)"
        elif name == "direction" and percentage < 1:
            status = "❌ 过低(辅助任务失效)"
        elif percentage > 50:
            status = "⚠️ 过高(可能主导)"
        else:
            status = "✅ 正常"
        
        print(f"  {name:9}: {weight:.6f} ({percentage:4.1f}%) {status}")
    
    # 3. 探索状态分析
    print(f"\n🎲 探索状态:")
    entropy_loss = abs(metrics["losses"]["entropy"])
    if entropy_loss < 0.001:
        print(f"  ❌ Entropy loss: {entropy_loss:.6f} (几乎无探索)")
    elif entropy_loss < 0.01:
        print(f"  ⚠️ Entropy loss: {entropy_loss:.6f} (探索不足)")
    else:
        print(f"  ✅ Entropy loss: {entropy_loss:.6f} (探索充分)")
    
    # 4. 方向学习分析  
    print(f"\n🧭 方向学习:")
    dir_acc = metrics["direction_accuracy"]
    random_acc = 1/9  # 9个方向的随机准确率
    
    if dir_acc <= random_acc * 1.2:  # 几乎等于随机
        print(f"  ❌ Direction accuracy: {dir_acc:.3f} (接近随机: {random_acc:.3f})")
    elif dir_acc < 0.5:
        print(f"  ⚠️ Direction accuracy: {dir_acc:.3f} (有改善但不足)")
    else:
        print(f"  ✅ Direction accuracy: {dir_acc:.3f} (良好)")
    
    # 5. 训练稳定性分析
    print(f"\n📈 训练稳定性:")
    high_cov_count = sum(1 for cov in metrics["cov"].values() if cov > 20)
    
    if high_cov_count >= 2:
        print(f"  ❌ {high_cov_count}个损失变异系数>20 (训练不稳定)")
    elif high_cov_count == 1:
        print(f"  ⚠️ {high_cov_count}个损失变异系数>20 (轻微不稳定)")
    else:
        print(f"  ✅ 变异系数正常 (训练稳定)")
    
    if metrics["explained_variance"] < 0:
        print(f"  ❌ Explained variance: {metrics['explained_variance']:.2f} (价值函数很差)")
    
    # 6. 策略更新分析
    print(f"\n🔄 策略更新:")
    if metrics["approx_kl"] < 1e-6:
        print(f"  ❌ KL散度: {metrics['approx_kl']:.2e} (策略几乎不更新)")
    elif metrics["clip_fraction"] == 0:
        print(f"  ⚠️ Clip fraction: 0 (更新幅度很小)")
    
    return severity

def provide_solutions(severity):
    """根据问题严重程度提供解决方案"""
    
    print(f"\n🔧 解决方案 (问题严重程度: {severity})")
    print("=" * 60)
    
    if severity == "严重":
        print("🚨 立即行动 - 训练基本失效:")
        print("  1. 停止当前训练")
        print("  2. 增加最小权重保护: min_weight = 0.1")
        print("  3. 大幅提升探索: ent_coef = 0.08")
        print("  4. 增强方向学习: direction_weight = 0.4")
        print("  5. 重新开始训练")
        
        print("\n🎯 具体修改:")
        print("  - 使用 improved_training_config_v2.py")
        print("  - 或者手动修改参数重新训练")
        
    elif severity == "中等":
        print("⚠️ 调整优化:")
        print("  1. 增加exploration权重")
        print("  2. 检查direction learning效果")
        print("  3. 降低norm_decay提高适应性")
        
    else:
        print("✅ 微调即可:")
        print("  1. 观察几个epoch看是否改善")
        print("  2. 可能只需要更多训练时间")

def show_expected_improvements():
    """显示修复后的预期改善"""
    
    print(f"\n📈 修复后预期改善")
    print("=" * 60)
    
    improvements = [
        ("ep_rew_mean", "-5.94", "> 0", "任务开始成功"),
        ("adaptive_weights/entropy", "0.007", "> 0.02", "恢复探索"),
        ("adaptive_weights/direction", "0.0002", "> 0.05", "方向学习生效"),
        ("direction_accuracy", "33.8%", "> 40%", "空间感知提升"),
        ("entropy_loss", "~0", "> 0.01", "探索活跃"),
        ("explained_variance", "-3.09", "> 0", "价值函数改善")
    ]
    
    print(f"{'指标':<25} {'当前':<10} {'目标':<10} {'含义'}")
    print("-" * 60)
    for metric, current, target, meaning in improvements:
        print(f"{metric:<25} {current:<10} {target:<10} {meaning}")

if __name__ == "__main__":
    severity = analyze_training_metrics()
    provide_solutions(severity)
    show_expected_improvements()
    
    print(f"\n💡 关键建议:")
    print("1. 当前训练已陷入局部最优，建议重新开始")
    print("2. 使用修复后的配置: improved_training_config_v2.py")
    print("3. 密切监控前50次迭代的指标变化")
    print("4. 如果仍有问题，可进一步调整参数")