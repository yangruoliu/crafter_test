#!/usr/bin/env python3
"""
训练指标快速参考和状态检查
"""

def check_metric_status(value, good_range, name):
    """检查指标状态"""
    if isinstance(good_range, tuple):
        min_val, max_val = good_range
        if min_val <= value <= max_val:
            status = "✅ 正常"
        elif value < min_val:
            status = "❌ 过低" 
        else:
            status = "⚠️ 过高"
    else:
        # 对于特殊情况
        if good_range == "positive":
            status = "✅ 正常" if value > 0 else "❌ 负值"
        elif good_range == "near_zero":
            status = "❌ 几乎为0" if abs(value) < 1e-6 else "✅ 正常"
        else:
            status = "？未知"
    
    return status

def analyze_your_metrics():
    """分析你提供的具体指标"""
    
    print("🔍 你的训练指标快速诊断")
    print("=" * 60)
    
    # 你的实际指标值
    your_metrics = {
        # 游戏表现
        "ep_len_mean": (168, (50, 500), "平均游戏长度"),
        "ep_rew_mean": (-5.94, "positive", "平均奖励"),
        
        # 自适应权重 (百分比)
        "adaptive_weights_policy": (16.4, (20, 40), "策略权重%"),
        "adaptive_weights_value": (14.4, (10, 30), "价值权重%"), 
        "adaptive_weights_entropy": (0.7, (2, 10), "探索权重%"),
        "adaptive_weights_direction": (0.02, (5, 20), "方向权重%"),
        
        # 核心训练指标
        "approx_kl": (3.47e-08, (1e-4, 1e-2), "策略变化度"),
        "clip_fraction": (0, (0.1, 0.3), "PPO裁剪比例"),
        "direction_accuracy": (33.8, (20, 50), "方向预测准确率%"),
        
        # 损失指标
        "entropy_loss": (-0.000512, (-2.0, -0.01), "探索损失"),
        "policy_gradient_loss": (0.0893, (0.01, 2.0), "策略损失"),
        "value_loss": (0.00233, (0.1, 100), "价值损失"),
        "direction_loss": (1.75, (1.0, 2.5), "方向损失"),
        
        # 稳定性指标
        "explained_variance": (-3.09, (0.0, 1.0), "价值函数质量"),
        "policy_cov": (10.7, (0.1, 5.0), "策略变异系数"),
        "value_cov": (31.3, (0.1, 5.0), "价值变异系数"),
        "entropy_cov": (23.1, (0.1, 5.0), "探索变异系数"),
        "direction_cov": (0.0788, (0.1, 5.0), "方向变异系数"),
    }
    
    # 按类别分析
    categories = {
        "🎮 游戏表现": ["ep_len_mean", "ep_rew_mean"],
        "⚖️ 权重分布": ["adaptive_weights_policy", "adaptive_weights_value", 
                      "adaptive_weights_entropy", "adaptive_weights_direction"],
        "📈 训练核心": ["approx_kl", "clip_fraction"],
        "🎯 任务指标": ["direction_accuracy", "direction_loss"],
        "📊 损失函数": ["entropy_loss", "policy_gradient_loss", "value_loss"],
        "📉 稳定性": ["explained_variance", "policy_cov", "value_cov", 
                    "entropy_cov", "direction_cov"]
    }
    
    problem_count = 0
    warning_count = 0
    
    for category, metrics in categories.items():
        print(f"\n{category}")
        print("-" * 40)
        
        for metric in metrics:
            if metric in your_metrics:
                value, good_range, description = your_metrics[metric]
                status = check_metric_status(value, good_range, metric)
                
                if "❌" in status:
                    problem_count += 1
                elif "⚠️" in status:
                    warning_count += 1
                
                print(f"  {description:<20}: {value:>10} {status}")
    
    # 总结
    print(f"\n🎯 总体评估")
    print("=" * 60)
    print(f"❌ 严重问题: {problem_count} 个")
    print(f"⚠️ 需要注意: {warning_count} 个")
    
    if problem_count >= 5:
        severity = "🚨 危急"
        action = "立即停止训练，使用修复配置重新开始"
    elif problem_count >= 3:
        severity = "⚠️ 严重"  
        action = "需要重大调整参数"
    elif problem_count >= 1:
        severity = "⚠️ 中等"
        action = "需要调整部分参数"
    else:
        severity = "✅ 良好"
        action = "继续训练并观察"
    
    print(f"严重程度: {severity}")
    print(f"建议行动: {action}")

def show_ideal_ranges():
    """显示理想的指标范围"""
    
    print(f"\n📋 理想指标范围速查表")
    print("=" * 60)
    
    ideal_ranges = {
        "🎮 游戏表现": {
            "ep_rew_mean": "> 0 且持续上升",
            "ep_len_mean": "100-300 (游戏相关)"
        },
        "⚖️ 权重分布": {
            "policy权重": "20%-40%",
            "value权重": "10%-30%", 
            "entropy权重": "2%-10%",
            "direction权重": "5%-20%"
        },
        "📈 训练活跃度": {
            "approx_kl": "0.0001-0.01",
            "clip_fraction": "0.1-0.3"
        },
        "🎯 任务学习": {
            "direction_accuracy": "> 40%",
            "direction_loss": "< 1.5"
        },
        "📊 探索状态": {
            "entropy_loss": "-0.1 到 -1.0"
        },
        "📉 训练稳定性": {
            "explained_variance": "0.5-1.0",
            "各变异系数": "0.5-3.0"
        }
    }
    
    for category, ranges in ideal_ranges.items():
        print(f"\n{category}")
        for metric, range_val in ranges.items():
            print(f"  {metric:<20}: {range_val}")

def show_troubleshooting():
    """显示常见问题排查"""
    
    print(f"\n🔧 常见问题快速排查")
    print("=" * 60)
    
    issues = {
        "ep_rew_mean < 0": "任务失败 → 检查奖励设计和环境配置",
        "entropy权重 < 2%": "探索不足 → 增加ent_coef和最小权重保护",
        "direction权重 < 5%": "辅助任务失效 → 增加direction_weight",
        "approx_kl < 1e-6": "策略停滞 → 增加learning_rate或重启训练",
        "clip_fraction = 0": "更新太小 → 增加learning_rate",
        "explained_variance < 0": "价值函数差 → 增加vf_coef或更多训练",
        "变异系数 > 10": "训练不稳定 → 降低learning_rate或调整权重"
    }
    
    for issue, solution in issues.items():
        print(f"• {issue}")
        print(f"  └─ {solution}")

if __name__ == "__main__":
    analyze_your_metrics()
    show_ideal_ranges()
    show_troubleshooting()
    
    print(f"\n💡 针对你的情况的具体建议:")
    print("1. 立即使用 improved_training_config_v2.py 重新训练")
    print("2. 重点监控前20次迭代的权重分布变化")  
    print("3. 期望看到 entropy 和 direction 权重显著增加")
    print("4. ep_rew_mean 应该在100次迭代内转正")
    print("5. 如果仍有问题，可进一步增加基础权重")