#!/usr/bin/env python3
"""
快速模型测试脚本 - 用于快速验证训练后的模型
"""

import gym
import crafter
import env_wrapper
from model_with_attn import CustomPPO
import numpy as np
import time

def quick_test_model(model_path: str, num_episodes: int = 5, render: bool = False):
    """
    快速测试训练后的模型
    
    Args:
        model_path: 训练好的模型路径
        num_episodes: 测试回合数
        render: 是否显示游戏画面
    """
    print(f"🚀 快速测试模型: {model_path}")
    print(f"测试回合数: {num_episodes}")
    
    # 设置环境
    env = gym.make("MyCrafter-v0")
    env = env_wrapper.MineStoneWrapper(env)
    env = env_wrapper.InitWrapper(env, init_items=["wood_pickaxe"], init_num=[1])
    env = env_wrapper.DirectionLabelWrapper(env, target_obj_id=7, target_obj_name="stone")
    
    # 加载模型
    try:
        model = CustomPPO.load(model_path, env=env)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试结果统计
    results = {
        'total_rewards': [],
        'episode_lengths': [],
        'completion_rates': []
    }
    
    for episode in range(num_episodes):
        print(f"\n--- 测试回合 {episode + 1}/{num_episodes} ---")
        
        obs = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        start_time = time.time()
        
        while not done and steps < 500:  # 限制最大步数为500
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(0.02)
        
        episode_time = time.time() - start_time
        
        # 记录结果
        results['total_rewards'].append(total_reward)
        results['episode_lengths'].append(steps)
        results['completion_rates'].append(done)
        
        print(f"   奖励: {total_reward:.2f}")
        print(f"   步数: {steps}")
        print(f"   完成: {'是' if done else '否'}")
        print(f"   耗时: {episode_time:.2f}s")
    
    # 生成总结报告
    print(f"\n" + "="*50)
    print(f"🎯 快速测试总结")
    print(f"="*50)
    
    avg_reward = np.mean(results['total_rewards'])
    std_reward = np.std(results['total_rewards'])
    avg_steps = np.mean(results['episode_lengths'])
    completion_rate = np.mean(results['completion_rates'])
    
    print(f"📊 性能指标:")
    print(f"   平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   奖励范围: [{np.min(results['total_rewards']):.2f}, {np.max(results['total_rewards']):.2f}]")
    print(f"   平均步数: {avg_steps:.1f}")
    print(f"   完成率: {completion_rate:.1%}")
    
    # 简单评级
    if avg_reward > 20 and completion_rate > 0.6:
        grade = "🟢 良好"
    elif avg_reward > 0 and completion_rate > 0.2:
        grade = "🟡 一般"
    else:
        grade = "🔴 需要改进"
    
    print(f"   模型评级: {grade}")
    
    env.close()
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_steps': avg_steps,
        'completion_rate': completion_rate,
        'grade': grade
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python quick_test_model.py <模型路径> [回合数] [是否渲染]")
        print("示例: python quick_test_model.py ./stone_with_direction_v4_20240804_063000 5 True")
        sys.exit(1)
    
    model_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    render = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    
    quick_test_model(model_path, num_episodes, render)