#!/usr/bin/env python3
"""
训练后模型测试脚本
支持多种测试模式和详细的性能分析
"""

import gym
import crafter
import env_wrapper
from model_with_attn import CustomResNet, CustomACPolicy, CustomPPO
import torch.nn as nn
import torch
import numpy as np
import os
import argparse
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

class ModelTester:
    def __init__(self, model_path: str, render: bool = False):
        """
        初始化模型测试器
        
        Args:
            model_path: 训练好的模型路径
            render: 是否渲染游戏画面
        """
        self.model_path = model_path
        self.render = render
        self.model = None
        self.env = None
        self.test_results = defaultdict(list)
        
        self._setup_environment()
        self._load_model()
    
    def _setup_environment(self):
        """设置测试环境"""
        print("🔧 设置测试环境...")
        
        # 创建环境 - 使用与训练相同的配置
        self.env = gym.make("MyCrafter-v0")
        
        # 应用与训练相同的wrapper
        self.env = env_wrapper.MineStoneWrapper(self.env)
        self.env = env_wrapper.InitWrapper(
            self.env, 
            init_items=["wood_pickaxe"], 
            init_num=[1]
        )
        self.env = env_wrapper.DirectionLabelWrapper(
            self.env, 
            target_obj_id=7,  # Stone object ID
            target_obj_name="stone"
        )
        
        print(f"✅ 环境设置完成")
        print(f"   观察空间: {self.env.observation_space}")
        print(f"   动作空间: {self.env.action_space}")
    
    def _load_model(self):
        """加载训练好的模型"""
        print(f"📦 加载模型: {self.model_path}")
        
        try:
            # 加载模型
            self.model = CustomPPO.load(self.model_path, env=self.env)
            print("✅ 模型加载成功")
            
            # 显示模型信息
            print(f"   模型类型: {type(self.model).__name__}")
            if hasattr(self.model, 'loss_normalization'):
                print(f"   损失归一化: {self.model.loss_normalization}")
            if hasattr(self.model, 'direction_weight'):
                print(f"   方向权重: {self.model.direction_weight}")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def run_single_episode(self, max_steps: int = 1000, verbose: bool = True) -> Dict:
        """
        运行单个测试回合
        
        Args:
            max_steps: 最大步数
            verbose: 是否显示详细信息
            
        Returns:
            回合统计信息
        """
        obs = self.env.reset()
        done = False
        step = 0
        total_reward = 0
        
        # 统计信息
        actions_taken = []
        rewards_received = []
        direction_predictions = []
        direction_accuracies = []
        
        if verbose:
            print(f"\n🎮 开始新回合...")
        
        while not done and step < max_steps:
            # 预测动作
            action, _states = self.model.predict(obs, deterministic=True)
            
            # 如果模型支持方向预测，获取方向信息
            direction_pred = None
            direction_accuracy = None
            
            if hasattr(self.model.policy, 'predict_direction'):
                try:
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        direction_logits = self.model.policy.predict_direction(obs_tensor)
                        direction_pred = torch.argmax(direction_logits, dim=1).item()
                        
                        # 如果环境提供真实方向标签
                        if hasattr(self.env, 'get_direction_label'):
                            true_direction = self.env.get_direction_label()
                            direction_accuracy = (direction_pred == true_direction)
                except:
                    pass
            
            # 执行动作
            obs, reward, done, info = self.env.step(action)
            
            # 记录统计信息
            actions_taken.append(action)
            rewards_received.append(reward)
            if direction_pred is not None:
                direction_predictions.append(direction_pred)
            if direction_accuracy is not None:
                direction_accuracies.append(direction_accuracy)
            
            total_reward += reward
            step += 1
            
            if self.render:
                self.env.render()
                time.sleep(0.05)  # 稍微减慢速度以便观察
            
            if verbose and step % 100 == 0:
                print(f"   步数: {step}, 累计奖励: {total_reward:.2f}")
        
        # 编译回合统计
        episode_stats = {
            'total_steps': step,
            'total_reward': total_reward,
            'average_reward': total_reward / step if step > 0 else 0,
            'actions_taken': actions_taken,
            'rewards_received': rewards_received,
            'final_info': info,
            'completed': done,
        }
        
        # 添加方向预测统计
        if direction_predictions:
            episode_stats['direction_predictions'] = direction_predictions
            episode_stats['avg_direction_accuracy'] = np.mean(direction_accuracies) if direction_accuracies else 0
        
        if verbose:
            print(f"✅ 回合结束:")
            print(f"   总步数: {step}")
            print(f"   总奖励: {total_reward:.2f}")
            print(f"   平均奖励: {episode_stats['average_reward']:.3f}")
            if 'avg_direction_accuracy' in episode_stats:
                print(f"   方向准确率: {episode_stats['avg_direction_accuracy']:.3f}")
            print(f"   是否完成: {done}")
        
        return episode_stats
    
    def run_multiple_episodes(self, num_episodes: int = 10, max_steps: int = 1000) -> Dict:
        """
        运行多个测试回合并生成统计报告
        
        Args:
            num_episodes: 测试回合数
            max_steps: 每回合最大步数
            
        Returns:
            汇总统计信息
        """
        print(f"\n🧪 开始批量测试 ({num_episodes} 回合)...")
        
        all_episode_stats = []
        
        for episode in range(num_episodes):
            print(f"\n--- 回合 {episode + 1}/{num_episodes} ---")
            
            episode_stats = self.run_single_episode(
                max_steps=max_steps, 
                verbose=(episode < 3)  # 只显示前3回合的详细信息
            )
            all_episode_stats.append(episode_stats)
            
            # 简要进度报告
            if episode >= 3:
                print(f"回合 {episode + 1}: 奖励={episode_stats['total_reward']:.2f}, "
                      f"步数={episode_stats['total_steps']}")
        
        # 生成汇总统计
        summary_stats = self._generate_summary_stats(all_episode_stats)
        
        return summary_stats
    
    def _generate_summary_stats(self, episode_stats: List[Dict]) -> Dict:
        """生成汇总统计信息"""
        print(f"\n📊 生成测试报告...")
        
        # 提取关键指标
        total_rewards = [ep['total_reward'] for ep in episode_stats]
        total_steps = [ep['total_steps'] for ep in episode_stats]
        avg_rewards = [ep['average_reward'] for ep in episode_stats]
        completion_rates = [ep['completed'] for ep in episode_stats]
        
        # 方向预测准确率
        direction_accuracies = [
            ep['avg_direction_accuracy'] 
            for ep in episode_stats 
            if 'avg_direction_accuracy' in ep
        ]
        
        summary = {
            'num_episodes': len(episode_stats),
            'reward_stats': {
                'mean': np.mean(total_rewards),
                'std': np.std(total_rewards),
                'min': np.min(total_rewards),
                'max': np.max(total_rewards),
                'median': np.median(total_rewards)
            },
            'steps_stats': {
                'mean': np.mean(total_steps),
                'std': np.std(total_steps),
                'min': np.min(total_steps),
                'max': np.max(total_steps),
                'median': np.median(total_steps)
            },
            'completion_rate': np.mean(completion_rates),
            'avg_reward_per_step': {
                'mean': np.mean(avg_rewards),
                'std': np.std(avg_rewards)
            }
        }
        
        if direction_accuracies:
            summary['direction_accuracy'] = {
                'mean': np.mean(direction_accuracies),
                'std': np.std(direction_accuracies)
            }
        
        return summary
    
    def print_test_report(self, summary_stats: Dict):
        """打印详细的测试报告"""
        print("\n" + "="*70)
        print("🎯 模型测试报告")
        print("="*70)
        
        print(f"📈 测试概况:")
        print(f"   测试回合数: {summary_stats['num_episodes']}")
        print(f"   任务完成率: {summary_stats['completion_rate']:.1%}")
        
        print(f"\n🏆 奖励统计:")
        reward_stats = summary_stats['reward_stats']
        print(f"   平均奖励: {reward_stats['mean']:.2f} ± {reward_stats['std']:.2f}")
        print(f"   奖励范围: [{reward_stats['min']:.2f}, {reward_stats['max']:.2f}]")
        print(f"   中位数奖励: {reward_stats['median']:.2f}")
        
        print(f"\n⏱️ 步数统计:")
        steps_stats = summary_stats['steps_stats']
        print(f"   平均步数: {steps_stats['mean']:.1f} ± {steps_stats['std']:.1f}")
        print(f"   步数范围: [{steps_stats['min']}, {steps_stats['max']}]")
        
        print(f"\n📊 效率指标:")
        efficiency = summary_stats['avg_reward_per_step']
        print(f"   平均每步奖励: {efficiency['mean']:.4f} ± {efficiency['std']:.4f}")
        
        if 'direction_accuracy' in summary_stats:
            print(f"\n🎯 方向预测:")
            dir_acc = summary_stats['direction_accuracy']
            print(f"   平均准确率: {dir_acc['mean']:.1%} ± {dir_acc['std']:.1%}")
        
        # 性能评级
        avg_reward = reward_stats['mean']
        completion_rate = summary_stats['completion_rate']
        
        print(f"\n🏅 模型性能评级:")
        if avg_reward > 50 and completion_rate > 0.8:
            grade = "A+ (优秀)"
        elif avg_reward > 20 and completion_rate > 0.6:
            grade = "B+ (良好)"
        elif avg_reward > 0 and completion_rate > 0.4:
            grade = "C+ (一般)"
        elif avg_reward > -20:
            grade = "D (较差)"
        else:
            grade = "F (失败)"
        
        print(f"   综合评级: {grade}")
        
        print("="*70)
    
    def save_test_results(self, summary_stats: Dict, output_path: str = None):
        """保存测试结果到文件"""
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"test_results_{timestamp}.json"
        
        # 添加模型信息
        test_report = {
            'model_path': self.model_path,
            'test_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'summary_stats': summary_stats
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)
        
        print(f"📁 测试结果已保存到: {output_path}")
    
    def benchmark_performance(self, num_episodes: int = 20):
        """性能基准测试"""
        print(f"\n🚀 开始性能基准测试...")
        
        # 测试不同的确定性设置
        results = {}
        
        print(f"\n测试确定性预测...")
        start_time = time.time()
        summary_deterministic = self.run_multiple_episodes(
            num_episodes=num_episodes//2, 
            max_steps=500
        )
        deterministic_time = time.time() - start_time
        results['deterministic'] = {
            'stats': summary_deterministic,
            'avg_time_per_episode': deterministic_time / (num_episodes//2)
        }
        
        print(f"\n🎲 测试随机性预测...")
        # 临时修改模型预测方式
        original_predict = self.model.predict
        
        def stochastic_predict(obs, deterministic=False):
            return original_predict(obs, deterministic=False)
        
        self.model.predict = stochastic_predict
        
        start_time = time.time()
        summary_stochastic = self.run_multiple_episodes(
            num_episodes=num_episodes//2, 
            max_steps=500
        )
        stochastic_time = time.time() - start_time
        results['stochastic'] = {
            'stats': summary_stochastic,
            'avg_time_per_episode': stochastic_time / (num_episodes//2)
        }
        
        # 恢复原始预测方法
        self.model.predict = original_predict
        
        # 比较结果
        print(f"\n📊 基准测试对比:")
        print(f"确定性预测:")
        print(f"  平均奖励: {results['deterministic']['stats']['reward_stats']['mean']:.2f}")
        print(f"  完成率: {results['deterministic']['stats']['completion_rate']:.1%}")
        print(f"  每回合耗时: {results['deterministic']['avg_time_per_episode']:.2f}s")
        
        print(f"随机性预测:")
        print(f"  平均奖励: {results['stochastic']['stats']['reward_stats']['mean']:.2f}")
        print(f"  完成率: {results['stochastic']['stats']['completion_rate']:.1%}")
        print(f"  每回合耗时: {results['stochastic']['avg_time_per_episode']:.2f}s")
        
        return results
    
    def close(self):
        """清理资源"""
        if self.env:
            self.env.close()

def main():
    parser = argparse.ArgumentParser(description='测试训练后的模型')
    parser.add_argument('model_path', help='训练好的模型路径')
    parser.add_argument('--episodes', '-e', type=int, default=10, 
                       help='测试回合数 (默认: 10)')
    parser.add_argument('--max-steps', '-s', type=int, default=1000,
                       help='每回合最大步数 (默认: 1000)')
    parser.add_argument('--render', '-r', action='store_true',
                       help='渲染游戏画面')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='运行性能基准测试')
    parser.add_argument('--output', '-o', type=str,
                       help='测试结果输出文件路径')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    # 创建测试器
    tester = ModelTester(args.model_path, render=args.render)
    
    try:
        if args.benchmark:
            # 性能基准测试
            benchmark_results = tester.benchmark_performance(num_episodes=args.episodes)
            
            # 保存基准测试结果
            if args.output:
                benchmark_output = args.output.replace('.json', '_benchmark.json')
                with open(benchmark_output, 'w') as f:
                    json.dump(benchmark_results, f, indent=2)
                print(f"📁 基准测试结果已保存到: {benchmark_output}")
        else:
            # 标准测试
            summary_stats = tester.run_multiple_episodes(
                num_episodes=args.episodes,
                max_steps=args.max_steps
            )
            
            # 打印报告
            tester.print_test_report(summary_stats)
            
            # 保存结果
            if args.output:
                tester.save_test_results(summary_stats, args.output)
            else:
                tester.save_test_results(summary_stats)
    
    finally:
        tester.close()

if __name__ == "__main__":
    main()