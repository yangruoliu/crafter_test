#!/usr/bin/env python3
"""
模型行为诊断脚本
深入分析模型的具体行为模式，找出效率低下的原因
"""

import gym
import crafter
import env_wrapper
from model_with_attn import CustomPPO
import numpy as np
import time
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import json

class ModelBehaviorDiagnostic:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.action_history = []
        self.reward_history = []
        self.position_history = []
        self.direction_predictions = []
        self.behavior_patterns = defaultdict(int)
        
        self._setup_environment()
        self._load_model()
    
    def _setup_environment(self):
        """设置诊断环境"""
        self.env = gym.make("MyCrafter-v0")
        self.env = env_wrapper.MineStoneWrapper(self.env)
        self.env = env_wrapper.InitWrapper(self.env, init_items=["wood_pickaxe"], init_num=[1])
        self.env = env_wrapper.DirectionLabelWrapper(self.env, target_obj_id=7, target_obj_name="stone")
    
    def _load_model(self):
        """加载模型"""
        try:
            self.model = CustomPPO.load(self.model_path, env=self.env)
            print(f"✅ 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def analyze_single_episode(self, max_steps=300, verbose=True):
        """详细分析单个回合的行为"""
        obs = self.env.reset()
        done = False
        step = 0
        total_reward = 0
        
        episode_data = {
            'actions': [],
            'rewards': [],
            'positions': [],
            'observations': [],
            'step_analysis': []
        }
        
        if verbose:
            print(f"\n🔍 开始详细行为分析...")
        
        while not done and step < max_steps:
            # 记录当前状态
            current_position = self._extract_agent_position(obs)
            
            # 预测动作
            action, _states = self.model.predict(obs, deterministic=True)
            
            # 执行动作
            new_obs, reward, done, info = self.env.step(action)
            
            # 记录数据
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['positions'].append(current_position)
            episode_data['observations'].append(obs.copy())
            
            # 分析这一步
            step_info = self._analyze_step(step, action, reward, current_position, obs, new_obs)
            episode_data['step_analysis'].append(step_info)
            
            total_reward += reward
            step += 1
            obs = new_obs
            
            if verbose and step % 50 == 0:
                print(f"   步数: {step}, 累计奖励: {total_reward:.2f}, 当前位置: {current_position}")
        
        # 回合结束分析
        episode_summary = self._summarize_episode(episode_data, total_reward, step, done)
        
        if verbose:
            self._print_episode_analysis(episode_summary)
        
        return episode_data, episode_summary
    
    def _extract_agent_position(self, obs):
        """从观察中提取智能体位置 (简化版本)"""
        # 这是一个简化的位置提取，实际可能需要根据环境具体实现
        # 这里返回观察的某些特征作为位置的近似
        return (obs.mean(), obs.std())  # 使用观察的统计特征作为位置代理
    
    def _analyze_step(self, step, action, reward, position, obs, new_obs):
        """分析单步行为"""
        return {
            'step': step,
            'action': action,
            'reward': reward,
            'position': position,
            'obs_change': np.sum(np.abs(new_obs - obs)),  # 观察变化程度
            'reward_type': 'positive' if reward > 0 else 'negative' if reward < 0 else 'zero'
        }
    
    def _summarize_episode(self, episode_data, total_reward, total_steps, completed):
        """总结回合表现"""
        actions = episode_data['actions']
        rewards = episode_data['rewards']
        
        # 动作分析
        action_counts = Counter(actions)
        most_common_actions = action_counts.most_common(3)
        
        # 奖励分析
        positive_rewards = [r for r in rewards if r > 0]
        negative_rewards = [r for r in rewards if r < 0]
        zero_rewards = [r for r in rewards if r == 0]
        
        # 效率分析
        reward_per_step = total_reward / total_steps if total_steps > 0 else 0
        
        # 行为模式分析
        behavior_patterns = self._identify_behavior_patterns(episode_data)
        
        return {
            'total_reward': total_reward,
            'total_steps': total_steps,
            'completed': completed,
            'reward_per_step': reward_per_step,
            'action_distribution': dict(action_counts),
            'most_common_actions': most_common_actions,
            'reward_breakdown': {
                'positive': len(positive_rewards),
                'negative': len(negative_rewards),
                'zero': len(zero_rewards),
                'pos_sum': sum(positive_rewards),
                'neg_sum': sum(negative_rewards)
            },
            'behavior_patterns': behavior_patterns,
            'efficiency_score': self._calculate_efficiency_score(total_reward, total_steps, completed)
        }
    
    def _identify_behavior_patterns(self, episode_data):
        """识别行为模式"""
        actions = episode_data['actions']
        patterns = {}
        
        # 检查重复动作
        consecutive_same = 0
        max_consecutive = 0
        for i in range(1, len(actions)):
            if actions[i] == actions[i-1]:
                consecutive_same += 1
                max_consecutive = max(max_consecutive, consecutive_same)
            else:
                consecutive_same = 0
        
        patterns['max_consecutive_same_action'] = max_consecutive
        
        # 检查循环行为 (简化版本)
        recent_actions = actions[-20:] if len(actions) >= 20 else actions
        patterns['recent_action_diversity'] = len(set(recent_actions)) / len(recent_actions) if recent_actions else 0
        
        # 检查探索 vs 利用
        action_entropy = self._calculate_action_entropy(actions)
        patterns['action_entropy'] = action_entropy
        
        return patterns
    
    def _calculate_action_entropy(self, actions):
        """计算动作熵，衡量探索程度"""
        if not actions:
            return 0
        action_counts = Counter(actions)
        total = len(actions)
        entropy = 0
        for count in action_counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        return entropy
    
    def _calculate_efficiency_score(self, reward, steps, completed):
        """计算效率得分"""
        if not completed:
            return 0.0
        
        # 基础完成分数
        completion_score = 10.0
        
        # 奖励效率 (奖励越高越好)
        reward_score = max(0, reward) * 0.1
        
        # 步数效率 (步数越少越好)
        step_penalty = max(0, steps - 100) * 0.01
        
        efficiency = completion_score + reward_score - step_penalty
        return max(0, efficiency)
    
    def _print_episode_analysis(self, summary):
        """打印回合分析结果"""
        print(f"\n" + "="*60)
        print(f"🎯 回合行为分析报告")
        print(f"="*60)
        
        print(f"📊 基本指标:")
        print(f"   总奖励: {summary['total_reward']:.2f}")
        print(f"   总步数: {summary['total_steps']}")
        print(f"   完成状态: {'✅' if summary['completed'] else '❌'}")
        print(f"   效率得分: {summary['efficiency_score']:.2f}")
        print(f"   每步奖励: {summary['reward_per_step']:.4f}")
        
        print(f"\n🎮 动作分析:")
        print(f"   最常用动作: {summary['most_common_actions']}")
        print(f"   动作多样性: {len(summary['action_distribution'])} 种不同动作")
        
        print(f"\n🏆 奖励分析:")
        rb = summary['reward_breakdown']
        print(f"   正奖励步数: {rb['positive']} (总计: {rb['pos_sum']:.2f})")
        print(f"   负奖励步数: {rb['negative']} (总计: {rb['neg_sum']:.2f})")
        print(f"   零奖励步数: {rb['zero']}")
        
        print(f"\n🔍 行为模式:")
        bp = summary['behavior_patterns']
        print(f"   最大连续相同动作: {bp['max_consecutive_same_action']}")
        print(f"   近期动作多样性: {bp['recent_action_diversity']:.3f}")
        print(f"   动作熵: {bp['action_entropy']:.3f}")
        
        # 问题诊断
        print(f"\n⚠️ 问题诊断:")
        issues = []
        
        if summary['reward_per_step'] < -0.03:
            issues.append("每步奖励过低，策略效率需要改进")
        
        if bp['max_consecutive_same_action'] > 10:
            issues.append(f"存在重复行为 (连续{bp['max_consecutive_same_action']}次相同动作)")
        
        if bp['action_entropy'] < 1.0:
            issues.append("动作多样性不足，可能陷入固定模式")
        
        if rb['negative'] > rb['positive'] * 2:
            issues.append("负奖励步数过多，行为选择有问题")
        
        if not issues:
            issues.append("未发现明显问题，可能需要更多训练时间")
        
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print(f"="*60)

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python diagnose_model_behavior.py <模型路径>")
        print("示例: python diagnose_model_behavior.py ./stone_with_direction_v4_20240804_070000")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print(f"🔍 开始诊断模型行为: {model_path}")
    
    diagnostic = ModelBehaviorDiagnostic(model_path)
    
    # 分析3个回合
    for episode in range(3):
        print(f"\n{'='*20} 回合 {episode + 1} {'='*20}")
        episode_data, summary = diagnostic.analyze_single_episode(verbose=True)
        
        # 保存详细数据 (可选)
        if episode == 0:  # 只保存第一回合的详细数据
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            with open(f"behavior_analysis_{timestamp}.json", 'w') as f:
                # 转换numpy数组为list以便JSON序列化
                serializable_data = {
                    'summary': summary,
                    'action_sequence': episode_data['actions'],
                    'reward_sequence': episode_data['rewards']
                }
                json.dump(serializable_data, f, indent=2)
            print(f"📁 详细数据已保存到 behavior_analysis_{timestamp}.json")

if __name__ == "__main__":
    main()