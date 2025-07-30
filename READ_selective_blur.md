# README_selective_blur.py
"""
方案一：选择性模糊使用指南

=== 核心思路 ===
通过对非目标物体进行图像模糊处理，使智能体更容易关注到重要的目标物体，
从而加速强化学习的训练过程。

=== 技术实现 ===
1. 利用游戏提供的semantic_map获取物体位置信息
2. 识别目标物体在当前视野中的位置
3. 对非目标区域应用高斯模糊，保持目标区域清晰
4. 与现有训练框架无缝集成

=== 使用步骤 ===

步骤1: 训练带选择性模糊的模型
```bash
python train_with_blur.py
```

步骤2: 测试模型效果
```bash
python test_with_blur.py
```

步骤3: 验证设置
```bash
python verify_blur_setup.py
```

=== 参数配置 ===

target_obj_id: 目标物体ID
- 3: stone (石头)
- 8: coal (煤炭) 
- 9: iron (铁矿)

blur_strength: 模糊强度 (建议15-25，必须是奇数)

=== 预期效果 ===
- 加速目标物体的发现和导航
- 提高训练效率
- 保持与现有框架的兼容性

=== 扩展可能 ===
- 支持多目标物体
- 动态调整模糊强度
- 结合其他视觉增强技术
"""

def show_usage_examples():
    """显示使用示例"""
    
    print(__doc__)
    
    print("\n=== 代码示例 ===")
    
    example_code = '''
# 基础使用示例
import gym
import env_wrapper

# 创建环境
env = gym.make("MyCrafter-v0")
env = env_wrapper.MineStoneWrapper(env)
env = env_wrapper.InitWrapper(env, ["wood_pickaxe"], [1])

# 应用选择性模糊 (针对石头)
env = env_wrapper.SelectiveBlurWrapper(
    env, 
    target_obj_id=3,           # 石头ID
    target_obj_name="stone",   # 便于调试
    blur_strength=15           # 模糊强度
)

# 正常使用
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)

# 查看处理效果
blur_info = info.get('selective_blur', {})
print(f"目标发现: {blur_info.get('target_found', False)}")
'''
    
    print(example_code)

if __name__ == "__main__":
    show_usage_examples()


'''
training_videos_20241201_143052/    # 训练视频
├── training_step_50000_*.mp4
├── training_step_100000_*.mp4
└── training_step_1000000_*.mp4

test_videos_20241201_143052/        # 测试视频
├── test_episode_1_*.mp4
├── test_episode_2_*.mp4
└── test_episode_3_*.mp4

comparison_videos_20241201_143052/  # 对比视频
├── no_blur/
│   ├── test_episode_1_*.mp4
│   └── test_episode_2_*.mp4
├── with_blur/
│   ├── test_episode_1_*.mp4
│   └── test_episode_2_*.mp4
└── side_by_side_comparison.mp4
'''

```
# 训练并录制视频
CUDA_VISIBLE_DEVICES=1 python train_with_blur.py

# 测试并录制视频
CUDA_VISIBLE_DEVICES=1 python test_with_blur.py
```
