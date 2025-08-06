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

blur_strength: 模糊强度 

=== 预期效果 ===
- 加速目标物体的发现和导航
- 提高训练效率
- 保持与现有框架的兼容性

=== 扩展可能 ===
- 支持多目标物体
- 动态调整模糊强度
- 结合其他视觉增强技术
"""

主要代码
```
env_wrapper.py # 最后的wrapper

env_wrapper_v_a_star.py 这个wrapper代码和上面一样的，是因为在服务器上跑的时候想做对比实验看，所以随便看一个就行

crafter_blur_manual_play.py # 这里是手动check blur 效果的代码

train_with_blur.py  

test_with_blur.py  
```
上述代码train 或者 test 的过程中会有一些视频和图片的记录，视频和图片的路径可能有点乱，这里没太细改，但是问题不大，可以看

建议test在本地进行，可以看效果。


```
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
```



```
CUDA_VISIBLE_DEVICES=1 python train_with_blur.py

CUDA_VISIBLE_DEVICES=1 python test_with_blur.py
```
# 个人记录

https://www.notion.so/crafter-23c1cc31a6428091b068ef91189906b9?source=copy_link

