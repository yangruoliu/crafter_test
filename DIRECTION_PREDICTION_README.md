# Direction Prediction Auxiliary Task for PPO

这个改进为原有的PPO模型增加了方向预测的辅助任务，用于预测游戏中重点关注物体（如石头）相对于主角的方位。

## 功能特点

### 1. 方向预测网络
- 增加了9分类的方向预测网络（8个方向 + None）
- 方向包括：上、下、左、右、左上、左下、右上、右下、无目标
- 网络输出每个方向的概率分布

### 2. 辅助Loss
- 在PPO的主要损失函数基础上增加了方向预测的交叉熵损失
- 可通过 `direction_weight` 参数调整辅助损失的权重
- 训练过程中会记录方向预测的准确率

### 3. 自动标签生成
- 从游戏内存自动获取主角位置和目标物体位置
- 计算最近目标物体相对于主角的方向作为真实标签
- 训练阶段自动生成标签，测试阶段无需标签

## 文件结构

### 核心文件
- `model_with_attn.py`: 改进的PPO模型，包含方向预测功能
- `env_wrapper.py`: 新增 `DirectionLabelWrapper` 用于生成方向标签
- `train_with_direction.py`: 带方向预测的训练脚本
- `test_with_direction.py`: 带方向预测的测试脚本

### 主要类和函数

#### CustomACPolicy
```python
class CustomACPolicy(ActorCriticPolicy):
    def __init__(self, *args, num_aux_classes: int = 9, **kwargs):
        # 添加方向预测网络
        self.direction_net = nn.Linear(features_dim, num_aux_classes)
```

#### DirectionLabelWrapper
```python
class DirectionLabelWrapper(gym.Wrapper):
    def __init__(self, env, target_obj_id, target_obj_name="stone"):
        # 自动生成方向标签的环境包装器
```

#### CustomPPO
```python
class CustomPPO(PPO):
    def __init__(self, *args, direction_weight: float = 0.3, **kwargs):
        # 支持方向预测辅助损失的PPO算法
```

## 使用方法

### 1. 训练模型

```bash
python train_with_direction.py
```

训练配置：
```python
config = {
    "total_timesteps": 1000000,
    "save_dir": "./stone_with_direction",
    "init_items": ["wood_pickaxe"],
    "init_num": [1],
    "target_obj_id": 7,  # 石头的物体ID
    "target_obj_name": "stone",
    "direction_weight": 0.3  # 方向预测损失权重
}
```

### 2. 测试模型

```bash
python test_with_direction.py
```

测试会显示：
- 游戏画面
- 实时方向预测结果
- 方向预测准确率
- 平均奖励

### 3. 自定义目标物体

可以通过修改 `target_obj_id` 来关注不同的物体：
- 7: 石头 (stone)
- 8: 煤炭 (coal)
- 9: 铁矿 (iron)
- 其他物体ID可查看游戏代码

## 方向编码

方向标签使用以下编码：
- 0: 上 (Up)
- 1: 右上 (Up-Right)
- 2: 右 (Right)
- 3: 右下 (Down-Right)
- 4: 下 (Down)
- 5: 左下 (Down-Left)
- 6: 左 (Left)
- 7: 左上 (Up-Left)
- 8: 无目标 (None)

## 参数配置

### 训练参数
- `direction_weight`: 方向预测损失的权重 (默认: 0.3)
- `num_aux_classes`: 方向分类数量 (固定: 9)
- `search_radius`: 搜索目标物体的半径 (默认: 15)

### 模型参数
- `features_dim`: 特征维度 (默认: 1024)
- `learning_rate`: 学习率 (默认: 3e-4)
- `batch_size`: 批大小 (默认: 512)

## 监控指标

训练过程中会记录以下指标：
- `train/direction_loss`: 方向预测损失
- `train/direction_accuracy`: 方向预测准确率
- `train/policy_gradient_loss`: 策略梯度损失
- `train/value_loss`: 价值函数损失
- `train/entropy_loss`: 熵损失

## 注意事项

1. **环境包装顺序**: `DirectionLabelWrapper` 应该放在最后，确保能获取正确的游戏信息
2. **观测空间**: 使用字典观测空间 `{'obs': image, 'direction_label': int}`
3. **设备兼容**: 确保所有张量在相同设备上（CPU/GPU）
4. **内存使用**: 方向标签存储会略微增加内存使用

## 扩展可能

1. **多目标支持**: 可扩展为同时预测多个物体的方向
2. **距离预测**: 除方向外还可预测到目标的距离
3. **动态目标**: 根据任务动态切换关注的目标物体
4. **注意力机制**: 结合视觉注意力机制提升预测精度

## 故障排除

### 常见问题

1. **模型加载失败**: 确保模型路径正确且模型文件存在
2. **方向标签不准确**: 检查 `target_obj_id` 是否正确
3. **训练不收敛**: 尝试调整 `direction_weight` 参数
4. **内存不足**: 减少 `batch_size` 或 `n_steps`

### 调试建议

1. 使用较小的 `total_timesteps` 进行快速测试
2. 启用 `verbose=1` 查看训练日志
3. 检查方向预测准确率是否在合理范围内
4. 可视化测试结果验证方向预测是否正确 