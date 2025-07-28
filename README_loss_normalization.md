# 损失归一化功能说明

## 概述

本项目已集成了基于文献研究的损失归一化功能，用于解决多损失训练中不同损失函数量级差异导致的权重失效问题。

## 主要特性

### 1. 动态损失归一化
- 使用指数移动平均（EWMA）跟踪各损失的统计信息
- 自动归一化不同量级的损失函数
- 确保权重系数设置的有效性

### 2. 自适应权重调整
- 基于变异系数（Coefficient of Variation）动态调整权重
- 变异度高的损失获得更多权重，促进学习
- 自动平衡各损失的相对重要性

### 3. 详细监控和日志
- 记录移动平均值、变异系数和自适应权重
- 便于分析和调试训练过程

## 使用方法

### 1. 启用损失归一化

在创建CustomPPO时设置参数：

```python
model = CustomPPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    direction_weight=0.3,
    loss_normalization=True,  # 启用损失归一化
    norm_decay=0.99,          # EWMA衰减因子
    # ... 其他参数
)
```

### 2. 配置参数说明

- `loss_normalization`: bool, 是否启用损失归一化（默认True）
- `norm_decay`: float, EWMA衰减因子（默认0.99，值越大历史信息权重越高）

### 3. 监控训练过程

训练过程中会自动记录以下指标：

#### 损失统计信息
- `train/loss_norm/policy_loss_ma`: Policy损失移动平均
- `train/loss_norm/value_loss_ma`: Value损失移动平均  
- `train/loss_norm/entropy_loss_ma`: Entropy损失移动平均
- `train/loss_norm/direction_loss_ma`: Direction损失移动平均

#### 变异系数
- `train/loss_norm/policy_cov`: Policy损失变异系数
- `train/loss_norm/value_cov`: Value损失变异系数
- `train/loss_norm/entropy_cov`: Entropy损失变异系数
- `train/loss_norm/direction_cov`: Direction损失变异系数

#### 自适应权重
- `train/adaptive_weights/policy`: Policy损失自适应权重
- `train/adaptive_weights/value`: Value损失自适应权重
- `train/adaptive_weights/entropy`: Entropy损失自适应权重
- `train/adaptive_weights/direction`: Direction损失自适应权重

## 工作原理

### 1. 统计跟踪
```python
# 每次训练步骤更新损失统计
self._update_loss_statistics(policy_loss, value_loss, entropy_loss, direction_loss)
```

### 2. 归一化计算
```python
# 使用移动平均归一化损失
normalized_loss = current_loss / (moving_average + eps)
```

### 3. 自适应权重
```python
# 基于变异系数计算权重
coefficient_of_variation = std_dev / mean
adaptive_weight = cov / total_cov * original_weight
```

### 4. 最终损失
```python
# 组合归一化后的损失
total_loss = (adaptive_policy_weight * policy_loss_norm + 
              adaptive_entropy_weight * entropy_loss_norm + 
              adaptive_value_weight * value_loss_norm + 
              adaptive_direction_weight * direction_loss_norm)
```

## 预期效果

### 1. 权重有效性
- 原始权重设置（如direction_weight=0.3）变得有意义
- 不再被其他损失的量级"淹没"

### 2. 训练稳定性
- 各损失函数在训练中得到平衡的关注
- 减少某个损失主导训练的问题

### 3. 性能提升
- 方向检测任务真正参与训练过程
- 整体模型性能改善

## 对比分析

### 修改前
```python
# 直接加权，存在量级问题
loss = policy_loss + 0.01*entropy_loss + 0.5*value_loss + 0.3*direction_loss
# 如果direction_loss数值很小，0.3的权重实际上没有作用
```

### 修改后
```python
# 归一化后再加权，确保权重有效性
loss = w_p*policy_norm + w_e*entropy_norm + w_v*value_norm + w_d*direction_norm
# 每个损失都在相同量级上，权重设置真正发挥作用
```

## 注意事项

1. **初始化期间**: 前10个训练步骤使用原始权重，之后启用归一化
2. **参数调优**: 可能需要重新调整权重系数，因为归一化后的效果不同
3. **监控重要**: 观察自适应权重的变化，确保符合预期
4. **性能对比**: 建议进行消融实验，对比归一化前后的效果

## 测试

运行测试脚本验证功能：

```bash
python test_loss_normalization.py
```

这将展示损失归一化和自适应权重计算的工作过程。