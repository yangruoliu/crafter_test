# 🧪 训练后模型测试指南

本指南提供了完整的模型测试方案，帮助您评估训练后模型的性能。

## 📁 测试脚本概览

### 1. `test_trained_model.py` - 完整测试套件
功能最全面的测试脚本，支持：
- 详细的性能分析
- 多回合批量测试
- 基准性能对比
- 方向预测准确率测试
- 结果导出和可视化

### 2. `quick_test_model.py` - 快速测试
简化的测试脚本，适合：
- 快速验证模型是否正常工作
- 简单的性能评估
- 开发过程中的快速检查

## 🚀 快速开始

### 方法1: 快速测试 (推荐新手)

```bash
# 基本测试 (5回合)
python quick_test_model.py ./your_model_path

# 带渲染的测试 (观看AI玩游戏)
python quick_test_model.py ./your_model_path 3 true

# 更多回合的测试
python quick_test_model.py ./your_model_path 10 false
```

### 方法2: 完整测试

```bash
# 标准测试 (10回合)
python test_trained_model.py ./your_model_path

# 带渲染的详细测试
python test_trained_model.py ./your_model_path --render --episodes 5

# 性能基准测试
python test_trained_model.py ./your_model_path --benchmark --episodes 20

# 长时间测试并保存结果
python test_trained_model.py ./your_model_path --episodes 50 --max-steps 1000 --output results.json
```

## 📊 测试指标说明

### 🏆 核心性能指标

| 指标 | 说明 | 理想范围 | 问题诊断 |
|------|------|----------|----------|
| **平均奖励** | 每回合获得的平均奖励 | > 20 | < 0 表示策略很差 |
| **完成率** | 成功完成任务的回合比例 | > 60% | < 20% 表示任务学习失败 |
| **平均步数** | 每回合的平均步数 | 150-300 | > 500 可能陷入循环 |
| **每步奖励** | 效率指标 | > 0.1 | < 0 表示行为随机 |

### 🎯 辅助指标

| 指标 | 说明 | 理想范围 |
|------|------|----------|
| **方向准确率** | 方向预测的准确性 | > 40% |
| **回合耗时** | 计算效率 | < 10s/回合 |
| **奖励标准差** | 性能稳定性 | 越小越好 |

## 🎮 常见测试场景

### 场景1: 训练过程中的检查点测试
```bash
# 每训练1000步测试一次
python quick_test_model.py ./checkpoints/model_1000_steps 3
python quick_test_model.py ./checkpoints/model_2000_steps 3
python quick_test_model.py ./checkpoints/model_3000_steps 3
```

### 场景2: 最终模型全面评估
```bash
# 完整的性能评估
python test_trained_model.py ./final_model --episodes 30 --output final_results.json

# 基准测试对比
python test_trained_model.py ./final_model --benchmark --episodes 40
```

### 场景3: 多个模型对比
```bash
# 测试不同版本的模型
python test_trained_model.py ./model_v1 --episodes 20 --output v1_results.json
python test_trained_model.py ./model_v2 --episodes 20 --output v2_results.json
python test_trained_model.py ./model_v3 --episodes 20 --output v3_results.json
```

### 场景4: 调试模式 (观察AI行为)
```bash
# 观察AI的具体行为
python test_trained_model.py ./your_model --render --episodes 3 --max-steps 200
```

## 📈 性能评级标准

### 🟢 优秀 (A+)
- 平均奖励 > 50
- 完成率 > 80%
- 每步奖励 > 0.2
- 行为稳定且高效

### 🟡 良好 (B+)
- 平均奖励 > 20
- 完成率 > 60%
- 每步奖励 > 0.1
- 基本掌握任务要领

### 🟠 一般 (C+)
- 平均奖励 > 0
- 完成率 > 40%
- 每步奖励 > 0.05
- 部分理解任务，但效率低

### 🔴 较差 (D)
- 平均奖励 > -20
- 完成率 > 10%
- 有一定学习成果，但还需改进

### ⚫ 失败 (F)
- 平均奖励 <= -20
- 完成率 <= 10%
- 基本没有学到有效策略

## 🔍 问题诊断指南

### 问题1: 平均奖励为负
**可能原因:**
- 模型训练不充分
- 超参数设置问题
- 环境奖励设计有问题

**解决方案:**
- 继续训练更多步数
- 调整学习率和其他超参数
- 检查奖励函数设计

### 问题2: 完成率很低但奖励不错
**可能原因:**
- 模型学会了获得中间奖励，但没学会完成最终任务
- 探索不足，陷入局部最优

**解决方案:**
- 增加探索权重 (entropy coefficient)
- 调整奖励结构，增加完成任务的奖励

### 问题3: 回合步数过多
**可能原因:**
- 模型陷入重复行为
- 没有学会有效的任务完成策略

**解决方案:**
- 增加步数惩罚
- 检查动作空间和状态表示
- 考虑增加curriculum learning

### 问题4: 方向准确率低
**可能原因:**
- 方向损失权重太低
- 方向预测网络训练不充分

**解决方案:**
- 增加 `direction_weight` 参数
- 检查方向标签的正确性
- 确保损失归一化正常工作

## 💡 高级测试技巧

### 1. 长期稳定性测试
```bash
# 测试100回合，检查长期表现
python test_trained_model.py ./model --episodes 100 --output long_term_test.json
```

### 2. 不同难度测试
修改测试脚本中的环境配置，测试模型在不同初始条件下的表现。

### 3. 对抗性测试
在测试环境中添加随机干扰，测试模型的鲁棒性。

### 4. 可视化分析
使用保存的结果文件，创建性能曲线和统计图表。

## 📋 测试清单

使用V4模型时，重点检查以下指标：

- [ ] 平均奖励是否 > 0 (基本要求)
- [ ] 完成率是否 > 20% (任务理解)
- [ ] 方向准确率是否 > 30% (方向学习)
- [ ] 步数是否合理 (< 400)
- [ ] 是否还有明显的探索行为
- [ ] 权重分布是否平衡 (如果可以获取训练日志)

## 🚨 注意事项

1. **环境一致性**: 确保测试环境配置与训练时完全一致
2. **随机种子**: 多次测试以获得可靠的统计结果
3. **硬件影响**: GPU/CPU的差异可能影响模型行为
4. **版本兼容**: 确保使用相同版本的依赖库

## 📞 故障排除

如果测试脚本运行出错：

1. **检查依赖**: 确保所有必要的包都已安装
2. **路径问题**: 验证模型文件路径是否正确
3. **内存不足**: 尝试减少测试回合数
4. **环境问题**: 确认Crafter环境正确安装

通过这些测试方法，您可以全面评估模型的性能，发现问题并指导进一步的改进！🎯