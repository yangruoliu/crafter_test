# 🔄 自动GitHub同步指南

本项目已配置完整的GitHub自动同步机制，确保所有代码修改都能及时同步到远程仓库。

## 🚀 使用方法

### 1. 快速同步命令

```bash
# 最简单的方式 - 使用自动生成的提交消息
./sync

# 带自定义提交消息
./sync "修复了训练稳定性问题"

# 或者使用Python脚本
python3 auto_sync_github.py
python3 auto_sync_github.py --message "你的提交消息"
```

### 2. 自动同步功能

✅ **已经配置好的自动功能**：
- 每次使用git commit后，会自动推送到GitHub
- 不需要手动执行git push
- 失败时会显示错误信息

## 📋 同步方式总览

| 方式 | 命令 | 适用场景 |
|------|------|----------|
| **快速同步** | `./sync` | 日常开发，使用自动消息 |
| **带消息同步** | `./sync "消息"` | 重要修改，需要说明 |
| **Python脚本** | `python3 auto_sync_github.py` | 调试或自定义需求 |
| **自动同步** | 自动触发 | git commit后自动执行 |

## 🔧 技术细节

### 自动同步脚本功能
- ✅ 检查git状态
- ✅ 自动添加所有更改 (git add .)
- ✅ 提交更改 (git commit)
- ✅ 推送到远程分支 (git push)
- ✅ 错误处理和状态报告

### 目标分支
所有更改都会同步到：
```
https://github.com/yangruoliu/crafter_test/tree/cursor/check-ppo-loss-normalization-for-direction-loss-c7d3
```

## 📖 使用示例

### 场景1: 修改了训练配置
```bash
# 修改improved_training_config_v4.py后
./sync "调整V4配置参数，优化探索权重"
```

### 场景2: 添加了新功能
```bash
# 创建新文件后
./sync "添加模型可视化分析工具"
```

### 场景3: 修复了bug
```bash
# 修复代码后
./sync "修复损失权重计算中的数值不稳定问题"
```

### 场景4: 日常提交
```bash
# 小修改，使用自动消息
./sync
```

## ⚠️ 注意事项

1. **提交前检查**: 确保代码能正常运行
2. **敏感信息**: 不要提交包含密码或密钥的文件
3. **大文件**: 避免提交超大的模型文件或数据集
4. **网络问题**: 如果推送失败，脚本会显示错误信息

## 🛠️ 故障排除

### 推送失败
```bash
# 查看详细错误信息
python3 auto_sync_github.py --message "测试提交"
```

### 重新设置自动同步
```bash
# 如果自动同步失效，重新设置
python3 auto_sync_github.py --setup
```

### 检查git状态
```bash
git status
git log --oneline -5
```

## 🎯 最佳实践

1. **频繁同步**: 每次重要修改后立即同步
2. **清晰消息**: 使用描述性的提交消息
3. **测试后提交**: 确保代码能正常运行再提交
4. **分阶段提交**: 将大的改动分成多个小的提交

现在您可以专注于代码开发，同步工作已经完全自动化！🚀