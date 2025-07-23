## 项目说明

目前仍然处于探索阶段，所以代码封装以及文件命名没有十分规范。

### 主要文件

1. `model.py`

主要包含了PPO模型的定义

2. `train.py`

主要包含训练脚本

3. `test.py`

测试单个模型的脚本，可以在config字典中设置`render=True`可视化显示测试过程（远程连接服务器时可视化可能得设置端口转发）

4. `human_demonstrations.py`

人手动玩游戏，让LLM总结规则

5. `llm_agent.py`

我们目前的方法，整合所有好训练的子模型，让LLM根据规则调用模型，效果比baseline要好很多

6. `planning.py`

LLM根据规则做出任务规划，定义一系列子任务（尚不完善）

7. `learn_sub_models`

根据规划出的一系列子任务进行训练

8. `env_wrapper.py`

定义了一些环境wrapper，主要用于reward shaping和自定义环境，比较常用的：

* `InitWrapper`: 用于对装备栏初始化，如：

```python
env = InitWrapper(env, ["stone_pickaxe", "wood"], [1, 2])
```
表示初始环境装备栏中加入一个木头镐和两个木头，可以在`train.py`以及`test.py`的config字典中进行设置

9. `info.txt`

环境step后返回的内存信息的一个示例

10. 其它

包含一些环境以及测试的代码，以及不太成功的尝试，可暂时忽略

### 注意事项

1. 本项目在原始的crafter环境中做了一些修改，所以与通过`pip`安装的crafter环境可能会不兼容，建议新建一个环境，然后安装相关依赖：

```bash
pip install -r requirements.txt
```
2. 本项目使用的线上模型是deepseek671B，请在运行前设置环境变量`DEEPSEEK_API_KEY`为您的api_key

验证方法：

```bash
echo $DEEPSEEK_API_KEY
```
如果输出为您的api_key则代表环境变量成功设置
# crafter_test
