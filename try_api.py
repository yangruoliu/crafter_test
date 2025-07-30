# # 导入OpenAI模块，用于调用OpenAI的API
# from openai import OpenAI
#
# # 使用指定的API密钥和基础URL创建OpenAI客户端实例
# client = OpenAI(
#     # api_key:这是用于身份验证的 API 密钥。您需要使用有效的 API 密钥来向 DeepSeek 的 API 进行身份验证，以调用其服务。
#     #  base_url:这是 DeepSeek API 的基础 URL。OpenAI 客户端将使用此 URL 来发送请求到 DeepSeek 的服务器。
#     api_key="sk-cfc4d64210cf4645938e367d8f023ab9", base_url='https://api.deepseek.com/v1'
# )
#
# # 调用OpenAI的聊天完成接口
# response = client.chat.completions.create(
#     model="deepseek-chat",  # 指定使用的模型名称
#     messages=[  # 定义消息列表，包含系统角色和用户角色的消息
#         {"role": "system", "content": "我司AI智能助手，有什么需要帮助的吗？"},  # 系统角色消息
#         {"role": "user", "content": "什么是fastAPI"},  # 用户角色消息
#     ],
#     stream=False  # 设置非流式响应，回答的结果一次性输出
#     # stream=True  # 设置流式响应，回答的一段一段输出
#
# )
#
# # 非流式输出
# print(response.choices[0].message.content)
# # 流式输出
# # for chunk in response:
# #     print(chunk.choices[0].delta.content, end="", flush=True)

import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量（推荐方式）
load_dotenv()

client = OpenAI(
    api_key=os.getenv("Qwen_API_KEY"),  # 从环境变量获取密钥，防止API-key的泄露
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 调用通义千问
completion = client.chat.completions.create(
    model="qwen-plus",  # 通义千问的模型名称
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "阿里云公司的简介"}
    ],
    stream=True,  # 流式输出，适合长文本响应
    extra_body={"enable_thinking": False}  # 控制思考过程
)

# 非流式输出
# print(completion.choices[0].message.content)
# 流式输出
for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)