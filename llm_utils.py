from ollama import chat
from config import SYS_PROMPT
from openai import OpenAI
import os

#sk-a82539b067ad4a64967adce8747bd530

def llm_chat(prompt, system_prompt = SYS_PROMPT, model="qwen2.5:7b"):

    messages = []

    sys_prompt = {}
    sys_prompt['role'] = "system"
    sys_prompt['content'] = system_prompt

    messages.append(sys_prompt)

    user_prompt = {}
    user_prompt['role'] = "user"
    user_prompt['content'] = prompt

    messages.append(user_prompt)

    if model == "deepseek-chat":

        #api_key = os.getenv("DEEPSEEK_API_KEY")
        api_key = "sk-a82539b067ad4a64967adce8747bd530"

        if not api_key:
            raise ValueError("No DEEPSEEK_API_KEY is defined")

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(model = model, messages = messages, stream = False)
        text = response.choices[0].message.content

    else:

        response = chat(model=model, messages=messages)
        text = response["message"]["content"]

    return text


def compose_user_prompt(obs, rule_set):

    return "Here is your observation:\n\n" + obs + "\n\n Here is current rule set:\n\n" + rule_set
