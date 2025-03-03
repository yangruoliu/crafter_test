from ollama import chat
from config import SYS_PROMPT

def llm_chat(prompt, model="qwen2.5:7b"):

    messages = []

    sys_prompt = {}
    sys_prompt['role'] = "system"
    sys_prompt['content'] = SYS_PROMPT

    messages.append(sys_prompt)

    user_prompt = {}
    user_prompt['role'] = "user"
    user_prompt['content'] = prompt

    messages.append(user_prompt)

    response = chat(model=model, messages=messages)

    text = response["message"]["content"]

    return text


def compose_user_prompt(obs, rule_set):

    return "Here is your observation:\n\n" + obs + "\n\n Here is current rule set:\n\n" + rule_set
