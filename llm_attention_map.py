import re
from sre_compile import dis
import numpy as np
import matplotlib.pyplot as plt
import torch

import llm_utils
import llm_prompt
import json
from PIL import Image
import cv2


KEY1 = "object" 
KEY2 = "direction"
KEY3 = "distance"
KEY4 = "priority"

pattern_string = rf"""
( 
    "{KEY1}"\s*:\s*(?P<value1>.*?)
    \s*,\s*
    "{KEY2}"\s*:\s*(?P<value2>.*?)
    \s*,\s*
    "{KEY3}"\s*:\s*(?P<value3>.*?)
    \s*,\s*
    "{KEY4}"\s*:\s*(?P<value4>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)
)
"""

def visualize_matrix(matrix_data):
    if not isinstance(matrix_data, np.ndarray):
        matrix_data = np.array(matrix_data, dtype=float)

    if matrix_data.shape != (7, 9):
        raise ValueError("输入矩阵必须是 7x9 的维度。")

    if np.any(matrix_data < 0) or np.any(matrix_data > 1):
        raise ValueError("矩阵中的元素值必须在0到1的区间内。")

    grid_to_display = np.zeros((9, 9))

    grid_to_display[0:7, :] = matrix_data

    fig = plt.figure(figsize=(2, 2))  # 可以调整尺寸，保持宽高比为1:1
    ax = fig.add_axes([0, 0, 1, 1])   # 使图像内容填满整个figure区域

    ax.imshow(grid_to_display, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')

    ax.axis('off')

    fig.savefig("mask", dpi=32)

    print(np.array(fig))

    plt.show()
    return fig


def direction_offset(direction):

    if "north-west" in direction:
        return (-0.5, -0.5)
    if "north-east" in direction:
        return (-0.5, 0.5)
    if "south-west" in direction:
        return (0.5, -0.5)
    if "south-east" in direction:
        return (0.5, 0.5)
    if "west" in direction:
        return (0, -1)
    if "east" in direction:
        return (0, 1)
    if "north" in direction:
        return (-1, 0)
    if "south" in direction:
        return (1, 0)
    return (0, 0)

def parse_llm_text(text):

    attn_map = np.zeros((7, 9), dtype=np.float32)
    # pattern = re.compile(r'(\w+) at \((\d+),(\d+)\): ([0-9.]+)')
    regex = re.compile(pattern_string, re.VERBOSE | re.DOTALL)
    matches = regex.finditer(text)
    direction = []
    distance = []
    priority = []
    for match in matches:
        direction.append(match.group('value2'))
        distance.append(match.group('value3'))
        priority.append(match.group('value4'))
    attn_map[3,4] = 0.9
    try:
        for i in range(len(direction)):
            x = 3
            y = 4
            offset = direction_offset(direction[i])
            dist = float(distance[i])
            x += dist * offset[0]
            y += dist * offset[1]
            x = int(x)
            y = int(y)
            print(x, " ", y)
            if 0 <= x < 7 and 0 <= y < 9:
                attn_map[x, y] = float(priority[i])

    except Exception as e:
        print(e)

    return attn_map

def build_attn_map(directions_list, distances_list):
    

    attn_map = np.zeros((9, 9), dtype=np.float32)
    attn_map[3, 4] = 0.5
    try:
        for i in range(len(distances_list)):
            x = 3
            y = 4
            offset = direction_offset(directions_list[i])
            dist = float(distances_list[i])
            x += dist * offset[0]
            y += dist * offset[1]
            x = int(x)
            y = int(y)
            if 0 <= x < 7 and 0 <= y < 9:
                attn_map[x, y] = 1

    except Exception as e:
        print(e)
    return attn_map

def blur(state_img, attn_map):

    result = state_img.copy()

    for i in range(attn_map.shape[0]):
        for j in range(attn_map.shape[1]):
            y1 = i * 7
            y2 = (i + 1) * 7
            x1 = j * 7
            x2 = (j + 1) * 7

            region = state_img[y1:y2, x1:x2, :]

            attention_value = attn_map[i, j]
            if attention_value == 0:
                blurred_region = cv2.GaussianBlur(region, (7, 7), 0)
                result[y1:y2, x1:x2, :] = blurred_region
    np.save("MAN",  result)
    return result

def convert_to_rgb_image_pil(arr_9x9):

    assert arr_9x9.shape == (9, 9), "Input must be of shape (9, 9)"
    assert np.all((arr_9x9 >= 0) & (arr_9x9 <= 1)), "Values must be in [0, 1]"

    # Convert to 0–255 and invert (so 0 is white and 1 is black)
    grayscale = (1 - arr_9x9) * 255
    grayscale = grayscale.astype(np.uint8)

    # Create a grayscale PIL image
    img = Image.fromarray(grayscale, mode='L')

    # Resize to 64x64
    img_resized = img.resize((64, 64), resample=Image.NEAREST)

    # Convert to RGB
    img_rgb = img_resized.convert('RGB')

    # Convert back to NumPy array
    return np.array(img_rgb)


def obs_to_attn_map(obs):

    objects_list, distances_list, directions_list = parse_seen_objects(obs)

    attn = build_attn_map(directions_list, distances_list)

    return attn


def query_llm(prompt):

    try:
        response = llm_utils.llm_chat(prompt, system_prompt="", model="deepseek-chat")
        print("LLM Response:\n", response)
    except Exception as e:
        print("LLM query failed, returning empty attention.", e)
        return np.zeros((7, 9), dtype=np.float32)

    return parse_llm_text(response)


def parse_seen_objects(info_text):

    objects_list = []
    distances_list = []
    directions_list = []

    in_see_section = False
    # 正则表达式用于匹配 "- <物体名称> <距离> steps to your <方位>" 格式的行
    # (.+?)     - 捕获组1: 物体名称 (非贪婪匹配，允许名称中包含空格)
    # (\d+)      - 捕获组2: 距离 (一个或多个数字)
    # steps to your - 固定文本
    # ([\w-]+) - 捕获组3: 方位 (一个或多个字母、数字、下划线或连字符，例如 north, north-west)
    pattern = re.compile(r"(.+?) (\d+) steps to your ([\w-]+)")

    for line in info_text.splitlines():
        stripped_line = line.strip()

        if stripped_line == "You see:":
            in_see_section = True
            continue

        if in_see_section:
            if stripped_line.startswith("- "):
                # 移除行首的 "- "
                content = stripped_line[2:]
                match = pattern.match(content)
                if match:
                    object_name = match.group(1)
                    distance = int(match.group(2))
                    direction = match.group(3)

                    objects_list.append(object_name)
                    distances_list.append(distance)
                    directions_list.append(direction)
                # else: # 如果行以"- "开头但不匹配完整模式，可以选择忽略或记录错误
            elif not stripped_line: # 空行表示 "You see:" 区域的结束
                in_see_section = False
                break # 通常物体列表结束后会有空行，可以停止解析
            else: # 如果是不以"- "开头的非空行，也表示 "You see:" 区域的结束
                in_see_section = False
                break

    if len(objects_list) != len(distances_list) or len(distances_list) != len(directions_list):
        return [], [], []
    
    return objects_list, distances_list, directions_list


if __name__ == "__main__":

    rules = open("rules.txt", 'r').read()
    goal = "mine iron"
    obs = "You took action do.\n\n\n\nYou see:\n- grass 2 steps to your south\n- stone 1 steps to your west\n- path 1 steps to your north\n- tree 4 steps to your south-east\n- iron 5 steps to your north-west\n\nYou face path at your front.\n\nYour status:\n- health: 7/9\n- food: 5/9\n- drink: 4/9\n- energy: 9/9\n\nYou have nothing in your inventory."

    attn = obs_to_attn_map(obs)
    print(attn)
    state_img = np.load("gaga.npy")

    blurred_img = blur(state_img, attn)
    img = Image.fromarray(blurred_img)
    img.show()
    # img.save("KOBE.png")

    # img = convert_to_rgb_image_pil(attn)
    # img = Image.fromarray(img)
    # img.save("mask.png")
    # img.show()

    # prompt = llm_prompt.compose_llm_attention_prompt(rules, obs, goal)

    # attn = query_llm(prompt)
    # visualize_attention(attn)
    # visualize_matrix(attn)

    # attn_tensor = torch.tensor(attn).unsqueeze(0)  # shape: (1, 7, 9)
    # print("Attention Tensor Shape:", attn_tensor.shape)

