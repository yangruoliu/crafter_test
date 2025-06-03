import re

def parse_seen_objects(info_text):

    objects_list = []
    distances_list = []
    directions_list = []

    in_see_section = False


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


def get_label(info, obj_name):

    obj_list, _, _ = parse_seen_objects(info)
    
    if obj_name in obj_list:
        return 1
    return 0

