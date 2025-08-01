from PIL.Image import Palette
from config import SYS_PROMPT
import llm_utils
import gym
import numpy as np
import llm_prompt
import random
from llm_attention_map import parse_seen_objects, build_attn_map, convert_to_rgb_image_pil
import matplotlib.pyplot as plt
from gym import spaces

import cv2

def face_at(obs):

    try:
        return obs.split()[obs.split().index("face") + 1]
    except ValueError as _:
        pass
    return ""


class LabelGeneratingWrapper(gym.Wrapper):
    """
    This wrapper calls an external function to get a label for each step
    and packages the observation and label into a Dict observation space.
    """
    def __init__(self, env: gym.Env, get_label_func, target_obj, num_aux_classes: int):
        super().__init__(env)
        self.get_label_func = get_label_func
        self.target_obj = target_obj

        # Define the new observation space as a dictionary
        self.observation_space = spaces.Dict({
            'obs': env.observation_space,
            'label': spaces.Discrete(num_aux_classes)
        })

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        label = 0
        dict_obs = {'obs': obs, 'label': label}
        return dict_obs

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        label = self.get_label_func(info['obs'], self.target_obj)
        dict_obs = {'obs': obs, 'label': label}
        return dict_obs, reward, terminated, info


class myFrameStack(gym.Wrapper):

    def __init__(self, env, stack_size=2):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = []
        
        original_shape = env.observation_space.shape  # (64, 64, 3)
        print(original_shape)
        new_shape = (original_shape[0], original_shape[1], original_shape[2]*stack_size)  # (64, 64, 3*stack_size)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.frames = [obs] * self.stack_size 
        return np.concatenate(self.frames, axis=-1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        stacked_obs = np.concatenate(self.frames, axis=-1)
        return stacked_obs, reward, done, info

class AttentionMapWrapper(gym.Wrapper):

    def __init__(self, env, obj_list, stack_size=2):
        super().__init__(env)
        self.stack_size = stack_size
        self.obj_list = obj_list
        
        original_shape = env.observation_space.shape  # (64, 64, 3)
        new_shape = (original_shape[0], original_shape[1], original_shape[2]*stack_size)  # (64, 64, 3*stack_size)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs = self.env.reset()
        white_img = np.full((64, 64, 3), 255, dtype=np.uint8)
        return np.concatenate([obs, white_img], axis=-1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        objects_list, distances_list, directions_list = parse_seen_objects(info['obs'])
        
        indices = [i for i, obj in enumerate(objects_list) if obj in self.obj_list]

        distances_list = [distances_list[i] for i in indices]
        directions_list = [directions_list[i] for i in indices]

        attn = build_attn_map(directions_list, distances_list)

        attn = convert_to_rgb_image_pil(attn)
            
        obs_with_attn = np.concatenate([obs, attn], axis=-1)

        # plt.imshow(np.hstack((obs, attn)))
        # plt.axis('off')
        # plt.pause(1)
        # plt.close()

        return obs_with_attn, reward, done, info


class LLMWrapper(gym.Wrapper):

    def __init__(self, env, model="qwen2.5:7b"):
        super().__init__(env)
        self.rule_set = "{}"
        self.cur_step = 0
        self.model = model

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        if reward >= 1 or reward <= -1:
            user_prompt = llm_utils.compose_user_prompt(info["obs"] + info['history'], self.rule_set)
            self.rule_set = llm_utils.llm_chat(user_prompt, model=self.model)
            print(self.rule_set)
        
        self.cur_step += 1

        return obs, reward, done, info

class LLMSubtaskWrapper(gym.Wrapper):

    def __init__(self, env, current_goal, model="qwen2.5:7b"):
        super().__init__(env)
        self.rule_set = "{}"
        self.cur_step = 0
        self.model = model
        self.current_goal = current_goal

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        if reward >= 1 or reward <= -1:
            user_prompt = llm_utils.compose_user_prompt(info["obs"] + info["history"], self.rule_set)
            user_prompt += "\n\nHere is the current_goal: " + self.current_goal
            self.rule_set = llm_utils.llm_chat(user_prompt, system_prompt=llm_prompt.EXPLORATION_PROMPT, model=self.model)
            # if "deepseek" in self.model:
            #     index = self.rule_set.find("</think>")
            #     self.rule_set = self.rule_set[index+8:]
            print(self.rule_set)
        
        self.cur_step += 1

        return obs, reward, done, info

class NavigationWrapper(gym.Wrapper):

    def __init__(self, env, obj_index):
        super().__init__(env)
        self.target_obj = obj_index

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        # reward = 0
        player_pos = info['player_pos']

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)
        
        for i in range(left_index, right_index, 1):
            for j in range(up_index, down_index, 1):
                if (info['semantic'][i][j] == self.target_obj):
                    reward = 1000
                    done = True
                    return obs, reward, done, info
        
        return obs, reward, done, info



class MovementWrapper(gym.Wrapper):

    def __init__(self, env, generate_rule=True):
        super().__init__(env)
        self.prev_pos = np.array([32, 32])

    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # move = ["move_up", "move_down", "move_left", "move_right"]
        # if info["action"] in move:
        #     reward += 0.1

        play_pos = info["player_pos"]
        if not np.array_equal(play_pos, self.prev_pos):
            reward += 0.1
        else:
            reward -= 0.1

        self.prev_pos = play_pos

        return obs, reward, done, info

class DrinkWaterWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_drink = 10

    def reset(self, **kwargs):
        self.prev_drink = 10
        return self.env.reset()

    def step(self, action):
        
        obs, reward, done, info = self.env.step(action)

        reward = 0
        if info["inventory"]["health"] == 0:
            reward -= 1000

        if info["inventory"]["drink"] > self.prev_drink:
            reward += 1000
            done = True
        self.prev_drink = info["inventory"]["drink"]

        return obs, reward, done, info

class MineStoneWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_stone = 0
        self.prev_pos = np.array([32, 32])
    
    def reset(self, **kwargs):
        self.prev_stone = 0
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        play_pos = info["player_pos"]
        if np.array_equal(play_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = play_pos

        num_stone = info["inventory"]["stone"]
        if num_stone > self.prev_stone:
            reward += 1000
            done = True
        self.prev_stone = num_stone

        return obs, reward, done, info

class WoodWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_wood = 0
    
    def reset(self, **kwargs):
        self.prev_wood = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        reward = 0
        if info["inventory"]["health"] == 0:
            reward -= 1000

        num_wood = info["inventory"]["wood"]
        if num_wood > self.prev_wood:
            reward += 1000
            done = True
        self.prev_wood = num_wood

        return obs, reward, done, info

class WoodPickaxeWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_wood_pickaxe = 0
        
    
    def reset(self, **kwargs):
        self.prev_wood_pickaxe = 0
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        play_pos = info["player_pos"]
        if np.array_equal(play_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = play_pos

        num_wood_pickaxe = info["inventory"]["wood_pickaxe"]
        if num_wood_pickaxe > self.prev_wood_pickaxe:
            reward += 1000
            # done = True

        self.prev_wood_pickaxe = num_wood_pickaxe

        return obs, reward, done, info

class StoneSwordWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_stone_pickaxe = 0
        self.prev_stone_sword = 0
    
    def reset(self, **kwargs):
        self.prev_stone_pickaxe = 0
        self.prev_stone_sword = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # reward = 0
        # if info["inventory"]["health"] == 0:
        #     reward -= 1000

        num_stone_pickaxe = info["inventory"]["stone_pickaxe"]
        num_stone_sword = info["inventory"]["stone_sword"]
        if num_stone_pickaxe > self.prev_stone_pickaxe and num_stone_pickaxe == 1:
            reward += 1000
            # done = True
        # if num_stone_sword > self.prev_stone_sword and num_stone_sword == 1:
        #     reward += 1000
        # if num_stone_pickaxe and num_stone_pickaxe:
        #     done = True

        self.prev_stone_pickaxe = num_stone_pickaxe
        self.num_stone_sword = num_stone_sword

        return obs, reward, done, info

class InitWrapper(gym.Wrapper):

    def __init__(self, env, init_items=[], init_num=[], init_center="mid"):
        super().__init__(env)
        self.init_items = init_items
        self.init_num = init_num
        self.init_center = init_center

    def reset(self, **kwargs):

        self.env.reset_aux(init_items=self.init_items, init_num=self.init_num, init_center=self.init_center)
        return self.env.reset()

class MineCoalWrapper(gym.Wrapper):

    def __init__(self, env, navigation_model):
        super().__init__(env)
        self.model = navigation_model
        self.prev_coal = 0

    def reset(self, **kwargs):

        self.prev_coal = 0
        
        obs = self.env.reset(**kwargs)

        valid = False

        for _ in range(100):

            if not valid:

                action, _  = self.model.predict(obs, deterministic=True)
                obs, _, _, info = self.env.step(action)

                player_pos = info['player_pos']

                left_index = max(0, player_pos[0] - 4)
                right_index = min(64, player_pos[0] + 4)
                up_index = max(0, player_pos[1] - 3)
                down_index = min(64, player_pos[1] + 3)
                
                for i in range(left_index, right_index, 1):
                    if not valid:
                        for j in range(up_index, down_index, 1):
                            if (info['semantic'][i][j] == 8):
                                valid = True
                                break
        return obs
    
    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # reward = 0
        num_coal = info["inventory"]["coal"]

        if num_coal > self.prev_coal:
            reward += 1000
            done = True
        self.prev_coal = num_coal

        return obs, reward, done, info

class FurnaceWrapper(gym.Wrapper):

    def __init__(self, env):
        self.prev_pos = np.array([32, 32])
        super().__init__(env)
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
 
        play_pos = info["player_pos"]
        if np.array_equal(play_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = play_pos

        if face_at(info["obs"]) == "furnace":
            reward += 1000
            done = True

        return obs, reward, done, info

class MineIronWrapper(gym.Wrapper):

    def __init__(self, env, navigation_model):
        super().__init__(env)
        self.model = navigation_model
        self.prev_iron = 0

    def reset(self, **kwargs):

        self.prev_iron = 0
        
        obs = self.env.reset(**kwargs)

        valid = False

        for _ in range(100):

            if not valid:

                action, _  = self.model.predict(obs, deterministic=True)
                obs, _, _, info = self.env.step(action)

                player_pos = info['player_pos']

                left_index = max(0, player_pos[0] - 4)
                right_index = min(64, player_pos[0] + 4)
                up_index = max(0, player_pos[1] - 3)
                down_index = min(64, player_pos[1] + 3)
                
                for i in range(left_index, right_index, 1):
                    if not valid:
                        for j in range(up_index, down_index, 1):
                            if (info['semantic'][i][j] == 9):
                                valid = True
                                break
        return obs
    
    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # reward = 0
        num_iron = info["inventory"]["iron"]

        if num_iron > self.prev_iron:
            reward += 1000
            done = True
        self.prev_iron = num_iron

        return obs, reward, done, info


class MineCoalWrapper2(gym.Wrapper):

    def __init__(self, env, obj_index=8):
        super().__init__(env)
        self.target_obj = obj_index
        self.prev_coal = 0
        self.find_coal = False
        self.prev_pos = np.array([32, 32])

    def reset(self, **kwargs):
        self.prev_coal = 0
        self.find_coal = False
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        # reward = 0
        player_pos = info['player_pos']

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)
        
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos

        if not self.find_coal:

            for i in range(left_index, right_index, 1):
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 100
                        self.find_coal = True
                        break
        
        num_coal = info["inventory"]["coal"]
        if num_coal > self.prev_coal:
            reward += 10000
            done = True
        self.prev_coal = num_coal
        
        return obs, reward, done, info

class MineIronWrapper2(gym.Wrapper):

    def __init__(self, env, obj_index=9):
        super().__init__(env)
        self.target_obj = obj_index
        self.prev_iron = 0
        self.find_iron = False
        self.prev_pos = np.array([32, 32])

    def reset(self, **kwargs):
        self.prev_iron = 0
        self.find_iron = False
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        # reward = 0
        player_pos = info['player_pos']

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)
        
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos

        if not self.find_iron:

            for i in range(left_index, right_index, 1):
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 100
                        self.find_iron = True
                        break
        
        num_iron = info["inventory"]["iron"]
        if num_iron > self.prev_iron:
            reward += 10000
            done = True
        self.prev_iron = num_iron
        
        return obs, reward, done, info


def get_direction_label(player_pos, target_obj_id, semantic_map):
    """
    Calculate the direction label for the target object relative to the player.
    
    Args:
        player_pos: (x, y) position of the player
        target_obj_id: ID of the target object to focus on
        semantic_map: semantic map from game info
    
    Returns:
        direction_label: int from 0-8 representing direction
                        0=up, 1=up-right, 2=right, 3=down-right, 
                        4=down, 5=down-left, 6=left, 7=up-left, 8=none
    """
    # Find the closest target object
    target_positions = []
    
    # Search in a reasonable range around the player
    search_radius = 15  # Adjust based on your game view
    px, py = player_pos
    
    for x in range(max(0, px - search_radius), min(semantic_map.shape[0], px + search_radius + 1)):
        for y in range(max(0, py - search_radius), min(semantic_map.shape[1], py + search_radius + 1)):
            if semantic_map[x, y] == target_obj_id:
                target_positions.append((x, y))
    
    if not target_positions:
        return 8  # None - no target object found
    
    # Find the closest target object
    min_distance = float('inf')
    closest_target = None
    
    for target_pos in target_positions:
        distance = abs(target_pos[0] - px) + abs(target_pos[1] - py)  # Manhattan distance
        if distance < min_distance:
            min_distance = distance
            closest_target = target_pos
    
    if closest_target is None:
        return 8  # None
    
    # Calculate direction
    dx = closest_target[0] - px
    dy = closest_target[1] - py
    
    # If target is at the same position as player
    if dx == 0 and dy == 0:
        return 8  # None
    
    # Convert to direction label
    # We use 8-direction system + none
    if dx == 0:  # Vertical movement only
        if dy > 0:
            return 4  # down
        else:
            return 0  # up
    elif dy == 0:  # Horizontal movement only  
        if dx > 0:
            return 2  # right
        else:
            return 6  # left
    else:  # Diagonal movement
        if dx > 0 and dy > 0:
            return 3  # down-right
        elif dx > 0 and dy < 0:
            return 1  # up-right
        elif dx < 0 and dy > 0:
            return 5  # down-left
        else:  # dx < 0 and dy < 0
            return 7  # up-left


class DirectionLabelWrapper(gym.Wrapper):
    """
    Wrapper that generates direction labels for the target object the agent should focus on.
    The direction label indicates where the target object is relative to the player.
    """
    
    def __init__(self, env, target_obj_id, target_obj_name="stone"):
        super().__init__(env)
        self.target_obj_id = target_obj_id  # e.g., 7 for stone
        self.target_obj_name = target_obj_name
        
        # Modify observation space to include direction labels
        self.observation_space = spaces.Dict({
            'obs': env.observation_space,
            'direction_label': spaces.Discrete(9)  # 9 classes: 8 directions + none
        })
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Initial direction label (no target visible)
        dict_obs = {'obs': obs, 'direction_label': 8}  # 8 = None
        return dict_obs
    
    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        
        # Calculate direction label
        player_pos = info['player_pos']
        semantic_map = info['semantic']
        direction_label = get_direction_label(player_pos, self.target_obj_id, semantic_map)
        
        # Add direction label to info for use in training
        info['direction_label'] = direction_label
        
        dict_obs = {'obs': obs, 'direction_label': direction_label}
        return dict_obs, reward, terminated, info


def call_wrapper(env, wrapper_list, init_wrapper_params):

    for wrapper in wrapper_list:
        if wrapper == "MineStoneWrapper":
            env = MineStoneWrapper(env)
        if wrapper == "InitWrapper":
            env = InitWrapper(env, init_wrapper_params[0], init_wrapper_params[1])

    return env

    



class SelectiveBlurWrapper(gym.Wrapper):
    """
    方案一：选择性模糊包装器
    
    核心思路：利用游戏语义地图信息，对非目标物体区域进行模糊处理，
    使智能体更容易关注到重要的目标物体，从而加速训练。
    
    技术实现：
    1. 从游戏info中获取semantic_map和player_pos
    2. 识别目标物体在视野中的位置
    3. 对非目标区域应用高斯模糊
    4. 保持目标物体区域清晰
    """
    
    def __init__(self, env, target_obj_id, target_obj_name="stone", blur_strength=5):
        """
        初始化选择性模糊包装器
        
        Args:
            env: 被包装的环境
            target_obj_id: 目标物体的ID (例如: 3=stone, 8=coal, 9=iron)
            target_obj_name: 目标物体名称 (用于调试)
            blur_strength: 模糊强度，设置为是奇数 (默认5) why?
        """
        super().__init__(env)
        self.target_obj_id = target_obj_id
        self.target_obj_name = target_obj_name
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        
        # 物体ID到名称的映射（基于Crafter游戏）
        self.id_to_name = {
            0: 'none', 1: 'water', 2: 'grass', 3: 'stone', 4: 'path', 5: 'sand',
            6: 'tree', 7: 'lava', 8: 'coal', 9: 'iron', 10: 'diamond',
            11: 'table', 12: 'furnace', 13: 'player', 14: 'cow', 15: 'zombie',
            16: 'skeleton', 17: 'arrow', 18: 'plant'
        }
        
        # print(f"SelectiveBlurWrapper initialized:")
        # print(f"  Target object: {target_obj_name} (ID: {target_obj_id})")
        # print(f"  Blur strength: {blur_strength}")

    def _get_target_mask(self, semantic_map, player_pos, view_size, image_shape):
        """
        创建目标物体遮罩
        
        Args:
            semantic_map: 游戏语义地图 (64x64)
            player_pos: 玩家位置 [x, y]
            view_size: 视野大小 [width, height]
            image_shape: 图像形状 [height, width, channels]
        
        Returns:
            target_mask: 目标物体遮罩，1表示目标物体区域，0表示其他区域
        """
        px, py = player_pos
        view_w, view_h = view_size
        
        # 计算视野范围
        half_w, half_h = view_w // 2, view_h // 2
        x1 = max(0, px - half_w)
        y1 = max(0, py - half_h)
        x2 = min(semantic_map.shape[0], px + half_w + 1)
        y2 = min(semantic_map.shape[1], py + half_h + 1)
        
        # 提取视野区域的语义地图
        view_semantic = semantic_map[x1:x2, y1:y2]
        
        # 创建目标物体遮罩
        target_positions = (view_semantic == self.target_obj_id)
        semantic_mask = target_positions.astype(np.uint8)
        
        # 将语义遮罩缩放到图像尺寸
        img_h, img_w = image_shape[:2]
        if semantic_mask.shape[0] > 0 and semantic_mask.shape[1] > 0:
            semantic_mask = semantic_mask.T
            
            target_mask = cv2.resize(
                semantic_mask.astype(np.float32),
                (img_w, img_h),
                interpolation=cv2.INTER_LINEAR
            )
            # 应用阈值以保持二值特性
            target_mask = (target_mask > 0.3).astype(np.uint8)
        else:
            target_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        return target_mask

    def _apply_selective_blur(self, image, target_mask):
        """
        应用选择性模糊
        
        Args:
            image: 原始图像
            target_mask: 目标物体遮罩
            
        Returns:
            processed_image: 处理后的图像
        """
        # 创建模糊版本
        blurred = cv2.GaussianBlur(image, (self.blur_strength, self.blur_strength), 0)
        
        # 对遮罩进行轻微平滑处理
        smooth_mask = cv2.GaussianBlur(target_mask.astype(np.float32), (5, 5), 0)
        
        # 扩展遮罩到三个通道
        mask_3d = np.stack([smooth_mask] * 3, axis=2)
        
        # 选择性混合：目标区域保持原图，其他区域使用模糊图
        result = mask_3d * image + (1 - mask_3d) * blurred
        
        return result.astype(np.uint8)

    def reset(self, **kwargs):
        """重置环境"""
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """环境步进，应用选择性模糊"""
        obs, reward, done, info = self.env.step(action)
        
        # 获取必要的游戏信息
        semantic_map = info.get('semantic', None)
        player_pos = info.get('player_pos', [32, 32])
        view_size = info.get('view', [9, 9])
        
        if semantic_map is not None:
            try:
                # 创建目标物体遮罩
                target_mask = self._get_target_mask(semantic_map, player_pos, view_size, obs.shape)
                
                # 应用选择性模糊
                processed_obs = self._apply_selective_blur(obs, target_mask)
                
                # 添加调试信息
                target_found = np.sum(target_mask) > 0
                info['selective_blur'] = {
                    'target_obj_id': self.target_obj_id,
                    'target_obj_name': self.target_obj_name,
                    'target_found': target_found,
                    'target_pixels': int(np.sum(target_mask)),
                    'blur_strength': self.blur_strength
                }
                
                return processed_obs, reward, done, info
                
            except Exception as e:
                print(f"SelectiveBlurWrapper error: {e}")
                # 如果处理失败，返回原图像
                return obs, reward, done, info
        
        return obs, reward, done, info