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

import heapq 

# from skimage.draw import line as bresenham_line

import cv2

import os

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

    """
    
    def __init__(self, env, target_obj_id, target_obj_name="stone", blur_strength=3):
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

        # 定义世界视野在垂直方向上所占的比例 (7行世界 / 9行总视野)
        self.WORLD_VIEW_HEIGHT_RATIO = 7.0 / 9.0

    # def _get_target_mask(self, semantic_map, player_pos, view_size, image_shape):
    #     """
    #     创建目标物体遮罩
        
    #     Args:
    #         semantic_map: 游戏语义地图 (64x64)
    #         player_pos: 玩家位置 [x, y]
    #         view_size: 视野大小 [width, height]
    #         image_shape: 图像形状 [height, width, channels]
        
    #     Returns:
    #         target_mask: 目标物体遮罩，1表示目标物体区域，0表示其他区域
    #     """
    #     px, py = player_pos
    #     view_w, view_h = view_size
        
    #     # 计算视野范围
    #     half_w, half_h = view_w // 2, view_h // 2
    #     x1 = max(0, px - half_w)
    #     y1 = max(0, py - half_h)
    #     x2 = min(semantic_map.shape[0], px + half_w + 1)
    #     y2 = min(semantic_map.shape[1], py + half_h + 1)
        
    #     # 提取视野区域的语义地图
    #     view_semantic = semantic_map[x1:x2, y1:y2]
    #     print(f"----   view_semantic: {view_semantic}")
        
    #     # 创建目标物体遮罩
    #     target_positions = (view_semantic == self.target_obj_id) | (view_semantic == 13)
    #     semantic_mask = target_positions.astype(np.uint8)
        
    #     # 将语义遮罩缩放到图像尺寸
    #     img_h, img_w = image_shape[:2]
    #     if semantic_mask.shape[0] > 0 and semantic_mask.shape[1] > 0:
    #         semantic_mask = semantic_mask.T
            
    #         target_mask = cv2.resize(
    #             semantic_mask.astype(np.float32),
    #             (img_w, img_h),
    #             interpolation=cv2.INTER_LINEAR
    #         )
    #         # 应用阈值以保持二值特性
    #         target_mask = (target_mask > 0.3).astype(np.uint8)
    #     else:
    #         target_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
    #     return target_mask

    def _get_proportional_clear_mask(self, semantic_map, player_pos, view_size, image_shape):
        """
        这个函数将屏幕分为“世界视野”和“UI物品栏”两部分，并为每个部分生成正确的蒙版，
        最后将它们组合成一个完整的64x64蒙版。

        错误的原因，截取的位置不对。
        """
        img_h, img_w = image_shape[:2] # 64, 64

        world_view_px_h = int(round(img_h * self.WORLD_VIEW_HEIGHT_RATIO))

        inventory_px_h = img_h - world_view_px_h

        px, py = player_pos
        view_w, _ = view_size # 9
        semantic_view_h = 7
        half_w = view_w // 2
        half_h = semantic_view_h // 2

        x1, x2 = max(0, px - half_w), min(semantic_map.shape[1], px + half_w + 1)
        y1, y2 = max(0, py - half_h), min(semantic_map.shape[0], py + half_h + 1)

        view_semantic = semantic_map[x1:x2, y1:y2]

        player_id = 13

        clear_semantic_mask = ((view_semantic == self.target_obj_id) | (view_semantic == player_id)).astype(np.uint8)
        
        target_indices = list(zip(*np.where(view_semantic == self.target_obj_id)))
        
        if target_indices:
            # 玩家在视野中心的位置
            player_pos_in_view = (half_w, half_h) # (x, y) 格式

            # 找到最近的目标物体
            closest_target = min(target_indices, key=lambda pos: self._heuristic(pos, player_pos_in_view))
            
            # 转置语义地图以匹配 (y, x) 坐标系
            view_semantic_transposed = view_semantic.T
            
            # A* 寻路的起点和终点需要是 (y, x) 格式
            start_node = (player_pos_in_view[1], player_pos_in_view[0])
            goal_node = (closest_target[1], closest_target[0])
            
            # 调用 A* 算法寻找路径
            path = self._find_shortest_path(view_semantic_transposed, start_node, goal_node)
            
            # 如果找到路径，将路径上的点也加入清晰蒙版
            if path:
                for y, x in path:
                    # 注意：在原始的 (宽度, 高度) 蒙版上更新
                    if 0 <= x < clear_semantic_mask.shape[0] and 0 <= y < clear_semantic_mask.shape[1]:
                       clear_semantic_mask[x, y] = True


        clear_semantic_mask = clear_semantic_mask.T.astype(np.uint8)
        # 将7x9语义蒙版缩放到世界视野的像素尺寸 (64x50) ---
        # 注意：cv2.resize的尺寸参数是(宽度, 高度)
        world_view_mask = cv2.resize(
            clear_semantic_mask,
            (img_w, world_view_px_h), # 缩放到 (64, 50)
            interpolation=cv2.INTER_NEAREST # 使用最近邻插值，保持清晰的边界
        )

        final_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # 将64x50的世界蒙版放到最终蒙版的上半部分
        final_mask[0:world_view_px_h, :] = world_view_mask
        
        # 将最终蒙版的下半部分（UI区域）全部设为1（清晰）
        final_mask[world_view_px_h:img_h, :] = 1 # 1代表清晰
        
        return final_mask


    def _get_target_mask(self, semantic_map, player_pos, view_size, image_shape):
        """
        创建目标物体遮罩，包含目标物体、主人公和最短路径
        
        Args:
            semantic_map: 游戏语义地图 (64x64)
            player_pos: 玩家位置 [x, y]
            view_size: 视野大小 [width, height]
            image_shape: 图像形状 [height, width, channels]
        
        Returns:
            target_mask: 目标物体遮罩，1表示清晰区域，0表示模糊区域
        """
        px, py = player_pos
        view_w, view_h = view_size
        
        # 计算视野范围
        half_w, half_h = view_w // 2, view_h // 2 - 1  # 这里是因为视野范围是以角色为中心的 7 * 9， 最下面两行是角色所有物。
        x1 = max(0, px - half_w)
        y1 = max(0, py - half_h)
        x2 = min(semantic_map.shape[0], px + half_w + 1)
        y2 = min(semantic_map.shape[1], py + half_h + 1)
        # print(f"----   Player at ({px}, {py}) in 7×9 view")
        # print(f"----  854  View range: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        # ----   Player at (32, 32) in 7×9 view
        # ----   View range: x1=28, y1=28, x2=37, y2=37
    
        # print(f"----   x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
        # print(f"semantic_map.shape: {semantic_map.shape}")
        # print(f"semantic_map: {semantic_map}")
        
        # 提取视野区域的语义地图
        view_semantic = semantic_map[x1:x2, y1:y2]
        # print(f"----  864 view_semantic: {view_semantic}")  # 其实到这里的view_semantic 都是没问题的，

        
        
        # 创建基础遮罩：目标物体和主人公
        target_positions = (view_semantic == self.target_obj_id) | (view_semantic == 13)
        # print(f"----   target_positions: {target_positions}")
        semantic_mask = target_positions.astype(np.uint8)
        print(f"----   semantic_mask: {semantic_mask}")
        
        # 如果视野内有目标物体，计算最短路径
        # if np.any(view_semantic == self.target_obj_id):
        #     # 找到目标物体的位置
        #     target_y, target_x = np.where(view_semantic == self.target_obj_id)
        #     if len(target_y) > 0 and len(target_x) > 0:
        #         # 取第一个目标物体位置
        #         target_pos = (target_y[0], target_x[0])
                
        #         # 计算从主人公到目标物体的最短路径
        #         path_mask = self._find_shortest_path(view_semantic, target_pos, (half_h, half_w))
                
        #         # 合并路径到遮罩中
        #         semantic_mask = semantic_mask | path_mask
        #         print(f"----   semantic_mask2: {semantic_mask}")
        
        # 将语义遮罩缩放到图像尺寸
        img_h, img_w = image_shape[:2]
        # print(f"----   img_h: {img_h}, img_w: {img_w}")  (64, 64)
        if semantic_mask.shape[0] > 0 and semantic_mask.shape[1] > 0:
            semantic_mask = semantic_mask.T
            print(f"----   semantic_mask.T: {semantic_mask}")
            
            target_mask = cv2.resize(
                semantic_mask.astype(np.float32),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            )
            # 应用阈值以保持二值特性
            target_mask = (target_mask > 0.5).astype(np.uint8)

            print(f"---- 904 --  target_mask: {target_mask}")
            # os.makedirs("target_mask", exist_ok=True)
            # path = os.path.join("target_mask", f"target_mask_{self.target_obj_name}.txt")

            # with open(path, 'w') as f:
            #     for row in target_mask:
            #         row_str = " ".join([str(int(val)) for val in row])
            #         f.write(row_str + "\n")

            # print(f"Target mask saved to: {path}")
        else:
            target_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        return target_mask
    
    def _debug_mask_alignment(self, semantic_map, player_pos, view_size, target_mask):
        """
        调试遮罩对齐问题
        """
        print("=== Debug Mask Alignment ===")
        print(f"Player position: {player_pos}")
        print(f"View size: {view_size}")
        print(f"Semantic map shape: {semantic_map.shape}")
        print(f"Target mask shape: {target_mask.shape}")
        
        # 检查玩家位置周围的遮罩值
        px, py = player_pos
        if 0 <= px < target_mask.shape[1] and 0 <= py < target_mask.shape[0]:
            center_value = target_mask[py, px]
            print(f"Mask value at player position: {center_value}")
        
        # 检查目标物体的位置
        target_y, target_x = np.where(semantic_map == self.target_obj_id)
        if len(target_y) > 0:
            print(f"Target object found at: ({target_x[0]}, {target_y[0]})")
            # 检查对应位置的遮罩值
            if (0 <= target_x[0] < target_mask.shape[1] and 
                0 <= target_y[0] < target_mask.shape[0]):
                target_mask_value = target_mask[target_y[0], target_x[0]]
                print(f"Mask value at target position: {target_mask_value}")
        
        print("===========================")

    def _find_shortest_path(self, grid, start, goal):
        """
        使用A*算法找到从主人公到目标物体的最短路径
        
        Args:
                grid: 视野的语义地图 (view_semantic.T)，已经是 (高度, 宽度) 格式。
                start: 起点坐标 (y, x)。
                goal: 终点坐标 (y, x)。
                
            Returns:
                包含路径坐标的列表，如果找不到路径则返回空列表。
        """
        # 定义在语义地图上哪些物体是不可通过的障碍物
        # ID: 1=water, 6=tree, 7=lava (根据Crafter游戏定义调整)
        obstacles_ids = {1, 6, 7}

        # 初始化
        open_set = [(0, start)]  # 优先队列，存储 (f_score, position)
        came_from = {}          # 记录路径，用于回溯

        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:

            current_f, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dy, current[1] + dx)

                if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    continue

                if grid[neighbor[0], neighbor[1]] in obstacles_ids:
                    continue

                tentative_g_score = g_score.get(current, float('inf')) + 1

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []



    def _heuristic(self, pos1, pos2):
        """
        曼哈顿距离启发式函数
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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

        # 添加调试信息
        # print(f"----   Real player position: {player_pos}")  # (32, 32)
        # print(f"----   view_size: {view_size}")  # (9, 9)
        # print(f"----   Info keys: {list(info.keys())}")

        # processed_obs = obs
        
        if semantic_map is not None:
            try:
                # 创建目标物体遮罩
                target_mask = self._get_proportional_clear_mask(semantic_map, player_pos, view_size, obs.shape)

                # self._debug_mask_alignment(semantic_map, player_pos, view_size, target_mask)
                
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