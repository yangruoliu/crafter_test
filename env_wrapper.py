import llm_utils
import gym
import numpy as np

class LLMWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.rule_set = "{}"
        self.cur_step = 0

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        if self.cur_step % 200 == 0:
            user_prompt = llm_utils.compose_user_prompt(info["obs"], self.rule_set)
            self.rule_set = llm_utils.llm_chat(user_prompt)
        
        self.cur_step += 1

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
