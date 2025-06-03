from httpx import get
import matplotlib.pyplot as plt
import time
from PIL import Image
import gym
from pydantic_core.core_schema import invalid_schema
from stable_baselines3 import PPO
from torch import ge, inverse
# import crafter
from crafter import crafter
import env_wrapper
import os
import llm_prompt
import llm_utils
import numpy as np
from tqdm import tqdm

def show(fig, image_display, obs):

    img = Image.fromarray(obs)

    image_display.set_data(img)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    time.sleep(0.2)

def get_item_number(info, item):

    return info["inventory"][item]

def create_model_description(available_models):

    description = {}
    for model in available_models:
        if model == "model1":
            description["model1"] = "craft a wood pickaxe"
        if model == "model2":
            description["model2"] = "collect a stone"
        if model == "model3":
            description["model3"] = "crafter a stone pickaxe"
        if model == "model6":
            description["model6"] = "collect a coal"
        if model == "model7":
            description["model7"] = "place a furnace"
        if model == "model9":
            description["model9"] = "collect an iron"

    return str(description)

def available_models(info):

    available_models = ["model1"]
    if get_item_number(info, "wood_pickaxe") >= 1:
        available_models.append("model2")
        available_models.append("model6")
        if get_item_number(info, "stone") >= 1:
            available_models.append("model3")
            if get_item_number(info, "stone") >= 4 and get_item_number(info, "coal") >= 1 and info["achievements"]["place_furnace"] == 0:
                available_models.append("model7")
        if get_item_number(info, "stone_pickaxe") >= 1:
            available_models.append("model9")
    return available_models


def choose_model(goal, info, last_model_call, model_list, rules, model_description):

    llm_response = llm_utils.llm_chat(model="deepseek-chat", prompt=llm_prompt.compose_llm_agent_prompt(rules=rules, model_description=model_description, current_goal=goal, info=info, last_model_call=last_model_call), system_prompt="")

    print(llm_response)

    model = model1
    for key in model_list.keys():
        if "call " + str(key) in llm_response:
            model = model_list[key]
            print("calling " + key)
            return model, key

    return model, "model1"

def face_at(obs):

    try:
        return obs.split()[obs.split().index("face") + 1]
    except ValueError as _:
        pass
    return ""

def is_finished(info, last_info):

    return info["inventory"]["iron"] > last_info["inventory"]["iron"] or info["inventory"]["wood_sword"] > last_info["inventory"]["wood_sword"] or info["inventory"]["wood_pickaxe"] > last_info["inventory"]["wood_pickaxe"] or info["inventory"]["stone"] > last_info["inventory"]["stone"] or info["inventory"]["stone_pickaxe"] > last_info["inventory"]["stone_pickaxe"] or info["inventory"]["coal"] > last_info["inventory"]["coal"] or (face_at(info["obs"]) == "furnace" and info["achievements"]["place_furnace"] == 0)

def is_current_goal_achieved(goal, info, last_info):

    if "wood_sword" in goal:
        return get_item_number(info, "wood_sword") > get_item_number(last_info, "wood_sword")
    elif "wood_pickaxe" in goal:
        return get_item_number(info, "wood_pickaxe") > get_item_number(last_info, "wood_pickaxe")
    elif "stone_pickaxe" in goal:
        return get_item_number(info, "stone_pickaxe") > get_item_number(last_info, "stone_pickaxe")
    elif "stone" in goal:
        return get_item_number(info, "stone") > get_item_number(last_info, "stone")
    elif "coal" in goal:
        return get_item_number(info, "coal") > get_item_number(last_info, "coal")
    elif "furnace" in goal:
        return face_at(info["obs"]) == "furnace"
    elif "iron" in goal:
        return get_item_number(info, "iron") > get_item_number(last_info, "iron")
    print("invalid goal!\n")
    return True

def not_moved(prev_locations):
    
    for i in range(1, len(prev_locations)):
        if not np.array_equal(prev_locations[i], prev_locations[i-1]):
            return False
    return True


def test(env, model_list, num_episodes, rules, model_description, goal_list, stack_size=1, mode="usual", last_model_call="", render=True):

    if goal_list == []:
        print("Please make sure there is at least one goal in goal_list")
        return 

    num_goals = len(goal_list)
    index = 0

    print("Testing...")
    print("current_goal: ", goal_list[index])

    total_rewards = []

    for episode in tqdm(range(num_episodes)):

        obs = env.reset()

        if render:

            plt.ion()
            fig, ax = plt.subplots()
            image_display = ax.imshow(obs)
            plt.show(block=False)

        done = False
        episode_reward = 0

        frames = [obs] * stack_size 
        model = model_list["model0"]
        model_name = "model0"
        last_model_call = ""
        last_info = ""
        prev_locations = [np.array([0, 0])] * 5 + [np.array([1, 1])] * 5

        is_first_epoch = True

        while not done:

            if not_moved(prev_locations):
                action = np.random.choice([1, 2, 3, 4])

            else:
                action, _ = model.predict(np.concatenate(frames, axis=-1), deterministic=True)

            frames.pop(0)
            frames.append(obs)

            obs, reward, done, info = env.step(action)

            prev_locations.pop(0)
            prev_locations.append(info["player_pos"])

            if is_first_epoch:

                is_first_epoch = False
                last_info = info
                continue

            if index != num_goals and is_current_goal_achieved(goal_list[index], info, last_info):

                index += 1

            if index >= num_goals:

                model = model8 

            else:

                if mode == "usual":

                    model, model_name = choose_model(goal_list[index], info, last_model_call, model_list, rules, model_description)
                    last_model_call = model_name

                elif mode == "lazy":

                    if is_finished(info, last_info):

                        model_description = create_model_description(available_models(info))
                        print(model_description)

                        model, model_name = choose_model(goal_list[index], info, last_model_call, model_list, rules, model_description)
                        last_model_call = model_name

            last_info = info
            episode_reward += reward

            if render:
                
                show(fig, image_display, obs)
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        total_rewards.append(episode_reward)

    return total_rewards

if __name__ == "__main__":


    env = gym.make("MyCrafter-v0")

    env = crafter.Recorder(
        env, "our_method_res",
        save_stats = True,
        save_video = False,
        save_episode = False,
    )
    # env = env_wrapper.MineStoneWrapper(env)
    env = env_wrapper.InitWrapper(env, init_items=[], init_num=[])


    model_description = '''{
"model1": {"description": "craft a wood pickaxe", "requirement": "None"},
"model2": {"description": "collect a stone", "requirement": "1 wood pickaxe"},
"model3": {"description": "craft a stone pickaxe", "requirement": "1 stone and 1 wood_pickaxe"},
"model6": {"description": "collect a coal", "requirement": "1 wood_pickaxe"}
"model7": {"description": "place a furnace", "requirement": "1 stone_pickaxe and 1 wood_pickaxe and 1 coal and 4 stones"}
}
    '''
    rules = open("rules.txt", 'r').read()


    model0 = PPO.load(os.path.join("RL_models", "original_agent"))
    model1 = PPO.load(os.path.join("RL_models", "wood_pickaxe"))
    model2 = PPO.load(os.path.join("RL_models", "stone"))
    model3 = PPO.load(os.path.join("RL_models", "stone_pickaxe"))
    # model4 = PPO.load("water_model")
    # model5 = PPO.load("navigation_coal")
    model6 = PPO.load(os.path.join("RL_models", "coal"))
    model7 = PPO.load(os.path.join("RL_models", "furnace"))
    model8 = PPO.load(os.path.join("RL_models", "original_agent"))
    model9 = PPO.load(os.path.join("RL_models", "iron"))

    model_list = {}
    model_list = {"model1": model1,
                  "model2": model2,
                  "model3": model3,
                  # "model4": model4,
                  # "model5": model5,
                  "model6": model6,
                  "model7": model7,
                  "model8": model8,
                  "model9": model9,
                  "model0": model8
                 }

    test_episodes = 100 
    render = False
    goal_list = ["wood_pickaxe", "stone_pickaxe", "stone", "coal", "furnace", "iron"]
    mode = "lazy"
    stack_size = 1

    total_rewards = test(env, model_list, test_episodes, rules=rules, model_description=model_description, goal_list=goal_list, render=render, last_model_call="", mode=mode, stack_size=stack_size)

    average_reward = sum(total_rewards) / test_episodes
    print(f"Average reward over {test_episodes} episodes: {average_reward}")
