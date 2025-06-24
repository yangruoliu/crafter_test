import matplotlib.pyplot as plt
import time
from PIL import Image
import gym
from stable_baselines3 import PPO
import crafter
import env_wrapper
import os
import llm_prompt

def show(fig, image_display, obs):

    img = Image.fromarray(obs)

    image_display.set_data(img)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    time.sleep(0.2)

def get_item_number(info, item):

    return info["inventory"][item]


def test(env, model_list, num_episodes, render=True):

    print("Testing...")

    total_rewards = []

    for episode in range(num_episodes):

        obs = env.reset()

        if render:

            plt.ion()
            fig, ax = plt.subplots()
            image_display = ax.imshow(obs)
            plt.show(block=False)

        done = False
        episode_reward = 0

        model = model_list["base"]
        need_water = False

        while not done:

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            if get_item_number(info, "wood_pickaxe") > 0:

                if get_item_number(info, "stone_pickaxe") == 0 and get_item_number(info, "stone") == 0:
                    model = model_list["stone"]
                else:
                    model = model_list["stone_pickaxe"]

            if get_item_number(info, "stone_pickaxe") > 0:

                model = model_list["base"]
                

            episode_reward += reward

            if render:
                
                show(fig, image_display, obs)
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        total_rewards.append(episode_reward)

    return total_rewards

if __name__ == "__main__":


    env = gym.make("MyCrafter-v0")
    # env = env_wrapper.MineStoneWrapper(env)
    env = env_wrapper.InitWrapper(env, init_items=[], init_num=[])


    model_description = '''{
"model1": {"description": "craft a wood pickaxe", "requirement": "None"},
"model2": {"description": "mine stone", "requirement": "1 wood pickaxe"},
"model3": {"description": "craft a stone pickaxe", "requirement": "1 stone"},
}
    '''
    rules = open("rules.txt", 'r').read()


    model1 = PPO.load(os.path.join("RL_models", "orinal_agent"))
    model2 = PPO.load(os.path.join("RL_models", "stone"))
    model3 = PPO.load(os.path.join("RL_models", "stone_pickaxe"))
    model5 = PPO.load(os.path.join("RL_models", "wood_pickaxe"))

    model_list = {}
    model_list = {"base": model1,
                  "stone": model2,
                  "stone_pickaxe": model3,
                  "wood_pickaxe": model5}

    test_episodes = 1
    render = True

    total_rewards = test(env, model_list, test_episodes, render=render)

    average_reward = sum(total_rewards) / test_episodes
    print(f"Average reward over {test_episodes} episodes: {average_reward}")
