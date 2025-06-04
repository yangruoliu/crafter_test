import matplotlib.pyplot as plt
import time
from PIL import Image
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils import deterministic
# import crafter
from crafter import crafter
import env_wrapper
import os
import numpy as np
from tqdm import tqdm
from llm_attention_map import parse_seen_objects, build_attn_map, convert_to_rgb_image_pil
from my_label_oracle import get_label

def test(env, model, num_episodes, render=True):

    print("Testing...")

    total_rewards = []

    for episode in tqdm(range(num_episodes)):

        obs = env.reset()

        if render:

            plt.ion()
            fig, ax = plt.subplots()
            image_display = ax.imshow(obs['obs'])
            plt.show(block=False)

        done = False
        episode_reward = 0

        while not done:

            action, _ = model.predict(obs, deterministic=False)

            classification_results = model.policy.predict_with_aux(obs, deterministic=True)

            predicted_class = classification_results['predicted_class'][0]
            print(predicted_class)


            obs, reward, done, info = env.step(action)

            episode_reward += reward

            if render:

                img = Image.fromarray(obs['obs'])

                image_display.set_data(img)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                time.sleep(0.2)
                # plt.close()

        if render:
            plt.close()

        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        total_rewards.append(episode_reward)

    return total_rewards

if __name__ == "__main__":


    generate_rule = False
    # model = "deepseek-r1:8b"
    model = "qwen2.5:7b"

    env = gym.make("MyCrafter-v0")
    env = env_wrapper.LabelGeneratingWrapper(env, get_label_func=get_label, target_obj="tree", num_aux_classes=2)
    # env = crafter.Recorder(
    #     env, "base_model_res",
    #     save_stats = False,
    #     save_video = False,
    #     save_episode = False,
    # )
    # env = env_wrapper.DrinkWaterWrapper(env)
    # env = env_wrapper.MakeStoneSwordWrapper(env)
    # env = env_wrapper.NavigationWrapper(env, obj_index=9)
    # env = env_wrapper.MineIronWrapper(env, navigation_model=PPO.load("navigation_iron"))
    # env = env_wrapper.InitWrapper(env, init_items=["wood_pickaxe"], init_num=[1])
    # env = env_wrapper.StoneSwordWrapper(env)

    if generate_rule:
        env = env_wrapper.LLMWrapper(env, model=model)

    # model = PPO.load(os.path.join("RL_models", "stone.zip"))
    model = PPO.load("wood0.02")
    stack_size = 1
    with_attn = False
    test_episodes = 1 
    render = True

    total_rewards = test(env, model, test_episodes, render=render)

    average_reward = sum(total_rewards) / test_episodes
    print(f"Average reward over {test_episodes} episodes: {average_reward}")

    if generate_rule:
        # save rules
        with open("rules.txt", "w") as f:
            f.write(env.rule_set)
