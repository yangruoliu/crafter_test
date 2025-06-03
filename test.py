import matplotlib.pyplot as plt
import time
from PIL import Image
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# import crafter
from crafter import crafter
import env_wrapper
import os
import numpy as np
from tqdm import tqdm
from llm_attention_map import parse_seen_objects, build_attn_map, convert_to_rgb_image_pil

def test(env, model, num_episodes, attn_obj_list=[], stack_size=2, render=True, with_attn=False):

    print("Testing...")

    total_rewards = []

    for episode in tqdm(range(num_episodes)):

        obs = env.reset()

        attn = np.full((64, 64, 3), 255, dtype=np.uint8)

        if render:

            plt.ion()
            fig, ax = plt.subplots()
            if with_attn:
                image_display = ax.imshow(np.hstack((obs, attn)))
            else:
                image_display = ax.imshow(obs)
            plt.show(block=False)

        done = False
        episode_reward = 0
        frames = [obs] * stack_size 

        while not done:

            if with_attn:
                action, _ = model.predict(np.concatenate([obs, attn], axis=-1), deterministic=False)
            else:
                action, _ = model.predict(np.concatenate(frames, axis=-1), deterministic=False)

            frames.pop(0)
            frames.append(obs)

            obs, reward, done, info = env.step(action)

            if with_attn:

                objects_list, distances_list, directions_list = parse_seen_objects(info['obs'])
                
                indices = [i for i, obj in enumerate(objects_list) if obj in attn_obj_list]

                distances_list = [distances_list[i] for i in indices]
                directions_list = [directions_list[i] for i in indices]

                attn = build_attn_map(directions_list, distances_list)

                attn = convert_to_rgb_image_pil(attn)

            episode_reward += reward

            if render:

                if with_attn:
                    img = Image.fromarray(np.hstack((obs, attn)))

                else:
                    img = Image.fromarray(obs)

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
    env = env_wrapper.InitWrapper(env, init_items=["wood_pickaxe"], init_num=[1])
    # env = env_wrapper.StoneSwordWrapper(env)

    if generate_rule:
        env = env_wrapper.LLMWrapper(env, model=model)

    # model = PPO.load(os.path.join("RL_models", "stone.zip"))
    model = PPO.load("stone_pickaxe_attn")
    stack_size = 2
    with_attn = True
    test_episodes = 1 
    render = True

    total_rewards = test(env, model, test_episodes, render=render, stack_size=stack_size, with_attn=with_attn, attn_obj_list=["stone"])

    average_reward = sum(total_rewards) / test_episodes
    print(f"Average reward over {test_episodes} episodes: {average_reward}")

    if generate_rule:
        # save rules
        with open("rules.txt", "w") as f:
            f.write(env.rule_set)
