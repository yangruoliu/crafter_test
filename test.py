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

def test(env, model, num_episodes, stack_size=2, render=True):

    print("Testing...")

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

        while not done:

            action, _ = model.predict(np.concatenate(frames, axis=-1), deterministic=False)

            frames.pop(0)
            frames.append(obs)

            obs, reward, done, info = env.step(action)

            episode_reward += reward

            if render:

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

    config = {
        "generate_rule": False,
        "test_episodes": 1,
        "recorder": False,
        "recorder_res_path": "base_model_res",
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "render": True,
        "stack_size": 1,
        "model_name": "stone"
    }

    generate_rule = config["generate_rule"]

    env = gym.make("MyCrafter-v0")
    if config["recorder"]:
        env = crafter.Recorder(
            env, config["recorder_res_path"],
            save_stats = False,
            save_video = true,
            save_episode = False,
        )
    env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=config["init_num"])
    # env = env_wrapper.WoodPickaxeWrapper(env)

    if generate_rule:
        env = env_wrapper.LLMWrapper(env, model=model)

    model = PPO.load(os.path.join("RL_models", config["model_name"]))
    stack_size = config["stack_size"]
    test_episodes = config["test_episodes"]
    render = config["render"]

    total_rewards = test(env, model, test_episodes, render=render, stack_size=stack_size)

    average_reward = sum(total_rewards) / test_episodes
    print(f"Average reward over {test_episodes} episodes: {average_reward}")

    if generate_rule:
        # save rules
        with open("rules.txt", "w") as f:
            f.write(env.rule_set)
