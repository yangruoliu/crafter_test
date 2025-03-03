import matplotlib.pyplot as plt
import time
from PIL import Image
import gym
from stable_baselines3 import PPO
import crafter
import env_wrapper

def test(env, model, num_episodes, render=True):

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

        while not done:

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            episode_reward += reward

            if render:

                img = Image.fromarray(obs)

                image_display.set_data(img)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                time.sleep(0.2)
                # plt.close()

        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        total_rewards.append(episode_reward)

    return total_rewards

if __name__ == "__main__":


    env = gym.make("MyCrafter-v0")
    env = env_wrapper.DrinkWaterWrapper(env)
    model = PPO.load("water_model")

    test_episodes = 1
    render = True

    total_rewards = test(env, model, test_episodes, render=render)

    average_reward = sum(total_rewards) / test_episodes
    print(f"Average reward over {test_episodes} episodes: {average_reward}")
