import gym
import crafter
from stable_baselines3 import PPO
import env_wrapper
import test
from tqdm import tqdm
from model import CustomResNet, CustomACPolicy, CustomPPO, TQDMProgressBar
import torch.nn as nn
import torch

if __name__ == "__main__":

    env = gym.make("MyCrafter-v0") 
    # env = gym.make("MyCrafter-v1")

    generate_rule = False

    if generate_rule:
        env = env_wrapper.LLMWrapper(env)
    # env = env_wrapper.MovementWrapper(env)
    env = env_wrapper.DrinkWaterWrapper(env)

    policy_kwargs = {
        "features_extractor_class": CustomResNet,
        "features_extractor_kwargs": {"features_dim": 1024},
        "activation_fn": nn.ReLU,
        "net_arch": [],
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"eps": 1e-5}
    }

    model = CustomPPO(
        CustomACPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=3,
        gamma=0.95,
        gae_lambda=0.65,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        normalize_advantage=False
    )

    total_timesteps = 1000000

    model.learn(total_timesteps=total_timesteps, callback=TQDMProgressBar(total_timesteps=total_timesteps))

    model.save("water_model")

    test_episodes = 1
    render = True
    save_rule = False

    # test
    total_rewards = test.test(env, model, test_episodes, render=render)

    average_reward = sum(total_rewards) / test_episodes
    print(f"Average reward over {test_episodes} episodes: {average_reward}")

    if save_rule:
        # save rules
        with open("rules.txt", "w") as f:
            f.write(env.rule_set)

    env.close()
