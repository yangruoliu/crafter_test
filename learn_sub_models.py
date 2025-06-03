import planning
import gym
from model import CustomResNet, CustomACPolicy, CustomPPO, TQDMProgressBar
import torch.nn as nn
import torch
import crafter
import env_wrapper
import os
import test
from stable_baselines3 import PPO

OBJECTS_TABLE = ["health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"]


def train_model(env, save_path, total_timesteps=1000000):

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

    total_timesteps = total_timesteps

    model.learn(total_timesteps=total_timesteps, callback=TQDMProgressBar(total_timesteps=total_timesteps))

    model.save(save_path)

    env.close()

def build_reward_wrapper(wrapper_name):

    locals_dict = {}
    exec("env1 = {}(env)".format(wrapper_name), globals(), locals_dict)
    return locals_dict['env1'] 

def build_init_wrapper(env, init_items, init_num):

    env = env_wrapper.InitWrapper(env, init_items, init_nume)
    return env


# rules = open("rules_mine_stones.txt", 'r').read()
tasks_list = ["craft an iron pickaxe"]
# (tasks_list, command_list) = planning.plan(tasks_list=tasks_list, num_step=3, rules=rules)
object_list = ["wood", "wood_pickaxe", "stone", "stone_pickaxe", "iron"]

for i, object in enumerate(object_list):

    wrapper_name = "wrapper_" + object
    try:
        exec(planning.define_wrapper(object))
    except Exception as e:
        print("{} is not defined".format(wrapper_name))

    env = gym.make("MyCrafter-v0")
    env = build_init_wrapper(env, init_items=object_list[:i], init_num=[1 for _ in range(i)])
    env = build_reward_wrapper(wrapper_name)

    save_path = os.path.join("RL_models", object)

    train_model(env, save_path=save_path, total_timesteps=1000000)

    model = PPO.load(save_path)

    test_episodes = 1
    render = True

    total_rewards = test.test(env, model, test_episodes, render=render)

    average_reward = sum(total_rewards) / test_episodes
    print(f"Average reward over {test_episodes} episodes: {average_reward}")




















