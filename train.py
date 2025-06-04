import gym
import crafter
import env_wrapper
from model import CustomResNet, CustomACPolicy, CustomPPO, TQDMProgressBar
import torch.nn as nn
import torch
from stable_baselines3 import PPO
from gym.wrappers import FrameStack
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, log_path: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_path = log_path
        self.rewards = []

        # 创建保存目录
        os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 记录每一步的奖励
        if 'episode' in self.locals:
            ep_info = self.locals['infos'][0].get('episode')
            if ep_info is not None:
                self.rewards.append(ep_info['r'])

        # 每 check_freq 步保存一下当前累计的奖励
        if self.n_calls % self.check_freq == 0:
            np.save(os.path.join(self.log_path, "rewards.npy"), np.array(self.rewards))
        return True


if __name__ == "__main__":

    env = gym.make("MyCrafter-v0") 
    # env = env_wrapper.myFrameStack(env, stack_size=4)
    # env = gym.make("MyCrafter-v1")
    # env =   TransposeImage(env)
    # env = gym.wrappers.FrameStack(env, num_stack=2, lz4_compress=False)
    # env = env_wrapper.FlattenStackWrapper(env)

    # env = env_wrapper.MovementWrapper(env)
    # env = env_wrapper.DrinkWaterWrapper(env)
    # env = env_wrapper.AttentionMapWrapper(env, obj_list=["stone"], stack_size=2)
    # env = env_wrapper.MineStoneWrapper(env)
    env = env_wrapper.WoodWrapper(env)
    # env = env_wrapper.StoneSwordWrapper(env)
    # env = env_wrapper.MineCoalWrapper(env, navigation_model=PPO.load("navigation_coal"))
    # env = env_wrapper.MineIronWrapper(env, navigation_model=PPO.load("navigation_iron"))
    # env = env_wrapper.NavigationWrapper(env, 9)
    # env = env_wrapper.MineIronWrapper2(env)
    env = env_wrapper.InitWrapper(env, init_items=[], init_num=[])
    # env = env_wrapper.FurnaceWrapper(env)
    # env = env_wrapper.WoodPickaxeWrapper(env)

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

    # model.learn(total_timesteps=total_timesteps, callback=TQDMProgressBar(total_timesteps=total_timesteps))
    model.learn(total_timesteps=total_timesteps, callback=RewardLoggerCallback(check_freq=1000, log_path="./log/"), progress_bar=True)

    model.save("wood")

    env.close()
