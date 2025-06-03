import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from tqdm import tqdm
from stable_baselines3 import PPO



class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        residual = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual

class CustomResNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)
        # print(observation_space.shape)
        c, h, w = observation_space.shape
        
        self.conv_net = nn.Sequential(
            # Stack 1
            nn.Conv2d(c, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResNetBlock(64),
            ResNetBlock(64),
            
            # Stack 2
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResNetBlock(64),
            ResNetBlock(64),
            
            # Stack 3
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResNetBlock(128),
            ResNetBlock(128),
        )
        
        self.dense_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Re-initialize last linear layers
        nn.init.orthogonal_(self.dense_net[-2].weight, gain=1.0)
        nn.init.constant_(self.dense_net[-2].bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.conv_net(observations)
        return self.dense_net(features)


class CustomACPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Modify final layer initialization
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.orthogonal_(self.value_net.weight, gain=0.1)
        nn.init.constant_(self.action_net.bias, 0)
        nn.init.constant_(self.value_net.bias, 0)


class EWMARolloutBuffer(RolloutBuffer):
    def __init__(self, *args, ewma_decay=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewma_decay = ewma_decay
        self.ewma_mean = 0.0
        self.ewma_var = 1.0
        self.ewma_count = 0

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        # Compute returns and advantages using parent method
        super().compute_returns_and_advantage(last_values, dones)
        
        # Update EWMA statistics using returns from the current rollout
        returns = self.returns.flatten()
        batch_mean = np.mean(returns)
        batch_var = np.var(returns)
        
        if self.ewma_count == 0:
            self.ewma_mean = batch_mean
            self.ewma_var = batch_var
        else:
            # Apply EWMA update
            self.ewma_mean = self.ewma_decay * self.ewma_mean + (1 - self.ewma_decay) * batch_mean
            self.ewma_var = self.ewma_decay * self.ewma_var + (1 - self.ewma_decay) * batch_var
        
        self.ewma_count += 1
        
        # Normalize returns with EWMA stats
        self.returns = (self.returns - self.ewma_mean) / (np.sqrt(self.ewma_var) + 1e-8)


class CustomPPO(PPO):
    def _setup_model(self) -> None:
        super()._setup_model()  # Initialize default components
        # Replace the buffer with the custom EWMA buffer
        self.rollout_buffer = EWMARolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            ewma_decay=0.99  # As per your settings
        )

class TQDMProgressBar(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.ewma_mean = 0.0
        self.ewma_var = 1.0

    def _on_training_start(self):
        # Initialize progress bar
        self.pbar = tqdm(total=self.total_timesteps, dynamic_ncols=True)
        self.pbar.set_description("Training PPO")

    def _on_step(self):
        # Update progress bar
        self.pbar.update(self.training_env.num_envs)  # For parallel environments
        return True

    def _on_rollout_end(self):
        # Add custom metrics (modify with your actual metrics)
        stats = {
            "ep_rew_mean": np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
            "ewma_mean": self.model.rollout_buffer.ewma_mean,
            "ewma_var": self.model.rollout_buffer.ewma_var,
            "entropy": self.model.logger.name_to_value["train/entropy_loss"]
        }
        self.pbar.set_postfix(stats)

    def _on_training_end(self):
        # Close progress bar
        self.pbar.close()
