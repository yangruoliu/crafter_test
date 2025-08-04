from importlib.metadata import distribution
from gymnasium.envs.registration import EnvCreator
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from torch.nn.modules import adaptive
from tqdm import tqdm
from stable_baselines3 import PPO
from gym import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.utils import  explained_variance



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
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 1024):
        super().__init__(observation_space['obs'], features_dim)
        # print(observation_space.shape)
        c, h, w = observation_space["obs"].shape
        
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

    def forward(self, observations) -> torch.Tensor:
        obs = observations['obs']
        
        # 确保输入是 4D: [B, C, H, W]
        if obs.dim() == 5:  # [B, N, C, H, W] -> [B*N, C, H, W]
            obs = obs.view(-1, *obs.shape[-3:])
        elif obs.dim() == 3:  # [C, H, W] -> [1, C, H, W]
            obs = obs.unsqueeze(0)
        
        features = self.conv_net(obs)
        return self.dense_net(features)


class CustomACPolicy(ActorCriticPolicy):
    def __init__(self, *args, num_aux_classes: int = 9, **kwargs):
        super().__init__(*args, **kwargs)
        # Modify final layer initialization
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.orthogonal_(self.value_net.weight, gain=0.1)
        nn.init.constant_(self.action_net.bias, 0)
        nn.init.constant_(self.value_net.bias, 0)

        features_dim = self.features_extractor.features_dim
        # Direction prediction network for 9 classes (8 directions + None)
        self.direction_net = nn.Linear(features_dim, num_aux_classes)
        
        # Initialize direction prediction network
        nn.init.orthogonal_(self.direction_net.weight, gain=1.0)
        nn.init.constant_(self.direction_net.bias, 0)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate actions according to the current policy,
        given the observations and actions.
        
        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log probability of the action, entropy of the action distribution, 
                 and direction prediction logits
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        # Direction prediction
        direction_logits = self.direction_net(features)

        return values, log_prob, distribution.entropy(), direction_logits

    @torch.no_grad()
    def predict_with_direction(self, obs: dict[str, torch.Tensor], deterministic: bool = True):
        """
        Get the policy action and direction prediction from an observation.
        
        :param obs: the input observation
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: the model's action, direction probabilities, and predicted direction class
        """
        self.set_training_mode(False)
        obs_tensor, _ = self.obs_to_tensor(obs)

        features = self.extract_features(obs_tensor)
        latent_pi, _ = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        direction_logits = self.direction_net(features)
        direction_probabilities = F.softmax(direction_logits, dim=1)
        predicted_direction = torch.argmax(direction_probabilities, dim=1)

        direction_results = {
            "logits": direction_logits.cpu().numpy(),
            "probabilities": direction_probabilities.cpu().numpy(),
            "predicted_direction": predicted_direction.cpu().numpy()
        }

        return actions.cpu().numpy(), direction_results

class EWMARolloutBuffer(DictRolloutBuffer):
    def __init__(self, *args, ewma_decay=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewma_decay = ewma_decay
        self.ewma_mean = 0.0
        self.ewma_var = 1.0
        self.ewma_count = 0
        
        # Add storage for direction labels
        self.direction_labels = None

    def reset(self) -> None:
        """
        Reset the rollout buffer.
        """
        super().reset()
        self.direction_labels = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

    def add(
        self,
        obs: dict,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        direction_label: np.ndarray = None,
    ) -> None:
        """
        Add elements to the rollout buffer.
        
        :param obs: Observation
        :param action: Action
        :param reward: 
        :param episode_start: Start of episode signal.
        :param value: estimated value of the observation
        :param log_prob: log probability of the action
        :param direction_label: direction label for auxiliary task
        """
        # Call parent add method
        super().add(obs, action, reward, episode_start, value, log_prob)
        
        # Store direction labels if provided
        if direction_label is not None:
            if isinstance(direction_label, torch.Tensor):
                direction_label = direction_label.cpu().numpy()
            self.direction_labels[self.pos - 1] = direction_label.copy()

    def get(self, batch_size: int = None):
        """
        Generator that returns batches of rollout data.
        """
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        # Prepare to return mini-batches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < len(indices):
            batch_indices = indices[start_idx : start_idx + batch_size]
            
            # Get observations, actions, etc. from parent data structures
            yield self._get_samples(batch_indices)
            start_idx += batch_size

    def _get_samples(self, batch_indices):
        """
        Get samples for given indices
        """
        from stable_baselines3.common.buffers import RolloutBufferSamples
        from collections import namedtuple
        
        # Get observations (handling dict observation spaces)
        observations = {}
        for key in self.observations.keys():
            observations[key] = self.observations[key][batch_indices]
        
        # Get other standard rollout data
        actions = self.actions[batch_indices]
        values = self.values[batch_indices].flatten()
        log_probs = self.log_probs[batch_indices].flatten()
        advantages = self.advantages[batch_indices].flatten()
        returns = self.returns[batch_indices].flatten()
        
        # Convert to tensors
        observations = {key: self.to_torch(obs) for key, obs in observations.items()}
        actions = self.to_torch(actions)
        values = self.to_torch(values)
        log_probs = self.to_torch(log_probs)
        advantages = self.to_torch(advantages)
        returns = self.to_torch(returns)
        
        # Create custom batch with direction labels
        ExtendedRolloutBufferSamples = namedtuple(
            "ExtendedRolloutBufferSamples",
            RolloutBufferSamples._fields + ("direction_labels",)
        )
        
        # Get direction labels if available
        direction_labels = None
        if self.direction_labels is not None:
            direction_labels = self.to_torch(self.direction_labels.flatten()[batch_indices])
        
        batch = ExtendedRolloutBufferSamples(
            observations=observations,
            actions=actions,
            old_values=values,
            old_log_prob=log_probs,
            advantages=advantages,
            returns=returns,
            direction_labels=direction_labels
        )
        
        return batch

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

    def __init__(self, *args, aux_weight: float = 0.5, direction_weight: float = 0.3, 
                 loss_normalization: bool = True, norm_decay: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_weight = aux_weight
        self.direction_weight = direction_weight
        self.aux_loss_fn = torch.nn.CrossEntropyLoss()
        self.direction_loss_fn = torch.nn.CrossEntropyLoss()
        
        # Loss normalization parameters
        self.loss_normalization = loss_normalization
        self.norm_decay = norm_decay
        
        # Initialize loss moving averages for normalization
        self.policy_loss_ma = None
        self.value_loss_ma = None
        self.entropy_loss_ma = None
        self.direction_loss_ma = None
        
        # Initialize loss moving variances for coefficient of variation
        self.policy_loss_var = None
        self.value_loss_var = None
        self.entropy_loss_var = None
        self.direction_loss_var = None
        
        # Step counter for initialization
        self.loss_norm_steps = 0

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

    def _update_loss_statistics(self, policy_loss: float, value_loss: float, 
                               entropy_loss: float, direction_loss: float) -> None:
        """
        Update moving averages and variances for loss normalization.
        Uses exponential moving average (EWMA) for robust statistics.
        """
        self.loss_norm_steps += 1
        
        losses = [policy_loss, value_loss, entropy_loss, direction_loss]
        moving_averages = [self.policy_loss_ma, self.value_loss_ma, 
                          self.entropy_loss_ma, self.direction_loss_ma]
        moving_variances = [self.policy_loss_var, self.value_loss_var,
                           self.entropy_loss_var, self.direction_loss_var]
        
        # Initialize on first step
        if self.loss_norm_steps == 1:
            self.policy_loss_ma = policy_loss
            self.value_loss_ma = value_loss
            self.entropy_loss_ma = entropy_loss
            self.direction_loss_ma = direction_loss
            
            self.policy_loss_var = 0.0
            self.value_loss_var = 0.0
            self.entropy_loss_var = 0.0
            self.direction_loss_var = 0.0
        else:
            # Update moving averages and variances using EWMA
            updated_mas = []
            updated_vars = []
            
            for loss, ma, var in zip(losses, moving_averages, moving_variances):
                # Update moving average
                new_ma = self.norm_decay * ma + (1 - self.norm_decay) * loss
                # Update moving variance (Welford's online algorithm adaptation)
                new_var = self.norm_decay * var + (1 - self.norm_decay) * (loss - ma) ** 2
                
                updated_mas.append(new_ma)
                updated_vars.append(new_var)
            
            self.policy_loss_ma, self.value_loss_ma, self.entropy_loss_ma, self.direction_loss_ma = updated_mas
            self.policy_loss_var, self.value_loss_var, self.entropy_loss_var, self.direction_loss_var = updated_vars

    def _normalize_losses(self, policy_loss: torch.Tensor, value_loss: torch.Tensor,
                         entropy_loss: torch.Tensor, direction_loss: torch.Tensor) -> tuple:
        """
        Normalize losses using moving averages to handle scale differences.
        Returns normalized losses and adaptive weights based on coefficient of variation.
        """
        if not self.loss_normalization or self.loss_norm_steps < 10:
            # Skip normalization for first few steps to build statistics
            return policy_loss, value_loss, entropy_loss, direction_loss, 1.0, self.ent_coef, self.vf_coef, self.direction_weight
        
        # Normalize by moving averages (avoid division by zero)
        eps = 1e-8
        
        policy_loss_norm = policy_loss / (self.policy_loss_ma + eps)
        value_loss_norm = value_loss / (self.value_loss_ma + eps)
        entropy_loss_norm = entropy_loss / (self.entropy_loss_ma + eps)
        direction_loss_norm = direction_loss / (self.direction_loss_ma + eps)
        
        # Calculate coefficient of variation for adaptive weighting
        # Use absolute values to handle negative loss means (like entropy)
        policy_std = (self.policy_loss_var ** 0.5)
        value_std = (self.value_loss_var ** 0.5)
        entropy_std = (self.entropy_loss_var ** 0.5)
        direction_std = (self.direction_loss_var ** 0.5)
        
        policy_cov = policy_std / (abs(self.policy_loss_ma) + eps)
        value_cov = value_std / (abs(self.value_loss_ma) + eps)
        entropy_cov = entropy_std / (abs(self.entropy_loss_ma) + eps)
        direction_cov = direction_std / (abs(self.direction_loss_ma) + eps)
        
        # Ensure all coefficients are positive
        policy_cov = max(0.0, policy_cov)
        value_cov = max(0.0, value_cov)
        entropy_cov = max(0.0, entropy_cov)
        direction_cov = max(0.0, direction_cov)
        
        # Adaptive weights based on coefficient of variation
        # Higher CoV means higher uncertainty, should get more weight
        # Apply smoothing to prevent extreme weight distributions
        policy_cov_smooth = policy_cov ** 0.5  # Take square root to reduce extreme values
        value_cov_smooth = value_cov ** 0.5
        entropy_cov_smooth = entropy_cov ** 0.5
        direction_cov_smooth = direction_cov ** 0.5
        
        total_cov = policy_cov_smooth + value_cov_smooth + entropy_cov_smooth + direction_cov_smooth + eps
        
        # Base weights from original coefficients with adaptive scaling
        base_policy_weight = 1.0
        base_value_weight = self.vf_coef
        base_entropy_weight = self.ent_coef
        base_direction_weight = self.direction_weight
        
        # Adaptive weights combining base weights with variation-based scaling
        adaptive_policy_weight = (policy_cov_smooth / total_cov) * base_policy_weight
        adaptive_value_weight = (value_cov_smooth / total_cov) * base_value_weight
        adaptive_entropy_weight = (entropy_cov_smooth / total_cov) * base_entropy_weight
        adaptive_direction_weight = (direction_cov_smooth / total_cov) * base_direction_weight
        
        # Ensure minimum weights to prevent complete suppression
        min_weight = 0.15  # 增加最小权重保护，确保每个损失都有足够参与度
        adaptive_policy_weight = max(min_weight * base_policy_weight, adaptive_policy_weight)
        adaptive_value_weight = max(min_weight * base_value_weight, adaptive_value_weight)
        adaptive_entropy_weight = max(min_weight * base_entropy_weight, adaptive_entropy_weight)
        adaptive_direction_weight = max(min_weight * base_direction_weight, adaptive_direction_weight)
        
        # Add maximum weight limit to prevent single loss domination
        max_single_weight = 0.4  # 防止任何单一损失超过40%
        adaptive_policy_weight = min(max_single_weight, adaptive_policy_weight)
        adaptive_value_weight = min(max_single_weight, adaptive_value_weight)
        adaptive_entropy_weight = min(max_single_weight, adaptive_entropy_weight)
        adaptive_direction_weight = min(max_single_weight, adaptive_direction_weight)
        
        # Renormalize weights to ensure they sum to reasonable total
        total_weight = (adaptive_policy_weight + adaptive_value_weight + 
                       adaptive_entropy_weight + adaptive_direction_weight)
        
        if total_weight > 0:
            adaptive_policy_weight /= total_weight
            adaptive_value_weight /= total_weight
            adaptive_entropy_weight /= total_weight
            adaptive_direction_weight /= total_weight
        
        # Apply final scaling with base weights
        adaptive_policy_weight *= base_policy_weight
        adaptive_value_weight *= base_value_weight
        adaptive_entropy_weight *= base_entropy_weight
        adaptive_direction_weight *= base_direction_weight
        
        return (policy_loss_norm, value_loss_norm, entropy_loss_norm, direction_loss_norm,
                adaptive_policy_weight, adaptive_entropy_weight, adaptive_value_weight, adaptive_direction_weight)

    def collect_rollouts(
        self, env, callback, rollout_buffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a rollout buffer.
        Modified to collect direction labels during rollout.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
           with torch.no_grad():
            if isinstance(self._last_obs, dict):
                obs_tensor = to_tensor_dict(self._last_obs, self.device)
            else:
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
            actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Extract direction labels from environment info if available
            direction_labels = None
            if len(infos) > 0 and 'direction_label' in infos[0]:
                direction_labels = np.array([info.get('direction_label', 8) for info in infos])  # 8 = None

            rollout_buffer.add(
                self._last_obs, 
                actions, 
                rewards, 
                self._last_episode_starts, 
                values, 
                log_probs,
                direction_label=direction_labels
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            if isinstance(new_obs, dict):
                obs_tensor = to_tensor_dict(new_obs, self.device)
            else:
                obs_tensor = torch.as_tensor(new_obs).to(self.device)
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Modified to include direction prediction auxiliary loss.
        """
        # Switch to train mode (this affects batch norm / dropout)

        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, direction_losses = [], [], []
        clip_fractions = []
        direction_accuracies = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy, direction_logits = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # Direction prediction auxiliary loss
                direction_loss = torch.tensor(0.0, device=self.device)
                direction_accuracy = 0.0
                if hasattr(rollout_data, 'direction_labels') and rollout_data.direction_labels is not None:
                    direction_loss = self.direction_loss_fn(direction_logits, rollout_data.direction_labels)
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        direction_pred = torch.argmax(direction_logits, dim=1)
                        direction_accuracy = (direction_pred == rollout_data.direction_labels).float().mean().item()
                    
                    direction_losses.append(direction_loss.item())
                    direction_accuracies.append(direction_accuracy)

                # Update loss statistics for normalization
                self._update_loss_statistics(
                    policy_loss.item(), 
                    value_loss.item(), 
                    entropy_loss.item(), 
                    direction_loss.item()
                )

                # Normalize losses and get adaptive weights
                (policy_loss_norm, value_loss_norm, entropy_loss_norm, direction_loss_norm,
                 adaptive_policy_weight, adaptive_entropy_weight, adaptive_value_weight, adaptive_direction_weight) = self._normalize_losses(
                    policy_loss, value_loss, entropy_loss, direction_loss
                )

                # Total loss combines normalized losses with adaptive weights
                if self.loss_normalization and self.loss_norm_steps >= 10:
                    loss = (adaptive_policy_weight * policy_loss_norm + 
                           adaptive_entropy_weight * entropy_loss_norm + 
                           adaptive_value_weight * value_loss_norm + 
                           adaptive_direction_weight * direction_loss_norm)
                else:
                    # Fallback to original weighting for initial steps
                    loss = (policy_loss + 
                           self.ent_coef * entropy_loss + 
                           self.vf_coef * value_loss + 
                           self.direction_weight * direction_loss)

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        
        # Log direction prediction metrics
        if direction_losses:
            self.logger.record("train/direction_loss", np.mean(direction_losses))
            self.logger.record("train/direction_accuracy", np.mean(direction_accuracies))
        
        # Log loss normalization statistics
        if self.loss_normalization and self.loss_norm_steps >= 10:
            self.logger.record("train/loss_norm/policy_loss_ma", self.policy_loss_ma)
            self.logger.record("train/loss_norm/value_loss_ma", self.value_loss_ma)
            self.logger.record("train/loss_norm/entropy_loss_ma", self.entropy_loss_ma)
            self.logger.record("train/loss_norm/direction_loss_ma", self.direction_loss_ma)
            
            # Log coefficient of variations
            eps = 1e-8
            policy_cov = max(0.0, (self.policy_loss_var ** 0.5) / (abs(self.policy_loss_ma) + eps))
            value_cov = max(0.0, (self.value_loss_var ** 0.5) / (abs(self.value_loss_ma) + eps))
            entropy_cov = max(0.0, (self.entropy_loss_var ** 0.5) / (abs(self.entropy_loss_ma) + eps))
            direction_cov = max(0.0, (self.direction_loss_var ** 0.5) / (abs(self.direction_loss_ma) + eps))
            
            self.logger.record("train/loss_norm/policy_cov", policy_cov)
            self.logger.record("train/loss_norm/value_cov", value_cov)
            self.logger.record("train/loss_norm/entropy_cov", entropy_cov)
            self.logger.record("train/loss_norm/direction_cov", direction_cov)
            
            # Log adaptive weights
            total_cov = policy_cov + value_cov + entropy_cov + direction_cov + eps
            self.logger.record("train/adaptive_weights/policy", policy_cov / total_cov)
            self.logger.record("train/adaptive_weights/value", value_cov / total_cov * self.vf_coef)
            self.logger.record("train/adaptive_weights/entropy", entropy_cov / total_cov * self.ent_coef)
            self.logger.record("train/adaptive_weights/direction", direction_cov / total_cov * self.direction_weight)
        
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
    
    # def train(self):
    #     super().train()
    

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

def to_tensor_dict(obs_dict, device):
    result = {}
    for k, v in obs_dict.items():
        tensor = torch.as_tensor(v).to(device)
        if k == 'obs' and tensor.dim() == 5:  # 修正5D到4D
            tensor = tensor.view(-1, *tensor.shape[-3:])
        result[k] = tensor
    return result
