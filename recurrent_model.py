import gym # 或者 import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import collections
import crafter
import matplotlib.pyplot as plt
from PIL import Image
import time
import env_wrapper

# --- 超参数 ---
CONFIG = {
    "env_id": "MyCrafter-v0", # 您的自定义环境 ID
    "seed": 42,
    "total_training_timesteps": 3000000,
    "n_steps_rollout": 512,
    "learning_rate": 2.5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.65,
    "ppo_epochs": 4,
    "ppo_clip_val": 0.1,
    "training_sequence_length": 32,
    "training_batch_size": 16,
    "value_loss_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "lstm_hidden_size": 1024,
    "log_interval": 1,
    "eval_interval": 100,
    "num_eval_episodes": 3,
    "save_path": "RE",
    "wrapper_list": [],
    "init_wrapper_params": [[], []]
}

assert CONFIG["n_steps_rollout"] % CONFIG["training_sequence_length"] == 0, \
    "n_steps_rollout must be a multiple of training_sequence_length"
num_sequences_per_rollout = CONFIG["n_steps_rollout"] // CONFIG["training_sequence_length"]
assert num_sequences_per_rollout % CONFIG["training_batch_size"] == 0, \
    "num_sequences_per_rollout must be a multiple of training_batch_size"

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

# --- 1. 模型定义 (RecurrentActorCritic) ---
class RecurrentActorCritic(nn.Module):
    def __init__(self, input_channels, num_actions, lstm_hidden_size):
        super(RecurrentActorCritic, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

        self.cnn = nn.Sequential(
            # Stack 1
            nn.Conv2d(input_channels, 64, 3, stride=1, padding=1),
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
            nn.Linear(256, 1024),
            nn.ReLU()
        )

        # CNN 期望 CHW 格式: (input_channels, 64, 64)
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), # 输入 (N, C, 64, 64)
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        cnn_output_size = 64 * 4 * 4 # 1024

        self.lstm = nn.LSTM(cnn_output_size, lstm_hidden_size, batch_first=False)
        self.actor_head = nn.Linear(lstm_hidden_size, num_actions)
        self.critic_head = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x, hidden_state):
        # x 必须是 CHW 格式: (seq_len, batch_size, C, H, W)
        # hidden_state: (h_0, c_0)
        seq_len, batch_size, c, h, w = x.size()
        cnn_in = x.view(seq_len * batch_size, c, h, w) # (N_total, C, H, W)
        cnn_out = self.cnn(cnn_in)
        cnn_out = self.dense_net(cnn_out)
        lstm_in = cnn_out.view(seq_len, batch_size, -1)
        lstm_out, next_hidden_state = self.lstm(lstm_in, hidden_state)
        action_logits = self.actor_head(lstm_out)
        value = self.critic_head(lstm_out)
        return action_logits, value, next_hidden_state

    def initial_hidden_state(self, batch_size, device):
        h_0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c_0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (h_0, c_0)

# --- 2. Rollout Buffer (适配循环网络) ---
class RecurrentRolloutBuffer:
    def __init__(self, n_steps, obs_shape_chw, action_shape, lstm_hidden_size, gae_lambda, gamma, device):
        self.n_steps = n_steps
        self.obs_shape_chw = obs_shape_chw # 存储 CHW 格式的形状
        self.action_shape = action_shape
        self.lstm_hidden_size = lstm_hidden_size
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.device = device

        # Buffer 存储 CHW 格式的观测数据
        self.observations = torch.zeros((self.n_steps,) + self.obs_shape_chw, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.n_steps,) + action_shape, dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.n_steps, dtype=torch.bool, device=self.device)
        self.log_probs = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
        
        self.h_states = torch.zeros((self.n_steps, 1, self.lstm_hidden_size), dtype=torch.float32, device=self.device)
        self.c_states = torch.zeros((self.n_steps, 1, self.lstm_hidden_size), dtype=torch.float32, device=self.device)

        self.ptr = 0
        self.rollout_filled = False

    def add(self, obs_chw_tensor, action, reward, done, log_prob, value, h_state, c_state):
        # obs_chw_tensor 应该是已经转换为 CHW 格式的 tensor
        if self.ptr >= self.n_steps:
            print("Warning: Buffer overflow")
            self.ptr = 0

        self.observations[self.ptr] = obs_chw_tensor
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.bool, device=self.device)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value.squeeze()
        self.h_states[self.ptr] = h_state.squeeze(1) 
        self.c_states[self.ptr] = c_state.squeeze(1)

        self.ptr += 1
        if self.ptr == self.n_steps:
            self.rollout_filled = True

    def compute_returns_and_advantages(self, last_value, last_done):
        last_gae_lam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].float()
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns = self.advantages + self.values
        self.rollout_filled = False
        self.ptr = 0

    def get_training_batches(self, training_sequence_length, training_batch_size):
        num_total_sequences = self.n_steps // training_sequence_length
        all_sequence_indices = np.arange(num_total_sequences)
        np.random.shuffle(all_sequence_indices)

        for i in range(0, num_total_sequences, training_batch_size):
            batch_indices = all_sequence_indices[i : i + training_batch_size]
            if len(batch_indices) == 0: continue
            current_batch_size = len(batch_indices)

            obs_batch = torch.zeros((training_sequence_length, current_batch_size) + self.obs_shape_chw, device=self.device)
            act_batch = torch.zeros((training_sequence_length, current_batch_size) + self.action_shape, device=self.device, dtype=torch.int64)
            logp_batch = torch.zeros((training_sequence_length, current_batch_size), device=self.device)
            adv_batch = torch.zeros((training_sequence_length, current_batch_size), device=self.device)
            ret_batch = torch.zeros((training_sequence_length, current_batch_size), device=self.device)
            h_init_batch = torch.zeros((1, current_batch_size, self.lstm_hidden_size), device=self.device)
            c_init_batch = torch.zeros((1, current_batch_size, self.lstm_hidden_size), device=self.device)

            for j, seq_idx in enumerate(batch_indices):
                start_idx = seq_idx * training_sequence_length
                end_idx = start_idx + training_sequence_length
                
                obs_batch[:, j, ...] = self.observations[start_idx:end_idx]
                act_batch[:, j, ...] = self.actions[start_idx:end_idx]
                logp_batch[:, j] = self.log_probs[start_idx:end_idx]
                adv_batch[:, j] = self.advantages[start_idx:end_idx]
                ret_batch[:, j] = self.returns[start_idx:end_idx]
                h_init_batch[0, j, :] = self.h_states[start_idx]
                c_init_batch[0, j, :] = self.c_states[start_idx]

            adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
            yield obs_batch, act_batch, logp_batch, adv_batch, ret_batch, (h_init_batch, c_init_batch)

# --- 3. 模型训练 ---
def train():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        env = gym.make(CONFIG["env_id"])
        env = env_wrapper.call_wrapper(env, CONFIG["wrapper_list"], CONFIG["init_wrapper_params"])
    except Exception as e:
        print(f"Error: Could not create environment '{CONFIG['env_id']}'. {e}")
        print("Please ensure your custom environment is registered and working correctly.")
        return

    actual_env_obs_shape_hwc = env.observation_space.shape # e.g., (64, 64, 3)
    print(f"Raw environment HWC observation space shape: {actual_env_obs_shape_hwc}")

    # 确认是 HWC 且通道数为 3
    if not (len(actual_env_obs_shape_hwc) == 3 and actual_env_obs_shape_hwc[2] == 3):
        raise ValueError(f"Environment observation shape {actual_env_obs_shape_hwc} is not HWC with 3 channels (e.g., H, W, 3).")

    # 转换为 CHW 格式 (Channels, Height, Width)
    # 例如 (64, 64, 3) -> (3, 64, 64)
    obs_shape_chw = (actual_env_obs_shape_hwc[2], actual_env_obs_shape_hwc[0], actual_env_obs_shape_hwc[1])
    input_channels_for_model = actual_env_obs_shape_hwc[2] # Should be 3

    print(f"Processed CHW observation shape for network: {obs_shape_chw}")
    print(f"Input channels for model: {input_channels_for_model}")

    if not isinstance(env.action_space, gym.spaces.Discrete):
        print("Error: Action space must be Discrete.")
        env.close()
        return
    num_actions = env.action_space.n

    model = RecurrentActorCritic(input_channels_for_model, num_actions, CONFIG["lstm_hidden_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], eps=1e-5)
    
    action_shape = ()
    buffer = RecurrentRolloutBuffer(
        CONFIG["n_steps_rollout"], obs_shape_chw, action_shape,
        CONFIG["lstm_hidden_size"], CONFIG["gae_lambda"], CONFIG["gamma"], device
    )

    current_obs_np_hwc = env.reset()
    current_obs_np_chw = current_obs_np_hwc.transpose(2, 0, 1) # HWC to CHW
    current_obs_tensor_chw = torch.tensor(current_obs_np_chw, dtype=torch.float32, device=device)
    
    current_h, current_c = model.initial_hidden_state(batch_size=1, device=device)
    
    episode_rewards = collections.deque(maxlen=10)
    current_episode_reward = 0
    num_updates = CONFIG["total_training_timesteps"] // CONFIG["n_steps_rollout"]
    global_step = 0

    for update_idx in range(1, num_updates + 1):
        model.eval()
        for step in range(CONFIG["n_steps_rollout"]):
            global_step +=1
            # current_obs_tensor_chw 已经是 CHW tensor
            obs_for_model = current_obs_tensor_chw.unsqueeze(0).unsqueeze(0) # (1, 1, C, H, W)
            
            with torch.no_grad():
                action_logits, value, (next_h, next_c) = model(obs_for_model, (current_h, current_c))
            
            dist = Categorical(logits=action_logits.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_item = action.item()

            next_obs_np_hwc, reward, done, info = env.step(action_item)
            next_obs_np_chw = next_obs_np_hwc.transpose(2, 0, 1) # HWC to CHW
            next_obs_tensor_chw = torch.tensor(next_obs_np_chw, dtype=torch.float32, device=device)

            # buffer.add 需要 CHW tensor
            buffer.add(current_obs_tensor_chw, action, reward, done, log_prob, value, current_h, current_c)
            
            current_obs_tensor_chw = next_obs_tensor_chw
            current_h, current_c = next_h, next_c
            current_episode_reward += reward

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                current_obs_np_hwc = env.reset()
                current_obs_np_chw = current_obs_np_hwc.transpose(2, 0, 1) # HWC to CHW
                current_obs_tensor_chw = torch.tensor(current_obs_np_chw, dtype=torch.float32, device=device)
                current_h, current_c = model.initial_hidden_state(batch_size=1, device=device)
        
        with torch.no_grad():
            obs_for_model = current_obs_tensor_chw.unsqueeze(0).unsqueeze(0)
            _, last_value, _ = model(obs_for_model, (current_h, current_c))
        buffer.compute_returns_and_advantages(last_value.squeeze(), done)

        model.train()
        for _ in range(CONFIG["ppo_epochs"]):
            for obs_b, act_b, logp_b, adv_b, ret_b, (h_init_b, c_init_b) in \
                buffer.get_training_batches(CONFIG["training_sequence_length"], CONFIG["training_batch_size"]):
                
                new_logits, new_values, _ = model(obs_b, (h_init_b, c_init_b))
                new_dist = Categorical(logits=new_logits)
                new_log_probs = new_dist.log_prob(act_b)
                
                ratio = torch.exp(new_log_probs - logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - CONFIG["ppo_clip_val"], 1.0 + CONFIG["ppo_clip_val"]) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                
                new_values = new_values.squeeze(-1)
                value_loss = F.mse_loss(new_values, ret_b)
                entropy_loss = -new_dist.entropy().mean()
                total_loss = policy_loss + CONFIG["value_loss_coef"] * value_loss + CONFIG["entropy_coef"] * entropy_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                optimizer.step()

        if update_idx % CONFIG["log_interval"] == 0 and len(episode_rewards) > 0:
            avg_reward = np.mean(list(episode_rewards))
            print(f"Update: {update_idx}/{num_updates}, Timesteps: {global_step}, Avg. Reward (last 10): {avg_reward:.2f}")
            if 'policy_loss' in locals(): # Check if losses are defined (i.e., at least one PPO epoch ran)
                 print(f"Losses -> Policy: {policy_loss.item():.3f}, Value: {value_loss.item():.3f}, Entropy: {-entropy_loss.item():.3f}")


        if update_idx % CONFIG["eval_interval"] == 0:
            print(f"\n--- Evaluating model at update {update_idx} ---")
            evaluate_model(model, CONFIG["env_id"], CONFIG["num_eval_episodes"], device, CONFIG["seed"] + update_idx, render=False)
            print("---------------------------------------\n")

    print("Training finished.")
    env.close()
    return model

# --- 4. 模型在环境中测试 ---
def evaluate_model(trained_model, env_id, num_episodes, device, seed, render=False):
    eval_env = gym.make(env_id) #, render_mode="human")
    eval_env = env_wrapper.call_wrapper(eval_env, ["InitWrapper"], CONFIG["init_wrapper_params"])
    
    eval_obs_shape_hwc = eval_env.observation_space.shape
    # 确认评估环境也是 HWC
    is_hwc_eval = (len(eval_obs_shape_hwc) == 3 and eval_obs_shape_hwc[2] == 3)
    if not is_hwc_eval:
        print(f"Warning: Evaluation environment observation shape {eval_obs_shape_hwc} is not HWC with 3 channels as expected.")
        # Handle accordingly or raise error if strict HWC is required by this function's logic

    trained_model.eval()
    total_rewards = []

    for episode in range(num_episodes):
        obs_np_hwc = eval_env.reset()
        if is_hwc_eval:
            obs_np_chw = obs_np_hwc.transpose(2, 0, 1)
        else: # Assume it's already CHW or handle other formats
            obs_np_chw = obs_np_hwc

        current_eval_h, current_eval_c = trained_model.initial_hidden_state(batch_size=1, device=device)
        done = False
        episode_reward = 0

        if render:

            plt.ion()
            fig, ax = plt.subplots()
            image_display = ax.imshow(obs_np_hwc)
            plt.show(block=False)
        
        while not done:
            obs_tensor_chw = torch.tensor(obs_np_chw, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                action_logits, _, (next_h, next_c) = trained_model(obs_tensor_chw, (current_eval_h, current_eval_c))
            
            action = torch.argmax(action_logits.squeeze(0), dim=-1)
            
            next_obs_np_hwc, reward, done, info = eval_env.step(action.item())
            if is_hwc_eval:
                obs_np_chw = next_obs_np_hwc.transpose(2, 0, 1)
            else:
                obs_np_chw = next_obs_np_hwc # Update current obs to next_obs

            image = Image.fromarray(next_obs_np_hwc)

            if render:
                image_display.set_data(image)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                time.sleep(0.2)

            current_eval_h, current_eval_c = next_h, next_c
            episode_reward += reward
            
            if done:
                 if done: current_eval_h, current_eval_c = trained_model.initial_hidden_state(batch_size=1, device=device)

        if render:
            plt.close()

        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"Average evaluation reward over {num_episodes} episodes: {avg_reward:.2f}")
    eval_env.close()
    return avg_reward


if __name__ == "__main__":
    print("Starting Recurrent PPO training for HWC environment...")
    # trained_policy_model = train()

    # torch.save(trained_policy_model.state_dict(), CONFIG["save_path"])
    
    MODEL_SAVE_PATH = CONFIG["save_path"] 

    dummy_env_for_params = gym.make(CONFIG["env_id"])
    dummy_env_for_params = env_wrapper.call_wrapper(dummy_env_for_params, ["InitWrapper"], CONFIG["init_wrapper_params"])
    dummy_obs_shape_hwc = dummy_env_for_params.observation_space.shape
    loaded_model_input_channels = dummy_obs_shape_hwc[2]
    loaded_model_num_actions = dummy_env_for_params.action_space.n
    dummy_env_for_params.close()


    trained_policy_model = RecurrentActorCritic(
        input_channels=loaded_model_input_channels, # 和训练时一样
        num_actions=loaded_model_num_actions,     # 和训练时一样
        lstm_hidden_size=CONFIG["lstm_hidden_size"]
    )
    
    device_to_use = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. 加载 state_dict
    trained_policy_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device_to_use))
    trained_policy_model.to(device_to_use)
    trained_policy_model.eval() # 确保设置为评估模式
    print(f"Model loaded successfully from {MODEL_SAVE_PATH}")

    print("\n--- Final Evaluation of Trained Model ---")
    evaluate_model(trained_policy_model, CONFIG["env_id"], 5, 
                       torch.device("cuda" if torch.cuda.is_available() else "cpu"), CONFIG["seed"] + 1000, render=True)
