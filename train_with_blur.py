# train_with_blur.py
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
import time

def train_with_selective_blur():
    """
    Train agent with selective blur
    
    Training approach:
    1. Apply selective blur wrapper to highlight target objects
    2. Use same PPO algorithm as original training
    3. Accelerate learning through image preprocessing
    """
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    config = {
        "total_timesteps": 2000000,
        "save_dir": os.path.join("RL_models_zdl", f"stone_with_blur_{timestamp}"),
        "model_name": f"stone_with_blur_{timestamp}",
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "target_obj_id": 3,        # Stone ID
        "target_obj_name": "stone",
        "blur_strength": 2         # Blur strength
    }

    os.makedirs("RL_models_zdl", exist_ok=True)

    print("=== Starting selective blur training ===")
    print(f"Target object: {config['target_obj_name']} (ID: {config['target_obj_id']})")
    print(f"Blur strength: {config['blur_strength']}")
    print(f"Training steps: {config['total_timesteps']}")
    print(f"Save directory: {config['save_dir']}")

    # Create environment
    env = gym.make("MyCrafter-v0")
    print("Base environment created")

    # Apply existing wrappers
    env = env_wrapper.MineStoneWrapper(env)
    env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=config["init_num"])
    print("Task wrappers applied")

    # Apply selective blur wrapper (key improvement)
    env = env_wrapper.SelectiveBlurWrapper(
        env,
        target_obj_id=config["target_obj_id"],
        target_obj_name=config["target_obj_name"],
        blur_strength=config["blur_strength"]
    )
    print("Selective blur wrapper applied")

    # Use same model configuration as original training
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
    print("PPO model created")

    # Create training monitoring callback
    class BlurTrainingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.blur_stats = []

        def _on_step(self) -> bool:
            # Collect selective blur statistics
            if hasattr(self.training_env, 'get_attr'):
                try:
                    infos = self.training_env.get_attr('_last_info')
                    if infos and len(infos) > 0:
                        info = infos[0]
                        if 'selective_blur' in info:
                            blur_info = info['selective_blur']
                            self.blur_stats.append({
                                'step': self.num_timesteps,
                                'target_found': blur_info.get('target_found', False),
                                'target_pixels': blur_info.get('target_pixels', 0)
                            })
                except:
                    pass
            return True

        def _on_training_end(self) -> None:
            if self.blur_stats:
                target_found_rate = sum(1 for s in self.blur_stats if s['target_found']) / len(self.blur_stats)
                avg_target_pixels = np.mean([s['target_pixels'] for s in self.blur_stats])
                print(f"\n=== Selective blur statistics ===")
                print(f"Target object detection rate: {target_found_rate:.2%}")
                print(f"Average target pixels: {avg_target_pixels:.1f}")

    # Start training
    callback = BlurTrainingCallback()
    print("Starting training...")
    
    model.learn(
        total_timesteps=config["total_timesteps"], 
        progress_bar=True,
        callback=callback
    )

    # Save model
    model.save(config["save_dir"])
    print(f"Model saved to: {config['save_dir']}")

    # Save configuration
    config_path = config["save_dir"] + "_config.txt"
    with open(config_path, 'w') as f:
        f.write("=== Selective blur training configuration ===\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"Configuration saved to: {config_path}")

    # Save latest model record
    latest_model_file = os.path.join("RL_models", "latest_blur_model.txt")
    with open(latest_model_file, 'w') as f:
        f.write(config["model_name"])
    print(f"Latest model record saved to: {latest_model_file}")

    env.close()
    print("=== Training completed ===")

if __name__ == "__main__":
    train_with_selective_blur()