# test_with_blur.py
import matplotlib.pyplot as plt
import time
from PIL import Image
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from crafter import crafter
import env_wrapper
import os
import numpy as np
from tqdm import tqdm
import glob
import cv2
from datetime import datetime

def test_with_blur(env, model, num_episodes, stack_size=2, render=True, save_video=False, video_dir="test_videos"):
    """
    Test model with selective blur functionality
    
    Args:
        env: Test environment
        model: Trained model
        num_episodes: Number of test episodes
        stack_size: Number of frames to stack
        render: Whether to display rendering
        save_video: Whether to save video
        video_dir: Directory to save videos
    """
    print("Testing with selective blur...")

    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        print(f"Videos will be saved to: {video_dir}")

    total_rewards = []
    blur_stats = []

    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        
        # Initialize video writer if saving video
        video_writer = None
        if save_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(video_dir, f"episode_{episode+1}_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (obs.shape[1], obs.shape[0]))
            print(f"Recording episode {episode+1} to: {video_path}")

        if render:
            plt.ion()
            fig, ax = plt.subplots()
            image_display = ax.imshow(obs)
            ax.set_title("Game Screen with Selective Blur")
            plt.show(block=False)

        done = False
        episode_reward = 0
        frames = [obs] * stack_size
        episode_blur_stats = []

        while not done:
            # Predict action
            action, _ = model.predict(np.concatenate(frames, axis=-1), deterministic=False)
            
            # Ensure action is scalar
            if isinstance(action, np.ndarray):
                action = int(action.item())
            elif hasattr(action, '__len__') and len(action) == 1:
                action = int(action[0])
            else:
                action = int(action)

            # Execute action
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            # Update frame stack
            frames.pop(0)
            frames.append(obs)

            # Save frame to video
            if save_video and video_writer is not None:
                frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            # Collect selective blur statistics
            blur_info = info.get('selective_blur', {})
            if blur_info:
                episode_blur_stats.append({
                    'target_found': blur_info.get('target_found', False),
                    'target_pixels': blur_info.get('target_pixels', 0),
                    'target_obj_name': blur_info.get('target_obj_name', 'unknown')
                })

            if render:
                img = Image.fromarray(obs)
                image_display.set_data(img)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                time.sleep(0.2)

        # Close video writer
        if save_video and video_writer is not None:
            video_writer.release()
            print(f"Episode {episode+1} video saved")

        if render:
            plt.close()

        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        total_rewards.append(episode_reward)
        blur_stats.extend(episode_blur_stats)

    return total_rewards, blur_stats

def plot_rewards(total_rewards, save_plot=True, plot_filename="reward_plot.png"):
    """
    Plot rewards as a line chart
    
    Args:
        total_rewards: List of rewards from all episodes
        save_plot: Whether to save the plot
        plot_filename: Filename to save the plot
    """
    plt.figure(figsize=(10, 6))
    episodes = range(1, len(total_rewards) + 1)
    
    plt.plot(episodes, total_rewards, 'o-', linewidth=2, markersize=6, color='blue')
    plt.title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add average line
    avg_reward = np.mean(total_rewards)
    plt.axhline(y=avg_reward, color='red', linestyle='--', alpha=0.7, 
                label=f'Average: {avg_reward:.2f}')
    plt.legend()
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(total_rewards):.2f}\nStd: {np.std(total_rewards):.2f}\nMin: {min(total_rewards):.2f}\nMax: {max(total_rewards):.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Reward plot saved as: {plot_filename}")
    
    plt.show()

def find_blur_model():
    """
    Find blur training model
    Priority:
    1. Model recorded in latest_blur_model.txt
    2. Latest blur model in RL_models directory
    3. Original stone model as fallback
    """
    
    # Method 1: Find from record file
    latest_file = os.path.join("RL_models_zdl", "latest_blur_model.txt")
    if os.path.exists(latest_file):
        try:
            with open(latest_file, 'r') as f:
                model_name = f.read().strip()
            model_path = os.path.join("RL_models_zdl", model_name)
            if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
                print(f"Found latest blur model: {model_name}")
                return model_name
        except:
            pass
    
    # Method 2: Search for blur models in RL_models directory
    print("Searching for blur models in RL_models directory...")
    
    # Search for .zip files
    zip_pattern = os.path.join("RL_models_zdl", "stone_with_blur_*.zip")
    zip_models = glob.glob(zip_pattern)
    
    # Search for directories
    dir_pattern = os.path.join("RL_models_zdl", "stone_with_blur_*")
    all_matches = glob.glob(dir_pattern)
    # Filter out config files, keep only directories
    dir_models = [m for m in all_matches if os.path.isdir(m) and not m.endswith('.txt')]
    
    all_blur_models = zip_models + dir_models
    
    if all_blur_models:
        print(f"Found candidate models: {[os.path.basename(m) for m in all_blur_models]}")
        # Find latest model
        latest_model = max(all_blur_models, key=os.path.getctime)
        model_name = os.path.basename(latest_model)
        # Remove .zip suffix if present
        if model_name.endswith('.zip'):
            model_name = model_name[:-4]
        print(f"Found blur model: {model_name}")
        return model_name
    
    # Method 3: Fallback to original model
    print("No blur model found, using original stone model")
    return "stone"

if __name__ == "__main__":
    config = {
        "generate_rule": False,
        "test_episodes": 50,
        "recorder": False,
        "recorder_res_path": "our_method_res",
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "render": True,
        "stack_size": 1,
        "model_name": "stone_with_blur_v1",  # Will be auto-selected
        "save_video": True,
        "video_dir": "test_videos",
        "target_obj_id": 3,
        "target_obj_name": "stone",
        "blur_strength": 6
    }

    generate_rule = config["generate_rule"]

    env = gym.make("MyCrafter-v0")
    if config["recorder"]:
        env = crafter.Recorder(
            env, config["recorder_res_path"],
            save_stats=False,
            save_video=True,
            save_episode=False,
        )
    env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=config["init_num"])

    env = env_wrapper.MineStoneWrapper(env)

    # Auto-select model if not specified
    if config["model_name"] is None:
        config["model_name"] = find_blur_model()

    # Load model
    try:
        if "blur" in config["model_name"].lower():
            model_path = os.path.join("RL_models_zdl", config["model_name"])
        else:
            model_path = os.path.join("RL_models", config["model_name"])
        model = PPO.load(model_path)
        print(f"Model loaded successfully: {config['model_name']}")
    except Exception as e:
        print(f"Model loading failed: {e}")
        print(f"Attempted to load: {model_path}")
        exit()

    # Apply selective blur wrapper for blur models
    if "blur" in config["model_name"].lower():
        env = env_wrapper.SelectiveBlurWrapper(env, target_obj_id=config["target_obj_id"],
                                             target_obj_name=config["target_obj_name"],
                                             blur_strength=config["blur_strength"])
        print("Applied selective blur environment")

    if generate_rule:
        env = env_wrapper.LLMWrapper(env, model=model)

    stack_size = config["stack_size"]
    test_episodes = config["test_episodes"]
    render = config["render"]
    save_video = config["save_video"]
    video_dir = config["video_dir"]

    total_rewards, blur_stats = test_with_blur(
        env, model, test_episodes, render=render, stack_size=stack_size,
        save_video=save_video, video_dir=video_dir
    )

    average_reward = sum(total_rewards) / test_episodes
    print(f"Average reward over {test_episodes} episodes: {average_reward}")

    # Plot rewards
    plot_rewards(total_rewards, save_plot=True, plot_filename="blur_reward_plot.png")

    # Print blur statistics if available
    if blur_stats:
        target_found_rate = sum(1 for s in blur_stats if s['target_found']) / len(blur_stats)
        avg_target_pixels = np.mean([s['target_pixels'] for s in blur_stats])
        print(f"Selective blur statistics:")
        print(f"  Target detection rate: {target_found_rate:.1%}")
        print(f"  Average target pixels: {avg_target_pixels:.1f}")

    if generate_rule:
        # save rules
        with open("rules.txt", "w") as f:
            f.write(env.rule_set)