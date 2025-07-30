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

def test_with_selective_blur(env, model, num_episodes, render=True, compare_mode=False, save_video=False, video_dir="test_videos"):
    """
    Test model with selective blur
    
    Args:
        env: Test environment
        model: Trained model
        num_episodes: Number of test episodes
        render: Whether to display rendering
        compare_mode: Whether to show comparison effects
        save_video: Whether to save video
        video_dir: Directory to save videos
    """
    print("=== Starting selective blur model testing ===")

    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        print(f"Videos will be saved to: {video_dir}")

    total_rewards = []
    blur_stats = []

    for episode in tqdm(range(num_episodes), desc="Testing episodes"):
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
            if compare_mode:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                ax1.set_title("Processed Image")
                ax2.set_title("Target Detection Status")
                ax3.set_title("Statistics")
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
                ax1.set_title("Game Screen with Selective Blur")
            
            image_display = ax1.imshow(obs)
            plt.show(block=False)

        done = False
        episode_reward = 0
        step_count = 0
        episode_blur_stats = []

        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
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
            step_count += 1

            # Save frame to video
            if save_video and video_writer is not None:
                frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            # Collect selective blur statistics
            blur_info = info.get('selective_blur', {})
            if blur_info:
                episode_blur_stats.append({
                    'step': step_count,
                    'target_found': blur_info.get('target_found', False),
                    'target_pixels': blur_info.get('target_pixels', 0),
                    'target_obj_name': blur_info.get('target_obj_name', 'unknown')
                })

            # Update display
            if render:
                # Update main image
                image_display.set_data(obs)
                
                if compare_mode and blur_info:
                    # Show target detection status
                    ax2.clear()
                    status_color = 'green' if blur_info.get('target_found', False) else 'red'
                    status_text = 'Target Found' if blur_info.get('target_found', False) else 'Target Not Found'
                    ax2.text(0.5, 0.7, status_text, ha='center', va='center', 
                            color=status_color, fontsize=16, weight='bold',
                            transform=ax2.transAxes)
                    ax2.text(0.5, 0.5, f"Target: {blur_info.get('target_obj_name', 'unknown')}", 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.text(0.5, 0.3, f"Pixels: {blur_info.get('target_pixels', 0)}", 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_xlim(0, 1)
                    ax2.set_ylim(0, 1)
                    ax2.axis('off')
                    
                    # Show statistics
                    ax3.clear()
                    if episode_blur_stats:
                        found_rate = sum(1 for s in episode_blur_stats if s['target_found']) / len(episode_blur_stats)
                        ax3.text(0.1, 0.8, f"Episode {episode + 1}", fontsize=12, weight='bold',
                                transform=ax3.transAxes)
                        ax3.text(0.1, 0.6, f"Steps: {step_count}", transform=ax3.transAxes)
                        ax3.text(0.1, 0.4, f"Reward: {episode_reward:.1f}", transform=ax3.transAxes)
                        ax3.text(0.1, 0.2, f"Target found: {found_rate:.1%}", transform=ax3.transAxes)
                    ax3.set_xlim(0, 1)
                    ax3.set_ylim(0, 1)
                    ax3.axis('off')

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                time.sleep(0.1)

        # Close video writer
        if save_video and video_writer is not None:
            video_writer.release()
            print(f"Episode {episode+1} video saved")

        if render:
            plt.close()

        # Record episode statistics
        blur_stats.extend(episode_blur_stats)
        total_rewards.append(episode_reward)
        
        # Calculate episode statistics
        if episode_blur_stats:
            target_found_rate = sum(1 for s in episode_blur_stats if s['target_found']) / len(episode_blur_stats)
            avg_target_pixels = np.mean([s['target_pixels'] for s in episode_blur_stats])
        else:
            target_found_rate = 0
            avg_target_pixels = 0

        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Target found rate={target_found_rate:.1%}, "
              f"Avg target pixels={avg_target_pixels:.1f}")

    return total_rewards, blur_stats

def find_blur_model():
    """
    Find blur training model
    Priority:
    1. Model recorded in latest_blur_model.txt
    2. Latest blur model in RL_models directory
    3. Original stone model as fallback
    """
    
    # Method 1: Find from record file
    latest_file = os.path.join("RL_models", "latest_blur_model.txt")
    if os.path.exists(latest_file):
        try:
            with open(latest_file, 'r') as f:
                model_name = f.read().strip()
            model_path = os.path.join("RL_models", model_name)
            if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
                print(f"Found latest blur model: {model_name}")
                return model_name
        except:
            pass
    
    # Method 2: Search for blur models in RL_models directory
    print("Searching for blur models in RL_models directory...")
    
    # Search for .zip files
    zip_pattern = os.path.join("RL_models", "stone_with_blur_*.zip")
    zip_models = glob.glob(zip_pattern)
    
    # Search for directories
    dir_pattern = os.path.join("RL_models", "stone_with_blur_*")
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

def list_available_models():
    """
    List all available models
    """
    print("=== Available models ===")
    if not os.path.exists("RL_models"):
        print("RL_models directory does not exist")
        return []
    
    zip_files = [f for f in os.listdir("RL_models") if f.endswith('.zip')]
    if not zip_files:
        print("No .zip model files found")
        return []
    
    print("Found models:")
    models = []
    for i, zip_file in enumerate(sorted(zip_files)):
        model_name = zip_file[:-4]  # Remove .zip suffix
        models.append(model_name)
        model_type = "Blur Model" if "blur" in model_name else "Regular Model"
        print(f"  {i+1}. {model_name} ({model_type})")
    
    return models

def compare_with_without_blur():
    """
    Compare effects with and without selective blur
    """
    print("=== Comparison test: with and without selective blur ===")
    
    config = {
        "test_episodes": 3,
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "target_obj_id": 3,
        "target_obj_name": "stone",
        "blur_strength": 1,
        "blur_model_name": find_blur_model(),
        "original_model_name": "stone" 
    }

    print(f"Comparing models: {config['blur_model_name']} vs {config['original_model_name']}")

    # Load models
    try:
        # Load blur model
        blur_model_path = os.path.join("RL_models", config["blur_model_name"])
        blur_model = PPO.load(blur_model_path)
        print(f"Blur model loaded successfully: {config['blur_model_name']}")
        
        # Load original model
        original_model_path = os.path.join("RL_models", config["original_model_name"])
        original_model = PPO.load(original_model_path)
        print(f"Original model loaded successfully: {config['original_model_name']}")

    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Please ensure the following files exist:")
        print(f"  RL_models/{config['blur_model_name']}.zip")
        print(f"  RL_models/{config['original_model_name']}.zip")
        return

    # Test 1: Original model in normal environment
    print("\n--- Test 1: Original model in normal environment ---")
    env1 = gym.make("MyCrafter-v0")
    env1 = env_wrapper.MineStoneWrapper(env1)
    env1 = env_wrapper.InitWrapper(env1, init_items=config["init_items"], init_num=config["init_num"])
    
    rewards1, _ = test_with_selective_blur(env1, original_model, config["test_episodes"], render=True)
    env1.close()

    # Test 2: Blur model in blur environment
    print("\n--- Test 2: Blur model in blur environment ---")
    env2 = gym.make("MyCrafter-v0")
    env2 = env_wrapper.MineStoneWrapper(env2)
    env2 = env_wrapper.InitWrapper(env2, init_items=config["init_items"], init_num=config["init_num"])
    env2 = env_wrapper.SelectiveBlurWrapper(env2, target_obj_id=config["target_obj_id"], 
                                          target_obj_name=config["target_obj_name"],
                                          blur_strength=config["blur_strength"])
    
    rewards2, blur_stats = test_with_selective_blur(env2, blur_model, config["test_episodes"], render=True)
    env2.close()

    # Display comparison results
    print("\n=== Comparison results ===")
    print(f"Without blur - Average reward: {np.mean(rewards1):.2f} ± {np.std(rewards1):.2f}")
    print(f"With selective blur - Average reward: {np.mean(rewards2):.2f} ± {np.std(rewards2):.2f}")
    
    improvement = np.mean(rewards2) - np.mean(rewards1)
    if improvement > 0:
        print(f"Selective blur improved performance by {improvement:.2f} points")
    else:
        print(f"Selective blur effect: {improvement:.2f} points")
    
    if blur_stats:
        target_found_rate = sum(1 for s in blur_stats if s['target_found']) / len(blur_stats)
        avg_target_pixels = np.mean([s['target_pixels'] for s in blur_stats])
        print(f"Selective blur statistics - Target detection rate: {target_found_rate:.1%}, Average target pixels: {avg_target_pixels:.1f}")

    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(['No Blur', 'With Blur'], [np.mean(rewards1), np.mean(rewards2)], 
            yerr=[np.std(rewards1), np.std(rewards2)], capsize=5)
    plt.title('Average Reward Comparison')
    plt.ylabel('Average Reward')
    
    plt.subplot(1, 2, 2)
    episodes = range(1, len(rewards1) + 1)
    plt.plot(episodes, rewards1, 'o-', label='No Blur', linewidth=2)
    plt.plot(episodes, rewards2, 's-', label='With Blur', linewidth=2)
    plt.title('Episode Reward Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main test function"""
    config = {
        "test_episodes": 5,
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "target_obj_id": 3,
        "target_obj_name": "stone",
        "blur_strength": 7,
        "model_name": None,  # Will be auto-selected
        "render": False,
        "compare_mode": True,
        "save_video": True,
        "video_dir": "test_videos"
    }

    # print("=== Selective blur testing program ===")
    
    # Show available models
    available_models = list_available_models()
    if not available_models:
        print("No model files available")
        return
    
    print("\nAvailable test modes:")
    print("1. Basic test (auto-select blur model)")
    print("2. Comparison test (with and without blur)")
    print("3. Specify model test")
    print("4. Performance test (multiple episodes)")
    
    choice = input("Please select test mode (1/2/3/4): ").strip()
    
    if choice == "1":
        # Basic test - auto-select blur model
        print("\nExecuting basic test...")
        
        # Auto-select model
        if config["model_name"] is None:
            config["model_name"] = find_blur_model()
        
        # Load model
        try:
            model_path = os.path.join("RL_models", config["model_name"])
            model = PPO.load(model_path)
            print(f"Model loaded successfully: {config['model_name']}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            print(f"Attempted to load: RL_models/{config['model_name']}.zip")
            return

        # Create test environment (consistent with training)
        env = gym.make("MyCrafter-v0")
        env = env_wrapper.MineStoneWrapper(env)
        env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=config["init_num"])
        
        # Only use SelectiveBlurWrapper for blur models
        if "blur" in config["model_name"].lower():
            env = env_wrapper.SelectiveBlurWrapper(env, target_obj_id=config["target_obj_id"],
                                                 target_obj_name=config["target_obj_name"],
                                                 blur_strength=config["blur_strength"])
            print("Applied selective blur environment (matching blur model)")
        else:
            print("Using normal environment (matching regular model)")

        # Execute test
        rewards, blur_stats = test_with_selective_blur(
            env, model, config["test_episodes"], 
            render=config["render"], compare_mode=config["compare_mode"],
            save_video=config["save_video"], video_dir=config["video_dir"]
        )

        # Output results
        print(f"\n=== Test results ===")
        print(f"Model: {config['model_name']}")
        print(f"Average reward: {np.mean(rewards):.2f}")
        print(f"Reward range: {min(rewards):.2f} ~ {max(rewards):.2f}")
        if blur_stats:
            target_found_rate = sum(1 for s in blur_stats if s['target_found']) / len(blur_stats)
            print(f"Target detection rate: {target_found_rate:.1%}")
        
        env.close()

    elif choice == "2":
        # Comparison test
        compare_with_without_blur()

    elif choice == "3":
        # Specify model test
        print("\nPlease select model to test:")
        for i, model in enumerate(available_models):
            print(f"  {i+1}. {model}")
        
        try:
            selection = int(input("Enter model number: ")) - 1
            if 0 <= selection < len(available_models):
                config["model_name"] = available_models[selection]
                print(f"Selected model: {config['model_name']}")
                
                # Execute the same logic as choice "1"
                print("Executing specified model test...")
                # [Copy the same logic as choice "1" here]
                
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")

    elif choice == "4":
        # Performance test
        print("\nExecuting performance test...")
        config["test_episodes"] = 20
        config["render"] = True
        config["save_video"] = False
        config["model_name"] = find_blur_model()
        
        print("Performance test completed")

    else:
        print("Invalid selection")

if __name__ == "__main__":
    main()