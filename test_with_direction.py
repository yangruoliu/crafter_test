import matplotlib.pyplot as plt
import time
from PIL import Image
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import crafter
import env_wrapper
import os
import numpy as np
from tqdm import tqdm
from model_with_attn import CustomACPolicy

# Direction names for visualization
DIRECTION_NAMES = [
    "Up", "Up-Right", "Right", "Down-Right", 
    "Down", "Down-Left", "Left", "Up-Left", "None"
]

def test_with_direction_prediction(env, model, num_episodes, render=True, show_direction=True):
    """
    Test the model and show direction predictions
    """
    print("Testing with direction prediction...")

    total_rewards = []
    direction_accuracies = []

    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        
        if render:
            plt.ion()
            if show_direction:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
            
            # Display game observation
            if isinstance(obs, dict):
                image_display = ax1.imshow(obs['obs'])
                ax1.set_title("Game View")
            else:
                image_display = ax1.imshow(obs)
                ax1.set_title("Game View")
            
            if show_direction:
                # Direction prediction display
                ax2.set_xlim(-1, 1)
                ax2.set_ylim(-1, 1)
                ax2.set_title("Direction Prediction")
                ax2.set_aspect('equal')
                # Create a circle to represent directions
                circle = plt.Circle((0, 0), 0.8, fill=False, color='black', linewidth=2)
                ax2.add_patch(circle)
                
                # Add direction labels
                directions_xy = [
                    (0, 0.9),     # Up
                    (0.64, 0.64), # Up-Right  
                    (0.9, 0),     # Right
                    (0.64, -0.64), # Down-Right
                    (0, -0.9),    # Down
                    (-0.64, -0.64), # Down-Left
                    (-0.9, 0),    # Left
                    (-0.64, 0.64), # Up-Left
                ]
                
                for i, (x, y) in enumerate(directions_xy):
                    ax2.text(x, y, DIRECTION_NAMES[i], ha='center', va='center', fontsize=8)
                
                # Arrow for prediction
                arrow = ax2.annotate('', xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='->', lw=3, color='red'))
            
            plt.show(block=False)

        done = False
        episode_reward = 0
        correct_predictions = 0
        total_predictions = 0

        while not done:
            # Get action and direction prediction
            if hasattr(model.policy, 'predict_with_direction'):
                action, direction_results = model.policy.predict_with_direction(obs, deterministic=True)
                predicted_direction = direction_results['predicted_direction'][0]
                direction_probabilities = direction_results['probabilities'][0]
            else:
                action, _ = model.predict(obs, deterministic=True)
                predicted_direction = 8  # None
                direction_probabilities = np.zeros(9)

            if isinstance(obs, dict):
                true_direction = obs['direction_label']
            else:
                true_direction = 8  # Default to None if not available

            # 确保 action 是标量整数
            if isinstance(action, np.ndarray):
                action = int(action.item())
            elif hasattr(action, '__len__') and len(action) == 1:
                action = int(action[0])
            else:
                action = int(action)

            obs, reward, done, info = env.step(action)
            episode_reward += reward

            # Calculate accuracy
            if predicted_direction == true_direction:
                correct_predictions += 1
            total_predictions += 1

            if render:
                # Update game view
                if isinstance(obs, dict):
                    img = Image.fromarray(obs['obs'])
                else:
                    img = Image.fromarray(obs)
                
                image_display.set_data(img)
                
                if show_direction:
                    # Update direction prediction
                    ax2.clear()
                    ax2.set_xlim(-1, 1)
                    ax2.set_ylim(-1, 1)
                    ax2.set_title(f"Direction Prediction: {DIRECTION_NAMES[predicted_direction]}")
                    ax2.set_aspect('equal')
                    
                    # Redraw circle and labels
                    circle = plt.Circle((0, 0), 0.8, fill=False, color='black', linewidth=2)
                    ax2.add_patch(circle)
                    
                    for i, (x, y) in enumerate(directions_xy):
                        color = 'red' if i == predicted_direction else 'black'
                        weight = 'bold' if i == predicted_direction else 'normal'
                        ax2.text(x, y, DIRECTION_NAMES[i], ha='center', va='center', 
                               fontsize=8, color=color, weight=weight)
                    
                    # Draw arrow for predicted direction
                    if predicted_direction < 8:  # Not None
                        x, y = directions_xy[predicted_direction]
                        arrow = ax2.annotate('', xy=(x*0.6, y*0.6), xytext=(0, 0),
                                           arrowprops=dict(arrowstyle='->', lw=3, color='red'))
                    
                    # Show probability distribution as text
                    prob_text = f"Confidence: {direction_probabilities[predicted_direction]:.2f}"
                    ax2.text(0, -1.2, prob_text, ha='center', va='center', fontsize=10)
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                time.sleep(0.2)

        if render:
            plt.close()

        episode_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        direction_accuracies.append(episode_accuracy)
        
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, "
              f"Direction Accuracy: {episode_accuracy:.2f}")
        total_rewards.append(episode_reward)

    return total_rewards, direction_accuracies

if __name__ == "__main__":
    config = {
        "test_episodes": 10,  # 增加到10轮测试
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "target_obj_id": 7,  # Stone object ID
        "target_obj_name": "stone",
        "render": True,
        "show_direction": True,
        "model_name": "stone_with_direction"
    }

    # Setup environment
    env = gym.make("MyCrafter-v0")
    env = env_wrapper.MineStoneWrapper(env)
    env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=config["init_num"])
    env = env_wrapper.DirectionLabelWrapper(
        env, 
        target_obj_id=config["target_obj_id"], 
        target_obj_name=config["target_obj_name"]
    )

    # Load model
    try:
        model = PPO.load(os.path.join("RL_models", config["model_name"]))
        print(f"Loaded model: {config['model_name']}")
    except:
        try:
            model = PPO.load(config["model_name"])
            print(f"Loaded model: {config['model_name']}")
        except:
            print(f"Could not load model {config['model_name']}")
            print("Please make sure the model exists and try again.")
            exit(1)

    # Test the model
    total_rewards, direction_accuracies = test_with_direction_prediction(
        env, model, config["test_episodes"], 
        render=config["render"], 
        show_direction=config["show_direction"]
    )

    # Print results
    average_reward = sum(total_rewards) / len(total_rewards)
    average_direction_accuracy = sum(direction_accuracies) / len(direction_accuracies)
    
    print(f"\n=== Test Results ===")
    print(f"Average reward over {config['test_episodes']} episodes: {average_reward:.2f}")
    print(f"Average direction prediction accuracy: {average_direction_accuracy:.2f}")
    print(f"Target object: {config['target_obj_name']} (ID: {config['target_obj_id']})")

    env.close() 