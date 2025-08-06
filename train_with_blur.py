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
import cv2
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback
import time
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import json

class RewardPlotCallback(BaseCallback):
    """训练过程中的reward绘制回调 """
    
    def __init__(self, save_interval=1000, plot_filename="training_rewards.png", verbose=0):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.plot_filename = plot_filename
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """每步回调 - 使用更可靠的方法获取训练信息"""
        # Überprüfen Sie jeden Umgebung in der VecEnv
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                # Informationen aus dem info-Wörterbuch abrufen, wenn eine Episode beendet ist
                info = self.locals["infos"][i]
                if 'episode' in info:
                    episode_info = info['episode']
                    self.episode_rewards.append(episode_info['r'])
                    self.episode_lengths.append(episode_info['l'])
                    
                    # Diagramm in regelmäßigen Abständen speichern
                    if len(self.episode_rewards) % self.save_interval == 0:
                        self._plot_rewards()
        return True

    def _plot_rewards(self):
        """绘制reward折线图 """
        if len(self.episode_rewards) < 5:
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from datetime import datetime
            import json

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            episodes = range(1, len(self.episode_rewards) + 1)

            plt.plot(episodes, self.episode_rewards, 'b-', linewidth=1, alpha=0.3, label='Raw')

            # 移动平均线
            if len(self.episode_rewards) >= 10:
                window_size = min(20, len(self.episode_rewards) // 3)
                moving_avg = np.convolve(
                    self.episode_rewards,
                    np.ones(window_size) / window_size,
                    mode='valid'
                )


                x_axis_moving_avg = range(window_size, len(self.episode_rewards) - len(moving_avg) + window_size + len(
                    moving_avg))  #

                x_axis_for_moving_avg = np.arange(window_size, len(self.episode_rewards) + 1)

                plt.plot(
                    x_axis_for_moving_avg,
                    moving_avg,
                    'r-',
                    linewidth=2,
                    label=f'{window_size}-episode Avg'
                )

            plt.title('Training Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 副图 - Reward分布
            plt.subplot(1, 2, 2)
            plt.hist(
                self.episode_rewards,
                bins=min(15, len(set(self.episode_rewards)) if len(set(self.episode_rewards)) > 1 else 1),
                alpha=0.7,
                color='green',
                edgecolor='black'
            )
            plt.title('Reward Distribution')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

            if self.episode_rewards:
                plt.text(
                    0.05, 0.95,
                    f"Episodes: {len(self.episode_rewards)}\n"
                    f"Avg: {np.mean(self.episode_rewards):.2f}\n"
                    f"Max: {max(self.episode_rewards):.2f}\n"
                    f"Min: {min(self.episode_rewards):.2f}",
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

            plt.tight_layout()
            plt.savefig(self.plot_filename, dpi=200, bbox_inches='tight')
            plt.close()

            data = {
                'episode_rewards': [float(r) for r in self.episode_rewards],
                'episode_lengths': [int(l) for l in self.episode_lengths],
                'total_steps': self.num_timesteps,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.plot_filename.replace('.png', '.json'), 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Fehler beim Plotten der Belohnungen: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_training_end(self) -> None:
        """训练结束时绘制最终图表"""
        self._plot_rewards()
        print(f"Finaler Belohnungs-Plot gespeichert als: {self.plot_filename}")


class SelectiveBlurWrapperWithRender(env_wrapper.SelectiveBlurWrapper):
    """带render方法的SelectiveBlurWrapper - 用于训练时的数据保存"""
    def __init__(self, env, target_obj_id, target_obj_name="stone", blur_strength=5):
        super().__init__(env, target_obj_id, target_obj_name, blur_strength)
        # 确保metadata正确传递
        if self.metadata is None:
            self.metadata = {
                'render_modes': ['rgb_array'],
                'render_fps': 30
            }
        # 保存原始和处理后的图像
        self._last_original_obs = None
        self._last_blurred_obs = None
        self._last_mask = None
    
    # def _get_target_mask(self, semantic_map, player_pos, view_size, image_shape):
    #     """
    #     重写目标物体遮罩创建方法，修复坐标系问题
    #     """
    #     px, py = player_pos
    #     view_w, view_h = view_size
        
    #     # 计算视野范围
    #     half_w, half_h = view_w // 2, view_h // 2
    #     x1 = max(0, px - half_w)
    #     y1 = max(0, py - half_h)
    #     x2 = min(semantic_map.shape[0], px + half_w + 1)
    #     y2 = min(semantic_map.shape[1], py + half_h + 1)
        
    #     # 提取视野区域的语义地图
    #     view_semantic = semantic_map[x1:x2, y1:y2]
        
    #     # 创建目标物体遮罩
    #     target_positions = (view_semantic == self.target_obj_id)
    #     semantic_mask = target_positions.astype(np.uint8)
        
    #     # 将语义遮罩缩放到图像尺寸
    #     img_h, img_w = image_shape[:2]
    #     if semantic_mask.shape[0] > 0 and semantic_mask.shape[1] > 0:
    #         # 直接转置：语义地图通常是 [y, x] 格式，需要转换为 [x, y] 格式以匹配图像
    #         semantic_mask = semantic_mask.T  # 直接转置
            
    #         target_mask = cv2.resize(
    #             semantic_mask.astype(np.float32),
    #             (img_w, img_h),  # OpenCV格式: (width, height)
    #             interpolation=cv2.INTER_NEAREST  # 使用NEAREST避免插值导致的问题
    #         )
    #         # 应用阈值以保持二值特性
    #         target_mask = (target_mask > 0.5).astype(np.uint8)
    #     else:
    #         target_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
    #     return target_mask
    
    def render(self, size=None):
        """返回blur处理后的图像"""
        if hasattr(self, '_last_blurred_obs') and self._last_blurred_obs is not None:
            return self._last_blurred_obs
        return self.env.render(size)
    
    def get_original_obs(self):
        """获取原始观察结果"""
        return self._last_original_obs
    
    def get_last_mask(self):
        """获取最后的mask"""
        return self._last_mask
    
    def step(self, action):
        """重写step方法，同时保存原始和blur处理的图像"""

        original_obs, reward, done, info = self.env.step(action)
        

        self._last_original_obs = original_obs.copy()
        

        semantic_map = info.get('semantic', None)
        player_pos = info.get('player_pos', [32, 32])
        view_size = info.get('view', [9, 9])
        
        if semantic_map is not None:
            try:
                # 创建目标物体遮罩
                target_mask = self._get_proportional_clear_mask(semantic_map, player_pos, view_size, original_obs.shape)
                self._last_mask = target_mask.copy()

                blurred_obs = self._apply_selective_blur(original_obs, target_mask)
                

                self._last_blurred_obs = blurred_obs.copy()
                
                # 添加调试信息
                target_found = np.sum(target_mask) > 0
                info['selective_blur'] = {
                    'target_obj_id': self.target_obj_id,
                    'target_obj_name': self.target_obj_name,
                    'target_found': target_found,
                    'target_pixels': int(np.sum(target_mask)),
                    'blur_strength': self.blur_strength
                }
                
                return blurred_obs, reward, done, info
                
            except Exception as e:
                print(f"SelectiveBlurWrapper error: {e}")

                self._last_blurred_obs = original_obs.copy()
                self._last_mask = np.ones((original_obs.shape[0], original_obs.shape[1]), dtype=np.uint8)
                return original_obs, reward, done, info

        self._last_blurred_obs = original_obs.copy()
        self._last_mask = np.ones((original_obs.shape[0], original_obs.shape[1]), dtype=np.uint8)
        return original_obs, reward, done, info
    
    def reset(self, **kwargs):
        """重置时清除缓存"""
        obs = self.env.reset(**kwargs)
        self._last_original_obs = obs.copy()
        self._last_blurred_obs = obs.copy()
        self._last_mask = np.ones((obs.shape[0], obs.shape[1]), dtype=np.uint8)
        return obs

def save_blur_matrix(mask, step_count, target_name, target_found, target_pixels, save_dir="blur_matrices"):
    """
    保存blur矩阵到文件
    """
    os.makedirs(save_dir, exist_ok=True)
    

    npy_file = os.path.join(save_dir, f"blur_mask_step_{step_count:04d}.npy")
    np.save(npy_file, mask)
    

    txt_file = os.path.join(save_dir, f"blur_mask_step_{step_count:04d}.txt")
    with open(txt_file, 'w') as f:
        f.write(f"Step {step_count} - Blur Mask Matrix\n")
        f.write(f"Target: {target_name} | Found: {target_found} | Pixels: {target_pixels}\n")
        f.write(f"Matrix shape: {mask.shape} (Height={mask.shape[0]}, Width={mask.shape[1]})\n")
        f.write("=" * 50 + "\n")
        
        for i, row in enumerate(mask):
            row_str = " ".join([f"{int(val)}" for val in row])
            f.write(f"Row {i:2d}: {row_str}\n")
        
        f.write("=" * 50 + "\n")
        f.write(f"Clear pixels (1): {np.sum(mask)} | Blur pixels (0): {mask.size - np.sum(mask)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def create_comparison_image(original, blurred, mask, target_name, step_count):
    """
    创建对比图像：原图 | blur图 | mask
    """
    h, w = original.shape[:2]

    mask_visual = np.zeros((h, w, 3), dtype=np.uint8)
    if mask is not None:
        # 确保mask尺寸正确
        if mask.shape != (h, w):
            mask_resized = cv2.resize(
                mask.astype(np.float32), 
                (w, h),  # OpenCV格式: (width, height)
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask.astype(np.float32)
        
        # 创建颜色映射：0=红色(模糊区域)，1=绿色(清晰区域)
        mask_visual[:, :, 0] = (1 - mask_resized) * 255  # 红色通道：模糊区域
        mask_visual[:, :, 1] = mask_resized * 255         # 绿色通道：清晰区域
        mask_visual[:, :, 2] = 0                          # 蓝色通道：始终为0
    
    # 水平拼接三个图像
    comparison = np.hstack([original, blurred, mask_visual])
    return comparison

class TrainingVideoCallback(BaseCallback):
    """训练过程中的视频录制和数据保存回调"""
    
    def __init__(self, 
                 save_video=True, 
                 save_matrices=True, 
                 save_interval=100,
                 video_size=(600, 600),
                 video_fps=10,
                 timestamp=None,
                 verbose=0):
        super().__init__(verbose)
        
        self.save_video = save_video
        self.save_matrices = save_matrices
        self.save_interval = save_interval
        self.video_size = video_size
        self.video_fps = video_fps
        self.timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        
        # 创建保存目录
        if save_video:
            self.video_dir = f"game_videos_{self.timestamp}"
            os.makedirs(self.video_dir, exist_ok=True)
            self.video_path = os.path.join(self.video_dir, f"training_{self.timestamp}.mp4")
            
        if save_matrices:
            self.matrix_dir = f"blur_matrices_{self.timestamp}"
            os.makedirs(self.matrix_dir, exist_ok=True)
            
        # 视频录制
        self.video_writer = None
        self.step_count = 0
        self.blur_stats = []
        
    def _on_training_start(self) -> None:
        """训练开始时初始化视频录制"""
        if self.save_video:
            try:

                codecs = ['mp4v', 'avc1', 'H264', 'XVID']
                self.video_writer = None
                
                for codec in codecs:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video_width = self.video_size[0] * 3
                        video_height = self.video_size[1]

                        video_ext = '.mp4' if codec in ['mp4v', 'avc1', 'H264'] else '.avi'
                        video_path = self.video_path.replace('.mp4', video_ext)
                        
                        self.video_writer = cv2.VideoWriter(
                            video_path,
                            fourcc,
                            self.video_fps,
                            (video_width, video_height)
                        )
                        
                        if self.video_writer.isOpened():
                            print(f"Video recording initialized with codec {codec}: {video_path}")
                            self.video_path = video_path  # 更新路径
                            break
                        else:
                            self.video_writer.release()
                            self.video_writer = None
                            
                    except Exception as e:
                        print(f"Failed to initialize with codec {codec}: {e}")
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                
                if self.video_writer is None:
                    print("Warning: Failed to initialize video writer with any codec")
                    print("Video recording will be disabled")
                    self.save_video = False
                    
            except Exception as e:
                print(f"Error initializing video recording: {e}")
                self.video_writer = None
                self.save_video = False

    def _on_step(self) -> bool:
        """每步回调"""
        self.step_count += 1
        
        # 获取环境信息
        if hasattr(self.training_env, 'get_attr'):
            try:
                # 获取blur wrapper
                envs = self.training_env.get_attr('env')
                if envs and len(envs) > 0:
                    blur_wrapper = None
                    env = envs[0]
                    
                    # 找到SelectiveBlurWrapper
                    while hasattr(env, 'env'):
                        if isinstance(env, SelectiveBlurWrapperWithRender):
                            blur_wrapper = env
                            break
                        env = env.env
                    
                    if blur_wrapper:
                        original_obs = blur_wrapper.get_original_obs()
                        blurred_obs = blur_wrapper.render()
                        mask = blur_wrapper.get_last_mask()
                        
                        if original_obs is not None and blurred_obs is not None and mask is not None:

                            if self.save_matrices and self.step_count % self.save_interval == 0:
                                target_found = np.sum(mask) > 0
                                target_pixels = int(np.sum(mask))
                                save_blur_matrix(
                                    mask, 
                                    self.step_count, 
                                    blur_wrapper.target_obj_name,
                                    target_found,
                                    target_pixels,
                                    self.matrix_dir
                                )
                            
                            # 录制视频
                            if self.save_video and self.video_writer and self.video_writer.isOpened():
                                comparison_img = create_comparison_image(
                                    original_obs, 
                                    blurred_obs, 
                                    mask,
                                    blur_wrapper.target_obj_name,
                                    self.step_count
                                )

                                if comparison_img.shape[:2] != (self.video_size[1], self.video_size[0] * 3):
                                    comparison_img = cv2.resize(
                                        comparison_img,
                                        (self.video_size[0] * 3, self.video_size[1]),
                                        interpolation=cv2.INTER_NEAREST
                                    )

                                video_frame = cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR)
                                self.video_writer.write(video_frame)

                        if hasattr(blur_wrapper, '_last_info'):
                            info = blur_wrapper._last_info
                            if 'selective_blur' in info:
                                blur_info = info['selective_blur']
                                self.blur_stats.append({
                                    'step': self.num_timesteps,
                                    'target_found': blur_info.get('target_found', False),
                                    'target_pixels': blur_info.get('target_pixels', 0)
                                })
                        
            except Exception as e:
                if self.verbose > 0:
                    print(f"Callback error: {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """训练结束时清理资源"""
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved: {self.video_path}")

        if self.blur_stats:
            target_found_rate = sum(1 for s in self.blur_stats if s['target_found']) / len(self.blur_stats)
            avg_target_pixels = np.mean([s['target_pixels'] for s in self.blur_stats])
            print(f"\n=== Selective blur statistics ===")
            print(f"Target object detection rate: {target_found_rate:.2%}")
            print(f"Average target pixels: {avg_target_pixels:.1f}")
            print(f"Total steps processed: {len(self.blur_stats)}")

def train_with_selective_blur():
    """
    Train agent with selective blur with video recording and data saving
    """

    parser = argparse.ArgumentParser(description='Train PPO with selective blur')
    parser.add_argument('--target_obj_id', type=int, default=3, help='Target object ID (3=stone)')
    parser.add_argument('--target_obj_name', type=str, default='stone', help='Target object name')
    parser.add_argument('--blur_strength', type=int, default=3, help='Blur strength (odd number)')
    parser.add_argument('--total_timesteps', type=int, default=2000000, help='Total training timesteps')
    parser.add_argument('--save_video', type=bool, default=True, help='Save training video')
    parser.add_argument('--save_matrices', type=bool, default=True, help='Save blur matrices')
    parser.add_argument('--video_size', type=int, nargs=2, default=[600, 600], help='Video size')
    parser.add_argument('--video_fps', type=int, default=10, help='Video frame rate')
    parser.add_argument('--save_interval', type=int, default=100, help='Save matrices every N steps')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    config = {
        "total_timesteps": args.total_timesteps,
        "save_dir": os.path.join("RL_models_zdl", f"stone_with_blur_v1"),
        "model_name": f"stone_with_blur_v1",
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "target_obj_id": args.target_obj_id,
        "target_obj_name": args.target_obj_name,
        "blur_strength": args.blur_strength,
        "timestamp": timestamp
    }

    os.makedirs("RL_models_zdl", exist_ok=True)

    print("=== Starting selective blur training with video recording ===")
    print(f"Target object: {config['target_obj_name']} (ID: {config['target_obj_id']})")
    print(f"Blur strength: {config['blur_strength']}")
    print(f"Training steps: {config['total_timesteps']}")
    print(f"Save directory: {config['save_dir']}")
    print(f"Timestamp: {timestamp}")
    print(f"Video recording: {args.save_video}")
    print(f"Matrix saving: {args.save_matrices}")

    # Create environment
    env = gym.make("MyCrafter-v0")
    print("Base environment created")

    # Apply existing wrappers
    env = env_wrapper.MineStoneWrapper(env)
    env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=config["init_num"])
    print("Task wrappers applied")

    # Apply selective blur wrapper with render capability
    env = SelectiveBlurWrapperWithRender(
        env,
        target_obj_id=config["target_obj_id"],
        target_obj_name=config["target_obj_name"],
        blur_strength=config["blur_strength"]
    )
    print("Selective blur wrapper with render applied")

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
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        normalize_advantage=True
    )
    print("PPO model created")

    # Create video recording callback
    callback = TrainingVideoCallback(
        save_video=args.save_video,
        save_matrices=args.save_matrices,
        save_interval=args.save_interval,
        video_size=tuple(args.video_size),
        video_fps=args.video_fps,
        timestamp=timestamp,
        verbose=1
    )

    reward_callback = RewardPlotCallback(
        save_interval=1000,
        plot_filename=f"training_rewards_{timestamp}.png",
        verbose=1
    )

    # Start training
    print("Starting training with video recording...")
    
    model.learn(
        total_timesteps=config["total_timesteps"], 
        progress_bar=True,
        callback=[callback, reward_callback]
    )

    # Save model
    model.save(config["save_dir"])
    
    config_path = config["save_dir"] + "_config.txt"
    with open(config_path, 'w') as f:
        f.write("=== Selective blur training configuration ===\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nent_coef: {args.ent_coef}\n")
        f.write(f"learning_rate: {args.learning_rate}\n")
    
    env.close()
    print("=== Training completed ===")

if __name__ == "__main__":
    train_with_selective_blur()