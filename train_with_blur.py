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
    
    def _get_target_mask(self, semantic_map, player_pos, view_size, image_shape):
        """
        重写目标物体遮罩创建方法，修复坐标系问题
        """
        px, py = player_pos
        view_w, view_h = view_size
        
        # 计算视野范围
        half_w, half_h = view_w // 2, view_h // 2
        x1 = max(0, px - half_w)
        y1 = max(0, py - half_h)
        x2 = min(semantic_map.shape[0], px + half_w + 1)
        y2 = min(semantic_map.shape[1], py + half_h + 1)
        
        # 提取视野区域的语义地图
        view_semantic = semantic_map[x1:x2, y1:y2]
        
        # 创建目标物体遮罩
        target_positions = (view_semantic == self.target_obj_id)
        semantic_mask = target_positions.astype(np.uint8)
        
        # 将语义遮罩缩放到图像尺寸
        img_h, img_w = image_shape[:2]
        if semantic_mask.shape[0] > 0 and semantic_mask.shape[1] > 0:
            # 直接转置：语义地图通常是 [y, x] 格式，需要转换为 [x, y] 格式以匹配图像
            semantic_mask = semantic_mask.T  # 直接转置
            
            target_mask = cv2.resize(
                semantic_mask.astype(np.float32),
                (img_w, img_h),  # OpenCV格式: (width, height)
                interpolation=cv2.INTER_NEAREST  # 使用NEAREST避免插值导致的问题
            )
            # 应用阈值以保持二值特性
            target_mask = (target_mask > 0.5).astype(np.uint8)
        else:
            target_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        return target_mask
    
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
        # 先获取原始观察结果
        original_obs, reward, done, info = self.env.step(action)
        
        # 保存原始图像
        self._last_original_obs = original_obs.copy()
        
        # 获取必要的游戏信息
        semantic_map = info.get('semantic', None)
        player_pos = info.get('player_pos', [32, 32])
        view_size = info.get('view', [9, 9])
        
        if semantic_map is not None:
            try:
                # 创建目标物体遮罩
                target_mask = self._get_target_mask(semantic_map, player_pos, view_size, original_obs.shape)
                self._last_mask = target_mask.copy()
                
                # 应用选择性模糊
                blurred_obs = self._apply_selective_blur(original_obs, target_mask)
                
                # 保存blur处理后的图像
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
                # 如果处理失败，返回原图像
                self._last_blurred_obs = original_obs.copy()
                self._last_mask = np.ones((original_obs.shape[0], original_obs.shape[1]), dtype=np.uint8)
                return original_obs, reward, done, info
        
        # 如果没有语义地图，原样返回
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
    
    # 保存为numpy文件
    npy_file = os.path.join(save_dir, f"blur_mask_step_{step_count:04d}.npy")
    np.save(npy_file, mask)
    
    # 保存为文本文件
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
    
    # 创建mask可视化（放大到与原图相同尺寸）
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
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # 对比视频尺寸：宽度 × 3
                video_width = self.video_size[0] * 3
                video_height = self.video_size[1]
                
                self.video_writer = cv2.VideoWriter(
                    self.video_path,
                    fourcc,
                    self.video_fps,
                    (video_width, video_height)
                )
                
                if self.video_writer.isOpened():
                    print(f"Video recording initialized: {self.video_path}")
                else:
                    print("Failed to initialize video writer")
                    self.video_writer = None
                    
            except Exception as e:
                print(f"Error initializing video recording: {e}")
                self.video_writer = None

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
                            # 保存矩阵
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
                                
                                # 调整尺寸
                                if comparison_img.shape[:2] != (self.video_size[1], self.video_size[0] * 3):
                                    comparison_img = cv2.resize(
                                        comparison_img,
                                        (self.video_size[0] * 3, self.video_size[1]),
                                        interpolation=cv2.INTER_NEAREST
                                    )
                                
                                # 录制视频（OpenCV使用BGR格式）
                                video_frame = cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR)
                                self.video_writer.write(video_frame)
                        
                        # 收集统计信息
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
        
        # 打印统计信息
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
    
    # 解析命令行参数
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
    
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    config = {
        "total_timesteps": args.total_timesteps,
        "save_dir": os.path.join("RL_models_zdl", f"stone_with_blur_{timestamp}"),
        "model_name": f"stone_with_blur_{timestamp}",
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

    # Start training
    print("Starting training with video recording...")
    
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
        f.write("\n=== Command line arguments ===\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"Configuration saved to: {config_path}")

    # Save latest model record
    latest_model_file = os.path.join("RL_models", "latest_blur_model.txt")
    os.makedirs("RL_models", exist_ok=True)
    with open(latest_model_file, 'w') as f:
        f.write(config["model_name"])
    print(f"Latest model record saved to: {latest_model_file}")

    env.close()
    print("=== Training completed ===")

if __name__ == "__main__":
    train_with_selective_blur()