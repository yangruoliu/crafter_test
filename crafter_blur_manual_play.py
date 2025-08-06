import argparse
import numpy as np
import os
import cv2
from datetime import datetime
from crafter import crafter
try:
    import pygame
except ImportError:
    print('Please install the pygame package to use the GUI.')
    raise

from PIL import Image
import env_wrapper
import gym
from stable_baselines3 import PPO

class MetadataFixWrapper(gym.Wrapper):
    """修复metadata问题的wrapper"""
    def __init__(self, env):
        super().__init__(env)
        # 确保metadata是一个字典
        if self.metadata is None:
            self.metadata = {
                'render_modes': ['rgb_array'],
                'render_fps': 30
            }

class SelectiveBlurWrapperWithRender(env_wrapper.SelectiveBlurWrapper):
    """带render方法的SelectiveBlurWrapper - 保存原始和blur图像，修复坐标问题"""
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

        # # 添加调试信息
        # print(f"----   Real player position: {player_pos}")
        # print(f"----   view_size: {view_size}")
        # print(f"----   Info keys: {list(info.keys())}")
        
        if semantic_map is not None:
            try:
                # 创建目标物体遮罩
                target_mask = self._get_proportional_clear_mask(semantic_map, player_pos, view_size, original_obs.shape)
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

def fix_metadata_chain(env):
    """修复整个wrapper链的metadata"""
    current_env = env
    while hasattr(current_env, 'env'):
        if current_env.metadata is None:
            current_env.metadata = {
                'render_modes': ['rgb_array'],
                'render_fps': 30
            }
        current_env = current_env.env
    
    # 修复最底层的环境
    if current_env.metadata is None:
        current_env.metadata = {
            'render_modes': ['rgb_array'],
            'render_fps': 30
        }
    
    return env

def print_blur_matrix(mask, step_count, target_name, target_found, target_pixels):
    """
    在终端打印blur矩阵
    """
    print(f"\n{'='*50}")
    print(f"Step {step_count} - Blur Mask Matrix")
    print(f"Target: {target_name} | Found: {target_found} | Pixels: {target_pixels}")
    print(f"Matrix shape: {mask.shape} (Height={mask.shape[0]}, Width={mask.shape[1]})")
    print(f"{'='*50}")
    
    # 打印矩阵（每行显示）
    for i, row in enumerate(mask):
        row_str = " ".join([f"{int(val)}" for val in row])
        print(f"Row {i:2d}: {row_str}")
    
    print(f"{'='*50}")
    print(f"Clear pixels (1): {np.sum(mask)} | Blur pixels (0): {mask.size - np.sum(mask)}")
    print(f"{'='*50}\n")

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
    
    print(f"Blur matrix saved: {txt_file} and {npy_file}")

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
            # print(f"Warning: Mask shape {mask.shape} != image shape {(h, w)}, resizing...")
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
    
    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    
    cv2.putText(comparison, "Original", (10, 20), font, font_scale, color, thickness)
    cv2.putText(comparison, "Blurred", (w + 10, 20), font, font_scale, color, thickness)
    cv2.putText(comparison, "Mask (R=blur, G=clear)", (2*w + 10, 20), font, font_scale, color, thickness)
    cv2.putText(comparison, f"Step {step_count} - {target_name}", (10, h - 10), font, font_scale, color, thickness)
    
    return comparison

def main():
    boolean = lambda x: bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser(description='Manual Crafter with Selective Blur and Video Recording')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
    parser.add_argument('--view', type=int, nargs=2, default=(9, 9))
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--health', type=int, default=9)
    parser.add_argument('--window', type=int, nargs=2, default=(600, 600))
    parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
    parser.add_argument('--record', type=str, default=None)
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--wait', type=boolean, default=False)
    parser.add_argument('--death', type=str, default='reset', choices=['continue', 'reset', 'quit'])
    
    # Blur相关参数
    parser.add_argument('--target_obj_id', type=int, default=6, help='Target object ID (3=stone, 8=coal, 9=iron)')
    parser.add_argument('--target_obj_name', type=str, default='tree', help='Target object name')
    parser.add_argument('--blur_strength', type=int, default=6, help='Blur strength (must be odd)')
    parser.add_argument('--save_matrices', type=boolean, default=True, help='Save blur matrices to files')
    parser.add_argument('--matrix_save_dir', type=str, default='blur_matrices', help='Directory to save matrices')
    
    # 视频录制参数
    parser.add_argument('--save_video', type=boolean, default=True, help='Save video recording')
    parser.add_argument('--video_fps', type=int, default=10, help='Video recording FPS')
    parser.add_argument('--video_dir', type=str, default='game_videos', help='Directory to save videos')
    parser.add_argument('--show_comparison', type=boolean, default=True, help='Show comparison view')
    
    args = parser.parse_args()

    keymap = {
        pygame.K_a: 'move_left',
        pygame.K_d: 'move_right',
        pygame.K_w: 'move_up',
        pygame.K_s: 'move_down',
        pygame.K_SPACE: 'do',
        pygame.K_TAB: 'sleep',
        pygame.K_r: 'place_stone',
        pygame.K_t: 'place_table',
        pygame.K_f: 'place_furnace',
        pygame.K_p: 'place_plant',
        pygame.K_1: 'make_wood_pickaxe',
        pygame.K_2: 'make_stone_pickaxe',
        pygame.K_3: 'make_iron_pickaxe',
        pygame.K_4: 'make_wood_sword',
        pygame.K_5: 'make_stone_sword',
        pygame.K_6: 'make_iron_sword',
    }
    
    print('=== Manual Crafter with Selective Blur and Video Recording ===')
    print('Actions:')
    for key, action in keymap.items():
        print(f'  {pygame.key.name(key)}: {action}')
    
    print(f'\nBlur Settings:')
    print(f'  Target object: {args.target_obj_name} (ID: {args.target_obj_id})')
    print(f'  Blur strength: {args.blur_strength}')
    print(f'  Save matrices: {args.save_matrices}')
    print(f'  Show comparison: {args.show_comparison}')
    
    print(f'\nVideo Settings:')
    print(f'  Save video: {args.save_video}')
    print(f'  Video FPS: {args.video_fps}')
    
    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.save_matrices:
        matrix_save_dir = f"{args.matrix_save_dir}_{timestamp}"
        print(f'  Matrix save directory: {matrix_save_dir}')
    
    if args.save_video:
        video_save_dir = f"{args.video_dir}_{timestamp}"
        os.makedirs(video_save_dir, exist_ok=True)
        video_path = os.path.join(video_save_dir, f"gameplay_{timestamp}.mp4")
        print(f'  Video save path: {video_path}')
    
    print('  Blur matrix will be printed when target is found or every 5 steps')
    print('  Press M to manually print and save current blur matrix')
    print('  Press V to toggle video recording on/off')
    print('  Press C to toggle comparison view')
    print('  Press Q or ESC to quit\n')

    # 调整窗口大小以适应对比视图
    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]
    
    if args.show_comparison:
        # 对比模式：宽度 × 3
        display_window = (args.window[0] * 3, args.window[1])
    else:
        display_window = tuple(args.window)

    # 创建环境
    print("Creating environment...")
    env = gym.make("MyCrafter-v0")
    print("Environment created, fixing metadata...")
    
    # 首先修复基础环境的metadata
    env = MetadataFixWrapper(env)
    print("Base metadata fixed")
    
    # 添加InitWrapper
    env = env_wrapper.InitWrapper(env, init_items=["stone_pickaxe"], init_num=[1], init_center=6)
    print("InitWrapper added")
    
    # 添加SelectiveBlurWrapper
    env = SelectiveBlurWrapperWithRender(
        env, 
        target_obj_id=args.target_obj_id,
        target_obj_name=args.target_obj_name,
        blur_strength=args.blur_strength
    )
    print("SelectiveBlurWrapper added")
    
    # 修复整个wrapper链的metadata
    env = fix_metadata_chain(env)
    print("Metadata chain fixed")
    
    # 重置环境
    print("Resetting environment...")
    obs = env.reset()
    print("Environment reset complete")
    
    achievements = set()
    duration = 0
    return_ = 0
    was_done = False
    step_count = 0
    last_info = None
    show_comparison = args.show_comparison

    # 初始化保存目录
    if args.save_matrices:
        matrix_save_dir = f"{args.matrix_save_dir}_{timestamp}"
        os.makedirs(matrix_save_dir, exist_ok=True)
        print(f"Matrices will be saved to: {matrix_save_dir}")

    # 初始化视频录制
    video_writer = None
    recording_active = args.save_video
    
    if args.save_video:
        try:
            print("Initializing video recording...")
            # 获取第一帧来确定视频尺寸
            initial_frame = obs.copy()
            print(f"Initial frame shape: {initial_frame.shape}")
            
            if size != args.window:
                initial_frame = Image.fromarray(initial_frame)
                initial_frame = initial_frame.resize(args.window, resample=Image.NEAREST)
                initial_frame = np.array(initial_frame)
                print(f"Resized frame shape: {initial_frame.shape}")
            
            # 如果是对比模式，调整视频尺寸
            if show_comparison:
                video_width = initial_frame.shape[1] * 3
                video_height = initial_frame.shape[0]
            else:
                video_width = initial_frame.shape[1]
                video_height = initial_frame.shape[0]
            
            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                video_path, 
                fourcc, 
                args.video_fps, 
                (video_width, video_height)
            )
            
            if video_writer.isOpened():
                print(f"Video recording initialized successfully: {video_path}")
                print(f"Video size: {video_width}x{video_height}")
            else:
                print("Failed to initialize video writer")
                video_writer = None
                recording_active = False
                
        except Exception as e:
            print(f"Error initializing video recording: {e}")
            video_writer = None
            recording_active = False

    pygame.init()
    screen = pygame.display.set_mode(display_window)
    pygame.display.set_caption('Crafter with Selective Blur - Recording Enabled' if args.save_video else 'Crafter with Selective Blur')
    clock = pygame.time.Clock()
    running = True
    
    print("Starting game loop...")
    
    try:
        while running:
            # Rendering - 获取原始、blur图像和mask
            try:
                # 获取原始图像和blur图像
                original_obs = env.get_original_obs() if hasattr(env, 'get_original_obs') else obs
                blurred_obs = obs  # step返回的是blur处理后的观察结果
                current_mask = env.get_last_mask() if hasattr(env, 'get_last_mask') else None
                
                # 调整大小
                if size != args.window:
                    original_display = Image.fromarray(original_obs)
                    original_display = original_display.resize(args.window, resample=Image.NEAREST)
                    original_display = np.array(original_display)
                    
                    blurred_display = Image.fromarray(blurred_obs)
                    blurred_display = blurred_display.resize(args.window, resample=Image.NEAREST)
                    blurred_display = np.array(blurred_display)
                else:
                    original_display = original_obs
                    blurred_display = blurred_obs
                
                # 准备显示图像
                if show_comparison:
                    # 创建对比图像：原图 | blur图 | mask
                    display_image = create_comparison_image(
                        original_display, 
                        blurred_display, 
                        current_mask,  # 使用直接获取的mask
                        args.target_obj_name, 
                        step_count
                    )
                else:
                    # 只显示blur图像
                    display_image = blurred_display
                
                # 显示到pygame窗口
                surface = pygame.surfarray.make_surface(display_image.transpose((1, 0, 2)))
                screen.blit(surface, (0, 0))
                
                # 添加录制状态指示
                if recording_active and video_writer is not None:
                    # 在右上角添加红色录制指示
                    pygame.draw.circle(screen, (255, 0, 0), (display_window[0] - 30, 30), 10)
                    font = pygame.font.Font(None, 24)
                    text = font.render("REC", True, (255, 255, 255))
                    screen.blit(text, (display_window[0] - 60, 45))
                
                pygame.display.flip()
                
                # 保存视频帧
                if recording_active and video_writer is not None:
                    # 转换为BGR格式（OpenCV格式）
                    frame_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                    
            except Exception as e:
                print(f"Render error: {e}")
                
            clock.tick(args.fps)

            # Keyboard input.
            action = None
            manual_matrix_save = False
            toggle_recording = False
            toggle_comparison = False
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_m:  # 手动保存矩阵
                        manual_matrix_save = True
                    elif event.key == pygame.K_v:  # 切换录制状态
                        toggle_recording = True
                    elif event.key == pygame.K_c:  # 切换对比视图
                        toggle_comparison = True
                    elif event.key in keymap.keys():
                        action = keymap[event.key]
            
            # 处理录制切换
            if toggle_recording:
                if video_writer is not None:
                    recording_active = not recording_active
                    status = "ON" if recording_active else "OFF"
                    print(f"Video recording toggled {status}")
                    # 更新窗口标题
                    title = f'Crafter with Selective Blur - Recording {"ON" if recording_active else "OFF"}'
                    pygame.display.set_caption(title)
                else:
                    print("Video writer not initialized")
            
            # 处理对比视图切换
            if toggle_comparison:
                show_comparison = not show_comparison
                print(f"Comparison view: {'ON' if show_comparison else 'OFF'}")
                # 调整窗口大小
                if show_comparison:
                    new_window = (args.window[0] * 3, args.window[1])
                else:
                    new_window = tuple(args.window)
                screen = pygame.display.set_mode(new_window)
                    
            if action is None and not manual_matrix_save:
                pressed = pygame.key.get_pressed()
                for key, action_name in keymap.items():
                    if pressed[key]:
                        action = action_name
                        break
                else:
                    if args.wait:
                        continue
                    else:
                        action = 'noop'

            # Environment step.
            if action is not None:
                print(f"Action: {action}")
                obs, reward, done, info = env.step(env.action_names.index(action))
                duration += 1
                step_count += 1
                last_info = info

                # 处理selective blur信息
                blur_info = info.get('selective_blur', None)
                if blur_info is not None:
                    target_found = blur_info.get('target_found', False)
                    target_pixels = blur_info.get('target_pixels', 0)
                    target_name = blur_info.get('target_obj_name', 'unknown')
                    
                    # 每5步或找到目标时，打印并保存blur矩阵
                    if target_found or step_count % 5 == 0:
                        semantic_map = info.get('semantic', None)
                        player_pos = info.get('player_pos', [32, 32])
                        view_size = info.get('view', [9, 9])
                        
                        if semantic_map is not None:
                            mask = env._get_target_mask(semantic_map, player_pos, view_size, obs.shape)
                            
                            # # 在终端打印blur矩阵
                            # print_blur_matrix(mask, step_count, target_name, target_found, target_pixels)
                            
                            # 保存blur矩阵
                            if args.save_matrices:
                                save_blur_matrix(mask, step_count, target_name, target_found, target_pixels, matrix_save_dir)
                    
                    # 打印基本信息
                    if target_found:
                        print(f"✓ Step {step_count}: Target {target_name} found! Pixels: {target_pixels}")

                if hasattr(env, 'cur_step') and env.cur_step > 0 and env.cur_step % 100 == 0:
                    print(f'Time step: {env.cur_step}')
                if reward:
                    print(f'Reward: {reward}')
                    return_ += reward

                # Episode end.
                if done and not was_done:
                    was_done = True
                    print('Episode done!')
                    print('Duration:', duration)
                    print('Return:', return_)
                    if args.death == 'quit':
                        running = False
                    if args.death == 'reset':
                        print('\nStarting a new episode.')
                        obs = env.reset()
                        achievements = set()
                        was_done = False
                        duration = 0
                        return_ = 0
                        step_count = 0
                    if args.death == 'continue':
                        pass
            
            # 手动保存blur矩阵
            if manual_matrix_save and last_info is not None:
                try:
                    blur_info = last_info.get('selective_blur', None)
                    if blur_info is not None:
                        semantic_map = last_info.get('semantic', None)
                        player_pos = last_info.get('player_pos', [32, 32])
                        view_size = last_info.get('view', [9, 9])
                        
                        if semantic_map is not None:
                            mask = env._get_target_mask(semantic_map, player_pos, view_size, obs.shape)
                            
                            target_name = args.target_obj_name
                            target_found = np.sum(mask) > 0
                            target_pixels = int(np.sum(mask))
                            
                            # # 打印矩阵
                            # print_blur_matrix(mask, step_count, target_name, target_found, target_pixels)
                            
                            # 保存矩阵
                            if args.save_matrices:
                                save_blur_matrix(mask, step_count, target_name, target_found, target_pixels, matrix_save_dir)
                            
                            print("Manual matrix print and save completed!")
                        else:
                            print("No semantic map available for manual save")
                    else:
                        print("No blur info available for manual save")
                except Exception as e:
                    print(f"Error during manual matrix save: {e}")

    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭视频录制
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved: {video_path}")
        
        pygame.quit()
        print(f"\nGame ended after {step_count} steps")
        print(f"Total reward: {return_:.2f}")
        if args.save_matrices:
            print(f"All blur matrices saved in: {matrix_save_dir}")
        if args.save_video and video_writer is not None:
            print(f"Video saved: {video_path}")

if __name__ == '__main__':
    main()