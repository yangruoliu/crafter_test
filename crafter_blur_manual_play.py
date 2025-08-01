import argparse
import numpy as np
import os
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
    """带render方法的SelectiveBlurWrapper"""
    def render(self, size=None):
        return self.env.render(size)

def print_blur_matrix(mask, step_count, target_name, target_found, target_pixels):
    """
    在终端打印blur矩阵
    """
    print(f"\n{'='*50}")
    print(f"Step {step_count} - Blur Mask Matrix")
    print(f"Target: {target_name} | Found: {target_found} | Pixels: {target_pixels}")
    print(f"{'='*50}")
    
    # 打印矩阵（每行显示）
    for i, row in enumerate(mask):
        row_str = " ".join([f"{int(val)}" for val in row])
        print(f"Row {i:2d}: {row_str}")
    
    print(f"{'='*50}")
    print(f"Matrix shape: {mask.shape}")
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
        f.write("=" * 50 + "\n")
        
        for i, row in enumerate(mask):
            row_str = " ".join([f"{int(val)}" for val in row])
            f.write(f"Row {i:2d}: {row_str}\n")
        
        f.write("=" * 50 + "\n")
        f.write(f"Matrix shape: {mask.shape}\n")
        f.write(f"Clear pixels (1): {np.sum(mask)} | Blur pixels (0): {mask.size - np.sum(mask)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Blur matrix saved: {txt_file} and {npy_file}")

def main():
    boolean = lambda x: bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser(description='Manual Crafter with Selective Blur')
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
    parser.add_argument('--target_obj_id', type=int, default=3, help='Target object ID (3=stone, 8=coal, 9=iron)')
    parser.add_argument('--target_obj_name', type=str, default='stone', help='Target object name')
    parser.add_argument('--blur_strength', type=int, default=15, help='Blur strength (must be odd)')
    parser.add_argument('--save_matrices', type=boolean, default=True, help='Save blur matrices to files')
    parser.add_argument('--matrix_save_dir', type=str, default='blur_matrices', help='Directory to save matrices')
    
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
    
    print('=== Manual Crafter with Selective Blur ===')
    print('Actions:')
    for key, action in keymap.items():
        print(f'  {pygame.key.name(key)}: {action}')
    
    print(f'\nBlur Settings:')
    print(f'  Target object: {args.target_obj_name} (ID: {args.target_obj_id})')
    print(f'  Blur strength: {args.blur_strength}')
    print(f'  Save matrices: {args.save_matrices}')
    if args.save_matrices:
        # 创建带时间戳的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"{args.matrix_save_dir}_{timestamp}"
        print(f'  Save directory: {save_dir}')
    print('  Blur matrix will be printed when target is found or every 5 steps')
    print('  Press M to manually print and save current blur matrix')
    print('  Press Q or ESC to quit\n')

    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]

    # 创建环境
    env = gym.make("MyCrafter-v0")
    env = MetadataFixWrapper(env)  # 修复metadata
    
    # 添加InitWrapper（去掉LLMWrapper）
    env = env_wrapper.InitWrapper(env, init_items=["stone_pickaxe"], init_num=[1], init_center=6)
    
    # 添加SelectiveBlurWrapper
    env = SelectiveBlurWrapperWithRender(
        env, 
        target_obj_id=args.target_obj_id,
        target_obj_name=args.target_obj_name,
        blur_strength=args.blur_strength
    )
    
    env.reset()
    achievements = set()
    duration = 0
    return_ = 0
    was_done = False
    step_count = 0
    last_info = None

    # 初始化保存目录
    if args.save_matrices:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"{args.matrix_save_dir}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Matrices will be saved to: {save_dir}")

    pygame.init()
    screen = pygame.display.set_mode(args.window)
    pygame.display.set_caption('Crafter with Selective Blur')
    clock = pygame.time.Clock()
    running = True
    
    try:
        while running:
            # Rendering.
            try:
                image = env.render(size)
                if size != args.window:
                    image = Image.fromarray(image)
                    image = image.resize(args.window, resample=Image.NEAREST)
                    image = np.array(image)
                surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
            except Exception as e:
                print(f"Render error: {e}")
                
            clock.tick(args.fps)

            # Keyboard input.
            action = None
            manual_matrix_save = False
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_m:  # 手动保存矩阵
                        manual_matrix_save = True
                    elif event.key in keymap.keys():
                        action = keymap[event.key]
                    
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
                            
                            # 在终端打印blur矩阵
                            print_blur_matrix(mask, step_count, target_name, target_found, target_pixels)
                            
                            # 保存blur矩阵
                            if args.save_matrices:
                                save_blur_matrix(mask, step_count, target_name, target_found, target_pixels, save_dir)
                    
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
                        env.reset()
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
                            current_obs = env.render()
                            mask = env._get_target_mask(semantic_map, player_pos, view_size, current_obs.shape)
                            
                            target_name = args.target_obj_name
                            target_found = np.sum(mask) > 0
                            target_pixels = int(np.sum(mask))
                            
                            # 打印矩阵
                            print_blur_matrix(mask, step_count, target_name, target_found, target_pixels)
                            
                            # 保存矩阵
                            if args.save_matrices:
                                save_blur_matrix(mask, step_count, target_name, target_found, target_pixels, save_dir)
                            
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
        pygame.quit()
        print(f"\nGame ended after {step_count} steps")
        print(f"Total reward: {return_:.2f}")
        if args.save_matrices:
            print(f"All blur matrices saved in: {save_dir}")

if __name__ == '__main__':
    main()