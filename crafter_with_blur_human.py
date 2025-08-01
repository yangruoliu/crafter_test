import argparse
import numpy as np
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

def print_blur_matrix(mask, step_count, target_name, target_found, target_pixels):
    """
    在终端打印blur矩阵
    
    Args:
        mask: blur mask矩阵
        step_count: 当前步数
        target_name: 目标物体名称
        target_found: 是否找到目标
        target_pixels: 目标像素数量
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

def main():
    boolean = lambda x: bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser(description='Human demonstrations with selective blur')
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
    parser.add_argument('--blur_strength', type=int, default=5, help='Blur strength (must be odd)')
    parser.add_argument('--enable_blur', type=boolean, default=True, help='Enable selective blur')
    
    args = parser.parse_args()

    # 按键映射（与human_demonstrations.py保持一致）
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
    
    print('=== Human Demonstrations with Selective Blur ===')
    print('Actions:')
    for key, action in keymap.items():
        print(f'  {pygame.key.name(key)}: {action}')
    
    if args.enable_blur:
        print(f'\nBlur Settings:')
        print(f'  Target object: {args.target_obj_name} (ID: {args.target_obj_id})')
        print(f'  Blur strength: {args.blur_strength}')
        print('  Blur matrix will be printed when target is found or every 10 steps')
    print('  Press M to manually print current blur matrix')
    print('  Press Q or ESC to quit\n')

    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]

    # 创建环境（按照human_demonstrations.py的方式）
    env = gym.make("MyCrafter-v0")
    
    # 可选：添加LLMWrapper（注释掉以避免依赖问题）
    # env = env_wrapper.LLMWrapper(env, model="deepseek-chat")
    
    # 添加InitWrapper
    env = env_wrapper.InitWrapper(env, init_items=["stone_pickaxe"], init_num=[1], init_center=6)
    
    # 可选：添加NavigationWrapper
    # env = env_wrapper.NavigationWrapper(env, obj_index=9)
    
    # 重要：添加SelectiveBlurWrapper（如果启用）
    if args.enable_blur:

        class SelectiveBlurWrapperWithRender(env_wrapper.SelectiveBlurWrapper):
            def render(self, size=None):
                """确保render方法正确传递"""
                return self.env.render(size)
        
        env = SelectiveBlurWrapperWithRender(
            env, 
            target_obj_id=args.target_obj_id,
            target_obj_name=args.target_obj_name,
            blur_strength=args.blur_strength
        )
        # env = env_wrapper.SelectiveBlurWrapper(
        #     env, 
        #     target_obj_id=args.target_obj_id,
        #     target_obj_name=args.target_obj_name,
        #     blur_strength=args.blur_strength
        # )
    
    env.reset()
    achievements = set()
    duration = 0
    return_ = 0
    was_done = False
    step_count = 0  # 添加步数计数

    # 初始化pygame
    pygame.init()
    screen = pygame.display.set_mode(args.window)
    pygame.display.set_caption('Crafter with Selective Blur - Human Control')
    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            # Rendering.
            image = env.render(size)
            if size != args.window:
                image = Image.fromarray(image)
                image = image.resize(args.window, resample=Image.NEAREST)
                image = np.array(image)
            surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(args.fps)

            # Keyboard input.
            action = None
            manual_print_mask = False
            pygame.event.pump()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_m:  # 手动打印mask
                        manual_print_mask = True
                    elif event.key in keymap.keys():
                        action = keymap[event.key]
            
            if action is None and not manual_print_mask:
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

                # 处理selective blur信息
                if args.enable_blur:
                    blur_info = info.get('selective_blur', None)
                    if blur_info is not None:
                        target_found = blur_info.get('target_found', False)
                        target_pixels = blur_info.get('target_pixels', 0)
                        target_name = blur_info.get('target_obj_name', 'unknown')
                        
                        # 如果找到目标或者每10步，打印blur矩阵
                        if target_found or step_count % 10 == 0:
                            semantic_map = info.get('semantic', None)
                            player_pos = info.get('player_pos', [32, 32])
                            view_size = info.get('view', [9, 9])
                            
                            if semantic_map is not None:
                                mask = env._get_target_mask(semantic_map, player_pos, view_size, obs.shape)
                                print_blur_matrix(mask, step_count, target_name, target_found, target_pixels)
                        
                        # 打印基本信息
                        if target_found:
                            print(f"✓ Step {step_count}: Target {target_name} found! Pixels: {target_pixels}")

                # 步数统计
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
                        step_count = 0  # 重置步数
                    if args.death == 'continue':
                        pass
            
            # 手动打印blur矩阵
            if manual_print_mask and args.enable_blur:
                try:
                    # 获取当前环境信息
                    current_obs = env.render()  # 获取当前观察
                    if hasattr(env, '_env'):  # 检查是否有wrapped环境
                        # 尝试从环境中获取信息
                        env_info = getattr(env, '_last_info', None)
                        if env_info is None:
                            print("No environment info available for manual mask print")
                        else:
                            semantic_map = env_info.get('semantic', None)
                            player_pos = env_info.get('player_pos', [32, 32])
                            view_size = env_info.get('view', [9, 9])
                            
                            if semantic_map is not None:
                                mask = env._get_target_mask(semantic_map, player_pos, view_size, current_obs.shape)
                                print_blur_matrix(mask, step_count, args.target_obj_name, 
                                                np.sum(mask) > 0, int(np.sum(mask)))
                                print("Manual mask print completed!")
                            else:
                                print("No semantic map available for manual print")
                    else:
                        print("Manual print not available - no environment wrapper detected")
                except Exception as e:
                    print(f"Error during manual mask print: {e}")

    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        pygame.quit()
        print(f"\nGame ended after {step_count} steps")
        print(f"Total reward: {return_:.2f}")

def demo_auto_play():
    """
    自动游戏演示（使用随机动作）
    """
    print("=== Auto-play Demo with Selective Blur ===")
    
    # 配置参数
    target_obj_id = 3  # stone
    target_obj_name = "stone"
    blur_strength = 15
    max_steps = 200
    
    print(f"Target object: {target_obj_name} (ID: {target_obj_id})")
    print(f"Blur strength: {blur_strength}")
    print(f"Max steps: {max_steps}")
    
    # 创建环境（按照项目方式）
    env = gym.make("MyCrafter-v0")
    env = env_wrapper.InitWrapper(env, init_items=["stone_pickaxe"], init_num=[1], init_center=6)
    env = env_wrapper.SelectiveBlurWrapper(
        env, 
        target_obj_id=target_obj_id,
        target_obj_name=target_obj_name,
        blur_strength=blur_strength
    )
    
    obs = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    print("\nStarting auto-play...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while not done and step_count < max_steps:
            # 随机选择动作
            action = env.action_space.sample()
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            step_count += 1
            total_reward += reward
            
            # 处理blur信息
            blur_info = info.get('selective_blur', None)
            if blur_info is not None:
                target_found = blur_info.get('target_found', False)
                target_pixels = blur_info.get('target_pixels', 0)
                target_name = blur_info.get('target_obj_name', 'unknown')
                
                # 如果找到目标或者每20步，打印blur矩阵
                if target_found or step_count % 20 == 0:
                    semantic_map = info.get('semantic', None)
                    player_pos = info.get('player_pos', [32, 32])
                    view_size = info.get('view', [9, 9])
                    
                    if semantic_map is not None:
                        mask = env._get_target_mask(semantic_map, player_pos, view_size, obs.shape)
                        print_blur_matrix(mask, step_count, target_name, target_found, target_pixels)
                
                if target_found:
                    print(f"✓ Step {step_count}: Target {target_name} found! Pixels: {target_pixels}")
                elif step_count % 50 == 0:
                    print(f"Step {step_count}: Searching for {target_name}... Reward: {total_reward:.2f}")
            
            if reward > 0:
                print(f"Reward at step {step_count}: {reward}")
    
    except KeyboardInterrupt:
        print("\nAuto-play interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        env.close()
        print(f"\nAuto-play ended after {step_count} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward per step: {total_reward/max(step_count, 1):.4f}")

if __name__ == "__main__":
    import sys
    
    print("Choose demo mode:")
    print("1. Manual control (human demonstrations with blur)")
    print("2. Auto-play with random actions")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        demo_auto_play()
    else:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            demo_auto_play()
        else:
            print("Invalid choice, running manual control demo")
            main()