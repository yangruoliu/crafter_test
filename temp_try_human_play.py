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

def main():
    boolean = lambda x: bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--death', type=str, default='reset', choices=[
        'continue', 'reset', 'quit'])
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
    
    print('Actions:')
    for key, action in keymap.items():
        print(f'  {pygame.key.name(key)}: {action}')

    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]

    # 创建环境并修复metadata
    env = gym.make("MyCrafter-v0")
    env = MetadataFixWrapper(env)  # 添加metadata修复wrapper
    
    # 添加其他wrapper
    env = env_wrapper.LLMWrapper(env, model="deepseek-chat")
    env = env_wrapper.InitWrapper(env, init_items=["stone_pickaxe"], init_num=[1], init_center=6)
    
    env.reset()
    achievements = set()
    duration = 0
    return_ = 0
    was_done = False

    pygame.init()
    screen = pygame.display.set_mode(args.window)
    clock = pygame.time.Clock()
    running = True
    
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
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
                action = keymap[event.key]
                
        if action is None:
            pressed = pygame.key.get_pressed()
            for key, action in keymap.items():
                if pressed[key]:
                    break
            else:
                if args.wait:
                    continue
                else:
                    action = 'noop'

        # Environment step.
        print(action)
        _, reward, done, _ = env.step(env.action_names.index(action))
        duration += 1

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
            if args.death == 'continue':
                pass

    pygame.quit()

if __name__ == '__main__':
    main()