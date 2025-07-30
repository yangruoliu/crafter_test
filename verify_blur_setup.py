# verify_blur_setup.py
"""
快速验证选择性模糊设置是否正确
"""
import gym
import env_wrapper
import numpy as np
import matplotlib.pyplot as plt

def verify_selective_blur_setup():
    """验证选择性模糊设置"""
    print("=== 验证选择性模糊设置 ===")
    
    try:
        # 创建环境
        env = gym.make("MyCrafter-v0")
        env = env_wrapper.MineStoneWrapper(env)
        env = env_wrapper.InitWrapper(env, ["wood_pickaxe"], [1])
        env = env_wrapper.SelectiveBlurWrapper(env, target_obj_id=3, target_obj_name="stone")
        
        print("✓ 环境创建成功")
        
        # 重置环境
        obs = env.reset()
        print(f"✓ 环境重置成功，观察形状: {obs.shape}")
        
        # 执行几步
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            blur_info = info.get('selective_blur', {})
            if blur_info:
                print(f"步骤 {i+1}: 目标={blur_info.get('target_obj_name', 'unknown')}, "
                      f"发现={blur_info.get('target_found', False)}, "
                      f"像素数={blur_info.get('target_pixels', 0)}")
            else:
                print(f"步骤 {i+1}: 无模糊信息")
                
            if done:
                print("任务完成!")
                break
        
        # 显示最后一帧
        plt.figure(figsize=(8, 6))
        plt.imshow(obs)
        plt.title("选择性模糊效果验证")
        plt.axis('off')
        plt.show()
        
        env.close()
        print("✓ 验证完成")
        
    except Exception as e:
        print(f"✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_selective_blur_setup()