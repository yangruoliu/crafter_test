#!/usr/bin/env python3
import argparse
import os
import sys
import time
from typing import Optional

try:
    import gym
except Exception:
    import gymnasium as gym
import numpy as np
import cv2

# Ensure local modules are available
import crafter  # noqa: F401
import env_wrapper
from my_label_oracle import get_label

try:
    import model_with_attn as mwa
except Exception:
    mwa = None

CustomPPO = getattr(mwa, "CustomPPO", None)


def build_env_for_task(task_id: str, model_kind: str) -> gym.Env:
    env = gym.make("MyCrafter-v0")

    # Task-specific wrappers
    if task_id == "mine_stone":
        env = env_wrapper.MineStoneWrapper(env)
    elif task_id == "mine_coal":
        env = env_wrapper.MineCoalWrapper2(env, obj_index=8)
    elif task_id == "mine_iron":
        env = env_wrapper.MineIronWrapper2(env, obj_index=9)
    elif task_id == "drink_water":
        env = env_wrapper.DrinkWaterWrapper(env)
    elif task_id == "wood_pickaxe":
        env = env_wrapper.WoodPickaxeWrapper(env)
    else:
        raise SystemExit(f"Unknown task id: {task_id}")

    # Final wrapper depends on model kind to ensure observation space matches the model
    if model_kind == "direction":
        # default to stone target (7)
        env = env_wrapper.DirectionLabelWrapper(env, target_obj_id=7, target_obj_name="target")
    elif model_kind == "no_direction":
        env = env_wrapper.LabelGeneratingWrapper(env, get_label_func=get_label, target_obj="stone", num_aux_classes=2)
    else:
        raise SystemExit(f"Unknown model kind: {model_kind}")

    return env


def load_model(model_path: str, env: gym.Env, device: str = "cpu"):
    if CustomPPO is None:
        raise SystemExit("CustomPPO not available (model_with_attn.py not importable)")
    # Prefer load without env for direction models; but here we don't know kind reliably.
    # Use with env for maximal compatibility.
    return CustomPPO.load(model_path, env=env, device=device)


def to_action(model, obs, model_kind: str):
    # Supports dict observations for direction models
    if model_kind == "direction" and hasattr(model, "policy") and hasattr(model.policy, "predict_with_direction"):
        actions, _ = model.policy.predict_with_direction(obs, deterministic=True)
        if isinstance(actions, np.ndarray):
            return int(actions.reshape(-1)[0])
        return int(actions)
    else:
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            return int(action.reshape(-1)[0])
        return int(action)


def main():
    parser = argparse.ArgumentParser(description="Visualize agent playing the Crafter tasks with live rendering")
    parser.add_argument("--model", type=str, required=True, help="Path to model zip file")
    parser.add_argument("--kind", type=str, choices=["direction", "no_direction"], required=True, help="Model kind")
    parser.add_argument("--task", type=str, default="mine_stone", help="Task: mine_stone, mine_coal, mine_iron, drink_water, wood_pickaxe")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda:0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--size", type=int, default=512, help="Render window size (square)")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS for display")
    parser.add_argument("--headless", action="store_true", help="Run without GUI, suitable for servers without display")
    parser.add_argument("--video-out", type=str, default=None, help="If set, save a video to this path (mp4)")
    parser.add_argument("--video-fps", type=int, default=None, help="FPS for saved video; default to --fps")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    if args.device.startswith("cpu"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    env = build_env_for_task(args.task, args.kind)
    model = load_model(args.model, env, device=args.device)

    # Determine headless mode
    no_display = not os.environ.get("DISPLAY")
    headless = args.headless or no_display

    delay_ms = max(1, int(1000 / max(1, args.fps)))

    # Setup video writer if requested
    writer = None
    if args.video_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vfps = args.video_fps if args.video_fps is not None else args.fps
        writer = cv2.VideoWriter(args.video_out, fourcc, float(vfps), (args.size, args.size))
        if not writer.isOpened():
            print(f"Warning: failed to open video writer at {args.video_out}")
            writer = None

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            while not done and steps < args.max_steps:
                action = to_action(model, obs, args.kind)
                obs, reward, done, info = env.step(action)
                ep_reward += float(reward)
                steps += 1

                # Render and show
                try:
                    frame = env.render((args.size, args.size))
                except Exception:
                    # Fallback: if obs is image-like
                    if isinstance(obs, dict) and "obs" in obs:
                        frame = obs["obs"]
                    else:
                        frame = obs
                    if frame is not None and frame.ndim == 3 and frame.shape[-1] in (1, 3):
                        pass
                    else:
                        frame = np.zeros((args.size, args.size, 3), dtype=np.uint8)

                # Convert to BGR for OpenCV if needed
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                if frame.shape[-1] == 3:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    bgr = frame

                # Resize to target window/video size if needed
                if bgr.shape[1] != args.size or bgr.shape[0] != args.size:
                    bgr = cv2.resize(bgr, (args.size, args.size), interpolation=cv2.INTER_NEAREST)

                if writer is not None:
                    writer.write(bgr)

                if not headless:
                    cv2.imshow("Crafter Viewer", bgr)
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key == ord('q'):
                        done = True
                        break

            print(f"Episode {ep+1}/{args.episodes} reward={ep_reward:.1f} steps={steps}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()