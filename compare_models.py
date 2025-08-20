#!/usr/bin/env python3
import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

try:
    import gym
except Exception:
    import gymnasium as gym
import numpy as np

# Ensure local modules are available
import crafter  # noqa: F401
import env_wrapper
from my_label_oracle import get_label

# Optional plotting (only import when needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import custom algo/policy implementations
try:
    import model_with_attn as mwa
except Exception:
    mwa = None

# Register alias for potential pickled paths like "model.model_with_attn"
if mwa is not None:
    sys.modules.setdefault("model.model_with_attn", mwa)
    sys.modules.setdefault("model_with_attn", mwa)

CustomPPO = getattr(mwa, "CustomPPO", None)


@dataclass
class ModelSpec:
    name: str
    path: str
    kind: str  # one of: "direction" or "no_direction"


@dataclass
class TaskSpec:
    id: str
    description: str
    direction_obj_id: Optional[int]
    aux_label_name: Optional[str]
    needs_pickaxe: bool = False
    init_items: Optional[List[str]] = None
    init_num: Optional[List[int]] = None


def build_env_for_task(task: TaskSpec, model_kind: str) -> gym.Env:
    env = gym.make("MyCrafter-v0")

    # Task-specific wrappers
    if task.id == "mine_stone":
        env = env_wrapper.MineStoneWrapper(env)
    elif task.id == "mine_coal":
        env = env_wrapper.MineCoalWrapper2(env, obj_index=8)
    elif task.id == "mine_iron":
        env = env_wrapper.MineIronWrapper2(env, obj_index=9)
    elif task.id == "drink_water":
        env = env_wrapper.DrinkWaterWrapper(env)
    elif task.id == "wood_pickaxe":
        env = env_wrapper.WoodPickaxeWrapper(env)
    else:
        raise ValueError(f"Unknown task id: {task.id}")

    # Initialization items
    if task.needs_pickaxe:
        env = env_wrapper.InitWrapper(env, init_items=["wood_pickaxe"], init_num=[1])

    if task.init_items is not None and task.init_num is not None:
        env = env_wrapper.InitWrapper(env, init_items=task.init_items, init_num=task.init_num)

    # Final wrapper depends on model kind to ensure observation space matches the model
    if model_kind in ("direction"):
        # Use DirectionLabelWrapper; default to stone(7) when not specified
        target_id = task.direction_obj_id if task.direction_obj_id is not None else 7
        env = env_wrapper.DirectionLabelWrapper(env, target_obj_id=target_id, target_obj_name="target")
    elif model_kind == "no_direction":
        # Use LabelGeneratingWrapper; default label name if missing
        label_name = task.aux_label_name if task.aux_label_name is not None else "stone"
        env = env_wrapper.LabelGeneratingWrapper(env, get_label_func=get_label, target_obj=label_name, num_aux_classes=2)
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    return env


def load_model(model_spec: ModelSpec, env: gym.Env, device: str = "cpu"):
    if CustomPPO is None:
        raise RuntimeError("CustomPPO not available (model_with_attn.py not importable)")

    # Prefer loading without env first to avoid potential hangs during wrapper setup
    try:
        print(f"  Loading model '{model_spec.name}' from {model_spec.path} on device={device} (no env) ...", flush=True)
        model = CustomPPO.load(model_spec.path, device=device)
        print(f"  Model weights loaded. Now attaching env ...", flush=True)
        model.set_env(env)
        print(f"  Env attached to model '{model_spec.name}'.", flush=True)
        return model
    except Exception as e:
        # As a fallback, try loading with env directly
        try:
            print(f"  Fallback: load with env for '{model_spec.name}' ...", flush=True)
            model = CustomPPO.load(model_spec.path, env=env, device=device)
            print(f"  Loaded model '{model_spec.name}' with env.", flush=True)
            return model
        except Exception as e2:
            raise RuntimeError(f"Failed to load model '{model_spec.name}' from {model_spec.path}: {e} / {e2}")


def evaluate_on_env(model, env: gym.Env, model_kind: str, num_episodes: int = 10, max_steps: int = 1000) -> Dict:
    total_rewards: List[float] = []
    total_steps: List[int] = []
    completions: List[bool] = []

    # Direction metrics if available
    dir_accs: List[float] = []
    # Per-episode data for plotting
    episode_rewards: List[float] = []
    episode_success: List[int] = []

    for ep_idx in range(num_episodes):
        print(f"  Resetting env for episode {ep_idx+1}/{num_episodes} ...", flush=True)
        obs = env.reset()
        print(f"  Env reset ok for episode {ep_idx+1}/{num_episodes}.", flush=True)
        done = False
        ep_reward = 0.0
        steps = 0
        dir_hits = 0
        dir_count = 0
        success_happened = False

        while not done and steps < max_steps:
            if model_kind in ("direction") and hasattr(model, "policy") and hasattr(model.policy, "predict_with_direction"):
                actions, dir_results = model.policy.predict_with_direction(obs, deterministic=True)
                if isinstance(actions, np.ndarray):
                    if actions.ndim == 0:
                        action = int(actions)
                    else:
                        action = int(actions.reshape(-1)[0])
                else:
                    action = int(actions)
            else:
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    if action.ndim == 0:
                        action = int(action)
                    else:
                        action = int(action.reshape(-1)[0])
                else:
                    action = int(action)
                dir_results = None

            obs, reward, done, info = env.step(action)

            ep_reward += float(reward)
            steps += 1

            # Success detection: prefer wrapper flag; fallback to large positive reward spike
            try:
                if isinstance(info, dict) and (info.get("success", False) or float(reward) >= 900.0):
                    success_happened = True
            except Exception:
                pass

            # Heartbeat for long episodes
            if steps % 200 == 0:
                print(f"    ep {ep_idx+1}/{num_episodes}: {steps}/{max_steps} steps", flush=True)

            if dir_results is not None:
                try:
                    if isinstance(obs, dict) and "direction_label" in obs:
                        pred_dir = int(dir_results["predicted_direction"].reshape(-1)[0])
                        true_dir = int(obs["direction_label"])  # 0..8
                        dir_hits += int(pred_dir == true_dir)
                        dir_count += 1
                except Exception:
                    pass

        # Per-episode progress line
        print(
            f"  ep {ep_idx+1}/{num_episodes}: reward={ep_reward:.1f}, success={1 if success_happened else 0}, steps={steps}",
            flush=True,
        )

        total_rewards.append(ep_reward)
        total_steps.append(steps)
        completions.append(bool(success_happened))
        episode_rewards.append(ep_reward)
        episode_success.append(1 if success_happened else 0)

        if dir_count > 0:
            dir_accs.append(dir_hits / dir_count)

    summary: Dict = {
        "reward_mean": float(np.mean(total_rewards)),
        "reward_std": float(np.std(total_rewards)),
        "reward_min": float(np.min(total_rewards) if total_rewards else 0.0),
        "reward_max": float(np.max(total_rewards) if total_rewards else 0.0),
        "reward_median": float(np.median(total_rewards) if total_rewards else 0.0),
        "steps_mean": float(np.mean(total_steps)),
        "steps_std": float(np.std(total_steps)),
        "steps_min": int(np.min(total_steps) if total_steps else 0),
        "steps_max": int(np.max(total_steps) if total_steps else 0),
        "steps_median": float(np.median(total_steps) if total_steps else 0.0),
        "completion_rate": float(np.mean(completions)),
        "episodes": num_episodes,
        "episodes_rewards": episode_rewards,
        "episodes_success": episode_success,
        "cumulative_completion": (np.cumsum(episode_success) / np.maximum(1, np.arange(1, len(episode_success) + 1))).tolist() if episode_success else [],
        "reward_auc": float(np.trapz(episode_rewards, dx=1.0)),
        "first_success_episode": int(np.argmax(episode_success) + 1) if any(episode_success) else None,
    }
    if len(dir_accs) > 0:
        summary["direction_accuracy_mean"] = float(np.mean(dir_accs))
        summary["direction_accuracy_std"] = float(np.std(dir_accs))

    return summary


def plot_single_task_curves(results: Dict[str, Dict[str, Dict[str, Any]]], task_id: str, plot_path: str) -> None:
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(2, 1, 1)
    for model_name, task_map in results.items():
        s = task_map.get(task_id, {})
        rewards = s.get("episodes_rewards", [])
        if rewards:
            ax1.plot(range(1, len(rewards) + 1), rewards, label=model_name)
    ax1.set_title(f"Reward Curve per Episode - Task: {task_id}")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    for model_name, task_map in results.items():
        s = task_map.get(task_id, {})
        success = s.get("episodes_success", [])
        if success:
            success = np.array(success, dtype=float)
            cum_rate = np.cumsum(success) / np.maximum(1, np.arange(1, len(success) + 1))
            ax2.plot(range(1, len(success) + 1), cum_rate, label=model_name)
    ax2.set_title("Cumulative Completion Rate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Completion Rate")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def print_comparison_table(results: Dict[str, Dict[str, Dict]]):
    model_names = list(results.keys())
    task_ids = sorted({t for m in model_names for t in results[m].keys()})

    def fmt(v: Optional[float], pct=False):
        if v is None:
            return "-"
        if pct:
            return f"{v*100:.1f}%"
        return f"{v:.2f}"

    print("\n=== 对比结果 (按任务) ===")
    for task in task_ids:
        print(f"\n--- 任务: {task} ---")
        header = f"{'模型':<20} {'完成率':>8} {'回报均值±std':>16} {'步数均值±std':>16} {'方向准确率(均值)':>18}"
        print(header)
        print("-" * len(header))
        for model_name in model_names:
            s = results[model_name].get(task, {})
            acc = s.get("direction_accuracy_mean")
            line = (
                f"{model_name:<20} "
                f"{fmt(s.get('completion_rate'), pct=True):>8} "
                f"{fmt(s.get('reward_mean'))}±{fmt(s.get('reward_std')):>8} "
                f"{fmt(s.get('steps_mean'))}±{fmt(s.get('steps_std')):>8} "
                f"{fmt(acc if acc is not None else None):>18}"
            )
            print(line)


def main():
    parser = argparse.ArgumentParser(description="对比6种训练方案模型在多任务上的指标")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "指定一个模型，格式: name:path:kind，其中kind ∈ {direction_dynamic, direction_fixed, attn_only}. "
            "可多次提供此参数以添加多个模型"
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["mine_stone", "mine_coal", "mine_iron"],
        help="要评测的任务列表，可选: mine_stone, mine_coal, mine_iron, drink_water, wood_pickaxe",
    )
    parser.add_argument("--episodes", type=int, default=10, help="每个任务的评测回合数")
    parser.add_argument("--max-steps", type=int, default=1000, help="每回合最大步数")
    parser.add_argument("--save-json", type=str, default=None, help="将结果保存到指定json路径")
    parser.add_argument("--plot-path", type=str, default=None, help="当评测单一任务时，保存Reward/完成率曲线图的路径")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "cuda:0", "cuda:1"], help="加载模型所用设备，默认cpu")

    args = parser.parse_args()

    # Parse models
    model_specs: List[ModelSpec] = []
    for m in args.model:
        try:
            name, path, kind = m.split(":", 2)
        except ValueError:
            raise SystemExit(f"--model 参数格式错误: {m}. 正确格式为 name:path:kind")
        if not os.path.exists(path):
            raise SystemExit(f"模型路径不存在: {path}")
        model_specs.append(ModelSpec(name=name, path=path, kind=kind))

    # Build task list
    task_db: Dict[str, TaskSpec] = {
        "mine_stone": TaskSpec(
            id="mine_stone", description="挖石头", direction_obj_id=7, aux_label_name="stone", needs_pickaxe=True
        ),
        "mine_coal": TaskSpec(
            id="mine_coal", description="挖煤炭", direction_obj_id=8, aux_label_name="coal", needs_pickaxe=True
        ),
        "mine_iron": TaskSpec(
            id="mine_iron", description="挖铁矿", direction_obj_id=9, aux_label_name="iron", needs_pickaxe=True
        ),
        "drink_water": TaskSpec(
            id="drink_water", description="喝水", direction_obj_id=None, aux_label_name="water", needs_pickaxe=False
        ),
        "wood_pickaxe": TaskSpec(
            id="wood_pickaxe", description="制作木镐", direction_obj_id=None, aux_label_name="tree", needs_pickaxe=False,
            init_items=["wood", "stone"], init_num=[1, 1]
        ),
    }

    tasks: List[TaskSpec] = []
    for t in args.tasks:
        if t not in task_db:
            raise SystemExit(f"未知任务: {t}")
        tasks.append(task_db[t])

    all_results: Dict[str, Dict[str, Dict]] = {}

    for spec in model_specs:
        model_results: Dict[str, Dict] = {}
        print(f"\n==== 评测模型: {spec.name} ({spec.kind}) ====")
        for task in tasks:
            print(f"\n>> 任务: {task.id} - {task.description}")
            print(f"  Building env for task '{task.id}' ...", flush=True)
            env = build_env_for_task(task, spec.kind)
            print(f"  Env ready for task '{task.id}'.", flush=True)
            model = load_model(spec, env, device=args.device)
            summary = evaluate_on_env(
                model, env, model_kind=spec.kind, num_episodes=args.episodes, max_steps=args.max_steps
            )
            model_results[task.id] = summary
            try:
                env.close()
            except Exception:
                pass

            acc_str = (
                f", 方向准确率: {summary['direction_accuracy_mean']*100:.1f}%"
                if "direction_accuracy_mean" in summary
                else ""
            )
            print(
                f"完成率: {summary['completion_rate']*100:.1f}%, 奖励: {summary['reward_mean']:.2f}±{summary['reward_std']:.2f}, "
                f"步数: {summary['steps_mean']:.1f}±{summary['steps_std']:.1f}{acc_str}"
            )

        all_results[spec.name] = model_results

    print_comparison_table(all_results)

    if args.save_json:
        import json
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"results": all_results, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.save_json}")

    if args.plot_path and len(tasks) == 1:
        task_id = tasks[0].id
        plot_single_task_curves(all_results, task_id=task_id, plot_path=args.plot_path)
        print(f"曲线图已保存到: {args.plot_path}")


if __name__ == "__main__":
    main()