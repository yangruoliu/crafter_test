#!/usr/bin/env python3
import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import gym
import numpy as np

# Ensure local modules are available
import crafter  # noqa: F401
import env_wrapper
from my_label_oracle import get_label

# Import custom algo/policy implementations
# Add alias to be compatible with models saved with different module paths
import importlib

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
    kind: str  # one of: "direction_dynamic", "direction_fixed", "attn_only"


@dataclass
class TaskSpec:
    id: str
    description: str
    # object id used by DirectionLabelWrapper (if applicable)
    direction_obj_id: Optional[int]
    # label string for LabelGeneratingWrapper (if applicable)
    aux_label_name: Optional[str]
    # whether to give starting wood_pickaxe = 1
    needs_pickaxe: bool = False
    # optional per-task init items
    init_items: Optional[List[str]] = None
    init_num: Optional[List[int]] = None


def build_env_for_task(task: TaskSpec, model_kind: str) -> gym.Env:
    env = gym.make("MyCrafter-v0")

    # Task-specific wrappers
    if task.id == "mine_stone":
        env = env_wrapper.MineStoneWrapper(env)
    elif task.id == "mine_coal":
        # Use the simplified wrapper that does not require a navigation model
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
    if model_kind in ("direction_dynamic", "direction_fixed"):
        # Use DirectionLabelWrapper; default to stone(7) when not specified
        target_id = task.direction_obj_id if task.direction_obj_id is not None else 7
        env = env_wrapper.DirectionLabelWrapper(env, target_obj_id=target_id, target_obj_name="target")
    elif model_kind == "attn_only":
        # Use LabelGeneratingWrapper; default label name if missing
        label_name = task.aux_label_name if task.aux_label_name is not None else "stone"
        env = env_wrapper.LabelGeneratingWrapper(env, get_label_func=get_label, target_obj=label_name, num_aux_classes=2)
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    return env


def load_model(model_spec: ModelSpec, env: gym.Env):
    # We expect all three trainings to use the same CustomPPO class defined in model_with_attn.py
    if CustomPPO is None:
        raise RuntimeError("CustomPPO not available (model_with_attn.py not importable)")

    try:
        model = CustomPPO.load(model_spec.path, env=env)
        return model
    except Exception as e:
        # Try without passing env (will set later)
        try:
            model = CustomPPO.load(model_spec.path)
            model.set_env(env)
            return model
        except Exception as e2:
            raise RuntimeError(f"Failed to load model '{model_spec.name}' from {model_spec.path}: {e} / {e2}")


def evaluate_on_env(model, env: gym.Env, model_kind: str, num_episodes: int = 10, max_steps: int = 1000) -> Dict:
    total_rewards: List[float] = []
    total_steps: List[int] = []
    completions: List[bool] = []

    # Direction metrics if available
    dir_accs: List[float] = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        dir_hits = 0
        dir_count = 0

        while not done and steps < max_steps:
            if model_kind in ("direction_dynamic", "direction_fixed") and hasattr(model, "policy") \
               and hasattr(model.policy, "predict_with_direction"):
                # Use the helper to also retrieve direction prediction
                actions, dir_results = model.policy.predict_with_direction(obs, deterministic=True)
                # Ensure scalar action
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

            # Accumulate
            ep_reward += float(reward)
            steps += 1

            # Direction accuracy if obs contains label and we produced a prediction
            if dir_results is not None:
                try:
                    if isinstance(obs, dict) and "direction_label" in obs:
                        pred_dir = int(dir_results["predicted_direction"].reshape(-1)[0])
                        true_dir = int(obs["direction_label"])  # 0..8
                        dir_hits += int(pred_dir == true_dir)
                        dir_count += 1
                except Exception:
                    pass

        total_rewards.append(ep_reward)
        total_steps.append(steps)
        completions.append(bool(done))

        if dir_count > 0:
            dir_accs.append(dir_hits / dir_count)

    summary: Dict = {
        "reward_mean": float(np.mean(total_rewards)),
        "reward_std": float(np.std(total_rewards)),
        "steps_mean": float(np.mean(total_steps)),
        "steps_std": float(np.std(total_steps)),
        "completion_rate": float(np.mean(completions)),
        "episodes": num_episodes,
    }
    if len(dir_accs) > 0:
        summary["direction_accuracy_mean"] = float(np.mean(dir_accs))
        summary["direction_accuracy_std"] = float(np.std(dir_accs))

    return summary


def print_comparison_table(results: Dict[str, Dict[str, Dict]]):
    # results[model_name][task_id] -> summary dict
    model_names = list(results.keys())
    # collect all tasks
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
    parser = argparse.ArgumentParser(description="对比三种训练方案模型在多任务上的指标")
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
            env = build_env_for_task(task, spec.kind)
            model = load_model(spec, env)
            summary = evaluate_on_env(
                model, env, model_kind=spec.kind, num_episodes=args.episodes, max_steps=args.max_steps
            )
            model_results[task.id] = summary
            # Cleanup
            try:
                env.close()
            except Exception:
                pass

            # Pretty print small summary
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


if __name__ == "__main__":
    main()