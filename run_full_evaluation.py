#!/usr/bin/env python3
"""
Batch evaluation runner for five tasks across three models.
- Tasks: mine_stone, mine_coal, mine_iron, drink_water, wood_pickaxe
- Models (defaults):
  - v4 dynamic: /home/crafter_zelda/crafter_test/stone_with_direction_v4_20250804_145854.zip
  - v5 dynamic: /home/crafter_zelda/crafter_test/stone_with_direction_v5_20250804_153025.zip
  - base fixed: /home/crafter_zelda/crafter_test/stone_with_direction.zip

This script calls compare_models.py per task to generate:
- Per-task JSON metrics (full episode-level stats)
- Per-task PNG plots (reward curve + cumulative completion)
And also one all-tasks JSON summary in a single run (no plot).

Usage:
  python3 run_full_evaluation.py \
    --episodes 50 \
    --max-steps 1000 \
    --out-dir /home/crafter_zelda/crafter_test/full_eval

You can override model paths via flags, see --help.
"""

import argparse
import os
import subprocess
import sys
import time
from typing import List

DEFAULT_MODELS = {
    "v1": ("direction", os.path.abspath(os.path.join(os.path.dirname(__file__), "stone_with_direction_improved_20250813_142225.zip"))),
    "v2": ("direction", os.path.abspath(os.path.join(os.path.dirname(__file__), "stone_with_direction_fixed_20250813_142210.zip"))),
    "v3": ("direction", os.path.abspath(os.path.join(os.path.dirname(__file__), "stone_with_direction_final_20250813_142007.zip"))),
    "v4": ("direction", os.path.abspath(os.path.join(os.path.dirname(__file__), "stone_with_direction_v4_20250813_141852.zip"))),
    "v5": ("direction", os.path.abspath(os.path.join(os.path.dirname(__file__), "stone_with_direction_v5_20250813_122504.zip"))),
    "base": ("no_direction", os.path.abspath(os.path.join(os.path.dirname(__file__), "stone.zip"))),
}

TASKS: List[str] = [
    "mine_stone",
    "mine_coal",
    "mine_iron",
    "drink_water",
    "wood_pickaxe",
]


def build_model_args(models: dict) -> List[str]:
    args: List[str] = []
    for name, (kind, path) in models.items():
        args.extend(["--model", f"{name}:{path}:{kind}"])
    return args


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    # Stream output line-by-line to avoid long periods with no logs
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with code {proc.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation and plots for five tasks and three models")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--out-dir", type=str, default=None, help="Output base directory; default under model dir with timestamp")

    parser.add_argument("--model-v1", type=str, default=DEFAULT_MODELS["v1"][1], help="Path to v1 model")
    parser.add_argument("--model-v2", type=str, default=DEFAULT_MODELS["v2"][1], help="Path to v2 model")
    parser.add_argument("--model-v3", type=str, default=DEFAULT_MODELS["v3"][1], help="Path to v3 model")
 
    parser.add_argument("--model-v4", type=str, default=DEFAULT_MODELS["v4"][1], help="Path to v4 dynamic model")
    parser.add_argument("--model-v5", type=str, default=DEFAULT_MODELS["v5"][1], help="Path to v5 dynamic model")
    parser.add_argument("--model-base", type=str, default=DEFAULT_MODELS["base"][1], help="Path to base fixed-weight model")

    args = parser.parse_args()

    # Compose model dict
    models = {
        "v1": ("direction", args.model_v1),
        "v2": ("direction", args.model_v2),
        "v3": ("direction", args.model_v3),
        "v4": ("direction", args.model_v4),
        "v5": ("direction", args.model_v5),
        "base": ("no_direction", args.model_base),
    }

    # Resolve output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"/home/crafter_zelda/crafter_test/full_eval_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    compare_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "compare_models.py"))
    if not os.path.exists(compare_script):
        raise SystemExit(f"compare_models.py not found at {compare_script}")

    # Preflight: verify model files exist to fail fast with a clear message
    missing: List[str] = []
    for name, (_kind, path) in models.items():
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")
    if missing:
        print("The following model paths do not exist:")
        for m in missing:
            print(f"  - {m}")
        print("Please pass valid paths via --model-v1/--model-v2/--model-v3/--model-v4/--model-v5/--model-base.")
        raise SystemExit(1)

    # 1) Run all tasks together for a consolidated JSON
    all_json = os.path.join(out_dir, "compare_all.json")
    cmd_all = [
        sys.executable,
        "-u",  # unbuffered for real-time logs
        compare_script,
        *build_model_args(models),
        "--tasks",
        *TASKS,
        "--episodes", str(args.episodes),
        "--max-steps", str(args.max_steps),
        "--save-json", all_json,
    ]
    run_cmd(cmd_all)

    # 2) Run each task individually to produce per-task JSON and PNG
    for task in TASKS:
        task_json = os.path.join(out_dir, f"{task}.json")
        task_png = os.path.join(out_dir, f"{task}.png")
        cmd_task = [
            sys.executable,
            "-u",  # unbuffered for real-time logs
            compare_script,
            *build_model_args(models),
            "--tasks", task,
            "--episodes", str(args.episodes),
            "--max-steps", str(args.max_steps),
            "--save-json", task_json,
            "--plot-path", task_png,
        ]
        run_cmd(cmd_task)

    print("\nDone.")
    print("Outputs:")
    print(f"  All-tasks JSON: {all_json}")
    for task in TASKS:
        print(f"  {task}: {os.path.join(out_dir, f'{task}.json')} | {os.path.join(out_dir, f'{task}.png')}")


if __name__ == "__main__":
    main()