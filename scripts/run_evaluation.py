#!/usr/bin/env python3
"""Run evaluation on generated tasks with an expert agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openclaw_env.core.environment import OpenClawEnv
from openclaw_env.core.task import load_task_ids
from openclaw_env.evaluation.metrics import compute_metrics


def run_expert_agent(env: OpenClawEnv, verbose: bool = False) -> bool:
    """Run the expert solution for a task."""
    task = env.task
    if not task or not task.ground_truth:
        return False

    total = len(task.ground_truth.solution_commands)
    for idx, command in enumerate(task.ground_truth.solution_commands, 1):
        if verbose:
            print(f"      [{idx}/{total}] $ {command}", flush=True)
        obs, reward, done, info = env.step(command)
        if verbose:
            print(
                f"         -> exit={obs.exit_code} done={done} "
                f"stdout={(obs.command_output or '').strip()[:100]}",
                flush=True,
            )
        if done:
            break

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run expert-agent evaluation on a task split."
    )
    parser.add_argument("split", nargs="?", default="dev", help="Dataset split")
    parser.add_argument(
        "--mode",
        choices=["mock", "multi", "real", "hybrid"],
        default="multi",
        help="Backend mode used by OpenClawEnv",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-task and per-command progress",
    )
    parser.add_argument(
        "--task-data-dir",
        default=str(
            Path(__file__).parent.parent
            / "openclaw_env"
            / "data"
        ),
        help=(
            "Task data root containing tasks/ and datasets/. "
            "Default: openclaw_env/data"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.task_data_dir)
    split = args.split

    task_ids = load_task_ids(split, data_dir=data_dir)
    if not task_ids:
        print(f"No tasks found for split '{split}'")
        return

    print(
        f"Evaluating {len(task_ids)} tasks from '{split}' split "
        f"(mode={args.mode})..."
    )

    results = []
    total_tasks = len(task_ids)
    for i, task_id in enumerate(task_ids, 1):
        print(f"  [START {i:4d}/{total_tasks}] {task_id}", flush=True)
        try:
            with OpenClawEnv(task_id=task_id, task_data_dir=data_dir, backend=args.mode) as env:
                env.reset()
                run_expert_agent(env, verbose=args.verbose)
                eval_result = env.evaluate()

                task = env.task
                task_meta = {
                    "task_id": task_id,
                    "domains": task.domains if task else [],
                    "difficulty": task.difficulty if task else 0,
                }
                results.append((task_meta, eval_result))

                status = "PASS" if eval_result.success else "FAIL"
                print(
                    f"  [DONE  {i:4d}/{total_tasks}] [{status}] {task_id} "
                    f"(score: {eval_result.score:.2f})",
                    flush=True,
                )
        except Exception as e:
            print(f"  [DONE  {i:4d}/{total_tasks}] [ERROR] {task_id}: {e}", flush=True)

    # Compute aggregate metrics
    metrics = compute_metrics(results)
    print(f"\n{'='*50}")
    print(f"Results: {metrics.passed_tasks}/{metrics.total_tasks} passed "
          f"(TGC: {metrics.tgc*100:.1f}%)")
    print(f"Average score: {metrics.avg_score:.4f}")

    print("\nBy domain:")
    for domain, dm in sorted(metrics.by_domain.items()):
        print(f"  {domain}: {dm.passed}/{dm.total} (TGC: {dm.tgc*100:.1f}%)")

    print("\nBy difficulty:")
    for diff, dfm in sorted(metrics.by_difficulty.items()):
        print(f"  Level {diff}: {dfm.passed}/{dfm.total} (TGC: {dfm.tgc*100:.1f}%)")

    # Save report
    report_path = data_dir / "trajectories" / f"eval_{split}_{args.mode}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
