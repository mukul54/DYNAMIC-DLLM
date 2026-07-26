#!/usr/bin/env python3
"""Collect accuracy + throughput + tokens/step from a directory of runs.

Pairs each lm-eval ``results_*.json`` with the ``speed.json`` written by the
SpeedMeter and prints one row per run:

    python metrics/summarize_runs.py ./gsm8k512_log
    python metrics/summarize_runs.py ./gsm8k512_log --metric exact_match,flexible-extract
    python metrics/summarize_runs.py ./gsm8k512_log --csv results.csv

The run name is the directory passed as ``--output_path`` for that run.
"""

import argparse
import csv
import glob
import json
import os
from typing import Optional


def _latest_results_json(run_dir: str) -> Optional[str]:
    """lm-eval nests results under <output_path>/<sanitized model name>/."""
    matches = glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True)
    return max(matches, key=os.path.getmtime) if matches else None


def _pick_metric(task_results: dict, preferred: Optional[str]) -> tuple:
    """Return (metric_name, value). Prefers flexible-extract exact_match."""
    if preferred and preferred in task_results:
        return preferred, task_results[preferred]

    order = [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "pass_at_1,create_test",
        "acc,none",
    ]
    for key in order:
        if key in task_results:
            return key, task_results[key]

    for key, value in task_results.items():
        if isinstance(value, float) and not key.endswith("_stderr"):
            return key, value
    return "n/a", float("nan")


def collect(root: str, metric: Optional[str]) -> list:
    rows = []
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
    )
    for run_dir in run_dirs:
        row = {"run": os.path.basename(run_dir)}

        results_path = _latest_results_json(run_dir)
        if results_path:
            with open(results_path) as f:
                results = json.load(f).get("results", {})
            for task, task_results in results.items():
                if not isinstance(task_results, dict):
                    continue
                metric_name, value = _pick_metric(task_results, metric)
                row["task"] = task
                row["metric"] = metric_name
                # lm-eval reports fractions; the paper's tables are percentages.
                row["accuracy"] = value * 100 if isinstance(value, float) else value
                break

        speed_path = os.path.join(run_dir, "speed.json")
        if os.path.exists(speed_path):
            with open(speed_path) as f:
                speed = json.load(f)
            row.update(
                {
                    "gen_length": speed.get("gen_length"),
                    "throughput_tok_s": speed.get("throughput_tok_s"),
                    "throughput_tok_s_full": speed.get("throughput_tok_s_full"),
                    "tokens_per_step": speed.get("tokens_per_step"),
                    "avg_nfe": speed.get("avg_nfe_per_sequence"),
                    "gen_time_s": speed.get("generation_time_s"),
                    "sequences": speed.get("sequences"),
                }
            )

        if len(row) > 1:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="directory containing one sub-directory per run")
    parser.add_argument(
        "--metric",
        default=None,
        help="explicit lm-eval metric key, e.g. 'exact_match,flexible-extract'",
    )
    parser.add_argument("--baseline", default=None, help="run name to compute speedup against")
    parser.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = parser.parse_args()

    rows = collect(args.root, args.metric)
    if not rows:
        print(f"No runs found under {args.root}")
        return

    names = [r["run"] for r in rows]
    # Speedups in the paper are quoted against greedy no-cache decoding.
    default_baseline = next(
        (n for n in ("nocache_greedy", "no_cache", "baseline") if n in names), names[0]
    )
    baseline_name = args.baseline or default_baseline
    baseline = next(
        (r for r in rows if r["run"] == baseline_name and r.get("throughput_tok_s")), None
    )
    base_tps = baseline["throughput_tok_s"] if baseline else None

    header = f"{'run':<20} {'acc %':>8} {'tok/s':>10} {'speedup':>9} {'tok/step':>9} {'NFE':>8} {'time s':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        tps = r.get("throughput_tok_s")
        speedup = f"{tps / base_tps:.1f}x" if (tps and base_tps) else "-"
        print(
            f"{r['run']:<20} "
            f"{r.get('accuracy', float('nan')):>8.2f} "
            f"{tps if tps is not None else float('nan'):>10.1f} "
            f"{speedup:>9} "
            f"{r.get('tokens_per_step', float('nan')):>9.2f} "
            f"{r.get('avg_nfe', float('nan')):>8.1f} "
            f"{r.get('gen_time_s', float('nan')):>10.1f}"
        )

    metrics_used = {r.get("metric") for r in rows if r.get("metric")}
    if metrics_used:
        print(f"\naccuracy metric: {', '.join(sorted(metrics_used))}")
    print(f"speedup baseline: {baseline_name}")
    print("throughput counts tokens up to the first <eos> (Fast-dLLM protocol);")
    print("tokens/step = gen_length / NFE, so fixed-schedule greedy decoding is 1.00.")

    if args.csv:
        fieldnames = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
