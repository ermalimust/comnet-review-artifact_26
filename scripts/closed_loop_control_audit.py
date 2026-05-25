#!/usr/bin/env python3
"""Sequential closed-loop audit for the action-selection case study.

The original same-budget control table asks whether scores select useful
windows under a fixed trigger budget. This script adds a lightweight sequential
audit: at each decision interval k, a policy compares the current score with a
rolling threshold formed from previous scores only. The action effect is still a
toy payoff model, not a deployment scheduler, but the trigger is online and the
threshold is updated without current or future outcomes.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

EPS = 1e-12

CONTROL_POLICIES = [
    "proposed_safe_trigger",
    "direct_risk_trigger",
    "ewma_persistence_trigger",
    "static_utilization_trigger",
    "random_trigger",
]

POLICY_LABELS = {
    "proposed_safe_trigger": "Rolling calibrated risk",
    "direct_risk_trigger": "Rolling direct risk",
    "ewma_persistence_trigger": "Rolling EWMA",
    "static_utilization_trigger": "Rolling utilization",
    "random_trigger": "Rolling random",
}

CONTROL_METRICS = [
    "target_violation_reduction",
    "p95_delay_reduction",
    "trigger_rate",
    "background_cost",
    "false_alarm_cost",
    "base_violation",
    "controlled_violation",
    "base_p95",
    "controlled_p95",
]


Row = Dict[str, str]


def read_csv(path: Path) -> List[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def safe_float(row: Row, key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return 0.0
    return float(value)


def group_by_method(rows: Iterable[Row]) -> Dict[str, List[Row]]:
    grouped: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    for items in grouped.values():
        items.sort(key=lambda row: int(float(row["window_id"])))
    return grouped


def method_for_policy(by_method: Dict[str, List[Row]], policy: str) -> Tuple[str, str]:
    reference = "proposed_adaptive_margin" if "proposed_adaptive_margin" in by_method else "proposed_laq"
    if policy == "proposed_safe_trigger":
        return reference, "pred_risk_safe"
    if policy == "direct_risk_trigger":
        direct = (
            "scenario_adaptive_direct_risk"
            if "scenario_adaptive_direct_risk" in by_method
            else "direct_risk_conformal"
        )
        return direct, "pred_risk_safe"
    if policy == "ewma_persistence_trigger":
        return "persistence_ewma_safe", "pred_risk_safe"
    if policy == "static_utilization_trigger":
        return reference, "pred_rho"
    raise ValueError(f"No score source for policy {policy}")


def apply_toy_action(row: Row, action: int, high_risk: int) -> Tuple[float, float]:
    base_violation = safe_float(row, "emp_violation")
    base_p95 = safe_float(row, "emp_p95")
    violation_benefit = 0.45 if high_risk else 0.15
    p95_benefit = 0.30 if high_risk else 0.10
    controlled_violation = base_violation * (1.0 - action * violation_benefit)
    controlled_p95 = base_p95 * (1.0 - action * p95_benefit)
    return controlled_violation, controlled_p95


def evaluate_policy_for_scenario(
    seed: int,
    scenario: str,
    policy: str,
    by_method: Dict[str, List[Row]],
    rolling_window: int,
    threshold_quantile: float,
    high_risk_quantile: float,
) -> List[Dict[str, object]]:
    reference = "proposed_adaptive_margin" if "proposed_adaptive_margin" in by_method else "proposed_laq"
    reference_rows = [row for row in by_method.get(reference, []) if row["scenario"] == scenario]
    reference_rows.sort(key=lambda row: int(float(row["window_id"])))
    if len(reference_rows) <= rolling_window:
        return []

    high_threshold = percentile([safe_float(row, "emp_risk") for row in reference_rows], high_risk_quantile)
    rng = random.Random(1700003 + 997 * seed + sum(ord(ch) for ch in scenario + policy))

    if policy == "random_trigger":
        score_rows = reference_rows
        scores = [rng.random() for _ in score_rows]
    else:
        method, key = method_for_policy(by_method, policy)
        score_rows = [row for row in by_method.get(method, []) if row["scenario"] == scenario]
        score_rows.sort(key=lambda row: int(float(row["window_id"])))
        scores = [safe_float(row, key) for row in score_rows]

    rows_by_id = {int(float(row["window_id"])): row for row in reference_rows}
    traces: List[Dict[str, object]] = []
    usable = min(len(score_rows), len(scores))
    for idx in range(rolling_window, usable):
        score_row = score_rows[idx]
        affected_id = int(float(score_row["window_id"]))
        if affected_id not in rows_by_id:
            continue
        past_scores = scores[idx - rolling_window : idx]
        threshold = percentile(past_scores, threshold_quantile)
        action = int(scores[idx] >= threshold)
        affected = rows_by_id[affected_id]
        high_risk = int(safe_float(affected, "emp_risk") >= high_threshold)
        controlled_violation, controlled_p95 = apply_toy_action(affected, action, high_risk)
        false_alarm = int(action and not high_risk)
        traces.append(
            {
                "seed": seed,
                "scenario": scenario,
                "policy": policy,
                "decision_window_id": score_row["window_id"],
                "threshold": threshold,
                "score": scores[idx],
                "action": action,
                "high_risk": high_risk,
                "base_violation": safe_float(affected, "emp_violation"),
                "controlled_violation": controlled_violation,
                "base_p95": safe_float(affected, "emp_p95"),
                "controlled_p95": controlled_p95,
                "false_alarm": false_alarm,
            }
        )
    return traces


def metric_row(seed: int, policy: str, traces: Sequence[Dict[str, object]]) -> Dict[str, object]:
    n = max(1, len(traces))
    base_violation = mean([float(row["base_violation"]) for row in traces])
    controlled_violation = mean([float(row["controlled_violation"]) for row in traces])
    base_p95 = mean([float(row["base_p95"]) for row in traces])
    controlled_p95 = mean([float(row["controlled_p95"]) for row in traces])
    trigger_rate = mean([float(row["action"]) for row in traces])
    false_alarm_cost = mean([float(row["false_alarm"]) for row in traces])
    return {
        "seed": seed,
        "policy": policy,
        "n": n,
        "target_violation_reduction": (base_violation - controlled_violation) / max(base_violation, EPS),
        "p95_delay_reduction": (base_p95 - controlled_p95) / max(base_p95, EPS),
        "trigger_rate": trigger_rate,
        "background_cost": 0.12 * trigger_rate,
        "false_alarm_cost": false_alarm_cost,
        "base_violation": base_violation,
        "controlled_violation": controlled_violation,
        "base_p95": base_p95,
        "controlled_p95": controlled_p95,
    }


def aggregate_metrics(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    output: List[Dict[str, object]] = []
    for policy in sorted(grouped):
        items = grouped[policy]
        out: Dict[str, object] = {
            "policy": policy,
            "runs": len(items),
            "seeds": ";".join(str(row["seed"]) for row in sorted(items, key=lambda r: int(r["seed"]))),
        }
        for metric in CONTROL_METRICS:
            values = [float(row[metric]) for row in items]
            out[f"{metric}_mean"] = f"{mean(values):.10g}"
            out[f"{metric}_std"] = f"{sample_std(values):.10g}"
        output.append(out)
    return output


def fmt_metric(row: Dict[str, object], metric: str) -> str:
    mu = float(row[f"{metric}_mean"])
    sigma = float(row[f"{metric}_std"])
    return f"{mu:.4f} $\\pm$ {sigma:.4f}"


def write_latex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    by_policy = {str(row["policy"]): row for row in rows}
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Policy & Viol. red. & $p95$ red. & Trigger & Bg. cost & False alarms \\",
        r"\hline",
    ]
    for policy in CONTROL_POLICIES:
        if policy not in by_policy:
            continue
        row = by_policy[policy]
        lines.append(
            f"{POLICY_LABELS[policy]} & "
            + f"{fmt_metric(row, 'target_violation_reduction')} & "
            + f"{fmt_metric(row, 'p95_delay_reduction')} & "
            + f"{fmt_metric(row, 'trigger_rate')} & "
            + f"{fmt_metric(row, 'background_cost')} & "
            + f"{fmt_metric(row, 'false_alarm_cost')} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_seed(path: Path) -> int:
    name = path.name
    marker = "seed"
    if marker not in name:
        raise ValueError(f"Cannot infer seed from directory name: {path}")
    suffix = name.split(marker, 1)[1].split("_", 1)[0]
    return int(suffix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("paper1_draft/experiment_outputs"),
        help="Root containing cn_hardening_seed*_margin22 directories.",
    )
    parser.add_argument(
        "--run-glob",
        default="cn_hardening_seed*_margin22",
        help="Directory glob used under input-root.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("paper1_draft/experiment_outputs/revision_audits"),
    )
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--threshold-quantile", type=float, default=0.80)
    parser.add_argument("--high-risk-quantile", type=float, default=0.90)
    args = parser.parse_args()

    all_traces: List[Dict[str, object]] = []
    per_seed_metrics: List[Dict[str, object]] = []

    run_dirs = sorted(args.input_root.glob(args.run_glob), key=infer_seed)
    if not run_dirs:
        raise SystemExit(f"No run directories matched {args.input_root / args.run_glob}")

    for run_dir in run_dirs:
        seed = infer_seed(run_dir)
        rows = read_csv(run_dir / "window_predictions.csv")
        by_method = group_by_method(rows)
        scenarios = sorted({row["scenario"] for row in rows})
        for policy in CONTROL_POLICIES:
            policy_traces: List[Dict[str, object]] = []
            for scenario in scenarios:
                policy_traces.extend(
                    evaluate_policy_for_scenario(
                        seed=seed,
                        scenario=scenario,
                        policy=policy,
                        by_method=by_method,
                        rolling_window=args.rolling_window,
                        threshold_quantile=args.threshold_quantile,
                        high_risk_quantile=args.high_risk_quantile,
                    )
                )
            all_traces.extend(policy_traces)
            per_seed_metrics.append(metric_row(seed, policy, policy_traces))

    aggregate = aggregate_metrics(per_seed_metrics)
    write_csv(all_traces, args.outdir / "closed_loop_control_traces.csv")
    write_csv(per_seed_metrics, args.outdir / "closed_loop_control_metrics_by_seed.csv")
    write_csv(aggregate, args.outdir / "closed_loop_control_summary.csv")
    write_latex(aggregate, args.outdir / "closed_loop_control_table.tex")
    print(f"Read {len(run_dirs)} seed runs")
    print(f"Wrote {len(all_traces)} closed-loop trace rows")
    print(f"Wrote {args.outdir / 'closed_loop_control_summary.csv'}")
    print(f"Wrote {args.outdir / 'closed_loop_control_table.tex'}")


if __name__ == "__main__":
    main()
