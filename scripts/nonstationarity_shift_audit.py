#!/usr/bin/env python
"""Mid-trace distribution-shift audit for the Computer Networks revision.

The audit builds a sequential DES trace whose operating regime changes halfway
through the run. It compares the original fixed validation calibration with a
rolling residual calibration that updates after observed windows become
available. The goal is a compact reviewer-facing stress test for non-stationary
arrival, service, interruption, and drift conditions.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

import run_simulation as sim


PHASES = [
    ("pre_shift", "Pre-shift", 0, 80),
    ("post_early", "Post early", 80, 120),
    ("post_mid", "Post mid", 120, 160),
    ("post_late", "Post late", 160, 240),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--train-windows", type=int, default=1200)
    parser.add_argument("--trace-windows", type=int, default=240)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--violation-tau", type=float, default=0.45)
    parser.add_argument("--margin-scale", type=float, default=2.2)
    parser.add_argument("--rolling-window", type=int, default=40)
    parser.add_argument(
        "--outdir",
        default=r"paper1_draft/experiment_outputs/revision_audits",
    )
    parser.add_argument(
        "--table-dir",
        default=r"paper1_draft/computer_networks_revision/tables/des",
    )
    return parser.parse_args()


def phase_for_index(idx: int) -> Tuple[str, str]:
    for phase_id, label, start, end in PHASES:
        if start <= idx < end:
            return phase_id, label
    return PHASES[-1][0], PHASES[-1][1]


def fit_pipeline(
    seed: int,
    args: argparse.Namespace,
) -> Tuple[sim.MultiTargetLogModel, sim.ThetaProjector, float, List[float]]:
    rng = random.Random(seed)
    train, _ = sim.generate_dataset(
        rng,
        args.train_windows,
        0,
        args.horizon,
        args.violation_tau,
    )
    fit_samples, val_samples = sim.split_train_validation(train)
    model = sim.MultiTargetLogModel(sim.THETA_KEYS, alpha=0.05)
    model.fit(fit_samples, source="theta")
    projector = sim.ThetaProjector(eta=0.06)
    projector.fit(fit_samples)
    residuals = []
    for sample in val_samples:
        theta = projector.project(model.predict(sample))
        pred = sim.queueing_analyzer(theta, violation_tau=args.violation_tau)
        residuals.append(abs(pred["risk"] - sample.empirical["risk"]))
    base_margin = sim.percentile(residuals, 0.90) if residuals else 0.0
    return model, projector, base_margin, residuals


def generate_trace(seed: int, args: argparse.Namespace) -> List[sim.WindowSample]:
    rng = random.Random(7919 + seed)
    pre = sim.Scenario(
        "pre_shift",
        load_scale=1.0,
        vacation_scale=1.0,
        burst_scale=1.0,
        drift_level=0.0,
    )
    post = sim.Scenario(
        "post_shift",
        load_scale=1.60,
        vacation_scale=1.95,
        burst_scale=1.55,
        drift_level=1.25,
        channel_scale=1.15,
    )
    trace = []
    switch = args.trace_windows // 3
    for idx in range(args.trace_windows):
        scenario = pre if idx < switch else post
        trace.append(sim.simulate_window(rng, scenario, "trace", args.horizon, args.violation_tau))
    return trace


def evaluate_seed(seed: int, args: argparse.Namespace) -> List[Dict[str, object]]:
    model, projector, base_margin, residuals = fit_pipeline(seed, args)
    rolling: Deque[float] = deque(residuals[-args.rolling_window :], maxlen=args.rolling_window)
    rows = []
    for idx, sample in enumerate(generate_trace(seed, args)):
        theta = projector.project(model.predict(sample))
        nominal = sim.queueing_analyzer(theta, violation_tau=args.violation_tau)
        adaptive_multiplier = sim.descriptor_adaptive_multiplier(sample)
        rolling_margin = sim.percentile(list(rolling), 0.90) if rolling else base_margin
        phase_id, phase_label = phase_for_index(idx)
        methods = {
            "fixed_validation": base_margin,
            "rolling_calibration": rolling_margin,
        }
        residual = abs(nominal["risk"] - sample.empirical["risk"])
        for method, margin in methods.items():
            safe_risk = nominal["risk"] + args.margin_scale * margin * adaptive_multiplier
            rows.append(
                {
                    "seed": seed,
                    "window_id": idx,
                    "phase": phase_id,
                    "phase_label": phase_label,
                    "method": method,
                    "emp_risk": sample.empirical["risk"],
                    "pred_risk": nominal["risk"],
                    "pred_risk_safe": safe_risk,
                    "risk_abs_error": abs(nominal["risk"] - sample.empirical["risk"]),
                    "safe_covers": 1.0 if safe_risk >= sample.empirical["risk"] else 0.0,
                    "shortfall": max(0.0, sample.empirical["risk"] - safe_risk),
                    "margin": margin,
                    "adaptive_multiplier": adaptive_multiplier,
                }
            )
        rolling.append(residual)
    return rows


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["phase"]), str(row["method"]))].append(row)

    output = []
    order = {phase_id: idx for idx, (phase_id, _, _, _) in enumerate(PHASES)}
    for (phase, method), items in sorted(grouped.items(), key=lambda item: (order[item[0][0]], item[0][1])):
        output.append(
            {
                "phase": phase,
                "phase_label": str(items[0]["phase_label"]),
                "method": method,
                "n": len(items),
                "coverage": mean([float(row["safe_covers"]) for row in items]),
                "mean_shortfall": mean([float(row["shortfall"]) for row in items]),
                "risk_mae": mean([float(row["risk_abs_error"]) for row in items]),
                "mean_margin": mean([float(row["margin"]) for row in items]),
            }
        )
    return output


def method_value(rows: Sequence[Dict[str, object]], phase: str, method: str, key: str) -> float:
    for row in rows:
        if row["phase"] == phase and row["method"] == method:
            return float(row[key])
    return 0.0


def write_latex(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Trace phase & Fixed cov. & Rolling cov. & Fixed shortfall & Rolling shortfall \\",
        r"\hline",
    ]
    for phase, label, _, _ in PHASES:
        fixed_cov = method_value(rows, phase, "fixed_validation", "coverage")
        rolling_cov = method_value(rows, phase, "rolling_calibration", "coverage")
        fixed_short = method_value(rows, phase, "fixed_validation", "mean_shortfall")
        rolling_short = method_value(rows, phase, "rolling_calibration", "mean_shortfall")
        lines.append(
            f"{label} & {fixed_cov:.3f} & {rolling_cov:.3f} & "
            f"{fixed_short:.4f} & {rolling_short:.4f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        rows.extend(evaluate_seed(seed, args))
    aggregate_rows = aggregate(rows)
    outdir = Path(args.outdir)
    table_dir = Path(args.table_dir)
    write_csv(outdir / "nonstationarity_shift_window_predictions.csv", rows)
    write_csv(outdir / "nonstationarity_shift_summary.csv", aggregate_rows)
    write_latex(outdir / "nonstationarity_shift_table.tex", aggregate_rows)
    write_latex(table_dir / "nonstationarity_shift_table.tex", aggregate_rows)
    print(f"Wrote {outdir / 'nonstationarity_shift_summary.csv'}")
    print(f"Wrote {table_dir / 'nonstationarity_shift_table.tex'}")


if __name__ == "__main__":
    main()
