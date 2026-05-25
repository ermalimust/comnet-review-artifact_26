"""Frequency identifiability audit for unavailable-interval descriptors.

The reviewer concern is that the first two unavailable-interval moments
``v1`` and ``v2`` do not encode interruption frequency. This audit fixes the
active-service windows and the duration distribution of individual unavailable
intervals, then repeats the same duration template at low/mid/high frequencies.
Thus ``v1`` and ``v2`` stay fixed while ``u`` changes with interruption count.
The audit tests whether the revised descriptor tuple uses ``u`` to separate
systems that share the same unavailable-interval duration moments.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import run_simulation as sim


LEVELS = [
    ("low", 1),
    ("mid", 2),
    ("high", 3),
]


def parse_seed_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def second_moment(values: Sequence[float]) -> float:
    return sum(v * v for v in values) / len(values) if values else 0.0


def repeated_durations(template: Sequence[float], repetitions: int) -> List[float]:
    return [duration for _ in range(repetitions) for duration in template]


def place_even(durations: Sequence[float], horizon: float) -> List[Tuple[float, float]]:
    total = sum(durations)
    slack = max(0.0, horizon - total)
    gap = slack / (len(durations) + 1)
    t = gap
    out: List[Tuple[float, float]] = []
    for duration in durations:
        out.append((t, t + duration))
        t += duration + gap
    return out


def simulate_controlled_window(
    arrivals: Sequence[float],
    active_services: Sequence[float],
    vacations: Sequence[Tuple[float, float]],
    violation_tau: float,
) -> Dict[str, float]:
    current = 0.0
    delays: List[float] = []
    for arrival, active_service in zip(arrivals, active_services):
        start = max(arrival, current)
        start = sim.advance_to_available(start, vacations)
        completion = sim.serve_with_vacations(start, active_service, vacations)
        current = completion
        delays.append(max(0.0, completion - arrival))

    delay_mean = sim.mean(delays, 0.0)
    p95 = sim.percentile(delays, 0.95)
    p99 = sim.percentile(delays, 0.99)
    violation = sum(1 for delay in delays if delay > violation_tau) / len(delays) if delays else 0.0
    risk = delay_mean + 0.35 * p95 + 1.5 * violation
    return {
        "delay_mean": delay_mean,
        "p95": p95,
        "p99": p99,
        "violation": violation,
        "risk": risk,
    }


def make_base_window(
    rng: random.Random,
    horizon: float,
) -> Tuple[List[float], List[float], List[float], Dict[str, float]]:
    service_mean = rng.uniform(0.046, 0.062)
    service_cv = rng.uniform(0.62, 0.86)
    lam = rng.uniform(7.0, 9.8)
    arrivals = sim.generate_arrivals(rng, lam, burst_scale=1.10, horizon=horizon)
    active_services = [sim.lognormal_with_mean_cv(rng, service_mean, service_cv) for _ in arrivals]

    template_count = 4
    base_duration = rng.uniform(0.13, 0.20)
    template = [base_duration * rng.uniform(0.82, 1.18) for _ in range(template_count)]

    high_total = sum(repeated_durations(template, LEVELS[-1][1]))
    if high_total > 0.28 * horizon:
        scale = 0.28 * horizon / high_total
        template = [duration * scale for duration in template]

    theta_base = {
        "lambda": max(sim.EPS, len(arrivals) / horizon),
        "m1": sim.mean(active_services, service_mean),
        "m2": second_moment(active_services),
        "v1": sim.mean(template, 0.0),
        "v2": second_moment(template),
        "b": sim.interarrival_burstiness(arrivals),
    }
    return arrivals, active_services, template, theta_base


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return sim.percentile(values, q)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(max(0.0, var))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Frequency & Count & Mean $u$ & Emp. risk (s) & Nominal MAE (s) & Cal. coverage \\",
        r"\hline",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['level']} & "
            + f"{float(row['count_mean']):.1f} & "
            + f"{float(row['u_mean']):.3f} & "
            + f"{float(row['emp_risk_mean']):.4f} $\\pm$ {float(row['emp_risk_std']):.4f} & "
            + f"{float(row['nominal_mae_mean']):.4f} $\\pm$ {float(row['nominal_mae_std']):.4f} & "
            + f"{float(row['coverage_mean']):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> None:
    seeds = parse_seed_list(args.seeds)
    rows: List[Dict[str, object]] = []

    for seed in seeds:
        rng = random.Random(seed)
        for window_idx in range(args.windows_per_seed):
            arrivals, active_services, template, theta_base = make_base_window(rng, args.horizon)
            for level, repetitions in LEVELS:
                durations = repeated_durations(template, repetitions)
                vacations = place_even(durations, args.horizon)
                theta = dict(theta_base)
                theta["u"] = max(0.0, min(0.95, sum(durations) / args.horizon))
                pred = sim.queueing_analyzer(theta, risk_margin=0.0, violation_tau=args.violation_tau)
                empirical = simulate_controlled_window(arrivals, active_services, vacations, args.violation_tau)
                residual = empirical["risk"] - pred["risk"]
                split = "calibration" if window_idx < args.calibration_windows else "test"
                rows.append(
                    {
                        "seed": seed,
                        "window": window_idx,
                        "split": split,
                        "level": level,
                        "interruption_count": len(durations),
                        "interruption_frequency": len(durations) / args.horizon,
                        "lambda": theta["lambda"],
                        "m1": theta["m1"],
                        "m2": theta["m2"],
                        "v1": theta["v1"],
                        "v2": theta["v2"],
                        "u": theta["u"],
                        "b": theta["b"],
                        "pred_risk": pred["risk"],
                        "emp_risk": empirical["risk"],
                        "emp_delay": empirical["delay_mean"],
                        "emp_p95": empirical["p95"],
                        "emp_violation": empirical["violation"],
                        "residual": residual,
                        "abs_residual": abs(residual),
                    }
                )

    calibration_abs = [float(row["abs_residual"]) for row in rows if row["split"] == "calibration"]
    margin = percentile(calibration_abs, args.coverage_quantile)

    summary_rows: List[Dict[str, object]] = []
    for level, _repetitions in LEVELS:
        selected = [row for row in rows if row["split"] == "test" and row["level"] == level]
        emp_risk = [float(row["emp_risk"]) for row in selected]
        nominal_abs = [abs(float(row["pred_risk"]) - float(row["emp_risk"])) for row in selected]
        coverage = [
            1.0 if float(row["pred_risk"]) + margin >= float(row["emp_risk"]) else 0.0
            for row in selected
        ]
        counts = [float(row["interruption_count"]) for row in selected]
        occupancies = [float(row["u"]) for row in selected]
        emp_mean, emp_std = mean_std(emp_risk)
        mae_mean, mae_std = mean_std(nominal_abs)
        count_mean, _count_std = mean_std(counts)
        u_mean, _u_std = mean_std(occupancies)
        summary_rows.append(
            {
                "level": level,
                "n": len(selected),
                "calibration_margin": margin,
                "count_mean": count_mean,
                "u_mean": u_mean,
                "emp_risk_mean": emp_mean,
                "emp_risk_std": emp_std,
                "nominal_mae_mean": mae_mean,
                "nominal_mae_std": mae_std,
                "coverage_mean": sum(coverage) / len(coverage) if coverage else 0.0,
            }
        )

    outdir = Path(args.outdir)
    table_dir = Path(args.table_dir) if args.table_dir else None
    write_csv(outdir / "identifiability_window_predictions.csv", rows)
    write_csv(outdir / "identifiability_summary.csv", summary_rows)
    write_latex(summary_rows, outdir / "identifiability_audit_table.tex")
    if table_dir:
        write_latex(summary_rows, table_dir / "identifiability_audit_table.tex")

    print(f"Wrote {outdir / 'identifiability_summary.csv'}")
    print(f"Wrote {outdir / 'identifiability_audit_table.tex'}")
    print(f"Calibration margin: {margin:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default="7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83",
    )
    parser.add_argument("--windows-per-seed", type=int, default=120)
    parser.add_argument("--calibration-windows", type=int, default=40)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--violation-tau", type=float, default=0.45)
    parser.add_argument("--coverage-quantile", type=float, default=0.90)
    parser.add_argument("--outdir", default="paper1_draft/experiment_outputs/revision_audits")
    parser.add_argument("--table-dir", default="paper1_draft/computer_networks_revision/tables/des")
    return parser.parse_args()


if __name__ == "__main__":
    run_audit(parse_args())
