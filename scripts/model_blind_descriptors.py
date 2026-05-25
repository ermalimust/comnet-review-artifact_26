#!/usr/bin/env python
"""Observable-descriptor audit for the Computer Networks revision.

This audit tests a model-blind variant of the queueing-input pipeline. Instead
of training on simulator-provided latent queueing descriptors, it constructs a
coarse descriptor vector directly from observable noisy window features and
then applies the same analytical map and validation calibration.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import run_simulation as sim


SCENARIOS = ["overall", "load_high", "vacation_high", "drift_strong"]
METHODS = ["learned_descriptor", "observable_descriptor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--train-windows", type=int, default=1200)
    parser.add_argument("--test-windows", type=int, default=250)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--violation-tau", type=float, default=0.45)
    parser.add_argument("--margin-scale", type=float, default=2.2)
    parser.add_argument(
        "--outdir",
        default=r"paper1_draft/experiment_outputs/revision_audits",
    )
    parser.add_argument(
        "--table-dir",
        default=r"paper1_draft/computer_networks_revision/tables/des",
    )
    return parser.parse_args()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def observable_theta(sample: sim.WindowSample) -> Dict[str, float]:
    """Build a queueing-input vector from observed features only.

    The feature vector is intentionally restricted to noisy measurements that a
    lightweight window monitor could expose: arrival-rate hint, service hint,
    unavailable-fraction hint, burstiness hint, channel/SNR-like hint, packet
    size hint, and traffic-class hint. It does not use the simulator's latent
    per-window ``theta`` values.
    """

    lambda_obs, service_hint, u_hint, b_hint, snr_like, packet_size_hint, target_video = sample.features[:7]
    m1 = clamp(service_hint, 0.003, 0.35)
    lam = clamp(lambda_obs, 0.05, 120.0)
    u = clamp(u_hint, 0.0, 0.92)
    b = clamp(b_hint, 0.35, 8.0)

    snr_penalty = clamp(1.0 - snr_like, 0.0, 1.0)
    size_penalty = clamp(packet_size_hint - 0.7, 0.0, 1.0)
    service_cv = clamp(0.42 + 0.28 * target_video + 0.10 * snr_penalty + 0.12 * size_penalty, 0.35, 1.35)
    m2 = m1 * m1 * (1.0 + service_cv * service_cv)

    # Coarse unavailable-interval moments inferred from occupancy only. The
    # scaling is deliberately simple so the audit measures the cost of using a
    # monitor-level proxy rather than simulator-provided interruption moments.
    v1 = clamp(0.018 + 0.16 * u, 0.0, 0.25)
    shape_factor = clamp(1.35 + 0.30 * b + 0.50 * u, 1.2, 4.0)
    v2 = shape_factor * v1 * v1

    return {
        "lambda": lam,
        "m1": m1,
        "m2": m2,
        "v1": v1,
        "v2": v2,
        "b": b,
        "u": u,
    }


def calibrate_observable_margin(
    samples: Sequence[sim.WindowSample],
    violation_tau: float,
) -> float:
    residuals = []
    for sample in samples:
        pred = sim.queueing_analyzer(observable_theta(sample), violation_tau=violation_tau)
        residuals.append(abs(pred["risk"] - sample.empirical["risk"]))
    return sim.percentile(residuals, 0.90) if residuals else 0.0


def row_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    screening = sim.screening_metrics(rows)
    return {
        "delay_mae": sim.mean([float(row["delay_abs_error"]) for row in rows]),
        "risk_mae": sim.mean([float(row["risk_abs_error"]) for row in rows]),
        "coverage": sim.mean([float(row["safe_covers"]) for row in rows]),
        "auroc": screening["screening_auroc"],
        "recall_at_10": screening["recall_at_10"],
    }


def evaluate_seed(seed: int, args: argparse.Namespace) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    train, test = sim.generate_dataset(
        rng,
        args.train_windows,
        args.test_windows,
        args.horizon,
        args.violation_tau,
    )
    fit_samples, val_samples = sim.split_train_validation(train)

    structured = sim.MultiTargetLogModel(sim.THETA_KEYS, alpha=0.05)
    structured.fit(fit_samples, source="theta")
    projector = sim.ThetaProjector(eta=0.06)
    projector.fit(fit_samples)

    learned_margin = sim.calibrate_margin(val_samples, structured, projector, args.violation_tau)
    observable_margin = calibrate_observable_margin(val_samples, args.violation_tau)

    prediction_rows: List[Dict[str, object]] = []
    for idx, sample in enumerate(test):
        if sample.scenario not in SCENARIOS:
            continue
        adaptive_multiplier = sim.descriptor_adaptive_multiplier(sample)

        learned_theta = projector.project(structured.predict(sample))
        learned_pred = sim.queueing_analyzer(
            learned_theta,
            risk_margin=args.margin_scale * learned_margin * adaptive_multiplier,
            violation_tau=args.violation_tau,
        )
        observable_pred = sim.queueing_analyzer(
            observable_theta(sample),
            risk_margin=args.margin_scale * observable_margin * adaptive_multiplier,
            violation_tau=args.violation_tau,
        )
        for method, pred in [
            ("learned_descriptor", learned_pred),
            ("observable_descriptor", observable_pred),
        ]:
            safe_risk = pred["risk_safe"]
            prediction_rows.append(
                {
                    "seed": seed,
                    "window_id": idx,
                    "scenario": sample.scenario,
                    "method": method,
                    "emp_delay_mean": sample.empirical["delay_mean"],
                    "pred_delay_mean": pred["delay_mean"],
                    "emp_risk": sample.empirical["risk"],
                    "pred_risk": pred["risk"],
                    "pred_risk_safe": safe_risk,
                    "emp_p95": sample.empirical["p95"],
                    "pred_p95": pred["p95"],
                    "emp_violation": sample.empirical["violation"],
                    "pred_violation": pred["violation"],
                    "delay_abs_error": abs(pred["delay_mean"] - sample.empirical["delay_mean"]),
                    "risk_abs_error": abs(pred["risk"] - sample.empirical["risk"]),
                    "safe_covers": 1.0 if safe_risk >= sample.empirical["risk"] else 0.0,
                }
            )

    output = []
    for scenario in SCENARIOS:
        for method in METHODS:
            rows = [row for row in prediction_rows if row["scenario"] == scenario and row["method"] == method]
            metrics = row_metrics(rows)
            output.append({"seed": seed, "scenario": scenario, "method": method, **metrics})
    return output


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario"]), str(row["method"]))].append(row)

    output = []
    for (scenario, method), items in sorted(grouped.items()):
        entry = {"scenario": scenario, "method": method, "seeds": ";".join(str(row["seed"]) for row in items)}
        for metric in ["delay_mae", "risk_mae", "coverage", "auroc", "recall_at_10"]:
            values = [float(row[metric]) for row in items]
            entry[metric] = mean(values)
        output.append(entry)
    return output


def read_pair(aggregate_rows: Sequence[Dict[str, object]], scenario: str, method: str, metric: str) -> float:
    for row in aggregate_rows:
        if row["scenario"] == scenario and row["method"] == method:
            return float(row[metric])
    return 0.0


def scenario_tex(name: str) -> str:
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def write_latex(path: Path, aggregate_rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Scenario & Risk MAE L/O & Coverage L/O & AUROC L/O & R@10\% L/O \\",
        r"\hline",
    ]
    for scenario in SCENARIOS:
        learned_risk = read_pair(aggregate_rows, scenario, "learned_descriptor", "risk_mae")
        obs_risk = read_pair(aggregate_rows, scenario, "observable_descriptor", "risk_mae")
        learned_cov = read_pair(aggregate_rows, scenario, "learned_descriptor", "coverage")
        obs_cov = read_pair(aggregate_rows, scenario, "observable_descriptor", "coverage")
        learned_auc = read_pair(aggregate_rows, scenario, "learned_descriptor", "auroc")
        obs_auc = read_pair(aggregate_rows, scenario, "observable_descriptor", "auroc")
        learned_rec = read_pair(aggregate_rows, scenario, "learned_descriptor", "recall_at_10")
        obs_rec = read_pair(aggregate_rows, scenario, "observable_descriptor", "recall_at_10")
        lines.append(
            f"{scenario_tex(scenario)} & "
            f"{learned_risk:.4f}/{obs_risk:.4f} & "
            f"{learned_cov:.3f}/{obs_cov:.3f} & "
            f"{learned_auc:.3f}/{obs_auc:.3f} & "
            f"{learned_rec:.3f}/{obs_rec:.3f} \\\\"
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
    all_rows: List[Dict[str, object]] = []
    for seed in seeds:
        all_rows.extend(evaluate_seed(seed, args))

    aggregate_rows = aggregate(all_rows)
    outdir = Path(args.outdir)
    table_dir = Path(args.table_dir)
    write_csv(outdir / "model_blind_descriptor_seed_summary.csv", all_rows)
    write_csv(outdir / "model_blind_descriptor_aggregate.csv", aggregate_rows)
    write_latex(outdir / "model_blind_descriptor_table.tex", aggregate_rows)
    write_latex(table_dir / "model_blind_descriptor_table.tex", aggregate_rows)
    print(f"Wrote {outdir / 'model_blind_descriptor_aggregate.csv'}")
    print(f"Wrote {table_dir / 'model_blind_descriptor_table.tex'}")


if __name__ == "__main__":
    main()
