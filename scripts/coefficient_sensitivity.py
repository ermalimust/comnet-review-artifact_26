#!/usr/bin/env python
"""Coefficient-sensitivity audit for the Computer Networks revision.

The audit reruns the existing DES learner with fixed seed splits and varies the
interruption-correction coefficients one at a time. It writes compact CSV and
LaTeX summaries for the revised manuscript.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import run_simulation as sim


BASE_COEFFS = {"c_v": 0.35, "c_u": 0.15, "c_b": 0.12, "delta_u": 0.08}
SCENARIOS = ["overall", "load_high", "vacation_high", "drift_strong"]
METRICS = ["delay_mae", "risk_mae", "coverage", "auroc", "recall_at_10"]


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


def variants() -> List[Tuple[str, str, Dict[str, float]]]:
    output = [("baseline", "Baseline", dict(BASE_COEFFS))]
    for key, label in [("c_v", r"$c_v$"), ("c_u", r"$c_u$"), ("c_b", r"$c_b$")]:
        for scale in [0.0, 0.5, 1.5, 2.0]:
            coeffs = dict(BASE_COEFFS)
            coeffs[key] = BASE_COEFFS[key] * scale
            output.append((f"{key}_{scale:g}x", f"{label} {scale:g}x", coeffs))
    for value in [0.04, 0.12, 0.16]:
        coeffs = dict(BASE_COEFFS)
        coeffs["delta_u"] = value
        output.append((f"delta_u_{value:g}", rf"$\delta_u={value:.2f}$", coeffs))
    return output


def queueing_analyzer_coeff(
    theta: Dict[str, float],
    coeffs: Dict[str, float],
    risk_margin: float = 0.0,
    violation_tau: float = 0.45,
) -> Dict[str, float]:
    lam = max(sim.EPS, theta["lambda"])
    m1 = max(sim.EPS, theta["m1"])
    m2 = max(m1 * m1, theta["m2"])
    v1 = max(0.0, theta["v1"])
    v2 = max(0.0, theta["v2"])
    b = max(0.0, theta["b"])
    u = max(0.0, min(0.95, theta["u"]))

    rho = lam * m1
    stable_gap = max(0.04, 1.0 - min(rho, 0.96))
    mg1_wait = lam * m2 / (2.0 * stable_gap)
    residual_vac = v2 / (2.0 * v1) if v1 > sim.EPS else 0.0
    vacation_wait = (
        coeffs["c_v"] * u * residual_vac / max(coeffs["delta_u"], 1.0 - u)
        + coeffs["c_u"] * u * m1
    )
    burst_wait = coeffs["c_b"] * max(0.0, b - 1.0) * m1

    wait = mg1_wait + vacation_wait + burst_wait
    delay = wait + m1
    service_var = max(0.0, m2 - m1 * m1)
    tail_scale = math.sqrt(service_var + v2 + 0.25 * wait * wait + sim.EPS)
    p95 = delay + 1.64 * tail_scale
    p99 = delay + 2.33 * tail_scale
    if violation_tau <= delay:
        violation = min(1.0, 0.50 + (delay - violation_tau) / max(delay + tail_scale, sim.EPS))
    else:
        violation = math.exp(-(violation_tau - delay) / max(tail_scale, 0.02))
    violation = max(0.0, min(1.0, violation))
    risk = sim.metric_risk(delay, p95, violation)
    return {
        "delay_mean": delay,
        "risk": risk,
        "risk_safe": risk + risk_margin,
        "p95": p95,
        "p99": p99,
        "violation": violation,
        "rho": rho,
    }


def calibrate_margin_coeff(
    samples: Sequence[sim.WindowSample],
    model: sim.MultiTargetLogModel,
    projector: sim.ThetaProjector,
    coeffs: Dict[str, float],
    violation_tau: float,
) -> float:
    residuals = []
    for sample in samples:
        pred_theta = projector.project(model.predict(sample))
        pred = queueing_analyzer_coeff(pred_theta, coeffs, violation_tau=violation_tau)
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

    output = []
    for variant_id, variant_label, coeffs in variants():
        base_margin = calibrate_margin_coeff(
            val_samples,
            structured,
            projector,
            coeffs,
            args.violation_tau,
        )
        prediction_rows = []
        for idx, sample in enumerate(test):
            if sample.scenario not in SCENARIOS:
                continue
            theta = projector.project(structured.predict(sample))
            adaptive_margin = args.margin_scale * base_margin * sim.descriptor_adaptive_multiplier(sample)
            pred = queueing_analyzer_coeff(
                theta,
                coeffs,
                risk_margin=adaptive_margin,
                violation_tau=args.violation_tau,
            )
            safe_risk = pred["risk_safe"]
            prediction_rows.append(
                {
                    "seed": seed,
                    "window_id": idx,
                    "scenario": sample.scenario,
                    "variant": variant_id,
                    "variant_label": variant_label,
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

        for scenario in SCENARIOS:
            rows = [row for row in prediction_rows if row["scenario"] == scenario]
            metrics = row_metrics(rows)
            output.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "variant": variant_id,
                    "variant_label": variant_label,
                    **metrics,
                }
            )
    return output


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def aggregate(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario"]), str(row["variant"]))].append(row)

    output = []
    for (scenario, variant), items in sorted(grouped.items()):
        out: Dict[str, object] = {
            "scenario": scenario,
            "variant": variant,
            "variant_label": items[0]["variant_label"],
            "runs": len(items),
            "seeds": ";".join(str(item["seed"]) for item in sorted(items, key=lambda row: int(row["seed"]))),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = sample_std(values)
        output.append(out)
    return output


def range_text(values: Sequence[float], decimals: int) -> str:
    return f"{min(values):.{decimals}f}--{max(values):.{decimals}f}"


def scenario_tex(name: str) -> str:
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def summarize_ranges(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output = []
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        baseline = next(row for row in scenario_rows if row["variant"] == "baseline")
        output.append(
            {
                "scenario": scenario,
                "risk_base": baseline["risk_mae_mean"],
                "risk_range": range_text([float(row["risk_mae_mean"]) for row in scenario_rows], 4),
                "coverage_base": baseline["coverage_mean"],
                "coverage_range": range_text([float(row["coverage_mean"]) for row in scenario_rows], 3),
                "auroc_base": baseline["auroc_mean"],
                "auroc_range": range_text([float(row["auroc_mean"]) for row in scenario_rows], 3),
                "recall_base": baseline["recall_at_10_mean"],
                "recall_range": range_text([float(row["recall_at_10_mean"]) for row in scenario_rows], 3),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Scenario & Risk MAE base/range & Coverage base/range & AUROC base/range & R@10\% base/range \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{scenario_tex(str(row['scenario']))} & "
            f"{float(row['risk_base']):.4f} / {row['risk_range']} & "
            f"{float(row['coverage_base']):.3f} / {row['coverage_range']} & "
            f"{float(row['auroc_base']):.3f} / {row['auroc_range']} & "
            f"{float(row['recall_base']):.3f} / {row['recall_range']} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    all_rows: List[Dict[str, object]] = []
    for seed in seeds:
        all_rows.extend(evaluate_seed(seed, args))
    aggregate_rows = aggregate(all_rows)
    range_rows = summarize_ranges(aggregate_rows)

    outdir = Path(args.outdir)
    table_dir = Path(args.table_dir)
    write_csv(outdir / "coefficient_sensitivity_seed_summary.csv", all_rows)
    write_csv(outdir / "coefficient_sensitivity_aggregate.csv", aggregate_rows)
    write_csv(outdir / "coefficient_sensitivity_ranges.csv", range_rows)
    write_latex(range_rows, outdir / "coefficient_sensitivity_table.tex")
    write_latex(range_rows, table_dir / "coefficient_sensitivity_table.tex")

    print(f"Wrote {outdir / 'coefficient_sensitivity_ranges.csv'}")
    print(f"Wrote {table_dir / 'coefficient_sensitivity_table.tex'}")


if __name__ == "__main__":
    main()
