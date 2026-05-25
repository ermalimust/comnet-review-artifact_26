#!/usr/bin/env python
"""Revision-stage statistical and robustness audits.

This script reads existing multi-seed CSV outputs and produces compact LaTeX
tables for the revised manuscript. It intentionally uses only the Python
standard library so the audit can be rerun as part of the artifact package.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SEED_COMPARISONS = [
    ("overall", "risk_mae", "proposed_adaptive_margin", "static_queueing", "Risk MAE vs static", "lower"),
    ("load_high", "safe_coverage", "proposed_adaptive_margin", "proposed_laq", "Coverage vs global", "higher"),
    ("vacation_high", "safe_coverage", "proposed_adaptive_margin", "proposed_laq", "Coverage vs global", "higher"),
    ("drift_strong", "safe_coverage", "proposed_adaptive_margin", "proposed_laq", "Coverage vs global", "higher"),
    ("drift_strong", "risk_mae", "proposed_adaptive_margin", "direct_risk_conformal", "Risk MAE vs direct", "lower"),
    ("drift_strong", "recall_at_10", "proposed_adaptive_margin", "direct_risk_conformal", "R@10 vs direct", "higher"),
]

WEIGHT_SCENARIOS = ["overall", "load_high", "vacation_high", "drift_strong"]
BETAS = [0.20, 0.35, 0.50]
OMEGAS = [1.00, 1.50, 2.00]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-pattern",
        default=r"paper1_draft/experiment_outputs/cn_seed*_margin22",
        help="Glob pattern for per-seed DES output directories.",
    )
    parser.add_argument(
        "--outdir",
        default=r"paper1_draft/experiment_outputs/revision_audits",
        help="Directory for CSV and LaTeX audit outputs.",
    )
    parser.add_argument(
        "--table-dir",
        default=r"paper1_draft/computer_networks_revision/tables/des",
        help="Optional manuscript table directory to receive LaTeX snippets.",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def seed_from_path(path: Path) -> str:
    match = re.search(r"seed(\d+)", path.name)
    return match.group(1) if match else path.name


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def average_ranks(abs_values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(abs_values), key=lambda item: item[1])
    ranks = [0.0] * len(abs_values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and abs(indexed[j][1] - indexed[i][1]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def wilcoxon_exact_p(diffs: Sequence[float]) -> float:
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    ranks = average_ranks([abs(d) for d in nonzero])
    w_plus = sum(rank for rank, diff in zip(ranks, nonzero) if diff > 0)
    total = sum(ranks)
    observed = min(w_plus, total - w_plus)
    count = 0
    extreme = 0
    for mask in range(1 << n):
        w = sum(ranks[i] for i in range(n) if mask & (1 << i))
        stat = min(w, total - w)
        count += 1
        if stat <= observed + 1e-12:
            extreme += 1
    return min(1.0, extreme / count)


def bootstrap_ci(values: Sequence[float], draws: int) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(20260525)
    means = []
    n = len(values)
    for _ in range(draws):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    return percentile(means, 0.025), percentile(means, 0.975)


def scenario_tex(name: str) -> str:
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def collect_seed_rows(seed_dirs: Sequence[Path]) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    rows: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for run_dir in seed_dirs:
        seed = seed_from_path(run_dir)
        for row in read_csv(run_dir / "summary_metrics.csv"):
            rows[(seed, row["scenario"], row["method"])] = row
    return rows


def statistical_audit(seed_dirs: Sequence[Path], draws: int) -> List[Dict[str, object]]:
    by_key = collect_seed_rows(seed_dirs)
    seeds = sorted({seed_from_path(path) for path in seed_dirs}, key=lambda value: int(value))
    output: List[Dict[str, object]] = []
    for scenario, metric, proposed, comparator, label, direction in SEED_COMPARISONS:
        improvements = []
        used = []
        for seed in seeds:
            p_key = (seed, scenario, proposed)
            c_key = (seed, scenario, comparator)
            if p_key not in by_key or c_key not in by_key:
                continue
            p_value = float(by_key[p_key][metric])
            c_value = float(by_key[c_key][metric])
            improvement = c_value - p_value if direction == "lower" else p_value - c_value
            improvements.append(improvement)
            used.append(seed)
        ci_lo, ci_hi = bootstrap_ci(improvements, draws)
        output.append(
            {
                "scenario": scenario,
                "comparison": label,
                "metric": metric,
                "seeds": ";".join(used),
                "mean_improvement": mean(improvements),
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "wilcoxon_p": wilcoxon_exact_p(improvements),
            }
        )
    return output


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif abs(pos - neg) < 1e-12:
                wins += 0.5
            total += 1
    return wins / total


def top_decile_metrics(emp_scores: Sequence[float], pred_scores: Sequence[float]) -> Tuple[float, float]:
    n = len(emp_scores)
    k = max(1, int(round(0.10 * n)))
    empirical_top = set(sorted(range(n), key=lambda i: emp_scores[i], reverse=True)[:k])
    predicted_top = set(sorted(range(n), key=lambda i: pred_scores[i], reverse=True)[:k])
    recall = len(empirical_top & predicted_top) / k
    labels = [1 if i in empirical_top else 0 for i in range(n)]
    return auroc(pred_scores, labels), recall


def metric_range(values: Sequence[float]) -> str:
    return f"{min(values):.3f}--{max(values):.3f}"


def risk_weight_audit(seed_dirs: Sequence[Path]) -> List[Dict[str, object]]:
    rows = []
    for run_dir in seed_dirs:
        seed = seed_from_path(run_dir)
        for row in read_csv(run_dir / "window_predictions.csv"):
            if row["method"] == "proposed_adaptive_margin" and row["scenario"] in WEIGHT_SCENARIOS:
                row["seed"] = seed
                rows.append(row)

    output = []
    for scenario in WEIGHT_SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        seed_values = sorted({row["seed"] for row in scenario_rows}, key=lambda value: int(value))
        grid = []
        baseline = None
        for beta in BETAS:
            for omega in OMEGAS:
                seed_coverages = []
                seed_aurocs = []
                seed_recalls = []
                for seed in seed_values:
                    seed_rows = [row for row in scenario_rows if row["seed"] == seed]
                    emp_scores = [
                        float(row["emp_delay_mean"]) + beta * float(row["emp_p95"]) + omega * float(row["emp_violation"])
                        for row in seed_rows
                    ]
                    pred_nominal = [
                        float(row["pred_delay_mean"]) + beta * float(row["pred_p95"]) + omega * float(row["pred_violation"])
                        for row in seed_rows
                    ]
                    margin_add = [
                        float(row["pred_risk_safe"]) - float(row["pred_risk"])
                        for row in seed_rows
                    ]
                    pred_safe = [score + margin for score, margin in zip(pred_nominal, margin_add)]
                    seed_coverages.append(mean([1.0 if pred >= emp else 0.0 for pred, emp in zip(pred_safe, emp_scores)]))
                    auc, recall = top_decile_metrics(emp_scores, pred_safe)
                    seed_aurocs.append(auc)
                    seed_recalls.append(recall)
                coverage = mean(seed_coverages)
                auc = mean(seed_aurocs)
                recall = mean(seed_recalls)
                item = {"beta": beta, "omega": omega, "coverage": coverage, "auroc": auc, "recall_at_10": recall}
                grid.append(item)
                if abs(beta - 0.35) < 1e-12 and abs(omega - 1.50) < 1e-12:
                    baseline = item
        assert baseline is not None
        output.append(
            {
                "scenario": scenario,
                "baseline_auroc": baseline["auroc"],
                "auroc_range": metric_range([item["auroc"] for item in grid]),
                "baseline_recall": baseline["recall_at_10"],
                "recall_range": metric_range([item["recall_at_10"] for item in grid]),
                "baseline_coverage": baseline["coverage"],
                "coverage_range": metric_range([item["coverage"] for item in grid]),
            }
        )
    return output


def write_statistical_latex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"Scenario & Comparison & Mean improvement & 95\% bootstrap CI & $p_{\mathrm{WSR}}$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{scenario_tex(str(row['scenario']))} & {row['comparison']} & "
            f"{float(row['mean_improvement']):.4f} & "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] & "
            f"{float(row['wilcoxon_p']):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_weight_latex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Scenario & AUROC base/range & R@10\% base/range & Coverage base/range \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{scenario_tex(str(row['scenario']))} & "
            f"{float(row['baseline_auroc']):.3f} / {row['auroc_range']} & "
            f"{float(row['baseline_recall']):.3f} / {row['recall_range']} & "
            f"{float(row['baseline_coverage']):.3f} / {row['coverage_range']} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_dirs = sorted((Path(path) for path in glob.glob(args.seed_pattern)), key=seed_from_path)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories match {args.seed_pattern!r}")

    outdir = Path(args.outdir)
    table_dir = Path(args.table_dir) if args.table_dir else outdir

    stats = statistical_audit(seed_dirs, args.bootstrap)
    weights = risk_weight_audit(seed_dirs)

    write_csv(outdir / "statistical_audit_summary.csv", stats)
    write_csv(outdir / "risk_weight_sensitivity_summary.csv", weights)
    write_statistical_latex(stats, outdir / "statistical_audit_table.tex")
    write_weight_latex(weights, outdir / "risk_weight_sensitivity_table.tex")
    write_statistical_latex(stats, table_dir / "statistical_audit_table.tex")
    write_weight_latex(weights, table_dir / "risk_weight_sensitivity_table.tex")

    print(f"Wrote {outdir / 'statistical_audit_summary.csv'}")
    print(f"Wrote {outdir / 'risk_weight_sensitivity_summary.csv'}")
    print(f"Wrote {table_dir / 'statistical_audit_table.tex'}")
    print(f"Wrote {table_dir / 'risk_weight_sensitivity_table.tex'}")


if __name__ == "__main__":
    main()
