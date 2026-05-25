#!/usr/bin/env python
"""Seed-fold NS-3 held-out validation for the Computer Networks revision.

This script reuses the lag-feature evaluation in ``evaluate_ns3_heldout.py`` but
rotates held-out seed pairs. It summarizes fold-to-fold variability so the NS-3
packet-trace validation is not tied to a single seed split.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import evaluate_ns3_heldout as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        default=r"paper1_draft/experiment_outputs/ns3_validation_extended/ns3_window_metrics.csv",
        help="NS-3 window metrics CSV.",
    )
    parser.add_argument(
        "--outdir",
        default=r"paper1_draft/experiment_outputs/ns3_validation_extended",
        help="Output directory.",
    )
    parser.add_argument(
        "--table-dir",
        default=r"paper1_draft/computer_networks_revision/tables/ns3",
        help="Manuscript table directory.",
    )
    parser.add_argument("--fold-size", type=int, default=2)
    return parser.parse_args()


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def folds_from_seeds(seeds: Sequence[str], fold_size: int) -> List[List[str]]:
    ordered = sorted(seeds, key=lambda value: int(value))
    return [ordered[idx : idx + fold_size] for idx in range(0, len(ordered), fold_size)]


def aggregate(summary_rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = list(summary_rows)
    output = []
    for method in base.METHODS:
        items = [row for row in rows if row["method"] == method]
        out: Dict[str, object] = {
            "method": method,
            "folds": len(items),
            "heldout_seeds": ";".join(str(row["test_seeds"]) for row in items),
        }
        for metric in ["risk_mae", "p95_mae", "violation_mae", "safe_coverage", "screening_auroc", "recall_at_10"]:
            values = [float(row[metric]) for row in items]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = sample_std(values)
        output.append(out)
    return output


def fmt_pm(row: Dict[str, object], metric: str, decimals: int = 3, scale: float = 1.0) -> str:
    mu = scale * float(row[f"{metric}_mean"])
    sd = scale * float(row[f"{metric}_std"])
    return f"{mu:.{decimals}f} $\\pm$ {sd:.{decimals}f}"


def write_latex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    by_method = {str(row["method"]): row for row in rows}
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & Risk MAE (s) & $p95$ MAE (ms) & Coverage & R@10\% \\",
        r"\hline",
    ]
    for method in base.METHODS:
        row = by_method[method]
        lines.append(
            f"{base.display_method(method)} & "
            f"{fmt_pm(row, 'risk_mae', decimals=4)} & "
            f"{fmt_pm(row, 'p95_mae', decimals=3, scale=1000.0)} & "
            f"{fmt_pm(row, 'safe_coverage', decimals=3)} & "
            f"{fmt_pm(row, 'recall_at_10', decimals=3)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    samples = base.build_samples(base.read_csv(Path(args.window)))
    seeds = sorted({str(sample["seed"]) for sample in samples}, key=lambda value: int(value))
    folds = folds_from_seeds(seeds, args.fold_size)
    prediction_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for fold_idx, test_seeds in enumerate(folds, start=1):
        train_seeds = [seed for seed in seeds if seed not in test_seeds]
        fold_predictions, fold_summary = base.evaluate(samples, train_seeds, test_seeds)
        for row in fold_predictions:
            row["fold"] = fold_idx
            row["train_seeds"] = ";".join(train_seeds)
            row["test_seeds"] = ";".join(test_seeds)
            prediction_rows.append(row)
        for row in fold_summary:
            row["fold"] = fold_idx
            row["train_seeds"] = ";".join(train_seeds)
            row["test_seeds"] = ";".join(test_seeds)
            summary_rows.append(row)

    aggregate_rows = aggregate(summary_rows)
    outdir = Path(args.outdir)
    table_dir = Path(args.table_dir)
    write_csv(prediction_rows, outdir / "ns3_seedfold_window_predictions.csv")
    write_csv(summary_rows, outdir / "ns3_seedfold_metrics_by_fold.csv")
    write_csv(aggregate_rows, outdir / "ns3_seedfold_metrics_summary.csv")
    write_latex(aggregate_rows, outdir / "ns3_seedfold_screening_table.tex")
    write_latex(aggregate_rows, table_dir / "ns3_seedfold_screening_table.tex")
    print(f"Wrote {outdir / 'ns3_seedfold_metrics_summary.csv'}")
    print(f"Wrote {table_dir / 'ns3_seedfold_screening_table.tex'}")


if __name__ == "__main__":
    main()
