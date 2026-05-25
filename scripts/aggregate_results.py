"""Aggregate multi-seed simulation summaries.

The script intentionally uses only the Python standard library so the
experiment pipeline stays reproducible on a clean Windows terminal.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METRICS = [
    "delay_mae",
    "risk_mae",
    "p95_mae",
    "p99_mae",
    "violation_mae",
    "safe_coverage",
    "avg_conservativeness",
    "screening_auroc",
    "precision_at_10",
    "recall_at_10",
    "calibration_ece",
]
CONTROL_METRICS = [
    "target_violation_reduction",
    "p95_delay_reduction",
    "trigger_rate",
    "background_cost",
    "false_alarm_cost",
]

DEFAULT_SCENARIOS = ["overall", "drift_strong", "load_high", "vacation_high"]
HARDCASE_SCENARIOS = ["load_high", "vacation_high", "drift_strong"]
EXTENDED_SCENARIOS = [
    "overall",
    "load_low",
    "load_high",
    "vacation_low",
    "vacation_high",
    "drift_mild",
    "drift_strong",
    "traffic_mix_video_heavy",
    "traffic_mix_c2_heavy",
]
DEFAULT_METHODS = ["proposed_laq", "static_queueing", "oracle_queueing", "blackbox_delay"]
CN_METHODS = [
    "proposed_adaptive_margin",
    "proposed_laq",
    "scenario_adaptive_direct_risk",
    "direct_risk_conformal",
    "direct_p95_conformal",
    "persistence_ewma_safe",
    "kingman_no_vacation",
    "tail_metric_ridge",
    "static_queueing",
]
HARDENING_METHODS = [
    "proposed_adaptive_margin",
    "proposed_laq",
    "scenario_adaptive_direct_risk",
    "direct_risk_conformal",
    "persistence_ewma_safe",
    "direct_p95_conformal",
    "static_queueing",
]
EXTENDED_COMPARISON_METHODS = [
    "proposed_adaptive_margin",
    "direct_risk_conformal",
    "persistence_ewma_safe",
    "kingman_no_vacation",
]
DEFAULT_ABLATION_METHODS = [
    "proposed_laq",
    "ablation_no_margin",
    "ablation_no_vacation",
    "ablation_m1_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default=r"paper1_draft/experiment_outputs/main_seed*_margin22",
        help="Glob pattern for experiment output directories.",
    )
    parser.add_argument(
        "--outdir",
        default=r"paper1_draft/experiment_outputs/multiseed_margin22",
        help="Directory for aggregate CSV and LaTeX snippets.",
    )
    return parser.parse_args()


def seed_from_path(path: Path) -> str:
    match = re.search(r"seed(\d+)", path.name)
    return match.group(1) if match else path.name


def read_rows(run_dir: Path) -> List[Dict[str, str]]:
    csv_path = run_dir / "summary_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary file: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seed = seed_from_path(run_dir)
    for row in rows:
        row["seed"] = seed
    return rows


def read_optional_csv(run_dir: Path, name: str) -> List[Dict[str, str]]:
    csv_path = run_dir / name
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seed = seed_from_path(run_dir)
    for row in rows:
        row["seed"] = seed
    return rows


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def aggregate(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["method"])].append(row)

    output = []
    for (scenario, method), items in sorted(grouped.items()):
        out: Dict[str, str] = {
            "scenario": scenario,
            "method": method,
            "runs": str(len(items)),
            "seeds": ";".join(sorted(row["seed"] for row in items)),
        }
        if "n" in items[0]:
            n_values = [float(row["n"]) for row in items]
            out["window_n_mean"] = f"{mean(n_values):.6g}"
        for metric in METRICS:
            values = [float(row[metric]) for row in items if row.get(metric) not in (None, "")]
            out[f"{metric}_mean"] = f"{mean(values):.10g}"
            out[f"{metric}_std"] = f"{sample_std(values):.10g}"
        output.append(out)
    return output


def aggregate_control(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)

    output = []
    for policy, items in sorted(grouped.items()):
        out: Dict[str, str] = {
            "policy": policy,
            "runs": str(len(items)),
            "seeds": ";".join(sorted(row["seed"] for row in items)),
        }
        for metric in CONTROL_METRICS:
            values = [float(row[metric]) for row in items if row.get(metric) not in (None, "")]
            out[f"{metric}_mean"] = f"{mean(values):.10g}"
            out[f"{metric}_std"] = f"{sample_std(values):.10g}"
        output.append(out)
    return output


def write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    if not rows:
        raise ValueError("No aggregate rows to write.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pm(row: Dict[str, str], metric: str) -> str:
    mu = float(row[f"{metric}_mean"])
    sigma = float(row[f"{metric}_std"])
    decimals = 3 if metric == "safe_coverage" else 4
    return f"{mu:.{decimals}f} $\\pm$ {sigma:.{decimals}f}"


def display_method(method: str) -> str:
    return {
        "proposed_laq": "Global margin",
        "static_queueing": "Static queueing",
        "oracle_queueing": "Oracle inputs (nominal)",
        "blackbox_delay": "Black-box",
        "proposed_adaptive_margin": "Adaptive margin (ours)",
        "direct_risk_conformal": "Direct conformal",
        "scenario_adaptive_direct_risk": "Adaptive direct risk",
        "direct_p95_conformal": "Direct $p95$ conformal",
        "persistence_ewma_safe": "EWMA persistence",
        "tail_metric_ridge": "Tail regression",
        "kingman_no_vacation": "Kingman no interruption",
        "ablation_no_margin": "No safety margin",
        "ablation_no_vacation": "No interruption modeling",
        "ablation_m1_only": "Only $m_1$",
    }.get(method, method.replace("_", " "))


def display_policy(policy: str) -> str:
    return {
        "proposed_safe_trigger": "Calibrated risk trigger",
        "direct_risk_trigger": "Direct risk trigger",
        "ewma_persistence_trigger": "EWMA trigger",
        "static_utilization_trigger": "Utilization trigger",
        "random_trigger": "Random trigger",
    }.get(policy, policy.replace("_", " "))


def display_scenario(scenario: str) -> str:
    return scenario.replace("_", r"\_")


def write_main_latex(rows: List[Dict[str, str]], out_path: Path) -> None:
    by_key = {(row["scenario"], row["method"]): row for row in rows}
    lines = [
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"Scenario & Method & Delay MAE (s) & Risk MAE (s) & Coverage \\",
        r"\hline",
    ]
    for scenario in DEFAULT_SCENARIOS:
        first = True
        for method in DEFAULT_METHODS:
            row = by_key[(scenario, method)]
            prefix = "" if first else ""
            lines.append(
                prefix
                + rf"\texttt{{{scenario.replace('_', r'\_')}}} & {display_method(method)} & "
                + f"{fmt_pm(row, 'delay_mae')} & {fmt_pm(row, 'risk_mae')} & {fmt_pm(row, 'safe_coverage')} \\\\"
            )
            first = False
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_metric(row: Dict[str, str], metric: str) -> str:
    mu = float(row[f"{metric}_mean"])
    sigma = float(row[f"{metric}_std"])
    return f"{mu:.3f} $\\pm$ {sigma:.3f}"


def write_cn_main_latex(rows: List[Dict[str, str]], out_path: Path) -> bool:
    by_key = {(row["scenario"], row["method"]): row for row in rows}
    if not all((scenario, method) in by_key for scenario in DEFAULT_SCENARIOS for method in CN_METHODS):
        return False

    lines = [
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"Scenario & Method & Delay MAE (s) & Risk MAE (s) & Coverage \\",
        r"\hline",
    ]
    for scenario in DEFAULT_SCENARIOS:
        for method in CN_METHODS:
            row = by_key[(scenario, method)]
            lines.append(
                rf"\texttt{{{scenario.replace('_', r'\_')}}} & {display_method(method)} & "
                + f"{fmt_pm(row, 'delay_mae')} & {fmt_pm(row, 'risk_mae')} & {fmt_pm(row, 'safe_coverage')} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def write_screening_latex(rows: List[Dict[str, str]], out_path: Path) -> bool:
    by_key = {(row["scenario"], row["method"]): row for row in rows}
    if not all((scenario, method) in by_key for scenario in DEFAULT_SCENARIOS for method in CN_METHODS):
        return False

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Scenario & Method & AUROC & P@10\% & R@10\% & Calib. error (s) \\",
        r"\hline",
    ]
    for scenario in DEFAULT_SCENARIOS:
        for method in CN_METHODS:
            row = by_key[(scenario, method)]
            lines.append(
                rf"\texttt{{{scenario.replace('_', r'\_')}}} & {display_method(method)} & "
                + f"{fmt_metric(row, 'screening_auroc')} & "
                + f"{fmt_metric(row, 'precision_at_10')} & "
                + f"{fmt_metric(row, 'recall_at_10')} & "
                + f"{fmt_metric(row, 'calibration_ece')} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def write_extended_scenario_latex(rows: List[Dict[str, str]], out_path: Path) -> bool:
    by_key = {(row["scenario"], row["method"]): row for row in rows}
    required = [
        (scenario, method)
        for scenario in EXTENDED_SCENARIOS
        for method in EXTENDED_COMPARISON_METHODS
    ]
    if not all(key in by_key for key in required):
        return False

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Scenario & Method & AUROC & R@10\% & Coverage & Risk MAE (s) \\",
        r"\hline",
    ]
    for scenario in EXTENDED_SCENARIOS:
        for method in EXTENDED_COMPARISON_METHODS:
            row = by_key[(scenario, method)]
            lines.append(
                rf"\texttt{{{display_scenario(scenario)}}} & {display_method(method)} & "
                + f"{fmt_metric(row, 'screening_auroc')} & "
                + f"{fmt_metric(row, 'recall_at_10')} & "
                + f"{fmt_pm(row, 'safe_coverage')} & "
                + f"{fmt_pm(row, 'risk_mae')} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def write_hardcase_latex(rows: List[Dict[str, str]], out_path: Path) -> bool:
    by_key = {(row["scenario"], row["method"]): row for row in rows}
    required = [
        (scenario, method)
        for scenario in HARDCASE_SCENARIOS
        for method in HARDENING_METHODS
    ]
    if not all(key in by_key for key in required):
        return False

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Scenario & Method & Coverage & Risk difference (s) & Risk MAE (s) & R@10\% \\",
        r"\hline",
    ]
    for scenario in HARDCASE_SCENARIOS:
        for method in HARDENING_METHODS:
            row = by_key[(scenario, method)]
            lines.append(
                rf"\texttt{{{display_scenario(scenario)}}} & {display_method(method)} & "
                + f"{fmt_pm(row, 'safe_coverage')} & "
                + f"{fmt_pm(row, 'avg_conservativeness')} & "
                + f"{fmt_pm(row, 'risk_mae')} & "
                + f"{fmt_metric(row, 'recall_at_10')} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def write_control_latex(rows: List[Dict[str, str]], out_path: Path) -> bool:
    if not rows:
        return False
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Policy & Viol. red. & $p95$ red. & Trigger & Bg. cost & False alarms \\",
        r"\hline",
    ]
    order = [
        "proposed_safe_trigger",
        "direct_risk_trigger",
        "ewma_persistence_trigger",
        "static_utilization_trigger",
        "random_trigger",
    ]
    by_policy = {row["policy"]: row for row in rows}
    for policy in order:
        if policy not in by_policy:
            continue
        row = by_policy[policy]
        lines.append(
            f"{display_policy(policy)} & "
            + f"{fmt_metric(row, 'target_violation_reduction')} & "
            + f"{fmt_metric(row, 'p95_delay_reduction')} & "
            + f"{fmt_metric(row, 'trigger_rate')} & "
            + f"{fmt_metric(row, 'background_cost')} & "
            + f"{fmt_metric(row, 'false_alarm_cost')} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def write_ablation_latex(rows: List[Dict[str, str]], out_path: Path) -> bool:
    by_key = {(row["scenario"], row["method"]): row for row in rows}
    if not all((scenario, method) in by_key for scenario in DEFAULT_SCENARIOS for method in DEFAULT_ABLATION_METHODS):
        return False

    lines = [
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"Scenario & Variant & Delay MAE (s) & Risk MAE (s) & Coverage \\",
        r"\hline",
    ]
    for scenario in DEFAULT_SCENARIOS:
        for method in DEFAULT_ABLATION_METHODS:
            row = by_key[(scenario, method)]
            lines.append(
                rf"\texttt{{{scenario.replace('_', r'\_')}}} & {display_method(method)} & "
                + f"{fmt_pm(row, 'delay_mae')} & {fmt_pm(row, 'risk_mae')} & {fmt_pm(row, 'safe_coverage')} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def print_key_summary(rows: List[Dict[str, str]]) -> None:
    by_key = {(row["scenario"], row["method"]): row for row in rows}
    for scenario in DEFAULT_SCENARIOS:
        print(f"[{scenario}]")
        available_ablation = all((scenario, method) in by_key for method in DEFAULT_ABLATION_METHODS)
        methods = DEFAULT_ABLATION_METHODS if available_ablation else DEFAULT_METHODS
        for method in methods:
            row = by_key[(scenario, method)]
            print(
                f"  {method:16s} "
                f"delay={fmt_pm(row, 'delay_mae')} "
                f"risk={fmt_pm(row, 'risk_mae')} "
                f"coverage={fmt_pm(row, 'safe_coverage')}"
            )


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path) for path in sorted(glob.glob(args.pattern))]
    if not run_dirs:
        raise SystemExit(f"No run directories matched pattern: {args.pattern}")

    all_rows: List[Dict[str, str]] = []
    all_control_rows: List[Dict[str, str]] = []
    for run_dir in run_dirs:
        all_rows.extend(read_rows(run_dir))
        all_control_rows.extend(read_optional_csv(run_dir, "control_policy_metrics.csv"))

    outdir = Path(args.outdir)
    aggregate_rows = aggregate(all_rows)
    write_csv(aggregate_rows, outdir / "aggregate_summary.csv")
    aggregate_control_rows = aggregate_control(all_control_rows) if all_control_rows else []
    if aggregate_control_rows:
        write_csv(aggregate_control_rows, outdir / "control_policy_summary.csv")
    write_main_latex(aggregate_rows, outdir / "main_results_table.tex")
    wrote_cn_main = write_cn_main_latex(aggregate_rows, outdir / "cn_main_results_table.tex")
    wrote_screening = write_screening_latex(aggregate_rows, outdir / "screening_metrics_table.tex")
    wrote_extended = write_extended_scenario_latex(aggregate_rows, outdir / "extended_scenario_table.tex")
    wrote_hardcase = write_hardcase_latex(aggregate_rows, outdir / "hardcase_coverage_table.tex")
    wrote_control = write_control_latex(aggregate_control_rows, outdir / "control_policy_table.tex")
    wrote_ablation = write_ablation_latex(aggregate_rows, outdir / "ablation_results_table.tex")

    print(f"Aggregated {len(run_dirs)} runs: {', '.join(path.name for path in run_dirs)}")
    print(f"Wrote: {outdir / 'aggregate_summary.csv'}")
    if aggregate_control_rows:
        print(f"Wrote: {outdir / 'control_policy_summary.csv'}")
    print(f"Wrote: {outdir / 'main_results_table.tex'}")
    if wrote_cn_main:
        print(f"Wrote: {outdir / 'cn_main_results_table.tex'}")
    if wrote_screening:
        print(f"Wrote: {outdir / 'screening_metrics_table.tex'}")
    if wrote_extended:
        print(f"Wrote: {outdir / 'extended_scenario_table.tex'}")
    if wrote_hardcase:
        print(f"Wrote: {outdir / 'hardcase_coverage_table.tex'}")
    if wrote_control:
        print(f"Wrote: {outdir / 'control_policy_table.tex'}")
    if wrote_ablation:
        print(f"Wrote: {outdir / 'ablation_results_table.tex'}")
    print_key_summary(aggregate_rows)


if __name__ == "__main__":
    main()
