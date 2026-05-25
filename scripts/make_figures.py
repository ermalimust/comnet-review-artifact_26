"""Generate publication-style figures from existing experiment CSV files.

The script intentionally avoids matplotlib so the project stays dependency-free.
It writes standalone PGFPlots/TikZ sources that can be compiled into vector PDFs
by XeLaTeX or Tectonic.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SCENARIOS = ["overall", "drift_strong", "load_high", "vacation_high"]
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
RISK_VALIDITY_SCENARIOS = [
    "overall",
    "load_high",
    "vacation_high",
    "drift_strong",
    "traffic_mix_video_heavy",
    "traffic_mix_c2_heavy",
]
NS3_BRIDGE_SCENARIOS = [
    "overall",
    "load_high",
    "vacation_high",
    "drift_strong",
    "traffic_mix_video_heavy",
    "traffic_mix_c2_heavy",
]
SCENARIO_LABELS = {
    "overall": r"Overall",
    "drift_strong": r"Strong\\drift",
    "load_high": r"High\\load",
    "vacation_high": r"High\\vacation",
    "load_low": r"Low\\load",
    "vacation_low": r"Low\\vacation",
    "drift_mild": r"Mild\\drift",
    "traffic_mix_video_heavy": r"Video\\heavy",
    "traffic_mix_c2_heavy": r"C2\\heavy",
}

MAIN_METHODS = ["proposed_adaptive_margin", "proposed_laq", "static_queueing", "blackbox_delay"]
CN_METHODS = [
    "proposed_adaptive_margin",
    "proposed_laq",
    "scenario_adaptive_direct_risk",
    "direct_risk_conformal",
    "direct_p95_conformal",
    "persistence_ewma_safe",
    "kingman_no_vacation",
    "tail_metric_ridge",
]
ABLATION_METHODS = ["proposed_laq", "ablation_no_margin", "ablation_no_vacation", "ablation_m1_only"]
HARDCASE_SCENARIOS = ["load_high", "vacation_high", "drift_strong"]
ADAPTIVE_METHODS = [
    "proposed_adaptive_margin",
    "proposed_laq",
    "scenario_adaptive_direct_risk",
    "persistence_ewma_safe",
]
CONTROL_POLICIES = [
    "proposed_safe_trigger",
    "direct_risk_trigger",
    "static_utilization_trigger",
    "ewma_persistence_trigger",
    "random_trigger",
]
NS3_HELDOUT_METHODS = [
    "ns3_lagged_safe",
    "ns3_direct_nomargin",
    "ns3_ewma_safe",
    "ns3_background_rate",
]
METHOD_LABELS = {
    "proposed_adaptive_margin": "Adaptive margin",
    "proposed_laq": "Global margin",
    "scenario_adaptive_direct_risk": "Adaptive direct",
    "static_queueing": "Static",
    "oracle_queueing": "Oracle inputs",
    "blackbox_delay": "Black-box",
    "direct_risk_conformal": "Direct conformal",
    "direct_p95_conformal": "Direct $p95$",
    "persistence_ewma_safe": "EWMA",
    "tail_metric_ridge": "Tail regression",
    "kingman_no_vacation": "Kingman",
    "ablation_no_margin": "No margin",
    "ablation_no_vacation": "No interruption",
    "ablation_m1_only": "Only $m_1$",
}
METHOD_COLORS = {
    "proposed_adaptive_margin": "tealgreen",
    "proposed_laq": "proposedblue",
    "scenario_adaptive_direct_risk": "deepviolet",
    "static_queueing": "staticgray",
    "oracle_queueing": "oraclegreen",
    "blackbox_delay": "blackboxorange",
    "direct_risk_conformal": "nomarginred",
    "direct_p95_conformal": "vacationgold",
    "persistence_ewma_safe": "monepurple",
    "tail_metric_ridge": "vacationgold",
    "kingman_no_vacation": "monepurple",
    "ablation_no_margin": "nomarginred",
    "ablation_no_vacation": "vacationgold",
    "ablation_m1_only": "monepurple",
}
SCENARIO_COLORS = {
    "overall": "proposedblue",
    "drift_strong": "nomarginred",
    "load_high": "oraclegreen",
    "vacation_high": "blackboxorange",
    "load_low": "staticgray",
    "vacation_low": "vacationgold",
    "drift_mild": "monepurple",
    "traffic_mix_video_heavy": "tealgreen",
    "traffic_mix_c2_heavy": "deepviolet",
}
CONTROL_POLICY_LABELS = {
    "proposed_safe_trigger": "Calibrated risk",
    "direct_risk_trigger": "Direct risk",
    "static_utilization_trigger": "Utilization",
    "ewma_persistence_trigger": "EWMA",
    "random_trigger": "Random",
}
CONTROL_POLICY_COLORS = {
    "proposed_safe_trigger": "tealgreen",
    "direct_risk_trigger": "nomarginred",
    "static_utilization_trigger": "oraclegreen",
    "ewma_persistence_trigger": "monepurple",
    "random_trigger": "staticgray",
}
NS3_HELDOUT_LABELS = {
    "ns3_lagged_safe": "Lagged cal.",
    "ns3_direct_nomargin": "Lagged no margin",
    "ns3_ewma_safe": "EWMA cal.",
    "ns3_background_rate": "Background",
}
NS3_HELDOUT_COLORS = {
    "ns3_lagged_safe": "tealgreen",
    "ns3_direct_nomargin": "nomarginred",
    "ns3_ewma_safe": "monepurple",
    "ns3_background_rate": "staticgray",
}


SummaryTable = Dict[str, Dict[str, Dict[str, float]]]
Row = Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-summary",
        default=r"paper1_draft/experiment_outputs/multiseed_margin22/aggregate_summary.csv",
        help="Aggregate summary CSV for main baseline comparison.",
    )
    parser.add_argument(
        "--ablation-summary",
        default=r"paper1_draft/experiment_outputs/ablation_multiseed_margin22/aggregate_summary.csv",
        help="Aggregate summary CSV for ablation comparison.",
    )
    parser.add_argument(
        "--main-window-glob",
        default=r"paper1_draft/experiment_outputs/main_seed*_margin22/window_predictions.csv",
        help="Glob pattern for main per-window prediction CSV files.",
    )
    parser.add_argument(
        "--ablation-window-glob",
        default=r"paper1_draft/experiment_outputs/ablation_seed*_margin22/window_predictions.csv",
        help="Glob pattern for ablation per-window prediction CSV files.",
    )
    parser.add_argument(
        "--ns3-window",
        default=r"paper1_draft/experiment_outputs/ns3_validation/ns3_window_metrics.csv",
        help="NS-3 window-metric CSV.",
    )
    parser.add_argument(
        "--ns3-summary",
        default=r"paper1_draft/experiment_outputs/ns3_validation/ns3_summary.csv",
        help="NS-3 aggregate summary CSV.",
    )
    parser.add_argument(
        "--ns3-packet-glob",
        default=r"paper1_draft/experiment_outputs/ns3_validation/packets/*_packets.csv",
        help="Glob pattern for NS-3 packet CSV files.",
    )
    parser.add_argument(
        "--control-summary",
        default=r"paper1_draft/experiment_outputs/cn_hardening_multiseed_margin22/control_policy_summary.csv",
        help="Aggregate control-policy CSV for resource-control figures.",
    )
    parser.add_argument(
        "--ns3-heldout-summary",
        default=r"paper1_draft/experiment_outputs/ns3_validation_extended/ns3_shadow_prediction_metrics.csv",
        help="NS-3 held-out predictor summary CSV.",
    )
    parser.add_argument(
        "--outdir",
        default=r"paper1_draft/figures",
        help="Output directory for generated PGFPlots sources.",
    )
    parser.add_argument(
        "--figure-prefix",
        default="",
        help="Optional prefix for generated figure filenames.",
    )
    return parser.parse_args()


def read_summary(path: Path) -> SummaryTable:
    table: SummaryTable = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scenario = row["scenario"]
            method = row["method"]
            table.setdefault(scenario, {})[method] = {
                "delay_mae": float(row["delay_mae_mean"]),
                "delay_mae_std": float(row["delay_mae_std"]),
                "risk_mae": float(row["risk_mae_mean"]),
                "risk_mae_std": float(row["risk_mae_std"]),
                "p95_mae": float(row["p95_mae_mean"]),
                "p95_mae_std": float(row["p95_mae_std"]),
                "coverage": float(row["safe_coverage_mean"]),
                "coverage_std": float(row["safe_coverage_std"]),
                "conservativeness": float(row["avg_conservativeness_mean"]),
                "conservativeness_std": float(row["avg_conservativeness_std"]),
                "screening_auroc": float(row.get("screening_auroc_mean", 0.0)),
                "screening_auroc_std": float(row.get("screening_auroc_std", 0.0)),
                "precision_at_10": float(row.get("precision_at_10_mean", 0.0)),
                "precision_at_10_std": float(row.get("precision_at_10_std", 0.0)),
                "recall_at_10": float(row.get("recall_at_10_mean", 0.0)),
                "recall_at_10_std": float(row.get("recall_at_10_std", 0.0)),
                "calibration_ece": float(row.get("calibration_ece_mean", 0.0)),
                "calibration_ece_std": float(row.get("calibration_ece_std", 0.0)),
            }
    return table


def read_rows(pattern: str) -> List[Row]:
    rows: List[Row] = []
    for path in sorted(glob.glob(pattern)):
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def read_csv(path: Path) -> List[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_optional_csv(path: Path) -> List[Row]:
    if not path.exists():
        return []
    return read_csv(path)


def safe_float(row: Row, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def ratio_std(value: float, value_std: float, base: float, base_std: float, same_series: bool = False) -> float:
    if same_series:
        return 0.0
    value = max(value, 1e-12)
    base = max(base, 1e-12)
    ratio = value / base
    return ratio * math.sqrt((value_std / value) ** 2 + (base_std / base) ** 2)


def metric_value(
    table: SummaryTable,
    scenario: str,
    method: str,
    metric: str,
    normalized_to: str | None = None,
) -> float:
    value = table[scenario][method][metric]
    if normalized_to is not None:
        base = table[scenario][normalized_to][metric]
        return value / max(base, 1e-12)
    return value


def max_metric(
    table: SummaryTable,
    scenarios: Sequence[str],
    methods: Sequence[str],
    metric: str,
    normalized_to: str | None = None,
    floor: float = 0.0,
) -> float:
    values = [
        metric_value(table, scenario, method, metric, normalized_to)
        for scenario in scenarios
        for method in methods
        if scenario in table and method in table[scenario]
    ]
    return max([floor] + values)


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


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return sum((value - mu) ** 2 for value in values) / len(values)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mu = mean(xs)
    y_mu = mean(ys)
    numerator = sum((x - x_mu) * (y - y_mu) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - x_mu) ** 2 for x in xs) * sum((y - y_mu) ** 2 for y in ys))
    return numerator / denom if denom > 0 else 0.0


def ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        rank = 0.5 * (idx + end - 1) + 1.0
        for original_idx, _ in indexed[idx:end]:
            output[original_idx] = rank
        idx = end
    return output


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return pearson(ranks(xs), ranks(ys))


def top_decile_lift(scores: Sequence[float], values: Sequence[float]) -> float:
    if len(scores) != len(values) or not scores:
        return 0.0
    pairs = sorted(zip(scores, values), key=lambda item: item[0], reverse=True)
    top_n = max(1, math.ceil(0.10 * len(pairs)))
    top = [value for _, value in pairs[:top_n]]
    rest = [value for _, value in pairs[top_n:]]
    return mean(top) / max(mean(rest), 1e-12)


def top_decile_gap(scores: Sequence[float], values: Sequence[float]) -> float:
    if len(scores) != len(values) or not scores:
        return 0.0
    pairs = sorted(zip(scores, values), key=lambda item: item[0], reverse=True)
    top_n = max(1, math.ceil(0.10 * len(pairs)))
    top = [value for _, value in pairs[:top_n]]
    rest = [value for _, value in pairs[top_n:]]
    return mean(top) - mean(rest)


def tex_header() -> str:
    return r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{proposedblue}{RGB}{46,94,170}
\definecolor{staticgray}{RGB}{128,136,148}
\definecolor{oraclegreen}{RGB}{69,145,98}
\definecolor{blackboxorange}{RGB}{218,132,55}
\definecolor{nomarginred}{RGB}{186,78,68}
\definecolor{vacationgold}{RGB}{204,151,55}
\definecolor{monepurple}{RGB}{126,93,159}
\definecolor{tealgreen}{RGB}{35,143,135}
\definecolor{deepviolet}{RGB}{98,80,154}
\begin{document}
"""


def tex_footer() -> str:
    return r"\end{document}" + "\n"


def coords_for_bars(
    table: SummaryTable,
    methods: Sequence[str],
    metric: str,
    normalized_to: str | None = None,
    add_legend: bool = True,
) -> str:
    plots: List[str] = []
    for method in methods:
        coords: List[str] = []
        for scenario in SCENARIOS:
            value = metric_value(table, scenario, method, metric, normalized_to)
            coords.append(f"({scenario},{value:.4f})")
        legend_entry = rf"\addlegendentry{{{METHOD_LABELS[method]}}}" if add_legend else ""
        plots.append(
            rf"""\addplot+[
    ybar,
    bar width=3.9pt,
    fill={METHOD_COLORS[method]},
    draw=black!45,
    line width=0.2pt
] coordinates {{
{chr(10).join(coords)}
}};
{legend_entry}
"""
        )
    return "\n".join(plots)


def bar_axis_options(ylabel: str, ymax: float, legend: bool = False, ymin: float = 0.0) -> str:
    options = [
        "ybar",
        f"ymin={ymin:.3f}",
        f"ymax={ymax:.3f}",
        f"ylabel={{{ylabel}}}",
    ]
    if legend:
        options.append(
            r"legend style={at={(0.5,1.30)}, anchor=south, legend columns=4, "
            r"draw=none, fill=none, font=\scriptsize, /tikz/every even column/.append style={column sep=0.11cm}}"
        )
    labels = ",".join(f"{{{SCENARIO_LABELS[scenario]}}}" for scenario in SCENARIOS)
    symbolic = ",".join(SCENARIOS)
    options.extend(
        [
            f"symbolic x coords={{{symbolic}}}",
            "xtick=data",
            f"xticklabels={{{labels}}}",
            r"x tick label style={font=\scriptsize, align=center, text depth=0.25ex}",
            "enlarge x limits=0.14",
        ]
    )
    return ",\n    ".join(options) + ","


def scenario_axis_options(
    scenarios: Sequence[str],
    ylabel: str,
    ymax: float,
    legend: bool = False,
    ymin: float = 0.0,
) -> str:
    options = [
        "ybar",
        f"ymin={ymin:.3f}",
        f"ymax={ymax:.3f}",
        f"ylabel={{{ylabel}}}",
    ]
    if legend:
        options.append(
            r"legend style={at={(0.5,1.30)}, anchor=south, legend columns=4, "
            r"draw=none, fill=none, font=\scriptsize, /tikz/every even column/.append style={column sep=0.10cm}}"
        )
    labels = ",".join(f"{{{SCENARIO_LABELS[scenario]}}}" for scenario in scenarios)
    symbolic = ",".join(scenarios)
    options.extend(
        [
            f"symbolic x coords={{{symbolic}}}",
            "xtick=data",
            f"xticklabels={{{labels}}}",
            r"x tick label style={font=\scriptsize, align=center, text depth=0.25ex}",
            "enlarge x limits=0.14",
        ]
    )
    return ",\n    ".join(options) + ","


def coords_for_scenario_bars(
    table: SummaryTable,
    scenarios: Sequence[str],
    methods: Sequence[str],
    metric: str,
    add_legend: bool = True,
) -> str:
    plots: List[str] = []
    for method in methods:
        coords = [
            f"({scenario},{table[scenario][method][metric]:.4f})"
            for scenario in scenarios
            if scenario in table and method in table[scenario]
        ]
        if not coords:
            continue
        legend_entry = rf"\addlegendentry{{{METHOD_LABELS[method]}}}" if add_legend else ""
        plots.append(
            rf"""\addplot+[
    ybar,
    bar width=5.0pt,
    fill={METHOD_COLORS[method]},
    draw=black!45,
    line width=0.2pt
] coordinates {{
{chr(10).join(coords)}
}};
{legend_entry}
"""
        )
    return "\n".join(plots)


def reference_line_for_scenarios(scenarios: Sequence[str], y: float) -> str:
    if not scenarios:
        return ""
    return (
        rf"\addplot+[black!55, densely dashed, line width=0.6pt, mark=none, forget plot] "
        rf"coordinates {{({scenarios[0]},{y:.4f}) ({scenarios[-1]},{y:.4f})}};"
    )


def reference_line(y: float) -> str:
    return rf"\addplot+[black!55, densely dashed, line width=0.6pt, mark=none, forget plot] coordinates {{(overall,{y:.4f}) (vacation_high,{y:.4f})}};"


def write_grouped_bar_chart(
    out_path: Path,
    table: SummaryTable,
    methods: Sequence[str],
    metric: str,
    ylabel: str,
    ymax: float,
    normalized_to: str | None = None,
    reference: float | None = None,
) -> None:
    ref = reference_line(reference) if reference is not None else ""
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=7.7in,
    height=4.25in,
    {bar_axis_options(ylabel, ymax, legend=True)}
    label style={{font=\Large}},
    tick label style={{font=\large}},
    ymajorgrids=true,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.55pt}},
    tick style={{black!65, line width=0.55pt}},
]
{coords_for_bars(table, methods, metric, normalized_to)}
{ref}
\end{{axis}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def write_main_performance_landscape(out_path: Path, table: SummaryTable) -> None:
    delay_ymax = 1.12 * max_metric(table, SCENARIOS, MAIN_METHODS, "delay_mae", "static_queueing", 1.0)
    risk_ymax = 1.12 * max_metric(table, SCENARIOS, MAIN_METHODS, "risk_mae", "static_queueing", 1.0)
    p95_ymax = 1.12 * max_metric(table, SCENARIOS, MAIN_METHODS, "p95_mae", "static_queueing", 1.0)
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 2, horizontal sep=1.10cm, vertical sep=1.75cm}},
    width=3.55in,
    height=2.45in,
    ymajorgrids=true,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.45pt}},
    tick style={{black!65, line width=0.45pt}},
    label style={{font=\scriptsize}},
    tick label style={{font=\scriptsize}},
    title style={{font=\small, yshift=-0.5ex}},
]
\nextgroupplot[
    title={{(a) Delay error}},
    {bar_axis_options("MAE / static", delay_ymax, legend=True)}
]
{coords_for_bars(table, MAIN_METHODS, "delay_mae", "static_queueing")}

\nextgroupplot[
    title={{(b) Risk error}},
    {bar_axis_options("MAE / static", risk_ymax)}
]
{coords_for_bars(table, MAIN_METHODS, "risk_mae", "static_queueing", add_legend=False)}

\nextgroupplot[
    title={{(c) Coverage}},
    {bar_axis_options("Coverage", 1.08)}
]
{coords_for_bars(table, MAIN_METHODS, "coverage", add_legend=False)}
{reference_line(0.90)}

\nextgroupplot[
    title={{(d) Tail-delay error}},
    {bar_axis_options("$p95$ MAE / static", p95_ymax)}
]
{coords_for_bars(table, MAIN_METHODS, "p95_mae", "static_queueing", add_legend=False)}
\end{{groupplot}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def binned_calibration(rows: Sequence[Row], scenario: str, risk_col: str, bins: int = 10) -> str:
    selected = [
        (safe_float(row, risk_col), safe_float(row, "emp_risk"))
        for row in rows
        if row["scenario"] == scenario and row["method"] == "proposed_laq"
    ]
    selected.sort(key=lambda item: item[0])
    if not selected:
        return ""
    coords: List[str] = []
    n = len(selected)
    for idx in range(bins):
        chunk = selected[idx * n // bins : (idx + 1) * n // bins]
        coords.append(f"({mean([x for x, _ in chunk]):.4f},{mean([y for _, y in chunk]):.4f})")
    return "\n".join(coords)


def ecdf_coords(values: Sequence[float], points: int = 55) -> str:
    if not values:
        return ""
    ordered = sorted(values)
    coords: List[str] = []
    for idx in range(points):
        p = idx / (points - 1)
        coords.append(f"({percentile(ordered, p):.4f},{p:.4f})")
    return "\n".join(coords)


def conservativeness_values(
    rows: Sequence[Row],
    scenario: str,
    method: str,
    lower: float | None = None,
    upper: float | None = None,
) -> List[float]:
    values: List[float] = []
    for row in rows:
        if row["scenario"] != scenario or row["method"] != method:
            continue
        value = safe_float(row, "conservativeness")
        if not math.isfinite(value):
            continue
        if lower is not None:
            value = max(lower, value)
        if upper is not None:
            value = min(upper, value)
        values.append(value)
    return values


def write_risk_calibration(out_path: Path, rows: Sequence[Row]) -> None:
    overall_xmax = 0.90
    drift_xmax = 6.20
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 2, horizontal sep=1.15cm, vertical sep=1.75cm}},
    width=3.55in,
    height=2.45in,
    ymajorgrids=true,
    xmajorgrids=true,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.45pt}},
    tick style={{black!65, line width=0.45pt}},
    label style={{font=\scriptsize}},
    tick label style={{font=\scriptsize}},
    title style={{font=\small, yshift=-0.5ex}},
]
\nextgroupplot[
    title={{(a) Overall calibration}},
    xlabel={{Predicted risk}},
    ylabel={{Empirical risk}},
    xmin=0, xmax={overall_xmax:.2f},
    ymin=0, ymax={overall_xmax:.2f},
    legend style={{at={{(0.5,1.27)}}, anchor=south, legend columns=3, draw=none, fill=none, font=\scriptsize}},
]
\addplot+[black!55, densely dashed, line width=0.6pt, mark=none] coordinates {{(0,0) ({overall_xmax:.2f},{overall_xmax:.2f})}};
\addlegendentry{{Ideal}}
\addplot+[proposedblue, mark=*, mark size=1.3pt, line width=0.65pt] coordinates {{
{binned_calibration(rows, "overall", "pred_risk")}
}};
\addlegendentry{{Nominal}}
\addplot+[nomarginred, mark=square*, mark size=1.3pt, line width=0.65pt] coordinates {{
{binned_calibration(rows, "overall", "pred_risk_safe")}
}};
\addlegendentry{{Safe}}

\nextgroupplot[
    title={{(b) Strong-drift calibration}},
    xlabel={{Predicted risk}},
    ylabel={{Empirical risk}},
    xmin=0, xmax={drift_xmax:.2f},
    ymin=0, ymax={drift_xmax:.2f},
]
\addplot+[black!55, densely dashed, line width=0.6pt, mark=none] coordinates {{(0,0) ({drift_xmax:.2f},{drift_xmax:.2f})}};
\addplot+[proposedblue, mark=*, mark size=1.3pt, line width=0.65pt] coordinates {{
{binned_calibration(rows, "drift_strong", "pred_risk")}
}};
\addplot+[nomarginred, mark=square*, mark size=1.3pt, line width=0.65pt] coordinates {{
{binned_calibration(rows, "drift_strong", "pred_risk_safe")}
}};

\nextgroupplot[
    title={{(c) Overall risk difference}},
    xlabel={{Predicted risk $-$ empirical risk}},
    ylabel={{CDF}},
    xmin=-1.0, xmax=1.5,
    ymin=0, ymax=1.02,
    legend style={{at={{(0.03,0.97)}}, anchor=north west, legend columns=1, draw=none, fill=none, font=\scriptsize}},
]
\addplot+[proposedblue, line width=0.75pt, mark=none] coordinates {{
{ecdf_coords(conservativeness_values(rows, "overall", "proposed_laq", -1.0, 1.5))}
}};
\addlegendentry{{Proposed safe}}
\addplot+[blackboxorange, line width=0.75pt, mark=none, densely dashed] coordinates {{
{ecdf_coords(conservativeness_values(rows, "overall", "blackbox_delay", -1.0, 1.5))}
}};
\addlegendentry{{Black-box}}
\addplot+[black!50, dotted, line width=0.6pt, mark=none, forget plot] coordinates {{(0,0) (0,1.02)}};

\nextgroupplot[
    title={{(d) Strong-drift risk difference}},
    xlabel={{Predicted risk $-$ empirical risk}},
    ylabel={{CDF}},
    xmin=-8.0, xmax=5.0,
    ymin=0, ymax=1.02,
]
\addplot+[proposedblue, line width=0.75pt, mark=none] coordinates {{
{ecdf_coords(conservativeness_values(rows, "drift_strong", "proposed_laq", -8.0, 5.0))}
}};
\addplot+[blackboxorange, line width=0.75pt, mark=none, densely dashed] coordinates {{
{ecdf_coords(conservativeness_values(rows, "drift_strong", "blackbox_delay", -8.0, 5.0))}
}};
\addplot+[black!50, dotted, line width=0.6pt, mark=none, forget plot] coordinates {{(0,0) (0,1.02)}};
\end{{groupplot}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def risk_proxy_rows(rows: Sequence[Row], scenario: str) -> List[Row]:
    return [row for row in rows if row["scenario"] == scenario and row["method"] == "proposed_laq"]


def risk_proxy_stat_map(rows: Sequence[Row]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for scenario in RISK_VALIDITY_SCENARIOS:
        selected = risk_proxy_rows(rows, scenario)
        emp_risk = [safe_float(row, "emp_risk") for row in selected]
        emp_p95 = [safe_float(row, "emp_p95") for row in selected]
        emp_p99 = [safe_float(row, "emp_p99") for row in selected]
        emp_violation = [safe_float(row, "emp_violation") for row in selected]
        stats[scenario] = {
            "rho_p95": spearman(emp_risk, emp_p95),
            "rho_p99": spearman(emp_risk, emp_p99),
            "rho_violation": spearman(emp_risk, emp_violation),
            "p95_lift": top_decile_lift(emp_risk, emp_p95),
            "violation_gap": top_decile_gap(emp_risk, emp_violation),
        }
    return stats


def coords_from_stats(stats: Dict[str, Dict[str, float]], metric: str) -> str:
    coords = []
    for scenario in RISK_VALIDITY_SCENARIOS:
        coords.append(f"({scenario},{stats[scenario][metric]:.4f})")
    return "\n".join(coords)


def risk_validity_axis_options(ylabel: str, ymax: float, ymin: float = 0.0) -> str:
    labels = ",".join(f"{{{SCENARIO_LABELS[scenario]}}}" for scenario in RISK_VALIDITY_SCENARIOS)
    symbolic = ",".join(RISK_VALIDITY_SCENARIOS)
    return rf"""ybar,
    ymin={ymin:.3f},
    ymax={ymax:.3f},
    ylabel={{{ylabel}}},
    symbolic x coords={{{symbolic}}},
    xtick=data,
    xticklabels={{{labels}}},
    x tick label style={{font=\scriptsize, align=center, text depth=0.25ex}},
    enlarge x limits=0.12,
    xmajorgrids=false,"""


def write_risk_proxy_validity(out_path: Path, table_path: Path, rows: Sequence[Row]) -> None:
    stats = risk_proxy_stat_map(rows)
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Scenario & $\rho_s(R_{\mathrm{emp}},p95)$ & $\rho_s(R_{\mathrm{emp}},p99)$ & $\rho_s(R_{\mathrm{emp}},P_{\mathrm{viol}})$ & $p95$ lift & Viol. gap \\",
        r"\hline",
    ]
    for scenario in RISK_VALIDITY_SCENARIOS:
        item = stats[scenario]
        lines.append(
            rf"\texttt{{{scenario.replace('_', r'\_')}}} & "
            + f"{item['rho_p95']:.3f} & {item['rho_p99']:.3f} & {item['rho_violation']:.3f} & "
            + f"{item['p95_lift']:.2f} & {item['violation_gap']:.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    p95_lift_ymax = 1.15 * max(1.0, max(stats[scenario]["p95_lift"] for scenario in RISK_VALIDITY_SCENARIOS))
    viol_gap_ymax = 1.15 * max(0.10, max(stats[scenario]["violation_gap"] for scenario in RISK_VALIDITY_SCENARIOS))
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 2, horizontal sep=1.10cm, vertical sep=1.80cm}},
    width=3.55in,
    height=2.45in,
    ymajorgrids=true,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.45pt}},
    tick style={{black!65, line width=0.45pt}},
    label style={{font=\scriptsize}},
    tick label style={{font=\scriptsize}},
    title style={{font=\small, yshift=-0.5ex}},
]
\nextgroupplot[
    title={{(a) Risk--$p95$ association}},
    {risk_validity_axis_options(r"Spearman $\rho_s$", 1.02, -0.05)}
]
\addplot+[ybar, bar width=7pt, fill=proposedblue, draw=black!45, line width=0.2pt] coordinates {{
{coords_from_stats(stats, "rho_p95")}
}};

\nextgroupplot[
    title={{(b) Risk--violation association}},
    {risk_validity_axis_options(r"Spearman $\rho_s$", 1.02, -0.05)}
]
\addplot+[ybar, bar width=7pt, fill=nomarginred, draw=black!45, line width=0.2pt] coordinates {{
{coords_from_stats(stats, "rho_violation")}
}};

\nextgroupplot[
    title={{(c) Top-risk $p95$ lift}},
    {risk_validity_axis_options(r"Top decile / rest", p95_lift_ymax)}
]
\addplot+[ybar, bar width=7pt, fill=oraclegreen, draw=black!45, line width=0.2pt] coordinates {{
{coords_from_stats(stats, "p95_lift")}
}};
\addplot+[black!55, densely dashed, line width=0.6pt, mark=none, forget plot] coordinates {{(overall,1.0) (traffic_mix_c2_heavy,1.0)}};

\nextgroupplot[
    title={{(d) Top-risk violation gap}},
    {risk_validity_axis_options(r"Top decile $-$ rest", viol_gap_ymax)}
]
\addplot+[ybar, bar width=7pt, fill=blackboxorange, draw=black!45, line width=0.2pt] coordinates {{
{coords_from_stats(stats, "violation_gap")}
}};
\end{{groupplot}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def ns3_window_values(rows: Sequence[Row], scenario: str, key: str) -> List[float]:
    return [safe_float(row, key) for row in rows if row["scenario"] == scenario]


def ns3_p95_vs_background(rows: Sequence[Row], scenario: str) -> str:
    coords: List[str] = []
    for row in rows:
        if row["scenario"] == scenario:
            coords.append(f"({safe_float(row, 'background_rate'):.4f},{1000.0 * safe_float(row, 'p95_delay'):.4f})")
    return "\n".join(coords)


def ns3_mean_by_window(rows: Sequence[Row], scenario: str, key: str) -> str:
    by_window: Dict[int, List[float]] = {}
    for row in rows:
        if row["scenario"] == scenario:
            by_window.setdefault(int(row["window_id"]), []).append(safe_float(row, key))
    coords = [f"({window_id},{mean(values):.4f})" for window_id, values in sorted(by_window.items())]
    return "\n".join(coords)


def packet_delay_by_scenario(packet_rows: Sequence[Row], scenario: str) -> List[float]:
    return [
        1000.0 * safe_float(row, "delay")
        for row in packet_rows
        if row["scenario"] == scenario and row.get("target") == "1" and row.get("received") == "1"
    ]


def ns3_summary_bar(summary_rows: Sequence[Row], key: str, multiplier: float = 1.0) -> str:
    coords = []
    by_scenario = {row["scenario"]: row for row in summary_rows}
    for scenario in SCENARIOS:
        row = by_scenario[scenario]
        coords.append(f"({scenario},{multiplier * safe_float(row, key):.4f})")
    return "\n".join(coords)


def write_ns3_packet_validation(out_path: Path, window_rows: Sequence[Row], summary_rows: Sequence[Row], packet_rows: Sequence[Row]) -> None:
    labels = ",".join(f"{{{SCENARIO_LABELS[scenario]}}}" for scenario in SCENARIOS)
    symbolic = ",".join(SCENARIOS)
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 2, horizontal sep=1.15cm, vertical sep=1.75cm}},
    width=3.55in,
    height=2.45in,
    ymajorgrids=true,
    xmajorgrids=true,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.45pt}},
    tick style={{black!65, line width=0.45pt}},
    label style={{font=\scriptsize}},
    tick label style={{font=\scriptsize}},
    title style={{font=\small, yshift=-0.5ex}},
]
\nextgroupplot[
    title={{(a) Background occupancy}},
    xlabel={{Background rate (pkt/s)}},
    ylabel={{$p95$ delay (ms)}},
    xmin=0, xmax=110,
    ymin=0, ymax=25,
    legend style={{at={{(0.5,1.27)}}, anchor=south, legend columns=2, draw=none, fill=none, font=\scriptsize}},
]
\addplot+[only marks, mark=*, mark size=1.2pt, opacity=0.55, proposedblue] coordinates {{
{ns3_p95_vs_background(window_rows, "overall")}
}};
\addlegendentry{{Overall}}
\addplot+[only marks, mark=square*, mark size=1.2pt, opacity=0.55, blackboxorange] coordinates {{
{ns3_p95_vs_background(window_rows, "vacation_high")}
}};
\addlegendentry{{High vacation}}

\nextgroupplot[
    title={{(b) Mobility-loss signature}},
    xlabel={{Window index}},
    ylabel={{Violation probability}},
    xmin=0, xmax=11,
    ymin=0, ymax=1.02,
]
\addplot+[nomarginred, mark=*, mark size=1.2pt, line width=0.75pt] coordinates {{
{ns3_mean_by_window(window_rows, "drift_strong", "violation_prob")}
}};

\nextgroupplot[
    title={{(c) Target-packet delay CDF}},
    xlabel={{Delay (ms)}},
    ylabel={{CDF}},
    xmin=0, xmax=35,
    ymin=0, ymax=1.02,
    legend style={{at={{(0.03,0.08)}}, anchor=south west, legend columns=1, draw=none, fill=none, font=\scriptsize}},
]
\addplot+[proposedblue, line width=0.75pt, mark=none] coordinates {{
{ecdf_coords(packet_delay_by_scenario(packet_rows, "overall"))}
}};
\addlegendentry{{Overall}}
\addplot+[oraclegreen, line width=0.75pt, mark=none] coordinates {{
{ecdf_coords(packet_delay_by_scenario(packet_rows, "load_high"))}
}};
\addlegendentry{{High load}}
\addplot+[blackboxorange, line width=0.75pt, mark=none] coordinates {{
{ecdf_coords(packet_delay_by_scenario(packet_rows, "vacation_high"))}
}};
\addlegendentry{{High vacation}}

\nextgroupplot[
    title={{(d) Loss-inclusive violations}},
    ybar,
    ymin=0,
    ymax=1.02,
    ylabel={{Violation probability}},
    symbolic x coords={{{symbolic}}},
    xtick=data,
    xticklabels={{{labels}}},
    x tick label style={{font=\scriptsize, align=center, text depth=0.25ex}},
    enlarge x limits=0.16,
    xmajorgrids=false,
]
\addplot+[
    ybar,
    bar width=9pt,
    fill=proposedblue,
    draw=black!45,
    line width=0.2pt
] coordinates {{
{ns3_summary_bar(summary_rows, "violation_prob_mean")}
}};
\end{{groupplot}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def available_ns3_bridge_scenarios(summary_rows: Sequence[Row]) -> List[str]:
    available = {row["scenario"] for row in summary_rows}
    return [scenario for scenario in NS3_BRIDGE_SCENARIOS if scenario in available]


def ns3_window_scatter_coords(rows: Sequence[Row], scenario: str, x_key: str, y_key: str, y_multiplier: float = 1.0) -> str:
    coords: List[str] = []
    for row in rows:
        if row["scenario"] == scenario:
            coords.append(
                f"({safe_float(row, x_key):.4f},{y_multiplier * safe_float(row, y_key):.4f})"
            )
    return "\n".join(coords)


def ns3_bridge_scatter_plots(
    window_rows: Sequence[Row],
    scenarios: Sequence[str],
    x_key: str,
    y_key: str,
    y_multiplier: float = 1.0,
    legend: bool = False,
) -> str:
    plots: List[str] = []
    markers = ["*", "square*", "triangle*", "diamond*", "pentagon*", "otimes*"]
    for idx, scenario in enumerate(scenarios):
        legend_label = SCENARIO_LABELS[scenario].replace(r"\\", " ")
        legend_entry = rf"\addlegendentry{{{legend_label}}}" if legend else ""
        plots.append(
            rf"""\addplot+[only marks, mark={markers[idx % len(markers)]}, mark size=1.15pt, opacity=0.55, {SCENARIO_COLORS[scenario]}] coordinates {{
{ns3_window_scatter_coords(window_rows, scenario, x_key, y_key, y_multiplier)}
}};
{legend_entry}
"""
        )
    return "\n".join(plots)


def ns3_bridge_summary_coords(summary_rows: Sequence[Row], scenarios: Sequence[str], key: str, multiplier: float = 1.0) -> str:
    by_scenario = {row["scenario"]: row for row in summary_rows}
    coords = []
    for scenario in scenarios:
        row = by_scenario[scenario]
        coords.append(f"({scenario},{multiplier * safe_float(row, key):.4f})")
    return "\n".join(coords)


def ns3_bridge_bar_options(scenarios: Sequence[str], ylabel: str, ymax: float) -> str:
    labels = ",".join(f"{{{SCENARIO_LABELS[scenario]}}}" for scenario in scenarios)
    symbolic = ",".join(scenarios)
    return rf"""ybar,
    ymin=0,
    ymax={ymax:.3f},
    ylabel={{{ylabel}}},
    symbolic x coords={{{symbolic}}},
    xtick=data,
    xticklabels={{{labels}}},
    x tick label style={{font=\scriptsize, align=center, text depth=0.25ex}},
    enlarge x limits=0.12,
    xmajorgrids=false,"""


def write_ns3_interruption_bridge(out_path: Path, window_rows: Sequence[Row], summary_rows: Sequence[Row]) -> None:
    scenarios = available_ns3_bridge_scenarios(summary_rows)
    if not scenarios:
        return
    max_background = 1.12 * max(1.0, max(safe_float(row, "background_rate") for row in window_rows))
    max_p95_window = 1.12 * max(1.0, max(1000.0 * safe_float(row, "p95_delay") for row in window_rows))
    max_p95_summary = 1.12 * max(1.0, max(1000.0 * safe_float(row, "p95_delay_mean") for row in summary_rows))
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 2, horizontal sep=1.15cm, vertical sep=1.80cm}},
    width=3.55in,
    height=2.45in,
    ymajorgrids=true,
    xmajorgrids=true,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.45pt}},
    tick style={{black!65, line width=0.45pt}},
    label style={{font=\scriptsize}},
    tick label style={{font=\scriptsize}},
    title style={{font=\small, yshift=-0.5ex}},
]
\nextgroupplot[
    title={{(a) Occupancy--delay bridge}},
    xlabel={{Background rate (pkt/s)}},
    ylabel={{$p95$ delay (ms)}},
    xmin=0, xmax={max_background:.1f},
    ymin=0, ymax={max_p95_window:.1f},
    legend style={{at={{(0.5,1.30)}}, anchor=south, legend columns=3, draw=none, fill=none, font=\scriptsize}},
]
{ns3_bridge_scatter_plots(window_rows, scenarios, "background_rate", "p95_delay", 1000.0, legend=True)}

\nextgroupplot[
    title={{(b) Loss--violation bridge}},
    xlabel={{Loss rate}},
    ylabel={{Violation probability}},
    xmin=0, xmax=1.02,
    ymin=0, ymax=1.02,
]
{ns3_bridge_scatter_plots(window_rows, scenarios, "loss_rate", "violation_prob")}

\nextgroupplot[
    title={{(c) Scenario tail delay}},
    {ns3_bridge_bar_options(scenarios, r"$p95$ delay (ms)", max_p95_summary)}
]
\addplot+[ybar, bar width=7pt, fill=proposedblue, draw=black!45, line width=0.2pt] coordinates {{
{ns3_bridge_summary_coords(summary_rows, scenarios, "p95_delay_mean", 1000.0)}
}};

\nextgroupplot[
    title={{(d) Scenario violation}},
    {ns3_bridge_bar_options(scenarios, r"Violation probability", 1.02)}
]
\addplot+[ybar, bar width=7pt, fill=nomarginred, draw=black!45, line width=0.2pt] coordinates {{
{ns3_bridge_summary_coords(summary_rows, scenarios, "violation_prob_mean")}
}};
\end{{groupplot}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def ablation_risk_ymax(table: SummaryTable) -> float:
    return 1.15 * max_metric(table, SCENARIOS, ABLATION_METHODS, "risk_mae", "proposed_laq", 1.0)


def write_ablation_mechanism(out_path: Path, table: SummaryTable, rows: Sequence[Row]) -> None:
    risk_ymax = ablation_risk_ymax(table)
    delay_ymax = 1.15 * max_metric(table, SCENARIOS, ABLATION_METHODS, "delay_mae", "proposed_laq", 1.0)
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 2, horizontal sep=1.10cm, vertical sep=1.75cm}},
    width=3.55in,
    height=2.45in,
    ymajorgrids=true,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.45pt}},
    tick style={{black!65, line width=0.45pt}},
    label style={{font=\scriptsize}},
    tick label style={{font=\scriptsize}},
    title style={{font=\small, yshift=-0.5ex}},
]
\nextgroupplot[
    title={{(a) Coverage}},
    {bar_axis_options("Coverage", 1.08, legend=True)}
]
{coords_for_bars(table, ABLATION_METHODS, "coverage")}
{reference_line(0.90)}

\nextgroupplot[
    title={{(b) Risk error}},
    {bar_axis_options("Risk MAE / proposed", risk_ymax)}
]
{coords_for_bars(table, ABLATION_METHODS, "risk_mae", "proposed_laq", add_legend=False)}

\nextgroupplot[
    title={{(c) Delay error}},
    {bar_axis_options("Delay MAE / proposed", delay_ymax)}
]
{coords_for_bars(table, ABLATION_METHODS, "delay_mae", "proposed_laq", add_legend=False)}

\nextgroupplot[
    title={{(d) Overall risk difference}},
    xlabel={{Predicted risk $-$ empirical risk}},
    ylabel={{CDF}},
    xmin=-1.0, xmax=1.6,
    ymin=0, ymax=1.02,
    legend style={{at={{(0.03,0.97)}}, anchor=north west, legend columns=1, draw=none, fill=none, font=\scriptsize}},
]
\addplot+[proposedblue, line width=0.75pt, mark=none] coordinates {{
{ecdf_coords(conservativeness_values(rows, "overall", "proposed_laq"))}
}};
\addlegendentry{{Proposed}}
\addplot+[nomarginred, line width=0.75pt, mark=none, densely dashed] coordinates {{
{ecdf_coords(conservativeness_values(rows, "overall", "ablation_no_margin"))}
}};
\addlegendentry{{No margin}}
\addplot+[vacationgold, line width=0.75pt, mark=none, dotted] coordinates {{
{ecdf_coords(conservativeness_values(rows, "overall", "ablation_no_vacation"))}
}};
\addplot+[monepurple, line width=0.75pt, mark=none, dashdotted] coordinates {{
{ecdf_coords(conservativeness_values(rows, "overall", "ablation_m1_only"))}
}};
\addplot+[black!50, dotted, line width=0.6pt, mark=none, forget plot] coordinates {{(0,0) (0,1.02)}};
\end{{groupplot}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def write_screening_effectiveness(out_path: Path, table: SummaryTable) -> None:
    auroc_ymax = 1.02
    precision_ymax = 1.02
    recall_ymax = 1.02
    ece_ymax = 1.15 * max_metric(table, SCENARIOS, CN_METHODS, "calibration_ece", None, 0.1)
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 2, horizontal sep=1.10cm, vertical sep=1.75cm}},
    width=3.55in,
    height=2.45in,
    ymajorgrids=true,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.45pt}},
    tick style={{black!65, line width=0.45pt}},
    label style={{font=\scriptsize}},
    tick label style={{font=\scriptsize}},
    title style={{font=\small, yshift=-0.5ex}},
]
\nextgroupplot[
    title={{(a) High-risk ranking}},
    {bar_axis_options("AUROC", auroc_ymax, legend=True)}
]
{coords_for_bars(table, CN_METHODS, "screening_auroc")}

\nextgroupplot[
    title={{(b) Top-10\% precision}},
    {bar_axis_options("Precision", precision_ymax)}
]
{coords_for_bars(table, CN_METHODS, "precision_at_10", add_legend=False)}

\nextgroupplot[
    title={{(c) Top-10\% recall}},
    {bar_axis_options("Recall", recall_ymax)}
]
{coords_for_bars(table, CN_METHODS, "recall_at_10", add_legend=False)}

\nextgroupplot[
    title={{(d) Calibration error}},
    {bar_axis_options("Mean bin error", ece_ymax)}
]
{coords_for_bars(table, CN_METHODS, "calibration_ece", add_legend=False)}
\end{{groupplot}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def write_adaptive_margin_tradeoff(out_path: Path, table: SummaryTable) -> None:
    max_conserv = max(
        table[scenario][method]["conservativeness"]
        for scenario in HARDCASE_SCENARIOS
        for method in ADAPTIVE_METHODS
        if scenario in table and method in table[scenario]
    )
    min_conserv = min(
        table[scenario][method]["conservativeness"]
        for scenario in HARDCASE_SCENARIOS
        for method in ADAPTIVE_METHODS
        if scenario in table and method in table[scenario]
    )
    max_risk = max(
        table[scenario][method]["risk_mae"]
        for scenario in HARDCASE_SCENARIOS
        for method in ADAPTIVE_METHODS
        if scenario in table and method in table[scenario]
    )
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 2, horizontal sep=1.10cm, vertical sep=1.75cm}},
    width=3.55in,
    height=2.45in,
    ymajorgrids=true,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.45pt}},
    tick style={{black!65, line width=0.45pt}},
    label style={{font=\scriptsize}},
    tick label style={{font=\scriptsize}},
    title style={{font=\small, yshift=-0.5ex}},
]
\nextgroupplot[
    title={{(a) Coverage in stress scenarios}},
    {scenario_axis_options(HARDCASE_SCENARIOS, "Coverage", 1.08, legend=True)}
]
{coords_for_scenario_bars(table, HARDCASE_SCENARIOS, ADAPTIVE_METHODS, "coverage")}
{reference_line_for_scenarios(HARDCASE_SCENARIOS, 0.90)}

\nextgroupplot[
    title={{(b) Risk difference}},
    {scenario_axis_options(HARDCASE_SCENARIOS, r"Calibrated risk $-$ empirical risk", 1.15 * max(0.20, max_conserv), ymin=1.15 * min(-0.10, min_conserv))}
]
{coords_for_scenario_bars(table, HARDCASE_SCENARIOS, ADAPTIVE_METHODS, "conservativeness", add_legend=False)}
\addplot+[black!50, dotted, line width=0.6pt, mark=none, forget plot] coordinates {{(load_high,0) (drift_strong,0)}};

\nextgroupplot[
    title={{(c) Risk error}},
    {scenario_axis_options(HARDCASE_SCENARIOS, "Risk MAE", 1.15 * max_risk)}
]
{coords_for_scenario_bars(table, HARDCASE_SCENARIOS, ADAPTIVE_METHODS, "risk_mae", add_legend=False)}

\nextgroupplot[
    title={{(d) Top-10\% recall}},
    {scenario_axis_options(HARDCASE_SCENARIOS, "Recall", 1.02)}
]
{coords_for_scenario_bars(table, HARDCASE_SCENARIOS, ADAPTIVE_METHODS, "recall_at_10", add_legend=False)}
\end{{groupplot}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def policy_axis_options(ylabel: str, ymax: float, legend: bool = False) -> str:
    labels = ",".join(f"{{{CONTROL_POLICY_LABELS[policy]}}}" for policy in CONTROL_POLICIES)
    symbolic = ",".join(CONTROL_POLICIES)
    options = [
        "ybar",
        "ymin=0",
        f"ymax={ymax:.3f}",
        f"ylabel={{{ylabel}}}",
        f"symbolic x coords={{{symbolic}}}",
        f"xtick={{{symbolic}}}",
        f"xticklabels={{{labels}}}",
        r"x tick label style={font=\scriptsize, rotate=22, anchor=east}",
        "enlarge x limits=0.16",
    ]
    if legend:
        options.append(
            r"legend style={at={(0.5,1.28)}, anchor=south, legend columns=5, draw=none, fill=none, font=\scriptsize}"
        )
    return ",\n    ".join(options) + ","


def control_value(rows_by_policy: Dict[str, Row], policy: str, metric: str) -> float:
    return safe_float(rows_by_policy.get(policy, {}), f"{metric}_mean")


def control_policy_bars(rows_by_policy: Dict[str, Row], metric: str, add_legend: bool = True) -> str:
    plots: List[str] = []
    for policy in CONTROL_POLICIES:
        if policy not in rows_by_policy:
            continue
        value = control_value(rows_by_policy, policy, metric)
        legend_entry = rf"\addlegendentry{{{CONTROL_POLICY_LABELS[policy]}}}" if add_legend else ""
        plots.append(
            rf"""\addplot+[ybar, bar width=8pt, fill={CONTROL_POLICY_COLORS[policy]}, draw=black!45, line width=0.2pt] coordinates {{
({policy},{value:.4f})
}};
{legend_entry}
"""
        )
    return "\n".join(plots)


def control_scatter_plots(rows_by_policy: Dict[str, Row], x_metric: str, y_metric: str) -> str:
    plots: List[str] = []
    markers = ["*", "square*", "triangle*", "diamond*", "otimes*"]
    for idx, policy in enumerate(CONTROL_POLICIES):
        if policy not in rows_by_policy:
            continue
        x_value = control_value(rows_by_policy, policy, x_metric)
        y_value = control_value(rows_by_policy, policy, y_metric)
        plots.append(
            rf"""\addplot+[only marks, mark={markers[idx]}, mark size=2.1pt, {CONTROL_POLICY_COLORS[policy]}] coordinates {{
({x_value:.4f},{y_value:.4f})
}};
\addlegendentry{{{CONTROL_POLICY_LABELS[policy]}}}
"""
        )
    return "\n".join(plots)


def write_control_case_study(out_path: Path, rows: Sequence[Row]) -> None:
    if not rows:
        return
    rows_by_policy = {row["policy"]: row for row in rows}
    max_violation = 1.18 * max(control_value(rows_by_policy, policy, "target_violation_reduction") for policy in CONTROL_POLICIES)
    max_p95 = 1.18 * max(control_value(rows_by_policy, policy, "p95_delay_reduction") for policy in CONTROL_POLICIES)
    max_false = 1.18 * max(control_value(rows_by_policy, policy, "false_alarm_cost") for policy in CONTROL_POLICIES)
    max_bg = 1.18 * max(control_value(rows_by_policy, policy, "background_cost") for policy in CONTROL_POLICIES)
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 2, horizontal sep=1.15cm, vertical sep=1.80cm}},
    width=3.55in,
    height=2.45in,
    ymajorgrids=true,
    xmajorgrids=false,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.45pt}},
    tick style={{black!65, line width=0.45pt}},
    label style={{font=\scriptsize}},
    tick label style={{font=\scriptsize}},
    title style={{font=\small, yshift=-0.5ex}},
]
\nextgroupplot[
    title={{(a) Violation reduction}},
    {policy_axis_options("Reduction", max_violation, legend=True)}
]
{control_policy_bars(rows_by_policy, "target_violation_reduction")}

\nextgroupplot[
    title={{(b) Tail-delay reduction}},
    {policy_axis_options(r"$p95$ reduction", max_p95)}
]
{control_policy_bars(rows_by_policy, "p95_delay_reduction", add_legend=False)}

\nextgroupplot[
    title={{(c) False-alarm cost}},
    {policy_axis_options("False alarms", max_false)}
]
{control_policy_bars(rows_by_policy, "false_alarm_cost", add_legend=False)}

\nextgroupplot[
    title={{(d) Same-cost ranking}},
    xlabel={{Background cost}},
    ylabel={{Violation reduction}},
    xmin=0, xmax={max_bg:.4f},
    ymin=0, ymax={max_violation:.4f},
    legend style={{at={{(0.02,0.98)}}, anchor=north west, legend columns=1, draw=none, fill=none, font=\scriptsize}},
]
{control_scatter_plots(rows_by_policy, "background_cost", "target_violation_reduction")}
\end{{groupplot}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def ns3_heldout_axis_options(ylabel: str, ymax: float, legend: bool = False) -> str:
    labels = ",".join(f"{{{NS3_HELDOUT_LABELS[method]}}}" for method in NS3_HELDOUT_METHODS)
    symbolic = ",".join(NS3_HELDOUT_METHODS)
    options = [
        "ybar",
        "ymin=0",
        f"ymax={ymax:.3f}",
        f"ylabel={{{ylabel}}}",
        f"symbolic x coords={{{symbolic}}}",
        f"xtick={{{symbolic}}}",
        f"xticklabels={{{labels}}}",
        r"x tick label style={font=\scriptsize, rotate=22, anchor=east}",
        "enlarge x limits=0.16",
    ]
    if legend:
        options.append(
            r"legend style={at={(0.5,1.28)}, anchor=south, legend columns=4, draw=none, fill=none, font=\scriptsize}"
        )
    return ",\n    ".join(options) + ","


def ns3_metric_value(row_by_method: Dict[str, Row], method: str, metric: str, multiplier: float = 1.0) -> float:
    return multiplier * safe_float(row_by_method.get(method, {}), metric)


def ns3_heldout_bars(
    row_by_method: Dict[str, Row],
    metric: str,
    multiplier: float = 1.0,
    add_legend: bool = True,
) -> str:
    plots: List[str] = []
    for method in NS3_HELDOUT_METHODS:
        if method not in row_by_method:
            continue
        value = ns3_metric_value(row_by_method, method, metric, multiplier)
        legend_entry = rf"\addlegendentry{{{NS3_HELDOUT_LABELS[method]}}}" if add_legend else ""
        plots.append(
            rf"""\addplot+[ybar, bar width=8pt, fill={NS3_HELDOUT_COLORS[method]}, draw=black!45, line width=0.2pt] coordinates {{
({method},{value:.4f})
}};
{legend_entry}
"""
        )
    return "\n".join(plots)


def write_ns3_heldout_validation(out_path: Path, rows: Sequence[Row]) -> None:
    if not rows:
        return
    row_by_method = {row["method"]: row for row in rows}
    max_risk = 1.18 * max(ns3_metric_value(row_by_method, method, "risk_mae") for method in NS3_HELDOUT_METHODS)
    max_p95 = 1.18 * max(ns3_metric_value(row_by_method, method, "p95_mae", 1000.0) for method in NS3_HELDOUT_METHODS)
    content = rf"""{tex_header()}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 2, horizontal sep=1.15cm, vertical sep=1.80cm}},
    width=3.55in,
    height=2.45in,
    ymajorgrids=true,
    xmajorgrids=false,
    grid style={{black!12}},
    axis x line*=bottom,
    axis y line*=left,
    axis line style={{black!65, line width=0.45pt}},
    tick style={{black!65, line width=0.45pt}},
    label style={{font=\scriptsize}},
    tick label style={{font=\scriptsize}},
    title style={{font=\small, yshift=-0.5ex}},
]
\nextgroupplot[
    title={{(a) NS-3 risk error}},
    {ns3_heldout_axis_options("Risk MAE", max_risk, legend=True)}
]
{ns3_heldout_bars(row_by_method, "risk_mae")}

\nextgroupplot[
    title={{(b) NS-3 tail-delay error}},
    {ns3_heldout_axis_options(r"$p95$ MAE (ms)", max_p95)}
]
{ns3_heldout_bars(row_by_method, "p95_mae", multiplier=1000.0, add_legend=False)}

\nextgroupplot[
    title={{(c) Held-out coverage}},
    {ns3_heldout_axis_options("Coverage", 1.08)}
]
{ns3_heldout_bars(row_by_method, "safe_coverage", add_legend=False)}
\addplot+[black!55, densely dashed, line width=0.6pt, mark=none, forget plot] coordinates {{(ns3_lagged_safe,0.90) (ns3_background_rate,0.90)}};

\nextgroupplot[
    title={{(d) High-risk recall}},
    {ns3_heldout_axis_options("R@10\\%", 1.02)}
]
{ns3_heldout_bars(row_by_method, "recall_at_10", add_legend=False)}
\end{{groupplot}}
\end{{tikzpicture}}
{tex_footer()}"""
    out_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    main_table = read_summary(Path(args.main_summary))
    ablation_table = read_summary(Path(args.ablation_summary))
    main_rows = read_rows(args.main_window_glob)
    ablation_rows = read_rows(args.ablation_window_glob)
    ns3_windows = read_csv(Path(args.ns3_window))
    ns3_summary = read_csv(Path(args.ns3_summary))
    ns3_packets = read_rows(args.ns3_packet_glob)
    control_summary = read_optional_csv(Path(args.control_summary))
    ns3_heldout_summary = read_optional_csv(Path(args.ns3_heldout_summary))

    write_grouped_bar_chart(
        outdir / "main_delay_mae_normalized.tex",
        main_table,
        MAIN_METHODS,
        "delay_mae",
        "Delay MAE / static queueing",
        ymax=1.55,
        normalized_to="static_queueing",
    )
    write_grouped_bar_chart(
        outdir / "main_risk_mae_normalized.tex",
        main_table,
        MAIN_METHODS,
        "risk_mae",
        "Risk MAE / static queueing",
        ymax=3.85,
        normalized_to="static_queueing",
    )
    write_grouped_bar_chart(
        outdir / "main_safe_coverage_bars.tex",
        main_table,
        MAIN_METHODS,
        "coverage",
        "Coverage",
        ymax=1.08,
        reference=0.90,
    )
    write_grouped_bar_chart(
        outdir / "ablation_coverage_bars.tex",
        ablation_table,
        ABLATION_METHODS,
        "coverage",
        "Coverage",
        ymax=1.08,
        reference=0.90,
    )

    prefix = args.figure_prefix
    write_main_performance_landscape(outdir / f"{prefix}main_performance_landscape.tex", main_table)
    if all(method in main_table["overall"] for method in CN_METHODS):
        write_screening_effectiveness(outdir / f"{prefix}screening_effectiveness.tex", main_table)
    write_risk_calibration(outdir / f"{prefix}risk_calibration_windows.tex", main_rows)
    write_risk_proxy_validity(
        outdir / f"{prefix}risk_proxy_validity.tex",
        Path(args.main_summary).parent / "risk_proxy_validity_table.tex",
        main_rows,
    )
    write_ns3_packet_validation(outdir / f"{prefix}ns3_packet_validation.tex", ns3_windows, ns3_summary, ns3_packets)
    write_ns3_interruption_bridge(outdir / f"{prefix}ns3_interruption_bridge.tex", ns3_windows, ns3_summary)
    write_ablation_mechanism(outdir / f"{prefix}ablation_mechanism.tex", ablation_table, ablation_rows)
    if all(method in main_table.get("overall", {}) for method in ADAPTIVE_METHODS):
        write_adaptive_margin_tradeoff(outdir / f"{prefix}adaptive_margin_tradeoff.tex", main_table)
    write_control_case_study(outdir / f"{prefix}control_case_study.tex", control_summary)
    write_ns3_heldout_validation(outdir / f"{prefix}ns3_heldout_validation.tex", ns3_heldout_summary)

    print(f"Wrote PGFPlots sources to: {outdir}")
    for path in sorted(outdir.glob("*.tex")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
