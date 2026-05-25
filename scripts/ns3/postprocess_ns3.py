#!/usr/bin/env python
"""Postprocess NS-3 UAV vehicular vacation traces.

The NS-3 scratch program writes packet-level CSV files. This script aggregates
them into window-level and scenario-level delay-risk metrics that can be used
as supplemental evidence in the paper.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SUMMARY_METRICS = [
    "target_packets",
    "target_received",
    "loss_rate",
    "target_rate",
    "background_rate",
    "mean_delay",
    "p95_delay",
    "p99_delay",
    "received_violation_prob",
    "violation_prob",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more packet CSV files or glob patterns.",
    )
    parser.add_argument(
        "--outdir",
        default=r"paper1_draft/experiment_outputs/ns3_validation",
        help="Output directory.",
    )
    parser.add_argument("--window", type=float, default=10.0, help="Window length in seconds.")
    parser.add_argument("--warmup", type=float, default=5.0, help="Ignore packets before this time.")
    parser.add_argument(
        "--violation-tau",
        type=float,
        default=0.45,
        help="Delay threshold for violation probability.",
    )
    return parser.parse_args()


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


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


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def read_packet_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing packet trace: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(path)
                rows.append(row)
    return rows


def safe_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def window_id(tx_time: float, warmup: float, window: float) -> int:
    return int(math.floor((tx_time - warmup) / window))


def build_window_metrics(
    rows: Iterable[Dict[str, str]],
    warmup: float,
    window: float,
    violation_tau: float,
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        tx_time = safe_float(row, "tx_time")
        if tx_time < warmup:
            continue
        scenario = row["scenario"]
        seed = row.get("seed", "unknown")
        wid = window_id(tx_time, warmup, window)
        groups[(scenario, seed, wid)].append(row)

    output: List[Dict[str, object]] = []
    for (scenario, seed, wid), items in sorted(groups.items()):
        target_rows = [row for row in items if row.get("target") == "1"]
        bg_rows = [row for row in items if row.get("target") != "1"]
        received_target = [row for row in target_rows if row.get("received") == "1"]
        delays = [safe_float(row, "delay") for row in received_target]
        target_count = len(target_rows)
        target_received = len(received_target)
        loss_rate = 1.0 - (target_received / target_count) if target_count else 0.0
        received_violations = sum(1 for delay in delays if delay > violation_tau)
        received_violation = received_violations / len(delays) if delays else 0.0
        deadline_or_loss_violation = (
            (target_count - target_received + received_violations) / target_count
            if target_count
            else 0.0
        )

        output.append(
            {
                "scenario": scenario,
                "seed": seed,
                "window_id": wid,
                "window_start": warmup + wid * window,
                "window_end": warmup + (wid + 1) * window,
                "target_packets": target_count,
                "target_received": target_received,
                "background_packets": len(bg_rows),
                "loss_rate": loss_rate,
                "target_rate": target_count / window,
                "background_rate": len(bg_rows) / window,
                "mean_delay": mean(delays),
                "p95_delay": percentile(delays, 0.95),
                "p99_delay": percentile(delays, 0.99),
                "received_violation_prob": received_violation,
                "violation_prob": deadline_or_loss_violation,
            }
        )
    return output


def aggregate_windows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["scenario"])].append(row)

    output: List[Dict[str, object]] = []
    for scenario, items in sorted(groups.items()):
        out: Dict[str, object] = {
            "scenario": scenario,
            "windows": len(items),
            "seeds": ";".join(sorted({str(row["seed"]) for row in items})),
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in items]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = sample_std(values)
        output.append(out)
    return output


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pm(row: Dict[str, object], metric: str, decimals: int = 4, scale: float = 1.0) -> str:
    mu = scale * float(row[f"{metric}_mean"])
    sigma = scale * float(row[f"{metric}_std"])
    return f"{mu:.{decimals}f} $\\pm$ {sigma:.{decimals}f}"


def write_latex_summary(rows: Sequence[Dict[str, object]], path: Path) -> None:
    scenario_order = ["overall", "load_high", "vacation_high", "drift_strong"]
    by_scenario = {str(row["scenario"]): row for row in rows}
    ordered = [by_scenario[name] for name in scenario_order if name in by_scenario]
    ordered.extend(row for row in rows if str(row["scenario"]) not in scenario_order)

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Scenario &",
        r"\multicolumn{1}{c}{\shortstack{Target rate\\(pkt/s)}} &",
        r"\multicolumn{1}{c}{\shortstack{Loss\\rate}} &",
        r"\multicolumn{1}{c}{\shortstack{Mean delay\\(ms)}} &",
        r"\multicolumn{1}{c}{\shortstack{$p95$ delay\\(ms)}} &",
        r"\multicolumn{1}{c}{\shortstack{Deadline/loss\\viol.}} \\",
        r"\hline",
    ]
    for row in ordered:
        scenario = str(row["scenario"]).replace("_", r"\_")
        lines.append(
            rf"\texttt{{{scenario}}} & "
            + f"{fmt_pm(row, 'target_rate', 2)} & "
            + f"{fmt_pm(row, 'loss_rate', 3)} & "
            + f"{fmt_pm(row, 'mean_delay', 4, scale=1000.0)} & "
            + f"{fmt_pm(row, 'p95_delay', 4, scale=1000.0)} & "
            + f"{fmt_pm(row, 'violation_prob', 3)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.input)
    packet_rows = read_packet_rows(paths)
    window_rows = build_window_metrics(packet_rows, args.warmup, args.window, args.violation_tau)
    summary_rows = aggregate_windows(window_rows)

    outdir = Path(args.outdir)
    write_csv(window_rows, outdir / "ns3_window_metrics.csv")
    write_csv(summary_rows, outdir / "ns3_summary.csv")
    write_latex_summary(summary_rows, outdir / "ns3_validation_table.tex")

    print(f"Read {len(packet_rows)} packet rows from {len(paths)} file(s).")
    print(f"Wrote {len(window_rows)} window rows to {outdir / 'ns3_window_metrics.csv'}")
    print(f"Wrote {len(summary_rows)} summary rows to {outdir / 'ns3_summary.csv'}")
    print(f"Wrote {outdir / 'ns3_validation_table.tex'}")


if __name__ == "__main__":
    main()
