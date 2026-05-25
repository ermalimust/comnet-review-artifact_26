#!/usr/bin/env python
"""Held-out NS-3 window-level screening evaluation.

The script uses postprocessed NS-3 window metrics. It trains lightweight
lag-feature predictors on calibration seeds and evaluates high-risk screening on
held-out seeds. It intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


EPS = 1e-9
TARGET_KEYS = ["mean_delay", "p95_delay", "violation_prob"]
METHODS = ["ns3_lagged_safe", "ns3_direct_nomargin", "ns3_ewma_safe", "ns3_background_rate"]


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
    parser.add_argument("--train-seeds", default="7,11,13")
    parser.add_argument("--test-seeds", default="17,19")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


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


def mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


class Standardizer:
    def __init__(self) -> None:
        self.mean: List[float] = []
        self.std: List[float] = []

    def fit(self, rows: Sequence[Sequence[float]]) -> None:
        self.mean = []
        self.std = []
        for j in range(len(rows[0])):
            col = [row[j] for row in rows]
            mu = mean(col)
            sigma = math.sqrt(mean([(value - mu) ** 2 for value in col]))
            self.mean.append(mu)
            self.std.append(sigma if sigma > EPS else 1.0)

    def transform_one(self, row: Sequence[float]) -> List[float]:
        return [(row[j] - self.mean[j]) / self.std[j] for j in range(len(row))]

    def transform(self, rows: Sequence[Sequence[float]]) -> List[List[float]]:
        return [self.transform_one(row) for row in rows]


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < EPS:
            aug[pivot][col] = EPS
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [value / div for value in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < EPS:
                continue
            aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(n + 1)]
    return [aug[i][-1] for i in range(n)]


class RidgeRegressor:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.coef: List[float] = []

    def fit(self, x_rows: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        x_aug = [[1.0] + list(row) for row in x_rows]
        p = len(x_aug[0])
        xtx = [[0.0 for _ in range(p)] for _ in range(p)]
        xty = [0.0 for _ in range(p)]
        for row, target in zip(x_aug, y):
            for i in range(p):
                xty[i] += row[i] * target
                for j in range(p):
                    xtx[i][j] += row[i] * row[j]
        for i in range(1, p):
            xtx[i][i] += self.alpha
        self.coef = solve_linear_system(xtx, xty)

    def predict_one(self, x: Sequence[float]) -> float:
        return sum(coef * value for coef, value in zip(self.coef, [1.0] + list(x)))


class MultiTargetLogModel:
    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = list(keys)
        self.scaler = Standardizer()
        self.models: Dict[str, RidgeRegressor] = {}

    def fit(self, rows: Sequence[Dict[str, object]]) -> None:
        x_rows = [row["features"] for row in rows]
        self.scaler.fit(x_rows)
        x_scaled = self.scaler.transform(x_rows)
        for key in self.keys:
            model = RidgeRegressor(alpha=0.08)
            y = [math.log(max(float(row[key]), EPS)) for row in rows]
            model.fit(x_scaled, y)
            self.models[key] = model

    def predict(self, features: Sequence[float]) -> Dict[str, float]:
        x = self.scaler.transform_one(features)
        return {key: max(EPS, math.exp(model.predict_one(x))) for key, model in self.models.items()}


def risk_score(mean_delay: float, p95_delay: float, violation: float) -> float:
    return max(0.0, mean_delay) + 0.35 * max(0.0, p95_delay) + 1.5 * max(0.0, min(1.0, violation))


def build_samples(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    ordered = sorted(rows, key=lambda row: (row["scenario"], int(row["seed"]), int(row["window_id"])))
    previous_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    samples: List[Dict[str, object]] = []
    for row in ordered:
        key = (row["scenario"], row["seed"])
        prev = previous_by_key.get(key, row)
        features = [
            safe_float(prev, "target_rate"),
            safe_float(prev, "background_rate"),
            safe_float(prev, "loss_rate"),
            safe_float(prev, "mean_delay"),
            safe_float(prev, "p95_delay"),
            safe_float(prev, "p99_delay"),
            safe_float(prev, "violation_prob"),
        ]
        mean_delay = safe_float(row, "mean_delay")
        p95_delay = safe_float(row, "p95_delay")
        violation = safe_float(row, "violation_prob")
        samples.append(
            {
                "scenario": row["scenario"],
                "seed": row["seed"],
                "window_id": row["window_id"],
                "features": features,
                "mean_delay": mean_delay,
                "p95_delay": p95_delay,
                "violation_prob": violation,
                "risk": risk_score(mean_delay, p95_delay, violation),
                "background_rate": safe_float(row, "background_rate"),
            }
        )
        previous_by_key[key] = row
    return samples


def screening_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(ranked):
        j = idx + 1
        while j < len(ranked) and abs(ranked[j][0] - ranked[idx][0]) <= 1e-12:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in ranked[idx:j])
        idx = j
    return (rank_sum - positives * (positives + 1) / 2.0) / max(EPS, positives * negatives)


def metrics_for_method(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    empirical = [float(row["emp_risk"]) for row in rows]
    scores = [float(row["pred_risk_safe"]) for row in rows]
    threshold = percentile(empirical, 0.90)
    labels = [1 if value >= threshold else 0 for value in empirical]
    top_n = max(1, math.ceil(0.10 * len(rows)))
    order = sorted(range(len(rows)), key=lambda idx: scores[idx], reverse=True)
    hits = sum(labels[idx] for idx in order[:top_n])
    total_hits = max(1, sum(labels))
    return {
        "n": len(rows),
        "risk_mae": mean([abs(float(row["pred_risk"]) - float(row["emp_risk"])) for row in rows]),
        "p95_mae": mean([abs(float(row["pred_p95"]) - float(row["emp_p95"])) for row in rows]),
        "violation_mae": mean(
            [abs(float(row["pred_violation"]) - float(row["emp_violation"])) for row in rows]
        ),
        "safe_coverage": mean([1.0 if float(row["pred_risk_safe"]) >= float(row["emp_risk"]) else 0.0 for row in rows]),
        "avg_conservativeness": mean([float(row["pred_risk_safe"]) - float(row["emp_risk"]) for row in rows]),
        "screening_auroc": screening_auroc(labels, scores),
        "recall_at_10": hits / total_hits,
    }


def display_method(method: str) -> str:
    return {
        "ns3_lagged_safe": "Lagged calibrated predictor",
        "ns3_direct_nomargin": "Lagged no margin",
        "ns3_ewma_safe": "EWMA calibrated predictor",
        "ns3_background_rate": "Background-rate trigger",
    }.get(method, method.replace("_", " "))


def write_latex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Method &",
        r"\multicolumn{1}{c}{Risk MAE (s)} &",
        r"\multicolumn{1}{c}{$p95$ MAE (ms)} &",
        r"\multicolumn{1}{c}{Viol. MAE} &",
        r"\multicolumn{1}{c}{Coverage} &",
        r"\multicolumn{1}{c}{R@10\%} \\",
        r"\hline",
    ]
    by_method = {str(row["method"]): row for row in rows}
    for method in METHODS:
        row = by_method[method]
        lines.append(
            f"{display_method(method)} & "
            + f"{float(row['risk_mae']):.4f} & "
            + f"{1000.0 * float(row['p95_mae']):.3f} & "
            + f"{float(row['violation_mae']):.3f} & "
            + f"{float(row['safe_coverage']):.3f} & "
            + f"{float(row['recall_at_10']):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(samples: Sequence[Dict[str, object]], train_seeds: Sequence[str], test_seeds: Sequence[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train = [sample for sample in samples if str(sample["seed"]) in train_seeds]
    test = [sample for sample in samples if str(sample["seed"]) in test_seeds]
    model = MultiTargetLogModel(TARGET_KEYS)
    model.fit(train)

    train_preds = [model.predict(sample["features"]) for sample in train]
    train_risk_preds = [
        risk_score(pred["mean_delay"], pred["p95_delay"], pred["violation_prob"])
        for pred in train_preds
    ]
    margin = percentile(
        [max(0.0, float(sample["risk"]) - pred) for sample, pred in zip(train, train_risk_preds)],
        0.90,
    )
    ewma_margin = percentile(
        [
            max(0.0, float(sample["risk"]) - risk_score(*ewma_tuple(train, idx)))
            for idx, sample in enumerate(train)
        ],
        0.90,
    )
    bg_scale = mean([float(sample["risk"]) for sample in train]) / max(
        mean([float(sample["background_rate"]) for sample in train]), EPS
    )

    prediction_rows: List[Dict[str, object]] = []
    for idx, sample in enumerate(test):
        pred = model.predict(sample["features"])
        pred_risk = risk_score(pred["mean_delay"], pred["p95_delay"], pred["violation_prob"])
        previous_mean, previous_p95, previous_violation = ewma_tuple(test, idx)
        previous_risk = risk_score(previous_mean, previous_p95, previous_violation)
        bg_risk = bg_scale * float(sample["background_rate"])
        method_preds = {
            "ns3_lagged_safe": (
                pred["mean_delay"],
                pred["p95_delay"],
                pred["violation_prob"],
                pred_risk,
                pred_risk + margin,
            ),
            "ns3_direct_nomargin": (
                pred["mean_delay"],
                pred["p95_delay"],
                pred["violation_prob"],
                pred_risk,
                pred_risk,
            ),
            "ns3_ewma_safe": (
                previous_mean,
                previous_p95,
                previous_violation,
                previous_risk,
                previous_risk + ewma_margin,
            ),
            "ns3_background_rate": (
                pred["mean_delay"],
                pred["p95_delay"],
                pred["violation_prob"],
                bg_risk,
                bg_risk + margin,
            ),
        }
        for method, values in method_preds.items():
            mean_delay, p95_delay, violation, risk, risk_safe = values
            prediction_rows.append(
                {
                    "scenario": sample["scenario"],
                    "seed": sample["seed"],
                    "window_id": sample["window_id"],
                    "method": method,
                    "emp_risk": sample["risk"],
                    "pred_risk": risk,
                    "pred_risk_safe": risk_safe,
                    "emp_p95": sample["p95_delay"],
                    "pred_p95": p95_delay,
                    "emp_violation": sample["violation_prob"],
                    "pred_violation": violation,
                }
            )

    summary_rows = []
    for method in METHODS:
        method_rows = [row for row in prediction_rows if row["method"] == method]
        summary_rows.append({"method": method, **metrics_for_method(method_rows)})
    return prediction_rows, summary_rows


def ewma_tuple(samples: Sequence[Dict[str, object]], idx: int) -> Tuple[float, float, float]:
    scenario = samples[idx]["scenario"]
    if idx == 0 or samples[idx - 1]["scenario"] != scenario or samples[idx - 1]["seed"] != samples[idx]["seed"]:
        prior = [sample for sample in samples[:idx] if sample["scenario"] == scenario]
        if not prior:
            prior = samples[: max(1, idx)]
        return (
            mean([float(sample["mean_delay"]) for sample in prior], float(samples[idx]["mean_delay"])),
            mean([float(sample["p95_delay"]) for sample in prior], float(samples[idx]["p95_delay"])),
            mean([float(sample["violation_prob"]) for sample in prior], float(samples[idx]["violation_prob"])),
        )
    prev = samples[idx - 1]
    return float(prev["mean_delay"]), float(prev["p95_delay"]), float(prev["violation_prob"])


def main() -> None:
    args = parse_args()
    train_seeds = [item.strip() for item in args.train_seeds.split(",") if item.strip()]
    test_seeds = [item.strip() for item in args.test_seeds.split(",") if item.strip()]
    samples = build_samples(read_csv(Path(args.window)))
    prediction_rows, summary_rows = evaluate(samples, train_seeds, test_seeds)
    outdir = Path(args.outdir)
    write_csv(prediction_rows, outdir / "ns3_shadow_window_predictions.csv")
    write_csv(summary_rows, outdir / "ns3_shadow_prediction_metrics.csv")
    write_latex(summary_rows, outdir / "ns3_shadow_screening_table.tex")
    print(f"Wrote {len(prediction_rows)} held-out NS-3 prediction rows.")
    print(f"Wrote {outdir / 'ns3_shadow_prediction_metrics.csv'}")
    print(f"Wrote {outdir / 'ns3_shadow_screening_table.tex'}")


if __name__ == "__main__":
    main()
