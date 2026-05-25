#!/usr/bin/env python
"""Run dependency-free simulation experiments for Paper 1.

The goal of this script is to create a reproducible first experiment pipeline:

1. Generate packet/window-level samples from a lightweight discrete-event model.
2. Train a structured-input estimator for queueing parameters.
3. Compare proposed, static queueing, oracle queueing, and black-box baselines.
4. Write CSV/JSON outputs for later plotting and paper integration.

This is not the final calibrated simulator. It is the first runnable scaffold
that keeps the paper's modeling split intact:

    X -> learned queueing inputs -> queueing analyzer

rather than

    X -> delay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple


EPS = 1e-9
THETA_KEYS = ["lambda", "m1", "m2", "v1", "v2", "b", "u"]
METRIC_KEYS = ["delay_mean", "risk", "p95", "p99", "violation"]
SCREENING_KEYS = ["screening_auroc", "precision_at_10", "recall_at_10", "calibration_ece"]
CONTROL_POLICIES = [
    "proposed_safe_trigger",
    "direct_risk_trigger",
    "ewma_persistence_trigger",
    "static_utilization_trigger",
    "random_trigger",
]


@dataclass
class Scenario:
    name: str
    load_scale: float = 1.0
    vacation_scale: float = 1.0
    burst_scale: float = 1.0
    drift_level: float = 0.0
    video_prob: float = 0.65
    channel_scale: float = 1.0


@dataclass
class WindowSample:
    scenario: str
    split: str
    target: str
    features: List[float]
    theta: Dict[str, float]
    empirical: Dict[str, float]


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


def second_moment(values: Sequence[float], default: float = 0.0) -> float:
    return sum(v * v for v in values) / len(values) if values else default


def lognormal_with_mean_cv(rng: random.Random, mean_value: float, cv: float) -> float:
    cv = max(cv, 0.05)
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(max(mean_value, EPS)) - 0.5 * sigma2
    return rng.lognormvariate(mu, sigma)


def merge_intervals(intervals: Iterable[Tuple[float, float]], horizon: float) -> List[Tuple[float, float]]:
    clipped = []
    for start, end in intervals:
        start = max(0.0, min(horizon, start))
        end = max(0.0, min(horizon, end))
        if end > start:
            clipped.append((start, end))
    clipped.sort()
    merged: List[Tuple[float, float]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def generate_vacations(rng: random.Random, scenario: Scenario, horizon: float) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []

    # Rule-driven streams. Durations are intentionally small relative to a window,
    # then swept through vacation_scale.
    streams = [
        (1.0, 0.030, "remote_id"),
        (0.55, 0.018, "telemetry"),
        (0.22, 0.008, "c2_heartbeat"),
    ]
    for period, duration, _ in streams:
        phase = rng.uniform(0.0, period)
        t = phase
        while t < horizon:
            jitter = rng.uniform(-0.08 * period, 0.08 * period)
            start = max(0.0, t + jitter)
            dur = duration * scenario.vacation_scale * rng.uniform(0.75, 1.35)
            intervals.append((start, start + dur))
            t += period

    # Random channel-unavailability bursts.
    channel_rate = 0.35 * scenario.channel_scale * scenario.vacation_scale
    t = rng.expovariate(max(channel_rate, EPS)) if channel_rate > 0 else horizon + 1.0
    while t < horizon:
        dur = rng.uniform(0.015, 0.075) * scenario.vacation_scale
        intervals.append((t, t + dur))
        t += rng.expovariate(max(channel_rate, EPS))

    return merge_intervals(intervals, horizon)


def advance_to_available(t: float, vacations: Sequence[Tuple[float, float]]) -> float:
    changed = True
    while changed:
        changed = False
        for start, end in vacations:
            if start <= t < end:
                t = end
                changed = True
                break
    return t


def serve_with_vacations(start: float, active_service: float, vacations: Sequence[Tuple[float, float]]) -> float:
    t = advance_to_available(start, vacations)
    remaining = active_service
    while remaining > EPS:
        next_vac = None
        for vac_start, vac_end in vacations:
            if vac_end <= t:
                continue
            if vac_start >= t:
                next_vac = (vac_start, vac_end)
                break
            if vac_start <= t < vac_end:
                t = vac_end
                next_vac = None
                break

        if next_vac is None:
            t += remaining
            remaining = 0.0
            continue

        vac_start, vac_end = next_vac
        available = max(0.0, vac_start - t)
        if available >= remaining:
            t += remaining
            remaining = 0.0
        else:
            remaining -= available
            t = vac_end
    return t


def generate_arrivals(rng: random.Random, lam: float, burst_scale: float, horizon: float) -> List[float]:
    arrivals: List[float] = []
    t = 0.0
    while True:
        t += rng.expovariate(max(lam, EPS))
        if t >= horizon:
            break
        arrivals.append(t)

        # Occasional local bursts, representing event-triggered C2 or video bursts.
        if rng.random() < min(0.20, 0.04 * burst_scale):
            cluster = rng.randint(1, max(1, int(round(2 * burst_scale))))
            for _ in range(cluster):
                extra = t + rng.uniform(0.001, 0.060)
                if extra < horizon:
                    arrivals.append(extra)
    arrivals.sort()
    return arrivals


def interarrival_burstiness(arrivals: Sequence[float]) -> float:
    if len(arrivals) < 3:
        return 1.0
    gaps = [b - a for a, b in zip(arrivals[:-1], arrivals[1:]) if b > a]
    if len(gaps) < 2:
        return 1.0
    m = mean(gaps, 1.0)
    if m <= EPS:
        return 1.0
    return max(0.2, min(8.0, statistics.pstdev(gaps) / m))


def simulate_window(rng: random.Random, scenario: Scenario, split: str, horizon: float, violation_tau: float) -> WindowSample:
    target = "video" if rng.random() < scenario.video_prob else "event_c2"
    target_video = 1.0 if target == "video" else 0.0

    base_service = 0.050 if target == "video" else 0.025
    base_cv = 0.75 if target == "video" else 0.45
    link_factor = rng.lognormvariate(0.0, 0.18 + 0.12 * scenario.drift_level)
    service_mean = base_service * scenario.channel_scale * link_factor
    service_cv = base_cv * (1.0 + 0.25 * scenario.burst_scale)

    desired_rho = min(0.92, 0.38 * scenario.load_scale * rng.uniform(0.85, 1.20))
    lam = max(0.1, desired_rho / max(service_mean, EPS))
    if scenario.drift_level > 0:
        lam *= rng.lognormvariate(0.0, 0.35 * scenario.drift_level)
        service_mean *= rng.lognormvariate(0.0, 0.22 * scenario.drift_level)

    vacations = generate_vacations(rng, scenario, horizon)
    arrivals = generate_arrivals(rng, lam, scenario.burst_scale, horizon)

    current = 0.0
    waits: List[float] = []
    services: List[float] = []
    delays: List[float] = []
    for arrival in arrivals:
        active_service = lognormal_with_mean_cv(rng, service_mean, service_cv)
        start = max(arrival, current)
        start = advance_to_available(start, vacations)
        completion = serve_with_vacations(start, active_service, vacations)
        current = completion

        waits.append(max(0.0, start - arrival))
        services.append(max(EPS, completion - start))
        delays.append(max(0.0, completion - arrival))

    vac_durations = [end - start for start, end in vacations]
    busy_time = sum(vac_durations)
    empirical_delay = mean(delays, 0.0)
    empirical_p95 = percentile(delays, 0.95)
    empirical_p99 = percentile(delays, 0.99)
    violation = sum(1 for d in delays if d > violation_tau) / len(delays) if delays else 0.0
    empirical_risk = empirical_delay + 0.35 * empirical_p95 + 1.5 * violation

    theta = {
        "lambda": max(EPS, len(arrivals) / horizon),
        "m1": mean(services, service_mean),
        "m2": second_moment(services, service_mean * service_mean * (1.0 + service_cv * service_cv)),
        "v1": mean(vac_durations, 0.0),
        "v2": second_moment(vac_durations, 0.0),
        "b": interarrival_burstiness(arrivals),
        "u": max(0.0, min(0.95, busy_time / horizon)),
    }

    # Observable, noisy window-level features. These mimic what a lightweight
    # estimator could receive from recent packet/context measurements.
    obs_noise = 0.10 + 0.10 * scenario.drift_level
    lambda_obs = theta["lambda"] * rng.lognormvariate(0.0, obs_noise)
    service_hint = theta["m1"] * rng.lognormvariate(0.0, obs_noise)
    vacation_hint = theta["u"] + rng.gauss(0.0, 0.025 + 0.030 * scenario.drift_level)
    burst_hint = theta["b"] + rng.gauss(0.0, 0.15 + 0.10 * scenario.drift_level)
    snr_like = 1.0 / max(link_factor, EPS) + rng.gauss(0.0, 0.08)
    packet_size_hint = (1.0 if target == "video" else 0.45) * rng.lognormvariate(0.0, 0.08)

    features = [
        lambda_obs,
        service_hint,
        max(0.0, min(1.0, vacation_hint)),
        max(0.0, burst_hint),
        snr_like,
        packet_size_hint,
        target_video,
        scenario.load_scale,
        scenario.vacation_scale,
        scenario.drift_level,
    ]

    empirical = {
        "delay_mean": empirical_delay,
        "risk": empirical_risk,
        "p95": empirical_p95,
        "p99": empirical_p99,
        "violation": violation,
    }
    return WindowSample(scenario.name, split, target, features, theta, empirical)


class Standardizer:
    def __init__(self) -> None:
        self.mean: List[float] = []
        self.std: List[float] = []

    def fit(self, rows: Sequence[Sequence[float]]) -> None:
        n_features = len(rows[0])
        self.mean = []
        self.std = []
        for j in range(n_features):
            col = [row[j] for row in rows]
            m = mean(col, 0.0)
            s = statistics.pstdev(col) if len(col) > 1 else 1.0
            self.mean.append(m)
            self.std.append(s if s > EPS else 1.0)

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
        aug[col] = [x / div for x in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < EPS:
                continue
            aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(n + 1)]
    return [aug[i][-1] for i in range(n)]


class RidgeRegressor:
    def __init__(self, alpha: float = 1e-3) -> None:
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
        row = [1.0] + list(x)
        return sum(c * v for c, v in zip(self.coef, row))


class MultiTargetLogModel:
    def __init__(self, keys: Sequence[str], alpha: float = 1e-3) -> None:
        self.keys = list(keys)
        self.alpha = alpha
        self.scaler = Standardizer()
        self.models: Dict[str, RidgeRegressor] = {}

    def fit(self, samples: Sequence[WindowSample], source: str) -> None:
        x = [sample.features for sample in samples]
        self.scaler.fit(x)
        xs = self.scaler.transform(x)
        for key in self.keys:
            model = RidgeRegressor(self.alpha)
            if source == "theta":
                y = [math.log(max(sample.theta[key], EPS)) for sample in samples]
            else:
                y = [math.log(max(sample.empirical[key], EPS)) for sample in samples]
            model.fit(xs, y)
            self.models[key] = model

    def predict(self, sample: WindowSample) -> Dict[str, float]:
        x = self.scaler.transform_one(sample.features)
        out = {}
        for key, model in self.models.items():
            out[key] = max(EPS, math.exp(model.predict_one(x)))
        return out


class ThetaProjector:
    """Project learned queueing inputs back to a bounded feasible domain.

    The theorem in the paper is explicitly local: both true and learned inputs
    are assumed to lie in a bounded domain away from instability. A lightweight
    regressor can extrapolate badly under strong drift, so the experiment should
    enforce the same modeling contract before passing learned inputs to the
    queueing analyzer.
    """

    def __init__(self, eta: float = 0.06) -> None:
        self.eta = eta
        self.bounds: Dict[str, Tuple[float, float]] = {}

    def fit(self, samples: Sequence[WindowSample]) -> None:
        for key in THETA_KEYS:
            values = [sample.theta[key] for sample in samples]
            lo = percentile(values, 0.005)
            hi = percentile(values, 0.995)
            lower = max(EPS, 0.50 * lo)
            upper = max(lower * 1.05, 1.75 * hi)
            if key in {"v1", "v2", "b", "u"}:
                lower = 0.0
            if key == "u":
                upper = min(0.92, max(0.05, hi + 0.10))
            self.bounds[key] = (lower, upper)

    def project(self, theta: Dict[str, float]) -> Dict[str, float]:
        projected = {}
        for key in THETA_KEYS:
            lower, upper = self.bounds[key]
            projected[key] = min(max(theta.get(key, lower), lower), upper)

        projected["m2"] = max(projected["m2"], projected["m1"] * projected["m1"])
        projected["u"] = min(max(projected["u"], 0.0), 0.92)

        max_rho = 1.0 - self.eta
        rho = projected["lambda"] * projected["m1"]
        if rho > max_rho:
            projected["lambda"] = max_rho / max(projected["m1"], EPS)
        return projected

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {key: {"lower": lo, "upper": hi} for key, (lo, hi) in self.bounds.items()}


def queueing_analyzer(theta: Dict[str, float], risk_margin: float = 0.0, violation_tau: float = 0.5) -> Dict[str, float]:
    lam = max(EPS, theta["lambda"])
    m1 = max(EPS, theta["m1"])
    m2 = max(m1 * m1, theta["m2"])
    v1 = max(0.0, theta["v1"])
    v2 = max(0.0, theta["v2"])
    b = max(0.0, theta["b"])
    u = max(0.0, min(0.95, theta["u"]))

    rho = lam * m1
    stable_gap = max(0.04, 1.0 - min(rho, 0.96))
    mg1_wait = lam * m2 / (2.0 * stable_gap)
    residual_vac = v2 / (2.0 * v1) if v1 > EPS else 0.0
    # The simulated effective service time already absorbs interruptions that
    # occur after service starts. The vacation term should therefore mainly
    # represent residual unavailability before service can start, not double
    # count all rule-driven occupancy.
    vacation_wait = 0.35 * u * residual_vac / max(0.08, 1.0 - u) + 0.15 * u * m1
    burst_wait = 0.12 * max(0.0, b - 1.0) * m1

    wait = mg1_wait + vacation_wait + burst_wait
    delay = wait + m1
    service_var = max(0.0, m2 - m1 * m1)
    tail_scale = math.sqrt(service_var + v2 + 0.25 * wait * wait + EPS)
    p95 = delay + 1.64 * tail_scale
    p99 = delay + 2.33 * tail_scale
    if violation_tau <= delay:
        violation = min(1.0, 0.50 + (delay - violation_tau) / max(delay + tail_scale, EPS))
    else:
        violation = math.exp(-(violation_tau - delay) / max(tail_scale, 0.02))
    violation = max(0.0, min(1.0, violation))

    risk = delay + 0.35 * p95 + 1.5 * violation
    return {
        "delay_mean": delay,
        "risk": risk,
        "risk_safe": risk + risk_margin,
        "p95": p95,
        "p99": p99,
        "violation": violation,
        "rho": rho,
    }


def kingman_no_interruption_analyzer(theta: Dict[str, float], risk_margin: float = 0.0, violation_tau: float = 0.5) -> Dict[str, float]:
    """A conventional G/G/1-style approximation without interruption terms."""
    lam = max(EPS, theta["lambda"])
    m1 = max(EPS, theta["m1"])
    m2 = max(m1 * m1, theta["m2"])
    burst_cv = max(0.2, theta.get("b", 1.0))
    service_cv2 = max(0.0, (m2 - m1 * m1) / max(m1 * m1, EPS))
    arrival_cv2 = burst_cv * burst_cv
    rho = min(0.96, lam * m1)
    stable_gap = max(0.04, 1.0 - rho)
    wait = (rho / stable_gap) * 0.5 * (arrival_cv2 + service_cv2) * m1
    delay = wait + m1
    tail_scale = math.sqrt(max(0.0, m2 - m1 * m1) + 0.25 * wait * wait + EPS)
    p95 = delay + 1.64 * tail_scale
    p99 = delay + 2.33 * tail_scale
    if violation_tau <= delay:
        violation = min(1.0, 0.50 + (delay - violation_tau) / max(delay + tail_scale, EPS))
    else:
        violation = math.exp(-(violation_tau - delay) / max(tail_scale, 0.02))
    violation = max(0.0, min(1.0, violation))
    risk = delay + 0.35 * p95 + 1.5 * violation
    return {
        "delay_mean": delay,
        "risk": risk,
        "risk_safe": risk + risk_margin,
        "p95": p95,
        "p99": p99,
        "violation": violation,
        "rho": rho,
    }


def average_theta(samples: Sequence[WindowSample]) -> Dict[str, float]:
    return {key: mean([sample.theta[key] for sample in samples], EPS) for key in THETA_KEYS}


def make_scenarios() -> List[Scenario]:
    return [
        Scenario("overall", load_scale=1.0, vacation_scale=1.0, burst_scale=1.0, drift_level=0.0),
        Scenario("traffic_mix_video_heavy", load_scale=1.0, vacation_scale=1.0, burst_scale=1.2, video_prob=0.90),
        Scenario("traffic_mix_c2_heavy", load_scale=0.9, vacation_scale=1.0, burst_scale=1.5, video_prob=0.25),
        Scenario("load_low", load_scale=0.65, vacation_scale=1.0, burst_scale=0.9),
        Scenario("load_high", load_scale=1.75, vacation_scale=1.0, burst_scale=1.2),
        Scenario("vacation_low", load_scale=1.0, vacation_scale=0.45, burst_scale=1.0),
        Scenario("vacation_high", load_scale=1.0, vacation_scale=2.25, burst_scale=1.0),
        Scenario("drift_mild", load_scale=1.1, vacation_scale=1.2, burst_scale=1.2, drift_level=0.7),
        Scenario("drift_strong", load_scale=1.25, vacation_scale=1.8, burst_scale=1.6, drift_level=1.4, channel_scale=1.2),
    ]


def generate_dataset(
    rng: random.Random,
    train_n: int,
    test_n_per_scenario: int,
    horizon: float,
    violation_tau: float,
) -> Tuple[List[WindowSample], List[WindowSample]]:
    train_scenario = Scenario("train_base", load_scale=1.0, vacation_scale=1.0, burst_scale=1.0, drift_level=0.0)
    train = [simulate_window(rng, train_scenario, "train", horizon, violation_tau) for _ in range(train_n)]
    test: List[WindowSample] = []
    for scenario in make_scenarios():
        for _ in range(test_n_per_scenario):
            test.append(simulate_window(rng, scenario, "test", horizon, violation_tau))
    return train, test


def calibrate_margin(
    samples: Sequence[WindowSample],
    model: MultiTargetLogModel,
    projector: ThetaProjector,
    violation_tau: float,
    theta_transform=None,
) -> float:
    residuals: List[float] = []
    for sample in samples:
        pred_theta = projector.project(model.predict(sample))
        if theta_transform is not None:
            pred_theta = theta_transform(pred_theta)
        pred = queueing_analyzer(pred_theta, violation_tau=violation_tau)
        residuals.append(abs(pred["risk"] - sample.empirical["risk"]))
    return percentile(residuals, 0.90) if residuals else 0.0


def calibrate_direct_margin(
    samples: Sequence[WindowSample],
    model: MultiTargetLogModel,
    score_fn,
    quantile: float = 0.90,
) -> float:
    shortfalls: List[float] = []
    for sample in samples:
        pred = model.predict(sample)
        score = score_fn(pred)
        shortfalls.append(max(0.0, sample.empirical["risk"] - score))
    return percentile(shortfalls, quantile) if shortfalls else 0.0


def calibrate_metric_margin(
    samples: Sequence[WindowSample],
    model: MultiTargetLogModel,
    key: str,
    quantile: float = 0.90,
) -> float:
    shortfalls: List[float] = []
    for sample in samples:
        pred = model.predict(sample)
        shortfalls.append(max(0.0, sample.empirical[key] - pred[key]))
    return percentile(shortfalls, quantile) if shortfalls else 0.0


def descriptor_adaptive_multiplier(sample: WindowSample) -> float:
    load = max(0.0, sample.features[7])
    vacation = max(0.0, sample.features[8])
    drift = max(0.0, sample.features[9])
    unavailable_hint = max(0.0, sample.features[2])
    return (
        1.0
        + 0.70 * max(0.0, load - 1.0)
        + 0.45 * max(0.0, vacation - 1.0)
        + 1.05 * drift
        + 0.35 * max(0.0, unavailable_hint - 0.08)
    )


def calibrate_persistence_margin(samples: Sequence[WindowSample], quantile: float = 0.90) -> float:
    if not samples:
        return 0.0
    state = mean([sample.empirical["risk"] for sample in samples], 0.0)
    shortfalls: List[float] = []
    alpha = 0.35
    for sample in samples:
        shortfalls.append(max(0.0, sample.empirical["risk"] - state))
        state = alpha * sample.empirical["risk"] + (1.0 - alpha) * state
    return percentile(shortfalls, quantile)


def tail_metric_score(pred: Dict[str, float]) -> float:
    delay = max(0.0, pred["delay_mean"])
    p95 = min(max(pred["p95"], delay), 10.0)
    violation = min(max(pred["violation"], 0.0), 1.0)
    return delay + 0.35 * p95 + 1.5 * violation


def metric_risk(delay: float, p95: float, violation: float) -> float:
    return max(0.0, delay) + 0.35 * max(0.0, p95) + 1.5 * max(0.0, min(1.0, violation))


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


def screening_metrics(group_rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    empirical = [float(row["emp_risk"]) for row in group_rows]
    scores = [float(row["pred_risk_safe"]) for row in group_rows]
    threshold = percentile(empirical, 0.90)
    labels = [1 if value >= threshold else 0 for value in empirical]
    top_n = max(1, int(math.ceil(0.10 * len(group_rows))))
    order = sorted(range(len(group_rows)), key=lambda i: scores[i], reverse=True)
    selected = order[:top_n]
    selected_hits = sum(labels[i] for i in selected)
    total_hits = max(1, sum(labels))
    sorted_pairs = sorted(zip(scores, empirical), key=lambda item: item[0])
    bins = 10
    ece_terms: List[float] = []
    for idx in range(bins):
        chunk = sorted_pairs[idx * len(sorted_pairs) // bins : (idx + 1) * len(sorted_pairs) // bins]
        if not chunk:
            continue
        ece_terms.append(abs(mean([score for score, _ in chunk]) - mean([risk for _, risk in chunk])))
    return {
        "screening_auroc": screening_auroc(labels, scores),
        "precision_at_10": selected_hits / top_n,
        "recall_at_10": selected_hits / total_hits,
        "calibration_ece": mean(ece_terms),
    }


def group_threshold(values: Sequence[float], q: float = 0.80) -> float:
    return percentile(values, q) if values else 0.0


def evaluate_control_policies(rows: Sequence[Dict[str, object]], seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    reference_method = "proposed_adaptive_margin" if "proposed_adaptive_margin" in by_method else "proposed_laq"
    policy_sources = {
        "proposed_safe_trigger": (reference_method, "pred_risk_safe"),
        "direct_risk_trigger": (
            "scenario_adaptive_direct_risk"
            if "scenario_adaptive_direct_risk" in by_method
            else "direct_risk_conformal",
            "pred_risk_safe",
        ),
        "ewma_persistence_trigger": ("persistence_ewma_safe", "pred_risk_safe"),
        "static_utilization_trigger": (reference_method, "pred_rho"),
    }

    traces: List[Dict[str, object]] = []
    metrics: List[Dict[str, object]] = []
    scenarios = sorted({str(row["scenario"]) for row in rows})
    rng = random.Random(1000003 + seed)

    proposed_counts: Dict[str, int] = {}
    for scenario in scenarios:
        ref_rows = [row for row in by_method.get(reference_method, []) if str(row["scenario"]) == scenario]
        scores = [float(row["pred_risk_safe"]) for row in ref_rows]
        threshold = group_threshold(scores)
        proposed_counts[scenario] = sum(1 for score in scores if score >= threshold)

    for policy in CONTROL_POLICIES:
        policy_trace: List[Dict[str, object]] = []
        for scenario in scenarios:
            ref_rows = [row for row in by_method.get(reference_method, []) if str(row["scenario"]) == scenario]
            high_threshold = group_threshold([float(row["emp_risk"]) for row in ref_rows], 0.90)
            if policy == "random_trigger":
                candidate_rows = ref_rows
                keyed = [(rng.random(), row) for row in candidate_rows]
                selected_ids = {
                    int(row["window_id"])
                    for _, row in sorted(keyed, key=lambda item: item[0], reverse=True)[: proposed_counts.get(scenario, 0)]
                }
                for row in candidate_rows:
                    action = int(int(row["window_id"]) in selected_ids)
                    high_risk = int(float(row["emp_risk"]) >= high_threshold)
                    policy_trace.append(control_trace_row(seed, scenario, policy, row, action, high_risk))
                continue

            method, key = policy_sources[policy]
            candidate_rows = [row for row in by_method.get(method, []) if str(row["scenario"]) == scenario]
            scores = [safe_row_float(row, key) for row in candidate_rows]
            threshold = group_threshold(scores)
            for row, score in zip(candidate_rows, scores):
                action = int(score >= threshold)
                high_risk = int(float(row["emp_risk"]) >= high_threshold)
                policy_trace.append(control_trace_row(seed, scenario, policy, row, action, high_risk))

        traces.extend(policy_trace)
        metrics.append(control_metric_row(seed, policy, policy_trace))

    return metrics, traces


def safe_row_float(row: Dict[str, object], key: str) -> float:
    value = row.get(key, 0.0)
    if value in ("", None):
        return 0.0
    return float(value)


def control_trace_row(
    seed: int,
    scenario: str,
    policy: str,
    row: Dict[str, object],
    action: int,
    high_risk: int,
) -> Dict[str, object]:
    base_violation = float(row["emp_violation"])
    base_p95 = float(row["emp_p95"])
    benefit = 0.45 if high_risk else 0.15
    p95_benefit = 0.30 if high_risk else 0.10
    controlled_violation = base_violation * (1.0 - action * benefit)
    controlled_p95 = base_p95 * (1.0 - action * p95_benefit)
    return {
        "seed": seed,
        "scenario": scenario,
        "policy": policy,
        "window_id": row["window_id"],
        "action": action,
        "high_risk": high_risk,
        "emp_risk": row["emp_risk"],
        "base_violation": base_violation,
        "controlled_violation": controlled_violation,
        "base_p95": base_p95,
        "controlled_p95": controlled_p95,
        "false_alarm": 1 if action and not high_risk else 0,
    }


def control_metric_row(seed: int, policy: str, traces: Sequence[Dict[str, object]]) -> Dict[str, object]:
    n = max(1, len(traces))
    base_violation = mean([float(row["base_violation"]) for row in traces])
    controlled_violation = mean([float(row["controlled_violation"]) for row in traces])
    base_p95 = mean([float(row["base_p95"]) for row in traces])
    controlled_p95 = mean([float(row["controlled_p95"]) for row in traces])
    action_rate = mean([float(row["action"]) for row in traces])
    false_alarm_rate = mean([float(row["false_alarm"]) for row in traces])
    return {
        "seed": seed,
        "policy": policy,
        "n": n,
        "target_violation_reduction": (base_violation - controlled_violation) / max(base_violation, EPS),
        "p95_delay_reduction": (base_p95 - controlled_p95) / max(base_p95, EPS),
        "trigger_rate": action_rate,
        "background_cost": 0.12 * action_rate,
        "false_alarm_cost": false_alarm_rate,
        "base_violation": base_violation,
        "controlled_violation": controlled_violation,
        "base_p95": base_p95,
        "controlled_p95": controlled_p95,
    }


def remove_vacation_theta(theta: Dict[str, float]) -> Dict[str, float]:
    transformed = dict(theta)
    transformed["v1"] = 0.0
    transformed["v2"] = 0.0
    transformed["u"] = 0.0
    return transformed


def m1_only_theta(theta: Dict[str, float]) -> Dict[str, float]:
    transformed = remove_vacation_theta(theta)
    transformed["m2"] = transformed["m1"] * transformed["m1"]
    transformed["b"] = 1.0
    return transformed


def split_train_validation(samples: Sequence[WindowSample], val_frac: float = 0.20) -> Tuple[List[WindowSample], List[WindowSample]]:
    n_val = max(1, int(round(len(samples) * val_frac)))
    return list(samples[:-n_val]), list(samples[-n_val:])


def evaluate(
    train: Sequence[WindowSample],
    test: Sequence[WindowSample],
    violation_tau: float,
    margin_scale: float,
    include_ablations: bool,
    seed: int = 0,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, float],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    fit_samples, val_samples = split_train_validation(train)

    structured = MultiTargetLogModel(THETA_KEYS, alpha=0.05)
    structured.fit(fit_samples, source="theta")
    projector = ThetaProjector(eta=0.06)
    projector.fit(fit_samples)
    blackbox = MultiTargetLogModel(METRIC_KEYS, alpha=0.05)
    blackbox.fit(fit_samples, source="empirical")

    base_margin = calibrate_margin(val_samples, structured, projector, violation_tau)
    direct_conformal_margin = calibrate_direct_margin(val_samples, blackbox, lambda pred: pred["risk"])
    tail_metric_margin = calibrate_direct_margin(val_samples, blackbox, tail_metric_score)
    direct_p95_margin = calibrate_metric_margin(val_samples, blackbox, "p95")
    persistence_margin = calibrate_persistence_margin(val_samples)
    kingman_margin = calibrate_margin(
        val_samples,
        structured,
        projector,
        violation_tau,
        theta_transform=m1_only_theta,
    )
    no_vacation_margin = calibrate_margin(
        val_samples,
        structured,
        projector,
        violation_tau,
        theta_transform=remove_vacation_theta,
    )
    m1_only_margin = calibrate_margin(
        val_samples,
        structured,
        projector,
        violation_tau,
        theta_transform=m1_only_theta,
    )
    static_theta = average_theta(fit_samples)
    ewma_initial = {
        key: mean([sample.empirical[key] for sample in val_samples], 0.0)
        for key in METRIC_KEYS
    }
    ewma_by_scenario: Dict[str, Dict[str, float]] = {}

    rows: List[Dict[str, object]] = []

    for idx, sample in enumerate(test):
        drift_multiplier = 1.0 + 0.45 * max(0.0, sample.features[-1])
        adaptive_multiplier = descriptor_adaptive_multiplier(sample)
        margin = margin_scale * base_margin * drift_multiplier
        adaptive_margin = margin_scale * base_margin * adaptive_multiplier
        no_vacation_margin_k = margin_scale * no_vacation_margin * drift_multiplier
        m1_only_margin_k = margin_scale * m1_only_margin * drift_multiplier

        predictions: Dict[str, Dict[str, float]] = {}
        predictions["oracle_queueing"] = queueing_analyzer(sample.theta, risk_margin=0.0, violation_tau=violation_tau)
        predictions["static_queueing"] = queueing_analyzer(static_theta, risk_margin=0.0, violation_tau=violation_tau)

        proposed_theta = projector.project(structured.predict(sample))
        predictions["proposed_laq"] = queueing_analyzer(proposed_theta, risk_margin=margin, violation_tau=violation_tau)
        predictions["proposed_adaptive_margin"] = queueing_analyzer(
            proposed_theta,
            risk_margin=adaptive_margin,
            violation_tau=violation_tau,
        )

        if include_ablations:
            predictions["ablation_no_margin"] = queueing_analyzer(
                proposed_theta,
                risk_margin=0.0,
                violation_tau=violation_tau,
            )
            predictions["ablation_no_vacation"] = queueing_analyzer(
                remove_vacation_theta(proposed_theta),
                risk_margin=no_vacation_margin_k,
                violation_tau=violation_tau,
            )
            predictions["ablation_m1_only"] = queueing_analyzer(
                m1_only_theta(proposed_theta),
                risk_margin=m1_only_margin_k,
                violation_tau=violation_tau,
            )

        bb_pred = blackbox.predict(sample)
        bb_pred["risk_safe"] = bb_pred["risk"]
        predictions["blackbox_delay"] = bb_pred

        direct_pred = dict(bb_pred)
        direct_pred["risk_safe"] = direct_pred["risk"] + margin_scale * direct_conformal_margin * drift_multiplier
        predictions["direct_risk_conformal"] = direct_pred

        scenario_direct_pred = dict(bb_pred)
        scenario_direct_pred["risk_safe"] = (
            scenario_direct_pred["risk"] + margin_scale * direct_conformal_margin * adaptive_multiplier
        )
        predictions["scenario_adaptive_direct_risk"] = scenario_direct_pred

        tail_pred = dict(bb_pred)
        tail_pred["risk"] = tail_metric_score(tail_pred)
        tail_pred["risk_safe"] = tail_pred["risk"] + margin_scale * tail_metric_margin * drift_multiplier
        predictions["tail_metric_ridge"] = tail_pred

        direct_p95_pred = dict(bb_pred)
        direct_p95_pred["risk"] = metric_risk(
            direct_p95_pred["delay_mean"],
            direct_p95_pred["p95"],
            direct_p95_pred["violation"],
        )
        direct_p95_safe = direct_p95_pred["p95"] + margin_scale * direct_p95_margin * adaptive_multiplier
        direct_p95_pred["risk_safe"] = metric_risk(
            direct_p95_pred["delay_mean"],
            direct_p95_safe,
            direct_p95_pred["violation"],
        )
        predictions["direct_p95_conformal"] = direct_p95_pred

        ewma_state = ewma_by_scenario.setdefault(sample.scenario, dict(ewma_initial))
        ewma_pred = {
            "delay_mean": ewma_state["delay_mean"],
            "risk": ewma_state["risk"],
            "risk_safe": ewma_state["risk"] + margin_scale * persistence_margin * adaptive_multiplier,
            "p95": ewma_state["p95"],
            "p99": ewma_state["p99"],
            "violation": min(max(ewma_state["violation"], 0.0), 1.0),
            "rho": sample.features[0] * sample.features[1],
        }
        predictions["persistence_ewma_safe"] = ewma_pred

        predictions["kingman_no_vacation"] = kingman_no_interruption_analyzer(
            m1_only_theta(proposed_theta),
            risk_margin=margin_scale * kingman_margin * drift_multiplier,
            violation_tau=violation_tau,
        )

        for method, pred in predictions.items():
            safe_risk = pred.get("risk_safe", pred["risk"])
            rows.append(
                {
                    "window_id": idx,
                    "scenario": sample.scenario,
                    "target": sample.target,
                    "method": method,
                    "emp_delay_mean": sample.empirical["delay_mean"],
                    "pred_delay_mean": pred["delay_mean"],
                    "emp_risk": sample.empirical["risk"],
                    "pred_risk": pred["risk"],
                    "pred_risk_safe": safe_risk,
                    "pred_rho": pred.get("rho", ""),
                    "emp_p95": sample.empirical["p95"],
                    "pred_p95": pred["p95"],
                    "emp_p99": sample.empirical["p99"],
                    "pred_p99": pred["p99"],
                    "emp_violation": sample.empirical["violation"],
                    "pred_violation": pred["violation"],
                    "delay_abs_error": abs(pred["delay_mean"] - sample.empirical["delay_mean"]),
                    "risk_abs_error": abs(pred["risk"] - sample.empirical["risk"]),
                    "p95_abs_error": abs(pred["p95"] - sample.empirical["p95"]),
                    "p99_abs_error": abs(pred["p99"] - sample.empirical["p99"]),
                    "violation_abs_error": abs(pred["violation"] - sample.empirical["violation"]),
                    "safe_covers": 1.0 if safe_risk >= sample.empirical["risk"] else 0.0,
                    "conservativeness": safe_risk - sample.empirical["risk"],
                }
            )
        alpha = 0.35
        for key in METRIC_KEYS:
            ewma_state[key] = alpha * sample.empirical[key] + (1.0 - alpha) * ewma_state[key]

    summary: List[Dict[str, object]] = []
    groups = sorted({(row["scenario"], row["method"]) for row in rows})
    for scenario, method in groups:
        group_rows = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
        summary.append(
            {
                "scenario": scenario,
                "method": method,
                "n": len(group_rows),
                "delay_mae": mean([float(row["delay_abs_error"]) for row in group_rows]),
                "risk_mae": mean([float(row["risk_abs_error"]) for row in group_rows]),
                "p95_mae": mean([float(row["p95_abs_error"]) for row in group_rows]),
                "p99_mae": mean([float(row["p99_abs_error"]) for row in group_rows]),
                "violation_mae": mean([float(row["violation_abs_error"]) for row in group_rows]),
                "safe_coverage": mean([float(row["safe_covers"]) for row in group_rows]),
                "avg_conservativeness": mean([float(row["conservativeness"]) for row in group_rows]),
                **screening_metrics(group_rows),
            }
        )

    calibration = {
        "base_risk_margin": base_margin,
        "direct_conformal_margin": direct_conformal_margin,
        "tail_metric_margin": tail_metric_margin,
        "direct_p95_margin": direct_p95_margin,
        "persistence_margin": persistence_margin,
        "kingman_no_vacation_margin": kingman_margin,
        "ablation_no_vacation_margin": no_vacation_margin,
        "ablation_m1_only_margin": m1_only_margin,
        "margin_scale": margin_scale,
        "include_ablations": include_ablations,
        "violation_tau": violation_tau,
        "theta_projector": projector.as_dict(),
    }
    control_metrics, control_traces = evaluate_control_policies(rows, seed=seed)
    return rows, summary, calibration, control_metrics, control_traces


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_console_summary(summary: Sequence[Dict[str, object]]) -> None:
    print("\nTop-level summary (delay_mae / risk_mae / coverage):")
    scenarios = sorted({str(row["scenario"]) for row in summary})
    methods = [
        "proposed_laq",
        "proposed_adaptive_margin",
        "static_queueing",
        "oracle_queueing",
        "blackbox_delay",
        "direct_risk_conformal",
        "scenario_adaptive_direct_risk",
        "direct_p95_conformal",
        "tail_metric_ridge",
        "persistence_ewma_safe",
        "kingman_no_vacation",
        "ablation_no_margin",
        "ablation_no_vacation",
        "ablation_m1_only",
    ]
    for scenario in scenarios:
        print(f"\n[{scenario}]")
        for method in methods:
            matches = [row for row in summary if row["scenario"] == scenario and row["method"] == method]
            if not matches:
                continue
            row = matches[0]
            print(
                f"  {method:18s} "
                f"delay_mae={float(row['delay_mae']):.4f} "
                f"risk_mae={float(row['risk_mae']):.4f} "
                f"coverage={float(row['safe_coverage']):.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 1 simulation experiments.")
    parser.add_argument("--preset", choices=["quick", "main"], default="quick")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", default="")
    parser.add_argument("--train-windows", type=int, default=0)
    parser.add_argument("--test-windows", type=int, default=0, help="Test windows per scenario.")
    parser.add_argument("--horizon", type=float, default=10.0, help="Window length in seconds.")
    parser.add_argument("--violation-tau", type=float, default=0.45, help="Delay threshold for violation metric.")
    parser.add_argument("--margin-scale", type=float, default=1.0, help="Multiplier for the calibrated safety margin.")
    parser.add_argument("--include-ablations", action="store_true", help="Also report ablation variants of the proposed method.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preset == "quick":
        train_n = args.train_windows or 180
        test_n = args.test_windows or 45
    else:
        train_n = args.train_windows or 1200
        test_n = args.test_windows or 250

    if args.outdir:
        outdir = args.outdir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join("paper1_draft", "experiment_outputs", f"{args.preset}_{stamp}")

    rng = random.Random(args.seed)
    train, test = generate_dataset(rng, train_n, test_n, args.horizon, args.violation_tau)
    rows, summary, calibration, control_metrics, control_traces = evaluate(
        train,
        test,
        args.violation_tau,
        args.margin_scale,
        args.include_ablations,
        seed=args.seed,
    )

    os.makedirs(outdir, exist_ok=True)
    write_csv(os.path.join(outdir, "window_predictions.csv"), rows)
    write_csv(os.path.join(outdir, "summary_metrics.csv"), summary)
    write_csv(os.path.join(outdir, "control_policy_metrics.csv"), control_metrics)
    write_csv(os.path.join(outdir, "control_action_traces.csv"), control_traces)
    with open(os.path.join(outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "preset": args.preset,
                "seed": args.seed,
                "train_windows": train_n,
                "test_windows_per_scenario": test_n,
                "horizon": args.horizon,
                "margin_scale": args.margin_scale,
                "include_ablations": args.include_ablations,
                "outdir": outdir,
                "calibration": calibration,
                "scenarios": [scenario.__dict__ for scenario in make_scenarios()],
            },
            f,
            indent=2,
        )

    print_console_summary(summary)
    print(f"\nWrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
