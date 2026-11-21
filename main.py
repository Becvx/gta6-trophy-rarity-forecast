#!/usr/bin/env python3
"""
GTA VI trophy rarity forecast entry.

Forecasts each trophy's earn rate at the case target (2026-11-26T00:00:00Z)
as five quantiles: 5%, 25%, 50%, 75%, 95%. Score = mean pinball loss on logit
scale over every trophy and level (lower is better).

Approach:
  - For each trophy, fit a linear trend of logit(rate) vs time (hours).
  - Predict the logit rate at the target from the fitted trend.
  - Spread: house spread (0.5 * sqrt(days)) + residual sigma, so the spread
    stays wide and robust.
  - Convert back to percentages and emit the five quantiles.

Stdlib only. Deterministic: no random, so two runs produce the same bytes.
"""

import datetime
import json
import math
import os
import sys
from pathlib import Path

INPUT_DIR = 'C:/Users/ravel/gta6-entry/gta6-trophy-rarity-forecast-01-kit/input'
OUTPUT_DIR = 'C:/Users/ravel/gta6-entry/gta6-trophy-rarity-forecast-01-kit/out'

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA = 0.5
CLAMP_MIN = 0.01
CLAMP_MAX = 99.99


def parse_iso(s: str) -> float:
    dt = datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    return (dt - epoch).total_seconds() / 3600.0


def logit(pct: float) -> float:
    p = max(CLAMP_MIN, min(CLAMP_MAX, pct))
    return math.log(p / (100.0 - p))


def sigmoid(x: float) -> float:
    return 100.0 / (1.0 + math.exp(-x))


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    b = num / den if den != 0 else 0.0
    return my - b * mx, b


def residual_variance(xs: list[float], ys: list[float], a: float, b: float) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    return sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 1)


def forecast_trophy(rates: list[float], times_hours: list[float], horizon_hours: float) -> list[float]:
    valid = [(t, r) for t, r in zip(times_hours, rates) if r is not None]
    if not valid:
        d = horizon_hours / 1440.0
        spread = SIGMA * math.sqrt(d)
        return [round4(100.0 / (1.0 + math.exp(-(logit(r) + z * spread)))) for z in Z]

    xs = [t for t, _ in valid]
    ys = [logit(r) for _, r in valid]
    a, b = linear_fit(xs, ys)

    x0 = min(xs)
    x_pred = x0 + horizon_hours
    logit_pred = a + b * x_pred

    horizon_days = horizon_hours / 1440.0
    spread_base = SIGMA * math.sqrt(horizon_days)
    var_resid = residual_variance(xs, ys, a, b)
    spread_resid = math.sqrt(max(var_resid, 1e-9))
    spread = spread_base + spread_resid

    return [round4(sigmoid(logit_pred + z * spread)) for z in Z]


def round4(x: float) -> float:
    return round(x, 4)


def main() -> None:
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in sorted(input_dir.glob('*.in')):
        with open(input_file, 'r') as f:
            case = json.load(f)

        if case.get('levels') != LEVELS:
            raise ValueError(f"{input_file.name} asks for levels {case.get('levels')}, not {LEVELS}")

        horizon_hours = case['horizon_minutes'] / 60.0
        trophies = case['trophies']
        history = case['history']

        trophy_rates = [[] for _ in trophies]
        times_by_trophy = [[] for _ in trophies]
        if history:
            first_time = parse_iso(history[0]['at'])
            for read in history:
                t = parse_iso(read['at']) - first_time
                for i, rate in enumerate(read.get('rates', []) or []):
                    trophy_rates[i].append(rate)
                    times_by_trophy[i].append(t)

        trophies_out = []
        for i, trophy in enumerate(trophies):
            q = forecast_trophy(trophy_rates[i], times_by_trophy[i], horizon_hours)
            trophies_out.append({'id': trophy['id'], 'q': q})

        output = {'trophies': trophies_out}
        out_file = output_dir / f"{case['id']}.out"
        with open(out_file, 'w') as f:
            json.dump(output, f)
            f.write('\n')


if __name__ == '__main__':
    main()
