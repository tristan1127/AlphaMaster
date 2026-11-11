"""Generate deterministic synthetic OHLCV Parquet bars for an offline UI smoke test.

NOT market data. NEVER use the output to evaluate a trading edge or strategy PnL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n_bars: int, seed: int = 42) -> pd.DataFrame:
    if n_bars < 300:
        raise ValueError("Need at least 300 bars for AlphaMaster's current MIN_BARS")
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(loc=0.0, scale=0.003, size=n_bars)
    opens = np.empty(n_bars, dtype=float)
    closes = np.empty(n_bars, dtype=float)
    for i in range(n_bars):
        opens[i] = 100.0 if i == 0 else closes[i - 1]
        closes[i] = opens[i] * np.exp(log_ret[i])
    upper = np.maximum(opens, closes)
    lower = np.minimum(opens, closes)
    highs = upper * (1.0 + rng.uniform(0.0001, 0.003, n_bars))
    lows = lower * (1.0 - rng.uniform(0.0001, 0.003, n_bars))
    ts = pd.date_range("2023-01-01", periods=n_bars, freq="h", tz="UTC")
    seconds = ts.astype("int64") // 1_000_000_000
    return pd.DataFrame({
        "time": seconds,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": rng.integers(100, 5000, size=n_bars),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fake OHLCV data for an offline smoke test only")
    parser.add_argument("--output", default="data/DEMO_H1.parquet")
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.bars, args.seed)
    df.to_parquet(output, index=False)
    print(f"Synthetic test bars: {len(df)} -> {output.resolve()}")
    print("FAKE DATA: for installation/UI testing only, not financial research or trading.")


if __name__ == "__main__":
    main()
