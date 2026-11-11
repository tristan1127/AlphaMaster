# macOS AlphaMaster offline research quickstart

This guide is for macOS 13+ and Python 3.11, using local Parquet OHLCV only.
**No MT5, no live brokerage credentials, no automatic order execution.**
It describes a best-effort environment; the full application has not been end-to-end tested on macOS by this contributor.

## 1. Install Python and get the corrected research branch

Install Python 3.11 (e.g. using Homebrew with `brew install python@3.11`), then:

```bash
git clone --branch fix/walk-forward-purge-20261111 https://github.com/tristan1127/AlphaMaster.git
cd AlphaMaster
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-research-macos.txt
```

The repository's original `requirements.txt` includes `MetaTrader5`, for which the official Python wheels are Windows-only. The **research** requirements intentionally exclude it. Do not use `python run.py` (live MT5 execution).

## 2. Inspect the regression tests

```bash
python -m pytest -q tests/unit/test_walk_forward_gap.py tests/unit/test_backtest.py
```

These are focused checks; passing them does **not** prove that strategy backtests are correct.

## 3. Generate fake data solely to check that the UI starts

```bash
python scripts/make_synthetic_parquet.py --output data/DEMO_H1.parquet
python run_web.py --port 8765
```

Open `http://127.0.0.1:8765`. Choose the generated `data/DEMO_H1.parquet` (absolute path is more reliable if a macOS native picker is unavailable).

Fake data is a random walk, **not** real financial market data; ignore any training PnL/Sharpe shown for it.

You may alternatively run a short CLI installation smoke test (still synthetic data only):

```bash
python - <<'PY'
from model_core.config import ModelConfig
ModelConfig.TRAIN_STEPS = 5  # smoke test only; default is much larger
from train_file import train_from_file
train_from_file("data/DEMO_H1.parquet", from_scratch=True)
PY
```

## 4. Real historical data: required Parquet schema

Use a file named `BTCUSDT_H1.parquet`, `AAPL_D1.parquet`, etc. containing at least **300** chronologically ordered bars and these columns:

| column | interpretation |
| --- | --- |
| `time` | Unix seconds, not milliseconds |
| `open` | opening price |
| `high` | maximum price (>= open/close) |
| `low` | minimum price (<= open/close) |
| `close` | closing price |
| `tick_volume` or `volume` | bar volume |

Prefer thousands of bars and a long time span for actual factor research. Validate timezone, duplicates, missing bars, adjustments and instrument identifiers yourself.

```bash
python train_file.py --data-file /absolute/path/to/BTCUSDT_H1.parquet
```

The default `TRAIN_STEPS` is **9000**; do not start a full training job before confirming that everything imports and the tests pass. Training may use substantial CPU/time.

## 5. Research validity checklist

1. Keep the last 20–30% of chronological data untouched for **final** validation. The in-engine Walk-Forward score is repeatedly used to select formulas, so it is not an unbiased final holdout.
2. Do not train or tune on this final holdout, including repeatedly consulting its metrics.
3. Audit positions/returns: the project target is `log(open[t+2]/open[t+1])`, and costs are charged when the position changes. Verify this timing against the intended executable fills and bid/ask data.
4. Check no look-ahead in normalization, indicators and strategy selection. Include survivorship, data adjustment, spreads and realistic slippage in later research.
5. Compare net-of-cost performance with buy-and-hold, simple time-series momentum, and a randomized or shuffled signal.
6. Start with **paper trading only**. No strategy performance claim follows from installing or running this software.

## Boundaries

The macOS research installation excludes MT5 and potentially optional live data sources. TradingView/tvdatafeed, third-party APIs and Linux/Windows-specific modules are not included in this minimal setup. The native file picker may need macOS Tk support; CLI Parquet training avoids that GUI requirement.
