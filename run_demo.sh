#!/usr/bin/env bash
set -Eeuo pipefail

# Change to repo root (the folder where this script resides)
cd -- "$(dirname "$0")"

echo
echo "=== HKML Demo: one-click run (install -> make tiny DB -> custom Top-K -> A/B) ==="

# Pick python
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

echo "[1/4] Installing package..."
"$PY" -m pip install -U pip
"$PY" -m pip install -e .

echo
echo "[2/4] Creating tiny offline DuckDB (no internet)..."
if [[ -f scripts/make_tiny_db.py ]]; then
  mkdir -p data
  "$PY" scripts/make_tiny_db.py
else
  echo "  - scripts/make_tiny_db.py not found; skipping (assume data/tiny.duckdb already exists)."
fi

echo
echo "[3/4] Running Top-K with custom strategy & A/B test..."
export HKML_LOOKBACK=${HKML_LOOKBACK:-60}
export HKML_TOPK=${HKML_TOPK:-2}
mkdir -p "$PWD/outputs"

"$PY" -m hkml.simple_cli --db data/tiny.duckdb --table ohlcv_hk --rebal W       --weights-mod "$PWD/my_simple_strategy.py" --tc-bps 10 --outdir "$PWD/outputs"       --abtest --ab-block 10 --ab-B 2000

echo
echo "Done. Artifacts saved to: outputs/"
echo "  - equity_curve.png, excess_equity.png, bootstrap_hist.png"
echo "  - metrics.json, abtest.json"
