@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: Change to repo root (the folder where this script resides)
cd /d "%~dp0" || (echo [ERROR] Cannot change directory to script location.& exit /b 1)

echo.
echo === HKML Demo: one-click run (install -> make tiny DB -> custom Top-K -> A/B) ===

echo [1/4] Installing package...
python -m pip install -U pip || goto :err
pip install -e . || goto :err

echo.
echo [2/4] Creating tiny offline DuckDB (no internet)...
python scripts\make_tiny_db.py --out data\tiny.duckdb --table ohlcv_hk ^
  --tickers 0700.HK,3690.HK,9988.HK,0005.HK,0001.HK,1299.HK,0011.HK --rows 252 --start 2020-01-01 --seed 42 || goto :err

echo.
echo [3/4] Configuring strategy parameters...
if not defined HKML_LOOKBACK set HKML_LOOKBACK=60
if not defined HKML_TOPK set HKML_TOPK=2
echo     LOOKBACK=%HKML_LOOKBACK%
echo     TOP_K   =%HKML_TOPK%

echo.
echo [4/4] Running backtest + A/B (block bootstrap)...
python -m hkml.simple_cli --db "%CD%\data\tiny.duckdb" --table ohlcv_hk --rebal W ^
  --weights-mod "%CD%\my_simple_strategy.py" --tc-bps 10 --outdir "%CD%\outputs" ^
  --abtest --ab-block 10 --ab-B 2000 || goto :err

echo.
echo Done. Artifacts saved to: outputs\
echo   - equity_curve.png, excess_equity.png, bootstrap_hist.png
echo   - metrics.json, abtest.json
echo.
exit /b 0

:err
echo.
echo [FAILED] A step failed. Please review the error messages above.
exit /b 1
