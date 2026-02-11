#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Data Deduplication - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.0.0
# Description:
#   CLI menu to run the main Data Deduplication pipelines:
#   - run FastAPI (routes defined in src/core/service.py)
#   - print registered routes
#   - run unit tests
#   - run quick smoke checks
###############################################################################

## ============================================================
## DEFAULT ENV
## ============================================================

: "${API_HOST:=0.0.0.0}"
: "${API_PORT:=8080}"
: "${LOG_LEVEL:=INFO}"

## ============================================================
## HELPERS
## ============================================================

print_header() {
  echo "============================================================"
  echo " Data Deduplication - Pipeline Menu"
  echo "============================================================"
  echo " Host : ${API_HOST}"
  echo " Port : ${API_PORT}"
  echo "------------------------------------------------------------"
}

pause() {
  read -r -p "Press Enter to continue..." _
}

## ============================================================
## ACTIONS
## ============================================================

run_api_dev() {
  echo "## Starting FastAPI (uvicorn dev runner)"
  export API_HOST API_PORT LOG_LEVEL
  uvicorn src.core.service:app --host "${API_HOST}" --port "${API_PORT}"
}

print_routes() {
  echo "## Listing registered routes"
  python - <<'PY'
from src.core.service import create_app

app = create_app()

routes = []
for r in app.routes:
    methods = getattr(r, "methods", None)
    path = getattr(r, "path", None)
    if not methods or not path:
        continue

    clean_methods = sorted([m for m in methods if m not in ("HEAD", "OPTIONS")])
    if not clean_methods:
        continue

    routes.append((",".join(clean_methods), path))

for methods, path in sorted(routes, key=lambda x: x[1]):
    print(f"{methods:12s} {path}")
PY
}

run_unit_tests() {
  echo "## Running unit tests (pytest)"
  pytest -q
}

smoke_check() {
  echo "## Smoke check: import + app creation + pipeline import"
  python - <<'PY'
from src.core.service import create_app
from src.pipeline import run_pipeline

app = create_app()
assert app is not None

res = run_pipeline("unknownFunction", payload={})
assert isinstance(res, dict)
assert res.get("code") == "404"

print("OK: smoke check passed")
PY
}

run_eda() {
  echo "## EDA: generate Plotly HTML reports from a CSV"
  read -r -p "Enter CSV path (e.g. data/raw/my_file.csv): " csv_path
  read -r -p "Enter output dir (default: eda_outputs): " out_dir
  out_dir="${out_dir:-eda_outputs}"

  python - <<PY
from pathlib import Path
import pandas as pd

from src.eda.plots import statistics_plots

csv_path = Path(r"${csv_path}").expanduser().resolve()
out_dir = Path(r"${out_dir}").expanduser().resolve()

if not csv_path.exists():
    raise SystemExit(f"CSV not found: {csv_path}")

df = pd.read_csv(csv_path, sep=",", encoding="utf-8", low_memory=False)

statistics_plots(
    df=df,
    output_dir=out_dir,
    open_in_browser=False,
)

print(f"OK: EDA plots written to: {out_dir}")
PY
}

## ============================================================
## MENU
## ============================================================

while true; do
  print_header
  echo "Choose an option:"
  echo "  1) Run FastAPI (dev)"
  echo "  2) Print registered routes"
  echo "  3) Run unit tests"
  echo "  4) Run smoke check"
  echo "  5) Run EDA (Plotly HTML)"
  echo "  0) Exit"
  echo "------------------------------------------------------------"

  read -r -p "Enter choice: " choice
  echo ""

  case "${choice}" in
    1)
      run_api_dev
      pause
      ;;
    2)
      print_routes
      pause
      ;;
    3)
      run_unit_tests
      pause
      ;;
    4)
      smoke_check
      pause
      ;;
    5)
      run_eda
      pause
      ;;
    0)
      echo "Bye."
      exit 0
      ;;
    *)
      echo "Invalid choice. Please try again."
      pause
      ;;
  esac

  echo ""
done
