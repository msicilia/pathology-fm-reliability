#!/bin/sh
# Run every analysis stage on the cached embeddings (./embeddings) and write the
# result files to ./results, the figures to ./results/figures and the tables to
# ./results/tables.
#
# Run from any directory:
#     sh scripts/run_pipeline.sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p results

uv run python src/pipeline/ood_sweep.py
uv run python src/analysis/probe_metrics.py
uv run python src/analysis/ood_summary.py
uv run python src/analysis/blur_accuracy.py
uv run python src/analysis/probe_sensitivity.py
uv run python src/analysis/scorer_sensitivity.py
uv run python src/analysis/gate_operating_point.py
uv run python src/analysis/slide_aggregation.py
uv run python src/analysis/label_coarsening.py
uv run python src/analysis/sicap_without_benign.py
uv run python src/figures/make_figures.py
uv run python src/figures/make_tables.py
