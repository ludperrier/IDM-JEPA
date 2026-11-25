#!/bin/bash
# Regenerate the paper's planning figures and result tables from
# analysis_out_updated, which carries the step-matched baseline checkpoints
# (epochs 160/32/16/11) alongside the epoch-8 evaluations.
set -euo pipefail

cd "$(dirname "$0")"

"${PYTHON:-python}" plot_planning_benchmark.py \
    --sweep-tag v3 --sweep-tag v10 --sweep-tag v11 --sweep-tag v12 --sweep-tag v13 \
    --sweep-tag v16 \
    --epoch 8 --solver cem --benchmark both \
    --planning-dir analysis_out_updated/planning_eval \
    --output-dir analysis_out_updated/planning_benchmark/v3_v10_v11_v12_v13

for benchmark in per_transition per_episode; do
    for extension in pdf png; do
        cp "analysis_out_updated/planning_benchmark/v3_v10_v11_v12_v13/planning_accuracy_vs_label_fraction_${benchmark}.${extension}" \
           "paper/figs/planning_accuracy_vs_label_fraction_${benchmark}.${extension}"
    done
done

"${PYTHON:-python}" build_paper_tables.py
