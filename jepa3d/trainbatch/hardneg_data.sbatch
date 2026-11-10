#!/bin/bash
#SBATCH --job-name=hardneg_data
#SBATCH --partition=geo
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/hardneg_v1/logs/data_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/hardneg_v1/logs/data_%j.err

# hardneg_v1 manifests: the f8 mix with same-question wrong answers added and
# 1 scene in 20 held out (jepa3d/tools/add_hard_negatives.py). CPU only.
#   experiments/hardneg_v1/data/train_f8.jsonl (+ .stats.json)
#   experiments/hardneg_v1/data/holdout_f8.jsonl
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/hardneg_v1/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/hardneg_v1"

cd "$ROOT"
srun singularity exec --bind /glob/g01-cache:/glob/g01-cache "$CONTAINER" \
    python3 jepa3d/tools/add_hard_negatives.py \
    --input data/mix_vsi_spar_train_f8.jsonl \
    --output-train "$EXP/data/train_f8.jsonl" \
    --output-holdout "$EXP/data/holdout_f8.jsonl"
wc -l "$EXP"/data/*.jsonl
echo "[done] $(date)"
