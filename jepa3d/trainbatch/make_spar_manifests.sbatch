#!/bin/bash
#SBATCH --job-name=spar3d_manifests
#SBATCH --partition=geo
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/data/manifests_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/data/manifests_%j.err

# Posed SPAR manifests for the 3DRAE X-encoder (jepa3d/tools/make_spar_manifest.py):
#   data/spar3d_train.jsonl + data/spar3d_val.jsonl   <- vjepa201 uni8 SFT rows
#   data/spar3d_eval9k.jsonl                          <- vjepa201 9k SPAR eval rows
# CPU only. Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/data

set -euo pipefail

ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

cd "$ROOT"
mkdir -p data
SIF="srun singularity exec --bind /glob/g01-cache:/glob/g01-cache $CONTAINER"

echo "[train manifest] start $(date)"
$SIF python3 jepa3d/tools/make_spar_manifest.py \
    --input "$VJEPA/data/spar_sftqa_sftvljepa_uni8.jsonl" \
    --output data/spar3d_train.jsonl \
    --val-out data/spar3d_val.jsonl \
    --workers 16

echo "[eval manifest] start $(date)"
$SIF python3 jepa3d/tools/make_spar_manifest.py \
    --input "$VJEPA/data/spar_sftqa_eval_old_3k.jsonl" \
    --output data/spar3d_eval9k.jsonl \
    --workers 16

wc -l data/spar3d_*.jsonl
echo "[done] $(date)"
