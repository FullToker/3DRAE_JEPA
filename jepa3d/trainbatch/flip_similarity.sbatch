#!/bin/bash
#SBATCH --job-name=flip_sim
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/flip_similarity_check/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/flip_similarity_check/logs/%x_%j.err

# flip_similarity_check: cosine similarity, in the Y-encoder's space, between a sentence
# and the same sentence with its spatial relation flipped (jepa3d/tools/check_flip_similarity.py).
# No training. Pretrained embeddinggemma, and the Y-encoder + y_proj of hardneg_v1
# (trained with same-question wrong answers as negatives).
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/flip_similarity_check/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/flip_similarity_check"

cd "$ROOT"
srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache "$CONTAINER" \
    python3 jepa3d/tools/check_flip_similarity.py \
    --cache-dir "$VJEPA/ckpts" \
    --manifest experiments/hardneg_v1/data/holdout_f8.jsonl \
    --checkpoint hardneg_v1=experiments/hardneg_v1/ckpt/final.pt \
    --output "$EXP/results/flip_similarity.json"
echo "[done] $(date)"
