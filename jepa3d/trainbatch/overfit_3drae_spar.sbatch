#!/bin/bash
#SBATCH --job-name=overfit_3drae_spar
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_spar/overfit_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_spar/overfit_%j.err

# Plumbing + OOM check before sft_3drae_spar.sbatch: one frozen batch of
# train.batch_size (32) samples repeated for OVERFIT_STEPS steps on 1 GPU,
# then the eval script on a few hundred rows against the overfit checkpoint
# (accuracy meaningless, only checks train -> checkpoint -> infer).
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_spar

set -euo pipefail
OVERFIT_STEPS=${OVERFIT_STEPS:-200}

ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_3drae_spar"
CONFIG="jepa3d/configs/overfit_3drae_spar.yaml"

cd "$ROOT"
mkdir -p "$LOG_DIR"
nvidia-smi dmon -s pucvmet -d 30 > "$LOG_DIR/gpu_monitor_overfit_${SLURM_JOB_ID}.log" 2>&1 &

SIF="srun singularity exec \
    --nv \
    --env-file $VJEPA/.env/hf.env \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env VJEPA_ROOT=$VJEPA \
    -H $VJEPA:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

echo "[overfit-train] start $(date)"
$SIF accelerate launch --num_processes 1 jepa3d/train.py \
    --config "$CONFIG" --overfit --overfit-steps "$OVERFIT_STEPS"
echo "[overfit-train] done $(date)"

echo "[overfit-infer] start $(date)"
$SIF python3 jepa3d/infer_spar.py \
    --config "$CONFIG" \
    --input data/spar3d_eval9k.jsonl \
    --output-dir outputs/overfit_3drae_spar/infer_eval9k_head \
    --limit 256
echo "[overfit-infer] done $(date)"
