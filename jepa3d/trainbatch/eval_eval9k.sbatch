#!/bin/bash
#SBATCH --job-name=eval9k
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_mix/eval9k_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_mix/eval9k_%j.err

# SPAR eval9k (7,870 scannet/scannetpp rows, same file as job 12300) on the
# VSI+SPAR mix checkpoints, so they can be compared with sft_3drae_spar.
# Output: <run output_dir>/infer_eval9k_<ckpt>/{metrics.json,predictions.jsonl}

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
cd "$ROOT"

SIF="singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA \
    -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER"

for CONFIG in jepa3d/configs/sft_3drae_mix_f8.yaml jepa3d/configs/sft_3drae_mix_f8_long.yaml; do
    OUT=$(sed -n 's|^ *output_dir: ${RAE_ROOT}/||p' "$CONFIG")
    CKPT="$OUT/final.pt"
    echo "[eval9k $CKPT] start $(date)"
    srun $SIF python3 jepa3d/infer_spar.py --config "$CONFIG" --checkpoint "$CKPT" \
        --input data/spar3d_eval9k.jsonl --output-dir "$OUT/infer_eval9k_final"
    echo "[eval9k $CKPT] done $(date)"
done
