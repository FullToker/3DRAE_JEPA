#!/bin/bash
#SBATCH --job-name=slot_chain
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slot_chain_check/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slot_chain_check/logs/%x_%j.err

# slot_chain_check: name -> 3D position with no training, dino.txt picks the
# patch and the existing point probe locates it (jepa3d/tools/chain_teacher_point.py).
# To compare with the category -> position probe (outputs/geo_probe/real, 0.95 m).
# A smoke run on 8 scenes comes first.
#   SMOKE=1   only the smoke run
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slot_chain_check/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/slot_chain_check"

cd "$ROOT"
RUN="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
COMMON="jepa3d/tools/chain_teacher_point.py --stats data/frame_stats/scannet.jsonl
        --labels data/geo_probe/labels_val.jsonl --train-labels data/geo_probe/labels_train.jsonl
        --probe outputs/geo_probe/point_3drae_real/final.pt"
$RUN $COMMON --limit 8 --output "$EXP/results/chain_smoke"
if [ "${SMOKE:-0}" != 1 ]; then
    $RUN $COMMON --output "$EXP/results/chain"
fi
echo "[done] $(date)"
