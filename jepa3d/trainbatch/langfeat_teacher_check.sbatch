#!/bin/bash
#SBATCH --job-name=teacher_check
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/langfeat_teacher_check/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/langfeat_teacher_check/logs/%x_%j.err

# langfeat_teacher_check: does dino.txt ground category names in our views
# (jepa3d/tools/check_teacher_grounding.py)? No training. EmbodiedScan val
# ScanNet scenes, the geometry probe's validation samples.
#   SIZES="224x308 448x616"   teacher input sizes, one result file each
#   EXTRA="--limit 8"         smoke run
# Weights: checkpoints/dinotxt (dinov2 checkout + hub/checkpoints).
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/langfeat_teacher_check/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/langfeat_teacher_check"

cd "$ROOT"
RUN="srun singularity exec --nv --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
for size in ${SIZES:-224x308 448x616}; do
    $RUN jepa3d/tools/check_teacher_grounding.py --stats data/frame_stats/scannet.jsonl \
        --labels data/geo_probe/labels_val.jsonl --train-labels data/geo_probe/labels_train.jsonl \
        --size ${size%x*} ${size#*x} ${EXTRA:-} \
        --output "$EXP/results/dinotxt_${size}${TAG:-}.json"
done
echo "[done] $(date)"
