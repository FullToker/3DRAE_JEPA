#!/bin/bash
#SBATCH --job-name=hit_rate
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/teacher_hit_rate/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/teacher_hit_rate/logs/%x_%j.err

# teacher_hit_rate: does another score or another way of picking the patch raise
# dino.txt's hit rate (62% of in-view objects)? jepa3d/tools/improve_teacher_grounding.py,
# nothing is trained. A smoke run on 8 scenes comes first.
#   SMOKE=1   only the smoke run
#   PROBE     point probe that lifts the patches (default: the centre-prompt probe)
#   TAG       suffix of the result files (default: improve)
# Without a free GPU: sbatch --gres=gpu:0 --cpus-per-task=40 <this file> (slow; fine for the
# training-free runs and the smoke run).
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/teacher_hit_rate/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/teacher_hit_rate"

cd "$ROOT"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK > 8 ? SLURM_CPUS_PER_TASK - 8 : 1))   # 8 loader workers
RUN="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
PROBE=${PROBE:-outputs/geo_probe/point_3drae_real/final.pt}
TAG=${TAG:-improve}
COMMON="jepa3d/tools/improve_teacher_grounding.py --stats data/frame_stats/scannet.jsonl
        --labels data/geo_probe/labels_val.jsonl --train-labels data/geo_probe/labels_train.jsonl --probe $PROBE"
$RUN $COMMON --limit 8 --output "$EXP/results/${TAG}_smoke"
if [ "${SMOKE:-0}" != 1 ]; then
    $RUN $COMMON --output "$EXP/results/$TAG"
fi
echo "[done] $(date)"
