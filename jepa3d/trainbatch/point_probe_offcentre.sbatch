#!/bin/bash
#SBATCH --job-name=pt_offcentre
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/point_probe_offcentre/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/point_probe_offcentre/logs/%x_%j.err

# point_probe_offcentre: retrain the point probe with prompts anywhere on the object
# (geo_probe/train.py --box-prompts SHARE), then lift dino.txt's patches with it
# (improve_teacher_grounding.py). The centre-prompt probe: 0.27 m at the centre,
# 0.47 m at a random patch in the box. A smoke run (8 scenes, 2 epochs) comes first.
#   SHARE     share of training prompts moved off the centre (default 0.5)
#   SMOKE=1   only the smoke run
# Without a free GPU: sbatch --gres=gpu:0 --cpus-per-task=40 <this file> (slow; fine for the
# training-free runs and the smoke run).
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/point_probe_offcentre/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/point_probe_offcentre"

cd "$ROOT"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK > 8 ? SLURM_CPUS_PER_TASK - 8 : 1))   # 8 loader workers
RUN="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
SHARE=${SHARE:-0.5}
NAME="box$(echo "$SHARE" | tr -d .)"
TRAIN="jepa3d/geo_probe/train.py --stats data/frame_stats/scannet.jsonl --query point --box-prompts $SHARE
       --train-labels data/geo_probe/labels_train.jsonl --val-labels data/geo_probe/labels_val.jsonl"
CHAIN="jepa3d/tools/improve_teacher_grounding.py --stats data/frame_stats/scannet.jsonl
       --labels data/geo_probe/labels_val.jsonl --train-labels data/geo_probe/labels_train.jsonl"
rm -rf "$EXP/ckpt/smoke_$NAME"
$RUN $TRAIN --limit 8 --epochs 2 --eval-every 1 --batch-size 4 --output "$EXP/ckpt/smoke_$NAME"
$RUN $CHAIN --limit 8 --probe "$EXP/ckpt/smoke_$NAME/final.pt" --output "$EXP/results/smoke_$NAME"
if [ "${SMOKE:-0}" != 1 ]; then
    $RUN $TRAIN --output "$EXP/ckpt/$NAME"
    $RUN $CHAIN --probe "$EXP/ckpt/$NAME/final.pt" --output "$EXP/results/$NAME"
fi
rm -rf "$EXP/ckpt/smoke_$NAME"
echo "[done] $(date)"
