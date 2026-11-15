#!/bin/bash
#SBATCH --job-name=siglip2_probe
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/siglip2_probe/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/siglip2_probe/logs/%x_%j.err

# siglip2_probe: the geometry probes of jepa3d/geo_probe, rerun on the SigLIP2
# 3DRAE (configs/models/siglip2_feature.yaml: SigLIP2 encoder + its own neck).
# Same data, probe and schedule as the DINOv2 runs in outputs/geo_probe.
#   RUN=category        category name -> position, 3DRAE-SigLIP2 tokens   (DINOv2 3DRAE: outputs/geo_probe/real)
#   RUN=point           point prompt  -> position                          (point_3drae_real)
#   RUN=reverse         ray -> category, attention over the tokens          (reverse_3drae_real)
#   RUN=reverse_index   ray -> category from the indexed SigLIP2 patch      (reverse_index_dino)
#   SMOKE=1             16 scenes, 2 epochs, output runs/smoke_<RUN>
# Each run first does the smoke version, so a crash shows within minutes.
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/siglip2_probe/logs

set -euo pipefail
RUN=${RUN:-category}
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/siglip2_probe"

cd "$ROOT"
PY="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
COMMON="--stats data/frame_stats/scannet.jsonl --feature-config configs/models/siglip2_feature.yaml
        --train-labels data/geo_probe/labels_train.jsonl --val-labels data/geo_probe/labels_val.jsonl"

case "$RUN" in
    category)       CMD="jepa3d/geo_probe/train.py --tokens real --memory 3drae --query category" ;;
    point)          CMD="jepa3d/geo_probe/train.py --tokens real --memory 3drae --query point" ;;
    reverse)        CMD="jepa3d/geo_probe/train_reverse.py --tokens real --memory 3drae --readout attention" ;;
    reverse_index)  CMD="jepa3d/geo_probe/train_reverse.py --memory dino --readout index" ;;
    *) echo "unknown RUN=$RUN" >&2; exit 1 ;;
esac

rm -rf "$EXP/runs/smoke_$RUN"
$PY $CMD $COMMON --limit 16 --epochs 2 --eval-every 1 --output "$EXP/runs/smoke_$RUN"
if [ "${SMOKE:-0}" != 1 ]; then
    $PY $CMD $COMMON --output "$EXP/runs/$RUN"
fi
echo "[done] $(date)"
