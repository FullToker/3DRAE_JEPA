#!/bin/bash
#SBATCH --job-name=geo_probe
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/geo_probe/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/geo_probe/%x_%j.err

# Geometry probe stages, see jepa3d/geo_probe/README.md.
#   STAGE=labels                 build labels, write projection sheets
#   STAGE=overfit                8 train scenes, fixed views, evaluated on those scenes
#   STAGE=train TOKENS=real      main run (TOKENS=shuffled: category-prior baseline)
#   MEMORY=dino                  DINOv2 patches + rays instead of 3DRAE neck tokens

set -euo pipefail
STAGE=${STAGE:-labels}
TOKENS=${TOKENS:-real}
MEMORY=${MEMORY:-3drae}
NAME=$([ "$MEMORY" = 3drae ] && echo "$TOKENS" || echo "${MEMORY}_$TOKENS")

ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

cd "$ROOT"
mkdir -p logs/geo_probe
RUN="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
COMMON="--stats data/frame_stats/scannet.jsonl"

case "$STAGE" in
    labels)
        $RUN jepa3d/geo_probe/labels.py --output-dir data/geo_probe
        $RUN jepa3d/geo_probe/check_labels.py --labels data/geo_probe/labels_val.jsonl \
            --output logs/geo_probe/label_check --scenes 20
        ;;
    overfit)
        $RUN jepa3d/geo_probe/train.py $COMMON --tokens real \
            --train-labels data/geo_probe/labels_train.jsonl --val-labels data/geo_probe/labels_train.jsonl \
            --limit 8 --fixed-views --val-seeds 1 --epochs 1000 --batch-size 8 --eval-every 250 \
            --output outputs/geo_probe/overfit
        ;;
    train)
        $RUN jepa3d/geo_probe/train.py $COMMON --tokens "$TOKENS" --memory "$MEMORY" \
            --train-labels data/geo_probe/labels_train.jsonl --val-labels data/geo_probe/labels_val.jsonl \
            --output "outputs/geo_probe/$NAME"
        ;;
    *)
        echo "unknown STAGE=$STAGE" >&2
        exit 1
        ;;
esac
