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
#   STAGE=analyze                error breakdown of RUNS (default: real dino_real)
#   STAGE=report                 execute vggt_probe_report.ipynb into outputs/geo_probe
#   MEMORY=dino                  DINOv2 patches + rays instead of 3DRAE neck tokens
#   MEMORY=vggt                  VGGT aggregator patch tokens + rays
#   MEMORY_SIZE="378 518"        input size of a dino / vggt memory (default 224 308)
#   QUERY=point                  query = DINOv2 patch + ray at the object's projected centre
#   EXTRA="--limit 16 --epochs 2" OUT=smoke   extra train.py arguments, output name
#   ATTENTION_RUNS="dino_real"   analyze: also measure attention on the object's patches

set -euo pipefail
STAGE=${STAGE:-labels}
TOKENS=${TOKENS:-real}
MEMORY=${MEMORY:-3drae}
QUERY=${QUERY:-category}
MEMORY_SIZE=${MEMORY_SIZE:-}
MEM="$MEMORY${MEMORY_SIZE##* }"          # vggt518 for MEMORY_SIZE="378 518"
NAME=$([ "$MEM" = 3drae ] && echo "$TOKENS" || echo "${MEM}_$TOKENS")
[ "$QUERY" = point ] && NAME="point_${MEM}_$TOKENS"
NAME=${OUT:-$NAME}

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
        $RUN jepa3d/geo_probe/train.py $COMMON --tokens "$TOKENS" --memory "$MEMORY" --query "$QUERY" \
            ${MEMORY_SIZE:+--memory-size $MEMORY_SIZE} ${EXTRA:-} \
            --train-labels data/geo_probe/labels_train.jsonl --val-labels data/geo_probe/labels_val.jsonl \
            --output "outputs/geo_probe/$NAME"
        ;;
    analyze)
        for run in ${RUNS:-real dino_real}; do
            echo "=== $run"
            ATT=$(echo " ${ATTENTION_RUNS:-} " | grep -q " $run " && echo --attention || true)
            $RUN jepa3d/geo_probe/analyze.py $COMMON --run "outputs/geo_probe/$run" $ATT \
                --train-labels data/geo_probe/labels_train.jsonl --val-labels data/geo_probe/labels_val.jsonl
        done
        ;;
    report)
        $RUN -m jupyter nbconvert --to notebook --execute jepa3d/geo_probe/vggt_probe_report.ipynb \
            --output "$ROOT/outputs/geo_probe/vggt_probe_report.ipynb"
        ;;
    *)
        echo "unknown STAGE=$STAGE" >&2
        exit 1
        ;;
esac
