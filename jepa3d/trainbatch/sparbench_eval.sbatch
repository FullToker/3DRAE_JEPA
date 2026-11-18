#!/bin/bash
#SBATCH --job-name=sparbench_eval
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/sparbench_eval/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/sparbench_eval/logs/%x_%j.err

# A second evaluation set for trained models: SPAR-Bench (data/spar3d_eval9k.jsonl,
# 7,870 posed multiple-choice rows of ScanNet / ScanNet++ with 1, 2, 3 or 8 views,
# 24 question types; no training on its question formats).
#   1. object slots for its rows (dino.txt + box05 point probe, as slots_v2's data)
#      -> experiments/sparbench_eval/data/spar3d_eval9k.jsonl
#   2. each run of RUNS (default "hardneg_v1 slots_v2") with real / shuffled / zero
#      scene tokens, and the slot controls for models with slots
#      -> experiments/sparbench_eval/results/<run>.json
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/sparbench_eval/logs
#   sbatch jepa3d/trainbatch/sparbench_eval.sbatch

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/sparbench_eval"
SRC="experiments/hardneg_v1/data"
MANIFEST="$EXP/data/spar3d_eval9k.jsonl"

cd "$ROOT"
mkdir -p "$EXP/data" "$EXP/results"
PY="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
if [ ! -f "$MANIFEST" ]; then
    # the vocabulary manifests of slots_data.sbatch: the same object names as in training
    $PY jepa3d/tools/precompute_slots.py --train-labels data/geo_probe/labels_train.jsonl \
        --probe experiments/point_probe_offcentre/ckpt/box05/final.pt \
        --vocab-manifests $SRC/train_f8.jsonl $SRC/holdout_f8.jsonl data/vsibench_test_posed_f8.jsonl data/vsibench_debiased_posed_f8.jsonl \
        --manifest data/spar3d_eval9k.jsonl --output "$MANIFEST" ${LIMIT:+--limit "$LIMIT"}
fi
for run in ${RUNS:-hardneg_v1 slots_v2}; do
    echo "[$run] start $(date)"
    $PY jepa3d/tools/eval_holdout.py --config "jepa3d/configs/$run.yaml" --checkpoint "experiments/$run/ckpt/final.pt" \
        --eval "$MANIFEST" --output "$EXP/results/$run.json" ${LIMIT:+--limit "$LIMIT"}
done
echo "[done] $(date)"
