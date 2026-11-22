#!/bin/bash
#SBATCH --job-name=mmsibench_eval
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/mmsibench_eval/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/mmsibench_eval/logs/%x_%j.err

# MMSI-Bench (1,000 multiple-choice questions, 2-10 unposed images) for trained models.
# The benchmark has no camera poses: jepa3d/tools/make_mmsibench_manifest.py gives every
# view the same placeholder camera, so this measures the models with wrong geometry input.
#   1. manifest with placeholder cameras, then object slots for its rows
#      -> experiments/mmsibench_eval/data/mmsibench.jsonl
#   2. each run of RUNS with real / shuffled / zero scene tokens (and the slot controls)
#      -> experiments/mmsibench_eval/results/<run>.json
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/mmsibench_eval/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/mmsibench_eval"
SRC="experiments/hardneg_v1/data"
MANIFEST="$EXP/data/mmsibench.jsonl"

cd "$ROOT"
mkdir -p "$EXP/data" "$EXP/results"
PY="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
if [ ! -f "$MANIFEST" ]; then
    $PY jepa3d/tools/make_mmsibench_manifest.py --input "$VJEPA/data/mmsibench_manifest.jsonl" --output "$EXP/data/mmsibench_raw.jsonl"
    $PY jepa3d/tools/precompute_slots.py --train-labels data/geo_probe/labels_train.jsonl \
        --probe experiments/point_probe_offcentre/ckpt/box05/final.pt \
        --vocab-manifests $SRC/train_f8.jsonl $SRC/holdout_f8.jsonl data/vsibench_test_posed_f8.jsonl data/vsibench_debiased_posed_f8.jsonl \
        --manifest "$EXP/data/mmsibench_raw.jsonl" --output "$MANIFEST" ${LIMIT:+--limit "$LIMIT"}
fi
for run in ${RUNS:-hardneg_v1 slots_v2 balnum_v1 delex_v1}; do
    echo "[$run] start $(date)"
    $PY jepa3d/tools/eval_holdout.py --config "jepa3d/configs/$run.yaml" --checkpoint "experiments/$run/ckpt/final.pt" \
        --eval "$MANIFEST" --output "$EXP/results/$run.json" ${LIMIT:+--limit "$LIMIT"}
done
echo "[done] $(date)"
