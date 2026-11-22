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
# The benchmark has no camera poses. CAMERAS=vggt (default): poses and intrinsics estimated
# with VGGT, stored once next to the dataset in vjepa201/source_data/mmsibench_vggt/
# (jepa3d/tools/estimate_vggt_poses.py; used for this benchmark only). CAMERAS=placeholder:
# the same camera for every view, as a lower reference.
#   1. manifest with cameras, then object slots for its rows
#      -> experiments/mmsibench_eval/data/mmsibench_<cameras>.jsonl
#   2. each run of RUNS with real / shuffled / zero scene tokens (and the slot controls)
#      -> experiments/mmsibench_eval/results/<cameras>/<run>.json
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/mmsibench_eval/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/mmsibench_eval"
SRC="experiments/hardneg_v1/data"
CAMERAS=${CAMERAS:-vggt}
POSES="$VJEPA/source_data/mmsibench_vggt"
MANIFEST="$EXP/data/mmsibench_$CAMERAS.jsonl"
RAW="$EXP/data/mmsibench_${CAMERAS}_raw.jsonl"

cd "$ROOT"
mkdir -p "$EXP/data" "$EXP/results/$CAMERAS"
PY="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
if [ ! -f "$MANIFEST" ]; then
    if [ "$CAMERAS" = vggt ] && [ ! -f "$POSES/poses.jsonl" ]; then
        $PY jepa3d/tools/estimate_vggt_poses.py --manifest "$VJEPA/data/mmsibench_manifest.jsonl" --output "$POSES"
    fi
    $PY jepa3d/tools/make_mmsibench_manifest.py --input "$VJEPA/data/mmsibench_manifest.jsonl" --output "$RAW" \
        $([ "$CAMERAS" = vggt ] && echo "--poses $POSES/poses.jsonl")
    $PY jepa3d/tools/precompute_slots.py --train-labels data/geo_probe/labels_train.jsonl \
        --probe experiments/point_probe_offcentre/ckpt/box05/final.pt \
        --vocab-manifests $SRC/train_f8.jsonl $SRC/holdout_f8.jsonl data/vsibench_test_posed_f8.jsonl data/vsibench_debiased_posed_f8.jsonl \
        --manifest "$RAW" --output "$MANIFEST" ${LIMIT:+--limit "$LIMIT"}
fi
for run in ${RUNS:-hardneg_v1 slots_v2 balnum_v1 delex_v1}; do
    echo "[$run] start $(date)"
    $PY jepa3d/tools/eval_holdout.py --config "jepa3d/configs/$run.yaml" --checkpoint "experiments/$run/ckpt/final.pt" \
        --eval "$MANIFEST" --output "$EXP/results/$CAMERAS/$run.json" ${LIMIT:+--limit "$LIMIT"}
done
echo "[done] $(date)"
