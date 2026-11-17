#!/bin/bash
#SBATCH --job-name=slots_data
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slots_v1/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slots_v1/logs/%x_%j.err

# slots_v1 data: the hardneg_v1 manifests and the VSI-Bench manifests with object
# slots added (jepa3d/tools/precompute_slots.py; dino.txt + the box05 point probe,
# nothing trained) -> experiments/slots_v1/data/
#   PART=smoke   50 frame groups of the holdout -> data/smoke_f8.jsonl
#   PART=eval    holdout_f8, vsibench_test_posed_f8, vsibench_debiased_posed_f8
#   PART=train   train_f8, shard SHARD (i/n, default 0/1) -> train_f8.jsonl.part<i>
#                (slots_train.sbatch concatenates the parts)
# e.g.  sbatch --export=ALL,PART=train,SHARD=0/4 jepa3d/trainbatch/slots_data.sbatch
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slots_v1/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/slots_v1"
SRC="experiments/hardneg_v1/data"
PART=${PART:-smoke}

cd "$ROOT"
mkdir -p "$EXP/data"
RUN="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
# The same vocabulary manifests in every run: all parts must find the same object names.
COMMON="jepa3d/tools/precompute_slots.py --train-labels data/geo_probe/labels_train.jsonl
    --probe experiments/point_probe_offcentre/ckpt/box05/final.pt
    --vocab-manifests $SRC/train_f8.jsonl $SRC/holdout_f8.jsonl data/vsibench_test_posed_f8.jsonl data/vsibench_debiased_posed_f8.jsonl"

case "$PART" in
smoke)
    $RUN $COMMON --manifest "$SRC/holdout_f8.jsonl" --limit 50 --output "$EXP/data/smoke_f8.jsonl"
    ;;
eval)
    $RUN $COMMON --manifest "$SRC/holdout_f8.jsonl" --output "$EXP/data/holdout_f8.jsonl"
    for split in test debiased; do
        $RUN $COMMON --manifest "data/vsibench_${split}_posed_f8.jsonl" --output "$EXP/data/vsibench_${split}_posed_f8.jsonl"
    done
    ;;
train)
    $RUN $COMMON --manifest "$SRC/train_f8.jsonl" --shard "${SHARD:-0/1}" --output "$EXP/data/train_f8.jsonl"
    ;;
*)
    echo "unknown PART=$PART" >&2
    exit 1
    ;;
esac
echo "[done] $(date)"
