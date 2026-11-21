#!/bin/bash
#SBATCH --job-name=balnum_data
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/balnum_v1/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/balnum_v1/logs/%x_%j.err

# balnum_v1 data -> experiments/balnum_v1/data/
#   answer_weights.json                           inverse answer-frequency weights and numeric medians of the
#                                                 training manifest (jepa3d/balance.py)
#   vsibench_{test,debiased}_numeric_posed_f8.jsonl   VSI-Bench's numeric questions on posed ScanNet / ScanNet++
#                                                 scenes (make_vsibench_manifest.py --numeric), with object slots
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/balnum_v1/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/balnum_v1"
SRC="experiments/hardneg_v1/data"
VSIB="$VJEPA/source_data/vsibench"

cd "$ROOT"
mkdir -p "$EXP/data"
PY="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"

$PY jepa3d/balance.py --manifest experiments/slots_v2/data/train_f8.jsonl --output "$EXP/data/answer_weights.json"
for split in test:test.jsonl debiased:test_debiased.parquet; do
    raw="$EXP/data/vsibench_${split%%:*}_numeric_raw_f8.jsonl"
    $PY jepa3d/tools/make_vsibench_manifest.py --numeric --input "$VSIB/${split#*:}" --num-views 8 \
        --stats data/frame_stats/scannet.jsonl data/frame_stats/scannetpp.jsonl --output "$raw"
    # the vocabulary manifests of slots_data.sbatch: the same object names as in training
    $PY jepa3d/tools/precompute_slots.py --train-labels data/geo_probe/labels_train.jsonl \
        --probe experiments/point_probe_offcentre/ckpt/box05/final.pt \
        --vocab-manifests $SRC/train_f8.jsonl $SRC/holdout_f8.jsonl data/vsibench_test_posed_f8.jsonl data/vsibench_debiased_posed_f8.jsonl \
        --manifest "$raw" --output "$EXP/data/vsibench_${split%%:*}_numeric_posed_f8.jsonl"
    rm "$raw"
done
echo "[done] $(date)"
