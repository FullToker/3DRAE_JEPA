#!/bin/bash
#SBATCH --job-name=slot_vsibench
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slot_vsibench_check/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slot_vsibench_check/logs/%x_%j.err

# slot_vsibench_check: VSI-Bench answers computed from object-slot positions
# (dino.txt patch -> point probe), jepa3d/tools/slots_answer_vsibench.py. Nothing
# is trained. A smoke run on 40 questions comes first.
#   SMOKE=1   only the smoke run
#   PROBE     point probe (default: the centre-prompt probe)
#   TAG       prefix of the result files (default: centre)
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slot_vsibench_check/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/slot_vsibench_check"

cd "$ROOT"
RUN="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
PROBE=${PROBE:-outputs/geo_probe/point_3drae_real/final.pt}
TAG=${TAG:-centre}
COMMON="jepa3d/tools/slots_answer_vsibench.py --train-labels data/geo_probe/labels_train.jsonl --probe $PROBE"
$RUN $COMMON --manifest data/vsibench_test_posed_f8.jsonl --limit 40 --output "$EXP/results/${TAG}_smoke"
if [ "${SMOKE:-0}" != 1 ]; then
    for split in test debiased; do
        $RUN $COMMON --manifest "data/vsibench_${split}_posed_f8.jsonl" --output "$EXP/results/${TAG}_$split"
    done
fi
echo "[done] $(date)"
