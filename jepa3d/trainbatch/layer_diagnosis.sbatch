#!/bin/bash
#SBATCH --job-name=layer_diagnosis
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/layer_diagnosis/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/layer_diagnosis/logs/%x_%j.err

# Layer-by-layer diagnosis of trained predictors (jepa3d/tools/diagnose_layers.py):
# does the question interact with the scene tokens, where does a name's position
# live. No training. Both models read the slots_v1 holdout manifest (the same
# rows as hardneg_v1's, plus the slots that mark the object names).
#   RUNS="slots_v1 hardneg_v1" (default; a run reads its own holdout manifest if it has one)   BATCHES=64 (of 16 rows)
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/layer_diagnosis/logs
#   sbatch jepa3d/trainbatch/layer_diagnosis.sbatch

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/layer_diagnosis"

cd "$ROOT"
mkdir -p "$EXP/results"
for run in ${RUNS:-slots_v1 hardneg_v1}; do
    echo "[$run] start $(date)"
    srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit \
        --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3 jepa3d/tools/diagnose_layers.py \
        --config "jepa3d/configs/$run.yaml" --checkpoint "experiments/$run/ckpt/final.pt" \
        --manifest "$( [ -f experiments/$run/data/holdout_f8.jsonl ] && echo experiments/$run || echo experiments/slots_v1 )/data/holdout_f8.jsonl" --batches "${BATCHES:-64}" \
        --output "$EXP/results/$run.json" > "$EXP/logs/$run.txt"
done
echo "[done] $(date)"
