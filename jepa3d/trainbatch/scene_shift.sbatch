#!/bin/bash
#SBATCH --job-name=scene_shift
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=02:00:00

# What another scene changes in the prediction (jepa3d/tools/scene_shift_check.py), for the checkpoints
# of RUNS (default: scneg_v1 and its base objptr_v1_l2) -> experiments/<run>/results/scene_shift.json
# Pass --output/--error in experiments/scneg_v1/logs.

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
DATA=${DATA:-experiments/objptr_v1/data}
cd "$ROOT"
for run in ${RUNS:-scneg_v1 objptr_v1_l2}; do
    srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit \
        --bind /glob/g01-cache:/glob/g01-cache "$CONTAINER" \
        python3 jepa3d/tools/scene_shift_check.py --config "jepa3d/configs/$run.yaml" \
        --checkpoint "experiments/$run/ckpt/final.pt" \
        --manifests "$DATA/holdout_f8.jsonl" "$DATA/vsibench_test_posed_f8.jsonl" \
        --output "experiments/$run/results/scene_shift.json" ${LIMIT:+--limit "$LIMIT"}
done
echo "[done] $(date)"
