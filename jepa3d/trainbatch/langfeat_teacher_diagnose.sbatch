#!/bin/bash
#SBATCH --job-name=teacher_diag
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/langfeat_teacher_check/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/langfeat_teacher_check/logs/%x_%j.err

# langfeat_teacher_check, diagnosis: is the 0.6 m of the teacher check the model
# or the measurement (jepa3d/tools/diagnose_teacher_grounding.py)? No training.
# Scores every object under text half x template x score type and adds the
# "ray passes through the 3D box" measure. A smoke run on 8 scenes comes first.
#   SIZES="224x308"     teacher input sizes, one result set each
#   SMOKE=1             only the smoke run
# Results: experiments/langfeat_teacher_check/results/diagnose_<size>.{json,objects.jsonl,examples.npz}

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/langfeat_teacher_check"

cd "$ROOT"
RUN="srun singularity exec --nv --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
COMMON="jepa3d/tools/diagnose_teacher_grounding.py --stats data/frame_stats/scannet.jsonl
        --labels data/geo_probe/labels_val.jsonl --train-labels data/geo_probe/labels_train.jsonl"
$RUN $COMMON --limit 8 --dump 2 --output "$EXP/results/diagnose_smoke"
if [ "${SMOKE:-0}" != 1 ]; then
    for size in ${SIZES:-224x308}; do
        $RUN $COMMON --size ${size%x*} ${size#*x} --output "$EXP/results/diagnose_${size}"
    done
fi
echo "[done] $(date)"
