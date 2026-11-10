#!/bin/bash
#SBATCH --job-name=hardneg_v1
#SBATCH --partition=geo
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/hardneg_v1/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/hardneg_v1/logs/%x_%j.err

# hardneg_v1 (jepa3d/configs/hardneg_v1.yaml). Prereq: hardneg_data.sbatch done.
#
#   MODE=overfit  one frozen batch for OVERFIT_STEPS steps on 1 GPU; both
#                 infonce_loss (ckpt_overfit/train_log.jsonl) and hard_loss
#                 (ckpt_overfit/hardneg_log.jsonl) should go to ~0. Then the eval
#                 script on 64 rows (plumbing only); the checkpoint is deleted.
#       sbatch --gres=gpu:1 --cpus-per-task=10 --mem=96G --job-name=hardneg_overfit \
#           --export=ALL,MODE=overfit jepa3d/trainbatch/hardneg_train.sbatch
#   MODE=train    (default) 3-GPU training, then eval of ckpt/final.pt
#       sbatch jepa3d/trainbatch/hardneg_train.sbatch
#   MODE=eval     eval only
#       sbatch --gres=gpu:1 --cpus-per-task=10 --mem=96G --job-name=hardneg_eval \
#           --export=ALL,MODE=eval jepa3d/trainbatch/hardneg_train.sbatch
#
# Eval (jepa3d/tools/eval_holdout.py): holdout multiple choice in long and
# short answer format plus VSI-Bench test / debiased, each with real, shuffled
# and zero scene tokens -> experiments/hardneg_v1/results/eval_<ckpt>.json
# Resume is not supported by the trainer; a rerun starts from scratch.

set -euo pipefail
MODE=${MODE:-train}
NUM_GPUS=${NUM_GPUS:-3}
OVERFIT_STEPS=${OVERFIT_STEPS:-200}

ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="experiments/hardneg_v1"
CONFIG="jepa3d/configs/hardneg_v1.yaml"

cd "$ROOT"
mkdir -p "$EXP/logs" "$EXP/results"
MONITOR_LOG="$EXP/logs/gpu_monitor_${SLURM_JOB_ID}.log"
nvidia-smi dmon -s pucvmet -d 30 > "$MONITOR_LOG" 2>&1 &
( while true; do
    printf '[MEM %s] used=%s / total=%s\n' "$(date +%T)" \
        "$(free -h | awk '/Mem:/{print $3}')" "$(free -h | awk '/Mem:/{print $2}')" >> "$MONITOR_LOG"
    sleep 60
done ) &

SIF="srun singularity exec \
    --nv \
    --env-file $VJEPA/.env/hf.env \
    --env MLFLOW_ALLOW_FILE_STORE=true \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env VJEPA_ROOT=$VJEPA \
    -H $VJEPA:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

run_eval() {   # <checkpoint> <output json> [extra eval_holdout.py args]
    local ckpt=$1 out=$2
    shift 2
    echo "[eval $ckpt] start $(date)"
    $SIF python3 jepa3d/tools/eval_holdout.py --config "$CONFIG" --checkpoint "$ckpt" \
        --holdout "$EXP/data/holdout_f8.jsonl" \
        --eval data/vsibench_test_posed_f8.jsonl data/vsibench_debiased_posed_f8.jsonl \
        --output "$out" "$@"
}

case "$MODE" in
overfit)
    $SIF accelerate launch --num_processes 1 jepa3d/train.py --config "$CONFIG" \
        --overfit --overfit-steps "$OVERFIT_STEPS" --output-dir "$EXP/ckpt_overfit"
    run_eval "$EXP/ckpt_overfit/final.pt" "$EXP/results/eval_overfit_plumbing.json" --limit 64
    rm "$EXP/ckpt_overfit/final.pt"
    tail -n 3 "$EXP/ckpt_overfit/train_log.jsonl" "$EXP/ckpt_overfit/hardneg_log.jsonl"
    ;;
train)
    echo "[train] start $(date)"
    $SIF accelerate launch --multi_gpu --num_processes "$NUM_GPUS" jepa3d/train.py --config "$CONFIG"
    echo "[train] done $(date)"
    run_eval "$EXP/ckpt/final.pt" "$EXP/results/eval_final.json"
    ;;
eval)
    CHECKPOINT=${CHECKPOINT:-$EXP/ckpt/final.pt}
    run_eval "$CHECKPOINT" "$EXP/results/eval_$(basename "$CHECKPOINT" .pt).json"
    ;;
*)
    echo "unknown MODE=$MODE" >&2
    exit 1
    ;;
esac
echo "[done] $(date)"
