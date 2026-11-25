#!/bin/bash
#SBATCH --job-name=e1_oracle
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/e1_oracle/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/e1_oracle/logs/%x_%j.err

# E1 (jepa3d/oracle.py, jepa3d/configs/e1_oracle_{a,b}.yaml): true positions as tokens.
#
# The code runs from its own worktree (branch e1_oracle), so the main checkout
# and the jobs started from it are not touched. Results go to
# experiments/e1_oracle/ of the main checkout. Create the logs folder first:
#     mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/e1_oracle/logs
#
#   MODE=data     manifests (CPU, minutes): check "rule_agreement" in the log before training
#       sbatch --gres=gpu:0 --cpus-per-task=4 --mem=32G --job-name=e1_data \
#           --export=ALL,MODE=data jepa3d/trainbatch/e1_oracle.sbatch
#   MODE=overfit  one frozen batch, hard_loss should go to ~0; then the eval script on 64 rows
#       sbatch --job-name=e1_overfit_b --export=ALL,MODE=overfit,ARM=b jepa3d/trainbatch/e1_oracle.sbatch
#   MODE=train    training, then eval of ckpt/final.pt
#       arm B (positions only, 1 GPU):
#       sbatch --job-name=e1_b --export=ALL,MODE=train,ARM=b jepa3d/trainbatch/e1_oracle.sbatch
#       arm A (positions + 3DRAE scene tokens, 3 GPUs):
#       sbatch --gres=gpu:3 --cpus-per-task=30 --mem=240G --job-name=e1_a \
#           --export=ALL,MODE=train,ARM=a,NUM_GPUS=3 jepa3d/trainbatch/e1_oracle.sbatch
#       arm D (arm B + direction classification head, 1 GPU):
#       sbatch --job-name=e1_d --export=ALL,MODE=train,ARM=d jepa3d/trainbatch/e1_oracle.sbatch
#   MODE=eval     eval only
#   MODE=probe    jepa3d/tools/oracle_probe.py: a small MLP on the positions arm B is handed (minutes)
#       sbatch --job-name=e1_probe --export=ALL,MODE=probe jepa3d/trainbatch/e1_oracle.sbatch
#
# Eval (jepa3d/tools/eval_holdout.py --delexicalize, holdout only: VSI-Bench scenes
# have no object positions): real / shuffled / zero scene tokens, positions of
# another row (slots_swapped), none (slots_none), and each name with another
# name's position of its own question (slots_permuted).

set -euo pipefail
MODE=${MODE:-train}
ARM=${ARM:-b}
NUM_GPUS=${NUM_GPUS:-1}
OVERFIT_STEPS=${OVERFIT_STEPS:-300}

MAIN="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
CODE="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA_e1"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
EXP="$MAIN/experiments/e1_oracle"
DATA="$EXP/data"
RUN="$EXP/$ARM"
CONFIG="jepa3d/configs/e1_oracle_$ARM.yaml"

cd "$CODE"
[ -e checkpoints ] || ln -s "$MAIN/checkpoints" checkpoints   # 3DRAE weights, not tracked
mkdir -p "$EXP/logs" "$DATA" "$RUN/results"

SIF="srun singularity exec \
    --nv \
    --env-file $VJEPA/.env/hf.env \
    --env MLFLOW_ALLOW_FILE_STORE=true \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env VJEPA_ROOT=$VJEPA \
    --env MAIN_ROOT=$MAIN \
    -H $VJEPA:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

run_eval() {   # <checkpoint> <output json> [extra eval_holdout.py args]
    local ckpt=$1 out=$2
    shift 2
    echo "[eval $ckpt] start $(date)"
    $SIF python3 jepa3d/tools/eval_holdout.py --config "$CONFIG" --checkpoint "$ckpt" \
        --holdout "$DATA/holdout_f8.jsonl" --delexicalize --output "$out" "$@"
}

case "$MODE" in
data)
    $SIF python3 jepa3d/tools/make_oracle_manifest.py --data "$MAIN/experiments/ground_v1/data" --out "$DATA"
    wc -l "$DATA"/*.jsonl
    ;;
overfit)
    $SIF accelerate launch --num_processes 1 jepa3d/train.py --config "$CONFIG" \
        --overfit --overfit-steps "$OVERFIT_STEPS" --output-dir "$RUN/ckpt_overfit"
    run_eval "$RUN/ckpt_overfit/final.pt" "$RUN/results/eval_overfit_plumbing.json" --limit 64
    rm "$RUN/ckpt_overfit/final.pt"
    tail -n 3 "$RUN/ckpt_overfit/hardneg_log.jsonl"
    ;;
train)
    echo "[train] start $(date)"
    if [ "$NUM_GPUS" -gt 1 ]; then
        $SIF accelerate launch --multi_gpu --num_processes "$NUM_GPUS" jepa3d/train.py --config "$CONFIG"
    else
        $SIF accelerate launch --num_processes 1 jepa3d/train.py --config "$CONFIG"
    fi
    echo "[train] done $(date)"
    run_eval "$RUN/ckpt/final.pt" "$RUN/results/eval_final.json"
    ;;
eval)
    CHECKPOINT=${CHECKPOINT:-$RUN/ckpt/final.pt}
    run_eval "$CHECKPOINT" "$RUN/results/eval_$(basename "$CHECKPOINT" .pt).json"
    ;;
probe)
    $SIF python3 jepa3d/tools/oracle_probe.py --config "$CONFIG" --output "$EXP/probe/results/probe.json"
    ;;
*)
    echo "unknown MODE=$MODE" >&2
    exit 1
    ;;
esac
echo "[done] $(date)"
