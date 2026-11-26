#!/bin/bash
#SBATCH --job-name=slots_v1
#SBATCH --partition=geo
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slots_v1/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/slots_v1/logs/%x_%j.err

# slots_v1 (jepa3d/configs/slots_v1.yaml). Prereq: slots_data.sbatch done.
#
#   MODE=overfit  one frozen batch for OVERFIT_STEPS steps on 1 GPU; both
#                 infonce_loss (ckpt_overfit/train_log.jsonl) and hard_loss
#                 (ckpt_overfit/hardneg_log.jsonl) should go to ~0. Then the eval
#                 script on 64 rows (plumbing only); the checkpoint is deleted.
#       sbatch --gres=gpu:1 --cpus-per-task=10 --mem=96G --job-name=slots_overfit \
#           --export=ALL,MODE=overfit jepa3d/trainbatch/slots_train.sbatch
#                 TRAIN_MANIFEST=<jsonl> takes the batch from another manifest
#                 (e.g. the holdout, before the training manifest exists).
#   MODE=train    (default) 3-GPU training, then eval of ckpt/final.pt
#       sbatch jepa3d/trainbatch/slots_train.sbatch
#   MODE=eval     eval only
#       sbatch --gres=gpu:1 --cpus-per-task=10 --mem=96G --job-name=slots_eval \
#           --export=ALL,MODE=eval jepa3d/trainbatch/slots_train.sbatch
#   MODE=baseline objptr_v1: the model as built, no weights loaded -> results/eval_untrained.json
#                 (how often dino.txt's own choice of cell is on the object)
#       sbatch --gres=gpu:1 --cpus-per-task=10 --mem=96G --job-name=objptr_baseline \
#           --export=ALL,MODE=baseline,NAME=objptr_v1 jepa3d/trainbatch/slots_train.sbatch
#   MODE=diagnose do the trained slot vectors depend on the positions, does the
#                 prediction depend on them (jepa3d/tools/diagnose_slot_encoder.py)
#       sbatch --gres=gpu:1 --cpus-per-task=10 --mem=96G --job-name=slots_diagnose \
#           --export=ALL,MODE=diagnose jepa3d/trainbatch/slots_train.sbatch
#
# Eval (jepa3d/tools/eval_holdout.py): holdout multiple choice in long and
# short answer format plus VSI-Bench test / debiased, each with real, shuffled
# and zero scene tokens, with swapped slots, without slots and with nothing of
# the row's own images -> experiments/slots_v1/results/eval_<ckpt>.json
# Resume is not supported by the trainer; a rerun starts from scratch.

set -euo pipefail
MODE=${MODE:-train}
NUM_GPUS=${NUM_GPUS:-3}
OVERFIT_STEPS=${OVERFIT_STEPS:-200}

ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
NAME=${NAME:-slots_v1}   # slots_v1 | slots_v2 (give sbatch --output and --error in its logs folder)
EXP="experiments/$NAME"
CONFIG="jepa3d/configs/$NAME.yaml"
DATA=${DATA:-$EXP/data}   # manifests with slots; an experiment that reuses another one's passes DATA=experiments/<other>/data
# DELEX_EVAL=1: train mode evaluates a second time with all rows delexicalized (jepa3d/delex.py),
#   eval mode only that way -> results/eval_<ckpt>_delex.json
# MODES="<mode> ...": only these visual inputs / controls; SUFFIX=<text>: appended to the eval file's name
#   (MODE=eval of an old checkpoint with a new control must not overwrite its eval_final.json)
# NUMERIC="<jsonl> ..." WEIGHTS=<json of jepa3d/balance.py>: numeric questions for models with a numeric head

cd "$ROOT"
mkdir -p "$EXP/logs" "$EXP/results"
if [ ! -f "$DATA/train_f8.jsonl" ] && [ "$MODE" = train ]; then   # shards of slots_data.sbatch -> one manifest
    cat "$DATA"/train_f8.jsonl.part[0-9] > "$DATA/train_f8.jsonl.tmp"
    mv "$DATA/train_f8.jsonl.tmp" "$DATA/train_f8.jsonl"
    rm "$DATA"/train_f8.jsonl.part[0-9]
    wc -l "$DATA/train_f8.jsonl"
fi
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
        --holdout "$DATA/holdout_f8.jsonl" \
        --eval "$DATA/vsibench_test_posed_f8.jsonl" "$DATA/vsibench_debiased_posed_f8.jsonl" \
        ${NUMERIC:+--numeric $NUMERIC} ${WEIGHTS:+--answer-weights "$WEIGHTS"} ${MODES:+--modes $MODES} \
        --output "$out" "$@"
}

case "$MODE" in
overfit)
    $SIF accelerate launch --num_processes 1 jepa3d/train.py --config "$CONFIG" \
        --overfit --overfit-steps "$OVERFIT_STEPS" --output-dir "$EXP/ckpt_overfit" \
        ${TRAIN_MANIFEST:+--train-manifest "$TRAIN_MANIFEST"}
    run_eval "$EXP/ckpt_overfit/final.pt" "$EXP/results/eval_overfit_plumbing.json" --limit 64
    rm "$EXP/ckpt_overfit/final.pt"
    tail -n 3 "$EXP/ckpt_overfit/train_log.jsonl" "$EXP/ckpt_overfit/hardneg_log.jsonl"
    ;;
train)
    echo "[train] start $(date)"
    $SIF accelerate launch --multi_gpu --num_processes "$NUM_GPUS" jepa3d/train.py --config "$CONFIG"
    echo "[train] done $(date)"
    run_eval "$EXP/ckpt/final.pt" "$EXP/results/eval_final.json"
    if [ -n "${DELEX_EVAL:-}" ]; then
        run_eval "$EXP/ckpt/final.pt" "$EXP/results/eval_final_delex.json" --delexicalize
    fi
    ;;
baseline)   # object pointers before any training: where does the untrained choice land (1 GPU, holdout only)
    $SIF python3 jepa3d/tools/eval_holdout.py --config "$CONFIG" --checkpoint none --modes real shuffled \
        --holdout "$DATA/holdout_f8.jsonl" --output "$EXP/results/eval_untrained.json"
    ;;
eval)
    CHECKPOINT=${CHECKPOINT:-$EXP/ckpt/final.pt}
    if [ -n "${DELEX_EVAL:-}" ]; then
        run_eval "$CHECKPOINT" "$EXP/results/eval_$(basename "$CHECKPOINT" .pt)_delex${SUFFIX:-}.json" --delexicalize
    else
        run_eval "$CHECKPOINT" "$EXP/results/eval_$(basename "$CHECKPOINT" .pt)${SUFFIX:-}.json"
    fi
    ;;
diagnose)
    CHECKPOINT=${CHECKPOINT:-$EXP/ckpt/final.pt}
    $SIF python3 jepa3d/tools/diagnose_slot_encoder.py --config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --manifest "$EXP/data/holdout_f8.jsonl" --output "$EXP/results/diagnose_slot_encoder.json"
    ;;
*)
    echo "unknown MODE=$MODE" >&2
    exit 1
    ;;
esac
echo "[done] $(date)"
