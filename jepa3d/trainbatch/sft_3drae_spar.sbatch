#!/bin/bash
#SBATCH --job-name=sft_3drae_spar
#SBATCH --partition=geo
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=7-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_spar/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_spar/%j.err

# Variant A: frozen 3DRAE X-encoder, SFT on posed SPAR (3 GPUs), then the
# 9k SPAR eval and the same-subset re-score of the old vjepa / fusion_gv runs.
# Prereqs: make_spar_manifests.sbatch done, overfit_3drae_spar.sbatch passed.
# Re-run eval only: RUN_TRAIN=0 sbatch jepa3d/trainbatch/sft_3drae_spar.sbatch
# Resume is not supported by the trainer; a rerun starts from scratch.

set -euo pipefail
RUN_TRAIN=${RUN_TRAIN:-1}
NUM_GPUS=${NUM_GPUS:-3}

ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_3drae_spar"
CONFIG="jepa3d/configs/sft_3drae_spar.yaml"
EVAL_DIR="outputs/sft_3drae_spar/infer_eval9k"

cd "$ROOT"
mkdir -p "$LOG_DIR"
MONITOR_LOG="$LOG_DIR/gpu_monitor_${SLURM_JOB_ID}.log"
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

if [ "$RUN_TRAIN" = "1" ]; then
    echo "[train] start $(date)"
    $SIF accelerate launch --multi_gpu --num_processes "$NUM_GPUS" jepa3d/train.py --config "$CONFIG"
    echo "[train] done $(date)"
fi

echo "[infer-eval9k] start $(date)"
$SIF python3 jepa3d/infer_spar.py --config "$CONFIG" \
    --input data/spar3d_eval9k.jsonl --output-dir "$EVAL_DIR"
echo "[infer-eval9k] done $(date)"

for baseline in vjepa fusiongv; do
    $SIF python3 jepa3d/tools/rescore_baseline.py --posed data/spar3d_eval9k.jsonl \
        --predictions "$VJEPA/outputs/inference/spar_eval_old_${baseline}_sample9k/predictions.jsonl" \
        --output "$EVAL_DIR/baseline_${baseline}_metrics.json"
done
