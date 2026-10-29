#!/bin/bash
#SBATCH --job-name=sft_3drae_mix
#SBATCH --partition=geo
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_mix/%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_mix/%j.err

# 3DRAE X-encoder SFT on VSI-590K + strict SPAR, N posed views (3 GPUs), then
# VSI-Bench eval (full + debiased, MC rows of ScanNet / ScanNet++).
# Prereqs: make_mix_manifests.sbatch done, overfit_3drae_mix.sbatch passed
# (it also builds the VSI-Bench manifests). Queue after the overfit job:
#   sbatch --dependency=afterok:<overfit job> jepa3d/trainbatch/sft_3drae_mix.sbatch
# Re-run eval only: RUN_TRAIN=0 sbatch jepa3d/trainbatch/sft_3drae_mix.sbatch
# Resume is not supported by the trainer; a rerun starts from scratch.

set -euo pipefail
RUN_TRAIN=${RUN_TRAIN:-1}
NUM_GPUS=${NUM_GPUS:-3}
N=${NUM_VIEWS:-8}

ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_3drae_mix"
CONFIG="jepa3d/configs/sft_3drae_mix_f${N}.yaml"
OUT="outputs/sft_3drae_mix_f${N}"

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

for split in test debiased; do
    echo "[infer-vsibench-$split] start $(date)"
    $SIF python3 jepa3d/infer_spar.py --config "$CONFIG" \
        --input "data/vsibench_${split}_posed_f${N}.jsonl" --output-dir "$OUT/infer_vsibench_${split}"
done
echo "[done] $(date)"
