#!/bin/bash
#SBATCH --job-name=overfit_3drae_mix
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_mix/overfit_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_mix/overfit_%j.err

# Before sft_3drae_mix.sbatch: builds the posed VSI-Bench eval manifests
# (raw MC questions / options, letter prefix removed), then one frozen batch
# of the f8 mix repeated for OVERFIT_STEPS steps on 1 GPU (loss should go to
# ~0; also an OOM check for 8 views), then the eval script on 256 VSI-Bench
# rows against the overfit checkpoint (accuracy meaningless, plumbing only).
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_mix

set -euo pipefail
OVERFIT_STEPS=${OVERFIT_STEPS:-200}
N=${NUM_VIEWS:-8}

ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
LOG_DIR="$ROOT/logs/sft_3drae_mix"
CONFIG="jepa3d/configs/overfit_3drae_mix_f${N}.yaml"
VSIB="$VJEPA/source_data/vsibench"

cd "$ROOT"
mkdir -p "$LOG_DIR"
nvidia-smi dmon -s pucvmet -d 30 > "$LOG_DIR/gpu_monitor_overfit_${SLURM_JOB_ID}.log" 2>&1 &

CPU="srun singularity exec --bind /glob/g01-cache:/glob/g01-cache --env VJEPA_ROOT=$VJEPA $CONTAINER"
SIF="srun singularity exec \
    --nv \
    --env-file $VJEPA/.env/hf.env \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env VJEPA_ROOT=$VJEPA \
    -H $VJEPA:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER"

echo "[vsibench manifests f$N] start $(date)"
for split in test:test.jsonl debiased:test_debiased.parquet; do
    $CPU python3 jepa3d/tools/make_vsibench_manifest.py --input "$VSIB/${split#*:}" --num-views "$N" \
        --stats data/frame_stats/scannet.jsonl data/frame_stats/scannetpp.jsonl \
        --output "data/vsibench_${split%%:*}_posed_f${N}.jsonl"
done

echo "[overfit-train] start $(date)"
$SIF accelerate launch --num_processes 1 jepa3d/train.py \
    --config "$CONFIG" --overfit --overfit-steps "$OVERFIT_STEPS"
echo "[overfit-train] done $(date)"

echo "[overfit-infer] start $(date)"
$SIF python3 jepa3d/infer_spar.py \
    --config "$CONFIG" \
    --input "data/vsibench_test_posed_f${N}.jsonl" \
    --output-dir "outputs/overfit_3drae_mix_f${N}/infer_vsibench_head" \
    --limit 256
echo "[overfit-infer] done $(date)"
