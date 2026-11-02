#!/bin/bash
#SBATCH --job-name=blind_check
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_mix/blind_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/sft_3drae_mix/blind_%j.err

# Real vs shuffled vs zero visual tokens on one checkpoint (tools/blind_check.py).
# CONFIG / CHECKPOINT override the defaults below.

set -euo pipefail
N=${NUM_VIEWS:-8}
cd /glob/g01-cache/pf/Yushuo/3DRAE_JEPA
CONFIG=${CONFIG:-jepa3d/configs/sft_3drae_mix_f8_long.yaml}
OUT=$(sed -n 's|^ *output_dir: ${RAE_ROOT}/||p' "$CONFIG")
CHECKPOINT=${CHECKPOINT:-$OUT/final.pt}

ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

cd "$ROOT"
srun singularity exec \
    --nv \
    --env-file $VJEPA/.env/hf.env \
    --env VJEPA_ROOT=$VJEPA \
    -H $VJEPA:/home/go98qit \
    --bind /glob/g01-cache:/glob/g01-cache \
    $CONTAINER python3 jepa3d/tools/blind_check.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --val "data/spar_video_val_f${N}.jsonl" \
    --eval "data/vsibench_test_posed_f${N}.jsonl" "data/vsibench_debiased_posed_f${N}.jsonl" \
    --output "$OUT/blind_check_$(basename "$CHECKPOINT" .pt).json"
