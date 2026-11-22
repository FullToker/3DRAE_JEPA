#!/bin/bash
#SBATCH --job-name=label_count
#SBATCH --partition=geo
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/grounding_label_count/logs/count_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/grounding_label_count/logs/count_%j.err

# grounding_label_count: how many training rows can get a true 3D position
# (EmbodiedScan box centre) for the objects their question names
# (jepa3d/tools/count_box_labels.py). CPU only, read-only on the manifests and
# source_data. Writes experiments/grounding_label_count/results/summary.json.
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/grounding_label_count/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
DATA="${DATA:-experiments/slots_v2/data}"

cd "$ROOT"
srun singularity exec --bind /glob/g01-cache:/glob/g01-cache "$CONTAINER" \
    python3 jepa3d/tools/count_box_labels.py \
    --manifests "$DATA/train_f8.jsonl" "$DATA/holdout_f8.jsonl" \
    --output experiments/grounding_label_count/results/summary.json
echo "[done] $(date)"
