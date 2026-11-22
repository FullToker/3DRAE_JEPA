#!/bin/bash
#SBATCH --job-name=ground_data
#SBATCH --partition=geo
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/ground_v1/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/ground_v1/logs/%x_%j.err

# ground_v1 data: true 3D positions for the objects the questions name
# (jepa3d/tools/build_ground_labels.py): EmbodiedScan box centres on ScanNet,
# centres rebuilt from SPAR's own point annotations on ScanNet++. CPU only.
# Reads source_data and experiments/slots_v2/data, writes only experiments/ground_v1/data/:
#   scene_objects.json  train_f8.jsonl  holdout_f8.jsonl  labels.stats.json
# and links the VSI-Bench manifests of slots_v2 there (they get no labels).
#   LIMIT=<n>  debug: the first n scenes, prints the checks, writes nothing
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/ground_v1/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
SRC="experiments/slots_v2/data"
OUT="experiments/ground_v1/data"

cd "$ROOT"
mkdir -p "$OUT"
srun singularity exec --env VJEPA_ROOT=$VJEPA --bind /glob/g01-cache:/glob/g01-cache "$CONTAINER" \
    python3 jepa3d/tools/build_ground_labels.py \
    --manifests "$SRC/train_f8.jsonl" "$SRC/holdout_f8.jsonl" \
    --output-dir "$OUT" --workers 26 ${LIMIT:+--limit-scenes "$LIMIT"}
if [ -z "${LIMIT:-}" ]; then
    for split in test debiased; do
        ln -sf "$ROOT/$SRC/vsibench_${split}_posed_f8.jsonl" "$OUT/vsibench_${split}_posed_f8.jsonl"
    done
fi
echo "[done] $(date)"
