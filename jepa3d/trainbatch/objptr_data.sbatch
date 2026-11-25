#!/bin/bash
#SBATCH --job-name=objptr_data
#SBATCH --partition=geo
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/objptr_v1/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/objptr_v1/logs/%x_%j.err

# objptr_v1 data (jepa3d/tools/add_visibility.py): for every labelled object name of
# ground_v1's manifests, the views of the row that show the object's centre, with a
# depth-map test for whether it is hidden there. CPU only.
# Reads experiments/ground_v1/data and the depth maps under source_data, writes only
# experiments/objptr_v1/data/: train_f8.jsonl  holdout_f8.jsonl  visibility.stats.json
# and links scene_objects.json and the VSI-Bench manifests there.
#   LIMIT=<n>  debug: the first n rows of each manifest, prints the tally, writes nothing
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/objptr_v1/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
SRC="experiments/ground_v1/data"
OUT="experiments/objptr_v1/data"

cd "$ROOT"
mkdir -p "$OUT"
srun singularity exec --env VJEPA_ROOT=$VJEPA --bind /glob/g01-cache:/glob/g01-cache "$CONTAINER" \
    python3 jepa3d/tools/add_visibility.py --objects "$SRC/scene_objects.json" \
    --manifests "$SRC/holdout_f8.jsonl" "$SRC/train_f8.jsonl" \
    --output-dir "$OUT" --workers 26 ${LIMIT:+--limit "$LIMIT"}
if [ -z "${LIMIT:-}" ]; then
    ln -sf "$ROOT/$SRC/scene_objects.json" "$OUT/scene_objects.json"
    for split in test debiased; do
        ln -sf "$ROOT/experiments/slots_v2/data/vsibench_${split}_posed_f8.jsonl" "$OUT/vsibench_${split}_posed_f8.jsonl"
    done
fi
echo "[done] $(date)"
