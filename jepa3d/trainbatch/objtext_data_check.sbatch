#!/bin/bash
#SBATCH --job-name=objtext_check
#SBATCH --partition=geo
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/objtext_data_check/logs/check_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/objtext_data_check/logs/check_%j.err

# objtext_data_check: which object-level text on disk can supervise
# "region <-> text" (jepa3d/tools/check_objtext_data.py). CPU only, read-only
# on source_data. Writes experiments/objtext_data_check/results/summary.json.
# Create the log folder once before the first submission:
#   mkdir -p /glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/objtext_data_check/logs

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

cd "$ROOT"
srun singularity exec --bind /glob/g01-cache:/glob/g01-cache "$CONTAINER" \
    python3 jepa3d/tools/check_objtext_data.py \
    --output experiments/objtext_data_check/results/summary.json
echo "[done] $(date)"
