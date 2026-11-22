#!/bin/bash
#SBATCH --job-name=ground_show
#SBATCH --partition=geo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/ground_v1/logs/%x_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/experiments/ground_v1/logs/%x_%j.err

# ground_v1 pictures (jepa3d/tools/show_ground.py) -> experiments/ground_v1/results/figs/
#   PART=labels  (default) holdout rows with their true centres drawn into the views; no GPU needed:
#       sbatch --gres=gpu:0 jepa3d/trainbatch/ground_show.sbatch
#   PART=pred    the same rows with the trained model's centres, and predictions.json for all holdout names
#       sbatch --export=ALL,PART=pred jepa3d/trainbatch/ground_show.sbatch

set -euo pipefail
ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
VJEPA="/glob/g01-cache/pf/Yushuo/vjepa201"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"
NAME=${NAME:-ground_v1}
EXP="experiments/$NAME"
PART=${PART:-labels}

cd "$ROOT"
RUN="srun singularity exec --nv --env-file $VJEPA/.env/hf.env --env VJEPA_ROOT=$VJEPA -H $VJEPA:/home/go98qit --bind /glob/g01-cache:/glob/g01-cache $CONTAINER python3"
COMMON="jepa3d/tools/show_ground.py --config jepa3d/configs/$NAME.yaml --manifest $EXP/data/holdout_f8.jsonl --output-dir $EXP/results/figs"
case "$PART" in
labels) $RUN $COMMON ;;
pred)   $RUN $COMMON --checkpoint "${CHECKPOINT:-$EXP/ckpt/final.pt}" ;;
*)      echo "unknown PART=$PART" >&2; exit 1 ;;
esac
echo "[done] $(date)"
