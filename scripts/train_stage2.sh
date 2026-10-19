#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG="${1:-configs/training/dino2_stage2_scannet.yaml}"
OUTPUT_DIR="${2:-outputs/stage2}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$NPROC_PER_NODE" \
  scripts/train_stage2.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  "${@:3}"
