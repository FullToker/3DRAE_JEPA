#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG="${1:-configs/training/dino2_stat_scannet.yaml}"
OUTPUT="${2:-outputs/stage1/stats.safetensors}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$NPROC_PER_NODE" \
  scripts/calculate_stat.py \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  "${@:3}"
