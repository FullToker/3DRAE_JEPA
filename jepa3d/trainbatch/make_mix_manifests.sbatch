#!/bin/bash
#SBATCH --job-name=mix_manifests
#SBATCH --partition=geo
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/data/mix_%j.out
#SBATCH --error=/glob/g01-cache/pf/Yushuo/3DRAE_JEPA/logs/data/mix_%j.err

# VSI-590K (ScanNet + ScanNet++) + strict SPAR video QA -> posed N-view manifests
# (see jepa3d/docs/train_data_vsi_spar.md). CPU only.
#   data/frame_stats/{scannet,scannetpp}.jsonl   per-frame quality, built once
#   data/vsi590k_sn_f${N}.jsonl
#   data/spar_video_{train,val}_f${N}.jsonl
#   data/mix_vsi_spar_train_f${N}.jsonl (+ .stats.json)
# Other view counts: NUM_VIEWS=16 sbatch jepa3d/trainbatch/make_mix_manifests.sbatch
# Debug run on a few scenes: LIMIT=20 sbatch ...

set -euo pipefail
N=${NUM_VIEWS:-8}
LIMIT=${LIMIT:-0}

ROOT="/glob/g01-cache/pf/Yushuo/3DRAE_JEPA"
CONTAINER="/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif"

cd "$ROOT"
mkdir -p data/frame_stats
SIF="srun singularity exec --bind /glob/g01-cache:/glob/g01-cache $CONTAINER"
SUFFIX=$([ "$LIMIT" = "0" ] && echo "" || echo "_debug")
STATS="data/frame_stats/scannet${SUFFIX}.jsonl data/frame_stats/scannetpp${SUFFIX}.jsonl"

for subset in scannet scannetpp; do
    out="data/frame_stats/${subset}${SUFFIX}.jsonl"
    if [ ! -s "$out" ]; then
        echo "[frame stats $subset] start $(date)"
        $SIF python3 jepa3d/tools/frame_stats.py --subset "$subset" --output "$out" --limit "$LIMIT"
    fi
done

echo "[vsi590k f$N] start $(date)"
$SIF python3 jepa3d/tools/sample_vsi590k.py --num-views "$N" --stats $STATS \
    --output "data/vsi590k_sn_f${N}${SUFFIX}.jsonl" --limit "$LIMIT"

echo "[spar train f$N] start $(date)"
$SIF python3 jepa3d/tools/sample_spar_video.py --split train --num-views "$N" --stats $STATS \
    --output "data/spar_video_train_f${N}${SUFFIX}.jsonl" --limit "$LIMIT"
# SPAR's scannetpp val files are byte copies of its train files -> val = scannet only
echo "[spar val f$N] start $(date)"
$SIF python3 jepa3d/tools/sample_spar_video.py --split val --subsets scannet --num-views "$N" --stats $STATS \
    --output "data/spar_video_val_f${N}${SUFFIX}.jsonl" --limit "$LIMIT"

echo "[mix f$N] start $(date)"
$SIF python3 jepa3d/tools/mix_manifests.py \
    --inputs "data/vsi590k_sn_f${N}${SUFFIX}.jsonl" "data/spar_video_train_f${N}${SUFFIX}.jsonl" \
    --output "data/mix_vsi_spar_train_f${N}${SUFFIX}.jsonl"

wc -l data/*_f${N}${SUFFIX}.jsonl
echo "[done] $(date)"
