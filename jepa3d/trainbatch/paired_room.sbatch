#!/bin/bash
#SBATCH --job-name=paired_room
#SBATCH --partition=geo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00

set -euo pipefail
repo_root=${RAE_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repository root}}
vjepa_root=${VJEPA_ROOT:-/glob/g01-cache/pf/Yushuo/vjepa201}
container=${VJEPA_CONTAINER:-/glob/g01-cache/pf/Yushuo/container/vjepa21v2.sif}
run_dir=${PAIRED_OUT:-$repo_root/experiments/paired_room/run_$SLURM_JOB_ID}
args=(run --out "$run_dir"
      --baseline-revision "${HARDNEG_RAE_REV:?set to the job 12610 3DRAE_JEPA commit}"
      --sibling-revision "${HARDNEG_VJEPA_REV:?set to the job 12610 sibling commit}")
if [[ -n ${ROOM_ALIASES:-} ]]; then
    args+=(--aliases "$ROOM_ALIASES")
fi
cd "$repo_root"
srun --ntasks=1 singularity exec --nv --bind /glob/g01-cache:/glob/g01-cache \
    --env "VJEPA_ROOT=$vjepa_root" --env "RAE_ROOT=$repo_root" --env HF_HUB_OFFLINE=1 \
    --env MLFLOW_ALLOW_FILE_STORE=true --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$container" python3 -m jepa3d.tools.paired_room "${args[@]}"
