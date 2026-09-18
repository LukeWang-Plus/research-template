#!/usr/bin/env bash
#SBATCH --job-name=research
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=research-%j.log

set -euo pipefail

# Submit from the project root after uv sync and explicit data preparation.
# Set partition/account/module directives for your own cluster.
cd "$SLURM_SUBMIT_DIR"

uv run research train \
    "${1:?Usage: sbatch scripts/train.slurm CONFIG}" \
    --id "slurm-${SLURM_JOB_ID}" \
    --set 'device="cuda:0"'
