#!/bin/bash
#SBATCH --job-name=epigen
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

EXPERIMENT=${1:?Usage: sbatch train.slurm <experiment_dir> <config_name> [hydra overrides...]}
CONFIG=${2:?Usage: sbatch train.slurm <experiment_dir> <config_name> [hydra overrides...]}
shift 2
PORT=$((29500 + SLURM_JOB_ID % 10000))

export PYTHONUNBUFFERED=1

torchrun --nproc_per_node=2 --master_port "$PORT" -m datagen.train \
  --config-path "../configs/$EXPERIMENT" --config-name "$CONFIG" "$@"
