#!/bin/bash -l
#
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --partition=a100
#SBATCH --job-name=multi_stem
#SBATCH --output=jobs/%x-%j.out
#SBATCH --error=jobs/%x-%j.err
#SBATCH --time=23:59:00
#SBATCH --export=NONE
#SBATCH --gres=gpu:a100:2
#SBATCH -C a100_80


unset SLURM_EXPORT_ENV

module load python cuda/12.8.1
export HTTP_PROXY=http://proxy:80   # needed for wandb
export HTTPS_PROXY=http://proxy:80

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/hpc/f102ac/f102ac13/dev/SEMambaXLinOSS:$PYTHONPATH"

srun --unbuffered --ntasks=1 python -u train.py \
  --config recipes/non_selective/LinOSS.yaml \
  --exp_folder /home/atuin/f102ac/f102ac13/SEMambaBackbones/ \
  --exp_name LinOSS
