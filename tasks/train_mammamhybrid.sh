#!/bin/bash -l
#
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --partition=a100
#SBATCH --job-name=mammamhybrid
#SBATCH --output=jobs/%x-%j.out
#SBATCH --error=jobs/%x-%j.err
#SBATCH --time=23:59:00
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --open-mode=append
#SBATCH --export=NONE
#SBATCH --gres=gpu:a100:2
#SBATCH -C a100_80

source tasks/_env.sh
source tasks/_requeue.sh

run_with_requeue srun --unbuffered --ntasks=1 python -u train.py \
  --config recipes/hybrid/MamMamParallel.yaml \
  --exp_folder /home/atuin/f102ac/f102ac13/SEMambaBackbones/ \
  --exp_name MambaMambaHybrid
