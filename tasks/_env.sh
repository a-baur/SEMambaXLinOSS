# Shared runtime environment for the tasks/train_*.sh sbatch scripts.
#
# Sourced after the #SBATCH headers so the per-model scripts differ only in
# job-name, config and exp_name. Keeping module/venv/proxy/PYTHONPATH setup in
# one place avoids the drift that let cpus-per-task diverge across scripts.

unset SLURM_EXPORT_ENV

module load python cuda/12.8.1

export HTTP_PROXY=http://proxy:80    # needed for wandb
export HTTPS_PROXY=http://proxy:80

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/hpc/f102ac/f102ac13/dev/SEMambaXLinOSS:$PYTHONPATH"

# One intra-op thread per process. With num_gpus DDP ranks each spawning
# num_workers dataloaders, the PyTorch default (all cores per process) heavily
# oversubscribes the allocation; pin it to 1 and size CPUs via --cpus-per-task
# (>= num_gpus * num_workers) instead.
export OMP_NUM_THREADS=1
