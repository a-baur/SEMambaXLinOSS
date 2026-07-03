#!/bin/bash -l
#
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --partition=a100
#SBATCH --job-name=mamba3
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

set -euo pipefail   # fail fast on any staging/check error

# stage data on node-local $TMPDIR
# DATASET_NAME="EARS-WHAM_v2_16k"
# TARBALL="/home/atuin/f102ac/shared/${DATASET_NAME}.tar"

# echo "Staging data to $TMPDIR"
#mkdir -p "$TMPDIR/data"
#tar xf "$TARBALL" -C "$TMPDIR/data"

#export DATA_ROOT="$TMPDIR/data/${DATASET_NAME}"
#export DATA_ROOT_ORIG="/home/atuin/f102ac/shared/EARS-WHAM_v2_16k/"
#echo "DATA_ROOT=$DATA_ROOT"
#echo "DATA_ROOT_ORIG=$DATA_ROOT_ORIG"

# Pre-flight: confirm remapped paths actually exist before launching the job.
#python -c "
#import json, os
#from dataloaders.dataloader_vctk import remap_to_data_root
#dr, orig = os.environ['DATA_ROOT'], os.environ['DATA_ROOT_ORIG']
#for split in ('train', 'valid'):
#    c = json.load(open(f'data/ears_wham_16k/{split}_clean.json'))
#    n = json.load(open(f'data/ears_wham_16k/{split}_noisy.json'))
#    for paths in (c, n):
#        p = remap_to_data_root(paths[:1], dr, orig)[0]
#        assert os.path.exists(p), f'remap miss: {p}'
#print('remap OK')
#"

run_with_requeue srun --unbuffered --ntasks=1 python -u train.py \
  --config recipes/selective/Mamba3.yaml \
  --exp_folder /home/atuin/f102ac/f102ac13/SEMambaBackbones/ \
  --exp_name Mamba3
