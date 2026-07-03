# Sourced by the tasks/train_*.sh sbatch scripts to enable auto-requeue.
#
# SLURM caps jobs at 24 h. `run_with_requeue <cmd...>` runs the training command
# under a SIGUSR1 handler that requeues the job just before the wall-clock limit.
# The requeued job keeps the same JobID and re-runs the same script, so train.py
# resumes from the latest checkpoint (model + optimizer + wandb run).
#
# Pair it with these directives in the calling sbatch script:
#   #SBATCH --requeue                 # allow the job to go back in the queue
#   #SBATCH --signal=B:SIGUSR1@120    # signal the batch shell 120 s before the limit
#   #SBATCH --open-mode=append        # don't truncate the .out/.err on restart
#
# Note: SLURM delivers the signal only to this batch shell (the B: prefix). bash
# runs traps only between commands, so the workload must run in the background
# with `wait` for the trap to fire promptly -- run_with_requeue handles that.
#
# Guard against a job that fails on startup and would otherwise requeue forever:
# after MAX_RESTARTS requeues the final allocation runs without the trap, so it
# either finishes training or exits for good. Override the cap per-script by
# setting MAX_RESTARTS before calling run_with_requeue.

: "${MAX_RESTARTS:=10}"

_requeue_handler() {
    echo "[requeue] caught SIGUSR1 at $(date -Is); requeuing job ${SLURM_JOB_ID}" \
         "(restart #${SLURM_RESTART_COUNT:-0})"
    scontrol requeue "${SLURM_JOB_ID}"
}

run_with_requeue() {
    if [ "${SLURM_RESTART_COUNT:-0}" -ge "${MAX_RESTARTS}" ]; then
        echo "[requeue] restart #${SLURM_RESTART_COUNT:-0} >= MAX_RESTARTS=${MAX_RESTARTS};" \
             "running without auto-requeue (final allocation)."
    else
        trap _requeue_handler SIGUSR1
    fi
    "$@" &
    wait "$!"
}
