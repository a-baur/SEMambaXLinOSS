#!/usr/bin/env python
"""Standalone debug harness for the training-time validation.

Validation runs on rank 0 only (see ``train.validate``): rank 0 scores the full
validation set while the other ranks wait at a ``dist.barrier()``. So this harness is
single-process -- it builds the generator + an ``Evaluator`` + the full validation
loader and calls the *real* ``train.validate()`` on one GPU. The point is to verify,
without launching a full training run, that:

  * validation completes without erroring, and
  * it is fast enough that the other ranks' barrier wait does not exceed the NCCL
    watchdog / process-group timeout while rank 0 validates.

It reports the utterance count / wall time / throughput / peak GPU memory and the
averaged PESQ+UTMOS, plus a verdict against the configured timeout budget.

Run from the repo root (so imports resolve like train.py):

    python debug_validation.py --config recipes/latest.yaml
    # quick smoke test on a 60-utterance subset, two passes (watch the 2nd: the
    # first pays one-off CUDA/Triton kernel compilation):
    python debug_validation.py --config recipes/latest.yaml --max-utts 60 --repeats 2
    # score with real weights so the PESQ/UTMOS numbers are meaningful:
    python debug_validation.py --config recipes/latest.yaml --checkpoint exp/MyRun

Notes
-----
* ``--timeout`` (default 300 s) is the wall-time budget the validation must stay
  under -- it stands in for the production process-group timeout (3600 s via
  ``dist_cfg.timeout_seconds``) / NCCL watchdog that bounds the other ranks' wait.
"""

import argparse
import os
import tempfile
import time

import torch
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

# Reuse the REAL code paths so this tests what training actually runs.
import train as T
from models.generator import SEMamba
from utils.metrics import Evaluator
from utils.util import initialize_seed, load_config, scan_checkpoint


def _fmt_mb(x):
    return f"{x / 1024:.2f} GB" if x >= 1024 else f"{x:.0f} MB"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="recipes/latest.yaml")
    p.add_argument("--checkpoint", default=None, help="exp dir or g_*.pth for real weights")
    p.add_argument("--max-utts", type=int, default=0, help="cap the val set for a quick smoke test (0 = full)")
    p.add_argument("--repeats", type=int, default=2, help="validation passes; watch pass 2 (warm)")
    p.add_argument("--timeout", type=int, default=300, help="wall-time budget in s the validation must stay under")
    args = p.parse_args()

    if torch.cuda.device_count() < 1:
        raise SystemExit("Need at least 1 GPU.")

    cfg = load_config(args.config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    initialize_seed(cfg["env_setting"]["seed"])

    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress_factor = cfg["model_cfg"]["compress_factor"]
    stft_params = (n_fft, hop_size, win_size, compress_factor)

    # --- model (plain, unwrapped -- validate() only wraps to unwrap DDP) -----------
    generator = SEMamba(cfg).to(device)
    if args.checkpoint:
        # Accept either an exp dir (pick the latest g_*.pth) or a direct .pth path.
        path = args.checkpoint
        if os.path.isdir(path):
            path = scan_checkpoint(path, "g_")
            if path is None:
                raise SystemExit(f"No g_*.pth checkpoint found under {args.checkpoint}")
        sd = torch.load(path, map_location=device)
        generator.load_state_dict(sd.get("generator", sd), strict=False)
        print(f"loaded generator weights from {path}")

    # --- evaluator. Validation scores a whole val_batch_size batch of fixed-length
    # crops at once, so PESQ forks a pool across the batch (mirrors train.py) and
    # UTMOS overlaps on the GPU. -----------------------------------------------------
    val_batch_size = cfg["training_cfg"].get("val_batch_size", 1)
    pesq_n_processes = max(1, min(val_batch_size, os.cpu_count() or 1))
    evaluator = Evaluator(
        sr=cfg["stft_cfg"]["sampling_rate"], pesq_n_processes=pesq_n_processes
    ).to(device)

    # --- validation loader (the REAL create_dataloader, full set) ------------------
    validset = T.create_dataset(cfg, train=False, split=False, device=device)
    full_len = len(validset)
    if args.max_utts and args.max_utts < full_len:
        validset = Subset(validset, list(range(args.max_utts)))
    loader = T.create_dataloader(validset, cfg, train=False)

    print(
        f"\n=== validation debug ===\n"
        f"  config            : {args.config}\n"
        f"  mixer (ssm)       : {cfg['model_cfg'].get('ssm')}\n"
        f"  val utterances    : {len(validset)} of {full_len}"
        f"{' (subset)' if len(validset) < full_len else ''}\n"
        f"  timeout budget    : {args.timeout} s   "
        f"(prod default 3600 s via dist_cfg.timeout_seconds)\n"
        f"  repeats           : {args.repeats}\n",
        flush=True,
    )

    # validate() writes scalars / example spectrograms via the SummaryWriter (rank 0
    # always owns one in training), so give it a throwaway logdir here.
    best = (0.0, 0, 0.0, 0)
    with tempfile.TemporaryDirectory() as logdir:
        sw = SummaryWriter(logdir)
        for r in range(args.repeats):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            t0 = time.time()

            # This is the exact function training calls.
            best = T.validate(
                generator, evaluator, loader, cfg, device, sw, steps=0,
                stft_params=stft_params, best=best,
            )

            torch.cuda.synchronize(device)
            elapsed = time.time() - t0
            peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

            n_utts = len(validset)
            ups = n_utts / elapsed if elapsed > 0 else 0.0
            tag = "warm" if r > 0 else "cold (incl. kernel compile)"
            print(f"--- pass {r + 1}/{args.repeats}  [{tag}] ---")
            print(
                f"  {n_utts} utts | {elapsed:.2f} s | {ups:.2f} utts/s | "
                f"peak {_fmt_mb(peak_mb)} | PESQ {best[0]:.4f} (@{best[1]}) | "
                f"UTMOS {best[2]:.4f}"
            )
            # Verdict: at the real interval this runs while ranks>0 wait on the
            # barrier; it must finish well under the PG timeout.
            verdict = "OK" if elapsed < 0.8 * args.timeout else "SLOW -> would risk timeout"
            print(f"  verdict: wall {elapsed:.1f}s vs timeout {args.timeout}s  ->  {verdict}\n", flush=True)
        sw.close()


if __name__ == "__main__":
    main()
