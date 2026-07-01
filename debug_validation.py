#!/usr/bin/env python
"""Standalone debug harness for the sharded training-time validation.

It mirrors ``train.py`` exactly -- spawns one process per GPU (``num_gpus`` from the
config), builds the DDP-wrapped generator + a per-rank ``Evaluator`` + the sharded
validation loader, and calls the *real* ``train.validate()``. The point is to verify,
without launching a full training run, that:

  * validation completes across ALL ranks without deadlocking on the all-reduce
    (a mismatch there is exactly what tripped the NCCL watchdog), and
  * it is fast enough that the rank that finishes first does not sit idle past the
    collective timeout.

It reports, per rank, the shard size / wall time / throughput / peak GPU memory, and
on rank 0 the averaged PESQ+UTMOS and a verdict against the configured DDP timeout.

Run from the repo root (so imports resolve like train.py):

    python debug_validation.py --config recipes/latest.yaml
    # quick smoke test on a 60-utterance subset, two passes (watch the 2nd: the
    # first pays one-off CUDA/Triton kernel compilation):
    python debug_validation.py --config recipes/latest.yaml --max-utts 60 --repeats 2
    # score with real weights so the PESQ/UTMOS numbers are meaningful:
    python debug_validation.py --config recipes/latest.yaml --checkpoint exp/MyRun

Notes
-----
* Uses its own rendezvous port (``--port``, default 19500) so it will not clash with
  a training job already bound to the config's ``dist_url``.
* Sets a short process-group timeout (``--timeout``, default 300 s) so a genuine hang
  fails fast here instead of blocking for the production 3600 s.
"""

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Subset

# Reuse the REAL code paths so this tests what training actually runs.
import train as T
from models.generator import SEMamba
from utils.metrics import Evaluator
from utils.util import initialize_seed, load_config, scan_checkpoint


def _fmt_mb(x):
    return f"{x / 1024:.2f} GB" if x >= 1024 else f"{x:.0f} MB"


def worker(rank, args, cfg):
    num_gpus = cfg["env_setting"]["num_gpus"]

    # --- distributed setup, mirroring train.initialize_process_group but with our
    # own port + a short timeout so a hang is reported quickly. --------------------
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    world_size = cfg["env_setting"]["dist_cfg"]["world_size"] * num_gpus
    dist.init_process_group(
        backend=cfg["env_setting"]["dist_cfg"]["dist_backend"],
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{args.port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=args.timeout),
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    initialize_seed(cfg["env_setting"]["seed"])

    is_rank0 = rank == 0
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress_factor = cfg["model_cfg"]["compress_factor"]
    stft_params = (n_fft, hop_size, win_size, compress_factor)

    # --- model (DDP-wrapped, exactly as in training so validate() exercises the
    # unwrap path) -----------------------------------------------------------------
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
        if is_rank0:
            print(f"[rank0] loaded generator weights from {path}")
    if num_gpus > 1 and not args.no_ddp:
        generator = DistributedDataParallel(generator, device_ids=[rank])

    # --- evaluator (per rank). Validation scores a whole val_batch_size batch of
    # fixed-length crops at once, so PESQ forks a pool across the batch (mirrors
    # train.py) and UTMOS overlaps on the GPU. --------------------------------------
    val_batch_size = cfg["training_cfg"].get("val_batch_size", 1)
    pesq_n_processes = max(1, min(val_batch_size, os.cpu_count() or 1))
    evaluator = Evaluator(
        sr=cfg["stft_cfg"]["sampling_rate"], pesq_n_processes=pesq_n_processes
    ).to(device)

    # --- sharded validation loader (the REAL create_dataloader) -------------------
    validset = T.create_dataset(cfg, train=False, split=False, device=device)
    full_len = len(validset)
    if args.max_utts and args.max_utts < full_len:
        validset = Subset(validset, list(range(args.max_utts)))
    loader = T.create_dataloader(validset, cfg, train=False, rank=rank, world_size=world_size)
    n_shard = len(loader.sampler)

    if is_rank0:
        print(
            f"\n=== validation debug ===\n"
            f"  config            : {args.config}\n"
            f"  mixer (ssm)       : {cfg['model_cfg'].get('ssm')}\n"
            f"  world_size        : {world_size}  (num_gpus={num_gpus})\n"
            f"  val utterances    : {len(validset)} of {full_len}"
            f"{' (subset)' if len(validset) < full_len else ''}\n"
            f"  pg timeout        : {args.timeout} s   "
            f"(prod default 3600 s via dist_cfg.timeout_seconds)\n"
            f"  repeats           : {args.repeats}\n",
            flush=True,
        )

    best = (0.0, 0, 0.0, 0)
    for r in range(args.repeats):
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        torch.cuda.synchronize(device)
        t0 = time.time()

        # sw=None -> no TensorBoard writes / no example-spectrogram logging, so we
        # time pure validation. This is the exact function training calls.
        best = T.validate(
            generator, evaluator, loader, cfg, device, None, steps=0,
            stft_params=stft_params, best=best,
        )

        torch.cuda.synchronize(device)
        elapsed = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

        # Gather per-rank (shard, elapsed_ms, peak_mb) onto rank 0 for a clean table.
        stats = torch.tensor([float(n_shard), elapsed * 1000.0, peak_mb], device=device)
        gathered = [torch.zeros_like(stats) for _ in range(world_size)]
        dist.all_gather(gathered, stats)

        if is_rank0:
            tag = "warm" if r > 0 else "cold (incl. kernel compile)"
            print(f"--- pass {r + 1}/{args.repeats}  [{tag}] ---")
            total_utts = 0
            slowest = 0.0
            for i, g in enumerate(gathered):
                sh, ms, mb = int(g[0]), g[1].item(), g[2].item()
                total_utts += sh
                slowest = max(slowest, ms)
                ups = sh / (ms / 1000.0) if ms > 0 else 0.0
                print(
                    f"  rank {i}: {sh:4d} utts | {ms / 1000:7.2f} s | "
                    f"{ups:5.2f} utts/s | peak {_fmt_mb(mb)}"
                )
            wall = slowest / 1000.0
            print(
                f"  TOTAL : {total_utts} utts across {world_size} ranks | "
                f"wall {wall:.2f} s | {total_utts / wall:.2f} utts/s | "
                f"PESQ {best[0]:.4f} (@{best[1]}) | UTMOS {best[2]:.4f}"
            )
            # Verdict: at the real interval this runs while ranks>0 wait on the next
            # training collective; it must finish well under the PG timeout.
            verdict = "OK" if wall < 0.8 * args.timeout else "SLOW -> would risk timeout"
            print(f"  verdict: wall {wall:.1f}s vs timeout {args.timeout}s  ->  {verdict}\n", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="recipes/latest.yaml")
    p.add_argument("--checkpoint", default=None, help="exp dir or g_*.pth for real weights")
    p.add_argument("--max-utts", type=int, default=0, help="cap the val set for a quick smoke test (0 = full)")
    p.add_argument("--repeats", type=int, default=2, help="validation passes; watch pass 2 (warm)")
    p.add_argument("--timeout", type=int, default=300, help="process-group timeout in s (fail-fast on hang)")
    p.add_argument("--port", type=int, default=19500, help="rendezvous port (avoid clashing with training)")
    p.add_argument("--num-gpus", type=int, default=0, help="override config num_gpus (0 = use config)")
    p.add_argument("--no-ddp", action="store_true", help="skip the DDP wrap (isolate model vs DDP issues)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.num_gpus:
        cfg["env_setting"]["num_gpus"] = args.num_gpus
    num_gpus = cfg["env_setting"]["num_gpus"]

    avail = torch.cuda.device_count()
    if num_gpus > avail:
        print(f"WARNING: num_gpus={num_gpus} > available {avail}; using {avail}")
        cfg["env_setting"]["num_gpus"] = num_gpus = avail
    if num_gpus < 1:
        raise SystemExit("Need at least 1 GPU.")

    if num_gpus > 1:
        mp.spawn(worker, nprocs=num_gpus, args=(args, cfg))
    else:
        worker(0, args, cfg)


if __name__ == "__main__":
    main()
