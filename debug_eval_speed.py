#!/usr/bin/env python3
"""Quick diagnostics for slow in-training validation.

Run this on BOTH the slow (HPC) and fast (org GPU server) machines and diff the
output. It probes the environment factors that make this repo's validation loop
(single GPU, batch_size=1, CPU-bound perceptual metrics, single-worker reads of
full-length utterances) sensitive to where it runs, then times each stage so you
can see which one actually dominates.

    python debug_eval_speed.py                      # env + micro-benchmarks
    python debug_eval_speed.py --metrics            # also time the real Evaluator
    python debug_eval_speed.py --data data/ears_wham_16k/valid_clean.json  # I/O test

Nothing here needs the trained model or the full config; stages that can't run
(missing deps / data) are skipped with a note instead of crashing.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import time

import numpy as np


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 1. Environment: cores actually granted, clock, SLURM/cgroup limits
# --------------------------------------------------------------------------- #
def report_env():
    section("ENVIRONMENT / CPU")
    print(f"host            : {platform.node()}")
    print(f"python          : {platform.python_version()}")
    print(f"os.cpu_count()  : {os.cpu_count()}")
    try:
        # cores this process may actually use (respects cgroup/cpuset pinning)
        print(f"sched affinity  : {len(os.sched_getaffinity(0))} cores usable")
    except AttributeError:
        pass

    # cgroup v2 / v1 CPU quota -- this is what SLURM uses to cap you
    quota = None
    v2 = "/sys/fs/cgroup/cpu.max"
    if os.path.exists(v2):
        try:
            q, p = open(v2).read().split()
            quota = "unlimited" if q == "max" else f"{int(q) / int(p):.1f} cores"
        except Exception:
            pass
    if quota:
        print(f"cgroup cpu.max  : {quota}")

    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_ID", "SLURM_NODELIST",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(var):
            print(f"{var:<16}: {os.environ[var]}")

    model = sh("lscpu | grep -E 'Model name'")
    mhz = sh("lscpu | grep -E 'CPU max MHz|CPU MHz'")
    if model:
        print(model.strip())
    if mhz:
        print(mhz.strip(), " <-- single-thread metrics (PESQ) are clock-bound")

    try:
        import torch
        print(f"torch           : {torch.__version__} (cuda {torch.version.cuda})")
        print(f"torch threads   : {torch.get_num_threads()}")
    except ImportError:
        print("torch           : NOT INSTALLED")


# --------------------------------------------------------------------------- #
# 2. GPU: model, MIG slice, power cap, clocks, who else is on it
# --------------------------------------------------------------------------- #
def report_gpu():
    section("GPU")
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found")
        return
    print(sh("nvidia-smi -L") or "")
    q = sh("nvidia-smi --query-gpu=name,power.limit,power.draw,clocks.sm,"
           "clocks.max.sm,utilization.gpu,memory.used,memory.total "
           "--format=csv,noheader")
    if q:
        print("name, pwr.limit, pwr.draw, sm.clk, sm.max, util%, mem.used, mem.total")
        print(q)
    procs = sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    if procs:
        n = len([l for l in procs.splitlines() if l.strip()])
        print(f"compute procs on GPU: {n}  (>1 => shared/contended)")
    if "MIG" in (sh("nvidia-smi -L") or ""):
        print("** MIG slice detected -- you have a fraction of the card **")


# --------------------------------------------------------------------------- #
# 3. Storage: filesystem type + read throughput for the val data
# --------------------------------------------------------------------------- #
def report_storage(data_path):
    section("STORAGE")
    target = data_path or os.getcwd()
    real = os.path.realpath(target if os.path.exists(target) else os.getcwd())
    fstype = sh(f"stat -f -c %T {real}")
    print(f"path            : {real}")
    print(f"fs type         : {fstype}  "
          f"({'NETWORK/PARALLEL -- likely I/O bound' if fstype in {'lustre','nfs','gpfs','smb2','fuseblk'} else 'local'})")
    mnt = sh(f"df -hT {real} | tail -1")
    if mnt:
        print(mnt)

    if data_path and os.path.exists(data_path):
        try:
            entries = json.load(open(data_path))
            paths = list(entries.values()) if isinstance(entries, dict) else entries
            paths = [p if isinstance(p, str) else p[0] for p in paths][:50]
        except Exception as e:
            print(f"could not parse {data_path}: {e}")
            return
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            print("no listed files exist on this machine -- skipping read test")
            return
        # cold-ish read: time opening + reading the first N files single-threaded,
        # which is exactly what the val loader (num_workers=1) does.
        t0 = time.perf_counter()
        nbytes = 0
        for p in existing:
            with open(p, "rb") as f:
                nbytes += len(f.read())
        dt = time.perf_counter() - t0
        print(f"read {len(existing)} files / {nbytes/1e6:.1f} MB in {dt:.2f}s "
              f"=> {nbytes/1e6/dt:.0f} MB/s, {dt/len(existing)*1e3:.1f} ms/file")
        print("  (low MB/s or high ms/file on a network FS => storage is the problem)")


# --------------------------------------------------------------------------- #
# 4. CPU single-thread micro-bench (proxy for PESQ/STFT metric cost)
# --------------------------------------------------------------------------- #
def bench_cpu():
    section("CPU SINGLE-THREAD MICRO-BENCH (PESQ-like)")
    # repeated FFTs on ~6s of 16k audio -- single threaded, clock sensitive
    x = np.random.randn(96000).astype(np.float64)
    t0 = time.perf_counter()
    for _ in range(200):
        np.abs(np.fft.rfft(x[:1024 * 90].reshape(90, 1024), axis=-1))
    dt = time.perf_counter() - t0
    print(f"200 STFT-ish iters: {dt:.3f}s  ({dt/200*1e3:.2f} ms/iter)")
    print("  compare across machines: slower here ~= lower CPU clock / fewer cores")


# --------------------------------------------------------------------------- #
# 5. GPU bs=1 latency micro-bench (validation runs at batch_size=1)
# --------------------------------------------------------------------------- #
def bench_gpu():
    section("GPU batch_size=1 LATENCY MICRO-BENCH")
    try:
        import torch
    except ImportError:
        print("torch not installed -- skipping")
        return
    if not torch.cuda.is_available():
        print("CUDA not available -- skipping")
        return
    dev = torch.device("cuda")
    # stand-in for a long, narrow sequential workload (bs=1, long time axis)
    net = torch.nn.Sequential(
        torch.nn.Conv1d(64, 64, 5, padding=2), torch.nn.GELU(),
        torch.nn.Conv1d(64, 64, 5, padding=2),
    ).to(dev)
    x = torch.randn(1, 64, 96000, device=dev)  # ~6s at 16k, bs=1
    with torch.no_grad():
        for _ in range(3):  # warmup
            net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            y = net(x)
        # mimic the per-metric .item() syncs that serialize the real loop
        for _ in range(20):
            float(y.mean())
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"20 fwd @ bs=1, len=96000 + .item() syncs: {dt:.3f}s "
          f"({dt/20*1e3:.1f} ms/iter)")
    print("  if this is fast but real validation is slow => metrics/IO, not the model")


# --------------------------------------------------------------------------- #
# 6. The REAL metric stage (PESQ / UTMOS / DistillMOS), timed individually
# --------------------------------------------------------------------------- #
def bench_metrics(sr):
    section("REAL EVALUATOR METRIC STAGE (per-metric timing)")
    try:
        import torch
        from utils.metrics import Evaluator
    except Exception as e:
        print(f"cannot import Evaluator ({e}) -- run from repo root with deps installed")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ev = Evaluator(sr=sr).to(dev)
    except Exception as e:
        print(f"Evaluator init failed: {e}")
        return
    clean = torch.randn(1, sr * 6, device=dev).clamp(-1, 1)
    pred = (clean + 0.05 * torch.randn_like(clean)).clamp(-1, 1)

    # warmup (loads hub weights, allocs)
    try:
        ev.compute(clean, pred, exclude=("nisqa", "estoi"))
    except Exception as e:
        print(f"warmup compute failed: {e}")

    def t(fn, name, n=5):
        try:
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            if dev == "cuda":
                torch.cuda.synchronize()
            print(f"  {name:<12}: {(time.perf_counter()-t0)/n*1e3:7.1f} ms/utt")
        except Exception as e:
            print(f"  {name:<12}: FAILED ({e})")

    print(f"device={dev}, 6s utterance, batch_size=1 (as in training validation)")
    t(lambda: ev._mrstft(pred.unsqueeze(1), clean.unsqueeze(1)).item(), "mrstft")
    t(lambda: ev._pesq(pred, clean).item(), "pesq(CPU)")   # runs on CPU regardless
    t(lambda: ev._utmos(pred, sr).item(), "utmos")
    t(lambda: ev._sisdr(pred, clean).item(), "sisdr")
    t(lambda: ev._lsd(clean, pred), "lsd")
    with torch.no_grad():
        t(lambda: ev._distillmos(pred).mean().item(), "distillmos")
    t(lambda: ev.compute(clean, pred, exclude=("nisqa", "estoi")), "FULL compute")
    print("  whichever line dominates is your bottleneck; PESQ tends to be CPU/clock bound")


# --------------------------------------------------------------------------- #
# 7. OLD vs NEW validation-loop pattern (what actually changed in train.py)
# --------------------------------------------------------------------------- #
def bench_loop(sr, n_utts, val_bs, n_jobs):
    section("VALIDATION-LOOP PATTERN: OLD vs NEW")
    try:
        import torch
        from joblib import Parallel, delayed
        from utils.metrics import Evaluator
        from utils.pesq_utils import pesq_wb
    except Exception as e:
        print(f"cannot import repo deps ({e}) -- run from repo root")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = Evaluator(sr=sr).to(dev)
    # synthetic full-length utterances (metric stage only; excludes the model fwd)
    utts = []
    for _ in range(n_utts):
        c = torch.randn(1, sr * 6, device=dev).clamp(-1, 1)
        utts.append((c, (c + 0.05 * torch.randn_like(c)).clamp(-1, 1)))
    print(f"device={dev}, {n_utts} utts of 6s, val_bs={val_bs}, pesq n_jobs={n_jobs}")

    def warmup():
        ev.compute(utts[0][0], utts[0][1], exclude=("nisqa", "estoi"), as_tensor=False)
    warmup()
    if dev == "cuda":
        torch.cuda.synchronize()

    # --- OLD: bs=1, .item() per metric per utterance, serial PESQ ---
    t0 = time.perf_counter()
    acc = {k: 0.0 for k in ("mrstft", "pesq", "utmos", "distillmos", "sisdr", "lsd")}
    for c, p in utts:
        m = ev.compute(c, p, exclude=("nisqa", "estoi"), as_tensor=False)
        for k in acc:
            acc[k] += getattr(m, k)
    if dev == "cuda":
        torch.cuda.synchronize()
    old = time.perf_counter() - t0

    # --- NEW: accumulate GPU metrics as tensors, one sync; PESQ in parallel ---
    # Warm the loky pool first; in training it persists across validation passes,
    # so the one-time spawn cost should not be charged to the steady-state timing.
    with Parallel(n_jobs=n_jobs) as _warm:
        _warm(delayed(pesq_wb)(sr, utts[0][0][0].cpu().numpy(), utts[0][1][0].cpu().numpy())
              for _ in range(n_jobs))
    t0 = time.perf_counter()
    sums = {k: torch.zeros((), device=dev) for k in ("mrstft", "utmos", "distillmos", "sisdr", "lsd")}
    pesq_sum = 0.0
    with Parallel(n_jobs=n_jobs) as run_pesq:
        for i in range(0, len(utts), val_bs):
            chunk = utts[i:i + val_bs]
            refs, degs = [], []
            for c, p in chunk:
                m = ev.compute(c, p, exclude=("nisqa", "estoi", "pesq"), as_tensor=True)
                for k in sums:
                    sums[k] += getattr(m, k)
                refs.append(c[0].cpu().numpy())
                degs.append(p[0].cpu().numpy())
            pesq_sum += float(sum(run_pesq(delayed(pesq_wb)(sr, r, d) for r, d in zip(refs, degs))))
    _ = {k: v.item() for k, v in sums.items()}  # single sync
    if dev == "cuda":
        torch.cuda.synchronize()
    new = time.perf_counter() - t0

    print(f"  OLD (per-utt .item + serial PESQ): {old:7.2f}s  ({old/n_utts*1e3:6.1f} ms/utt)")
    print(f"  NEW (batched accum + parallel PESQ): {new:7.2f}s  ({new/n_utts*1e3:6.1f} ms/utt)")
    print(f"  speedup: {old/new:.1f}x   (excludes the model forward, which also benefits from batching)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="valid_clean.json to probe storage/read speed")
    ap.add_argument("--metrics", action="store_true",
                    help="time the real Evaluator (needs repo deps)")
    ap.add_argument("--loop", action="store_true",
                    help="benchmark OLD vs NEW validation-loop pattern (needs repo deps)")
    ap.add_argument("--n-utts", type=int, default=40, help="utterances for --loop")
    ap.add_argument("--val-bs", type=int, default=8, help="validation batch size for --loop")
    ap.add_argument("--n-jobs", type=int, default=16, help="parallel PESQ workers for --loop")
    ap.add_argument("--sr", type=int, default=16000, help="sampling rate")
    args = ap.parse_args()

    report_env()
    report_gpu()
    report_storage(args.data)
    bench_cpu()
    bench_gpu()
    if args.metrics:
        bench_metrics(args.sr)
    if args.loop:
        bench_loop(args.sr, args.n_utts, args.val_bs, args.n_jobs)
    print("\nDone. Run on both machines and diff the output.\n")


if __name__ == "__main__":
    main()
