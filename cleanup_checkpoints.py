#!/usr/bin/env python3
"""Prune per-run checkpoints under ``exp/``, keeping only the ones worth saving.

For every run directory ``exp/<run>/`` this keeps three checkpoints and deletes
the rest (both the ``g_<step>.pth`` generator and the paired ``do_<step>.pth``
discriminator):

  1. the **latest** checkpoint (highest step present on disk),
  2. the checkpoint with the **highest PESQ**, and
  3. the checkpoint with the **highest PESQ among epochs <= 200**.

PESQ and epoch for each step are recovered from the wandb stdout logs
(``wandb/run-*/files/output.log``). Each validation pass writes, in order::

    Saving checkpoint to exp/<run>/g_<step>.pth
    Validation @ <step> steps: ...
    VALIDATION | ... | PESQ Score: <pesq> (max. ...) | ...

and epochs come from ``Epoch: <n>`` markers. Logs from resumed runs are merged
in timestamp order, so a step's PESQ/epoch is taken from the run that produced it.

Safe by default: prints a plan and deletes nothing unless ``--execute`` is given.

Usage::

    python cleanup_checkpoints.py                 # dry run over exp/
    python cleanup_checkpoints.py --execute        # actually delete
    python cleanup_checkpoints.py --run SelectiveLRU_GammaNorm --execute
    python cleanup_checkpoints.py --exp-dir exp --max-epoch 200
"""

import argparse
import glob
import os
import re
import sys

CKPT_RE = re.compile(r"^(g|do)_(\d+)\.pth$")
EPOCH_RE = re.compile(r"^Epoch:\s*(\d+)")
SAVE_G_RE = re.compile(r"Saving checkpoint to .*[/\\]g_(\d+)\.pth")
VAL_STEP_RE = re.compile(r"Validation @ (\d+) steps")
PESQ_RE = re.compile(r"VALIDATION\b.*PESQ Score:\s*([0-9]+\.?[0-9]*)")


def parse_run_logs(run_dir):
    """Return ``{step: {"pesq": float|None, "epoch": int|None}}`` for one run.

    Merges every ``wandb/run-*/files/output.log`` under ``run_dir`` in filename
    (timestamp) order so later resumes overwrite earlier data for the same step.
    """
    logs = sorted(
        glob.glob(os.path.join(run_dir, "wandb", "run-*", "files", "output.log"))
    )
    info = {}
    for log_path in logs:
        cur_epoch = None
        last_step = None
        try:
            with open(log_path, "r", errors="replace") as f:
                for line in f:
                    m = EPOCH_RE.search(line)
                    if m:
                        cur_epoch = int(m.group(1))
                        continue
                    m = SAVE_G_RE.search(line)
                    if m:
                        last_step = int(m.group(1))
                        info.setdefault(last_step, {"pesq": None, "epoch": None})
                        if cur_epoch is not None:
                            info[last_step]["epoch"] = cur_epoch
                        continue
                    m = VAL_STEP_RE.search(line)
                    if m:
                        last_step = int(m.group(1))
                        continue
                    m = PESQ_RE.search(line)
                    if m and last_step is not None:
                        rec = info.setdefault(
                            last_step, {"pesq": None, "epoch": None}
                        )
                        rec["pesq"] = float(m.group(1))
                        if rec["epoch"] is None and cur_epoch is not None:
                            rec["epoch"] = cur_epoch
        except OSError as e:
            print(f"  warning: could not read {log_path}: {e}", file=sys.stderr)
    return info


def disk_steps(run_dir):
    """Return ``{step: {"g": path|None, "do": path|None}}`` for checkpoints on disk."""
    steps = {}
    for name in os.listdir(run_dir):
        m = CKPT_RE.match(name)
        if not m:
            continue
        kind, step = m.group(1), int(m.group(2))
        steps.setdefault(step, {"g": None, "do": None})[kind] = os.path.join(
            run_dir, name
        )
    return steps


def choose_keep(steps, info, max_epoch):
    """Return ``{step: [reasons]}`` for the checkpoints to keep in one run."""
    keep = {}

    def add(step, reason):
        if step is not None:
            keep.setdefault(step, []).append(reason)

    # 1. Latest checkpoint on disk.
    add(max(steps), "latest")

    # 2. Highest PESQ among on-disk checkpoints with a known PESQ.
    scored = [(s, info.get(s, {}).get("pesq")) for s in steps]
    scored = [(s, p) for s, p in scored if p is not None]
    if scored:
        best = max(scored, key=lambda sp: sp[1])
        add(best[0], f"best PESQ ({best[1]:.4f})")

        # 3. Highest PESQ among on-disk checkpoints with epoch <= max_epoch.
        early = [
            (s, p)
            for s, p in scored
            if (info.get(s, {}).get("epoch") is not None
                and info[s]["epoch"] <= max_epoch)
        ]
        if early:
            best_early = max(early, key=lambda sp: sp[1])
            ep = info[best_early[0]]["epoch"]
            add(best_early[0], f"best PESQ<=ep{max_epoch} ({best_early[1]:.4f} @ ep{ep})")

    return keep


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024


def process_run(run_dir, max_epoch, execute):
    """Plan (and optionally perform) cleanup for a single run directory."""
    steps = disk_steps(run_dir)
    if not steps:
        return 0, 0
    info = parse_run_logs(run_dir)
    keep = choose_keep(steps, info, max_epoch)

    print(f"\n{run_dir}  ({len(steps)} checkpoint step(s) on disk)")
    for step in sorted(keep):
        pesq = info.get(step, {}).get("pesq")
        ep = info.get(step, {}).get("epoch")
        pesq_s = f"{pesq:.4f}" if pesq is not None else "?"
        print(
            f"  KEEP   step {step:>9}  PESQ={pesq_s}  epoch={ep}  "
            f"[{', '.join(keep[step])}]"
        )

    freed = 0
    n_deleted = 0
    for step in sorted(steps):
        if step in keep:
            continue
        for kind in ("g", "do"):
            path = steps[step][kind]
            if path and os.path.exists(path):
                freed += os.path.getsize(path)
                n_deleted += 1
                if execute:
                    os.remove(path)
        pesq = info.get(step, {}).get("pesq")
        pesq_s = f"{pesq:.4f}" if pesq is not None else "?"
        verb = "DELETED" if execute else "would delete"
        print(f"  {verb} step {step:>9}  PESQ={pesq_s}")

    action = "freed" if execute else "would free"
    print(f"  -> {action} {human_size(freed)} across {n_deleted} file(s)")
    return freed, n_deleted


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default="exp",
                    help="parent directory holding run subdirs (default: exp)")
    ap.add_argument("--run", default=None,
                    help="clean only this run subdir (default: all runs under --exp-dir)")
    ap.add_argument("--max-epoch", type=int, default=200,
                    help="epoch ceiling for the third kept checkpoint (default: 200)")
    ap.add_argument("--execute", action="store_true",
                    help="actually delete files (default: dry run, deletes nothing)")
    args = ap.parse_args()

    if args.run:
        run_dirs = [os.path.join(args.exp_dir, args.run)]
    else:
        run_dirs = sorted(
            d for d in glob.glob(os.path.join(args.exp_dir, "*")) if os.path.isdir(d)
        )

    if not args.execute:
        print("DRY RUN -- no files will be deleted. Re-run with --execute to apply.")

    total_freed = 0
    total_deleted = 0
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            print(f"skipping {run_dir}: not a directory", file=sys.stderr)
            continue
        freed, n = process_run(run_dir, args.max_epoch, args.execute)
        total_freed += freed
        total_deleted += n

    action = "Freed" if args.execute else "Would free"
    print(f"\n{action} {human_size(total_freed)} total across "
          f"{total_deleted} file(s) in {len(run_dirs)} run(s).")
    if not args.execute and total_deleted:
        print("Re-run with --execute to delete.")


if __name__ == "__main__":
    main()
