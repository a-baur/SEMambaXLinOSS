"""Report parameter count, GFLOPs, and wall-clock timing of the SEMamba generator.

Builds the model exactly like ``evaluate.py`` (same config/checkpoint plumbing),
then runs the generator on a single synthetic utterance to measure FLOPs via
PyTorch's built-in ``FlopCounterMode`` and to time the forward and backward passes.

Supports the same checkpoint layouts as ``evaluate.py``:
  * a single ``.pth`` file,
  * a checkpoint directory (holds ``*.pth`` + ``config.yaml`` or ``g_*.pth``),
  * a *parent* directory of such checkpoint dirs (e.g. ``ckpts/``): every
    checkpoint under it is profiled and a comparison table is printed.

Examples:
    # From an exp dir (uses the co-located config.yaml; weights optional for FLOPs)
    python profile_model.py --checkpoint exp/MyRun

    # From a bare config, no trained weights needed
    python profile_model.py --config recipes/latest-test.yaml --no-load-weights

    # Profile + compare every checkpoint under a parent directory
    python profile_model.py --checkpoint ckpts
"""

import argparse
import json
import os
import time
import warnings

import torch
from torch.utils.flop_counter import FlopCounterMode

# ignore mamba_ssm deprecation warning
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.custom_(fwd|bwd).*is deprecated.*",
    category=FutureWarning,
)

from evaluate import (
    find_checkpoint_dirs,
    is_checkpoint_dir,
    load_generator,
    resolve_checkpoint_and_config,
)
from models.generator import SEMamba
from models.stfts import mag_phase_stft
from utils.util import load_config


def human(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000.0:
            return f"{n:.3f}{unit}"
        n /= 1000.0
    return f"{n:.3f}P"


def scalar_loss(out) -> torch.Tensor:
    """A cheap scalar to backprop from (mag, pha, com) so backward has work to do."""
    mag, pha, com = out
    return mag.float().pow(2).mean() + pha.float().pow(2).mean() + com.float().pow(2).mean()


def time_passes(model, mag, pha, runs: int, warmup: int, device: torch.device):
    """Median wall-clock per pass (ms) for forward-only and forward+backward.

    Timed *outside* FlopCounterMode: the counter installs a TorchDispatchMode that
    intercepts every ATen op, so timing under it would measure that overhead, not
    the model.
    """
    cuda = device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    # --- forward only (no grad) ---
    fwd = []
    with torch.no_grad():
        for i in range(warmup + runs):
            sync()
            t0 = time.perf_counter()
            model(mag, pha)
            sync()
            if i >= warmup:
                fwd.append(time.perf_counter() - t0)

    # --- forward + backward (grad enabled) ---
    fwd_bwd = []
    for i in range(warmup + runs):
        model.zero_grad(set_to_none=True)
        sync()
        t0 = time.perf_counter()
        loss = scalar_loss(model(mag, pha))
        loss.backward()
        sync()
        if i >= warmup:
            fwd_bwd.append(time.perf_counter() - t0)

    fwd.sort()
    fwd_bwd.sort()
    return fwd[len(fwd) // 2] * 1e3, fwd_bwd[len(fwd_bwd) // 2] * 1e3


def mixer_name(cfg: dict) -> str:
    """Human-readable mixer, matching TFMambaBlock's per-axis resolution.

    Mixer is keyed 'ssm' (default 'mamba'); hybrid configs override per axis
    via 'time_mixer'/'freq_mixer' sub-dicts.
    """
    mcfg = cfg["model_cfg"]
    time_ssm = mcfg.get("time_mixer", mcfg).get("ssm", "mamba")
    freq_ssm = mcfg.get("freq_mixer", mcfg).get("ssm", "mamba")
    return time_ssm if time_ssm == freq_ssm else f"time={time_ssm}, freq={freq_ssm}"


def profile_checkpoint(
    ckpt_file: str | None,
    config_file: str,
    args,
    device: torch.device,
) -> dict:
    """Profile one checkpoint/config; print the report and return a summary dict.

    ``ckpt_file`` may be ``None`` (config-only) or skipped via ``--no-load-weights``,
    in which case a randomly-initialised model is built (FLOPs/params are unaffected).
    """
    print(f"Config     : {config_file}")
    cfg = load_config(config_file)

    if ckpt_file is not None and not args.no_load_weights:
        print(f"Checkpoint : {ckpt_file}")
        model = load_generator(ckpt_file, cfg, device)
    else:
        model = SEMamba(cfg).to(device).eval()
        print("Checkpoint : (none — random init; FLOPs/params unaffected)")

    # --- Parameter count ---
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    mixer = mixer_name(cfg)

    print()
    print(f"Mixer        : {mixer}")
    print(f"Parameters   : {human(total)}  ({total:,} total, {trainable:,} trainable)")

    # --- Build a single real-shaped input via the STFT front end ---
    sr = cfg["stft_cfg"]["sampling_rate"]
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress = cfg["model_cfg"]["compress_factor"]

    num_samples = int(args.duration * sr)
    wav = torch.randn(1, num_samples, device=device)
    mag, pha, _ = mag_phase_stft(wav, n_fft, hop_size, win_size, compress)
    F, T = mag.shape[-2], mag.shape[-1]
    print(f"Input        : {args.duration:g}s @ {sr} Hz -> mag/pha [1, {F}, {T}]")

    # --- FLOP count: forward only, then forward+backward ---
    # FlopCounterMode tallies whatever ATen ops execute inside its context, so a
    # forward under no_grad gives fwd FLOPs, and a forward+backward gives the sum.
    fwd_counter = FlopCounterMode(display=False)
    with torch.no_grad(), fwd_counter:
        model(mag, pha)
    fwd_flops = fwd_counter.get_total_flops()  # counts MACs as 2 FLOPs already

    fb_counter = FlopCounterMode(display=False)
    with fb_counter:
        model.zero_grad(set_to_none=True)
        scalar_loss(model(mag, pha)).backward()
    fwd_bwd_flops = fb_counter.get_total_flops()
    bwd_flops = fwd_bwd_flops - fwd_flops

    print(f"FLOPs (fwd)  : {human(fwd_flops)}  ({fwd_flops:,})")
    print(f"FLOPs (bwd)  : {human(bwd_flops)}  ({bwd_flops:,})")
    print(f"FLOPs (f+b)  : {human(fwd_bwd_flops)}  ({fwd_bwd_flops:,})")
    print(f"GFLOPs       : fwd {fwd_flops / 1e9:.3f} | bwd {bwd_flops / 1e9:.3f} "
          f"| f+b {fwd_bwd_flops / 1e9:.3f}  (single sample)")

    # --- Wall-clock timing (separate from the FLOP counter, see time_passes) ---
    fwd_ms, fwd_bwd_ms = time_passes(model, mag, pha, args.runs, args.warmup, device)
    bwd_ms = fwd_bwd_ms - fwd_ms
    print()
    print(f"Time (fwd)   : {fwd_ms:.3f} ms")
    print(f"Time (bwd)   : {bwd_ms:.3f} ms")
    print(f"Time (f+b)   : {fwd_bwd_ms:.3f} ms"
          f"  (median of {args.runs} runs, {args.warmup} warmup)")

    return {
        "checkpoint": ckpt_file,
        "config": config_file,
        "mixer": mixer,
        "params_total": total,
        "params_trainable": trainable,
        "gflops_fwd": fwd_flops / 1e9,
        "gflops_bwd": bwd_flops / 1e9,
        "gflops_fb": fwd_bwd_flops / 1e9,
        "time_fwd_ms": fwd_ms,
        "time_bwd_ms": bwd_ms,
        "time_fb_ms": fwd_bwd_ms,
    }


# (column label, summary key, format spec) for the comparison table.
COMPARE_COLUMNS = (
    ("Mixer", "mixer", ">18s"),
    ("Params", "params_total", ">12,d"),
    ("GFLOPs(fwd)", "gflops_fwd", ">11.3f"),
    ("GFLOPs(f+b)", "gflops_fb", ">11.3f"),
    ("Fwd ms", "time_fwd_ms", ">10.3f"),
    ("Bwd ms", "time_bwd_ms", ">10.3f"),
    ("F+b ms", "time_fb_ms", ">10.3f"),
)


def print_comparison_table(summaries: dict[str, dict]):
    """Print a name x profile-metric table, one row per checkpoint."""
    if not summaries:
        return
    name_w = max(len("checkpoint"), max(len(n) for n in summaries))
    widths = [max(len(lbl), 18) for lbl, _, _ in COMPARE_COLUMNS]
    header = "  ".join(
        [f"{'checkpoint':<{name_w}}"]
        + [f"{lbl:>{w}}" for (lbl, _, _), w in zip(COMPARE_COLUMNS, widths)]
    )
    print("\n" + "=" * len(header))
    print("COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, summ in summaries.items():
        cells = [f"{name:<{name_w}}"]
        for (_, key, spec), w in zip(COMPARE_COLUMNS, widths):
            cells.append(f"{format(summ[key], spec):>{w}}")
        print("  ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="A .pth file, a checkpoint dir (with *.pth + config.yaml or g_*.pth), or a "
        "parent directory of such dirs (profiles + compares all). Used to locate config "
        "and (optionally) load weights. FLOPs/params do not depend on the weights.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config. Defaults to config.yaml next to the checkpoint.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Length of the single synthetic sample, in seconds (sets the time axis).",
    )
    parser.add_argument(
        "--no-load-weights",
        action="store_true",
        help="Skip loading checkpoint weights (just build from config).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Timed iterations (median reported) for forward and backward passes.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed warmup iterations before timing (lets CUDA kernels autotune).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="If set (compare mode), write the gathered summaries to "
        "<output_dir>/profile_comparison.json.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.config is None and args.checkpoint is None:
        parser.error("provide --config and/or --checkpoint")

    device = torch.device(args.device)

    # Compare mode: a parent directory of checkpoint dirs (ckpts/mamba, ...).
    if (
        args.checkpoint is not None
        and os.path.isdir(args.checkpoint)
        and not is_checkpoint_dir(args.checkpoint)
    ):
        ckpt_dirs = find_checkpoint_dirs(args.checkpoint)
        if not ckpt_dirs:
            raise FileNotFoundError(
                f"No checkpoint directories (with *.pth) found under {args.checkpoint}"
            )
        print(f"Profiling {len(ckpt_dirs)} checkpoints under {args.checkpoint}:")
        for d in ckpt_dirs:
            print(f"  - {os.path.basename(d)}")

        summaries: dict[str, dict] = {}
        for d in ckpt_dirs:
            name = os.path.basename(d.rstrip("/"))
            print(f"\n{'#' * 60}\n# {name}\n{'#' * 60}")
            ckpt_file, config_file = resolve_checkpoint_and_config(d, args.config)
            summaries[name] = profile_checkpoint(ckpt_file, config_file, args, device)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            comparison_json = os.path.join(args.output_dir, "profile_comparison.json")
            with open(comparison_json, "w") as f:
                json.dump(summaries, f, indent=2)
        print_comparison_table(summaries)
        if args.output_dir:
            print(f"\nComparison   : {comparison_json}")
        print(
            "\nNote: FlopCounterMode tracks ATen ops; custom Mamba/LinOSS scan CUDA/"
            "Triton\nkernels are not counted, so the recurrence cost is excluded from "
            "FLOPs\n(but is included in the wall-clock timing)."
        )
        return

    # Single-checkpoint (or config-only) mode.
    ckpt_file = None
    if args.checkpoint is not None:
        ckpt_file, config_file = resolve_checkpoint_and_config(
            args.checkpoint, args.config
        )
    else:
        config_file = args.config

    profile_checkpoint(ckpt_file, config_file, args, device)

    print(
        "\nNote: FlopCounterMode tracks ATen ops; custom Mamba/LinOSS scan CUDA/"
        "Triton\nkernels are not counted, so the recurrence cost is excluded from "
        "FLOPs\n(but is included in the wall-clock timing). Backward FLOPs are the "
        "difference\nf+b minus fwd."
    )


if __name__ == "__main__":
    main()
