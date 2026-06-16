"""Report parameter count, GFLOPs, and wall-clock timing of the SEMamba generator.

Builds the model exactly like ``evaluate.py`` (same config/checkpoint plumbing),
then runs the generator on a single synthetic utterance to measure FLOPs via
PyTorch's built-in ``FlopCounterMode`` and to time the forward and backward passes.

Examples:
    # From an exp dir (uses the co-located config.yaml; weights optional for FLOPs)
    python profile_model.py --checkpoint exp/MyRun

    # From a bare config, no trained weights needed
    python profile_model.py --config recipes/la  -test.yaml --no-load-weights
"""

import argparse
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

from evaluate import load_generator, resolve_checkpoint_and_config
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="A .pth file or an exp dir with g_*.pth. Used to locate config and "
        "(optionally) load weights. FLOPs/params do not depend on the weights.",
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
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.config is None and args.checkpoint is None:
        parser.error("provide --config and/or --checkpoint")

    device = torch.device(args.device)

    # Resolve config (and checkpoint) the same way evaluate.py does.
    ckpt_file = None
    if args.checkpoint is not None:
        ckpt_file, config_file = resolve_checkpoint_and_config(
            args.checkpoint, args.config
        )
    else:
        config_file = args.config
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

    # Mixer is keyed 'ssm' (default 'mamba'); hybrid configs override per axis
    # via 'time_mixer'/'freq_mixer' sub-dicts (see TFMambaBlock).
    mcfg = cfg["model_cfg"]
    time_ssm = mcfg.get("time_mixer", mcfg).get("ssm", "mamba")
    freq_ssm = mcfg.get("freq_mixer", mcfg).get("ssm", "mamba")
    mixer = time_ssm if time_ssm == freq_ssm else f"time={time_ssm}, freq={freq_ssm}"

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

    print(
        "\nNote: FlopCounterMode tracks ATen ops; custom Mamba/LinOSS scan CUDA/"
        "Triton\nkernels are not counted, so the recurrence cost is excluded from "
        "FLOPs\n(but is included in the wall-clock timing). Backward FLOPs are the "
        "difference\nf+b minus fwd."
    )


if __name__ == "__main__":
    main()
