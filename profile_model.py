"""Report parameter count and GFLOPs of the SEMamba generator for one sample.

Builds the model exactly like ``evaluate.py`` (same config/checkpoint plumbing),
then runs the generator on a single synthetic utterance to measure FLOPs via
PyTorch's built-in ``FlopCounterMode``.

Examples:
    # From an exp dir (uses the co-located config.yaml; weights optional for FLOPs)
    python profile_model.py --checkpoint exp/MyRun

    # From a bare config, no trained weights needed
    python profile_model.py --config recipes/la  -test.yaml --no-load-weights
"""

import argparse

import torch
from torch.utils.flop_counter import FlopCounterMode

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

    # --- FLOP count for one forward pass ---
    flop_counter = FlopCounterMode(display=False)
    with torch.no_grad(), flop_counter:
        model(mag, pha)
    flops = flop_counter.get_total_flops()  # counts MACs as 2 FLOPs already

    print(f"FLOPs        : {human(flops)}  ({flops:,})")
    print(f"GFLOPs       : {flops / 1e9:.3f}  (single sample, fwd only)")
    print(
        "\nNote: FlopCounterMode tracks ATen ops; custom Mamba/LinOSS scan CUDA/"
        "Triton\nkernels are not counted, so the recurrence cost is excluded."
    )


if __name__ == "__main__":
    main()
