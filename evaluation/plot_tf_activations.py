"""Plot per-channel activations flowing through the time-frequency blocks.

For one noisy/clean utterance pair this runs the generator and captures the
running ``[B, C, T, F]`` representation right after each ``TFMambaBlock``'s
**time** sub-mixer (post residual) and after its **frequency** sub-mixer, then
reduces the channel axis to a single time-frequency energy map so it can be
shown like a spectrogram.

Layout (two columns):
  * row 0    : clean spectrogram | noisy spectrogram
  * row i+1  : activation after TF-block ``i`` time mixer | after freq mixer

Example
-------
    python evaluation/plot_tf_activations.py \
        --checkpoint exp/SelectiveLRUMIMO \
        --index 0 \
        --output eval_out/tf_activations.png
"""

import argparse
import glob
import os
import sys
import warnings

# Allow running as `python evaluation/plot_tf_activations.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ignore mamba_ssm deprecation warning (matches evaluate.py)
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.custom_(fwd|bwd).*is deprecated.*",
    category=FutureWarning,
)

import librosa
import numpy as np
import torch

from models.generator import SEMamba
from models.stfts import mag_phase_stft
from utils.util import load_config


def resolve_checkpoint_and_config(ckpt_arg: str, config_arg: str | None):
    """Return (ckpt_file, config_file); mirrors evaluate.py's resolver."""
    if os.path.isdir(ckpt_arg):
        # exp/ runs use step-numbered g_*.pth; ckpts/<name>/ hold a single
        # <name>.pth. Prefer the latest g_*.pth, else any other .pth.
        candidates = sorted(glob.glob(os.path.join(ckpt_arg, "g_????????.pth")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(ckpt_arg, "*.pth")))
        if not candidates:
            raise FileNotFoundError(f"No .pth checkpoints under {ckpt_arg}")
        ckpt_file = candidates[-1]
        default_config = os.path.join(ckpt_arg, "config.yaml")
    else:
        if not os.path.isfile(ckpt_arg):
            raise FileNotFoundError(ckpt_arg)
        ckpt_file = ckpt_arg
        default_config = os.path.join(os.path.dirname(ckpt_arg), "config.yaml")
    config_file = config_arg or default_config
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config not found at {config_file}; pass --config explicitly.")
    return ckpt_file, config_file


def load_generator(ckpt_file: str, cfg: dict, device: torch.device) -> SEMamba:
    model = SEMamba(cfg).to(device)
    state = torch.load(ckpt_file, map_location=device)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def resolve_pair(args, cfg):
    """Return (noisy_path, clean_path) from explicit paths or a test-set index."""
    if args.noisy is not None:
        return args.noisy, args.clean
    import json

    from dataloaders.dataloader_vctk import (
        _common_root,
        extract_identifier,
        get_clean_path_for_noisy,
    )

    clean_json = args.test_clean_json or cfg["data_cfg"]["test_clean_json"]
    noisy_json = args.test_noisy_json or cfg["data_cfg"]["test_noisy_json"]
    with open(clean_json) as f:
        clean_paths = json.load(f)
    with open(noisy_json) as f:
        noisy_paths = json.load(f)
    clean_root = _common_root(clean_paths)
    noisy_root = _common_root(noisy_paths)
    clean_by_id = {extract_identifier(p, clean_root): p for p in clean_paths}
    noisy_sorted = sorted(noisy_paths, key=os.path.basename)
    noisy = noisy_sorted[args.index]
    clean = get_clean_path_for_noisy(noisy, noisy_root, clean_by_id)
    return noisy, clean


class TFActivationCapture:
    """Re-implements ``TFMambaBlock.forward`` to grab the two intermediate maps.

    Each captured tensor is the running ``[B, C, T, F]`` representation after
    the time (post-residual) and after the frequency sub-mixer, i.e. exactly
    what flows on to the next block.
    """

    def __init__(self, model: SEMamba):
        self.model = model
        self.records = []  # list of (after_time, after_freq), one per TF block

    def __enter__(self):
        for block in self.model.TSMamba:
            block.forward = self._make_forward(block)
        return self

    def _make_forward(self, block):
        records = self.records

        def forward(x):
            b, c, t, f = x.size()
            x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
            x = block.tlinear(block.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
            after_time = x.view(b, f, t, c).permute(0, 3, 2, 1).contiguous()  # [B,C,T,F]
            x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
            x = block.flinear(block.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
            after_freq = x.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()  # [B,C,T,F]
            records.append((after_time.detach().cpu(), after_freq.detach().cpu()))
            return after_freq

        return forward

    def __exit__(self, *exc):
        # Drop the per-instance override so the class method is used again.
        for block in self.model.TSMamba:
            del block.forward
        return False


def reduce_channels(act: torch.Tensor, mode: str, per_channel_norm: str) -> np.ndarray:
    """Reduce ``[B, C, T, F]`` (batch 0) over channels to a ``[F, T]`` map.

    ``per_channel_norm`` rescales each channel over its own ``(T, F)`` support
    *before* aggregating so no single loud channel dominates the map:
      * ``std`` -- z-score each channel (zero mean, unit std): the map then
        reads as "how many std devs from its own baseline is each channel here",
        pooled over channels.
      * ``max`` -- divide each channel by its own peak |value| (scale to [-1,1]).
      * ``none`` -- raw activations (a few high-variance channels can dominate).

    ``mode`` is the channel aggregation. ``median`` (of |a|) is robust to a
    handful of outlier channels even without normalization.
    """
    a = act[0].clone()  # [C, T, F]
    if per_channel_norm == "std":
        mu = a.mean(dim=(1, 2), keepdim=True)
        sd = a.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        a = (a - mu) / sd
    elif per_channel_norm == "max":
        peak = a.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        a = a / peak
    elif per_channel_norm != "none":
        raise ValueError(f"Unknown per_channel_norm {per_channel_norm!r}")

    if mode == "l2":
        m = a.pow(2).mean(dim=0).sqrt()
    elif mode == "mean_abs":
        m = a.abs().mean(dim=0)
    elif mode == "median":
        m = a.abs().median(dim=0).values
    elif mode == "max":
        m = a.abs().amax(dim=0)
    else:
        raise ValueError(f"Unknown reduce mode {mode!r}")
    return m.transpose(0, 1).numpy()  # [F, T]


def to_db(mag: np.ndarray, ref: float, floor_db: float = -80.0) -> np.ndarray:
    """Log-scale a magnitude spectrogram to dB relative to ``ref``, with a floor."""
    db = 20.0 * np.log10(np.maximum(mag, 1e-12) / (ref + 1e-12))
    return np.maximum(db, floor_db)


def has_checkpoint(d: str) -> bool:
    """Whether directory ``d`` directly contains a loadable .pth checkpoint."""
    return bool(
        glob.glob(os.path.join(d, "g_????????.pth")) or glob.glob(os.path.join(d, "*.pth"))
    )


def expand_checkpoints(path: str) -> list[str]:
    """Resolve a path to a list of checkpoint args.

    Accepts a ``.pth`` file, a single checkpoint dir (``exp/Run`` or
    ``ckpts/mamba``), or a *parent* dir (``ckpts/``) whose immediate subdirs are
    checkpoint dirs -- in which case every such subdir is returned.
    """
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    if has_checkpoint(path):
        return [path]
    subs = sorted(
        p for p in glob.glob(os.path.join(path, "*")) if os.path.isdir(p) and has_checkpoint(p)
    )
    if not subs:
        raise FileNotFoundError(f"No .pth checkpoints found in {path} or its immediate subdirs")
    return subs


def run_name_for(ckpt_arg: str) -> str:
    """Stable, filesystem-safe name for a checkpoint's output file."""
    p = ckpt_arg.rstrip("/")
    if os.path.isdir(p):
        return os.path.basename(p)
    parent = os.path.basename(os.path.dirname(p))
    stem = os.path.splitext(os.path.basename(p))[0]
    return f"{parent}_{stem}" if parent else stem


def render_one(ckpt_arg: str, args, plt, device: torch.device) -> None:
    """Load one checkpoint, run it on the chosen utterance, and write its plot."""
    ckpt_file, config_file = resolve_checkpoint_and_config(ckpt_arg, args.config)
    cfg = load_config(config_file)
    sr = cfg["stft_cfg"]["sampling_rate"]
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress = cfg["model_cfg"]["compress_factor"]

    print(f"checkpoint={ckpt_file}\nconfig={config_file}\ndevice={device}")
    model = load_generator(ckpt_file, cfg, device)

    noisy_path, clean_path = resolve_pair(args, cfg)
    if clean_path is None:
        raise SystemExit(f"No clean reference for noisy file {noisy_path!r}")
    print(f"noisy={noisy_path}\nclean={clean_path}")

    noisy_np, _ = librosa.load(noisy_path, sr=sr)
    clean_np, _ = librosa.load(clean_path, sr=sr)
    n = min(len(noisy_np), len(clean_np), int(args.max_seconds * sr))
    noisy_np, clean_np = noisy_np[:n], clean_np[:n]

    noisy_t = torch.from_numpy(noisy_np).float().to(device)
    # Match evaluate.py's per-utterance energy normalization before the STFT.
    norm = torch.sqrt(len(noisy_t) / torch.sum(noisy_t**2.0))
    noisy_in = (noisy_t * norm).unsqueeze(0)

    noisy_mag, noisy_pha, _ = mag_phase_stft(noisy_in, n_fft, hop_size, win_size, compress)
    clean_in = (torch.from_numpy(clean_np).float().to(device)).unsqueeze(0)
    clean_mag, _, _ = mag_phase_stft(clean_in, n_fft, hop_size, win_size, compress)

    with TFActivationCapture(model) as cap, torch.no_grad():
        model(noisy_mag, noisy_pha)
    records = cap.records
    num_layers = len(records)
    print(f"captured {num_layers} TF blocks")

    # Spectrograms: [F, T] display maps, dB on a fixed [-80, 0] scale shared by
    # both panels (common 0 dB reference => clean/noisy directly comparable).
    clean_lin = clean_mag[0].detach().cpu().numpy()
    noisy_lin = noisy_mag[0].detach().cpu().numpy()
    spec_ref = max(clean_lin.max(), noisy_lin.max())
    clean_disp = to_db(clean_lin, spec_ref)
    noisy_disp = to_db(noisy_lin, spec_ref)

    # Build every activation map up front so color limits can be derived from
    # the actual data (and optionally shared across all panels).
    act_maps = []  # [(t_map, f_map), ...] one per TF block
    for after_time, after_freq in records:
        act_maps.append((
            reduce_channels(after_time, args.reduce, args.per_channel_norm),
            reduce_channels(after_freq, args.reduce, args.per_channel_norm),
        ))

    lo, hi = args.clip
    shared_lims = None
    if args.shared_scale:
        pooled = np.concatenate([m.ravel() for pair in act_maps for m in pair])
        shared_lims = (float(np.percentile(pooled, lo)), float(np.percentile(pooled, hi)))

    def act_limits(data):
        # Robust per-map limits so one hot pixel can't wash out the panel.
        if shared_lims is not None:
            return shared_lims
        return float(np.percentile(data, lo)), float(np.percentile(data, hi))

    n_rows = 1 + num_layers
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(11, 2.6 * n_rows), squeeze=False, constrained_layout=True
    )

    extent_spec = [0, n / sr, 0, sr / 2000.0]  # x: seconds, y: kHz

    def show(ax, data, title, cmap, is_spec):
        vmin, vmax = (-80.0, 0.0) if is_spec else act_limits(data)
        im = ax.imshow(
            data, origin="lower", aspect="auto", cmap=cmap,
            extent=extent_spec if is_spec else None, vmin=vmin, vmax=vmax,
        )
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if is_spec:
            ax.set_ylabel("kHz")
        else:
            ax.set_ylabel("freq bin")
        return im

    show(axes[0][0], clean_disp, "clean spectrogram (dB)", args.cmap, True)
    show(axes[0][1], noisy_disp, "noisy spectrogram (dB)", args.cmap, True)

    unit = {"std": "z", "max": "norm", "none": "raw"}[args.per_channel_norm]
    for i, (t_map, f_map) in enumerate(act_maps):
        tag = f"{args.reduce}/{unit}"
        show(axes[i + 1][0], t_map, f"block {i}: after time mixer ({tag})", args.cmap, False)
        show(axes[i + 1][1], f_map, f"block {i}: after freq mixer ({tag})", args.cmap, False)

    axes[-1][0].set_xlabel("time frame")
    axes[-1][1].set_xlabel("time frame")

    fig.suptitle(
        f"TF-block channel activations — {os.path.basename(noisy_path)}\n{ckpt_file}",
        fontsize=11,
    )

    out_path = args.out_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help=".pth file, a checkpoint dir (exp/Run, ckpts/mamba), or a "
                             "parent dir (ckpts/) whose subdirs are checkpoints")
    parser.add_argument("--config", default=None, help="config.yaml (default: next to checkpoint)")
    parser.add_argument("--noisy", default=None, help="Explicit noisy wav (overrides --index)")
    parser.add_argument("--clean", default=None, help="Explicit clean wav (paired with --noisy)")
    parser.add_argument("--index", type=int, default=0, help="Test-set pair index when no --noisy")
    parser.add_argument("--test_clean_json", default=None)
    parser.add_argument("--test_noisy_json", default=None)
    parser.add_argument("--output", default=None,
                        help="Output PNG for a single checkpoint (default: "
                             "<output_dir>/tf_act_<name>.png)")
    parser.add_argument("--output_dir", default="eval_out",
                        help="Directory for per-checkpoint PNGs when several are rendered")
    parser.add_argument("--reduce", choices=["l2", "mean_abs", "median", "max"], default="l2",
                        help="Channel aggregation ('median' is robust to outlier channels)")
    parser.add_argument("--per_channel_norm", choices=["std", "max", "none"], default="std",
                        help="Rescale each channel before aggregating so one loud "
                             "channel can't dominate (default: z-score)")
    parser.add_argument("--clip", type=float, nargs=2, metavar=("LO", "HI"), default=(2.0, 98.0),
                        help="Percentiles for robust color limits, per activation map")
    parser.add_argument("--shared_scale", action="store_true",
                        help="Use one common color scale across all activation panels "
                             "(makes magnitudes comparable between layers)")
    parser.add_argument("--max_seconds", type=float, default=4.0,
                        help="Trim the utterance to at most this many seconds")
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device(args.device)
    checkpoints = expand_checkpoints(args.checkpoint)
    multi = len(checkpoints) > 1
    if multi and args.output:
        print("Ignoring --output for a multi-checkpoint run; using --output_dir instead.")
    print(f"rendering {len(checkpoints)} checkpoint(s): {', '.join(map(run_name_for, checkpoints))}")

    failures = []
    for ckpt_arg in checkpoints:
        if not multi and args.output:
            args.out_path = args.output
        else:
            args.out_path = os.path.join(args.output_dir, f"tf_act_{run_name_for(ckpt_arg)}.png")
        try:
            render_one(ckpt_arg, args, plt, device)
        except Exception as e:  # keep the batch alive if one checkpoint is broken
            failures.append((ckpt_arg, f"{type(e).__name__}: {e}"))
            print(f"FAILED {ckpt_arg}: {type(e).__name__}: {e}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if failures:
        print(f"\n{len(failures)}/{len(checkpoints)} checkpoint(s) failed:")
        for ckpt_arg, msg in failures:
            print(f"  {run_name_for(ckpt_arg)}: {msg}")


if __name__ == "__main__":
    main()
