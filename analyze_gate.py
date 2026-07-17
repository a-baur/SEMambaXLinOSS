"""Visualize the LinOSS/Mamba fusion gate over the input spectrogram.

For a parallel-mode hybrid run (``recipes/hybrid/LinMamParallel.yaml``, e.g.
``exp/LinMamHybrid``) every ``HybridBlock`` fuses a LinOSS and a Mamba backbone
with a learned per-channel gate (``models/mamba_block.py``)::

    x1 = backbone_1(x)   # LinOSS
    x2 = backbone_2(x)   # Mamba
    g  = sigmoid(gate(cat([x1, x2])))
    x  = g * x1 + (1 - g) * x2       # g -> 1 = LinOSS, g -> 0 = Mamba

This script captures ``g`` with forward hooks (no model edits), reduces it over
channels, reshapes it back to the time-frequency plane, and renders it as a
heatmap aligned to the input spectrogram.

Caveat: ``g`` mixes each block's *update*, not the final output -- each block
also carries a residual + ConvTranspose1d projection + skip. The heatmap is a
faithful proxy for "where the block leaned on LinOSS vs Mamba," not a literal
decomposition of the enhanced spectrogram.

Outputs (per utterance, under ``--output_dir``):
  * ``<utt>_summary.png``       -- spectrogram + layer-averaged gate mean/std maps
  * ``<utt>_perlayer.png``      -- per-TFMambaBlock grid (time col, freq col)
  * ``<utt>_corr.png``          -- gate vs spectral-feature correlation matrices
  * ``<utt>_correlations.json`` -- all Pearson r values (frame/freq/bin + per-layer)
  * ``<utt>_gate.npz``          -- raw ``time_map``/``freq_map``/``spectrogram`` +
                                   per-layer stacks, for downstream custom plots
  * ``<utt>_bimodal.png``       -- per-channel gate histograms (with --dump-channels)
  * ``<utt>_ablation.{png,npz,json}`` -- gate pinned to Mamba-only (g=0) vs
                                   LinOSS-only (g=1): output-effect and (with a
                                   clean ref) per-bin error-reduction maps +
                                   per-regime mag/phase losses (with --ablate)
  * ``oscillators.{png,json,npz}`` -- run-level: the backbone_1 LinOSS oscillator
                                   spectrum (resonance freqs/harmonic spacings/Q)
                                   joined with the gate's acoustic-frequency
                                   profile and the speech-F0 present (--oscillators)

Color note: the gate maps cluster near 0.5, so the mean panels use a diverging
map tightened symmetrically about 0.5 (white = balanced) rather than the full
0..1 range -- the annotated ``color [lo, hi]`` range makes the scaling explicit.
"""

import argparse
import contextlib
import json
import math
import os

import librosa
import matplotlib

matplotlib.use("Agg")  # headless: save only, never plt.show()
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from evaluate import build_pair_list, load_generator, resolve_checkpoint_and_config
from models.linoss import LinOSS
from models.loss import phase_losses
from models.mamba_block import HybridBlock
from models.stfts import mag_phase_stft
from utils.util import load_config

# Logit magnitude that drives sigmoid to a hard 0/1 for the branch ablation.
_GATE_OVERRIDE_LOGIT = 30.0


@contextlib.contextmanager
def gate_override(model, branch: str | None):
    """Temporarily pin every ``HybridBlock`` gate to one branch.

    ``branch`` is ``"linoss"`` (g=1 => ``x = backbone_1``), ``"mamba"`` (g=0 =>
    ``x = backbone_2``) or ``None`` (leave the learned gate untouched). Implemented
    with forward hooks on the ``gate`` submodule that *replace* its pre-sigmoid
    logits with +/-BIG, so ``sigmoid`` saturates to 1/0 -- no model edits.
    Assumes backbone_1 = LinOSS, backbone_2 = Mamba (as in LinMamParallel.yaml).
    """
    handles = []
    if branch is not None:
        fill = _GATE_OVERRIDE_LOGIT if branch == "linoss" else -_GATE_OVERRIDE_LOGIT

        def hook(_m, _i, out):
            return torch.full_like(out, fill)

        for tfblock in model.TSMamba:
            for mamba_block in (tfblock.time_mamba, tfblock.freq_mamba):
                for blocks in (mamba_block.forward_blocks, mamba_block.backward_blocks):
                    handles.append(blocks[0].gate.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def register_gate_hooks(model) -> tuple[dict, list]:
    """Hook every ``HybridBlock.gate`` and stash its pre-sigmoid logits.

    Returns ``(store, handles)`` where ``store`` maps
    ``(layer_idx, axis, direction)`` -> captured tensor (filled on forward), and
    ``handles`` are the hook handles to ``.remove()`` afterwards.

    Raises if the run is not a parallel hybrid, so pointing this at a plain
    Mamba/LinOSS checkpoint fails with a clear message rather than an empty plot.
    """
    store: dict = {}
    handles = []

    def make_hook(key):
        def hook(_module, _inp, out):
            store[key] = out.detach().float().cpu()

        return hook

    n_hooked = 0
    for li, tfblock in enumerate(model.TSMamba):
        for axis, mamba_block in (("time", tfblock.time_mamba), ("freq", tfblock.freq_mamba)):
            for direction, blocks in (
                ("fwd", mamba_block.forward_blocks),
                ("bwd", mamba_block.backward_blocks),
            ):
                hb = blocks[0]
                if not (isinstance(hb, HybridBlock) and hb.mode == "parallel"):
                    for h in handles:
                        h.remove()
                    raise ValueError(
                        "This checkpoint is not a parallel hybrid run: "
                        f"TSMamba[{li}].{axis}_mamba.{direction} block is "
                        f"{type(hb).__name__}"
                        + (f" (mode={hb.mode!r})" if isinstance(hb, HybridBlock) else "")
                        + ". analyze_gate.py only applies to 'ssm: hybrid' with "
                        "'mode: parallel' (see recipes/hybrid/LinMamParallel.yaml)."
                    )
                handles.append(hb.gate.register_forward_hook(make_hook((li, axis, direction))))
                n_hooked += 1

    if n_hooked == 0:
        raise ValueError("No TFMambaBlocks found on the model; nothing to hook.")
    return store, handles


def reshape_gate(reduced: torch.Tensor, axis: str, direction: str, T: int) -> torch.Tensor:
    """Fold a channel-reduced gate ``[batch_fold, L]`` back to an ``[F', T]`` map.

    The captured gate is for a *single* (batch=1) utterance:
      * time axis: batch fold is F', sequence is T -> ``view(1, F', T)``.
      * freq axis: batch fold is T, sequence is F' -> ``view(1, T, F')`` then T.
    The backward block ran on a flipped sequence, so its sequence axis is
    reversed and flipped back here.
    """
    if axis == "time":
        m = reduced.view(1, -1, T)[0]  # [F', T]
        if direction == "bwd":
            m = m.flip(-1)  # undo the seq flip along T
    else:  # freq
        m = reduced.view(1, T, -1)[0].transpose(0, 1)  # [T, F'] -> [F', T]
        if direction == "bwd":
            m = m.flip(0)  # undo the seq flip along F'
    return m  # [F', T]


def gate_channel_stats(logits: torch.Tensor, axis: str, direction: str, T: int):
    """Return ``(mean_map, std_map, chan_mean)`` for one captured gate.

    ``mean_map``/``std_map`` are folded to ``[F', T]``:
      * ``mean`` is the per-bin LinOSS share;
      * ``std`` is the dispersion of the gate *across channels* at that bin --
        small std means the whole channel bank agrees on the LinOSS/Mamba split,
        large std means the mean hides a mixed decision.
    ``chan_mean`` is ``[dim]``: each channel's gate averaged over every
    time-frequency position, i.e. that channel's *typical* LinOSS share. Its
    distribution over channels is what the bimodality check inspects (two
    clusters near 0/1 = dedicated LinOSS vs Mamba channels; one broad hump =
    every channel hedges).
    """
    g = torch.sigmoid(logits)  # [batch_fold, L, dim]
    mean_map = reshape_gate(g.mean(dim=-1), axis, direction, T)
    std_map = reshape_gate(g.std(dim=-1), axis, direction, T)
    chan_mean = g.mean(dim=(0, 1))  # [dim]
    return mean_map, std_map, chan_mean


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient (sample-corrected).

    ``BC = (skew^2 + 1) / (kurt + 3(n-1)^2 / ((n-2)(n-3)))``. Ranges in (0, 1];
    the uniform-distribution reference is 5/9 ~ 0.555, so ``BC > 0.555`` is the
    usual "looks bimodal" threshold and a tight unimodal hump sits well below it.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return float("nan")
    d = x - x.mean()
    var = (d**2).mean()
    if var <= 0:
        return float("nan")  # all channels identical -> degenerate, not bimodal
    skew = (d**3).mean() / var**1.5
    kurt = (d**4).mean() / var**2 - 3.0  # excess kurtosis
    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((skew**2 + 1.0) / denom)


def lift_freq(m: torch.Tensor, F_bins: int, T: int) -> np.ndarray:
    """Interpolate an ``[F', T]`` map to the input freq resolution ``[F_bins, T]``."""
    lifted = F.interpolate(m[None, None], size=(F_bins, T), mode="nearest")[0, 0]
    return lifted.numpy()


# ---------------------------------------------------------------------------
# Color scaling -- the gate maps cluster in a narrow band around 0.5, so the
# full 0..1 range washes them out. Tighten the diverging mean maps symmetrically
# around 0.5 (white stays "balanced") and clip the std maps to a robust max.
# ---------------------------------------------------------------------------
def sym_limits(arrays, center: float = 0.5, pct: float = 99.0, min_half: float = 0.04):
    """Symmetric ``(vmin, vmax)`` about ``center`` covering ``pct``% of |x-center|."""
    dev = np.concatenate([np.abs(np.asarray(a, float).ravel() - center) for a in arrays])
    half = max(float(np.percentile(dev, pct)), min_half)
    return center - half, center + half


def robust_vmax(arrays, pct: float = 99.0, floor: float = 1e-6) -> float:
    """Upper color limit at the ``pct``th percentile (ignores a few hot outliers)."""
    v = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    return max(float(np.percentile(v, pct)), floor)


# ---------------------------------------------------------------------------
# Spectral features + gate/feature correlations
# ---------------------------------------------------------------------------
FRAME_FEATURES = (
    "log_energy",
    "centroid",
    "bandwidth",
    "flatness",
    "rolloff",
    "flux",
)
FREQ_FEATURES = ("freq_hz", "time_avg_log_energy")
GATE_MAPS = (
    ("time_map", "time mean"),
    ("time_std_map", "time std"),
    ("freq_map", "freq mean"),
    ("freq_std_map", "freq std"),
)


def pearson(a, b) -> float:
    """Pearson r with guards for length mismatch and zero variance."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size != b.size or a.size < 2 or a.std() <= 0 or b.std() <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spectral_features(mag_lin: np.ndarray, sr: int, n_fft: int) -> dict:
    """Per-frame, per-freq and per-bin spectral descriptors of a linear-mag STFT.

    ``mag_lin`` is ``[F, T]`` linear magnitude (compression undone). Returns a
    dict with keys ``frame`` ([T] each), ``freq`` ([F] each) and ``bin``
    (``logmag`` ``[F, T]``), all aligned to the gate maps (same STFT).
    """
    eps = 1e-10
    S = np.asarray(mag_lin, float)
    logmag = 20.0 * np.log10(S + eps)  # [F, T]

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=n_fft)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=n_fft)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, n_fft=n_fft)[0]
    # Spectral flux: half-wave-rectified frame-to-frame magnitude change.
    diff = np.diff(S, axis=1, prepend=S[:, :1])
    flux = np.sqrt((np.maximum(diff, 0.0) ** 2).sum(axis=0))
    frame_log_energy = 10.0 * np.log10((S**2).mean(axis=0) + eps)

    return {
        "frame": {
            "log_energy": frame_log_energy,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "flatness": flatness,
            "rolloff": rolloff,
            "flux": flux,
        },
        "freq": {
            "freq_hz": librosa.fft_frequencies(sr=sr, n_fft=n_fft),
            "time_avg_log_energy": logmag.mean(axis=1),
        },
        "bin": {"logmag": logmag},
    }


def gate_spectral_correlations(res: dict, feats: dict) -> dict:
    """Pearson correlations between each gate map and the spectral features.

    Three granularities per gate map:
      * ``per_frame`` -- gate averaged over freq ([T]) vs each frame feature;
      * ``per_freq``  -- gate averaged over time ([F]) vs each freq feature;
      * ``per_bin``   -- full ``[F, T]`` gate vs per-bin log-magnitude.
    Plus ``per_layer`` frame/freq correlations to track how the coupling evolves
    with depth. All values are Pearson r (or NaN when a series is degenerate).
    """
    out = {"per_frame": {}, "per_freq": {}, "per_bin": {}, "per_layer": {}}
    for key, _label in GATE_MAPS:
        m = res[key]  # [F, T]
        frame_gate = m.mean(axis=0)  # [T]
        freq_gate = m.mean(axis=1)  # [F]
        out["per_frame"][key] = {
            f: pearson(frame_gate, feats["frame"][f]) for f in FRAME_FEATURES
        }
        out["per_freq"][key] = {
            f: pearson(freq_gate, feats["freq"][f]) for f in FREQ_FEATURES
        }
        out["per_bin"][key] = {"logmag": pearson(m, feats["bin"]["logmag"])}

    # Per-layer (mean gate only), to see the coupling's depth trend.
    for axis, stack_key in (("time", "time_layers"), ("freq", "freq_layers")):
        stack = res[stack_key]  # [L, F, T]
        out["per_layer"][axis] = {
            "frame": {
                f: [pearson(stack[li].mean(axis=0), feats["frame"][f]) for li in range(len(stack))]
                for f in FRAME_FEATURES
            },
            "freq": {
                f: [pearson(stack[li].mean(axis=1), feats["freq"][f]) for li in range(len(stack))]
                for f in FREQ_FEATURES
            },
        }
    return out


def _round_nested(obj, ndigits: int = 3):
    """Recursively round floats in nested dict/list for compact JSON."""
    if isinstance(obj, dict):
        return {k: _round_nested(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_nested(v, ndigits) for v in obj]
    if isinstance(obj, float):
        return None if math.isnan(obj) else round(obj, ndigits)
    return obj


@torch.no_grad()
def analyze_utterance(model, store, noisy_wav, cfg, device, clean_wav=None):
    """Run one waveform through the model and build the aligned gate maps.

    Returns a dict with:
      * ``spectrogram``       -- input (noisy) log-mag ``[F, T]``
      * ``clean_spectrogram`` -- clean-reference log-mag ``[F, T]`` (same
        normalization as the noisy input), or ``None`` if no ``clean_wav``
      * ``time_map``/``freq_map`` -- layer+direction averaged ``[F, T]`` LinOSS share
      * ``time_std_map``/``freq_std_map`` -- layer+direction averaged across-channel std
      * ``time_layers``/``freq_layers`` -- per-layer fwd/bwd-averaged mean ``[L, F, T]``
      * ``time_std_layers``/``freq_std_layers`` -- per-layer fwd/bwd-averaged std ``[L, F, T]``
      * ``time_chan``/``freq_chan`` -- per-layer, fwd/bwd-averaged per-channel
        LinOSS share ``[L, dim]`` (for the bimodality check)
      * ``features`` -- spectral descriptors aligned to the gate maps
        (:func:`spectral_features`), for the correlation analysis
    """
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress = cfg["model_cfg"]["compress_factor"]
    sr = cfg["stft_cfg"]["sampling_rate"]

    norm = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav**2.0))
    noisy = (noisy_wav * norm).unsqueeze(0).to(device)

    noisy_mag, noisy_pha, _ = mag_phase_stft(noisy, n_fft, hop_size, win_size, compress)
    F_bins, T = noisy_mag.shape[1], noisy_mag.shape[2]

    # Linear magnitude (undo the ^compress) for physically-meaningful features.
    mag_lin = noisy_mag[0].cpu().numpy().astype(np.float64) ** (1.0 / compress)
    features = spectral_features(mag_lin, sr, n_fft)

    # Clean reference spectrogram in the SAME normalized/compressed domain as the
    # noisy input, so the two panels are directly comparable on one dB scale.
    clean_spectrogram = None
    if clean_wav is not None:
        clean = (clean_wav.to(device) * norm).unsqueeze(0)
        clean_mag, _, _ = mag_phase_stft(clean, n_fft, hop_size, win_size, compress)
        clean_spectrogram = 20.0 * np.log10(clean_mag[0].cpu().numpy() + 1e-8)

    store.clear()
    model(noisy_mag, noisy_pha)  # populates store via hooks

    n_layers = len(model.TSMamba)
    # Average fwd/bwd within each (layer, axis) for both channel-mean and
    # channel-std, then lift F' -> F.
    layer_maps = {"time": {"mean": [], "std": [], "chan": []}, "freq": {"mean": [], "std": [], "chan": []}}
    for axis in ("time", "freq"):
        for li in range(n_layers):
            mean_dirs, std_dirs, chan_dirs = [], [], []
            for d in ("fwd", "bwd"):
                mean_m, std_m, chan_m = gate_channel_stats(store[(li, axis, d)], axis, d, T)
                mean_dirs.append(mean_m)
                std_dirs.append(std_m)
                chan_dirs.append(chan_m)
            mean_avg = torch.stack(mean_dirs, 0).mean(0)  # [F', T]
            std_avg = torch.stack(std_dirs, 0).mean(0)  # [F', T]
            chan_avg = torch.stack(chan_dirs, 0).mean(0)  # [dim]
            layer_maps[axis]["mean"].append(lift_freq(mean_avg, F_bins, T))
            layer_maps[axis]["std"].append(lift_freq(std_avg, F_bins, T))
            layer_maps[axis]["chan"].append(chan_avg.numpy())

    time_layers = np.stack(layer_maps["time"]["mean"], 0)  # [L, F, T]
    freq_layers = np.stack(layer_maps["freq"]["mean"], 0)
    time_std_layers = np.stack(layer_maps["time"]["std"], 0)
    freq_std_layers = np.stack(layer_maps["freq"]["std"], 0)
    time_chan = np.stack(layer_maps["time"]["chan"], 0)  # [L, dim]
    freq_chan = np.stack(layer_maps["freq"]["chan"], 0)

    return {
        "spectrogram": 20.0 * np.log10(noisy_mag[0].cpu().numpy() + 1e-8),
        "clean_spectrogram": clean_spectrogram,
        "time_map": time_layers.mean(0),
        "freq_map": freq_layers.mean(0),
        "time_std_map": time_std_layers.mean(0),
        "freq_std_map": freq_std_layers.mean(0),
        "time_layers": time_layers,
        "freq_layers": freq_layers,
        "time_std_layers": time_std_layers,
        "freq_std_layers": freq_std_layers,
        "time_chan": time_chan,
        "freq_chan": freq_chan,
        "features": features,
    }


def plot_summary(res, out_path, title):
    """Spectrogram(s) + per-bin channel-mean and channel-std gate maps.

    Stacked panels sharing the time axis: input (noisy) spectrogram, an optional
    clean-reference spectrogram, then the channel-*mean* (LinOSS share) and
    channel-*std* (cross-channel dispersion) for the time mixer, then the same
    pair for the freq mixer. Every panel -- spectrograms included -- carries its
    own colorbar (the spectrograms get a dB bar) so all panels share one width.
    """
    # Tighten color ranges to the data: mean maps symmetric about 0.5 (white =
    # balanced), std maps clipped to a robust max -- both shared across time/freq
    # so the panels stay comparable.
    m_lo, m_hi = sym_limits([res["time_map"], res["freq_map"]])
    std_vmax = robust_vmax([res["time_std_map"], res["freq_std_map"]])

    # Spectrogram panels (noisy, and clean if available), on one shared dB scale.
    specs = [("spectrogram", "input (noisy) log-mag spectrogram")]
    if res.get("clean_spectrogram") is not None:
        specs.append(("clean_spectrogram", "clean-reference log-mag spectrogram"))
    spec_stack = np.concatenate([res[k].ravel() for k, _ in specs])
    s_lo, s_hi = np.percentile(spec_stack, [1.0, 99.0])

    n_panels = len(specs) + 4
    fig, axs = plt.subplots(n_panels, 1, figsize=(11, 2.8 * n_panels), sharex=True)
    for i, (key, name) in enumerate(specs):
        im = axs[i].imshow(
            res[key], origin="lower", aspect="auto", cmap="magma", vmin=s_lo, vmax=s_hi
        )
        axs[i].set_title(f"{title}  --  {name}" if i == 0 else name)
        axs[i].set_ylabel("freq bin")
        fig.colorbar(im, ax=axs[i], extend="both").set_label("dB")

    off = len(specs)
    panels = (
        (axs[off + 0], res["time_map"], "time-mixer gate: channel mean", "mean", "coolwarm", m_lo, m_hi),
        (axs[off + 1], res["time_std_map"], "time-mixer gate: channel std", "std", "viridis", 0.0, std_vmax),
        (axs[off + 2], res["freq_map"], "freq-mixer gate: channel mean", "mean", "coolwarm", m_lo, m_hi),
        (axs[off + 3], res["freq_std_map"], "freq-mixer gate: channel std", "std", "viridis", 0.0, std_vmax),
    )
    for ax, m, name, kind, cmap, vmin, vmax in panels:
        im = ax.imshow(m, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        rng = f"[{vmin:.2f}, {vmax:.2f}]"
        ax.set_title(f"{name}  (mean over layers & fwd/bwd; color {rng})")
        ax.set_ylabel("freq bin")
        cb = fig.colorbar(im, ax=ax, extend="both" if kind == "mean" else "max")
        cb.set_label(
            "1 = LinOSS   /   0 = Mamba" if kind == "mean" else "cross-channel std (0 = agree)"
        )
    axs[-1].set_xlabel("time frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_perlayer(res, out_path, title):
    """Grid: one row per TFMambaBlock, time col + freq col (fwd/bwd averaged)."""
    n_layers = res["time_layers"].shape[0]
    # Shared symmetric range about 0.5 across all layers/axes for contrast.
    v_lo, v_hi = sym_limits([res["time_layers"], res["freq_layers"]])
    fig, axs = plt.subplots(
        n_layers, 2, figsize=(11, 2.4 * n_layers + 0.6), sharex=True, squeeze=False
    )
    im = None
    for li in range(n_layers):
        for col, (key, name) in enumerate(
            ((("time_layers"), "time"), (("freq_layers"), "freq"))
        ):
            ax = axs[li][col]
            im = ax.imshow(
                res[key][li], origin="lower", aspect="auto", cmap="coolwarm", vmin=v_lo, vmax=v_hi
            )
            ax.set_ylabel(f"L{li}\nfreq bin")
            if li == 0:
                ax.set_title(f"{name}-mixer gate")
    for col in range(2):
        axs[-1][col].set_xlabel("time frame")
    fig.suptitle(
        f"{title}  --  per-layer gate  (1 = LinOSS / 0 = Mamba; color [{v_lo:.2f}, {v_hi:.2f}])"
    )
    fig.tight_layout(rect=(0, 0, 0.92, 1.0))
    cbar_ax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
    fig.colorbar(im, cax=cbar_ax, extend="both").set_label("1 = LinOSS   /   0 = Mamba")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_bimodal(res, out_path, title):
    """Per-channel gate histograms -- the bimodality check.

    Grid of one row per TFMambaBlock, time col + freq col. Each cell is the
    distribution over the ``dim`` channels of that channel's mean LinOSS share
    (0 = always Mamba, 1 = always LinOSS). Two clusters near the edges => the
    network has dedicated LinOSS vs Mamba channels; a single central hump =>
    every channel hedges. Sarle's BC (>0.555 ~ bimodal) is annotated per cell.
    """
    tc, fc = res["time_chan"], res["freq_chan"]  # [L, dim]
    n_layers = tc.shape[0]
    bins = np.linspace(0.0, 1.0, 26)
    fig, axs = plt.subplots(
        n_layers, 2, figsize=(10, 2.2 * n_layers + 0.6), sharex=True, squeeze=False
    )
    for li in range(n_layers):
        for col, (arr, name, color) in enumerate(
            ((tc, "time", "#4c72b0"), (fc, "freq", "#dd8452"))
        ):
            ax = axs[li][col]
            vals = arr[li]
            ax.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=0.4)
            ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8)
            bc = bimodality_coefficient(vals)
            ax.set_title(
                f"{name}-mixer  L{li}   mean={vals.mean():.2f}  BC={bc:.2f}", fontsize=9
            )
            ax.set_ylabel("channels")
    for col in range(2):
        axs[-1][col].set_xlabel("per-channel LinOSS share  (0 = Mamba, 1 = LinOSS)")
    fig.suptitle(f"{title}  --  per-channel gate distribution  (BC > 0.555 ~ bimodal)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _annotated_matrix(ax, M, row_labels, col_labels, title):
    """Draw a correlation matrix (RdBu_r, -1..1) with per-cell value labels."""
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "n/a" if np.isnan(v) else f"{v:+.2f}"
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=7,
                color="white" if not np.isnan(v) and abs(v) > 0.55 else "black",
            )
    return im


def plot_correlations(corr, out_path, title):
    """Two annotated matrices: gate-vs-frame-feature and gate-vs-freq/bin-feature."""
    rows = [label for _key, label in GATE_MAPS]
    frame_cols = list(FRAME_FEATURES)
    freqbin_cols = list(FREQ_FEATURES) + ["logmag (per-bin)"]

    M1 = np.array(
        [[corr["per_frame"][k][f] for f in FRAME_FEATURES] for k, _ in GATE_MAPS], float
    )
    M2 = np.array(
        [
            [corr["per_freq"][k][f] for f in FREQ_FEATURES] + [corr["per_bin"][k]["logmag"]]
            for k, _ in GATE_MAPS
        ],
        float,
    )

    fig, axs = plt.subplots(
        1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [len(frame_cols), len(freqbin_cols)]}
    )
    _annotated_matrix(axs[0], M1, rows, frame_cols, "gate (freq-averaged) vs per-frame feature")
    im = _annotated_matrix(
        axs[1], M2, rows, freqbin_cols, "gate vs per-freq / per-bin feature"
    )
    fig.suptitle(f"{title}  --  gate / spectral-feature correlation (Pearson r)")
    fig.tight_layout(rect=(0, 0, 0.95, 0.96))
    cbar_ax = fig.add_axes((0.96, 0.15, 0.012, 0.7))
    fig.colorbar(im, cax=cbar_ax).set_label("Pearson r")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def ablate_utterance(model, noisy_wav, clean_wav, cfg, device):
    """Rerun the model with the gate pinned to each branch; measure the effect.

    Produces, in the compressed-STFT domain the model works in:
      * ``logmag`` per regime (``learned``/``mamba``/``linoss``), ``[F, T]``;
      * ``add_effect_db`` = dB(learned) - dB(mamba): what *adding* LinOSS does to
        the output magnitude, spatially;
      * ``branch_diff_db`` = dB(linoss) - dB(mamba): how differently the two
        branches would denoise on their own.
    When ``clean_wav`` is given it also returns error-reduction maps and per-regime
    magnitude/phase losses (the causal "where does LinOSS help" signal):
      * ``benefit_add`` = err(mamba) - err(learned): positive where the learned
        gate (i.e. having LinOSS available) lowers the per-bin magnitude error.
    """
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress = cfg["model_cfg"]["compress_factor"]
    eps = 1e-8

    norm = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav**2.0))
    noisy = (noisy_wav * norm).unsqueeze(0).to(device)
    noisy_mag, noisy_pha, _ = mag_phase_stft(noisy, n_fft, hop_size, win_size, compress)

    regimes = {"learned": None, "mamba": "mamba", "linoss": "linoss"}
    mags, phas = {}, {}
    for name, branch in regimes.items():
        with gate_override(model, branch):
            mag_g, pha_g, _ = model(noisy_mag, noisy_pha)
        mags[name] = mag_g[0]  # [F, T]
        phas[name] = pha_g[0]

    def db(x):
        return 20.0 * np.log10(x.cpu().numpy() + eps)

    res = {
        "noisy_logmag": db(noisy_mag[0]),
        "logmag": {k: db(v) for k, v in mags.items()},
        "add_effect_db": db(mags["learned"]) - db(mags["mamba"]),
        "branch_diff_db": db(mags["linoss"]) - db(mags["mamba"]),
        "has_clean": clean_wav is not None,
    }

    if clean_wav is None:
        return res

    # Clean reference in the SAME normalized/compressed domain as the outputs.
    clean = (clean_wav.to(device) * norm).unsqueeze(0)
    clean_mag, clean_pha, _ = mag_phase_stft(clean, n_fft, hop_size, win_size, compress)
    T = min(clean_mag.shape[-1], noisy_mag.shape[-1])
    clean_mag_t, clean_pha_t = clean_mag[..., :T], clean_pha[..., :T]

    err, metrics = {}, {}
    for name in regimes:
        m = mags[name][..., :T].unsqueeze(0)  # [1, F, T]
        p = phas[name][..., :T].unsqueeze(0)
        err[name] = ((m[0] - clean_mag_t[0]) ** 2).cpu().numpy()  # [F, T]
        ip, gd, iaf = phase_losses(clean_pha_t, p, cfg)
        ip, gd, iaf = ip.item(), gd.item(), iaf.item()
        metrics[name] = {
            "mag_mse": F.mse_loss(m, clean_mag_t).item(),
            "phase_ip": ip,
            "phase_gd": gd,
            "phase_iaf": iaf,
            "phase_sum": ip + gd + iaf,
        }

    # Positive => LinOSS availability helps (Mamba-only error minus learned error).
    benefit = {
        "add_mag_mse": metrics["mamba"]["mag_mse"] - metrics["learned"]["mag_mse"],
        "add_phase_sum": metrics["mamba"]["phase_sum"] - metrics["learned"]["phase_sum"],
        "linoss_only_mag_mse": metrics["mamba"]["mag_mse"] - metrics["linoss"]["mag_mse"],
        "linoss_only_phase_sum": metrics["mamba"]["phase_sum"] - metrics["linoss"]["phase_sum"],
    }
    res.update(
        {
            "clean_logmag": db(clean_mag_t[0]),
            "err": {k: v for k, v in err.items()},
            "benefit_add": err["mamba"] - err["learned"],  # >0 where LinOSS helps
            "benefit_linoss_only": err["mamba"] - err["linoss"],
            "metrics": metrics,
            "benefit": benefit,
        }
    )
    return res


def _sym0(a, pct: float = 99.0, floor: float = 1e-9) -> float:
    """Symmetric-about-0 color half-range at a robust percentile of |a|."""
    return max(float(np.percentile(np.abs(np.asarray(a, float)), pct)), floor)


def plot_ablation(abl, out_path, title):
    """Spatially-resolved branch-ablation panels (+ metrics banner if clean)."""
    has_clean = abl["has_clean"]
    add_v = _sym0(abl["add_effect_db"])
    rows = [("noisy_logmag", "noisy log-mag spectrogram", "magma", None, None)]
    if has_clean:
        # Noisy + clean share one dB scale so the reference is directly comparable.
        spec_stack = np.concatenate([abl["noisy_logmag"].ravel(), abl["clean_logmag"].ravel()])
        s_lo, s_hi = np.percentile(spec_stack, [1.0, 99.0])
        rows[0] = ("noisy_logmag", "noisy log-mag spectrogram", "magma", s_lo, s_hi)
        rows.append(("clean_logmag", "clean-reference log-mag spectrogram", "magma", s_lo, s_hi))
    rows.append(
        ("add_effect_db", "LinOSS effect on output: dB(learned) - dB(Mamba-only)",
         "coolwarm", -add_v, add_v)
    )
    if has_clean:
        ben_v = _sym0(abl["benefit_add"])
        rows.append(
            ("benefit_add", "LinOSS magnitude benefit: err(Mamba-only) - err(learned)  (red = LinOSS helps)",
             "RdBu_r", -ben_v, ben_v)
        )

    fig, axs = plt.subplots(len(rows), 1, figsize=(11, 2.9 * len(rows) + 0.8), sharex=True)
    if len(rows) == 1:
        axs = [axs]
    for ax, (key, name, cmap, vmin, vmax) in zip(axs, rows):
        im = ax.imshow(abl[key], origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("freq bin")
        fig.colorbar(im, ax=ax)
    axs[-1].set_xlabel("time frame")

    banner = f"{title}  --  branch ablation (g=0 Mamba-only / g=1 LinOSS-only)"
    if has_clean:
        m = abl["metrics"]
        b = abl["benefit"]
        banner += (
            f"\nmag_MSE  learned={m['learned']['mag_mse']:.4f}  mamba={m['mamba']['mag_mse']:.4f}  "
            f"linoss={m['linoss']['mag_mse']:.4f}   |   "
            f"phase_sum  learned={m['learned']['phase_sum']:.3f}  mamba={m['mamba']['phase_sum']:.3f}  "
            f"linoss={m['linoss']['phase_sum']:.3f}"
            f"\nLinOSS benefit (Mamba-only - learned; >0 helps):  "
            f"mag_MSE={b['add_mag_mse']:+.4f}   phase_sum={b['add_phase_sum']:+.3f}"
        )
    fig.suptitle(banner, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if has_clean else 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LinOSS oscillator spectrum vs gate frequency profile (the "why LinOSS helps"
# join): the backbone_1 LinOSS in every HybridBlock is a bank of damped
# oscillators. We reconstruct each mode's discrete eigenvalue lambda = rho*e^{iw}
# EXACTLY as the forward pass constrains (A, G) (the a_from_g stability
# projection), read off its resonance, and overlay it on where the gate actually
# routes to LinOSS across the acoustic spectrum.
# ---------------------------------------------------------------------------
def _linoss_modes(ln: LinOSS):
    """Per-mode ``(omega [rad/step], rho)`` for a damped-IMEX LinOSS bank.

    Rebuilds the effective ``A``/``G`` with the same relu + a_from_g clamping the
    forward pass uses, forms the 2x2 IMEX transition ``M`` per mode, and returns
    its complex eigenvalue ``lambda = rho * e^{i*omega}`` via ``det = 1/(1+dt*G)``
    (= ``|lambda|^2``) and ``trace = M11 + M22``. All modes are oscillatory by
    construction (the projection keeps ``A`` in the complex-root band), so
    ``cos(omega) = trace / (2*sqrt(det))`` is always in ``[-1, 1]``.
    """
    if not (ln.damping and ln.discretization == "IMEX" and ln.A_from_G):
        raise ValueError(
            "oscillator readout supports the damped-IMEX LinOSS (a_from_g) "
            "backbone only; got "
            f"discretization={ln.discretization!r} damping={ln.damping} a_from_g={ln.A_from_G}"
        )
    with torch.no_grad():
        steps = torch.sigmoid(ln.steps)
        G = torch.relu(ln.G_diag)
        sq = torch.sqrt(1.0 + steps * G)
        A_low = (2.0 + steps * G - 2.0 * sq) / steps**2
        A_high = (2.0 + steps * G + 2.0 * sq) / steps**2
        A = A_low + torch.relu(ln.A_diag - A_low) - torch.relu(ln.A_diag - A_high)
        det = 1.0 / (1.0 + steps * G)  # |lambda|^2
        trace = det + (1.0 - steps**2 * det * A)  # M11 + M22
        cos_w = (trace / (2.0 * torch.sqrt(det))).clamp(-1.0, 1.0)
        omega = torch.arccos(cos_w)
        rho = torch.sqrt(det)
    return omega.cpu().numpy(), rho.cpu().numpy()


def linoss_oscillators(model, cfg, f_prime: int) -> dict:
    """Pool every backbone_1 LinOSS bank per axis and map modes to physics.

    Returns ``{'time': {...}, 'freq': {...}, 'meta': {...}}``. Each axis dict has
    ``omega``/``rho``/``Q`` and ``phys_hz``:
      * time mixer -- one step = one STFT frame, so ``phys_hz`` is a *temporal
        modulation* frequency ``omega/2pi * frame_rate`` (Nyquist = frame_rate/2);
      * freq mixer -- one step = one (downsampled) frequency bin, so the mode is
        a comb along frequency; ``phys_hz`` is its *harmonic spacing*
        ``(2pi/omega) * df'`` ~ the F0 it resonates with (min = Nyquist 2 bins).
    ``Q = omega / (2*decay)`` (decay = -ln rho) is the resonance sharpness.
    """
    sr = cfg["stft_cfg"]["sampling_rate"]
    hop = cfg["stft_cfg"]["hop_size"]
    frame_rate = sr / hop
    df_prime = (sr / 2.0) / (f_prime - 1)  # downsampled freq-bin width (Hz)

    out: dict = {}
    for axis, attr in (("time", "time_mamba"), ("freq", "freq_mamba")):
        ws, rhos = [], []
        for li in range(len(model.TSMamba)):
            mb = getattr(model.TSMamba[li], attr)
            for blocks in (mb.forward_blocks, mb.backward_blocks):
                bb = blocks[0].backbone_1
                ln = getattr(bb, "ssm", None)
                if not isinstance(ln, LinOSS):
                    raise ValueError(
                        f"TSMamba[{li}].{attr} backbone_1 is not a LinOSS "
                        f"(got {type(bb).__name__}); oscillator readout expects "
                        "LinMamParallel-style hybrid (backbone_1 = LinOSS)."
                    )
                w, rho = _linoss_modes(ln)
                ws.append(w)
                rhos.append(rho)
        w = np.concatenate(ws)
        rho = np.concatenate(rhos)
        decay = -np.log(np.clip(rho, 1e-12, 1.0))  # per-step decay rate
        Q = np.where(decay > 0, w / (2.0 * decay), np.inf)
        rec = {"omega": w, "rho": rho, "Q": Q}
        if axis == "time":
            rec["phys_hz"] = w / (2.0 * np.pi) * frame_rate
        else:
            rec["phys_hz"] = np.where(
                w > 1e-6, (2.0 * np.pi / np.maximum(w, 1e-12)) * df_prime, np.inf
            )
        out[axis] = rec

    out["meta"] = {
        "frame_rate_hz": float(frame_rate),
        "df_prime_hz": float(df_prime),
        "f_prime": int(f_prime),
        "nyquist_spacing_hz": float(2.0 * df_prime),
        "n_modes_per_axis": int(w.size),
    }
    return out


def speech_f0(
    wav_np: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 500.0,
    frame_length: int = 1024,
    hop: int = 256,
) -> np.ndarray:
    """Voiced-frame F0 (Hz) via a pure-numpy cepstral pitch detector.

    Deliberately avoids ``librosa.pyin`` (its numba ``localmin`` gufunc segfaults
    in this environment). For each Hann-windowed frame it takes the real cepstrum
    ``irfft(log|rfft|)`` and picks the peak quefrency in the ``[sr/fmax, sr/fmin]``
    lag band; a frame is called voiced when that peak is prominent (>3 sigma above
    the band mean) and the frame has enough energy. Approximate, but adequate for
    the distribution overlay -- returns the voiced F0 samples (empty if none).
    """
    x = np.asarray(wav_np, dtype=np.float64)
    lag_min = int(np.floor(sr / fmax))
    lag_max = min(int(np.ceil(sr / fmin)), frame_length - 1)
    if x.size < frame_length or lag_max <= lag_min:
        return np.array([])

    win = np.hanning(frame_length)
    starts = range(0, x.size - frame_length + 1, hop)
    f0s, proms, energies = [], [], []
    for s in starts:
        frame = x[s : s + frame_length] * win
        energies.append(float(np.sqrt(np.mean(frame**2))))
        spec = np.abs(np.fft.rfft(frame)) + 1e-10
        cep = np.fft.irfft(np.log(spec))
        seg = cep[lag_min : lag_max + 1]
        k = int(np.argmax(seg))
        f0s.append(sr / (lag_min + k))
        proms.append((seg[k] - seg.mean()) / (seg.std() + 1e-9))

    f0s, proms, energies = np.array(f0s), np.array(proms), np.array(energies)
    pos = energies[energies > 0]
    if pos.size == 0:
        return np.array([])
    voiced = (energies > 0.1 * np.median(pos)) & (proms > 3.0)
    return f0s[voiced]


def plot_oscillators(osc, gate_prof, f0_samples, out_path, title):
    """Join the LinOSS oscillator spectrum with the gate's frequency profile.

    Rows: (0-left) freq-mixer harmonic-spacing density overlaid on the speech-F0
    density -- do the oscillators resonate where speech pitch is? (0-right) time-
    mixer temporal-modulation density; (1) per-axis resonance sharpness Q; (2) the
    gate's LinOSS share vs acoustic frequency with mean input log-energy for
    reference -- where the model actually routes to LinOSS.
    """
    meta = osc["meta"]
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.8, 1.1], hspace=0.5, wspace=0.22)

    # Row 0 left -- the join: freq-osc harmonic spacing vs speech F0 (shared Hz).
    ax = fig.add_subplot(gs[0, 0])
    sp = osc["freq"]["phys_hz"]
    sp = sp[np.isfinite(sp)]
    disp_max = 700.0
    over = float((sp > disp_max).mean())  # near-DC / very-wide-comb modes off-scale
    bins = np.linspace(0.0, disp_max, 60)
    # No clipping: hist drops out-of-range values, so the shown [0, 700] Hz band
    # keeps its true shape instead of piling the tail into the last bin.
    ax.hist(
        sp, bins=bins, density=True, color="#4c72b0", alpha=0.8,
        label=f"LinOSS freq-osc spacing (n={sp.size}, {over * 100:.0f}% >{disp_max:.0f})",
    )
    if f0_samples.size:
        ax.hist(
            f0_samples, bins=bins, density=True, histtype="step",
            color="#c44e52", lw=2.0, label=f"speech F0, voiced (n={f0_samples.size})",
        )
    ax.axvspan(80, 400, color="k", alpha=0.06)
    ax.axvline(meta["nyquist_spacing_hz"], color="gray", ls=":", lw=1.2)
    ax.text(
        meta["nyquist_spacing_hz"] + 6, ax.get_ylim()[1] * 0.55, "Nyquist\n(2 bins)",
        fontsize=7, color="gray",
    )
    frac = float(((sp >= 80) & (sp <= 400)).mean())
    ax.set_title(
        f"freq-mixer oscillators: harmonic-comb resonance\n"
        f"{frac * 100:.0f}% resonate in speech-F0 band [80, 400] Hz",
        fontsize=10,
    )
    ax.set_xlabel("resonant harmonic spacing  ~ F0  (Hz)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)

    # Row 0 right -- time-mixer temporal modulation rate.
    ax = fig.add_subplot(gs[0, 1])
    tf = osc["time"]["phys_hz"]
    nyq = meta["frame_rate_hz"] / 2.0
    ax.hist(
        tf, bins=np.linspace(0.0, nyq, 60), density=True, color="#55a868", alpha=0.85,
        label=f"LinOSS time-osc (n={tf.size})",
    )
    ax.axvspan(2, 16, color="k", alpha=0.06)
    ax.text(9, ax.get_ylim()[1] * 0.9, "syllabic\n2-16 Hz", fontsize=7, ha="center")
    ax.set_title(
        f"time-mixer oscillators: temporal-modulation rate\n"
        f"(frame rate {meta['frame_rate_hz']:.0f} Hz, Nyquist {nyq:.0f} Hz)",
        fontsize=10,
    )
    ax.set_xlabel("modulation frequency (Hz)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)

    # Row 1 -- resonance sharpness Q per axis.
    for col, (axis, color, name) in enumerate(
        (("freq", "#4c72b0", "freq"), ("time", "#55a868", "time"))
    ):
        ax = fig.add_subplot(gs[1, col])
        Q = osc[axis]["Q"]
        Q = Q[np.isfinite(Q)]
        qmax = float(np.percentile(Q, 97)) if Q.size else 1.0
        # range= drops the long tail rather than clipping it into the last bin.
        ax.hist(Q, bins=np.linspace(0.0, qmax, 50), color=color, alpha=0.85)
        ax.axvline(0.5, color="k", ls="--", lw=0.8)  # Q=0.5 = critically damped
        ax.set_title(
            f"{name}-mixer oscillator sharpness Q  (median {np.median(Q):.2f})", fontsize=10
        )
        ax.set_xlabel("Q = omega / (2*decay)   (higher = sharper / longer ringing)")
        ax.set_ylabel("modes")

    # Row 2 -- gate LinOSS share vs acoustic frequency (+ input energy reference).
    ax = fig.add_subplot(gs[2, :])
    fr = gate_prof["freqs"]
    ax.plot(fr, gate_prof["time_share"], color="#55a868", lw=2, label="time-mixer LinOSS share")
    ax.plot(fr, gate_prof["freq_share"], color="#4c72b0", lw=2, label="freq-mixer LinOSS share")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, float(fr[-1]))
    ax.set_xlabel("acoustic frequency (Hz)")
    ax.set_ylabel("LinOSS share (gate)")
    ax.set_title(
        f"gate frequency profile (avg over {gate_prof['n_utt']} utt): "
        "where the model routes to LinOSS across the spectrum",
        fontsize=10,
    )
    ax2 = ax.twinx()
    ax2.fill_between(fr, gate_prof["logenergy"], color="gray", alpha=0.15)
    ax2.plot(fr, gate_prof["logenergy"], color="gray", lw=1, alpha=0.5)
    ax2.set_ylabel("mean input log-energy (dB)", color="gray")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"{title}  --  LinOSS oscillator spectrum vs gate frequency profile", fontsize=12
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def select_pairs(args, cfg):
    """Return ``[(noisy_path, clean_path_or_None), ...]``.

    Explicit ``--input`` wavs (optionally paired positionally with ``--clean``),
    else the first ``--num`` (noisy, clean) pairs from the test set -- reusing
    ``evaluate.build_pair_list`` so the clean matching is identical to eval.
    """
    if args.input:
        if args.clean and len(args.clean) != len(args.input):
            raise SystemExit("--clean must list the same number of paths as --input")
        cleans = args.clean if args.clean else [None] * len(args.input)
        return list(zip(args.input, cleans))
    pairs = build_pair_list(cfg, None, None)  # [(noisy, clean)], sorted by basename
    return pairs[: args.num]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="exp/LinMamHybrid",
        help="Path to a .pth file or an exp directory containing g_*.pth.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config. Defaults to config.yaml next to the checkpoint.",
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Explicit noisy wav path(s). If omitted, first --num from the test set.",
    )
    parser.add_argument(
        "--clean",
        nargs="*",
        default=None,
        help="Clean reference wav(s) paired positionally with --input (for --ablate "
        "benefit maps). Ignored in test-set mode, which pairs clean automatically.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=3,
        help="Number of test-set utterances to analyze when --input is not given.",
    )
    parser.add_argument("--output_dir", default="eval_out/gate_analysis")
    parser.add_argument(
        "--dump-channels",
        action="store_true",
        help="Also save per-channel gate vectors to the npz and write a "
        "<utt>_bimodal.png histogram (bimodality check).",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Also rerun the model with the gate pinned to Mamba-only (g=0) and "
        "LinOSS-only (g=1); write <utt>_ablation.png/.npz/.json. With a clean "
        "reference this gives the causal per-bin 'where LinOSS helps' map.",
    )
    parser.add_argument(
        "--oscillators",
        action="store_true",
        help="Also extract the backbone_1 LinOSS oscillator spectrum (resonance "
        "frequencies, harmonic spacings, Q) and join it with the gate's acoustic-"
        "frequency profile (avg over the analyzed utterances). Writes one "
        "oscillators.{png,json,npz} for the run.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_file, config_file = resolve_checkpoint_and_config(args.checkpoint, args.config)
    print(f"Checkpoint : {ckpt_file}")
    print(f"Config     : {config_file}")

    cfg = load_config(config_file)
    sr = cfg["stft_cfg"]["sampling_rate"]
    model = load_generator(ckpt_file, cfg, device)

    store, handles = register_gate_hooks(model)
    print(f"Hooked {len(handles)} gates ({len(model.TSMamba)} TF blocks x time/freq x fwd/bwd)")

    pairs = select_pairs(args, cfg)
    if not pairs:
        raise SystemExit("No input utterances resolved (empty --input and empty test set).")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Analyzing {len(pairs)} utterance(s) at {sr} Hz -> {args.output_dir}\n")

    # Run-level oscillator/gate-profile accumulator (populated when --oscillators).
    osc_acc = (
        {"time_share": 0.0, "freq_share": 0.0, "logenergy": 0.0, "n": 0,
         "f0": [], "freqs": None, "f_prime": None}
        if args.oscillators
        else None
    )

    try:
        for path, clean_path in pairs:
            base = os.path.splitext(os.path.basename(path))[0]
            noisy_np, _ = librosa.load(path, sr=sr)
            noisy_t = torch.from_numpy(noisy_np).float()
            clean_t = None
            if clean_path is not None:
                clean_np, _ = librosa.load(clean_path, sr=sr)
                clean_t = torch.from_numpy(clean_np).float()
            res = analyze_utterance(model, store, noisy_t, cfg, device, clean_wav=clean_t)

            if osc_acc is not None:
                # Accumulate the gate's time-averaged acoustic-frequency profile
                # and the speech pitch actually present (clean if available).
                osc_acc["time_share"] += res["time_map"].mean(axis=1)
                osc_acc["freq_share"] += res["freq_map"].mean(axis=1)
                osc_acc["logenergy"] += res["features"]["freq"]["time_avg_log_energy"]
                osc_acc["n"] += 1
                if osc_acc["freqs"] is None:
                    osc_acc["freqs"] = librosa.fft_frequencies(
                        sr=sr, n_fft=cfg["stft_cfg"]["n_fft"]
                    )
                    osc_acc["f_prime"] = int(store[(0, "freq", "fwd")].shape[1])
                f0_src = clean_np if clean_t is not None else noisy_np
                osc_acc["f0"].append(speech_f0(f0_src, sr))

            summary_png = os.path.join(args.output_dir, f"{base}_summary.png")
            perlayer_png = os.path.join(args.output_dir, f"{base}_perlayer.png")
            corr_png = os.path.join(args.output_dir, f"{base}_corr.png")
            corr_json = os.path.join(args.output_dir, f"{base}_correlations.json")
            npz_path = os.path.join(args.output_dir, f"{base}_gate.npz")
            plot_summary(res, summary_png, base)
            plot_perlayer(res, perlayer_png, base)

            # Spectral-feature correlations: compute, store JSON, plot matrices.
            corr = gate_spectral_correlations(res, res["features"])
            with open(corr_json, "w") as f:
                json.dump(_round_nested(corr), f, indent=2)
            plot_correlations(corr, corr_png, base)

            npz_arrays = {
                "spectrogram": res["spectrogram"],
                "time_map": res["time_map"],
                "freq_map": res["freq_map"],
                "time_std_map": res["time_std_map"],
                "freq_std_map": res["freq_std_map"],
                "time_layers": res["time_layers"],
                "freq_layers": res["freq_layers"],
                "time_std_layers": res["time_std_layers"],
                "freq_std_layers": res["freq_std_layers"],
            }
            outputs = [
                os.path.basename(summary_png),
                os.path.basename(perlayer_png),
                os.path.basename(corr_png),
                os.path.basename(corr_json),
            ]
            extra = ""
            if args.dump_channels:
                npz_arrays["time_chan"] = res["time_chan"]
                npz_arrays["freq_chan"] = res["freq_chan"]
                bimodal_png = os.path.join(args.output_dir, f"{base}_bimodal.png")
                plot_bimodal(res, bimodal_png, base)
                outputs.append(os.path.basename(bimodal_png))
                t_bc = bimodality_coefficient(res["time_chan"].ravel())
                f_bc = bimodality_coefficient(res["freq_chan"].ravel())
                extra = f"  |  bimodality BC  time={t_bc:.2f}  freq={f_bc:.2f}"

            abl_extra = ""
            if args.ablate:
                if clean_t is None:
                    print(f"    [ablate] no clean ref for {base}; output-effect maps only")
                abl = ablate_utterance(model, noisy_t, clean_t, cfg, device)
                abl_png = os.path.join(args.output_dir, f"{base}_ablation.png")
                abl_npz = os.path.join(args.output_dir, f"{base}_ablation.npz")
                plot_ablation(abl, abl_png, base)
                abl_npz_arrays = {
                    "noisy_logmag": abl["noisy_logmag"],
                    "add_effect_db": abl["add_effect_db"],
                    "branch_diff_db": abl["branch_diff_db"],
                    **{f"logmag_{k}": v for k, v in abl["logmag"].items()},
                }
                if abl["has_clean"]:
                    abl_npz_arrays["benefit_add"] = abl["benefit_add"]
                    abl_npz_arrays["benefit_linoss_only"] = abl["benefit_linoss_only"]
                    abl_json = os.path.join(args.output_dir, f"{base}_ablation.json")
                    with open(abl_json, "w") as f:
                        json.dump(
                            _round_nested({"metrics": abl["metrics"], "benefit": abl["benefit"]}, 5),
                            f,
                            indent=2,
                        )
                    outputs.append(os.path.basename(abl_json))
                    b = abl["benefit"]
                    abl_extra = (
                        f"\n      LinOSS benefit (Mamba-only - learned; >0 helps):  "
                        f"mag_MSE={b['add_mag_mse']:+.4f}  phase_sum={b['add_phase_sum']:+.3f}"
                    )
                np.savez_compressed(abl_npz, **abl_npz_arrays)
                outputs.extend([os.path.basename(abl_png), os.path.basename(abl_npz)])

            np.savez_compressed(npz_path, **npz_arrays)
            outputs.append(os.path.basename(npz_path))

            t_share, f_share = float(res["time_map"].mean()), float(res["freq_map"].mean())
            t_std, f_std = float(res["time_std_map"].mean()), float(res["freq_std_map"].mean())
            # Headline correlations: freq-mean gate vs frame energy, time-std vs energy.
            c_fe = corr["per_frame"]["freq_map"]["log_energy"]
            c_ts = corr["per_frame"]["time_std_map"]["log_energy"]
            print(
                f"  {base}: LinOSS share  time={t_share:.3f}  freq={f_share:.3f}  |  "
                f"channel std  time={t_std:.3f}  freq={f_std:.3f}{extra}\n"
                f"      corr(freq-mean gate, frame energy)={c_fe:+.2f}  "
                f"corr(time-std gate, frame energy)={c_ts:+.2f}{abl_extra}\n"
                f"      -> {', '.join(outputs)}"
            )
    finally:
        for h in handles:
            h.remove()

    # Run-level LinOSS oscillator spectrum vs the averaged gate frequency profile.
    if osc_acc is not None and osc_acc["n"] > 0:
        gate_prof = {
            "freqs": osc_acc["freqs"],
            "time_share": osc_acc["time_share"] / osc_acc["n"],
            "freq_share": osc_acc["freq_share"] / osc_acc["n"],
            "logenergy": osc_acc["logenergy"] / osc_acc["n"],
            "n_utt": osc_acc["n"],
        }
        f0_all = [a for a in osc_acc["f0"] if a.size]
        f0_samples = np.concatenate(f0_all) if f0_all else np.array([])
        osc = linoss_oscillators(model, cfg, osc_acc["f_prime"])

        osc_png = os.path.join(args.output_dir, "oscillators.png")
        osc_json = os.path.join(args.output_dir, "oscillators.json")
        osc_npz = os.path.join(args.output_dir, "oscillators.npz")
        run_name = os.path.basename(os.path.dirname(config_file)) or "run"
        plot_oscillators(osc, gate_prof, f0_samples, osc_png, run_name)

        sp = osc["freq"]["phys_hz"]
        sp_fin = sp[np.isfinite(sp)]
        summary = {
            "meta": osc["meta"],
            "n_utterances": osc_acc["n"],
            "freq_frac_in_F0band_80_400": float(((sp >= 80) & (sp <= 400)).mean()),
            "freq_spacing_hz_pct": {
                str(p): float(np.percentile(sp_fin, p)) for p in (5, 50, 95)
            },
            "time_mod_hz_pct": {
                str(p): float(np.percentile(osc["time"]["phys_hz"], p)) for p in (5, 50, 95)
            },
            "Q_median": {
                ax: float(np.median(osc[ax]["Q"][np.isfinite(osc[ax]["Q"])]))
                for ax in ("time", "freq")
            },
        }
        with open(osc_json, "w") as f:
            json.dump(_round_nested(summary, 3), f, indent=2)
        osc_arrays = {
            f"{ax}_{k}": osc[ax][k]
            for ax in ("time", "freq")
            for k in ("omega", "rho", "Q", "phys_hz")
        }
        np.savez_compressed(
            osc_npz,
            **osc_arrays,
            gate_freqs=gate_prof["freqs"],
            gate_time_share=gate_prof["time_share"],
            gate_freq_share=gate_prof["freq_share"],
            gate_logenergy=gate_prof["logenergy"],
            f0_samples=f0_samples,
        )
        print(
            f"\nOscillators: {summary['freq_frac_in_F0band_80_400'] * 100:.0f}% of freq-mixer "
            f"modes resonate in speech-F0 [80,400] Hz  (median spacing "
            f"{summary['freq_spacing_hz_pct']['50']:.0f} Hz);  time-mixer median modulation "
            f"{summary['time_mod_hz_pct']['50']:.0f} Hz\n"
            f"      -> oscillators.png, oscillators.json, oscillators.npz"
        )

    print(
        "\nNote: g measures each block's *update* mix, not a literal decomposition of the "
        "enhanced output\n(blocks also carry a residual + ConvTranspose1d projection + skip)."
    )


if __name__ == "__main__":
    main()
