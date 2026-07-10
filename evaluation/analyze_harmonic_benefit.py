"""Is LinOSS's oscillator bias actually paying off, and where?

Training-free test of the "LinOSS = harmonic specialist / Mamba = transient
specialist" hypothesis. For each test utterance we enhance it with a *pure
LinOSS* and a *pure Mamba* generator, then compare their per-frame magnitude
error against the clean reference, conditioned on whether each frame is
**voiced/harmonic** (cepstral peak prominence, CPP) or unvoiced/transient.

The oscillator inductive bias predicts LinOSS's overall deficit vs Mamba is
concentrated on *unvoiced/transient* frames, while on *voiced harmonic* frames
LinOSS is competitive or ahead (its resonator bank is a native harmonic-comb
matched filter along frequency).

Outputs (under ``--output_dir``):
  * ``harmonic_benefit.png`` -- (1) mean per-frame mag error, LinOSS vs Mamba,
    split voiced/unvoiced; (2) benefit = err_Mamba - err_LinOSS vs acoustic
    frequency, split voiced/unvoiced; (3) benefit vs harmonicity (CPP bins).
  * ``harmonic_benefit.json`` -- the headline aggregates.
  * ``harmonic_benefit.npz`` -- raw per-frame arrays for custom plots.

Runs on GPU (Mamba/LinOSS have no CPU forward path). Reuses evaluate.py's
checkpoint/config resolution, generator loading, enhancement preprocessing and
test-pair listing so it stays consistent with the eval numbers.
"""

import argparse
import json
import os

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluate import (
    build_pair_list,
    enhance,
    load_generator,
    resolve_checkpoint_and_config,
)
from models.stfts import mag_phase_stft
from utils.util import load_config


def frame_voicing(
    clean_np: np.ndarray,
    sr: int,
    n_frames: int,
    hop: int,
    win: int,
    fmin: float = 65.0,
    fmax: float = 500.0,
) -> np.ndarray:
    """Cepstral peak prominence (CPP) per STFT frame, aligned to center=True.

    Frame ``t`` is centered at sample ``t*hop`` (torch.stft center=True). We take
    a Hann-windowed ``win``-sample slice there, form the real cepstrum
    ``irfft(log|rfft|))``, and score the strongest peak in the pitch quefrency
    band ``[sr/fmax, sr/fmin]`` as a z-score above the band's own mean. Pure
    numpy -- librosa.pyin segfaults via numba in this environment.
    """
    pad = win // 2
    x = np.pad(clean_np, (pad, pad), mode="reflect")
    hann = np.hanning(win)
    lag_lo = int(np.floor(sr / fmax))
    lag_hi = int(np.ceil(sr / fmin))
    cpp = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        c = t * hop  # center in original signal == c+pad in padded
        frame = x[c : c + win]
        if frame.shape[0] < win:
            frame = np.pad(frame, (0, win - frame.shape[0]))
        frame = frame * hann
        if not np.any(frame):
            continue
        spec = np.log(np.abs(np.fft.rfft(frame)) + 1e-10)
        cep = np.fft.irfft(spec)
        hi = min(lag_hi, cep.shape[0] - 1)
        if hi <= lag_lo:
            continue
        seg = cep[lag_lo:hi]
        peak = seg.max()
        cpp[t] = (peak - seg.mean()) / (seg.std() + 1e-9)
    return cpp


def analyze_pair(model_lin, model_mam, clean_np, noisy_np, cfg, device):
    """Per-frame magnitude + phase errors for both models + CPP, one utterance.

    Returns a dict of ``(F, T)`` arrays: ``mag_lin``/``mag_mam`` (compressed STFT
    magnitude L1 error vs clean), ``pha_lin``/``pha_mam`` (wrapped circular phase
    error |angle(e^{i(clean-enh)})|), and ``w`` (clean magnitude, the weight for
    phase -- phase is meaningless in silent bins). Plus ``cpp`` shaped ``(T,)``.
    """
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop = cfg["stft_cfg"]["hop_size"]
    win = cfg["stft_cfg"]["win_size"]
    compress = cfg["model_cfg"]["compress_factor"]
    sr = cfg["stft_cfg"]["sampling_rate"]

    noisy_t = torch.from_numpy(noisy_np).float().to(device)
    enh_lin = enhance(model_lin, noisy_t, cfg)
    enh_mam = enhance(model_mam, noisy_t, cfg)

    clean_t = torch.from_numpy(clean_np).float().to(device)
    n = min(clean_t.size(-1), enh_lin.size(-1), enh_mam.size(-1))
    clean_t, enh_lin, enh_mam = clean_t[:n], enh_lin[:n], enh_mam[:n]

    def analyze(w):
        m, p, _ = mag_phase_stft(w.unsqueeze(0), n_fft, hop, win, compress)
        return m[0], p[0]  # (F, T), (F, T)

    clean_mag, clean_pha = analyze(clean_t)
    lin_mag, lin_pha = analyze(enh_lin)
    mam_mag, mam_pha = analyze(enh_mam)

    def pha_err(p):  # wrapped circular distance in [0, pi]
        d = clean_pha - p
        return torch.atan2(torch.sin(d), torch.cos(d)).abs()

    T = clean_mag.shape[1]
    cpp = frame_voicing(clean_t.cpu().numpy(), sr, T, hop, win)
    return {
        "mag_lin": (lin_mag - clean_mag).abs().cpu().numpy(),
        "mag_mam": (mam_mag - clean_mag).abs().cpu().numpy(),
        "pha_lin": pha_err(lin_pha).cpu().numpy(),
        "pha_mam": pha_err(mam_pha).cpu().numpy(),
        "w": clean_mag.cpu().numpy(),
        "cpp": cpp,
    }


def main():
    """CLI: run the voiced/unvoiced LinOSS-vs-Mamba benefit analysis."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--linoss", default="exp/LinOSS", help=".pth or exp dir (pure LinOSS)")
    ap.add_argument("--mamba", default="exp/Mamba", help=".pth or exp dir (pure Mamba)")
    ap.add_argument("--config", default=None, help="defaults to LinOSS's config.yaml")
    ap.add_argument("--test_clean_json", default=None)
    ap.add_argument("--test_noisy_json", default=None)
    ap.add_argument("--num", type=int, default=50, help="test utterances (first N by basename)")
    ap.add_argument("--cpp_thresh", type=float, default=3.0, help="voiced if CPP z-score >")
    ap.add_argument("--output_dir", default="eval_out/harmonic_benefit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    lin_ckpt, lin_cfg_file = resolve_checkpoint_and_config(args.linoss, args.config)
    mam_ckpt, _ = resolve_checkpoint_and_config(args.mamba, None)
    cfg = load_config(lin_cfg_file)
    sr = cfg["stft_cfg"]["sampling_rate"]
    print(f"LinOSS : {lin_ckpt}")
    print(f"Mamba  : {mam_ckpt}")

    model_lin = load_generator(lin_ckpt, load_config(lin_cfg_file), device)
    mam_cfg_file = resolve_checkpoint_and_config(args.mamba, None)[1]
    model_mam = load_generator(mam_ckpt, load_config(mam_cfg_file), device)

    pairs = build_pair_list(cfg, args.test_clean_json, args.test_noisy_json)[: args.num]
    print(f"Analyzing {len(pairs)} utterances at {sr} Hz")

    # Two error metrics: "mag" (unweighted) and "pha" (magnitude-weighted, since
    # phase is meaningless in silent bins). Each accumulates per-frame scalars
    # (split voiced/unvoiced) and a weighted per-(F,) benefit sum + weight sum.
    metrics = ("mag", "pha")
    acc = {
        m: {
            "fr_lin_v": [], "fr_mam_v": [], "fr_lin_u": [], "fr_mam_u": [],
            "num_v": 0.0, "den_v": 0.0, "num_u": 0.0, "den_u": 0.0, "ben": [],
        }
        for m in metrics
    }
    band_nv, band_nu = 0, 0
    cpp_all = []

    for i, (noisy_path, clean_path) in enumerate(pairs):
        noisy_np, _ = librosa.load(noisy_path, sr=sr)
        clean_np, _ = librosa.load(clean_path, sr=sr)
        r = analyze_pair(model_lin, model_mam, clean_np, noisy_np, cfg, device)
        cpp = r["cpp"]
        voiced = cpp > args.cpp_thresh
        cpp_all.append(cpp)
        band_nv += int(voiced.sum())
        band_nu += int((~voiced).sum())

        for m in metrics:
            el, em = r[f"{m}_lin"], r[f"{m}_mam"]  # (F, T)
            w = np.ones_like(el) if m == "mag" else r["w"]  # phase weight = clean mag
            wsum = w.sum(axis=0) + 1e-12  # (T,)
            lin_f = (w * el).sum(axis=0) / wsum  # (T,) weighted mean over freq
            mam_f = (w * em).sum(axis=0) / wsum
            a = acc[m]
            a["fr_lin_v"].append(lin_f[voiced])
            a["fr_mam_v"].append(mam_f[voiced])
            a["fr_lin_u"].append(lin_f[~voiced])
            a["fr_mam_u"].append(mam_f[~voiced])
            a["ben"].append(mam_f - lin_f)  # >0 => LinOSS better this frame
            dnum = w * (em - el)  # (F,T) weighted benefit numerator per bin
            if voiced.any():
                a["num_v"] = a["num_v"] + dnum[:, voiced].sum(axis=1)
                a["den_v"] = a["den_v"] + w[:, voiced].sum(axis=1)
            if (~voiced).any():
                a["num_u"] = a["num_u"] + dnum[:, ~voiced].sum(axis=1)
                a["den_u"] = a["den_u"] + w[:, ~voiced].sum(axis=1)
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(pairs)}]")

    cat = np.concatenate
    cpp_cat = cat(cpp_all)
    freqs = np.linspace(0, sr / 2, r["mag_lin"].shape[0])
    summary = {
        "n_utterances": len(pairs),
        "cpp_thresh": args.cpp_thresh,
        "n_frames_voiced": band_nv,
        "n_frames_unvoiced": band_nu,
    }
    plot_data = {}
    for m in metrics:
        a = acc[m]
        ev_l, ev_m = cat(a["fr_lin_v"]), cat(a["fr_mam_v"])
        eu_l, eu_m = cat(a["fr_lin_u"]), cat(a["fr_mam_u"])
        band_v = a["num_v"] / np.maximum(a["den_v"], 1e-12)
        band_u = a["num_u"] / np.maximum(a["den_u"], 1e-12)
        summary[m] = {
            "voiced": {
                "err_linoss": float(ev_l.mean()),
                "err_mamba": float(ev_m.mean()),
                "benefit_linoss": float((ev_m - ev_l).mean()),
            },
            "unvoiced": {
                "err_linoss": float(eu_l.mean()),
                "err_mamba": float(eu_m.mean()),
                "benefit_linoss": float((eu_m - eu_l).mean()),
            },
        }
        plot_data[m] = {"band_v": band_v, "band_u": band_u, "ben": cat(a["ben"])}

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "harmonic_benefit.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(args.output_dir, "harmonic_benefit.npz"),
        freqs=freqs, cpp=cpp_cat,
        mag_band_voiced=plot_data["mag"]["band_v"],
        mag_band_unvoiced=plot_data["mag"]["band_u"],
        pha_band_voiced=plot_data["pha"]["band_v"],
        pha_band_unvoiced=plot_data["pha"]["band_u"],
        mag_frame_benefit=plot_data["mag"]["ben"],
        pha_frame_benefit=plot_data["pha"]["ben"],
    )

    _plot(summary, plot_data, freqs, cpp_cat, args,
          os.path.join(args.output_dir, "harmonic_benefit.png"))

    labels = {"mag": "magnitude L1", "pha": "phase (mag-weighted rad)"}
    for m in metrics:
        v, u = summary[m]["voiced"], summary[m]["unvoiced"]
        print(f"\n[{labels[m]}]")
        print(
            f"  Voiced  ({band_nv} fr): LinOSS {v['err_linoss']:.4f} vs Mamba "
            f"{v['err_mamba']:.4f}  -> LinOSS benefit {v['benefit_linoss']:+.4f}"
        )
        print(
            f"  Unvoiced({band_nu} fr): LinOSS {u['err_linoss']:.4f} vs Mamba "
            f"{u['err_mamba']:.4f}  -> LinOSS benefit {u['benefit_linoss']:+.4f}"
        )
        favors_voiced = v["benefit_linoss"] > u["benefit_linoss"]
        print(
            f"  -> LinOSS relatively favors {'VOICED' if favors_voiced else 'UNVOICED'} "
            f"frames on {labels[m]}"
        )
    print(f"\n-> {args.output_dir}/harmonic_benefit.png")


def _panel_bars(ax, sm, label):
    """Grouped voiced/unvoiced error bars for one metric."""
    v, u = sm["voiced"], sm["unvoiced"]
    x = np.arange(2)
    w = 0.38
    ax.bar(x - w / 2, [v["err_linoss"], u["err_linoss"]], w, label="LinOSS", color="#d1495b")
    ax.bar(x + w / 2, [v["err_mamba"], u["err_mamba"]], w, label="Mamba", color="#30638e")
    ax.set_xticks(x)
    ax.set_xticklabels(["voiced", "unvoiced"])
    ax.set_ylabel(f"mean per-frame {label} error")
    ax.set_title(f"{label}: error by frame type (lower better)")
    ax.legend()
    for xi, (a, b) in enumerate(
        [(v["err_linoss"], v["err_mamba"]), (u["err_linoss"], u["err_mamba"])]
    ):
        ax.annotate(f"benefit {b - a:+.4f}", (xi, max(a, b)),
                    ha="center", va="bottom", fontsize=9)


def _plot(summary, plot_data, freqs, cpp, args, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    labels = {"mag": "magnitude L1", "pha": "phase (rad)"}
    lo, hi = np.percentile(cpp, [1, 99])
    edges = np.linspace(lo, hi, 21)
    idx = np.digitize(cpp, edges)

    for row, m in enumerate(("mag", "pha")):
        pd = plot_data[m]
        _panel_bars(axes[row, 0], summary[m], labels[m])

        # benefit vs acoustic frequency, split voiced/unvoiced
        ax = axes[row, 1]
        ax.plot(freqs, pd["band_v"], color="#d1495b", label="voiced")
        ax.plot(freqs, pd["band_u"], color="#30638e", label="unvoiced")
        ax.axhline(0, color="k", lw=0.7)
        ax.axvspan(80, 400, color="gold", alpha=0.15, label="F0 band 80-400 Hz")
        ax.set_xlabel("acoustic frequency (Hz)")
        ax.set_ylabel("benefit = Mamba - LinOSS  (>0 LinOSS better)")
        ax.set_title(f"{labels[m]}: where LinOSS wins across the spectrum")
        ax.legend()

        # benefit vs harmonicity (CPP), binned
        ax = axes[row, 2]
        centers, means = [], []
        for b in range(1, len(edges)):
            sel = idx == b
            if sel.sum() >= 20:
                centers.append(0.5 * (edges[b - 1] + edges[b]))
                means.append(pd["ben"][sel].mean())
        ax.plot(centers, means, "o-", color="#3c1642")
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(args.cpp_thresh, color="gray", ls="--",
                   label=f"voiced thresh {args.cpp_thresh}")
        ax.set_xlabel("harmonicity  (cepstral peak prominence)")
        ax.set_ylabel("mean per-frame LinOSS benefit")
        ax.set_title(f"{labels[m]}: benefit vs harmonicity")
        ax.legend()

    fig.suptitle(
        "LinOSS harmonic-oscillator bias: magnitude vs phase, by frame voicing",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
