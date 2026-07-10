"""Measure theta-band entrainment of the time-axis LinOSS oscillator bank.

The static placement analysis (``analyze_linoss.py``) tells you whether the
time-axis bank *owns* resonators near the speech syllabic/theta band (~4-8 Hz at
the 160 Hz frame rate). This script asks the dynamic question: when the trained
model actually processes speech, do the per-oscillator state trajectories
*phase-lock* to the speech's theta-band amplitude modulation (the syllabic
rhythm)? That is entrainment, and it is stimulus dependent -- it cannot be read
off the poles.

What it does, per utterance:
  1. Drives the generator with the chosen source (``--source noisy|clean``),
     capturing the input to every **time-axis** ``LinOSS`` module via a forward
     pre-hook.
  2. Recomputes each module's per-oscillator complex state trajectory ``ys``
     (shape ``(F', T, state_dim)``) using LinOSS's own recurrence helpers, so it
     stays in lock-step with the real forward pass.
  3. Untangles bidirectionality: the ``backward_blocks`` module is fed a
     time-reversed sequence (see ``MambaBlock.forward``), so its trajectory is
     flipped back to normal time order before any correlation.
  4. Builds the reference syllabic envelope ``env(t)`` = log broadband power of
     the magnitude spectrogram (of the drive, or of clean speech with
     ``--ref clean``), at the 160 Hz frame rate.
  5. Correlates each oscillator's real output against ``env`` in the theta band:
       * theta phase-locking value (PLV)  -- band-pass both to theta, Hilbert,
         PLV = |<exp(i(phi_osc - phi_env))>| pooled over freq-bin sequences,
         time and utterances. PLV->1 = locked.
       * envelope-oscillator magnitude-squared coherence (Welch, pooling the F'
         frequency-bin sequences as extra segments), as a full modulation
         spectrum so the theta peak is visible.

The decisive read is **entrainment vs. intrinsic resonant frequency**: genuine
entrainment shows PLV/coherence peaking for the oscillators whose *own* natural
frequency (from ``analyze_linoss``) lands in theta -- resonance, not passthrough.

Outputs (under --output_dir):
  * entrainment.csv          -- one row per (module, oscillator): intrinsic
    freq_hz, lambda_mag, coupling, theta_plv, theta_msc.
  * plv_vs_intrinsic.png     -- theta PLV & theta coherence vs intrinsic freq.
  * coherence_curves.png     -- coupling-weighted coherence spectrum per layer.
  * coherence_heatmap.png    -- per-oscillator coherence, sorted by intrinsic Hz.

Usage:
  python analyze_theta_entrainment.py --checkpoint exp/MyRun \
      --test_noisy_json data/ears_wham_16k/test_noisy.json \
      --output_dir eval_out/theta_entrainment
  # entrainment to *clean* speech rhythm even though it is the drive:
  python analyze_theta_entrainment.py --checkpoint exp/MyRun --source clean
  # driven by noisy, but locking measured against the clean envelope:
  python analyze_theta_entrainment.py --checkpoint exp/MyRun --source noisy --ref clean
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import librosa
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import signal as sps

# Reuse the checkpoint/extraction machinery so this stays in sync with the
# static analysis and the real forward pass.
from evaluation.analyze_linoss import (
    _effective_params,
    _eig_2x2,
    _module_metadata,
    load_generator,
    resolve_checkpoint_and_config,
)
from evaluation.evaluate import build_pair_list
from models.linoss.linoss import (
    LinOSS,
    _apply_damped_linoss_imex,
    _apply_linoss_im,
    _apply_linoss_imex,
    _project_input,
)
from models.stfts import mag_phase_stft
from utils.util import load_config


# --------------------------------------------------------------------------- #
# Per-oscillator state trajectory (a faithful re-run of LinOSS.forward up to the
# pre-readout state ``ys``, reusing the module's own recurrence helpers).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def linoss_states(m: LinOSS, x: torch.Tensor) -> torch.Tensor:
    """Return the complex per-oscillator state trajectory ``ys``.

    Mirrors ``LinOSS.forward`` exactly but stops before the ``C`` readout, so the
    output is the (batch, time, state_dim) complex bank state -- one trajectory
    per oscillator. Honors the module's ``use_triton`` flag: on a shared GPU the
    fused scan avoids the kernel-launch storm of the sequential Python loop.
    """
    tk = m.use_triton
    steps = torch.sigmoid(m.steps)
    B_complex = torch.complex(m.B[..., 0], m.B[..., 1])
    Bu = _project_input(B_complex, x)

    if m.discretization == "IM":
        A_diag = torch.relu(m.A_diag)
        return _apply_linoss_im(A_diag, Bu, steps, use_triton=tk)

    if m.damping:
        if m.A_from_G:
            G_diag = torch.relu(m.G_diag)
            sqrt_term = torch.sqrt(1.0 + steps * G_diag)
            A_low = (2.0 + steps * G_diag - 2.0 * sqrt_term) / steps ** 2
            A_high = (2.0 + steps * G_diag + 2.0 * sqrt_term) / steps ** 2
            A_diag = A_low + torch.relu(m.A_diag - A_low) - torch.relu(m.A_diag - A_high)
        else:
            A_diag = torch.relu(m.A_diag)
            G_low = torch.relu(steps * A_diag - 2 * torch.sqrt(A_diag))
            G_high = steps * A_diag + 2 * torch.sqrt(A_diag)
            G_diag = torch.clamp(m.G_diag, min=G_low, max=G_high)
        return _apply_damped_linoss_imex(A_diag, G_diag, Bu, steps, use_triton=tk)

    A_diag = torch.relu(m.A_diag)
    return _apply_linoss_imex(A_diag, Bu, steps, use_triton=tk)


def intrinsic_freqs(m: LinOSS, frame_rate: float) -> dict:
    """Per-oscillator intrinsic resonant freq (Hz), |lambda|, oscillatory flag,
    and B/C coupling -- identical definitions to analyze_linoss.analyze_module."""
    p = _effective_params(m)
    lam, _, osc = _eig_2x2(p["M11"], p["M12"], p["M21"], p["M22"])
    angle = np.abs(np.angle(lam))
    freq_hz = angle / (2.0 * np.pi) * frame_rate
    B = torch.complex(m.B[..., 0], m.B[..., 1]).detach().cpu().numpy()
    C = torch.complex(m.C[..., 0], m.C[..., 1]).detach().cpu().numpy()
    coupling = np.abs(B).mean(axis=1) * np.abs(C).mean(axis=0)
    return {
        "freq_hz": freq_hz,
        "lambda_mag": np.abs(lam),
        "oscillatory": osc,
        "coupling": coupling,
    }


# --------------------------------------------------------------------------- #
# Reference envelope + entrainment metrics.
# --------------------------------------------------------------------------- #
def syllabic_envelope(mag: torch.Tensor, compress: float) -> np.ndarray:
    """Log broadband power envelope from a (1, F, T) compressed magnitude.

    Decompresses to linear magnitude, sums power over frequency, returns log
    power per frame -- the temporal modulation carrier whose 4-8 Hz content is
    the syllabic/theta rhythm. Mean-removed so DC does not dominate filtering.
    """
    mag_lin = mag.detach().float().cpu().numpy()[0] ** (1.0 / compress)  # (F, T)
    power = (mag_lin ** 2).sum(axis=0)                                   # (T,)
    env = np.log(power + 1e-8)
    return env - env.mean()


def _bandpass(x: np.ndarray, band, fs: float, axis: int) -> np.ndarray:
    b, a = sps.butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, x, axis=axis)


def osc_signal(ys: np.ndarray, mode: str) -> np.ndarray:
    """Scalar per-oscillator trajectory used for locking. ``real`` = Re(state)
    (carrier/phase entrainment); ``abs`` = |state| (amplitude entrainment)."""
    return np.real(ys) if mode == "real" else np.abs(ys)


def match_len(env: np.ndarray, T: int) -> np.ndarray:
    """Crop/tile a donor envelope to length ``T`` without time-rescaling, so its
    modulation power spectrum (the theta-dominant shape) is preserved -- only its
    temporal alignment with the states is destroyed. The mismatch null thus
    isolates genuine locking from shared envelope spectral shape."""
    if len(env) == T:
        return env
    if len(env) > T:
        return env[:T]
    return np.tile(env, int(np.ceil(T / len(env))))[:T]


def update_plv(acc: dict, env: np.ndarray, sig: np.ndarray, band, fs: float):
    """Accumulate complex PLV sum/count for one (utterance, module).

    env: (T,)   sig: (rows, T, N) real.  Pools rows and time.
    """
    env_ph = np.angle(sps.hilbert(_bandpass(env, band, fs, axis=0)))      # (T,)
    sig_ph = np.angle(sps.hilbert(_bandpass(sig, band, fs, axis=1), axis=1))  # (rows,T,N)
    dphi = sig_ph - env_ph[None, :, None]
    acc["sum"] = acc.get("sum", 0.0) + np.exp(1j * dphi).sum(axis=(0, 1))  # (N,)
    acc["count"] = acc.get("count", 0) + sig.shape[0] * sig.shape[1]


def update_coherence(acc: dict, env: np.ndarray, sig: np.ndarray, fs: float, nperseg: int):
    """Accumulate Welch cross/auto spectra for envelope-oscillator coherence.

    The F' frequency-bin sequences (rows) are pooled as independent segments
    alongside the within-row Welch segments. ``env`` is shared across rows, so
    its auto-spectrum is counted ``rows`` times per Welch segment.
    """
    T = env.shape[0]
    if T < nperseg:
        return
    rows, _, N = sig.shape
    win = np.hanning(nperseg)
    step = nperseg // 2
    Sxy = acc.get("Sxy", 0.0)
    Sxx = acc.get("Sxx", 0.0)
    Syy = acc.get("Syy", 0.0)
    for start in range(0, T - nperseg + 1, step):
        sl = slice(start, start + nperseg)
        Xe = np.fft.rfft(win * env[sl])                         # (Fk,)
        Yo = np.fft.rfft(win[None, :, None] * sig[:, sl, :], axis=1)  # (rows,Fk,N)
        Sxy = Sxy + (np.conj(Xe)[None, :, None] * Yo).sum(axis=0)    # (Fk,N)
        Sxx = Sxx + rows * (np.abs(Xe) ** 2)                         # (Fk,)
        Syy = Syy + (np.abs(Yo) ** 2).sum(axis=0)                    # (Fk,N)
    acc["Sxy"], acc["Sxx"], acc["Syy"] = Sxy, Sxx, Syy
    acc["freqs"] = np.fft.rfftfreq(nperseg, 1.0 / fs)


def finalize_coherence(acc: dict):
    """Magnitude-squared coherence (Fk, N) and freq grid, or (None, None)."""
    if "Sxy" not in acc:
        return None, None
    msc = (np.abs(acc["Sxy"]) ** 2) / (acc["Sxx"][:, None] * acc["Syy"] + 1e-20)
    return msc, acc["freqs"]


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="exp/LinOSS_EARS",
                        help="A g_*.pth file or an exp directory containing one.")
    parser.add_argument("--config", default=None,
                        help="YAML config. Defaults to config.yaml next to the ckpt.")
    parser.add_argument("--test_noisy_json", default="data/ears_wham_16k/test_noisy.json")
    parser.add_argument("--test_clean_json", default=None,
                        help="Defaults to the config's test_clean_json.")
    parser.add_argument("--output_dir", default="eval_out/theta_entrainment")
    parser.add_argument("--source", choices=["noisy", "clean"], default="noisy",
                        help="Which audio drives the model.")
    parser.add_argument("--ref", choices=["drive", "clean"], default="drive",
                        help="Reference envelope: the drive's, or always clean.")
    parser.add_argument("--signal", choices=["real", "abs"], default="real",
                        help="Oscillator trajectory: Re(state) or |state|.")
    parser.add_argument("--band", type=float, nargs=2, default=(4.0, 8.0),
                        metavar=("LO", "HI"), help="Entrainment band in Hz.")
    parser.add_argument("--null", choices=["none", "shuffle"], default="none",
                        help="shuffle: also score each utterance's states against a "
                             "different utterance's envelope (mismatch null), and "
                             "report matched-minus-null entrainment.")
    parser.add_argument("--nperseg", type=int, default=128,
                        help="Welch segment length (frames) for coherence.")
    parser.add_argument("--max_utts", type=int, default=200,
                        help="Cap on utterances (theta stats converge well before all).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    band = tuple(args.band)
    device = torch.device(args.device)
    ckpt_file, config_file = resolve_checkpoint_and_config(args.checkpoint, args.config)
    print(f"Checkpoint : {ckpt_file}")
    print(f"Config     : {config_file}")

    cfg = load_config(config_file)
    stft = cfg["stft_cfg"]
    sr, n_fft, hop, win = (stft["sampling_rate"], stft["n_fft"],
                           stft["hop_size"], stft["win_size"])
    compress = cfg["model_cfg"]["compress_factor"]
    frame_rate = sr / hop
    print(f"Frame rate : {frame_rate:g} Hz   theta band: {band[0]:g}-{band[1]:g} Hz   "
          f"source={args.source} ref={args.ref} signal={args.signal}")

    model = load_generator(ckpt_file, cfg, device)

    # Time-axis LinOSS modules only.
    time_mods = [(name, m) for name, m in model.named_modules()
                 if isinstance(m, LinOSS) and _module_metadata(name)["axis"] == "time"]
    if not time_mods:
        raise SystemExit("No time-axis LinOSS modules found in this checkpoint.")
    print(f"Found {len(time_mods)} time-axis LinOSS modules "
          f"({sum(m.state_dim for _, m in time_mods)} oscillators)")

    # Capture each module's input via a pre-hook (b=1 per utterance, so rows=F').
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, m in time_mods:
        def hook(mod, inp, _name=name):
            captured[_name] = inp[0].detach()
        handles.append(m.register_forward_pre_hook(hook))

    pairs = build_pair_list(cfg, args.test_clean_json, args.test_noisy_json)
    if args.max_utts:
        pairs = pairs[: args.max_utts]
    print(f"Processing {len(pairs)} utterances")

    plv_acc = {name: {} for name, _ in time_mods}
    coh_acc = {name: {} for name, _ in time_mods}
    null_plv_acc = {name: {} for name, _ in time_mods}
    null_coh_acc = {name: {} for name, _ in time_mods}
    prev_env = None  # donor envelope for the mismatch null

    for i, (noisy_path, clean_path) in enumerate(pairs):
        drive_path = clean_path if args.source == "clean" else noisy_path
        drive_np, _ = librosa.load(drive_path, sr=sr)
        drive = torch.from_numpy(drive_np).float().to(device)
        norm = torch.sqrt(len(drive) / torch.sum(drive ** 2.0 + 1e-12))
        drive_n = (drive * norm).unsqueeze(0)

        drive_mag, drive_pha, _ = mag_phase_stft(drive_n, n_fft, hop, win, compress)

        if args.ref == "clean" and args.source != "clean":
            clean_np, _ = librosa.load(clean_path, sr=sr)
            clean = torch.from_numpy(clean_np).float().to(device)
            cnorm = torch.sqrt(len(clean) / torch.sum(clean ** 2.0 + 1e-12))
            ref_mag, _, _ = mag_phase_stft((clean * cnorm).unsqueeze(0),
                                           n_fft, hop, win, compress)
        else:
            ref_mag = drive_mag

        captured.clear()
        with torch.no_grad():
            model(drive_mag, drive_pha)

        env = syllabic_envelope(ref_mag, compress)  # (T,)
        # Donor envelope (a *different* utterance) for the mismatch null.
        do_null = args.null == "shuffle" and prev_env is not None
        null_env = match_len(prev_env, env.shape[0]) if do_null else None

        for name, m in time_mods:
            # LinOSS sees (b*F', T, C); driving one utterance => b=1, rows=F'.
            x = captured[name]                       # (F', T, inner_dim)
            ys = linoss_states(m, x).cpu().numpy()   # (F', T, N) complex
            if _module_metadata(name)["direction"] == "bwd":
                ys = ys[:, ::-1, :]                  # undo time reversal
            sig = osc_signal(ys, args.signal)        # (F', T, N) real
            # Align env length to the trajectory (time is preserved, but guard).
            if env.shape[0] != sig.shape[1]:
                env_u = np.interp(np.linspace(0, 1, sig.shape[1]),
                                  np.linspace(0, 1, env.shape[0]), env)
            else:
                env_u = env
            update_plv(plv_acc[name], env_u, sig, band, frame_rate)
            update_coherence(coh_acc[name], env_u, sig, frame_rate, args.nperseg)
            if do_null:
                null_u = match_len(null_env, sig.shape[1])
                update_plv(null_plv_acc[name], null_u, sig, band, frame_rate)
                update_coherence(null_coh_acc[name], null_u, sig, frame_rate, args.nperseg)

        prev_env = env
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(pairs)}]")

    for h in handles:
        h.remove()

    # ----------------------------------------------------------------- collate
    do_null = args.null == "shuffle"

    def theta_plv_of(acc, name, m):
        return np.abs(acc[name].get("sum", np.zeros(m.d_state))) / max(
            acc[name].get("count", 1), 1)

    def theta_msc_of(acc, name, freqs_band):
        msc, freqs = finalize_coherence(acc[name])
        if msc is None:
            return None, None
        return msc, freqs

    rows = []
    heat_msc, heat_freq, heat_intrinsic, heat_coupling, heat_layer = [], None, [], [], []
    null_msc_stack = []
    for name, m in time_mods:
        meta = _module_metadata(name)
        intr = intrinsic_freqs(m, frame_rate)
        plv = theta_plv_of(plv_acc, name, m)
        msc, freqs = theta_msc_of(coh_acc, name, band)
        plv_null = theta_plv_of(null_plv_acc, name, m) if do_null else None
        msc_null, _ = theta_msc_of(null_coh_acc, name, band) if do_null else (None, None)
        if msc is not None:
            in_band = (freqs >= band[0]) & (freqs <= band[1])
            theta_msc = msc[in_band].mean(axis=0)
            theta_msc_null = (msc_null[in_band].mean(axis=0)
                              if msc_null is not None else np.full(m.state_dim, np.nan))
            heat_msc.append(msc.T)              # (N, Fk)
            heat_freq = freqs
            heat_intrinsic.append(intr["freq_hz"])
            heat_coupling.append(intr["coupling"])
            heat_layer.append(np.full(m.state_dim, meta["layer"]))
            null_msc_stack.append((msc_null.T if msc_null is not None
                                   else np.full((m.state_dim, len(freqs)), np.nan)))
        else:
            theta_msc = np.full(m.state_dim, np.nan)
            theta_msc_null = np.full(m.state_dim, np.nan)
        for n in range(m.state_dim):
            row = {
                "module": name, "layer": meta["layer"], "direction": meta["direction"],
                "channel": n, "intrinsic_freq_hz": intr["freq_hz"][n],
                "lambda_mag": intr["lambda_mag"][n], "oscillatory": bool(intr["oscillatory"][n]),
                "coupling": intr["coupling"][n], "theta_plv": plv[n],
                "theta_msc": theta_msc[n],
            }
            if do_null:
                row["theta_plv_null"] = plv_null[n]
                row["theta_msc_null"] = theta_msc_null[n]
                row["theta_plv_delta"] = plv[n] - plv_null[n]
                row["theta_msc_delta"] = theta_msc[n] - theta_msc_null[n]
            rows.append(row)
    df = pd.DataFrame(rows)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "entrainment.csv")
    df.to_csv(csv_path, index=False)

    _plot_vs_intrinsic(df, band, os.path.join(args.output_dir, "plv_vs_intrinsic.png"))
    if heat_msc:
        null_stack = null_msc_stack if do_null else None
        _plot_coherence_curves(
            heat_msc, heat_freq, heat_coupling, heat_layer, band, frame_rate,
            os.path.join(args.output_dir, "coherence_curves.png"), null_stack)
        _plot_coherence_heatmap(
            heat_msc, heat_freq, heat_intrinsic, band, frame_rate,
            os.path.join(args.output_dir, "coherence_heatmap.png"))

    # ----------------------------------------------------------------- summary
    osc = df[df["oscillatory"]].copy()
    in_theta = osc["intrinsic_freq_hz"].between(*band)
    w = osc["coupling"].to_numpy()

    def wmean(mask, col):
        s = osc[mask]
        ww = s["coupling"].to_numpy()
        v = s[col].to_numpy()
        ok = ~np.isnan(v)
        return float(np.average(v[ok], weights=ww[ok])) if ok.any() and ww[ok].sum() else float("nan")

    print("\nCoupling-weighted entrainment (oscillatory channels):")
    print(f"  intrinsic-in-theta  : PLV={wmean(in_theta, 'theta_plv'):.3f}  "
          f"MSC={wmean(in_theta, 'theta_msc'):.3f}  (n={int(in_theta.sum())})")
    print(f"  intrinsic-off-theta : PLV={wmean(~in_theta, 'theta_plv'):.3f}  "
          f"MSC={wmean(~in_theta, 'theta_msc'):.3f}  (n={int((~in_theta).sum())})")
    if do_null:
        all_mask = np.ones(len(osc), dtype=bool)
        print("\nMatched vs shuffle null (all oscillatory, coupling-weighted):")
        print(f"  matched : PLV={wmean(all_mask, 'theta_plv'):.3f}  "
              f"MSC={wmean(all_mask, 'theta_msc'):.3f}")
        print(f"  null    : PLV={wmean(all_mask, 'theta_plv_null'):.3f}  "
              f"MSC={wmean(all_mask, 'theta_msc_null'):.3f}")
        print(f"  delta   : PLV={wmean(all_mask, 'theta_plv_delta'):+.3f}  "
              f"MSC={wmean(all_mask, 'theta_msc_delta'):+.3f}  "
              f"(>0 ==> genuine locking beyond shared envelope shape)")
    print("\nPer layer/direction (intrinsic-in-theta only):")
    sub = osc[in_theta]
    if len(sub):
        print(sub.groupby(["layer", "direction"]).agg(
            n=("channel", "size"),
            med_intrinsic_hz=("intrinsic_freq_hz", "median"),
            mean_plv=("theta_plv", "mean"),
            mean_msc=("theta_msc", "mean"),
        ).round(3).to_string())
    print(f"\nWrote:\n  {csv_path}")
    for f in ("plv_vs_intrinsic.png", "coherence_curves.png", "coherence_heatmap.png"):
        p = os.path.join(args.output_dir, f)
        if os.path.exists(p):
            print(f"  {p}")


# --------------------------------------------------------------------------- #
# Plots.
# --------------------------------------------------------------------------- #
def _plot_vs_intrinsic(df: pd.DataFrame, band, out_path: str):
    osc = df[df["oscillatory"]]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, col, title in ((axs[0], "theta_plv", f"theta PLV ({band[0]:g}-{band[1]:g} Hz)"),
                           (axs[1], "theta_msc", "theta coherence (MSC)")):
        sizes = 8 + 200 * (osc["coupling"] / (osc["coupling"].max() + 1e-12))
        for direction, marker in (("fwd", "o"), ("bwd", "^")):
            s = osc[osc["direction"] == direction]
            sc = ax.scatter(s["intrinsic_freq_hz"], s[col], c=s["layer"], s=sizes[s.index],
                            cmap="viridis", marker=marker, alpha=0.75, edgecolors="none",
                            label=direction)
        ax.axvspan(band[0], band[1], color="green", alpha=0.12, label="theta band")
        ax.set_xlabel("intrinsic resonant freq (Hz)")
        ax.set_ylabel(col)
        ax.set_title(title + "  vs intrinsic freq (size=coupling)")
        ax.legend(fontsize=8)
        fig.colorbar(sc, ax=ax, label="layer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_coherence_curves(heat_msc, freqs, couplings, layers, band, frame_rate,
                           out_path, null_stack=None):
    msc = np.concatenate(heat_msc, axis=0)          # (Ntot, Fk)
    w = np.concatenate(couplings)
    lay = np.concatenate(layers)
    nyq = frame_rate / 2.0
    keep = freqs <= min(nyq, 40.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for layer in sorted(np.unique(lay)):
        sel = lay == layer
        curve = np.average(msc[sel][:, keep], axis=0, weights=w[sel] + 1e-12)
        ax.plot(freqs[keep], curve, alpha=0.6, label=f"layer {layer}")
    overall = np.average(msc[:, keep], axis=0, weights=w + 1e-12)
    ax.plot(freqs[keep], overall, color="k", lw=2.2, label="overall (matched)")
    if null_stack is not None:
        msc_null = np.concatenate(null_stack, axis=0)
        null_overall = np.average(msc_null[:, keep], axis=0, weights=w + 1e-12)
        ax.plot(freqs[keep], null_overall, color="k", lw=1.8, ls="--",
                label="overall (shuffle null)")
    ax.axvspan(band[0], band[1], color="green", alpha=0.12, label="theta band")
    ax.set_xlabel("modulation frequency (Hz)")
    ax.set_ylabel("coupling-weighted coherence (MSC)")
    ax.set_title("envelope-oscillator coherence spectrum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_coherence_heatmap(heat_msc, freqs, intrinsic, band, frame_rate, out_path):
    msc = np.concatenate(heat_msc, axis=0)          # (Ntot, Fk)
    intr = np.concatenate(intrinsic)
    order = np.argsort(intr)
    nyq = frame_rate / 2.0
    keep = freqs <= min(nyq, 40.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(msc[order][:, keep], origin="lower", aspect="auto", cmap="magma",
                   vmin=0, vmax=np.percentile(msc[:, keep], 99) or 1.0,
                   extent=[freqs[keep].min(), freqs[keep].max(), 0, len(order)])
    ax.axvline(band[0], color="cyan", ls="--", lw=1)
    ax.axvline(band[1], color="cyan", ls="--", lw=1)
    ax.set_xlabel("modulation frequency (Hz)")
    ax.set_ylabel("oscillator (sorted by intrinsic freq, low->high)")
    ax.set_title("per-oscillator envelope coherence (theta band dashed)")
    fig.colorbar(im, ax=ax, label="coherence (MSC)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
