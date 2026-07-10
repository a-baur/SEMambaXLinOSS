"""Coupling and phase-locking analysis for the selective LRU mixer (S-LinOSS).

The base ``LinOSS`` bank is linear time-invariant: every mode has *fixed* poles,
so ``analyze_linoss.py`` can read its physics straight off the transition matrix,
and ``analyze_theta_entrainment.py`` then asks whether those fixed resonators
phase-lock to the speech syllabic rhythm. The selective LRU (``MambOSS`` in
``models/linoss/selective_linoss.py``) has *no* fixed poles: its per-step
eigenvalue ``lambda_k = nu_k * exp(i theta_k)`` is produced from the input token,

    nu_k    = sigmoid(W_nu    u_k + c_nu)     (selective damping,   |lambda_k|)
    theta_k =          W_theta u_k + c_theta  (selective frequency, arg lambda_k)

so its "intrinsic frequency" only exists as a *baseline* (the zero-input bias
term) plus an input-driven swing. This script analyzes both halves of that.

It produces three views:

  1. **Coupling / placement (static, no data).** Per mode: the baseline pole
     ``lambda0 = sigmoid(c_nu) * exp(i c_theta)`` (baseline |lambda| and resonant
     frequency), the B/C input/output coupling (identical definition to
     ``analyze_linoss``), and the *selectivity gain* ``||W_nu||`` / ``||W_theta||``
     -- how strongly the input can move each mode's damping and frequency.

  2. **Selectivity in action (dynamic).** Driving the model with speech and
     recomputing each time-axis module's ``nu_k`` / ``theta_k`` / state ``h_k``
     trajectories, the *realized* per-mode damping and instantaneous frequency
     (mean and swing), so you can see how much of the selectivity budget the
     trained model actually spends.

  3. **Phase-locking (dynamic).** Three entrainment questions against the
     syllabic (theta-band, ~4-8 Hz) envelope of the drive:
       * **state** locking -- does Re(h_k) (the oscillator output) phase-lock to
         the syllabic rhythm? (the same question ``analyze_theta_entrainment``
         asks of LinOSS).
       * **nu** locking -- does the selective *forgetting gate* track the rhythm
         (drop at onsets, hold during sustained content)?
       * **theta** locking -- does the selective *instantaneous frequency* track
         the rhythm (input-controlled chirp, spec section 7)?
     Each via theta-band PLV and a full envelope coherence (MSC) spectrum, reusing
     the metric machinery of ``analyze_theta_entrainment``. An optional shuffle
     null (``--null shuffle``) separates genuine locking from shared envelope
     spectral shape.

Outputs (under --output_dir):
  * selective_modes.csv        -- one row per (module, mode): baseline pole,
    coupling, selectivity gains, realized nu/theta stats, and the three lockings.
  * baseline_poles.png         -- baseline poles in the z-plane (size = coupling).
  * selectivity.png            -- baseline vs realized damping/frequency swing.
  * entrainment_vs_intrinsic.png -- state PLV/MSC vs baseline resonant freq.
  * coordinate_entrainment.png -- coupling-weighted coherence spectra for state,
    nu and theta (the decisive selective-vs-LTI figure).
  * coherence_heatmap.png      -- per-mode state coherence, sorted by baseline Hz.

Usage:
  python -m evaluation.analyze_selective_entrainment --checkpoint exp/SLinOSS_EARS \
      --test_noisy_json data/ears_wham_16k/test_noisy.json \
      --output_dir eval_out/selective_entrainment
  # coupling/placement only, no dataset needed:
  python -m evaluation.analyze_selective_entrainment --checkpoint exp/SLinOSS_EARS --max_utts 0
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless, matches utils/viz.py
import librosa
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

# Shared machinery, kept in lock-step with the static analysis, the LinOSS
# entrainment script, and the real forward pass.
from evaluation.analyze_linoss import (
    _module_metadata,
    load_generator,
    resolve_checkpoint_and_config,
)
from evaluation.analyze_theta_entrainment import (
    finalize_coherence,
    match_len,
    osc_signal,
    syllabic_envelope,
    update_coherence,
    update_plv,
)
from evaluation.evaluate import build_pair_list
from models.linoss.linoss import _project_input
from selective_lru.selective_lru import _selective_recurrence
from selective_lru import SelectiveLRUMIMO
from models.stfts import mag_phase_stft
from utils.util import load_config

# Signals whose locking to the syllabic envelope we measure. "state" is the
# oscillator output (the LinOSS question); "nu"/"theta" are the selective spectral
# coordinates -- the locking the LTI LinOSS cannot have by construction.
SIGNALS = ("state", "nu", "theta")


# --------------------------------------------------------------------------- #
# Faithful re-run of MambOSS.forward up to (but not including) the C readout.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def selective_trajectories(m: SelectiveLRUMIMO, x: torch.Tensor):
    """Return ``(nu, theta, h)`` trajectories exactly as ``MambOSS.forward``.

    ``nu`` (B,T,N) real in (0,1) is the per-step damping ``|lambda_k|``; ``theta``
    (B,T,N) real is the per-step phase ``arg lambda_k``; ``h`` (B,T,N) complex is
    the pre-readout state. Honors the module's ``use_triton`` flag so on a shared
    GPU the fused scan avoids the kernel-launch storm of the Python loop.
    """
    nu = torch.sigmoid(m.nu_proj(x))             # (B, T, N) in (0, 1)
    theta = m.theta_proj(x)                      # (B, T, N)
    lam = torch.polar(nu, theta)                 # (B, T, N) complex, |lam| = nu

    B_complex = torch.complex(m.B[..., 0], m.B[..., 1])
    Bu = _project_input(B_complex, x)            # (B, T, N) complex
    if m.normalize_input:
        Bu = Bu * torch.sqrt((1.0 - nu ** 2).clamp_min(1e-6))

    if m.use_triton:
        from selective_lru.selective_lru_mimo_triton import selective_scan_triton
        h = selective_scan_triton(lam, Bu)
    else:
        h = _selective_recurrence(lam, Bu)
    return nu, theta, h


def _wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    """Wrap an unbounded phase (rad) to (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def baseline_and_coupling(m: SelectiveLRUMIMO, frame_rate: float, axis: str) -> dict:
    """Static, input-independent per-mode quantities.

    The baseline pole ``lambda0 = sigmoid(c_nu) * exp(i c_theta)`` is what each
    mode reduces to at zero selective contribution -- the direct analogue of the
    LTI LinOSS pole. Coupling matches ``analyze_linoss.analyze_module``; the
    selectivity gains are the row norms of the selective projection weights.
    """
    c_nu = m.nu_proj.bias.detach().cpu().numpy()
    c_theta = m.theta_proj.bias.detach().cpu().numpy()
    nu0 = 1.0 / (1.0 + np.exp(-c_nu))                       # sigmoid
    # Folded baseline frequency: the real readout makes +/-theta equivalent.
    cyc0 = np.abs(_wrap_to_pi(c_theta)) / (2.0 * np.pi)     # cycles/step
    with np.errstate(divide="ignore"):
        tau0 = np.where(nu0 < 1.0, -1.0 / np.log(np.clip(nu0, 1e-12, 1.0)), np.inf)

    B = torch.complex(m.B[..., 0], m.B[..., 1]).detach().cpu().numpy()   # (N, F)
    C = torch.complex(m.C[..., 0], m.C[..., 1]).detach().cpu().numpy()   # (F, N)
    b_energy = np.abs(B).mean(axis=1)
    c_energy = np.abs(C).mean(axis=0)

    # Selectivity gains: how far the input can swing damping / phase per step.
    nu_gain = np.linalg.norm(m.nu_proj.weight.detach().cpu().numpy(), axis=1)
    theta_gain = np.linalg.norm(m.theta_proj.weight.detach().cpu().numpy(), axis=1)

    out = {
        "baseline_lambda_mag": nu0,
        "baseline_cycles_per_step": cyc0,
        "baseline_tau_steps": tau0,
        "b_energy": b_energy,
        "c_energy": c_energy,
        "coupling": b_energy * c_energy,
        "nu_sel_gain": nu_gain,
        "theta_sel_gain": theta_gain,
    }
    if axis == "time":
        out["baseline_freq_hz"] = cyc0 * frame_rate
        out["baseline_tau_sec"] = tau0 / frame_rate
    return out


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="exp/SLinOSS_EARS",
                        help="A g_*.pth file or an exp directory containing one.")
    parser.add_argument("--config", default=None,
                        help="YAML config. Defaults to config.yaml next to the ckpt.")
    parser.add_argument("--test_noisy_json", default="data/ears_wham_16k/test_noisy.json")
    parser.add_argument("--test_clean_json", default=None,
                        help="Defaults to the config's test_clean_json.")
    parser.add_argument("--output_dir", default="eval_out/selective_entrainment")
    parser.add_argument("--source", choices=["noisy", "clean"], default="noisy",
                        help="Which audio drives the model.")
    parser.add_argument("--ref", choices=["drive", "clean"], default="drive",
                        help="Reference envelope: the drive's, or always clean.")
    parser.add_argument("--signal", choices=["real", "abs"], default="real",
                        help="State trajectory for locking: Re(h) (carrier/phase) "
                             "or |h| (amplitude).")
    parser.add_argument("--band", type=float, nargs=2, default=(4.0, 8.0),
                        metavar=("LO", "HI"), help="Entrainment band in Hz.")
    parser.add_argument("--null", choices=["none", "shuffle"], default="none",
                        help="shuffle: also score each utterance's signals against a "
                             "different utterance's envelope (mismatch null).")
    parser.add_argument("--nperseg", type=int, default=128,
                        help="Welch segment length (frames) for coherence.")
    parser.add_argument("--max_utts", type=int, default=200,
                        help="Cap on utterances; 0 = static coupling analysis only.")
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

    model = load_generator(ckpt_file, cfg, device)

    selective_mods = [(name, m) for name, m in model.named_modules()
                      if isinstance(m, SelectiveLRUMIMO)]
    if not selective_mods:
        raise SystemExit("No selective LinOSS (MambOSS) modules found in this "
                         "checkpoint. Is this a 'selective_linoss' run?")
    time_mods = [(n, m) for n, m in selective_mods
                 if _module_metadata(n)["axis"] == "time"]
    print(f"Found {len(selective_mods)} selective modules "
          f"({sum(m.state_dim for _, m in selective_mods)} modes); "
          f"{len(time_mods)} on the time axis.")
    print(f"Frame rate : {frame_rate:g} Hz   theta band: {band[0]:g}-{band[1]:g} Hz   "
          f"source={args.source} ref={args.ref} signal={args.signal}")

    do_dynamic = args.max_utts > 0 and bool(time_mods)
    plv_acc = {n: {s: {} for s in SIGNALS} for n, _ in time_mods}
    coh_acc = {n: {s: {} for s in SIGNALS} for n, _ in time_mods}
    null_plv_acc = {n: {s: {} for s in SIGNALS} for n, _ in time_mods}
    null_coh_acc = {n: {s: {} for s in SIGNALS} for n, _ in time_mods}
    # Running sums for the realized nu / instantaneous-frequency statistics.
    stat_acc = {n: {"nu_sum": 0.0, "nu_sq": 0.0, "f_sum": 0.0, "f_sq": 0.0, "cnt": 0}
                for n, _ in time_mods}

    if do_dynamic:
        captured: dict[str, torch.Tensor] = {}
        handles = []
        for name, m in time_mods:
            def hook(mod, inp, _name=name):
                captured[_name] = inp[0].detach()
            handles.append(m.register_forward_pre_hook(hook))

        pairs = build_pair_list(cfg, args.test_clean_json, args.test_noisy_json)
        pairs = pairs[: args.max_utts]
        print(f"Processing {len(pairs)} utterances")
        prev_env = None  # donor envelope for the mismatch null

        for i, (noisy_path, clean_path) in enumerate(pairs):
            drive_path = clean_path if args.source == "clean" else noisy_path
            drive_np, _ = librosa.load(drive_path, sr=sr)
            drive = torch.from_numpy(drive_np).float().to(device)
            norm = torch.sqrt(len(drive) / torch.sum(drive ** 2.0 + 1e-12))
            drive_mag, drive_pha, _ = mag_phase_stft((drive * norm).unsqueeze(0),
                                                     n_fft, hop, win, compress)

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
            do_null = args.null == "shuffle" and prev_env is not None
            null_env = match_len(prev_env, env.shape[0]) if do_null else None

            for name, m in time_mods:
                x = captured[name]                       # (F', T, inner_dim)
                nu, theta, h = selective_trajectories(m, x)
                nu = nu.cpu().numpy()
                theta = theta.cpu().numpy()
                h = h.cpu().numpy()
                if _module_metadata(name)["direction"] == "bwd":
                    nu, theta, h = nu[:, ::-1], theta[:, ::-1], h[:, ::-1]

                # Instantaneous modulation frequency in Hz (folded, real-readout).
                inst_freq = np.abs(_wrap_to_pi(theta)) / (2.0 * np.pi) * frame_rate
                sigs = {
                    "state": osc_signal(h, args.signal),
                    "nu": nu,
                    "theta": inst_freq,
                }

                # Accumulate realized statistics (pooled over rows and time).
                acc = stat_acc[name]
                acc["nu_sum"] += nu.sum(axis=(0, 1))
                acc["nu_sq"] += (nu ** 2).sum(axis=(0, 1))
                acc["f_sum"] += inst_freq.sum(axis=(0, 1))
                acc["f_sq"] += (inst_freq ** 2).sum(axis=(0, 1))
                acc["cnt"] += nu.shape[0] * nu.shape[1]

                # Align envelope length to the trajectory (time is preserved).
                T = sigs["state"].shape[1]
                env_u = (env if env.shape[0] == T else
                         np.interp(np.linspace(0, 1, T),
                                   np.linspace(0, 1, env.shape[0]), env))
                for s in SIGNALS:
                    update_plv(plv_acc[name][s], env_u, sigs[s], band, frame_rate)
                    update_coherence(coh_acc[name][s], env_u, sigs[s],
                                     frame_rate, args.nperseg)
                    if do_null:
                        null_u = match_len(null_env, T)
                        update_plv(null_plv_acc[name][s], null_u, sigs[s],
                                   band, frame_rate)
                        update_coherence(null_coh_acc[name][s], null_u, sigs[s],
                                         frame_rate, args.nperseg)

            prev_env = env
            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(pairs)}]")

        for hd in handles:
            hd.remove()
    elif args.max_utts > 0:
        print("No time-axis selective modules; running static analysis only.")

    # ----------------------------------------------------------------- collate
    def plv_of(acc, name, s, n):
        return np.abs(acc[name][s].get("sum", np.zeros(n))) / max(
            acc[name][s].get("count", 1), 1)

    def msc_spectrum(acc, name, s):
        return finalize_coherence(acc[name][s])  # (msc, freqs) or (None, None)

    def theta_band_msc(acc, name, s, n):
        msc, freqs = msc_spectrum(acc, name, s)
        if msc is None:
            return np.full(n, np.nan)
        in_band = (freqs >= band[0]) & (freqs <= band[1])
        return msc[in_band].mean(axis=0)

    rows = []
    # Coherence-spectrum stacks for the curve plot (time axis, "matched").
    spec = {s: {"msc": [], "w": [], "lay": [], "freq": None} for s in SIGNALS}
    spec_null = {s: [] for s in SIGNALS}
    heat = {"msc": [], "freq": None, "baseline_hz": []}
    time_names = {n for n, _ in time_mods}
    do_null = args.null == "shuffle" and do_dynamic

    for name, m in selective_mods:
        meta = _module_metadata(name)
        bc = baseline_and_coupling(m, frame_rate, meta["axis"])
        is_time = name in time_names
        n = m.state_dim

        if is_time and do_dynamic:
            acc = stat_acc[name]
            cnt = max(acc["cnt"], 1)
            nu_mean = acc["nu_sum"] / cnt
            nu_std = np.sqrt(np.maximum(acc["nu_sq"] / cnt - nu_mean ** 2, 0.0))
            f_mean = acc["f_sum"] / cnt
            f_std = np.sqrt(np.maximum(acc["f_sq"] / cnt - f_mean ** 2, 0.0))
            plv = {s: plv_of(plv_acc, name, s, n) for s in SIGNALS}
            tmsc = {s: theta_band_msc(coh_acc, name, s, n) for s in SIGNALS}
            if do_null:
                plv_n = {s: plv_of(null_plv_acc, name, s, n) for s in SIGNALS}
                tmsc_n = {s: theta_band_msc(null_coh_acc, name, s, n) for s in SIGNALS}
            # Stash spectra for the aggregate plots (state coherence drives heatmap).
            for s in SIGNALS:
                msc, freqs = msc_spectrum(coh_acc, name, s)
                if msc is not None:
                    spec[s]["msc"].append(msc.T)        # (N, Fk)
                    spec[s]["w"].append(bc["coupling"])
                    spec[s]["lay"].append(np.full(n, meta["layer"]))
                    spec[s]["freq"] = freqs
                    if do_null:
                        mn, _ = msc_spectrum(null_coh_acc, name, s)
                        spec_null[s].append(mn.T if mn is not None
                                            else np.full((n, len(freqs)), np.nan))
            msc_state, freqs_state = msc_spectrum(coh_acc, name, "state")
            if msc_state is not None:
                heat["msc"].append(msc_state.T)
                heat["freq"] = freqs_state
                heat["baseline_hz"].append(bc.get("baseline_freq_hz", np.full(n, np.nan)))
        else:
            nu_mean = nu_std = f_mean = f_std = np.full(n, np.nan)
            plv = {s: np.full(n, np.nan) for s in SIGNALS}
            tmsc = {s: np.full(n, np.nan) for s in SIGNALS}
            plv_n = tmsc_n = {s: np.full(n, np.nan) for s in SIGNALS}

        for k in range(n):
            row = {
                "module": name, "layer": meta["layer"], "axis": meta["axis"],
                "direction": meta["direction"], "channel": k,
                "baseline_lambda_mag": bc["baseline_lambda_mag"][k],
                "baseline_cycles_per_step": bc["baseline_cycles_per_step"][k],
                "baseline_freq_hz": bc.get("baseline_freq_hz",
                                           np.full(n, np.nan))[k],
                "baseline_tau_steps": bc["baseline_tau_steps"][k],
                "b_energy": bc["b_energy"][k], "c_energy": bc["c_energy"][k],
                "coupling": bc["coupling"][k],
                "nu_sel_gain": bc["nu_sel_gain"][k],
                "theta_sel_gain": bc["theta_sel_gain"][k],
                "realized_nu_mean": nu_mean[k], "realized_nu_std": nu_std[k],
                "realized_freq_hz_mean": f_mean[k], "realized_freq_hz_std": f_std[k],
                "state_plv": plv["state"][k], "state_msc": tmsc["state"][k],
                "nu_plv": plv["nu"][k], "nu_msc": tmsc["nu"][k],
                "theta_plv": plv["theta"][k], "theta_msc": tmsc["theta"][k],
            }
            if do_null:
                for s in SIGNALS:
                    row[f"{s}_plv_delta"] = plv[s][k] - plv_n[s][k]
                    row[f"{s}_msc_delta"] = tmsc[s][k] - tmsc_n[s][k]
            rows.append(row)
    df = pd.DataFrame(rows)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "selective_modes.csv")
    df.to_csv(csv_path, index=False)

    written = [csv_path]
    p = os.path.join(args.output_dir, "baseline_poles.png")
    plot_baseline_poles(df, p); written.append(p)
    p = os.path.join(args.output_dir, "selectivity.png")
    plot_selectivity(df, do_dynamic, p); written.append(p)
    if do_dynamic:
        p = os.path.join(args.output_dir, "entrainment_vs_intrinsic.png")
        plot_entrainment_vs_intrinsic(df, band, p); written.append(p)
        if any(spec[s]["msc"] for s in SIGNALS):
            p = os.path.join(args.output_dir, "coordinate_entrainment.png")
            plot_coordinate_entrainment(spec, spec_null, band, frame_rate, p)
            written.append(p)
        if heat["msc"]:
            p = os.path.join(args.output_dir, "coherence_heatmap.png")
            plot_coherence_heatmap(heat, band, frame_rate, p); written.append(p)

    print_summary(df, band, do_dynamic, do_null)
    print("\nWrote:\n  " + "\n  ".join(written))


# --------------------------------------------------------------------------- #
# Plots.
# --------------------------------------------------------------------------- #
def plot_baseline_poles(df: pd.DataFrame, out_path: str):
    """Baseline poles lambda0 = nu0 e^{i theta0} in the z-plane, per axis."""
    axes_present = [a for a in ("time", "freq") if (df["axis"] == a).any()]
    fig, axs = plt.subplots(1, len(axes_present), figsize=(6 * len(axes_present), 6),
                            squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax, axis_name in zip(axs[0], axes_present):
        sub = df[df["axis"] == axis_name]
        ax.plot(np.cos(theta), np.sin(theta), color="k", lw=0.8)
        ax.axhline(0, color="0.8", lw=0.5)
        ax.axvline(0, color="0.8", lw=0.5)
        ang = sub["baseline_cycles_per_step"] * 2.0 * np.pi
        re = sub["baseline_lambda_mag"] * np.cos(ang)
        im = sub["baseline_lambda_mag"] * np.sin(ang)
        sizes = 8 + 200 * (sub["coupling"] / (sub["coupling"].max() + 1e-12))
        sc = ax.scatter(re, im, c=sub["layer"], s=sizes, cmap="viridis",
                        alpha=0.75, edgecolors="none")
        ax.set_aspect("equal"); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{axis_name} axis baseline poles (size = coupling)")
        ax.set_xlabel("Re"); ax.set_ylabel("Im")
        fig.colorbar(sc, ax=ax, label="layer")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_selectivity(df: pd.DataFrame, do_dynamic: bool, out_path: str):
    """Baseline placement vs how much selectivity the model actually spends."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].hist(df["baseline_lambda_mag"], bins=40, color="steelblue")
    axs[0, 0].set_title("baseline |lambda| = sigmoid(c_nu)")
    axs[0, 0].set_xlabel("|lambda0|")

    tm = df[df["axis"] == "time"]
    axs[0, 1].hist(tm["baseline_freq_hz"].dropna(), bins=40, color="purple", alpha=0.7)
    axs[0, 1].set_title("time axis baseline resonant freq")
    axs[0, 1].set_xlabel("Hz")

    if do_dynamic and tm["realized_nu_std"].notna().any():
        axs[1, 0].hist(tm["realized_nu_std"].dropna(), bins=40, color="seagreen")
        axs[1, 0].set_title("realized nu swing (std over speech)")
        axs[1, 0].set_xlabel("std(nu_k)  -- selective forgetting in use")
        axs[1, 1].hist(tm["realized_freq_hz_std"].dropna(), bins=40, color="indianred")
        axs[1, 1].set_title("realized instantaneous-freq swing")
        axs[1, 1].set_xlabel("std(freq_k) [Hz]  -- selective chirp in use")
    else:
        axs[1, 0].hist(df["nu_sel_gain"], bins=40, color="seagreen")
        axs[1, 0].set_title("nu selectivity gain ||W_nu||")
        axs[1, 0].set_xlabel("||W_nu row||")
        axs[1, 1].hist(df["theta_sel_gain"], bins=40, color="indianred")
        axs[1, 1].set_title("theta selectivity gain ||W_theta||")
        axs[1, 1].set_xlabel("||W_theta row||")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_entrainment_vs_intrinsic(df: pd.DataFrame, band, out_path: str):
    """State PLV / coherence vs baseline resonant freq -- resonance vs passthrough."""
    sub = df[(df["axis"] == "time") & df["state_plv"].notna()]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, col, title in ((axs[0], "state_plv", f"state PLV ({band[0]:g}-{band[1]:g} Hz)"),
                           (axs[1], "state_msc", "state coherence (MSC)")):
        sizes = 8 + 200 * (sub["coupling"] / (sub["coupling"].max() + 1e-12))
        for direction, marker in (("fwd", "o"), ("bwd", "^")):
            s = sub[sub["direction"] == direction]
            if not len(s):
                continue
            sc = ax.scatter(s["baseline_freq_hz"], s[col], c=s["layer"],
                            s=sizes[s.index], cmap="viridis", marker=marker,
                            alpha=0.75, edgecolors="none", label=direction)
        ax.axvspan(band[0], band[1], color="green", alpha=0.12, label="theta band")
        ax.set_xlabel("baseline resonant freq (Hz)"); ax.set_ylabel(col)
        ax.set_title(title + "  vs baseline freq (size=coupling)")
        ax.legend(fontsize=8)
        if "sc" in dir():
            fig.colorbar(sc, ax=ax, label="layer")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_coordinate_entrainment(spec, spec_null, band, frame_rate, out_path):
    """Coupling-weighted coherence spectra for state, nu and theta.

    The state curve is the LinOSS-style entrainment; the nu and theta curves are
    the locking of the *selective coordinates themselves* -- present only because
    the transition is input-dependent. A theta-band bump on nu/theta is the
    signature of selectivity tracking the syllabic rhythm.
    """
    nyq = frame_rate / 2.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"state": "k", "nu": "seagreen", "theta": "indianred"}
    for s in SIGNALS:
        if not spec[s]["msc"]:
            continue
        msc = np.concatenate(spec[s]["msc"], axis=0)
        w = np.concatenate(spec[s]["w"])
        freqs = spec[s]["freq"]
        keep = freqs <= min(nyq, 40.0)
        curve = np.average(msc[:, keep], axis=0, weights=w + 1e-12)
        ax.plot(freqs[keep], curve, color=colors[s], lw=2.0, label=f"{s} (matched)")
        if spec_null[s]:
            mn = np.concatenate(spec_null[s], axis=0)
            null_curve = np.average(mn[:, keep], axis=0, weights=w + 1e-12)
            ax.plot(freqs[keep], null_curve, color=colors[s], lw=1.3, ls="--",
                    label=f"{s} (null)")
    ax.axvspan(band[0], band[1], color="green", alpha=0.12, label="theta band")
    ax.set_xlabel("modulation frequency (Hz)")
    ax.set_ylabel("coupling-weighted coherence (MSC)")
    ax.set_title("envelope coherence: oscillator state vs selective nu / theta")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_coherence_heatmap(heat, band, frame_rate, out_path):
    """Per-mode state coherence spectrum, sorted by baseline resonant freq."""
    msc = np.concatenate(heat["msc"], axis=0)            # (Ntot, Fk)
    base = np.concatenate(heat["baseline_hz"])
    order = np.argsort(np.nan_to_num(base, nan=np.inf))
    freqs = heat["freq"]
    keep = freqs <= min(frame_rate / 2.0, 40.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(msc[order][:, keep], origin="lower", aspect="auto", cmap="magma",
                   vmin=0, vmax=np.percentile(msc[:, keep], 99) or 1.0,
                   extent=[freqs[keep].min(), freqs[keep].max(), 0, len(order)])
    ax.axvline(band[0], color="cyan", ls="--", lw=1)
    ax.axvline(band[1], color="cyan", ls="--", lw=1)
    ax.set_xlabel("modulation frequency (Hz)")
    ax.set_ylabel("mode (sorted by baseline resonant freq, low->high)")
    ax.set_title("per-mode state coherence (theta band dashed)")
    fig.colorbar(im, ax=ax, label="coherence (MSC)")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
def print_summary(df, band, do_dynamic, do_null):
    print("\nBaseline placement (time axis):")
    tm = df[df["axis"] == "time"]
    if len(tm):
        in_theta = tm["baseline_freq_hz"].between(*band)
        print(f"  modes with baseline freq in {band[0]:g}-{band[1]:g} Hz: "
              f"{int(in_theta.sum())}/{len(tm)}")
        print(f"  median baseline |lambda| = {tm['baseline_lambda_mag'].median():.3f}")
    if not do_dynamic:
        print("\n(static coupling analysis only; pass --max_utts > 0 for "
              "phase-locking)")
        return

    osc = df[(df["axis"] == "time") & df["state_plv"].notna()].copy()
    w = osc["coupling"].to_numpy()

    def wmean(mask, col):
        s = osc[mask]; ww = s["coupling"].to_numpy(); v = s[col].to_numpy()
        ok = ~np.isnan(v)
        return float(np.average(v[ok], weights=ww[ok])) if ok.any() and ww[ok].sum() else float("nan")

    in_theta = osc["baseline_freq_hz"].between(*band)
    print("\nCoupling-weighted phase-locking (time-axis modes):")
    for s in SIGNALS:
        print(f"  {s:5s}: PLV={wmean(np.ones(len(osc), bool), f'{s}_plv'):.3f}  "
              f"MSC={wmean(np.ones(len(osc), bool), f'{s}_msc'):.3f}")
    print("\nState locking, baseline-in-theta vs off-theta (resonance check):")
    print(f"  in-theta  : PLV={wmean(in_theta, 'state_plv'):.3f}  "
          f"MSC={wmean(in_theta, 'state_msc'):.3f}  (n={int(in_theta.sum())})")
    print(f"  off-theta : PLV={wmean(~in_theta, 'state_plv'):.3f}  "
          f"MSC={wmean(~in_theta, 'state_msc'):.3f}  (n={int((~in_theta).sum())})")
    print("\nRealized selectivity (coupling-weighted means):")
    print(f"  nu swing std        = {wmean(np.ones(len(osc), bool), 'realized_nu_std'):.4f}")
    print(f"  inst-freq swing std = {wmean(np.ones(len(osc), bool), 'realized_freq_hz_std'):.3f} Hz")
    if do_null:
        print("\nMatched-minus-null (>0 ==> genuine locking beyond envelope shape):")
        for s in SIGNALS:
            print(f"  {s:5s}: dPLV={wmean(np.ones(len(osc), bool), f'{s}_plv_delta'):+.3f}  "
                  f"dMSC={wmean(np.ones(len(osc), bool), f'{s}_msc_delta'):+.3f}")


if __name__ == "__main__":
    main()
