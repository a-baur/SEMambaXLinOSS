"""Analyze the learned oscillators and damping of LinOSS mixers in a checkpoint.

Each ``LinOSS`` module is a bank of ``state_dim`` independent damped 2nd-order
oscillators. This script loads a trained SEMamba generator, finds every LinOSS
mixer, reconstructs the *effective* (post-activation, post-clamp) parameters
exactly as the forward pass does, and derives physically meaningful quantities
from the eigenvalues of each oscillator's discrete 2x2 transition matrix.

For every oscillator we report:
  * lambda_mag   -- |eigenvalue|, the per-step decay (1 = undamped/persistent).
  * lambda_angle -- arg(eigenvalue), the per-step oscillation phase advance.
  * zeta         -- continuous-time damping ratio G / (2*sqrt(A)).
  * an axis-aware physical reading:
      - time axis: oscillation frequency in Hz and decay time constant in s
        (frame rate = sampling_rate / hop_size);
      - freq axis: spectral period and correlation bandwidth in Hz
        (bin spacing = sampling_rate / n_fft).
  * b_energy / c_energy -- how strongly the oscillator couples to the input /
    output channels, so near-dead oscillators can be down-weighted.

Outputs (under --output_dir):
  * oscillators.csv  -- one row per oscillator with all of the above.
  * poles.png        -- pole plot in the z-plane vs the unit circle.
  * distributions.png-- histograms of |lambda|, zeta, step, frequency.
  * freq_vs_decay.png-- frequency vs decay scatter (per axis), sized by coupling.

Usage:
  python analyze_linoss.py --checkpoint exp/LinOSS_EARS \
      --output_dir eval_out/linoss_analysis
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")  # headless, matches utils/viz.py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from models.generator import SEMamba
from models.linoss.linoss import LinOSS
from utils.util import load_config


# --------------------------------------------------------------------------- #
# Checkpoint / config resolution (mirrors evaluate.py, kept local to avoid the
# heavy librosa / UTMOS imports that pulling in evaluate.py would trigger).
# --------------------------------------------------------------------------- #
def resolve_checkpoint_and_config(ckpt_arg: str, config_arg: str | None):
    if os.path.isdir(ckpt_arg):
        candidates = sorted(glob.glob(os.path.join(ckpt_arg, "g_????????.pth")))
        if not candidates:
            raise FileNotFoundError(f"No g_*.pth checkpoints under {ckpt_arg}")
        ckpt_file = candidates[-1]
        default_config = os.path.join(ckpt_arg, "config.yaml")
    else:
        if not os.path.isfile(ckpt_arg):
            raise FileNotFoundError(ckpt_arg)
        ckpt_file = ckpt_arg
        default_config = os.path.join(os.path.dirname(ckpt_arg), "config.yaml")
    config_file = config_arg or default_config
    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            f"Config not found at {config_file}; pass --config explicitly."
        )
    return ckpt_file, config_file


def load_generator(ckpt_file: str, cfg: dict, device: torch.device) -> SEMamba:
    model = SEMamba(cfg).to(device)
    state = torch.load(ckpt_file, map_location=device)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Per-module oscillator extraction.
# --------------------------------------------------------------------------- #
def _module_metadata(name: str) -> dict:
    """Parse a LinOSS module's dotted path into (layer, axis, direction).

    Paths look like:
        TSMamba.0.time_mamba.forward_blocks.0.mixer.ssm
    """
    parts = name.split(".")
    layer = next((int(p) for prev, p in zip(parts, parts[1:])
                  if prev == "TSMamba" and p.isdigit()), -1)
    if "time_mamba" in parts:
        axis = "time"
    elif "freq_mamba" in parts:
        axis = "freq"
    else:
        axis = "?"
    if "forward_blocks" in parts:
        direction = "fwd"
    elif "backward_blocks" in parts:
        direction = "bwd"
    else:
        direction = "?"
    return {"layer": layer, "axis": axis, "direction": direction}


def _effective_params(m: LinOSS):
    """Reconstruct (M-matrix entries, A_eff, G_eff, step) exactly as forward().

    Returns numpy arrays of shape (state_dim,). M entries are the per-oscillator
    diagonal entries of the 2x2 discrete transition matrix.
    """
    step = torch.sigmoid(m.steps)

    if m.discretization == "IM":
        A = torch.relu(m.A_diag)
        G = torch.zeros_like(A)
        schur = 1.0 / (1.0 + step ** 2 * A)
        M11 = 1.0 - step ** 2 * A * schur
        M12 = -step * A * schur
        M21 = step * schur
        M22 = schur
        F1, F2 = M11 * step, M21 * step
    elif m.damping:
        G = torch.relu(m.G_diag)
        sqrt_term = torch.sqrt(1.0 + step * G)
        A_low = (2.0 + step * G - 2.0 * sqrt_term) / step ** 2
        A_high = (2.0 + step * G + 2.0 * sqrt_term) / step ** 2
        A = A_low + torch.relu(m.A_diag - A_low) - torch.relu(m.A_diag - A_high)
        inv_S = 1.0 / (1.0 + step * G)
        M11 = inv_S
        M12 = -step * inv_S * A
        M21 = step * inv_S
        M22 = 1.0 - step ** 2 * inv_S * A
        F1, F2 = step * inv_S, step ** 2 * inv_S
    else:  # IMEX, no damping
        A = torch.relu(m.A_diag)
        G = torch.zeros_like(A)
        M11 = torch.ones_like(A)
        M12 = -step * A
        M21 = step
        M22 = 1.0 - step ** 2 * A
        F1, F2 = step, step ** 2

    to_np = lambda t: t.detach().cpu().numpy()
    # F1/F2 are the per-channel gains by which the (scalar) projected input Bu
    # drives the two state components: F_t = [F1; F2] * Bu_t. The output reads
    # the second component, so the SISO transfer is H(z)=[0,1](I-z^-1 M)^-1[F1;F2].
    return {
        "M11": to_np(M11), "M12": to_np(M12),
        "M21": to_np(M21), "M22": to_np(M22),
        "F1": to_np(F1), "F2": to_np(F2),
        "A": to_np(A), "G": to_np(G), "step": to_np(step),
    }


def _eig_2x2(M11, M12, M21, M22):
    """Closed-form eigenvalues of per-channel 2x2 matrices (complex)."""
    tr = M11 + M22
    det = M11 * M22 - M12 * M21
    disc = (tr ** 2 - 4.0 * det).astype(np.complex128)
    root = np.sqrt(disc)
    lam1 = (tr + root) / 2.0
    lam2 = (tr - root) / 2.0
    # Pick the eigenvalue with the larger magnitude as the dominant pole;
    # for a complex-conjugate pair they share a magnitude, so take the one
    # with non-negative imaginary part for a consistent angle sign.
    mag1, mag2 = np.abs(lam1), np.abs(lam2)
    dominant = np.where(mag1 >= mag2, lam1, lam2)
    # For oscillatory (complex) pairs, prefer the +imag representative.
    osc = np.abs(dominant.imag) > 1e-9
    dominant = np.where(osc & (dominant.imag < 0), np.conjugate(dominant), dominant)
    spectral_radius = np.maximum(mag1, mag2)
    return dominant, spectral_radius, osc


def transfer_function(p: dict, omega: np.ndarray) -> np.ndarray:
    """Per-oscillator SISO frequency response |H(e^{jw})|, complex.

    H(z) = [0,1] (I - z^-1 M)^-1 [F1; F2], evaluated at z = e^{jw}.
    Returns an (N_channels, len(omega)) complex array.
    """
    q = np.exp(-1j * omega)[None, :]                       # (1, W), z^-1
    M11 = p["M11"][:, None]; M12 = p["M12"][:, None]
    M21 = p["M21"][:, None]; M22 = p["M22"][:, None]
    F1 = p["F1"][:, None]; F2 = p["F2"][:, None]
    num = q * M21 * F1 + (1.0 - q * M11) * F2
    den = (1.0 - q * M11) * (1.0 - q * M22) - q ** 2 * M12 * M21
    return num / den


def impulse_response(p: dict, n_steps: int) -> np.ndarray:
    """Per-oscillator impulse response y_t = [0,1] M^t [F1; F2], real.

    Returns an (N_channels, n_steps) real array (the output-component trajectory
    in response to a unit impulse at t=0).
    """
    y1 = p["F1"].astype(np.float64).copy()
    y2 = p["F2"].astype(np.float64).copy()
    out = np.empty((y1.shape[0], n_steps), dtype=np.float64)
    out[:, 0] = y2
    for t in range(1, n_steps):
        y1, y2 = (p["M11"] * y1 + p["M12"] * y2,
                  p["M21"] * y1 + p["M22"] * y2)
        out[:, t] = y2
    return out


def analyze_module(name: str, m: LinOSS, stft: dict) -> pd.DataFrame:
    meta = _module_metadata(name)
    p = _effective_params(m)
    lam, spec_radius, osc = _eig_2x2(p["M11"], p["M12"], p["M21"], p["M22"])

    mag = np.abs(lam)
    angle = np.abs(np.angle(lam))  # rad/step, in [0, pi]

    # Decay time constant in steps (frames/bins): |lambda| = exp(-1/tau).
    with np.errstate(divide="ignore"):
        tau_steps = np.where(mag < 1.0, -1.0 / np.log(np.clip(mag, 1e-12, 1.0)), np.inf)

    # Continuous-time damping ratio zeta = G / (2*sqrt(A)).
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = p["G"] / (2.0 * np.sqrt(np.clip(p["A"], 1e-12, None)))

    cycles_per_step = angle / (2.0 * np.pi)

    sr = stft["sampling_rate"]
    hop = stft["hop_size"]
    n_fft = stft["n_fft"]
    frame_rate = sr / hop          # Hz, for the time axis
    bin_hz = sr / n_fft            # Hz per frequency bin, for the freq axis

    # Axis-aware physical readings. The time axis carries a true temporal
    # modulation frequency (Hz); the freq axis carries a *spectral* modulation
    # rate expressed as cycles per kHz (bounded in [0, 0.5 * 1000/bin_hz]),
    # which stays finite for near-DC oscillators unlike a spectral period.
    freq_hz = np.full_like(mag, np.nan)            # time axis only
    spec_cyc_per_khz = np.full_like(mag, np.nan)   # freq axis only
    if meta["axis"] == "time":
        freq_hz = cycles_per_step * frame_rate
        tau_phys = tau_steps / frame_rate          # seconds
        phys_unit = "tau_sec"
    elif meta["axis"] == "freq":
        spec_cyc_per_khz = cycles_per_step * (1000.0 / bin_hz)
        tau_phys = tau_steps * bin_hz              # correlation bandwidth in Hz
        phys_unit = "bandwidth_hz"
    else:
        tau_phys = np.full_like(mag, np.nan)
        phys_unit = "na"

    # Resonance sharpness (pole-based Q) and effective receptive field.
    # Q = w0 / bandwidth ~= angle / (2 (1 - |lambda|)) for a pole near the
    # unit circle; large Q = narrowband resonator, small Q = broad/damped.
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.where(osc, angle / (2.0 * np.maximum(1.0 - mag, 1e-12)), np.nan)
    # Receptive field: steps until the |lambda|^t envelope falls to -20 dB.
    with np.errstate(divide="ignore"):
        rf_steps = np.where(mag < 1.0,
                            np.log(0.1) / np.log(np.clip(mag, 1e-12, 1.0)),
                            np.inf)
    rf_ms = rf_steps / frame_rate * 1000.0 if meta["axis"] == "time" else np.full_like(mag, np.nan)

    # Peak gain of each oscillator's frequency response (dB).
    omega = np.linspace(0.0, np.pi, 512)
    H = transfer_function(p, omega)
    peak_gain_db = 20.0 * np.log10(np.abs(H).max(axis=1) + 1e-12)

    # Input/output coupling strength per oscillator.
    B = torch.complex(m.B[..., 0], m.B[..., 1]).detach().cpu().numpy()  # (N, F)
    C = torch.complex(m.C[..., 0], m.C[..., 1]).detach().cpu().numpy()  # (F, N)
    b_energy = np.abs(B).mean(axis=1)           # per state channel
    c_energy = np.abs(C).mean(axis=0)
    coupling = b_energy * c_energy

    n = m.state_dim
    df = pd.DataFrame({
        "module": name,
        "layer": meta["layer"],
        "axis": meta["axis"],
        "direction": meta["direction"],
        "channel": np.arange(n),
        "discretization": m.discretization,
        "damping": bool(m.damping),
        "step": p["step"],
        "A_eff": p["A"],
        "G_eff": p["G"],
        "zeta": zeta,
        "lambda_mag": mag,
        "lambda_angle_rad": angle,
        "spectral_radius": spec_radius,
        "oscillatory": osc,
        "tau_steps": tau_steps,
        "cycles_per_step": cycles_per_step,
        "freq_hz": freq_hz,
        "spec_cyc_per_khz": spec_cyc_per_khz,
        phys_unit: tau_phys,
        "Q": Q,
        "rf_steps": rf_steps,
        "rf_ms": rf_ms,
        "peak_gain_db": peak_gain_db,
        "b_energy": b_energy,
        "c_energy": c_energy,
        "coupling": coupling,
    })
    return df


# --------------------------------------------------------------------------- #
# Plotting.
# --------------------------------------------------------------------------- #
def plot_poles(df: pd.DataFrame, out_path: str):
    axes_present = [a for a in ("time", "freq") if (df["axis"] == a).any()]
    fig, axs = plt.subplots(1, len(axes_present), figsize=(6 * len(axes_present), 6),
                            squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax, axis_name in zip(axs[0], axes_present):
        sub = df[df["axis"] == axis_name]
        ax.plot(np.cos(theta), np.sin(theta), color="k", lw=0.8)
        ax.axhline(0, color="0.8", lw=0.5)
        ax.axvline(0, color="0.8", lw=0.5)
        re = sub["lambda_mag"] * np.cos(sub["lambda_angle_rad"])
        im = sub["lambda_mag"] * np.sin(sub["lambda_angle_rad"])
        sizes = 8 + 200 * (sub["coupling"] / (sub["coupling"].max() + 1e-12))
        sc = ax.scatter(re, im, c=sub["layer"], s=sizes, cmap="viridis",
                        alpha=0.75, edgecolors="none")
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.05, 1.05)  # upper half-plane (we fold conjugates)
        ax.set_title(f"{axis_name} axis poles (size = coupling)")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        fig.colorbar(sc, ax=ax, label="layer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, out_path: str, init_r_min: float | None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    osc = df[df["oscillatory"]]

    axs[0, 0].hist(df["lambda_mag"], bins=40, color="steelblue")
    if init_r_min is not None:
        axs[0, 0].axvspan(init_r_min, 1.0, color="orange", alpha=0.2,
                          label=f"init range [{init_r_min:g}, 1]")
        axs[0, 0].legend()
    axs[0, 0].set_title("|lambda| (per-step decay)")
    axs[0, 0].set_xlabel("|lambda|")

    zeta = df["zeta"].replace([np.inf, -np.inf], np.nan).dropna()
    axs[0, 1].hist(np.clip(zeta, 0, np.percentile(zeta, 99) if len(zeta) else 1),
                   bins=40, color="indianred")
    axs[0, 1].axvline(1.0, color="k", ls="--", lw=1, label="critical (zeta=1)")
    axs[0, 1].legend()
    axs[0, 1].set_title("damping ratio zeta = G / (2 sqrt(A))")
    axs[0, 1].set_xlabel("zeta")

    axs[1, 0].hist(df["step"], bins=40, color="seagreen")
    axs[1, 0].set_title("step (Delta = sigmoid(steps))")
    axs[1, 0].set_xlabel("step")

    time_osc = osc[osc["axis"] == "time"]
    if len(time_osc):
        axs[1, 1].hist(time_osc["freq_hz"].dropna(), bins=40, color="purple",
                       alpha=0.7)
        axs[1, 1].set_title("time axis: oscillation frequency")
        axs[1, 1].set_xlabel("Hz")
    else:
        freq_osc = osc[osc["axis"] == "freq"]
        axs[1, 1].hist(freq_osc["spec_cyc_per_khz"].dropna(), bins=40,
                       color="darkorange", alpha=0.7)
        axs[1, 1].set_title("freq axis: spectral modulation")
        axs[1, 1].set_xlabel("cycles per kHz")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_freq_vs_decay(df: pd.DataFrame, out_path: str):
    osc = df[df["oscillatory"]]
    axes_present = [a for a in ("time", "freq") if (osc["axis"] == a).any()]
    if not axes_present:
        return
    fig, axs = plt.subplots(1, len(axes_present), figsize=(7 * len(axes_present), 5),
                            squeeze=False)
    for ax, axis_name in zip(axs[0], axes_present):
        sub = osc[osc["axis"] == axis_name]
        ycol = "tau_sec" if axis_name == "time" else "bandwidth_hz"
        xcol = "freq_hz" if axis_name == "time" else "spec_cyc_per_khz"
        y = sub[ycol].replace([np.inf, -np.inf], np.nan)
        sizes = 8 + 200 * (sub["coupling"] / (sub["coupling"].max() + 1e-12))
        sc = ax.scatter(sub[xcol], y, c=sub["layer"], s=sizes,
                        cmap="viridis", alpha=0.75, edgecolors="none")
        ax.set_yscale("log")
        ax.set_xlabel("oscillation freq (Hz)" if axis_name == "time"
                      else "spectral modulation (cycles/kHz)")
        ax.set_ylabel("decay time tau (s)" if axis_name == "time"
                      else "correlation bandwidth (Hz)")
        ax.set_title(f"{axis_name} axis: frequency vs decay")
        fig.colorbar(sc, ax=ax, label="layer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# Speech temporal modulation spectrum peaks around the ~4 Hz syllabic rate,
# with most intelligibility-relevant energy in the 2-16 Hz band. Marked on the
# time-axis modulation transfer plot to see whether LinOSS allocates resonators
# there.
SPEECH_MOD_BAND_HZ = (2.0, 16.0)
SPEECH_MOD_PEAK_HZ = 4.0


def _axis_freq_grid(omega: np.ndarray, axis_name: str, stft: dict):
    """Map normalized angular frequency (rad/step) to a physical x-axis."""
    sr, n_fft, hop = stft["sampling_rate"], stft["n_fft"], stft["hop_size"]
    if axis_name == "time":
        # omega in [0, pi] -> [0, Nyquist] Hz, Nyquist = frame_rate / 2.
        return omega / np.pi * (sr / hop / 2.0), "temporal modulation (Hz)"
    bin_hz = sr / n_fft
    return omega / (2 * np.pi) * (1000.0 / bin_hz), "spectral modulation (cycles/kHz)"


def _collect_axis(linoss_mods, axis_name, omega, n_ir):
    """Stack per-oscillator responses for every module on one axis."""
    Hs, irs, couplings, layers = [], [], [], []
    for name, m in linoss_mods:
        if _module_metadata(name)["axis"] != axis_name:
            continue
        p = _effective_params(m)
        Hs.append(transfer_function(p, omega))
        irs.append(impulse_response(p, n_ir))
        B = torch.complex(m.B[..., 0], m.B[..., 1]).detach().cpu().numpy()
        C = torch.complex(m.C[..., 0], m.C[..., 1]).detach().cpu().numpy()
        couplings.append(np.abs(B).mean(1) * np.abs(C).mean(0))
        layers.append(np.full(m.d_state, _module_metadata(name)["layer"]))
    if not Hs:
        return None
    return {
        "H": np.concatenate(Hs, 0),
        "ir": np.concatenate(irs, 0),
        "coupling": np.concatenate(couplings),
        "layer": np.concatenate(layers),
    }


def _axes_present(linoss_mods):
    return [a for a in ("time", "freq")
            if any(_module_metadata(n)["axis"] == a for n, _ in linoss_mods)]


def plot_freq_response(linoss_mods, stft, out_path):
    """Filterbank view: each oscillator's frequency response, peak-normalized,
    stacked and sorted by resonant frequency. A clean diagonal ridge means the
    bank tiles the modulation axis like a wavelet/Gabor filterbank."""
    omega = np.linspace(0.0, np.pi, 512)
    present = _axes_present(linoss_mods)
    fig, axs = plt.subplots(1, len(present), figsize=(7 * len(present), 6),
                            squeeze=False)
    for ax, axis_name in zip(axs[0], present):
        d = _collect_axis(linoss_mods, axis_name, omega, 1)
        mag = np.abs(d["H"])
        magdb = 20.0 * np.log10(mag + 1e-12)
        magdb = magdb - magdb.max(axis=1, keepdims=True)   # peak-normalize per row
        order = np.argsort(mag.argmax(axis=1))             # sort by resonant freq
        x, xlabel = _axis_freq_grid(omega, axis_name, stft)
        im = ax.imshow(magdb[order], origin="lower", aspect="auto", cmap="magma",
                       vmin=-40, vmax=0,
                       extent=[x.min(), x.max(), 0, len(order)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("oscillator (sorted by resonant freq)")
        ax.set_title(f"{axis_name} axis: per-oscillator frequency response")
        fig.colorbar(im, ax=ax, label="|H| (dB, peak-normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_modulation_transfer(linoss_mods, stft, out_path):
    """Coupling-weighted aggregate |H| per layer -- the modulation transfer
    function of each block. For the time axis this is the temporal modulation
    passband; the 2-16 Hz speech syllabic band is shaded for reference."""
    omega = np.linspace(0.0, np.pi, 512)
    present = _axes_present(linoss_mods)
    fig, axs = plt.subplots(1, len(present), figsize=(7 * len(present), 5),
                            squeeze=False)
    for ax, axis_name in zip(axs[0], present):
        d = _collect_axis(linoss_mods, axis_name, omega, 1)
        mag, w, layers = np.abs(d["H"]), d["coupling"], d["layer"]
        x, xlabel = _axis_freq_grid(omega, axis_name, stft)
        for layer in sorted(np.unique(layers)):
            sel = layers == layer
            mtf = (w[sel, None] * mag[sel]).sum(0) / (w[sel].sum() + 1e-12)
            ax.plot(x, mtf, alpha=0.55, label=f"layer {layer}")
        overall = (w[:, None] * mag).sum(0) / (w.sum() + 1e-12)
        ax.plot(x, overall, color="k", lw=2.2, label="overall")
        if axis_name == "time":
            ax.axvspan(*SPEECH_MOD_BAND_HZ, color="green", alpha=0.12,
                       label="speech 2-16 Hz")
            ax.axvline(SPEECH_MOD_PEAK_HZ, color="green", ls="--", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("coupling-weighted |H|")
        ax.set_title(f"{axis_name} axis: modulation transfer function")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_impulse_responses(linoss_mods, stft, out_path, top_k=48):
    """Learned kernels: impulse responses of the most strongly-coupled
    oscillators, peak-normalized and sorted by dominant frequency. Reveals the
    decaying-sinusoid (Gabor-like) kernels the bank applies along each axis."""
    sr, n_fft, hop = stft["sampling_rate"], stft["n_fft"], stft["hop_size"]
    frame_rate, bin_hz = sr / hop, sr / n_fft
    present = _axes_present(linoss_mods)
    fig, axs = plt.subplots(1, len(present), figsize=(7 * len(present), 6),
                            squeeze=False)
    for ax, axis_name in zip(axs[0], present):
        n_ir = 160 if axis_name == "time" else (n_fft // 2 + 1)  # 1.0 s / full band
        d = _collect_axis(linoss_mods, axis_name, np.array([0.0]), n_ir)
        ir, coup = d["ir"], d["coupling"]
        idx = np.argsort(coup)[::-1][:min(top_k, len(coup))]
        ir = ir[idx]
        dom = np.abs(np.fft.rfft(ir, axis=1))[:, 1:].argmax(1)  # skip DC
        ir = ir[np.argsort(dom)]
        ir = ir / (np.abs(ir).max(axis=1, keepdims=True) + 1e-12)
        if axis_name == "time":
            x_max, xlabel = n_ir / frame_rate * 1000.0, "time lag (ms)"
        else:
            x_max, xlabel = n_ir * bin_hz / 1000.0, "frequency lag (kHz)"
        im = ax.imshow(ir, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-1, vmax=1, extent=[0, x_max, 0, len(ir)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"top-{len(ir)} coupled oscillators (sorted by freq)")
        ax.set_title(f"{axis_name} axis: learned impulse-response kernels")
        fig.colorbar(im, ax=ax, label="normalized response")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _init_r_min(cfg: dict) -> float | None:
    """Best-effort fetch of the LinOSS init radius for the |lambda| overlay."""
    mc = cfg["model_cfg"]
    for key in ("time_mixer", "freq_mixer"):
        sub = mc.get(key)
        if isinstance(sub, dict) and "linoss_r_min" in sub:
            return float(sub["linoss_r_min"])
    return float(mc["linoss_r_min"]) if "linoss_r_min" in mc else None


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="exp/LinOSS_EARS",
                        help="A g_*.pth file or an exp directory containing one.")
    parser.add_argument("--config", default=None,
                        help="YAML config. Defaults to config.yaml next to the ckpt.")
    parser.add_argument("--output_dir", default="eval_out/linoss_analysis")
    parser.add_argument("--device", default="cpu",
                        help="cpu is fine; analysis is tiny.")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_file, config_file = resolve_checkpoint_and_config(args.checkpoint, args.config)
    print(f"Checkpoint : {ckpt_file}")
    print(f"Config     : {config_file}")

    cfg = load_config(config_file)
    model = load_generator(ckpt_file, cfg, device)

    linoss_mods = [(name, m) for name, m in model.named_modules()
                   if isinstance(m, LinOSS)]
    if not linoss_mods:
        raise SystemExit("No LinOSS modules found in this checkpoint "
                         "(pure-Mamba model?). Nothing to analyze.")
    print(f"Found {len(linoss_mods)} LinOSS modules "
          f"({sum(m.state_dim for _, m in linoss_mods)} oscillators total)")

    dfs = [analyze_module(name, m, cfg["stft_cfg"]) for name, m in linoss_mods]
    df = pd.concat(dfs, ignore_index=True)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "oscillators.csv")
    df.to_csv(csv_path, index=False)

    stft = cfg["stft_cfg"]
    outputs = {
        "poles.png": lambda p: plot_poles(df, p),
        "distributions.png": lambda p: plot_distributions(df, p, _init_r_min(cfg)),
        "freq_vs_decay.png": lambda p: plot_freq_vs_decay(df, p),
        "freq_response.png": lambda p: plot_freq_response(linoss_mods, stft, p),
        "modulation_transfer.png": lambda p: plot_modulation_transfer(linoss_mods, stft, p),
        "impulse_responses.png": lambda p: plot_impulse_responses(linoss_mods, stft, p),
    }
    for fname, fn in outputs.items():
        fn(os.path.join(args.output_dir, fname))

    # Console summary per (axis, layer).
    print("\nPer axis/layer summary (oscillatory channels only):")
    osc = df[df["oscillatory"]]
    summary = osc.groupby(["axis", "layer"]).agg(
        n_osc=("channel", "size"),
        med_freq_hz=("freq_hz", "median"),          # time axis
        med_cyc_per_khz=("spec_cyc_per_khz", "median"),  # freq axis
        med_lambda=("lambda_mag", "median"),
        med_zeta=("zeta", "median"),
        med_Q=("Q", "median"),
        med_rf_ms=("rf_ms", "median"),              # time axis
    ).round(3)
    print(summary.to_string())
    frac_osc = df["oscillatory"].mean()
    print(f"\nOscillatory fraction: {frac_osc:.1%} "
          f"({df['oscillatory'].sum()}/{len(df)})")
    print("\nWrote:\n  " + csv_path + "\n  "
          + "\n  ".join(os.path.join(args.output_dir, f) for f in outputs))


if __name__ == "__main__":
    main()
