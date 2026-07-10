"""Plot per-channel TF activations, output spectrograms, and error summaries for signals.

For each probe signal this renders the input/output magnitude spectrograms, the
magnitude error (out - in), and a **phase-error** panel (anti-wrapped output vs
input phase, masked to the active region) — the phase panel is the one that
surfaces the voiced-frame deficit that separates LinOSS from Mamba.

Pass exactly two checkpoints to also get a side-by-side comparison figure per
signal (model A | model B | A - B) for both magnitude and phase.

Example
-------
    # single model
    python evaluation/signal_response_viz.py --checkpoint ckpts/mamba \
        --signals all --plot_mode both --index 5
    # compare two models side-by-side (deterministic noise via --seed)
    python evaluation/signal_response_viz.py \
        --checkpoint ckpts/mamba ckpts/linoss --signals all --seed 0
"""

import argparse
import os
import sys
import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from models.stfts import mag_phase_stft
from models.loss import anti_wrapping_function
from utils.util import load_config
from plot_tf_activations import (
    TFActivationCapture,
    reduce_channels,
    to_db,
    resolve_checkpoint_and_config,
    load_generator,
    run_name_for,
    expand_checkpoints,
    resolve_pair,
)

# --- Synthetic Signal Generators ---

def apply_time_mask(signal, sr, duration, start_frac=1/3, end_frac=2/3, fade_len=0.02):
    mask = np.zeros_like(signal)
    start_idx = int(sr * duration * start_frac)
    end_idx = int(sr * duration * end_frac)
    fade_samples = int(sr * fade_len)

    mask[start_idx:end_idx] = 1.0
    if fade_samples > 0:
        window = np.hanning(fade_samples * 2)
        mask[start_idx:start_idx+fade_samples] = window[:fade_samples]
        mask[end_idx-fade_samples:end_idx] = window[fade_samples:]

    return signal * mask

def generate_vowel(sr, duration=2.0, f0=200):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    for i in range(1, 10):
        signal += (1.0 / i) * np.sin(2 * np.pi * (f0 * i) * t)
    return apply_time_mask(signal, sr, duration)

def generate_fricative(sr, duration=2.0):
    noise = np.random.randn(int(sr * duration))
    return apply_time_mask(noise, sr, duration)

def generate_plosive(sr, duration=2.0):
    signal = np.zeros(int(sr * duration))
    burst_start = int(sr * (duration / 2.0)) - int(sr * 0.01)
    burst_end = burst_start + int(sr * 0.02)
    signal[burst_start:burst_end] = np.random.randn(burst_end - burst_start)
    return signal

def generate_chirp(sr, duration=2.0, f_start=20, f_end=8000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return apply_time_mask(np.sin(phase), sr, duration, start_frac=0.1, end_frac=0.9)

def generate_step(sr, duration=2.0):
    noise = np.random.randn(int(sr * duration))
    mid = int(sr * (duration / 2.0))
    noise[mid:] = 0.0
    return apply_time_mask(noise, sr, duration, start_frac=0.2, end_frac=0.8)

def generate_am_noise(sr, duration=2.0, mod_rate=4.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.randn(int(sr * duration))
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * mod_rate * t))
    return apply_time_mask(noise * envelope, sr, duration, start_frac=0.1, end_frac=0.9)

def generate_dynamic_f0(sr, duration=2.0, f_base=200, vibrato_rate=5.0, vibrato_depth=20):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    phase = 2 * np.pi * f_base * t - (vibrato_depth / vibrato_rate) * np.cos(2 * np.pi * vibrato_rate * t)
    sig = np.zeros_like(t)
    for i in range(1, 6):
        sig += (1.0 / i) * np.sin(i * phase)
    return apply_time_mask(sig, sr, duration)

def generate_two_tone(sr, duration=2.0, f1=250, f2=3000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone1 = np.sin(2 * np.pi * f1 * t)
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 2 * t))
    tone2 = np.sin(2 * np.pi * f2 * t) * env
    return apply_time_mask(tone1 + tone2, sr, duration, start_frac=0.1, end_frac=0.9)

def normalize_energy(sig):
    rms = np.sqrt(np.mean(sig**2))
    return sig / (rms + 1e-8)


# --- Core Execution & Plotting ---

def plot_summary_bars(summary_stats, output_dir, mean_key, std_key, ylabel, title, fname):
    """Grouped bar chart of one per-signal scalar metric, one bar group per model.

    ``summary_stats[model][signal]`` is a dict; ``mean_key`` is the bar height and
    ``std_key`` (may be ``None``) the error bar.
    """
    models = [m for m in summary_stats.keys() if summary_stats[m]]
    if not models:
        return

    signals = list(summary_stats[models[0]].keys())
    if not signals:
        return

    x = np.arange(len(signals))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(10, len(signals) * 1.5), 6))

    for i, model in enumerate(models):
        means = [summary_stats[model].get(sig, {}).get(mean_key, np.nan) for sig in signals]
        if std_key is not None:
            stds = [summary_stats[model].get(sig, {}).get(std_key, np.nan) for sig in signals]
        else:
            stds = None

        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=model, capsize=4, alpha=0.85)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(signals, rotation=45, ha="right")

    ax.legend(title="Checkpoints", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote summary plot to {out_path}")


def render_signal(model, sig_np, sig_name, cfg, device, args, ckpt_name):
    sr = cfg["stft_cfg"]["sampling_rate"]
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress = cfg["model_cfg"]["compress_factor"]

    sig_t = torch.from_numpy(sig_np).float().to(device)
    norm = torch.sqrt(len(sig_t) / torch.sum(sig_t**2.0 + 1e-8))
    sig_in = (sig_t * norm).unsqueeze(0)

    mag, pha, _ = mag_phase_stft(sig_in, n_fft, hop_size, win_size, compress)

    with TFActivationCapture(model) as cap, torch.no_grad():
        mag_out, pha_out, complex_out = model(mag, pha)

    # Anti-wrapped phase error (out vs input), same operator as the training
    # phase loss. In near-silence the phase is meaningless, so we mask the panel
    # to the active region and weight the scalar metric by input magnitude.
    phase_err = anti_wrapping_function(pha_out - pha)[0].detach().cpu().numpy()  # [F, T]

    mag = torch.pow(mag, 1 / compress)
    mag_out = torch.pow(mag_out, 1 / compress)

    records = cap.records
    num_layers = len(records)

    sig_lin = mag[0].detach().cpu().numpy()
    out_lin = mag_out[0].detach().cpu().numpy()

    spec_ref = max(sig_lin.max(), out_lin.max())
    sig_disp = to_db(sig_lin, spec_ref)
    out_disp = to_db(out_lin, spec_ref)

    error_disp = out_disp - sig_disp
    mean_err = float(np.mean(error_disp))
    std_err = float(np.std(error_disp))

    # Two summaries: the mean over the *whole* field (shows distortion that
    # propagates into originally-silent regions) and an input-magnitude-weighted
    # mean (the in-band phase error, i.e. on the signal's own support).
    full_phase_err = float(np.mean(phase_err))
    w_phase_err = float((sig_lin * phase_err).sum() / (sig_lin.sum() + 1e-12))

    extent_spec = [0, len(sig_np) / sr, 0, sr / 2000.0]
    safe_name = sig_name.replace(' ', '_').lower()

    sig_dir = os.path.join(args.output_dir, safe_name)
    os.makedirs(sig_dir, exist_ok=True)

    def show_panel(ax, data, title, cmap, is_spec, vmin, vmax):
        im = ax.imshow(
            data, origin="lower", aspect="auto", cmap=cmap,
            extent=extent_spec if is_spec else None, vmin=vmin, vmax=vmax,
        )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("kHz" if is_spec else "freq bin")
        return im

    if args.plot_mode in ["output", "both"]:
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.5), constrained_layout=True)

        im1 = show_panel(axes[0], sig_disp, f"{sig_name} Input (dB)", args.cmap, True, -80.0, 0.0)
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.02)

        im2 = show_panel(axes[1], out_disp, f"{sig_name} Model Output (dB)", args.cmap, True, -80.0, 0.0)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.02)

        err_lim = float(max(abs(np.percentile(error_disp, 99)), abs(np.percentile(error_disp, 1))))
        err_lim = max(err_lim, 1.0)

        title_err = f"Mag Error (Out - In)\nMean: {mean_err:.2f} dB | Std: {std_err:.2f} dB"
        im3 = show_panel(axes[2], error_disp, title_err, "RdBu_r", True, -err_lim, err_lim)
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.02, label="Δ dB")

        title_pha = (f"Phase Error |Δφ| (full field)\n"
                     f"full: {full_phase_err:.3f} | in-band: {w_phase_err:.3f} rad")
        im4 = show_panel(axes[3], phase_err, title_pha, "viridis", True, 0.0, np.pi)
        fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.02, label="rad")

        for ax in axes:
            ax.set_xlabel("time (s)")

        fig.suptitle(f"Model Output Spectrograms — {sig_name}\n{ckpt_name}", fontsize=11)
        out_path = os.path.join(sig_dir, f"out_spec_{ckpt_name}_{safe_name}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)

    if args.plot_mode in ["activations", "both"]:
        act_maps = []
        for after_time, after_freq in records:
            act_maps.append((
                reduce_channels(after_time, args.reduce, args.per_channel_norm),
                reduce_channels(after_freq, args.reduce, args.per_channel_norm),
            ))

        lo, hi = args.clip
        def act_limits(data):
            return float(np.percentile(data, lo)), float(np.percentile(data, hi))

        n_rows = 1 + num_layers
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(11, 2.6 * n_rows), squeeze=False, constrained_layout=True
        )

        im_in = show_panel(axes[0][0], sig_disp, f"{sig_name} Input (dB)", args.cmap, True, -80.0, 0.0)
        fig.colorbar(im_in, ax=axes[0][0], fraction=0.046, pad=0.02)

        im_out = show_panel(axes[0][1], out_disp, f"{sig_name} Model Output (dB)", args.cmap, True, -80.0, 0.0)
        fig.colorbar(im_out, ax=axes[0][1], fraction=0.046, pad=0.02)

        unit = {"std": "z", "max": "norm", "none": "raw"}[args.per_channel_norm]
        for i, (t_map, f_map) in enumerate(act_maps):
            tag = f"{args.reduce}/{unit}"
            v_t = act_limits(t_map)
            v_f = act_limits(f_map)

            im_t = show_panel(axes[i + 1][0], t_map, f"block {i}: after time mixer ({tag})", args.cmap, False, v_t[0], v_t[1])
            fig.colorbar(im_t, ax=axes[i + 1][0], fraction=0.046, pad=0.02)

            im_f = show_panel(axes[i + 1][1], f_map, f"block {i}: after freq mixer ({tag})", args.cmap, False, v_f[0], v_f[1])
            fig.colorbar(im_f, ax=axes[i + 1][1], fraction=0.046, pad=0.02)

        axes[-1][0].set_xlabel("time frame")
        axes[-1][1].set_xlabel("time frame")
        fig.suptitle(f"TF-block channel activations — {sig_name}\n{ckpt_name}", fontsize=11)

        out_path = os.path.join(sig_dir, f"tf_act_{ckpt_name}_{safe_name}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)

    return {
        "sig_lin": sig_lin,
        "out_lin": out_lin,
        "phase_err": phase_err,
        "extent": extent_spec,
        "mean_err": mean_err,
        "std_err": std_err,
        "w_phase_err": w_phase_err,
        "full_phase_err": full_phase_err,
    }


def render_comparison(res_a, res_b, name_a, name_b, sig_name, args):
    """Side-by-side comparison of two models on one signal.

    Two rows (magnitude dB, phase error) × four columns
    (input | model A | model B | A - B). The A - B column uses a diverging
    scale so wherever the two mixers diverge lights up directly.
    """
    extent = res_a["extent"]
    safe_name = sig_name.replace(" ", "_").lower()

    # Shared dB reference across input + both outputs so the diff is unbiased.
    spec_ref = max(res_a["sig_lin"].max(), res_a["out_lin"].max(), res_b["out_lin"].max())
    in_disp = to_db(res_a["sig_lin"], spec_ref)
    out_a = to_db(res_a["out_lin"], spec_ref)
    out_b = to_db(res_b["out_lin"], spec_ref)
    mag_diff = out_a - out_b
    mdiff_lim = max(float(np.percentile(np.abs(mag_diff), 99)), 1.0)

    # Full-field phase error (no masking) so distortion propagating outside the
    # signal's own support is visible.
    pe_a, pe_b = res_a["phase_err"], res_b["phase_err"]
    pdiff = pe_a - pe_b
    pdiff_lim = max(float(np.percentile(np.abs(pdiff), 99)), 0.05)

    fig, axes = plt.subplots(2, 4, figsize=(20, 7), constrained_layout=True)

    def panel(ax, data, title, cmap, vmin, vmax, cbar_label):
        im = ax.imshow(
            data, origin="lower", aspect="auto", cmap=cmap,
            extent=extent, vmin=vmin, vmax=vmax,
        )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("kHz")
        ax.set_xlabel("time (s)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)

    # Row 0: magnitude (dB)
    panel(axes[0][0], in_disp, "Input (dB)", args.cmap, -80.0, 0.0, "dB")
    panel(axes[0][1], out_a, f"A: {name_a} out (dB)", args.cmap, -80.0, 0.0, "dB")
    panel(axes[0][2], out_b, f"B: {name_b} out (dB)", args.cmap, -80.0, 0.0, "dB")
    panel(axes[0][3], mag_diff, "Mag A - B", "RdBu_r", -mdiff_lim, mdiff_lim, "Δ dB")

    # Row 1: phase error (rad), full field (unmasked) to show propagation.
    axes[1][0].axis("off")
    axes[1][0].text(0.5, 0.5, "Phase error\n|Δφ| vs input\n(full field —\nno masking)",
                    ha="center", va="center", fontsize=11, transform=axes[1][0].transAxes)
    pa = res_a["full_phase_err"]
    pb = res_b["full_phase_err"]
    panel(axes[1][1], pe_a, f"A phase err (full {pa:.3f} rad)", "viridis", 0.0, np.pi, "rad")
    panel(axes[1][2], pe_b, f"B phase err (full {pb:.3f} rad)", "viridis", 0.0, np.pi, "rad")
    panel(axes[1][3], pdiff, "Phase err A - B", "RdBu_r", -pdiff_lim, pdiff_lim, "Δ rad")

    fig.suptitle(f"Model comparison — {sig_name}\nA = {name_a}   |   B = {name_b}", fontsize=12)

    sig_dir = os.path.join(args.output_dir, safe_name)
    os.makedirs(sig_dir, exist_ok=True)
    out_path = os.path.join(sig_dir, f"compare_{name_a}_vs_{name_b}_{safe_name}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    ALL_SIGNALS = {
        "vowel": ("Vowel", generate_vowel),
        "fricative": ("Fricative", generate_fricative),
        "plosive": ("Plosive", generate_plosive),
        "chirp": ("Chirp Sweep", generate_chirp),
        "step": ("Step Drop", generate_step),
        "am_noise": ("AM Noise", generate_am_noise),
        "dynamic_f0": ("Dynamic F0", generate_dynamic_f0),
        "two_tone": ("Two-Tone Mix", generate_two_tone),
        "real_clean": ("Real Clean Speech", None),
        "real_noisy": ("Real Noisy Speech", None),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, nargs="+",
                        help="One or more checkpoint .pth files/dirs. Pass exactly two "
                             "(or a parent dir with two subdirs) to also get side-by-side "
                             "comparison figures.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default="eval_out/phonetic_plots")

    parser.add_argument("--signals", nargs="+", default=["all"],
                        choices=list(ALL_SIGNALS.keys()) + ["all"],
                        help="Which signals to generate and analyze")
    parser.add_argument("--plot_mode", choices=["activations", "output", "both"], default="both",
                        help="What to plot: layer activations, final model output, or both")

    parser.add_argument("--index", type=int, default=0, help="Test-set pair index for real signals")
    parser.add_argument("--noisy", default=None, help="Explicit noisy wav")
    parser.add_argument("--clean", default=None, help="Explicit clean wav")
    parser.add_argument("--test_clean_json", default=None)
    parser.add_argument("--test_noisy_json", default=None)
    parser.add_argument("--max_seconds", type=float, default=2.0, help="Trim real utterance to match synthetics")

    parser.add_argument("--reduce", default="l2")
    parser.add_argument("--per_channel_norm", default="std")
    parser.add_argument("--clip", type=float, nargs=2, default=(2.0, 98.0))
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the noise-based synthetic signals so runs (and the "
                             "two models being compared) see identical inputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Deterministic noise for fricative/plosive/step/am_noise. Signals are built
    # once and reused across checkpoints, so both compared models see identical
    # inputs; seeding also makes results reproducible across invocations.
    np.random.seed(args.seed)

    checkpoints = [c for arg in args.checkpoint for c in expand_checkpoints(arg)]
    print(f"rendering {len(checkpoints)} checkpoint(s)...")

    sigs_to_run = list(ALL_SIGNALS.keys()) if "all" in args.signals else args.signals

    _, config_file0 = resolve_checkpoint_and_config(checkpoints[0], args.config)
    cfg0 = load_config(config_file0)
    sr = cfg0["stft_cfg"].get("sampling_rate", 16000)

    signals_np = {}
    for sig_key in sigs_to_run:
        display_name, generator_fn = ALL_SIGNALS[sig_key]

        if sig_key in ["real_clean", "real_noisy"]:
            try:
                noisy_path, clean_path = resolve_pair(args, cfg0)
                target_path = clean_path if sig_key == "real_clean" else noisy_path

                if target_path is None:
                    print(f"Warning: Could not resolve path for {sig_key} (index {args.index}). Skipping.")
                    continue

                real_np, _ = librosa.load(target_path, sr=sr)
                max_samples = int(args.max_seconds * sr)
                if len(real_np) > max_samples:
                    real_np = real_np[:max_samples]

                signals_np[display_name] = normalize_energy(real_np)
            except Exception as e:
                print(f"Warning: Failed to load {sig_key} utterance: {e}")
                continue
        else:
            signals_np[display_name] = normalize_energy(generator_fn(sr))

    failures = []
    summary_stats = {}
    all_results = {}  # ckpt_name -> {sig_name -> result dict} for the comparison
    ckpt_order = []

    for ckpt_arg in checkpoints:
        ckpt_name = run_name_for(ckpt_arg)
        summary_stats[ckpt_name] = {}
        all_results[ckpt_name] = {}

        try:
            ckpt_file, config_file = resolve_checkpoint_and_config(ckpt_arg, args.config)
            cfg = load_config(config_file)
            model = load_generator(ckpt_file, cfg, device)

            for sig_name, sig_np in signals_np.items():
                res = render_signal(model, sig_np, sig_name, cfg, device, args, ckpt_name)
                summary_stats[ckpt_name][sig_name] = {
                    "mean_err": res["mean_err"],
                    "std_err": res["std_err"],
                    "w_phase_err": res["w_phase_err"],
                }
                all_results[ckpt_name][sig_name] = res
                print(f"  [{ckpt_name}] {sig_name}: done")

            ckpt_order.append(ckpt_name)
            del model
        except Exception as e:
            failures.append((ckpt_name, f"{type(e).__name__}: {e}"))
            print(f"FAILED {ckpt_name}: {type(e).__name__}: {e}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if summary_stats:
        plot_summary_bars(
            summary_stats, args.output_dir, "mean_err", "std_err",
            "Mean Error (Out - In) [dB]", "Average Magnitude Error per Signal and Model",
            "error_summary.png",
        )
        plot_summary_bars(
            summary_stats, args.output_dir, "w_phase_err", None,
            "Mag-weighted mean |Δφ| [rad]", "Average Phase Error per Signal and Model",
            "phase_error_summary.png",
        )

    # Side-by-side comparison for the first two successful checkpoints.
    if len(ckpt_order) >= 2:
        name_a, name_b = ckpt_order[0], ckpt_order[1]
        if len(ckpt_order) > 2:
            print(f"\ncomparing first two checkpoints ({name_a} vs {name_b}); "
                  f"{len(ckpt_order) - 2} more rendered individually only.")
        for sig_name in signals_np:
            res_a = all_results[name_a].get(sig_name)
            res_b = all_results[name_b].get(sig_name)
            if res_a is None or res_b is None:
                continue
            out_path = render_comparison(res_a, res_b, name_a, name_b, sig_name, args)
            print(f"  compare {sig_name}: {out_path}")

    if failures:
        print(f"\n{len(failures)}/{len(checkpoints)} checkpoint(s) failed:")
        for name, msg in failures:
            print(f"  {name}: {msg}")

if __name__ == "__main__":
    main()