"""Measure model layer activations for different phonetic signal types.

Generates synthetic signals representing vowels (harmonics), fricatives (noise),
and plosives (bursts), feeds them through a given model, and plots the layer-wise
activation energies to compare how the network responds to each signal class.

Example
-------
    python evaluation/measure_phonetic_response.py --checkpoint ckpts/my_model.pth
"""

import argparse
import os
import sys

# Allow running from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.stfts import mag_phase_stft
from utils.util import load_config
from plot_tf_activations import (
    TFActivationCapture,
    load_generator,
    resolve_checkpoint_and_config,
    run_name_for,
)

def apply_time_mask(signal, sr, duration, start_frac=1 / 3, end_frac=2 / 3, fade_len=0.02):
    """Zeros out the signal outside the window, applying a smooth fade to prevent clicks."""
    mask = np.zeros_like(signal)
    start_idx = int(sr * duration * start_frac)
    end_idx = int(sr * duration * end_frac)
    fade_samples = int(sr * fade_len)

    mask[start_idx:end_idx] = 1.0

    # Apply a Hanning taper to the edges
    if fade_samples > 0:
        window = np.hanning(fade_samples * 2)
        mask[start_idx : start_idx + fade_samples] = window[:fade_samples]
        mask[end_idx - fade_samples : end_idx] = window[fade_samples:]

    return signal * mask


def generate_vowel(sr, duration=2.0, f0=200):
    """Harmonic signal restricted to the middle third of the timeline."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    for i in range(1, 10):
        signal += (1.0 / i) * np.sin(2 * np.pi * (f0 * i) * t)

    return apply_time_mask(signal, sr, duration, 1 / 3, 2 / 3)


def generate_fricative(sr, duration=2.0):
    """White noise restricted to the middle third of the timeline."""
    noise = np.random.randn(int(sr * duration))
    return apply_time_mask(noise, sr, duration, 1 / 3, 2 / 3)


def generate_plosive(sr, duration=2.0):
    """Plosive burst centered exactly at the midpoint of the timeline."""
    signal = np.zeros(int(sr * duration))

    # Center the 20ms burst exactly at duration / 2
    burst_start = int(sr * (duration / 2.0)) - int(sr * 0.01)
    burst_end = burst_start + int(sr * 0.02)
    signal[burst_start:burst_end] = np.random.randn(burst_end - burst_start)

    return signal


def normalize_energy(sig):
    """RMS normalization so all signals have equivalent starting energy."""
    rms = np.sqrt(np.mean(sig**2))
    return sig / (rms + 1e-8)


def get_layer_responses(model, waveform, cfg, device):
    """Passes waveform through the model and computes mean activation magnitude per layer."""
    sr = cfg["stft_cfg"]["sampling_rate"]
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress = cfg["model_cfg"]["compress_factor"]

    wave_t = torch.from_numpy(waveform).float().to(device)
    norm = torch.sqrt(len(wave_t) / torch.sum(wave_t**2.0 + 1e-8))
    wave_in = (wave_t * norm).unsqueeze(0)

    mag, pha, _ = mag_phase_stft(wave_in, n_fft, hop_size, win_size, compress)

    with TFActivationCapture(model) as capt, torch.no_grad():
        model(mag, pha)

    layer_energies = []
    labels = []

    # We use the 'after_freq' representations (the output of each block)
    for i, (_, after_freq) in enumerate(capt.records):
        # Mean absolute activation as the response metric
        mean_activation = after_freq.abs().mean().item()
        layer_energies.append(mean_activation)
        labels.append(f"blk{i}")

    return layer_energies, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--config", default=None, help="Override config.yaml")
    parser.add_argument("--output_dir", default="eval_out/phonetic_response")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model
    ckpt_file, config_file = resolve_checkpoint_and_config(args.checkpoint, args.config)
    cfg = load_config(config_file)
    model = load_generator(ckpt_file, cfg, device)
    model.eval()
    name = run_name_for(args.checkpoint)

    sr = cfg["stft_cfg"].get("sampling_rate", 16000)

    # Generate test signals
    signals = {
        "Vowel (Harmonic)": normalize_energy(generate_vowel(sr)),
        "Fricative (Noise)": normalize_energy(generate_fricative(sr)),
        "Plosive (Burst)": normalize_energy(generate_plosive(sr)),
    }

    # Run inferences
    responses = {}
    labels = None
    for sig_name, sig in signals.items():
        energies, layer_labels = get_layer_responses(model, sig, cfg, device)
        responses[sig_name] = energies
        if labels is None:
            labels = layer_labels

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))

    markers = ["o", "s", "^"]
    for i, (sig_name, energies) in enumerate(responses.items()):
        ax.plot(x, energies, marker=markers[i], label=sig_name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Mean Absolute Activation")
    ax.set_title(f"Layer-wise Response by Phonetic Class\n({name})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, f"{name}_phonetic_responses.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved response plot to {out_path}")


if __name__ == "__main__":
    main()
