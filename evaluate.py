"""Evaluate a SEMamba generator checkpoint on a test set.

Supports two checkpoint layouts:
  * single-file ckpts (e.g. ``ckpts/SEMamba_advanced.pth``)
  * exp-run directories produced by ``train.py`` (e.g. ``exp/LinOSS``),
    where the latest ``g_????????.pth`` is picked automatically and the
    co-located ``config.yaml`` is used by default.

Outputs:
  * ``<output_dir>/metrics.json`` -- per-utterance PESQ / MR-STFT / UTMOS
    plus a summary block with the mean of each.
  * ``<output_dir>/samples/<utt>_{noisy,enhanced,clean}.wav`` -- a fixed,
    deterministic subset of enhanced clips with their noisy / clean refs.
"""

import argparse
import glob
import json
import math
import os
import warnings

import librosa
import soundfile as sf
import torch

from models.generator import SEMamba
from models.stfts import mag_phase_istft, mag_phase_stft
from utils.metrics import Evaluator
from utils.util import load_config


def resolve_checkpoint_and_config(ckpt_arg: str, config_arg: str | None):
    """Return (ckpt_file, config_file).

    ``ckpt_arg`` may be a ``.pth`` file or an exp directory; in the latter
    case the most recent ``g_????????.pth`` is selected (mirrors
    ``utils.util.scan_checkpoint``).
    """
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
    # Both ckpts/*.pth and exp/.../g_*.pth wrap weights under "generator".
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def enhance(model: SEMamba, noisy_wav: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Run the generator on a single 1D noisy waveform, return 1D enhanced."""
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress = cfg["model_cfg"]["compress_factor"]

    norm = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0))
    noisy = (noisy_wav * norm).unsqueeze(0)

    noisy_mag, noisy_pha, _ = mag_phase_stft(noisy, n_fft, hop_size, win_size, compress)
    mag_g, pha_g, _ = model(noisy_mag, noisy_pha)
    audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress)
    return (audio_g / norm).squeeze(0)


def build_pair_list(cfg: dict, test_clean_json: str | None, test_noisy_json: str | None):
    clean_json = test_clean_json or cfg["data_cfg"]["test_clean_json"]
    noisy_json = test_noisy_json or cfg["data_cfg"]["test_noisy_json"]
    with open(clean_json) as f:
        clean_paths = json.load(f)
    with open(noisy_json) as f:
        noisy_paths = json.load(f)

    clean_by_name = {os.path.basename(p): p for p in clean_paths}
    # Sort by basename so the "fixed samples" subset is reproducible across runs.
    noisy_sorted = sorted(noisy_paths, key=os.path.basename)

    pairs = []
    missing = 0
    for noisy in noisy_sorted:
        clean = clean_by_name.get(os.path.basename(noisy))
        if clean is None:
            missing += 1
            continue
        pairs.append((noisy, clean))
    if missing:
        warnings.warn(f"{missing} noisy files had no matching clean reference; skipped.")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="ckpts/SEMamba_advanced.pth",
        help="Path to a .pth file or an exp directory containing g_*.pth.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config. Defaults to config.yaml next to the checkpoint.",
    )
    parser.add_argument("--test_clean_json", default=None)
    parser.add_argument("--test_noisy_json", default=None)
    parser.add_argument("--output_dir", default="eval_out")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of fixed enhanced clips to save (alongside noisy/clean refs).",
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
    evaluator = Evaluator(sr=sr).to(device)

    pairs = build_pair_list(cfg, args.test_clean_json, args.test_noisy_json)
    print(f"Evaluating {len(pairs)} utterances at {sr} Hz")

    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sample_indices = set(range(min(args.num_samples, len(pairs))))

    metric_names = ("pesq", "mrstft", "utmos")
    sums = {m: 0.0 for m in metric_names}
    counts = {m: 0 for m in metric_names}
    per_utt = []

    for i, (noisy_path, clean_path) in enumerate(pairs):
        noisy_np, _ = librosa.load(noisy_path, sr=sr)
        clean_np, _ = librosa.load(clean_path, sr=sr)
        noisy_t = torch.from_numpy(noisy_np).float().to(device)
        clean_t = torch.from_numpy(clean_np).float().to(device)

        enhanced = enhance(model, noisy_t, cfg)
        n = min(enhanced.size(-1), clean_t.size(-1))
        # Evaluator.compute expects batched (B, T) tensors.
        enhanced_b = enhanced[..., :n].unsqueeze(0)
        clean_b = clean_t[..., :n].unsqueeze(0)
        metrics = evaluator.compute(clean_b, enhanced_b)
        # PESQ sentinel from utils/metrics.py on degenerate utterances.
        pesq = float("nan") if metrics.pesq == -1.0 else metrics.pesq

        row = {
            "noisy": noisy_path,
            "clean": clean_path,
            "pesq": pesq,
            "mrstft": metrics.mrstft,
            "utmos": metrics.utmos,
        }
        per_utt.append(row)
        for m in metric_names:
            v = row[m]
            if v is not None and not math.isnan(v):
                sums[m] += v
                counts[m] += 1

        if i in sample_indices:
            base = os.path.splitext(os.path.basename(noisy_path))[0]
            sf.write(
                os.path.join(samples_dir, f"{base}_enhanced.wav"),
                enhanced.cpu().numpy(),
                sr,
                "PCM_16",
            )
            sf.write(
                os.path.join(samples_dir, f"{base}_noisy.wav"),
                noisy_np,
                sr,
                "PCM_16",
            )
            sf.write(
                os.path.join(samples_dir, f"{base}_clean.wav"),
                clean_np,
                sr,
                "PCM_16",
            )

        if (i + 1) % 50 == 0:
            running = {m: sums[m] / max(counts[m], 1) for m in metric_names}
            print(
                f"  [{i + 1}/{len(pairs)}] "
                f"PESQ={running['pesq']:.4f}  "
                f"MR-STFT={running['mrstft']:.4f}  "
                f"UTMOS={running['utmos']:.4f}"
            )

    means = {m: sums[m] / max(counts[m], 1) for m in metric_names}
    summary = {
        "checkpoint": ckpt_file,
        "config": config_file,
        "sampling_rate": sr,
        "num_utterances": len(pairs),
        "valid_counts": counts,
        "mean_pesq": means["pesq"],
        "mean_mrstft": means["mrstft"],
        "mean_utmos": means["utmos"],
        "samples_dir": samples_dir,
    }
    out_json = os.path.join(args.output_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_utterance": per_utt}, f, indent=2)

    print()
    print(f"Mean PESQ    : {means['pesq']:.4f} over {counts['pesq']}/{len(pairs)} utts")
    print(f"Mean MR-STFT : {means['mrstft']:.4f} over {counts['mrstft']}/{len(pairs)} utts")
    print(f"Mean UTMOS   : {means['utmos']:.4f} over {counts['utmos']}/{len(pairs)} utts")
    print(f"Samples      : {samples_dir} ({len(sample_indices)} clips)")
    print(f"Per-utt log  : {out_json}")


if __name__ == "__main__":
    main()
