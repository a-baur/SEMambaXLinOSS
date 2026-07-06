"""Evaluate a SEMamba generator checkpoint on a test set.

Supports three checkpoint layouts:
  * single-file ckpts (e.g. ``ckpts/SEMamba_advanced.pth``)
  * checkpoint directories holding a single named ``*.pth`` + ``config.yaml``
    (e.g. ``ckpts/mamba/``) or exp-run directories produced by ``train.py``
    (e.g. ``exp/LinOSS``), where the latest ``g_????????.pth`` is picked
    automatically and the co-located ``config.yaml`` is used by default.
  * a *parent* directory of such checkpoint directories (e.g. ``ckpts/``):
    every checkpoint under it is evaluated and compared. Each run's outputs
    land in ``<output_dir>/<name>/`` and a combined ``comparison.json`` plus a
    printed table rank the checkpoints side by side.

Outputs:
  * ``<output_dir>/metrics.json`` -- per-utterance PESQ / MR-STFT / UTMOS
    plus a summary block with the mean and variance of each.
  * ``<output_dir>/samples/<utt>_{noisy,enhanced,clean}.wav`` -- a fixed,
    deterministic subset of enhanced clips with their noisy / clean refs.
  * ``<output_dir>/comparison.json`` (compare mode only) -- the per-checkpoint
    summary blocks gathered into one file.
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
import torch.nn.functional as F

from dataloaders.dataloader_vctk import (
    _common_root,
    extract_identifier,
    get_clean_path_for_noisy,
)
from utils.metrics import Evaluator
from utils.util import load_config
from models.generator import SEMamba
from models.loss import phase_losses
from models.stfts import mag_phase_istft, mag_phase_stft


def resolve_checkpoint_and_config(ckpt_arg: str, config_arg: str | None):
    """Return (ckpt_file, config_file).

    ``ckpt_arg`` may be a ``.pth`` file or an exp directory; in the latter
    case the most recent ``g_????????.pth`` is selected (mirrors
    ``utils.util.scan_checkpoint``).
    """
    if os.path.isdir(ckpt_arg):
        candidates = sorted(glob.glob(os.path.join(ckpt_arg, "g_????????.pth")))
        if not candidates:
            # Named single-file ckpts (e.g. ckpts/mamba/mamba.pth).
            candidates = sorted(glob.glob(os.path.join(ckpt_arg, "*.pth")))
        if not candidates:
            raise FileNotFoundError(f"No *.pth checkpoints under {ckpt_arg}")
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


def is_checkpoint_dir(path: str) -> bool:
    """True if ``path`` directly holds a ``.pth`` (a single checkpoint)."""
    return os.path.isdir(path) and bool(glob.glob(os.path.join(path, "*.pth")))


def find_checkpoint_dirs(root: str) -> list[str]:
    """Return sorted immediate subdirectories of ``root`` that are checkpoints."""
    subdirs = sorted(
        p for p in glob.glob(os.path.join(root, "*")) if is_checkpoint_dir(p)
    )
    return subdirs


METRIC_NAMES = (
    "pesq",
    "mrstft",
    "utmos",
    "distillmos",
    "dnsmospro",
    "nisqa",
    "sisdr",
    "lsd",
    "estoi",
    "magnitude",
    "phase",
    "phase_ip",
    "phase_gd",
    "phase_iaf",
    "complex",
)

METRIC_LABELS = {
    "pesq": "PESQ",
    "mrstft": "MR-STFT",
    "utmos": "UTMOS",
    "distillmos": "DistillMOS",
    "dnsmospro": "DNSMOSPro",
    "nisqa": "NISQA",
    "sisdr": "SI-SDR",
    "lsd": "LSD",
    "estoi": "ESTOI",
    "magnitude": "Magnitude",
    "phase": "Phase",
    "phase_ip": "Phase-IP",
    "phase_gd": "Phase-GD",
    "phase_iaf": "Phase-IAF",
    "complex": "Complex",
}


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

    # Pair clean/noisy with the same dataset-relative, SNR-suffix-stripped key the
    # training dataloader uses, so VCTK-DEMAND (identical basenames) and EARS-WHAM
    # (per-speaker dirs + "_<snr>dB" suffix) both match correctly.
    clean_root = _common_root(clean_paths)
    noisy_root = _common_root(noisy_paths)
    clean_by_id = {extract_identifier(p, clean_root): p for p in clean_paths}
    # Sort by basename so the "fixed samples" subset is reproducible across runs.
    noisy_sorted = sorted(noisy_paths, key=os.path.basename)

    pairs = []
    missing = 0
    for noisy in noisy_sorted:
        clean = get_clean_path_for_noisy(noisy, noisy_root, clean_by_id)
        if clean is None:
            missing += 1
            continue
        pairs.append((noisy, clean))
    if missing:
        warnings.warn(f"{missing} noisy files had no matching clean reference; skipped.")
    return pairs


def evaluate_checkpoint(
    ckpt_file: str,
    config_file: str,
    output_dir: str,
    args,
    device: torch.device,
    evaluator_cache: dict,
) -> dict:
    """Evaluate one checkpoint, write ``<output_dir>/metrics.json`` + samples.

    Returns the summary block (means/variances/counts). ``evaluator_cache``
    memoizes one ``Evaluator`` per sampling rate so compare mode does not
    rebuild the (expensive) metric models per checkpoint.
    """
    print(f"Checkpoint : {ckpt_file}")
    print(f"Config     : {config_file}")

    cfg = load_config(config_file)
    sr = cfg["stft_cfg"]["sampling_rate"]

    model = load_generator(ckpt_file, cfg, device)
    if sr not in evaluator_cache:
        evaluator_cache[sr] = Evaluator(sr=sr).to(device)
    evaluator = evaluator_cache[sr]

    pairs = build_pair_list(cfg, args.test_clean_json, args.test_noisy_json)
    print(f"Evaluating {len(pairs)} utterances at {sr} Hz")

    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sample_indices = set(range(min(args.num_samples, len(pairs))))

    metric_names = METRIC_NAMES
    sums = {m: 0.0 for m in metric_names}
    sumsq = {m: 0.0 for m in metric_names}
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

        # Spectral-domain training losses (magnitude / phase / complex), computed in
        # the same compressed STFT domain as train.py's validation. Both waveforms are
        # in the un-normalized domain here, so re-analysing them is self-consistent.
        n_fft = cfg["stft_cfg"]["n_fft"]
        hop_size = cfg["stft_cfg"]["hop_size"]
        win_size = cfg["stft_cfg"]["win_size"]
        compress = cfg["model_cfg"]["compress_factor"]
        clean_mag, clean_pha, clean_com = mag_phase_stft(
            clean_b, n_fft, hop_size, win_size, compress
        )
        enh_mag, enh_pha, enh_com = mag_phase_stft(
            enhanced_b, n_fft, hop_size, win_size, compress
        )
        ip, gd, iaf = phase_losses(clean_pha, enh_pha, cfg)
        ip, gd, iaf = ip.item(), gd.item(), iaf.item()

        row = {
            "noisy": noisy_path,
            "clean": clean_path,
            "pesq": pesq,
            "mrstft": metrics.mrstft,
            "utmos": metrics.utmos,
            "distillmos": metrics.distillmos,
            "dnsmospro": metrics.dnsmospro,
            "nisqa": metrics.nisqa,
            "sisdr": metrics.sisdr,
            "lsd": metrics.lsd,
            "estoi": metrics.estoi,
            "magnitude": F.mse_loss(clean_mag, enh_mag).item(),
            "phase": ip + gd + iaf,
            "phase_ip": ip,
            "phase_gd": gd,
            "phase_iaf": iaf,
            "complex": F.mse_loss(clean_com, enh_com).item(),
        }
        per_utt.append(row)
        for m in metric_names:
            v = row[m]
            if v is not None and not math.isnan(v):
                sums[m] += v
                sumsq[m] += v * v
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
            stats = "  ".join(f"{m.upper()}={running[m]:.4f}" for m in metric_names)
            print(f"  [{i + 1}/{len(pairs)}] {stats}")

    means = {m: sums[m] / max(counts[m], 1) for m in metric_names}
    # Sample variance (ddof=1); 0.0 when fewer than two valid values.
    variances = {
        m: (sumsq[m] - sums[m] * sums[m] / counts[m]) / (counts[m] - 1)
        if counts[m] > 1
        else 0.0
        for m in metric_names
    }
    summary = {
        "checkpoint": ckpt_file,
        "config": config_file,
        "sampling_rate": sr,
        "num_utterances": len(pairs),
        "valid_counts": counts,
        **{f"mean_{m}": means[m] for m in metric_names},
        **{f"var_{m}": variances[m] for m in metric_names},
        "samples_dir": samples_dir,
    }
    out_json = os.path.join(output_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_utterance": per_utt}, f, indent=2)

    print()
    for m in metric_names:
        print(
            f"Mean {METRIC_LABELS[m]:<10}: {means[m]:.4f} (var {variances[m]:.4g}) "
            f"over {counts[m]}/{len(pairs)} utts"
        )
    print(f"Samples      : {samples_dir} ({len(sample_indices)} clips)")
    print(f"Per-utt log  : {out_json}")
    return summary


def print_comparison_table(summaries: dict[str, dict]):
    """Print a name x mean-metric table, one row per checkpoint."""
    if not summaries:
        return
    # Column widths driven by the longest checkpoint name.
    name_w = max(len("checkpoint"), max(len(n) for n in summaries))
    header = "  ".join(
        [f"{'checkpoint':<{name_w}}"] + [f"{METRIC_LABELS[m]:>10}" for m in METRIC_NAMES]
    )
    print("\n" + "=" * len(header))
    print("COMPARISON (means)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, summ in summaries.items():
        cells = [f"{name:<{name_w}}"]
        for m in METRIC_NAMES:
            cells.append(f"{summ[f'mean_{m}']:>10.4f}")
        print("  ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="ckpts/SEMamba_advanced.pth",
        help=(
            "A .pth file, a single checkpoint directory (holds *.pth + config.yaml "
            "or g_*.pth), or a parent directory of such dirs (evaluates + compares all)."
        ),
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
    evaluator_cache: dict = {}

    # Compare mode: a parent directory of checkpoint dirs (ckpts/mamba, ...).
    if os.path.isdir(args.checkpoint) and not is_checkpoint_dir(args.checkpoint):
        ckpt_dirs = find_checkpoint_dirs(args.checkpoint)
        if not ckpt_dirs:
            raise FileNotFoundError(
                f"No checkpoint directories (with *.pth) found under {args.checkpoint}"
            )
        print(f"Comparing {len(ckpt_dirs)} checkpoints under {args.checkpoint}:")
        for d in ckpt_dirs:
            print(f"  - {os.path.basename(d)}")

        summaries: dict[str, dict] = {}
        for d in ckpt_dirs:
            name = os.path.basename(d.rstrip("/"))
            print(f"\n{'#' * 60}\n# {name}\n{'#' * 60}")
            ckpt_file, config_file = resolve_checkpoint_and_config(d, args.config)
            out_dir = os.path.join(args.output_dir, name)
            summaries[name] = evaluate_checkpoint(
                ckpt_file, config_file, out_dir, args, device, evaluator_cache
            )

        comparison_json = os.path.join(args.output_dir, "comparison.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(comparison_json, "w") as f:
            json.dump(summaries, f, indent=2)
        print_comparison_table(summaries)
        print(f"\nComparison   : {comparison_json}")
        return

    # Single-checkpoint mode.
    ckpt_file, config_file = resolve_checkpoint_and_config(args.checkpoint, args.config)
    evaluate_checkpoint(
        ckpt_file, config_file, args.output_dir, args, device, evaluator_cache
    )


if __name__ == "__main__":
    main()
