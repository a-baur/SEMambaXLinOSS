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
    plus a summary block with the mean and variance of each. When an SNR
    manifest is supplied (``--snr_json``), the summary also carries a
    ``per_snr`` block breaking every mean/variance down by input-SNR category.
  * ``<output_dir>/samples/<cat>/<utt>_{noisy,enhanced,clean}.wav`` -- a fixed,
    deterministic subset of enhanced clips with their noisy / clean refs. With
    an SNR manifest the subset is drawn evenly from each SNR category (the
    first ``--num_samples`` utterances of each); otherwise the first
    ``--num_samples`` utterances overall.
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


def _fmt_edge(x: float) -> str:
    """Render an SNR bin edge without a trailing ``.0`` (e.g. ``5`` not ``5.0``)."""
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def snr_category(snr: float | None, edges: list[float]) -> str:
    """Label the SNR bin ``snr`` falls in, given ascending ``edges``.

    Edges ``[0, 5, 10]`` produce ``"<0dB"``, ``"0-5dB"``, ``"5-10dB"``,
    ``">=10dB"``. A missing SNR maps to ``"unknown"``.
    """
    if snr is None or (isinstance(snr, float) and math.isnan(snr)):
        return "unknown"
    if snr < edges[0]:
        return f"<{_fmt_edge(edges[0])}dB"
    for lo, hi in zip(edges, edges[1:]):
        if lo <= snr < hi:
            return f"{_fmt_edge(lo)}-{_fmt_edge(hi)}dB"
    return f">={_fmt_edge(edges[-1])}dB"


def snr_category_order(edges: list[float]) -> list[str]:
    """The category labels in ascending-SNR order (``unknown`` last)."""
    labels = [f"<{_fmt_edge(edges[0])}dB"]
    labels += [f"{_fmt_edge(lo)}-{_fmt_edge(hi)}dB" for lo, hi in zip(edges, edges[1:])]
    labels += [f">={_fmt_edge(edges[-1])}dB", "unknown"]
    return labels


def load_snr_map(snr_json: str | None) -> dict[str, float] | None:
    """Load ``snr_json`` into an ``identifier -> snr`` dict, or ``None``.

    Keys are reduced to the same SNR-suffix-stripped, dataset-relative
    identifier the dataloader uses (via ``extract_identifier``), so a noisy
    wav path can be matched to its SNR the same way it is matched to its clean
    reference. Returns ``None`` if no manifest is given or it cannot be read.
    """
    if not snr_json:
        return None
    if not os.path.isfile(snr_json):
        warnings.warn(f"SNR manifest not found at {snr_json}; skipping stratification.")
        return None
    with open(snr_json) as f:
        raw = json.load(f)
    root = _common_root(list(raw.keys()))
    return {extract_identifier(k, root): float(v) for k, v in raw.items()}


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


def build_pair_list(
    cfg: dict,
    test_clean_json: str | None,
    test_noisy_json: str | None,
    snr_by_id: dict[str, float] | None = None,
):
    """Return a list of ``(noisy_path, clean_path, snr)`` triples.

    ``snr`` is the input SNR (dB) looked up from ``snr_by_id`` for the noisy
    file, or ``None`` when no manifest is given or the file is absent from it.
    """
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
    snr_missing = 0
    for noisy in noisy_sorted:
        clean = get_clean_path_for_noisy(noisy, noisy_root, clean_by_id)
        if clean is None:
            missing += 1
            continue
        snr = None
        if snr_by_id is not None:
            snr = snr_by_id.get(extract_identifier(noisy, noisy_root))
            if snr is None:
                snr_missing += 1
        pairs.append((noisy, clean, snr))
    if missing:
        warnings.warn(f"{missing} noisy files had no matching clean reference; skipped.")
    if snr_by_id is not None and snr_missing:
        warnings.warn(f"{snr_missing} noisy files had no SNR in the manifest; binned as 'unknown'.")
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

    snr_by_id = getattr(args, "snr_by_id", None)
    snr_edges = getattr(args, "snr_edges", None)
    stratify = snr_by_id is not None and snr_edges is not None

    pairs = build_pair_list(cfg, args.test_clean_json, args.test_noisy_json, snr_by_id)
    print(f"Evaluating {len(pairs)} utterances at {sr} Hz")

    # Category per pair (None -> a single implicit "all" bucket when not stratifying).
    if stratify:
        pair_cats = [snr_category(snr, snr_edges) for (_, _, snr) in pairs]
    else:
        pair_cats = [None] * len(pairs)

    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    # Sample the first ``num_samples`` utterances of each category (or of the whole
    # set when not stratifying), so every SNR band is represented in the saved clips.
    sample_indices: set[int] = set()
    seen_per_cat: dict[str | None, int] = {}
    for idx, cat in enumerate(pair_cats):
        taken = seen_per_cat.get(cat, 0)
        if taken < args.num_samples:
            sample_indices.add(idx)
            seen_per_cat[cat] = taken + 1

    metric_names = METRIC_NAMES
    sums = {m: 0.0 for m in metric_names}
    sumsq = {m: 0.0 for m in metric_names}
    counts = {m: 0 for m in metric_names}
    # Per-category accumulators, created lazily as categories appear.
    strat_sums: dict[str, dict[str, float]] = {}
    strat_sumsq: dict[str, dict[str, float]] = {}
    strat_counts: dict[str, dict[str, int]] = {}
    strat_utts: dict[str, int] = {}
    per_utt = []

    for i, (noisy_path, clean_path, snr) in enumerate(pairs):
        cat = pair_cats[i]
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
            "snr": snr,
            "snr_category": cat,
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
        if stratify:
            strat_utts[cat] = strat_utts.get(cat, 0) + 1
            csum = strat_sums.setdefault(cat, {m: 0.0 for m in metric_names})
            csumsq = strat_sumsq.setdefault(cat, {m: 0.0 for m in metric_names})
            ccount = strat_counts.setdefault(cat, {m: 0 for m in metric_names})
        for m in metric_names:
            v = row[m]
            if v is not None and not math.isnan(v):
                sums[m] += v
                sumsq[m] += v * v
                counts[m] += 1
                if stratify:
                    csum[m] += v
                    csumsq[m] += v * v
                    ccount[m] += 1

        if i in sample_indices:
            # Group saved clips by SNR category (flat "samples/" when not stratifying).
            dst = os.path.join(samples_dir, cat) if stratify else samples_dir
            os.makedirs(dst, exist_ok=True)
            base = os.path.splitext(os.path.basename(noisy_path))[0]
            sf.write(
                os.path.join(dst, f"{base}_enhanced.wav"),
                enhanced.cpu().numpy(),
                sr,
                "PCM_16",
            )
            sf.write(
                os.path.join(dst, f"{base}_noisy.wav"),
                noisy_np,
                sr,
                "PCM_16",
            )
            sf.write(
                os.path.join(dst, f"{base}_clean.wav"),
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

    if stratify:
        # Per-category means/variances, in ascending-SNR order (empty bins dropped).
        per_snr = {}
        for cat in snr_category_order(snr_edges):
            if cat not in strat_utts:
                continue
            cc, cs, css = strat_counts[cat], strat_sums[cat], strat_sumsq[cat]
            cmeans = {m: cs[m] / max(cc[m], 1) for m in metric_names}
            cvars = {
                m: (css[m] - cs[m] * cs[m] / cc[m]) / (cc[m] - 1) if cc[m] > 1 else 0.0
                for m in metric_names
            }
            per_snr[cat] = {
                "num_utterances": strat_utts[cat],
                "valid_counts": cc,
                **{f"mean_{m}": cmeans[m] for m in metric_names},
                **{f"var_{m}": cvars[m] for m in metric_names},
            }
        summary["per_snr"] = per_snr

    out_json = os.path.join(output_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_utterance": per_utt}, f, indent=2)

    print()
    for m in metric_names:
        print(
            f"Mean {METRIC_LABELS[m]:<10}: {means[m]:.4f} (var {variances[m]:.4g}) "
            f"over {counts[m]}/{len(pairs)} utts"
        )
    if stratify:
        print_stratified_table(summary["per_snr"])
    print(f"Samples      : {samples_dir} ({len(sample_indices)} clips)")
    print(f"Per-utt log  : {out_json}")
    return summary


def load_existing_summary(output_dir: str) -> dict | None:
    """Return the stored summary block if ``<output_dir>/metrics.json`` exists.

    ``metrics.json`` is written only after a checkpoint finishes evaluating, so
    its presence marks a completed run whose result can be reused as-is.
    """
    out_json = os.path.join(output_dir, "metrics.json")
    if not os.path.isfile(out_json):
        return None
    try:
        with open(out_json) as f:
            return json.load(f).get("summary")
    except (json.JSONDecodeError, OSError):
        return None


def print_stratified_table(per_snr: dict[str, dict]):
    """Print an SNR-category x mean-metric table, one row per category."""
    if not per_snr:
        return
    cat_w = max(len("SNR"), max(len(c) for c in per_snr))
    header = "  ".join(
        [f"{'SNR':<{cat_w}}", f"{'N':>4}"]
        + [f"{METRIC_LABELS[m]:>10}" for m in METRIC_NAMES]
    )
    print("\n" + "=" * len(header))
    print("PER-SNR BREAKDOWN (means)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cat, block in per_snr.items():
        cells = [f"{cat:<{cat_w}}", f"{block['num_utterances']:>4}"]
        for m in METRIC_NAMES:
            cells.append(f"{block[f'mean_{m}']:>10.4f}")
        print("  ".join(cells))


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
        help="Enhanced clips to save (with noisy/clean refs). With an SNR manifest, "
        "this many per SNR category; otherwise this many overall.",
    )
    parser.add_argument(
        "--snr_json",
        default="data/snr/test_snr.json",
        help="JSON mapping dataset-relative noisy paths to input SNR (dB); enables "
        "per-SNR stratified metrics and sampling. Pass '' to disable.",
    )
    parser.add_argument(
        "--snr_bins",
        default="0,5,10,15",
        help="Ascending, comma-separated SNR bin edges (dB) for stratification.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate checkpoints even if their output metrics.json already exists "
        "(default: skip and reuse the existing result).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    evaluator_cache: dict = {}

    # SNR stratification is opt-in and best-effort: a missing/empty manifest just
    # falls back to unstratified metrics + first-N sampling.
    args.snr_by_id = load_snr_map(args.snr_json)
    args.snr_edges = None
    if args.snr_by_id is not None:
        args.snr_edges = sorted(float(e) for e in args.snr_bins.split(",") if e.strip())
        if not args.snr_edges:
            warnings.warn("--snr_bins parsed to no edges; disabling stratification.")
            args.snr_by_id = None

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
            out_dir = os.path.join(args.output_dir, name)
            if not args.force:
                existing = load_existing_summary(out_dir)
                if existing is not None:
                    print(f"Skipping (found {os.path.join(out_dir, 'metrics.json')}); "
                          "pass --force to re-evaluate.")
                    summaries[name] = existing
                    continue
            ckpt_file, config_file = resolve_checkpoint_and_config(d, args.config)
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
    if not args.force and load_existing_summary(args.output_dir) is not None:
        print(f"Skipping (found {os.path.join(args.output_dir, 'metrics.json')}); "
              "pass --force to re-evaluate.")
        return
    ckpt_file, config_file = resolve_checkpoint_and_config(args.checkpoint, args.config)
    evaluate_checkpoint(
        ckpt_file, config_file, args.output_dir, args, device, evaluator_cache
    )


if __name__ == "__main__":
    main()
