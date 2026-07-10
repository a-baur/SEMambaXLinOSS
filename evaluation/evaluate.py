"""Evaluate a SEMamba generator checkpoint on a test set.

Supports two checkpoint layouts:
  * single-file ckpts (e.g. ``ckpts/SEMamba_advanced.pth``)
  * exp-run directories produced by ``train.py`` (e.g. ``exp/LinOSS``),
    where the latest ``g_????????.pth`` is picked automatically and the
    co-located ``config.yaml`` is used by default.

Multiple checkpoints may be passed at once; when more than one GPU is
available they are evaluated in parallel (one process per GPU, checkpoints
assigned round-robin).

Outputs (per checkpoint; written to ``<output_dir>/<run_name>`` when more
than one checkpoint is given, else directly to ``<output_dir>``):
  * ``metrics.json`` -- per-utterance PESQ / MR-STFT / UTMOS plus a summary
    block with the mean and variance of each.
  * ``samples/<utt>_{noisy,enhanced,clean}.wav`` -- a fixed, deterministic
    subset of enhanced clips with their noisy / clean refs.

In addition, every checkpoint appends one row (mean + variance of each
metric) to a shared ``--results_file`` CSV, so several runs accumulate into a
single comparison table.
"""

import argparse
import csv
import fcntl
import glob
import json
import math
import os
import warnings

# ignore mamba_ssm deprecation warning
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.custom_(fwd|bwd).*is deprecated.*",
    category=FutureWarning,
)

import librosa
import soundfile as sf
import torch
import torch.multiprocessing
from dataloaders.dataloader_vctk import (
    _common_root,
    extract_identifier,
    get_clean_path_for_noisy,
)
from models.generator import SEMamba
from models.stfts import mag_phase_istft, mag_phase_stft
from utils.metrics import (
    LOWER_IS_BETTER,
    METRIC_LABELS,
    METRIC_NAMES,
    PESQ_FAILED,
    Evaluator,
    metric_metadata,
)
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
        raise FileNotFoundError(f"Config not found at {config_file}; pass --config explicitly.")
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

    norm = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav**2.0))
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


def run_name_for(ckpt_arg: str) -> str:
    """Stable, filesystem-safe name for a checkpoint's per-run output dir."""
    p = ckpt_arg.rstrip("/")
    if os.path.isdir(p):
        return os.path.basename(p)
    # A bare g_*.pth lives inside an exp dir; prefer that dir's name.
    parent = os.path.basename(os.path.dirname(p))
    stem = os.path.splitext(os.path.basename(p))[0]
    return f"{parent}_{stem}" if parent else stem


def out_dir_for(ckpt_arg: str, output_dir: str, multi: bool) -> str:
    """Per-checkpoint output dir: a named subdir when comparing several models."""
    return os.path.join(output_dir, run_name_for(ckpt_arg)) if multi else output_dir


def append_results(results_file: str, row: dict) -> None:
    """Append one CSV row to ``results_file`` under an exclusive file lock.

    The lock makes this safe to call from several parallel worker processes;
    the header is written only when the file is empty.
    """
    parent = os.path.dirname(results_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(results_file, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_sample(samples_dir, noisy_path, enhanced, noisy_np, clean_np, sr):
    """Write the enhanced / noisy / clean wav triplet for one sampled utterance."""
    base = os.path.splitext(os.path.basename(noisy_path))[0]
    sf.write(
        os.path.join(samples_dir, f"{base}_enhanced.wav"), enhanced.cpu().numpy(), sr, "PCM_16"
    )
    sf.write(os.path.join(samples_dir, f"{base}_noisy.wav"), noisy_np, sr, "PCM_16")
    sf.write(os.path.join(samples_dir, f"{base}_clean.wav"), clean_np, sr, "PCM_16")


def evaluate_pairs(model, evaluator, cfg, pairs, device, sr, samples_dir, sample_indices, name):
    """Run the model over every (noisy, clean) pair, accumulating metric stats.

    Returns ``(per_utt, sums, sumsq, counts)`` where ``sums``/``sumsq``/``counts``
    are per-metric running totals over the non-NaN entries (for mean + variance).
    """
    sums = dict.fromkeys(METRIC_NAMES, 0.0)
    sumsq = dict.fromkeys(METRIC_NAMES, 0.0)
    counts = dict.fromkeys(METRIC_NAMES, 0)
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

        row = {"noisy": noisy_path, "clean": clean_path}
        for m in METRIC_NAMES:
            v = metrics[m]
            # PESQ sentinel from utils/metrics.py on degenerate utterances -> NaN.
            row[m] = float("nan") if (m == "pesq" and v == PESQ_FAILED) else v
        per_utt.append(row)
        for m in METRIC_NAMES:
            v = row[m]
            if v is not None and not math.isnan(v):
                sums[m] += v
                sumsq[m] += v * v
                counts[m] += 1

        if i in sample_indices:
            write_sample(samples_dir, noisy_path, enhanced, noisy_np, clean_np, sr)

        if (i + 1) % 50 == 0:
            running = {m: sums[m] / max(counts[m], 1) for m in METRIC_NAMES}
            stats = "  ".join(f"{m.upper()}={running[m]:.4f}" for m in METRIC_NAMES)
            print(f"  [{name}][{i + 1}/{len(pairs)}] {stats}")

    return per_utt, sums, sumsq, counts


def run_one(ckpt_arg: str, args, device: torch.device, multi: bool) -> dict:
    """Evaluate a single checkpoint; write metrics.json and return its summary.

    When ``args.results_file`` is set the summary (mean + variance per metric)
    is also appended as a row to that shared CSV.
    """
    name = run_name_for(ckpt_arg)
    ckpt_file, config_file = resolve_checkpoint_and_config(ckpt_arg, args.config)
    out_dir = out_dir_for(ckpt_arg, args.output_dir, multi)
    print(f"[{name}] checkpoint={ckpt_file} config={config_file} device={device}")

    cfg = load_config(config_file)
    sr = cfg["stft_cfg"]["sampling_rate"]

    model = load_generator(ckpt_file, cfg, device)
    evaluator = Evaluator(sr=sr).to(device)

    pairs = build_pair_list(cfg, args.test_clean_json, args.test_noisy_json)
    print(f"[{name}] evaluating {len(pairs)} utterances at {sr} Hz")

    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sample_indices = set(range(min(args.num_samples, len(pairs))))

    per_utt, sums, sumsq, counts = evaluate_pairs(
        model, evaluator, cfg, pairs, device, sr, samples_dir, sample_indices, name
    )

    means = {m: sums[m] / max(counts[m], 1) for m in METRIC_NAMES}
    # Sample variance (ddof=1); 0.0 when fewer than two valid values.
    variances = {
        m: (sumsq[m] - sums[m] * sums[m] / counts[m]) / (counts[m] - 1) if counts[m] > 1 else 0.0
        for m in METRIC_NAMES
    }

    summary = {
        "checkpoint": ckpt_file,
        "config": config_file,
        "sampling_rate": sr,
        "num_utterances": len(pairs),
        "valid_counts": counts,
        "samples_dir": samples_dir,
        # Self-describing metric set so downstream tooling (plots) needs no
        # knowledge of the metric registry.
        "metrics_meta": metric_metadata(),
    }
    for m in METRIC_NAMES:
        summary[f"mean_{m}"] = means[m]
        summary[f"var_{m}"] = variances[m]

    out_json = os.path.join(out_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_utterance": per_utt}, f, indent=2)

    if args.results_file:
        results_row = {
            "run_name": name,
            "checkpoint": ckpt_file,
            "config": config_file,
            "num_utterances": len(pairs),
        }
        for m in METRIC_NAMES:
            results_row[f"mean_{m}"] = means[m]
            results_row[f"var_{m}"] = variances[m]
            results_row[f"n_{m}"] = counts[m]
        append_results(args.results_file, results_row)

    print(f"\n[{name}] results")
    for m in METRIC_NAMES:
        print(
            f"  {METRIC_LABELS[m]:<10}: {means[m]:.4f} (var {variances[m]:.4g}) "
            f"over {counts[m]}/{len(pairs)} utts"
        )
    print(f"  samples : {samples_dir} ({len(sample_indices)} clips)")
    print(f"  per-utt : {out_json}")
    if args.results_file:
        print(f"  results : {args.results_file}")
    return summary


def load_per_utterance(out_dir: str) -> dict:
    """Load a checkpoint's per-utterance metrics from ``out_dir/metrics.json``.

    Returns ``{metric: {utt_key: value}}`` keyed by the noisy-file path (the
    stable per-utterance identifier shared across models on the same test set).
    """
    with open(os.path.join(out_dir, "metrics.json")) as f:
        data = json.load(f)
    by_metric = {m: {} for m in METRIC_NAMES}
    for row in data["per_utterance"]:
        key = row["noisy"]
        for m in METRIC_NAMES:
            by_metric[m][key] = row.get(m)
    return by_metric


def correct_pvalues(pvals: list, method: str):
    """Multiple-comparison correction. ``method`` in {holm, fdr_bh, none}.

    NaN p-values pass through as NaN and are excluded from the family size.
    """
    import numpy as np

    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    pm = p[mask]
    m = pm.size
    if m == 0:
        return out.tolist()
    order = np.argsort(pm)
    corr = np.empty(m)
    if method == "holm":
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * pm[idx])
            corr[idx] = min(running, 1.0)
    elif method == "fdr_bh":
        ranked = pm[order]
        adj = ranked * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        corr[order] = np.minimum(adj, 1.0)
    else:  # "none"
        corr = pm
    out[mask] = corr
    return out.tolist()


def paired_test(a, b, test: str):
    """Paired significance test on aligned vectors ``a``, ``b``.

    Returns ``(statistic, p_value)``; ``(nan, nan)`` when the test is undefined
    (e.g. all-zero differences for Wilcoxon, or fewer than two pairs).
    """
    import numpy as np
    from scipy import stats

    if a.size < 2:
        return float("nan"), float("nan")
    if test == "ttest":
        stat, p = stats.ttest_rel(a, b)
    else:  # wilcoxon signed-rank
        if np.allclose(a, b):
            return 0.0, 1.0
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            return float("nan"), float("nan")
    return float(stat), float(p)


def compute_significance(models: list, tests: list, correction: str, alpha: float) -> list:
    """Pairwise paired significance tests across models.

    ``models`` is a list of ``(name, per_utt)`` where ``per_utt`` comes from
    :func:`load_per_utterance`. For each metric and each requested test, every
    model pair is compared on the utterances both scored with a finite value;
    p-values are corrected within each ``(metric, test)`` family.
    """
    import numpy as np

    rows = []
    for metric in METRIC_NAMES:
        for test in tests:
            family = []  # collect rows for this (metric, test) before correcting
            for ai in range(len(models)):
                for bi in range(ai + 1, len(models)):
                    name_a, pa = models[ai]
                    name_b, pb = models[bi]
                    da, db = pa[metric], pb[metric]
                    keys = [
                        k
                        for k in da
                        if k in db
                        and da[k] is not None
                        and db[k] is not None
                        and not math.isnan(da[k])
                        and not math.isnan(db[k])
                    ]
                    a = np.array([da[k] for k in keys], dtype=float)
                    b = np.array([db[k] for k in keys], dtype=float)
                    stat, p = paired_test(a, b, test)
                    mean_a = float(a.mean()) if a.size else float("nan")
                    mean_b = float(b.mean()) if b.size else float("nan")
                    family.append(
                        {
                            "metric": metric,
                            "test": test,
                            "model_a": name_a,
                            "model_b": name_b,
                            "n_paired": len(keys),
                            "mean_a": mean_a,
                            "mean_b": mean_b,
                            "mean_diff": mean_a - mean_b,
                            "statistic": stat,
                            "p_value": p,
                        }
                    )
            corrected = correct_pvalues([r["p_value"] for r in family], correction)
            for r, pc in zip(family, corrected):
                r["p_corrected"] = pc
                r["significant"] = bool(pc == pc and pc < alpha)  # pc==pc filters NaN
                # Which model is favored (per metric direction) when significant.
                if r["significant"] and r["mean_a"] != r["mean_b"]:
                    a_better = (r["mean_a"] < r["mean_b"]) == (metric in LOWER_IS_BETTER)
                    r["favored"] = r["model_a"] if a_better else r["model_b"]
                else:
                    r["favored"] = ""
            rows.extend(family)
    return rows


def run_significance(models: list, output_dir: str, tests: list, correction: str, alpha: float):
    """Compute pairwise significance and write significance.csv + console table."""
    if len(models) < 2:
        print("Significance test needs >= 2 checkpoints; skipping.")
        return
    rows = compute_significance(models, tests, correction, alpha)

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "significance.csv")
    fields = [
        "metric",
        "test",
        "model_a",
        "model_b",
        "n_paired",
        "mean_a",
        "mean_b",
        "mean_diff",
        "statistic",
        "p_value",
        "p_corrected",
        "significant",
        "favored",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSignificance ({'/'.join(tests)}, {correction} correction, alpha={alpha})")
    for metric in METRIC_NAMES:
        mrows = [r for r in rows if r["metric"] == metric]
        sig = [r for r in mrows if r["significant"]]
        if not sig:
            print(f"  {METRIC_LABELS[metric]:<10}: no significant pairwise differences")
            continue
        print(f"  {METRIC_LABELS[metric]}:")
        for r in sig:
            print(
                f"    {r['model_a']} vs {r['model_b']} [{r['test']}]: "
                f"diff={r['mean_diff']:+.4f} p_corr={r['p_corrected']:.2e} "
                f"-> {r['favored']} better"
            )
    print(f"  table: {out_csv}")


def _worker(gpu_id: int, ckpt_list: list, args) -> None:
    """Evaluate an assigned list of checkpoints sequentially on one GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    for ckpt_arg in ckpt_list:
        try:
            run_one(ckpt_arg, args, device, multi=True)
        except Exception as e:  # keep sibling checkpoints alive on failure
            print(f"[gpu{gpu_id}] FAILED {ckpt_arg}: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        default=["ckpts/SEMamba_advanced.pth"],
        help=(
            "One or more .pth files or exp directories containing g_*.pth. "
            "Multiple checkpoints are evaluated in parallel across GPUs."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config. Defaults to config.yaml next to the checkpoint.",
    )
    parser.add_argument("--test_clean_json", default="data/ears_wham_16k/test_clean.json")
    parser.add_argument("--test_noisy_json", default="data/ears_wham_16k/test_noisy.json")
    parser.add_argument("--output_dir", default="eval_out")
    parser.add_argument(
        "--results_file",
        default=None,
        help=(
            "CSV to append one summary row (mean + variance per metric) per "
            "checkpoint. Defaults to <output_dir>/results.csv."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of fixed enhanced clips to save (alongside noisy/clean refs).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--significance_test",
        choices=["wilcoxon", "ttest", "both", "none"],
        default="wilcoxon",
        help=(
            "Paired per-utterance significance test between every model pair "
            "(needs >= 2 checkpoints). 'none' disables it."
        ),
    )
    parser.add_argument(
        "--significance_correction",
        choices=["holm", "fdr_bh", "none"],
        default="holm",
        help="Multiple-comparison correction applied within each metric.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for the corrected p-value.",
    )
    parser.add_argument(
        "--significance_only",
        action="store_true",
        help="Skip evaluation; compute significance from existing metrics.json files.",
    )
    args = parser.parse_args()

    if args.results_file is None:
        args.results_file = os.path.join(args.output_dir, "results.csv")

    checkpoints = args.checkpoint
    multi = len(checkpoints) > 1

    if not args.significance_only:
        dispatch_eval(checkpoints, args, multi)

    # Pairwise significance testing across the evaluated checkpoints.
    if args.significance_test != "none" and multi:
        gather_and_test(checkpoints, args, multi)


def dispatch_eval(checkpoints: list, args, multi: bool) -> None:
    """Evaluate every checkpoint, in parallel across GPUs when possible."""
    device = torch.device(args.device)
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0

    # Parallelise only when we have several checkpoints AND several GPUs;
    # sharing one GPU across processes risks OOM, so fall back to sequential.
    if multi and n_gpu > 1:
        n_workers = min(n_gpu, len(checkpoints))
        buckets = [[] for _ in range(n_workers)]
        for i, ckpt in enumerate(checkpoints):
            buckets[i % n_workers].append(ckpt)
        print(f"Evaluating {len(checkpoints)} checkpoints across {n_workers} GPUs")

        ctx = torch.multiprocessing.get_context("spawn")
        procs = []
        for gpu_id, bucket in enumerate(buckets):
            p = ctx.Process(target=_worker, args=(gpu_id, bucket, args))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        if multi:
            print(f"Evaluating {len(checkpoints)} checkpoints sequentially on {device}")
        for ckpt in checkpoints:
            run_one(ckpt, args, device, multi=multi)

    if multi:
        print(f"\nResults table: {args.results_file}")


def gather_and_test(checkpoints: list, args, multi: bool) -> None:
    """Load each checkpoint's per-utterance metrics and run pairwise tests."""
    tests = ["wilcoxon", "ttest"] if args.significance_test == "both" else [args.significance_test]
    models = []
    for ckpt in checkpoints:
        out_dir = out_dir_for(ckpt, args.output_dir, multi)
        try:
            models.append((run_name_for(ckpt), load_per_utterance(out_dir)))
        except FileNotFoundError:
            print(f"  no metrics.json under {out_dir}; excluded from significance test")
    run_significance(models, args.output_dir, tests, args.significance_correction, args.alpha)


if __name__ == "__main__":
    main()
