"""Compare model representations with linear CKA (Centered Kernel Alignment).

Runs several trained generators on the *same* set of utterances, captures the
running ``[B, C, T, F]`` representation after each TF sub-block, and measures how
similar every model's representation is to every other's using linear CKA.

CKA is invariant to orthogonal transforms, isotropic scaling, and channel
permutation, so it compares "what a layer computes" without needing the two
models to share a channel basis (or even a channel count). Every ``(t, f)`` cell
of the representation is treated as one sample and the ``C`` channels as its
features; samples are pooled over a handful of utterances.

Outputs (under ``--output_dir``):
  * ``cka_model_matrix.png`` -- model x model heatmap of the mean matched-layer
    CKA (one number per model pair): the quick "who resembles whom" overview.
  * ``cka_layerwise.png`` -- an N x N grid of L x L cross-layer CKA heatmaps,
    one per model pair (diagonal panels are each model's self-similarity across
    its own layers).
  * ``cka.npz`` -- raw arrays (per-pair L x L matrices, model summary, labels).

Requires the compared checkpoints to share the STFT grid (n_fft/hop) so the
representation ``(T, F)`` positions align; this is asserted at runtime.

Example
-------
    python evaluation/compare_cka.py --checkpoint ckpts --num_utts 8 \
        --output_dir eval_out/cka
"""

import argparse
import os
import sys

# Allow running from the repo root and importing the sibling plotting module.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import librosa
import numpy as np
import torch

from models.stfts import mag_phase_stft
from utils.util import load_config

from plot_tf_activations import (
    TFActivationCapture,
    expand_checkpoints,
    load_generator,
    resolve_checkpoint_and_config,
    run_name_for,
)


def build_pairs(args, cfg: dict, num: int):
    """Return up to ``num`` (noisy, clean) test pairs, sorted by basename."""
    import json

    from dataloaders.dataloader_vctk import (
        _common_root,
        extract_identifier,
        get_clean_path_for_noisy,
    )

    clean_json = args.test_clean_json or cfg["data_cfg"]["test_clean_json"]
    noisy_json = args.test_noisy_json or cfg["data_cfg"]["test_noisy_json"]
    with open(clean_json) as f:
        clean_paths = json.load(f)
    with open(noisy_json) as f:
        noisy_paths = json.load(f)
    clean_root = _common_root(clean_paths)
    noisy_root = _common_root(noisy_paths)
    clean_by_id = {extract_identifier(p, clean_root): p for p in clean_paths}
    noisy_sorted = sorted(noisy_paths, key=os.path.basename)

    pairs = []
    for noisy in noisy_sorted:
        clean = get_clean_path_for_noisy(noisy, noisy_root, clean_by_id)
        if clean is not None:
            pairs.append((noisy, clean))
        if len(pairs) >= num:
            break
    return pairs


def utt_to_input(noisy_path: str, cfg: dict, device: torch.device, max_seconds: float):
    """Load a noisy wav and return its (mag, pha) STFT inputs for the model."""
    sr = cfg["stft_cfg"]["sampling_rate"]
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress = cfg["model_cfg"]["compress_factor"]

    noisy_np, _ = librosa.load(noisy_path, sr=sr)
    noisy_np = noisy_np[: int(max_seconds * sr)]
    noisy_t = torch.from_numpy(noisy_np).float().to(device)
    # Match evaluate.py's per-utterance energy normalization.
    norm = torch.sqrt(len(noisy_t) / torch.sum(noisy_t**2.0))
    noisy_in = (noisy_t * norm).unsqueeze(0)
    mag, pha, _ = mag_phase_stft(noisy_in, n_fft, hop_size, win_size, compress)
    return mag, pha


def select_representations(records, mode: str):
    """Pick which captured tensors to compare, with matching labels.

    ``records`` is a list of ``(after_time, after_freq)`` per TF block.
    ``block`` -> one representation per block (its output, after the freq mixer).
    ``both``  -> the after-time and after-freq maps, giving a finer depth axis.
    """
    reps, labels = [], []
    for i, (after_time, after_freq) in enumerate(records):
        if mode == "both":
            reps += [after_time, after_freq]
            labels += [f"b{i}.t", f"b{i}.f"]
        else:  # "block"
            reps.append(after_freq)
            labels.append(f"blk{i}")
    return reps, labels


def rep_to_samples(rep: torch.Tensor) -> np.ndarray:
    """Flatten ``[1, C, T, F]`` to a ``(T*F, C)`` sample x feature matrix."""
    c = rep.shape[1]
    return rep[0].reshape(c, -1).transpose(0, 1).to(torch.float32).numpy()


def capture_model_reps(model, cfg, pairs, device, args, shape_cache, idx_cache):
    """Return this model's pooled sample matrices, one per layer.

    Samples are subsampled per utterance with a per-utterance-seeded index that
    is shared across models (so the same ``(t, f)`` cells are compared), then
    concatenated across utterances.
    """
    reps_mode = args.representations
    cap = args.max_samples_per_utt
    layer_chunks = None
    labels = None

    for u, (noisy_path, _clean) in enumerate(pairs):
        mag, pha = utt_to_input(noisy_path, cfg, device, args.max_seconds)
        with TFActivationCapture(model) as capt, torch.no_grad():
            model(mag, pha)
        reps, labels = select_representations(capt.records, reps_mode)

        _, _, t, f = reps[0].shape
        n = t * f
        if u not in shape_cache:
            shape_cache[u] = (t, f)
            rng = np.random.default_rng(1000 + u)
            idx_cache[u] = rng.permutation(n)[:cap] if n > cap else np.arange(n)
        elif shape_cache[u] != (t, f):
            raise ValueError(
                f"utterance {u}: representation shape {(t, f)} != {shape_cache[u]} "
                "from an earlier model (mismatched STFT grid); cannot align CKA samples."
            )
        idx = idx_cache[u]

        if layer_chunks is None:
            layer_chunks = [[] for _ in reps]
        for li, rep in enumerate(reps):
            layer_chunks[li].append(rep_to_samples(rep)[idx])

    return [np.concatenate(c, axis=0) for c in layer_chunks], labels


def center_and_norm(X: np.ndarray):
    """Column-center ``X`` and return ``(Xc, ||Xc^T Xc||_F)`` for CKA reuse."""
    Xc = X - X.mean(axis=0, keepdims=True)
    gram = Xc.T @ Xc
    self_norm = float(np.sqrt((gram * gram).sum()))
    return Xc, self_norm


def linear_cka(Xc, self_x, Yc, self_y) -> float:
    """Linear CKA from pre-centered matrices and their self-norms."""
    c = Xc.T @ Yc
    hsic = float((c * c).sum())
    denom = self_x * self_y
    return hsic / denom if denom > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Parent dir (ckpts/), a checkpoint dir, or a .pth file")
    parser.add_argument("--config", default=None, help="Override config.yaml for all checkpoints")
    parser.add_argument("--num_utts", type=int, default=8,
                        help="Number of test utterances to pool samples over")
    parser.add_argument("--test_clean_json", default=None)
    parser.add_argument("--test_noisy_json", default=None)
    parser.add_argument("--representations", choices=["block", "both"], default="block",
                        help="'block': one rep per TF block; 'both': after-time and after-freq")
    parser.add_argument("--max_samples_per_utt", type=int, default=4000,
                        help="Random (t,f) cells sampled per utterance (shared across models)")
    parser.add_argument("--max_seconds", type=float, default=6.0)
    parser.add_argument("--output_dir", default="eval_out/cka")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device(args.device)
    checkpoints = expand_checkpoints(args.checkpoint)
    if len(checkpoints) < 2:
        raise SystemExit("CKA needs >= 2 checkpoints to compare.")

    # Build the shared utterance list from the first checkpoint's config.
    _, cfg0_file = resolve_checkpoint_and_config(checkpoints[0], args.config)
    cfg0 = load_config(cfg0_file)
    pairs = build_pairs(args, cfg0, args.num_utts)
    if not pairs:
        raise SystemExit("No test pairs found; check the dataset JSONs.")
    print(f"comparing {len(checkpoints)} checkpoints over {len(pairs)} utterances")

    shape_cache, idx_cache = {}, {}
    reps_by_model = {}  # name -> list over layers of (N, C) arrays
    labels = None
    failures = []

    for ckpt_arg in checkpoints:
        name = run_name_for(ckpt_arg)
        try:
            ckpt_file, config_file = resolve_checkpoint_and_config(ckpt_arg, args.config)
            cfg = load_config(config_file)
            model = load_generator(ckpt_file, cfg, device)
            reps, labels = capture_model_reps(
                model, cfg, pairs, device, args, shape_cache, idx_cache
            )
            reps_by_model[name] = reps
            print(f"  [{name}] captured {len(reps)} layers, "
                  f"{reps[0].shape[0]} samples x {reps[0].shape[1]} channels")
            del model
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    names = list(reps_by_model)
    if len(names) < 2:
        raise SystemExit("Fewer than two checkpoints loaded successfully; nothing to compare.")
    n_models = len(names)
    n_layers = len(labels)

    # Pre-center each (model, layer) once and cache its self-norm.
    centered = {n: [center_and_norm(X) for X in reps_by_model[n]] for n in names}

    # Cross-layer CKA cube for every model pair, plus the matched-layer summary.
    pair_mats = np.zeros((n_models, n_models, n_layers, n_layers), dtype=np.float32)
    summary = np.zeros((n_models, n_models), dtype=np.float32)
    for ai, a in enumerate(names):
        for bi, b in enumerate(names):
            M = pair_mats[ai, bi]
            for i in range(n_layers):
                Xc, sx = centered[a][i]
                for j in range(n_layers):
                    Yc, sy = centered[b][j]
                    M[i, j] = linear_cka(Xc, sx, Yc, sy)
            summary[ai, bi] = float(np.mean(np.diag(M)))

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, "cka.npz"),
        names=np.array(names), labels=np.array(labels),
        pair_matrices=pair_mats, summary=summary,
    )

    _plot_summary(plt, summary, names, args)
    _plot_layerwise(plt, pair_mats, names, labels, args)

    # Console table of the matched-layer summary.
    print("\nmean matched-layer CKA:")
    print("            " + "  ".join(f"{n[:8]:>8}" for n in names))
    for ai, a in enumerate(names):
        print(f"  {a[:10]:>10}  " + "  ".join(f"{summary[ai, bi]:8.3f}" for bi in range(n_models)))
    print(f"\nwrote {args.output_dir}/cka_model_matrix.png, cka_layerwise.png, cka.npz")
    if failures:
        print(f"{len(failures)} checkpoint(s) failed: " + ", ".join(n for n, _ in failures))


def _plot_summary(plt, summary, names, args):
    """model x model heatmap of mean matched-layer CKA, annotated."""
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1.5), constrained_layout=True)
    im = ax.imshow(summary, cmap=args.cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = summary[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 0.6 else "black")
    ax.set_title("Mean matched-layer linear CKA", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CKA")
    fig.savefig(os.path.join(args.output_dir, "cka_model_matrix.png"), dpi=140)
    plt.close(fig)


def _plot_layerwise(plt, pair_mats, names, labels, args):
    """N x N grid of L x L cross-layer CKA heatmaps."""
    n = len(names)
    L = len(labels)
    fig, axes = plt.subplots(
        n, n, figsize=(1.6 * n + 1.5, 1.6 * n + 1.5), squeeze=False, constrained_layout=True
    )
    im = None
    for ai in range(n):
        for bi in range(n):
            ax = axes[ai][bi]
            im = ax.imshow(pair_mats[ai, bi], cmap=args.cmap, vmin=0.0, vmax=1.0,
                           origin="upper", aspect="equal")
            if ai == 0:
                ax.set_title(names[bi], fontsize=9)
            if bi == 0:
                ax.set_ylabel(names[ai], fontsize=9)
            if L <= 6:
                ax.set_xticks(range(L)); ax.set_yticks(range(L))
                ax.set_xticklabels(labels, fontsize=5, rotation=90)
                ax.set_yticklabels(labels, fontsize=5)
            else:
                ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="CKA (layer i of row vs layer j of col)")
    fig.suptitle("Cross-layer linear CKA between models", fontsize=12)
    fig.savefig(os.path.join(args.output_dir, "cka_layerwise.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
