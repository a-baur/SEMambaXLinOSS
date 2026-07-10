"""Clean-vs-noisy representation convergence across models.

Feeds each generator the *clean* and the *noisy* version of the same utterance,
captures the running ``[B, C, T, F]`` representation after each TF sub-block, and
measures how close the noisy-driven representation is to the clean-driven one at
each depth. A good enhancer should pull its deep representation toward the one it
would build from clean speech; *where* on the depth axis (and how much) each
architecture does this is the comparison.

Two complementary per-block metrics, pooled over ``--num_utts`` utterances:
  * ``CKA(noisy, clean)`` -- structural alignment in [0, 1] (scale/rotation
    invariant); higher = more converged.
  * relative L2 distance ``||H_noisy - H_clean||_F / ||H_clean||_F`` -- absolute
    closeness in the model's own basis (both reps share it, so this is
    meaningful); lower = more converged.

Outputs (under ``--output_dir``):
  * ``convergence_curves.png`` -- both metrics vs block depth, one line per model.
  * ``difference_maps.png`` -- for one reference utterance, the channel-collapsed
    ``||H_noisy - H_clean||`` per model x block: where the correction lives.
  * ``convergence.npz`` -- raw per-model curves.

Clean and noisy are trimmed to the same length so their ``(t, f)`` positions
align; each input is energy-normalized independently, matching inference.

Example
-------
    python evaluation/compare_clean_noisy.py --checkpoint ckpts --num_utts 8 \
        --output_dir eval_out/convergence
"""

import argparse
import os
import sys

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
    reduce_channels,
    resolve_checkpoint_and_config,
    run_name_for,
)
from compare_cka import (
    build_pairs,
    center_and_norm,
    linear_cka,
    rep_to_samples,
    select_representations,
)


def pair_to_inputs(noisy_path, clean_path, cfg, device, max_seconds):
    """Load a (noisy, clean) pair, trimmed to equal length, as STFT inputs.

    Each signal is energy-normalized independently (as at inference), so the
    representations differ only through content + a per-signal gain.
    """
    sr = cfg["stft_cfg"]["sampling_rate"]
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress = cfg["model_cfg"]["compress_factor"]

    noisy_np, _ = librosa.load(noisy_path, sr=sr)
    clean_np, _ = librosa.load(clean_path, sr=sr)
    n = min(len(noisy_np), len(clean_np), int(max_seconds * sr))
    noisy_np, clean_np = noisy_np[:n], clean_np[:n]

    def to_input(x):
        t = torch.from_numpy(x).float().to(device)
        norm = torch.sqrt(len(t) / torch.sum(t**2.0))
        return (t * norm).unsqueeze(0)

    mag_n, pha_n, _ = mag_phase_stft(to_input(noisy_np), n_fft, hop_size, win_size, compress)
    mag_c, pha_c, _ = mag_phase_stft(to_input(clean_np), n_fft, hop_size, win_size, compress)
    return mag_n, pha_n, mag_c, pha_c


def capture(model, mag, pha):
    """Run the model once and return the list of ``(after_time, after_freq)``."""
    with TFActivationCapture(model) as capt, torch.no_grad():
        model(mag, pha)
    return capt.records


def analyze_model(model, cfg, pairs, device, args, shape_cache, idx_cache):
    """Return per-layer convergence metrics for one model.

    Returns ``(labels, cka, rel, cos, diff_maps)`` where the three metric lists
    are one value per layer and ``diff_maps`` are the reference-utterance
    channel-collapsed difference maps (``[F, T]`` per layer).
    """
    cap = args.max_samples_per_utt
    clean_chunks = noisy_chunks = None
    rel_acc = cos_acc = None
    labels = None
    diff_maps = None
    eps = 1e-8

    for u, (noisy_path, clean_path) in enumerate(pairs):
        mag_n, pha_n, mag_c, pha_c = pair_to_inputs(
            noisy_path, clean_path, cfg, device, args.max_seconds
        )
        reps_n, labels = select_representations(capture(model, mag_n, pha_n), args.representations)
        reps_c, _ = select_representations(capture(model, mag_c, pha_c), args.representations)
        n_layers = len(reps_n)

        _, _, t, f = reps_n[0].shape
        n = t * f
        if u not in shape_cache:
            shape_cache[u] = (t, f)
            rng = np.random.default_rng(2000 + u)
            idx_cache[u] = rng.permutation(n)[:cap] if n > cap else np.arange(n)
        elif shape_cache[u] != (t, f):
            raise ValueError(
                f"utterance {u}: shape {(t, f)} != {shape_cache[u]} from an earlier model."
            )
        idx = idx_cache[u]

        if clean_chunks is None:
            clean_chunks = [[] for _ in range(n_layers)]
            noisy_chunks = [[] for _ in range(n_layers)]
            rel_acc = [[] for _ in range(n_layers)]
            cos_acc = [[] for _ in range(n_layers)]

        store_diff = u == args.ref_index
        if store_diff:
            diff_maps = []
        for li in range(n_layers):
            hn = reps_n[li][0]  # [C, T, F]
            hc = reps_c[li][0]
            diff = hn - hc
            hc_norm = hc.norm().clamp_min(eps)
            rel_acc[li].append((diff.norm() / hc_norm).item())
            cos = (hn.flatten() @ hc.flatten()) / (hn.norm() * hc.norm()).clamp_min(eps)
            cos_acc[li].append(cos.item())

            noisy_chunks[li].append(rep_to_samples(reps_n[li])[idx])
            clean_chunks[li].append(rep_to_samples(reps_c[li])[idx])

            if store_diff:
                # |H_noisy - H_clean| collapsed over channels: where the model
                # rewrites its representation. Per-panel robust-scaled at plot time.
                diff_maps.append(reduce_channels(reps_n[li] - reps_c[li], "l2", "none"))

    cka = []
    for li in range(len(labels)):
        Xc_n, sn = center_and_norm(np.concatenate(noisy_chunks[li], axis=0))
        Xc_c, sc = center_and_norm(np.concatenate(clean_chunks[li], axis=0))
        cka.append(linear_cka(Xc_n, sn, Xc_c, sc))
    rel = [float(np.mean(v)) for v in rel_acc]
    cos = [float(np.mean(v)) for v in cos_acc]
    return labels, cka, rel, cos, diff_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Parent dir (ckpts/), a checkpoint dir, or a .pth file")
    parser.add_argument("--config", default=None, help="Override config.yaml for all checkpoints")
    parser.add_argument("--num_utts", type=int, default=8)
    parser.add_argument("--ref_index", type=int, default=0,
                        help="Utterance (0-based) used for the difference-map figure")
    parser.add_argument("--test_clean_json", default=None)
    parser.add_argument("--test_noisy_json", default=None)
    parser.add_argument("--representations", choices=["block", "both"], default="block")
    parser.add_argument("--max_samples_per_utt", type=int, default=4000)
    parser.add_argument("--max_seconds", type=float, default=6.0)
    parser.add_argument("--output_dir", default="eval_out/convergence")
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device(args.device)
    checkpoints = expand_checkpoints(args.checkpoint)

    _, cfg0_file = resolve_checkpoint_and_config(checkpoints[0], args.config)
    cfg0 = load_config(cfg0_file)
    pairs = build_pairs(args, cfg0, args.num_utts)
    if not pairs:
        raise SystemExit("No test pairs found; check the dataset JSONs.")
    if not 0 <= args.ref_index < len(pairs):
        args.ref_index = 0
    print(f"analyzing {len(checkpoints)} checkpoints over {len(pairs)} utterances")

    shape_cache, idx_cache = {}, {}
    results = {}  # name -> dict(cka, rel, cos, diff_maps)
    labels = None
    failures = []

    for ckpt_arg in checkpoints:
        name = run_name_for(ckpt_arg)
        try:
            ckpt_file, config_file = resolve_checkpoint_and_config(ckpt_arg, args.config)
            cfg = load_config(config_file)
            model = load_generator(ckpt_file, cfg, device)
            labels, cka, rel, cos, diff_maps = analyze_model(
                model, cfg, pairs, device, args, shape_cache, idx_cache
            )
            results[name] = {"cka": cka, "rel": rel, "cos": cos, "diff_maps": diff_maps}
            print(f"  [{name}] CKA(n,c)={[round(x, 3) for x in cka]} "
                  f"relL2={[round(x, 3) for x in rel]}")
            del model
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not results:
        raise SystemExit("No checkpoints loaded successfully.")

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, "convergence.npz"),
        names=np.array(list(results)), labels=np.array(labels),
        cka=np.array([results[n]["cka"] for n in results]),
        rel=np.array([results[n]["rel"] for n in results]),
        cos=np.array([results[n]["cos"] for n in results]),
    )

    _plot_curves(plt, results, labels, args)
    _plot_diff_maps(plt, results, labels, args, pairs)

    print(f"\nwrote {args.output_dir}/convergence_curves.png, difference_maps.png, convergence.npz")
    if failures:
        print(f"{len(failures)} checkpoint(s) failed: " + ", ".join(n for n, _ in failures))


def _plot_curves(plt, results, labels, args):
    """CKA(noisy,clean) and relative L2 distance vs block depth, per model."""
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name, r in results.items():
        ax1.plot(x, r["cka"], marker="o", label=name)
        ax2.plot(x, r["rel"], marker="o", label=name)
    ax1.set_title("CKA(noisy, clean) — higher = more converged")
    ax1.set_ylabel("linear CKA"); ax1.set_ylim(0, 1.02)
    ax2.set_title("relative L2 distance ‖Hₙ−H_c‖ / ‖H_c‖ — lower = more converged")
    ax2.set_ylabel("relative distance")
    for ax in (ax1, ax2):
        ax.set_xlabel("TF block")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Clean-vs-noisy representation convergence with depth", fontsize=13)
    fig.savefig(os.path.join(args.output_dir, "convergence_curves.png"), dpi=140)
    plt.close(fig)


def _plot_diff_maps(plt, results, labels, args, pairs):
    """model x block grid of channel-collapsed |H_noisy - H_clean| maps."""
    names = [n for n in results if results[n]["diff_maps"] is not None]
    if not names:
        return
    L = len(labels)
    fig, axes = plt.subplots(
        len(names), L, figsize=(2.4 * L + 1, 2.1 * len(names) + 1),
        squeeze=False, constrained_layout=True,
    )
    for ai, name in enumerate(names):
        for li in range(L):
            ax = axes[ai][li]
            m = results[name]["diff_maps"][li]
            vmin, vmax = np.percentile(m, 2), np.percentile(m, 98)
            ax.imshow(m, origin="lower", aspect="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)
            if ai == 0:
                ax.set_title(labels[li], fontsize=9)
            if li == 0:
                ax.set_ylabel(name, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    ref_name = os.path.basename(pairs[args.ref_index][0])
    fig.suptitle(f"Representation change ‖noisy − clean‖ per block — {ref_name}", fontsize=12)
    fig.savefig(os.path.join(args.output_dir, "difference_maps.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
