"""Visualize a multi-model comparison produced by ``evaluate.py``.

Reads the per-checkpoint ``metrics.json`` files under a comparison directory
(e.g. ``eval_out/cmp/<model>/metrics.json``) plus the ``significance.csv``
written by ``evaluate.py``, and renders figures that surface the *performance
differences* between models. Because every model is scored on the same test
utterances, the comparison is paired -- the plots lean on that:

  * ``distributions.png`` -- per-utterance metric distributions (box + mean)
    for every model, one panel per metric. The spread, not just the mean.
  * ``paired_diff_vs_<baseline>.png`` -- the distribution of per-utterance
    differences (model - baseline), oriented so >0 always means "better than
    baseline", with the Wilcoxon corrected p-value and win-rate annotated.
    This is the paired quantity the significance test is computed on.
  * ``significance_heatmap.png`` -- metric x model grid of win-rate over the
    baseline, annotated with the mean delta and significance stars.

The metric set, labels, and directions are read from each run's
``metrics.json`` (the ``metrics_meta`` block ``evaluate.py`` embeds), so adding
a metric to the evaluator surfaces here automatically -- no edit needed.

Standalone (no torch / GPU needed):
    python plot_comparison.py --comparison_dir eval_out/cmp --baseline Mamba
"""

import argparse
import json
import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fallback direction for runs predating the embedded metrics_meta block.
_FALLBACK_LOWER_IS_BETTER = {"mrstft", "lsd"}


@dataclass(frozen=True)
class Metric:
    """Display metadata for one metric, as discovered from the run data."""

    name: str
    label: str
    lower_is_better: bool

    @property
    def sign(self) -> int:
        """+1 if higher is better, -1 if lower is better (orients diffs)."""
        return -1 if self.lower_is_better else 1

    @property
    def arrow(self) -> str:
        """Short direction hint for panel titles."""
        return "down=better" if self.lower_is_better else "up=better"


def metrics_from_json(data: dict) -> list[Metric]:
    """Metric metadata for one run: from ``metrics_meta`` if present, else inferred."""
    meta = data.get("summary", {}).get("metrics_meta")
    if meta:
        return [Metric(m["name"], m["label"], bool(m["lower_is_better"])) for m in meta]
    # Older runs: infer names from the per-utterance keys, guess label/direction.
    keys = [k for k in data["per_utterance"][0] if k not in ("noisy", "clean")]
    return [Metric(k, k.upper(), k in _FALLBACK_LOWER_IS_BETTER) for k in keys]


def stars(p) -> str:
    """Significance stars for a (corrected) p-value."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def discover_models(comparison_dir: str):
    """Return ``(models, metrics)``.

    ``models`` is ``{model_name: {metric: {utt_key: value}}}``; ``metrics`` is the
    list of :class:`Metric` discovered from the first run's ``metrics.json``.
    """
    models, metrics = {}, None
    for name in sorted(os.listdir(comparison_dir)):
        mpath = os.path.join(comparison_dir, name, "metrics.json")
        if not os.path.isfile(mpath):
            continue
        with open(mpath) as f:
            data = json.load(f)
        if metrics is None:
            metrics = metrics_from_json(data)
        by_metric = {m.name: {} for m in metrics}
        for row in data["per_utterance"]:
            for m in metrics:
                by_metric[m.name][row["noisy"]] = row.get(m.name)
        models[name] = by_metric
    return models, (metrics or [])


def aligned_frame(models: dict, metrics: list) -> pd.DataFrame:
    """Long-form DataFrame (utt, model, metric, value) on the shared utterances."""
    key_metric = metrics[0].name
    common = None
    for by_metric in models.values():
        keys = set(by_metric[key_metric])
        common = keys if common is None else (common & keys)
    common = sorted(common)

    records = []
    for name, by_metric in models.items():
        for m in metrics:
            d = by_metric[m.name]
            for k in common:
                v = d.get(k)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    records.append((k, name, m.name, v))
    return pd.DataFrame(records, columns=["utt", "model", "metric", "value"])


def load_significance(comparison_dir: str) -> dict:
    """Index significance.csv by (metric, frozenset{a,b}) -> row dict (Wilcoxon)."""
    path = os.path.join(comparison_dir, "significance.csv")
    if not os.path.isfile(path):
        return {}
    sig = pd.read_csv(path)
    sig = sig[sig["test"] == "wilcoxon"] if "test" in sig.columns else sig
    out = {}
    for _, r in sig.iterrows():
        out[(r["metric"], frozenset({r["model_a"], r["model_b"]}))] = r.to_dict()
    return out


def rank_metric(metrics: list) -> Metric:
    """Metric used to order models left-to-right: PESQ if present, else the first."""
    return next((m for m in metrics if m.name == "pesq"), metrics[0])


def model_order(df: pd.DataFrame, metrics: list, baseline: str) -> list:
    """Baseline first, then remaining models best-to-worst on the ranking metric."""
    rm = rank_metric(metrics)
    means = df[df.metric == rm.name].groupby("model")["value"].mean() * rm.sign
    ranked = list(means.sort_values(ascending=False).index)
    others = [m for m in ranked if m != baseline]
    return ([baseline] + others) if baseline in ranked else ranked


def _grid(n):
    ncol = 4 if n > 3 else n
    nrow = math.ceil(n / ncol)
    return nrow, ncol


def _panel_axes(metrics):
    nrow, ncol = _grid(len(metrics))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    return fig, axes


def plot_distributions(df, metrics, order, colors, out_path):
    """Box plots of per-utterance values per model, one panel per metric."""
    fig, axes = _panel_axes(metrics)
    for ax, m in zip(axes, metrics):
        data = [df[(df.metric == m.name) & (df.model == mdl)]["value"].to_numpy() for mdl in order]
        bp = ax.boxplot(
            data,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "k", "markersize": 4},
            medianprops={"color": "k"},
            widths=0.6,
            patch_artist=True,
            showfliers=False,
        )
        for patch, mdl in zip(bp["boxes"], order):
            patch.set_facecolor(colors[mdl])
            patch.set_alpha(0.75)
        ax.set_title(f"{m.label}  ({m.arrow})", fontsize=10)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)
    fig.suptitle("Per-utterance metric distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_paired_diff(df, metrics, order, colors, baseline, sig, out_path):
    """Violin of per-utterance (model - baseline), oriented so >0 = better."""
    others = [m for m in order if m != baseline]
    wide = df.pivot_table(index="utt", columns=["model", "metric"], values="value")
    fig, axes = _panel_axes(metrics)
    for ax, m in zip(axes, metrics):
        deltas, labels = [], []
        for mdl in others:
            pair = wide[[(mdl, m.name), (baseline, m.name)]].dropna()
            d = m.sign * (pair[(mdl, m.name)] - pair[(baseline, m.name)]).to_numpy()
            deltas.append(d)
            rec = sig.get((m.name, frozenset({mdl, baseline})))
            p = rec["p_corrected"] if rec else float("nan")
            winrate = float(np.mean(d > 0)) * 100
            labels.append(f"{mdl}\n{stars(p)}  {winrate:.0f}% win")
        parts = ax.violinplot(deltas, showmeans=True, showextrema=False)
        for body, mdl in zip(parts["bodies"], others):
            body.set_facecolor(colors[mdl])
            body.set_alpha(0.7)
        ax.axhline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax.set_title(m.label, fontsize=10)
        ax.set_xticks(range(1, len(others) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Per-utterance improvement over {baseline}  (>0 = better than {baseline})", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_significance_heatmap(df, metrics, order, baseline, sig, out_path):
    """Metric x model heatmap: win-rate vs baseline, annotated with delta + stars."""
    others = [m for m in order if m != baseline]
    wide = df.pivot_table(index="utt", columns=["model", "metric"], values="value")
    win = np.full((len(metrics), len(others)), np.nan)
    annot = [["" for _ in others] for _ in metrics]
    for i, m in enumerate(metrics):
        for j, mdl in enumerate(others):
            pair = wide[[(mdl, m.name), (baseline, m.name)]].dropna()
            a = pair[(mdl, m.name)].to_numpy()
            b = pair[(baseline, m.name)].to_numpy()
            win[i, j] = np.mean(m.sign * (a - b) > 0) * 100
            rec = sig.get((m.name, frozenset({mdl, baseline})))
            p = rec["p_corrected"] if rec else float("nan")
            annot[i][j] = f"{np.mean(a - b):+.3f}\n{stars(p)}"

    fig, ax = plt.subplots(figsize=(1.6 * len(others) + 2.5, 0.6 * len(metrics) + 2))
    im = ax.imshow(win, cmap="RdBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(others)))
    ax.set_xticklabels(others, fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.label for m in metrics], fontsize=10)
    for i in range(len(metrics)):
        for j in range(len(others)):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"% of utterances better than {baseline}")
    ax.set_title(
        f"Performance vs {baseline}\n(cell: mean delta, raw units; ***/**/* = Holm-corrected p)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_dir", default="eval_out/cmp")
    parser.add_argument(
        "--baseline", default=None, help="Reference model for paired diffs (default: best-ranked)."
    )
    parser.add_argument("--output_dir", default=None, help="Default: <comparison_dir>/figures.")
    args = parser.parse_args()

    models, metrics = discover_models(args.comparison_dir)
    if len(models) < 2:
        raise SystemExit(f"Need >= 2 models with metrics.json under {args.comparison_dir}")
    if not metrics:
        raise SystemExit("No metrics found in metrics.json files.")
    df = aligned_frame(models, metrics)
    sig = load_significance(args.comparison_dir)

    baseline = args.baseline
    if baseline is None or baseline not in models:
        rm = rank_metric(metrics)
        baseline = (df[df.metric == rm.name].groupby("model")["value"].mean() * rm.sign).idxmax()
    order = model_order(df, metrics, baseline)
    cmap = plt.get_cmap("tab10")
    colors = {mdl: cmap(i % 10) for i, mdl in enumerate(order)}

    out_dir = args.output_dir or os.path.join(args.comparison_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"Models: {order}  | metrics: {[m.name for m in metrics]}  | "
        f"shared utterances: {df.utt.nunique()}  | baseline: {baseline}"
    )

    plot_distributions(df, metrics, order, colors, os.path.join(out_dir, "distributions.png"))
    plot_paired_diff(
        df,
        metrics,
        order,
        colors,
        baseline,
        sig,
        os.path.join(out_dir, f"paired_diff_vs_{baseline}.png"),
    )
    plot_significance_heatmap(
        df, metrics, order, baseline, sig, os.path.join(out_dir, "significance_heatmap.png")
    )
    print(f"Wrote 3 figures to {out_dir}")


if __name__ == "__main__":
    main()
