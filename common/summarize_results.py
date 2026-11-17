r"""
Collect the metrics of all training runs into one comparison table and figure.

Reads every `<results>/<run>/metrics.csv` written by `train_catboost.py` and
writes, in `<results>/`:

    summary.csv    long table: target, features, r2, mae, rmse, ...
    summary.md     test R^2 (and MAE) as markdown tables, targets x feature sets
    summary.png    test R^2 per target, one small panel per feature set

Usage
-----
    python common/summarize_results.py --results results [--order porosity ph ...]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, BLUE = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"


def md_table(wide, fmt):
    head = "| target | " + " | ".join(wide.columns) + " |"
    sep = "|---|" + "---:|" * wide.shape[1]
    rows = ["| " + str(i) + " | " + " | ".join(fmt.format(v) for v in r) + " |"
            for i, r in zip(wide.index, wide.values)]
    return "\n".join([head, sep] + rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--order", nargs="+", default=[], help="order of the feature sets")
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.results, "*", "metrics.csv")))
    if not files:
        sys.exit(f"no metrics.csv under {a.results}/ - run 03_train.sh first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sets = [s for s in a.order if s in set(df.features)] + \
           [s for s in dict.fromkeys(df.features) if s not in a.order]
    targets = list(dict.fromkeys(df.target))
    df.to_csv(os.path.join(a.results, "summary.csv"), index=False, float_format="%.6g")

    r2 = df.pivot(index="target", columns="features", values="r2").loc[targets, sets]
    mae = df.pivot(index="target", columns="features", values="mae").loc[targets, sets]
    r2m = pd.concat([r2, r2.mean().to_frame("**mean**").T])
    maem = pd.concat([mae, mae.mean().to_frame("**mean**").T])
    n = df.iloc[0]
    text = (f"Single split, seed-fixed: {int(n.n_train)} train / {int(n.n_val)} validation / "
            f"{int(n.n_test)} test structures. Metrics on the test set.\n\n"
            f"**Test R²**\n\n{md_table(r2m, '{:.3f}')}\n\n"
            f"**Test MAE**\n\n{md_table(maem, '{:.4g}')}\n")
    with open(os.path.join(a.results, "summary.md"), "w") as fh:
        fh.write(text)
    print(text)

    # small multiples: one panel per feature set, shared axis, one hue
    lo = min(0.0, max(-1.0, float(np.nanmin(r2.values)) - 0.05))
    fig, axes = plt.subplots(1, len(sets), figsize=(2.3 * len(sets) + 1.2, 0.34 * len(targets) + 1.6),
                             sharey=True, squeeze=False)
    for ax, s in zip(axes[0], sets):
        vals = r2[s].values[::-1]
        ax.barh(range(len(targets)), np.clip(vals, lo, None), color=BLUE, height=0.62)
        for i, v in enumerate(vals):
            ax.text(max(v, 0) + 0.03, i, f"{v:.2f}", va="center", ha="left", fontsize=8, color=INK)
        ax.axvline(0, color="#c3c2b7", lw=1)
        ax.set_xlim(lo, 1.22)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks(range(len(targets)), targets[::-1])
        ax.set_title(s, loc="left", fontsize=10, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("test R²", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "summary.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print(f"-> {a.results}/summary.csv, summary.md, summary.png")


if __name__ == "__main__":
    main()
