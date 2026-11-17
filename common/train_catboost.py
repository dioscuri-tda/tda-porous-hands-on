r"""
Train CatBoost on topological descriptors - deliberately the simplest setup.

* one fixed train / validation / test split (default 60 / 20 / 20 %, `--seed`);
* one single-target CatBoostRegressor per target (constant columns are dropped,
  and for wide tables only a random subset of the features is tried per split);
* the validation set is used only for early stopping (best iteration);
* metrics (R^2, MAE, RMSE) are computed on the test set, which the model never saw.

The split depends only on `--seed` and on the structure names, so different
feature sets are compared on exactly the same test structures.

Feature selection
-----------------
`--features` takes group names. A group is either a column prefix (`ph` selects
every column starting with `ph_`, `ecp` every `ecp_...`, `thr` every `thr_...`)
or an exact column name (`porosity`, `solid_fraction`). `--directions`
optionally keeps only descriptor columns of the given direction tags
(e.g. `--directions 1_0_0 0_1_0 0_0_1`).

Outputs, in `<outdir>/<label>/`
-------------------------------
    metrics.csv        one row per target: r2, mae, rmse, best_iteration, sizes
    predictions.csv    name, split, <target>_true, <target>_pred (all structures)
    parity.png         predicted vs. true on the test set, one panel per target
    importance.png     share of CatBoost feature importance per descriptor group
                       and direction, one row per target
    config.json        the arguments of the run

Usage
-----
    python common/train_catboost.py --dataset dataset.csv \
        --targets C_xx_xx C_yy_yy C_zz_zz --features solid_fraction ph \
        --label ph+porosity --outdir results
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
})

DIRECTION_RE = re.compile(r"^([a-z]+)_(-?\d+_-?\d+_-?\d+)_\d+$")


def select_features(columns, groups, directions):
    """Columns of the requested groups (prefix `g_` or exact name), in table order."""
    out = []
    for c in columns:
        m = DIRECTION_RE.match(c)
        for g in groups:
            if c == g:
                out.append(c)
            elif m and m.group(1) == g and (not directions or m.group(2) in directions):
                out.append(c)
    return list(dict.fromkeys(out))


def make_split(names, seed, val_frac, test_frac):
    """Deterministic split of the (sorted) structure names -> dict name: split."""
    names = sorted(names)
    rest, test = train_test_split(names, test_size=test_frac, random_state=seed)
    train, val = train_test_split(rest, test_size=val_frac / (1.0 - test_frac), random_state=seed)
    return {**{n: "train" for n in train}, **{n: "val" for n in val}, **{n: "test" for n in test}}


def importance_groups(features, importance):
    """Sum the feature importance per (descriptor group, direction) -> Series in %."""
    keys = []
    for f in features:
        m = DIRECTION_RE.match(f)
        keys.append(f"{m.group(1)} [{m.group(2).replace('_', ' ')}]" if m else f)
    s = pd.Series(importance, index=keys).groupby(level=0, sort=False).sum()
    return 100.0 * s / max(s.sum(), 1e-12)


def plot_parity(pred, targets, metrics, path, title):
    n = len(targets)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow), squeeze=False)
    test = pred[pred.split == "test"]
    for ax, t in zip(axes.ravel(), targets):
        y, p = test[f"{t}_true"], test[f"{t}_pred"]
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#c3c2b7", lw=1, zorder=1)
        ax.scatter(y, p, s=28, color=BLUE, edgecolor="white", linewidth=0.8, zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        m = metrics.loc[t]
        ax.set_title(f"{t}   R² = {m.r2:.2f}   MAE = {m.mae:.3g}", loc="left")
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{title} — test set ({len(test)} structures)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp, path, title):
    """imp: DataFrame targets x (group, direction), in %; sequential single-hue heatmap."""
    fig, ax = plt.subplots(figsize=(max(5.0, 0.42 * imp.shape[1] + 2.2), 0.42 * imp.shape[0] + 2.4))
    im = ax.imshow(imp.values, cmap="Blues", vmin=0, vmax=max(imp.values.max(), 1e-9), aspect="auto")
    ax.set_xticks(range(imp.shape[1]), imp.columns, rotation=60, ha="right")
    ax.set_yticks(range(imp.shape[0]), imp.index)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("share of feature importance [%]")
    cb.outline.set_visible(False)
    ax.set_title(f"{title} — where the model looks", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="table written by build_dataset.py")
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--features", nargs="+", required=True,
                   help="feature groups: column prefixes (ph, ecp, thr) or exact columns")
    p.add_argument("--directions", nargs="+", default=[],
                   help="keep only these direction tags, e.g. 1_0_0 (default: all)")
    p.add_argument("--label", default=None, help="name of the run (default: features joined by +)")
    p.add_argument("--outdir", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--early-stopping", type=int, default=100)
    p.add_argument("--rsm", type=float, default=None,
                   help="fraction of features tried at each split (CatBoost `rsm`). "
                        "Default: 1 up to 500 features, otherwise 500 / n_features - "
                        "the descriptors are wide (thousands of columns) and highly "
                        "redundant, and this makes training many times faster")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    a = p.parse_args()

    label = a.label or "+".join(a.features)
    out = os.path.join(a.outdir, label)
    os.makedirs(out, exist_ok=True)

    df = pd.read_csv(a.dataset)
    missing = [t for t in a.targets if t not in df.columns]
    if missing:
        sys.exit(f"targets not in the dataset: {missing}")
    features = select_features(df.columns, a.features, set(a.directions))
    absent = [g for g in a.features if not select_features(df.columns, [g], set(a.directions))]
    if absent:
        sys.exit(f"no columns for feature group(s) {absent} in {a.dataset}\n"
                 f"  (the 'ecp' group needs the Julia step; rerun without --no-julia "
                 f"or build the dataset with --reference)")

    # constant columns (e.g. empty corners of the persistence images) carry nothing
    features = [c for c in features if df[c].nunique() > 1]
    rsm = a.rsm if a.rsm is not None else min(1.0, 500.0 / len(features))

    split = df.name.map(make_split(df.name, a.seed, a.val_frac, a.test_frac))
    tr, va, te = (split == "train").values, (split == "val").values, (split == "test").values
    X = df[features]
    print(f"[{label}] {len(features)} features, {len(a.targets)} targets, "
          f"train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}")

    pred = pd.DataFrame({"name": df.name, "split": split})
    rows, imps = [], {}
    for t in a.targets:
        y = df[t].values
        model = CatBoostRegressor(
            loss_function="RMSE", iterations=a.iterations, learning_rate=a.learning_rate,
            depth=a.depth, rsm=rsm, random_seed=a.seed, thread_count=a.threads, verbose=0,
            allow_writing_files=False)
        model.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True,
                  early_stopping_rounds=a.early_stopping)
        yhat = model.predict(X)
        pred[f"{t}_true"], pred[f"{t}_pred"] = y, yhat
        row = {"target": t, "features": label,
               "r2": r2_score(y[te], yhat[te]),
               "mae": mean_absolute_error(y[te], yhat[te]),
               "rmse": float(np.sqrt(np.mean((y[te] - yhat[te]) ** 2))),
               "r2_val": r2_score(y[va], yhat[va]),
               "best_iteration": model.get_best_iteration(),
               "n_features": len(features), "n_train": int(tr.sum()),
               "n_val": int(va.sum()), "n_test": int(te.sum())}
        rows.append(row)
        imps[t] = importance_groups(features, model.get_feature_importance())
        print(f"    {t:10s} test R2 = {row['r2']:6.3f}   MAE = {row['mae']:.4g}   "
              f"(best iteration {row['best_iteration']})")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(out, "metrics.csv"), index=False, float_format="%.6g")
    pred.to_csv(os.path.join(out, "predictions.csv"), index=False, float_format="%.6g")
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump({**vars(a), "label": label, "rsm": rsm, "n_features": len(features)}, fh, indent=2)

    plot_parity(pred, a.targets, metrics.set_index("target"), os.path.join(out, "parity.png"), label)
    imp = pd.DataFrame(imps).T.fillna(0.0)
    if imp.shape[1] > 1:
        plot_importance(imp, os.path.join(out, "importance.png"), label)
    print(f"    mean test R2 = {metrics.r2.mean():.3f}  -> {out}/")


if __name__ == "__main__":
    main()
