#!/usr/bin/env python3
"""Figure 3E-F and Supplementary Figure 4 A-F: how the motion thresholds affect
how much awake data is kept.

Uses the same framewise displacement (FD) input as 01_figure3D.py (--bids-dir or
--runs-table), awake runs only (pictures and videos), per age and task.

  Figure 3E / 3F   frames censored per run (%), FD >= 1.5 mm, for every awake run;
                   violin + runs, videos vs pictures; 3E = 2 months, 3F = 9 months.
  Supp. Fig. 4 A-B runs included (%) as the run inclusion threshold (median FD <
                   threshold) varies from 0.25 to 3 mm; A = videos, B = pictures,
                   one line per age.
  Supp. Fig. 4 C-F frames censored within included runs (%), for each run inclusion
                   threshold (x) and frame censoring threshold (lines; frames with
                   FD >= threshold are censored); C-D = 2 months, E-F = 9 months
                   (videos, pictures).

The chosen thresholds (1.5 mm for both) are marked. Run types not in the data
(e.g. videos in OpenNeuro ds006883 v1.0.1) are left out of the figures.

With --runs-table, a 'fmap_same_session' column (1/0), if present, keeps only runs
with field maps from the same session, as in the original analysis. The shared
runs all have them.

Outputs (outputs/motion/figure3EF_S4_openneuro_v<version>/, or --out):
  figure3E_censoring_per_run_2M.png, figure3F_censoring_per_run_9M.png
  figureS4AB_runs_included.png, figureS4CD_frames_censored_2M.png, figureS4EF_frames_censored_9M.png
  threshold_sensitivity.tsv   per age x task x run threshold x frame threshold:
                              runs total / included, % included, frames in included
                              runs, frames censored, % censored
  censoring_per_run.tsv       per run: age, task, % frames censored at 1.5 mm
                              (plus run_name unless --aggregate-only)

This reproduces motion_threshold_sensitivity_v14.py from the original FOUNDCOG
analysis.

Usage (from the repository root):
  python code/motion/02_figure3EF_S4.py --bids-dir /path/to/ds006883
"""
import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
fig3d = importlib.import_module("01_figure3D")  # shared FD reading and OpenNeuro release

THRESHOLDS = np.round(np.arange(0.25, 3.01, 0.25), 2)  # 0.25 ... 3.00 mm
CHOSEN = fig3d.THRESHOLD  # 1.5 mm, for both run inclusion and frame censoring
TASKS = ["videos", "pictures"]
TASK_LABELS = {"videos": "Videos", "pictures": "Pictures"}
AGES = ["2M", "9M"]
AGE_LABELS = {"2M": "2 months", "9M": "9 months"}
AGE_COLORS = {"2M": plt.get_cmap("Set2")(0 / 7), "9M": "teal"}
GUIDE_COLORS = {"2M": "#BB8FB7", "9M": "#78206E"}
TASK_COLORS = {"videos": "#ADC9DB", "pictures": "#AEADCB"}


# ---------------------------------------------------------------- data

def awake_runs(runs):
    """Awake runs as a table: age, task, run_name, fd (array), median FD."""
    rows = []
    for r in runs:
        if r["state"] != "awake":
            continue
        if "fmap_same_session" in r and not bool(int(r["fmap_same_session"])):
            continue
        fd = r["fd"][np.isfinite(r["fd"])]
        if fd.size == 0:
            continue
        rows.append({"age": r["age"], "task": r["task"], "run_name": r["run_name"],
                     "fd": fd, "median_fd": float(np.median(fd))})
    return pd.DataFrame(rows)


def threshold_sensitivity(df):
    rows = []
    for (age, task), g in df.groupby(["age", "task"], sort=False):
        for run_thr in THRESHOLDS:
            kept = g[g["median_fd"] < run_thr]
            frames = np.concatenate(kept["fd"].to_list()) if len(kept) else np.array([])
            for frame_thr in THRESHOLDS:
                n_cens = int(np.count_nonzero(frames >= frame_thr))
                rows.append({"age": age, "task": task,
                             "run_median_threshold_mm": run_thr, "frame_censor_threshold_mm": frame_thr,
                             "n_total_runs": len(g), "n_included_runs": len(kept),
                             "pct_runs_included": 100 * len(kept) / len(g),
                             "n_frames_in_included_runs": frames.size, "n_frames_censored": n_cens,
                             "pct_frames_censored": 100 * n_cens / frames.size if frames.size else np.nan})
    return pd.DataFrame(rows)


def censoring_per_run(df):
    return df.assign(pct_frames_censored=[100 * np.mean(fd >= CHOSEN) for fd in df["fd"]])[
        ["age", "task", "run_name", "pct_frames_censored"]]


# ---------------------------------------------------------------- plots

def style_axes(ax):
    sns.despine(ax=ax)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.set_axisbelow(True)


def l_guide(ax, x, y, color):
    """Dashed L from the x-axis up to the point and across to the y-axis, with a star."""
    x0 = 0.20  # left edge of the x-axis (set in threshold_axes)
    ax.plot([x, x], [0, y], ls="--", color=color, lw=1.6, zorder=4)
    ax.plot([x0, x], [y, y], ls="--", color=color, lw=1.6, zorder=4)
    ax.plot(x, y, marker="*", markersize=9, color=color, markeredgecolor="none", ls="None", zorder=6)


def threshold_axes(axes, ylabel):
    for ax in axes:
        ax.set_xlim(0.20, 3.05)
        ax.set_xticks(THRESHOLDS)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Run median-FWD inclusion threshold (mm)")
        style_axes(ax)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
        ax.yaxis.grid(True, ls="--", which="both", color="lightgrey", alpha=0.7, zorder=0)
        ax.xaxis.grid(True, ls="--", which="major", color="lightgrey", alpha=0.7, zorder=0)
    axes[0].set_ylabel(ylabel)


def plot_runs_included(stats, tasks, path):
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks) + 1.5, 4), sharey=True, squeeze=False)
    axes = axes[0]
    runs = stats.drop_duplicates(["age", "task", "run_median_threshold_mm"])
    for ax, task in zip(axes, tasks):
        for age in AGES:
            d = runs[(runs["task"] == task) & (runs["age"] == age)].sort_values("run_median_threshold_mm")
            if d.empty:
                continue
            ax.plot(d["run_median_threshold_mm"], d["pct_runs_included"], marker="o", markersize=4,
                    lw=2, alpha=0.8, color=AGE_COLORS[age], label=AGE_LABELS[age])
            h = d[np.isclose(d["run_median_threshold_mm"], CHOSEN)]
            if len(h):
                l_guide(ax, CHOSEN, float(h["pct_runs_included"].iloc[0]), GUIDE_COLORS[age])
    threshold_axes(axes, "Runs included (%)")
    axes[-1].legend(title="Age", bbox_to_anchor=(1.03, 1), loc="upper left", fontsize=9, title_fontsize=10)
    handles = [Line2D([0], [0], color=GUIDE_COLORS[a], lw=1.6, marker="*", markersize=10,
                      markeredgecolor="none", label=a.replace("M", "-months")) for a in AGES]
    fig.legend(title="Chosen threshold", title_fontsize=10, handles=handles, loc="upper right",
               bbox_to_anchor=(1.02, 0.73), fontsize=9, frameon=True)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_frames_censored(stats, age, tasks, path):
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks) + 1.5, 4), sharey=True, squeeze=False)
    axes = axes[0]
    d_age = stats[stats["age"] == age]
    colors = dict(zip(THRESHOLDS, plt.get_cmap("Spectral")(np.linspace(0.05, 0.95, len(THRESHOLDS)))))
    markers = dict(zip(THRESHOLDS, ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "d"]))
    for ax, task in zip(axes, tasks):
        d = d_age[d_age["task"] == task]
        for thr, line in d.groupby("frame_censor_threshold_mm"):
            line = line.sort_values("run_median_threshold_mm")
            ax.plot(line["run_median_threshold_mm"], line["pct_frames_censored"], label=f"{thr:.2f}",
                    lw=1.8, marker=markers[thr], markersize=4, color=colors[thr])
        h = d[np.isclose(d["frame_censor_threshold_mm"], CHOSEN) & np.isclose(d["run_median_threshold_mm"], CHOSEN)]
        if len(h):
            l_guide(ax, CHOSEN, float(h["pct_frames_censored"].iloc[0]), GUIDE_COLORS[age])
    threshold_axes(axes, "Frames censored within included runs (%)")
    axes[-1].legend(title="Frame censor\nthreshold (mm)", bbox_to_anchor=(1.03, 1), loc="upper left",
                    fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_censoring_violins(per_run, age, tasks, path, rng):
    d = per_run[per_run["age"] == age]
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(tasks), 3))  # 4 x 3 in for videos + pictures
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.yaxis.grid(True, ls="--", which="both", color="darkgrey", alpha=0.7, zorder=0)
    # each violin's peak at full width, as in the original figure (one task per x position)
    norm = {"density_norm": "width"} if tuple(map(int, sns.__version__.split(".")[:2])) >= (0, 13) else {"scale": "width"}
    sns.violinplot(data=d, x="task", y="pct_frames_censored", order=tasks, hue="task", palette=TASK_COLORS,
                   linewidth=0, cut=0, inner="quart", dodge=False, ax=ax, **norm)
    for ln in ax.lines:                      # quartiles dashed ...
        ln.set(linestyle="--", linewidth=1, color="darkslategray", alpha=0.8)
    for ln in ax.lines[1::3]:                # ... median solid
        ln.set(linestyle="-", linewidth=1.5)
    for pc in ax.collections[:len(tasks)]:
        pc.set_edgecolor("none")
        pc.set_alpha(0.35)
    for x, task in enumerate(tasks):
        y = d.loc[d["task"] == task, "pct_frames_censored"]
        ax.scatter(x + rng.uniform(-0.11, 0.11, len(y)), y, color=TASK_COLORS[task], alpha=0.45,
                   zorder=5, s=12, edgecolors="none")
    ax.axhline(50, color="darkslategray", ls="--", lw=1, alpha=0.8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("Frames censored per run (%)", fontsize=12)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=12)
    if ax.legend_ is not None:
        ax.legend_.remove()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bids-dir", type=Path, help="root of the BIDS dataset (ds006883)")
    src.add_argument("--runs-table", type=Path, help="alternative input table (see 01_figure3D.py)")
    ap.add_argument("--openneuro-version", help="OpenNeuro release of --bids-dir (default as in 01_figure3D.py)")
    ap.add_argument("--out", type=Path, help="output folder (default: outputs/motion/figure3EF_S4_openneuro_v<version>; "
                    "required with --runs-table)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="leave run names out of censoring_per_run.tsv (for data whose run names must not be shared)")
    ap.add_argument("--precise", action="store_true", help="write the tables without rounding")
    a = ap.parse_args()

    if a.bids_dir:
        fd_dir = a.bids_dir / fig3d.FD_SUBDIR
        if not fd_dir.is_dir():
            raise SystemExit(f"{fd_dir} not found: --bids-dir should be the dataset root")
        tag = fig3d.release_tag(a.bids_dir)
        if tag and a.openneuro_version and tag != a.openneuro_version:
            raise SystemExit(f"--openneuro-version {a.openneuro_version} but {a.bids_dir} is at release {tag}")
        version = a.openneuro_version or tag or fig3d.OPENNEURO_VERSION
        a.out = a.out or fig3d.OUT_DIR / f"figure3EF_S4_openneuro_v{version.replace('.', '-')}"
        runs = fig3d.read_fd_dir(fd_dir, asleep_tasks=[])
    else:
        if a.out is None:
            raise SystemExit("--out is required with --runs-table")
        runs = fig3d.read_runs_table(a.runs_table)
    a.out.mkdir(parents=True, exist_ok=True)

    df = awake_runs(runs)
    tasks = [t for t in TASKS if t in set(df["task"])]
    stats = threshold_sensitivity(df)
    per_run = censoring_per_run(df)

    stats_out = stats if a.precise else stats.round({"pct_runs_included": 2, "pct_frames_censored": 2})
    stats_out.to_csv(a.out / "threshold_sensitivity.tsv", sep="\t", index=False)
    pr_out = per_run.drop(columns="run_name") if a.aggregate_only else per_run
    (pr_out if a.precise else pr_out.round({"pct_frames_censored": 2})).to_csv(
        a.out / "censoring_per_run.tsv", sep="\t", index=False)

    rng = np.random.default_rng(42)
    ages = [g for g in AGES if g in set(df["age"])]
    for panel, age in [("3E", "2M"), ("3F", "9M")]:
        if age in ages:
            plot_censoring_violins(per_run, age, tasks, a.out / f"figure{panel}_censoring_per_run_{age}.png", rng)
    plot_runs_included(stats, tasks, a.out / "figureS4AB_runs_included.png")
    for panel, age in [("S4CD", "2M"), ("S4EF", "9M")]:
        if age in ages:
            plot_frames_censored(stats, age, tasks, a.out / f"figure{panel}_frames_censored_{age}.png")

    chosen = stats[np.isclose(stats["run_median_threshold_mm"], CHOSEN) & np.isclose(stats["frame_censor_threshold_mm"], CHOSEN)]
    print(chosen[["age", "task", "n_total_runs", "n_included_runs", "pct_runs_included", "pct_frames_censored"]]
          .round(2).to_string(index=False))
    print(f"Tasks found: {tasks}. Outputs -> {a.out}")


if __name__ == "__main__":
    main()
