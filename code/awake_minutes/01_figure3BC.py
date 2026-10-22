#!/usr/bin/env python3
"""Figure 3B-C: minutes of awake fMRI acquired per infant.

  Figure 3B  awake minutes per infant at 2 and 9 months (violins of infants with
             any awake data), with each infant's two visits joined:
               teal   scanned at both ages
               grey   came back at 9 months but no awake data were acquired (0 min)
               black  not scanned at 9 months
  Figure 3C  the same split by task (videos | pictures), per age.

Awake minutes = number of fMRI volumes in the infant's awake runs (pictures, videos)
x TR (0.61 s) / 60.

Input, either
  --bids-dir    the OpenNeuro dataset (ds006883). The number of volumes in each run is
                the number of framewise-displacement values + 1, from
                derivatives/foundcog_preproc/motion_fwd. The same infant has the same
                number at both ages (sub-2057 = sub-9057). Infants who were never
                scanned or came back without data are not in the dataset, so every
                infant here is teal or black.
  --runs-table  TSV with infant, age (2M/9M), task, n_volumes (one row per run, or per
                infant x task). A row with n_volumes 0 (task may be empty) marks an
                infant who came for that visit but has no awake data.

Outputs (outputs/awake_minutes/figure3BC_openneuro_v<version>/, or --out):
  figure3B_awake_minutes.png, figure3C_awake_minutes_by_task.png
  scan_time_summary.tsv       per age (3B) and age x task (3C): infants, infants with
                              awake data, total hours, median and quartiles of minutes
                              (infants with awake data only)
  scan_time_categories.tsv    infants per age and category (one per infant, from both visits):
                                data_2M_and_9M          awake data at both ages (teal)
                                no_data_2M_data_9M      awake data at 9 months only (teal)
                                data_2M_no_data_9M      came back at 9 months, no awake data
                                                        acquired (grey)
                                data_2M_no_return_9M    no 9-month visit (black)
                                no_data_2M_no_return_9M came at 2 months, no awake data, no
                                                        9-month visit (black)
  scan_time_correlation.tsv   Pearson r between 2- and 9-month minutes, infants scanned
                              at 2 months who came back (teal + grey)
  minutes_per_infant.tsv      (unless --aggregate-only) one row per infant x age x task

This reproduces time_catplot.py and time_catplot_bytask.py from the original FOUNDCOG
analysis.

Usage (from the repository root):
  python code/awake_minutes/01_figure3BC.py --bids-dir /path/to/ds006883
"""
import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "motion"))
fig3d = importlib.import_module("01_figure3D")  # FD file reading and OpenNeuro release

TR_S = 0.61
AGES = ["2M", "9M"]
AGE_LABELS = {"2M": "2-months", "9M": "9-months"}
AGE_COLORS = {"2M": plt.get_cmap("Set2")(0 / 7), "9M": "teal"}
NINE_MONTH_COLORS = {"data_9M": "teal", "no_data_9M": "lightslategrey", "no_return_9M": "black"}
TASKS = ["videos", "pictures"]
TASK_COLORS = {"videos": plt.get_cmap("Blues")(0.6), "pictures": plt.get_cmap("Purples")(0.6)}
# Violin widths as in the original figures: each violin's peak at full width, then (3C)
# narrowed within each age in proportion to its number of infants
NORM = {"density_norm": "width"} if tuple(map(int, sns.__version__.split(".")[:2])) >= (0, 13) else {"scale": "width"}


# ---------------------------------------------------------------- data

def runs_from_bids(bids_dir):
    rows = []
    for r in fig3d.read_fd_dir(bids_dir / fig3d.FD_SUBDIR, asleep_tasks=[]):
        if r["state"] == "awake":
            rows.append({"infant": r["sub"][5:], "age": r["age"], "task": r["task"], "n_volumes": len(r["fd"]) + 1})
    return pd.DataFrame(rows)


def per_infant(runs):
    """Minutes per infant x age x task, and per infant x age (all awake runs)."""
    runs = runs.assign(task=runs["task"].fillna(""), minutes=runs["n_volumes"] * TR_S / 60)
    by_task = runs.groupby(["infant", "age", "task"], as_index=False)["minutes"].sum()
    total = runs.groupby(["infant", "age"], as_index=False)["minutes"].sum()
    # one category per infant, from its 2- and 9-month rows (a row with 0 min = visit without data)
    two = total[total["age"] == "2M"].set_index("infant")["minutes"]
    nine = total[total["age"] == "9M"].set_index("infant")["minutes"]

    def category(i):
        p2 = "data_2M" if two.get(i, 0) > 0 else "no_data_2M"
        p9 = "data_9M" if nine.get(i, 0) > 0 else "no_data_9M" if i in nine.index else "no_return_9M"
        return "data_2M_and_9M" if (p2, p9) == ("data_2M", "data_9M") else f"{p2}_{p9}"
    total["category"] = total["infant"].map(category)
    return total, by_task[by_task["task"] != ""]


def summarise(total, by_task, precise):
    rows = []
    for (age, task), g in list(total.assign(task="all_awake").groupby(["age", "task"])) + list(by_task.groupby(["age", "task"])):
        m = g["minutes"]
        pos = m[m > 0]
        rows.append({"figure": "3B" if task == "all_awake" else "3C", "age": age, "task": task,
                     "n_infants": len(g), "n_infants_with_awake_data": len(pos), "total_hours": m.sum() / 60,
                     "median_min": pos.median(), "q10_min": pos.quantile(0.10), "q25_min": pos.quantile(0.25),
                     "q75_min": pos.quantile(0.75)})
    s = pd.DataFrame(rows)
    s["age"] = pd.Categorical(s["age"], AGES, ordered=True)
    s = s.sort_values(["figure", "age", "task"])
    return s if precise else s.round({c: 2 for c in ["total_hours", "median_min", "q10_min", "q25_min", "q75_min"]})


def correlation(total):
    w = total[~total["category"].str.endswith("no_return_9M")].pivot(index="infant", columns="age", values="minutes").dropna()
    if len(w) < 3:
        return pd.DataFrame()
    r, p = pearsonr(w["2M"], w["9M"])
    slope, intercept = np.polyfit(w["2M"], w["9M"], 1)
    return pd.DataFrame([{"n_infants": len(w), "r": r, "r_squared": r ** 2, "p": p,
                          "slope": slope, "intercept": intercept}])


# ---------------------------------------------------------------- plots

def scale_by_count(ax, counts, split, width=0.8):
    """Violin widths as in the original figures (seaborn < 0.13, scale="count"): each violin
    is drawn with its peak at full width (density_norm="width"), then narrowed within each x
    position in proportion to its number of observations. counts: per hue level, in hue
    order; one list, or one list per x position. Call straight after sns.violinplot."""
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    k = counts.shape[1]
    factor = counts / counts.max(axis=1, keepdims=True)   # within each x position

    def f(tick, h):
        return factor[min(tick, len(factor) - 1) if len(factor) > 1 else 0, h]

    def locate(xs):
        """Hue index and centre line of the violin a shape (or quartile line) belongs to."""
        mid = (np.min(xs) + np.max(xs)) / 2
        tick = round(mid)
        if split:
            return (0 if mid < tick else 1), tick
        offsets = -width / 2 + width / k * (np.arange(k) + 0.5)
        h = int(np.argmin(np.abs(offsets - (mid - tick))))
        return h, tick + offsets[h]

    for pc in ax.collections:
        if not isinstance(pc, PolyCollection):
            continue
        for path in pc.get_paths():
            v = path.vertices
            if len(v) == 0:
                continue
            h, c = locate(v[:, 0])
            v[:, 0] = c + (v[:, 0] - c) * f(int(round(c)), h)
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        if len(x) != 2 or x[0] == x[1]:
            continue  # only the horizontal quartile lines
        h, c = locate(x)
        line.set_xdata(c + (x - c) * f(int(round(c)), h))


def style(ax, ylabel_step, label_every):
    ax.yaxis.set_major_locator(plt.MultipleLocator(ylabel_step))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}" if x % label_every == 0 else ""))
    ax.yaxis.grid(True, linestyle="--", which="both", color="darkgrey", alpha=0.7, zorder=0)


def style_violins(ax, outline_width):
    for ln in ax.lines:                      # quartiles dashed, median solid
        ln.set(linestyle="--", linewidth=1, color="darkslategray", alpha=0.8)
    for ln in ax.lines[1::3]:
        ln.set(linestyle="-", linewidth=1.5)
    for pc in ax.collections:                # outline in the fill colour, semi-transparent
        pc.set_edgecolor(pc.get_facecolor())
        pc.set_linewidth(outline_width)
        pc.set_alpha(0.5)


def finish(ax, path, legend=False):
    sns.despine(ax=ax)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.set_ylabel("Awake Minutes", fontsize=12)
    ax.set_xlabel("")
    ax.set_xticks(range(len(AGES)))
    ax.set_xticklabels([AGE_LABELS[a] for a in AGES], fontsize=12)
    if not legend and ax.legend_ is not None:
        ax.legend_.remove()
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)


def plot_3b(total, path, rng):
    fig, ax = plt.subplots(figsize=(4, 3))
    style(ax, 5, 10)
    d = total[total["minutes"] > 0]
    sns.violinplot(data=d, x="age", y="minutes", order=AGES, hue="age", hue_order=AGES, palette=AGE_COLORS,
                   linewidth=1, cut=0, inner="quart", dodge=False, ax=ax, **NORM)
    style_violins(ax, 1)
    # each infant's visits, jittered, joined if seen at both ages
    total = total.assign(x=total["age"].map({a: i for i, a in enumerate(AGES)}) + rng.uniform(-0.07, 0.07, len(total)))
    nine_part = total["category"].map(lambda c: "no_return_9M" if c.endswith("no_return_9M")
                                      else "no_data_9M" if c.endswith("no_data_9M") else "data_9M")
    for part, col in NINE_MONTH_COLORS.items():
        sub = total[nine_part == part]
        if part == "no_return_9M":
            ax.scatter(sub["x"], sub["minutes"], color=col, alpha=0.4, zorder=5, s=10)
            continue
        for _, g in sub.groupby("infant"):
            g = g.sort_values("age")
            if len(g) > 1:
                ax.plot(g["x"], g["minutes"], color=col, alpha=0.2, zorder=5, linewidth=0.7)
            ax.scatter(g["x"], g["minutes"], color=col, alpha=0.8, zorder=5, s=10)
    finish(ax, path)


def plot_3c(by_task, path):
    fig, ax = plt.subplots(figsize=(4, 3))
    style(ax, 2.5, 5)
    d = by_task[by_task["minutes"] > 0]
    tasks = [t for t in TASKS if t in set(d["task"])]
    sns.violinplot(data=d, x="age", y="minutes", order=AGES, hue="task", hue_order=tasks, palette=TASK_COLORS,
                   split=len(tasks) > 1, linewidth=1.2, cut=0, inner="quart", ax=ax, **NORM)
    n = d.groupby(["age", "task"]).size()
    scale_by_count(ax, [[n.get((age, t), 0) for t in tasks] for age in AGES], split=len(tasks) > 1)
    style_violins(ax, 1.5)
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [lab.capitalize() for lab in labels], title="Task", fontsize=10, title_fontsize=10,
              bbox_to_anchor=(1.13, 1.02))
    finish(ax, path, legend=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bids-dir", type=Path, help="root of the BIDS dataset (ds006883)")
    src.add_argument("--runs-table", type=Path, help="TSV: infant, age (2M/9M), task, n_volumes")
    ap.add_argument("--openneuro-version", help="OpenNeuro release of --bids-dir (default as in 01_figure3D.py)")
    ap.add_argument("--out", type=Path, help="output folder (default: outputs/awake_minutes/figure3BC_openneuro_v<version>; "
                    "required with --runs-table)")
    ap.add_argument("--aggregate-only", action="store_true", help="do not write minutes_per_infant.tsv")
    ap.add_argument("--precise", action="store_true", help="write the tables without rounding")
    a = ap.parse_args()

    if a.bids_dir:
        if not (a.bids_dir / fig3d.FD_SUBDIR).is_dir():
            raise SystemExit(f"{a.bids_dir / fig3d.FD_SUBDIR} not found: --bids-dir should be the dataset root")
        tag = fig3d.release_tag(a.bids_dir)
        if tag and a.openneuro_version and tag != a.openneuro_version:
            raise SystemExit(f"--openneuro-version {a.openneuro_version} but {a.bids_dir} is at release {tag}")
        version = a.openneuro_version or tag or fig3d.OPENNEURO_VERSION
        a.out = a.out or fig3d.REPO / "outputs" / "awake_minutes" / f"figure3BC_openneuro_v{version.replace('.', '-')}"
        runs = runs_from_bids(a.bids_dir)
    else:
        if a.out is None:
            raise SystemExit("--out is required with --runs-table")
        runs = pd.read_csv(a.runs_table, sep="\t", dtype={"infant": str, "task": str})
    a.out.mkdir(parents=True, exist_ok=True)

    total, by_task = per_infant(runs)
    summary = summarise(total, by_task, a.precise)
    summary.to_csv(a.out / "scan_time_summary.tsv", sep="\t", index=False)
    cats = total.groupby(["age", "category"]).size().rename("n_infants").reset_index()
    cats.to_csv(a.out / "scan_time_categories.tsv", sep="\t", index=False)
    corr = correlation(total)
    (corr if a.precise else corr.round(4)).to_csv(a.out / "scan_time_correlation.tsv", sep="\t", index=False)
    if not a.aggregate_only:
        by_task.round({"minutes": 3}).to_csv(a.out / "minutes_per_infant.tsv", sep="\t", index=False)

    plot_3b(total, a.out / "figure3B_awake_minutes.png", np.random.default_rng(42))
    plot_3c(by_task, a.out / "figure3C_awake_minutes_by_task.png")

    print(summary.to_string(index=False))
    print(cats.to_string(index=False))
    if len(corr):
        print(corr.round(4).to_string(index=False))
    print(f"Outputs -> {a.out}")


if __name__ == "__main__":
    main()
