#!/usr/bin/env python3
"""Figure 3D: head motion (framewise displacement, FD) during awake and asleep scans.

Reads per-volume FD (Power et al. 2012) from the BIDS dataset shared on OpenNeuro
(ds006883), given as --bids-dir (the dataset root):

  <bids-dir>/derivatives/foundcog_preproc/motion_fwd/sub-*/sub-*_ses-*_task-*_run-*_fd_power_2012.txt

Each file has a header line, then one FD value (mm) per volume from the second
volume on. Every task present is used, so the figure fills in as more of the
dataset is released.

Groups, per age (2 and 9 months, from the participant ID: sub-2xxx / sub-9xxx):
  All awake       every awake run (tasks: pictures, videos)
  Included awake  awake runs with median FD < 1.5 mm (the inclusion threshold)
  Asleep          resting-state runs (tasks rest5 and rest10; see --asleep-tasks)

Outputs go to outputs/motion/figure3D_openneuro_v<version>/ (e.g. v1-0-1), named after the
OpenNeuro release of the data: --openneuro-version if given, else the dataset's git tag
if it is a DataLad/git clone, else OPENNEURO_VERSION below (the current release).
--out overrides the folder.
  figure3D_fd_raincloud.png   FD of every frame, by age and group
  fd_per_run.tsv              (unless --aggregate-only) one row per run: median FD, % frames above threshold, included
  fd_summary.tsv              per age x group x task: runs, infants, median and IQR of
                              run-median FD (frame-level for asleep runs), % frames censored

This reproduces plot_awake_motion_attempt_overlap_includes_fmaps_v10_by_cohort.py
from the original FOUNDCOG analysis. The published figure also includes runs
that are not shared (and, for 'All awake', requires field maps for every run,
which all shared runs have).

Usage (from the repository root):
  python code/motion/01_figure3D.py --bids-dir /path/to/ds006883
"""
import argparse
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs" / "motion"
OPENNEURO_VERSION = "1.0.1"  # current ds006883 release - update with each new release
FD_SUBDIR = Path("derivatives") / "foundcog_preproc" / "motion_fwd"
THRESHOLD = 1.5  # mm: runs with median FD below this are included; frames above it are censored
AWAKE_TASKS = ["pictures", "videos"]
AGES = {"2M": "2-months", "9M": "9-months"}
FNAME = re.compile(r"^(?P<sub>sub-[A-Za-z0-9]+)_(?P<ses>ses-\w+?)_task-(?P<task>\w+?)_(?:dir-\w+?_)?run-(?P<run>\d+)_fd_power_2012\.txt$")


def lighten(color, amount):
    """Mix a colour with white (amount = share of white)."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


COLORS = {
    ("2M", "all"): lighten(plt.get_cmap("Set2")(0), 0.4),
    ("2M", "included"): plt.get_cmap("Set2")(0),
    ("2M", "asleep"): "lightgray",
    ("9M", "all"): lighten("teal", 0.1),
    ("9M", "included"): "teal",
    ("9M", "asleep"): "darkgray",
}
GROUP_LABELS = {"all": "All awake", "included": "Included awake", "asleep": "Asleep"}


# ---------------------------------------------------------------- data

def read_fd_dir(fd_dir, asleep_tasks):
    runs = []
    for f in sorted(Path(fd_dir).glob("sub-*/*_fd_power_2012.txt")):
        m = FNAME.match(f.name)
        if not m:
            continue
        task = m.group("task")
        state = "awake" if task in AWAKE_TASKS else "asleep" if task in asleep_tasks else None
        if state is None:
            continue
        fd = pd.read_csv(f, sep=r"\s+")["FramewiseDisplacement"].to_numpy(dtype=float)
        fd = fd[~np.isnan(fd)]
        sub = m.group("sub")
        runs.append({"run_name": f.name.replace("_fd_power_2012.txt", ""), "sub": sub,
                     "age": {"2": "2M", "9": "9M"}.get(sub[4], "?"), "task": task, "state": state, "fd": fd})
    if not runs:
        raise SystemExit(f"No FD files found in {fd_dir}")
    return runs


def read_runs_table(path):
    """Alternative input: TSV with run_name, sub, age (2M/9M), task, state (awake/asleep),
    fd (FD values separated by ';')."""
    df = pd.read_csv(path, sep="\t")
    return [dict(r, fd=np.array([float(v) for v in str(r["fd"]).split(";") if v != ""]))
            for r in df.to_dict("records")]


def per_run_table(runs):
    rows = []
    for r in runs:
        med = float(np.median(r["fd"])) if len(r["fd"]) else np.nan
        rows.append({"run_name": r["run_name"], "age": r["age"], "task": r["task"], "state": r["state"],
                     "n_frames": len(r["fd"]), "median_fd_mm": med,
                     "pct_frames_above_threshold": float((r["fd"] >= THRESHOLD).mean() * 100) if len(r["fd"]) else np.nan,
                     "included": int(r["state"] == "awake" and med < THRESHOLD)})
    return pd.DataFrame(rows)


def frames_long(runs, per_run):
    """One row per frame and group. Included runs appear in both 'all' and 'included'."""
    parts = []
    for r, included in zip(runs, per_run["included"]):  # same order: matched by position, not name
        groups = ["asleep"] if r["state"] == "asleep" else (["all", "included"] if included else ["all"])
        for g in groups:
            parts.append(pd.DataFrame({"fd": r["fd"], "age": r["age"], "group": g, "task": r["task"],
                                       "sub": r["sub"], "run_name": r["run_name"]}))
    return pd.concat(parts, ignore_index=True)


def summary(runs, per_run, precise=False):
    fds = [r["fd"] for r in runs]  # matched to per_run rows by position, not name
    pr = per_run.assign(sub=[r["sub"] for r in runs])
    out = []
    for age in AGES:
        for group in ["all", "included", "asleep"]:
            for task in ["all_tasks"] + (AWAKE_TASKS if group != "asleep" else []):
                g = pr[pr["age"] == age]
                g = g[g["state"] == "asleep"] if group == "asleep" else g[g["state"] == "awake"]
                if group == "included":
                    g = g[g["included"] == 1]
                if task != "all_tasks":
                    g = g[g["task"] == task]
                if g.empty:
                    continue
                frames = np.concatenate([fds[i] for i in g.index])
                # awake: distribution of run medians; asleep: distribution of frames (as in the original)
                vals = frames if group == "asleep" else g["median_fd_mm"].to_numpy()
                out.append({"age": age, "group": GROUP_LABELS[group], "task": task, "n_runs": len(g),
                            "n_infants": g["sub"].nunique(), "summary_of": "frames" if group == "asleep" else "run medians",
                            "median_fd_mm": float(np.median(vals)),
                            "q1_mm": float(np.quantile(vals, 0.25)),
                            "q3_mm": float(np.quantile(vals, 0.75)),
                            "pct_frames_censored": float((frames >= THRESHOLD).mean() * 100)})
    out = pd.DataFrame(out)
    if not precise and not out.empty:
        out = out.round({"median_fd_mm": 3, "q1_mm": 3, "q3_mm": 3, "pct_frames_censored": 2})
    return out


# ---------------------------------------------------------------- plot

def raincloud(fr, path, rng):
    """Horizontal raincloud: for each age, overlapping half-violins above a baseline, and a
    box plot + scattered frames per group below it. Violin height is proportional to the number
    of frames, relative to the largest group of the same age (seaborn's density_norm='count',
    one plot per age)."""
    groups = [g for g in ["all", "included", "asleep"] if g in fr["group"].unique()]
    ages = [a for a in AGES if a in fr["age"].unique()]
    xs = np.linspace(0, 5, 500)
    band = 1.0
    fig, ax = plt.subplots(figsize=(6, 8))
    for ai, age in enumerate(ages):
        base = -ai * band * 2.2
        data = {g: fr[(fr["age"] == age) & (fr["group"] == g)]["fd"].to_numpy() for g in groups}
        n_max = max(len(d) for d in data.values())  # largest group of THIS age fills the band
        for gi, g in enumerate(groups):
            d = data[g]
            if len(d) < 2:
                continue
            col = COLORS[(age, g)]
            dens = gaussian_kde(d, bw_method=0.6)(xs)  # seaborn-style bw scale factor
            # as seaborn density_norm='count': peak height ~ number of frames, relative within age
            h = dens / dens.max() * 0.9 * band * len(d) / n_max
            ax.fill_between(xs, base, base + h, color=col, alpha=0.4, lw=0, zorder=2)
            ax.plot(xs, base + h, color="dimgray", lw=0.8, alpha=0.8, zorder=3)
            y = base - 0.25 - gi * 0.3
            jit = rng.uniform(-0.1, 0.1, len(d))
            ax.scatter(d, y + jit, s=1.5, color=col, alpha=0.05, lw=0, zorder=1)
            box = dict(positions=[y], widths=0.18, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=col, edgecolor="black", lw=1),
                       medianprops=dict(color="black", lw=1), whiskerprops=dict(lw=1),
                       capprops=dict(lw=1), manage_ticks=False, zorder=4)
            try:
                ax.boxplot(d, orientation="horizontal", **box)  # matplotlib >= 3.10
            except TypeError:
                ax.boxplot(d, vert=False, **box)
        handles = [Patch(facecolor=COLORS[(age, g)], edgecolor="gray", alpha=0.8) for g in groups]
        leg = ax.legend(handles, [GROUP_LABELS[g] for g in groups], title=AGES[age], loc="upper right",
                        bbox_to_anchor=(1.0, 1 - ai * 0.45), fontsize=12, title_fontsize=12)
        ax.add_artist(leg)
    ax.set_xlim(0, 5)
    ax.set_ylim(-(len(ages) - 1) * band * 2.2 - 1.2, band * 1.0)
    ax.axvline(THRESHOLD, color="slategrey", ls="--", alpha=0.9)
    ax.set_xlabel("Framewise Displacement (mm)", fontsize=18)
    ax.set_yticks([])
    sns.despine(ax=ax)
    ax.spines["left"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.tick_params(axis="x", labelsize=13)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.grid(which="both", axis="x", alpha=0.3, ls="--", lw=1, zorder=-10)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def release_tag(bids_dir):
    """OpenNeuro release of a DataLad/git clone (its tag, e.g. '1.0.1'), or None."""
    try:
        r = subprocess.run(["git", "-C", str(bids_dir), "describe", "--tags", "--exact-match"],
                           capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return r.stdout.strip() or None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bids-dir", type=Path, help="root of the BIDS dataset (ds006883)")
    src.add_argument("--runs-table", type=Path, help="alternative input table (see read_runs_table)")
    ap.add_argument("--asleep-tasks", nargs="*", default=["rest5", "rest10"],
                    help="task names of asleep (resting-state) runs; all of them are used")
    ap.add_argument("--openneuro-version", help=f"OpenNeuro release of --bids-dir (default: the dataset's git tag if any, "
                    f"else {OPENNEURO_VERSION})")
    ap.add_argument("--out", type=Path, help="output folder (default: outputs/motion/figure3D_openneuro_v<version>; "
                    "required with --runs-table)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="do not write fd_per_run.tsv (for data whose run names must not be shared)")
    ap.add_argument("--precise", action="store_true", help="write the summary without rounding")
    a = ap.parse_args()

    if a.bids_dir:
        fd_dir = a.bids_dir / FD_SUBDIR
        if not fd_dir.is_dir():
            raise SystemExit(f"{fd_dir} not found: --bids-dir should be the dataset root "
                             f"(the folder containing derivatives/)")
        tag = release_tag(a.bids_dir)
        if tag and a.openneuro_version and tag != a.openneuro_version:
            raise SystemExit(f"--openneuro-version {a.openneuro_version} but {a.bids_dir} is at release {tag}")
        version = a.openneuro_version or tag or OPENNEURO_VERSION
        if a.out is None:
            a.out = OUT_DIR / f"figure3D_openneuro_v{version.replace('.', '-')}"
        runs = read_fd_dir(fd_dir, a.asleep_tasks)
    else:
        if a.out is None:
            raise SystemExit("--out is required with --runs-table")
        runs = read_runs_table(a.runs_table)
    a.out.mkdir(parents=True, exist_ok=True)
    per_run = per_run_table(runs)
    dup = per_run["run_name"][per_run["run_name"].duplicated()].unique()
    if len(dup):
        print(f"Note: {len(dup)} run names occur more than once (each run is still counted separately): "
              f"{', '.join(dup[:5])}{' ...' if len(dup) > 5 else ''}")
    if not a.aggregate_only:
        per_run.round({"median_fd_mm": 3, "pct_frames_above_threshold": 2}).to_csv(
            a.out / "fd_per_run.tsv", sep="\t", index=False)
    s = summary(runs, per_run, precise=a.precise)
    s.to_csv(a.out / "fd_summary.tsv", sep="\t", index=False)
    raincloud(frames_long(runs, per_run), a.out / "figure3D_fd_raincloud.png", np.random.default_rng(42))

    print(s[s["task"] == "all_tasks"].to_string(index=False))
    print(f"Tasks found: {sorted(per_run['task'].unique())}. Outputs -> {a.out}")


if __name__ == "__main__":
    main()
