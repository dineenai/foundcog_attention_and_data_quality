#!/usr/bin/env python3
"""Step 3: Figure 5 panels B-F - attentive state by age, task and run.

Reads outputs/attentive_state/tag_counts_per_run.tsv (step 2). For every run, the proportion of
the run's seconds (%) in each category is:

  Interested          interest_scanning
  Fussing             upset
  Sleepy              sleepy
  Asleep              asleep
  Eyes closed awake   eyes_closed_awake
  Looking away        looking_away_internal + _camera + _foreignobject
                      (total time looking away)

Runs with no seconds left after baby_out truncation are dropped.

Panels
  B  all runs, 2 vs 9 months. Clustered OLS per category (proportion ~ age,
     standard errors clustered by infant), BH-FDR across the 6 categories.
  C  2 months, all runs, videos vs pictures. Same test (proportion ~ task),
     BH-FDR across categories.
  D  as C, 9 months.
  E  2 months, session 1, infants with all of first video run, first picture
     run and second video run; Interested, Fussing, Sleepy, Asleep. Friedman
     test across the three runs, then pairwise Wilcoxon signed-rank tests
     (stars show uncorrected Wilcoxon p < .05 / .01 / .001).
  F  as E, 9 months.

Exclusions are read from code/attentive_state/exclusions.tsv (participant_id, applies_to, reason):
applies_to = 'all' removes the infant from every panel; 'run_comparison'
removes them from panels E-F only.

Outputs in outputs/attentive_state/figure5_sharable_runs/ (or --out): one PNG per panel, and for each panel the run and
infant counts, median/IQR per group and category, and the test results.

This reproduces plot_tags_by_age.py.py and PlotTagsFunctions.py from the
original FOUNDCOG pipeline. Run on the shared data (consenting control infants
only), counts and statistics differ from the published figure, which also
includes NICU infants and controls whose caregivers did not consent to sharing.

Usage (from the repository root):
  python code/attentive_state/03_figure5.py
"""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "attentive_state"
HERE = Path(__file__).resolve().parent

LOOKING_AWAY_COLS = ["looking_away_internal", "looking_away_camera", "looking_away_foreignobject"]

CATEGORIES = {  # label: tag columns
    "Interested": ["interest_scanning"],
    "Fussing": ["upset"],
    "Sleepy": ["sleepy"],
    "Asleep": ["asleep"],
    "Eyes closed\nawake": ["eyes_closed_awake"],
    "Looking\naway": LOOKING_AWAY_COLS,
}
CATS_ALL = list(CATEGORIES)                                  # panels B-D
CATS_RUNS = ["Interested", "Fussing", "Sleepy", "Asleep"]    # panels E-F
RUNS_COMPARED = ["S1_Videos_R1", "S1_Pictures_R1", "S1_Videos_R2"]
RUN_LABELS = {"S1_Videos_R1": "First Video Run", "S1_Pictures_R1": "First Picture Run",
              "S1_Videos_R2": "Second Video Run"}
AGE_LABELS = {"2M": "2-months", "9M": "9-months"}
MIN_RUNS_PER_GROUP_E_F = 10  # run types with fewer runs are not plotted
Y_TOP_E_F = 132  # y-axis top for panels E-F (labels stop at 100; the rest is for significance bars)
Y_TOP_B_D = 112


# ---------------------------------------------------------------- data

def load_runs(counts_path, exclusions_path):
    df = pd.read_csv(counts_path, sep="\t")
    df = df[df["n_timepoints"] > 0].copy()
    df["sub"] = df["run_name"].str.split("_").str[0]
    df["age"] = df["age_months"].map({2: "2M", 9: "9M"})
    df["task"] = df["task"].str.capitalize()
    df["ses_task_run"] = ("S" + df["session"].str.replace("ses-", "") + "_" + df["task"] + "_R"
                          + df["run"].astype(int).astype(str))

    excl = pd.read_csv(exclusions_path, sep="\t", comment="#") if exclusions_path.exists() else \
        pd.DataFrame(columns=["participant_id", "applies_to", "reason"])
    bad = set(excl["applies_to"]) - {"all", "run_comparison"}
    if bad:
        raise ValueError(f"{exclusions_path}: applies_to must be 'all' or 'run_comparison', found {bad}")
    drop_all = set(excl.loc[excl["applies_to"] == "all", "participant_id"])
    drop_runs = set(excl.loc[excl["applies_to"] == "run_comparison", "participant_id"])
    n_before = df["sub"].nunique()
    df = df[~df["sub"].isin(drop_all)]
    print(f"Excluded from all panels: {n_before - df['sub'].nunique()} infants present in the data "
          f"({len(drop_all)} listed)")
    return df, drop_runs


def proportions_long(df, categories):
    """One row per run x category: % of the run's seconds."""
    rows = []
    for cat in categories:
        p = df[CATEGORIES[cat]].sum(axis=1) / df["n_timepoints"] * 100
        rows.append(df[["run_name", "sub", "age", "task", "ses_task_run", "n_timepoints"]]
                    .assign(Tag=cat, **{"Proportion (%)": p.values}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- stats

def clustered_ols(long, group_col, min_per_group):
    """Per category: proportion ~ group, SEs clustered by infant; BH-FDR across categories."""
    res = []
    for tag, d in long.groupby("Tag", sort=False):
        levels = sorted(d[group_col].unique())
        if len(levels) != 2 or d.groupby(group_col).size().min() < min_per_group:
            continue
        model = smf.ols(f"Q('Proportion (%)') ~ {group_col}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["sub"]})
        coef = f"{group_col}[T.{levels[1]}]"
        g1, g2 = (d.loc[d[group_col] == lv, "Proportion (%)"] for lv in levels)
        res.append({"Tag": tag.replace("\n", " "), "Group1": levels[0], "Group2": levels[1],
                    "coef": model.params[coef], "p-val": model.pvalues[coef],
                    "Median_Group1": g1.median(), "Median_Group2": g2.median(),
                    "Zero_pct_Group1": (g1 == 0).mean(), "Zero_pct_Group2": (g2 == 0).mean()})
    res = pd.DataFrame(res)
    if not res.empty:
        res["p_corr"] = multipletests(res["p-val"], method="fdr_bh")[1]
    return res


def friedman_wilcoxon(long):
    levels = sorted(long["ses_task_run"].unique())
    res = []
    for tag, d in long.groupby("Tag", sort=False):
        frame = d.pivot_table(index="sub", columns="ses_task_run", values="Proportion (%)")[levels].dropna()
        _, p_fried = friedmanchisquare(*[frame[c].values for c in levels])
        for a, b in itertools.combinations(levels, 2):
            _, p = wilcoxon(frame[a], frame[b])
            res.append({"Tag": tag.replace("\n", " "), "Comparison": f"{a} vs {b}", "n_infants": len(frame),
                        "p_friedman": p_fried, "p_wilcoxon": p})
    return pd.DataFrame(res)


def median_iqr(long, group_col):
    s = (long.groupby([group_col, "Tag"], sort=False)["Proportion (%)"]
         .agg(median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
         .round(3).reset_index())
    s["Tag"] = s["Tag"].str.replace("\n", " ")
    return s


# ---------------------------------------------------------------- plotting

def style_axes(ax, ylim):
    ax.set_ylim(ylim)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.yaxis.grid(True, linestyle="--", which="both", color="lightgrey", alpha=0.7)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.5)
    ax.set_ylabel("Proportion of Time (%)", fontsize=16)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=16)


def split_violin(long, group_col, hue_order, colors, legend_title, legend_labels, stats, path):
    fig, ax = plt.subplots(figsize=(len(CATS_ALL) + 3, 5))
    sns.violinplot(x="Tag", y="Proportion (%)", hue=group_col, order=CATS_ALL, hue_order=hue_order,
                   data=long, palette=colors, split=True, inner="quart", cut=0, density_norm="count",
                   legend=False, ax=ax)
    for i, pc in enumerate(ax.collections):
        pc.set_edgecolor(colors[hue_order[i % 2]])
        pc.set_linewidth(1.5)
        pc.set_alpha(0.7)
    for i, line in enumerate(ax.lines):
        median = i % 3 == 1
        line.set_linestyle("-" if median else "--")
        line.set_linewidth(2 if median else 1)
        line.set_color("darkslategray")
    # Fixed y-range (labels stop at 100) so panels are the same size with or without stars
    style_axes(ax, (0, Y_TOP_B_D))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 5), minor=True)

    if stats is not None and not stats.empty:  # bar across both halves + one star per category with BH-corrected p < .05
        xt = [t.get_text() for t in ax.get_xticklabels()]
        y, tick = 104, 1.5
        for _, r in stats[stats["p_corr"] < 0.05].iterrows():
            label = next(c for c in CATS_ALL if c.replace("\n", " ") == r["Tag"])
            x = ax.get_xticks()[xt.index(label)]
            x1, x2 = x - 0.2, x + 0.2
            ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], c="k", lw=1.4, zorder=300, clip_on=False)
            ax.text(x, y + 0.3, "*", ha="center", va="bottom", fontsize=16, zorder=300, clip_on=False)

    handles = [Patch(facecolor=colors[h], edgecolor=colors[h], alpha=0.7) for h in hue_order]
    ax.legend(handles, legend_labels, title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=16, title_fontsize=16)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_violins(long, stats, n_infants, path):
    counts = long.drop_duplicates(["age", "run_name"])["ses_task_run"].value_counts()
    keep = [r for r in RUNS_COMPARED if counts.get(r, 0) >= MIN_RUNS_PER_GROUP_E_F]
    data = long[long["ses_task_run"].isin(keep)]
    blues, purples = sns.color_palette("Blues_r", 6)[3:], sns.color_palette("Purples_r", 6)[3:]
    vids, pics = [r for r in keep if "Videos" in r], [r for r in keep if "Pictures" in r]
    palette = {**dict(zip(vids, blues)), **dict(zip(pics, purples))}

    fig, ax = plt.subplots(figsize=(len(CATS_RUNS) * 1.9 + 4, 5))
    sns.violinplot(x="Tag", y="Proportion (%)", hue="ses_task_run", hue_order=keep, order=CATS_RUNS,
                   data=data, palette=palette, cut=0, inner="quart", linewidth=0.5, dodge=True, ax=ax)
    for pc in ax.collections:
        pc.set_edgecolor(pc.get_facecolor())
        pc.set_linewidth(1.5)
        pc.set_alpha(0.7)
    for i, line in enumerate(ax.lines):
        median = i % 3 == 1
        line.set_linestyle("-" if median else "dotted")
        line.set_linewidth(1.5)
        line.set_color("black" if median else "slategray")
    # Fixed y-range so panels are the same size with or without significance bars:
    # room above 100 for up to three bar levels (all pairs of three runs).
    top = Y_TOP_E_F
    style_axes(ax, (0, top))

    if stats is not None and not stats.empty:  # tiered bars for pairwise Wilcoxon p < .05
        offsets = dict(zip(keep, np.linspace(-0.3, 0.3, len(keep))))
        layer_gap, tick, gap_above_data = 9, 1.5, 4  # in % units
        for xi, cat in enumerate(CATS_RUNS):
            sig = stats[(stats["Tag"] == cat) & (stats["p_wilcoxon"] < 0.05)]
            ymax = data.loc[data["Tag"] == cat, "Proportion (%)"].max()
            pairs = []
            for _, r in sig.iterrows():
                a, b = r["Comparison"].split(" vs ")
                if a in offsets and b in offsets:
                    x1, x2 = sorted((xi + offsets[a], xi + offsets[b]))
                    pairs.append((x2 - x1, x1, x2, r["p_wilcoxon"]))
            # shortest bars lowest, each on its own level, stars sitting above their bar
            for layer, (_, x1, x2, p) in enumerate(sorted(pairs)):
                y = ymax + gap_above_data + layer * layer_gap
                top = max(top, y + layer_gap)
                ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], c="k", lw=1.4, zorder=300, clip_on=False)
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                ax.text((x1 + x2) / 2, y + 0.3, stars, ha="center", va="bottom", fontsize=16,
                        zorder=300, clip_on=False)
        ax.set_ylim(top=top)
    # Space above 100 is only for the significance bars: label and grid 0-100 only
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 5), minor=True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [RUN_LABELS.get(l, l) for l in labels], title=f"Run ({n_infants} infants)",
              bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16, title_fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- panels

PRECISE = False  # --precise: write stats unrounded (used to check against the original pipeline)


def save(df, path):
    df.to_csv(path, sep="\t", index=False)


def save_stats(df, path):
    """Round for reading: estimates and medians to 3 dp, % zeros to 1 dp, p-values to 4 s.f."""
    df = df.copy()
    if not PRECISE and not df.empty:
        for c in df.columns:
            if c.startswith("Zero_pct"):
                df[c] = (df[c] * 100).round(1)
            elif c.startswith(("p-val", "p_corr", "p_friedman", "p_wilcoxon")):
                df[c] = df[c].map(lambda v: float(f"{v:.4g}"))
            elif c == "coef" or c.startswith("Median"):
                df[c] = df[c].round(3)
        df = df.rename(columns={c: c.replace("Zero_pct", "Pct_zero") for c in df.columns})
    df.to_csv(path, sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", default=OUT / "tag_counts_per_run.tsv", type=Path)
    ap.add_argument("--exclusions", default=HERE / "exclusions.tsv", type=Path)
    ap.add_argument("--out", default=OUT / "figure5_sharable_runs", type=Path)
    ap.add_argument("--precise", action="store_true", help="write statistics without rounding")
    a = ap.parse_args()
    globals()["PRECISE"] = a.precise
    a.out.mkdir(parents=True, exist_ok=True)

    runs, drop_for_run_comparison = load_runs(a.counts, a.exclusions)
    summary = []

    # B: age
    long = proportions_long(runs, CATS_ALL)
    stats = clustered_ols(long, "age", min_per_group=3)
    n_runs = long.drop_duplicates(["age", "run_name"]).groupby("age").size()
    split_violin(long, "age", ["2M", "9M"], {"2M": plt.get_cmap("Set2")(0), "9M": "teal"},
                 "Age (n runs)", [f"{AGE_LABELS[g]} ({n_runs.get(g, 0)})" for g in ["2M", "9M"]],
                 stats, a.out / "panel_B_age.png")
    save_stats(stats, a.out / "panel_B_stats_clustered_ols_age.tsv")
    save(median_iqr(long, "age"), a.out / "panel_B_median_iqr.tsv")
    summary += [{"panel": "B", "group": g, "n_runs": int(n_runs.get(g, 0)),
                 "n_infants": long.loc[long["age"] == g, "sub"].nunique()} for g in ["2M", "9M"]]

    # C, D: task within age
    for panel, age in (("C", "2M"), ("D", "9M")):
        la = long[long["age"] == age]
        stats = clustered_ols(la, "task", min_per_group=2)
        n_runs = la.drop_duplicates(["age", "run_name"]).groupby("task").size()
        split_violin(la, "task", ["Videos", "Pictures"],
                     {"Videos": plt.get_cmap("Blues")(0.6), "Pictures": plt.get_cmap("Purples")(0.6)},
                     "Task (n runs)", [f"{t} ({n_runs.get(t, 0)})" for t in ["Videos", "Pictures"]],
                     stats, a.out / f"panel_{panel}_task_{age}.png")
        save_stats(stats, a.out / f"panel_{panel}_stats_clustered_ols_task.tsv")
        save(median_iqr(la, "task"), a.out / f"panel_{panel}_median_iqr.tsv")
        summary += [{"panel": panel, "group": t, "n_runs": int(n_runs.get(t, 0)),
                     "n_infants": la.loc[la["task"] == t, "sub"].nunique()} for t in ["Videos", "Pictures"]]

    # E, F: first video, first picture, second video run of session 1
    for panel, age in (("E", "2M"), ("F", "9M")):
        r = runs[(runs["age"] == age) & runs["ses_task_run"].isin(RUNS_COMPARED)]
        complete = r.groupby("sub")["ses_task_run"].apply(set)
        subs = [s for s, have in complete.items() if set(RUNS_COMPARED) <= have]
        r = r[r["sub"].isin(subs) & ~r["sub"].isin(drop_for_run_comparison)]
        lr = proportions_long(r, CATS_RUNS)
        n_inf = r["sub"].nunique()
        stats = friedman_wilcoxon(lr) if n_inf else pd.DataFrame()
        run_violins(lr, stats, n_inf, a.out / f"panel_{panel}_runs_{age}.png")
        save_stats(stats, a.out / f"panel_{panel}_stats_friedman_wilcoxon.tsv")
        save(median_iqr(lr, "ses_task_run"), a.out / f"panel_{panel}_median_iqr.tsv")
        summary.append({"panel": panel, "group": "+".join(RUNS_COMPARED), "n_runs": len(r), "n_infants": n_inf})

    summary = pd.DataFrame(summary)
    save(summary, a.out / "panel_counts.tsv")
    print(summary.to_string(index=False))
    print(f"Figures, stats and summaries -> {a.out}")


if __name__ == "__main__":
    main()
