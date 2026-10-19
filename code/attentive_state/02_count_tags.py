#!/usr/bin/env python3
"""Step 2: count attentive-state tags per run and summarise them as proportions.

Reads outputs/attentive_state/tag_timeseries.tsv (from step 1) and writes:

  tag_counts_per_run.tsv   one row per run: age, session, task, run, number of
                           seconds (n_timepoints), seconds with each tag, and
                           seconds with each combination of tags below
  tag_proportions.tsv      for each age x run type (videos, pictures, all runs):
                           % of all seconds with each tag or combination, in
                           four sets ('main', 'detailed', 'technical', 'summary')

Combinations (counts of seconds in which the condition holds):
  sleepy_AND/NOT_interest_scanning, vocalising_AND/NOT_upset,
  upset_AND_eyes_open (eyes_closed_awake = 0) [AND/NOT interest_scanning],
  eyes_closed_awake_AND_sleepy (should always be 0), eyes_closed_awake_AND/NOT_upset,
  interest_scanning with each combination of upset and sleepy,
  looking_away_ANY = looking_away_internal + _camera + _foreignobject (a sum of counts).

The 'summary' set is the mutually exclusive breakdown used to describe how
infants spent the run, plus 'remainder' (100 minus the rest: time with none of
them, i.e. passively attending).

This reproduces emotional_state_stats_revised.py from the original FOUNDCOG
pipeline. The shared data include control infants only, so results are given
per age and run type rather than split by NICU / control group.

Usage (from the repository root):
  python code/attentive_state/02_count_tags.py
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "attentive_state"

TAG_NAMES = [
    "adult_hand", "asleep", "baby_out", "interest_enjoyment", "eyes_blocked",
    "eyes_closed_awake", "interest_scanning", "object_in_mouth", "sleepy",
    "upset", "vocalising", "looking_away_internal", "looking_away_camera",
    "looking_away_foreignobject",
]

# Columns left out of the 'summary' set (as in the original analysis), so that
# the remaining categories do not overlap.
SUMMARY_DROP = [
    "adult_hand", "looking_away_ANY", "baby_out", "interest_enjoyment", "vocalising", "upset",
    "interest_scanning_AND_sleepy_AND_upset", "interest_scanning", "object_in_mouth",
    "vocalising_AND_upset", "vocalising_NOT_upset", "eyes_closed_awake",
    "eyes_closed_awake_AND_sleepy", "upset_AND_eyes_open", "sleepy",
    "sleepy_AND_interest_scanning", "upset_AND_eyes_open_AND_interest_scanning",
]
TECHNICAL = ["adult_hand", "object_in_mouth"]
RUN_RE = re.compile(r"^sub-(?P<age>[29])\d{3}_(?P<session>ses-\w+?)_task-(?P<task>\w+?)_(?:dir-\w+?_)?run-(?P<run>\d+)$")


def count_run(df):
    """Counts for one run, in the column order of the original tagCounts-detailed files."""
    on = {t: df[t] == 1 for t in TAG_NAMES}
    off = {t: df[t] == 0 for t in TAG_NAMES}
    c = {"n_timepoints": len(df)}
    c.update({t: int(np.count_nonzero(df[t])) for t in TAG_NAMES})
    combos = {
        "sleepy_AND_interest_scanning": on["sleepy"] & on["interest_scanning"],
        "sleepy_NOT_interest_scanning": on["sleepy"] & off["interest_scanning"],
        "vocalising_AND_upset": on["vocalising"] & on["upset"],
        "vocalising_NOT_upset": on["vocalising"] & off["upset"],
        "upset_AND_eyes_open": on["upset"] & off["eyes_closed_awake"],
        "upset_AND_eyes_open_AND_interest_scanning": on["upset"] & off["eyes_closed_awake"] & on["interest_scanning"],
        "upset_AND_eyes_open_NOT_interest_scanning": on["upset"] & off["eyes_closed_awake"] & off["interest_scanning"],
        "eyes_closed_awake_AND_sleepy": on["eyes_closed_awake"] & on["sleepy"],
        "eyes_closed_awake_AND_upset": on["eyes_closed_awake"] & on["upset"],
        "eyes_closed_awake_NOT_upset": on["eyes_closed_awake"] & off["upset"],
        "interest_scanning_AND_upset_NOT_sleepy": on["interest_scanning"] & on["upset"] & off["sleepy"],
        "interest_scanning_AND_sleepy_AND_upset": on["interest_scanning"] & on["upset"] & on["sleepy"],
        "interest_scanning_AND_sleepy_NOT_upset": on["interest_scanning"] & off["upset"] & on["sleepy"],
        "interest_scanning_NOT_upset_NOT_sleepy": on["interest_scanning"] & off["upset"] & off["sleepy"],
    }
    c.update({k: int(v.sum()) for k, v in combos.items()})
    c["looking_away_ANY"] = c["looking_away_internal"] + c["looking_away_camera"] + c["looking_away_foreignobject"]
    return c


def proportions(counts, cols):
    """% of all seconds in the group (pooled over runs)."""
    return counts[cols].sum() / counts["n_timepoints"].sum() * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT, type=Path)
    a = ap.parse_args()

    ts = pd.read_csv(a.out / "tag_timeseries.tsv", sep="\t")
    rows = []
    for run, df in ts.groupby("run_name", sort=True):
        m = RUN_RE.match(run)
        if not m:
            raise ValueError(f"Unexpected run name: {run}")
        rows.append({"run_name": run, "age_months": int(m.group("age")), "session": m.group("session"),
                     "task": m.group("task"), "run": m.group("run"), **count_run(df)})
    counts = pd.DataFrame(rows)
    counts.to_csv(a.out / "tag_counts_per_run.tsv", sep="\t", index=False)

    count_cols = [c for c in counts.columns if c not in ("run_name", "age_months", "session", "task", "run", "n_timepoints")]
    sets = {
        "main": TAG_NAMES,
        "detailed": count_cols,
        "technical": TECHNICAL,
        "summary": [c for c in count_cols if c not in SUMMARY_DROP],
    }
    out = []
    for age in sorted(counts["age_months"].unique()):
        for run_type, sel in (("videos", counts["task"] == "videos"), ("pictures", counts["task"] == "pictures"),
                              ("all_runs", counts["task"].notna())):
            g = counts[(counts["age_months"] == age) & sel]
            if g.empty:
                continue
            for set_name, cols in sets.items():
                p = proportions(g, cols)
                if set_name == "summary":
                    p["remainder"] = 100 - p.sum()
                for measure, value in p.items():
                    out.append({"age_months": age, "run_type": run_type, "set": set_name, "measure": measure,
                                "percent_of_time": value, "n_runs": len(g), "n_seconds": int(g["n_timepoints"].sum())})
    pd.DataFrame(out).to_csv(a.out / "tag_proportions.tsv", sep="\t", index=False)

    print(f"{len(counts)} runs -> {a.out / 'tag_counts_per_run.tsv'}")
    print(f"Proportions for {counts['age_months'].nunique()} ages x 3 run types -> {a.out / 'tag_proportions.tsv'}")
    bad = counts[counts["eyes_closed_awake_AND_sleepy"] > 0]
    if len(bad):
        print(f"  Note: {len(bad)} runs have seconds tagged both eyes_closed_awake and sleepy "
              f"(expected never): {', '.join(bad['run_name'])}")


if __name__ == "__main__":
    main()
