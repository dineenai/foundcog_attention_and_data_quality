#!/usr/bin/env python3
"""Step 1: convert attentive-state tag files into 1-second time series.

For every tag file in data/attentive_state_tags/ this:
  1. makes one row per second of the tagged video (video length from
     tag_run_metadata.tsv, rounded up to a whole second);
  2. marks each tag as present (1) or absent (0) in each second: tag start and
     end times are rounded to the nearest second and the tag covers
     [start, end), or just the start second if both round to the same second;
  3. removes every second from the first 'baby_out' second onwards (the infant
     was taken out of the scanner), so only time with the infant in the scanner
     is analysed. Runs where this happens are listed for checking.

Outputs (in outputs/, created if needed):
  tag_timeseries.tsv        run_name, second, then one 0/1 column per tag
  baby_out_truncation.tsv   runs cut at baby_out, with where and how much

This reproduces longformat_infant_videos_main.py and
baby_in_infant_videos_for_analysis.py from the original FOUNDCOG pipeline.

Usage (from the repository root):
  python code/01_tags_to_timeseries.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BIN_S = 1  # bin size in seconds

# Tag names, in the order of the ELAN tagging template (template_3.txt). Names in
# the tag files are matched after lower-casing and replacing spaces and hyphens
# with underscores.
TAG_NAMES = [
    "adult_hand", "asleep", "baby_out", "interest_enjoyment", "eyes_blocked",
    "eyes_closed_awake", "interest_scanning", "object_in_mouth", "sleepy",
    "upset", "vocalising", "looking_away_internal", "looking_away_camera",
    "looking_away_foreignobject",
]


def clean_tag_name(name):
    return str(name).strip().replace("-", "_").replace(" ", "_").lower()


def read_intervals(path):
    """Return the tag intervals in an ELAN tab-delimited export as
    (tag, start_s, end_s) rows. Exports have 6 or 9 columns depending on the
    ELAN export settings; the tag name, start, end and duration are kept."""
    data = pd.read_csv(path, sep="\t", header=None)
    if data.shape[1] == 6:
        data = data.drop(columns=data.columns[[1, 5]])
    elif data.shape[1] == 9:
        data = data.drop(columns=data.columns[[1, 2, 4, 6, 8]])
    else:
        raise ValueError(f"{path.name}: expected 6 or 9 columns, found {data.shape[1]}")
    data.columns = ["tag", "start_s", "end_s", "duration_s"]
    data["tag"] = data["tag"].map(clean_tag_name)
    return data


def to_timeseries(intervals, n_bins, path):
    ts = np.zeros((n_bins, len(TAG_NAMES)), dtype=int)
    col = {t: i for i, t in enumerate(TAG_NAMES)}
    for tag, start, end in intervals[["tag", "start_s", "end_s"]].itertuples(index=False):
        if tag not in col:
            raise ValueError(f"{path.name}: unknown tag '{tag}'")
        b0, b1 = int(np.round(start / BIN_S)), int(np.round(end / BIN_S))
        if b0 == b1:
            if 0 <= b0 < n_bins:
                ts[b0, col[tag]] = 1
        else:
            ts[max(b0, 0):max(min(b1, n_bins), 0), col[tag]] = 1
    return ts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=REPO / "data" / "attentive_state_tags", type=Path)
    ap.add_argument("--out", default=REPO / "outputs", type=Path)
    a = ap.parse_args()

    meta = pd.read_csv(a.data / "tag_run_metadata.tsv", sep="\t", index_col="run_name")
    files = sorted(a.data.glob("sub-*/ses-*/func/*_tags.txt"))
    if not files:
        sys.exit(f"No tag files found in {a.data}")

    frames, truncated = [], []
    for path in files:
        run = path.name[: -len("_tags.txt")]
        if run not in meta.index:
            sys.exit(f"{run} is not in tag_run_metadata.tsv")
        n_bins = int(np.round(np.ceil(meta.at[run, "video_duration_s"]) / BIN_S))
        ts = to_timeseries(read_intervals(path), n_bins, path)

        out_rows = np.flatnonzero(ts[:, TAG_NAMES.index("baby_out")] == 1)
        if out_rows.size:
            first, last = int(out_rows[0]), int(out_rows[-1])
            truncated.append({"run_name": run, "video_bins": n_bins, "first_baby_out_s": first,
                              "last_baby_out_s": last, "seconds_kept": first,
                              "baby_out_contiguous": int(last - first + 1 == out_rows.size)})
            ts = ts[:first]

        df = pd.DataFrame(ts, columns=TAG_NAMES)
        df.insert(0, "second", np.arange(len(df)) * BIN_S)
        df.insert(0, "run_name", run)
        frames.append(df)

    a.out.mkdir(parents=True, exist_ok=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(a.out / "tag_timeseries.tsv", sep="\t", index=False)
    pd.DataFrame(truncated, columns=["run_name", "video_bins", "first_baby_out_s", "last_baby_out_s",
                                     "seconds_kept", "baby_out_contiguous"]
                 ).to_csv(a.out / "baby_out_truncation.tsv", sep="\t", index=False)

    print(f"{len(files)} runs, {len(out)} seconds in total ({len(out) / 3600:.1f} h) -> {a.out / 'tag_timeseries.tsv'}")
    print(f"{len(truncated)} runs cut at baby_out -> {a.out / 'baby_out_truncation.tsv'}")
    odd = [t["run_name"] for t in truncated
           if not t["baby_out_contiguous"] or t["last_baby_out_s"] < t["video_bins"] - 5]
    if odd:
        print(f"  Check these {len(odd)}: baby_out is not one block running to the end of the video, "
              f"so it may be an accidental tag: {', '.join(odd)}")


if __name__ == "__main__":
    main()
