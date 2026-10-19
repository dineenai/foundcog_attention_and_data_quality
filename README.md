# FOUNDCOG attention and data quality

Code and attentive-state tags accompanying:

> Dineen, Á. T., O'Doherty, C., Truzzi, A., King, G., Zaadnoordijk, L., D'Arcy, E.-L., Caldinelli, C., Kravchenko, A., Holloway, T., White, J., Wade, A. R., De Burca, A., Healy, M., Burke, A., Campion, K., Murphy, C., Herbert, K., Ambre, C., Wadhwa, A., Markey, O., Rico-Picó, J., O'Hara, D., Byrne, A. T., Tarrant, A., Foran, A., Molloy, E. J., & Cusack, R. (2026). Awake Infant Functional MRI For Visual Neuroscience: Reflections from the Foundations of Cognition Project. *Frontiers in Neuroscience*, *20*, 1852835. https://doi.org/10.3389/fnins.2026.1852835

<details>
<summary>BibTeX</summary>

```bibtex
@article{dineen2026awake,
  title   = {Awake Infant Functional {MRI} For Visual Neuroscience: Reflections from the Foundations of Cognition Project},
  author  = {Dineen, {\'A}ine Travers and O'Doherty, Cliona and Truzzi, Anna and King, Graham and Zaadnoordijk, Lorijn and D'Arcy, Enna-Louise and Caldinelli, Chiara and Kravchenko, Anna and Holloway, Tamrin and White, Jessica and Wade, Alex R. and De Burca, Aoibhe and Healy, Maebh and Burke, Amy and Campion, Kevin and Murphy, Cameron and Herbert, Katie and Ambre, Claire and Wadhwa, Anisha and Markey, Oscar and Rico-Pic{\'o}, Josu{\'e} and O'Hara, Deirdre and Byrne, Angela T. and Tarrant, Ailbhe and Foran, Adrienne and Molloy, Eleanor J. and Cusack, Rhodri},
  journal = {Frontiers in Neuroscience},
  volume  = {20},
  pages   = {1852835},
  year    = {2026},
  publisher = {Frontiers},
  doi     = {10.3389/fnins.2026.1852835}
}
```

</details>

The code reproduces the figures on head motion (Figure 3D–F, Supplementary Figure 4) and attentive state (Figure 5 B–F). It can be run on the data shared here and on OpenNeuro. Each figure is also provided as reported in the paper, computed on the full cohort (group-level results only).

```
code/
  motion/            01_figure3D.py, 02_figure3EF_S4.py
  attentive_state/   01_tags_to_timeseries.py, 02_count_tags.py, 03_figure5.py
data/
  attentive_state_tags/   tag files and tag_run_metadata.tsv
outputs/
  motion/            figure3D_* and figure3EF_S4_* (openneuro_v1-0-1/, full_cohort/)
  attentive_state/   tag time series and counts, figure5_sharable_runs/, figure5_full_cohort/
```

All scripts are run from the repository root. Requirements: Python 3, numpy, pandas, scipy, statsmodels, matplotlib, seaborn.

## Motion (Figure 3D)

`code/motion/01_figure3D.py` plots head motion as framewise displacement (FD; Power et al., 2012) of every frame, at 2 and 9 months, in three groups:

- **All awake:** every awake run (pictures and videos tasks).
- **Included awake:** awake runs with median FD < 1.5 mm, the inclusion threshold.
- **Asleep:** resting-state runs acquired while the infant slept (rest5 and rest10 tasks).

The figure is a raincloud plot: half-violins of FD across frames, with a box plot and the individual frames below. The dashed line marks the 1.5 mm threshold. Violin heights are proportional to the number of frames, relative to the largest group of the same age. The script also writes `fd_summary.tsv`: for each age, group and task, the number of runs and infants, the median and interquartile range of run-median FD (of frames, for asleep runs), and the percentage of frames censored (FD ≥ 1.5 mm). For data that may be shared, it also writes `fd_per_run.tsv` with one row per run.

The FD values are read from the OpenNeuro dataset **ds006883**:

- Release used here: **v1.0.1**, https://openneuro.org/datasets/ds006883/versions/1.0.1
- This release contains the pictures runs of 126 infants (84 at 2 months, 42 at 9 months). The videos and resting-state runs will be added in later releases.

To reproduce, download the dataset (or the `derivatives/foundcog_preproc/motion_fwd/` folder), then run:

```bash
python code/motion/01_figure3D.py --bids-dir /path/to/ds006883
```

Output goes to `outputs/motion/figure3D_openneuro_v<release>/`. The release is taken from the dataset's version tag if it was downloaded with DataLad, and otherwise defaults to v1.0.1. Use `--openneuro-version` to set it explicitly.

| Folder | Data |
|---|---|
| `outputs/motion/figure3D_openneuro_v1-0-1/` | OpenNeuro ds006883 v1.0.1 (pictures runs only) |
| `outputs/motion/figure3D_full_cohort/` | All infants in the paper (control and NICU), all awake and asleep runs, as published (group-level summary only) |

## Motion thresholds (Figure 3E–F, Supplementary Figure 4)

`code/motion/02_figure3EF_S4.py` shows how the two motion thresholds affect how much awake data is kept. It uses the same FD input and options as `01_figure3D.py`, and awake runs only.

- **Figure 3E–F:** the percentage of frames censored (FD ≥ 1.5 mm) in each run, for videos and pictures runs (3E: 2 months, 3F: 9 months).
- **Supplementary Figure 4 A–B:** the percentage of runs included as the run inclusion threshold (median FD below the threshold) varies from 0.25 to 3 mm, for videos (A) and pictures (B).
- **Supplementary Figure 4 C–F:** the percentage of frames censored within the included runs, for each run inclusion threshold and frame censoring threshold (C–D: 2 months, E–F: 9 months; videos, pictures).

The chosen 1.5 mm thresholds are marked. As in the paper, only runs with field maps from the same session are used. The tables `threshold_sensitivity.tsv` and `censoring_per_run.tsv` hold the plotted values.

```bash
python code/motion/02_figure3EF_S4.py --bids-dir /path/to/ds006883
```

| Folder | Data |
|---|---|
| `outputs/motion/figure3EF_S4_openneuro_v1-0-1/` | OpenNeuro ds006883 v1.0.1 (pictures runs only, so no videos panels) |
| `outputs/motion/figure3EF_S4_full_cohort/` | All infants in the paper, as published (no run names) |

## Attentive state (Figure 5 B–F)

### Tags

Infants' attentive state was coded from the in-scanner video recordings in ELAN. The tags are shared in `data/attentive_state_tags/`: 413 tag files from 112 infants, one per tagged run, for infants whose caregivers consented to data sharing.

- Files: `sub-<ID>/ses-<N>/func/sub-<ID>_ses-<N>_task-<task>_dir-<dir>_run-<NNN>_tags.txt`. Each file is an ELAN tab-delimited export with one row per tagged interval: tag name, start, end and duration (s), relative to the start of the tagged (cropped) camera video. Add `video_onset_s` from `tag_run_metadata.tsv` to align tag times with the run's `events.tsv` on OpenNeuro.
- Participant IDs: `sub-2xxx` for infants scanned at 2 months, `sub-9xxx` for infants scanned at 9 months. Run names match OpenNeuro ds006883.
- Tags: interest_scanning, interest_enjoyment, upset, sleepy, asleep, eyes_closed_awake, vocalising, looking_away_internal, looking_away_camera, looking_away_foreignobject, eyes_blocked, adult_hand, object_in_mouth, baby_out. Definitions are in Table 1 of the paper. <!-- TODO: add codebook -->

#### Examples

<table>
  <tr>
    <th>Interested 2-month-old</th>
    <th>Sleepy 2-month-old falls asleep</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/2709f15e-2500-440b-beb6-133f553ab7ee" width="360" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/429d7279-ac34-41cf-abfa-59ebe723ad7e" width="360" controls></video></td>
  </tr>
</table>


Shared with the caregivers' permission. © 2026 Cusack Lab, CC BY-NC-ND 4.0 (see [media/LICENSE](media/LICENSE)).

`tag_run_metadata.tsv` gives, for each tagged run:

| Column | Meaning |
|---|---|
| `run_name` | run, as in the tag file name |
| `start_precision` | how precisely the video is aligned to the run: `millisecond` (recording started automatically with the scan) or `second` (started and cropped manually; to about 1 s) |
| `video_onset_s` | time (s) in the run's `events.tsv` at which tag time 0 occurs; add it to tag times to align them with the run |
| `recording_starts_late`, `recording_ends_early` | the recording misses the start or end of the run |
| `complete_recording` | the recording covers the whole run |
| `run_duration_s` | duration of the run (s) |
| `video_duration_s` | duration of the tagged (cropped) video (s) |

### Analysis

```bash
python code/attentive_state/01_tags_to_timeseries.py   # tags -> 1 s time series
python code/attentive_state/02_count_tags.py           # seconds with each tag, per run
python code/attentive_state/03_figure5.py              # Figure 5 B-F and statistics
```

1. **Time series:** each tag file becomes a 1-second time series, one 0/1 column per tag. Time from the first `baby_out` tag onwards (infant taken out of the scanner) is removed.
2. **Counts:** for each run, the number of seconds with each tag and with combinations of tags. Proportions are given per age and run type.
3. **Figure 5:** the percentage of each run spent in each state. *Looking away* is the total time across the three looking-away tags.
   - Panel B compares 2 and 9 months.
   - Panels C–D compare videos and pictures runs within each age.
   - Panels B–D use linear models with standard errors clustered by infant, with Benjamini–Hochberg correction across states.
   - Panels E–F compare the first videos, first pictures and second videos run of session 1, in infants who had all three. They use a Friedman test, then Wilcoxon signed-rank tests (uncorrected).

| Folder | Data |
|---|---|
| `outputs/attentive_state/figure5_sharable_runs/` | the tags shared here |
| `outputs/attentive_state/figure5_full_cohort/` | all infants in the paper (group-level results only) |

The published figure also includes NICU infants and infants whose caregivers did not consent to sharing. Results from the shared tags therefore differ somewhat from the paper.

## To be added in the next release

- Noise ceiling analysis (Figure 4)
- Efficiency of the animacy contrast with respect to run length (Supplementary Figures 7 and 8)

## Contact

Questions or enquiries: [dineenai@tcd.ie](mailto:dineenai@tcd.ie)

## Licence

- Code and outputs: BSD 3-Clause License, as for the FOUNDCOG paradigm. See [LICENSE](LICENSE).
- Attentive-state tags and run metadata in `data/attentive_state_tags/`: CC BY 4.0, © 2026 Cusack Lab. See [data/attentive_state_tags/LICENSE](data/attentive_state_tags/LICENSE). Please cite the paper above when using them.
- Images and videos in `media/`: CC BY-NC-ND 4.0, © 2026 Cusack Lab. See [media/LICENSE](media/LICENSE). Credit as: "[filename]" by Cusack Lab, © 2026, licensed under CC BY-NC-ND 4.0.

## Citation

If you use the code or tags, please cite the paper above. <!-- TODO: Zenodo DOI for this repository -->
