# Tommy John Surgery Dataset — Data Dictionary

**File:** `tommy_john.parquet` (primary surgery-level data)
**Companion:** `tommy_john_surgeons.parquet` (surgeon-level aggregates)
**Source:** Jon Roegele's [Tommy John Surgery List](https://docs.google.com/spreadsheets/d/1gQujXQQGOVNaiuwSN680Hq-FDVsCwvN-3AazykOBON0/) (XLSX export, pulled 2026-05-20)
**Rows:** 2,695 surgeries × 35 columns
**Pipeline:** `etl_tj.py`

## Why parquet, not CSV

The upstream Google Sheet has two structural issues that break naive CSV export:

1. **Duplicate column names.** Six column headers (`G`, `GS`, `IP`, `K-BB%`, `ERA-`, `FIP-`) appear twice. The XLSX inspection shows there is no merged group label disambiguating them in the source — they are genuinely ambiguous. Furthermore these columns are populated for only **59 of 2,696 rows (2.2%)**, and inspection of populated rows suggests inconsistent semantics. **They are dropped from this parquet.** If you need pre-TJ workload, join to Lahman or pybaseball via `mlbam_id` / `fangraphs_id`.

2. **Type information is lost in CSV.** Surgery and return dates round-trip cleanly only as datetime columns; the raw months-since-surgery column is often blank even when both dates are present.

Parquet preserves dtypes, makes column semantics explicit via the schema, and is ~50% smaller on disk.

## Columns

### Identifiers

| Column | Type | Description |
|---|---|---|
| `player` | string | Player name (free text — same name across multiple-surgery rows). |
| `mlbam_id` | Int64 | MLB Advanced Media player ID. Join key for pybaseball / Statcast. |
| `fangraphs_id` | Int64 | FanGraphs player ID. Join key for FanGraphs and Lahman. |

### Surgery details

| Column | Type | Description |
|---|---|---|
| `surgery_date` | datetime | Date of UCL reconstruction. |
| `surgery_year` | int32 | Calendar year of surgery (derived). For era effects. |
| `surgery_doy_frac` | float64 | Day-of-year / 365.25 ∈ [0,1]. For quarter-of-season effect (Honoki et al. 2025). |
| `age` | Int64 | Age in years at surgery. |
| `team` | string | Affiliated team / org (MLB abbreviation when at MLB level). |
| `level` | category | One of `MLB`, `AAA`, `AA`, `A+`, `A`, `A-`, `Rk`, `Coll`, `HS`. |
| `position` | category | Primary position. 92% are `P` (pitcher). |
| `throws` | category | `L` or `R`. |
| `country` | string | Country of origin. |
| `high_school` | string | Free text. |
| `colleges` | string | Free text; blank for HS-direct players. |
| `surgeon` | string | Free text, raw. Includes multi-surgeon entries separated by `/`. |
| `surgeon_group` | category | Collapsed: top 8 named surgeons + `Other` + `Unknown`. Use for modeling. |

### Outcome

| Column | Type | Description |
|---|---|---|
| `return_date` | datetime | Date of return to the *same* level the player was at when injured. Blank ⇒ no return observed. |
| `time_months` | float64 | Months from surgery to return (if observed) or to data-pull date (2026-05-20) if not. Computed as `(end - start).days / 30.4375`. **Re-computed from dates — do not use `recovery_months_raw`, which is missing for ~870 rows that have both dates.** |
| `event` | int8 | 1 = return to same level observed; 0 = not observed. |
| `competing_event` | int8 | 1 = `event=0` AND `active=0`; player did not return AND is no longer active. See "Censoring decisions" below. |
| `recovery_months_raw` | float64 | The original `Recovery Time (months)` column from the spreadsheet. Often blank even when computable. Kept for cross-checking; **prefer `time_months`**. |

### Recurrent-event structure

| Column | Type | Description |
|---|---|---|
| `surgery_number` | int64 | 1 for first TJ, 2 for second, etc. (ordered by `surgery_date` within `player`). |
| `revision` | int8 | 1 if `surgery_number >= 2`. |

### Post-TJ summary stats

These are populated by Roegele only for MLB-level surgeries with completed return.

| Column | Type | Description |
|---|---|---|
| `post_tj_mlb_g` | float64 | Total MLB games played after return. |
| `post_tj_mlb_ip_pa` | float64 | Total MLB IP (pitchers) or PA (hitters) after return. |
| `active` | Int64 | 1 if Roegele still considers the player active. Key signal for distinguishing genuine censoring from competing events. |

### Rehab milestones

Dates of intermediate rehab events. Sparse — populated for the more recent / higher-profile cases. Useful for landmark analyses but can introduce immortal-time bias if used naively as covariates.

| Column | Type | Description |
|---|---|---|
| `rehab_started_throwing` | datetime | Date player began throwing program. |
| `rehab_mound` | datetime | First date back on a mound. |
| `rehab_bullpen` | datetime | First bullpen session. |
| `rehab_live_hitters` | datetime | First live batting practice. |
| `rehab_game` | datetime | First competitive game in rehab assignment. |
| `setback1_date` | datetime | First documented setback. |
| `setback1_type` | string | Free-text setback description. |
| `setback2_date` | datetime | Second setback, if any. |
| `setback2_type` | string | Free-text setback description. |

## Censoring decisions — read this before modeling

A naïve `event=1` if `return_date` is observed gives an 81.7% event rate, which sounds reasonable but **hides a heterogeneous censoring structure**:

- **152 of 492 (31%)** non-return rows are 2023+ surgeries with `active=1` → genuine right-censoring (still in rehab as of data pull).
- **314 of 492 (64%)** are pre-2023 surgeries with `active=0` → really a *competing event*: the player exited play without ever returning to that level.
- **The remainder** are pre-2023 surgeries marked `active=1` — a mixed group of long-rehab cases and players who returned at a different level.

This shows up directly in the data: the **median follow-up time for "censored" rows is 94.8 months (~8 years)**, vs. 18.9 months for observed-event rows. Treating all 492 as standard right-censored observations will bias survival-curve estimates toward optimism.

Three modeling choices, in increasing order of correctness:

1. **Fixed horizon (simplest, good for tutorial).** Pick a maximum follow-up, e.g., 36 or 60 months. Any row censored beyond that becomes `event=0, time_months=<horizon>`. Defensible because the empirical 95th percentile of observed return times is well under 5 years.
2. **Use `competing_event` as a competing risk.** Fit a cause-specific or subdistribution model; Sparapani et al. 2020 (*SMMR* 29:57–77) extend BART to this case.
3. **Truncate to recent surgeries.** Restrict to `surgery_year >= 2010` (or similar), where data quality and the `active` flag are most reliable.

For the tutorial I recommend option **1 with a 36-month horizon**, with a brief mention of option 2 as a natural extension. This avoids competing-risks machinery while still giving honest censoring.

## Surgeon table

`tommy_john_surgeons.parquet` is Roegele's surgeon-level aggregation, 46 surgeons:

| Column | Type | Description |
|---|---|---|
| `surgeon` | string | Surgeon name. |
| `mlb_surgeries` | Int64 | Count at MLB level. |
| `milb_surgeries` | Int64 | Count at MiLB level. |
| `draft_surgeries` | Int64 | Count of pre-draft amateurs. |
| `total_surgeries` | Int64 | Total. |
| `mlb_return_pct` | float64 | Roegele's "% who returned to MLB" for that surgeon's MLB cases. |
| `revision_required_pct` | float64 | % of that surgeon's cases that subsequently required revision TJ. |

These are descriptive only — not adjusted for case mix. Confounded with era (older surgeons saw older protocols), level mix, and selection (top surgeons get more difficult cases).

## Known data-quality caveats

- **Surgeon coverage is sparse outside MLB.** 1,803 of 2,695 rows (67%) have `surgeon_group = 'Unknown'`, mostly amateur and lower-minor-league surgeries. For surgeon-effect modeling, restrict to MLB-level or MLB+AAA.
- **Roegele's "return to same level" definition** is specific: an MLB player must re-enter an MLB game; a HS pitcher must return to high-school play. A pitcher who has MLB surgery and returns only at AAA is `event=0` here.
- **Free-text surgeon column** has variant spellings. `surgeon_group` normalizes the top 8 but anything else lands in `Other`.
- **Some surgeries pre-1990 have only year-level precision.** `surgery_date` will show January 1 of that year in those cases.
- **`active` is Roegele-curated** and reflects his judgement at data-pull time. It is not an MLB-Advanced-Media derived field.

## Reproducing the parquet

```bash
curl -sL -o tj.xlsx \
  "https://docs.google.com/spreadsheets/d/1gQujXQQGOVNaiuwSN680Hq-FDVsCwvN-3AazykOBON0/export?format=xlsx"
python etl_tj.py
```
