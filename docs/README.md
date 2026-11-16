# Population synthesis fit dashboard

An interactive companion to the paper *Feasibility-Constrained Mixed-Integer
Expansion of Candidate Pools for Synthetic Population Generation* (v8
corrected-control run).

**Live site:** <https://zahern.github.io/Pop_Synth_Cell/>

It lets a reader navigate the paper's arms, see the per-zone fit on a map,
inspect the per-table RSSZ for a selected zone, and compare cohort-level
observed and synthesized marginal shares.

## What is in it

- **Map**: 371 SA2 polygons (362 fitted zones + 9 outside the cohort in gray),
  coloured by per-zone 19-table RSSZ for the selected configuration. Hover for the zone
  name, code, RSSZ and rank; click to select; scroll to zoom; drag to pan. The
  **Random zone** button picks a fitted zone at random.
- **Method selector**: the six configurations of the paper: CP pool with SA
  and IPU, and the expanded pool with SA, IPU, MIP and unique MIP.
- **Zone detail**: per-zone RSSZ, rank, per-table RSSZ bars (19 tables), and the
  method's median/mean RSSZ, held-out mean RSSZ, joint
  TVD, identical-person share and single-zone share.
- **Observed vs synthesized**: for the selected table, the cohort-level
  observed and synthesized category shares and their difference in percentage
  points.
- **Marginal progression (field of study)**: for the selected zone, the
  `qalfp` table's target, raw SA4 microdata, CP pool, MIP pool and fitted
  CP-MIP+fallback percentages. This is the same progression as the paper's zone
  table for SA2 112011236.
- **Method comparison**: the full configuration table; click a row to switch the map.
- **Method descriptions**: each configuration's pool and fitting method, with the
  CP construction, expansion MIP, fallback block and fitting MIP stated explicitly.

## Static figures

- `fig_rssz_by_arm.png` - per-zone RSSZ boxplots for the unified CP--MIP method
  and its three block ablations.
- `fig_zone_progression_all.png` - four-table marginal progression for SA2
  112011236: each pool stage against the census marginal.
- `fig_method_table_heatmap.png` - mean per-zone RSSZ by configuration and
  control table, with the best configuration starred in each column.
- `fig_table_delta_heatmap.png` - per-table RSSZ change relative to the CP pool
  with SA for the other four configurations.
- `fig_pool_tvd_by_size.png` - pool-target marginal distance by zone population
  quintile, before fitting.
- `fig_pool_score_summary.png` - marginal distance, zero-cell coverage and their
  equal-weight composite for each pool stage.
- `fig_pool_progression_by_size.png` - field-of-study and highest-education
  progression for five zones spanning the size distribution.

## Data and caveats

- Fitted cohort: **362 zones**. Nine zones outside it (four with no usable target
  and five non-residential/POE-special) are shown in gray and excluded from cohort
  means.
- Fitting basis: **19 in-sample control tables**.
- Held-out: **7 control tables**, mean RSSZ per arm.
- Marginal shares are **cohort-level** across the 362 fitted zones. The map and
  per-table bars are the per-zone fit.
- Sources (v8 run root `runs/2026-11-13_v8`):
  - `tables/zone_table_rssz.csv` (per-zone/per-table RSSZ)
  - `tables/arm_summary.csv` (summary)
  - `paper_figures/marginal_category_shares.csv` (observed vs synthesized)
  - `audit/heldout_rssz_all.csv` (held-out)
  - `audit/full_cohort_audit_v8.csv` (joint/diversity)
  - `synth_pop/boundaries/SA2_2021_AUST_GDA2020.shp` (boundaries)

## Run it locally

The site is fully static: open `index.html` directly, or serve the folder:

```bat
cd docs
python -m http.server 8765
```

then open <http://127.0.0.1:8765/>.

## Rebuild the data

```bat
python build_dashboard_data_v8.py
python build_progression_qalfp_v8.py
```

Both scripts read the v8 run over the `Z:` share and rewrite `data/zones.js`,
`data/results.js` and `data/progression_qalfp.js`. They use geopandas to
simplify the SA2 boundaries to ~100 m.

## Publishing

The site is the `docs/` folder on the `paper-revision` branch. Enable it once
under **Settings -> Pages -> Build and deployment -> Deploy from a branch**,
then choose branch `paper-revision` and folder `/docs`. Every push that changes
`docs/` redeploys automatically.

`plot_v8_figures.py`, `plot_method_table_heatmap.py` and
`zone_progression_all_v8.py` are kept here as the generators for the static
figures.
