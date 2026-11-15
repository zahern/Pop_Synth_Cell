# Population synthesis fit dashboard

An interactive companion to the paper *Feasibility-Constrained Mixed-Integer
Expansion of Candidate Pools for Synthetic Population Generation* (v8
corrected-control run).

**Live site:** <https://zahern.github.io/Pop_Synth_Cell/>

It lets a reader navigate the paper's arms, see the per-zone fit on a map,
inspect the per-table RSSZ for a selected zone, and compare cohort-level
observed and synthesized marginal shares.

## What is in it

- **Map**: 371 SA2 polygons (362 fitted zones + 5 special zones in gray),
  coloured by per-zone 19-table RSSZ for the selected method. Hover for the zone
  name, code, RSSZ and rank; click to select; scroll to zoom; drag to pan. The
  **Random zone** button picks a fitted zone at random.
- **Method selector**: the arms with a v8 fit. SA-Only and IPU-Only appear once
  their baseline jobs finish.
- **Zone detail**: per-zone RSSZ, rank, per-table RSSZ bars (19 tables), and the
  method's median/mean RSSZ (19- and 11-table bases), held-out mean RSSZ, joint
  TVD, identical-person share and single-zone share.
- **Observed vs synthesized**: for the selected table, the cohort-level
  observed and synthesized category shares and their difference in percentage
  points.
- **Marginal progression (field of study)**: for the selected zone, the
  `qalfp` table's target, raw SA4 microdata, CP pool, MIP pool and fitted
  CP-MIP+fallback percentages. This is the same progression as the paper's zone
  table for SA2 112011236.
- **Method comparison**: the full arm table; click a row to switch the map.
- **Method descriptions**: each arm's pool and fitting method, with the two MIP
  roles (expansion MIP vs fitting MIP) stated explicitly.

## Static figures

- `fig_rssz_by_arm.png` - per-zone RSSZ boxplots, split by pool coverage, with
  the fitting method on each row.
- `fig_zone_progression_all.png` - six-table marginal progression for SA2
  112011236: census target against the raw SA4 slice, CP pool and expanded pool.
- `fig_method_table_heatmap.png` - mean per-zone RSSZ by method and control
  table, with the best arm starred in each column.

## Data and caveats

- Fitted cohort: **362 zones**. The five special zones are shown in gray and
  excluded from cohort means. The four no-target zones are omitted.
- Fitting basis: **19 in-sample control tables**; the 11 shared tables are shown
  in the method comparison.
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
