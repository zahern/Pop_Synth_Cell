# Reproducing the paper

Every reported number is computed on the paper's **367-zone fitted cohort**
(`pbs/zones_371.txt` minus `301031014`, `304041102`, `310021279`,
`311031315`) and, for the main table, the **11 control tables shared by every
arm**. The held-out tables are never used in fitting.

The commands below assume the working directory is this repository root and
that `config.py` points at the census microdata. `python` means the `syn_env`
environment; on the HPC every python command is run inside a PBS job (the
`pbs/*.pbs` scripts), never on the login node.

## Pipeline

### 1. CP feasible pool

```bash
python build_feasible_pool_cp.py \
    --census "$CENSUS_CSV" \
    --control-attrs SEXP AGEP MSTP HEAP OCCP INCP LFSP EMPP HRSP QALFP QALLP \
                    RLHP MTWP NPRD VEHRD MRERD RNTRD STRD HIND BEDRD HHCD \
    --min-count 5 --target-per-cell 5 --max-infill-cells 30000 \
    --coverage-tables --coverage-per-category 5 \
    --out outputs/feasible_pool.csv
```

The exact production call, including the fusion step that creates
`LINKED_DATA_national.csv`, is in `pbs/stage_cp_pool.pbs`. The fusion script
itself belongs to the companion microdata pipeline and is not part of this
repository; the fused census is an input (see README, Data).

### 2. MIP pool expansion

The released paper arms use the 300-donor / 32-shell menu; the final run uses
2,000 donors. The job chain is:

```bash
DONORS_PER_CELL=300 SHELLS_PER_CELL=32 INCLUDE_PARTIAL=1 \
    bash submit_mip_expansion_sharded.sh
```

`submit_mip_expansion_sharded.sh` splits the 371-zone scope into 20 disjoint
shards, runs `pbs/stage_mip_expand_shard.pbs`, then merges with
`pbs/merge_expansion_shards.pbs` and (optionally) attaches HHTS attributes with
`pbs/attach_hhts_to_pool.pbs`.

### 3. Composition-aware pool

```bash
python make_composition_aware_pool.py
```

The script filters the appended menu (`_source == "mip_joint_menu"`) to remove
size-7 households and any household carrying `MTWP=17`. Edit `SRC` / `OUT` at
the top of the file for a different input pool.

### 4. Zone fitting

```bash
bash submit_cp_sa_chunks.sh      # CP + SA, 20 chunks
bash submit_cp_ipu.sh            # CP + IPU, 20 chunks
bash submit_cp_mip.sh            # CP + integer-count MIP, 20 chunks
```

The base arms are produced by the same submitters with `POOL=outputs/feasible_pool.csv`
(CP-only) or the raw-microdata settings in `pbs/stage_cp_sa_only.pbs`. The
expansion variants (coverage-only, composition) are selected by
`POOL`/`RESULTS_DIR`; the exact commands are recorded in the job logs under
`pbs_logs/` on the HPC (not committed).

### 5. Scores and audits

```bash
python score_paper_arms.py --latex       # Tables 9 and 11, 11-table RSSZ
python full_cohort_audit.py              # joint TVD, |Delta V|, clone audit
python make_fig_heldout.py               # held-out TVD table and figures
python heldout_rssz.py                   # held-out RSSZ by table
```

`score_paper_arms.py` reads the per-zone summaries written by the fitters
(`cp_pool_rssz_summary_chunk*.csv`); `--latex` prints the rows in the paper's
column order. `full_cohort_audit.py` reads the released populations.

## Paper tables

| Paper item | Script | Output |
|---|---|---|
| `tab:cohorts` | hand-written from `pbs/zones_371.txt` and control-table coverage | — |
| `tab:strategy_roles` | design table, hand-written | — |
| `tab:pool_support_stats` | `make_paper_boards.py` | `outputs/paper_figures/` |
| `tab:pool_growth` | `make_paper_boards.py` | `outputs/paper_figures/` |
| `tab:expansion_composition` | `score_paper_arms.py --latex` | `outputs/paper_arms/paper_arms_summary.csv` |
| `tab:pool_fitter_matrix` | `score_paper_arms.py --latex` | `outputs/paper_arms/paper_arms_summary.csv` |
| `tab:cp_vs_mip` | `score_paper_arms.py --latex` | `outputs/paper_arms/paper_arms_summary.csv` |
| `tab:cp_sa_tail` | tail-run summaries (`outputs/cp_pool_results*`, 5,000 iterations) | `files/cp_sa_tail.tex` |
| `tab:pathological_zones` | `score_paper_arms.py` per-zone CSV | `outputs/paper_arms/paper_arms_zones.csv` |
| `tab:full_audit` | `full_cohort_audit.py` | `outputs/full_cohort_audit/full_cohort_audit.csv` |
| `tab:heldout` | `make_fig_heldout.py` | `outputs/paper_figures/heldout_tvd_summary.csv` |
| `tab:pool_expansion_audit` | `make_paper_boards.py` from `outputs/expansion_shards/*.report.json` | `files/pool_expansion_table.tex` |
| `tab:control_codebook`, `tab:hh_handling`, `tab:heap_categories` | hand-written | — |
| `tab:known-methods-comparison` | conceptual comparison, hand-written | — |
| `tab:matched-method-results` | 60-zone empirical comparison; see note below | — |

## Paper figures

| Paper item | Script | Output |
|---|---|---|
| `fig:flow`, `fig:cp_pool`, `fig:mip-expansion-process`, `fig:missing-cells-framework`, `fig:donor_chromosome` | TikZ in `files/*.tex` | compiled PDF |
| `fig:expansion_board` | `make_paper_boards.py` | `files/fig_board_expansion.pdf` |
| `fig:zero_cell_coverage` | `paper_figures.py` | `zero_cell_coverage.png` |
| `fig:pool_before_after` | `make_pool_before_after_fig.py` | `fig_pool_before_after.png` |
| `fig:pool_gain_ipu`, `fig:pool_gain_sa` | `make_pool_gain_plot.py` | `fig_pool_gain_ipu.png`, `fig_pool_gain_sa.png` |
| `fig:marginals_all_tables` | `make_marginal_histograms.py` | `fig_marginals_all_tables.png` |
| `fig:marginal_differences` | `make_fig_all_variables.py` / `make_fig4.py` | `fig4_marginals_by_arm.png` |
| `fig:fit_example` | `make_fit_example.py` | `fig_fit_example.png` |
| `fig:sa_convergence` | `make_sa_convergence.py` | `fig_sa_convergence.png` |
| `fig:heldout_tvd` and `fig:heldout_categories_*` | `make_fig_heldout.py` | `fig_heldout_*.png` |

## Notes and known limitations

- **`tab:matched-method-results` (60-zone empirical comparison).** The
  generator for this table is not in this repository; its numbers should be
  recomputed from the 60-zone outputs before the table is used in a release.
  Person-level TVD is unaffected by the household-count correction described
  below, but the HH TVD column must be regenerated with one row per household.
- **Household-level metrics.** Synthetic populations repeat dwelling attributes
  on every member row. `calculate_total_rssz_table` groups by `abshid` before
  scoring household tables; `benchmark_margins.zone_metrics` and
  `make_fig_heldout._counts` were corrected on 2026-11-04 to do the same.
  Any older HH TVD/SRMSE/MAE values produced before that date are inflated.
- **Held-out cohort.** `make_fig_heldout.py` restricts to the 367-zone fitted
  cohort (not the 368-zone intersection of the arm output directories), so the
  table matches the rest of the paper. `lfsp` is available for 259 of those
  zones.
- **Stochastic fits.** SA uses a fixed seed per zone but is still stochastic in
  the accepted move sequence; repeated runs differ slightly. IPU and the
  integer-count MIP are deterministic given the pool and the solver time limit.
- **Data.** The ABS microdata are not redistributable. Only the SA2 control
  tables and the small summary CSVs are committed.
