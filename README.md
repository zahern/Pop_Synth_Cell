# Pop_Synth_Cell

Code and paper source for:

> **Feasibility-Constrained Mixed-Integer Expansion of Candidate Pools for
> Synthetic Population Generation: Construction, Household Assembly, and
> Combinatorial Optimisation**
> Zeke Ahern, Helen Thompson, Alexander Paz

This is the constraint-first pipeline used in the paper. It builds a feasible
candidate pool from census microdata and SA2 control tables, expands the pool
with mixed-integer-selected donor--shell households for zero-capacity cells,
fits each SA2 zone by simulated annealing (SA), iterative proportional updating
(IPU), or a single integer-count MIP, and scores the arms on a common cohort.

## Pipeline

| Stage | What it does | Main entry point |
|---|---|---|
| 0 | Build the CP feasible pool from census microdata under the structural rules | `build_feasible_pool_cp.py` |
| 1 | Expand the pool for zero-capacity cells with donor--shell households | `mip_pool.py --mode joint`, `pbs/stage_mip_expand_shard.pbs` |
| 2 | Optional composition filter on the appended menu | `make_composition_aware_pool.py` |
| 3 | Fit every SA2 zone (SA / IPU / integer-count MIP) | `cp_pool_sa.py`, `cp_pool_ipu.py`, `cp_pool_mip.py` |
| 4 | Score the arms on the shared tables and the 367-zone cohort | `score_paper_arms.py` |
| 5 | Joint / diversity audit, held-out validation, CP-MIP sweep | `full_cohort_audit.py`, `heldout_rssz.py`, `make_fig_heldout.py`, `append_cp_mip_heldout.py`, `sweep_identity.py` |
| 6 | Paper figures and boards | `make_*.py`, `paper_figures.py` |

The PBS scripts in `pbs/` are the exact jobs used for the paper; the
`submit_*.sh` scripts are the login-node submitters. Python always runs on a
compute node, never on the login node.

## Repository layout

```
*.py                     pipeline, fitting, scoring and figure scripts
pbs/                     PBS job scripts and the 371-zone / 20-chunk zone lists
submit_*.sh              login-node submitters (qsub only)
data_sa2_controls/       full SA2 control tables (26 tables)
data_sa2_controls_paper19/   the 19 in-sample tables used for fitting
data_sa2_controls_heldout/   the 7 held-out validation tables
released_summaries/      small CSVs and structured job reports behind the paper's tables
  expansion_reports/     per-shard expansion reports and merge/attach reports
files/, main_edit3.tex, figures   paper source
```

The fitted zone populations are large and are not committed. The small
summaries needed to check the paper's tables are in `released_summaries/`.

## Environment

```bash
conda create -n syn_env python=3.11 -y
conda activate syn_env
pip install -r requirements.txt
```

The PBS scripts assume a conda environment named `syn_env` under
`/home/ahernz/miniconda3/envs`; override `CPU_ENV_PREFIX` and `CPU_ENV_NAME`,
or edit `config.py`. The submitters also default `SYNTH_POP` to the original
HPC path, so from a fresh clone export the repository root first:

```bash
export SYNTH_POP="$PWD"
export SA_CONTROL_DIR="$PWD/data_sa2_controls_paper19"
```

## Data

Three inputs are needed:

1. **ABS 2021 Census 5% Basic microdata** (dwelling, family and person CSVs).
   These are large and are not redistributed here. Obtain them from the ABS
   and set `config.POP_SYNTH_DIR` (or the `SYNTH_POP`/`CENSUS_CSV` environment
   variables used by the PBS scripts). The CP stage consumes the fused
   national file `LINKED_DATA_national.csv`; the fusion step
   (`step_0b_fuse_national.py`) belongs to the companion microdata pipeline and
   is not part of this repository.
2. **SA2 control tables** — included in `data_sa2_controls*/`.
3. **Zone lists** — `pbs/zones_371.txt` is the 371-zone scope; the fitted
   cohort is that list minus `301031014`, `304041102`, `310021279` and
   `311031315` (367 zones). `pbs/zone_chunks/chunk_0..19.txt` are the 20
   fitting chunks.

## Configuration

`config.py` centralises every path and default. On a new machine edit at least:

- `HPC_HOME` / `POP_SYNTH_DIR` (where the census microdata live);
- the conda prefix/environment used by the PBS scripts;
- `OUTPUTS_DIR` if you do not want outputs under `synth_pop/outputs/`.

The PBS scripts also accept `SYNTH_POP`, `CENSUS_CSV`, `SA_CONTROL_DIR`,
`POOL` and `RESULTS_DIR` overrides, which is how the paper's jobs were run.

## Reproduce the paper

`REPRODUCING.md` maps each paper table and figure to the script that produces
it and the command used. The short version:

```bash
# 1. CP feasible pool
python build_feasible_pool_cp.py ...        # exact args in pbs/stage_cp_pool.pbs

# 2. MIP expansion (2000 donors / 32 shells in the final run; the
#    released paper arms use the settings recorded in the job logs)
DONORS_PER_CELL=... SHELLS_PER_CELL=... bash submit_mip_expansion_sharded.sh

# 3. Composition-aware variant
python make_composition_aware_pool.py       # edit SRC/OUT for your pool

# 4. Zone fits (20 chunks each)
bash submit_cp_sa_chunks.sh
bash submit_cp_ipu.sh
bash submit_cp_mip.sh

# 5. Scores, audits and held-out table
python score_paper_arms.py --latex
python full_cohort_audit.py
python make_fig_heldout.py
python append_cp_mip_heldout.py
python heldout_rssz.py
python sweep_identity.py
python make_zone_size_fig.py
```

All reported numbers are computed on the same 367 zones and, for the main
table, the 11 control tables shared by every arm. The held-out tables
(`age5p`, `assnp`, `bedrd`, `hcfmf`, `hhcd`, `lfsp`, `tenlld`) are never used
in fitting.

## Build the paper

A clean clone has no `.bbl`, so run BibTeX before the final LaTeX passes:

```bash
pdflatex -interaction=nonstopmode main_edit3.tex
bibtex main_edit3
pdflatex -interaction=nonstopmode main_edit3.tex
pdflatex -interaction=nonstopmode main_edit3.tex
```

The source is self-contained apart from the ABS microdata and fitted
populations, which are not redistributed.

## Reproducibility notes

- Seeds are fixed in the scripts and in the PBS environment; the SA runs are
  stochastic, so repeated runs will differ slightly at the zone level.
- The paper reports the CP-only, MIP-expanded, composition-filtered,
  coverage-only, SA-Only and IPU-Only arms. The one-MIP arm
  (`cp_pool_mip.py` / `zone_counts_mip.py`) is included as the current
  extension; its sweep and full-cohort jobs are in `submit_cp_mip*.sh`.
- Dwelling-level attributes (`bedrd`, `hcfmf`, `hhcd`, `tenlld`, and the
  in-sample household tables) are counted once per household in every metric.
  `calculate_total_rssz_table` does this by grouping on `abshid`;
  `benchmark_margins.zone_metrics` and `make_fig_heldout._counts` do the same
  after the 2026-11 correction.
- The paper's Data Availability statement points to this repository. The
  release branch is `paper-revision`; the default branch should be set to it
  when the repository is created.

## Citation

If you use this code, please cite the paper above. The previous GAN+SA
pipeline is archived separately at `zahern/Pop_Synth_GANSA`.

## Contact

Zeke Ahern, Queensland University of Technology.
