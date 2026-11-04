"""
Pipeline path configuration
============================
Single source of truth for all paths.  Two operating modes:

  LOCAL  — running on Windows (z:/ drive).
  HPC    — running on the cluster under /home/ahernz/.  Set USE_HPC=1 env var to force.

RULE: All Python scripts AND their input/output data live inside
      Paper_Plan/synth_pop (this folder).  Nothing is read from or
      written to directories outside Paper_Plan.

  Files that must be present in synth_pop/:
    microdata_for_exp.csv              -- census microdata seed (already moved here)
    data_sa2_controls/                 -- SA2 control tables in flat layout
                                           (split individual/household subdirs optional)

  To copy control tables from the old sa_run_again_n location (once, on HPC):
    cp -r /home/ahernz/sa_run_again_n/data_sa2_controls  /home/ahernz/Paper_Plan/synth_pop/

All scripts import this module with:
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
"""

import os

# ---------------------------------------------------------------------------
# Mode switch -- auto-detected; override with env var USE_HPC=1
#
# SYNTH_ROOT takes precedence over every auto-detection below.  It exists
# because the QUT and Gadi layouts differ: on QUT the tree lives under
# /home/ahernz, on Gadi under /scratch/<project>/<user>. Hardcoding either one
# silently points every input path at a directory that does not exist.
# ---------------------------------------------------------------------------
import os as _os
_SYNTH_ROOT_ENV = _os.environ.get("SYNTH_ROOT", "").rstrip("/")

USE_HPC = (
    _os.environ.get("USE_HPC", "0") == "1"   # explicit override
    or "PBS_O_WORKDIR" in _os.environ         # running inside a PBS job
    or _os.path.isdir("/home/ahernz")         # on the QUT HPC login node
    or bool(_SYNTH_ROOT_ENV)
)

# ---------------------------------------------------------------------------
# Root directories
# ---------------------------------------------------------------------------
if _SYNTH_ROOT_ENV:
    # Explicit root (Gadi scratch, or any relocated tree).
    _ROOT         = _SYNTH_ROOT_ENV
    POP_SYNTH_DIR = os.path.join(_ROOT, "Population_Synth")
    HH_MATCH_DIR  = os.path.join(_ROOT, "Household_Match")
    SA_DIR        = os.path.join(_ROOT, "sa_run_again_n")
elif USE_HPC:
    _ROOT         = "/home/ahernz"
    POP_SYNTH_DIR = os.path.join(_ROOT, "Population_Synth")
    HH_MATCH_DIR  = os.path.join(_ROOT, "Household_Match")
    # SA_DIR is kept ONLY for Stage 3a (sa_multi.py still runs there and
    # writes zones_sa_*/ into that directory).  All other data now lives
    # inside synth_pop/ (HERE).
    SA_DIR        = os.path.join(_ROOT, "sa_run_again_n")
else:
    # On Windows: z:\ is the network home
    _ROOT         = "z:/"
    POP_SYNTH_DIR = os.path.join(_ROOT, "Population_Synth")
    HH_MATCH_DIR  = os.path.join(_ROOT, "Household_Match")
    SA_DIR        = os.path.join(_ROOT, "sa_run_again_n")

# The synth_pop working directory (this folder -- all self-contained scripts live here)
HERE          = os.path.dirname(os.path.abspath(__file__))
SYNTH_POP_DIR = HERE

# ---------------------------------------------------------------------------
# Key input files
# ---------------------------------------------------------------------------
CENSUS_DWELLING = os.path.join(POP_SYNTH_DIR, "CENSUS_2021_BASIC_dwelling.csv")
CENSUS_FAMILY   = os.path.join(POP_SYNTH_DIR, "CENSUS_2021_BASIC_family.csv")
CENSUS_PERSON   = os.path.join(POP_SYNTH_DIR, "CENSUS_2021_BASIC_person.csv")

LINKED_DATA     = os.path.join(POP_SYNTH_DIR, os.environ.get("LINKED_DATA_FILE", "LINKED_DATA.csv"))

# DATGAN output location (written by gan_train.py)
DATGAN_OUTPUT   = os.path.join(POP_SYNTH_DIR, "core_brisbane", "data", "DATGAN.csv")

# Household match output (written by houehold_family_match_c.py + household_repairs_e.py)
HH_MATCH_INPUT  = os.path.join(SYNTH_POP_DIR, "outputs", "hh_match", "DATGAN_samples.csv")
HH_MATCH_OUTPUT = os.path.join(SYNTH_POP_DIR, "outputs", "hh_match", "cleaned_dataset_hhid.csv")

# External mirror locations used by downstream population / tour code.
POPULATION_SA2_DIR = os.path.join("Z:/", "lanuse", "population_sa2")
if USE_HPC:
    ZONE_IDS_SA2_DIR    = os.path.join("/mnt/hpccs01/home/ahernz", "test_runs_tours", "code", "saved_data", "zone_ids_sa2_copy")
    ZONE_ORDER_REF_DIR  = os.path.join("/mnt/hpccs01/home/ahernz", "test_runs_tours", "code", "saved_data", "zone_ids_sa2")
else:
    ZONE_IDS_SA2_DIR    = os.path.join("Z:/", "test_runs_tours", "code", "saved_data", "zone_ids_sa2_copy")
    ZONE_ORDER_REF_DIR  = os.path.join("Z:/", "test_runs_tours", "code", "saved_data", "zone_ids_sa2")
# Target directory for the freshly generated order_N.csv exports (tour model input).
# Same layout as ZONE_ORDER_REF_DIR (order_N.csv per zone) so build_synth_population.py /
# census_bridge.py can consume it as a drop-in "redone" replacement for zone_ids_sa2.
ZONE_ORDER_EXPORT_DIR = ZONE_IDS_SA2_DIR

# ---------------------------------------------------------------------------
# new_pop: 507 SA2-zone scope (demand_'s zone_ids_sa_only_507.txt), keyed
# directly by SA2 code (no ORDER indirection). Override root with env var
# NEW_POP_DIR.
# ---------------------------------------------------------------------------
NEW_POP_DIR = os.environ.get(
    "NEW_POP_DIR",
    os.path.join("/mnt/hpccs01/home/ahernz", "test_runs_tours", "new_pop") if USE_HPC
    else os.path.join("Z:/", "test_runs_tours", "new_pop"),
)
NEW_POP_SAVED_DATA_DIR = os.path.join(NEW_POP_DIR, "saved_data")
ZONE_IDS_SA2_507_DIR   = os.path.join(NEW_POP_SAVED_DATA_DIR, "zone_ids_sa2")
ZONE_LIST_507_PATH     = os.path.join(SYNTH_POP_DIR, "zone_ids_sa_only_507.txt")

# ---------------------------------------------------------------------------
# SA / microdata inputs -- ALL inside synth_pop/ (Paper_Plan), not SA_DIR
# ---------------------------------------------------------------------------
# IPU-Only / SA-Only seed.
#
# microdata_for_exp.csv is a 244,536-row slice whose REGUCP codes only span
# 31-42, while csf_sa.csv maps zones to the full 1-60 range. Every zone outside
# that slice therefore missed its REGUCP lookup and fell back to the whole pool
# ("REGUCP 4.0 not found in GAN pool"), which is both a correctness bug and the
# main driver of the 4-hour per-zone IPU timeouts.
#
# LINKED_DATA_national.csv is the fused national census (1.25M rows, REGUCP
# 1-60) produced by step_0b_fuse_national.py. Prefer it; allow an env override.
# LINKED_DATA_clean.csv is LINKED_DATA.csv with the ABSHID_x/ABSHID_y merge
# artefacts collapsed (see Population_Synth/repair_linked_keys.py): same
# 1.25M rows, clean ABSHID, full REGUCP 1-60, and it still carries ABSPID.
# LINKED_DATA_national.csv covers REGUCP 1-60 too but drops ABSPID, which
# SA_2_level.py requires -> every IPU-Only zone failed with
# "missing required columns: ['abspid']".
_CLEAN_CENSUS = os.path.join(POP_SYNTH_DIR, "LINKED_DATA_clean.csv")
_NATIONAL_CENSUS = os.path.join(POP_SYNTH_DIR, "LINKED_DATA_national.csv")
if os.path.exists(_CLEAN_CENSUS):
    _DEFAULT_SEED = _CLEAN_CENSUS
elif os.path.exists(_NATIONAL_CENSUS):
    _DEFAULT_SEED = _NATIONAL_CENSUS
else:
    _DEFAULT_SEED = os.path.join(SYNTH_POP_DIR, "microdata_for_exp.csv")
CENSUS_MICRODATA = os.environ.get("CENSUS_MICRODATA_FILE", _DEFAULT_SEED)

# SA2 control tables — flat layout (all CSVs directly in data_sa2_controls/)
# Override to reproduce a specific paper basis, e.g. the 19 in-sample tables the
# manuscript reports (data_sa2_controls_paper19). Default: the full live set.
SA_CONTROL_DIR = os.environ.get(
    "SA_CONTROL_DIR", os.path.join(SYNTH_POP_DIR, "data_sa2_controls"))

# ---------------------------------------------------------------------------
# Hold-out declaration (must match pbs/paper_analysis.pbs)
# ---------------------------------------------------------------------------
# The paper fits on 19 in-sample tables and evaluates out-of-sample on 7 held-out
# tables. The live control directory holds all 26, so a fit that reads it directly
# silently optimises against the held-out tables and destroys the out-of-sample
# claim. These names let the loaders detect that case instead of trusting the
# environment to be set correctly.
IN_SAMPLE_TABLES = ["indp", "sexp", "typp", "mtwp", "occp", "mstp", "incp",
                    "stup", "rlhp", "hrsp", "heap", "qallp", "qalfp",
                    "rntrd", "mrerd", "vehrd", "nprd", "hind", "strd"]
HELD_OUT_TABLES = ["age5p", "assnp", "bedrd", "hcfmf", "hhcd", "lfsp", "tenlld"]


def holdout_violations(directory=None):
    """Return the held-out tables present in a control directory.

    Empty list means the directory is the in-sample basis.
    """
    d = directory or SA_CONTROL_DIR
    if not os.path.isdir(d):
        return []
    present = {f.lower().replace(".csv", "") for f in os.listdir(d)
               if f.lower().endswith(".csv")}
    return sorted(t for t in HELD_OUT_TABLES if f"sa2{t}" in present)
# ---------------------------------------------------------------------------
# QALFP: control categories the microdata cannot distinguish
# ---------------------------------------------------------------------------
# ABS 2021 QALFP (2-digit) has 16 categories: 12 fields of study plus the
# supplementary codes 00 (inadequately described), && (not stated), @@
# (not applicable) and VV (overseas visitor)
# (https://www.abs.gov.au/census/guide-census-data/census-dictionary/2021/
#  variables-topic/education-and-training/non-school-qualification-field-study-qalfp).
# The fused microdata column holds 15 codes: everything except Mixed Field
# Programmes, which is folded into 'inadequately described'. The evidence is the
# modal category, which sits at 14 in the microdata and at 15 in the control
# tables (47.9% vs 48.9% for Tweed), and code 12, which carries 0.6% of donors
# against 0.09% (Mixed Field) + 0.69% (inadequately described) in the targets.
#
# Consequence: control code 12 has no possible donor at all, and a fit can never
# place mass there however the pool is built.
#
# Two different questions follow, and they have different answers:
#  * COVERAGE -- the pool builder must not demand a category no census record can
#    carry, or every rebuild reports an unfillable cell. Merge there (below).
#  * SCORING -- merging changes the basis for every arm, so it was measured rather
#    than assumed: a paired re-score of the four published arms
#    (outputs/qalfp_rescore_merged.csv) found merging RAISES qalfp RSSZ in 81% of
#    zones, by 0.0008-0.0030 (0.7-2.3%). The unattainable cell was therefore not
#    inflating the residual, and the published scoring basis is left alone.
# Set SA_QALFP_MERGE_FOR_COVERAGE=0 to make the builder demand all 16 categories.
QALFP_MERGE_FOR_COVERAGE = os.environ.get(
    "SA_QALFP_MERGE_FOR_COVERAGE", "1") not in ("0", "", "false", "False")
QALFP_MERGED_CATEGORIES = {"qalfp": {12: 13}}


SA_CONTROL_IND = os.path.join(SA_CONTROL_DIR, "individual")   # exists only if manually split
SA_CONTROL_HH  = os.path.join(SA_CONTROL_DIR, "household")    # exists only if manually split

# New attribute control tables (EV type, Solar installation) - generated from external data
SA2_VEH_TYPE = os.path.join(SA_CONTROL_DIR, "sa2vehtype.csv")
SA2_SOLAR_TYPE = os.path.join(SA_CONTROL_DIR, "sa2solartype.csv")

# SA1 control tables (used for optional benchmark reruns at SA1 geography)
SA1_CONTROL_DIR = os.path.join(SYNTH_POP_DIR, "data_sa1_controls")

# ---------------------------------------------------------------------------
# Output directories (all under synth_pop/outputs/ so everything is local)
# ---------------------------------------------------------------------------
OUTPUTS_DIR         = os.path.join(SYNTH_POP_DIR, "outputs")

# SA-GAN results -- written locally by Stage 3a (sa_multi_wrapper.py calls SA_2_level.py)
# Override with env var SA_GAN_RESULTS_DIR to write to a different folder (e.g. for 507-zone run).
SA_GAN_RESULTS_DIR = os.environ.get(
    "SA_GAN_RESULTS_DIR",
    os.path.join(OUTPUTS_DIR, "sa_gan_results"),
)

GAN_VALIDATION_DIR  = os.path.join(OUTPUTS_DIR, "gan_validation")
SA_ONLY_RESULTS_DIR = os.environ.get(
    "SA_ONLY_RESULTS_DIR", os.path.join(OUTPUTS_DIR, "sa_only_results"))
# Result directories are overridable so a corrected rerun can write to a
# versioned directory instead of overwriting the previous (invalid) run.
CP_POOL_RESULTS_DIR = os.environ.get(
    "CP_POOL_RESULTS_DIR", os.path.join(OUTPUTS_DIR, "cp_pool_results"))
CP_IPU_RESULTS_DIR  = os.environ.get(
    "CP_IPU_RESULTS_DIR", os.path.join(OUTPUTS_DIR, "cp_ipu_results"))
CP_MIP_RESULTS_DIR  = os.environ.get(
    "CP_MIP_RESULTS_DIR", os.path.join(OUTPUTS_DIR, "cp_mip_results"))
GA_NSA_RESULTS_DIR  = os.path.join(OUTPUTS_DIR, "ga_nsa_results")
IPU_GAN_RESULTS_DIR  = os.path.join(OUTPUTS_DIR, "ipu_gan_results")
IPU_ONLY_RESULTS_DIR = os.environ.get(
    "IPU_ONLY_RESULTS_DIR", os.path.join(OUTPUTS_DIR, "ipu_only_results"))
RSSZ_OUTLIER_DIR    = os.path.join(OUTPUTS_DIR, "rssz_outlier")
HH_MATCH_OUT_DIR    = os.path.join(OUTPUTS_DIR, "hh_match")

# Legacy alias for scripts that reference the old var name
GA_IPU_RESULTS_DIR = IPU_GAN_RESULTS_DIR

# ---------------------------------------------------------------------------
# HPC environment (used only by submit_pipeline.sh, not by Python scripts)
# ---------------------------------------------------------------------------
CONDA_PROFILE = os.environ.get(
    "CONDA_PROFILE",
    "/home/ahernz/miniconda3/etc/profile.d/conda.sh",
)
ENV_GAN       = "popgangpu"
ENV_SA        = "syn_env"

# ---------------------------------------------------------------------------
# GAN training hyperparameters
# ---------------------------------------------------------------------------
GAN_EPOCHS      = 300     # 300 epochs; use 10 for a quick test
GAN_BATCH_SIZE  = 1116    # ~1% of dataset; reported in paper
GAN_SAMPLE_MULT = 10      # synthetic records = SAMPLE_MULT x original size
GAN_DATA_SOURCE = "census"  # "census" or "hhts"
# Mathematical (conditional-resampling) repair is the default: corrections are
# drawn from P(attribute | parents) in the training data restricted to feasible
# values instead of collapsing to a single "not applicable" code. Set False (or
# env GAN_FEASIBLE_RESAMPLE=0 / --no-feasible-resample) for legacy recoding.
GAN_FEASIBLE_RESAMPLE = True

# ---------------------------------------------------------------------------
# SA run parameters (Reviewer 2, P21: tolerance parameters defined here)
# ---------------------------------------------------------------------------
SA_MAX_ITER     = 1000
SA_INITIAL_TEMP = 100.0
SA_COOLING_RATE = 0.95
# Household controls are reconciled to HHCD when present. Set this explicitly
# for replication runs so the universe choice is recorded with the experiment.
SA_HOUSEHOLD_UNIVERSE = os.environ.get("SA_HOUSEHOLD_UNIVERSE", "reconciled")
# Reporting floor for paper statistics (see reporting_scope.py).
#
# Zones whose census population is below this are still synthesised and still
# exported for the tour model; they are only excluded from the reported
# marginal-fit statistics, where expected cell counts of well under 1 make TVD
# and SRMSE meaningless.  In the current study area the sizes jump 78 -> 391 ->
# 602 -> 851 persons, so any floor in roughly 400-850 selects the same zones.
MIN_REPORT_PERSONS = int(os.environ.get("MIN_REPORT_PERSONS", "500"))

# tol_k,m -- per-table tolerance fraction (C3 in the paper):
#   a zone is considered fitted for table k,m if RSSZ(k,m) < SA_TOLERANCE
# tol_k    -- aggregate tolerance (C4): mean RSSZ across all tables < SA_TOLERANCE
SA_TOLERANCE    = 0.05    # 5% -- chosen to allow minor rounding without inflating RSSZ
