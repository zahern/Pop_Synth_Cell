"""
Hard-constraint rules shared by the GAN repairer and the benchmark.

`gan_train.py` repairs raw GAN output against these rules, `gan_validation.py`
reports the pre/post-repair rate, and `benchmark_margins.py` applies the same
rules to every strategy so the "repair rate" column is directly comparable
across methods.

Two sets are defined.  `CONSTRAINT_CHECKS` is per-record: every predicate reads
one row at a time, which is what a repairer needs.  `HOUSEHOLD_CHECKS` needs the
members of one household together (a size code cannot be checked from a single
row), so it is applied where a household is validated or audited rather than by
the per-record repairer.

The category constants are imported from `constrained_sampling` (the module the
repairer and the CP pool builder actually use), so this file cannot drift out
of sync with them.  Only numpy/pandas are needed — no sklearn/scipy — so the
benchmark can use the rules without pulling in the whole GAN validation stack.

ABS coding notes
----------------
AGEP codes 0-14 are children aged 0-14.
EMPP codes 1,2,3 = employed; 4 = Not stated; 5 = Not applicable (child / not
    employed); 6 = Overseas visitor.  Children must be in {5,6}.
LFSP codes 1-4 = in labour force; 5 = Not applicable (child); 6 = overseas.
INCP codes 18 = Not applicable, 19 = Overseas visitor (children must be 18/19).
MSTP code 6 = Not applicable (children <15).
QALFP code 14 is the modal *resident* code carried by fused/source children
    (225,437/225,582 kids), so children legitimately carry {14, 15}.
CACF code 7 is a residual bucket, not a count — it must be excluded from the
    CACF <= CPRF comparison (62% of source families carry it).
"""

import pandas as pd

try:                                    # single source of truth
    from constrained_sampling import (
        AGEP_CHILD_THRESHOLD,
        EMPP_CHILD_VALID,
        LFSP_CHILD_VALID,
        QALFP_CHILD_VALID,
        QALLP_NOT_APPLIC,
        QALLP_OVERSEAS,
        OCCP_NOT_APPLIC,
        OCCP_OVERSEAS,
        HEAP_NOT_APPLIC,
        HEAP_OVERSEAS,
        INCP_NOT_APPLIC,
        INCP_OVERSEAS,
        MSTP_NOT_APPLIC,
        HRSP_NOT_APPLIC,
        EMPP_APPLICABLE_LFSP,
        CACF_RESIDUAL,
        RLHP_CHILD_VALID,
    )
except Exception:                       # pragma: no cover - defensive fallback
    AGEP_CHILD_THRESHOLD = 15
    EMPP_CHILD_VALID = (5, 6)
    LFSP_CHILD_VALID = (5, 6)
    QALFP_CHILD_VALID = (14, 15)
    QALLP_NOT_APPLIC, QALLP_OVERSEAS = 8, 9
    OCCP_NOT_APPLIC, OCCP_OVERSEAS = 11, 12
    HEAP_NOT_APPLIC, HEAP_OVERSEAS = 11, 12
    INCP_NOT_APPLIC, INCP_OVERSEAS = 18, 19
    MSTP_NOT_APPLIC = 6
    HRSP_NOT_APPLIC = 1
    EMPP_APPLICABLE_LFSP = (1,)
    CACF_RESIDUAL = 7
    RLHP_CHILD_VALID = (3, 9, 10, 11)

# HIND/HINASD not-stated codes (from gan_train.py)
HIND_NOT_STATED = (24, 25)
HINASD_NOT_STATED = (23, 24)

MRERD_NOT_APPLICABLE = 23
RNTRD_NOT_APPLICABLE = 27

# LFSP values outside the labour-force question that still legitimately carry
# hours worked.  5 = not applicable (<15, the question is not asked); 6 =
# overseas visitor.  In a 400k-row census slice 788 rows are LFSP=6 with
# HRSP=10, every one of them RLHP=11, i.e. the fused microdata records hours
# for visitors as a rule and not as an error, so treating them as a
# contradiction flags the input data rather than a defect in a synthesis.
LFSP_HOURS_EXEMPT = (5, 6)


def _false(df):
    return pd.Series(False, index=df.index)


CONSTRAINT_CHECKS = {
    # Child EMPP must be not-applicable / overseas; codes 1,2,3=employed and
    # 4=Not stated are invalid for a child.
    "child_bad_empp (AGEP<15, EMPP not in {5,6})":
        lambda df: (df["AGEP"] < AGEP_CHILD_THRESHOLD)
        & (~df["EMPP"].isin(list(EMPP_CHILD_VALID)))
        if "AGEP" in df.columns and "EMPP" in df.columns
        else _false(df),

    # Child HRSP (hours worked) must be 1 (0 hrs / not applicable) — B1 rule.
    "child_hours_worked (AGEP<15, HRSP != 1)":
        lambda df: (df["AGEP"] < AGEP_CHILD_THRESHOLD)
        & (df["HRSP"] != HRSP_NOT_APPLIC)
        if "AGEP" in df.columns and "HRSP" in df.columns
        else _false(df),

    # Child RLHP (relationship in household) must be in RLHP_CHILD_VALID.
    "child_relationship (AGEP<15, RLHP not in {3,9,10,11})":
        lambda df: (df["AGEP"] < AGEP_CHILD_THRESHOLD)
        & (~df["RLHP"].isin(list(RLHP_CHILD_VALID)))
        if "AGEP" in df.columns and "RLHP" in df.columns
        else _false(df),

    # EMPP (status in employment) is only applicable to employed persons
    # (LFSP=1). The GAN spills EMPP onto ~19% of not-employed records — the
    # largest driver of its marginal error (TVD 0.24->0.05 once repaired).
    "empp_without_employment (LFSP!=1, EMPP not in {5,6})":
        lambda df: (~df["LFSP"].isin(list(EMPP_APPLICABLE_LFSP)))
        & (~df["EMPP"].isin(list(EMPP_CHILD_VALID)))
        if "LFSP" in df.columns and "EMPP" in df.columns
        else _false(df),

    # HIND (household income) and HINASD encode the same income; their
    # not-stated status agrees with 0 exceptions in the microdata
    # (HINASD in {23,24} <=> HIND in {24,25}).
    "hind_hinasd_mismatch (HIND not-stated != HINASD not-stated)":
        lambda df: (df["HIND"].isin(HIND_NOT_STATED)
                    != df["HINASD"].isin(HINASD_NOT_STATED))
        if "HIND" in df.columns and "HINASD" in df.columns
        else _false(df),

    # HRSP (hours worked) is only applicable to employed persons (LFSP=1).
    "hrsp_without_employment (LFSP!=1, HRSP!=1)":
        lambda df: (~df["LFSP"].isin(list(EMPP_APPLICABLE_LFSP)
                                     + list(LFSP_HOURS_EXEMPT)))
        & (df["HRSP"] != HRSP_NOT_APPLIC)
        if "LFSP" in df.columns and "HRSP" in df.columns
        else _false(df),

    "child_in_labour_force (AGEP<15, LFSP not in {5,6})":
        lambda df: (df["AGEP"] < AGEP_CHILD_THRESHOLD)
        & (~df["LFSP"].isin(list(LFSP_CHILD_VALID)))
        if "AGEP" in df.columns and "LFSP" in df.columns
        else _false(df),

    "child_with_income (AGEP<15, INCP not in {18,19})":
        lambda df: (df["AGEP"] < AGEP_CHILD_THRESHOLD)
        & (~df["INCP"].isin([INCP_NOT_APPLIC, INCP_OVERSEAS]))
        if "AGEP" in df.columns and "INCP" in df.columns
        else _false(df),

    "child_with_occupation (AGEP<15, OCCP not in {11,12})":
        lambda df: (df["AGEP"] < AGEP_CHILD_THRESHOLD)
        & (~df["OCCP"].isin([OCCP_NOT_APPLIC, OCCP_OVERSEAS]))
        if "AGEP" in df.columns and "OCCP" in df.columns
        else _false(df),

    "child_with_education (AGEP<15, HEAP not in {11,12})":
        lambda df: (df["AGEP"] < AGEP_CHILD_THRESHOLD)
        & (~df["HEAP"].isin([HEAP_NOT_APPLIC, HEAP_OVERSEAS]))
        if "AGEP" in df.columns and "HEAP" in df.columns
        else _false(df),

    # QALFP 14 is the modal resident code for fused children, not just 15.
    "child_with_qualification (AGEP<15, QALFP not in {14,15})":
        lambda df: (df["AGEP"] < AGEP_CHILD_THRESHOLD)
        & (~df["QALFP"].isin(list(QALFP_CHILD_VALID)))
        if "AGEP" in df.columns and "QALFP" in df.columns
        else _false(df),

    "child_with_degree (AGEP<15, QALLP not in {8,9})":
        lambda df: (df["AGEP"] < AGEP_CHILD_THRESHOLD)
        & (~df["QALLP"].isin([QALLP_NOT_APPLIC, QALLP_OVERSEAS]))
        if "AGEP" in df.columns and "QALLP" in df.columns
        else _false(df),

    "child_married (AGEP<15, MSTP != 6)":
        lambda df: (df["AGEP"] < AGEP_CHILD_THRESHOLD)
        & (df["MSTP"] != MSTP_NOT_APPLIC)
        if "AGEP" in df.columns and "MSTP" in df.columns
        else _false(df),

    # CACF==7 is a residual bucket, not a count — comparing it numerically
    # flags ordinary couple families (all 271k source "violations" are CACF==7).
    "CACF > CPRF (children exceed family size)":
        lambda df: (df["CACF"] > df["CPRF"]) & (df["CACF"] != CACF_RESIDUAL)
        if "CACF" in df.columns and "CPRF" in df.columns
        else _false(df),

    "HIND==1 (negative income) but MRERD active":
        lambda df: (df["HIND"] == 1) & (df["MRERD"] != MRERD_NOT_APPLICABLE)
        if "HIND" in df.columns and "MRERD" in df.columns
        else _false(df),

    # Both tenure fields set to active codes simultaneously.
    # Uses != not-applicable sentinels (NOT > 0 — N/A codes 27 and 23 are > 0)
    "renter_with_mortgage (RNTRD and MRERD both active)":
        lambda df: (df["RNTRD"] != RNTRD_NOT_APPLICABLE)
        & (df["MRERD"] != MRERD_NOT_APPLICABLE)
        if "RNTRD" in df.columns and "MRERD" in df.columns
        else _false(df),
}


# ---------------------------------------------------------------------------
# Household-level checks
# ---------------------------------------------------------------------------
# These need every member of one household, so they cannot be written as the
# per-record predicates above: a repairer rewrites one record at a time and
# cannot make a household's size code agree with its members by editing a
# person.  They are still structural -- a household whose size code contradicts
# its members is not one the census could have produced -- so they are applied
# wherever a household is assembled and reported separately from the
# person-level rule set.

# NPRD coding in the fused microdata is not a plain count at the top: 1-5 are
# exact, 6 is the top code for a private dwelling (six or more), and 7 is "not
# applicable" -- it appears only in visitors-only and other non-classifiable
# dwellings (HHCD 5/6), whose usual-resident size is not a number.  The census
# control tables instead use 7 and 8 as open-ended counts.
NPRD_EXACT_MAX = 5
NPRD_TOP_CODE = 6
NPRD_NOT_APPLICABLE = 7
NPRD_EIGHT_PLUS = 8
NPRD_NOT_APPLICABLE_HHCD = (5, 6)


def nprd_matches_members(df):
    """Rows whose household size code (NPRD) contradicts the member count.

    A household of n members may not carry an exact count other than n.  The
    codes are read under both codings the pipeline meets: 6 needs at least six
    members, 7 needs at least seven *or* a dwelling with no usual-resident size
    (visitors-only / non-classifiable), and 8 needs at least eight.  A missing
    or unrecognised code is a violation.  Households are identified by ABSHID;
    without it (or NPRD) the check reports nothing rather than guessing, since
    household size is not defined there.
    """
    if "NPRD" not in df.columns or "ABSHID" not in df.columns:
        return _false(df)
    nprd = pd.to_numeric(df["NPRD"], errors="coerce")
    size = df.groupby("ABSHID")["ABSHID"].transform("size")
    bad = nprd.isna() | ~nprd.isin([1, 2, 3, 4, 5, NPRD_TOP_CODE,
                                    NPRD_NOT_APPLICABLE, NPRD_EIGHT_PLUS])
    bad |= (nprd >= 1) & (nprd <= NPRD_EXACT_MAX) & (nprd != size)
    bad |= (nprd == NPRD_TOP_CODE) & (size < NPRD_TOP_CODE)
    bad |= (nprd == NPRD_EIGHT_PLUS) & (size < NPRD_EIGHT_PLUS)
    no_usual_residents = pd.Series(False, index=df.index)
    if "HHCD" in df.columns:
        no_usual_residents = pd.to_numeric(
            df["HHCD"], errors="coerce").isin(list(NPRD_NOT_APPLICABLE_HHCD))
    bad |= ((nprd == NPRD_NOT_APPLICABLE) & ~no_usual_residents
            & (size < NPRD_NOT_APPLICABLE))
    return (bad & size.notna()).fillna(False)


HOUSEHOLD_CHECKS = {
    "nprd_matches_members (NPRD disagrees with the persons in the household)":
        nprd_matches_members,
}
