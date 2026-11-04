"""Align a synthesis seed with the ABS control coding and make it category-complete.

The SA / IPU branches fit a seed population against the SA2 control tables. Two
defects make that impossible for whole tables: the fused seed is one code behind
the ABS QALFP coding above 11, and a ~5% sample simply has no household for some
categories the marginals require. `SEED` reports the unmatched mass for those
cells, and SA_2_level's RSSZ does not score them at all, so the search looks like
it has converged when it has only converged on the part it can see.

This applies the same two fixes the CP pool builder uses
(`build_feasible_pool_cp.align_qalfp_to_control_codes` and
`complete_control_coverage`) to any seed CSV.

  python prepare_sa_seed.py --seed cleaned_dataset_hhid.csv \
      --out outputs/seeds/cleaned_dataset_hhid_aligned.csv \
      --census /path/LINKED_DATA_national.csv --coverage

Without --census, donors for missing categories are drawn from the seed itself,
so categories absent from the seed stay absent (and are reported as such).
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_feasible_pool_cp import (
    align_qalfp_to_control_codes,
    complete_control_coverage,
    control_coverage,
    load_required_categories,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", required=True, help="Seed population CSV (household structure required)")
    ap.add_argument("--out", required=True, help="Where to write the aligned seed")
    ap.add_argument("--census", default=None,
                    help="Census CSV used as donor source for missing categories")
    ap.add_argument("--coverage", action="store_true",
                    help="Add households for control categories the seed cannot represent")
    ap.add_argument("--coverage-per-category", type=int, default=5)
    ap.add_argument("--no-align", action="store_true", help="Skip the QALFP recode")
    ap.add_argument("--seed-int", type=int, default=0, help="RNG seed for donor sampling")
    a = ap.parse_args()

    seed = pd.read_csv(a.seed, low_memory=False)
    print(f"seed: {len(seed):,} rows x {seed.shape[1]} cols from {a.seed}")
    before = None
    if a.coverage:
        required, hh_attrs = load_required_categories()
        before = control_coverage(seed, required)
        gaps = {k: v["missing"] for k, v in before.items() if v["missing"]}
        print(f"coverage before: {sum(len(v) for v in gaps.values())} missing "
              f"categories in {len(gaps)} attributes")

    if a.coverage:
        donors = pd.read_csv(a.census, low_memory=False) if a.census else seed
        if a.census:
            print(f"donor census: {len(donors):,} rows")
        seed, stats = complete_control_coverage(
            seed, donors, required, hh_attrs=hh_attrs,
            per_category=a.coverage_per_category, seed=a.seed_int)
        print(f"coverage added {stats['added_rows']:,} rows / "
              f"{stats['added_households']:,} households "
              f"(copied {stats['copied']}, patched {stats['patched']})")
        for item in stats["unsourceable"]:
            print(f"  unsourceable: {item}")
        for item in stats["coverage_gaps"]:
            print(f"  unfillable : {item}")
    # Alignment comes last: coverage sources categories in census coding, and the
    # recode is what turns them into the control-table codes.
    if not a.no_align:
        seed, moved = align_qalfp_to_control_codes(seed)
        print(f"QALFP recoded for {moved:,} rows (codes >= 12 shifted +1)")

    if a.coverage:
        after = control_coverage(seed, required)
        left = {k: v["missing"] for k, v in after.items() if v["missing"]}
        if left:
            print("still missing after alignment (these cells cannot be fitted at all):")
            for attr, cats in left.items():
                print(f"  {attr}: {cats}")
        else:
            print("coverage: every control category has a donor")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    seed.to_csv(a.out, index=False)
    print(f"wrote {len(seed):,} rows to {a.out}")


if __name__ == "__main__":
    main()
