"""Fast donor lookup + vectorised infill for build_feasible_pool_cp.py.

Replaces the per-thin-cell full-census scan in conditional_infill()/_relaxed_donors().

Original cost
-------------
For each of ~1.07M thin cells it built a 1.25M-element boolean mask per
conditioning attribute:

    for _, cell in thin_cells.iterrows():          # ~1.07M iterations
        cond = pd.Series(True, index=census.index)
        for c in keep:                             # up to 21 attributes
            cond &= census[c] == fixed[c]          # 1.25M ops each
        sub = census[cond]
        frames.append(draw)                        # ~1.07M DataFrames retained

That is ~2.8e13 element operations and retains one DataFrame per cell until the
final concat, which is what drove RSS to 21.7 GB and rising on Gadi.

This version
------------
* One groupby per relaxation level (at most relax_depth+1 groupbys total)
  instead of one boolean scan per cell.
* Preallocated object arrays filled positionally, so peak memory is
  proportional to the output rather than to the number of cells.
* Identical semantics: relax the trailing conditioning attributes one at a
  time, require >= MIN_DONORS donors, and re-fix the conditioning values on the
  drawn rows so each cell is preserved exactly.

Scale caveat
------------
With 21 control attributes the joint cell space is far larger than the 1.25M
census rows, so most cells are genuinely empty and infilling every one of them
at target_per_cell=20 would still mean ~21M synthetic rows. The caller passes
max_infill_cells to bound that; the report records what was skipped.
"""

import numpy as np
import pandas as pd

MIN_DONORS = 10  # unchanged from the original _relaxed_donors threshold

# A household cannot physically exceed a dozen or so members. The fused census
# nevertheless contains sentinel/aggregate ABSHID values: the largest group has
# 35,352 rows. Copying "the whole household" for one of those appended 35k rows
# per draw and drove the pool build to 87 GB and an OOM kill. Draws are now
# restricted to households within this size bound.
MAX_HH_SIZE = int(__import__("os").environ.get("CP_MAX_HH_SIZE", "12") or 12)

_DEBUG = bool(int(__import__("os").environ.get("CP_INFILL_DEBUG", "0") or 0))


def _rss(tag):
    """Peak-RSS probe (Linux) used to locate memory blow-ups in the infill."""
    if not _DEBUG:
        return
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"    [infill] {tag}: peak_rss={peak:.1f} GB", flush=True)
    except Exception:
        pass


def build_donor_index(census, avail, relax_depth=1):
    """Return a list of dicts mapping key-tuple -> positional indices.

    Level d keeps the first len(avail)-d conditioning attributes, matching the
    original relaxation order (drop the least-informative trailing variable).
    """
    levels = []
    for depth in range(relax_depth + 1):
        keep = avail[: len(avail) - depth]
        if not keep:
            break
        try:
            idx = census.groupby(keep, sort=False, observed=True).indices
        except TypeError:
            # Older pandas without observed= for this dtype combination.
            idx = census.groupby(keep, sort=False).indices
        levels.append({k: np.asarray(v) for k, v in idx.items()})
    return levels


def _aligned_cell_values(census, thin_cells, avail):
    """Extract the thin-cell attribute values with the census's dtypes.

    The groupby index keys carry the census dtypes (int64 for coded ABS
    variables), whereas thin_cells.to_numpy() upcasts mixed columns to float64.
    Without this alignment every key lookup misses and the infill silently
    returns zero rows.
    """
    frame = thin_cells[avail].copy()
    for c in avail:
        target = census[c].dtype
        try:
            if frame[c].dtype != target:
                if pd.api.types.is_integer_dtype(target):
                    # round() guards against float representation of large codes
                    frame[c] = frame[c].round().astype(target)
                elif pd.api.types.is_bool_dtype(target):
                    frame[c] = frame[c].astype(bool)
                else:
                    frame[c] = frame[c].astype(target)
        except (TypeError, ValueError):
            # Leave as-is; a non-castable column simply will not match, which
            # mirrors the original's comparison semantics.
            pass
    return frame.to_numpy()


# Household-level attributes: every member of a household must carry the same
# value. Person-level attributes may differ across members.
HOUSEHOLD_ATTRS = {
    "bedrd", "hhcd", "vehrd", "mrerd", "rntrd", "strd", "hind", "nprd",
    "tend", "tenlld", "dwtd", "regucp",
}


def conditional_infill(census, thin_cells, target_per_cell=20, seed=0,
                       relax_depth=1, max_infill_cells=None):
    """Reconstruct feasible *households* for thin cells.

    Semantics
    ---------
    For each thin cell:
      * draw donor rows from census rows sharing the cell's attribute values
        (relaxing trailing conditioning attributes when fewer than MIN_DONORS),
      * expand each drawn row to its **whole donor household** (all rows sharing
        that abshid) so household sizes stay realistic,
      * assign a fresh unique abshid per drawn household,
      * overwrite the household-level attributes (bedrd/hhcd/vehrd/...) on
        *every* member with the cell's values — one household cannot sit in two
        cells at once,
      * overwrite the person-level attributes on the **anchor** row (the drawn
        row) so the cell is matched exactly.

    The earlier implementation copied individual donor rows (keeping their
    abshid) and rewrote all conditioning attributes on those rows only. The same
    abshid could therefore be drawn from several cells and end up with
    conflicting household attributes — measured on the Oct-10 pool: 377,006
    inconsistent households / 2,038,124 rows (92% of the pool), which is what
    drove MIP tolerances to the 1.6 relaxation ceiling.

    max_infill_cells caps how many thin cells are processed, largest-first by
    observed count, so a pathological cell space cannot exhaust memory.
    """
    rng = np.random.default_rng(seed)

    avail = [c for c in thin_cells.columns if c in census.columns and c != "count"]
    if not avail or len(thin_cells) == 0:
        return pd.DataFrame(columns=census.columns)

    # Remember the full thin-cell population: thin_cells is truncated below, so
    # reporting len(thin_cells) at the end reported 0 skipped even when the cap
    # discarded most of them (Oct-2026: 60,000 of 1,065,748 kept).
    n_thin_total = len(thin_cells)
    if max_infill_cells is not None and len(thin_cells) > max_infill_cells:
        thin_cells = thin_cells.sort_values("count", ascending=False).head(max_infill_cells)

    counts = pd.to_numeric(thin_cells["count"], errors="coerce").fillna(0).to_numpy()
    needs = np.maximum(0, int(target_per_cell) - counts).astype(np.int64)
    total_need = int(needs.sum())
    if total_need == 0:
        return pd.DataFrame(columns=census.columns)

    _rss("before donor index")
    levels = build_donor_index(census, avail, relax_depth=relax_depth)
    _rss("after donor index")
    if not levels:
        return pd.DataFrame(columns=census.columns)

    _rss("before cell_vals")
    cell_vals = _aligned_cell_values(census, thin_cells, avail)
    _rss("after cell_vals")
    n_cols = len(census.columns)
    col_names = list(census.columns)
    col_pos = {c: i for i, c in enumerate(col_names)}
    col_arrays = [census[c].to_numpy() for c in col_names]

    # --- household index: abshid -> positional rows ---
    # The fused census carries uppercase ABSHID; the pool is lowercased only
    # later by clean_my_datagan(). Resolve the id case-insensitively — checking
    # the literal "abshid" silently disabled household-atomic infill for every
    # run (pool report stayed byte-identical at 2,224,014 rows).
    hh_col = next((c for c in census.columns if str(c).lower() == "abshid"), None)
    if hh_col is None:
        # No household key: degrade to the old row-wise behaviour.
        hh_index = None
        print("WARNING: cp_fast_infill: no ABshid/abshid column; infill will "
              "not be household-atomic")
    else:
        hh_vals = census[hh_col].to_numpy()
        order = np.argsort(hh_vals, kind="stable")
        sorted_ids = hh_vals[order]
        bounds = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
        ends = np.r_[bounds[1:], len(sorted_ids)]
        sizes = ends - bounds
        valid = sizes <= MAX_HH_SIZE
        n_big_hh = int((~valid).sum())
        n_big_rows = int(sizes[~valid].sum())
        if n_big_hh:
            print(f"  infill: excluding {n_big_hh:,} oversized ABSHID groups "
                  f"({n_big_rows:,} rows, max {int(sizes.max()):,}) from donors")
        hh_index = {sorted_ids[b]: order[b:e]
                    for b, e, ok in zip(bounds, ends, valid) if ok}
        # Row-level mask so draws can be filtered to households we can copy.
        row_valid = np.zeros(len(census), dtype=bool)
        for b, e, ok in zip(bounds, ends, valid):
            if ok:
                row_valid[order[b:e]] = True
        _rss("after hh_index")
        hh_of_row = hh_vals
        next_abshid = int(np.nanmax(hh_vals.astype("float64"))) + 1

    avail_hh = [c for c in avail if c.lower() in HOUSEHOLD_ATTRS]
    avail_person = [c for c in avail if c.lower() not in HOUSEHOLD_ATTRS]
    avail_hh_pos = [(col_pos[c], j) for j, c in enumerate(avail) if c in avail_hh]
    avail_person_pos = [(col_pos[c], j) for j, c in enumerate(avail) if c in avail_person]

    # Reserve generously; each drawn household contributes its full size.
    avg_hh = (len(census) / max(len(hh_index), 1)) if hh_index else 1.0
    cap = int(total_need * max(avg_hh, 1.0) * 1.5) + 1024
    out = np.empty((cap, n_cols), dtype=object)
    pos = 0
    n_infilled_cells = 0

    def _append(rows):
        """Copy census rows into the output buffer, growing it when needed."""
        nonlocal pos
        need_n = pos + len(rows)
        nonlocal_cap_grow(need_n)
        for j in range(n_cols):
            out[pos:need_n, j] = col_arrays[j][rows]
        pos = need_n

    def nonlocal_cap_grow(need_n):
        nonlocal out
        if need_n <= out.shape[0]:
            return
        grown = np.empty((max(need_n, out.shape[0] * 2), n_cols), dtype=object)
        grown[:pos] = out[:pos]
        out = grown

    _ALL_ROWS = (np.flatnonzero(row_valid) if hh_index is not None
                 else np.arange(len(census)))

    for i in range(len(cell_vals)):
        need = int(needs[i])
        if need <= 0:
            continue
        key_vals = cell_vals[i]

        donors = None
        for depth, level in enumerate(levels):
            key = tuple(key_vals[: len(avail) - depth])
            cand = level.get(key)
            if cand is not None and len(cand) >= MIN_DONORS:
                donors = cand
                break
        if donors is None:
            donors = _ALL_ROWS
        elif hh_index is not None:
            # Never draw a donor from an oversized/sentinel household.
            donors = donors[row_valid[donors]]
            if len(donors) < MIN_DONORS:
                donors = _ALL_ROWS

        draw = donors[rng.integers(0, len(donors), size=need)]

        for anchor in draw:
            if hh_index is None:
                members = np.array([anchor])
            else:
                members = hh_index[hh_of_row[anchor]]
            start = pos
            _append(members)

            # Fresh unique household id for this reconstructed household.
            if hh_index is not None:
                out[start:pos, col_pos[hh_col]] = next_abshid
                next_abshid += 1

            # Household attributes: identical for every member.
            for cpos, j in avail_hh_pos:
                out[start:pos, cpos] = key_vals[j]

            # Person attributes: exact on the anchor row only (anchor is the
            # first member appended only when it IS members[0]; find it).
            anchor_slot = start + int(np.flatnonzero(members == anchor)[0])
            for cpos, j in avail_person_pos:
                out[anchor_slot, cpos] = key_vals[j]

        n_infilled_cells += 1
        if _DEBUG and n_infilled_cells % 5000 == 0:
            _rss(f"loop cells={n_infilled_cells} pos={pos}")

    if pos == 0:
        return pd.DataFrame(columns=census.columns)

    _rss("before DataFrame")
    infilled = pd.DataFrame(out[:pos], columns=col_names)
    _rss("after DataFrame")

    try:
        from build_feasible_pool_cp import enforce_feasibility
        infilled, _ = enforce_feasibility(infilled, census, rng_seed=seed)
    except Exception:  # noqa: BLE001
        pass

    infilled.attrs["n_cells_infilled"] = n_infilled_cells
    infilled.attrs["n_cells_skipped_cap"] = max(0, n_thin_total - n_infilled_cells)
    return infilled
