"""
SA-Only Baseline Experiment
============================
SA applied to original census microdata (no GAN) to isolate the GAN contribution.
Reviewer 2 (P31): required to complete Table 2.

All paths come from config.py.  Run from synth_pop/:
    python sa_only_baseline.py [--zones 301011001 ...] [--max-iter 1000]
"""

import argparse, os, random, math, sys, re, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config

# All dependencies are now local copies within synth_pop/
# (SA_2_level.py, CensusRecordImporter.py, helper_functions.py copied from sa_run_again_n)

import pandas as pd
from SA_2_level import (
    build_control_dicts,
    calculate_total_rssz,
    calculate_total_rssz_table,
    calculate_total_marginals,
    adjust_with_tolerance,
    normalize_counts,
    setup_logger,
)
from CensusRecordImporter import CensusControlTable
from helper_functions import Helper

OUTPUT_DIR = config.SA_ONLY_RESULTS_DIR
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


_HH_MARKERS = {"bedrd", "hhcd", "hind", "mrerd", "nprd", "rntrd", "strd", "tend", "tenlld", "vehrd"}

def _is_household_table(fname):
    stem = fname.replace(".csv", "").lower().lstrip("sa2")
    return any(m in stem for m in _HH_MARKERS)


# Control-table categories are decoded by CensusRecordImporter into a finer code
# set than the populations carry, because the importer applies its raw label map
# and never the collapsed schema used to build the pool:
#   hrsp  'Not stated' -> 101, 'Not applicable' -> 102, 'Overseas visitor' -> 103,
#         while the pool stores 9, 1, 10 (SA_2_level.map_hrsp_to_hrsp_m).
#   strd  Caravan/Cabin/Improvised/attached-shop -> 9, 'Not stated' -> 4 and
#         'Not applicable' -> 5, while the pool uses Other 4, Not stated 5,
#         Not applicable 6.
# Left uncorrected, every consumer reports phantom unmatched cells (hrsp:101 in
# 364/368 zones and strd:9 in 181/368 in the 2026-11 expansion audit).  Collapse
# once, at load, so the SA fit, the CP-MIP fit and the expansion audit all see
# the pool's categories.  Single lookup on the original code: 9->4 must not then
# become 5.
_POOL_CODE_COLLAPSE = {
    "hrsp": {101: 9, 102: 1, 103: 10},
    "strd": {9: 4, 4: 5, 5: 6},
}


def _ct_to_entry(ct, key):
    """Convert a validated CensusControlTable object to the dict format
    expected by SA_2_level.build_control_dicts:
        {"data": <DataFrame>, "keys": [<non-zone non-count columns>]}

    ABS census CSVs use generic column names '1-digit' or '2-digit' for the
    attribute dimension.  Rename those to the actual attribute name, derived
    by stripping the 'sa1'/'sa2' prefix from the filename key
    (e.g. 'sa2heap' → 'heap', 'sa2hhcd' → 'hhcd').
    """
    zone_col = getattr(ct, "sa_num", "sa2")
    df = ct.control_table.copy()
    attr = re.sub(r'^sa[12]', '', key)   # 'sa2heap' → 'heap'
    for generic in ("1-digit", "2-digit"):
        if generic in df.columns:
            df = df.rename(columns={generic: attr})
    collapse = _POOL_CODE_COLLAPSE.get(attr)
    if collapse and attr in df.columns:
        codes = pd.to_numeric(df[attr], errors="coerce")
        df[attr] = codes.map(
            lambda v: collapse.get(int(v), int(v)) if pd.notna(v) else v)
    keys = [c for c in df.columns if c not in (zone_col, "count")]
    # NB: control categories are NOT merged here. Merging QALFP 12+13 at load time
    # changes the scoring basis for every arm, and a paired re-score of the four
    # published arms (outputs/qalfp_rescore_merged.csv) showed it raises qalfp RSSZ
    # in 81% of zones by 0.7-2.3%, i.e. the unattainable cell was not inflating the
    # residual. The merge is applied only where it fixes a false requirement -- the
    # pool builder's coverage check (see config.QALFP_MERGE_FOR_COVERAGE).
    return {"data": df, "keys": keys}


def _normalize_zone_values(series):
    """Return a string-normalized zone-code series suitable for equality checks.

    Handles mixed inputs like integers/floats/strings and trims trailing ".0"
    introduced by numeric coercion.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def _load_sa2_to_regucp(csf_map_path=None):
    """Load SA2->REGUCP concordance (one regucp per sa2) from csf_sa.csv."""
    candidates = []
    if csf_map_path:
        candidates.append(csf_map_path)
    candidates.extend([
        os.path.join(HERE, "csf_sa.csv"),
        os.path.join(config.SA_DIR, "csf_sa.csv"),
    ])

    source = next((p for p in candidates if p and os.path.exists(p)), None)
    if source is None:
        return {}, None

    conc = pd.read_csv(source, dtype=str, low_memory=False)
    if "sa2" not in conc.columns or "regucp" not in conc.columns:
        return {}, source

    conc = conc[["sa2", "regucp"]].dropna().copy()
    conc["sa2"] = _normalize_zone_values(conc["sa2"])
    conc["regucp"] = _normalize_zone_values(conc["regucp"])
    conc = conc[(conc["sa2"] != "") & (conc["regucp"] != "")]

    # Most datasets have one regucp per sa2; if duplicates appear, keep first deterministically.
    conc = conc.drop_duplicates(["sa2", "regucp"]).sort_values(["sa2", "regucp"])
    sa2_to_regucp = conc.drop_duplicates("sa2").set_index("sa2")["regucp"].to_dict()

    # ---- fallback for SA2s the concordance omits -------------------------
    # csf_sa.csv covers 2,315 SA2s and omits 32 of the 371 case-study zones
    # (e.g. 301021550, 309071552-555). For an unmapped zone the fitter has no
    # geographic donor set and falls back to the WHOLE pool, so its population
    # is drawn from anywhere in South-East Queensland rather than its own area.
    # Fill the gap from the SA3 the zone belongs to (modal REGUCP of its mapped
    # siblings), then its SA4, and report how many were filled.
    ref_path = os.path.join(HERE, "sa2_zones_reference_357.csv")
    if os.path.exists(ref_path):
        try:
            ref = pd.read_csv(ref_path, dtype=str)
            ref.columns = [c.strip().lower() for c in ref.columns]
            zcol = "sa2_code" if "sa2_code" in ref.columns else ref.columns[0]
            ref[zcol] = _normalize_zone_values(ref[zcol])
            filled = {}
            for level in ("sa3_code", "sa4_code"):
                if level not in ref.columns:
                    continue
                known = ref[ref[zcol].isin(sa2_to_regucp)].copy()
                known["regucp"] = known[zcol].map(sa2_to_regucp)
                if known.empty:
                    continue
                modal = (known.groupby([level, "regucp"]).size()
                              .reset_index(name="n")
                              .sort_values([level, "n", "regucp"], ascending=[True, False, True])
                              .drop_duplicates(level)
                              .set_index(level)["regucp"].to_dict())
                for _, row in ref.iterrows():
                    z = row[zcol]
                    if z in sa2_to_regucp or z in filled:
                        continue
                    key = row.get(level)
                    if key in modal:
                        filled[z] = modal[key]
            if filled:
                sa2_to_regucp.update(filled)
                print(f"  [concordance] {os.path.basename(source)} omits {len(filled)} "
                      f"reference SA2(s); inherited REGUCP from SA3/SA4")
        except Exception as exc:
            print(f"  [concordance] SA3 fallback skipped: {exc}")

    return sa2_to_regucp, source


def _quiet_validate_control_table(ct):
    """Run control-table validation while suppressing expected missing-column mapping noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Column '.*' not found in the control table. Skipping mapping\.",
            category=UserWarning,
        )
        ct.validate_and_clean()


def _load_from_dir(directory):
    """Load all (non-copy) CSV control tables from a flat directory."""
    ind, hh = {}, {}
    seen = set()
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"):
            continue
        if " copy" in fname.lower() or fname.startswith("~"):
            continue                         # skip duplicate "- Copy" files
        key = fname.replace(".csv", "").lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            ct = CensusControlTable(os.path.join(directory, fname))
            # Rename generic ABS column names BEFORE validate_and_clean so mappings are applied
            _attr = re.sub(r'^sa[12]', '', key)
            for _generic in ("1-digit", "2-digit"):
                if _generic in ct.control_table.columns:
                    ct.control_table = ct.control_table.rename(columns={_generic: _attr})
            _quiet_validate_control_table(ct)
        except Exception as exc:
            print(f"  WARN: skipping {fname} - {exc}")
            continue
        entry = _ct_to_entry(ct, key)
        if _is_household_table(fname):
            hh[key] = entry
        else:
            ind[key] = entry
    return ind, hh


def load_control_tables():
    """Load control tables from split (individual/household) or flat layout.

    Warns when the directory contains held-out tables: fitting them would break
    the paper's out-of-sample claim, and the live directory holds all 26 tables
    while the published fits used the 19 in-sample ones.
    """
    violations = config.holdout_violations()
    if violations:
        print(f"  [HOLDOUT] control dir '{os.path.basename(config.SA_CONTROL_DIR)}' "
              f"contains held-out table(s) {violations}; fitting on this basis is "
              f"NOT the paper basis. Set SA_CONTROL_DIR to the 19-table directory.")
    if os.path.isdir(config.SA_CONTROL_IND) and os.path.isdir(config.SA_CONTROL_HH):
        ind, hh = {}, {}
        for fname in sorted(os.listdir(config.SA_CONTROL_IND)):
            if not fname.endswith(".csv") or " copy" in fname.lower():
                continue
            key = fname.replace(".csv", "").lower()
            try:
                ct = CensusControlTable(os.path.join(config.SA_CONTROL_IND, fname))
                _attr = re.sub(r'^sa[12]', '', key)
                for _generic in ("1-digit", "2-digit"):
                    if _generic in ct.control_table.columns:
                        ct.control_table = ct.control_table.rename(columns={_generic: _attr})
                _quiet_validate_control_table(ct)
                ind[key] = _ct_to_entry(ct, key)
            except Exception as exc:
                print(f"  WARN: skipping {fname} - {exc}")
        for fname in sorted(os.listdir(config.SA_CONTROL_HH)):
            if not fname.endswith(".csv") or " copy" in fname.lower():
                continue
            key = fname.replace(".csv", "").lower()
            try:
                ct = CensusControlTable(os.path.join(config.SA_CONTROL_HH, fname))
                _attr = re.sub(r'^sa[12]', '', key)
                for _generic in ("1-digit", "2-digit"):
                    if _generic in ct.control_table.columns:
                        ct.control_table = ct.control_table.rename(columns={_generic: _attr})
                _quiet_validate_control_table(ct)
                hh[key] = _ct_to_entry(ct, key)
            except Exception as exc:
                print(f"  WARN: skipping {fname} - {exc}")
        return ind, hh
    return _load_from_dir(config.SA_CONTROL_DIR)


def run_sa_zone(pop, ind_tables, hh_tables, zone, max_iter, initial_temp, cooling_rate,
                objective_mode="rssz_sum"):
    log_file = os.path.join(LOG_DIR, f"sa_only_{zone}.log")
    setup_logger(log_file)
    available_cols = set(pop.columns)

    usable_ind = {k: v for k, v in ind_tables.items() if set(v.get("keys", [])) <= available_cols}
    usable_hh = {k: v for k, v in hh_tables.items() if set(v.get("keys", [])) <= available_cols}

    dropped_ind = sorted(set(ind_tables.keys()) - set(usable_ind.keys()))
    dropped_hh = sorted(set(hh_tables.keys()) - set(usable_hh.keys()))
    if dropped_ind or dropped_hh:
        print(
            f"  Zone {zone}: using {len(usable_ind)}/{len(ind_tables)} individual and "
            f"{len(usable_hh)}/{len(hh_tables)} household tables based on available columns"
        )

    # Reconcile all active tables to one explicit dwelling universe before
    # building control dictionaries. Without this, calculate_total_marginals
    # and the household objective can use different totals (notably NPRD
    # versus HHCD), producing a systematic household under-count.
    # Only this zone is fitted, and the per-zone reconciliation is independent
    # across zones, so restrict the pass to it. Without this the fit rescales all
    # 368 zones for every single zone it fits, which dominated per-zone runtime.
    usable_ind, usable_hh = normalize_counts(
        usable_ind,
        usable_hh,
        audit_path=os.path.join(OUTPUT_DIR, f"zone_{zone}_control_balance_audit.csv"),
        audit_focus_zone=str(zone),
        zones=[zone],
    )

    ind_dicts = build_control_dicts(usable_ind, zone)
    hh_dicts  = build_control_dicts(usable_hh,  zone)
    if sum(1 for d in ind_dicts.values() if not d) >= 2:
        print(f"  Zone {zone}: skipped - insufficient control data")
        return None, None

    households  = pop.groupby("abshid")
    remaining   = set(pop["abshid"].unique())
    total_ind, _ = calculate_total_marginals(ind_dicts, hh_dicts)

    if total_ind <= 0:
        print(f"  Zone {zone}: skipped - total individual target is zero")
        return None, None

    solution, n = [], 0
    _seed_dir = os.environ.get("SA_SEED_FROM_IPU_DIR", "").strip()
    _seeded = False
    if _seed_dir:
        for fname in (f"zone_{zone}_synthetic_population.csv",
                      f"zone_{zone}.csv"):
            path = os.path.join(_seed_dir, fname)
            if not os.path.exists(path):
                continue
            try:
                seed = pd.read_csv(path)
            except Exception:
                continue
            if "abshid" not in seed.columns:
                continue
            ids = set(seed["abshid"].astype(str)) & set(
                pop["abshid"].astype(str))
            if not ids:
                continue
            id_map = {str(h): h for h in remaining}
            remove = {id_map[s] for s in ids if s in id_map}
            solution = pop[pop["abshid"].astype(str).isin(ids)].to_dict(
                "records")
            n = len(solution)
            remaining -= remove
            _seeded = True
            print(f"  Zone {zone}: seeded from CP-IPU "
                  f"({len(remove)} households, {n} persons)")
            break

    if not _seeded and os.environ.get("SA_SIZE_STRATIFIED", "0") not in (
            "0", "", "false", "False"):
        target = {}
        for key, count in hh_dicts.get("nprd", {}).items():
            try:
                code = int(key[0][1])
            except (TypeError, ValueError, IndexError):
                continue
            target[code] = target.get(code, 0) + int(count)
        hh_size = households.size().to_dict()
        by_size = {}
        for hh, size in hh_size.items():
            by_size.setdefault(int(size), []).append(hh)
        for size in sorted(target, key=target.get, reverse=True):
            if n >= total_ind:
                break
            avail = list(by_size.get(size, []))
            random.shuffle(avail)
            for hh in avail[:int(target[size])]:
                if n >= total_ind:
                    break
                data = households.get_group(hh).to_dict("records")
                solution.extend(data)
                n += len(data)
                remaining.discard(hh)
        print(f"  Zone {zone}: size-stratified seed "
              f"({n} persons, nprd targets {target})")

    if not solution:
        while n < total_ind and remaining:
            hh = random.choice(list(remaining)); remaining.remove(hh)
            data = households.get_group(hh).to_dict("records")
            solution.extend(data); n += len(data)

    if not solution:
        print(f"  Zone {zone}: skipped - no feasible initial household solution")
        return None, None

    active_table_count = len(ind_dicts) + len(hh_dicts)
    # --- Per-table objective weights (Oct-2026, joint tables) ---
    # SA_TABLE_WEIGHTS='{"sa2sexp_x_mstp": 0.5}' down-weights tables derived
    # from microdata (joints) so census marginals stay anchored. Keys are the
    # lowercase control categories; omitted tables weigh 1.0. Unset = legacy.
    _table_weights = None
    _raw_weights = os.environ.get("SA_TABLE_WEIGHTS", "").strip()
    if _raw_weights:
        try:
            import json as _json
            _table_weights = {str(k).lower(): float(v) for k, v in _json.loads(_raw_weights).items()}
            print(f"  Zone {zone}: table weights: {_table_weights}")
        except Exception as _exc:
            print(f"  Zone {zone}: WARNING: ignoring malformed SA_TABLE_WEIGHTS ({_exc})")

    def _obj(rows):
        tot = calculate_total_rssz(rows, ind_dicts, hh_dicts, weights=_table_weights)
        if objective_mode == "rssz_sum":
            return tot
        elif objective_mode == "rssz_mean_active":
            return tot / max(active_table_count, 1)
        raise ValueError(f"Unsupported objective_mode: {objective_mode}")

    current_cost = _obj(solution)
    best_sol, best_cost = list(solution), current_cost
    # Perturbations that leave the objective bit-identical are pointless motion
    # (symmetric swaps the tolerance-adjuster can't distinguish). They are
    # rejected up front and counted as no-ops instead of accepted moves, so
    # every accepted move provably changes the objective. Disable with
    # SA_SKIP_NEUTRAL=0 to recover legacy accept-everything behaviour.
    _skip_neutral = os.environ.get("SA_SKIP_NEUTRAL", "1") not in ("0", "", "false", "False")
    # --- Adaptive temperature calibration (Oct-2026) ---
    # Root cause of acc=1.000 random-walk: T0=100 with single-swap |dRSSZ| of
    # ~0.001-0.05 accepts every worsening move for the whole run
    # (exp(-0.02/36)~0.999 early, exp(-0.02/2)=0.99 at the floor), so `best`
    # only improves by luck (1-70/1000 iters). Calibrate T0 from measured
    # proposal deltas so P(accept a median-size worsening move) ~= 0.5 at the
    # start and the chain ends near-greedy. Disable with SA_AUTO_TEMP=0 to
    # recover the exact legacy schedule.
    if os.environ.get("SA_AUTO_TEMP", "1") not in ("0", "", "false", "False"):
        _requested_temp = initial_temp
        _probe_n = int(os.environ.get("SA_AUTO_TEMP_PROBES", "40"))
        _probe_diffs = []
        _hh_groups = pop.groupby("abshid")
        _sol_ids = [r["abshid"] for r in solution]
        for _ in range(_probe_n):
            _out = random.choice(_sol_ids)
            _cands = list(remaining - {_out})
            if not _cands:
                break
            _inn = random.choice(_cands)
            _trial = [r for r in solution if r["abshid"] != _out]
            _trial.extend(_hh_groups.get_group(_inn).to_dict("records"))
            _trial, _ = adjust_with_tolerance(
                _trial, len(_trial), total_ind, config.SA_TOLERANCE,
                pop.groupby("abshid"), remaining)
            _c = _obj(_trial)
            if _c > current_cost:
                _probe_diffs.append(_c - current_cost)
        if _probe_diffs:
            _med = float(sorted(_probe_diffs)[len(_probe_diffs) // 2])
            initial_temp = max(_med / math.log(2.0), 1e-6)
            print(
                f"  Zone {zone}: auto-temp calibrated T0={initial_temp:.4f} "
                f"(median worsening swap Δ={_med:.4f} over {len(_probe_diffs)} probes; "
                f"requested T0 was {_requested_temp:.4f})"
            )
        else:
            initial_temp = 0.05
            print(
                f"  Zone {zone}: auto-temp found no worsening swaps "
                f"(seed already near-optimal?) — using fallback T0={initial_temp:.4f}"
            )
    # Avoid temperature collapse too early (0.95 freezes by ~200 iterations).
    effective_cooling = max(cooling_rate, 0.995)
    if effective_cooling != cooling_rate:
        print(
            f"  Zone {zone}: cooling-rate {cooling_rate} is aggressive; "
            f"using {effective_cooling} to keep search exploratory"
        )
    temp = initial_temp
    # Floor scales with T0: the legacy max(1e-3, T0*0.02) froze at 2.0 for
    # T0=100, i.e. still fully exploratory at the end. End ~100x below T0 so
    # the final stretch is near-greedy.
    min_temp = max(1e-6, initial_temp * 0.01)

    accepted_moves = 0
    improved_moves = 0
    stalled_iters = 0
    noop_proposals = 0
    _record_history = os.environ.get("SA_COST_HISTORY", "0") not in ("0", "", "false", "False")
    cost_history = [(0, float(current_cost), float(best_cost))] if _record_history else None

    for it in range(max_iter):
        if cost_history is not None and it > 0:
            cost_history.append((it, float(current_cost), float(best_cost)))
        current_hh_ids = {r["abshid"] for r in solution}
        proposal_found = False

        # Retry a few times so each iteration evaluates a genuinely perturbed state.
        for _attempt in range(20):
            new_sol = list(solution)
            new_n = n
            candidate_remaining = set(remaining)
            in_solution_households = list({r["abshid"] for r in new_sol})

            feasible_moves = []
            if candidate_remaining:
                feasible_moves.append("add")
            if in_solution_households:
                feasible_moves.append("remove")
            if candidate_remaining and in_solution_households:
                feasible_moves.append("swap")

            if not feasible_moves:
                break

            move = random.choice(feasible_moves)

            if move == "add":
                hh = random.choice(list(candidate_remaining))
                candidate_remaining.remove(hh)
                data = households.get_group(hh).to_dict("records")
                new_sol.extend(data)
                new_n += len(data)
            elif move == "remove":
                hh = random.choice(in_solution_households)
                new_sol = [r for r in new_sol if r["abshid"] != hh]
                candidate_remaining.add(hh)
                new_n = len(new_sol)
            elif move == "swap":
                hh_out = random.choice(in_solution_households)
                swap_in_candidates = list(candidate_remaining - {hh_out})
                if not swap_in_candidates:
                    continue
                new_sol = [r for r in new_sol if r["abshid"] != hh_out]
                candidate_remaining.add(hh_out)
                hh_in = random.choice(swap_in_candidates)
                candidate_remaining.remove(hh_in)
                new_sol.extend(households.get_group(hh_in).to_dict("records"))
                new_n = len(new_sol)

            new_sol, new_n = adjust_with_tolerance(
                new_sol, new_n, total_ind, config.SA_TOLERANCE, households, candidate_remaining)

            if {r["abshid"] for r in new_sol} != current_hh_ids:
                proposal_found = True
                break

        if not proposal_found:
            noop_proposals += 1
            stalled_iters += 1
            temp = max(temp * effective_cooling, min_temp)
            continue

        new_cost = _obj(new_sol)
        diff = new_cost - current_cost
        # Neutral perturbations (bit-identical objective) are pointless motion:
        # they can never improve best and only diffuse the current solution.
        # Reject up front and count as no-ops so every accepted move provably
        # changes the objective. Disable with SA_SKIP_NEUTRAL=0.
        if _skip_neutral and abs(diff) <= 1e-9:
            noop_proposals += 1
            stalled_iters += 1
            temp = max(temp * effective_cooling, min_temp)
            continue
        if diff < 0 or random.random() < math.exp(-diff / max(temp, 1e-10)):
            solution, current_cost, n = new_sol, new_cost, new_n
            remaining = candidate_remaining
            accepted_moves += 1
            if current_cost < best_cost:
                best_sol, best_cost = list(solution), current_cost
                improved_moves += 1
                stalled_iters = 0
            else:
                stalled_iters += 1
        else:
            stalled_iters += 1

        temp = max(temp * effective_cooling, min_temp)

        # Reheat when stuck for a long stretch so SA can escape local minima.
        if stalled_iters > 250:
            temp = max(temp, initial_temp * 0.25)
            stalled_iters = 0
        if (it + 1) % 200 == 0:
            acc_rate = accepted_moves / float(it + 1)
            print(
                f"  Zone {zone}: iter {it+1}/{max_iter}  RSSZ={current_cost:.4f}  "
                f"best={best_cost:.4f}  temp={temp:.4f}  acc={acc_rate:.3f}"
            )

    if cost_history is not None:
        cost_history.append((max_iter, float(current_cost), float(best_cost)))

    if max_iter > 0:
        print(
            f"  Zone {zone}: accepted {accepted_moves}/{max_iter} moves; "
            f"improving accepts={improved_moves}; no-op retries={noop_proposals}"
        )

    # --- Greedy swap polish (Oct-2026) ---
    # Annealing with random proposals plateaus above the achievable floor.
    # This phase tries random household swaps and keeps ONLY strictly
    # improving ones, so best_cost decreases monotonically to a local optimum.
    # Disable with SA_POLISH=0 (then best_* are exactly the annealing result).
    polish_moves = 0
    if os.environ.get("SA_POLISH", "1") not in ("0", "", "false", "False"):
        _polish_budget = int(os.environ.get("SA_POLISH_STEPS", "200"))
        _best_rows, _best_n = list(best_sol), len(best_sol)
        _hh_groups = pop.groupby("abshid")
        _stall = 0
        for _ps in range(_polish_budget):
            _ids = list({r["abshid"] for r in _best_rows})
            _out = random.choice(_ids)
            _cands = list(set(pop["abshid"].unique()) - set(_ids))
            if not _cands:
                break
            _inn = random.choice(_cands)
            _trial = [r for r in _best_rows if r["abshid"] != _out]
            _trial.extend(_hh_groups.get_group(_inn).to_dict("records"))
            _trial, _trial_n = adjust_with_tolerance(
                _trial, len(_trial), total_ind, config.SA_TOLERANCE,
                _hh_groups, set(pop["abshid"].unique()) - {r["abshid"] for r in _trial})
            _c = _obj(_trial)
            if _c < best_cost - 1e-9:
                best_sol, best_cost, _best_n = list(_trial), _c, _trial_n
                polish_moves += 1
                _stall = 0
            else:
                _stall += 1
                if _stall > 60:
                    break
        if polish_moves:
            print(f"  Zone {zone}: polish banked {polish_moves} strictly-improving swaps; best={best_cost:.4f}")

    if cost_history is not None:
        _hist_dir = os.environ.get("SA_GAN_RESULTS_DIR", ".")
        os.makedirs(_hist_dir, exist_ok=True)
        pd.DataFrame(cost_history, columns=["iteration", "current_cost", "best_cost"]).to_csv(
            os.path.join(_hist_dir, f"zone_{zone}_cost_history.csv"), index=False)

    rssz_total, per_table = calculate_total_rssz_table(best_sol, ind_dicts, hh_dicts)
    dominant_table = None
    dominant_rssz = None
    if per_table:
        dominant_table = max(per_table, key=per_table.get)
        dominant_rssz = float(per_table[dominant_table])

    result_row = {
        "zone": zone,
        "objective_mode": objective_mode,
        "objective_value": best_cost,
        "rssz_total": rssz_total,
        "active_tables": active_table_count,
        "dominant_table": dominant_table,
        "dominant_table_rssz": dominant_rssz,
        **per_table,
    }
    return pd.DataFrame(best_sol), result_row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zones",        nargs="*", type=str)
    p.add_argument("--microdata",    type=str, default=config.CENSUS_MICRODATA,
                   help="Path to census microdata CSV (default: config.CENSUS_MICRODATA)")
    p.add_argument("--csf-map",      type=str, default=None,
                   help="Optional path to csf_sa.csv (SA2->REGUCP concordance)")
    p.add_argument("--max-iter",     type=int,   default=config.SA_MAX_ITER)
    p.add_argument("--initial-temp", type=float, default=config.SA_INITIAL_TEMP)
    p.add_argument("--cooling-rate", type=float, default=config.SA_COOLING_RATE)
    p.add_argument(
        "--objective",
        type=str,
        default="rssz_sum",
        choices=["rssz_sum", "rssz_mean_active"],
        help=(
            "Objective used during SA search: rssz_sum (legacy) or "
            "rssz_mean_active (rssz_sum divided by active table count)."
        ),
    )
    args = p.parse_args()

    print(f"Loading census microdata from: {args.microdata}")
    helper = Helper()
    microdata = helper.clean_my_datagan(args.microdata)
    print(f"  Records: {len(microdata):,}")

    ind_tables, hh_tables = load_control_tables()
    print(f"  Individual control tables ({len(ind_tables)}):")
    for k, v in ind_tables.items():
        print(f"    {k:30s}  keys={v['keys']}  rows={len(v['data'])}")
    print(f"  Household  control tables ({len(hh_tables)}):")
    for k, v in hh_tables.items():
        print(f"    {k:30s}  keys={v['keys']}  rows={len(v['data'])}")

    # Pre-check: show which control tables match the microdata pool
    _pool_cols = set(microdata.columns)
    _usable_ind = {k: v for k, v in ind_tables.items() if set(v.get("keys", [])) <= _pool_cols}
    _usable_hh  = {k: v for k, v in hh_tables.items()  if set(v.get("keys", [])) <= _pool_cols}
    _dropped_ind = sorted(set(ind_tables.keys()) - set(_usable_ind.keys()))
    _dropped_hh  = sorted(set(hh_tables.keys()) - set(_usable_hh.keys()))
    print(f"\n  Control table / pool linkage (SA-Only baseline):")
    print(f"    Will use : {len(_usable_ind)} individual + {len(_usable_hh)} household tables")
    if _dropped_ind or _dropped_hh:
        print(f"    Dropped (column absent from pool): ind={_dropped_ind}  hh={_dropped_hh}")
    _missing_keys = sorted(
        k for entry in list(ind_tables.values()) + list(hh_tables.values())
        for k in entry.get("keys", []) if k not in _pool_cols
    )
    print(f"    Control attrs missing from pool: {_missing_keys if _missing_keys else 'NONE - all good'}")

    if args.zones:
        zones = [str(z) for z in args.zones]
    else:
        zones = [str(z) for z in next(iter(ind_tables.values()))["data"]["sa2"].dropna().unique().tolist()]

    has_sa2 = "sa2" in microdata.columns
    has_regucp = "regucp" in microdata.columns
    if not has_sa2 and not has_regucp:
        raise ValueError(
            "Microdata has neither 'sa2' nor 'regucp' columns, so zones cannot be matched."
        )

    sa2_to_regucp, conc_source = _load_sa2_to_regucp(args.csf_map)
    if sa2_to_regucp:
        print(f"  Loaded SA2->REGUCP concordance: {conc_source}")
    else:
        print("  WARN: SA2->REGUCP concordance not available; SA-only can only use direct sa2/regucp matches.")

    overlap = set()
    zone_set = set(zones)
    if has_sa2:
        overlap |= set(_normalize_zone_values(microdata["sa2"]).dropna().unique().tolist()) & zone_set
    micro_reg_set = set(_normalize_zone_values(microdata["regucp"]).dropna().unique().tolist()) if has_regucp else set()
    if has_regucp:
        overlap |= set(_normalize_zone_values(microdata["regucp"]).dropna().unique().tolist()) & zone_set
        # Fallback path: map each SA2 zone to its regucp bucket via csf_sa.csv.
        if sa2_to_regucp:
            overlap |= {z for z in zones if sa2_to_regucp.get(z) in micro_reg_set}

    if not overlap:
        print(
            "  WARN: No overlapping zone codes between microdata ('sa2'/'regucp') and SA2 control tables.\n"
            "        SA-Only will skip all zones unless microdata includes matching SA2 codes."
        )
    elif not args.zones:
        zones = [z for z in zones if z in overlap]
        print(f"  Filtered zones to concordance-reachable subset: {len(zones)}")
    print(f"\nProcessing {len(zones)} zones")

    all_rssz = []
    skipped_zones = []
    for zone in zones:
        print(f"\nZone {zone}")
        zone_pop = pd.DataFrame()
        if has_sa2:
            sa2_mask = _normalize_zone_values(microdata["sa2"]) == zone
            zone_pop = microdata.loc[sa2_mask]
        if zone_pop.empty and has_regucp:
            regucp_mask = _normalize_zone_values(microdata["regucp"]) == zone
            zone_pop = microdata.loc[regucp_mask]
        if zone_pop.empty and has_regucp and sa2_to_regucp:
            mapped_reg = sa2_to_regucp.get(zone)
            if mapped_reg:
                regucp_mask = _normalize_zone_values(microdata["regucp"]) == mapped_reg
                zone_pop = microdata.loc[regucp_mask]
        if zone_pop.empty:
            # No direct SA2/REGUCP match (e.g. Tweed Heads/Toowoomba zones --
            # microdata_for_exp.csv predates the SEQ+Tweed+Toowoomba 357-zone
            # study area expansion and has zero rows tagged for them at all).
            # Fall back to the full microdata pool as the seed rather than
            # skipping the zone outright, matching SA_2_level.py's existing
            # "REGUCP not found -> fall back to full synthetic population"
            # behaviour for SA-GAN/IPU-GAN -- keeps all four strategies
            # comparable over the same zone set instead of silently dropping
            # this baseline's coverage of border zones.
            print(f"  WARNING: zone {zone} has no direct SA2/REGUCP match in microdata_for_exp.csv "
                  "-- falling back to full microdata pool as seed.")
            zone_pop = microdata.copy()

        # Isolate each zone: an unexpected error while fitting one zone (e.g. a
        # border zone whose control tables don't line up with the microdata)
        # must not abort the whole chunk and lose every other zone's results.
        try:
            result_df, rssz_info = run_sa_zone(
                zone_pop, ind_tables, hh_tables, zone,
                args.max_iter, args.initial_temp, args.cooling_rate,
                objective_mode=args.objective)
        except Exception as exc:
            print(f"  WARNING: zone {zone} skipped - SA fitting raised an error: {exc}")
            skipped_zones.append(zone)
            continue

        if result_df is not None:
            result_df.to_csv(os.path.join(OUTPUT_DIR, f"zone_{zone}.csv"), index=False)
            all_rssz.append(rssz_info)
        else:
            skipped_zones.append(zone)

    if skipped_zones:
        print(f"\n  Skipped {len(skipped_zones)}/{len(zones)} zone(s) with no usable "
              f"data: {skipped_zones}")

    if not all_rssz:
        print(
            "ERROR: SA-Only baseline produced zero usable zones in this chunk — every "
            "requested zone lacked control-table/microdata overlap or failed SA fitting."
        )
        sys.exit(1)

    summary = pd.DataFrame(all_rssz)
    # When running as one of several parallel PBS chunk jobs (CHUNK_ID set by
    # stage3b_sa_only.pbs), give the summary CSV a per-chunk filename so
    # concurrent chunks don't overwrite each other's rows via to_csv() on the
    # same shared path. Stage 4 (stage4_analysis.pbs) aggregates all per-chunk
    # files back into the canonical sa_only_rssz_summary.csv before use.
    _chunk_id = os.environ.get("CHUNK_ID", "").strip()
    summary_name = f"sa_only_rssz_summary_chunk{_chunk_id}.csv" if _chunk_id else "sa_only_rssz_summary.csv"
    out_path = os.path.join(OUTPUT_DIR, summary_name)
    summary.to_csv(out_path, index=False)
    vals = summary["rssz_total"]
    print(f"\n{'='*60}\nSA-Only RSSZ Summary\n{'='*60}")
    print(f"  Zones : {len(vals)}")
    print(f"  Mean  : {vals.mean():.4f}")
    print(f"  Median: {vals.median():.4f}")
    print(f"  Min   : {vals.min():.4f}")
    print(f"  Max   : {vals.max():.4f}")
    print(f"  Std   : {vals.std():.4f}")
    print(f"\nSaved: {out_path}")
    print(
        "\n[CHECK] Add the SA-Only row to Table 2 of the paper.\n"
        "If SA-Only RSSZ ~ SA-GAN RSSZ, most improvement comes from SA, not the GAN.\n"
        "If SA-Only RSSZ >> SA-GAN RSSZ, the GAN's richer microdata pool is the key driver."
    )


if __name__ == "__main__":
    main()
