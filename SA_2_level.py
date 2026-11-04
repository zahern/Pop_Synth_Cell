import argparse
import random
import math
import pandas as pd
from collections import Counter, defaultdict
from CensusRecordImporter import CensusControlTable
from helper_functions import Helper
import os
from scipy.stats import chi2
import logging
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt
from functools import partial
import copy
import sys


def _canonical_category_value(value):
    """Normalise category values so control and observed keys are comparable."""
    if pd.isna(value):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return stripped
        numeric = pd.to_numeric(stripped, errors="coerce")
        if pd.notna(numeric):
            if float(numeric).is_integer():
                return int(numeric)
            return float(numeric)
        return stripped

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int,)):
        return int(value)

    if isinstance(value, (float,)):
        if pd.isna(value):
            return None
        if float(value).is_integer():
            return int(value)
        return float(value)

    return value


def _looks_like_na_value(value):
    """Return True if a category value appears to be a Not Applicable/Not Stated bucket."""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"na", "n/a", "not applicable", "not stated"}:
        return True
    return ("not" in text and ("app" in text or "stated" in text))


def _zone_key(value):
    """Canonical string key for a zone code, independent of column dtype.

    The control tables do not agree on the dtype of their ``sa2`` column: six
    of them load it as ``str`` and the rest as ``int64``. Comparing a zone code
    from one family against the other silently matches nothing, which made
    ``_balance_zone_total_with_na`` believe the zone had no rows at all and
    inject a synthetic bucket holding the entire zone target. That inflated the
    table total by the full target (roughly doubled it). Normalising both sides
    to this key removes the dtype dependence.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _zone_mask(data, zone, zone_column="sa2"):
    """Boolean mask selecting one zone, robust to str/int dtype disagreement."""
    if zone_column not in data.columns:
        return pd.Series(False, index=data.index)
    return data[zone_column].map(_zone_key) == _zone_key(zone)


def _pick_na_bucket_index(zone_df, key_columns):
    """Find the best existing NA-like row index for a zone table slice."""
    if zone_df.empty:
        return None

    # 1) Prefer explicit textual NA/Not stated labels if present.
    for idx, row in zone_df.iterrows():
        for col in key_columns:
            if col in zone_df.columns and _looks_like_na_value(row[col]):
                return idx

    # 2) Fallback for numeric-coded tables: NA bins are typically the highest codes.
    numeric_keys = [
        col for col in key_columns
        if col in zone_df.columns and pd.api.types.is_numeric_dtype(zone_df[col])
    ]
    if numeric_keys and len(numeric_keys) == len(key_columns):
        ranked = pd.Series(0.0, index=zone_df.index)
        for col in numeric_keys:
            ranked = ranked + pd.to_numeric(zone_df[col], errors="coerce").fillna(-1)
        if not ranked.empty:
            return ranked.idxmax()

    return None


def _ensure_na_bucket(data, zone, key_columns, zone_column="sa2", count_column="count"):
    """Ensure a zone table has an NA row available for balancing totals."""
    zone_mask = _zone_mask(data, zone, zone_column)
    zone_df = data.loc[zone_mask]
    na_idx = _pick_na_bucket_index(zone_df, key_columns)
    if na_idx is not None:
        return data, na_idx, False

    # Create a synthetic NA row when one is missing.
    new_row = {col: 0 for col in data.columns}
    if zone_column in data.columns:
        new_row[zone_column] = zone
    if count_column in data.columns:
        new_row[count_column] = 0

    for col in key_columns:
        if col not in data.columns:
            continue
        if pd.api.types.is_numeric_dtype(data[col]):
            series = pd.to_numeric(data.loc[zone_mask, col], errors="coerce")
            new_row[col] = int(series.max()) + 1 if series.notna().any() else 0
        else:
            new_row[col] = "Not applicable"

    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
    return data, data.index[-1], True


def _balance_zone_total_with_na(
    data,
    zone,
    key_columns,
    target_total,
    zone_column="sa2",
    count_column="count",
):
    """Force a zone table total to target_total by adjusting NA (then largest bins if needed)."""
    zone_mask = _zone_mask(data, zone, zone_column)
    zone_idx = data.index[zone_mask]
    if len(zone_idx) == 0:
        # The zone is absent from this table. It therefore carries no target
        # for the zone, and inventing one is what doubled the individual
        # universe: the previous branch created a synthetic NA row and set it
        # to the WHOLE zone target instead of the residual, so the table total
        # became target + original. Leave the table untouched and record it.
        return data, {
            "before_total": 0,
            "target_total": int(target_total),
            "diff": 0,
            "na_added": 0,
            "na_removed": 0,
            "other_removed": 0,
            "na_bucket_created": False,
            "after_total": 0,
            "zone_absent": True,
        }

    current_total = int(pd.to_numeric(data.loc[zone_idx, count_column], errors="coerce").fillna(0).sum())
    diff = int(target_total) - current_total
    if diff == 0:
        return data, {
            "before_total": current_total,
            "target_total": int(target_total),
            "diff": 0,
            "na_added": 0,
            "na_removed": 0,
            "other_removed": 0,
            "na_bucket_created": False,
            "after_total": current_total,
        }

    data[count_column] = pd.to_numeric(data[count_column], errors="coerce").fillna(0).round().astype(int)
    data, na_idx, na_bucket_created = _ensure_na_bucket(
        data,
        zone,
        key_columns,
        zone_column=zone_column,
        count_column=count_column,
    )

    na_added = 0
    na_removed = 0
    other_removed = 0

    if diff > 0:
        data.loc[na_idx, count_column] = int(data.loc[na_idx, count_column]) + diff
        na_added = diff
        after_total = int(pd.to_numeric(data.loc[_zone_mask(data, zone, zone_column), count_column], errors="coerce").fillna(0).sum())
        return data, {
            "before_total": current_total,
            "target_total": int(target_total),
            "diff": diff,
            "na_added": na_added,
            "na_removed": na_removed,
            "other_removed": other_removed,
            "na_bucket_created": na_bucket_created,
            "after_total": after_total,
        }

    # diff < 0: remove surplus without allowing negative counts.
    remaining_to_remove = -diff

    na_available = int(data.loc[na_idx, count_column])
    take = min(na_available, remaining_to_remove)
    data.loc[na_idx, count_column] = na_available - take
    remaining_to_remove -= take
    na_removed += take

    if remaining_to_remove <= 0:
        after_total = int(pd.to_numeric(data.loc[_zone_mask(data, zone, zone_column), count_column], errors="coerce").fillna(0).sum())
        return data, {
            "before_total": current_total,
            "target_total": int(target_total),
            "diff": diff,
            "na_added": na_added,
            "na_removed": na_removed,
            "other_removed": other_removed,
            "na_bucket_created": na_bucket_created,
            "after_total": after_total,
        }

    candidates = data.loc[_zone_mask(data, zone, zone_column)].sort_values(count_column, ascending=False)
    for idx in candidates.index:
        if idx == na_idx:
            continue
        available = int(data.loc[idx, count_column])
        if available <= 0:
            continue
        take = min(available, remaining_to_remove)
        data.loc[idx, count_column] = available - take
        remaining_to_remove -= take
        other_removed += take
        if remaining_to_remove <= 0:
            break

    after_total = int(pd.to_numeric(data.loc[_zone_mask(data, zone, zone_column), count_column], errors="coerce").fillna(0).sum())
    return data, {
        "before_total": current_total,
        "target_total": int(target_total),
        "diff": diff,
        "na_added": na_added,
        "na_removed": na_removed,
        "other_removed": other_removed,
        "na_bucket_created": na_bucket_created,
        "after_total": after_total,
    }

def normalize_counts(
    individual_control_tables,
    household_control_tables,
    zone_column='sa2',
    count_column='count',
    audit_path=None,
    audit_focus_zone=None,
    zones=None,
):
    """
    Reconcile control-table totals to one explicit universe per SA zone.

    Individual controls are scaled to the largest available person total for backwards
    compatibility. Household controls use HHCD when present, because the
    household comparison and dwelling-count target are defined on that
    universe. If HHCD is unavailable, the largest available household total is
    used and recorded as a fallback in the audit.

    Parameters:
        individual_control_tables (dict): Dictionary of individual control tables.
        household_control_tables (dict): Dictionary of household control tables.
        zone_column (str): Column name representing SA zones (default: 'sa2').
        count_column (str): Column name representing counts (default: 'count').

    Returns:
        tuple: A tuple containing the updated individual_control_tables and household_control_tables.
    """

    # Create deep copies of the dictionaries to avoid modifying the originals
    import copy
    updated_individual_control_tables = copy.deepcopy(individual_control_tables)
    updated_household_control_tables = copy.deepcopy(household_control_tables)

    # Gather all unique SA zones from individual and household tables.
    # Zones are pooled as canonical keys, never as raw column values: the tables
    # mix str and int64 sa2 columns, and a raw set would hold both 112011236 and
    # "112011236" as separate zones while every per-zone comparison matched only
    # one family (see _zone_key).
    # ``zones`` restricts the pass to the zones the caller is actually fitting;
    # the per-zone work is independent, so the result is identical and a
    # single-zone fit no longer pays for all 368 zones.
    if zones is not None:
        all_zones = {_zone_key(z) for z in zones}
    else:
        all_zones = set()
        for control_table in updated_individual_control_tables.values():
            all_zones.update(_zone_key(v) for v in control_table['data'][zone_column].unique())
        for control_table in updated_household_control_tables.values():
            all_zones.update(_zone_key(v) for v in control_table['data'][zone_column].unique())

    audit_rows = []

    # Process each SA zone
    for zone in all_zones:
        # Step 1: Adjust individual counts
        max_individual_count = 0

        # Determine the max count for the current zone in individual tables
        for control_table in updated_individual_control_tables.values():
            data = control_table['data']
            filtered_data = data[_zone_mask(data, zone, zone_column)]
            max_individual_count = max(max_individual_count, filtered_data[count_column].sum())

        # Update all individual tables to match the max count
        for control_table in updated_individual_control_tables.values():
            data = control_table['data']
            filtered_data = data[_zone_mask(data, zone, zone_column)]
            current_total = filtered_data[count_column].sum()

            if current_total > 0:  # Only scale if the current total is greater than 0
                scale_factor = max_individual_count / current_total
                # Temporarily cast the column to float64 for scaling
                data[count_column] = data[count_column].astype(float)
                data.loc[filtered_data.index, count_column] *= scale_factor
                # Cast back to int64 after scaling
                data[count_column] = data[count_column].round().astype(int)
            else:
                # If the total is 0, set all counts to 0 (no scaling possible)
                data.loc[filtered_data.index, count_column] = 0

        # Force exact post-rounding agreement by balancing into NA bucket.
        for category, control_table in updated_individual_control_tables.items():
            data = control_table['data']
            keys = control_table.get('keys', [])
            balanced_data, detail = _balance_zone_total_with_na(
                data,
                zone,
                keys,
                max_individual_count,
                zone_column=zone_column,
                count_column=count_column,
            )
            control_table['data'] = balanced_data
            audit_rows.append({
                "zone": str(zone),
                "level": "individual",
                "table": str(category),
                "target_total": int(max_individual_count),
                **detail,
            })

        # Step 2: Reconcile household counts to one dwelling universe. The
        # household tables are not guaranteed to have identical universes
        # (e.g. NPRD can be smaller than HHCD), so selecting NPRD later as the
        # total would undo this reconciliation.
        max_household_count = 0
        household_target_source = "max_available"
        hhcd_key = next(
            (key for key in updated_household_control_tables
             if str(key).lower().endswith("hhcd")),
            None,
        )
        if hhcd_key is not None:
            hhcd_data = updated_household_control_tables[hhcd_key]["data"]
            hhcd_zone = hhcd_data[_zone_mask(hhcd_data, zone, zone_column)]
            max_household_count = int(
                pd.to_numeric(hhcd_zone[count_column], errors="coerce")
                .fillna(0)
                .sum()
            )
            household_target_source = str(hhcd_key)

        # Determine the max count for the current zone in household tables
        if hhcd_key is None:
            for control_table in updated_household_control_tables.values():
                data = control_table['data']
                filtered_data = data[_zone_mask(data, zone, zone_column)]
                max_household_count = max(
                    max_household_count,
                    filtered_data[count_column].sum(),
                )

        # Update all household tables to match the max count
        for control_table in updated_household_control_tables.values():
            data = control_table['data']
            filtered_data = data[_zone_mask(data, zone, zone_column)]
            current_total = filtered_data[count_column].sum()

            if current_total > 0:  # Only scale if the current total is greater than 0
                scale_factor = max_household_count / current_total
                # Temporarily cast the column to float64 for scaling
                data[count_column] = data[count_column].astype(float)
                data.loc[filtered_data.index, count_column] *= scale_factor
                # Cast back to int64 after scaling
                data[count_column] = data[count_column].round().astype(int)
            else:
                # If the total is 0, set all counts to 0 (no scaling possible)
                data.loc[filtered_data.index, count_column] = 0

        # Force exact post-rounding agreement by balancing into NA bucket.
        for category, control_table in updated_household_control_tables.items():
            data = control_table['data']
            keys = control_table.get('keys', [])
            balanced_data, detail = _balance_zone_total_with_na(
                data,
                zone,
                keys,
                max_household_count,
                zone_column=zone_column,
                count_column=count_column,
            )
            control_table['data'] = balanced_data
            audit_rows.append({
                "zone": str(zone),
                "level": "household",
                "table": str(category),
                "target_total": int(max_household_count),
                "target_source": household_target_source,
                **detail,
            })

    if audit_path:
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        audit_df = pd.DataFrame(audit_rows)
        if audit_focus_zone is not None:
            audit_df = audit_df[audit_df["zone"].astype(str) == str(audit_focus_zone)]
        audit_df = audit_df.sort_values(["zone", "level", "table"]).reset_index(drop=True)
        audit_df.to_csv(audit_path, index=False)

    return updated_individual_control_tables, updated_household_control_tables
# Example usage:
# updated_individual_tables, updated_household_tables = normalize_counts(individual_control_tables, household_control_tables)

# Example usage:
# updated_individual_tables, updated_household_tables = normalize_counts(individual_control_tables, household_control_tables)

# Example usage:
# updated_individual_tables, updated_household_tables = normalize_counts(individual_control_tables, household_control_tables)


def setup_logger(log_file):
    """
    Setup the logger to write logs to a file.

    Parameters:
    - log_file: Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("Logging started for simulated annealing.")


def preprocess_expected_counts_ipu(expected_counts):

    """
    Convert the keys of expected_counts into a format compatible with the synthetic population.

    Parameters:
    - expected_counts: A dictionary where keys are hierarchical tuples.

    Returns:
    - A new dictionary with keys organized by variable names and values.
    """
    processed_counts = {}
    for key, count in expected_counts.items():
        category, nested_keys = key

        # Retain the hierarchical structure but normalize the keys
        if isinstance(nested_keys, tuple) and all(isinstance(inner, tuple) for inner in nested_keys):
            # Hierarchical keys like (('strd', 1),)
            variable_names = tuple(col for col, _ in nested_keys)
        elif isinstance(nested_keys, tuple):
            # Normalized keys like (1,)
            variable_names = nested_keys
        else:
            # Single keys
            variable_names = (nested_keys,)

        # Store the processed keys and counts
        processed_counts[(category, tuple(variable_names))] = count

    return processed_counts

def preprocess_expected_counts(expected_counts):
    """
    Convert the keys of expected_counts from tuples of tuples to simple tuples.

    Parameters:
    - expected_counts: A dictionary where keys are tuples of tuples.

    Returns:
    - A new dictionary with keys converted to simple tuples.
    """
    processed_counts = defaultdict(int)
    for key, count in expected_counts.items():
        canonical_key = tuple(_canonical_category_value(value) for _, value in key)
        processed_counts[canonical_key] += int(count)
    return dict(processed_counts)

def calculate_ssz(observed_counts, expected_counts_u, total_count, critical_value):
    """
    Calculate the SSZ statistic for a single tabulation.

    Parameters:
    - observed_counts: A Counter object with observed counts.
    - expected_counts: A dictionary with expected counts from control tables.
    - total_count: Total number of entries in the tabulation.
    - critical_value: Critical value of the chi-squared distribution.

    Returns:
    - SSZ statistic as a float.
    """
    ssz = 0
    expected_counts = preprocess_expected_counts(expected_counts_u)

    for key, observed_count in observed_counts.items():
        expected_count = expected_counts.get(key, 0)
        #print(f"Key: {key}, Observed: {observed_count}, Expected: {expected_count}")  # Debugging line
        if expected_count > 0:
            z_score_squared = ((observed_count - expected_count) ** 2) / expected_count
            ssz += z_score_squared
    if total_count > 0 and critical_value > 0:
        ssz /= total_count
        ssz /= critical_value
    #print(f"SSZ: {ssz}")  # Debugging line
    return ssz


def get_observed_counts(solution_df, category_columns):
    """
    Calculate observed counts for a given category from the synthetic population.

    Parameters:
    - solution_df: DataFrame containing the synthetic population.
    - category_columns: List of columns to group by for the category.

    Returns:
    - A Counter object with observed counts.
    """
    cols = list(category_columns)
    if not cols:
        return Counter()
    # Canonicalise each column's *distinct* values once and look them up per cell,
    # instead of doing everything through iterrows().  iterrows() builds a pandas
    # Series for every row, which a cProfile of a single 60-iteration zone measured
    # at 9.8M Series constructions taking 447 of 612 seconds -- 73% of the run.
    # Control columns hold a handful of categories, so the canonicaliser now runs
    # tens of times per cost evaluation rather than thousands.  Keys stay tuples of
    # canonical values (matching preprocess_expected_counts) so the RSSZ objective
    # is unchanged.
    mapped = []
    for col in cols:
        series = solution_df[col]
        values = series.to_numpy(dtype=object, copy=False)
        # Normalise missing to None up front so every cell has a stable cache key
        # (NaN never compares equal to its own dict key) and the canonical None
        # that preprocess_expected_counts() produces is what gets counted.
        if getattr(series, "hasnans", False):
            mask = pd.isna(values)
            if mask.any():
                values = values.copy()
                values[mask] = None
        canon = {v: _canonical_category_value(v) for v in pd.unique(values)}
        mapped.append([canon[v] for v in values])
    if len(mapped) == 1:
        return Counter((v,) for v in mapped[0])
    return Counter(zip(*mapped))


def _rssz_table_diagnostics(control_dict, observed_counts, category_columns, missing_columns=None):
    """Return overlap diagnostics to explain inflated RSSZ from sparse key matches."""
    expected_counts = preprocess_expected_counts(control_dict)
    expected_keys = set(expected_counts.keys())
    observed_keys = set(observed_counts.keys())
    matched_keys = expected_keys & observed_keys

    expected_total = sum(expected_counts.values())
    unmatched_expected_total = sum(expected_counts[k] for k in expected_keys - matched_keys)

    return {
        "category_columns": ",".join(category_columns),
        "expected_keys": len(expected_keys),
        "observed_keys": len(observed_keys),
        "matched_keys": len(matched_keys),
        "expected_key_coverage": (len(matched_keys) / len(expected_keys)) if expected_keys else 0.0,
        "observed_key_coverage": (len(matched_keys) / len(observed_keys)) if observed_keys else 0.0,
        "expected_total": expected_total,
        "unmatched_expected_total": unmatched_expected_total,
        "unmatched_expected_share": (unmatched_expected_total / expected_total) if expected_total else 0.0,
        "missing_columns": ",".join(missing_columns or []),
    }


def calculate_rssz_for_table(solution_df, control_dict, category_columns, return_details=False):
    """
    Calculate the RSSZ for a single control table.

    Parameters:
    - solution_df: DataFrame containing the synthetic population.
    - control_dict: Control dictionary with expected counts.
    - category_columns: List of columns to group by for the category.

    Returns:
    - RSSZ value for the control table as a float.
    """
    min_expected_coverage = float(os.environ.get("SA_MIN_EXPECTED_KEY_COVERAGE", "0.0"))
    missing_columns = [col for col in category_columns if col not in solution_df.columns]
    if missing_columns:
        logging.warning(
            f"Skipping RSSZ table because solution_df is missing columns: {missing_columns}"
        )
        details = _rssz_table_diagnostics(
            control_dict=control_dict,
            observed_counts=Counter(),
            category_columns=category_columns,
            missing_columns=missing_columns,
        )
        details["rssz"] = 0.0
        details["coverage_threshold"] = min_expected_coverage
        details["excluded_reason"] = "missing_columns"
        if return_details:
            return 0.0, details
        return 0

    observed_counts = get_observed_counts(solution_df, category_columns)
    total_count = sum(control_dict.values())
    confidence_level = 0.95  # 95% confidence level
    # Clamp dof to at least 1 so chi2.ppf never returns nan (dof=0 is degenerate)
    dof = max(1, len(control_dict) - 1)
    critical_value = chi2.ppf(confidence_level, dof)

    details = _rssz_table_diagnostics(
        control_dict=control_dict,
        observed_counts=observed_counts,
        category_columns=category_columns,
        missing_columns=None,
    )
    details["coverage_threshold"] = min_expected_coverage

    if details["expected_key_coverage"] < min_expected_coverage:
        logging.info(
            "Excluding RSSZ table due to low expected-key coverage: cols=%s coverage=%.4f threshold=%.4f",
            category_columns,
            details["expected_key_coverage"],
            min_expected_coverage,
        )
        details["rssz"] = 0.0
        details["excluded_reason"] = "low_expected_key_coverage"
        if return_details:
            return 0.0, details
        return 0.0

    rssz = calculate_ssz(observed_counts, control_dict, total_count, critical_value)
    details["rssz"] = rssz
    details["excluded_reason"] = ""
    if return_details:
        return rssz, details

    return rssz


def _table_weight(weights, category):
    """Per-table objective multiplier (joint tables are down-weighted by default)."""
    if not weights:
        return 1.0
    try:
        return float(weights.get(category, weights.get(category.lower(), 1.0)))
    except (TypeError, ValueError):
        return 1.0


def calculate_total_rssz(solution, individual_control_dicts, household_control_dicts, weights=None):
    """
    Calculate the total RSSZ statistic for the given solution.

    Parameters:
    - solution: List of dictionaries representing the current synthetic population solution.
    - individual_control_dicts: Control dictionaries for individual-level attributes.
    - household_control_dicts: Control dictionaries for household-level attributes.
    - weights: Optional dict {control category: multiplier} to weight tables in
      the objective (e.g. down-weight joint tables derived from microdata so
      census marginals stay anchored).  None (default) = all tables weight 1.0,
      exactly the legacy behaviour.  Multi-column (joint) control tables work
      with no further changes: keys are (column, value) tuples regardless of
      how many columns the table has.

    Returns:
    - Total RSSZ value as a float.
    """
    solution_df = pd.DataFrame(solution)
    rssz_total = 0

    # Calculate RSSZ for individual-level attributes
    for category, control_dict in individual_control_dicts.items():
        if not control_dict:
            continue

        sample_key = next(iter(control_dict.keys()), None)
        if sample_key is None:
            continue  # Skip empty control dicts

        category_columns = [key[0] for key in sample_key]
        rssz_total += _table_weight(weights, category) * calculate_rssz_for_table(solution_df, control_dict, category_columns)

    # Calculate RSSZ for household-level attributes
    if "abshid" in solution_df.columns:
        household_df = solution_df.groupby("abshid").first()  # Use the first record per household
        for category, control_dict in household_control_dicts.items():
            if not control_dict:
                continue

            sample_key = next(iter(control_dict.keys()), None)
            if sample_key is None:
                continue  # Skip empty control dicts

            category_columns = [key[0] for key in sample_key]
            rssz_total += _table_weight(weights, category) * calculate_rssz_for_table(household_df, control_dict, category_columns)
    else:
        logging.warning("Skipping household-level RSSZ scoring because solution_df is missing 'abshid'.")

    return rssz_total

def calculate_total_rssz_table(solution, individual_control_dicts, household_control_dicts, return_diagnostics=False, weights=None):
    """
    Calculate the total RSSZ statistic for the given solution and save RSSZ values for each table.

    Parameters:
    - solution: List of dictionaries representing the current synthetic population solution.
    - individual_control_dicts: Control dictionaries for individual-level attributes.
    - household_control_dicts: Control dictionaries for household-level attributes.
    - weights: Optional per-table multipliers (see calculate_total_rssz).  The
      per-table breakdown is always reported UNWEIGHTED for comparability; only
      the returned total reflects the weights.

    Returns:
    - Total RSSZ value as a float.
    - Dictionary of RSSZ values for individual and household tables.
    - Optional diagnostics list when return_diagnostics=True.
    """
    solution_df = pd.DataFrame(solution)
    rssz_total = 0
    rssz_per_table = {}  # To store RSSZ values for each table
    diagnostics = []

    # Calculate RSSZ for individual-level attributes
    for category, control_dict in individual_control_dicts.items():
        if not control_dict:
            continue

        sample_key = next(iter(control_dict.keys()), None)
        if sample_key is None:
            continue  # Skip empty control dicts

        category_columns = [key[0] for key in sample_key]
        rssz_value, details = calculate_rssz_for_table(
            solution_df,
            control_dict,
            category_columns,
            return_details=True,
        )
        rssz_total += _table_weight(weights, category) * rssz_value
        rssz_per_table[f"Individual - {category}"] = rssz_value  # Save RSSZ value
        details["table"] = f"Individual - {category}"
        diagnostics.append(details)

    # Calculate RSSZ for household-level attributes
    if "abshid" in solution_df.columns:
        household_df = solution_df.groupby("abshid").first()  # Use the first record per household
        for category, control_dict in household_control_dicts.items():
            if not control_dict:
                continue

            sample_key = next(iter(control_dict.keys()), None)
            if sample_key is None:
                continue  # Skip empty control dicts

            category_columns = [key[0] for key in sample_key]
            rssz_value, details = calculate_rssz_for_table(
                household_df,
                control_dict,
                category_columns,
                return_details=True,
            )
            rssz_total += _table_weight(weights, category) * rssz_value
            rssz_per_table[f"Household - {category}"] = rssz_value  # Save RSSZ value
            details["table"] = f"Household - {category}"
            diagnostics.append(details)
    else:
        logging.warning("Skipping household-level RSSZ table scoring because solution_df is missing 'abshid'.")

    if return_diagnostics:
        return rssz_total, rssz_per_table, diagnostics

    return rssz_total, rssz_per_table




def calculate_total_marginals(individual_control_dicts, household_control_dicts):
    """
    Calculate the total number of individuals and households required
    based on the control tables.

    Parameters:
    - individual_control_dicts: Control dictionaries for individual-level attributes.
    - household_control_dicts: Control dictionaries for household-level attributes.

    Returns:
    - total_individuals: Total number of individuals required.
    - total_households: Total number of households required.
    """
    def _sum_numeric(value):
        """Recursively sum numeric leaves from nested containers."""
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, dict):
            return sum(_sum_numeric(v) for v in value.values())
        if isinstance(value, (list, tuple, set)):
            return sum(_sum_numeric(v) for v in value)
        return 0.0

    total_individuals = sum(
        _sum_numeric(control_dict) for control_dict in individual_control_dicts.values()
    )
    total_households = sum(
        _sum_numeric(control_dict) for control_dict in household_control_dicts.values()
    )
    # Use the sex-count table as the authoritative individual total.
    # Keys may be bare ('sexp') or prefixed ('sa2sexp') depending on the caller,
    # so match on suffix rather than exact equality.
    sexp_key = next(
        (k for k in individual_control_dicts if k == 'sexp' or k.endswith('sexp')), None
    )
    if sexp_key and individual_control_dicts[sexp_key]:
        total_individuals = _sum_numeric(individual_control_dicts[sexp_key])
    # Use HHCD as the reconciled dwelling-universe total. NPRD is a fallback
    # only when HHCD is not available; using NPRD unconditionally creates the
    # systematic household under-count when HHCD and NPRD have different
    # universes.
    household_key = next(
        (k for k in household_control_dicts if k == 'hhcd' or k.endswith('hhcd')),
        None,
    )
    if household_key is None:
        household_key = next(
            (k for k in household_control_dicts if k == 'nprd' or k.endswith('nprd')),
            None,
        )
    if household_key and household_control_dicts[household_key]:
        total_households = _sum_numeric(household_control_dicts[household_key])
    return int(round(total_individuals)), int(round(total_households))


def calculate_cost(solution, individual_control_dicts, household_control_dicts):
    """
    Calculate the cost of a given solution compared to individual and household control tables.

    Parameters:
    - solution: List of dictionaries representing the current synthetic population solution.
    - individual_control_dicts: Control dictionaries for individual-level attributes.
    - household_control_dicts: Control dictionaries for household-level attributes.

    Returns:
    - Total cost as an integer.
    """
    solution_df = pd.DataFrame(solution)
    total_cost = 0

    # Calculate cost for individual-level attributes
    for category, control_dict in individual_control_dicts.items():
        if not control_dict:
            continue

        sample_key = next(iter(control_dict.keys()), None)
        if sample_key is None:
            continue  # Skip empty control dicts

        category_columns = [key[0] for key in sample_key]
        missing_columns = [col for col in category_columns if col not in solution_df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in solution_df: {missing_columns}")

        actual_counts = Counter(
            tuple(row[col] for col in category_columns) for _, row in solution_df.iterrows()
        )

        for key, target_count in control_dict.items():
            key_tuple = tuple(value for _, value in key)
            actual_count = actual_counts.get(key_tuple, 0)
            total_cost += abs(actual_count - target_count)

    # Calculate cost for household-level attributes
    household_df = solution_df.groupby("abshid").first()  # Use the first record per household
    for category, control_dict in household_control_dicts.items():
        if not control_dict:
            continue

        sample_key = next(iter(control_dict.keys()), None)
        if sample_key is None:
            continue  # Skip empty control dicts

        category_columns = [key[0] for key in sample_key]
        missing_columns = [col for col in category_columns if col not in household_df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in household_df: {missing_columns}")

        actual_counts = Counter(
            tuple(row[col] for col in category_columns) for _, row in household_df.iterrows()
        )

        for key, target_count in control_dict.items():
            key_tuple = tuple(value for _, value in key)
            actual_count = actual_counts.get(key_tuple, 0)
            total_cost += abs(actual_count - target_count)

    return total_cost


def adjust_with_tolerance(new_solution, new_individuals_count, total_individuals, tolerance, households, remaining_households):
    """
    Adjust the new solution to ensure the total individuals count is within the tolerance range.

    Parameters:
    - new_solution: Current solution list.
    - new_individuals_count: Current count of individuals.
    - total_individuals: Target count of individuals.
    - tolerance: Allowed percentage tolerance (e.g., 0.02 for 2%).
    - households: Grouped households (DataFrameGroupBy object).
    - remaining_households: Set of remaining household IDs.

    Returns:
    - Adjusted new_solution and new_individuals_count.
    """
    if total_individuals <= 0:
        return new_solution, new_individuals_count

    while abs(new_individuals_count - total_individuals) / total_individuals > tolerance:
        if new_individuals_count > total_individuals:
            # Remove a household if over the limit
            selected_household = random.choice(
                list(set(row["abshid"] for row in new_solution))
            )
            new_solution = [
                row for row in new_solution if row["abshid"] != selected_household
            ]
            new_individuals_count = sum(1 for row in new_solution)
        elif new_individuals_count < total_individuals and remaining_households:
            # Add a household if under the limit
            selected_household = random.choice(list(remaining_households))
            remaining_households.remove(selected_household)
            household_data = households.get_group(selected_household).to_dict("records")
            new_solution.extend(household_data)
            new_individuals_count += len(household_data)
        else:
            break  # Break if no remaining households to add

    return new_solution, new_individuals_count


def csf_2016():
    """
    Mimics the R function to process CSF 2016 data using the PopulationSynthesiser R package.

    Returns:
    - Merged and processed pandas DataFrame with columns 'regucp' and 'sa2'.
    """
    merged =pd.read_csv('csf_sa.csv')

    return merged


def _built_form_weights_path():
    """Resolve the A2 weights file path, or None if unavailable."""
    try:
        import config as _cfg
        p = os.path.join(_cfg.OUTPUTS_DIR, "built_form", "zone_regucp_weights.csv")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    p = os.path.join("outputs", "built_form", "zone_regucp_weights.csv")
    return p if os.path.exists(p) else None


def load_built_form_dwelling_total(zone):
    """A2 hook: return the Overture dwelling-count estimate for a zone, or None.

    Reads outputs/built_form/zone_dwelling_targets.csv (produced by
    built_form_conditioning.py).  The dwelling estimate is an independent,
    building-footprint-derived household/dwelling total used to anchor the
    household-count target in SA fitting, addressing the household-count
    mismatch (IPU underfilling down to -38%) reported in the paper.
    """
    path = None
    try:
        import config as _cfg
        _p = os.path.join(_cfg.OUTPUTS_DIR, "built_form", "zone_dwelling_targets.csv")
        if os.path.exists(_p):
            path = _p
    except Exception:
        pass
    if path is None:
        _p = os.path.join("outputs", "built_form", "zone_dwelling_targets.csv")
        path = _p if os.path.exists(_p) else None
    if not path:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    col = next((c for c in ("sa2_code", "sa2", "zone") if c in df.columns), None)
    if col is None:
        return None
    zkey = str(zone).zfill(9)
    row = df[df[col].astype(str).str.zfill(9) == zkey]
    if row.empty:
        return None
    est = row.iloc[0].get("dwelling_estimate")
    try:
        est = int(round(float(est)))
    except Exception:
        return None
    return est if est > 0 else None


def built_form_weighted_seed(synthetic_population, zone):
    """
    A2 hook: seed a zone from REGUCP regions weighted by built-form similarity
    (outputs/built_form/zone_regucp_weights.csv) instead of the whole pool.

    Household-atomic: each household is assigned its modal member REGUCP (same
    logic as the single-REGUCP filter below), then only households whose modal
    REGUCP carries positive built-form weight for this zone are retained.  This
    keeps the inner-zone sampling tied to the zone's built form while still
    preserving whole households.

    Returns a sub-population DataFrame, or None when the weights file is absent
    or the zone has no built-form weights (caller keeps current behaviour).
    """
    weights_path = _built_form_weights_path()
    if not weights_path:
        return None
    try:
        import built_form_conditioning as bfc
        weights = bfc.load_zone_regucp_weights(weights_path)
    except Exception:
        return None
    w = weights.get(str(zone))
    if not w:
        return None
    if "abshid" not in synthetic_population.columns or "regucp" not in synthetic_population.columns:
        return None

    _regucp_num = pd.to_numeric(synthetic_population["regucp"], errors="coerce")
    _hh_mode = (
        pd.DataFrame({"abshid": synthetic_population["abshid"], "_r": _regucp_num})
        .groupby(["abshid", "_r"]).size().reset_index(name="_n")
        .sort_values("_n")
        .drop_duplicates("abshid", keep="last")
    )
    _hh_mode["_key"] = _hh_mode["_r"].apply(
        lambda r: str(int(r)) if pd.notna(r) else None)
    _hh_mode["_w"] = _hh_mode["_key"].map(w).fillna(0.0)
    eligible = _hh_mode[_hh_mode["_w"] > 0]
    if eligible.empty:
        return None
    keep_ids = set(eligible["abshid"])
    logging.info(
        f"Zone {zone}: built-form-weighted seed — kept {len(keep_ids):,} of "
        f"{_hh_mode['abshid'].nunique():,} households "
        f"({100.0 * len(keep_ids) / max(_hh_mode['abshid'].nunique(), 1):.1f}% of pool)."
    )
    return synthetic_population[synthetic_population["abshid"].isin(keep_ids)].copy()


def lp_warm_start_households(households_df, household_control_dicts, total_households, total_individuals, zone):
    """
    Mathematical-programming warm start for SA: use a linear program to pick
    a mix of households (grouped into archetypes by joint household-level
    control-attribute signature, e.g. nprd=2,vehrd=1,mrerd=6,...) that best
    matches every household control-table target simultaneously, before SA's
    local search takes over.

    Why this targets the two main failure modes directly:
      1. Population-total underfill: total persons = sum over nprd categories
         of (nprd_value * count_of_households_with_that_nprd), so an LP that
         matches the nprd control table closely automatically matches the
         population total too -- no separate person-count constraint needed.
      2. Household-attribute mismatch (nprd/vehrd/mrerd/strd/rntrd): SA can
         only rearrange whichever households a random initial draw happens to
         include; it can never manufacture a household archetype that a bad
         random start omitted entirely. Solving for the archetype mix
         explicitly, subject to actual pool supply per archetype, gives SA a
         starting point that is already close to optimal for what the pool
         can supply -- and if the pool genuinely lacks an archetype (e.g. too
         few 2-3 person households), the LP reports that as an explicit
         residual rather than silently under/over-filling.

    Returns
    -------
    (selected_ids, diagnostics) -- selected_ids is a list of abshid values to
    seed the initial SA solution, or (None, {"error": ...}) if the LP could
    not be built or solved (caller should fall back to greedy random init).
    """
    from scipy.optimize import linprog

    # Household-level attribute columns that at least one control table targets.
    attr_cols = []
    for control_dict in household_control_dicts.values():
        if not control_dict:
            continue
        sample_key = next(iter(control_dict.keys()))
        for col, _ in sample_key:
            if col not in attr_cols:
                attr_cols.append(col)
    attr_cols = [c for c in attr_cols if c in households_df.columns]
    if not attr_cols:
        return None, {"error": "no household control attributes present on candidate pool"}

    hh_size = households_df.groupby("abshid").size()
    hh_first = households_df.groupby("abshid").first().loc[hh_size.index]

    signatures = hh_first[attr_cols].apply(
        lambda row: tuple((c, _canonical_category_value(row[c])) for c in attr_cols), axis=1
    )
    archetypes = {}
    for abshid, sig in signatures.items():
        archetypes.setdefault(sig, []).append(abshid)

    arch_keys = list(archetypes.keys())
    n = len(arch_keys)
    if n == 0:
        return None, {"error": "no archetypes built from candidate pool"}
    supply = np.array([len(archetypes[k]) for k in arch_keys], dtype=float)
    sig_lookup = {c: i for i, c in enumerate(attr_cols)}

    # One row per (control table, category value) target: coefficient vector
    # over archetypes (1 where the archetype matches that category), and the
    # target count. Categories with no matching archetype in the pool are
    # skipped -- the LP cannot select what does not exist.
    rows, row_labels = [], []
    for category, control_dict in household_control_dicts.items():
        if not control_dict:
            continue
        sample_key = next(iter(control_dict.keys()))
        cols = [c for c, _ in sample_key]
        if not all(c in sig_lookup for c in cols):
            continue
        idxs = [sig_lookup[c] for c in cols]
        for key, target in control_dict.items():
            key_vals = tuple(v for _, v in key)
            coeff = np.array([
                1.0 if tuple(sig[j][1] for j in idxs) == key_vals else 0.0
                for sig in arch_keys
            ])
            if coeff.sum() == 0:
                continue
            rows.append((coeff, float(target)))
            row_labels.append(f"{category}={key_vals}")

    if not rows:
        return None, {"error": "no household control-table category matched the candidate pool"}

    m = len(rows)
    num_vars = n + m  # x_1..x_n archetype counts, d_1..d_m abs-deviation slacks
    c_obj = np.zeros(num_vars)
    for r, (_, target) in enumerate(rows):
        w = 1.0 / max(target, 1.0)
        if row_labels[r].startswith("nprd"):
            w *= 5.0  # nprd directly determines the population total -- prioritise it
        c_obj[n + r] = w

    A_ub_rows, b_ub_rows = [], []
    for r, (coeff, target) in enumerate(rows):
        row1 = np.zeros(num_vars); row1[:n] = coeff; row1[n + r] = -1.0
        A_ub_rows.append(row1); b_ub_rows.append(target)
        row2 = np.zeros(num_vars); row2[:n] = -coeff; row2[n + r] = -1.0
        A_ub_rows.append(row2); b_ub_rows.append(-target)

    # Soft-cap total selected households at min(target, total supply) so
    # supply-constrained zones degrade gracefully instead of the LP erroring.
    achievable_total = min(total_households, float(supply.sum()))
    total_row = np.zeros(num_vars); total_row[:n] = 1.0
    A_ub_rows.append(total_row);  b_ub_rows.append(float(total_households))
    A_ub_rows.append(-total_row); b_ub_rows.append(-achievable_total)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    bounds = [(0, s) for s in supply] + [(0, None)] * m

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None, {"error": f"LP solve failed: {res.message}"}

    x = res.x[:n]
    target_total = int(round(achievable_total))
    floors = np.floor(x).astype(int)
    remainder = x - floors
    deficit = target_total - int(floors.sum())
    counts = floors.copy()
    if deficit > 0:
        for i in np.argsort(-remainder):
            if deficit <= 0:
                break
            if counts[i] < supply[i]:
                counts[i] += 1
                deficit -= 1
    counts = np.minimum(counts, supply.astype(int))

    rng = random.Random((hash(str(zone)) & 0xffffffff))
    selected_ids = []
    for i, k in enumerate(arch_keys):
        pool = archetypes[k]
        n_take = int(counts[i])
        selected_ids.extend(pool if n_take >= len(pool) else rng.sample(pool, n_take))

    diagnostics = {
        "n_archetypes": n,
        "n_control_rows": m,
        "supply_total": int(supply.sum()),
        "target_households": total_households,
        "achievable_households": target_total,
        "selected_households": len(selected_ids),
        "lp_objective": float(res.fun),
    }
    return selected_ids, diagnostics


# ===========================================================================
# Attribute-mutation perturbation  (enable with env var SA_MUTATE=1)
# ---------------------------------------------------------------------------
# In addition to the household add/remove/swap moves, SA may mutate the
# attribute values of individuals or whole families to a different valid census
# category. This gives the annealer far more degrees of freedom to match the
# individual-level margins (which whole-household swaps can barely touch) AND
# makes the fitted records novel (no longer verbatim microdata rows), reducing
# cross-zone cloning and increasing novelty. Mutations are constraint-safe by
# construction:
#   * Individual mutations only touch ADULTS (AGEP code > 14) and only move
#     among the non-"not-applicable" categories, so the age-gate rules
#     (child income/occupation/education/... must be N/A) can never be violated.
#     The N/A code per attribute is learned data-drivenly as the value children
#     overwhelmingly carry.
#   * Family mutations only touch dwelling attributes (VEHRD, STRD) that have no
#     hard cross-attribute constraint, and apply the SAME value to every member
#     of the household, preserving within-household consistency.
# ===========================================================================
_SA_MUTATE_INDIV_ATTRS = ("incp", "heap", "occp", "mtwp", "hrsp", "indp",
                          "qallp", "qalfp", "typp", "stup")
_SA_MUTATE_HH_ATTRS    = ("vehrd", "strd")
_SA_CHILD_AGEP_MAX     = 14   # ABS AGEP codes 0..14 == ages 0..14 (< 15)


def _sa_mutate_enabled():
    return os.environ.get("SA_MUTATE", "0") not in ("0", "", "false", "False")


def _sa_is_single_control_attr(control_dicts, attr):
    """True if a single-attribute control table exists for `attr`."""
    for cd in control_dicts.values():
        if not cd:
            continue
        k0 = next(iter(cd), None)
        if k0 and len(k0) == 1 and k0[0][0] == attr:
            return True
    return False


def _sa_target(control_dicts, attr):
    """Census target counts per category for a single-attribute control table,
    keyed by str(category) so lookups are robust to str/int type differences."""
    for cd in control_dicts.values():
        if not cd:
            continue
        k0 = next(iter(cd), None)
        if k0 and len(k0) == 1 and k0[0][0] == attr:
            return {str(key[0][1]): int(cnt) for key, cnt in cd.items()}
    return None


def build_mutation_context(solution, individual_control_dicts, household_control_dicts):
    """Precompute mutable attributes, their valid categories, and (for age-gated
    individual attributes) the child 'not applicable' code, so proposed mutations
    are constraint-safe. Categories are taken from the population's own values
    (so they are census-valid, present in this zone, and dtype-consistent with the
    counting used by the RSSZ objective). Returns None if disabled/nothing mutable."""
    if not _sa_mutate_enabled():
        return None
    df = pd.DataFrame(solution)
    has_agep = "agep" in df.columns
    if has_agep:
        agep_num = pd.to_numeric(df["agep"], errors="coerce")
        child = agep_num <= _SA_CHILD_AGEP_MAX
    else:
        child = None
    ctx = {"indiv": {}, "hh": {}, "has_agep": has_agep}
    for attr in _SA_MUTATE_INDIV_ATTRS:
        if attr not in df.columns:
            continue
        if not _sa_is_single_control_attr(individual_control_dicts, attr):
            continue
        cats = [v for v in df[attr].dropna().unique().tolist()]
        if len(cats) < 2:
            continue
        na = None
        if child is not None and bool(child.any()):
            vc = df.loc[child, attr].value_counts()
            if len(vc) and (vc.iloc[0] / vc.sum()) > 0.8:
                na = vc.index[0]     # the code children overwhelmingly carry == N/A
        adult_cats = [c for c in cats if c != na]
        if len(adult_cats) >= 2:
            ctx["indiv"][attr] = {"cats": adult_cats, "na": na,
                                  "target": _sa_target(individual_control_dicts, attr) or {}}
    for attr in _SA_MUTATE_HH_ATTRS:
        if attr not in df.columns:
            continue
        if not _sa_is_single_control_attr(household_control_dicts, attr):
            continue
        cats = [v for v in df[attr].dropna().unique().tolist()]
        if len(cats) >= 2:
            ctx["hh"][attr] = {"cats": cats,
                               "target": _sa_target(household_control_dicts, attr) or {}}
    if not ctx["indiv"] and not ctx["hh"]:
        return None
    return ctx


def _sa_deficit_move(cats, target, observed):
    """Given valid categories, census target counts and current observed counts
    (both str-keyed), return (over_cat, under_cat): the most over- and most
    under-represented category relative to the census, or (None, None) if there
    is no meaningful deficit to correct."""
    tot_o = sum(observed.values()) or 1
    tot_t = sum(target.values()) or 1
    def dev(c):
        return observed.get(str(c), 0) / tot_o - target.get(str(c), 0) / tot_t
    over = max(cats, key=dev)
    under = min(cats, key=dev)
    if over == under or dev(over) <= 0 or dev(under) >= 0:
        return None, None
    return over, under


def _sa_is_adult(row, has_agep):
    if not has_agep:
        return True
    try:
        return float(row.get("agep", 99)) > _SA_CHILD_AGEP_MAX
    except (TypeError, ValueError):
        return True


def _sa_attr_deficit(solution, attr, info, level, has_agep):
    """Total absolute proportion deviation of `attr` from its census target
    (0 = perfectly matched). Used to focus the greedy polish on the worst tables."""
    if level == "indiv":
        observed = Counter(str(r.get(attr)) for r in solution)
    else:
        seen = {}
        for r in solution:
            seen.setdefault(r["abshid"], str(r.get(attr)))
        observed = Counter(seen.values())
    target = info.get("target") or {}
    to = sum(observed.values()) or 1
    tt = sum(target.values()) or 1
    cats = set(observed) | {str(c) for c in info["cats"]}
    return sum(abs(observed.get(c, 0) / to - target.get(c, 0) / tt) for c in cats)


def _sa_worst_attr(solution, ctx):
    """Return (level, attr) of the mutable table with the largest current deficit."""
    best, best_d = None, -1.0
    for level, group in (("indiv", ctx["indiv"]), ("hh", ctx["hh"])):
        for attr, info in group.items():
            d = _sa_attr_deficit(solution, attr, info, level, ctx["has_agep"])
            if d > best_d:
                best_d, best = d, (level, attr)
    return best


def propose_mutation(solution, ctx, focus_attr=None, batch=1):
    """Return a new solution with one constraint-safe attribute mutation, or None.
    DIRECTED: moves record(s) from the census-over-represented category to the
    census-under-represented category for the chosen attribute, which directly
    lowers that table's RSSZ (a random category is used only as a fallback).
    `focus_attr=(level, attr)` forces the move onto a specific table (used by the
    greedy polish to target the worst-fitting tables); otherwise it is random.
    `batch` moves up to that many records/households at once (polish uses >1 so each
    RSSZ evaluation makes real progress); the SA loop keeps batch=1.
      Individual: an adult's control attribute -> another non-N/A category.
      Family:     a whole household's VEHRD/STRD -> another category (all members)."""
    if focus_attr is not None:
        do_hh = focus_attr[0] == "hh"
    else:
        do_hh = bool(ctx["hh"]) and (not ctx["indiv"] or random.random() < 0.4)
    n = len(solution)

    if not do_hh and ctx["indiv"]:
        attr = focus_attr[1] if focus_attr is not None else random.choice(list(ctx["indiv"]))
        info = ctx["indiv"][attr]
        na, cats, target = info["na"], info["cats"], info["target"]
        observed = Counter(str(r.get(attr)) for r in solution)
        over, under = _sa_deficit_move(cats, target, observed) if target else (None, None)
        if over is not None:
            # move up to `batch` adults from the over- to the under-represented category
            idxs = [i for i, r in enumerate(solution)
                    if str(r.get(attr)) == str(over) and _sa_is_adult(r, ctx["has_agep"])]
            if idxs:
                new_solution = list(solution)
                for i in random.sample(idxs, min(batch, len(idxs))):
                    new_solution[i] = {**solution[i], attr: under}
                return new_solution
        # fallback: random adult, random non-N/A category
        for _ in range(30):
            i = random.randrange(n)
            p = solution[i]
            if not _sa_is_adult(p, ctx["has_agep"]):
                continue
            if na is not None and p.get(attr) == na:
                continue
            newv = random.choice(cats)
            if newv == p.get(attr):
                continue
            new_solution = list(solution)
            new_solution[i] = {**p, attr: newv}
            return new_solution
        return None

    if do_hh and ctx["hh"]:
        attr = focus_attr[1] if focus_attr is not None else random.choice(list(ctx["hh"]))
        info = ctx["hh"][attr]
        cats, target = info["cats"], info["target"]
        # observed at the household level (one row per household)
        seen = {}
        for r in solution:
            hid = r["abshid"]
            if hid not in seen:
                seen[hid] = str(r.get(attr))
        observed = Counter(seen.values())
        over, under = _sa_deficit_move(cats, target, observed) if target else (None, None)
        if over is not None:
            over_hids = [h for h, v in seen.items() if v == str(over)]
            if over_hids:
                move_hids = set(random.sample(over_hids, min(batch, len(over_hids))))
                new_solution = list(solution)
                for j, r in enumerate(solution):
                    if r["abshid"] in move_hids:
                        new_solution[j] = {**r, attr: under}
                return new_solution
        # fallback: random household, random category
        for _ in range(30):
            i = random.randrange(n)
            hid = solution[i]["abshid"]
            newv = random.choice(cats)
            if newv == solution[i].get(attr):
                continue
            new_solution = list(solution)
            for j, r in enumerate(solution):
                if r["abshid"] == hid:
                    new_solution[j] = {**r, attr: newv}
            return new_solution
        return None
    return None


def simulated_annealing_multi_control_tables(
    synthetic_population, individual_control_tables, household_control_tables, zone, max_iterations, initial_temp, cooling_rate, log_file
):
    # Deterministic stream when SA_RANDOM_SEED is set, so two runs that differ
    # only in the schedule see the same proposals and the comparison measures the
    # schedule rather than the RNG. The zone is folded into the seed so parallel
    # zones do not share a stream.
    _rng_seed = os.environ.get("SA_RANDOM_SEED", "").strip()
    if _rng_seed:
        random.seed(f"{_rng_seed}:{zone}")

    # Setup the logger
    setup_logger(log_file)
    individual_control_dicts = build_control_dicts(individual_control_tables, zone)
    empty_individual_controls = sum(1 for control_dict in individual_control_dicts.values() if not control_dict)
    tolerance = 0.05
    if empty_individual_controls:
        print(
            f"Warning: {empty_individual_controls} individual control dictionaries are empty for zone {zone}. "
            "Continuing with the available controls."
        )
    household_control_dicts = build_control_dicts(household_control_tables, zone)

    households = synthetic_population.groupby("abshid")
    remaining_households = set(synthetic_population["abshid"].unique())
    current_solution = []

    # Compute the total number of individuals required
    total_individuals, total_households = calculate_total_marginals(individual_control_dicts, household_control_dicts)
    if total_individuals <= 0:
        total_individuals = len(synthetic_population)
        logging.warning(
            f"Zone {zone} has no computed individual target; falling back to zone population size {total_individuals}."
        )

    # A2 hook: override the household-count target with the independent Overture
    # building-footprint dwelling estimate when available.  This anchors the
    # household total (consumed by the LP warm start and the sparse-zone cap) to
    # an external source, correcting the census dwelling-table truncation and the
    # household underfill seen in IPU (down to -38% in the paper's figure).
    _bf_dwell = load_built_form_dwelling_total(zone)
    if _bf_dwell and _bf_dwell > 0 and _bf_dwell != total_households:
        logging.info(
            f"Zone {zone}: household total {total_households} -> "
            f"Overture dwelling estimate {_bf_dwell}."
        )
        total_households = _bf_dwell

    # Cap the initial population ONLY for genuinely sparse zones (<50 dwellings
    # per the household-level nprd control table), to avoid overfilling e.g. a
    # zone with 4 dwellings with 800 individuals. NOTE: this previously read
    # "total_households < total_individuals", which is true for almost every
    # normal zone (dwelling count is always < person count once average
    # household size exceeds 1), so it was capping initialisation everywhere,
    # not just in sparse zones -- systematically underfilling the population
    # (e.g. zone 301011001: cap ~1.5x5,881 dwellings => far short of the
    # 16,581-person sa2sexp target once most seeded "households" are small).
    _DWELLING_SPARSE_THRESHOLD = 50
    _hh_limit = (
        int(total_households)
        if 0 < total_households < _DWELLING_SPARSE_THRESHOLD
        else 0
    )
    if _hh_limit > 0:
        logging.info(
            f"Zone {zone}: capping initial pop to {_hh_limit} households "
            f"(ind. total={total_individuals}, hh. total={total_households}) — "
            f"prevents overfill in sparsely populated zones"
        )
        # Use household-limited target: select up to _hh_limit households, then
        # SA will add/remove to converge around the correct household count.
        max_hh_for_init = int(_hh_limit * 1.5)
    else:
        max_hh_for_init = None

    # ------------------------------------------------------------------
    # Initial solution, part 1: LP warm start (mathematical-programming
    # guided). Picks a household mix whose household-level attributes best
    # match all household control tables + the total-household count before
    # SA's local search takes over -- see lp_warm_start_households() for why
    # this directly targets both population-total underfill and household-
    # attribute mismatch. Falls back silently to the pure greedy random init
    # below (unchanged) if the LP fails, scipy is unavailable, or
    # SA_LP_WARM_START=0.
    # ------------------------------------------------------------------
    current_individuals_count = 0
    current_households_count = 0
    if household_control_dicts and os.environ.get("SA_LP_WARM_START", "1") != "0":
        try:
            lp_selected_ids, lp_diag = lp_warm_start_households(
                synthetic_population, household_control_dicts, total_households, total_individuals, zone
            )
        except Exception as exc:
            lp_selected_ids, lp_diag = None, {"error": str(exc)}
        if lp_selected_ids:
            logging.info(f"Zone {zone}: LP warm start selected {len(lp_selected_ids)} "
                         f"households — {lp_diag}")
            for hh_id in lp_selected_ids:
                if hh_id not in remaining_households:
                    continue
                remaining_households.remove(hh_id)
                household_data = households.get_group(hh_id).to_dict("records")
                current_solution.extend(household_data)
                current_individuals_count += len(household_data)
                current_households_count += 1
        else:
            logging.info(f"Zone {zone}: LP warm start unavailable ({lp_diag}) — "
                         "using greedy random init only.")

    # Initial solution, part 2: top up (or, if the LP warm start above was
    # skipped/failed/undershot, build entirely) with greedy random selection
    # until the total number of individuals is reached.
    while current_individuals_count < total_individuals and remaining_households:
        if max_hh_for_init is not None and current_households_count >= max_hh_for_init:
            break
        selected_household = random.choice(list(remaining_households))
        remaining_households.remove(selected_household)
        household_data = households.get_group(selected_household).to_dict("records")
        current_solution.extend(household_data)
        current_individuals_count += len(household_data)
        current_households_count += 1

    # Calculate the initial cost
    #NEW COST current_cost = calculate_cost(current_solution, individual_control_dicts, household_control_dicts)
    current_cost = calculate_total_rssz(current_solution, individual_control_dicts, household_control_dicts)
    best_solution = current_solution
    best_cost = current_cost

    # --- Cooling schedule --------------------------------------------------
    # The objective is a sum of per-table RSSZ, so its scale is set by the zone:
    # O(0.1-1) for a well-seeded zone, O(10-100) for a badly seeded one. A fixed
    # temperature (100, cooled by 0.99) is therefore meaningless: after 200
    # iterations it sits near 13.5, and the cost differences a 1-3 household move
    # produces are 0.001-0.05, so exp(-d/T) ~ 1 and the chain accepts ~99% of
    # every proposal. Measured on 216 zones of the Oct-2026 run, best_cost moved a
    # median of 0.005 over the last 800 of 1000 iterations and got worse in 39% of
    # zones: it was a random walk, not annealing.
    #
    # Scale to the achieved cost instead: start where an uphill move of ~1% of the
    # cost is accepted about half the time, end where only near-flat moves pass,
    # and derive the cooling rate from the iteration budget. SA_TEMP_MODE=fixed
    # restores the old behaviour exactly.
    _temp_mode = os.environ.get("SA_TEMP_MODE", "cost").strip().lower()
    if _temp_mode == "fixed" or current_cost <= 0:
        temperature = initial_temp
        _t_final = initial_temp * (cooling_rate ** max(max_iterations - 1, 0))
    else:
        _temp_frac = float(os.environ.get("SA_TEMP_FRAC", "0.01"))
        # Final temperature is set from the observed move scale, not guessed: a
        # 1-3 household proposal moves the objective by ~0.05% of the total
        # (measured mean |dCost| 0.0002-0.0006 on costs of 0.5-0.9).
        #
        # Measured on two zones, 200 iterations, matched RNG seed, schedule only
        # (SA_ADAPT=0). Uphill acceptance over the final 20 iterations:
        #
        #   fixed T=100,0.99     42.1% / 47.4%   <- never becomes selective
        #   cost-scaled, Tf=5e-4  21.1% / 57.9%   <- still zone-dependent
        #   cost-scaled, Tf=1e-4  15.8% /  5.3%   <- uniformly greedy at the end
        #
        # 1e-4*cost puts a typical uphill move at ~5 dT, so it is rejected ~99%
        # of the time and the last third of the budget behaves as hill-climbing.
        # The final cost over those 200 iterations is a wash between 5e-4 and
        # 1e-4 (0.8023 vs 0.8140 on zone 1, 0.5052 vs 0.5025 on zone 2), so the
        # colder default is chosen for the selectivity property, not for a fit
        # gain it has not demonstrated at this budget.
        _final_frac = float(os.environ.get("SA_TEMP_FINAL_FRAC", "0.0001"))
        temperature = max(_temp_frac * current_cost, 1e-12)
        _t_final = max(_final_frac * current_cost, 1e-15)
        if max_iterations > 1 and temperature > _t_final:
            cooling_rate = (_t_final / temperature) ** (1.0 / max_iterations)
        else:
            _t_final = temperature

    # Log initial state
    logging.info(
        f"Zone: {zone}, Initial Cost: {current_cost}, Initial Temperature: {temperature:.3e} "
        f"(mode={_temp_mode}, final {_t_final:.3e}, cooling_rate={cooling_rate:.6f})"
    )

    # --- Adaptive control ---------------------------------------------------
    # A fixed schedule is still a guess about how much uphill movement the zone
    # needs. Measure it instead: whenever the acceptance rate over the last
    # window leaves the target band, adjust the temperature to bring it back.
    # Patiently-rejected moves mean the chain is frozen on an optimum this
    # neighbourhood cannot leave, so it reheats part-way (less each time, capped)
    # rather than burning the remaining budget rejecting everything; near-total
    # acceptance means it is too hot for this cost scale and gets extra cooling.
    # SA_ADAPT=0 restores purely schedule-driven behaviour.
    _adapt = os.environ.get("SA_ADAPT", "1").strip().lower() not in ("0", "false", "no")
    _acc_lo = float(os.environ.get("SA_ACC_LOW", "0.02"))
    _acc_hi = float(os.environ.get("SA_ACC_HIGH", "0.40"))
    _acc_win = max(10, int(os.environ.get("SA_ACC_WINDOW", "50")))
    _max_reheats = max(0, int(os.environ.get("SA_MAX_REHEATS", "4")))
    # Stagnation trigger: a frozen chain does not look frozen in the acceptance
    # rate (moves that change nothing cost nothing, so they are always accepted).
    # What identifies it is best_cost not moving for a long stretch, which is
    # exactly zone 313011362 in the Oct-2026 run: three improvements in the first
    # three iterations, then nothing for 197 more.
    _stag_window = int(os.environ.get("SA_STAG_WINDOW", "0"))
    if _stag_window <= 0:
        _stag_window = max(100, max_iterations // 5)
    _t_start = temperature
    _win_prop = _win_acc = 0
    _reheats = 0
    _best_seen = best_cost
    _last_improve = 0

    def _cool(cost_diff, accepted):
        """Cool by the schedule, then correct towards the target acceptance band.

        Only *uphill* proposals count towards the acceptance rate. Most
        household moves are neutral (swapping in a household with the same
        attribute mix leaves every cell unchanged), so raw acceptance sits near
        1.0 regardless of temperature -- measured at T=5e-7 on the forced test,
        where 55% of proposals were still "accepted". Neutral moves carry no
        information about whether the chain is too hot or too cold.
        """
        nonlocal temperature, _win_prop, _win_acc, _reheats, _best_seen, _last_improve
        temperature *= cooling_rate
        if best_cost < _best_seen - 1e-12:
            _best_seen = best_cost
            _last_improve = iteration
        if not _adapt:
            return
        if cost_diff is None or cost_diff <= 1e-12:
            return                      # neutral/no-op move: no temperature signal
        _win_prop += 1
        _win_acc += 1 if accepted else 0

        # Stagnation first: the chain is frozen on an optimum this neighbourhood
        # will not leave at the current temperature.
        if (_reheats < _max_reheats and temperature > 0
                and iteration - _last_improve >= _stag_window):
            _reheats += 1
            # Never reheat downwards: a warm-up must exceed the current
            # temperature or it is not a reheat at all.
            temperature = max(temperature, _t_start * (0.5 ** _reheats))
            _last_improve = iteration
            _win_prop = _win_acc = 0
            _msg = (f"  [adapt] iter {iteration + 1}: no best-cost improvement for "
                    f"{_stag_window} iterations -> reheat {_reheats}/{_max_reheats} "
                    f"to T={temperature:.2e}")
            logging.info(_msg)
            print(_msg, flush=True)   # the zone log is cleaned per zone; keep it visible
            return

        if _win_prop < _acc_win:
            return
        acc = _win_acc / float(_win_prop)
        if acc < _acc_lo and _reheats < _max_reheats and temperature > 0:
            _reheats += 1
            temperature = max(temperature, _t_start * (0.5 ** _reheats))
            _last_improve = iteration
            _msg = (f"  [adapt] iter {iteration + 1}: acceptance {acc:.3f} < {_acc_lo} "
                    f"-> reheat {_reheats}/{_max_reheats} to T={temperature:.2e}")
            logging.info(_msg)
            print(_msg, flush=True)
        elif acc > _acc_hi:
            temperature *= 0.5
            _msg = (f"  [adapt] iter {iteration + 1}: acceptance {acc:.3f} > {_acc_hi} "
                    f"-> extra cooling to T={temperature:.2e}")
            logging.info(_msg)
            print(_msg, flush=True)
        _win_prop = _win_acc = 0

    accepted_moves = 0
    improving_moves = 0
    noop_proposals = 0

    # ------------------------------------------------------------------
    # Directed-move support. Random 1-3-household perturbations change the
    # 19-table RSSZ by ~0.1% per move, so in large zones (5k+ households) a
    # 1,000-iteration budget cannot repair a seed whose household-attribute
    # mix is off (observed: zone 303051075 kept 699 flat-dwelling households
    # against a census target of 4 because undirected SA never found the
    # swaps). A directed move looks at the currently worst-fitting household
    # control cell (largest over-representation z-score), removes a household
    # from that cell, and adds one from the most under-represented cell.
    # Fraction of directed proposals is set by SA_DIRECTED_FRAC (default
    # 0.5; set 0 to disable and recover the old behaviour exactly).
    # ------------------------------------------------------------------
    directed_frac = float(os.environ.get("SA_DIRECTED_FRAC", "0.5"))
    hh_first_df = None
    directed_attr_tables = []
    if directed_frac > 0 and household_control_dicts:
        try:
            hh_first_df = synthetic_population.groupby("abshid").first()
            for category, control_dict in household_control_dicts.items():
                if not control_dict:
                    continue
                sample_key = next(iter(control_dict.keys()))
                cols = [c for c, _ in sample_key]
                # single-attribute tables only (all current household tables are)
                if len(cols) == 1 and cols[0] in hh_first_df.columns:
                    targets = {key[0][1]: float(v) for key, v in control_dict.items()}
                    directed_attr_tables.append((category, cols[0], targets))
        except Exception as exc:
            logging.warning(f"Zone {zone}: directed-move setup failed ({exc}); using random moves only.")
            hh_first_df = None
    directed_moves_made = 0

    # Attribute-mutation perturbation (SA_MUTATE=1): context is (re)built from the
    # seeded solution; None disables mutation moves. mutate_frac controls how often
    # a mutation is attempted instead of a household move.
    mutation_ctx = build_mutation_context(current_solution, individual_control_dicts, household_control_dicts)
    mutate_frac = float(os.environ.get("SA_MUTATE_FRAC", "0.3"))
    mutation_moves_made = 0
    # Optional per-iteration cost history for convergence plots from iteration 0
    # (SA_COST_HISTORY=1). Seeded with the initial cost so the curve starts at 0.
    _record_history = os.environ.get("SA_COST_HISTORY", "0") not in ("0", "", "false", "False")
    cost_history = [(0, float(current_cost), float(best_cost))] if _record_history else None

    def _propose_directed(solution_rows, candidate_ids):
        """Remove a household from the most over-represented household-table
        cell and add one from the most under-represented cell. Returns
        (new_solution, n_persons_delta) or None if no directed move exists."""
        if hh_first_df is None or not directed_attr_tables:
            return None
        sol_df = pd.DataFrame(solution_rows)
        if "abshid" not in sol_df.columns or sol_df.empty:
            return None
        sol_hh = sol_df.groupby("abshid").first()

        best_over, best_under = None, None   # (z, category, attr, value)
        for category, attr, targets in directed_attr_tables:
            if attr not in sol_hh.columns:
                continue
            obs = sol_hh[attr].value_counts()
            for val, target in targets.items():
                o = float(obs.get(val, 0.0))
                z = (o - target) / max(target, 1.0) ** 0.5
                if o > target and (best_over is None or z > best_over[0]):
                    best_over = (z, category, attr, val)
                if o < target and (best_under is None or z < best_under[0]):
                    best_under = (z, category, attr, val)
        if best_over is None or best_under is None:
            return None

        _, _, over_attr, over_val = best_over
        _, _, under_attr, under_val = best_under

        remove_ids = sol_hh.index[sol_hh[over_attr] == over_val].tolist()
        if not remove_ids:
            return None
        add_mask = hh_first_df[under_attr] == under_val
        add_ids = [h for h in hh_first_df.index[add_mask] if h in candidate_ids]
        if not add_ids:
            return None

        rm = random.choice(remove_ids)
        ad = random.choice(add_ids)
        rm_size = int((sol_df["abshid"] == rm).sum())
        # prefer an addition of the same household size so the person total
        # stays balanced (sample up to 50 candidates for a size match)
        for cand in random.sample(add_ids, min(50, len(add_ids))):
            if len(households.get_group(cand)) == rm_size:
                ad = cand
                break
        new_rows = [row for row in solution_rows if row["abshid"] != rm]
        added = households.get_group(ad).to_dict("records")
        new_rows.extend(added)
        return new_rows, rm, ad, len(added) - rm_size

    # Simulated annealing loop
    for iteration in range(max_iterations):
        current_hh_ids = {row["abshid"] for row in current_solution}
        proposal_found = False

        # Attribute-mutation proposal (SA_MUTATE=1): mutate an individual's or a
        # whole family's attribute to another valid census category. The abshid set
        # is unchanged, so this is handled here (not via the household-swap path).
        if mutation_ctx is not None and random.random() < mutate_frac:
            mutated = propose_mutation(current_solution, mutation_ctx)
            if mutated is not None:
                new_cost = calculate_total_rssz(mutated, individual_control_dicts, household_control_dicts)
                cost_diff = new_cost - current_cost
                accept_move = cost_diff < 0 or random.random() < math.exp(-cost_diff / max(temperature, 1e-10))
                mutation_moves_made += 1
                if (iteration + 1) % 200 == 0:
                    acc_rate = accepted_moves / float(iteration + 1)
                    print(
                        f"  Zone {zone}: iter {iteration+1}/{max_iterations}  "
                        f"RSSZ={current_cost:.4f}  best={best_cost:.4f}  "
                        f"temp={temperature:.2e}  acc={acc_rate:.3f}  (mut)",
                        flush=True,
                    )
                if accept_move:
                    current_solution = mutated
                    current_cost = new_cost
                    accepted_moves += 1
                    if current_cost < best_cost:
                        best_solution = current_solution
                        best_cost = current_cost
                        improving_moves += 1
                if cost_history is not None:
                    cost_history.append((iteration + 1, float(current_cost), float(best_cost)))
                _cool(cost_diff, accept_move)
                continue

        # Directed proposal: target the worst-fitting household control cell.
        if hh_first_df is not None and random.random() < directed_frac:
            directed = _propose_directed(current_solution, remaining_households)
            if directed is not None:
                new_rows, rm_id, ad_id, person_delta = directed
                candidate_remaining = set(remaining_households)
                candidate_remaining.add(rm_id)
                candidate_remaining.discard(ad_id)
                new_solution = new_rows
                new_individuals_count = current_individuals_count + person_delta
                new_solution, new_individuals_count = adjust_with_tolerance(
                    new_solution, new_individuals_count, total_individuals,
                    tolerance, households, candidate_remaining,
                )
                if {row["abshid"] for row in new_solution} != current_hh_ids:
                    directed_moves_made += 1
                    new_cost = calculate_total_rssz(new_solution, individual_control_dicts, household_control_dicts)
                    cost_diff = new_cost - current_cost
                    accept_move = cost_diff < 0 or random.random() < math.exp(-cost_diff / max(temperature, 1e-10))
                    logging.info(
                        f"[iter {iteration+1}/{max_iterations}] directed: cost={new_cost:.6f} "
                        f"Δ={cost_diff:+.4f}  temp={temperature:.2e}  → "
                        f"{'ACCEPT' if accept_move else 'REJECT'}  |  best={best_cost:.6f}"
                    )
                    if (iteration + 1) % 200 == 0:
                        acc_rate = accepted_moves / float(iteration + 1)
                        print(
                            f"  Zone {zone}: iter {iteration+1}/{max_iterations}  "
                            f"RSSZ={current_cost:.4f}  best={best_cost:.4f}  "
                            f"temp={temperature:.2e}  acc={acc_rate:.3f}",
                            flush=True,
                        )
                    if accept_move:
                        current_solution = new_solution
                        current_cost = new_cost
                        current_individuals_count = new_individuals_count
                        remaining_households = candidate_remaining
                        accepted_moves += 1
                        if current_cost < best_cost:
                            best_solution = current_solution
                            best_cost = current_cost
                            improving_moves += 1
                            logging.info(f"[iter {iteration+1}] ★ NEW BEST: {best_cost:.6f}")
                    if cost_history is not None:
                        cost_history.append((iteration + 1, float(current_cost), float(best_cost)))
                    _cool(cost_diff, accept_move)
                    continue
            # no directed move available this iteration — fall through to random

        # Retry proposal construction until we get a real perturbation.
        for _attempt in range(20):
            new_solution = list(current_solution)
            new_individuals_count = current_individuals_count
            candidate_remaining = set(remaining_households)
            in_solution_households = list({row["abshid"] for row in new_solution})

            feasible_moves = []
            if candidate_remaining:
                feasible_moves.append("add")
            if in_solution_households:
                feasible_moves.append("remove")
            if candidate_remaining and in_solution_households:
                feasible_moves.append("swap")

            if not feasible_moves:
                break

            perturbation_type = random.choice(feasible_moves)

            if perturbation_type == "add":
                num_to_add = random.randint(1, 3)
                for _ in range(num_to_add):
                    if not candidate_remaining:
                        break
                    selected_household = random.choice(list(candidate_remaining))
                    candidate_remaining.remove(selected_household)
                    household_data = households.get_group(selected_household).to_dict("records")
                    new_solution.extend(household_data)
                    new_individuals_count += len(household_data)

            elif perturbation_type == "remove":
                num_to_remove = random.randint(1, 3)
                for _ in range(num_to_remove):
                    in_solution_households = list({row["abshid"] for row in new_solution})
                    if not in_solution_households:
                        break
                    selected_household = random.choice(in_solution_households)
                    new_solution = [
                        row for row in new_solution if row["abshid"] != selected_household
                    ]
                    candidate_remaining.add(selected_household)
                    new_individuals_count = len(new_solution)

            elif perturbation_type == "swap":
                num_swaps = random.randint(1, 3)
                for _ in range(num_swaps):
                    in_solution_households = list({row["abshid"] for row in new_solution})
                    if not in_solution_households:
                        break
                    selected_household_to_remove = random.choice(in_solution_households)
                    swap_in_candidates = list(candidate_remaining - {selected_household_to_remove})
                    if not swap_in_candidates:
                        break
                    new_solution = [
                        row for row in new_solution if row["abshid"] != selected_household_to_remove
                    ]
                    candidate_remaining.add(selected_household_to_remove)
                    selected_household_to_add = random.choice(swap_in_candidates)
                    candidate_remaining.remove(selected_household_to_add)
                    household_data = households.get_group(selected_household_to_add).to_dict("records")
                    new_solution.extend(household_data)
                    new_individuals_count = len(new_solution)

            # Ensure the total number of individuals is within tolerance.
            new_solution, new_individuals_count = adjust_with_tolerance(
                new_solution,
                new_individuals_count,
                total_individuals,
                tolerance,
                households,
                candidate_remaining,
            )

            if {row["abshid"] for row in new_solution} != current_hh_ids:
                proposal_found = True
                break

        if not proposal_found:
            noop_proposals += 1
            if cost_history is not None:
                cost_history.append((iteration + 1, float(current_cost), float(best_cost)))
            _cool(None, False)
            continue

        # Calculate the new cost
        new_cost = calculate_total_rssz(new_solution, individual_control_dicts, household_control_dicts)
        cost_diff = new_cost - current_cost

        # Determine acceptance
        accept_move = cost_diff < 0 or random.random() < math.exp(-cost_diff / max(temperature, 1e-10))
        move_label = "ACCEPT" if accept_move else "REJECT"
        delta_label = f"Δ={cost_diff:+.4f}" if cost_diff >= 0 else f"Δ={cost_diff:.4f}"

        # Log the proposal and decision
        logging.info(
            f"[iter {iteration+1}/{max_iterations}] proposal: cost={new_cost:.6f} {delta_label}  "
            f"temp={temperature:.2e}  → {move_label}  |  best={best_cost:.6f}"
        )

        # Periodic stdout progress (visible in PBS job logs)
        if (iteration + 1) % 200 == 0:
            acc_rate = accepted_moves / float(iteration + 1)
            print(
                f"  Zone {zone}: iter {iteration+1}/{max_iterations}  "
                f"RSSZ={current_cost:.4f}  best={best_cost:.4f}  "
                f"temp={temperature:.2e}  acc={acc_rate:.3f}",
                flush=True,
            )

        # Accept the new solution based on the cost and temperature
        if accept_move:
            current_solution = new_solution
            current_cost = new_cost
            current_individuals_count = new_individuals_count
            remaining_households = candidate_remaining
            accepted_moves += 1
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
                improving_moves += 1
                logging.info(f"[iter {iteration+1}] ★ NEW BEST: {best_cost:.6f}")
        if cost_history is not None:
            cost_history.append((iteration + 1, float(current_cost), float(best_cost)))
        # Reduce the temperature (and adapt to the measured acceptance rate)
        _cool(cost_diff, accept_move)

    # -----------------------------------------------------------------------
    # Greedy directed-mutation polish (SA_POLISH=1, on by default when mutating).
    # Annealing plateaus near the seed because its residual RSSZ is concentrated in
    # a few hard individual tables (e.g. hrsp, qalfp) that random-attribute moves
    # rarely target. This phase deterministically applies directed mutations that
    # STRICTLY reduce RSSZ, biased toward the worst-fitting tables, driving the cost
    # to its achievable floor (a real, monotone convergence recorded in cost_history).
    # -----------------------------------------------------------------------
    polish_moves = 0
    if mutation_ctx is not None and os.environ.get("SA_POLISH", "1") not in ("0", "", "false", "False"):
        budget = int(os.environ.get("SA_POLISH_STEPS", "600"))
        pbatch = int(os.environ.get("SA_POLISH_BATCH", "12"))
        stall = 0
        for ps in range(budget):
            # bias attribute choice toward the worst-fitting table ~70% of the time
            focus = _sa_worst_attr(best_solution, mutation_ctx) if random.random() < 0.7 else None
            cand = propose_mutation(best_solution, mutation_ctx, focus_attr=focus, batch=pbatch)
            if cand is None:
                continue
            c = calculate_total_rssz(cand, individual_control_dicts, household_control_dicts)
            if c < best_cost - 1e-9:
                best_solution = cand
                best_cost = c
                current_cost = c
                polish_moves += 1
                stall = 0
                if cost_history is not None:
                    cost_history.append((max_iterations + ps + 1, float(c), float(best_cost)))
            else:
                stall += 1
                if stall > 80:      # sustained non-improvement => converged
                    break
        logging.info(f"[polish] {polish_moves} improving moves; final best cost {best_cost:.6f}")

    logging.info(
        f"Zone: {zone}, Final Best Cost: {best_cost}, Accepted: {accepted_moves}/{max_iterations}, "
        f"Improving: {improving_moves}, NoOpProposals: {noop_proposals}, "
        f"DirectedMoves: {directed_moves_made} MutationMoves: {mutation_moves_made} "
        f"PolishMoves: {polish_moves} Reheats: {_reheats}"
    )
    if cost_history is not None:
        try:
            _hist_dir = os.environ.get("SA_GAN_RESULTS_DIR", ".")
            os.makedirs(_hist_dir, exist_ok=True)
            pd.DataFrame(cost_history, columns=["iteration", "current_cost", "best_cost"]).to_csv(
                os.path.join(_hist_dir, f"zone_{zone}_cost_history.csv"), index=False)
        except Exception as _exc:
            logging.warning(f"Zone {zone}: could not write cost history: {_exc}")
    return pd.DataFrame(best_solution)

import numpy as np

def flatten_control_dicts(control_dicts):
    """
    Flatten nested control dictionaries into a single dictionary.

    Parameters:
    - control_dicts: Dictionary of control dictionaries.

    Returns:
    - Flattened dictionary with control totals.
    """
    flat_dict = {}
    for category, control_dict in control_dicts.items():
        for key, value in control_dict.items():
            flat_dict[(category, key)] = value
    return flat_dict


def calculate_observed_totals(synthetic_population, control_totals):
    """
    Calculate observed totals for each control group in the synthetic population.

    Parameters:
    - synthetic_population: DataFrame containing the synthetic population.
    - control_totals: Dictionary of control totals with keys that may be hierarchical.

    Returns:
    - observed_totals: Dictionary of observed totals.
    """
    observed_totals = {}

    for (category, nested_keys), target_total in control_totals.items():
        column_names, values = _flat_key_to_columns_values(nested_keys)

        # Check if all column names exist in the synthetic population
        missing_columns = [col for col in column_names if col not in synthetic_population.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in synthetic population: {missing_columns}")

        observed_totals[(category, nested_keys)] = _observed_total_for_key(
            synthetic_population, column_names, values
        )

    return observed_totals


def _flat_key_to_columns_values(nested_keys):
    """Extract (column_names, values) from a flattened control-dict key."""
    if isinstance(nested_keys, tuple):
        if all(isinstance(inner, tuple) for inner in nested_keys):
            # Hierarchical keys (e.g., (('sexp', 1), ('mstp', 2)))
            column_names = [col for col, _ in nested_keys]
            values = [val for _, val in nested_keys]
        else:
            # Flattened keys (e.g., (1, 2))
            column_names = [f"key_{i}" for i in range(len(nested_keys))]
            values = list(nested_keys)
    else:
        raise ValueError(f"Unexpected key structure: {nested_keys}")
    return column_names, values


def _observed_total_for_key(synthetic_population, column_names, values):
    """Weight-summed observed total for ONE control key (column_names/values).

    Split out of calculate_observed_totals() so the IPU loop can recompute a
    single category's observed total after updating its weight, instead of
    recomputing the totals for every category in the control set each time
    (see run_ipu()) -- that used to make each IPU iteration O(N^2) in the
    number of flattened control categories N, which for ~20-30 attributes
    (hundreds of flattened category-value entries) made a single zone take
    hours instead of seconds.
    """
    if len(column_names) == 1:
        filtered = synthetic_population[column_names].isin(values).any(axis=1)
        # NOTE: matches the original (pre-existing) behaviour of counting
        # matching rows rather than summing "weight" for this branch --
        # unchanged here; see separate note about single-column controls
        # not being weight-aware.
        return filtered.sum()
    else:
        grouped = synthetic_population.groupby(column_names).sum(numeric_only=True)
        return grouped.loc[tuple(values), "weight"] if tuple(values) in grouped.index else 0


def plot_predicted_vs_actual(category, control_dict, solution_df, save_path):
    """
    Plot predicted vs. actual counts for a given category.

    Parameters:
    - category: The name of the category being analyzed.
    - control_dict: The control dictionary with expected counts.
    - solution_df: DataFrame representing the synthetic population solution.
    - save_path: Directory to save the plot.

    Returns:
    - None. Saves the plot to the specified path.
    """
    # Extract category columns
    sample_key = next(iter(control_dict.keys()), None)
    if sample_key is None:
        print(f"No data available for category: {category}")
        return
    category_columns = [key[0] for key in sample_key]

    # Calculate observed counts
    observed_counts = Counter(
        tuple(row[col] for col in category_columns) for _, row in solution_df.iterrows()
    )

    # Prepare data for plotting
    keys = [tuple(value for _, value in key) for key in control_dict.keys()]
    predicted = [control_dict[key] for key in control_dict.keys()]
    actual = [observed_counts.get(key_tuple, 0) for key_tuple in keys]

    # Create a DataFrame for easier plotting
    data = pd.DataFrame({
        "Category": [str(key) for key in keys],
        "Predicted": predicted,
        "Actual": actual
    })

    # Sort the data by predicted counts for better visualization
    data = data.sort_values("Predicted", ascending=False)

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    indices = range(len(data))
    plt.bar(indices, data["Predicted"], width=bar_width, label="Predicted", color="blue", alpha=0.7)
    plt.bar(
        [i + bar_width for i in indices],
        data["Actual"],
        width=bar_width,
        label="Actual",
        color="orange",
        alpha=0.7
    )

    # Add labels, title, and legend
    plt.xlabel("Categories", fontsize=12)
    plt.ylabel("Counts", fontsize=12)
    plt.title(f"Predicted vs. Actual for {category}", fontsize=14)
    plt.xticks(
        [i + bar_width / 2 for i in indices],
        data["Category"],
        rotation=90,
        fontsize=10
    )
    plt.legend()

    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{category}_predicted_vs_actual.png")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Saved plot for category '{category}' at: {file_path}")


def plot_all_categories(control_dicts, solution, save_dir):
    """
    Generate and save predicted vs. actual plots for all categories.

    Parameters:
    - control_dicts: Dictionary of control dictionaries (individual or household level).
    - solution: DataFrame representing the synthetic population solution.
    - save_dir: Directory to save the plots.

    Returns:
    - None. Saves all plots to the specified directory.
    """
    for category, control_dict in control_dicts.items():
        if not control_dict:
            print(f"Skipping category '{category}': No data available.")
            continue

        plot_predicted_vs_actual(
            category=category,
            control_dict=control_dict,
            solution_df=solution,
            save_path=save_dir
        )

def generate_new_population_by_sampling(synthetic_population, total_individuals):
    """
    Generate a smaller synthetic population by weighted sampling.

    Parameters:
    - synthetic_population: DataFrame containing the synthetic population with adjusted weights.
    - total_individuals: The target total number of individuals in the new synthetic population.

    Returns:
    - new_population: DataFrame containing the new synthetic population.
    """
    if synthetic_population is None or len(synthetic_population) == 0:
        return pd.DataFrame(columns=getattr(synthetic_population, "columns", []))

    # Clean and normalize weights to sum to 1.
    # IPU can occasionally produce all-zero/NaN weights for sparse zones.
    weights = pd.to_numeric(synthetic_population.get("weight", 1.0), errors="coerce").fillna(0.0)
    weights = weights.clip(lower=0.0)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        weights = pd.Series(np.ones(len(synthetic_population), dtype=float), index=synthetic_population.index)
        total_weight = float(weights.sum())

    synthetic_population["normalized_weight"] = weights / total_weight

    target_n = int(max(0, total_individuals))
    if target_n == 0:
        return synthetic_population.iloc[0:0].drop(columns=["normalized_weight", "weight"], errors="ignore")

    sample_n = min(target_n, len(synthetic_population))

    if "abshid" in synthetic_population.columns:
        # Household-atomic weighted sampling: draw WHOLE households (all
        # members of an abshid together) until the person target is reached.
        # Sampling individual rows (the old behaviour) fragmented households
        # -- each partial household still counted once per its NPRD/VEHRD
        # label, inflating dwelling-level counts well above census totals.
        # Household weight = sum of member IPU weights, so a household's
        # selection probability aggregates its members' fitted weights.
        # Sequential weighted sampling without replacement is done via the
        # Gumbel top-k trick (equivalent to successive draws w/o replacement,
        # O(n log n) instead of np.random.choice's O(n^2) at this size).
        hh_weight = weights.groupby(synthetic_population["abshid"]).sum()
        hh_size = synthetic_population.groupby("abshid").size()
        w = hh_weight.to_numpy(dtype=float)
        w = np.where(w > 0, w, 1e-12)
        gumbel = -np.log(-np.log(np.random.uniform(size=len(w)) + 1e-300) + 1e-300)
        order = np.argsort(-(np.log(w) + gumbel))
        ordered_ids = hh_weight.index.to_numpy()[order]
        ordered_sizes = hh_size.reindex(ordered_ids).to_numpy()

        cum = np.cumsum(ordered_sizes)
        # Take households while the cumulative member count stays within the
        # target; allow the final household to overshoot only if stopping
        # before it would leave the population further from the target.
        n_take = int(np.searchsorted(cum, sample_n, side="left"))
        if n_take < len(cum):
            under = sample_n - (cum[n_take - 1] if n_take > 0 else 0)
            over = cum[n_take] - sample_n
            if over <= under:
                n_take += 1
        chosen = set(ordered_ids[:max(n_take, 1)])
        new_population = synthetic_population[
            synthetic_population["abshid"].isin(chosen)
        ].reset_index(drop=True)
    else:
        # No household identifier available -- person-level weighted sampling.
        sampled_indices = np.random.choice(
            synthetic_population.index,
            size=sample_n,
            replace=False,
            p=synthetic_population["normalized_weight"]
        )
        new_population = synthetic_population.loc[sampled_indices].reset_index(drop=True)

    # Drop unnecessary weight columns
    new_population = new_population.drop(columns=["normalized_weight", "weight"], errors="ignore")

    return new_population


def ipu(synthetic_population, individual_control_dicts, household_control_dicts, max_iterations=100, tolerance=1e-3):
    """
    Perform Iterative Proportional Updating (IPU) to adjust weights of synthetic households.

    Parameters:
    - synthetic_population: DataFrame containing the synthetic population.
    - individual_control_dicts: Control dictionaries for individual-level attributes.
    - household_control_dicts: Control dictionaries for household-level attributes.
    - max_iterations: Maximum number of iterations for the IPU algorithm.
    - tolerance: Convergence tolerance.

    Returns:
    - updated_population: DataFrame with updated weights.
    """
    # Initialize weights for each household
    # Ensure the synthetic_population DataFrame is filtered for valid rows
    #synthetic_population = synthetic_population[synthetic_population["col"] > 0].copy()
    synthetic_population.loc[:, "weight"] = 1.0
    #synthetic_population.loc[:, "weight"] = 1.0
    #synthetic_population["weight"] = 1.0
    # Flatten control dictionaries into a single dictionary for IPU
    control_totals = {}
    control_totals.update(flatten_control_dicts(individual_control_dicts))
    control_totals.update(flatten_control_dicts(household_control_dicts))


    # Preprocess control totals to ensure consistent key structure
    #preprocessed_control_totals = preprocess_expected_counts_ipu(normalized_control_totals)

    # Extract synthetic population marginals (observed totals)
    observed_totals = calculate_observed_totals(synthetic_population, control_totals)

    # Perform IPU iterations
    for iteration in range(max_iterations):
        max_diff = 0

        # Loop through each control total
        for category, target_total in control_totals.items():
            if category not in observed_totals:
                print(f"Skipping category: {category} (not found in observed_totals)")  # Debugging
                continue

            # Calculate the adjustment factor
            observed_total = observed_totals[category]
            adjustment_factor = target_total / observed_total if observed_total > 0 else 1

            # Update weights in the synthetic population
            synthetic_population.loc[:, "weight"] *= adjustment_factor

            # Recalculate the observed total for ONLY this category (not the
            # whole control set -- see _observed_total_for_key docstring).
            # Subsequent categories in this same iteration still see the
            # freshly-updated weights (their turn recomputes their own
            # entry), preserving the original sequential/Gauss-Seidel
            # update order; this only removes the redundant O(N) work of
            # recomputing every OTHER category's total on every step.
            category_name, nested_keys = category
            column_names, values = _flat_key_to_columns_values(nested_keys)
            observed_totals[category] = _observed_total_for_key(
                synthetic_population, column_names, values
            )

            if target_total == 0:
                # Handle division by zero
                if observed_totals[category] == 0:
                    diff = 0  # Perfect match if both are zero
                else:
                    diff = observed_totals[category]
            else:
                # Normal calculation of difference
                diff = abs(target_total - observed_totals[category]) / target_total
            max_diff = max(max_diff, diff)
        # Stop if the maximum difference is within the tolerance
        if max_diff < tolerance:
            print(f"IPU converged after {iteration + 1} iterations.")
            break

    return synthetic_population


def run_ipu(synthetic_population, individual_control_tables, household_control_tables, zone, max_iterations, tolerance):
    """
    Run the IPU algorithm to adjust weights of synthetic households.

    Parameters:
    - synthetic_population: DataFrame containing the synthetic population.
    - individual_control_tables: Control dictionaries for individual-level attributes.
    - household_control_tables: Control dictionaries for household-level attributes.
    - zone: Zone being processed.
    - max_iterations: Maximum number of iterations for IPU.
    - tolerance: Convergence tolerance.

    Returns:
    - updated_population: DataFrame with updated weights.
    """
    # Build control dictionaries for the zone
    individual_control_dicts = build_control_dicts(individual_control_tables, zone)
    household_control_dicts = build_control_dicts(household_control_tables, zone)

    # Run IPU
    updated_population = ipu(
        synthetic_population, individual_control_dicts, household_control_dicts, max_iterations, tolerance
    )

    return updated_population



def build_control_dicts(control_tables, zone):
    """
    Build control dictionaries for a specific zone.

    Parameters:
    - control_tables: Dictionary of control tables.
    - zone: Zone for which to build the control dictionaries.

    Returns:
    - Dictionary of control dictionaries.
    """
    control_dicts = {}
    zone_str = str(zone)
    skip_attrs_raw = os.environ.get("SA_SKIP_CONTROL_ATTRS", "")
    skip_attrs = {x.strip().lower() for x in skip_attrs_raw.split(",") if x.strip()}
    for category, control_table in control_tables.items():
        if category.lower() in skip_attrs:
            logging.info(f"Skipping control category '{category}' via SA_SKIP_CONTROL_ATTRS")
            continue
        control_dict = defaultdict(int)
        data_table = control_table["data"]
        keys = control_table["keys"]

        if "sa2" not in data_table.columns:
            raise KeyError(f"The column 'sa2' is missing in the control table for category '{category}'.")

        filtered_table = data_table[data_table["sa2"].astype(str) == zone_str]
        # Category merges for donor-resolution scoring are applied once at load
        # time (sa_only_baseline._ct_to_entry), so they are already in the table.
        for _, row in filtered_table.iterrows():
            key = tuple((col, _canonical_category_value(row[col])) for col in keys)
            # Skip rows whose mapped value is None (e.g. "Not applicable" or "Not stated"
            # entries that are excluded from SA targets by the CensusRecordImporter mapping).
            if any(v is None for _, v in key):
                continue
            control_dict[key] += int(row["count"])

        control_dicts[category] = control_dict
    return control_dicts



def _resolve_control_csv_path(csv_file_path):
    """Resolve a control CSV path, preferring the declared control basis.

    Search order:
    0) $SA_CONTROL_DIR/<filename>  -- the declared fitting basis (the 19-table
       paper basis).  Checked first so a caller cannot silently fall back to
       data_sa2_controls, which also contains the 7 held-out tables.
    1) control_tables/<filename>
    2) original provided path
    3) data_sa2_controls/<filename>
    """
    base_name = os.path.basename(csv_file_path)
    candidates = []
    basis = os.environ.get("SA_CONTROL_DIR")
    if basis:
        candidates.append(os.path.join(basis, base_name))
    candidates += [
        os.path.join("control_tables", base_name),
        csv_file_path,
        os.path.join("data_sa2_controls", base_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return csv_file_path


def create_control_tables_from_csv(file_paths, zone_column="sa2"):
    """
    Reads multiple CSV files, creates and validates CensusControlTable objects, 
    and organizes them into control tables.

    Parameters:
    - file_paths: Dictionary where keys are attribute names and values are file paths to CSV files.
                  Example: 
                  {
                      "marital_status": {"file": "sexstringnew.csv", "keys": ["sexp", "mstp"]},
                      "vehrd": {"file": "vehrd.csv", "keys": ["vehrd"]}
                  }
    - zone_column: The column name that indicates zones (default is "sa2").

    Returns:
    - control_tables: A dictionary of control tables organized by attribute.
                      Example:
                      {
                          "marital_status": {"keys": ["sexp", "mstp"], "data": validated_table},
                          "vehrd": {"keys": ["vehrd"], "data": validated_table}
                      }
    """
    control_tables = {}

    for attribute, config in file_paths.items():
        csv_file_path = _resolve_control_csv_path(config["file"])
        keys = config["keys"]

        print(f"Processing {attribute} from {csv_file_path}...")

        # Initialize the CensusControlTable object and validate the data
        control_table = CensusControlTable(csv_file_path)
        if attribute == 'relationship':
            print('hold')
        # Rename '1-digit' or '2-digit' columns to the attribute name before validation
        for col in control_table.control_table.columns:
            '''
            if col in ['rlhp']:
                control_table.condense_from_mapping('relationship', 'rlhp', {})
                print(1)
                '''
            if col in ["1-digit", "2-digit"]:
                print(f"Renaming column '{col}' to '{attribute}' in file {csv_file_path}")
                control_table.control_table = control_table.control_table.rename(columns={col: attribute})
            if col in ['sa2']:
                print(f"Converting column '{col}' to string in file {csv_file_path}")
                control_table.control_table["sa2"] = control_table.control_table["sa2"].astype(str)
        validated_table = control_table.validate_and_clean()


        # Ensure the zone column exists
        if zone_column not in validated_table.columns:
            raise KeyError(f"The column '{zone_column}' is missing in the file {csv_file_path}")
        
        # Rename '1-digit' or '2-digit' columns to the attribute name
        '''
        for col in validated_table.columns:
            if col in ["1-digit", "2-digit"]:
                print(f"Renaming column '{col}' to '{attribute}' in file {csv_file_path}")
                validated_table = validated_table.rename(columns={col: attribute})
        '''


        # Add the validated table to the control tables dictionary
        control_tables[attribute] = {"keys": keys, "data": validated_table}

    return control_tables
def validate_synthetic_population(synthetic_population, individual_control_dicts, household_control_dicts):
    """
    Validate the synthetic population against the control tables.

    Parameters:
    - synthetic_population: DataFrame containing the synthetic population.
    - individual_control_dicts: Control dictionaries for individual-level attributes.
    - household_control_dicts: Control dictionaries for household-level attributes.

    Returns:
    - validation_report: A DataFrame summarizing the comparison.
    """
    validation_results = []

    # Validate individual-level attributes
    for category, control_dict in individual_control_dicts.items():
        if not control_dict:
            continue

        sample_key = next(iter(control_dict.keys()), None)
        if sample_key is None:
            continue  # Skip empty control dicts

        category_columns = [key[0] for key in sample_key]
        actual_counts = Counter(
            tuple(row[col] for col in category_columns) for _, row in synthetic_population.iterrows()
        )

        for key, target_count in control_dict.items():
            key_tuple = tuple(value for _, value in key)
            #the problem here is that it reads the key tuple as string, but my counts are converted to ints as per the convert
            actual_count = actual_counts.get(key_tuple, 0)
            diff = actual_count - target_count
            percent_diff = (diff / target_count) * 100 if target_count > 0 else None

            validation_results.append({
                "Category": category,
                "Key": key_tuple,
                "Control Count": target_count,
                "Synthetic Count": actual_count,
                "Difference": diff,
                "Percent Difference (%)": percent_diff
            })

    # Validate household-level attributes
    household_df = synthetic_population.groupby("abshid").first()  # Use the first record per household
    for category, control_dict in household_control_dicts.items():
        if not control_dict:
            continue

        sample_key = next(iter(control_dict.keys()), None)
        if sample_key is None:
            continue  # Skip empty control dicts

        category_columns = [key[0] for key in sample_key]
        actual_counts = Counter(
            tuple(row[col] for col in category_columns) for _, row in household_df.iterrows()
        )

        for key, target_count in control_dict.items():
            key_tuple = tuple(value for _, value in key)
            actual_count = actual_counts.get(key_tuple, 0)
            diff = actual_count - target_count
            percent_diff = (diff / target_count) * 100 if target_count > 0 else None

            validation_results.append({
                "Category": category,
                "Key": key_tuple,
                "Control Count": target_count,
                "Synthetic Count": actual_count,
                "Difference": diff,
                "Percent Difference (%)": percent_diff
            })

    # Convert validation results to a DataFrame
    validation_report = pd.DataFrame(validation_results)
    return validation_report



def prepare_mosaic_data(solution, individual_control_dicts, household_control_dicts):
    """
    Prepare data for a mosaic plot comparing control and solution counts.

    Parameters:
    - solution: DataFrame representing the synthetic population solution.
    - individual_control_dicts: Control dictionaries for individual-level attributes.
    - household_control_dicts: Control dictionaries for household-level attributes.

    Returns:
    - A dictionary formatted for the mosaic plot.
    """
    mosaic_data = {}
    solution_df = pd.DataFrame(solution)
    # Process individual-level attributes
    for category, control_dict in individual_control_dicts.items():
        # Aggregate solution counts for the current category
        category_column = next(iter(control_dict.keys()))[0][0]  # Extract the column name
        solution_counts = solution_df[category_column].value_counts().to_dict()

        # Combine control and solution counts into mosaic_data
        for key, control_count in control_dict.items():
            value = key[0][1]  # Extract the specific value (e.g., 'Male', 'Female', etc.)
            solution_count = solution_counts.get(value, 0)  # Get solution count or default to 0
            mosaic_data[(f"Individual", category, value, "Control")] = control_count
            mosaic_data[(f"Individual", category, value, "Solution")] = solution_count

    # Process household-level attributes
    household_solution = solution_df.groupby("abshid").first()  # Use one record per household
    for category, control_dict in household_control_dicts.items():
        # Aggregate solution counts for the current category
        category_column = next(iter(control_dict.keys()))[0][0]  # Extract the column name
        solution_counts = household_solution[category_column].value_counts().to_dict()

        # Combine control and solution counts into mosaic_data
        for key, control_count in control_dict.items():
            value = key[0][1]  # Extract the specific value (e.g., '1-Person', '2-Person', etc.)
            solution_count = solution_counts.get(value, 0)  # Get solution count or default to 0
            mosaic_data[(f"Household", category, value, "Control")] = control_count
            mosaic_data[(f"Household", category, value, "Solution")] = solution_count

    return mosaic_data


def custom_labelizer(key, data):
    # Extract the subset of the key


    # Try to access the value
    try:
        value = data[key]
    except KeyError:
        print('skip')  # Debug: Print when a key is skipped
        return ""

    # Calculate the proportion
    total = sum(data.values())
    proportion = value / total
    if proportion < 0.05:
        return ""
    return f"{key[-1]} ({proportion:.2%})"

def plot_mosaic(mosaic_data, title_prefix, save_path=None):
    """
    Create and optionally save a mosaic plot using the prepared data.

            logging.warning(
                f"Skipping RSSZ table because solution_df is missing columns: {missing_columns}"
            )
            return 0
    - mosaic_data: Dictionary containing control and solution counts for each category.
    - title: Title of the plot.
    - save_path: Optional path to save the plot (e.g., 'mosaic_plot.png').

    Returns:
    - None. Displays and optionally saves the mosaic plot.
    """
    # Normalize the mosaic data for proportions

    # Extract all unique categories from the data
    """
        Create a single figure with subplots for each category in the data.

        Parameters:
        - mosaic_data: Dictionary containing control and solution counts for each category.
        - title_prefix: Prefix for the title of each subplot (e.g., "Control vs Solution for").
        - columns: Number of columns for subplots in the figure.
        - save_path: Optional path to save the figure (e.g., "mosaic_subplot.png").

        Returns:
        - None. Displays and optionally saves the figure.
        """
    # Extract all unique categories from the data
    categories = sorted(set(key[1] for key in mosaic_data.keys()))
    num_categories = len(categories)
    columns = 3
    # Determine the layout of subplots
    rows = (num_categories + columns - 1) // columns

    # Create the figure and axes for subplots
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for idx, category in enumerate(categories):
        # Filter data for the current category
        category_data = {
            k: v for k, v in mosaic_data.items() if k[1] == category and v > 0
        }

        # Check if any data remains after filtering
        if not category_data:
            print(f"No data available for category: {category}")
            continue

        # Debugging: Print filtered data for the current category
        print(f"Filtered Data for Category '{category}':", category_data)

        # Plot the mosaic for the current category
        ax = axes[idx]

        mosaic(category_data, gap=0.02, ax=ax, labelizer=lambda k:'')
        #ax.set_title(f"{title_prefix} {category}")

    # Hide any unused subplots
    for idx in range(len(categories), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Show the figure
    plt.show()




# Example Usage
if __name__ == "__main__":
    # Define control tables for different categories

    parser = argparse.ArgumentParser(description="Process a specific zone for synthetic population generation.")
    parser.add_argument("--index", type=int, default=0, help="Index of the zone to process (ignored when --zone is given).")
    parser.add_argument("--zone", type=str, default=None, help="SA2 zone code to process directly (overrides --index).")
    parser.add_argument("--method", type=str, default='sa', help="sa or ipu., ipuc")
    parser.add_argument("--dictionary_opts", type=int, default=0, help="index to pass a set of dictionary options")
    parser.add_argument(
        "--iterations", "--max_iterations",
        dest="iterations",
        type=int,
        default=9000,
        help="Number of iterations for the chosen method.",
    )
    parser.add_argument(
        "--temp-mode",
        type=str,
        default=os.environ.get("SA_TEMP_MODE", "cost"),
        choices=["cost", "fixed"],
        help="cost (default): scale the schedule to the achieved cost, starting at "
             "SA_TEMP_FRAC*cost and ending at SA_TEMP_FINAL_FRAC*cost, with the "
             "cooling rate derived from the iteration budget. fixed: use "
             "--initial-temp/--cooling-rate verbatim (pre-Oct-2026 behaviour).",
    )
    parser.add_argument(
        "--initial-temp", "--initial_temp",
        dest="initial_temp",
        type=float,
        default=100.0,
        help="Initial temperature (only used with --temp-mode fixed).",
    )
    parser.add_argument(
        "--cooling-rate", "--cooling_rate",
        dest="cooling_rate",
        type=float,
        default=0.99,
        help="Simulated-annealing cooling rate.",
    )
    parser.add_argument(
        "--control-scope", "--control_scope",
        type=str,
        default=os.environ.get("SA_CONTROL_SCOPE", "seq"),
        choices=["seq", "statewide"],
        help="Control-table geography scope: seq uses legacy control files, statewide uses data_sa2_controls/sa2*.csv",
    )
    parser.add_argument(
        "--seed-file", "--seed_file",
        type=str,
        default=None,
        help="Override the seed microdata CSV file (default: cleaned_dataset_hhid.csv for SA/IPU; "
             "census_microdata_fitting_sample.csv for ipuc).",
    )
    parser.add_argument(
        "--output-dir", "--output_dir",
        type=str,
        default=None,
        help="Override the output directory for zone files.  "
             "Defaults to zones_{method}_{dictionary_opts}/ in the current working directory.",
    )
    args = parser.parse_args()
    index = args.index
    method = args.method
    dictionary_opts = args.dictionary_opts
    control_scope = args.control_scope

    # Override output directory if requested (used by IPU comparison job)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    # Canonical output directory for this run (may differ from the SA default)
    _OUTDIR = args.output_dir if args.output_dir else f"zones_{method}_{dictionary_opts}"

    '''INDIVIDUAL'''
    # importy the data
    h = Helper()
    if args.seed_file:
        data_file = args.seed_file
    elif method == 'ipuc':
        print('doint census onlu')
        data_file = 'census_microdata_fitting_sample.csv'
        #ta_file = "census_csv.csv"
    else:
        data_file = 'cleaned_dataset_hhid.csv'
    # Detect whether data file uses AREAENUM as an alias for regucp
    _probe = pd.read_csv(data_file, nrows=0)
    _probe.columns = _probe.columns.str.lower()
    _regucp_col = 'regucp' if 'regucp' in _probe.columns else ('areaenum' if 'areaenum' in _probe.columns else 'regucp')
    require_col = ['absfid', 'abspid', 'abshid', _regucp_col]

    # Load ALL columns from the seed file so the full GAN variable set is available
    # for SA fitting, IPU weighting, and downstream tour-model consumers.
    # clean_my_datagan(None) returns every column lowercased + numeric-coerced.
    d = h.clean_my_datagan(data_file, None)
    # Verify required columns survived
    _missing_req = [c for c in require_col if c not in d.columns]
    if _missing_req:
        raise ValueError(f"Seed file {data_file!r} is missing required columns: {_missing_req}")
    # Normalise region column name to 'regucp' regardless of source alias
    if _regucp_col != 'regucp' and _regucp_col in d.columns:
        d = d.rename(columns={_regucp_col: 'regucp'})


    if control_scope == "statewide":
        # Statewide SA2 controls from data_sa2_controls/ use sa2*.csv names.
        # Keep only tables with robust name/key alignment to the synthetic population.
        file_paths_ind = {
            "mstp": {"file": "sa2mstp.csv", "keys": ["mstp"]},
            "occp": {"file": "sa2occp.csv", "keys": ["occp"]},
            "mtwp": {"file": "sa2mtwp.csv", "keys": ["mtwp"]},
            "heap": {"file": "sa2heap.csv", "keys": ["heap"]},
            "sexp": {"file": "sa2sexp.csv", "keys": ["sexp"]},
            "relationship": {"file": "sa2rlhp.csv", "keys": ["rlhp"]},
            "qallp": {"file": "sa2qallp.csv", "keys": ["qallp"]},
            "qalfp": {"file": "sa2qalfp.csv", "keys": ["qalfp"]},
            "incp": {"file": "sa2incp.csv", "keys": ["incp"]},
            "hrsp": {"file": "sa2hrsp.csv", "keys": ["hrsp"]},
            "indp": {"file": "sa2indp.csv", "keys": ["indp"]},
            "typp": {"file": "sa2typp.csv", "keys": ["typp"]},
            "stup": {"file": "sa2stup.csv", "keys": ["stup"]},
        }

        file_paths_hhd = {
            "nprd": {"file": "sa2nprd.csv", "keys": ["nprd"]},
            "hind": {"file": "sa2hind.csv", "keys": ["hind"]},
            "vehrd": {"file": "sa2vehrd.csv", "keys": ["vehrd"]},
            "mrerd": {"file": "sa2mrerd.csv", "keys": ["mrerd"]},
            "rntrd": {"file": "sa2rntrd.csv", "keys": ["rntrd"]},
            "strd": {"file": "sa2strd.csv", "keys": ["strd"]},
        }
    elif dictionary_opts == 0:
        'individual opts'
        file_paths_ind = {
        "marital_status": {"file": "sexstringnew.csv", "keys": ["sexp", "mstp"]},
        "occp": {"file": "saoccu.csv", "keys": ["occp"]},
        "mtwp": {"file": "mtwp.csv", "keys": ["mtwp"]},
        "heap": {"file": "heap.csv", "keys": ["heap"]},
        "sexp": {"file": "sex.csv", "keys": ["sexp"]},
        "relationship": {"file": "rlhp.csv", "keys": ["rlhp"]},
            "gngp": {"file": "gngp.csv", "keys": ["gngp"]},
            "empp": {"file": "empp.csv", "keys": ["empp"]},
            "qallp": {"file": "qallp.csv", "keys": ["qallp"]},
            "qalfp": {"file": "qalfp.csv", "keys": ["qalfp"]},
            "incp": {"file": "incp.csv", "keys": ["incp"]},
            "hrsp": {"file": "hrsp.csv", "keys": ["hrsp"]},
            "indp": {"file": "indp.csv", "keys": ["indp"]},
            "typp": {"file": "typp.csv", "keys": ["typp"]},
        "stup": {"file": "stup.csv", "keys": ["stup"]},
        "siemp": {"file": "siemp.csv", "keys": ["siemp"]},
        "spip": {"file": "spip.csv", "keys": ["spip"]},
        }

        ''' household files'''

        file_paths_hhd = {
        "nprd": {"file": "nprd.csv", "keys": ["nprd"]},
        "hind": {"file": "hind.csv", "keys": ["hind"]},
        "vehrd": {"file": "vehrd.csv", "keys": ["vehrd"]},
        "mrerd": {"file": "mrerd.csv", "keys": ["mrerd"]},
        "rntrd": {"file": "rntrd.csv", "keys": ["rntrd"]},
        "tenlld": {"file": "tenlld.csv", "keys": ["tenlld"]},
        "strd": {"file": "strd.csv", "keys": ["strd"]},
        }
    elif dictionary_opts == 1:
        'individual opts'
        file_paths_ind = {
            "marital_status": {"file": "sexstringnew.csv", "keys": ["sexp", "mstp"]},
            "occp": {"file": "saoccu.csv", "keys": ["occp"]},
            "mtwp": {"file": "mtwp.csv", "keys": ["mtwp"]},
            "heap": {"file": "heap.csv", "keys": ["heap"]},
            "qallp": {"file": "qallp.csv", "keys": ["qallp"]},
            "qalfp": {"file": "qalfp.csv", "keys": ["qalfp"]},
            "incp": {"file": "incp.csv", "keys": ["incp"]},
            "hrsp": {"file": "hrsp.csv", "keys": ["hrsp"]},
        }

        ''' household files'''

        file_paths_hhd = {
            "nprd": {"file": "nprd.csv", "keys": ["nprd"]},
            "hind": {"file": "hind.csv", "keys": ["hind"]},
            "vehrd": {"file": "vehrd.csv", "keys": ["vehrd"]},
            "mrerd": {"file": "mrerd.csv", "keys": ["mrerd"]},
        }
    elif dictionary_opts ==2:
        'individual opts'
        file_paths_ind = {
            "marital_status": {"file": "sexstringnew.csv", "keys": ["sexp", "mstp"]},
            "relationship": {"file": "rlhp.csv", "keys": ["rlhp"]},
            "empp": {"file": "empp.csv", "keys": ["empp"]},
            "hrsp": {"file": "hrsp.csv", "keys": ["hrsp"]},
        }

        ''' household files'''

        file_paths_hhd = {
            "vehrd": {"file": "vehrd.csv", "keys": ["vehrd"]},
            "mrerd": {"file": "mrerd.csv", "keys": ["mrerd"]},
            "hind": {"file": "hind.csv", "keys": ["hind"]},
        }
    elif dictionary_opts ==3:
        #benchmark test against paper
        'individual opts'
        file_paths_ind = {
            "marital_status": {"file": "sexstringnew.csv", "keys": ["sexp", "mstp"]},
            "occp": {"file": "saoccu.csv", "keys": ["occp"]},
            "mtwp": {"file": "mtwp.csv", "keys": ["mtwp"]},
            "heap": {"file": "heap.csv", "keys": ["heap"]}
        }

        ''' household files'''

        file_paths_hhd = {
            "nprd": {"file": "nprd.csv", "keys": ["nprd"]},
            "hind": {"file": "hind.csv", "keys": ["hind"]},
            "vehrd": {"file": "vehrd.csv", "keys": ["vehrd"]},
            "mrerd": {"file": "mrerd.csv", "keys": ["mrerd"]},
            "rntrd": {"file": "rntrd.csv", "keys": ["rntrd"]},
        }



    individual_control_tables = create_control_tables_from_csv(file_paths_ind)


    # Define helper function to determine the range category
    def get_range_category(key):
        # Special cases
        if key == "Not stated":
            return "Not stated"
        if key == "Not applicable":
            return "Less than 1 hour and Not applicable"
        if key == "Overseas visitor":
            return "Overseas visitor"

        # Check for ranges
        try:
            hours = int(key.split()[0])  # Extract numeric value from the key
            if hours < 1:
                return "Less than 1 hour and Not applicable"
            elif 1 <= hours <= 15:
                return "1-15 hours"
            elif 16 <= hours <= 24:
                return "16-24 hours"
            elif 25 <= hours <= 34:
                return "25-34 hours"
            elif 35 <= hours <= 39:
                return "35-39 hours"
            elif hours == 40:
                return "40 hours"
            elif 41 <= hours <= 49:
                return "41-49 hours"
            elif hours >= 50:
                return "50 hours and over"
        except ValueError:
            return None  # For unexpected cases
    # Map each hrsp key to the smaller dictionary
    # Initialize a mapping dictionary
    mapped_hrsp = {}


    # Function to map `hrsp` values to `hrsp_m` categories
    def map_hrsp_to_hrsp_m(hrsp_value):
        if hrsp_value == 101:
            return 9  # "Not stated"
        elif hrsp_value == 102:
            return 1  # "Less than 1 hour and Not applicable"
        elif hrsp_value == 103:
            return 10  # "Overseas visitor"
        elif hrsp_value <= 15:
            return 2  # "1-15 hours"
        elif 16 <= hrsp_value <= 24:
            return 3  # "16-24 hours"
        elif 25 <= hrsp_value <= 34:
            return 4  # "25-34 hours"
        elif 35 <= hrsp_value <= 39:
            return 5  # "35-39 hours"
        elif hrsp_value == 40:
            return 6  # "40 hours"
        elif 41 <= hrsp_value <= 49:
            return 7  # "41-49 hours"
        elif hrsp_value >= 50:
            return 8  # "50 hours and over"
        else:
            return None  # For unexpected cases


    hrsp_m_reverse = {
        1: "Less than 1 hour and Not applicable",
        2: "1-15 hours",
        3: "16-24 hours",
        4: "25-34 hours",
        5: "35-39 hours",
        6: "40 hours",
        7: "41-49 hours",
        8: "50 hours and over",
        9: "Not stated",
        10: "Overseas visitor"
    }

    # Map `hrsp` to `hrsp_m`
    if 'hrsp' in individual_control_tables:
        df_hrsp = individual_control_tables['hrsp']['data']
        if 'hrsp' in df_hrsp.columns:
            # Legacy SEQ files use numeric hour codes; statewide files may already
            # be categorical bins encoded as strings. Only remap in numeric mode.
            hrsp_numeric = pd.to_numeric(df_hrsp['hrsp'], errors='coerce')
            if hrsp_numeric.notna().all():
                max_code = int(hrsp_numeric.max())
                if max_code > 10:
                    # Individual hour codes (1-103 from SEQ files) - remap to bins
                    df_hrsp['hrsp_m'] = hrsp_numeric.astype(int).apply(map_hrsp_to_hrsp_m)
                    aggregated_df = df_hrsp.groupby(['sa2', 'hrsp_m'], as_index=False)['count'].sum()
                    aggregated_df.rename(columns={'hrsp_m': 'hrsp'}, inplace=True)
                    individual_control_tables['hrsp']['data'] = aggregated_df
                else:
                    # Already binned (1-10) by hrsp_m mapping in CensusRecordImporter - just aggregate
                    aggregated_df = df_hrsp.groupby(['sa2', 'hrsp'], as_index=False)['count'].sum()
                    individual_control_tables['hrsp']['data'] = aggregated_df

    #how to then put in the hrsp_




    household_control_tables = create_control_tables_from_csv(file_paths_hhd)

    # ── Diagnostic: check every control-table key is present in the pool ────
    # Catches attribute mismatches early (before the SA loop) so failures
    # are visible at startup rather than silently producing wrong RSSZ values.
    _pool_cols = set(d.columns)
    _all_ctrl = {**individual_control_tables, **household_control_tables}
    _missing_keys = []
    _present_keys = []
    for _ct_name, _ct_entry in _all_ctrl.items():
        for _key in _ct_entry.get("keys", []):
            if _key in _pool_cols:
                _present_keys.append(_key)
            else:
                _missing_keys.append((_ct_name, _key))
    print(f"\n  Control table / pool linkage ({control_scope} scope):")
    print(f"    Control tables loaded : {len(individual_control_tables)} individual + "
          f"{len(household_control_tables)} household")
    print(f"    Pool columns matched  : {len(_present_keys)}")
    if _missing_keys:
        print(f"    Control attrs missing from pool: "
              f"{[f'{n}:{k}' for n, k in _missing_keys]}")
    else:
        print(f"    Control attrs missing from pool: NONE - all good")
    del _pool_cols, _all_ctrl, _missing_keys, _present_keys, _ct_name, _ct_entry, _key





    # Process each zone

    # Compute the intersection of zones across all control tables
    # zones = set(control_table["sa2"].unique() for control_table in control_tables.values())

    '''
    zones = set(
        zone
        for control_table in individual_control_tables.values()
        for zone in (
            control_table["data"]["sa2"].unique()
            if isinstance(control_table, dict) and "data" in control_table
            else control_table["sa2"].unique()
        )
    )
    zone = [str(zone) for zone in zones]
    '''


    zones = sorted(
        set(
            zone
            for control_table in individual_control_tables.values()
            for zone in (
                control_table["data"]["sa2"].unique()
                if isinstance(control_table, dict) and "data" in control_table
                else control_table["sa2"].unique()
            )
        )
    )
    zone = [str(zone) for zone in zones]
    # Map the index (or --zone code) to the corresponding zone
    if args.zone is not None:
        zone = args.zone
        # Validate: warn if zone not in control-table universe, but proceed anyway
        zone_strs = [str(z) for z in zones]
        if zone not in zone_strs:
            print(f"Warning: zone {zone} not found in individual control tables; proceeding with available data.")
    else:
        if index < 0 or index >= len(zones):
            raise ValueError(f"Invalid index: {index}. Must be between 0 and {len(zones) - 1}.")
        zone = zones[index]


    # Parameters for simulated annealing
    max_iterations = args.iterations
    initial_temp = args.initial_temp
    cooling_rate = args.cooling_rate
    # Ensure the 'zones_sa' directory exists
    os.makedirs(_OUTDIR, exist_ok=True)

    # Process the specified zone
    print(f"\nProcessing Zone: {zone} (Index: {index})...")

    # Filter the synthetic population for the specified zone
    csf_data = csf_2016()
    filtered_zone_data = csf_data[csf_data["sa2"].astype(str).str.strip() == str(zone)].reset_index(drop=True)
    if len(filtered_zone_data) == 0:
        logging.warning(
            f"Zone {zone}: missing in csf_sa.csv; using full seed pool without REGUCP filtering."
        )
        regucp_values = None
    else:
        # csf_sa.csv stores regucp as strings ('42'); clean_my_datagan may contain float64 (42.0).
        # Cast both sides to float so '42' == 42.0 comparison works correctly.
        regucp_values = float(pd.to_numeric(filtered_zone_data.iloc[0]["regucp"], errors="coerce"))

    #TODO need to inspect the control_tables
    # Define the specific zone you want to filter for

    _balance_audit_path = os.path.join(_OUTDIR, f"zone_{zone}_control_balance_audit.csv")
    individual_control_tables, household_control_tables = normalize_counts(
        individual_control_tables,
        household_control_tables,
        audit_path=_balance_audit_path,
        audit_focus_zone=str(zone),
    )

    total_individual_controls = 0
    total_household_controls = 0

    # Iterate through each key in individual_control_tables
    for key, control_table in individual_control_tables.items():
        # Access the DataFrame for the current key
        data = control_table['data']

        # Filter the DataFrame for the specific sa2 zone
        filtered_data = data[data['sa2'] == zone]

        # Sum the 'count' column for the filtered rows
        total_count_p = filtered_data['count'].sum()

        total_individual_controls += int(total_count_p)

        # Print the result for the current key
        print(f"Total count for sa2 = {zone} in {key}: {total_count_p}")

    for key, control_table in household_control_tables.items():
        # Access the DataFrame for the current key
        data = control_table['data']

        # Filter the DataFrame for the specific sa2 zone
        filtered_data = data[data['sa2'] == zone]

        # Sum the 'count' column for the filtered rows
        total_count = filtered_data['count'].sum()

        total_household_controls += int(total_count)

        # Print the result for the current key
        print(f"Total count for sa2 = {zone} in {key}: {total_count}")

    # now do the same for


    synthetic_population = d.copy()
    if regucp_values is None:
        # A2 hook: prefer a built-form-weighted seed over the raw full pool.
        _bf_seed = built_form_weighted_seed(synthetic_population, zone)
        sub_population = _bf_seed if _bf_seed is not None else synthetic_population.copy()
    elif "abshid" in synthetic_population.columns:
        # Household-atomic REGUCP filter. REGUCP is a person-level attribute
        # and Stage 2's household matcher does not condition on it, so one
        # household's members can carry different REGUCP codes. Filtering
        # person rows directly (the old behaviour) sliced such households
        # apart, seeding SA/IPU with pre-fragmented "households" -- e.g. West
        # End's fitted output averaged 1.53 members/household against the
        # pool's 2.3, inflating distinct-household counts ~70% above census
        # dwelling totals. Assign each household its modal member REGUCP and
        # keep or drop the household WHOLE.
        _regucp_num = pd.to_numeric(synthetic_population["regucp"], errors="coerce")
        _hh_mode = (
            pd.DataFrame({"abshid": synthetic_population["abshid"], "_r": _regucp_num})
            .groupby(["abshid", "_r"]).size().reset_index(name="_n")
            .sort_values("_n")
            .drop_duplicates("abshid", keep="last")
            .set_index("abshid")["_r"]
        )
        _keep_ids = set(_hh_mode[_hh_mode == regucp_values].index)
        sub_population = synthetic_population[
            synthetic_population["abshid"].isin(_keep_ids)
        ].copy()
    else:
        sub_population = synthetic_population[
            pd.to_numeric(synthetic_population["regucp"], errors="coerce") == regucp_values
        ].copy()

    if sub_population.empty:
        total_controls = total_individual_controls + total_household_controls
        out_dir = _OUTDIR
        os.makedirs(out_dir, exist_ok=True)

        if total_controls == 0:
            # Legitimate zero-population zone: emit explicit zero artifacts and exit cleanly.
            print(
                f"Zone {zone} has zero control totals and no seed records; "
                "writing zero-population artifacts and skipping SA/IPU."
            )
            empty_cols = list(synthetic_population.columns)
            pd.DataFrame(columns=empty_cols).to_csv(
                os.path.join(out_dir, f"zone_{zone}_synthetic_population.csv"), index=False
            )
            pd.DataFrame(columns=[
                "Category", "Key", "Control Count", "Synthetic Count", "Difference", "Percent Difference (%)"
            ]).to_csv(os.path.join(out_dir, f"zone_{zone}_validation_report.csv"), index=False)
            pd.DataFrame({"zone": [zone], "rssz_value": [0.0]}).to_csv(
                os.path.join(out_dir, f"zone_{zone}_rssz_value.csv"), index=False
            )
            pd.DataFrame(columns=["Table", "RSSZ"]).to_csv(
                os.path.join(out_dir, f"zone_{zone}_rssz_values_cat.csv"), index=False
            )
            pd.DataFrame(columns=[
                "category_columns", "expected_keys", "observed_keys", "matched_keys",
                "expected_key_coverage", "observed_key_coverage", "expected_total",
                "unmatched_expected_total", "unmatched_expected_share", "missing_columns",
                "rssz", "coverage_threshold", "excluded_reason"
            ]).to_csv(os.path.join(out_dir, f"zone_{zone}_rssz_diagnostics.csv"), index=False)
            sys.exit(0)

        # REGUCP not found in the seed pool (e.g. Tweed Heads zones, REGUCP 5, not
        # in the GAN training states). Fall back to a built-form-weighted seed when
        # available, otherwise to a CAPPED sample of the pool.
        #
        # The cap matters: the old fallback used the whole pool, so on the national
        # census seed (1.25M rows) every unmatched zone paid a full-population IPU
        # (O(N x categories x iterations)) and was killed by the 4-hour per-zone
        # timeout.
        _cap = int(os.environ.get("IPU_FALLBACK_SEED_CAP", "60000"))
        logging.warning(
            f"Zone {zone}: REGUCP {regucp_values} not found in seed pool "
            f"({total_controls} controls, pool size {len(synthetic_population):,}). "
            f"Falling back to a capped seed (<= {_cap:,} rows)."
        )
        _bf_seed = built_form_weighted_seed(synthetic_population, zone)
        if _bf_seed is not None:
            sub_population = _bf_seed
        elif len(synthetic_population) > _cap:
            sub_population = synthetic_population.sample(
                n=_cap, random_state=0).reset_index(drop=True)
        else:
            sub_population = synthetic_population.copy()



    os.makedirs(_OUTDIR, exist_ok=True)
    '''IPU'''
    if method == 'ipu' or method == 'ipuc':
        # Use the full REGUCP-filtered pool for IPU weight adjustment.
        # (The old .iloc[0:total_count_p] sliced by the last control-table's count —
        # whichever table happened to be last in the dict — giving an arbitrary and
        # often tiny subset.  IPU needs the full pool so it can assign representative
        # weights before generate_new_population_by_sampling draws the right N.)
        sub_population_c = sub_population.copy()
        _avail_cols = set(sub_population_c.columns)
        _ipu_ind = {k: v for k, v in individual_control_tables.items()
                     if set(v.get("keys", [])) <= _avail_cols}
        _ipu_hh  = {k: v for k, v in household_control_tables.items()
                     if set(v.get("keys", [])) <= _avail_cols}
        _dropped_ind = len(individual_control_tables) - len(_ipu_ind)
        _dropped_hh  = len(household_control_tables) - len(_ipu_hh)
        if _dropped_ind or _dropped_hh:
            print(f"  IPU: dropped {_dropped_ind} ind + {_dropped_hh} hh control tables "
                  f"(columns absent from seed data); "
                  f"keeping {len(_ipu_ind)} ind + {len(_ipu_hh)} hh")
        # Run IPU for the zone.
        # IPU tolerance was pinned at 1e-10, which is unreachable in practice:
        # every zone ran the full 1000-iteration budget and hit the 4-hour
        # per-zone timeout (stage3c). 1e-6 is far below any reporting precision
        # and converges in a few dozen passes. Both knobs are overridable.
        _ipu_tol = float(os.environ.get("IPU_TOLERANCE", "1e-6"))
        _ipu_max_iter = int(os.environ.get("IPU_MAX_ITER", str(max_iterations)))
        _ipu_max_iter = max(1, min(_ipu_max_iter, int(max_iterations)))
        updated_population = run_ipu(
            sub_population_c,
            _ipu_ind,
            _ipu_hh,
            zone,
            max_iterations=_ipu_max_iter,
            tolerance=_ipu_tol
        )
        _ipu_ind_dicts = build_control_dicts(_ipu_ind, zone)
        _ipu_hh_dicts = build_control_dicts(_ipu_hh, zone)
        # Determine the target total number of individuals from the marginals
        total_individuals, _ = calculate_total_marginals(_ipu_ind_dicts, _ipu_hh_dicts)
        if total_individuals <= 0:
            total_individuals = len(updated_population)
            logging.warning(
                f"Zone {zone}: no positive IPU individual target from controls; "
                f"falling back to seed length {total_individuals}."
            )
        total_individuals = min(int(total_individuals), int(len(updated_population)))
        # Generate the new synthetic population
        new_population = generate_new_population_by_sampling(updated_population, total_individuals)

        cols_lower = {c.lower(): c for c in new_population.columns}
        if "abshid" not in cols_lower and "householdid" in cols_lower:
            new_population = new_population.rename(columns={cols_lower["householdid"]: "abshid"})
        if "abshid" not in new_population.columns:
            raise KeyError(
                "IPU output is missing 'abshid'; household-level comparison cannot run. "
                f"Seed columns were: {list(sub_population.columns)}"
            )

        # Step 3 save the new population to a CSV file
        new_population.to_csv(os.path.join(_OUTDIR, f"zone_{zone}_new_population.csv"), index=False)
        new_population.to_csv(os.path.join(_OUTDIR, f"zone_{zone}_synthetic_population.csv"), index=False)

        # Convert the new population to a list of dictionaries (if needed for the RSSZ function)
        new_population_records = new_population.to_dict("records")
        # Calculate RSSZ for the new population
        rssz_value = calculate_total_rssz(
            solution=new_population_records,
            individual_control_dicts=_ipu_ind_dicts,
            household_control_dicts=_ipu_hh_dicts
        )

        # Print the RSSZ value
        print(f"RSSZ for the new synthetic population: {rssz_value}")

        # Step 5: Save the RSSZ value
        with open(os.path.join(_OUTDIR, f"zone_{zone}_rssz_value.txt"), "w") as f:
            f.write(f"RSSZ for the new synthetic population: {rssz_value}\n")
        print(f"RSSZ value saved to 'zones_{method}_{dictionary_opts}/zone_{zone}_rssz_value.txt'.")

        rssz_total, rssz_per_table, rssz_diagnostics = calculate_total_rssz_table(
            solution=new_population_records,
            individual_control_dicts=_ipu_ind_dicts,
            household_control_dicts=_ipu_hh_dicts,
            return_diagnostics=True,
        )
        pd.DataFrame(list(rssz_per_table.items()), columns=["Table", "RSSZ"]).to_csv(
            os.path.join(_OUTDIR, f"zone_{zone}_rssz_values_cat.csv"), index=False
        )
        pd.DataFrame(rssz_diagnostics).sort_values("rssz", ascending=False).to_csv(
            os.path.join(_OUTDIR, f"zone_{zone}_rssz_diagnostics.csv"), index=False
        )
        pd.DataFrame({"zone": [zone], "rssz_value": [rssz_value], "rssz_total": [rssz_total]}).to_csv(
            os.path.join(_OUTDIR, f"zone_{zone}_rssz_value.csv"), index=False
        )

    ''' SA'''
    if method == 'sa':
        # Run simulated annealing for the zone

        log_file = os.path.join(_OUTDIR, f"zone_{zone}_log.txt")
        result = simulated_annealing_multi_control_tables(
            sub_population,
            individual_control_tables,
            household_control_tables,
            zone,
            max_iterations,
            initial_temp,
            cooling_rate,
            log_file
        )

        if result is None:
            print(f"Skipping Zone {zone}: No valid solution found.")
            exit()

        # Save the synthetic population for the zone

        result.to_csv(os.path.join(_OUTDIR, f"zone_{zone}_synthetic_population.csv"), index=False)
        print(f"Synthetic population for Zone {zone} saved to 'zones_sa/zone_{zone}_synthetic_population.csv'.")

        # Validate the synthetic population and save the validation report.
        # Some upstream data variants can omit household identifiers in edge cases.
        if result.empty:
            print(f"Zone {zone} produced an empty synthetic population; writing empty validation report.")
            validation_report = pd.DataFrame(columns=[
                "Category", "Key", "Control Count", "Synthetic Count", "Difference", "Percent Difference (%)"
            ])
        else:
            cols_lower = {c.lower(): c for c in result.columns}
            if "abshid" not in cols_lower and "householdid" in cols_lower:
                result = result.rename(columns={cols_lower["householdid"]: "abshid"})
            validation_report = validate_synthetic_population(
                synthetic_population=result,
                individual_control_dicts=build_control_dicts(individual_control_tables, zone),
                household_control_dicts=build_control_dicts(household_control_tables, zone),
            )
        validation_report.to_csv(os.path.join(_OUTDIR, f"zone_{zone}_validation_report.csv"), index=False)
        print(f"Validation report for Zone {zone} saved to 'zones_sa/zone_{zone}_validation_report.csv'.")
        print(f'saving rscc value')

        # Calculate and save the RSSZ value
        print(f"Calculating RSSZ value for Zone {zone}...")

        rssz_value = calculate_total_rssz(
            solution=result.to_dict("records"),
            individual_control_dicts=build_control_dicts(individual_control_tables, zone),
            household_control_dicts=build_control_dicts(household_control_tables, zone),
        )
        print('trying the tables')
        rssz_total, rssz_per_table, rssz_diagnostics = calculate_total_rssz_table(
            solution=result.to_dict("records"),
            individual_control_dicts=build_control_dicts(individual_control_tables, zone),
            household_control_dicts=build_control_dicts(household_control_tables, zone),
            return_diagnostics=True,
        )
        print("Total RSSZ:", rssz_total)
        print("RSSZ per table:", rssz_per_table)
        dfdd = pd.DataFrame(list(rssz_per_table.items()), columns=['Table', 'RSSZ'])

        # Save to CSV
        dfdd.to_csv(os.path.join(_OUTDIR, f"zone_{zone}_rssz_values_cat.csv"), index=False)
        pd.DataFrame(rssz_diagnostics).sort_values("rssz", ascending=False).to_csv(
            os.path.join(_OUTDIR, f"zone_{zone}_rssz_diagnostics.csv"), index=False
        )
        # Prepare mosaic data


        try:
            # Directory to save the plots
            save_directory = os.path.join(_OUTDIR, f"zone_{zone}_plots_predicted_vs_actual")
            # Convert result to DataFrame if needed
            solution_df = pd.DataFrame(result.to_dict("records"))


            # Plot for individual-level categories
            individual_save_dir = os.path.join(save_directory, "individual")
            plot_all_categories(control_dicts=build_control_dicts(individual_control_tables, zone), solution = solution_df, save_dir = individual_save_dir)

            # Plot for household-level categories
            household_save_dir = os.path.join(save_directory, "household")
            plot_all_categories(control_dicts=build_control_dicts(household_control_tables, zone), solution = solution_df, save_dir = household_save_dir)
        except Exception as e:
            print(e)
            print('coulld not pass')


        try:
            mosaic_data = prepare_mosaic_data(solution=result.to_dict("records"),
                                                                individual_control_dicts=build_control_dicts(individual_control_tables, zone),
                                                                household_control_dicts=build_control_dicts(household_control_tables, zone))



            # Plot the mosaic
            plot_mosaic(mosaic_data, title_prefix="Control vs Solution Counts (Household and Individual)",
                        save_path=os.path.join(_OUTDIR, f"zone_{zone}_mosaic_plot.png"))
        except:
            print('could not pass')

        # Print RSSZ results
        print("Total RSSZ:", rssz_total)




        # Save the RSSZ value to a CSV file
        rssz_data = {"zone": [zone], "rssz_value": [rssz_value]}
        rssz_df = pd.DataFrame(rssz_data)
        rssz_df.to_csv(os.path.join(_OUTDIR, f"zone_{zone}_rssz_value.csv"), index=False)

        print(f"RSSZ value for Zone {zone} saved to 'zones_{method}_{dictionary_opts}/zone_{zone}_rssz_value.csv'.")

        '''
        for zone in zones:
            a = csf_2016()
            filtered_a = a[a["sa2"] == int(zone)].reset_index(drop = True)
            if len(filtered_a) == 0:
                print('skipping', zone)
                continue
    
            filtered_b = filtered_a.reset_index(drop=True).copy()
            # If you want only the 'regucp' values for those rows
            regucp_values = filtered_a.iloc[0]['regucp']
            reg = filtered_a.iloc[0]['regucp']
            sub_pop = synthetic_population[synthetic_population['regucp'] == reg]
            print(f"\nProcessing Zone: {zone}...")
            result = simulated_annealing_multi_control_tables(
                sub_pop,
                individual_control_tables,
                household_control_tables,
                zone,
                max_iterations,
                initial_temp,
                cooling_rate,
            )
            if result is None:
                continue
            #how to create the folder zones_sa
            print(f"\nFinal Population for Zone {zone}:")
            result.to_csv(f'zones_sa/{zone}_r.csv')
            # Validate the synthetic population against the control tables
            print(f"\nValidating Population for Zone {zone}...")
            validation_report = validate_synthetic_population(
                synthetic_population=result,
                individual_control_dicts=build_control_dicts(individual_control_tables, zone),
                household_control_dicts=build_control_dicts(household_control_tables, zone),
            )
    
            # Save the validation report to a CSV file
            validation_report.to_csv(f'zones_sa/{zone}_validation_report.csv', index=False)
            print(f"Validation report for Zone {zone} saved to 'zones_sa/{zone}_validation_report.csv'")
    
        '''
