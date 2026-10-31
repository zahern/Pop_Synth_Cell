#!/usr/bin/env python3
"""Compare the information carried by standard, CP, and expanded pools."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ATTRIBUTES = (
    "sexp", "agep", "mstp", "heap", "occp", "incp", "lfsp", "empp",
    "hrsp", "qallp", "qalfp", "rlhp", "mtwp", "nprd", "vehrd", "mrerd",
    "rntrd", "strd", "hind", "bedrd", "hhcd", "tend",
)


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _resolve(df: pd.DataFrame, name: str):
    lookup = {str(c).lower(): c for c in df.columns}
    return lookup.get(name.lower())


def _atomicity(df: pd.DataFrame) -> dict:
    hh = _resolve(df, "abshid")
    if hh is None:
        return {"households": 0, "atomicity_violations": None}
    grouped = df.groupby(hh, dropna=False, sort=False)
    violations = 0
    for field in ("nprd", "vehrd", "strd", "rntrd", "mrerd", "bedrd", "hhcd", "tend"):
        column = _resolve(df, field)
        if column is not None:
            violations += int((grouped[column].nunique(dropna=False) > 1).sum())
    return {
        "households": int(df[hh].nunique(dropna=False)),
        "atomicity_violations": violations,
    }


def _summary(name: str, path: str, attributes: tuple[str, ...]) -> tuple[dict, list[dict]]:
    df = _read(path)
    atomic = _atomicity(df)
    row = {
        "pool": name,
        "path": path,
        "person_rows": len(df),
        **atomic,
        "mean_household_size": (
            len(df) / atomic["households"] if atomic["households"] else None
        ),
    }
    margins = []
    for attribute in attributes:
        column = _resolve(df, attribute)
        if column is None:
            continue
        counts = df[column].value_counts(dropna=False)
        total = counts.sum()
        for category, count in counts.items():
            margins.append({
                "pool": name,
                "attribute": attribute,
                "category": str(category),
                "count": int(count),
                "share": float(count / total) if total else 0.0,
            })
    return row, margins


def _tvd(margins: pd.DataFrame, reference: str) -> pd.DataFrame:
    base = margins[margins.pool == reference]
    rows = []
    for pool in sorted(margins.pool.unique()):
        if pool == reference:
            continue
        other = margins[margins.pool == pool]
        for attribute in sorted(set(base.attribute) & set(other.attribute)):
            left = base[base.attribute == attribute].set_index("category")["share"]
            right = other[other.attribute == attribute].set_index("category")["share"]
            joined = pd.concat([left, right], axis=1).fillna(0.0)
            rows.append({
                "pool": pool,
                "attribute": attribute,
                "tvd_vs_reference": float(
                    0.5 * np.abs(joined.iloc[:, 0] - joined.iloc[:, 1]).sum()
                ),
            })
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, margins: pd.DataFrame, tvd: pd.DataFrame,
          output: Path) -> None:
    labels = summary["pool"].tolist()
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].bar(labels, summary["person_rows"], color=colors[:len(labels)])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Candidate person rows")
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].set_ylabel("rows (log scale)")

    tvd_mean = tvd.groupby("pool", as_index=False)["tvd_vs_reference"].mean()
    axes[0, 1].bar(
        tvd_mean["pool"], tvd_mean["tvd_vs_reference"],
        color=colors[1:len(tvd_mean) + 1],
    )
    axes[0, 1].set_title("Marginal TVD versus standard microdata")
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 1].set_ylabel("mean TVD")

    atomic = summary["atomicity_violations"].fillna(0)
    axes[1, 0].bar(labels, atomic, color=colors[:len(labels)])
    axes[1, 0].set_title("Household-field atomicity violations")
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].set_ylabel("violating household-field cells")

    support = margins.groupby("pool")["category"].nunique().reindex(labels)
    axes[1, 1].bar(labels, support, color=colors[:len(labels)])
    axes[1, 1].set_title("Distinct supported categories")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].set_ylabel("attribute-category cells")

    fig.suptitle(
        "Pool support before zone propagation\n"
        "Pool composition changes candidate availability; the zone fitter allocates candidates",
        fontsize=13,
    )
    fig.savefig(output / "pool_component_comparison.png", dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microdata", required=True)
    parser.add_argument("--gan-pool", default=None)
    parser.add_argument("--mip-pool", required=True)
    parser.add_argument("--reconciled-pool", default=None)
    parser.add_argument("--cp-pool", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-figure-dir", default=None)
    parser.add_argument("--attributes", nargs="*", default=DEFAULT_ATTRIBUTES)
    args = parser.parse_args(argv)

    paths = {
        "standard_microdata": args.microdata,
    }
    if args.gan_pool:
        paths["gan_pool"] = args.gan_pool
    if args.reconciled_pool:
        paths["mip_reconciled_pool"] = args.reconciled_pool
    if args.cp_pool:
        paths["cp_pool"] = args.cp_pool
    paths["mip_expanded_pool"] = args.mip_pool
    summaries = []
    margins = []
    for name, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        summary, rows = _summary(name, path, tuple(args.attributes))
        summaries.append(summary)
        margins.extend(rows)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    margin_frame = pd.DataFrame(margins)
    summary_frame.to_csv(output / "pool_component_summary.csv", index=False)
    margin_frame.to_csv(output / "pool_component_margins.csv", index=False)
    tvd = _tvd(margin_frame, "standard_microdata")
    tvd.to_csv(
        output / "pool_component_tvd_vs_microdata.csv", index=False
    )
    _plot(summary_frame, margin_frame, tvd, output)
    if args.paper_figure_dir:
        figure_dir = Path(args.paper_figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            output / "pool_component_comparison.png",
            figure_dir / "pool_component_comparison.png",
        )
    (output / "pool_component_manifest.json").write_text(
        json.dumps({
            "reference": "standard_microdata",
            "pools": paths,
            "attributes": list(args.attributes),
            "atomicity_definition": "household fields constant within ABSHID",
        }, indent=2), encoding="utf-8"
    )
    print(f"Wrote component summary and marginal comparisons to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
