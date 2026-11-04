import glob
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mip_fit as mipf
from sa_only_baseline import load_control_tables

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = sorted(glob.glob(os.path.join(HERE, "outputs", "expansion_shards",
                                        "shard_*.report.json")))

zones = {}
for path in REPORTS:
    doc = json.load(open(path))
    for rec in doc.get("zone_reports", []) or []:
        if str(rec.get("status", "ok")).lower() != "ok":
            continue
        zones[str(rec["zone"])] = rec

ind, hh = load_control_tables()

rows = []
residual_by_code = Counter()
residual_mass = 0.0
total_mass = 0.0
total_cells = 0
for zone, rec in sorted(zones.items()):
    tgt = {}
    tgt.update(mipf.zone_targets(ind, zone))
    tgt.update(mipf.zone_targets(hh, zone))
    tm = float(sum(v for v in tgt.values() if v > 0))
    total_mass += tm
    total_cells += sum(1 for v in tgt.values() if v > 0)
    rm = 0.0
    for cell in rec.get("uncovered_cells") or []:
        attr, _, cat = str(cell).partition(":")
        residual_by_code[str(cell)] += 1
        hit = tgt.get((attr, cat))
        if hit is None:
            try:
                hit = tgt.get((attr, int(cat)))
            except ValueError:
                hit = None
        rm += float(hit or 0.0)
    residual_mass += rm
    zero = int(rec.get("n_zero_cells", 0) or 0)
    cov = int(rec.get("n_zero_cells_covered", 0) or 0)
    unc = int(rec.get("n_zero_cells_uncovered", 0) or 0)
    rows.append({
        "zone": zone,
        "zero": zero,
        "covered": cov,
        "uncovered": unc,
        "coverage": (cov / zero) if zero else 1.0,
        "candidates": int(rec.get("n_candidates", 0) or 0),
        "selected_hh": int(rec.get("n_selected_households", 0) or 0),
        "selected_persons": int(rec.get("n_selected_persons", 0) or 0),
        "appended_hh": int(rec.get("n_appended_menu_households", 0) or 0),
        "menu_hh": int(rec.get("n_menu_households", 0) or 0),
        "dev_sum": float(rec.get("marginal_deviation_sum") or 0.0),
        "dev_mean": float(rec.get("marginal_deviation_mean") or 0.0),
        "dev_max": float(rec.get("marginal_deviation_max") or 0.0),
        "residual_mass": rm,
        "total_mass": tm,
        "total_cells": sum(1 for v in tgt.values() if v > 0),
        "zero_mass_share": (rm / tm) if tm else 0.0,
    })


def q(vals, p):
    vals = sorted(vals)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    frac = k - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


zeros = [r["zero"] for r in rows]
covs = [r["coverage"] for r in rows]
sel_hh = [r["selected_hh"] for r in rows]
sel_p = [r["selected_persons"] for r in rows]
app = [r["appended_hh"] for r in rows]
dev = [r["dev_sum"] for r in rows]

summary = {
    "zones": len(rows),
    "reports": len(REPORTS),
    "zero_cells": sum(zeros),
    "covered_cells": sum(r["covered"] for r in rows),
    "uncovered_cells": sum(r["uncovered"] for r in rows),
    "coverage_pct": 100.0 * sum(r["covered"] for r in rows) / sum(zeros),
    "target_cells": total_cells,
    "zero_cell_rate_pct": 100.0 * sum(zeros) / total_cells,
    "zero_min": min(zeros), "zero_q1": q(zeros, 0.25),
    "zero_median": statistics.median(zeros), "zero_q3": q(zeros, 0.75),
    "zero_max": max(zeros), "zero_mean": statistics.mean(zeros),
    "coverage_min": min(covs), "coverage_q1": q(covs, 0.25),
    "coverage_median": statistics.median(covs), "coverage_q3": q(covs, 0.75),
    "coverage_max": max(covs), "coverage_mean": statistics.mean(covs),
    "zones_full": sum(1 for r in rows if r["uncovered"] == 0),
    "zones_residual": sum(1 for r in rows if r["uncovered"] > 0),
    "zones_residual_1": sum(1 for r in rows if r["uncovered"] == 1),
    "selected_hh_total": sum(sel_hh),
    "selected_persons_total": sum(sel_p),
    "selected_hh_median": statistics.median(sel_hh),
    "selected_hh_max": max(sel_hh),
    "selected_persons_median": statistics.median(sel_p),
    "selected_persons_max": max(sel_p),
    "appended_hh_total": sum(app),
    "appended_hh_median": statistics.median(app),
    "appended_hh_max": max(app),
    "menu_hh_total": sum(r["menu_hh"] for r in rows),
    "dev_sum_median": statistics.median(dev),
    "dev_sum_max": max(dev),
    "residual_mass": residual_mass,
    "total_target_mass": total_mass,
    "residual_mass_pct": 100.0 * residual_mass / total_mass,
    "residual_by_code": dict(residual_by_code.most_common()),
    "zero_count_distribution": dict(sorted(Counter(zeros).items())),
    "uncovered_count_distribution": dict(sorted(Counter(r["uncovered"] for r in rows).items())),
}
summary["selected_per_covered_cell"] = summary["selected_hh_total"] / summary["covered_cells"]
summary["persons_per_covered_cell"] = summary["selected_persons_total"] / summary["covered_cells"]

out = os.path.join(HERE, "outputs", "expansion_zone_stats.json")
json.dump({"summary": summary, "zones": rows}, open(out, "w"), indent=2)
print(json.dumps(summary, indent=2))
print("wrote", out)
