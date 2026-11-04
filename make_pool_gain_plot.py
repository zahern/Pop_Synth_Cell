"""Per-table marginal TVD for the two fixed-fitter pool contrasts.

Both panels in each figure compare one fitter on the raw microdata with the same
fitter on the constructed and expanded pool. qalfp is separated because its
residual is dominated by the supplementary-code recode applied on the CP route,
not by the pool; the remaining tables are the pool contrast proper.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "paper_figures")
CSV = os.path.join(FIG_DIR, "marginal_tvd_by_table.csv")

df = pd.read_csv(CSV)
piv = df.pivot_table(index="table", columns="arm", values="tvd")

RAW, POOL = "#c0392b", "#2e8b57"

PAIRS = [
    ("ipu_only", "cp_ipu", "IPU-Only", "CP-pool\n+ IPU",
     "IPU-Only", "CP-pool + IPU", "fig_pool_gain_ipu.png"),
    ("sa_only", "cp_sa", "SA-Only", "CP-pool\n+ SA",
     "SA-Only", "CP-pool + SA", "fig_pool_gain_sa.png"),
]

for raw_arm, pool_arm, raw_tick, pool_tick, raw_legend, pool_legend, out_name in PAIRS:
    pair = piv[[raw_arm, pool_arm]].dropna()
    pair["delta"] = pair[raw_arm] - pair[pool_arm]
    qalfp = pair.loc[["qalfp"]]
    rest = pair.drop(index="qalfp").sort_values("delta")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 2.1]})

    # Panel A: qalfp
    axA.bar([0], [qalfp[raw_arm]["qalfp"]], width=0.55, color=RAW)
    axA.bar([1], [qalfp[pool_arm]["qalfp"]], width=0.55, color=POOL)
    axA.set_xticks([0, 1])
    axA.set_xticklabels([raw_tick, pool_tick], fontsize=9)
    axA.set_ylabel("TVD (qalfp)")
    axA.set_ylim(0, 0.55)
    axA.annotate(f"{qalfp[raw_arm]['qalfp']:.3f}",
                 (0, qalfp[raw_arm]["qalfp"]), ha="center", va="bottom",
                 fontsize=9)
    axA.annotate(f"{qalfp[pool_arm]['qalfp']:.3f}",
                 (1, qalfp[pool_arm]["qalfp"]), ha="center", va="bottom",
                 fontsize=9)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # Panel B: remaining 18 tables
    y = range(len(rest))
    for i, (_tbl, row) in enumerate(rest.iterrows()):
        axB.plot([row[raw_arm], row[pool_arm]], [i, i],
                 color=POOL if row["delta"] > 0 else RAW, linewidth=1.8,
                 zorder=1)
    axB.scatter(rest[raw_arm], y, s=34, color=RAW, zorder=2,
                label=raw_legend)
    axB.scatter(rest[pool_arm], y, s=34, color=POOL, zorder=2,
                label=pool_legend)
    axB.set_yticks(list(y))
    axB.set_yticklabels(rest.index, fontsize=9)
    axB.set_xlabel("TVD against the census marginal")
    axB.legend(frameon=False, fontsize=8.5, loc="lower right")
    axB.grid(axis="x", linestyle=":", alpha=0.45)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, out_name)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    n_imp = int((rest["delta"] > 0).sum())
    print(f"{raw_arm} -> {pool_arm}")
    print(f"  qalfp        : {qalfp[raw_arm]['qalfp']:.4f} -> "
          f"{qalfp[pool_arm]['qalfp']:.4f}")
    print(f"  other 18 mean: {rest[raw_arm].mean():.4f} -> "
          f"{rest[pool_arm].mean():.4f} "
          f"(gain {rest['delta'].mean():+.4f})")
    print(f"  improved on rest: {n_imp}/{len(rest)}")
    print("  wrote", out)
