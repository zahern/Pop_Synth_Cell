"""Figure 4: census versus every non-generative arm for the heap table.

One grouped plot, five series per category: the census target plus SA-Only,
IPU-Only, CP-pool+SA and CP-pool+IPU.  The generative arms are excluded
by design: the paper treats generative augmentation only in the literature
review, so no result figure carries it.

The Not applicable category (code 11) is excluded from the plot because it is
the single largest category in the census marginal and would compress the other
bars; it remains in the per-category table.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_fig_all_variables import plot_table, FIG_DIR

TABLE = "heap"
LABELS = ["Postgrad", "Grad dip/cert", "Bachelor", "Adv diploma",
          "Cert III & IV", "Yr 10+", "Cert I & II", "Yr 9 or below",
          "Suppl. codes", "Not stated"]
CODE_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

df = pd.read_csv(os.path.join(FIG_DIR, "marginal_category_shares.csv"))
out = os.path.join(FIG_DIR, "fig4_marginals_by_arm.png")
plot_table(TABLE, df, out, code_order=CODE_ORDER, labels=LABELS,
           title="Highest educational attainment (heap)",
           width=9.6, font_scale=1.25)
print("wrote", out)
