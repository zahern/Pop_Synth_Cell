"""Pooled TVD for the raw-microdata SA and IPU baselines on the fitted cohort."""
import os
import sys

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
SYNTH = r"Z:\Paper_Plan\synth_pop"
sys.path.insert(0, SYNTH)

import make_marginal_histograms as mmh  # noqa: E402

mmh.ARMS.clear()
mmh.ARMS.update({
    "sa_only": [os.path.join(RUN, "raw_baselines", "sa_only")],
    "ipu_only": [os.path.join(RUN, "raw_baselines", "ipu_only")],
})
mmh.ARM_LABELS = {"sa_only": "Base SA", "ipu_only": "Base IPU"}
mmh.FIG_DIR = os.path.join(RUN, "paper_figures", "raw_pooled")
os.makedirs(mmh.FIG_DIR, exist_ok=True)

sys.argv = [
    "make_marginal_histograms",
    "--arms", "sa_only", "ipu_only",
    "--control-dir", os.path.join(SYNTH, "data_sa2_controls_v8"),
]
mmh.main()
