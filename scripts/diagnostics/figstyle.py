"""One place that decides how a REPORT figure looks.

Companion to `figpaths.py`, which decides where a figure is written. The same
rule applies: a figure's appearance is defined here, not re-invented as loose
rcParams and hex strings in each script.

Two things this fixes, both visible in the pre-2026-09-10 figures:

* **Baked-in titles.** Every diagnostic wrote a sentence across the top with
  `suptitle`. In the report that sentence is the caption, set by the document in
  the document's font, so a title inside the image is a second caption competing
  with it. Report figures carry no title. `report_style()` does not enforce this
  -- only removing the calls does -- but every script that imports this module
  should have none left.
* **Inconsistent scale.** Scripts used dpi 110 to 180 and whatever figure width
  was convenient, so identical 9-pt labels came out at different physical sizes
  once each image was scaled into the page. Build every figure at
  `TEXT_WIDTH_IN` and the point sizes below land as written.

Usage:

    from scripts.diagnostics.figstyle import (TEXT_WIDTH_IN, C_PRED, label,
                                              report_style)
    report_style()
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 5.0))
"""

# A4 is 210 mm wide; the guideline sets a 30 mm left and 20 mm right margin, so
# the text column is 160 mm. A figure built at this width needs no scaling in
# the document, which is what keeps its text the same size as every other
# figure's.
TEXT_WIDTH_IN = 160 / 25.4          # 6.30 in
HALF_WIDTH_IN = TEXT_WIDTH_IN / 2

# Validated with the dataviz palette checker (light surface, categorical):
# lightness band, chroma floor, CVD separation, normal-vision floor and contrast
# all PASS. Do not substitute by eye. Moved here from fixed_anchor_eval.py,
# which was the only figure using them.
C_REAL, C_PRED, C_NULL = "#2f6fb5", "#c2660a", "#7a5195"
C_INK, C_MUTED, C_GRID = "#1f2328", "#5a6572", "#dde1e7"

# Internal tags are what the code calls things; these are what a reader is told.
# A figure axis must never show a checkpoint name, a dict key or a Python
# expression -- `decode(predict(t0))` and `fit/pred_loss` both shipped that way.
LABELS = {
    "lewm_e70":            "JEPA encoder",
    "vit_random":          "untrained encoder",
    "volwin_e71":          "JEPA encoder",
    "vit_random_volwin":   "untrained encoder",
    "cls":                 "class token",
    "patchmean":           "patch mean",
    "emb":                 "projector output",
    "lewm_e70_cls":        "JEPA encoder, class token",
    "lewm_e70_patchmean":  "JEPA encoder, patch mean",
    "lewm_e70_emb":        "JEPA encoder, projected",
    "vit_random_cls":      "untrained encoder, class token",
    "vit_random_patchmean": "untrained encoder, patch mean",
    "vit_random_emb":      "untrained encoder, projected",
    "volwin_e71_cls":        "JEPA encoder, class token",
    "volwin_e71_patchmean":  "JEPA encoder, patch mean",
    "volwin_e71_emb":        "JEPA encoder, projected",
    "vit_random_volwin_cls": "untrained encoder, class token",
    "vit_random_volwin_patchmean": "untrained encoder, patch mean",
    "vit_random_volwin_emb": "untrained encoder, projected",
    "fit/pred_loss":       "training",
    "validate/pred_loss":  "validation",
    "fit/sigreg_loss":     "training",
    "validate/sigreg_loss": "validation",
    "current: per-slice min-max":    "per-slice\nmin-max",
    "fixed A: per-volume 2-98 pct":  "per-volume\n2nd-98th percentile",
    "fixed B: per-volume median/IQR": "per-volume\nmedian and IQR",
    "real t0\n(input)":    "Earlier Scan",
    "real t1\n(ground truth)": "Later Scan",
    "decode(real t1)\nCEILING": "Decoder Upper Bound\n(Later Scan Decoded)",
    "decode(real t0)\nCOPY-t0 BASELINE": "No-Change Baseline\n(Earlier Scan Decoded)",
    "decode(predict(t0))\nFULL PIPELINE": "Decoded Prediction",
    "real t1 - real t0\nTRUE CHANGE": "Real Difference\n(Later minus Earlier)",
    "decode(pred) - decode(t0)\nPREDICTED CHANGE": "Predicted Difference\n(Prediction minus Baseline)",
    "decode(real t1) - decode(t0)\nCHANGE CEILING": "Upper-Bound Difference\n(Upper Bound minus Baseline)",
}


def label(tag):
    """Report label for an internal tag; unknown tags pass through unchanged.

    Deliberately not an error: a script may legitimately print a patient id or a
    horizon, which have no entry and need none.
    """
    return LABELS.get(tag, tag)


def report_style():
    """Apply the report's typography to matplotlib. Call before creating a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        # The report body is sans-serif per the guideline. DejaVu Sans is
        # matplotlib's bundled default, so this renders identically anywhere
        # rather than depending on a font being installed on the cluster.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": C_GRID,
        "axes.labelcolor": C_MUTED,
        "text.color": C_INK,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
