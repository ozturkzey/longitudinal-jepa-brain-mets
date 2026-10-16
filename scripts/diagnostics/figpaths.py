"""
One place that decides where a diagnostic figure or table is written.

`results/diagnostics/` was flat until 2026-08-28 and had grown to 40 files
from five unrelated investigations, so a figure's name was the only clue to
what produced it. Output now goes in a topic subfolder, and this module is
the only definition of what those topics are -- adding a figure means picking
an existing topic, not inventing a path string in one more script.

Unknown topics raise rather than being created on the fly: a typo would
otherwise silently make a sixth folder and the file would be lost to whoever
looks for it later.

    from scripts.diagnostics.figpaths import figpath
    OUT_PNG = figpath("training", "training_curves_3096280.png")
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS = REPO_ROOT / "results" / "diagnostics"

TOPICS = {
    "training":      "encoder/predictor training runs and their per-job diagnostics",
    "decoder":       "decoder reconstructions and capacity tests",
    "prediction":    "end-to-end encoder -> predictor -> decoder output",
    "registration":  "inter-visit registration QC and transform comparison",
    "normalization": "slice intensity normalization",
    "probes":        "what a readout can recover from a frozen embedding",
    "schematic":     "drawn diagrams of the cohort and the model path, not measurements",
}


def figpath(topic, filename):
    """Absolute path to `filename` under results/diagnostics/<topic>/.

    Creates the topic directory if it does not exist. Raises on an unknown
    topic -- see the module docstring.
    """
    if topic not in TOPICS:
        raise ValueError(
            f"unknown diagnostics topic {topic!r}; expected one of "
            f"{sorted(TOPICS)}. Add it to figpaths.TOPICS if it is genuinely new."
        )
    out_dir = DIAGNOSTICS / topic
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename
