"""
Plot train/validate loss curves from a train.py run's log file.

Usage:
    ./.venv/bin/python scripts/diagnostics/prediction/plot_training_curves.py [path/to/output_log.txt] [out.png] [checkpoint_epoch]

Defaults to job 3088924's log (the first real brain-mets training run) and
writes to results/diagnostics/training/training_curves_3088924.png.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.diagnostics.figpaths import figpath  # noqa: E402
from scripts.diagnostics.figstyle import (C_MUTED, C_PRED, C_REAL,  # noqa: E402
                                          TEXT_WIDTH_IN, label, report_style)

report_style()
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUT = figpath("training", "training_curves_3088924.png")
# Optional third argument: the epoch of the checkpoint frozen for later stages,
# on the log's own epoch axis (job 4194822: log epoch 70 = weights_epoch_71.pt).
# The minimum is flat over roughly epochs 49-70, so it need not equal the
# lowest-validation epoch and is drawn as its own line.
CKPT_EPOCH = None

# The keys are looked up by name rather than by position: newer runs log
# fit/baseline_loss and fit/pred_loss_vs_baseline as well, and in a different
# order, which a fixed sequence of fields silently fails to match.
EPOCH_RE = re.compile(r"^\[Epoch (\d+)/\d+\] done in [\d.]+s \|")
KEYS = {"fit_pred": "fit/pred_loss", "fit_sigreg": "fit/sigreg_loss", "fit_loss": "fit/loss",
        "val_pred": "validate/pred_loss_epoch", "val_sigreg": "validate/sigreg_loss_epoch",
        "val_loss": "validate/loss_epoch"}


def parse_log(path):
    epochs, fit_pred, fit_sigreg, fit_loss = [], [], [], []
    val_pred, val_sigreg, val_loss = [], [], []
    series = {"fit_pred": fit_pred, "fit_sigreg": fit_sigreg, "fit_loss": fit_loss,
              "val_pred": val_pred, "val_sigreg": val_sigreg, "val_loss": val_loss}
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = EPOCH_RE.match(line)
            if not m:
                continue
            values = {}
            for name, key in KEYS.items():
                hit = re.search(re.escape(key) + r": ([\d.]+)", line)
                if hit is None:
                    raise ValueError(f"{key} missing from an epoch line in {path}")
                values[name] = float(hit.group(1))
            epochs.append(int(m.group(1)))
            for name, target in series.items():
                target.append(values[name])
    return {
        "epochs": epochs, "fit_pred": fit_pred, "fit_sigreg": fit_sigreg, "fit_loss": fit_loss,
        "val_pred": val_pred, "val_sigreg": val_sigreg, "val_loss": val_loss,
    }


def main():
    global CKPT_EPOCH
    if len(sys.argv) < 2:
        sys.exit("usage: plot_training_curves.py <training log> [out.png] [checkpoint_epoch]")
    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    CKPT_EPOCH = int(sys.argv[3]) if len(sys.argv) > 3 else None

    d = parse_log(log_path)
    n = len(d["epochs"])
    print(f"Parsed {n} epochs from {log_path}")
    assert n > 0, "No epoch lines matched -- check the log format/path"

    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 4.4), sharex=True,
                             constrained_layout=True)

    # Where validation bottoms out is the figure's whole point: past it,
    # training error keeps falling and validation error rises. Marked as the
    # single lowest epoch, which is what the curve shows. It is NOT the same
    # claim as "the checkpoint used": the minimum here is flat across roughly
    # epochs 49-70, and epoch 70 was chosen by a separate per-checkpoint sweep.
    # Do not relabel this line as that checkpoint.
    best = d["epochs"][int(min(range(n), key=lambda i: d["val_pred"][i]))]

    # Names follow the report's Section 3.4.5: L_pred and L_SIGReg.
    for ax, fit_key, val_key, ylab, letter, name in (
            (axes[0], "fit_pred", "val_pred",
             r"$L_\mathrm{pred}$ (mean squared error)", "a", "Prediction loss"),
            (axes[1], "fit_sigreg", "val_sigreg",
             r"$L_\mathrm{SIGReg}$", "b", "SIGReg loss")):
        ax.plot(d["epochs"], d[fit_key], color=C_REAL, linewidth=1.2,
                label=label("fit/pred_loss"))
        ax.plot(d["epochs"], d[val_key], color=C_PRED, linewidth=1.2,
                label=label("validate/pred_loss"))
        # Both vertical lines are named in panel a's legend; text annotations
        # next to them collided with the y-axis and with each other.
        ax.axvline(best, color=C_MUTED, linewidth=1.0, linestyle=(0, (3, 2)),
                   zorder=1, label=f"lowest validation loss (epoch {best})"
                   if letter == "a" else None)
        if CKPT_EPOCH is not None:
            ax.axvline(CKPT_EPOCH, color=C_MUTED, linewidth=1.0,
                       linestyle=(0, (1, 1.5)), zorder=1,
                       label=f"checkpoint used (epoch {CKPT_EPOCH})"
                       if letter == "a" else None)
        ax.set_ylabel(ylab)
        ax.set_title(f"{letter}   {name}", fontweight="bold", fontsize=8.5,
                     loc="left")
        ax.legend(labelcolor=C_MUTED)

    axes[1].set_xlabel("epoch")

    fig.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
