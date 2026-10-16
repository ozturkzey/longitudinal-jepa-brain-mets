"""Can a frozen embedding say where in the head its slice came from?

The upper-bound question. The encoder is given one slice at a time with no
positional input, so any recoverable position is anatomy the representation
carries. Slice position is far coarser than a lesion: an embedding that cannot
locate a slice along the head axis is not carrying lesion-scale detail either.

THREE READINGS, because within-patient registration means slice 12 of one
patient is not slice 12 of another (see extract_slice_embeddings.py):

  position MAE   regression on index / (n_slices - 1), reported in normalized
                 units and converted back to millimetres with each volume's own
                 slice thickness. The only cross-patient reading, and the one
                 the ragged stacks make hardest.
  rank corr      Spearman between predicted and true position INSIDE one
                 patient, then a median over patients. Compares nothing across
                 patients, so no cohort effect can produce it.
  ordering       given two slices of one patient, how often the higher one is
                 predicted higher. Chance is 50%.

FOUR CONTROLS, and none of them is optional. Every one is printed beside every
result:

  label-only     predict the training mean position. What doing nothing scores.
  pixel-only     ridge on three numbers computed from the raw pixels with no
                 learning: slice mean, standard deviation and non-zero fraction.
                 Brain cross-section shrinks towards the vertex and the skull
                 base, so this floor is not low. The encoder has to beat it.
  untrained      the same architecture at a fixed random seed, through the
                 identical path. Run it as its own --tag and compare.
  shuffled       the same predictions scored against positions permuted WITHIN
                 each patient. Must land at rank corr 0 and ordering 50%. If it
                 does not, the split or the scoring leaks.

Splits are by patient, never by slice: adjacent slices of one patient are nearly
the same image, so a slice-level split would be scoring memorization.

Run under .venv, after extract_slice_embeddings.py:
    ./.venv/bin/python scripts/diagnostics/probes/slice_position_probe.py \
        --tag lewm_e70 --tag vit_random
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from brain_mets.train_linear_probe import three_way_split  # noqa: E402
from scripts.diagnostics.figpaths import figpath  # noqa: E402
from scripts.diagnostics.figstyle import (C_INK, C_MUTED, C_PRED, C_REAL,  # noqa: E402
                                          TEXT_WIDTH_IN, label, report_style)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO_ROOT / "results" / "probe_slice_position" / "embeddings"
READOUTS = ("cls", "patchmean", "emb")

# Searched on the validation patients, never on test. Wide and coarse on purpose:
# the point is not to squeeze the last percent out of a readout, it is to keep a
# badly-scaled one from losing for the wrong reason.
ALPHAS = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)
BOOT_SEED = 0
N_BOOT = 2000


def ridge_fit(x, y, alpha):
    """Closed-form ridge with an unpenalised intercept.

    d is 192 or 768, so the normal equations are a small solve and nothing here
    needs an optimizer or a fitting loop that could differ between runs.
    """
    xm, ym = x.mean(0), y.mean()
    xc, yc = x - xm, y - ym
    d = xc.shape[1]
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(d), xc.T @ yc)
    return w, float(ym - xm @ w)


def predict(x, model):
    w, b = model
    return x @ w + b


def per_patient(pred, true_pos, patients):
    """(median rank correlation, ordering accuracy, n usable patients).

    A patient with fewer than three distinct positions is skipped rather than
    scored: two points give a rank correlation of exactly +/-1 by construction,
    which would inflate the median with no evidence behind it.
    """
    rhos, accs = [], []
    for pid in np.unique(patients):
        m = patients == pid
        t, p = true_pos[m], pred[m]
        if len(np.unique(t)) < 3:
            continue
        # A constant prediction has no ordering to score. That is the
        # predict-the-mean baseline, which is a legitimate row: it must read as
        # undefined, not as 0% correct, which would look like it gets every pair
        # backwards.
        if len(np.unique(p)) < 2:
            continue
        rho = spearmanr(t, p).statistic
        if not np.isnan(rho):
            rhos.append(rho)
        # Every ordered pair of distinct slices in this patient.
        dt = t[:, None] - t[None, :]
        dp = p[:, None] - p[None, :]
        keep = dt != 0
        accs.append(float((np.sign(dt[keep]) == np.sign(dp[keep])).mean()))
    if not accs:
        return float("nan"), float("nan"), 0
    return float(np.median(rhos)), float(np.mean(accs)), len(accs)


def score(pred, true_pos, patients, mm_span):
    mae = float(np.abs(pred - true_pos).mean())
    mae_mm = float((np.abs(pred - true_pos) * mm_span).mean())
    rho, acc, n_pat = per_patient(pred, true_pos, patients)
    return {"mae": mae, "mae_mm": mae_mm, "rho": rho, "acc": acc, "n_patients": n_pat}


def boot_ci(pred, true_pos, patients, mm_span, key):
    """Percentile CI resampling PATIENTS, not slices.

    Slices within a patient are not independent -- they are neighbouring images
    of one head -- so resampling them would report an interval several times too
    narrow.
    """
    ids = np.unique(patients)
    rng = np.random.RandomState(BOOT_SEED)
    index_of = {pid: np.flatnonzero(patients == pid) for pid in ids}
    values = []
    for _ in range(N_BOOT):
        take = np.concatenate([index_of[pid] for pid in rng.choice(ids, len(ids))])
        # Patients drawn twice would collide in per-patient grouping, so score
        # the resample as one pooled population for the pooled statistics and
        # skip the per-patient ones, which get their own spread from the median.
        if key in ("mae", "mae_mm"):
            err = np.abs(pred[take] - true_pos[take])
            values.append(float((err * mm_span[take]).mean() if key == "mae_mm"
                                else err.mean()))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (float("nan"),) * 2


def run_one(name, features, data, splits, label_width):
    fit_m, val_m, test_m = splits
    pos, patients, mm_span = data["position"], data["patient_id"], data["mm_span"]

    mu, sd = features[fit_m].mean(0), features[fit_m].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)          # a constant column carries nothing
    z = (features - mu) / sd

    best = min(ALPHAS, key=lambda a: np.abs(
        predict(z[val_m], ridge_fit(z[fit_m], pos[fit_m], a)) - pos[val_m]).mean())
    # Refit on fit+val once alpha is chosen, which is what the age/sex probe does
    # and keeps the two comparable.
    train_m = fit_m | val_m
    model = ridge_fit(z[train_m], pos[train_m], best)
    pred = predict(z[test_m], model)

    out = score(pred, pos[test_m], patients[test_m], mm_span[test_m])
    lo, hi = boot_ci(pred, pos[test_m], patients[test_m], mm_span[test_m], "mae_mm")

    # Same predictions, positions permuted within each patient. Anything above
    # chance here is leakage, not signal.
    rng = np.random.RandomState(BOOT_SEED)
    shuffled = pos[test_m].copy()
    for pid in np.unique(patients[test_m]):
        m = patients[test_m] == pid
        shuffled[m] = rng.permutation(shuffled[m])
    null = score(pred, shuffled, patients[test_m], mm_span[test_m])

    def fmt(value, spec):
        return f"{'--':>{spec.split('.')[0].lstrip('>')}}" if np.isnan(value) else f"{value:{spec}}"

    print(f"  {name:<{label_width}}{out['mae']:>8.3f}{out['mae_mm']:>9.1f}"
          f"  [{lo:.1f}, {hi:.1f}]{fmt(out['rho'], '>9.3f')}{fmt(out['acc'], '>9.1%')}"
          f"{fmt(null['rho'], '>10.3f')}{fmt(null['acc'], '>9.1%')}{best:>9g}")
    out["pred"], out["true"] = pred, pos[test_m]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", action="append", required=True,
                    help="encoder tag from extract_slice_embeddings.py; repeatable")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--fig", default=str(figpath("probes", "slice_position.png")),
                    help="output figure path; pass an empty string to skip it")
    args = ap.parse_args()

    first = np.load(Path(args.dir) / f"{args.tag[0]}_{READOUTS[0]}.npz", allow_pickle=True)
    patients = first["patient_id"]
    pos = first["position"].astype(np.float64)
    # A full-stack normalized step is (n_slices - 1) slices of mm_step each, so
    # an error of 1.0 in normalized units is this many millimetres.
    mm_span = (first["n_slices"].astype(np.float64) - 1) * first["mm_step"].astype(np.float64)
    pixel_features = np.stack([first["mean"], first["std"], first["frac_nonzero"]], axis=1)
    data = {"position": pos, "patient_id": patients, "mm_span": mm_span}

    fit_m, val_m, test_m = three_way_split(patients)
    n_test_pat = len(set(patients[test_m]))
    print(f"{len(pos):,} slices from {len(set(patients))} patients | "
          f"test: {test_m.sum():,} slices from {n_test_pat} patients")
    print(f"stacks {first['n_slices'].min()}-{first['n_slices'].max()} slices, "
          f"{first['mm_step'].min():.1f}-{first['mm_step'].max():.1f} mm/slice")

    rows = [("predict the mean position", None, None, np.zeros((len(pos), 1)) + 1e-9),
            ("pixel statistics only", None, None, pixel_features)]
    for tag in args.tag:
        for readout in READOUTS:
            path = Path(args.dir) / f"{tag}_{readout}.npz"
            if not path.exists():
                print(f"  (skipping {path.name}, not extracted)")
                continue
            emb = np.load(path, allow_pickle=True)
            if not np.array_equal(emb["patient_id"], patients):
                raise ValueError(f"{path.name} has a different row order than "
                                 f"{args.tag[0]}_{READOUTS[0]}.npz -- re-extract")
            rows.append((f"{tag}  {readout}", tag, readout,
                         emb["embeddings"].astype(np.float64)))

    width = max(len(n) for n, _, _, _ in rows) + 2
    print(f"\n  {'':<{width}}{'MAE':>8}{'MAE mm':>9}{'  95% CI':>16}"
          f"{'rank rho':>9}{'ordering':>9}{'null rho':>10}{'null':>9}{'alpha':>9}")
    collected = {"tags": args.tag, "rows": {}, "pixel_acc": float("nan")}
    for name, tag, readout, features in rows:
        out = run_one(name, features, data, (fit_m, val_m, test_m), width)
        if tag is not None:
            collected["rows"][(tag, readout)] = out
        elif name.startswith("pixel"):
            collected["pixel_acc"] = out["acc"]

    if args.fig:
        make_figure(collected, args.fig)


def make_figure(results, out_path):
    """Two panels: what the trained encoder recovers and what an untrained one
    recovers.

    A held-out slice's true position runs along x and the ridge readout's
    estimate along y, so a representation that carries position collapses onto
    the diagonal and one that does not spreads into a cloud. Same axes and same
    readout in both, so the only difference between the panels is whether the
    encoder was trained. The within-patient ordering bar chart that used to be
    panel c was dropped for the report (2026-10-08); its numbers are still
    printed by main().
    """
    import matplotlib.pyplot as plt

    report_style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN * 0.8, 2.4),
                             constrained_layout=True, sharey=True)
    # Extra space between the two panels: their x-axis labels and the 1.0 / 0.0
    # tick labels otherwise run into each other.
    fig.get_layout_engine().set(wspace=0.12)

    def panel(ax, letter, text):
        ax.set_title(f"{letter}   {text}", color=C_INK, fontsize=8.5,
                     fontweight="bold", loc="left")

    # The projector output ("emb"): the vector the predictor and the decoder
    # receive, and the row the report's Table 4.1 quotes.
    scatter_readout = "emb"
    tags = [t for t in results["tags"] if (t, scatter_readout) in results["rows"]]
    for ax, tag, letter in zip(axes, tags, "ab"):
        row = results["rows"][(tag, scatter_readout)]
        ax.plot([0, 1], [0, 1], color=C_INK, linewidth=1.0, zorder=3)
        ax.scatter(row["true"], row["pred"], s=1.5, alpha=0.06, color=C_PRED,
                   linewidths=0, zorder=2)
        ax.set_xlim(0, 1); ax.set_ylim(-0.15, 1.15)
        ax.set_xlabel("true position (0 = bottom, 1 = top)", fontsize=7)
        if letter == "a":
            ax.set_ylabel("estimated position")
        panel(ax, letter, label(tag))

    fig.savefig(out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
