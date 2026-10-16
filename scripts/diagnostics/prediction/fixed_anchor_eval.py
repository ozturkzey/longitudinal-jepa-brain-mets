"""Fixed-anchor evaluation: predict every later visit from the SAME real t0.

    real t0 ---------------> t1     dt =  43d
    real t0 ---------------> t2     dt = 106d
    real t0 ---------------> t3     dt = 183d

One `predict()` call per target, always from the real anchor, with the true
elapsed days as the covariate. Deliberately NOT a rollout (t0 -> e1 -> e2 -> ...):
rollout feeds the model its own output, so a bad result cannot be attributed to
the per-step model rather than to compounding, and on a model this marginal the
single-step version is the right instrument.

WHAT THIS ANSWERS. Does predicted change grow with elapsed time? The copy-anchor
baseline ("nothing changed since t0") gets weaker as the horizon grows, so the
ratio should IMPROVE with dt if the model has learned the time relationship at
all. How much signal there is to learn is itself modest: across 25,070 held-out
anchor->target pairs, elapsed time correlates with actual embedding-space change
at only r = 0.09, and median change is flat from one month (2.22) to two years
(2.55). An earlier r = 0.79 was quoted here; it came from 64 points, one of which
carried a change of 15.9 against a median of 0.81, and it falls to r = 0.02 when
that single point is removed.

EXTRAPOLATION IS REPORTED, NOT HIDDEN. Training pairs consecutive visits, whose
gaps have a median near 70 days. A 500-day anchor->t4 horizon is therefore a dt
the model never saw, and a bad score there may mean "never learned time" OR
"asked for a dt outside its training range". The summary prints each horizon
bucket's distance from the training dt distribution in standard deviations, so
the two cannot be confused. Buckets beyond ~2 sigma are flagged.

NULLS. A rising curve is not evidence on its own -- `--shuffle-dt` permutes dt
across targets (same marginal distribution, relationship destroyed) and
`--predictor-ckpt` omitted leaves the predictor at random init. If the real curve
does not separate from those, the model is not reading dt.

Usage (needs a trained predictor to be meaningful):
    ./.venv/bin/python scripts/diagnostics/prediction/fixed_anchor_eval.py \
        --predictor-ckpt <path>/weights_epoch_N.pt
"""

import argparse
import os
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from scripts.diagnostics.figpaths import figpath
from scripts.diagnostics.figstyle import (C_INK, C_MUTED, C_NULL, C_PRED,
                                          C_REAL, TEXT_WIDTH_IN, report_style)

# torch, hydra, stable_pretraining and the dataset classes are imported inside
# the functions that need them, not here. Importing them costs minutes on this
# cluster's shared filesystem, and --from-csv redraws a finished evaluation with
# nothing but numpy and matplotlib -- it should not pay for a model stack it
# never builds.

ROOT_DIR = os.environ["BRAINJEPA_REGISTERED"]
CLINICAL_EXCEL = "Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx"

# Horizon buckets in days. Chosen from the measured consecutive-gap distribution
# (median ~70d): the first two sit inside the training range, the last two are
# progressively further outside it. Boundaries are round numbers so the table
# reads clearly, not fitted to the data.
HORIZON_EDGES = [0, 60, 120, 240, 480, 10_000]

# Palette, typography and page width come from figstyle so that every report
# figure shares them; see that module for why they are not defined per script.


def training_action_stats(dataset, pairing):
    """Mean/std of the action column as the predictor's TRAINING run saw it.

    Must match training exactly or every normalized dt fed to the predictor is
    wrong. Until 2026-09-17 this always used the consecutive-pair population, so
    the first_two predictor was scored with a dt mean of ~103 d where it trained
    on ~83 d, and with a shifted age mean -- no error, just mis-scaled inputs.

    The sequence dataset's own column is NOT the right source: it carries a
    leading dt_days[0] = 0.0 per sequence (the anchor frame has no predecessor).
    - "consecutive": every t>=1 step. Both datasets bridge blank frames by
      summing the spanned gaps, so this is the pair population exactly (207,922).
    - "first_two": the pair dataset takes visits 0 and 1 and keeps the pair only
      if neither slice is blank. After blank trimming that is exactly the
      sequences whose first two kept frames ARE visits 0 and 1 (29,773).
    std uses ddof=1 because train.py's ZScoreNormalizer takes torch.std.
    """
    import os

    from brain_mets.brain_mets_sequence_dataset import MODALITY

    rows = []
    if pairing == "consecutive":
        for s in dataset.sequences:
            for t in range(1, len(s["paths"])):
                rows.append([s["dt_days"][t], s["age"]])
    elif pairing == "first_two":
        first_two = {}  # patient_id -> paths of visits 0 and 1, as the pair dataset builds them
        for s in dataset.sequences:
            pid = s["patient_id"]
            if pid not in first_two:
                patient_path = os.path.join(ROOT_DIR, pid)
                visits = sorted(
                    v for v in os.listdir(patient_path)
                    if os.path.isdir(os.path.join(patient_path, v))
                    and os.path.exists(os.path.join(patient_path, v, f"{MODALITY}.nii.gz")))
                first_two[pid] = [os.path.join(patient_path, v, f"{MODALITY}.nii.gz")
                                  for v in visits[:2]]
            if s["paths"][:2] == first_two[pid]:
                rows.append([s["dt_days"][1], s["age"], pid])
        # split_by_patient shuffles the sorted patient set, so the held-out
        # patients here equal the first_two run's only if both sets agree.
        train_patients = {r[2] for r in rows}
        if train_patients != set(dataset.patient_ids):
            raise RuntimeError(
                f"first_two training population has {len(train_patients)} patients, "
                f"the sequence dataset {len(set(dataset.patient_ids))}: the held-out "
                f"split would not match the one the predictor was trained with")
        rows = [r[:2] for r in rows]
    else:
        raise ValueError(f"unknown pairing {pairing!r}")
    a = np.asarray(rows, dtype=np.float64)
    print(f"  action stats from pairing={pairing}: {len(a):,} training transitions")
    return (a[:, 0].mean(), a[:, 0].std(ddof=1), a[:, 1].mean(), a[:, 1].std(ddof=1),
            a[:, 0])


def collect(model, dataset, val_idx, pairing, shuffle_dt=False, seed=0):
    """One record per (sequence, target visit)."""
    import torch
    from lewm_core.utils import embed_dataset

    torch.set_num_threads(16)
    payload = embed_dataset(model, dataset, val_idx)
    dt_mean, dt_std, age_mean, age_std, train_dt = training_action_stats(dataset, pairing)

    recs = []
    for k, ds_idx in enumerate(payload["index"]):
        seq = dataset.sequences[ds_idx]
        emb = payload["emb"][k]                     # (T, D)
        T = emb.shape[0]
        if T < 2:
            continue
        anchor = emb[0:1].unsqueeze(0)              # (1, 1, D)
        cum = 0.0
        for j in range(1, T):
            cum += float(seq["dt_days"][j])         # gap from the ANCHOR, in days
            recs.append({
                "patient_id": seq["patient_id"], "slice_index": seq["slice_index"],
                "target_visit": j, "n_visits": T, "dt_days": cum,
                # dt_model is what the MODEL is told; dt_days is what actually
                # elapsed. They are equal except under the shuffled null, which
                # is the whole point -- see the shuffle block below.
                "dt_model": cum,
                "_anchor": anchor, "_target": emb[j], "_age": float(seq["age"]),
            })

    if shuffle_dt:
        # Same marginal dt distribution, relationship to the target destroyed.
        #
        # ONLY dt_model is permuted. Until 2026-09-04 this overwrote dt_days --
        # the field the null curve is later BINNED BY -- so the null plotted
        # predicted-change against the very dt that produced it. That is the
        # same function the real curve plots, so the two were forced to overlap
        # whether or not the model used dt at all, and the flat null on the
        # 09-03 figures was an artifact of this line, not a result. A correct
        # null keeps the true elapsed time on the x-axis and feeds the model a
        # dt belonging to some other record.
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(recs))
        shuffled = [recs[p]["dt_model"] for p in perm]
        for r, v in zip(recs, shuffled):
            r["dt_model"] = v

    # Batch the predictor over all records sharing nothing -- each is (1,1,D).
    with torch.no_grad():
        for start in range(0, len(recs), 512):
            chunk = recs[start:start + 512]
            anchor = torch.cat([r["_anchor"] for r in chunk], dim=0)          # (B,1,D)
            act = torch.tensor(
                [[[(r["dt_model"] - dt_mean) / max(dt_std, 1e-8),
                   (r["_age"] - age_mean) / max(age_std, 1e-8)]] for r in chunk],
                dtype=torch.float32)                                          # (B,1,2)
            pred = model.predict(anchor, model.action_encoder(act))
            pred = pred[:, -1]                                                # (B,D)
            for i, r in enumerate(chunk):
                tgt, anc = r["_target"], r["_anchor"][0, 0]
                r["err_model"] = float(torch.linalg.norm(pred[i] - tgt))
                r["err_base"] = float(torch.linalg.norm(anc - tgt))
                r["d_pred"] = float(torch.linalg.norm(pred[i] - anc))
                r["d_real"] = float(torch.linalg.norm(tgt - anc))
    for r in recs:
        for key in ("_anchor", "_target", "_age"):
            r.pop(key)
    return recs, train_dt


def bootstrap_ratio(num, den, n_boot=2000, seed=0):
    """Ratio of MEANS with a percentile CI, resampling records.

    Ratio-of-means, not mean-of-ratios: a per-record num/den divides by ~0 on
    targets where the anchor already equals the target, which is what produced a
    standard error of +/-2.4 million in an earlier scorer.
    """
    num, den = np.asarray(num), np.asarray(den)
    point = num.mean() / den.mean()
    rng = np.random.RandomState(seed)
    boots = [num[s].mean() / den[s].mean()
             for s in (rng.randint(0, len(num), len(num)) for _ in range(n_boot))]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(recs, train_dt, label):
    from scipy.stats import spearmanr, wilcoxon

    dtv = np.array([r["dt_days"] for r in recs])
    mu, sd = train_dt.mean(), train_dt.std()
    print(f"\n{'=' * 78}\nFIXED-ANCHOR EVALUATION -- {label}\n{'=' * 78}")
    print(f"  {len(recs):,} anchor->target predictions | "
          f"{len({r['patient_id'] for r in recs})} held-out patients")
    print(f"  training per-step dt: mean {mu:.0f} d, sd {sd:.0f} d "
          f"(anything far above this is EXTRAPOLATION)")

    print(f"\n  {'horizon':<14}{'n':>7}{'model':>9}{'copy-t0':>9}"
          f"{'ratio':>8}{'  95% CI':>18}{'sigma':>7}")
    for lo, hi in zip(HORIZON_EDGES, HORIZON_EDGES[1:]):
        sel = [r for r in recs if lo <= r["dt_days"] < hi]
        if len(sel) < 30:
            continue
        m = np.array([r["err_model"] for r in sel])
        b = np.array([r["err_base"] for r in sel])
        pt, lo_ci, hi_ci = bootstrap_ratio(m, b)
        sigma = (np.median([r["dt_days"] for r in sel]) - mu) / max(sd, 1e-8)
        flag = "  <-- EXTRAPOLATED" if sigma > 2 else ""
        name = f"{lo}-{hi}d" if hi < 10_000 else f"{lo}d+"
        print(f"  {name:<14}{len(sel):>7,}{m.mean():>9.3f}{b.mean():>9.3f}"
              f"{pt:>8.4f}   [{lo_ci:.4f}, {hi_ci:.4f}]{sigma:>7.1f}{flag}")

    dpred = np.array([r["d_pred"] for r in recs])
    dreal = np.array([r["d_real"] for r in recs])
    print(f"\n  Does predicted change track elapsed time?")
    print(f"    corr(||d_pred||, dt) = {np.corrcoef(dpred, dtv)[0,1]:+.4f}"
          f"      <-- the number this evaluation exists for")
    print(f"    corr(||d_real||, dt) = {np.corrcoef(dreal, dtv)[0,1]:+.4f}"
          f"      (the ceiling: how much of it is there to find)")
    print(f"    ||d_pred|| / ||d_real|| = {dpred.mean()/max(dreal.mean(),1e-8):.3f}"
          f"   (MSE shrinks predictions toward the mean; <1 is expected)")

    # WITHIN-ANCHOR, and this is the test that actually settles the question.
    # The pooled correlation above mixes patients, so it can be moved by a
    # between-patient effect -- patients with long follow-up simply changing
    # more -- with no per-scan dt response at all. Within one (patient, slice)
    # the anchor embedding and the age are literally the same tensor across
    # every horizon, so dt is the ONLY input that differs and any spread in
    # ||d_pred|| is a dt response by construction. Reported as a median over
    # anchors plus the fraction pointing the right way, because a handful of
    # long timelines would otherwise dominate a pooled number.
    by_anchor = {}
    for r in recs:
        by_anchor.setdefault((r["patient_id"], r["slice_index"]), []).append(r)
    rhos, spans = [], []
    for rs in by_anchor.values():
        if len(rs) < 4:
            continue
        x = np.array([r["dt_days"] for r in rs])
        y = np.array([r["d_pred"] for r in rs])
        rho = spearmanr(x, y).statistic
        if not np.isnan(rho):
            rhos.append(rho)
            spans.append(y.max() / max(y.min(), 1e-9))
    if rhos:
        rhos, spans = np.array(rhos), np.array(spans)
        pval = wilcoxon(rhos).pvalue if len(rhos) > 10 else float("nan")
        print(f"\n  Within-anchor dt response ({len(rhos):,} anchors with >=4 horizons)")
        print(f"    median Spearman(dt, ||d_pred||) = {np.median(rhos):+.3f}"
              f"   <-- same scan, same age, only dt differs")
        print(f"    anchors with a positive slope   = {(rhos > 0).mean():.1%}"
              f"   (50% = no response)")
        print(f"    median ||d_pred|| max/min       = {np.median(spans):.2f}x"
              f"   (how far dt moves the prediction)")
        print(f"    Wilcoxon vs 0: p = {pval:.3g}")

    # Within-patient: does dt predict change INSIDE one patient's timeline? A
    # pooled correlation can be produced by "sicker patients are scanned more
    # often AND change more" with no knowledge of time at all. This cannot.
    slopes = []
    by_pat = {}
    for r in recs:
        by_pat.setdefault(r["patient_id"], []).append(r)
    for rs in by_pat.values():
        if len({r["dt_days"] for r in rs}) < 3:
            continue
        x = np.array([r["dt_days"] for r in rs])
        y = np.array([r["d_pred"] for r in rs])
        if x.std() > 0:
            slopes.append(np.polyfit(x, y, 1)[0])
    if slopes:
        s = np.array(slopes)
        print(f"    within-patient slope: median {np.median(s):+.3e} per day, "
              f"{100*(s > 0).mean():.0f}% of {len(s)} patients positive")
    return dtv, dpred, dreal


def _log_x(ax):
    """Log-scale the x axis only if something was actually plotted on it.

    An empty axis keeps matplotlib's default (-0.055, 0.055) limits, and
    set_xscale("log") on those raises "Data cannot be log-scaled because all
    values are <= 0" -- which kills the figure AFTER every number has been
    computed and printed. Panels go empty legitimately: binned() drops any
    horizon bucket with fewer than 20 samples, so a small --max-patients run,
    or a cohort with few long horizons, leaves nothing to draw.
    """
    if ax.has_data():
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "no horizon bucket had >= 20 samples",
                transform=ax.transAxes, ha="center", va="center",
                color=C_MUTED, fontsize=9)


def load_records(csv_path):
    """Rows written by an earlier run, back in the form `make_figure` expects.

    Lets the figure be rebuilt from a finished evaluation without re-running the
    model. Only the columns the figure reads are cast to float; `patient_id`
    stays a string. The 2026-09-03 CSVs predate the `dt_model` column, which no
    panel uses, so their absence is not an error.
    """
    numeric = ("dt_days", "err_model", "err_base", "d_pred", "d_real",
               "target_visit", "n_visits", "slice_index")
    recs = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rec = dict(row)
            for key in numeric:
                if key in rec:
                    rec[key] = float(rec[key])
            recs.append(rec)
    if not recs:
        raise ValueError(f"{csv_path} holds no rows")
    required = {"dt_days", "err_model", "err_base", "d_pred", "d_real", "patient_id"}
    missing = required - recs[0].keys()
    if missing:
        raise KeyError(f"{csv_path} is missing {sorted(missing)}")
    return recs


def make_figure(real_recs, null_recs, train_dt, out_path, absent=()):
    """Three panels, one claim each. No title: the report supplies a caption.

    Every axis here is drawn on a range that contains the answer, not zoomed to
    fill the panel. Panel c is the reason this matters: on a 0.98-1.00 axis the
    model's advantage over doing nothing looked like a curve with structure,
    when at full scale it is a flat line sitting on the baseline.

    `train_dt` may be None when plotting from a CSV, since the training interval
    distribution is a property of the training set and is not stored per record.
    It controls only the extrapolation marking, which is then omitted and named
    in `absent` rather than quietly dropped.
    """
    import matplotlib.pyplot as plt

    report_style()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.5),
                             constrained_layout=True)

    def panel(ax, letter, text):
        ax.set_title(f"{letter}   {text}", color=C_INK, fontsize=8.5,
                     fontweight="bold", loc="left")

    def binned(recs, key):
        x = np.array([r["dt_days"] for r in recs])
        y = np.array([r[key] for r in recs])
        edges = np.geomspace(max(x.min(), 1), x.max(), 12)
        cx, cy, lo, hi = [], [], [], []
        for a, b in zip(edges, edges[1:]):
            m = (x >= a) & (x < b)
            if m.sum() < 20:
                continue
            cx.append(np.sqrt(a * b)); cy.append(np.median(y[m]))
            lo.append(np.percentile(y[m], 25)); hi.append(np.percentile(y[m], 75))
        return np.array(cx), np.array(cy), np.array(lo), np.array(hi)

    dpred = np.array([r["d_pred"] for r in real_recs])
    dreal = np.array([r["d_real"] for r in real_recs])

    # --- a. is the prediction anything other than a copy? -------------------
    # A predictor that copied the anchor forward would put the whole orange
    # distribution on zero. Where it sits instead, against the blue one, is the
    # entire identity question in one panel.
    ax = axes[0]
    top = float(np.percentile(np.concatenate([dpred, dreal]), 97))
    bins = np.linspace(0, top, 55)
    for values, color, lab in ((dreal, C_REAL, "actual"),
                               (dpred, C_PRED, "predicted")):
        ax.hist(values, bins=bins, color=color, alpha=0.30, linewidth=0, zorder=2)
        ax.hist(values, bins=bins, histtype="step", color=color, linewidth=1.4,
                label=lab, zorder=3)
        ax.axvline(np.median(values), color=color, linewidth=1.0,
                   linestyle=(0, (3, 2)), zorder=4)
    ax.set_xlim(0, top)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.text(0.97, 0.97, "a copy of the anchor\nwould sit at 0",
            transform=ax.transAxes, ha="right", va="top",
            color=C_MUTED, fontsize=7)
    ax.set_xlabel("‖change‖ in embedding space")
    ax.legend(loc="center right", labelcolor=C_MUTED)
    panel(ax, "a", "Size of the change")

    # --- b. does either curve depend on elapsed time? -----------------------
    # Drawn from zero deliberately. On an axis that starts at the data, two flat
    # lines look like two trends; from zero they read as what they are, a small
    # constant under a large constant.
    ax = axes[1]
    for key, color, lab in (("d_real", C_REAL, "actual"),
                            ("d_pred", C_PRED, "predicted")):
        cx, cy, lo, hi = binned(real_recs, key)
        ax.fill_between(cx, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=1)
        ax.plot(cx, cy, color=color, linewidth=1.6, label=lab, zorder=3)
    if null_recs:
        cx, cy, _, _ = binned(null_recs, "d_pred")
        ax.plot(cx, cy, color=C_NULL, linewidth=1.6, linestyle=(0, (4, 2)),
                label="elapsed time shuffled", zorder=2)
    ax.set_ylim(0, None)
    _log_x(ax)
    ax.set_xlabel("elapsed time (days)")
    ax.set_ylabel("‖change‖ in embedding space")
    ax.legend(labelcolor=C_MUTED)
    panel(ax, "b", "Against elapsed time")

    # --- c. is it better than doing nothing? --------------------------------
    ax = axes[2]
    xs, pts, los, his, extrap = [], [], [], [], []
    mu, sd = (train_dt.mean(), train_dt.std()) if train_dt is not None else (None, None)
    for lo_e, hi_e in zip(HORIZON_EDGES, HORIZON_EDGES[1:]):
        sel = [r for r in real_recs if lo_e <= r["dt_days"] < hi_e]
        if len(sel) < 30:
            continue
        pt, l, h = bootstrap_ratio([r["err_model"] for r in sel],
                                   [r["err_base"] for r in sel])
        med = np.median([r["dt_days"] for r in sel])
        xs.append(med); pts.append(pt); los.append(l); his.append(h)
        extrap.append(False if mu is None else (med - mu) / max(sd, 1e-8) > 2)
    if xs:
        ax.axhline(1.0, color=C_INK, linewidth=1.2, zorder=2)
        ax.fill_between(xs, los, his, color=C_PRED, alpha=0.20, linewidth=0, zorder=1)
        ax.plot(xs, pts, color=C_PRED, linewidth=1.6, zorder=3)
        ax.scatter([x for x, e in zip(xs, extrap) if not e],
                   [p for p, e in zip(pts, extrap) if not e],
                   color=C_PRED, s=22, zorder=4, label="within training range")
        if any(extrap):
            ax.scatter([x for x, e in zip(xs, extrap) if e],
                       [p for p, e in zip(pts, extrap) if e],
                       facecolors="none", edgecolors=C_PRED, s=30, linewidth=1.4,
                       zorder=4, label="beyond training range")
            ax.legend(loc="lower left", labelcolor=C_MUTED)
        # A tenth either side of the baseline. The previous version let
        # matplotlib pick, which zoomed a 2% band to fill the panel and turned
        # "indistinguishable from doing nothing" into an apparent trend.
        span = 0.10
        ax.set_ylim(1 - span, 1 + span)
        ax.annotate("predicting no change at all", (xs[0], 1.0),
                    textcoords="offset points", xytext=(2, 4),
                    color=C_INK, fontsize=7)
        best = min(pts)
        ax.annotate(f"best is {1 - best:.1%} below it", (xs[int(np.argmin(pts))], best),
                    textcoords="offset points", xytext=(2, -11),
                    color=C_MUTED, fontsize=7)
    _log_x(ax)
    ax.set_xlabel("horizon (days)")
    ax.set_ylabel("error ÷ no-change error")
    panel(ax, "c", "Against doing nothing")

    # An absent control must be visible. Dropping the curve silently leaves a
    # figure that looks complete and is not.
    if absent:
        fig.text(0.0, -0.02, "Not drawn: " + "; ".join(absent),
                 color=C_MUTED, fontsize=7, ha="left", va="top")

    fig.savefig(out_path)
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="brain_mets_predictor_consecutive",
                    help="Supplies the data, cache and model. Must be a consecutive "
                         "config: the sequence dataset needs every visit, which only "
                         "the all-visits cache holds.")
    ap.add_argument("--train-config", default=None,
                    help="The config the predictor checkpoint was TRAINED with; its "
                         "pairing decides the dt/age z-score. Defaults to --config. "
                         "For the first_two arm pass "
                         "brain_mets_predictor_residual_volwin.")
    ap.add_argument("--predictor-ckpt", default=None,
                    help="Trained predictor weights. Omitted = random init, which is "
                         "the null: a flat dt response is the correct answer there.")
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--no-null", action="store_true",
                    help="Skip the dt-shuffled control (halves runtime).")
    ap.add_argument("--out-csv", type=Path,
                    default=Path("results/diagnostics/prediction/fixed_anchor.csv"))
    ap.add_argument("--fig", type=Path,
                    default=Path(figpath("prediction", "fixed_anchor.png")))
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="Rebuild the figure from a finished run's CSV. Loads no "
                         "model and no data, so the figure is provably the same "
                         "measurement rather than a re-derivation of it.")
    ap.add_argument("--null-csv", type=Path, default=None,
                    help="With --from-csv: the shuffled-elapsed-time control's own "
                         "CSV. Runs before 2026-09-10 did not save it separately, "
                         "so without this the control is omitted and said to be.")
    ap.add_argument("--train-dt-stats", default=None, metavar="MEAN,SD",
                    help="With --from-csv: the training interval mean and standard "
                         "deviation in days, from that run's own printout. Only the "
                         "extrapolation marking uses it; omitted, no horizon is "
                         "marked and the figure says so.")
    args = ap.parse_args()

    if args.from_csv:
        recs = load_records(args.from_csv)
        print(f"read {len(recs):,} rows from {args.from_csv} | "
              f"{len({r['patient_id'] for r in recs})} held-out patients")
        null_recs = load_records(args.null_csv) if args.null_csv else None
        train_dt = None
        if args.train_dt_stats:
            mean, sd = (float(v) for v in args.train_dt_stats.split(","))
            # make_figure needs only .mean() and .std() of this array, so two
            # points carrying exactly that mean and standard deviation are the
            # honest way to pass a statistic that was not stored per record.
            train_dt = np.array([mean - sd, mean + sd])
        absent = []
        if null_recs is None:
            absent.append("elapsed-time-shuffled control (not saved by this run)")
        if train_dt is None:
            absent.append("extrapolation marking (training interval range not stored)")
        make_figure(recs, null_recs, train_dt, args.fig, absent=absent)
        return

    from hydra import compose, initialize
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict
    from stable_pretraining import data as dt

    from brain_mets.brain_mets_pair_dataset import ACTION_DIM
    from brain_mets.brain_mets_sequence_dataset import BrainMetsSequenceDataset
    from lewm_core.utils import (get_img_preprocessor, load_frozen_parts,
                                 split_by_patient)

    # return_hydra_config + set_config is LOAD-BEARING. Every config under
    # config/train/data/ writes cache_path as ${hydra:runtime.cwd}/cache/...,
    # and a bare compose() leaves HydraConfig unset, so reading cache_path
    # raises InterpolationResolutionError before any work happens. train.py is
    # unaffected because @hydra.main sets it; only compose()-based scripts hit
    # this. Resolved here against the process cwd, which is the repo root.
    with initialize(version_base=None, config_path="../../../config/train"):
        cfg = compose(config_name=args.config, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        train_cfg = compose(config_name=args.train_config or args.config,
                            return_hydra_config=True)
    for c in (cfg, train_cfg):
        with open_dict(c):
            c.model.action_encoder.input_dim = ACTION_DIM

    # The two configs may differ in pairing and cache only. Anything else would
    # score the checkpoint under a setup it was not trained in.
    for key in ("seed", "train_split", "freeze_encoder_ckpt", "img_size",
                "data.dataset.normalization", "data.dataset.drop_blank_slices"):
        a, b = OmegaConf.select(cfg, key), OmegaConf.select(train_cfg, key)
        if a != b:
            raise ValueError(f"--config and --train-config disagree on {key}: {a!r} vs {b!r}")
    if OmegaConf.to_container(cfg.model) != OmegaConf.to_container(train_cfg.model):
        raise ValueError("--config and --train-config build different models")
    if train_cfg.data.dataset.get("dt_transform", "raw") != "raw":
        raise ValueError("collect() feeds raw days; a dt_transform run cannot be scored here")
    pairing = train_cfg.data.dataset.pairing

    # Sequence dataset, not the pair dataset: the anchor->t_j structure needs whole
    # timelines. It reads the same cache and the same normalization as the config.
    dataset = BrainMetsSequenceDataset(
        root_dir=ROOT_DIR, clinical_excel=CLINICAL_EXCEL,
        cache_path=cfg.data.dataset.cache_path,
        normalization=cfg.data.dataset.normalization,
        drop_blank_slices=cfg.data.dataset.get("drop_blank_slices", True),
        max_patients=args.max_patients)
    dataset.transform = dt.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size))

    _, val_idx = split_by_patient(dataset, cfg.seed, cfg.train_split)
    print(f"held-out: {len(val_idx):,} timelines from "
          f"{len({dataset.patient_ids[i] for i in val_idx})} patients")

    model = instantiate(cfg.model)
    load_frozen_parts(model, cfg.freeze_encoder_ckpt, tuple(cfg.freeze_parts))
    if args.predictor_ckpt:
        load_frozen_parts(model, args.predictor_ckpt,
                          ("predictor", "pred_proj", "action_encoder"))
        label = f"predictor = {Path(args.predictor_ckpt).name}"
    else:
        label = "predictor = RANDOM INIT (null)"
    model.eval()

    recs, train_dt = collect(model, dataset, val_idx, pairing)
    summarize(recs, train_dt, label)

    null_recs = None
    if not args.no_null:
        null_recs, _ = collect(model, dataset, val_idx, pairing, shuffle_dt=True)
        summarize(null_recs, train_dt, label + "  [dt SHUFFLED -- null]")

    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows):,} rows to {path}")

    write_csv(args.out_csv, recs)

    # The null is scored in this process and was previously discarded with it,
    # so --from-csv could never redraw the control and every rebuilt figure
    # carried the "not saved by this run" footnote. Persisting it beside the
    # real records is what makes --null-csv usable.
    if null_recs is not None:
        write_csv(args.out_csv.with_suffix(".null.csv"), null_recs)

    # The training interval distribution is a property of the training set, not
    # of any record, so it cannot be recovered from the CSV later. Saving it
    # beside the CSV is what lets --from-csv mark extrapolated horizons; without
    # it that marking is lost the moment the process exits.
    stats_path = args.out_csv.with_suffix(".train_dt.json")
    stats_path.write_text(json.dumps(
        {"mean_days": float(train_dt.mean()), "sd_days": float(train_dt.std(ddof=1)),
         "n_training_steps": int(train_dt.size), "config": args.config,
         "train_config": args.train_config or args.config, "pairing": pairing}, indent=2))
    print(f"Wrote {stats_path}")

    make_figure(recs, null_recs, train_dt, args.fig)


if __name__ == "__main__":
    main()
