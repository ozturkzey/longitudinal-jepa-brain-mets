"""
Linear probing on CACHED embeddings: for every encoder in results/embeddings/,
train two small linear heads (age regression + sex classification) on top of
its frozen, pre-extracted (N, D) embeddings, then compare which encoder's
representations carry the most clinical signal.

ASSUMPTIONS MADE:
    - Age is predicted as a REGRESSION (a number), not bucketed classes
    - Train/test split is done by PATIENT, not by visit, so no patient's
      brain appears in both train and test (avoids leakage)
    - Each results/embeddings/<name>.npz already holds one embedding vector
      per sample (produced by src/encoders/<name>.py's EncoderAdapter.run())
      -- this script does NOT run any encoder itself, only the linear heads
    - Because the embeddings are already cached (small: at most a few
      thousand rows), heads are trained with full-batch gradient descent --
      no DataLoader/mini-batching needed, unlike the old live-volume version

Every encoder is reported alongside LABEL-ONLY BASELINES on the same test
patients (majority sex class, mean training age). Read the deltas, not the raw
numbers: sex here is a ~59/41 split, so 59% accuracy is the do-nothing answer.

Usage:
    python3 train_linear_probe.py
    python3 train_linear_probe.py --embeddings-dir results/probe_cls/embeddings \
        --out-csv results/probe_cls/linear_probe_cls.csv \
        --out-plot results/probe_cls/linear_probe_cls.png
"""

import argparse
import glob
import os
import sys
from pathlib import Path

# Run as `python brain_mets/train_linear_probe.py` from the repo root, so the
# repo is not on sys.path by default and `scripts.diagnostics` would not import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnostics.figstyle import (C_INK, C_MUTED, C_PRED, C_REAL,
                                          TEXT_WIDTH_IN, report_style)

report_style()
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINICAL_EXCEL = os.path.join(REPO_ROOT, "Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx")
DEFAULT_EMBEDDINGS_DIR = os.path.join(REPO_ROOT, "results", "encoder_comparison", "embeddings")
DEFAULT_OUT_CSV = os.path.join(REPO_ROOT, "results", "encoder_comparison", "linear_probe_comparison.csv")
DEFAULT_OUT_PLOT = os.path.join(REPO_ROOT, "results", "encoder_comparison", "linear_probe_comparison.png")

EPOCHS = 3000
SEARCH_EPOCHS = 1500  # shorter fits used only during the weight_decay search below
LEARNING_RATE = 1e-3
TEST_FRACTION = 0.2
VAL_FRACTION_OF_TRAIN = 0.2  # carved out of the remaining 80% train patients, for model selection
SPLIT_SEED = 42
WEIGHT_DECAY_GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]


# ---------------------------------------------------------------------
# Load clinical data: patient_id -> (age, sex)  (unchanged from the
# original BrainIAC-only script)
# ---------------------------------------------------------------------
def load_clinical_lookup(excel_path, sex_policy="first"):
    """
    Build a dict: patient_id -> {"age": float, "sex": 0 or 1}.

    Age is taken from the patient's first row (age at first imaging); a patient
    legitimately has several ages across the sheet because they age between
    visits. Rows with age == -17 (a known anonymization artifact) are skipped.

    SEX IS NOT RELIABLE IN THIS RELEASE. Measured 2026-08-26: 510 of 1430
    patients are listed as BOTH "Male" and "Female" across their own rows. Sex
    does not change between visits, so that column is corrupted for at least a
    third of the cohort. Taking the first row silently resolves each such
    patient by coin flip, which is what the original version did and is why
    every sex probe has landed on the majority-class baseline.

    `sex_policy` makes the choice explicit instead of hidden:
        "first"      -- first row wins. The historical behaviour, kept as the
                        default so existing runs stay reproducible. Contaminated.
        "majority"   -- most frequent value across the patient's rows. Uses more
                        of the sheet, but cannot fix a column that is wrong.
        "consistent" -- keep only patients whose rows all agree. The only
                        uncontaminated option, and the one to use when asking
                        whether sex is decodable at all.

    The conflict count is always reported, so the contamination can never again
    pass unnoticed.
    """
    if sex_policy not in ("first", "majority", "consistent"):
        raise ValueError(f"unknown sex_policy {sex_policy!r}")

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb["Clinical_data"]

    ages, sex_votes = {}, {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        patient_id, _study_datetime, age, sex = row
        if patient_id is None or age is None or sex is None:
            continue
        if age == -17:  # known anonymization artifact, skip
            continue
        if sex not in ("Female", "Male"):
            raise ValueError(f"unexpected sex value {sex!r} for {patient_id}")
        ages.setdefault(patient_id, float(age))
        sex_votes.setdefault(patient_id, []).append(0 if sex == "Female" else 1)

    conflicted = {p for p, votes in sex_votes.items() if len(set(votes)) > 1}
    print(f"[clinical] {len(ages)} patients | {len(conflicted)} "
          f"({len(conflicted) / max(len(ages), 1):.0%}) have CONFLICTING sex across their own "
          f"rows | sex_policy={sex_policy}")

    lookup = {}
    for patient_id, votes in sex_votes.items():
        if sex_policy == "consistent" and patient_id in conflicted:
            continue
        if sex_policy == "majority":
            sex = int(np.bincount(votes).argmax())
        else:  # "first", and "consistent" where all votes agree anyway
            sex = votes[0]
        lookup[patient_id] = {"age": ages[patient_id], "sex": sex}

    if sex_policy == "consistent":
        print(f"[clinical] kept {len(lookup)} patients with self-consistent sex")
    return lookup


# ---------------------------------------------------------------------
# Patient-level fit/val/test split -- computed per embeddings file (from
# that file's own patient_ids), so it stays correct even if different
# encoders were run against different manifest snapshots. "val" is used
# only to pick weight_decay (never seen by the weights being fit); "fit"
# and "val" are combined into "train" for the final refit before the one
# held-out "test" evaluation each encoder is ultimately compared on.
# ---------------------------------------------------------------------
def three_way_split(patient_ids, test_fraction=TEST_FRACTION,
                     val_fraction_of_train=VAL_FRACTION_OF_TRAIN, seed=SPLIT_SEED):
    unique_ids = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)

    n_test = max(1, int(len(unique_ids) * test_fraction))
    test_patients = set(unique_ids[:n_test])
    remaining = unique_ids[n_test:]
    n_val = max(1, int(len(remaining) * val_fraction_of_train))
    val_patients = set(remaining[:n_val])
    fit_patients = set(remaining[n_val:])

    def mask_for(patients):
        return np.array([pid in patients for pid in patient_ids])

    return mask_for(fit_patients), mask_for(val_patients), mask_for(test_patients)


# ---------------------------------------------------------------------
# Linear heads on top of a fixed-dimension embedding vector.
# ---------------------------------------------------------------------
class LinearHeads(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.age_head = nn.Linear(embed_dim, 1)   # regression: one number
        self.sex_head = nn.Linear(embed_dim, 2)   # classification: 2 classes

    def forward(self, x):
        age_pred = self.age_head(x).squeeze(-1)  # (N,)
        sex_logits = self.sex_head(x)             # (N, 2)
        return age_pred, sex_logits


def evaluate(model, x, ages_norm, sexes, age_mean, age_std):
    """ages_norm is in normalized (z-scored) space; MAE is reported back in real years."""
    model.eval()
    with torch.no_grad():
        age_pred_norm, sex_logits = model(x)
        sex_pred = sex_logits.argmax(dim=1)
        sex_acc = (sex_pred == sexes).float().mean().item()
        age_mae = ((age_pred_norm - ages_norm) * age_std).abs().mean().item()
    return sex_acc, age_mae, sex_pred.tolist()


def label_only_baselines(sex_train, sex_test, age_train, age_test):
    """What the labels alone predict, ignoring the embeddings entirely:
    always guess the training set's majority sex, and always guess its mean age.

    Any encoder that does not beat these has shown no signal, however
    respectable its raw accuracy looks.
    """
    majority_sex = int(torch.bincount(sex_train, minlength=2).argmax())
    baseline_sex_acc = (sex_test == majority_sex).float().mean().item()
    baseline_age_mae = (age_test - age_train.mean()).abs().mean().item()
    return baseline_sex_acc, baseline_age_mae


def fit_heads(x_train, age_train_norm, sex_train, embed_dim, weight_decay, epochs):
    # Seeded head init: without this the weight_decay search is not reproducible
    # -- reruns on identical embeddings picked different wd (0.03 vs 0.1 for
    # mri_core) and moved test MAE by ~0.3 yr, which is the same size as the
    # gaps being compared.
    torch.manual_seed(SPLIT_SEED)
    model = LinearHeads(embed_dim=embed_dim)
    age_loss_fn = nn.MSELoss()
    sex_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

    for _ in range(epochs):
        model.train()
        age_pred, sex_logits = model(x_train)
        # Normalized age loss and cross-entropy are now on comparable scales (both ~O(1)),
        # so gradient descent doesn't implicitly prioritize one head over the other --
        # with raw (unnormalized) ages the MSE term is ~100x larger and dominates.
        loss = age_loss_fn(age_pred, age_train_norm) + sex_loss_fn(sex_logits, sex_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# ---------------------------------------------------------------------
# Train + evaluate the linear heads for one encoder's cached embeddings.
# ---------------------------------------------------------------------
def probe_one_encoder(name, npz_path, clinical_lookup):
    data = np.load(npz_path)
    embeddings = data["embeddings"].astype(np.float32)  # (N, D)
    patient_ids = data["patient_ids"]

    # Keep only samples whose patient has valid clinical labels.
    keep = [i for i, pid in enumerate(patient_ids) if str(pid) in clinical_lookup]
    if len(keep) < 10:
        print(f"[{name}] SKIPPED: only {len(keep)} samples have clinical labels")
        return None

    embeddings = embeddings[keep]
    patient_ids = [str(patient_ids[i]) for i in keep]
    ages = np.array([clinical_lookup[pid]["age"] for pid in patient_ids], dtype=np.float32)
    sexes = np.array([clinical_lookup[pid]["sex"] for pid in patient_ids], dtype=np.int64)
    embed_dim = embeddings.shape[1]

    fit_mask, val_mask, test_mask = three_way_split(patient_ids)
    train_mask = fit_mask | val_mask  # fit+val combined, for the final refit

    pids = np.array(patient_ids)
    assert not (set(pids[train_mask]) & set(pids[test_mask])), \
        f"{name}: patient leakage between train and test"
    print(f"[{name}] {len(embeddings)} labeled samples, dim={embed_dim}, "
          f"{fit_mask.sum()} fit / {val_mask.sum()} val / {test_mask.sum()} test (split by patient)")

    x = torch.from_numpy(embeddings)
    age_t = torch.from_numpy(ages)
    sex_t = torch.from_numpy(sexes)

    # Normalize age target for the loss (z-score) so its scale matches cross-entropy's --
    # stats computed from "fit" for the search phase, from "fit+val" for the final refit,
    # to avoid leaking held-out patients' label statistics into either fit.
    def normalize(mask_for_stats, ages_to_norm):
        mean, std = age_t[mask_for_stats].mean(), age_t[mask_for_stats].std()
        return (ages_to_norm - mean) / std, mean, std

    # --- weight_decay search on a held-out val split (never used to fit weights) ---
    age_fit_norm, fit_mean, fit_std = normalize(fit_mask, age_t[fit_mask])
    age_val_norm = (age_t[val_mask] - fit_mean) / fit_std
    best_wd, best_val_loss = WEIGHT_DECAY_GRID[0], float("inf")
    for wd in WEIGHT_DECAY_GRID:
        model = fit_heads(x[fit_mask], age_fit_norm, sex_t[fit_mask], embed_dim, wd, SEARCH_EPOCHS)
        _, val_age_mae, _ = evaluate(model, x[val_mask], age_val_norm, sex_t[val_mask], fit_mean, fit_std)
        with torch.no_grad():
            val_sex_loss = nn.functional.cross_entropy(model(x[val_mask])[1], sex_t[val_mask]).item()
        val_loss = (val_age_mae / fit_std.item()) ** 2 + val_sex_loss  # both terms ~O(1), same scale as training loss
        if val_loss < best_val_loss:
            best_val_loss, best_wd = val_loss, wd
    print(f"[{name}] selected weight_decay={best_wd} (val_age_mae normalized+val_sex_loss={best_val_loss:.3f})")

    # --- final refit on fit+val combined, evaluated once on the untouched test set ---
    age_train_norm, train_mean, train_std = normalize(train_mask, age_t[train_mask])
    age_test_norm = (age_t[test_mask] - train_mean) / train_std
    model = fit_heads(x[train_mask], age_train_norm, sex_t[train_mask], embed_dim, best_wd, EPOCHS)

    test_sex_acc, test_age_mae, sex_preds = evaluate(model, x[test_mask], age_test_norm, sex_t[test_mask], train_mean, train_std)

    # Per-patient test predictions, kept so two encoders can be compared with a
    # PAIRED test on the same patients. The MAE gaps between encoders here are
    # ~1 year against a per-patient error spread several times that, so an
    # unpaired eyeball comparison of two MAE numbers cannot tell a real
    # difference from split noise.
    with torch.no_grad():
        age_pred_years = (model(x[test_mask])[0] * train_std + train_mean).tolist()
    test_predictions = {
        "patient_id": [pid for pid, keep in zip(patient_ids, test_mask) if keep],
        "age_true": ages[test_mask].tolist(),
        "age_pred": age_pred_years,
        "sex_true": sexes[test_mask].tolist(),
        "sex_pred": sex_preds,
    }
    train_sex_acc, train_age_mae, _ = evaluate(model, x[train_mask], age_train_norm, sex_t[train_mask], train_mean, train_std)

    # Label-only baselines on the SAME test patients: what you get from the
    # labels alone, with the embeddings thrown away. Without these the table
    # reads like a ranking when it may be noise -- sex is a ~59/41 split here,
    # so "59% accuracy" is exactly the do-nothing answer, not a result.
    baseline_sex_acc, baseline_age_mae = label_only_baselines(
        sex_t[train_mask], sex_t[test_mask], age_t[train_mask], age_t[test_mask]
    )
    print(
        f"[{name}] TRAIN(sex_acc={train_sex_acc:.1%}, age_mae={train_age_mae:.1f}) | "
        f"TEST(sex_acc={test_sex_acc:.1%}, age_mae={test_age_mae:.1f}) | "
        f"BASELINE(sex_acc={baseline_sex_acc:.1%}, age_mae={baseline_age_mae:.1f})"
    )

    # Class-collapse check: if the model always predicts the same class,
    # a decent-looking accuracy can just be class imbalance, not real learning.
    n_pred_0, n_pred_1 = sex_preds.count(0), sex_preds.count(1)
    if n_pred_0 == 0 or n_pred_1 == 0:
        print(f"[{name}] WARNING: predicts only ONE sex class on test set -- looks like collapse, not real learning.")

    return {
        "encoder": name,
        "n_samples": len(embeddings),
        "embed_dim": embed_dim,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "weight_decay": best_wd,
        "test_sex_acc": test_sex_acc,
        "test_age_mae": test_age_mae,
        "train_sex_acc": train_sex_acc,
        "train_age_mae": train_age_mae,
        "baseline_sex_acc": baseline_sex_acc,
        "baseline_age_mae": baseline_age_mae,
        "test_predictions": test_predictions,
    }


READOUTS = (("cls", "class token"), ("patchmean", "patch mean"), ("emb", "projected"))
ENCODERS = (("lewm_e70", "JEPA encoder"), ("vit_random", "untrained encoder"))


def _group(results):
    """Split `<encoder>_<readout>` tags into a fixed (readout, encoder) grid.

    Bars were previously ordered by score, so the two panels could disagree on
    what stood where and neither told the reader which pairs to compare. The
    grid puts each readout's trained and untrained bar side by side, which is
    the comparison the probe exists to make.

    A tag matching no known encoder or readout is not dropped and not guessed
    at: it becomes its own single-bar group under its own name, so an unfamiliar
    encoder shows up plainly instead of being mislabelled as a familiar one.
    """
    by_tag = {r["encoder"]: r for r in results}
    groups, used = [], set()
    for readout, readout_label in READOUTS:
        bars = []
        for encoder, encoder_label in ENCODERS:
            tag = f"{encoder}_{readout}"
            if tag in by_tag:
                bars.append((encoder_label, by_tag[tag]))
                used.add(tag)
        if bars:
            groups.append((readout_label, bars))
    for tag, r in by_tag.items():
        if tag not in used:
            groups.append((tag, [(tag, r)]))
    return groups


def plot_comparison(results, out_plot):
    report_style()
    groups = _group(results)
    encoder_labels = [lab for _, lab in ENCODERS]
    colors = {encoder_labels[0]: C_PRED, encoder_labels[1]: C_MUTED}

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.9),
                             constrained_layout=True)
    width = 0.36

    for ax, key, base_key, scale, ylabel, letter, name, baseline_label in (
        (axes[0], "test_sex_acc", "baseline_sex_acc", 100, "accuracy (%)",
         "a", "Sex", "label-only baseline"),
        (axes[1], "test_age_mae", "baseline_age_mae", 1, "mean absolute error (years)",
         "b", "Age", "label-only baseline"),
    ):
        centers = []
        for g, (group_label, bars) in enumerate(groups):
            offsets = np.linspace(-width / 2, width / 2, len(bars)) if len(bars) > 1 else [0.0]
            centers.append(g)
            for (encoder_label, r), off in zip(bars, offsets):
                x = g + off
                ax.bar(x, r[key] * scale, width=width * 0.92,
                       color=colors.get(encoder_label, C_REAL), zorder=2,
                       label=encoder_label)
                # Per bar, not one line across the panel: the split, and with it
                # the baseline, is recomputed from each embeddings file's own
                # patient list, so the bars are not guaranteed to share one.
                ax.plot([x - width * 0.46, x + width * 0.46],
                        [r[base_key] * scale] * 2, color=C_INK, linewidth=1.4,
                        zorder=4, label=baseline_label)
        ax.set_xticks(centers)
        ax.set_xticklabels([g for g, _ in groups])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{letter}   {name}", fontweight="bold", fontsize=8.5, loc="left")
        ax.grid(axis="x", visible=False)

    # One legend for both panels, below them: each panel would otherwise repeat
    # the same three entries on top of its own bars, which is where the old
    # figure's legend box sat.
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="outside lower center",
               ncol=len(seen), labelcolor=C_MUTED)

    fig.savefig(out_plot)
    print(f"\nSaved comparison plot to {out_plot}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-plot", default=DEFAULT_OUT_PLOT)
    parser.add_argument("--sex-policy", default="first", choices=["first", "majority", "consistent"],
                        help="how to resolve the corrupted sex column; see load_clinical_lookup")
    args = parser.parse_args()

    print("Loading clinical data...")
    clinical_lookup = load_clinical_lookup(CLINICAL_EXCEL, sex_policy=args.sex_policy)
    print(f"  {len(clinical_lookup)} patients with valid age/sex")

    npz_paths = sorted(glob.glob(os.path.join(args.embeddings_dir, "*.npz")))
    if not npz_paths:
        print(f"No embeddings found in {args.embeddings_dir} -- run the encoder adapters first.")
        return
    print(f"Found {len(npz_paths)} cached embedding sets: {[os.path.basename(p) for p in npz_paths]}\n")

    results = []
    for npz_path in npz_paths:
        name = os.path.splitext(os.path.basename(npz_path))[0]
        result = probe_one_encoder(name, npz_path, clinical_lookup)
        if result is not None:
            results.append(result)

    if not results:
        print("No encoder produced usable results.")
        return

    results.sort(key=lambda r: r["test_sex_acc"], reverse=True)

    # "vs" columns are the whole point: a positive sex delta and a negative age
    # delta mean the embeddings carried signal the labels alone did not.
    print("\n=== Comparison (sorted by test sex accuracy) ===")
    print(f"{'encoder':<22}{'dim':>6}{'n':>7}{'wd':>7}{'sex_acc':>10}{'vs base':>10}"
          f"{'age_mae':>10}{'vs base':>10}")
    for r in results:
        print(f"{r['encoder']:<22}{r['embed_dim']:>6}{r['n_samples']:>7}{r['weight_decay']:>7}"
              f"{r['test_sex_acc']:>9.1%}{r['test_sex_acc'] - r['baseline_sex_acc']:>+9.1%}"
              f"{r['test_age_mae']:>10.1f}{r['test_age_mae'] - r['baseline_age_mae']:>+10.1f}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("encoder,embed_dim,n_samples,n_train,n_test,weight_decay,train_sex_acc,test_sex_acc,"
                "baseline_sex_acc,train_age_mae,test_age_mae,baseline_age_mae\n")
        for r in results:
            f.write(f"{r['encoder']},{r['embed_dim']},{r['n_samples']},{r['n_train']},{r['n_test']},{r['weight_decay']},"
                     f"{r['train_sex_acc']:.4f},{r['test_sex_acc']:.4f},{r['baseline_sex_acc']:.4f},"
                     f"{r['train_age_mae']:.4f},{r['test_age_mae']:.4f},{r['baseline_age_mae']:.4f}\n")
    print(f"\nSaved comparison table to {args.out_csv}")

    # Namespaced by the run's own CSV name: two runs over the same embeddings
    # (e.g. different --sex-policy) would otherwise overwrite each other's
    # per-patient files, and paired_probe_test.py would then silently compare
    # predictions from two different runs.
    run_name = os.path.splitext(os.path.basename(args.out_csv))[0]
    predictions_dir = os.path.join(os.path.dirname(args.out_csv), "predictions", run_name)
    os.makedirs(predictions_dir, exist_ok=True)
    for r in results:
        preds = r["test_predictions"]
        with open(os.path.join(predictions_dir, f"{r['encoder']}.csv"), "w") as f:
            f.write("patient_id,age_true,age_pred,sex_true,sex_pred\n")
            for row in zip(*preds.values()):
                f.write(f"{row[0]},{row[1]:.4f},{row[2]:.4f},{row[3]},{row[4]}\n")
    print(f"Saved per-patient test predictions to {predictions_dir}/")

    plot_comparison(results, args.out_plot)


if __name__ == "__main__":
    main()
