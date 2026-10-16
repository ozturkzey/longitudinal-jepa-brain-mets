"""Quantitative registration QC for the per-modality registered FLAIR tree.

Answers two questions that were never measured, only assumed:

1. **Did registration actually help?** `register_per_modality.py` runs
   `ants.registration(..., type_of_transform="SyN")` and keeps only
   `result["warpedmovout"]` -- it records NO similarity metric, so nothing on
   disk says whether any registration converged. Worse, `warpedmovout` always
   carries the FIXED image's grid, so the earlier "0/50 patients had shape
   inconsistency" check could only ever confirm shapes match. A SyN run that
   ran to completion but misaligned the anatomy produces a correctly-shaped,
   wrongly-aligned volume and passes that check.

   Measured here as before-vs-after similarity between a patient's first two
   FLAIR visits:
     - AFTER  = the registered tree (t1 already warped onto t0's grid).
     - BEFORE = the RAW t1 resampled onto raw t0's grid by header alone
                (`ants.resample_image_to_target`, no registration). That is the
                honest baseline: what you get from the scanner's own physical
                coordinates with no alignment work.
   Registration earned its compute iff AFTER materially exceeds BEFORE.

2. **Is t1 slice i the same anatomical plane as t0 slice i?** The pipeline
   indexes both volumes at the SAME slice_index by construction
   (`brain_mets_pair_dataset.py`, `compare_pooled_decode.py:162-163`), so the
   indices trivially match -- the real question is whether that index refers to
   the same anatomy. Measured by sweeping an offset k over [-MAX_OFFSET,
   +MAX_OFFSET] and recording which k maximizes corr(t0[i], t1[i+k]). A
   systematic k=0 peak means correspondence holds; a systematic non-zero peak
   is a slice shift; a flat/noisy profile means registration failed for that
   patient.

Why this matters: `compare_pooled_decode.py`'s row 6 (`real t1 - real t0`)
involves NO decoder at all, so whole-brain change there is either real biology,
intensity scaling, or exactly the misalignment this script tests for. Rows 7-8
are a separate question (the pooled bottleneck) and this script says nothing
about them.

All metrics are computed inside a brain mask. Without one, the large block of
(0, 0) background voxels present in both volumes inflates every correlation
regardless of alignment -- the same trap already documented for
BRAIN_MASK_THRESH in compare_pooled_decode.py.

Run under `venv` (needs antspyx + nibabel):
    ./venv/bin/python scripts/diagnostics/data/registration_qc.py --n-patients 30
"""

import argparse
import csv
import datetime
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import ants
import nibabel as nib

from brain_mets.slice_preprocessing import center_crop_to_square, slice_to_uint8_rgb
from scripts.diagnostics.figpaths import figpath
from scripts.diagnostics.figstyle import (TEXT_WIDTH_IN, label as report_label,
                                          report_style)
from scripts.diagnostics.masking import brain_mask

RAW_ROOT = Path(os.environ["BRAINJEPA_RAW"])
REG_ROOT = Path(os.environ["BRAINJEPA_REGISTERED"])
OUT_CSV = figpath("registration", "registration_qc.csv")
MODALITY = "FLAIR"
MAX_OFFSET = 3  # slice-shift sweep runs over k in [-3, +3]

# Display-window half-width for the median/IQR candidate, in IQRs. Chosen by
# scripts/diagnostics/data/clipping_sweep.py: k=5 clips 0.3% of lesion-proxy voxels,
# where the textbook k=2 and k=3 clip 100% and 77% of them. Anything narrower
# silently deletes the signal this project measures.
IQR_HALF_WIDTH = 5.0


def ncc(a, b, mask):
    """Normalized cross-correlation inside the mask (= Pearson r of the masked
    voxels). Returns nan when either side is constant there."""
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def dice(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    return float(2 * inter / total) if total else float("nan")


def slice_axis(shape):
    """The through-plane axis. These stacks are partial-coverage (22-48 slices
    at 4-6mm) against 256-512 in-plane, so the shortest axis is the slice axis
    -- same rule the datasets use."""
    return int(np.argmin(shape))


def offset_profile(vol_t0, vol_t1, mask_t0, axis):
    """For each slice i, find the offset k in [-MAX_OFFSET, MAX_OFFSET] whose
    t1 slice best matches t0's slice i.

    Returns (best_offsets, peak_sharpness):
      best_offsets   -- one k per evaluated slice
      peak_sharpness -- corr at the best k minus the mean corr over the other
                        offsets, averaged over slices. Near 0 means no real
                        peak: the slice profile is flat and the argmax is noise,
                        which is itself evidence of a failed registration.
    """
    n_slices = vol_t0.shape[axis]
    best_offsets, sharpness = [], []

    for i in range(n_slices):
        sl_t0 = np.take(vol_t0, i, axis=axis)
        sl_mask = np.take(mask_t0, i, axis=axis)
        # Skip slices with too little brain to correlate meaningfully -- mostly
        # the first/last few of a partial-coverage stack.
        if sl_mask.sum() < 100:
            continue

        corrs = {}
        for k in range(-MAX_OFFSET, MAX_OFFSET + 1):
            j = i + k
            if not (0 <= j < n_slices):
                continue
            sl_t1 = np.take(vol_t1, j, axis=axis)
            c = ncc(sl_t0, sl_t1, sl_mask)
            if not np.isnan(c):
                corrs[k] = c

        if len(corrs) < 2:
            continue
        best_k = max(corrs, key=corrs.get)
        others = [c for k, c in corrs.items() if k != best_k]
        best_offsets.append(best_k)
        sharpness.append(corrs[best_k] - float(np.mean(others)))

    if not best_offsets:
        return None, None
    return np.array(best_offsets), float(np.mean(sharpness))


def flair_timeline(root, patient_id, registered):
    """Sorted [(visit_date, path)] for this patient's FLAIR visits. The two
    trees name files differently: raw keeps the full
    {patient}_{date}_{time}_{MOD}.nii.gz, registered writes bare {MOD}.nii.gz."""
    patient_dir = root / patient_id
    if not patient_dir.is_dir():
        return []
    out = []
    for visit_date in sorted(os.listdir(patient_dir)):
        visit_dir = patient_dir / visit_date
        if not visit_dir.is_dir():
            continue
        if registered:
            path = visit_dir / f"{MODALITY}.nii.gz"
            if path.exists():
                out.append((visit_date, path))
        else:
            match = [f for f in os.listdir(visit_dir) if f.endswith(f"_{MODALITY}.nii.gz")]
            if match:
                out.append((visit_date, visit_dir / match[0]))
    return out


def evaluate_patient(patient_id):
    """One row of the QC table, or None if this patient can't be evaluated."""
    reg = flair_timeline(REG_ROOT, patient_id, registered=True)
    raw = flair_timeline(RAW_ROOT, patient_id, registered=False)
    if len(reg) < 2 or len(raw) < 2:
        return None

    # --- AFTER: the registered pair, already on a common grid ---
    (t0_date, t0_reg_path), (t1_date, t1_reg_path) = reg[0], reg[1]
    vol_t0 = nib.load(str(t0_reg_path)).get_fdata().astype(np.float32)
    vol_t1 = nib.load(str(t1_reg_path)).get_fdata().astype(np.float32)
    if vol_t0.shape != vol_t1.shape:
        # Should be impossible -- warpedmovout inherits the fixed grid. Worth
        # surfacing loudly rather than silently skipping if it ever happens.
        print(f"  {patient_id}: SHAPE MISMATCH in registered tree "
              f"{vol_t0.shape} vs {vol_t1.shape}, skipping")
        return None

    mask_t0, mask_t1 = brain_mask(vol_t0), brain_mask(vol_t1)
    if mask_t0 is None or mask_t1 is None:
        return None
    mask_union = np.logical_or(mask_t0, mask_t1)

    ncc_after = ncc(vol_t0, vol_t1, mask_union)

    # Intensity, not geometry. Rendering the worst-scoring patients showed
    # anatomy that lines up well while the WHOLE brain shifts brightness
    # between visits -- FLAIR intensity is not calibrated across scans.
    #
    # NCC on the raw volumes cannot see this: Pearson r is invariant to any
    # affine intensity change, so a globally brighter t1 scores identically.
    #
    # Per-slice min-max (slice_to_uint8_rgb, what the encoder is actually fed)
    # is ALSO affine within any one slice -- so on a single slice pair NCC
    # would be equally blind. What makes it visible here is that each slice
    # gets a DIFFERENT rescaling, chosen by its own min and max, and t0's
    # slice i gets a different one from t1's slice i. Pooling voxels that went
    # through different affine maps is no longer a single linear relation, so
    # the pooled correlation does move.
    #
    # Measured directly (25 patients, 512 slices): 37.5% of slices have their
    # min-max window change by >20% between visits, and median brain tissue
    # lands >=0.05 of the display range away from where it landed at t0 in 57%
    # of slices. A bright lesion sets the max; when it fades, every other voxel
    # in that slice is remapped -- a LOCAL change becomes a GLOBAL one before
    # the encoder sees anything.
    #
    # ncc_encoder therefore measures t0-vs-t1 correspondence in the space the
    # model actually works in. ncc_encoder << ncc_after means the preprocessing
    # is manufacturing apparent change out of a local one.
    axis = slice_axis(vol_t0.shape)
    enc_t0, enc_t1, enc_mask = [], [], []
    for i in range(vol_t0.shape[axis]):
        m = np.take(mask_t0, i, axis=axis)
        if m.sum() < 100:
            continue
        # Exactly the dataset's pipeline, minus the resize (which is a
        # monotone spatial op and would only blur both sides equally).
        a = slice_to_uint8_rgb(center_crop_to_square(np.take(vol_t0, i, axis=axis)))[..., 0]
        b = slice_to_uint8_rgb(center_crop_to_square(np.take(vol_t1, i, axis=axis)))[..., 0]
        enc_t0.append(a.astype(np.float32))
        enc_t1.append(b.astype(np.float32))
        enc_mask.append(center_crop_to_square(m))
    if not enc_t0:
        return None
    ncc_encoder = ncc(np.stack(enc_t0), np.stack(enc_t1), np.stack(enc_mask))

    mean_t0, mean_t1 = float(vol_t0[mask_t0].mean()), float(vol_t1[mask_t1].mean())
    intensity_ratio = mean_t1 / (mean_t0 or 1.0)

    dice_after = dice(mask_t0, mask_t1)
    mi_after = float(ants.image_mutual_information(
        ants.from_numpy(vol_t0), ants.from_numpy(vol_t1)))

    # --- BEFORE: raw t1 onto raw t0's grid, header only, no registration ---
    raw_t0 = ants.image_read(str(raw[0][1]))
    raw_t1 = ants.image_read(str(raw[1][1]))
    raw_t1_on_t0 = ants.resample_image_to_target(raw_t1, raw_t0, interp_type="linear")

    arr_t0, arr_t1 = raw_t0.numpy().astype(np.float32), raw_t1_on_t0.numpy().astype(np.float32)
    raw_mask_t0, raw_mask_t1 = brain_mask(arr_t0), brain_mask(arr_t1)
    if raw_mask_t0 is None or raw_mask_t1 is None:
        return None
    raw_union = np.logical_or(raw_mask_t0, raw_mask_t1)

    ncc_before = ncc(arr_t0, arr_t1, raw_union)
    dice_before = dice(raw_mask_t0, raw_mask_t1)
    mi_before = float(ants.image_mutual_information(raw_t0, raw_t1_on_t0))

    # --- slice correspondence, on the registered pair (axis set above) ---
    offsets, sharpness = offset_profile(vol_t0, vol_t1, mask_t0, axis)
    if offsets is None:
        return None

    return {
        "patient_id": patient_id,
        "t0_date": t0_date,
        "t1_date": t1_date,
        "n_slices": vol_t0.shape[axis],
        "n_slices_scored": len(offsets),
        "ncc_before": ncc_before,
        "ncc_after": ncc_after,
        "ncc_gain": ncc_after - ncc_before,
        "ncc_encoder": ncc_encoder,
        "intensity_ratio": intensity_ratio,
        "mi_before": mi_before,
        "mi_after": mi_after,
        "dice_before": dice_before,
        "dice_after": dice_after,
        "frac_offset_zero": float((offsets == 0).mean()),
        "mean_abs_offset": float(np.abs(offsets).mean()),
        "peak_sharpness": sharpness,
    }


def evaluate_patient_transitions(patient_id):
    """One row per CONSECUTIVE transition (t0->t1, t1->t2, ...), not one per patient.

    `evaluate_patient` above scores only each patient's FIRST transition, which is
    what registration_qc_full.csv contains: 1,120 rows against 7,601 transitions in
    the tree. That was sufficient while BrainMetsPairDataset only ever paired the
    two earliest visits. `pairing: consecutive` (2026-09-02) puts every later
    transition into training, so ~6,481 of them would otherwise enter unmeasured.

    Deliberately leaner than `evaluate_patient`: no "before" baseline and no
    normalization decomposition. Those answer "did SyN help?", which is settled and
    is a property of the registration itself. The question here is narrower -- is
    THIS transition usable -- and every visit is already on the patient's visit-1
    grid, so a header-only "before" resample is not a meaningful control for it.

    Returns [] when the patient cannot be scored, so callers can extend() blindly.
    """
    reg = flair_timeline(REG_ROOT, patient_id, registered=True)
    if len(reg) < 2:
        return []

    rows = []
    prev = None  # (date, vol, mask) -- carried so each volume is loaded once
    for t, (date, path) in enumerate(reg):
        vol = nib.load(str(path)).get_fdata().astype(np.float32)
        mask = brain_mask(vol)
        if mask is None:
            prev = None  # cannot bridge across an unmaskable volume
            continue
        if prev is not None:
            p_date, p_vol, p_mask = prev
            if p_vol.shape != vol.shape:
                # Impossible if registration used one fixed image throughout
                # (audited 2026-09-02: 0/1123 patients mismatch). Surface it
                # loudly rather than skip silently if it ever happens.
                print(f"  {patient_id}: SHAPE MISMATCH {p_date}->{date} "
                      f"{p_vol.shape} vs {vol.shape}, skipping transition")
            else:
                axis = slice_axis(vol.shape)
                union = np.logical_or(p_mask, mask)
                offsets, sharpness = offset_profile(p_vol, vol, p_mask, axis)
                rows.append({
                    "patient_id": patient_id,
                    "transition_index": t - 1,
                    "n_visits": len(reg),
                    "t0_date": p_date,
                    "t1_date": date,
                    "dt_days": (datetime.date.fromisoformat(date)
                                - datetime.date.fromisoformat(p_date)).days,
                    "n_slices": vol.shape[axis],
                    "n_slices_scored": 0 if offsets is None else len(offsets),
                    "ncc_after": ncc(p_vol, vol, union),
                    "dice_after": dice(p_mask, mask),
                    "frac_offset_zero": (float("nan") if offsets is None
                                         else float((offsets == 0).mean())),
                    "mean_abs_offset": (float("nan") if offsets is None
                                        else float(np.abs(offsets).mean())),
                    "peak_sharpness": float("nan") if sharpness is None else sharpness,
                })
        prev = (date, vol, mask)
    return rows


# Flagging thresholds live in transition_exclusion.py, which has no antspyx
# dependency and so is importable from `.venv` as well as this module's `venv`.
# One definition, so scoring and exclusion cannot drift apart.
from scripts.diagnostics.data.transition_exclusion import FLAG_DICE, FLAG_FRAC_ZERO  # noqa: E402


def summarize_transitions(rows):
    """Summary for --all-transitions. `summarize` cannot be reused: it reports
    before/after gains, and these rows have no "before" by construction."""
    def col(name):
        return np.array([r[name] for r in rows], dtype=np.float64)

    n = len(rows)
    n_pat = len({r["patient_id"] for r in rows})
    print(f"\n{'=' * 72}\nPER-TRANSITION QC -- {n} transitions, {n_pat} patients\n{'=' * 72}")

    ncc_a, dice_a = col("ncc_after"), col("dice_after")
    frac0, mabs, sharp = col("frac_offset_zero"), col("mean_abs_offset"), col("peak_sharpness")
    print(f"\n  NCC (brain mask)     median {np.nanmedian(ncc_a):+.3f}   "
          f"q05 {np.nanpercentile(ncc_a, 5):+.3f}")
    print(f"  Dice of brain masks  median {np.nanmedian(dice_a):.3f}   "
          f"q05 {np.nanpercentile(dice_a, 5):.3f}")
    print(f"  slices peaking k=0   median {np.nanmedian(frac0):.1%}   "
          f"q05 {np.nanpercentile(frac0, 5):.1%}")
    print(f"  mean |offset|        median {np.nanmedian(mabs):.2f} slices")
    print(f"  peak sharpness       median {np.nanmedian(sharp):.3f}")

    # Is the first transition special? It is the only one registration_qc_full.csv
    # ever measured, so the whole cohort table rests on it being representative.
    first = np.array([r["transition_index"] == 0 for r in rows])
    if first.any() and (~first).any():
        print(f"\n  {'':22}{'first (t0->t1)':>16}{'later':>12}")
        for label, c in (("NCC", ncc_a), ("Dice", dice_a), ("k=0 fraction", frac0)):
            print(f"  {label:<22}{np.nanmedian(c[first]):>16.3f}{np.nanmedian(c[~first]):>12.3f}")
        print(f"  {'n':<22}{int(first.sum()):>16,}{int((~first).sum()):>12,}")

    bad = [r for r in rows
           if (r["dice_after"] < FLAG_DICE
               or (r["frac_offset_zero"] == r["frac_offset_zero"]  # not nan
                   and r["frac_offset_zero"] < FLAG_FRAC_ZERO))]
    print(f"\n  FLAGGED (dice < {FLAG_DICE} or k=0 fraction < {FLAG_FRAC_ZERO:.0%}): "
          f"{len(bad)}/{n} ({len(bad)/max(n,1):.2%}) "
          f"across {len({r['patient_id'] for r in bad})} patients")
    for r in sorted(bad, key=lambda r: r["dice_after"])[:15]:
        print(f"    {r['patient_id']} {r['transition_index']:>3} "
              f"({r['t0_date']}->{r['t1_date']})  dice {r['dice_after']:.3f}  "
              f"k=0 {r['frac_offset_zero']:.0%}  ncc {r['ncc_after']:+.3f}")


def display_brain_mask(volume):
    """Brain mask used only to draw Figure 3: voxels at least half as bright as
    the median nonzero voxel. The QC scores keep `brain_mask` (Otsu), which is
    what produced Section 4.1. On a first visit, the registration's fixed image,
    Otsu can split inside the tissue and keep only its bright half; the window
    measured on it then draws the whole scan too dark, and the difference row
    shows a uniform brightening that is not there."""
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return None
    mask = volume > 0.5 * float(np.median(nonzero))
    return mask if mask.any() else None


def render_patient(patient_id, out_path, n_show=5, slices=None):
    """Render one patient's registered t0/t1 slices and their difference.

    The point is to check the METRIC against the eye: a number saying "poorly
    aligned" should correspond to a picture that looks poorly aligned. If they
    disagree, the metric is what is wrong.
    """
    report_style()
    import matplotlib.pyplot as plt

    reg = flair_timeline(REG_ROOT, patient_id, registered=True)
    if len(reg) < 2:
        raise SystemExit(f"{patient_id}: needs 2+ registered FLAIR visits")

    vol_t0 = nib.load(str(reg[0][1])).get_fdata().astype(np.float32)
    vol_t1 = nib.load(str(reg[1][1])).get_fdata().astype(np.float32)
    axis = slice_axis(vol_t0.shape)
    n_slices = vol_t0.shape[axis]

    # Evenly spaced through the middle 60% -- the end slices of a
    # partial-coverage stack are mostly air and show nothing useful.
    lo, hi = int(0.2 * n_slices), int(0.8 * n_slices)
    if slices:
        indices = np.array(sorted(slices), dtype=int)
        n_show = len(indices)
    else:
        indices = np.linspace(lo, max(lo + 1, hi - 1), n_show).astype(int)

    # Per-VOLUME median/IQR window -- the production intensity mapping. This
    # used per-slice min-max until 2026-09-14, which is wrong for this figure:
    # it rescales a whole slice by that slice's own brightest voxel, and on
    # FLAIR the brightest voxel is usually a lesion. A lesion that brightens or
    # fades then re-lights every other voxel in its slice, so the difference
    # row came out a whole-brain red wash that is rescaling, not anatomy.
    m_t0, m_t1 = display_brain_mask(vol_t0), display_brain_mask(vol_t1)
    nvol_t0 = norm_mediqr_per_volume(vol_t0, m_t0, axis)
    nvol_t1 = norm_mediqr_per_volume(vol_t1, m_t1, axis)
    # Background sits below the window, so it clips to a small constant that
    # differs slightly between the two scans. Left in, it tints the whole
    # outside and, worse, feeds the 99th-percentile colour scale below.
    vol_brain = m_t0 | m_t1

    fig, axes = plt.subplots(3, n_show,
                             figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 3 / n_show),
                             constrained_layout=True)
    row_names = ("Earlier Scan", "Later Scan", "Difference\n(later − earlier)")
    for c, i in enumerate(indices):
        n0 = np.take(nvol_t0, i, axis=axis)
        n1 = np.take(nvol_t1, i, axis=axis)
        mb = np.take(vol_brain, i, axis=axis)
        diff = np.where(mb, n1 - n0, np.nan)
        lim = (np.percentile(np.abs(diff[mb]), 99) if mb.any() else 1.0) or 1.0

        for r, (img, kw) in enumerate([
            (n0, dict(cmap="gray", vmin=0, vmax=1)),
            (n1, dict(cmap="gray", vmin=0, vmax=1)),
            (diff, dict(cmap="RdBu_r", vmin=-lim, vmax=lim)),
        ]):
            ax = axes[r, c]
            ax.imshow(img.T, origin="lower", **kw)
            # Not axis("off"): that would take the row labels with it. The three
            # row names were repeated on all fifteen panels before, in the
            # dataset's own shorthand.
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            # Slice numbers go under the bottom row; the figure has no title,
            # the report caption names it.
            if r == 2:
                ax.set_xlabel(f"slice {i}")
            if c == 0:
                ax.set_ylabel(row_names[r])

    # No suptitle: the report caption names the figure, and a title inside
    # the image would repeat it. Patient, visit dates and sequence are caption
    # material too.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote {out_path}")



# ---------------------------------------------------------------------------
# How much of the observed t0->t1 change is a normalization artifact, and how
# much is anatomy?
# ---------------------------------------------------------------------------
# Two decompositions, run on the same slices so they are directly comparable.
#
# 1. UNIFORM vs STRUCTURED. Any change map splits into a spatially constant
#    part (every brain voxel moves by the same amount -- the signature of a
#    rescaling) and the residual structure. Reported as a share of total change
#    energy, so it needs no choice of "correct" normalization to interpret.
#
# 2. CURRENT vs ROBUST normalization. The current pipeline min-maxes each slice
#    independently, so one bright lesion sets the window for 50k voxels. The
#    robust alternative windows on the MEDIAN and IQR of brain tissue, computed
#    once per volume: dominated by normal tissue, so a lesion appearing or
#    fading barely moves it. If the uniform share collapses under the robust
#    normalization while the structured part survives, the uniform part was
#    an artifact of the preprocessing and not something that happened to the
#    patient.
#
# Scale note: the two normalizations produce different units, so magnitudes are
# never compared across them directly -- only energy SHARES (scale-free) and
# the correlation between the two structured parts.

def robust_window(vol, mask):
    """Median/IQR of brain tissue -- the lesion-insensitive alternative to
    per-slice min/max. Computed per VOLUME so all its slices share a window."""
    vals = vol[mask]
    med = np.median(vals)
    iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
    return med, (iqr or 1.0)


def split_energy(d, mask):
    """Change map -> (uniform share, structured part). The uniform part is the
    mean over brain voxels; energy is the sum of squares, so the shares add
    to 1 exactly."""
    vals = d[mask]
    if vals.size == 0:
        return float("nan"), None
    g = vals.mean()
    total = float((vals ** 2).sum())
    if total == 0:
        return float("nan"), None
    uniform = float(g ** 2 * vals.size)
    return uniform / total, vals - g


def decompose_patient(patient_id):
    reg = flair_timeline(REG_ROOT, patient_id, registered=True)
    if len(reg) < 2:
        return None
    vol_t0 = nib.load(str(reg[0][1])).get_fdata().astype(np.float32)
    vol_t1 = nib.load(str(reg[1][1])).get_fdata().astype(np.float32)
    if vol_t0.shape != vol_t1.shape:
        return None
    mask_t0, mask_t1 = brain_mask(vol_t0), brain_mask(vol_t1)
    if mask_t0 is None or mask_t1 is None:
        return None
    mask = np.logical_and(mask_t0, mask_t1)  # tissue present in BOTH visits
    axis = slice_axis(vol_t0.shape)

    med0, iqr0 = robust_window(vol_t0, mask_t0)
    med1, iqr1 = robust_window(vol_t1, mask_t1)
    lo0, hi0 = np.percentile(vol_t0[mask_t0], [2, 98])
    lo1, hi1 = np.percentile(vol_t1[mask_t1], [2, 98])
    rng0, rng1 = (hi0 - lo0) or 1.0, (hi1 - lo1) or 1.0

    cur_u, pct_u, rob_u, cur_s, rob_s = [], [], [], [], []
    for i in range(vol_t0.shape[axis]):
        m = np.take(mask, i, axis=axis)
        if m.sum() < 100:
            continue
        a, b = np.take(vol_t0, i, axis=axis), np.take(vol_t1, i, axis=axis)

        # current: per-slice min-max, exactly slice_to_uint8_rgb's arithmetic
        ra, rb = np.ptp(a), np.ptp(b)
        if ra <= 0 or rb <= 0:
            continue
        d_cur = (b - b.min()) / rb - (a - a.min()) / ra

        # candidate A: per-volume 2-98 percentile of brain tissue
        d_pct = (b - lo1) / rng1 - (a - lo0) / rng0
        # candidate B: per-volume median/IQR, windowed at IQR_HALF_WIDTH and
        # CLIPPED -- i.e. exactly what would ship, not an unbounded z-score.
        w0 = 2 * IQR_HALF_WIDTH * iqr0
        w1 = 2 * IQR_HALF_WIDTH * iqr1
        b_n = np.clip((b - (med1 - IQR_HALF_WIDTH * iqr1)) / (w1 or 1.0), 0, 1)
        a_n = np.clip((a - (med0 - IQR_HALF_WIDTH * iqr0)) / (w0 or 1.0), 0, 1)
        d_rob = b_n - a_n

        u_c, s_c = split_energy(d_cur, m)
        u_p, _s_p = split_energy(d_pct, m)
        u_r, s_r = split_energy(d_rob, m)
        if s_c is None or s_r is None or np.isnan(u_p):
            continue
        cur_u.append(u_c); pct_u.append(u_p); rob_u.append(u_r)
        cur_s.append(s_c); rob_s.append(s_r)

    if not cur_u:
        return None
    sc, sr = np.concatenate(cur_s), np.concatenate(rob_s)
    struct_corr = float(np.corrcoef(sc, sr)[0, 1]) if sc.std() and sr.std() else float("nan")
    return {
        "patient_id": patient_id,
        "n_slices": len(cur_u),
        "uniform_share_current": float(np.mean(cur_u)),
        "uniform_share_pct": float(np.mean(pct_u)),
        "uniform_share_robust": float(np.mean(rob_u)),
        "structured_corr": struct_corr,
    }


def run_decomposition(n_patients):
    patient_ids = sorted(p for p in os.listdir(REG_ROOT) if (REG_ROOT / p).is_dir())
    if n_patients:
        patient_ids = patient_ids[:n_patients]
    rows = []
    for i, pid in enumerate(patient_ids, 1):
        r = decompose_patient(pid)
        if r is None:
            continue
        rows.append(r)
        print(f"[{i}/{len(patient_ids)}] {pid}: uniform share "
              f"{r['uniform_share_current']:.1%} -> A {r['uniform_share_pct']:.1%} "
              f"/ B {r['uniform_share_robust']:.1%}")
    if not rows:
        raise SystemExit("nothing to decompose")

    cur = np.array([r["uniform_share_current"] for r in rows])
    pct = np.array([r["uniform_share_pct"] for r in rows])
    rob = np.array([r["uniform_share_robust"] for r in rows])
    cor = np.array([r["structured_corr"] for r in rows])
    n = len(rows)

    print(f"\n{'=' * 72}\nARTIFACT vs ANATOMY -- {n} patients\n{'=' * 72}")
    print("\nShare of total change ENERGY that is a spatially UNIFORM shift")
    print("(a uniform shift is what a rescaling does; anatomy is never uniform)")
    print(f"  current  (per-slice min-max) : median {np.median(cur):6.1%}   "
          f"IQR {np.percentile(cur, 25):.1%}-{np.percentile(cur, 75):.1%}")
    print(f"  fixed A  (per-volume 2-98pct): median {np.median(pct):6.1%}   "
          f"IQR {np.percentile(pct, 25):.1%}-{np.percentile(pct, 75):.1%}   "
          f"improves {int((pct < cur).sum())}/{n}")
    print(f"  fixed B  (per-volume med/IQR): median {np.median(rob):6.1%}   "
          f"IQR {np.percentile(rob, 25):.1%}-{np.percentile(rob, 75):.1%}   "
          f"improves {int((rob < cur).sum())}/{n}")
    print(f"  patients where the uniform share DROPS under robust: "
          f"{int((rob < cur).sum())}/{n}")
    print(f"\n  => on a median patient, {np.median(cur):.1%} of what the pipeline calls")
    print(f"     'change' is a whole-brain shift; robust normalization leaves {np.median(rob):.1%}.")
    print(f"     The difference, {np.median(cur) - np.median(rob):.1%} of change energy, is")
    print("     preprocessing artifact rather than anatomy.")
    print("\nDoes the STRUCTURED (non-uniform) part survive renormalization?")
    print(f"  corr(structured_current, structured_robust): median {np.median(cor):+.3f}")
    print("  (high = the real spatial signal is the same under both, i.e. only the")
    print("   uniform component was affected; low = normalization also distorts structure)")



# ---------------------------------------------------------------------------
# Candidate normalizations, and a before/after figure
# ---------------------------------------------------------------------------
# All three map a volume to a display range. They differ ONLY in what sets the
# window, which is the whole point:
#
#   minmax_per_slice -- current pipeline. Window = that slice's own min and
#                       max, so the single brightest voxel (usually a lesion)
#                       sets the scale for ~50k voxels.
#   pct_per_volume   -- 2nd/98th percentile of BRAIN voxels over the whole
#                       volume. Still a min-max in spirit, but the percentiles
#                       ignore the extremes that cause the problem.
#   mediqr_per_volume-- median and IQR of brain voxels. Centred on normal
#                       tissue rather than on the range.
#
# Per-VOLUME (not per-slice) matters as much as robustness: it means every
# slice of one scan shares a window, so a lesion in one slice cannot re-light
# the rest of that scan.

def norm_minmax_per_slice(vol, mask, axis):
    out = np.zeros_like(vol, dtype=np.float32)
    for i in range(vol.shape[axis]):
        sl = np.take(vol, i, axis=axis)
        r = np.ptp(sl)
        idx = [slice(None)] * vol.ndim
        idx[axis] = i
        out[tuple(idx)] = (sl - sl.min()) / r if r > 0 else 0.0
    return out


def norm_pct_per_volume(vol, mask, axis):
    lo, hi = np.percentile(vol[mask], [2, 98])
    return ((vol - lo) / ((hi - lo) or 1.0)).astype(np.float32)


def norm_mediqr_per_volume(vol, mask, axis):
    """median +/- IQR_HALF_WIDTH*IQR, CLIPPED to [0,1] -- the normalization as
    it would actually be implemented, not an unbounded z-score. The clip is
    load-bearing: slice_to_uint8_rgb ends in `(sl*255).astype(np.uint8)`, which
    WRAPS on out-of-range input, so an unclipped bright lesion would come out
    black."""
    med, iqr = robust_window(vol, mask)
    lo, hi = med - IQR_HALF_WIDTH * iqr, med + IQR_HALF_WIDTH * iqr
    return np.clip((vol - lo) / ((hi - lo) or 1.0), 0, 1).astype(np.float32)


NORMALIZATIONS = [
    ("current: per-slice min-max", norm_minmax_per_slice),
    ("fixed A: per-volume 2-98 pct", norm_pct_per_volume),
    ("fixed B: per-volume median/IQR", norm_mediqr_per_volume),
]


def compare_normalizations(patient_ids, out_path):
    """One column per patient, showing the SAME slice's change map under each
    normalization. The slice shown is that patient's most-changed one, so the
    comparison is made where there is actually something to see."""
    report_style()
    import matplotlib.pyplot as plt

    cols = []
    for pid in patient_ids:
        reg = flair_timeline(REG_ROOT, pid, registered=True)
        if len(reg) < 2:
            print(f"  {pid}: <2 FLAIR visits, skipping")
            continue
        v0 = nib.load(str(reg[0][1])).get_fdata().astype(np.float32)
        v1 = nib.load(str(reg[1][1])).get_fdata().astype(np.float32)
        if v0.shape != v1.shape:
            continue
        m0, m1 = brain_mask(v0), brain_mask(v1)
        if m0 is None or m1 is None:
            continue
        mask = np.logical_and(m0, m1)
        axis = slice_axis(v0.shape)

        maps = {}
        for name, fn in NORMALIZATIONS:
            maps[name] = fn(v1, m1, axis) - fn(v0, m0, axis)

        # Pick the slice with the most STRUCTURED change under the robust
        # window -- i.e. where something actually happened, rather than the
        # slice with the biggest global shift (which would be picking the
        # artifact on purpose).
        ref = maps["fixed A: per-volume 2-98 pct"]
        best_i, best_score = None, -1.0
        for i in range(v0.shape[axis]):
            mm = np.take(mask, i, axis=axis)
            if mm.sum() < 500:
                continue
            d = np.take(ref, i, axis=axis)[mm]
            score = float(np.std(d - d.mean()))
            if score > best_score:
                best_i, best_score = i, score
        if best_i is None:
            continue

        mm = np.take(mask, best_i, axis=axis)
        col = {
            "pid": pid, "i": best_i, "mask": mm,
            "t0": np.take(norm_minmax_per_slice(v0, m0, axis), best_i, axis=axis),
            "t1": np.take(norm_minmax_per_slice(v1, m1, axis), best_i, axis=axis),
        }
        for name, _ in NORMALIZATIONS:
            d = np.take(maps[name], best_i, axis=axis)
            u, _rest = split_energy(d, mm)
            col[name] = (d, u)
        cols.append(col)

    if not cols:
        raise SystemExit("no patients could be rendered")

    n = len(cols)
    nrows = 2 + len(NORMALIZATIONS)
    # One patient is the report figure, and five stages stacked in a single
    # column come out taller than the page. Lay a single patient out across the
    # width instead; several patients stay one column each, as the working
    # figure needs.
    if n == 1:
        fig, row = plt.subplots(1, nrows,
                                figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN / nrows + 0.5),
                                constrained_layout=True)
        axes = np.array(row).reshape(nrows, 1)
    else:
        width = min(TEXT_WIDTH_IN, 1.9 * n)
        fig, axes = plt.subplots(nrows, n, figsize=(width, width * nrows / n),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)

    for c, col in enumerate(cols):
        mm = col["mask"]
        axes[0, c].imshow(col["t0"].T, origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"earlier scan (slice {col['i']})")
        axes[1, c].imshow(col["t1"].T, origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[1, c].set_title("later scan")

        for r, (name, _) in enumerate(NORMALIZATIONS, start=2):
            d, u = col[name]
            dm = np.where(mm, d, np.nan)          # show brain only
            lim = np.nanpercentile(np.abs(dm), 99) or 1.0
            axes[r, c].imshow(dm.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            # The uniform share IS this figure's measurement, not decoration,
            # so it stays on the panel it describes.
            axes[r, c].set_title(f"{report_label(name)}\nuniform share {u:.0%}")

    for ax in axes.ravel():
        ax.axis("off")

    # No title. What each panel's scale is, and that a lower uniform share is
    # better, belong to the caption.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote {out_path}")



def show_normalized_inputs(patient_id, out_path, n_show=5):
    """What the ENCODER is fed, under each normalization.

    --compare-norm shows the DIFFERENCE maps; this shows the images themselves.
    The question it answers by eye: does the same tissue look the same at t0 and
    t1? Under a per-slice window it need not, because each slice is rescaled by
    its own extremes. Under a per-volume window the whole scan shares one
    mapping, so tissue that did not change should not appear to.

    Every panel is drawn on ONE fixed grey scale per normalization (not
    per-panel autoscaling), because per-panel autoscaling would silently redo
    the very normalization being compared and hide the effect.
    """
    report_style()
    import matplotlib.pyplot as plt

    reg = flair_timeline(REG_ROOT, patient_id, registered=True)
    if len(reg) < 2:
        raise SystemExit(f"{patient_id}: needs 2+ registered FLAIR visits")
    v0 = nib.load(str(reg[0][1])).get_fdata().astype(np.float32)
    v1 = nib.load(str(reg[1][1])).get_fdata().astype(np.float32)
    m0, m1 = brain_mask(v0), brain_mask(v1)
    axis = slice_axis(v0.shape)
    n_slices = v0.shape[axis]
    lo, hi = int(0.25 * n_slices), int(0.75 * n_slices)
    indices = np.linspace(lo, max(lo + 1, hi - 1), n_show).astype(int)

    fig, axes = plt.subplots(2 * len(NORMALIZATIONS), n_show,
                             figsize=(TEXT_WIDTH_IN,
                                      TEXT_WIDTH_IN * 2 * len(NORMALIZATIONS) / n_show),
                             constrained_layout=True)

    for r, (name, fn) in enumerate(NORMALIZATIONS):
        a, b = fn(v0, m0, axis), fn(v1, m1, axis)
        # common display window from BOTH visits' brain voxels, so t0 and t1
        # are drawn on the same ruler -- the whole point of the comparison
        both = np.concatenate([a[m0], b[m1]])
        vmin, vmax = np.percentile(both, [1, 99])
        for c, i in enumerate(indices):
            for k, (vol, label) in enumerate([(a, "t0"), (b, "t1")]):
                ax = axes[2 * r + k, c]
                ax.imshow(np.take(vol, i, axis=axis).T, origin="lower",
                          cmap="gray", vmin=vmin, vmax=vmax)
                if c == 0:
                    ax.set_title(f"{report_label(name)}, {label}", loc="left")
    for ax in axes.ravel():
        ax.axis("off")

    # No title; the caption says whose scans these are and that each pair of
    # rows shares one grey scale.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote {out_path}")


def summarize(rows):
    def col(name):
        return np.array([r[name] for r in rows], dtype=np.float64)

    n = len(rows)
    print(f"\n{'=' * 72}\nREGISTRATION QC -- {n} patients\n{'=' * 72}")

    print("\n--- 1. Did registration help? (NCC between t0 and t1, brain mask) ---")
    before, after, gain = col("ncc_before"), col("ncc_after"), col("ncc_gain")
    print(f"  before (header alignment only) : median {np.nanmedian(before):+.3f}")
    print(f"  after  (ANTs SyN registration) : median {np.nanmedian(after):+.3f}")
    print(f"  gain                           : median {np.nanmedian(gain):+.3f}")
    n_worse = int((gain < 0).sum())
    print(f"  patients where registration did NOT help: {n_worse}/{n} ({n_worse / n:.0%})")

    enc, ratio = col("ncc_encoder"), col("intensity_ratio")
    print(f"\n  after, in the ENCODER's space  : median {np.nanmedian(enc):+.3f}"
          f"   ({np.nanmedian(enc - after):+.3f} vs raw NCC)")
    print(f"  t1/t0 mean brain intensity     : median {np.nanmedian(ratio):.2f}x"
          f"  (range {np.nanmin(ratio):.2f}-{np.nanmax(ratio):.2f})")
    print("  (encoder space = per-slice min-max, what the model is fed. A large")
    print("   DROP there means the preprocessing manufactures apparent change.)")

    print("\n  Dice of brain masks:  "
          f"before {np.nanmedian(col('dice_before')):.3f} -> after {np.nanmedian(col('dice_after')):.3f}")
    # ants.image_mutual_information returns MI as a COST, i.e. negated -- MORE
    # NEGATIVE means MORE similar. Spelled out because a table of numbers
    # getting more negative reads like a regression otherwise.
    print("  Mutual information:   "
          f"before {np.nanmedian(col('mi_before')):.3f} -> after {np.nanmedian(col('mi_after')):.3f}"
          "   (ANTs cost: MORE NEGATIVE = more similar)")

    print("\n--- 2. Slice correspondence: is t1 slice i the same plane as t0 slice i? ---")
    frac_zero, mean_abs = col("frac_offset_zero"), col("mean_abs_offset")
    sharp = col("peak_sharpness")
    print(f"  slices whose best match is at offset 0 : median {np.nanmedian(frac_zero):.1%} per patient")
    print(f"  mean |best offset|                     : median {np.nanmedian(mean_abs):.2f} slices")
    print(f"  peak sharpness (best corr - mean other): median {np.nanmedian(sharp):+.4f}")
    n_shifted = int((frac_zero < 0.5).sum())
    print(f"  patients where MOST slices peak away from 0: {n_shifted}/{n} ({n_shifted / n:.0%})")
    print("  (high frac-at-0 + positive sharpness = correspondence holds;")
    print("   sharpness near 0 = flat profile, the argmax is noise)")

    print("\n--- 3. Worst 10 patients by post-registration NCC ---")
    for r in sorted(rows, key=lambda r: r["ncc_after"])[:10]:
        print(f"  {r['patient_id']:20s} ncc {r['ncc_before']:+.3f} -> {r['ncc_after']:+.3f} "
              f"| offset0 {r['frac_offset_zero']:5.1%} | sharp {r['peak_sharpness']:+.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-patients", type=int, default=30,
                    help="How many patients to score (default 30, a fast sample). "
                         "Use 0 for the whole cohort -- submit that as a batch job.")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first N patients. With --n-patients this selects a "
                         "contiguous range, so a SLURM array can split the cohort across "
                         "tasks that each write their own CSV.")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--all-transitions", action="store_true",
                    help="Score EVERY consecutive transition (t0->t1, t1->t2, ...) "
                         "instead of only each patient's first. Needed because "
                         "`pairing: consecutive` trains on all of them, and the "
                         "existing full-cohort table covers 1,120 of 7,601. Emits "
                         "one row per transition; no before/after columns.")
    ap.add_argument("--decompose", action="store_true",
                    help="Split the observed t0->t1 change into a spatially uniform "
                         "component (normalization artifact) and structure (anatomy), "
                         "under both the current and a robust normalization.")
    ap.add_argument("--compare-norm", metavar="PATIENT_IDS", default=None,
                    help="Comma-separated patient ids: render the same slice's change map "
                         "under the current and two candidate normalizations, to "
                         "results/diagnostics/normalization/normalization_before_after.png")
    ap.add_argument("--show-normalized", metavar="PATIENT_ID", default=None,
                    help="Render the t0/t1 slices as the ENCODER sees them under each "
                         "normalization (not the difference maps), to "
                         "results/diagnostics/normalization/normalized_inputs_<id>.png")
    ap.add_argument("--render", metavar="PATIENT_ID", default=None,
                    help="Skip scoring; render this patient's t0/t1/difference slices "
                         "to results/diagnostics/registration/registration_qc_<id>.png, to check "
                         "metric against the eye.")
    ap.add_argument("--slices", metavar="I,J,K", default=None,
                    help="With --render: show these slice indices instead of "
                         "evenly spaced ones through the middle 60%% of the stack.")
    args = ap.parse_args()

    if args.show_normalized:
        show_normalized_inputs(
            args.show_normalized,
            figpath("normalization", f"normalized_inputs_{args.show_normalized}.png"))
        return

    if args.compare_norm:
        compare_normalizations(
            [p.strip() for p in args.compare_norm.split(",") if p.strip()],
            figpath("normalization", "normalization_before_after.png"))
        return

    if args.decompose:
        run_decomposition(args.n_patients)
        return

    if args.render:
        render_patient(args.render,
                       figpath("registration", f"registration_qc_{args.render}.png"),
                       slices=[int(x) for x in args.slices.split(",")]
                       if args.slices else None)
        return

    patient_ids = sorted(p for p in os.listdir(REG_ROOT) if (REG_ROOT / p).is_dir())
    total = len(patient_ids)
    patient_ids = patient_ids[args.offset:]
    if args.n_patients:
        patient_ids = patient_ids[:args.n_patients]
    print(f"Scoring {len(patient_ids)} patients from {REG_ROOT} "
          f"(offset {args.offset} of {total})")

    rows = []
    for i, patient_id in enumerate(patient_ids, 1):
        if args.all_transitions:
            got = evaluate_patient_transitions(patient_id)
            if not got:
                print(f"[{i}/{len(patient_ids)}] {patient_id}: skipped (needs 2+ FLAIR visits with brain)")
                continue
            rows.extend(got)
            worst = min(got, key=lambda r: r["dice_after"])
            print(f"[{i}/{len(patient_ids)}] {patient_id}: {len(got)} transitions | "
                  f"median ncc {np.nanmedian([r['ncc_after'] for r in got]):+.3f} | "
                  f"worst dice {worst['dice_after']:.3f} (t{worst['transition_index']})")
            continue

        row = evaluate_patient(patient_id)
        if row is None:
            print(f"[{i}/{len(patient_ids)}] {patient_id}: skipped (needs 2+ FLAIR visits with brain)")
            continue
        rows.append(row)
        print(f"[{i}/{len(patient_ids)}] {patient_id}: "
              f"ncc {row['ncc_before']:+.3f} -> {row['ncc_after']:+.3f} "
              f"| offset0 {row['frac_offset_zero']:5.1%}")

    if not rows:
        raise SystemExit("No patients could be scored -- check the tree paths.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (summarize_transitions if args.all_transitions else summarize)(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
