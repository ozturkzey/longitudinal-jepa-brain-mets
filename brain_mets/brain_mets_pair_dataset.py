"""
Paired (t0, t1) FLAIR-slice dataset for LeWM JEPA training on the
per-modality-registered Yale-Brain-Mets-Longitudinal tree.

One sample = one (patient, slice_index) transition between two registered FLAIR
visits (t0, t1) -- every slice along the volume's smallest-dimension axis is
used, not just the middle one, since ANTs registration resamples every visit
onto the EARLIEST visit's exact grid, so every slice index has a valid,
spatially-corresponding frame at every visit. (Audited 2026-09-02: 0 of 1,123
patients have any shape mismatch across their FLAIR visits.)

`pairing` chooses which transitions of a patient's timeline become samples:

  "first_two"   -- only (visit 0 -> visit 1), the original behaviour and still
                   the default, so no existing config changes meaning.
  "consecutive" -- every (visit k -> visit k+1).

Only 210 of 1,123 patients (19%) have exactly two FLAIR visits, so "first_two"
discards later visits for the other 81%: 36,138 pairs against ~241,700. The
reason to prefer "consecutive" is not the extra data though -- it is that
"first_two" gives each patient exactly ONE Δt, which confounds Δt with patient
identity. The model cannot then separate "a longer gap means more change" from
"this patient changes a lot". Consecutive pairing supplies several Δt values per
patient against their own anatomy, which is the only arrangement in which the
Δt -> change relationship is identifiable.

Duck-types stable_worldmodel's Dataset interface (__len__, __getitem__,
get_dim, get_col_data, .transform) without subclassing anything from
stable_worldmodel -- train.py instantiates this directly via Hydra's
_target_ (see config/train/data/brain_mets.yaml), bypassing
swm.data.load_dataset entirely (that path is HDF5/Lance/folder-format
specific and doesn't apply to a hand-rolled medical dataset).

Run under .venv (needs nibabel + openpyxl + torch).
"""

import datetime
import os

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from brain_mets.train_linear_probe import load_clinical_lookup

from brain_mets.slice_cache import build_slice_cache, load_slice_cache

from brain_mets.slice_preprocessing import (  # noqa: F401  (re-exported for existing importers)
    apply_window,
    brain_mask,
    center_crop_to_square,
    resize_square,
    slice_to_uint8_rgb,
    volume_window,
)

MODALITY = "FLAIR"

# The action/covariate vector is [dt_days, age]. ONE definition, imported by
# the sequence dataset and by every diagnostic that has to size the action
# encoder before its dataset exists. train.py derives it from
# dataset.get_dim("action") instead and needs no import.
#
# `sex` was the third dimension until 2026-08-27; it is corrupted for 35.4%
# of patients and was dropped rather than guessed at. When this
# number changes, delete cache/emb_*.pt -- they store the NORMALIZED action
# and will otherwise serve vectors of the old width without erroring.
ACTION_DIM = 2


class BrainMetsPairDataset(Dataset):
    def __init__(self, root_dir, clinical_excel, img_size=224, max_patients=None,
                 cache_path=None, drop_blank_slices=False,
                 normalization="minmax", pairing="first_two",
                 dt_transform="raw", **_ignored):
        # **_ignored swallows Hydra-instantiate-forwarded keys that are only
        # meaningful to train.py itself (e.g. keys_to_load), not to this class.
        # normalization: which intensity mapping the cached pixels use.
        #   "minmax"        -- per-slice min-max, the original. Reproduces every
        #                      run before 2026-08-27.
        #   "volume_window" -- per-volume median +/- 5*IQR of brain tissue,
        #                      clipped. Removes the ~30% of apparent
        #                      visit-to-visit change that per-slice min-max
        #                      manufactures.
        # Explicit, and defaulted to the historical value, so no existing config
        # silently changes meaning.
        if normalization not in ("minmax", "volume_window"):
            raise ValueError(f"unknown normalization {normalization!r}")
        # pairing: which transitions of a patient's visit timeline become samples.
        #   "first_two"   -- visit 0 -> visit 1 only. The original behaviour, and
        #                    the default so no existing config changes meaning.
        #   "consecutive" -- every visit k -> visit k+1 (see module docstring for
        #                    why: it decouples Δt from patient identity).
        # A cache built for one mode is valid for the other -- it is keyed by
        # (path, slice_index) -- but "consecutive" needs slices from EVERY visit,
        # so point cache_path at the all-visits cache, not the pair cache.
        if pairing not in ("first_two", "consecutive"):
            raise ValueError(f"unknown pairing {pairing!r}")
        # dt_transform: what happens to dt_days BEFORE the downstream z-score.
        #   "raw"   -- the value in days, the original behaviour and the default
        #              so no existing config changes meaning.
        #   "log1p" -- log(1 + dt_days).
        # Why this is not cosmetic: dt_days runs 1..1,982 with a median near 70,
        # so a z-score over the raw column is set by a handful of multi-year gaps
        # and squeezes the common few-month gaps -- the ones that carry almost
        # all the data -- into a narrow band near the low end. Under log1p the
        # spacing becomes multiplicative, which is how the quantity is actually
        # read clinically ("twice as long", not "300 days more"), and the
        # populated part of the range gets most of the axis.
        #
        # Applied HERE rather than in a transform because get_col_data() feeds
        # the z-score statistics: if the transform saw log values and the stats
        # were computed on raw days, the model would be handed a standardization
        # fitted to the wrong distribution, with nothing raising an error.
        if dt_transform not in ("raw", "log1p"):
            raise ValueError(f"unknown dt_transform {dt_transform!r}")
        self.dt_transform = dt_transform
        self.pairing = pairing
        self.normalization = normalization
        self.root_dir = root_dir
        self.img_size = img_size
        self.drop_blank_slices = drop_blank_slices
        self.clinical = load_clinical_lookup(clinical_excel)
        self.transform = None
        self.pairs = self._build_pairs(max_patients)

        # Eagerly decode+process every needed slice once, up front, and cache
        # it in memory -- this runs in the main process BEFORE DataLoader
        # forks worker processes, so every worker inherits the populated
        # cache for free via Linux fork()'s copy-on-write semantics. Without
        # this, the same NIfTI files get re-read/re-decoded from the
        # filesystem on every epoch (measured: ~1.67 min/epoch for the
        # single-slice version, dominating a real training run).
        needed = set()
        for p in self.pairs:
            needed.add((p["t0_path"], p["slice_index"]))
            needed.add((p["t1_path"], p["slice_index"]))

        # Optionally persist the processed cache to disk (cache_path) so
        # later runs don't redo this work at all. On load, only the keys
        # actually missing (e.g. newly-registered patients since the cache
        # was last saved) get (re)processed -- self-healing as the
        # registered dataset grows, no wasted repeat work on what's already
        # cached.
        # The cache stores PROCESSED pixels, so which normalization built it is
        # not recoverable by looking at them -- and serving min-max pixels to a
        # run configured for volume_window would be silently wrong, not an
        # error. Stamp it in the file and refuse a mismatch.
        # Load/build go through brain_mets/slice_cache.py, which is the one
        # definition of the cache format and is shared with the all-slices
        # dataset -- the cache is keyed by (path, slice_index), so the two
        # datasets can reuse each other's work.
        self._cache = load_slice_cache(cache_path, normalization)
        if self._cache:
            print(f"[BrainMetsPairDataset] loaded {len(self._cache)} cached slices "
                  f"from {cache_path} (normalization={normalization})")

        build_slice_cache(self._cache, needed, cache_path, normalization,
                          self.img_size, label="BrainMetsPairDataset")

        if drop_blank_slices:
            self._drop_blank_pairs()

    def _drop_blank_pairs(self):
        """Drop pairs where either frame is a fully blank slice (max == min).

        Filters `self.pairs` in memory only -- the on-disk cache is never
        modified (it takes over an hour to rebuild and BrainMetsSequenceDataset
        reads the same file).

        Two distinct cases, both of which have to go (measured 2026-08-26 on the
        113 held-out patients, 16.75% of pairs affected):

        - BOTH frames blank (~12.4% of pairs). t0 and t1 encode identically, so
          the "copy t0 forward" baseline scores an exact 0 and is unbeatable,
          while any real model scores worse. These teach nothing except "copy the
          input" and make the baseline look artificially strong.
        - ONE frame blank (~4.3%). The two volumes simply cover a different axial
          extent -- an acquisition/registration artifact, not biology, since a
          brain does not appear or vanish between visits. The baseline is
          catastrophically wrong on these (error 5.35 vs 2.93 on real slices), so
          almost anything beats it. Because train.py's metric is a ratio of
          SQUARED errors, these few huge-error pairs dominated the batch average
          and pushed the logged ratio below 1.0 while the model was actually
          losing on the 83% of pairs that are real brain. That is the specific
          artifact this filter removes.

        Under `pairing="consecutive"` the blank frame is dropped from the TIMELINE
        and the surrounding frames are re-paired across it, rather than both of its
        pairs being discarded. A blank frame only means that visit's volume did not
        cover this slice index; the visits either side of it are still valid
        observations of the same anatomy, and their gap is the sum of the two steps
        spanned. This matches BrainMetsSequenceDataset._drop_blank_frames, so the
        pair and sequence populations stay comparable. For `pairing="first_two"`
        the timeline is two frames long, bridging cannot apply, and the behaviour
        is identical to before this method was generalized.
        """
        before = len(self.pairs)
        self.pairs = self._pairs_from_timelines(blank_filter=True)
        n_pat = len({p["patient_id"] for p in self.pairs})
        print(f"[BrainMetsPairDataset] drop_blank_slices: {before:,} -> {len(self.pairs):,} pairs "
              f"({(before - len(self.pairs)) / max(before, 1) * 100:.1f}% dropped), "
              f"{n_pat} patients remain (pairing={self.pairing})")

    def _build_pairs(self, max_patients):
        """Build `self._timelines` -- one per (patient, slice_index) -- and derive
        the pair list from them.

        The timeline is kept as state rather than discarded because blank-frame
        filtering has to re-pair across dropped frames (see `_drop_blank_pairs`),
        which is impossible from a flat pair list.
        """
        self._timelines = []
        patient_ids = sorted(os.listdir(self.root_dir))
        if max_patients is not None:
            patient_ids = patient_ids[:max_patients]

        for patient_id in patient_ids:
            patient_path = os.path.join(self.root_dir, patient_id)
            if not os.path.isdir(patient_path) or patient_id not in self.clinical:
                continue

            visit_dates = sorted(
                v for v in os.listdir(patient_path)
                if os.path.isdir(os.path.join(patient_path, v))
                and os.path.exists(os.path.join(patient_path, v, f"{MODALITY}.nii.gz"))
            )
            if len(visit_dates) < 2:
                continue
            if self.pairing == "first_two":
                visit_dates = visit_dates[:2]

            paths = [os.path.join(patient_path, d, f"{MODALITY}.nii.gz")
                     for d in visit_dates]

            # Header-only shape reads (nibabel is lazy -- .shape doesn't touch
            # pixel data), one per VISIT rather than one per transition.
            # register_per_modality.py warps every later visit onto the EARLIEST
            # visit's grid, so all of a patient's volumes share a shape -- audited
            # 2026-09-02, 0 of 1,123 patients mismatch. min() is a defensive
            # guard, not the expected path.
            shapes = [nib.load(p).shape for p in paths]
            n_slices = min(s[int(np.argmin(s))] for s in shapes)

            dates = [datetime.date.fromisoformat(d) for d in visit_dates]
            # dt_steps[k] is the gap from visit k to visit k+1, so the gap
            # spanning visits a..b is sum(dt_steps[a:b]).
            dt_steps = [float((dates[k + 1] - dates[k]).days)
                        for k in range(len(dates) - 1)]
            age = self.clinical[patient_id]["age"]

            for slice_index in range(n_slices):
                self._timelines.append({
                    "patient_id": patient_id,
                    "paths": paths,
                    "dt_steps": dt_steps,
                    "slice_index": slice_index,
                    "age": age,
                })

        return self._pairs_from_timelines()

    @staticmethod
    def _pairs_of(tl, keep=None):
        """The consecutive pairs of one timeline, optionally over a subset of its
        frames.

        `keep` is an ascending list of frame indices; a pair that spans dropped
        frames carries the SUM of the steps it covers, so a two-visit gap is never
        mislabelled as a one-visit gap.
        """
        idx = range(len(tl["paths"])) if keep is None else keep
        idx = list(idx)
        return [{
            "patient_id": tl["patient_id"],
            "t0_path": tl["paths"][a],
            "t1_path": tl["paths"][b],
            "slice_index": tl["slice_index"],
            "dt_days": float(sum(tl["dt_steps"][a:b])),
            "age": tl["age"],
        } for a, b in zip(idx, idx[1:])]

    def _pairs_from_timelines(self, blank_filter=False):
        pairs = []
        for tl in self._timelines:
            if not blank_filter:
                pairs.extend(self._pairs_of(tl))
                continue
            keep = [t for t, p in enumerate(tl["paths"])
                    if not self._is_blank(p, tl["slice_index"])]
            if len(keep) < 2:
                continue
            pairs.extend(self._pairs_of(tl, keep))
        return pairs

    def _is_blank(self, path, slice_index):
        a = self._cache[(path, slice_index)]
        return a.max() == a.min()

    def __len__(self):
        return len(self.pairs)

    def _load_slice(self, path, slice_index):
        return self._cache[(path, slice_index)]

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pixels = torch.stack([
            self._load_slice(p["t0_path"], p["slice_index"]),
            self._load_slice(p["t1_path"], p["slice_index"]),
        ])  # (2,3,H,W)
        action_vec = torch.tensor(self._action_vec(p), dtype=torch.float32)
        action = action_vec.unsqueeze(0).repeat(2, 1)  # (2,2) -- same covariates at both steps
        sample = {"pixels": pixels, "action": action}
        return self.transform(sample) if self.transform else sample

    @property
    def patient_ids(self):
        """One patient_id per pair, aligned with self.pairs -- used by
        train.py to do a patient-grouped train/val split instead of a plain
        random split, so a patient's slices never end up split across both"""
        return [p["patient_id"] for p in self.pairs]

    def get_dim(self, col):
        assert col == "action", f"BrainMetsPairDataset only knows the 'action' column, got {col!r}"
        return ACTION_DIM

    def _action_vec(self, p):
        """The covariate vector for one pair: [dt, age], dt per `dt_transform`.

        THE ONLY place the action is constructed. __getitem__ serves it to the
        model and get_col_data() supplies the z-score statistics fitted to it;
        those two disagreeing is a silent failure -- the model would receive
        values standardized against a distribution they do not come from -- so
        they share this method rather than each building the vector.
        """
        dt = p["dt_days"]
        if self.dt_transform == "log1p":
            dt = float(np.log1p(dt))
        return [dt, p["age"]]

    def get_col_data(self, col):
        assert col == "action", f"BrainMetsPairDataset only knows the 'action' column, got {col!r}"
        return np.array([self._action_vec(p) for p in self.pairs], dtype=np.float32)
