"""Every FLAIR slice of every registered visit, as independent samples.

One sample = ONE slice. Not a pair, not a sequence.

Why this exists. `BrainMetsPairDataset` takes each patient's two earliest
visits, because the PREDICTOR needs a (t0, t1) pair. A decoder or autoencoder
needs neither the pairing nor the registration -- it reconstructs each image on
its own -- and inheriting those rules costs most of the data: 2,246 of the
8,724 registered FLAIR volumes, i.e. 26%.

Pairing consecutive visits (1-2, 2-3, 3-4, ...) would reach every volume, but
every middle visit would appear TWICE -- once as t0, once as t1 -- silently
weighting middle visits 2x against first and last. This class avoids that:
each (volume, slice_index) appears exactly once.

Shape. `pixels` is `(1, 3, img_size, img_size)` -- a length-1 time axis, kept
so the tensor layout matches what `JEPA.encode` and `train.py` expect of every
other dataset here. `action` is a zero vector of width `ACTION_DIM`: this
dataset has no Δt (there is no second visit in a sample) and the covariates go
only to the action encoder, which no reconstruction arm trains. It is present
so the batch shape is uniform.

The length-1 time axis means `train.py`'s `ctx_len = T - n_preds` is 0, so there
is no (context, target) pair to predict from. `lejepa_forward` guards on that and
omits `pred_loss`/`baseline_loss` entirely. Note `loss.pred.weight: 0.0` does NOT
on its own make this safe -- an empty tensor's `.pow(2).mean()` is NaN and
`0.0 * NaN` is still NaN, so before that guard existed this dataset produced a
NaN total loss from the first step, silently. Arms using it should still set
`loss.pred.weight: 0.0` to document that they train nothing but the decoder.

The `action` column is all zeros, i.e. zero-variance, which breaks
`get_column_normalizer` (it divides by `data.std(0)`). Configs using this dataset
must therefore set `keys_to_load: [pixels]` so no normalizer is built for it.

Shares `cache/`'s format with the pair dataset (keyed by `(path, slice_index)`,
see brain_mets/slice_cache.py), so the two can reuse each other's cached work
as long as they agree on `normalization` and `img_size`.

Run under .venv (needs nibabel + openpyxl + torch).
"""

import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from brain_mets.brain_mets_pair_dataset import ACTION_DIM, MODALITY
from brain_mets.slice_cache import build_slice_cache, load_slice_cache
from brain_mets.train_linear_probe import load_clinical_lookup


class BrainMetsSliceDataset(Dataset):
    def __init__(self, root_dir, clinical_excel, img_size=224, max_patients=None,
                 cache_path=None, drop_blank_slices=False,
                 normalization="minmax", require_clinical=False,
                 save_every=None, **_ignored):
        if normalization not in ("minmax", "volume_window"):
            raise ValueError(f"unknown normalization {normalization!r}")
        self.root_dir = root_dir
        self.img_size = img_size
        self.normalization = normalization
        self.transform = None
        # Reconstruction needs no labels, so a patient missing from the
        # clinical sheet is still perfectly usable imaging. The pair dataset
        # requires it because it needs `age` for the action vector.
        self.require_clinical = require_clinical
        self.clinical = load_clinical_lookup(clinical_excel) if require_clinical else {}

        self.samples = self._build_samples(max_patients)
        needed = {(s["path"], s["slice_index"]) for s in self.samples}

        self._cache = load_slice_cache(cache_path, normalization)
        if self._cache:
            print(f"[BrainMetsSliceDataset] loaded {len(self._cache):,} cached slices "
                  f"from {cache_path} (normalization={normalization})")
        build_slice_cache(self._cache, needed, cache_path, normalization,
                          self.img_size, save_every=save_every,
                          label="BrainMetsSliceDataset")

        if drop_blank_slices:
            self._drop_blank_slices()

    def _build_samples(self, max_patients):
        """Every (volume, slice_index) in the tree, each exactly once."""
        samples = []
        patient_ids = sorted(os.listdir(self.root_dir))
        if max_patients is not None:
            patient_ids = patient_ids[:max_patients]

        for patient_id in patient_ids:
            patient_path = os.path.join(self.root_dir, patient_id)
            if not os.path.isdir(patient_path):
                continue
            if self.require_clinical and patient_id not in self.clinical:
                continue
            for visit_date in sorted(os.listdir(patient_path)):
                path = os.path.join(patient_path, visit_date, f"{MODALITY}.nii.gz")
                if not os.path.exists(path):
                    continue
                # Header-only read: nibabel is lazy, .shape does not touch
                # pixel data, so this is cheap even across ~8,700 volumes.
                shape = nib.load(path).shape
                n_slices = shape[int(np.argmin(shape))]
                for slice_index in range(n_slices):
                    samples.append({"patient_id": patient_id,
                                    "visit_date": visit_date,
                                    "path": path,
                                    "slice_index": slice_index})
        return samples

    def _drop_blank_slices(self):
        """Drop slices that are exactly flat (max == min) -- volume edges.

        In-memory only; the on-disk cache is never modified. Any embedding
        reconstructs a black rectangle perfectly, so keeping them flatters the
        reported error without saying anything about brain reconstruction.
        """
        before = len(self.samples)
        self.samples = [s for s in self.samples
                        if self._cache[(s["path"], s["slice_index"])].max()
                        > self._cache[(s["path"], s["slice_index"])].min()]
        pct = 100 * (before - len(self.samples)) / max(before, 1)
        print(f"[BrainMetsSliceDataset] drop_blank_slices: {before:,} -> "
              f"{len(self.samples):,} slices ({pct:.1f}% dropped), "
              f"{len(set(self.patient_ids))} patients remain")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pixels = self._cache[(s["path"], s["slice_index"])].unsqueeze(0)  # (1,3,H,W)
        action = torch.zeros(1, ACTION_DIM, dtype=torch.float32)
        sample = {"pixels": pixels, "action": action}
        return self.transform(sample) if self.transform else sample

    @property
    def patient_ids(self):
        """One patient_id per sample -- train.py's patient-grouped split reads
        this, so no patient can land on both sides."""
        return [s["patient_id"] for s in self.samples]

    def get_dim(self, key):
        return {"action": ACTION_DIM}[key]

    def get_col_data(self, key):
        if key != "action":
            raise KeyError(key)
        return torch.zeros(len(self.samples), ACTION_DIM, dtype=torch.float32)
