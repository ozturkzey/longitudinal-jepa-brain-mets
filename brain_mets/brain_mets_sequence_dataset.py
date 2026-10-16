"""
Variable-length (all-history -> last-visit) FLAIR-slice sequence dataset for
LeWM JEPA training on the per-modality-registered Yale-Brain-Mets-Longitudinal
tree.

One sample = one (patient, slice_index) sequence spanning ALL of that
patient's registered FLAIR visits: every visit except the last is context,
the last visit is the prediction target. Registration aligns every visit
onto the patient's FIRST visit's grid (see register_per_modality.py's
register_modality_timeline -- fixed_date = timeline[0]), so slice_index
correspondence holds across a patient's FULL timeline, not just two visits
-- verified directly (2026-08-25) against real data: 0/50 checked patients
(including the 20 with the most visits) had any shape inconsistency across
their full registered timeline.

Sequence length varies per patient (real measured range: 2-51 visits,
median 5). Batches must still be internally uniform-length (tensors need a
consistent shape to stack) -- see visit_length_sampler.py's
VisitLengthBatchSampler, used instead of adding padding/attention-masking to
lewm_core's shared Attention class.

Same duck-typed stable_worldmodel Dataset interface as BrainMetsPairDataset.
Run under .venv (needs nibabel + openpyxl + torch).
"""

import datetime
import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from brain_mets.brain_mets_pair_dataset import ACTION_DIM
from brain_mets.slice_preprocessing import (
    apply_window,
    brain_mask,
    center_crop_to_square,
    resize_square,
    slice_to_uint8_rgb,
    volume_window,
)
from brain_mets.train_linear_probe import load_clinical_lookup

MODALITY = "FLAIR"


class VisitLengthBatchSampler(Sampler):
    """Groups samples by sequence length so every batch handed to the model
    has a uniform T (frame count) -- an alternative to padding +
    attention-masking for BrainMetsSequenceDataset's variable-length
    (all-history -> last-visit) sequences. Keeps lewm_core's shared
    Attention class untouched (it only supports plain causal masking, no
    padding mask).

    Trade-off, accepted deliberately: batches are grouped, not perfectly
    uniform in size -- a patient with a rare sequence length may end up in a
    small final batch for that group. No sample is ever padded or dropped
    for shape reasons; drop_last (when requested) only ever drops a
    genuinely-incomplete last batch within a length group, same semantics
    as the ordinary case."""

    def __init__(self, lengths, batch_size, shuffle=True, drop_last=False, generator=None,
                 max_frames_per_batch=None):
        """max_frames_per_batch: cap on `batch_size * T` per batch. Without it, a
        fixed batch_size means memory scales with sequence length and the longest
        groups blow up -- job 3180372 was OOM-killed at 93 GB before its first
        epoch because batch_size=128 (inherited from the T=2 pair config) means
        154 MB/batch at T=2 but 3.9 GB/batch at T=51, times workers x prefetch.

        With a budget, per-group batch size becomes `max_frames // T`, so every
        batch moves roughly the same number of frames and memory is flat across
        groups. Setting it to 2 * the pair-config batch_size reproduces the pair
        behaviour exactly at T=2 and shrinks batches only where they would
        otherwise be huge.
        """
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.max_frames_per_batch = max_frames_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        self._groups = {}
        for idx, length in enumerate(self.lengths):
            self._groups.setdefault(length, []).append(idx)

    def _batch_size_for(self, length):
        if self.max_frames_per_batch is None:
            return self.batch_size
        return max(1, min(self.batch_size, self.max_frames_per_batch // max(int(length), 1)))

    def __iter__(self):
        groups = {length: list(idxs) for length, idxs in self._groups.items()}

        if self.shuffle:
            for length, idxs in groups.items():
                perm = torch.randperm(len(idxs), generator=self.generator).tolist()
                groups[length] = [idxs[i] for i in perm]

        batches = []
        for length, idxs in groups.items():
            bs = self._batch_size_for(length)
            for i in range(0, len(idxs), bs):
                batch = idxs[i:i + bs]
                if self.drop_last and len(batch) < bs:
                    continue
                batches.append(batch)

        if self.shuffle:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[i] for i in order]

        yield from batches

    def __len__(self):
        total = 0
        for length, idxs in self._groups.items():
            bs = self._batch_size_for(length)
            total += len(idxs) // bs if self.drop_last else -(-len(idxs) // bs)
        return total


class BrainMetsSequenceDataset(Dataset):
    def __init__(self, root_dir, clinical_excel, img_size=224, max_patients=None,
                 cache_path=None, min_visits=2, drop_blank_slices=False,
                 normalization="minmax", **_ignored):
        # NOTE drop_blank_slices is an EXPLICIT parameter, not left to **_ignored.
        # The pair dataset gained the same flag on 2026-08-26; if this class only
        # had **_ignored, setting it in a data config would be silently swallowed
        # and you would get an unfiltered run that looks correctly configured.
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
        self.normalization = normalization
        self.root_dir = root_dir
        self.img_size = img_size
        self.clinical = load_clinical_lookup(clinical_excel)
        self.transform = None
        self.sequences = self._build_sequences(max_patients, min_visits)
        self.min_visits = min_visits
        self._drop_blank = drop_blank_slices

        needed = set()
        for s in self.sequences:
            for path in s["paths"]:
                needed.add((path, s["slice_index"]))

        # The cache stores PROCESSED pixels, so which normalization built it is
        # not recoverable by looking at them -- and serving min-max pixels to a
        # run configured for volume_window would be silently wrong, not an
        # error. Stamp it in the file and refuse a mismatch.
        meta_key = ("__meta__", "normalization")
        if cache_path is not None and os.path.exists(cache_path):
            self._cache = torch.load(cache_path)
            cached_norm = self._cache.pop(meta_key, "minmax")  # pre-stamp caches are min-max
            if cached_norm != normalization:
                raise ValueError(
                    f"BrainMetsSequenceDataset: cache at {cache_path} was built with "
                    f"normalization={cached_norm!r} but this run asks for "
                    f"{normalization!r}. Build a separate cache file -- reusing this "
                    f"one would silently feed the wrong pixels.")
            print(f"[BrainMetsSequenceDataset] loaded {len(self._cache)} cached slices "
                  f"from {cache_path} (normalization={cached_norm})")
        else:
            self._cache = {}

        missing = needed - self._cache.keys()
        if missing:
            print(f"[BrainMetsSequenceDataset] processing {len(missing)} new/missing slices...")
            path_to_indices = {}
            for path, slice_index in missing:
                path_to_indices.setdefault(path, set()).add(slice_index)

            for path, slice_indices in path_to_indices.items():
                volume = nib.load(path).get_fdata().astype(np.float32)
                axis = int(np.argmin(volume.shape))
                # One window per VOLUME, computed before the slice loop -- that
                # is what stops a lesion in one slice from re-scaling the rest
                # of the scan. Needs a BRAIN mask: over the whole volume the
                # background zeros would drag the median to 0.
                window = None
                if self.normalization == "volume_window":
                    mask = brain_mask(volume)
                    if mask is None:
                        raise ValueError(f"no brain found in {path} -- cannot window it")
                    window = volume_window(volume, mask)
                for slice_index in slice_indices:
                    sl = np.take(volume, slice_index, axis=axis)
                    sl = center_crop_to_square(sl)
                    rgb = (slice_to_uint8_rgb(sl) if window is None
                           else apply_window(sl, *window))
                    rgb = resize_square(rgb, self.img_size)
                    self._cache[(path, slice_index)] = torch.from_numpy(rgb).permute(2, 0, 1)

            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                self._cache[meta_key] = normalization
                torch.save(self._cache, cache_path)
                del self._cache[meta_key]
                print(f"[BrainMetsSequenceDataset] saved {len(self._cache)} cached slices to {cache_path}")

        if self._drop_blank:
            self._drop_blank_frames()

    def _build_sequences(self, max_patients, min_visits):
        sequences = []
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
            if len(visit_dates) < min_visits:
                continue

            paths = [os.path.join(patient_path, d, f"{MODALITY}.nii.gz") for d in visit_dates]

            # Step-wise Δt: time since the PREVIOUS visit in this sequence,
            # not cumulative from the first one -- the first (earliest)
            # context frame has no previous frame, dt=0.0 for that step
            # (same "no meaningful transition at a sequence boundary"
            # convention train.py already applies elsewhere via nan_to_num).
            dt_days = [0.0]
            for i in range(1, len(visit_dates)):
                delta = (
                    datetime.date.fromisoformat(visit_dates[i])
                    - datetime.date.fromisoformat(visit_dates[i - 1])
                ).days
                dt_days.append(float(delta))

            # Header-only shape reads (nibabel is lazy), min() across the
            # FULL timeline generalizes the pair-dataset's two-visit min().
            shapes = [nib.load(p).shape for p in paths]
            axes = [int(np.argmin(s)) for s in shapes]
            n_slices = min(s[a] for s, a in zip(shapes, axes))

            age = self.clinical[patient_id]["age"]

            for slice_index in range(n_slices):
                sequences.append({
                    "patient_id": patient_id,
                    "paths": paths,
                    "slice_index": slice_index,
                    "dt_days": dt_days,
                    "age": age,
                    "sequence_length": len(visit_dates),
                })
        return sequences

    def _drop_blank_frames(self):
        """Remove fully blank frames (max == min) from each sequence, keeping the
        rest of that patient's timeline.

        TRIMS frames rather than dropping whole sequences. Dropping any sequence
        containing a blank would discard ~61% of them (median length 5, ~17% of
        frames blank) and would remove essentially every long sequence, which is
        the exact data the all-history dataset exists to use. A blank frame only
        means that visit's volume did not cover this slice index; the patient's
        other visits at this index are still valid observations.

        Step-wise Δt is recomputed across the removed frames: dt_days[i] is the gap
        from the previous kept frame, so skipping frame k means the next kept gap is
        the SUM of the gaps spanned. Not doing this would silently mislabel a
        two-visit gap as a one-visit gap and corrupt the covariate the model
        conditions on.

        Sequences left with fewer than min_visits frames are dropped entirely.
        The on-disk cache is never modified.
        """
        before_seq = len(self.sequences)
        before_frames = sum(len(s["paths"]) for s in self.sequences)
        kept = []
        for s in self.sequences:
            keep = [t for t, p in enumerate(s["paths"])
                    if (lambda a: a.max() != a.min())(self._cache[(p, s["slice_index"])])]
            if len(keep) < self.min_visits:
                continue
            dt = [0.0] + [float(sum(s["dt_days"][a + 1:b + 1])) for a, b in zip(keep, keep[1:])]
            kept.append({**s,
                         "paths": [s["paths"][t] for t in keep],
                         "dt_days": dt,
                         "sequence_length": len(keep)})
        self.sequences = kept
        after_frames = sum(len(s["paths"]) for s in kept)
        n_pat = len({s["patient_id"] for s in kept})
        print(f"[BrainMetsSequenceDataset] drop_blank_slices: "
              f"{before_seq:,} -> {len(kept):,} sequences "
              f"({(before_seq - len(kept)) / max(before_seq, 1) * 100:.1f}% dropped), "
              f"{before_frames:,} -> {after_frames:,} frames "
              f"({(before_frames - after_frames) / max(before_frames, 1) * 100:.1f}% trimmed), "
              f"{n_pat} patients remain")

    def __len__(self):
        return len(self.sequences)

    def _load_slice(self, path, slice_index):
        return self._cache[(path, slice_index)]

    def __getitem__(self, idx):
        s = self.sequences[idx]
        pixels = torch.stack([
            self._load_slice(p, s["slice_index"]) for p in s["paths"]
        ])  # (T,3,H,W)
        T = len(s["paths"])
        action = torch.tensor(
            [[s["dt_days"][t], s["age"]] for t in range(T)],
            dtype=torch.float32,
        )  # (T,2) -- each row is THAT frame's own step-wise covariates
        sample = {"pixels": pixels, "action": action}
        return self.transform(sample) if self.transform else sample

    @property
    def patient_ids(self):
        return [s["patient_id"] for s in self.sequences]

    @property
    def sequence_lengths(self):
        """Aligned with self.sequences -- used by visit_length_sampler.py to
        group same-length sequences into uniform-shape batches."""
        return [s["sequence_length"] for s in self.sequences]

    def get_dim(self, col):
        assert col == "action", f"BrainMetsSequenceDataset only knows the 'action' column, got {col!r}"
        return ACTION_DIM

    def get_col_data(self, col):
        """Flattened across every frame of every sequence (not one row per
        sequence) so the z-score normalizer sees the real distribution of
        per-step Δt values, not just first-to-last-visit gaps."""
        assert col == "action", f"BrainMetsSequenceDataset only knows the 'action' column, got {col!r}"
        rows = []
        for s in self.sequences:
            for t in range(s["sequence_length"]):
                rows.append([s["dt_days"][t], s["age"]])
        return np.array(rows, dtype=np.float32)
