"""One definition of how a FLAIR slice cache is built and persisted.

The cache is a flat dict mapping `(nifti_path, slice_index)` to a processed
`(3, img_size, img_size)` uint8 tensor, plus a `("__meta__", "normalization")`
stamp. Nothing about it is pair-specific or sequence-specific: it is keyed by
the slice itself, so every dataset over this tree shares the same format and
can reuse each other's work.

Extracted from `BrainMetsPairDataset.__init__` (2026-08-31) when a second
dataset needed the identical logic. The behaviour is unchanged; only the
location moved, plus the incremental saving described below.

Two things here are load-bearing and easy to get wrong:

* **Decode each volume once.** Grouping the requested keys by file and pulling
  every needed slice out of one `get_fdata()` is ~32x faster than a per-slice
  loader, which would re-decode the same NIfTI once per slice.
* **One intensity window per VOLUME, not per slice.** That is the whole point
  of `volume_window`: a per-slice window lets a bright lesion re-scale every
  other voxel in its slice, so a local change becomes a global one before the
  encoder ever sees it. The window is computed from an Otsu brain mask, before
  the slice loop, and a volume with no findable brain raises rather than
  silently falling back to a different mapping.
"""

import os

import nibabel as nib
import numpy as np
import torch

from brain_mets.slice_preprocessing import (apply_window, brain_mask,
                                            center_crop_to_square,
                                            resize_square, slice_to_uint8_rgb,
                                            volume_window)

META_KEY = ("__meta__", "normalization")


def load_slice_cache(cache_path, normalization):
    """Return the on-disk cache, or {} if there is none.

    Refuses a cache built with a different intensity mapping: the file stores
    PROCESSED pixels, so which mapping produced them cannot be recovered by
    looking at them, and serving min-max pixels to a volume_window run would be
    silently wrong rather than an error.
    """
    if cache_path is None or not os.path.exists(cache_path):
        return {}
    cache = torch.load(cache_path)
    cached_norm = cache.pop(META_KEY, "minmax")  # pre-stamp caches are min-max
    if cached_norm != normalization:
        raise ValueError(
            f"slice cache at {cache_path} was built with "
            f"normalization={cached_norm!r} but this run asks for "
            f"{normalization!r}. Build a separate cache file -- reusing this one "
            f"would silently feed the wrong pixels.")
    return cache


def save_slice_cache(cache, cache_path, normalization):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache[META_KEY] = normalization
    try:
        torch.save(cache, cache_path)
    finally:
        del cache[META_KEY]


def build_slice_cache(cache, needed, cache_path, normalization, img_size,
                      save_every=None, label="slice cache"):
    """Fill `cache` with every key in `needed` that it does not already hold.

    Incremental in two senses: only missing keys are processed, and with
    `save_every` set the partial cache is written to disk every that-many
    volumes. The second matters for the all-visits build -- 8,724 volumes is
    hours of CPU, and a single save at the end means a failure near the end
    loses all of it. A partial file is a perfectly valid input to a later run,
    which simply picks up the remaining keys.
    """
    missing = needed - cache.keys()
    if not missing:
        return cache

    path_to_indices = {}
    for path, slice_index in missing:
        path_to_indices.setdefault(path, set()).add(slice_index)
    n_vol = len(path_to_indices)
    print(f"[{label}] processing {len(missing):,} new/missing slices "
          f"from {n_vol:,} volumes...", flush=True)

    for i, (path, slice_indices) in enumerate(path_to_indices.items(), start=1):
        volume = nib.load(path).get_fdata().astype(np.float32)
        axis = int(np.argmin(volume.shape))
        window = None
        if normalization == "volume_window":
            mask = brain_mask(volume)
            if mask is None:
                raise ValueError(f"no brain found in {path} -- cannot window it")
            window = volume_window(volume, mask)
        for slice_index in slice_indices:
            sl = np.take(volume, slice_index, axis=axis)
            sl = center_crop_to_square(sl)
            rgb = (slice_to_uint8_rgb(sl) if window is None
                   else apply_window(sl, *window))            # (H,W,3) uint8
            rgb = resize_square(rgb, img_size)
            cache[(path, slice_index)] = torch.from_numpy(rgb).permute(2, 0, 1)

        if save_every and cache_path is not None and i % save_every == 0:
            save_slice_cache(cache, cache_path, normalization)
            print(f"[{label}] {i:,}/{n_vol:,} volumes | "
                  f"{len(cache):,} slices cached (checkpointed)", flush=True)

    if cache_path is not None:
        save_slice_cache(cache, cache_path, normalization)
        print(f"[{label}] saved {len(cache):,} cached slices to {cache_path}",
              flush=True)
    return cache
