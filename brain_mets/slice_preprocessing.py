"""Per-slice image preprocessing shared by the brain-mets datasets.

Moved out of brain_mets_pair_dataset.py (2026-08-27) so that code needing only
the pixel pipeline does not have to import that module's clinical-data chain
(train_linear_probe -> openpyxl), which is absent from the antspyx `venv` that
registration_qc.py runs under. Same reasoning as scripts/diagnostics/masking.py.

These run in order -- crop to square, map intensities to uint8 RGB, resize --
and that order is what the cached slices were built with. Changing any of them
invalidates the 41.9 GB slice cache.

TWO intensity mappings live here, and which one a cache was built with is not
recoverable from the cache file, so it must be recorded alongside it:

  slice_to_uint8_rgb        per-SLICE min-max. The original. Its window is set
                            by that slice's brightest voxel, which on FLAIR is
                            usually a lesion -- so when a lesion fades between
                            visits the whole slice is remapped and a LOCAL
                            change becomes a GLOBAL one. Measured 2026-08-27:
                            this accounts for ~30% of all apparent
                            visit-to-visit change.
  volume_window / apply_window
                            per-VOLUME median +/- 5*IQR of brain tissue, then
                            clipped. Anchored on normal tissue, so a lesion
                            cannot move it.

Per-VOLUME, not per-patient, is deliberate. Sharing one window across a
patient's visits would leave the scanner-to-scanner intensity difference in --
which is the thing being removed. Normalizing each scan by its own tissue
statistics is what puts normal tissue at the same grey level in every scan, and
that is what makes two visits comparable.
"""

import numpy as np
from PIL import Image

# Display-window half-width in IQRs for the per-volume mapping. Chosen by
# scripts/diagnostics/data/clipping_sweep.py over 40 volumes: k=5 clips 0.3% of
# lesion-proxy voxels, where the textbook k=2 and k=3 clip 100% and 77% of
# them -- silently, with the images still looking correct.
IQR_HALF_WIDTH = 5.0


def center_crop_to_square(sl):
    """Native slice size/aspect ratio varies per patient (confirmed: e.g.
    256x256 vs 408x512). utils.get_img_preprocessor's Resize(img_size) only
    resizes the SHORT edge to img_size (torchvision aspect-preserving
    behavior) -- on a non-square input that leaves the long edge != img_size,
    which breaks batch collation across patients with different aspect
    ratios. Center-crop to square first so the downstream resize always
    lands on a uniform (img_size, img_size)."""
    h, w = sl.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return sl[top:top + side, left:left + side]


def _to_uint8_rgb(sl01):
    """A [0,1] float slice -> (H,W,3) uint8, grey replicated to 3 channels.

    The clip is not cosmetic. `.astype(np.uint8)` WRAPS on out-of-range input
    (300 becomes 44), it does not saturate -- so an unclipped bright lesion
    would come out BLACK. min-max never hit this because it guarantees [0,1] by
    construction; any other mapping must clip explicitly."""
    sl_uint8 = (np.clip(sl01, 0.0, 1.0) * 255).astype(np.uint8)
    return np.stack([sl_uint8, sl_uint8, sl_uint8], axis=-1)  # (H,W,3)


def slice_to_uint8_rgb(sl):
    """Per-SLICE min-max to [0,255] uint8 -- the ORIGINAL mapping, matching
    src/encoders/lewm.py's encode_one exactly.

    Kept so existing caches and the historical encoder comparison stay
    reproducible. For new work prefer volume_window + apply_window; see this
    module's docstring for why."""
    lo, hi = sl.min(), sl.max()
    sl = (sl - lo) / (hi - lo) if hi > lo else np.zeros_like(sl)
    return _to_uint8_rgb(sl)


def volume_window(volume, mask):
    """(lo, hi) display window for one volume: median +/- IQR_HALF_WIDTH*IQR of
    the voxels under `mask`.

    `mask` must be a BRAIN mask (scripts/diagnostics/masking.brain_mask). Over
    the whole volume the background zeros dominate, which would drag the median
    to 0 and make the window meaningless -- the statistic only describes normal
    tissue if it is computed on tissue.

    Computed ONCE per volume and reused for all its slices: that is what stops a
    lesion in one slice from re-lighting the rest of the scan.
    """
    vals = volume[mask]
    if vals.size == 0:
        raise ValueError("volume_window: empty brain mask -- refusing to guess a window")
    med = float(np.median(vals))
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25)) or 1.0

    # The lower bound is clamped at 0, not left at med - k*IQR. Measured on
    # real volumes: med - 5*IQR lands anywhere from -317 to +34 depending on the
    # patient, and where it is negative the background (value 0) renders as
    # GREY rather than black -- at a DIFFERENT grey per patient. That is a new
    # per-scan inconsistency, precisely the thing this window exists to remove.
    # FLAIR magnitude images are non-negative (verified: 0/7 sampled volumes
    # contain a negative voxel), so clamping costs no real data and it keeps
    # background at exactly 0 for every scan while giving the full range to
    # tissue.
    return max(0.0, med - IQR_HALF_WIDTH * iqr), med + IQR_HALF_WIDTH * iqr


def apply_window(sl, lo, hi):
    """Map one slice through a precomputed (lo, hi) window -> uint8 RGB.

    Every slice of a volume gets the SAME window, so unchanged tissue keeps the
    same grey level across slices and, more importantly, across visits."""
    span = (hi - lo) or 1.0
    return _to_uint8_rgb((sl - lo) / span)


def resize_square(rgb_uint8, img_size):
    """Resize an already-square uint8 RGB array to (img_size, img_size).

    Baking this into the cache-build step (instead of leaving it to the
    per-epoch utils.get_img_preprocessor transform) matters for two reasons:
    it fixes the cache's real memory footprint (native crop resolution runs
    up to 640x640 -- 2.81x more bytes/slice than 224x224, measured), and it
    removes a repeated per-epoch op. Reordering resize before the later
    ImageNet normalization is numerically safe: normalization is affine and
    bilinear interpolation is a linear combination of neighboring pixels, so
    the two operations commute.
    """
    return np.array(Image.fromarray(rgb_uint8).resize((img_size, img_size), Image.BILINEAR))


def otsu_threshold(values, n_bins=256):
    """Standard Otsu: the threshold maximizing between-class variance."""
    hist, edges = np.histogram(values, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_lo = np.cumsum(hist)
    weight_hi = hist.sum() - weight_lo
    valid = (weight_lo > 0) & (weight_hi > 0)

    cum_mean = np.cumsum(hist * centers)
    mean_lo = np.divide(cum_mean, weight_lo, out=np.zeros_like(cum_mean), where=weight_lo > 0)
    total = cum_mean[-1]
    mean_hi = np.divide(total - cum_mean, weight_hi, out=np.zeros_like(cum_mean), where=weight_hi > 0)

    between = weight_lo * weight_hi * (mean_lo - mean_hi) ** 2
    between[~valid] = -1
    return centers[int(np.argmax(between))]


def brain_mask(volume):
    """Otsu on the volume's nonzero voxels, or None if there is no brain.

    No connected-component filtering: callers that need the largest component
    (brain_size_probe, measuring physical size) apply it themselves. For
    correlation masking it would make each of two masks depend on the other's
    topology, which is not what a mask is for.
    """
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return None
    mask = volume > otsu_threshold(nonzero)
    return mask if mask.any() else None
