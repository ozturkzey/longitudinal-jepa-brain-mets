"""Brain-masking primitives -- re-exported from brain_mets.slice_preprocessing.

The implementations moved into the package (2026-08-27) so that the datasets can
compute a per-volume intensity window without brain_mets importing from
scripts/, which would invert the dependency direction. This module stays as the
import path the diagnostics already use.

Still safe from every venv: slice_preprocessing depends only on numpy + PIL.
"""

from brain_mets.slice_preprocessing import brain_mask, otsu_threshold  # noqa: F401

__all__ = ["brain_mask", "otsu_threshold"]
