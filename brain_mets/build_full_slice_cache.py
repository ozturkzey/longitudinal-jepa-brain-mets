"""
One-time (or incremental re-run) build of the full, uncapped all-slices
BrainMetsPairDataset disk cache -- all patients, every slice.
Already-cached slices are reused, not reprocessed -- see
BrainMetsPairDataset's incremental-fill logic.

--normalization picks the intensity mapping AND, deliberately, a different
output file per mapping. They must never share a path: the cache stores
PROCESSED pixels, so which mapping built it cannot be recovered by looking at
them. (The dataset also stamps the mapping into the file and refuses a
mismatch, but two separate files is the first line of defence -- and it keeps
the min-max cache intact as a rollback.)

Usage (from repo root):
    ./.venv/bin/python brain_mets/build_full_slice_cache.py
    ./.venv/bin/python brain_mets/build_full_slice_cache.py --normalization volume_window
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain_mets.brain_mets_pair_dataset import BrainMetsPairDataset

ROOT_DIR = os.environ["BRAINJEPA_REGISTERED"]
CLINICAL_EXCEL = "Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx"
CACHE_PATHS = {
    "minmax": "cache/brain_mets_slice_cache.pt",
    "volume_window": "cache/brain_mets_slice_cache_volwin.pt",
}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--normalization", choices=sorted(CACHE_PATHS), default="minmax")
args = ap.parse_args()
cache_path = CACHE_PATHS[args.normalization]
print(f"Building {args.normalization} cache -> {cache_path}")

dataset = BrainMetsPairDataset(root_dir=ROOT_DIR, clinical_excel=CLINICAL_EXCEL,
                                cache_path=cache_path,
                                normalization=args.normalization)
print(f"Full cache build done: {len(dataset)} pairs across {len(set(dataset.patient_ids))} patients, "
      f"{len(dataset._cache)} cached slices")
