"""Build the all-visits FLAIR slice cache: every slice of every registered volume.

The existing `build_full_slice_cache.py` covers only the PAIR population --
each patient's two earliest visits, 2,246 of the 8,724 registered FLAIR
volumes (26%). That restriction exists for the predictor and is useless to a
reconstruction arm, which reconstructs each slice independently.

Writes to a SEPARATE file from the pair cache on purpose. The pair cache
(`brain_mets_slice_cache_volwin.pt`, 10.9 GB) is the exact artifact the
2026-08-31 decoder results were measured on, and it stays byte-identical and
reproducible. This one is a superset, ~281,000 slices and ~42 GB.

Expect ~3.5-4 hours: 8,724 volumes at ~1.5 s each under `volume_window`, which
pays for an Otsu brain mask per volume that `minmax` does not (measured: the
volwin pair build did 2,246 volumes in 57 min; the minmax all-visits build did
~8,700 in 72 min without masking). It checkpoints every 500 volumes, so a
failure costs at most a few minutes of work and a resumed run picks up exactly
where it stopped.

Usage (from repo root, and see hpc/build_all_visits_cache.sbatch):
    ./.venv/bin/python brain_mets/build_all_visits_slice_cache.py
    ./.venv/bin/python brain_mets/build_all_visits_slice_cache.py --normalization minmax
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain_mets.brain_mets_slice_dataset import BrainMetsSliceDataset

ROOT_DIR = os.environ["BRAINJEPA_REGISTERED"]
CLINICAL_EXCEL = "Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx"
CACHE_PATHS = {
    "minmax": "cache/brain_mets_slice_cache_all.pt",
    "volume_window": "cache/brain_mets_slice_cache_volwin_all.pt",
}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--normalization", choices=sorted(CACHE_PATHS), default="volume_window")
ap.add_argument("--save-every", type=int, default=500,
                help="checkpoint the partial cache every N volumes")
ap.add_argument("--max-patients", type=int, default=None,
                help="smoke-test escape hatch; leave unset for the real build")
args = ap.parse_args()

cache_path = CACHE_PATHS[args.normalization]
print(f"Building all-visits {args.normalization} cache -> {cache_path}", flush=True)
t0 = time.time()

dataset = BrainMetsSliceDataset(
    root_dir=ROOT_DIR, clinical_excel=CLINICAL_EXCEL,
    cache_path=cache_path, normalization=args.normalization,
    max_patients=args.max_patients, save_every=args.save_every)

vols = len({s["path"] for s in dataset.samples})
print(f"\nDone in {(time.time() - t0) / 60:.1f} min: {len(dataset):,} slices across "
      f"{vols:,} volumes and {len(set(dataset.patient_ids)):,} patients; "
      f"{len(dataset._cache):,} slices in the cache file.", flush=True)
