"""Per-SLICE frozen-encoder embeddings, for the slice-position probe.

`brain_mets/extract_cls_embeddings.py` averages every slice of a patient into
one vector, because the age and sex probes ask a per-patient question. This asks
a per-slice one -- where in the head did this slice come from -- so it keeps one
row per slice and records where that slice actually sits.

WHY THE QUESTION IS WORTH ASKING. The encoder never sees a slice index. Each
slice is encoded as an independent image with no positional input, so anything a
readout can recover about position is anatomy the representation carries. It is
a far coarser fact than a lesion, which makes it an upper bound: an embedding
that cannot locate a slice along the head axis cannot be carrying lesion-scale
detail either.

WHAT POSITION MEANS HERE. Registration is within-patient
(`brain_mets/register_per_modality.py` warps later visits onto that patient's
earliest one), so there is no common template and slice 12 of one patient is not
slice 12 of another. The stacks are also ragged, 22 to 48 slices at 4 to 6 mm.
Two things are therefore recorded per slice and the probe uses both:

    position   index / (n_slices - 1), over the FULL stack, so dropping blank
               slices does not rescale it
    mm_step    that volume's own slice thickness, so a normalized error can be
               turned back into millimetres

TRIVIAL FEATURES, recorded here so the probe can run its own floor. Brain
cross-section shrinks towards the vertex and the skull base, so simple pixel
statistics already carry position. `mean`, `std` and `frac_nonzero` are computed
straight from the cached uint8 pixels with no learning and no free parameters.
An encoder has to beat these to have shown anything.

ONE VISIT PER PATIENT. Later visits of the same patient sit on the same
registered grid and are near-duplicates; they would add rows without adding
information. Population is the same as the other probes: a clinical record and
at least two FLAIR visits.

Run under .venv:
    ./.venv/bin/python scripts/diagnostics/probes/extract_slice_embeddings.py \
        --ckpt $STABLEWM_HOME/checkpoints/lewm_brain_mets/weights_epoch_70.pt \
        --tag lewm_e70
    ./.venv/bin/python scripts/diagnostics/probes/extract_slice_embeddings.py \
        --random-init --tag vit_random
"""

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from brain_mets.extract_cls_embeddings import (CLINICAL_EXCEL, ROOT_DIR,  # noqa: E402
                                               build_encoder, encode_slices,
                                               first_visit_paths)
from brain_mets.train_linear_probe import load_clinical_lookup  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "results" / "probe_slice_position" / "embeddings"

# The epoch-70 encoder was TRAINED on the per-slice min-max cache. Feeding it the
# volume-window pixels would score it on inputs it has never seen, which is the
# exact defect that invalidated the first age/sex probe. The cache is therefore
# an explicit argument, its name is stamped into every output file, and arms
# built from different caches must not be compared.
DEFAULT_CACHE = REPO_ROOT / "cache" / "brain_mets_slice_cache.pt"


def volume_geometry(path):
    """(n_slices, mm_per_slice) from the NIfTI header alone.

    nibabel is lazy, so this touches no pixel data. The slice axis is the
    shortest one, which is the convention `brain_mets/slice_cache.py` uses to
    cut the stack -- it must not be re-derived differently here or `position`
    would index a different axis than the cached pixels came from.
    """
    img = nib.load(path)
    axis = int(np.argmin(img.shape))
    return int(img.shape[axis]), float(img.header.get_zooms()[axis])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", help="JEPA checkpoint to load encoder.* and projector.* from")
    source.add_argument("--random-init", action="store_true",
                        help="untrained fixed-seed control, the comparison that matters")
    parser.add_argument("--tag", required=True, help="output filename prefix, e.g. lewm_e70")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--root", default=ROOT_DIR)
    parser.add_argument("--max-patients", type=int, default=None, help="smoke-test cap")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    clinical = load_clinical_lookup(str(CLINICAL_EXCEL))
    visits = first_visit_paths(args.root, clinical, args.max_patients)
    print(f"[extract] {len(visits)} patients with a clinical record and >=2 FLAIR visits")

    print(f"[extract] mmap-loading {args.cache} ...")
    cache = torch.load(args.cache, mmap=True, map_location="cpu")
    by_path = {}
    for path, slice_index in cache.keys():
        by_path.setdefault(path, []).append(slice_index)
    print(f"[extract] {len(cache):,} cached slices over {len(by_path):,} volumes")

    encoder, projector = build_encoder(args.ckpt)

    per_readout = {"cls": [], "patchmean": [], "emb": []}
    meta = {k: [] for k in ("patient_id", "slice_index", "n_slices", "position",
                            "mm_step", "mean", "std", "frac_nonzero")}

    for i, (patient_id, path) in enumerate(sorted(visits.items())):
        if path not in by_path:
            raise KeyError(f"{patient_id}: {path} is absent from {args.cache}")
        n_slices, mm_step = volume_geometry(path)

        indices, slices = [], []
        for s in sorted(by_path[path]):
            tile = cache[(path, s)]
            if tile.max() == tile.min():
                continue                      # blank frame outside the imaged extent
            indices.append(s)
            slices.append(tile)
        if len(slices) < 2:
            raise ValueError(f"{patient_id}: {len(slices)} non-blank slices in {path}")

        out = encode_slices(slices, encoder, projector, args.batch_size)
        for name, vectors in out.items():
            per_readout[name].append(vectors)

        for s, tile in zip(indices, slices):
            gray = tile[0].float()            # the three channels are copies
            meta["patient_id"].append(patient_id)
            meta["slice_index"].append(s)
            meta["n_slices"].append(n_slices)
            # Over the full stack, not over the surviving slices: a patient with
            # more blank edges would otherwise get a different ruler.
            meta["position"].append(s / max(n_slices - 1, 1))
            meta["mm_step"].append(mm_step)
            meta["mean"].append(float(gray.mean()) / 255.0)
            meta["std"].append(float(gray.std()) / 255.0)
            meta["frac_nonzero"].append(float((gray > 0).float().mean()))

        if (i + 1) % 100 == 0 or i + 1 == len(visits):
            print(f"  [{i + 1}/{len(visits)}] {patient_id} ({len(slices)} slices)")

    os.makedirs(args.out_dir, exist_ok=True)
    columns = {k: np.array(v) for k, v in meta.items()}
    n = len(columns["patient_id"])
    print(f"[extract] {n:,} slices from {len(set(columns['patient_id']))} patients | "
          f"{columns['n_slices'].min()}-{columns['n_slices'].max()} slices per stack, "
          f"{columns['mm_step'].min():.1f}-{columns['mm_step'].max():.1f} mm")

    for name, chunks in per_readout.items():
        embeddings = np.concatenate(chunks).astype(np.float32)
        if embeddings.shape[0] != n:
            raise ValueError(f"{name}: {embeddings.shape[0]} rows for {n} slices")
        if np.isnan(embeddings).any():
            raise ValueError(f"{name}: NaNs in embeddings")
        out_path = os.path.join(args.out_dir, f"{args.tag}_{name}.npz")
        np.savez(out_path, embeddings=embeddings, cache=os.path.basename(args.cache),
                 **columns)
        print(f"[extract] {name}: {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()
