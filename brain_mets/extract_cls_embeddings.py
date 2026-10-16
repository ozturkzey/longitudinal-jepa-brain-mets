"""
Per-patient frozen-encoder embeddings for the age/sex linear probe.

Produces ONE vector per patient by averaging the encoder's output over every
non-blank slice of that patient's FIRST registered FLAIR visit, reading pixels
straight out of cache/brain_mets_slice_cache.pt -- i.e. through the exact
preprocessing the encoder was trained on (center-crop to square -> resize 224 ->
uint8 RGB, done once by BrainMetsPairDataset; then /255 + ImageNet normalize,
verified here to match lewm_core.utils.get_img_preprocessor to 7e-7).

This matters: the earlier probe adapter (src/encoders/lewm_trained.py) read the
UNREGISTERED tree, skipped center_crop_to_square, and used Resize((224,224)),
which anisotropically squashes non-square slices. It scored the encoder on
inputs whose geometry it had never seen. That script is left alone -- this is
the corrected path, and its outputs are NOT comparable to
results/encoder_comparison/.

Three readouts come free from one forward pass (see lewm_core/jepa.py encode):
    cls        last_hidden_state[:, 0]          <- the encoder's own summary token
    patchmean  last_hidden_state[:, 1:].mean(1) <- average of the patch grid
    emb        projector(cls)                   <- the actual JEPA embedding

Run under .venv (needs stable_pretraining + transformers):
    ./.venv/bin/python brain_mets/extract_cls_embeddings.py \
        --ckpt $STABLEWM_HOME/checkpoints/lewm_brain_mets/weights_epoch_70.pt \
        --tag lewm_e70
    ./.venv/bin/python brain_mets/extract_cls_embeddings.py --random-init --tag vit_random
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain_mets.train_linear_probe import load_clinical_lookup  # noqa: E402
from lewm_core.module import MLP  # noqa: E402
from stable_pretraining.backbone.utils import vit_hf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = os.environ["BRAINJEPA_REGISTERED"]
CLINICAL_EXCEL = REPO_ROOT / "Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx"
CACHE_PATH = REPO_ROOT / "cache" / "brain_mets_slice_cache.pt"
OUT_DIR = REPO_ROOT / "results" / "probe_cls" / "embeddings"

MODALITY = "FLAIR"
IMG_SIZE = 224
PATCH_SIZE = 14
SIZE = "tiny"
EMBED_DIM = 192
PROJECTOR_HIDDEN_DIM = 2048  # config/train/model/lewm.yaml
RANDOM_INIT_SEED = 3072  # same value config/train/lewm.yaml uses, for a reproducible control

# dt.transforms.ToImage(**dataset_stats.ImageNet) == x.float()/255 then this
# normalization -- verified numerically against get_img_preprocessor (7.2e-07).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_encoder(ckpt_path):
    """ViT-tiny encoder + projector, both frozen and in eval mode.

    ckpt_path=None gives the untrained control: a fixed-seed random init, which
    is what the architecture + preprocessing score before JEPA training.

    The projector must be eval()'d as well as frozen -- it contains BatchNorm,
    whose running stats would otherwise keep moving and shift `emb` even with
    the weights held still.
    """
    torch.manual_seed(RANDOM_INIT_SEED)
    encoder = vit_hf(
        size=SIZE, patch_size=PATCH_SIZE, image_size=IMG_SIZE,
        pretrained=False, use_mask_token=False,
    )
    projector = MLP(
        input_dim=EMBED_DIM, hidden_dim=PROJECTOR_HIDDEN_DIM,
        output_dim=EMBED_DIM, norm_fn=torch.nn.BatchNorm1d,
    )

    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for module, prefix in ((encoder, "encoder."), (projector, "projector.")):
            part = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            module.load_state_dict(part, strict=True)  # strict: a silent partial load would invalidate the probe
            print(f"[extract] loaded {len(part)} {prefix}* tensors from {ckpt_path}")

    for module in (encoder, projector):
        module.eval().requires_grad_(False)
    return encoder.to(DEVICE), projector.to(DEVICE)


def first_visit_paths(root_dir, clinical, max_patients=None):
    """patient_id -> path of their FIRST registered FLAIR visit.

    Same population and same t0 rule as BrainMetsPairDataset._build_pairs:
    a clinical label, at least two FLAIR visits, earliest date first.
    """
    out = {}
    for patient_id in sorted(os.listdir(root_dir)):
        patient_path = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_path) or patient_id not in clinical:
            continue
        visit_dates = sorted(
            v for v in os.listdir(patient_path)
            if os.path.exists(os.path.join(patient_path, v, f"{MODALITY}.nii.gz"))
        )
        if len(visit_dates) < 2:
            continue
        out[patient_id] = os.path.join(patient_path, visit_dates[0], f"{MODALITY}.nii.gz")
        if max_patients is not None and len(out) >= max_patients:
            break
    return out


@torch.no_grad()
def encode_slices(slices, encoder, projector, batch_size):
    """(n,3,224,224) uint8 -> dict of (n, 192) arrays, one per readout."""
    readouts = {"cls": [], "patchmean": [], "emb": []}
    for start in range(0, len(slices), batch_size):
        batch = torch.stack(slices[start:start + batch_size]).float().div_(255.0)
        batch = ((batch - IMAGENET_MEAN) / IMAGENET_STD).to(DEVICE)

        tokens = encoder(batch, interpolate_pos_encoding=True).last_hidden_state
        cls = tokens[:, 0]
        readouts["cls"].append(cls.cpu())
        readouts["patchmean"].append(tokens[:, 1:].mean(dim=1).cpu())
        readouts["emb"].append(projector(cls).cpu())
    return {name: torch.cat(chunks).numpy() for name, chunks in readouts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", help="JEPA checkpoint to load encoder.* and projector.* from")
    source.add_argument("--random-init", action="store_true", help="untrained fixed-seed control")
    parser.add_argument("--tag", required=True, help="output filename prefix, e.g. lewm_e70")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--cache", default=str(CACHE_PATH))
    parser.add_argument("--root", default=ROOT_DIR)
    parser.add_argument("--max-patients", type=int, default=None, help="smoke-test cap")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"[extract] device={DEVICE}")
    clinical = load_clinical_lookup(str(CLINICAL_EXCEL))
    visits = first_visit_paths(args.root, clinical, args.max_patients)
    print(f"[extract] {len(visits)} patients with a clinical label and >=2 FLAIR visits")

    # mmap: the cache is 42GB on disk. Memory-mapping opens it in ~30s with
    # negligible resident memory, where a plain torch.load would need 42GB RAM.
    # Read-only -- the cache takes ~72 min to rebuild and BrainMetsSequenceDataset
    # reads the same file, so it is never written back.
    print(f"[extract] mmap-loading {args.cache} ...")
    cache = torch.load(args.cache, mmap=True, map_location="cpu")
    print(f"[extract] {len(cache)} cached slices")

    by_path = {}
    for path, slice_index in cache.keys():
        by_path.setdefault(path, []).append(slice_index)

    encoder, projector = build_encoder(args.ckpt)

    patient_ids, n_slices = [], []
    per_readout = {"cls": [], "patchmean": [], "emb": []}
    for i, (patient_id, path) in enumerate(sorted(visits.items())):
        if path not in by_path:
            raise KeyError(f"{patient_id}: {path} is absent from the cache -- rebuild it before probing")
        # Fully blank slices (max == min) are black frames outside the imaged
        # extent; averaging them in would dilute every patient's vector by a
        # patient-specific amount. Same criterion as
        # BrainMetsPairDataset._drop_blank_pairs.
        slices = [cache[(path, s)] for s in sorted(by_path[path])]
        slices = [t for t in slices if t.max() != t.min()]
        if not slices:
            raise ValueError(f"{patient_id}: every slice of {path} is blank")

        out = encode_slices(slices, encoder, projector, args.batch_size)
        for name, vectors in out.items():
            per_readout[name].append(vectors.mean(axis=0))
        patient_ids.append(patient_id)
        n_slices.append(len(slices))
        if (i + 1) % 100 == 0 or i + 1 == len(visits):
            print(f"  [{i + 1}/{len(visits)}] {patient_id} ({len(slices)} slices)")

    os.makedirs(args.out_dir, exist_ok=True)
    n_slices = np.array(n_slices)
    print(f"[extract] slices/patient: mean={n_slices.mean():.1f} min={n_slices.min()} max={n_slices.max()}")
    for name, vectors in per_readout.items():
        embeddings = np.stack(vectors).astype(np.float32)
        n_nan = int(np.isnan(embeddings).sum())
        if n_nan:
            raise ValueError(f"{name}: {n_nan} NaNs in embeddings")
        out_path = os.path.join(args.out_dir, f"{args.tag}_{name}.npz")
        np.savez(out_path, embeddings=embeddings, patient_ids=np.array(patient_ids), n_slices=n_slices)
        print(f"[extract] {name}: {embeddings.shape}, mean per-dim std="
              f"{embeddings.std(axis=0).mean():.4g} (near-zero would suggest collapse) -> {out_path}")


if __name__ == "__main__":
    main()
