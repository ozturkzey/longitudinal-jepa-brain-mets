"""How much of a brain slice survives in ONE pooled vector -- 768-d vs 192-d.

The question: decoding from the embedding ALONE, what
is the upper bound on the informational content of the embedding space? And
specifically, does RadioDINO's raw 768-d CLS token carry more than the 192-d
vector our projector reduces it to?

Two decoders answer it. They are the same network trained the same way on the
same slices; the only difference is the width of the vector each inverts:

    arm 768   frozen RadioDINO -> CLS (768-d)               -> decoder -> pixels
    arm 192   frozen RadioDINO -> frozen projector (192-d)  -> decoder -> pixels

Scored on held-out PATIENTS -- 113 people neither decoder ever saw. A pooled
decoder can memorize, so a training-set number would measure the wrong thing.

Three reference points, because an L1 in ImageNet-normalized units means
nothing on its own:

  * mean-image baseline -- L1 of ignoring the embedding entirely and emitting
    the training-set mean brain. Every arm must beat it; how far it beats it is
    the actual "informational content" number, reported as the % of the
    deviation-from-mean that the embedding recovers.
  * display-space L1 -- the same error in grey levels out of 255.
  * SSIM -- structure rather than intensity, which is what L1 is worst at.

Checkpoint selection is by best held-out recon_loss parsed from each job's
log, not by taking the last epoch on trust, and the choice is VERIFIED: the
recomputed val loss must match the logged one, which would fail loudly on an
epoch/checkpoint off-by-one.

Arms are given as repeated --arm/--log pairs, so this runs with one arm or
two. The 2026-08-31 two-arm deliverable is:

    ./.venv/bin/python scripts/diagnostics/decoder/decoder_information_bound.py \
        --arm radiodino768 --log /path/to/Logs_..._3353852_output.txt \
        --arm radiodino192 --log /path/to/Logs_..._3353853_output.txt

and the single-arm ceiling on every registered FLAIR slice is:

    ./.venv/bin/python scripts/diagnostics/decoder/decoder_information_bound.py \
        --arm radiodino768_all --log /path/to/Logs_..._<jobid>_output.txt \
        --dataset allvisits
"""

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics.figstyle import TEXT_WIDTH_IN, report_style

report_style()
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import open_dict
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

import stable_pretraining.data as dt
from brain_mets.brain_mets_pair_dataset import ACTION_DIM, BrainMetsPairDataset
from lewm_core.utils import (freeze_parts_inplace, get_column_normalizer,
                             get_img_preprocessor, is_blank, load_frozen_parts,
                             split_by_patient)
from scripts.diagnostics.figpaths import figpath

ROOT_DIR = os.environ["BRAINJEPA_REGISTERED"]
CLINICAL_EXCEL = str(REPO_ROOT / "Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx")
CACHE = str(REPO_ROOT / "cache" / "brain_mets_slice_cache_volwin.pt")
CACHE_ALL = str(REPO_ROOT / "cache" / "brain_mets_slice_cache_volwin_all.pt")
# Where SaveCkptCallback writes: $STABLEWM_HOME/checkpoints (set in env.sh).
CKPT_ROOT = Path(os.environ["STABLEWM_HOME"]) / "checkpoints"

# Display names for the arms this has been run on; any other arm falls back to
# its own name, so a new arm needs no edit here.
LABEL = {"radiodino768": "768-d (no projector)",
         "radiodino192": "192-d (projector)",
         "radiodino768_all": "768-d (no projector), all visits",
         "brain_mets_maedec_radiodino_patch": "196x768 patch grid, MAE decoder",
         "brain_mets_decoder_radiodino_patch": "196x768 patch grid"}


def config_name(arm):
    """`--arm` names the config the arm trained under. A bare name keeps the
    historical `brain_mets_decoder_<arm>` spelling; anything already starting
    with `brain_mets_` is taken verbatim, which is how the MAE-decoder arms
    (`brain_mets_maedec_*`) are addressed without renaming the old ones."""
    return arm if arm.startswith("brain_mets_") else f"brain_mets_decoder_{arm}"


def ckpt_dir_for(arm):
    """The arm's checkpoint directory, read from its OWN config.

    Until 2026-09-01 this was spelled `CKPT_ROOT / f"lewm_brain_mets_decoder_
    {arm}"`, which silently assumed every arm's `output_model_name` follows
    that one pattern. Two already do not:

        brain_mets_maedec_radiodino_patch -> lewm_brain_mets_maedec_radiodino_patch
        brain_mets_decoder_e70_matched    -> lewm_brain_mets_decoder_matched

    so both would have looked in a directory that does not exist and died on
    `missing ...weights_epoch_N.pt` -- a confusing failure that reads as "the
    training run never saved" rather than "this script guessed the path".
    `output_model_name` is what SaveCkptCallback actually writes under, so it
    is the single source of truth and is now the only thing consulted.
    """
    with initialize(version_base=None, config_path="../../../config/train"):
        cfg = compose(config_name=config_name(arm))
    return CKPT_ROOT / cfg.output_model_name


def decode_for(model, out, source):
    """Decode through the SAME path `train.py` fed this arm during training.

    Scoring an arm through a different path measures a decoder that was never
    trained, and it fails silently -- the run finishes and the number looks
    plausible. Until 2026-09-01 this script hardcoded `decode_prediction`, which
    is correct only for `source: pooled`; a `patch` arm scored through it would
    have been nonsense. The source now comes from the arm's own config.
    """
    if source == "patch":
        return model.decode(out["patch_emb"])
    if source == "pooled":
        return model.decode_prediction(out["emb"],
                                       num_patches=out["patch_emb"].size(2))
    raise ValueError(f"unknown loss.recon.source {source!r}")
BATCH = 128
# Four columns, not ten: ten scans across the report's text width leave each one
# too small to judge a reconstruction by, which is the only thing this figure is
# for.
N_FIG = 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

# `[Epoch 7/250] done in 61.2s | ... | validate/recon_loss_epoch: 0.1108`
EPOCH_RE = re.compile(r"\[Epoch (\d+)/\d+\] done.*?validate/recon_loss_epoch: ([0-9.]+)")


def to_display(img):
    """(3,H,W) ImageNet-normalized -> (H,W) grey in [0,1]. The three channels
    are identical (the dataset replicates one grey slice), so the channel mean
    is exact rather than an average over different information."""
    a = img.detach().cpu().float().numpy() * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(a, 0, 1).mean(0)


def best_epoch(log_path, ckpt_dir):
    """(lightning_epoch, logged_val_recon): the lowest held-out recon among the
    epochs that actually have a checkpoint on disk.

    Not the last epoch -- early stopping runs `patience` epochs past the
    optimum, so the final checkpoint is never the best one. And not the best
    LOGGED epoch either: these runs save every 5th (`ckpt_every`), so the global
    optimum may have no weights. Reports the gap between the two, which is the
    honest cost of coarse saving and should be within the curve's own noise.
    """
    text = Path(log_path).read_text(errors="ignore")
    hits = [(int(e), float(v)) for e, v in EPOCH_RE.findall(text)]
    if not hits:
        raise SystemExit(f"no epoch lines found in {log_path} -- is the job finished?")
    # weights_epoch_{k+1} is the state that produced [Epoch k]'s validation.
    have = {int(f.stem.split("_")[-1]) - 1 for f in ckpt_dir.glob("weights_epoch_*.pt")}
    saved = [h for h in hits if h[0] in have]
    if not saved:
        raise SystemExit(f"none of the {len(hits)} logged epochs has a checkpoint "
                         f"in {ckpt_dir}")
    ep, val = min(saved, key=lambda t: t[1])
    g_ep, g_val = min(hits, key=lambda t: t[1])
    print(f"  {Path(log_path).name}: {len(hits)} epochs logged, {len(saved)} with "
          f"weights; best saved = epoch {ep} at {val:.4f} "
          f"(best overall was epoch {g_ep} at {g_val:.4f}, "
          f"gap {val - g_val:+.4f})")
    return ep, val


def build(arm, ckpt):
    """Instantiate the arm as train.py did, then load the trained decoder."""
    with initialize(version_base=None, config_path="../../../config/train"):
        cfg = compose(config_name=config_name(arm))
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = ACTION_DIM
    model = instantiate(cfg.model)
    if cfg.get("freeze_encoder_ckpt"):
        load_frozen_parts(model, cfg.freeze_encoder_ckpt, tuple(cfg.freeze_parts))
    if cfg.get("freeze_untrained_parts"):
        freeze_parts_inplace(model, tuple(cfg.freeze_untrained_parts))
    load_frozen_parts(model, str(ckpt), ("decoder",))
    return cfg, model.eval()


@torch.no_grad()
def evaluate(model, dataset, indices, mean_img, device, source):
    """Held-out metrics for one arm, plus the baseline on the same frames."""
    l1_sum = base_sum = disp_sum = ssim_sum = 0.0
    n = 0
    for start in range(0, len(indices), BATCH):
        chunk = indices[start:start + BATCH]
        samples = [dataset[int(i)] for i in chunk]
        pixels = torch.stack([s["pixels"] for s in samples]).to(device)
        action = torch.nan_to_num(
            torch.stack([s["action"] for s in samples]), 0.0).to(device)

        out = model.encode({"pixels": pixels, "action": action})
        dec = decode_for(model, out, source)

        real = pixels.flatten(0, 1).float()          # (N*T,3,H,W)
        pred = dec.flatten(0, 1).float()
        l1_sum += (pred - real).abs().mean(dim=(1, 2, 3)).sum().item()
        base_sum += (mean_img - real).abs().mean(dim=(1, 2, 3)).sum().item()

        # display space: undo ImageNet normalization, clamp, grey levels of 255
        std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32)
        mu = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32)
        rd = (real * std + mu).clamp(0, 1).mean(1, keepdim=True)
        pd = (pred * std + mu).clamp(0, 1).mean(1, keepdim=True)
        disp_sum += (pd - rd).abs().mean(dim=(1, 2, 3)).sum().item() * 255
        ssim_sum += ssim(pd, rd, data_range=1.0, reduction="sum").item()
        n += real.size(0)
    return {"l1": l1_sum / n, "baseline": base_sum / n,
            "display_l1": disp_sum / n, "ssim": ssim_sum / n, "n_frames": n}


@torch.no_grad()
def val_recon_bf16(model, dataset, indices, device, source):
    """Held-out recon under autocast bf16 -- the precision `trainer.precision:
    bf16` validated in. Used ONLY to confirm the checkpoint is the epoch the log
    named. Scoring in fp32 and comparing against a bf16 log entry looks like a
    ~0.0026 mismatch that is entirely numerical: measured 2026-08-31, four
    checkpoints matched their logged epoch to 0.0000 in bf16 and to 0.0026 in
    fp32. Reported metrics stay fp32; only this check matches the log's units.
    """
    tot, n = 0.0, 0
    for start in range(0, len(indices), BATCH):
        chunk = indices[start:start + BATCH]
        px = torch.stack([dataset[int(i)]["pixels"] for i in chunk]).to(device)
        ac = torch.nan_to_num(
            torch.stack([dataset[int(i)]["action"] for i in chunk]), 0.0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.encode({"pixels": px, "action": ac})
            dec = decode_for(model, out, source)
        tot += (dec.float() - px.float()).abs().mean(dim=(2, 3, 4)).sum().item()
        n += px.size(0) * px.size(1)
    return tot / n


@torch.no_grad()
def train_mean_image(dataset, train_idx, device, n_sample=8192):
    """The mean brain over TRAINING frames -- what you would predict knowing
    the training set and nothing about this patient."""
    rng = np.random.RandomState(0)
    sample = rng.choice(train_idx, size=min(n_sample, len(train_idx)), replace=False)
    total, n = None, 0
    for start in range(0, len(sample), BATCH):
        px = torch.stack([dataset[int(i)]["pixels"]
                          for i in sample[start:start + BATCH]]).to(device).flatten(0, 1).float()
        total = px.sum(0) if total is None else total + px.sum(0)
        n += px.size(0)
    return (total / n).unsqueeze(0)                  # (1,3,H,W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME",
                    help="arm name, i.e. the config/train/brain_mets_decoder_<NAME>.yaml "
                         "it trained under. Repeat with a matching --log per arm.")
    ap.add_argument("--log", action="append", required=True, metavar="PATH",
                    help="that arm's SLURM output log, for best-epoch selection")
    ap.add_argument("--dataset", choices=("pairs", "allvisits"), default="pairs",
                    help="pairs: BrainMetsPairDataset on the pair cache (the "
                         "2026-08-31 default). allvisits: every registered FLAIR "
                         "slice, BrainMetsSliceDataset on the all-visits cache.")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |recomputed - logged| val recon before the "
                         "checkpoint choice is treated as an off-by-one")
    args = ap.parse_args()
    if len(args.arm) != len(args.log):
        raise SystemExit(f"got {len(args.arm)} --arm but {len(args.log)} --log; "
                         "they are positional pairs and must match")
    ARMS = args.arm
    logs = dict(zip(args.arm, args.log))
    label = lambda a: LABEL.get(a, a)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No GPU visible -- this needs one.")

    print("\n=== checkpoint selection (best held-out epoch, not the last) ===")
    chosen = {}
    for arm in ARMS:
        ckpt_dir = ckpt_dir_for(arm)
        ep, logged = best_epoch(logs[arm], ckpt_dir)
        ckpt = ckpt_dir / f"weights_epoch_{ep + 1}.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing {ckpt}")
        chosen[arm] = (ckpt, logged)

    if args.dataset == "allvisits":
        # Every registered FLAIR slice. require_clinical=True is what the
        # training config sets, and it must match here or the seeded split
        # lands on a different population -- see config/train/data/
        # brain_mets_allvisits.yaml for the measured 13-patient consequence.
        from brain_mets.brain_mets_slice_dataset import BrainMetsSliceDataset
        dataset = BrainMetsSliceDataset(root_dir=ROOT_DIR, clinical_excel=CLINICAL_EXCEL,
                                        img_size=224, cache_path=CACHE_ALL,
                                        normalization="volume_window",
                                        drop_blank_slices=True, require_clinical=True)
        # No action normalizer: this dataset's action column is all zeros, and
        # get_column_normalizer divides by its standard deviation.
        transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
    else:
        dataset = BrainMetsPairDataset(root_dir=ROOT_DIR, clinical_excel=CLINICAL_EXCEL,
                                       img_size=224, cache_path=CACHE,
                                       normalization="volume_window",
                                       drop_blank_slices=True)
        transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=224),
                      get_column_normalizer(dataset, "action", "action")]
    dataset.transform = dt.transforms.Compose(*transforms)

    # The split the arms trained under; every one of these configs carries
    # seed 3072 / train_split 0.9.
    with initialize(version_base=None, config_path="../../../config/train"):
        seed_cfg = compose(config_name=config_name(ARMS[0]))
    train_idx, val_idx = split_by_patient(dataset, seed_cfg.seed, seed_cfg.train_split)
    pids = dataset.patient_ids
    assert not ({pids[i] for i in train_idx} & {pids[i] for i in val_idx}), \
        "patient leak between train and val"
    n_val_patients = len({pids[i] for i in val_idx})
    print(f"\nheld-out: {len(val_idx):,} samples from "
          f"{n_val_patients} patients (never seen in training)")

    mean_img = train_mean_image(dataset, train_idx, device)

    results, recons = {}, {}
    for arm in ARMS:
        ckpt, logged = chosen[arm]
        print(f"\n=== {arm} ===")
        cfg, model = build(arm, ckpt)
        model.to(device)
        source = cfg.loss.recon.get("source", "pooled")
        print(f"  decode path: loss.recon.source={source}")
        r = evaluate(model, dataset, val_idx, mean_img, device, source)
        # The checkpoint really is the epoch the log named -- checked in the
        # log's own precision, see val_recon_bf16.
        check = val_recon_bf16(model, dataset, val_idx, device, source)
        drift = abs(check - logged)
        print(f"  recomputed val recon {check:.4f} (bf16, as logged) vs logged "
              f"{logged:.4f} (drift {drift:.4f}); fp32 {r['l1']:.4f}")
        # Only meaningful when we score the SAME frames the training run
        # validated on. An all-visits-trained decoder scored on the pair frames
        # (or the reverse) changes the frame set, so the recomputed number
        # legitimately differs from the logged one and the comparison says
        # nothing about the epoch->checkpoint mapping. Narrowed rather than
        # dropped: for a matched frame set this still catches an off-by-one.
        trained_on = ("allvisits" if "BrainMetsSliceDataset" in cfg.data.dataset._target_
                      else "pairs")
        if args.dataset == trained_on:
            assert drift < args.tol, (
                f"{arm}: recomputed val recon {check:.4f} does not match the logged "
                f"{logged:.4f}. The epoch->checkpoint mapping is wrong, so this would "
                f"report a different model than the one selected.")
        else:
            print(f"  checkpoint-mapping assertion SKIPPED: this arm validated on "
                  f"'{trained_on}' frames and is scored here on '{args.dataset}' "
                  f"frames, so the logged value is not the right reference. The "
                  f"mapping is verified by the --dataset {trained_on} run.")
        results[arm] = r
        recons[arm] = (model, source)

    # ---------------------------------------------------------------- table
    base = results[ARMS[0]]["baseline"]
    print(f"\n{'=' * 78}")
    print("INFORMATION CONTENT OF ONE POOLED VECTOR -- held-out patients")
    print(f"{'=' * 78}")
    print(f"  {'':<26}{'L1 (norm)':>12}{'L1 (grey/255)':>15}{'SSIM':>8}{'% of dev':>10}")
    print(f"  {'mean-image baseline':<26}{base:>12.4f}{'':>15}{'':>8}{0.0:>9.1f}%")
    for arm in ARMS:
        r = results[arm]
        pct = 100 * (1 - r["l1"] / r["baseline"])
        print(f"  {label(arm):<26}{r['l1']:>12.4f}{r['display_l1']:>15.2f}"
              f"{r['ssim']:>8.3f}{pct:>9.1f}%")
    if len(ARMS) == 2:
        a, b = (results[k]["l1"] for k in ARMS)
        print(f"\n  {label(ARMS[0])} / {label(ARMS[1])} = {a / b:.3f}   "
              f"({'the first is better' if a < b else 'the second is better or equal'})")
    print(f"  scored on {results[ARMS[0]]['n_frames']:,} held-out frames")

    # --------------------------------------------------------------- figure
    rng = np.random.RandomState(0)
    fig_idx, seen = [], set()
    for i in rng.permutation(val_idx):
        # patient_ids is a per-sample list on BOTH datasets; dataset.pairs
        # exists only on the pair one.
        pid = dataset.patient_ids[i]
        if pid in seen:      # one slice per patient, so the figure shows N_FIG
            continue         # different people rather than N_FIG slices of one
        seen.add(pid)
        fig_idx.append(int(i))
        if len(fig_idx) == N_FIG:
            break
    samples = [dataset[i] for i in fig_idx]
    pixels = torch.stack([s["pixels"] for s in samples]).to(device)
    action = torch.nan_to_num(torch.stack([s["action"] for s in samples]), 0.0).to(device)

    # allvisits samples carry ONE frame, pairs carry two (t0, t1); either way
    # the frame of interest is the last one.
    t_last = pixels.size(1) - 1
    rows = [("real" if t_last == 0 else "real t1", None)] + [(label(a), a) for a in ARMS]
    panels = {}
    with torch.no_grad():
        for arm in ARMS:
            arm_model, arm_source = recons[arm]
            out = arm_model.encode({"pixels": pixels, "action": action})
            panels[arm] = decode_for(arm_model, out, arm_source)[:, t_last]

    fig, axes = plt.subplots(len(rows), N_FIG,
                             figsize=(TEXT_WIDTH_IN, 1.62 * len(rows)))
    for j in range(N_FIG):
        axes[0, j].imshow(to_display(pixels[j, t_last]), cmap="gray", vmin=0, vmax=1)
        for r, (_, arm) in enumerate(rows[1:], start=1):
            axes[r, j].imshow(to_display(panels[arm][j]), cmap="gray", vmin=0, vmax=1)
    for r, (name, arm) in enumerate(rows):
        # Title only, not title + ylabel: the two carried the same text and the
        # row name was printed twice per row. The held-out L1 used to be
        # appended here; it belongs in the caption, where the report can state
        # it once in the document's own font. It is still printed to stdout and
        # written to the results table.
        axes[r, 0].set_title(name, fontweight="bold", fontsize=8.5, loc="left")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    # No title. What it used to say -- that every arm shares a decoder, data and
    # schedule, and what the mean-image baseline is -- is caption material, and
    # both numbers are printed above and saved with the results.
    fig.tight_layout()
    # Keep the bare name ONLY for the 2026-08-31 two-arm deliverable, so
    # re-running that reproduces results/diagnostics/decoder/information_bound.png
    # in place. Every other combination is named by its arms and frame set --
    # otherwise a single-arm re-run silently overwrites that figure.
    if ARMS == ["radiodino768", "radiodino192"] and args.dataset == "pairs":
        stem = "information_bound"
    else:
        stem = "information_bound_" + "_".join(ARMS) + f"_{args.dataset}"
    out_png = figpath("decoder", f"{stem}.png")
    fig.savefig(out_png)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
