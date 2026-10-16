"""End-to-end look at the staged pipeline: encoder -> predictor -> decoder.

Renders N held-out patients as eight rows. The first five separate the two things
that can go wrong, each adding one more stage of the real pipeline:

  1. real t0                        -- the input scan
  2. real t1                        -- the ground-truth answer
  3. decode(encode(t1).emb)         -- CEILING: the decoder inverting the REAL
                                       t1 pooled embedding. Row 3 vs row 2 is
                                       purely the cost of squeezing a scan
                                       through one 192-d vector -- the best
                                       this decoder can ever do, no predictor
                                       involved.
  4. decode(encode(t0).emb)         -- COPY-t0 BASELINE: the trivial "nothing
                                       changes" guess through the same decoder.
                                       Row 5 must beat THIS, not row 2.
  5. decode(predict(t0))            -- the actual end-to-end output. Row 5 vs
                                       row 3 is purely predictor error, since
                                       both rows use the same decoder on the
                                       same kind of input.

So: row 3 bad => the pooled bottleneck/decoder is the limit. Row 3 good but
row 4 bad => the decoder is fine and the predictor is the limit. This is the
same isolation logic as the 2026-08-25 test that retired the patch-grid
decoder, but now with a decoder actually trained on the space it is fed.

Rows 6-8 are signed, color-mapped CHANGE maps. Rows 3-5 all look like the same
blurry blob because the decoder's blur dominates every one of them; subtracting
decode(real t0) from each cancels that shared blur and leaves only the change:

  6. real t1 - real t0              -- TRUE CHANGE: what actually happened.
  7. decode(pred) - decode(t0)      -- PREDICTED CHANGE: what the pipeline says
                                       happens. Compare to row 8, not row 6.
  8. decode(real t1) - decode(t0)   -- CHANGE CEILING: the most change this
                                       decoder can express at all. Row 7 vs
                                       row 8 is predictor error alone (same
                                       decoder, same reference image).

Row 8 near-flat would mean the pooled 192-d bottleneck cannot represent
visit-to-visit change in the first place, which makes row 7 uninterpretable --
that is why the ceiling row is here and not just the two obvious ones.

TWO color scales, deliberately: row 6 lives in real-image intensity space, rows
7-8 in decoded space, whose range is far smaller. One shared scale would render
rows 7-8 uniformly white and hide the result. Rows 7 and 8 DO share a scale --
they are meant to be compared to each other.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.diagnostics.figstyle import (TEXT_WIDTH_IN, label as report_label,
                                          report_style)

report_style()
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import open_dict

from brain_mets.brain_mets_pair_dataset import BrainMetsPairDataset
from brain_mets.brain_mets_pair_dataset import ACTION_DIM
from lewm_core.utils import (get_column_normalizer, get_img_preprocessor, is_blank,
                             load_frozen_parts, split_by_patient)
from stable_pretraining import data as dt
from scripts.diagnostics.figpaths import figpath  # noqa: E402

ROOT_DIR = os.environ["BRAINJEPA_REGISTERED"]
CLINICAL_EXCEL = "Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx"
# Four columns, not ten: at the report's text width ten columns leave each scan
# about 15 mm across, which is too small to see a lesion in. Overridable for a
# working figure that is not going into the document.
N_SAMPLES = 4

# Parameterised 2026-08-28: there are now several decoder arms (radiodino,
# mricore, twin, random control) and editing constants per arm is how figures
# end up mislabelled. Defaults reproduce the original 2026-08-26 run.
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--config-name", default="brain_mets_decoder",
                 help="Hydra config the decoder was trained with.")
_ap.add_argument("--decoder-ckpt", required=True,
                 help="weights_epoch_N.pt from the decoder stage.")
_ap.add_argument("--predictor-ckpt", required=True,
                 help="Stage-2 predictor weights_epoch_N.pt, or 'none' to keep the predictor "
                      "that freeze_encoder_ckpt supplies.")
_ap.add_argument("--cache", default=None, help="Slice cache; must match the decoder's training data.")
_ap.add_argument("--data-config", default=None,
                 help="Data config to compose instead of the one --config-name brings. "
                      "REQUIRED when the predictor was trained on a different pairing: the "
                      "action z-score is fitted to whichever dataset is built here, and "
                      "first_two and consecutive have different dt distributions, so the "
                      "wrong one feeds the model a correctly-shaped but wrongly-scaled dt.")
_ap.add_argument("--residual", choices=["true", "false"], default=None,
                 help="Compose the residual predictor. Omit to keep the module default "
                      "(True, for the 2026-08-26 run); the encoder arms are NOT residual.")
_ap.add_argument("--out", default=None, help="Output png name (not path).")
_ap.add_argument("--columns", type=int, default=None,
                 help="Patients per figure. Defaults to 4, which is what fits the "
                      "report page; raise it for a working figure.")
_ap.add_argument("--pick", default=None,
                 help="1-indexed column numbers to DISPLAY, e.g. 2,3,4,8,10. Sampling "
                      "still draws --columns patients in the same deterministic order, "
                      "so a column number means the same patient across runs. Colour "
                      "scales are computed over the displayed columns only.")
_ap.add_argument("--score-only", action="store_true",
                 help="Score one clean slice from EVERY held-out patient (113) and exit "
                      "after printing the numbers, without drawing a figure. This is how "
                      "report Table 4.3 is produced. Before 2026-09-17 the same scoring "
                      "happened only by accident, via --columns 0, which then crashed in "
                      "plotting (job 4159191).")
_args = _ap.parse_args()

if _args.score_only and (_args.columns is not None or _args.pick):
    _ap.error("--score-only scores every held-out patient; it takes no --columns or --pick")
if _args.columns is not None:
    N_SAMPLES = _args.columns
if _args.score_only:
    N_SAMPLES = 0  # the sampling loop only stops early when len(idx) == N_SAMPLES

_stem = _args.config_name.replace("brain_mets_decoder", "pipeline")
OUT_PNG = figpath("prediction", _args.out or f"{_stem}.png")

# MUST match the checkpoint. residual_predict lives in the MODEL config, not in the
# weights, and the residual adds no parameters -- so loading residual weights into a
# non-residual model passes strict=True silently and evaluates f(emb) instead of
# emb + f(emb). That renders the raw predicted CHANGE as if it were the predicted
# embedding: near-zero, decoded as garbage, with no error raised.
RESIDUAL_PREDICTOR = True

# Brain voxels for the change-agreement correlation below. Background is exactly
# 0 after the dataset's per-slice min-max, so without this mask the huge block of
# identical (0, 0) pairs dominates both series and inflates the correlation toward
# 1 no matter what the model did. Display is NOT masked -- only the printed number.
BRAIN_MASK_THRESH = 0.02

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def denormalize(img):
    """(3,H,W) ImageNet-normalized tensor -> (H,W,3) uint8 for display."""
    img = img.numpy() * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def to_gray(img):
    """(3,H,W) ImageNet-normalized tensor -> (H,W) float in [0,1].

    denormalize() stops at uint8, which cannot hold a signed difference. The
    three channels are identical (the dataset replicates one grayscale slice,
    brain_mets_pair_dataset.py), so the channel mean is exact, not an average
    over different information. Clipping happens BEFORE the subtraction on
    purpose: the change maps then show the difference between the images
    actually displayed in the rows above, not between out-of-range floats.
    """
    return np.clip(img.numpy() * IMAGENET_STD + IMAGENET_MEAN, 0, 1).mean(0)


# CLI overrides must land BEFORE the cfg block reads RESIDUAL_PREDICTOR.
DECODER_CKPT = _args.decoder_ckpt
PREDICTOR_CKPT = None if _args.predictor_ckpt.lower() == "none" else _args.predictor_ckpt
if _args.residual is not None:
    RESIDUAL_PREDICTOR = (_args.residual == "true")

# return_hydra_config + set_config is LOAD-BEARING: config/train/data/*.yaml
# writes cache_path as ${hydra:runtime.cwd}/cache/..., and a bare compose()
# leaves HydraConfig unset so the first read of it raises. Same fix as
# fixed_anchor_eval.py.
with initialize(version_base=None, config_path="../../../config/train"):
    cfg = compose(config_name=_args.config_name,
                  overrides=([f"data={_args.data_config}"] if _args.data_config else []),
                  return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
with open_dict(cfg):
    cfg.model.action_encoder.input_dim = ACTION_DIM
    cfg.model.residual_predict = RESIDUAL_PREDICTOR

model = instantiate(cfg.model)
load_frozen_parts(model, cfg.freeze_encoder_ckpt, tuple(cfg.freeze_parts))

if DECODER_CKPT is not None:
    sd = torch.load(DECODER_CKPT, map_location="cpu", weights_only=False)
    decoder_sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
    model.decoder.load_state_dict(decoder_sd, strict=True)
    print(f"Loaded trained decoder from {DECODER_CKPT}")
else:
    print("DECODER_CKPT is None -- random-init decoder. Output will be noise; "
          "this run only checks the pipeline executes end to end.")

if PREDICTOR_CKPT is not None:
    sd = torch.load(PREDICTOR_CKPT, map_location="cpu", weights_only=False)
    for part in ("predictor", "pred_proj", "action_encoder"):
        part_sd = {k[len(part) + 1:]: v for k, v in sd.items() if k.startswith(f"{part}.")}
        getattr(model, part).load_state_dict(part_sd, strict=True)
    print(f"Loaded Stage-2 predictor from {PREDICTOR_CKPT} (residual={RESIDUAL_PREDICTOR})")
else:
    print("PREDICTOR_CKPT is None -- using the epoch-70 predictor "
          "(measured 5.4% worse than the copy-t0 baseline).")
model.eval()

dataset = BrainMetsPairDataset(root_dir=ROOT_DIR, clinical_excel=CLINICAL_EXCEL,
                               cache_path=_args.cache or cfg.data.dataset.cache_path,
                               pairing=cfg.data.dataset.get("pairing", "first_two"),
                               normalization=cfg.data.dataset.get("normalization", "minmax"))
transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
transforms.append(get_column_normalizer(dataset, "action", "action"))
dataset.transform = dt.transforms.Compose(*transforms)

# HELD-OUT patients only, and no blank frames. Sampling the whole dataset would
# mostly draw TRAINING patients (90% of it) and would put slices the model was
# fitted on into a figure meant to show generalization -- and ~17% of pairs have
# a blank frame, which would waste panels on black squares.
_, val_idx = split_by_patient(dataset, cfg.seed, cfg.train_split)
rng = np.random.RandomState(0)
idx, seen = [], set()
for i in rng.permutation(val_idx):
    px = torch.stack([dataset._cache[(dataset.pairs[i]["t0_path"], dataset.pairs[i]["slice_index"])],
                      dataset._cache[(dataset.pairs[i]["t1_path"], dataset.pairs[i]["slice_index"])]])
    if bool(is_blank(px.float()).any()):
        continue
    pid = dataset.pairs[i]["patient_id"]
    if pid in seen:          # one slice per patient, so the figure shows 10
        continue             # different people rather than 10 slices of one
    seen.add(pid)
    idx.append(int(i))
    if len(idx) == N_SAMPLES:
        break
print(f"figure samples: {len(idx)} clean pairs from {len(seen)} distinct held-out patients")
samples = [dataset[i] for i in idx]
pixels = torch.stack([s["pixels"] for s in samples])            # (N, T, 3, H, W)
action = torch.nan_to_num(torch.stack([s["action"] for s in samples]), 0.0)

with torch.no_grad():
    out = model.encode({"pixels": pixels, "action": action})
    emb, act_emb = out["emb"], out["act_emb"]
    n_patches = out["patch_emb"].size(2)

    # Row 3: decoder inverting the REAL t1 embedding (ceiling)
    recon_real = model.decode_prediction(emb[:, 1:2], num_patches=n_patches)[:, 0]

    # Row 4: the trivial "nothing changes" baseline, put through the SAME
    # decoder so it is compared like-for-like against row 5. Without this row
    # the figure cannot answer the only question that matters -- whether the
    # prediction beats simply copying the input forward. Comparing row 5 to
    # real t1 alone conflates predictor error with the pooled decoder's own
    # blur, which row 3 already shows is large.
    recon_copy_t0 = model.decode_prediction(emb[:, :1], num_patches=n_patches)[:, 0]

    # Row 5: full pipeline -- predict t1 from t0, then decode that
    pred_emb = model.predict(emb[:, :1], act_emb[:, :1])
    recon_pred = model.decode_prediction(pred_emb, num_patches=n_patches)[:, 0]

real_t0, real_t1 = pixels[:, 0], pixels[:, 1]
l1 = torch.nn.functional.l1_loss
print("\n=== pixel-space L1 (lower = closer) ===")
print(f"  ceiling   decode(real t1)  vs real t1 : {l1(recon_real, real_t1):.4f}")
print(f"  baseline  decode(real t0)  vs real t1 : {l1(recon_copy_t0, real_t1):.4f}   <- copy t0 forward")
print(f"  pipeline  decode(pred t1)  vs real t1 : {l1(recon_pred, real_t1):.4f}   <- must beat the line above")
print(f"  predictor decode(pred t1)  vs ceiling : {l1(recon_pred, recon_real):.4f}")
print(f"  reference real t0          vs real t1 : {l1(real_t0, real_t1):.4f}   (pixel no-change baseline)")
verdict = "BEATS" if l1(recon_pred, real_t1) < l1(recon_copy_t0, real_t1) else "LOSES TO"
print(f"\n  => the predictor {verdict} the copy-t0 baseline in pixel space.")

# ---- signed change maps (rows 6-8) --------------------------------------
# Every decoded row carries the same decoder blur, which is why rows 3-5 look
# alike. Subtracting decode(real t0) cancels that shared term and leaves only
# what the model claims changes between the visits.
g_t0, g_t1 = np.stack([to_gray(x) for x in real_t0]), np.stack([to_gray(x) for x in real_t1])
g_ceil = np.stack([to_gray(x) for x in recon_real])
g_copy = np.stack([to_gray(x) for x in recon_copy_t0])
g_pred = np.stack([to_gray(x) for x in recon_pred])

d_true = g_t1 - g_t0        # real-image space
d_pred = g_pred - g_copy    # decoded space
d_ceil = g_ceil - g_copy    # decoded space


mask = (g_t0 > BRAIN_MASK_THRESH) | (g_t1 > BRAIN_MASK_THRESH)

# Which columns are drawn. Sampling order is deterministic (RandomState(0)), so
# a column number identifies the same patient across runs.
COLS = ([int(x) - 1 for x in _args.pick.split(",")] if _args.pick
        else list(range(N_SAMPLES)))
assert all(0 <= c < N_SAMPLES for c in COLS), f"--pick out of range 1..{N_SAMPLES}"
N_COLS = len(COLS)


def sym_scale(*maps):
    """|d| 99th percentile -> symmetric limit, so one hot voxel cannot flatten
    the whole panel. Floored so an all-zero map cannot give vmin == vmax.

    Restricted to BRAIN voxels of the DISPLAYED columns. Pooling over every
    column let one high-change patient set everyone's scale, and including the
    large near-zero background dragged the percentile down, so brain tissue
    saturated to flat red. Both made the change maps unreadable."""
    sel = np.concatenate([np.abs(m[COLS])[mask[COLS]].ravel() for m in maps])
    v = np.percentile(sel, 99) if sel.size else 1e-6
    return max(float(v), 1e-6)


v_real = sym_scale(d_true)          # row 6 only -- different intensity space
v_dec = sym_scale(d_pred, d_ceil)   # shared: rows 7 and 8 must be comparable
print("\n=== change maps (rows 6-8) ===")
print(f"  magnitude mean |predicted change| : {np.abs(d_pred).mean():.4f}")
print(f"            mean |ceiling   change| : {np.abs(d_ceil).mean():.4f}   <- most this decoder can express")
print(f"            mean |true      change| : {np.abs(d_true).mean():.4f}   (real-image space, not comparable to the two above)")
print(f"  agreement corr(predicted, ceiling) : {np.corrcoef(d_pred[mask], d_ceil[mask])[0, 1]:+.3f}   <- predictor error alone")
print(f"            corr(predicted, true)    : {np.corrcoef(d_pred[mask], d_true[mask])[0, 1]:+.3f}   <- end to end")
print(f"            over {mask.mean():.1%} of pixels (brain mask, intensity > {BRAIN_MASK_THRESH})")
if _args.score_only:
    sys.exit(0)

# One ordered list rather than a grayscale block followed by a difference block,
# so every change map sits directly under the rows it is computed from. That
# gives three groups, each ending in its own change map: the real pair, then
# what the decoder can do with the real answer in hand, then what the model
# produces.
GRAY, DIFF = "gray", "diff"
rows = [
    (GRAY, real_t0, "real t0\n(input)", None, None),
    (GRAY, real_t1, "real t1\n(ground truth)", None, None),
    (DIFF, d_true, "real t1 - real t0\nTRUE CHANGE", v_real, "real intensity difference"),
    (GRAY, recon_copy_t0, "decode(real t0)\nCOPY-t0 BASELINE", None, None),
    (GRAY, recon_real, "decode(real t1)\nCEILING", None, None),
    (DIFF, d_ceil, "decode(real t1) - decode(t0)\nCHANGE CEILING", v_dec, "decoded intensity difference"),
    (GRAY, recon_pred, "decode(predict(t0))\nFULL PIPELINE", None, None),
    (DIFF, d_pred, "decode(pred) - decode(t0)\nPREDICTED CHANGE", v_dec, "decoded intensity difference"),
]
n_rows = len(rows)
# White space between the columns only, so the four patients read as separate
# images. No extra vertical space: rows stay as close as before (2026-10-08,
# user asked to keep the figure compact).
# Figure height is derived from the column width so every cell is square:
# with a fixed height the cells came out taller than the square images and
# left white bands between the rows.
_L, _R, _T, _B, _WS, _HS = 0.30, 0.89, 0.99, 0.01, 0.06, 0.04
_cell = TEXT_WIDTH_IN * (_R - _L) / (N_COLS + _WS * (N_COLS - 1))
fig = plt.figure(figsize=(TEXT_WIDTH_IN,
                          _cell * (n_rows + _HS * (n_rows - 1)) / (_T - _B)))
gs = fig.add_gridspec(n_rows, N_COLS, left=_L, right=_R, top=_T,
                      bottom=_B, wspace=_WS, hspace=_HS)
axes = np.array([[fig.add_subplot(gs[r, cc]) for cc in range(N_COLS)]
                 for r in range(n_rows)])
bars = []
for r, (kind, data, label, v, cbar_label) in enumerate(rows):
    for cc, c in enumerate(COLS):
        if kind == GRAY:
            axes[r, cc].imshow(denormalize(data[c]))
        else:
            # Background outside the brain is blanked, so it reads as page white
            # instead of picking up a small offset and tinting the panel.
            im = axes[r, cc].imshow(np.where(mask[c], data[c], np.nan),
                                    cmap="RdBu_r", vmin=-v, vmax=v)
        axes[r, cc].set_xticks([]); axes[r, cc].set_yticks([])
    axes[r, 0].set_ylabel(report_label(label), rotation=0, ha="right",
                          va="center", labelpad=6)
    if kind == DIFF:
        bars.append((r, im, cbar_label))
# No title: the report supplies one as a caption, which is also where the
# red-up / blue-down reading of the change maps belongs.

# One bar per change row. The two decoded rows are no longer adjacent, so a
# single shared bar would have to span the grayscale row between them and would
# read as if that row were on the same scale. They carry identical limits, and
# printing those limits twice is what shows the two rows are comparable.
for r, im, label in bars:
    pos = axes[r, -1].get_position()
    cb = fig.colorbar(im, cax=fig.add_axes([0.915, pos.y0, 0.010, pos.y1 - pos.y0]))
    cb.set_label(label)
    cb.ax.tick_params(labelsize=7)

Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}")
