"""
Pretrained 2D radiology encoders, wrapped so `lewm_core.jepa.JEPA` can use
them in place of its from-scratch ViT-tiny.

Why a wrapper is needed. `JEPA.encode()` calls

    output = self.encoder(pixels, interpolate_pos_encoding=True)
    pixels_emb = output.last_hidden_state[:, 0]

which is the HuggingFace `ViTModel` interface that
`stable_pretraining.backbone.utils.vit_hf` returns. RadioDINO is published as
a *timm* checkpoint (its config.json is `{"architecture":
"vit_base_patch16_224", "pretrained_cfg": ...}`, with no transformers
`model_type`), so `transformers.AutoModel` cannot load it and timm's own
`forward_features()` returns a bare `(N, 1+P, D)` tensor with no
`.last_hidden_state` and no `interpolate_pos_encoding` argument. This module
is the adapter between the two, and nothing else.

The encoder is frozen here rather than by `utils.load_frozen_parts`, which
only handles weights coming from one of our own checkpoints. That matters for
more than `requires_grad`: `JEPA.train()` re-`eval()`s exactly the modules
listed in `model._frozen_modules`, which `load_frozen_parts` populates and
nothing else does. A pretrained encoder frozen only by `requires_grad` would
be flipped back into train mode by Lightning at every epoch, re-enabling
dropout inside a backbone whose whole purpose is to be a fixed function. So
`train()` is overridden below.
"""

import contextlib
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# timm model ids for the encoders wired through TimmViTEncoder. MRI-CORE is not
# one of them -- it is a SAM-style checkpoint needing the vendored
# mazurowski-lab/mri_foundation repo, so it gets its own class below.
TIMM_MODEL_IDS = {
    "radiodino": "hf_hub:Snarcy/RadioDino-b16",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MRI_FOUNDATION_ROOT = os.path.join(REPO_ROOT, "mri_foundation")
MRI_CORE_CKPT = os.path.join(MRI_FOUNDATION_ROOT, "pretrained_weights", "MRI_CORE_vitb.pth")


class _Output:
    """The one attribute `JEPA.encode()` reads off its encoder's return value."""

    __slots__ = ("last_hidden_state",)

    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class TimmViTEncoder(nn.Module):
    """A frozen timm ViT presenting the HuggingFace `ViTModel` call interface.

    Returns an object with `.last_hidden_state` of shape (N, 1 + P, D): the
    CLS token first, then the patch grid -- the same layout `JEPA.encode()`
    assumes when it takes `[:, 0]` as the pooled vector and `[:, 1:]` as
    `patch_emb`.
    """

    def __init__(self, name, img_size=224, pretrained=True, trainable=False):
        """`trainable=False` is the default and every existing arm depends on
        it: those configs deliberately omit `encoder` from `freeze_parts` /
        `freeze_untrained_parts` because this class freezes itself.

        `trainable=True` exists for ONE thing -- the reconstruction-ceiling
        arm, which fine-tunes the encoder to measure how much a pooled vector
        can hold when the encoder is optimised for the job rather than for
        semantic invariance. It undoes all three freezes below. Note that
        flipping `requires_grad` from outside would NOT be enough: the forward
        runs under `torch.no_grad()`, so no graph is built and the encoder
        would silently stay fixed."""
        super().__init__()
        import timm

        if name not in TIMM_MODEL_IDS:
            raise ValueError(
                f"unknown pretrained encoder {name!r}; wired: {sorted(TIMM_MODEL_IDS)}"
            )
        if not pretrained:
            raise ValueError(
                "pretrained=False defeats the purpose of this class -- the whole "
                "point of Setup B is a frozen PRETRAINED backbone. Use "
                "stable_pretraining.backbone.utils.vit_hf for a from-scratch ViT."
            )
        self.name = name
        self.img_size = img_size
        self.trainable = trainable
        self.model = timm.create_model(TIMM_MODEL_IDS[name], pretrained=True, num_classes=0)
        self.embed_dim = self.model.num_features

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def train(self, mode=True):
        """Stay in eval permanently *when frozen*: the weights never update, so
        dropout and any norm running-stats must not be re-enabled by Lightning's
        per-epoch .train() call. See the module docstring. When `trainable`,
        this is an ordinary module and follows the mode it is given."""
        super().train(mode)
        if not self.trainable:
            self.model.eval()
        return self

    def forward(self, pixels, interpolate_pos_encoding=False, **_ignored):
        h, w = pixels.shape[-2:]
        if (h, w) != (self.img_size, self.img_size):
            # timm's vit_base_patch16_224 declares fixed_input_size=True, so the
            # position embedding is not resized for us. Raise rather than let a
            # silently-wrong grid through -- JEPA passes
            # interpolate_pos_encoding=True unconditionally, and honouring that
            # request is not something this backbone can do.
            raise ValueError(
                f"{self.name} expects {self.img_size}x{self.img_size} input, got {h}x{w}. "
                "This backbone has a fixed position embedding and cannot interpolate it."
            )
        # THE line that decides whether this encoder can be trained at all.
        # Under no_grad no graph is built, so a caller that merely flipped
        # requires_grad would get a silently-frozen encoder and a result that
        # looks like a real run.
        ctx = contextlib.nullcontext() if self.trainable else torch.no_grad()
        with ctx:
            tokens = self.model.forward_features(pixels)  # (N, 1 + P, D)
        return _Output(tokens)


def _mri_core_pos_embed(image_size, patch_size=16):
    """Re-derive MRI-CORE's position embedding at `image_size`.

    Two separate problems are handled here, both inherited from upstream:

    1. The vendored loader has a real bug -- when it remaps the DINOv2 teacher's
       `backbone.pos_embed` to `image_encoder.pos_embed` it stores the
       interpolated tensor under the ORIGINAL key, so `image_encoder.pos_embed`
       silently stays at its zero-initialised default. It shows up in
       `missing_keys` on load. This recomputes it correctly.
    2. The checkpoint was trained at 1024x1024 (a 64x64 token grid) and we run
       at 224 (14x14), so the grid is bilinearly resized. Standard practice, but
       it IS off-design use of this checkpoint -- unlike RadioDINO, which is
       native at 224. Record that when comparing the two arms.
    """
    raw = torch.load(MRI_CORE_CKPT, map_location="cpu", weights_only=False)["teacher"]
    patch_pos = raw["backbone.pos_embed"][:, 1:, :]  # drop the cls slot
    cur = int(math.sqrt(patch_pos.shape[1]))
    tgt = image_size // patch_size
    grid = patch_pos.reshape(1, cur, cur, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, (tgt, tgt), mode="bilinear", align_corners=False)
    return grid.permute(0, 2, 3, 1)  # (1, tgt, tgt, 768)


class MRICoreEncoder(nn.Module):
    """Frozen MRI-CORE (mazurowski-lab/mri_foundation) presenting the same
    interface as TimmViTEncoder.

    Two structural differences from a plain ViT, both forced by the checkpoint:

    * **No CLS token.** SAM's image encoder keeps a spatial (B, H, W, D) grid and
      has no cls slot -- the teacher checkpoint's `cls_token` lands in
      `unexpected_keys` and is discarded. So index 0 of the returned sequence is
      the MEAN of the patch tokens, which is the same pooled representation
      `src/encoders/mri_core.py` used for the encoder comparison (age MAE 6.72).
      RadioDINO uses a real CLS token; this asymmetry between the two pretrained
      arms is a property of the checkpoints, not a choice.
    * **The neck is skipped.** SAM's 768->256 conv neck is bolted on for
      segmentation and carries NO trained weights in this checkpoint, so
      `image_encoder(x)` would be dominated by a random conv. The forward below
      stops before it: patch_embed -> +pos_embed -> blocks.
    """

    def __init__(self, img_size=224, patch_size=16):
        super().__init__()
        if not os.path.exists(MRI_CORE_CKPT):
            raise FileNotFoundError(
                f"MRI-CORE checkpoint not found at {MRI_CORE_CKPT}. Clone "
                "mazurowski-lab/mri_foundation into <repo>/mri_foundation and "
                "download MRI_CORE_vitb.pth -- see src/encoders/mri_core.py."
            )
        self.img_size = img_size
        self.embed_dim = 768

        # The vendored builder hardcodes map_location='cuda:0', which crashes on
        # a CPU-only process. Patch torch.load for the duration of the build
        # only -- a module-level patch (as src/encoders/mri_core.py does) would
        # leak into Lightning's own checkpoint loading.
        if MRI_FOUNDATION_ROOT not in sys.path:
            sys.path.insert(0, MRI_FOUNDATION_ROOT)
        import cfg
        from models.sam import sam_model_registry

        real_load, saved_argv = torch.load, sys.argv
        torch.load = lambda *a, **k: real_load(*a, **{**k, "map_location": "cpu"})
        sys.argv = ["mri_core_encoder"]  # cfg.parse_args() reads sys.argv
        try:
            args = cfg.parse_args()
            args.if_encoder_adapter = False
            args.if_mask_decoder_adapter = False
            model = sam_model_registry["vit_b"](
                args, checkpoint=MRI_CORE_CKPT, num_classes=1,
                image_size=img_size, pretrained_sam=False,
            )
        finally:
            torch.load, sys.argv = real_load, saved_argv

        self.image_encoder = model.image_encoder
        with torch.no_grad():
            self.image_encoder.pos_embed.copy_(_mri_core_pos_embed(img_size, patch_size))

        for p in self.image_encoder.parameters():
            p.requires_grad_(False)
        self.image_encoder.eval()

    def train(self, mode=True):
        """Frozen permanently -- see TimmViTEncoder.train."""
        super().train(mode)
        self.image_encoder.eval()
        return self

    def forward(self, pixels, interpolate_pos_encoding=False, **_ignored):
        h, w = pixels.shape[-2:]
        if (h, w) != (self.img_size, self.img_size):
            raise ValueError(
                f"mri_core was built for {self.img_size}x{self.img_size} input, got {h}x{w}. "
                "The position embedding is baked at construction; rebuild with img_size."
            )
        with torch.no_grad():
            x = self.image_encoder.patch_embed(pixels)     # (N, H', W', D)
            x = x + self.image_encoder.pos_embed
            for blk in self.image_encoder.blocks:
                x = blk(x)
            tokens = x.flatten(1, 2)                       # (N, P, D)
            pooled = tokens.mean(dim=1, keepdim=True)      # stands in for CLS
            tokens = torch.cat([pooled, tokens], dim=1)    # (N, 1 + P, D)
        return _Output(tokens)
