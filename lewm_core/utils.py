from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

def split_by_patient(dataset, seed, train_split):
    """Patient-grouped train/val split: every sample of a patient lands on one
    side, so no patient is ever seen in both. Returns (train_idx, val_idx).

    This is the single definition of the split -- train.py and every diagnostic
    that wants to score a checkpoint on held-out patients must call this, so a
    diagnostic can never silently disagree with the split the model trained on.
    """
    patient_ids = dataset.patient_ids
    unique_patients = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_patients)
    train_patients = set(unique_patients[: int(len(unique_patients) * train_split)])
    train_idx = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_idx = [i for i, pid in enumerate(patient_ids) if pid not in train_patients]
    return train_idx, val_idx


def is_blank(pixels):
    """(..., C, H, W) -> (...) bool. True when a frame is spatially constant in
    every channel, i.e. carries no information.

    Note the per-channel std: after ImageNet normalization an all-zero slice
    becomes three *different* per-channel constants, so a std taken across the
    channel axis is NOT zero and silently misses every blank slice.
    """
    return pixels.std(dim=(-2, -1)).max(dim=-1).values < 1e-6


@torch.no_grad()
def embed_dataset(model, dataset, indices, batch_size=128, verbose=True):
    """Run the FROZEN encoder+projector over `indices` once and return
    (embs, actions) as lists of per-sample tensors.

    Worth caching because `emb = projector(CLS)` cannot change: both modules are
    frozen at epoch-70 in every Stage-2/Stage-3 config, so every diagnostic that
    re-encodes is recomputing an identical answer. Measured cost of not caching:
    ~8.5 min to load the 42 GB slice cache plus ~10 min of ViT forwards, per run.

    The action_encoder is deliberately NOT applied here -- it is trainable, so
    act_emb differs per checkpoint. The normalized `action` is stored instead and
    the (tiny, 156k-param) action_encoder is re-run at scoring time.

    Groups by sequence length so variable-T datasets can stack, with the same
    frame budget as VisitLengthBatchSampler.
    """
    lengths = getattr(dataset, "sequence_lengths", None)
    groups = {}
    for i in indices:
        groups.setdefault(2 if lengths is None else int(lengths[int(i)]), []).append(int(i))

    embs, actions, order, n_blank = [], [], [], []
    done = 0
    for T, members in sorted(groups.items()):
        bs = max(1, min(batch_size, (batch_size * 2) // max(T, 1)))
        for start in range(0, len(members), bs):
            chunk = members[start:start + bs]
            samples = [dataset[j] for j in chunk]
            pixels = torch.stack([s["pixels"] for s in samples])
            action = torch.nan_to_num(torch.stack([s["action"] for s in samples]), 0.0)
            emb = model.encode({"pixels": pixels, "action": action})["emb"]
            blank = is_blank(pixels).sum(dim=1)
            for k in range(len(chunk)):
                embs.append(emb[k].clone())
                actions.append(action[k].clone())
                n_blank.append(int(blank[k]))
            order.extend(chunk)
            done += len(chunk)
            if verbose and done % 2048 < bs:
                print(f"    embedded {done}/{len(indices)}", flush=True)
    return {"emb": embs, "action": actions, "index": order, "n_blank": n_blank}


def save_embedding_cache(path, payload):
    import os as _os
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    n = len(payload["emb"])
    mb = sum(e.numel() for e in payload["emb"]) * 4 / 1e6
    print(f"[embedding cache] saved {n:,} samples ({mb:.1f} MB) to {path}")


def load_embedding_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    print(f"[embedding cache] loaded {len(payload['emb']):,} samples from {path}")
    return payload


def summarize_scores(dp, db, n_blank, n_frames, seed=0):
    """Group per-sample errors into all/clean/one_blank/both_blank and bootstrap
    a CI on the ratio-of-means. Shared by score_predictor (encodes on the fly) and
    score_from_cache (uses precomputed frozen embeddings) so the two can never
    disagree on how a number is produced.

    A sample with EVERY frame blank and one with only ONE blank frame are opposite
    cases and must not be pooled: all-blank means consecutive frames encode
    identically, so the copy-forward baseline is exactly 0 and unbeatable; one-blank
    means adjacent frames are maximally different, so the baseline is terrible.
    """
    rng = np.random.default_rng(seed)
    groups_out = {
        "all": torch.ones_like(n_blank, dtype=torch.bool),
        "clean": n_blank == 0,
        "one_blank": n_blank == 1,
        "both_blank": (n_blank == n_frames) & (n_blank > 0),
    }

    def summarize(mask):
        if mask.sum() == 0:
            return {"n": 0}
        a, b = dp[mask].numpy(), db[mask].numpy()
        # Bootstrap the ratio-of-means. A per-sample ratio a/b is useless here:
        # all-blank samples have b ~ 0, so individual ratios explode and their
        # spread is a division artifact rather than anything real.
        boot = np.array([
            (lambda i: a[i].mean() / max(b[i].mean(), 1e-12))(rng.integers(0, len(a), len(a)))
            for _ in range(1000)
        ])
        return {
            "n": int(mask.sum()),
            "pred_dist": float(a.mean()),
            "baseline_dist": float(b.mean()),
            # Report BOTH aggregations: by Jensen's inequality the mean of
            # per-sample ratios is always >= the ratio of means, and near 1.0 they
            # can straddle it and disagree. "ratio" is the honest headline;
            # train.py logs the mean-of-ratios kind.
            "ratio": float(a.mean() / max(b.mean(), 1e-12)),
            "ratio_se": float(boot.std()),
            "ratio_ci95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
            "mean_of_ratios": float(np.mean(a / np.maximum(b, 1e-12))),
            "beats_baseline_frac": float((a < b).mean()),
        }

    result = {k: summarize(m) for k, m in groups_out.items()}
    result["blank_frac"] = float((n_blank > 0).float().mean())
    return result


@torch.no_grad()
def score_from_cache(model, cache, n=2048, seed=0, n_preds=1):
    """Same scoring as score_predictor, but from precomputed frozen embeddings.

    Skips the 42 GB slice-cache load and every ViT forward -- ~18 min -> ~2 s.
    Only the trainable action_encoder and predictor are run, so this is valid for
    ANY checkpoint sharing the frozen encoder+projector the cache was built with.
    """
    total = len(cache["emb"])
    sel = np.random.RandomState(seed).choice(total, size=min(n, total), replace=False)
    dp, db, nb, nf = [], [], [], []
    for i in sel:
        emb = cache["emb"][int(i)].unsqueeze(0)          # (1, T, D)
        act = cache["action"][int(i)].unsqueeze(0)       # (1, T, A)
        ctx_len = emb.size(1) - n_preds
        act_emb = model.action_encoder(act)
        pred = model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
        tgt, ctx = emb[:, n_preds:], emb[:, :ctx_len]
        dp.append((pred - tgt).norm(dim=-1).mean(dim=1))
        db.append((ctx - tgt).norm(dim=-1).mean(dim=1))
        nb.append(cache["n_blank"][int(i)])
        nf.append(emb.size(1))
    return summarize_scores(torch.cat(dp), torch.cat(db),
                            torch.tensor(nb), torch.tensor(nf), seed=seed)


@torch.no_grad()
def score_predictor(model, dataset, indices, n=1024, batch_size=128, seed=0, n_preds=1):
    """Score predict(t0) against the trivial "copy t0 forward" baseline.

    The ratio is the number that matters: < 1.0 means the predictor beats doing
    nothing, >= 1.0 means it does not. Reports a standard error alongside, since
    every generalization claim here rests on ~113 held-out patients and a point
    estimate off a single batch is not enough to call a winner.

    Returns a dict of means/SEs plus the same numbers restricted to non-blank
    frames -- blank frames are where the baseline scores an exact zero, so they
    flatter the baseline and must be reportable separately.
    """
    idx = np.random.RandomState(seed).choice(indices, size=min(n, len(indices)), replace=False)

    # Group by sequence length before batching. BrainMetsSequenceDataset yields
    # 2..51 frames per sample, and torch.stack cannot combine different T -- the
    # same reason training needs VisitLengthBatchSampler. Pair datasets have no
    # `sequence_lengths`, so they fall through as a single group and behave
    # exactly as before.
    lengths = getattr(dataset, "sequence_lengths", None)
    if lengths is None:
        groups = {2: [int(i) for i in idx]}
    else:
        groups = {}
        for i in idx:
            groups.setdefault(int(lengths[int(i)]), []).append(int(i))

    dp, db, n_blank_frames = [], [], []
    for T, members in sorted(groups.items()):
        # Frame budget, same reasoning as VisitLengthBatchSampler: a fixed sample
        # count means memory scales with T (job 3180372 OOMed on exactly this).
        bs = max(1, min(batch_size, (batch_size * 2) // max(T, 1)))
        for start in range(0, len(members), bs):
            chunk = members[start:start + bs]
            samples = [dataset[j] for j in chunk]
            pixels = torch.stack([s["pixels"] for s in samples])
            action = torch.nan_to_num(torch.stack([s["action"] for s in samples]), 0.0)
            out = model.encode({"pixels": pixels, "action": action})
            emb = out["emb"]
            ctx_len = emb.size(1) - n_preds
            pred = model.predict(emb[:, :ctx_len], out["act_emb"][:, :ctx_len])
            tgt, ctx = emb[:, n_preds:], emb[:, :ctx_len]
            # .mean(dim=1) averages over the T-1 transitions in a sequence, so a
            # long timeline counts once, not T-1 times -- otherwise patients with
            # many visits would dominate the population score.
            dp.append((pred - tgt).norm(dim=-1).mean(dim=1))
            db.append((ctx - tgt).norm(dim=-1).mean(dim=1))
            n_blank_frames.append(is_blank(pixels).sum(dim=1))
    dp, db, n_blank = torch.cat(dp), torch.cat(db), torch.cat(n_blank_frames)
    n_frames = torch.full_like(n_blank, 2) if lengths is None else torch.tensor(
        [int(lengths[i]) for grp in sorted(groups) for i in groups[grp]])
    result = summarize_scores(dp, db, n_blank, n_frames, seed=seed)
    return result


def load_frozen_parts(model, ckpt_path, parts=("encoder",)):
    """Load the named submodules' weights from a raw model.state_dict()-format
    checkpoint (SaveCkptCallback's format, see below) into an already-
    instantiated JEPA model, then freeze them permanently. Submodules NOT
    named stay at whatever fresh init Hydra gave them.

    `parts=("encoder",)` is the encoder-only case (train the rest from
    scratch on top of a fixed visual backbone). Freezing more than the
    encoder is what makes the *embedding space itself* fixed: `emb` is
    projector(CLS), so with a trainable projector the space the predictor
    targets -- and that a pooled-embedding decoder must learn to invert --
    drifts every step. Freeze ("encoder", "projector") to pin that space,
    plus ("predictor", "pred_proj", "action_encoder") to pin the prediction
    path too."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for name in parts:
        sub = getattr(model, name)
        prefix = f"{name}."
        sub_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = sub.load_state_dict(sub_sd, strict=True)
        assert not missing and not unexpected, (
            f"[load_frozen_parts] '{name}' checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
        for p in sub.parameters():
            p.requires_grad = False
        sub.eval()
        print(f"[load_frozen_parts] loaded + froze '{name}' from {ckpt_path} "
              f"({sum(p.numel() for p in sub.parameters()):,} params)")
    model._frozen_modules = list(parts)
    return model


def load_pretrained_checkpoint(model, repo_id):
    """Load a HF-Hub-hosted LeWM JEPA checkpoint's weights into an already-instantiated
    model of matching architecture (config/train/model/lewm.yaml is built to match
    quentinll/lewm-pusht's architecture exactly, so no rebuilding is needed here).
    Tries the raw state_dict first, then falls back to remapping ViT key names --
    installed transformers versions differ in which internal ViTModel layout they use
    (discovered + fixed the same issue in src/encoders/lewm_pretrained.py)."""
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(repo_id=repo_id, filename="weights.pt")
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(sd, strict=True)
        print(f"[checkpoint] loaded {repo_id} (raw keys, no remapping needed)")
    except RuntimeError:
        renamed = {}
        for k, v in sd.items():
            if k.startswith("encoder."):
                k = k.replace("encoder.encoder.layer.", "encoder.layers.")
                k = k.replace(".attention.attention.query.", ".attention.q_proj.")
                k = k.replace(".attention.attention.key.", ".attention.k_proj.")
                k = k.replace(".attention.attention.value.", ".attention.v_proj.")
                k = k.replace(".attention.output.dense.", ".attention.o_proj.")
                k = k.replace(".intermediate.dense.", ".mlp.fc1.")
                k = k.replace(".output.dense.", ".mlp.fc2.")
            renamed[k] = v
        model.load_state_dict(renamed, strict=True)
        print(f"[checkpoint] loaded {repo_id} (after ViT key remapping)")
    return model


def freeze_parts_inplace(model, parts):
    """Freeze named submodules at whatever weights they were constructed with,
    without loading a checkpoint.

    `load_frozen_parts` is the usual route, but it requires weights on disk. A
    frozen-RANDOM-encoder control has no checkpoint by definition: the point is
    untrained weights held fixed. Encoders that freeze themselves
    (TimmViTEncoder, MRICoreEncoder) do not need this; `vit_hf` returns a plain
    HuggingFace ViTModel that does not, so without this it would train and stop
    being a control.

    Sets `model._frozen_modules` the same way, which is what `JEPA.train()`
    reads to keep these in eval mode every epoch -- otherwise Lightning flips
    dropout back on in a module whose whole purpose is to be a fixed function.
    """
    for name in parts:
        sub = getattr(model, name)
        for p in sub.parameters():
            p.requires_grad = False
        sub.eval()
        print(f"[freeze_parts_inplace] froze '{name}' at init "
              f"({sum(p.numel() for p in sub.parameters()):,} params, NO checkpoint)")
    model._frozen_modules = list(model._frozen_modules) + [p for p in parts
                                                           if p not in model._frozen_modules]
    return model


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained.

    Refuses to start writing into a directory that already holds another run's
    checkpoints. `save_pretrained` writes to `checkpoints/<run_name>/` with no
    job-id namespacing, so two runs sharing an `output_model_name` silently
    overwrite each other epoch by epoch. That already destroyed one set of
    weights here: job 3175109 overwrote job 3172332's, and only the recorded
    scores survived. It is also one `--config-name
    =brain_mets` away from destroying `lewm_brain_mets/weights_epoch_70.pt`,
    which every downstream stage loads.

    Overwriting is never what is wanted here, so this raises rather than warns.
    Pass a distinct `output_model_name` per run.
    """

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

        import stable_worldmodel as swm
        run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_name)
        existing = sorted(run_dir.glob("weights_epoch_*.pt")) if run_dir.is_dir() else []
        if existing:
            raise FileExistsError(
                f"{run_dir} already holds {len(existing)} checkpoint(s) from an earlier "
                f"run (e.g. {existing[0].name}). Saving here would overwrite them epoch "
                f"by epoch. Set a distinct output_model_name -- e.g. "
                f"output_model_name={run_name}_<something> -- or move the old directory."
            )

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )