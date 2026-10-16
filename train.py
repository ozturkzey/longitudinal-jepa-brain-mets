import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from lewm_core.module import SIGReg
from lewm_core.utils import freeze_parts_inplace, get_column_normalizer, get_img_preprocessor, load_frozen_parts, load_pretrained_checkpoint, split_by_patient, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight
    # pred_loss's weight was implicitly 1.0 and unconfigurable. Harmless while
    # everything upstream of it is frozen, but an arm that TRAINS the encoder
    # (the reconstruction-ceiling arm) must be able to switch it off: pred_emb
    # and tgt_emb both come off `emb`, so with a trainable encoder this term
    # backpropagates into the encoder through a randomly-initialised frozen
    # predictor, and the cheapest way to shrink it is to blandify `emb` -- a
    # collapse pressure fighting the reconstruction objective. Note
    # loss.detach_target does NOT cover this: it closes the target branch, not
    # the context branch. Default 1.0, so every existing config is unchanged.
    pred_cfg = cfg.loss.get("pred")
    pred_w = 1.0 if pred_cfg is None else pred_cfg.get("weight", 1.0)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    # Context length derived from the real batch T, not a fixed config
    # value -- required for variable-length sequences (BrainMetsSequenceDataset
    # + VisitLengthBatchSampler), where T differs batch-to-batch. A fixed
    # cfg.history_size would be silently wrong once it doesn't match T
    # exactly: slicing past the end of a dim doesn't error in PyTorch, so
    # ctx_emb would end up including the target frame itself whenever
    # history_size >= T, letting the model see the answer it's meant to
    # predict. T - n_preds is always correct and is a no-op for the old
    # fixed-pair path (T=2, n_preds=1 -> ctx_len=1, unchanged).
    ctx_len = emb.size(1) - n_preds

    # SIGReg acts on `emb` itself, so it is defined whether or not there is a
    # prediction to make. The prediction terms are added to it below.
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = lambd * output["sigreg_loss"]

    # ctx_len == 0 means the dataset carries ONE frame per sample and there is
    # nothing to predict from -- BrainMetsSliceDataset, used by the
    # reconstruction-only arms. Guarding here is not an optimisation: without
    # it, ctx_emb and tgt_emb are empty (B, 0, D) tensors, `.pow(2).mean()` of
    # an empty tensor is NaN, and `pred_w * nan` is NaN even at pred_w == 0.0.
    # The total loss would be NaN from the first step with no error raised.
    # A single-frame arm has no prediction to make, so pred_loss/baseline_loss
    # are simply not produced -- the log_dict comprehension below filters on
    # key presence, so they are absent from the logs rather than reported as
    # meaningless zeros.
    if ctx_len > 0:
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, : ctx_len]

        tgt_emb = emb[:, n_preds:] # label
        if cfg.loss.get("detach_target", False):
            # Stop-gradient on the target branch. Without this, pred_loss has a
            # gradient path into the target as well as the prediction, so the loss
            # can be reduced by making emb_t1 *easier to predict* rather than by
            # predicting better -- the encoder gets told to blandify its own
            # targets. Published JEPAs stop-grad or EMA the target branch for
            # exactly this reason. Job 3096280 trained with this channel open.
            #
            # Defaults to False so the upstream pusht/lewm.yaml pipeline keeps its
            # original semantics; a no-op when the encoder+projector are frozen.
            tgt_emb = tgt_emb.detach()
        # LeWM loss
        if pred_w == 0.0:
            # The prediction path is logged but contributes nothing, so skip
            # building its graph rather than build one and multiply it away. Both
            # sides are detached: with a trainable encoder, tgt_emb carries
            # gradient too.
            with torch.no_grad():
                pred_emb = self.model.predict(ctx_emb, ctx_act)
            output["pred_loss"] = (pred_emb - tgt_emb.detach()).pow(2).mean()
        else:
            pred_emb = self.model.predict(ctx_emb, ctx_act) # pred
            output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        output["loss"] = output["loss"] + pred_w * output["pred_loss"]

        # The trivial "copy the last context frame forward" baseline, logged (never
        # trained on) so that losing to it is visible on the first validation epoch
        # rather than only afterwards via a separate script. ctx_emb and tgt_emb
        # are both (B, T - n_preds, D), so this is a like-for-like comparison.
        # ratio < 1.0 => the predictor beats doing nothing.
        output["baseline_loss"] = (ctx_emb - tgt_emb).pow(2).mean().detach()
        output["pred_loss_vs_baseline"] = output["pred_loss"].detach() / output["baseline_loss"].clamp(min=1e-8)
    # Reconstruction loss: decode the pooled embedding emb=projector(CLS),
    # broadcast to every patch position, back to pixels. That is exactly what
    # JEPA.decode_prediction feeds at inference, so the decoder is trained on
    # the same kind of input it will be used on. Training it here on REAL
    # t0/t1 embeddings (rather than predicted ones) is valid because
    # pred_loss above trains predict(t0) into that same emb space -- a
    # decoder that inverts real embeddings therefore inverts predicted ones,
    # and gets every real frame as a training example instead of only
    # predicted ones. Requires the projector to be frozen (see
    # utils.load_frozen_parts), or the target space drifts from under it.
    #
    # Off by default (no loss.recon block == a plain nn.Identity decoder,
    # e.g. lewm.yaml's pusht runs).
    recon_cfg = cfg.loss.get("recon")
    if recon_cfg is not None and not isinstance(self.model.decoder, torch.nn.Identity):
        real_pixels = batch["pixels"].float()  # (B, T, C, H, W), same tensor encode() saw
        # `pooled` is the production path and the default, so no existing config
        # changes. `patch` feeds the decoder the encoder's REAL per-patch grid
        # instead of one broadcast vector -- 196 genuinely different tokens
        # rather than 196 copies of the same one. That is not a pipeline the
        # predictor can drive (it emits one pooled vector), so this is a
        # DIAGNOSTIC arm only: an upper bound that says what the decoder can do
        # when position is not thrown away, which is what makes the pooled
        # numbers interpretable. Note the decoder's embed_dim must then be the
        # ENCODER's width, since patch tokens are taken before the projector.
        source = recon_cfg.get("source", "pooled")
        if source == "patch":
            decoded = self.model.decode(output["patch_emb"])
        elif source == "pooled":
            decoded = self.model.decode_prediction(emb, num_patches=output["patch_emb"].size(2))
        else:
            raise ValueError(f"loss.recon.source must be 'pooled' or 'patch', got {source!r}")
        output["recon_loss"] = F.l1_loss(decoded, real_pixels)
        output["loss"] = output["loss"] + recon_cfg.weight * output["recon_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True, prog_bar=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    if "_target_" in dataset_cfg:
        # Hand-rolled dataset (e.g. brain_mets_pair_dataset.BrainMetsPairDataset):
        # plain torch.utils.data.Dataset, not an HDF5/Lance/folder-format
        # swm.data.load_dataset can resolve -- instantiate it directly instead.
        dataset = hydra.utils.instantiate(cfg.data.dataset)
    else:
        dataset_name = dataset_cfg.pop("name")
        cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
        dataset = swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.get("frameskip", 1) * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    if hasattr(dataset, "patient_ids"):
        # Patient-grouped split: a plain random_split over samples would risk
        # putting the same patient's slices in both train and val once a
        # patient contributes many pairs (one per slice) -- always split by
        # patient_id, never by sample, for medical data.
        train_idx, val_idx = split_by_patient(dataset, cfg.seed, cfg.train_split)
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
    else:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )

    if hasattr(dataset, "sequence_lengths"):
        # Variable-length sequences (BrainMetsSequenceDataset): every batch
        # must be internally uniform-length (tensors need a consistent shape
        # to stack), so group same-length sequences instead of padding --
        # see visit_length_sampler.py for why this avoids touching
        # lewm_core's shared Attention class.
        from brain_mets.brain_mets_sequence_dataset import VisitLengthBatchSampler

        seq_lengths = dataset.sequence_lengths
        train_lengths = [seq_lengths[i] for i in train_idx]
        val_lengths = [seq_lengths[i] for i in val_idx]
        # max_frames_per_batch keeps memory flat across length groups; without it
        # batch_size=128 at T=51 needs ~3.9 GB/batch and OOM-kills the job.
        max_frames = cfg.loader.get("max_frames_per_batch", None)
        train_batch_sampler = VisitLengthBatchSampler(
            train_lengths, batch_size=cfg.loader.batch_size, shuffle=True,
            drop_last=True, generator=rnd_gen, max_frames_per_batch=max_frames,
        )
        val_batch_sampler = VisitLengthBatchSampler(
            val_lengths, batch_size=cfg.loader.batch_size, shuffle=False,
            drop_last=False, max_frames_per_batch=max_frames,
        )
        loader_kwargs = {k: v for k, v in cfg.loader.items()
                         if k not in ("batch_size", "max_frames_per_batch")}
        train = torch.utils.data.DataLoader(train_set, batch_sampler=train_batch_sampler, **loader_kwargs)
        val = torch.utils.data.DataLoader(val_set, batch_sampler=val_batch_sampler, **loader_kwargs)
    else:
        train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
        val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    if cfg.get("ckpt_id"):
        load_pretrained_checkpoint(world_model, cfg.ckpt_id)

    if cfg.get("freeze_encoder_ckpt"):
        load_frozen_parts(world_model, cfg.freeze_encoder_ckpt, tuple(cfg.get("freeze_parts", ("encoder",))))

    # Freeze modules at their INIT weights, no checkpoint -- for the
    # frozen-random-encoder control, where untrained-and-fixed is the point.
    if cfg.get("freeze_untrained_parts"):
        freeze_parts_inplace(world_model, tuple(cfg.freeze_untrained_parts))

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Every epoch by default, unchanged for every existing config. Configurable
    # because a checkpoint holds the WHOLE JEPA -- frozen encoder and all -- so
    # its size follows the model, not the trainable part: the 768-d decoder arm
    # writes 720 MB per epoch, and 250 of those on both arms is ~360 GB of
    # shared filesystem for two 30M-parameter decoders. Coarser saving is only
    # sound while the val curve is smooth enough that the best SAVED epoch and
    # the best epoch agree to within noise; check that before raising it.
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model,
        epoch_interval=cfg.get("ckpt_every", 1),
    )
    callbacks = [object_dump_callback]

    # Off by default (no early_stopping block == old behavior, e.g. lewm.yaml's
    # pusht runs) -- only opts in when a config explicitly asks for it, so this
    # doesn't change any pipeline that hasn't been told to use it. See
    # config/train/brain_mets.yaml, added after job 3096280's 300-epoch run
    # overfit past its epoch-~70 validation optimum.
    es_cfg = cfg.get("early_stopping")
    if es_cfg and es_cfg.get("enabled", False):
        # min_delta defaults to 0.0 in Lightning, which counts a 1e-7 improvement
        # as progress and resets the patience counter -- so on a converged, flat
        # curve the callback can never fire. Job 3271656 sat at 0.1108 for the
        # last fifth of its run and stopped only on max_epochs. Forwarded here
        # with a 0.0 default so every existing config keeps its behaviour.
        callbacks.append(EarlyStopping(
            monitor=es_cfg.monitor, mode=es_cfg.mode, patience=es_cfg.patience,
            min_delta=es_cfg.get("min_delta", 0.0),
        ))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
