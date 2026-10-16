"""JEPA Implementation"""

import copy

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        decoder=None,
        residual_predict=False,
        clone_target_encoder=False,
        num_preds=1,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.decoder = decoder or nn.Identity()
        self.residual_predict = residual_predict
        self._frozen_modules = []

        # Twin-encoder setup: the TARGET frame (t1) gets its own encoder instead
        # of being pushed through the same weights as the context frame (t0).
        # A true clone -- deepcopy, so both branches start from identical
        # weights and diverge only through training. Two independent random
        # inits would be a different experiment.
        #
        # The projector is deliberately NOT cloned. Both branches share it, so
        # both embeddings land in one 192-d space; that is what makes pred_loss
        # and the copy-t0 baseline meaningful. It also means the target is
        # produced by trainable weights even when both encoders are frozen --
        # see train.py's loss.detach_target.
        self.num_preds = num_preds
        self.target_encoder = copy.deepcopy(encoder) if clone_target_encoder else None

        if residual_predict:
            # Zero-init the LAST linear of the residual branch so predict()
            # returns exactly `emb` at initialization -- i.e. the model starts
            # as the exact identity map, which is the "copy t0 forward"
            # baseline it has to beat.
            #
            # Without this the residual does NOT start at the baseline:
            # pred_proj is a randomly-initialised Linear->BN->GELU->Linear, so
            # f(emb) is noise of substantial magnitude. Measured 2026-08-26:
            # ||predict(emb) - emb|| / ||emb|| = 0.203, i.e. 20% of the
            # embedding norm away from the identity before any training.
            #
            # Zeroing the output layer does not block learning -- the gradient
            # w.r.t. its weight is input^T @ grad_out, which is non-zero -- it
            # only makes the branch start silent. Same technique as
            # ConditionalBlock's AdaLN-zero gates (module.py:102), and standard
            # for zero-initialised residual branches (DiT, ControlNet).
            linears = [m for m in self.pred_proj.modules() if isinstance(m, nn.Linear)]
            if not linears:
                raise ValueError(
                    "residual_predict=True requires pred_proj to contain at least one "
                    f"nn.Linear to zero-initialise, found none in {type(self.pred_proj).__name__}"
                )
            nn.init.zeros_(linears[-1].weight)
            if linears[-1].bias is not None:
                nn.init.zeros_(linears[-1].bias)

    def train(self, mode=True):
        """Keep frozen submodules permanently in eval mode -- Lightning calls
        .train() on the whole module every epoch, which would otherwise flip
        their dropout/BatchNorm back on even though their weights never
        update. For BatchNorm-carrying parts (projector/pred_proj) this is
        not cosmetic: in train mode they would normalize with batch stats and
        keep updating running stats, so a "frozen" embedding space would
        still drift batch-to-batch. Populated by utils.load_frozen_parts."""
        super().train(mode)
        for name in self._frozen_modules:
            getattr(self, name).eval()
        return self

    def _encode_frames(self, encoder, pixels, b):
        """Run (b, t, ...) pixels through one encoder + the SHARED projector.

        Returns (pooled (b, t, D), patches (b, t, P, D)). Factored out so the
        single-encoder and twin-encoder paths cannot drift apart.
        """
        flat = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        output = encoder(flat, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)

        # Per-patch embeddings (pre-projection, excludes the cls token at index 0) --
        # `emb` above is a single summary vector per frame and has no spatial
        # detail left for a pixel decoder to reconstruct from; this keeps the full
        # patch grid around for that purpose instead of discarding it.
        patch_emb = output.last_hidden_state[:, 1:]  # (b*t, P, D)
        return (rearrange(emb, "(b t) d -> b t d", b=b),
                rearrange(patch_emb, "(b t) p d -> b t p d", b=b))

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys

        With `target_encoder` set, context frames and target frames go through
        DIFFERENT encoders (see __init__). Without it, every frame goes through
        `self.encoder`, exactly as before -- that path must stay unchanged,
        every existing checkpoint depends on it.
        """

        pixels = info['pixels'].float()
        b, t = pixels.shape[0], pixels.shape[1]

        if self.target_encoder is None:
            emb, patch_emb = self._encode_frames(self.encoder, pixels, b)
        else:
            # train.py splits the result as ctx = [:, :t - n] and tgt = [:, n:].
            # Routing them to separate encoders only makes sense when those two
            # slices PARTITION the frames -- every frame is either a context or
            # a target, never both and never neither. That holds exactly when
            # t == 2n (the t0/t1 pair case, t=2, n=1). Longer sequences make a
            # frame serve as both, and there is no correct encoder to pick, so
            # raise rather than quietly encode it twice or drop it.
            n = self.num_preds
            if t != 2 * n:
                raise ValueError(
                    f"twin encoder needs the context and target frames to partition the "
                    f"sequence, i.e. t == 2 * num_preds, but got t={t}, num_preds={n}. "
                    f"With t={t} a frame would be both context and target (or neither)."
                )
            ctx_emb, ctx_patch = self._encode_frames(self.encoder, pixels[:, :t - n], b)
            tgt_emb, tgt_patch = self._encode_frames(self.target_encoder, pixels[:, n:], b)
            # ctx occupies frames [0, t-n) and tgt [n, t), contiguous and in
            # order, so a plain cat restores the original frame ordering.
            emb = torch.cat([ctx_emb, tgt_emb], dim=1)
            patch_emb = torch.cat([ctx_patch, tgt_patch], dim=1)

        info["emb"] = emb
        info["patch_emb"] = patch_emb

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)

        With residual_predict=True the network predicts the CHANGE and it is
        added back to the input, so predict() returns `emb + f(emb, act_emb)`
        instead of `f(emb, act_emb)`.

        Why (2026-08-26): on this data the trivial "copy t0 forward" baseline is
        strong -- most brains barely change between visits -- and the model has
        been unable to beat it (clean-pair ratio 1.0143 [1.0047, 1.0244] after
        Stage-2). Predicting absolutely forces the network to spend capacity
        reconstructing the 192-d input before it can model any change, and that
        reconstruction error is roughly the margin it is losing by. Residual
        makes the identity map exact and free at init, so the whole predictor
        works on the change instead.

        This is also what the architecture already intends: ARPredictor uses
        AdaLN-zero (module.py:102), whose blocks start as an exact identity
        residual stack -- but pred_proj, a freshly-initialised 792k-param MLP
        with BatchNorm, then destroys that. This restores it end-to-end.

        Diagnostic value: the output IS the predicted change, so if it converges
        toward zero that says there is no predictable change signal in this
        encoder's embedding -- pointing at the encoder rather than the
        predictor, which the current formulation cannot distinguish.

        Defaults False so upstream LeWM (pusht/cube/tworoom) is unaffected.
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        if self.residual_predict:
            preds = preds + emb
        return preds

    def decode(self, patch_emb):
        """Decode a per-patch token grid back into pixels.
        patch_emb: (B, T, P, D) -- e.g. info["patch_emb"] from encode()
        returns: (B, T, C, H, W)
        """
        b = patch_emb.size(0)
        flat = rearrange(patch_emb, "b t p d -> (b t) p d")
        images = self.decoder(flat)  # (B*T, C, H, W)
        return rearrange(images, "(b t) c h w -> b t c h w", b=b)

    def decode_prediction(self, pred_emb, num_patches=256):
        """Bridge from the predictor's single pooled per-frame vector to the
        decoder's per-patch grid input. Upstream LeWM's predictor only ever
        outputs one pooled vector per frame -- broadcasting it across all
        `num_patches` positions is an honest reflection of that: no
        per-patch spatial detail is fabricated, every patch starts from the
        identical predicted vector and the decoder does whatever it can
        with that. Expect smoother/less detailed output than decode()'s
        real patch_emb, by design, not a bug.
        pred_emb: (B, T, D) -- e.g. JEPA.predict()'s output
        returns: (B, T, C, H, W)
        """
        broadcast = pred_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)  # (B,T,P,D)
        return self.decode(broadcast)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
