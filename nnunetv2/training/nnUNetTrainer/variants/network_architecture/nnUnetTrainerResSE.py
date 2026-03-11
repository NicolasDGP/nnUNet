"""
nnUNetTrainerResSE
==================
Trainer that plugs UNetResSE3D + TransformerQueryDecoder into nnUNet v2.

The trainer:
  1. Builds the model using nnUNet plans (so patch size, features, strides
     are all auto-configured exactly like the baseline)
  2. Runs the full forward pass through both modules
  3. Computes three losses:
       loss_seg   = Dice + Focal  on the segmentation head output
       loss_mask  = Dice + Focal  on the best matching query mask
       loss_cls   = CrossEntropy  on query class logits
  4. During validation / inference: only returns seg_logits so nnUNet's
     built-in evaluation pipeline works unchanged

Train command:
    nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerResSE
"""

from __future__ import annotations
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.custom_nets.transformer_query_decoder import  (
    TransformerQueryDecoder,
    get_final_probability_map,
)
from nnunetv2.utilities.helpers import dummy_context


# ─────────────────────────────────────────────────────────────────────────────
#  Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def _soft_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    pred   : [B, C, D, H, W]  raw logits
    target : [B, 1, D, H, W]  long labels
    """
    pred   = torch.softmax(pred, dim=1)
    C      = pred.shape[1]
    target_oh = F.one_hot(target.squeeze(1).long(), C).permute(0, 4, 1, 2, 3).float()
    dims   = tuple(range(2, pred.ndim))
    inter  = (pred * target_oh).sum(dims)
    union  = pred.sum(dims) + target_oh.sum(dims)
    return (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()


def _focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce  = F.cross_entropy(pred, target.squeeze(1).long(), reduction="none")
    pt  = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def _match_queries_to_gt(
    mask_logits: torch.Tensor,
    gt_mask:     torch.Tensor,
) -> torch.Tensor:
    """
    Greedy IoU match — for each batch item find the query whose
    sigmoid mask best overlaps the GT aneurysm region.

    Returns matched : [B]  long, value -1 means no aneurysm in that item.
    """
    B, N, DHW = mask_logits.shape
    gt_flat   = (gt_mask > 0).float().view(B, DHW)
    matched   = torch.full((B,), -1, dtype=torch.long, device=mask_logits.device)

    for b in range(B):
        if gt_flat[b].sum() == 0:
            continue
        probs  = torch.sigmoid(mask_logits[b])          # [N, DHW]
        tgt    = gt_flat[b]                              # [DHW]
        inter  = (probs * tgt).sum(dim=1)               # [N]
        union  = probs.sum(dim=1) + tgt.sum() - inter
        matched[b] = (inter / (union + 1e-5)).argmax()

    return matched


def _query_mask_loss(
    mask_logits:  torch.Tensor,
    gt_mask:      torch.Tensor,
    matched_idx:  torch.Tensor,
) -> torch.Tensor:
    """Dice + Focal on the matched query mask only."""
    B, N, DHW = mask_logits.shape
    gt_flat   = (gt_mask > 0).float().view(B, DHW)
    total = torch.tensor(0.0, device=mask_logits.device)
    count = 0

    for b in range(B):
        idx = matched_idx[b].item()
        if idx < 0:
            continue
        pred = mask_logits[b, int(idx)]    # [DHW]  logits
        tgt  = gt_flat[b]                  # [DHW]

        # Dice
        p      = torch.sigmoid(pred)
        smooth = 1e-5
        dice_l = 1.0 - (2*(p*tgt).sum() + smooth) / (p.sum() + tgt.sum() + smooth)

        # Focal
        bce     = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        focal_l = ((1 - torch.exp(-bce)) ** 2 * bce).mean()

        total += dice_l + focal_l
        count += 1

    return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _ResSEWithTransformer(nn.Module):
    """
    Wraps UNetResSE3D + TransformerQueryDecoder into one nn.Module.

    Training  → returns (seg_logits, class_logits, mask_logits, mask_quality)
    Inference → returns seg_logits only  (nnUNet predict pipeline works unchanged)
    """

    def __init__(self, unet: nn.Module, tqd: TransformerQueryDecoder):
        super().__init__()
        self.unet = unet
        self.tqd  = tqd

    def forward(self, x: torch.Tensor):
        # always get decoder features during training
        if self.training:
            seg_logits, decoder_feats = self.unet(x, return_decoder_features=True)
            class_logits, mask_logits, mask_quality = self.tqd(decoder_feats)
            return seg_logits, class_logits, mask_logits, mask_quality
        else:
            # inference — standard nnUNet output only
            seg_logits = self.unet(x, return_decoder_features=False)
            return seg_logits


# ─────────────────────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainerResSE(nnUNetTrainer):
    """
    nnU-Net v2 trainer for UNetResSE3D + TransformerQueryDecoder.

    Loss weights (tune if needed):
        LAMBDA_CLS  = weight on query classification loss
        LAMBDA_MASK = weight on query mask loss
    """

    LAMBDA_CLS:  float = 0.5
    LAMBDA_MASK: float = 1.0

    # ── build model ───────────────────────────────────────────────────────────
    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        # import here to avoid circular imports at module level
        from nnunetv2.custom_nets.unet_ressne_feature_maps import UNetResSE3D

        # pull architecture config straight from nnUNet plans
        # so patch size, features, strides are auto-configured
        cfg  = self.configuration_manager
        plan = cfg.configuration

        n_stages         = len(plan["conv_kernel_sizes"])
        features         = plan.get("unet_max_num_features", 320)
        features_per_stage = plan.get(
            "features_per_stage",
            [min(features, 32 * (2**i)) for i in range(n_stages)],
        )
        strides          = plan["pool_op_kernel_sizes"]
        # first stage never pools
        strides_full     = [[1,1,1]] + list(strides)
        kernel_sizes     = plan["conv_kernel_sizes"]

        unet = UNetResSE3D(
            input_channels        = num_input_channels,
            n_stages              = n_stages,
            features_per_stage    = features_per_stage,
            conv_op               = nn.Conv3d,
            kernel_sizes          = kernel_sizes,
            strides               = strides_full,
            n_conv_per_stage      = 2,
            n_conv_per_stage_decoder = 2,
            num_classes           = num_output_channels,
            norm_op               = nn.InstanceNorm3d,
            norm_op_kwargs        = {"eps": 1e-5, "affine": True},
            nonlin                = nn.LeakyReLU,
            nonlin_kwargs         = {"inplace": True},
            deep_supervision      = False,
            se_reduction          = 16,
        )

        # decoder has n_stages-1 levels
        # features_per_stage is already in ascending order
        # decoder outputs full-res-first so channels are features_per_stage[0..n_stages-2]
        if isinstance(features_per_stage, int):
            feat_list = [features_per_stage * (2**i) for i in range(n_stages)]
        else:
            feat_list = list(features_per_stage)

        # decoder feature channels, full-res-first
        decoder_channels = feat_list[:n_stages-1]   # [feat0, feat1, ..., feat_{n-2}]

        tqd = TransformerQueryDecoder(
            feature_channels = decoder_channels,
            num_queries      = 16,
            embed_dim        = 256,
            num_heads        = 8,
            ffn_dim          = 512,
            num_classes      = num_output_channels,
            dropout          = 0.0,
        )

        return _ResSEWithTransformer(unet, tqd)

    # ── training step ─────────────────────────────────────────────────────────
    def train_step(self, batch: dict) -> dict:
        data   = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = target[0]          # take full-res target only
        target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        ctx = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )

        with ctx:
            seg_logits, class_logits, mask_logits, mask_quality = self.network(data)

            # 1 — segmentation loss on UNet output
            loss_seg = _soft_dice_loss(seg_logits, target) + \
                       _focal_loss(seg_logits, target)

            # 2 — match queries → GT, then mask loss
            matched   = _match_queries_to_gt(mask_logits, target)
            loss_mask = _query_mask_loss(mask_logits, target, matched)

            # 3 — query classification loss
            B, N, _ = class_logits.shape
            cls_tgt  = torch.zeros(B, N, dtype=torch.long, device=self.device)
            for b in range(B):
                if matched[b] >= 0:
                    cls_tgt[b, matched[b]] = 1
            loss_cls = F.cross_entropy(
                class_logits.view(B * N, -1),
                cls_tgt.view(B * N),
            )

            total_loss = (
                loss_seg
                + self.LAMBDA_CLS  * loss_cls
                + self.LAMBDA_MASK * loss_mask
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": total_loss.detach().cpu().numpy()}

    # ── validation step ───────────────────────────────────────────────────────
    def validation_step(self, batch: dict) -> dict:
        data   = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            # network is in eval mode → returns seg_logits only
            seg_logits = self.network(data)

        # compute tp/fp/fn for Dice tracking
        pred      = torch.softmax(seg_logits, dim=1)
        C         = pred.shape[1]
        tgt_oh    = F.one_hot(target.squeeze(1).long(), C).permute(0,4,1,2,3).float()
        axes      = tuple(range(2, pred.ndim))

        tp = (pred * tgt_oh).sum(axes).detach().cpu().numpy()
        fp = (pred * (1 - tgt_oh)).sum(axes).detach().cpu().numpy()
        fn = ((1 - pred) * tgt_oh).sum(axes).detach().cpu().numpy()

        return {"tp_hard": tp, "fp_hard": fp, "fn_hard": fn}

    def set_deep_supervision_enabled(self, enabled: bool):
        pass    # handled internally