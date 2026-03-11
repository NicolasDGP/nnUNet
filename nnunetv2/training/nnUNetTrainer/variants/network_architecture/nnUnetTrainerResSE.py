"""
nnUNetTrainerResSE
==================
Custom nnU-Net trainer that uses UNetResSE3D (residual SE encoder-decoder)
combined with the TransformerQueryDecoder.

Loss = seg_loss (Dice + CE)
     + lambda_cls  * query_classification_loss (CrossEntropy)
     + lambda_mask * query_mask_loss           (Dice + Focal)

To train (once dataset is preprocessed):
    nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerResSE
"""

from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.custom_modules.unet_resse import UNetResSE3D
from nnunetv2.custom_modules.transformer_query_decoder import (
    TransformerQueryDecoder,
    get_final_probability_map,
)
from nnunetv2.utilities.helpers import dummy_context


# ── tiny helper losses ────────────────────────────────────────────────────────

def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                   smooth: float = 1e-5) -> torch.Tensor:
    """pred and target both [B, C, ...], pred is raw logits."""
    pred   = torch.softmax(pred, dim=1)
    target = F.one_hot(target.long().squeeze(1),
                       num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    dims   = tuple(range(2, pred.ndim))
    inter  = (pred * target).sum(dims)
    union  = pred.sum(dims) + target.sum(dims)
    dice   = 1.0 - (2.0 * inter + smooth) / (union + smooth)
    return dice.mean()


def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               gamma: float = 2.0) -> torch.Tensor:
    """pred: [B, C, ...] logits, target: [B, 1, ...] long."""
    ce   = F.cross_entropy(pred, target.squeeze(1).long(), reduction='none')
    pt   = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def query_mask_loss(mask_logits: torch.Tensor,
                    gt_mask: torch.Tensor,
                    matched_query_idx: torch.Tensor) -> torch.Tensor:
    """
    Dice + Focal loss on the masks of queries that were matched to GT.

    mask_logits      : [B, N, DHW]
    gt_mask          : [B, 1, D, H, W]  — binary aneurysm mask
    matched_query_idx: [B] — which query index is matched per batch item
                       (-1 means no aneurysm in this sample)
    """
    B, N, DHW = mask_logits.shape
    gt_flat = (gt_mask > 0).float().view(B, DHW)   # [B, DHW]

    total = torch.tensor(0.0, device=mask_logits.device)
    count = 0
    for b in range(B):
        idx = matched_query_idx[b].item()
        if idx < 0:
            continue                               # no aneurysm — skip
        pred = mask_logits[b, idx]                 # [DHW]  raw logit
        tgt  = gt_flat[b]                          # [DHW]  binary

        # Dice on sigmoid output
        p      = torch.sigmoid(pred)
        smooth = 1e-5
        inter  = (p * tgt).sum()
        dice_l = 1.0 - (2 * inter + smooth) / (p.sum() + tgt.sum() + smooth)

        # Focal on raw logit
        focal_l = F.binary_cross_entropy_with_logits(
            pred, tgt, reduction='none'
        )
        pt      = torch.exp(-focal_l)
        focal_l = ((1 - pt) ** 2 * focal_l).mean()

        total += dice_l + focal_l
        count += 1

    return total / max(count, 1)


def match_queries_to_gt(mask_logits: torch.Tensor,
                        gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Greedy IoU match: for each batch item find the query whose sigmoid mask
    overlaps the GT the most.  Returns index tensor [B], -1 if no GT.
    """
    B, N, DHW = mask_logits.shape
    gt_flat    = (gt_mask > 0).float().view(B, DHW)   # [B, DHW]
    matched    = torch.full((B,), -1, dtype=torch.long,
                            device=mask_logits.device)

    for b in range(B):
        if gt_flat[b].sum() == 0:
            continue                                   # no aneurysm
        probs    = torch.sigmoid(mask_logits[b])       # [N, DHW]
        tgt      = gt_flat[b]                          # [DHW]
        inter    = (probs * tgt).sum(dim=1)            # [N]
        union    = probs.sum(dim=1) + tgt.sum() - inter
        iou      = inter / (union + 1e-5)
        matched[b] = iou.argmax()

    return matched


# ── Trainer ───────────────────────────────────────────────────────────────────

class nnUNetTrainerResSE(nnUNetTrainer):
    """
    Drop-in nnU-Net trainer using UNetResSE3D + TransformerQueryDecoder.
    Everything else (data loading, augmentation, LR schedule) is inherited
    from the standard nnUNetTrainer.
    """

    # loss weights — tune these if needed
    LAMBDA_CLS  = 0.5
    LAMBDA_MASK = 1.0

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build and return the full model (UNet + Transformer head)."""

        unet = UNetResSE3D(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            base_features=32,
            depth=4,
            se_reduction=8,
            return_decoder_features=True,
        )

        # We wrap both modules in a small container so nnUNet sees ONE model
        tqd = TransformerQueryDecoder(
            num_queries=16,
            feature_dim=32,      # matches base_features
            num_heads=4,
            num_scales=3,        # depth - 1
            num_classes=num_output_channels,
            ffn_dim=128,
        )

        model = _ResSEWithTransformer(unet, tqd)
        return model

    # ── training step ─────────────────────────────────────────────────────────
    def train_step(self, batch: dict) -> dict:
        data   = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]

        # nnUNet can pass target as list (deep supervision) — take first
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        ctx = (autocast(self.device.type, enabled=True)
               if self.device.type == "cuda" else dummy_context())

        with ctx:
            seg_logits, class_logits, mask_logits, mask_quality = \
                self.network(data)

            # 1. Standard segmentation loss
            loss_seg = soft_dice_loss(seg_logits, target) + \
                       focal_loss(seg_logits, target)

            # 2. Match queries to GT, compute mask loss
            matched_idx = match_queries_to_gt(mask_logits, target)
            loss_mask   = query_mask_loss(mask_logits, target, matched_idx)

            # 3. Query classification loss
            #    matched query → class 1 (aneurysm), rest → class 0
            B, N, _ = class_logits.shape
            cls_targets = torch.zeros(B, N, dtype=torch.long,
                                      device=self.device)
            for b in range(B):
                if matched_idx[b] >= 0:
                    cls_targets[b, matched_idx[b]] = 1
            loss_cls = F.cross_entropy(
                class_logits.view(B * N, -1),
                cls_targets.view(B * N),
            )

            # 4. Total
            loss = (loss_seg
                    + self.LAMBDA_CLS  * loss_cls
                    + self.LAMBDA_MASK * loss_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": loss.detach().cpu().numpy()}

    def set_deep_supervision_enabled(self, enabled: bool):
        pass   # we handle this ourselves


# ── thin wrapper so nnUNet sees one .network object ──────────────────────────

class _ResSEWithTransformer(nn.Module):
    """Bundles UNetResSE3D and TransformerQueryDecoder into one nn.Module."""

    def __init__(self, unet: UNetResSE3D, tqd: TransformerQueryDecoder):
        super().__init__()
        self.unet = unet
        self.tqd  = tqd

    def forward(self, x: torch.Tensor):
        seg_logits, decoder_features = self.unet(x)
        class_logits, mask_logits, mask_quality = self.tqd(decoder_features)
        return seg_logits, class_logits, mask_logits, mask_quality