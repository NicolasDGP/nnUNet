# This thing trains the whole network with the transformer query decoder and everything

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch._dynamo import OptimizedModule
except Exception:  
    OptimizedModule = () 
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

# Network wrapper. Forward returns segmentation logits

class _QueryDecoderWrapper(nn.Module):
    """
    Wrap UNetResSE3DWithQueryDecoder so that:
      - forward(x) returns ONLY seg_logits (tensor or list for deep supervision) -> nnU-Net compatible
      - forward_dict(x) returns the full dict (seg + queries) -> used by our custom train/val steps
    """

    def __init__(self, query_model: nn.Module):
        super().__init__()
        self.query_model = query_model

        # nnU-Net predictor expects `inference_apply_nonlin` on the network.
        bb = getattr(query_model, "backbone", None)
        if bb is not None and hasattr(bb, "inference_apply_nonlin"):
            self.inference_apply_nonlin = bb.inference_apply_nonlin  

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        out = self.query_model(x)
        return out["seg_logits"]

    def forward_dict(self, x: torch.Tensor, **kwargs) -> dict:
        return self.query_model(x, **kwargs)


# Query auxiliary loss (semantic-GT only; no instance matching required)

@dataclass
class QueryLossWeights:
    union_bce: float = 1.0
    union_dice: float = 1.0
    cls: float = 0.5
    quality: float = 0.25
    best_iou: float = 0.25


class QueryAuxLoss(nn.Module):
    """
    Auxiliary supervision for the query head using only semantic segmentation ground truth.

    It trains:
      - union of query masks vs foreground GT (BCE + Dice)
      - at least one query overlaps the foreground (best IoU term)
      - query classification/quality using pseudo targets derived from IoU (DETACHED)
    """

    def __init__(
        self,
        ignore_label: Optional[int],
        has_regions: bool,
        pos_iou_thresh: float = 0.10,
        weights: QueryLossWeights = QueryLossWeights(),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.has_regions = bool(has_regions)
        self.pos_iou_thresh = float(pos_iou_thresh)
        self.w = weights
        self.eps = float(eps)

    @staticmethod
    def _first_target(target: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        return target[0] if isinstance(target, list) else target

    def _foreground_mask(self, target: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        t = self._first_target(target)

        # Standard nnU-Net: label map is (B,1,...) with integer labels.
        if not self.has_regions:
            if t.ndim >= 2 and t.shape[1] == 1:
                t = t[:, 0]
            fg = t > 0
            if self.ignore_label is not None:
                fg = fg & (t != int(self.ignore_label))
            return fg

        # Region-based: targets are typically (B,C,...) one-hot-ish
        if t.ndim < 2:
            raise ValueError("Region training expects target with channel dim")
        if self.ignore_label is not None and t.shape[1] > 1:
            union = (t[:, :-1] > 0.5).any(dim=1)  # best-effort: last channel might be ignore
        else:
            union = (t > 0.5).any(dim=1)
        return union

    def forward(
        self,
        outputs: dict,
        target: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, dict]:
        mask_logits: torch.Tensor = outputs["mask_logits"]      # (B, N, D, H, W)
        class_logits: torch.Tensor = outputs["class_logits"]    # (B, N, 2)
        mask_quality: torch.Tensor = outputs["mask_quality"]    # (B, N, 1)

        fg = self._foreground_mask(target).to(mask_logits.dtype)  # (B, D, H, W)
        fg = fg.unsqueeze(1)  # (B, 1, D, H, W)

        p = torch.sigmoid(mask_logits)  # (B, N, D, H, W)

        # IoU per query (DETACHED for pseudo labels)
        inter = (p * fg).flatten(2).sum(-1)
        union = (p + fg - p * fg).flatten(2).sum(-1) + self.eps
        iou = inter / union                      # (B, N)
        iou_detached = iou.detach().clamp(0, 1)  # pseudo target

        # Union supervision via smooth-max (logsumexp)
        union_logits = torch.logsumexp(mask_logits, dim=1)  # (B, D, H, W)
        fg_flat = fg[:, 0]

        union_bce = F.binary_cross_entropy_with_logits(union_logits, fg_flat, reduction="mean")

        union_prob = torch.sigmoid(union_logits)
        num = 2.0 * (union_prob * fg_flat).sum() + self.eps
        den = (union_prob + fg_flat).sum() + self.eps
        union_dice = 1.0 - (num / den)

        best_iou = 1.0 - iou.max(dim=1).values.mean()

        cls_target = (iou_detached > self.pos_iou_thresh).long()
        cls_loss = F.cross_entropy(class_logits.view(-1, 2), cls_target.view(-1), reduction="mean")

        qual_logits = mask_quality[..., 0]
        qual_loss = F.binary_cross_entropy_with_logits(qual_logits, iou_detached, reduction="mean")

        loss = (
            self.w.union_bce * union_bce
            + self.w.union_dice * union_dice
            + self.w.best_iou * best_iou
            + self.w.cls * cls_loss
            + self.w.quality * qual_loss
        )

        comps = {
            "q_union_bce": union_bce.detach(),
            "q_union_dice": union_dice.detach(),
            "q_best_iou": best_iou.detach(),
            "q_cls": cls_loss.detach(),
            "q_quality": qual_loss.detach(),
        }
        return loss, comps


# Trainer

class nnUNetTrainerResSEWithQueryDecoder(nnUNetTrainer):
    """
    Trainer for UNetResSE3DWithQueryDecoder.

    - Keeps nnU-Net expectations intact: network(x) -> segmentation logits
    - Trains query head via network.forward_dict(x) in train/val steps
    """

    BACKBONE_CLASS_NAME: str = "nnunetv2.custom_nets.unet_full_simple.UNetResSE3D"
    QUERY_MODEL_CLASS_NAME: str = "nnunetv2.custom_nets.unet_full_simple.UNetResSE3DWithQueryDecoder"

    # Practical defaults (3D mask logits are expensive; bump if you have VRAM)
    N_QUERIES: int = 16
    N_HEADS: int = 8
    N_LAYERS: Optional[int] = None
    D_MODEL: Optional[int] = None
    DIM_FFN: Optional[int] = None
    DROPOUT: float = 0.0

    QUERY_LOSS_WEIGHT: float = 0.1
    QUERY_POS_IOU_THRESH: float = 0.10
    QUERY_LOSS_COMPONENT_WEIGHTS: QueryLossWeights = QueryLossWeights()

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # 1) Build plan-driven backbone
        backbone = nnUNetTrainer.build_network_architecture(
            nnUNetTrainerResSEWithQueryDecoder.BACKBONE_CLASS_NAME,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

        # 2) Build query model around backbone
        import pydoc

        query_cls = pydoc.locate(nnUNetTrainerResSEWithQueryDecoder.QUERY_MODEL_CLASS_NAME)
        if query_cls is None:
            raise ImportError(
                f"Could not locate {nnUNetTrainerResSEWithQueryDecoder.QUERY_MODEL_CLASS_NAME}. "
                f"Is it importable?"
            )

        query_model = query_cls(
            backbone=backbone,
            n_queries=int(nnUNetTrainerResSEWithQueryDecoder.N_QUERIES),
            n_heads=int(nnUNetTrainerResSEWithQueryDecoder.N_HEADS),
            n_layers=nnUNetTrainerResSEWithQueryDecoder.N_LAYERS,
            d_model=nnUNetTrainerResSEWithQueryDecoder.D_MODEL,
            dim_ffn=nnUNetTrainerResSEWithQueryDecoder.DIM_FFN,
            dropout=float(nnUNetTrainerResSEWithQueryDecoder.DROPOUT),
        )

        # 3) Wrap so `forward` returns seg logits, and `forward_dict` returns full dict
        return _QueryDecoderWrapper(query_model)

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Default nnU-Net implementation assumes `mod.decoder.deep_supervision`.
        Our architecture stores it at `mod.query_model.backbone.deep_supervision`.
        """
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        if hasattr(mod, "query_model") and hasattr(mod.query_model, "backbone"):
            bb = mod.query_model.backbone
            if hasattr(bb, "deep_supervision"):
                bb.deep_supervision = enabled
                return

        # Fallback for safety (if someone swaps wrapper)
        if hasattr(mod, "decoder") and hasattr(mod.decoder, "deep_supervision"):
            mod.decoder.deep_supervision = enabled

    def initialize(self):  # called by nnU-Net
        super().initialize()
        self._init_query_loss()

    def _init_query_loss(self):
        ignore = getattr(self.label_manager, "ignore_label", None)
        has_regions = bool(getattr(self.label_manager, "has_regions", False))
        self.query_aux_loss = QueryAuxLoss(
            ignore_label=ignore,
            has_regions=has_regions,
            pos_iou_thresh=float(self.QUERY_POS_IOU_THRESH),
            weights=self.QUERY_LOSS_COMPONENT_WEIGHTS,
        )

    def _unwrap_network(self) -> nn.Module:
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def _forward_dict(self, x: torch.Tensor) -> dict:
        mod = self._unwrap_network()
        if not hasattr(mod, "forward_dict"):
            raise RuntimeError(
                "Network does not expose forward_dict; did build_network_architecture return _QueryDecoderWrapper?"
            )
        return mod.forward_dict(x)

    @staticmethod
    def _first_target(target: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        return target[0] if isinstance(target, list) else target

    @staticmethod
    def _seg_fullres_logits(seg_logits: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        return seg_logits[0] if isinstance(seg_logits, list) else seg_logits

    def _compute_tp_fp_fn_hard(
        self,
        seg_logits: Union[torch.Tensor, List[torch.Tensor]],
        target: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hard TP/FP/FN per foreground class (background excluded), returned on CPU.
        """
        logits = self._seg_fullres_logits(seg_logits)
        t = self._first_target(target)

        ignore = getattr(self.label_manager, "ignore_label", None)
        has_regions = bool(getattr(self.label_manager, "has_regions", False))

        if not has_regions:
            if t.ndim >= 2 and t.shape[1] == 1:
                gt = t[:, 0].long()
            else:
                gt = t.long()

            pred = logits.argmax(1).long()
            n_cls = int(logits.shape[1])

            valid = torch.ones_like(gt, dtype=torch.bool)
            if ignore is not None:
                valid = gt != int(ignore)

            tp, fp, fn = [], [], []
            for c in range(1, n_cls):
                pred_c = (pred == c) & valid
                gt_c = (gt == c) & valid
                tp.append((pred_c & gt_c).sum())
                fp.append((pred_c & (~gt_c)).sum())
                fn.append(((~pred_c) & gt_c).sum())
            return torch.stack(tp).cpu(), torch.stack(fp).cpu(), torch.stack(fn).cpu()

        # Region-based (best-effort; assumes last channel could be ignore)
        if t.ndim < 2:
            raise ValueError("Region training expects target with channel dim")

        gt = (t > 0.5)
        pred = (torch.sigmoid(logits) > 0.5)

        if ignore is not None and gt.shape[1] > 1:
            ignore_mask = gt[:, -1]
            gt = gt[:, :-1]
            pred = pred[:, :-1]
            valid = ~ignore_mask
        else:
            valid = torch.ones_like(gt[:, 0], dtype=torch.bool)

        tp = (pred & gt & valid.unsqueeze(1)).flatten(2).sum(-1).sum(0)
        fp = (pred & (~gt) & valid.unsqueeze(1)).flatten(2).sum(-1).sum(0)
        fn = ((~pred) & gt & valid.unsqueeze(1)).flatten(2).sum(-1).sum(0)
        return tp.cpu(), fp.cpu(), fn.cpu()

    def train_step(self, batch: dict) -> dict:
        if not hasattr(self, "query_aux_loss"):
            self._init_query_loss()

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            out_dict = self._forward_dict(data)
            seg_logits = out_dict["seg_logits"]

            l_seg = self.loss(seg_logits, target)
            l_q, q_comps = self.query_aux_loss(out_dict, target)
            l = l_seg + float(self.QUERY_LOSS_WEIGHT) * l_q

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        out = {
            "loss": float(l.detach().cpu()),
            "loss_seg": float(l_seg.detach().cpu()),
            "loss_query": float(l_q.detach().cpu()),
        }
        for k, v in q_comps.items():
            out[k] = float(v.cpu())
        return out

    @torch.no_grad()
    def validation_step(self, batch: dict) -> dict:
        if not hasattr(self, "query_aux_loss"):
            self._init_query_loss()

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            out_dict = self._forward_dict(data)
            seg_logits = out_dict["seg_logits"]

            l_seg = self.loss(seg_logits, target)
            l_q, _ = self.query_aux_loss(out_dict, target)
            l = l_seg + float(self.QUERY_LOSS_WEIGHT) * l_q

        tp, fp, fn = self._compute_tp_fp_fn_hard(seg_logits, target)

        return {
            "loss": float(l.detach().cpu()),
            "tp_hard": tp.numpy(),
            "fp_hard": fp.numpy(),
            "fn_hard": fn.numpy(),
        }

#__all__ = ["nnUNetTrainerResSEWithQueryDecoder", "QueryAuxLoss", "QueryLossWeights"]