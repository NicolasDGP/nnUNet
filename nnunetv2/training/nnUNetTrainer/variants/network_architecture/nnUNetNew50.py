from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch._dynamo import OptimizedModule
except Exception:
    OptimizedModule = ()

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


class _QueryDecoderWrapper(nn.Module):
    """
    nnU-Net compatible wrapper.

    network(x) returns segmentation logits only.
    network(x, return_dict=True) returns the full query model dict.

    Important:
    train_step calls self.network(..., return_dict=True), NOT the unwrapped module.
    This keeps DDP gradient synchronization intact.
    """

    def __init__(self, query_model: nn.Module):
        super().__init__()
        self.query_model = query_model

        bb = getattr(query_model, "backbone", None)
        if bb is not None and hasattr(bb, "inference_apply_nonlin"):
            self.inference_apply_nonlin = bb.inference_apply_nonlin

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, List[torch.Tensor], dict]:
        out = self.query_model(x, **kwargs)
        return out if return_dict else out["seg_logits"]

    def forward_dict(self, x: torch.Tensor, **kwargs: Any) -> dict:
        return self.forward(x, return_dict=True, **kwargs)


@dataclass(frozen=True)
class QueryLossWeights:
    union_bce: float = 1.0
    union_dice: float = 1.0
    cls: float = 0.5
    quality: float = 0.25
    best_iou: float = 0.25


class QueryAuxLoss(nn.Module):
    """
    Stable auxiliary supervision for the query head using semantic segmentation GT.

    Fixes:
    - query loss computed in float32
    - ignore-label voxels masked out
    - GT resized to query mask resolution if needed
    - stable normalized smooth-OR query union
    - empty-foreground patches do not receive bogus best-IoU pressure
    """

    def __init__(
        self,
        ignore_label: Optional[int],
        has_regions: bool,
        pos_iou_thresh: float = 0.10,
        weights: Optional[QueryLossWeights] = None,
        eps: float = 1e-6,
        smooth_or_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.ignore_label = ignore_label
        self.has_regions = bool(has_regions)
        self.pos_iou_thresh = float(pos_iou_thresh)
        self.w = weights if weights is not None else QueryLossWeights()
        self.eps = float(eps)
        self.smooth_or_temperature = float(smooth_or_temperature)

    @staticmethod
    def _first_target(
        target: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        return target[0] if isinstance(target, (list, tuple)) else target

    def _foreground_and_valid_mask(
        self,
        target: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
        spatial_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self._first_target(target)

        if not self.has_regions:
            if t.ndim >= 2 and t.shape[1] == 1:
                t = t[:, 0]

            valid = torch.ones_like(t, dtype=torch.bool)

            if self.ignore_label is not None:
                valid = t.ne(int(self.ignore_label))

            fg = t.gt(0) & valid

        else:
            if t.ndim < 2:
                raise ValueError("Region training expects target with channel dim")

            if self.ignore_label is not None and t.shape[1] > 1:
                ignore_mask = t[:, -1].gt(0.5)
                fg = t[:, :-1].gt(0.5).any(dim=1)
                valid = ~ignore_mask
            else:
                fg = t.gt(0.5).any(dim=1)
                valid = torch.ones_like(fg, dtype=torch.bool)

        if tuple(fg.shape[1:]) != tuple(spatial_shape):
            fg = F.interpolate(
                fg[:, None].float(),
                size=spatial_shape,
                mode="nearest",
            )[:, 0].bool()

            valid = F.interpolate(
                valid[:, None].float(),
                size=spatial_shape,
                mode="nearest",
            )[:, 0].bool()

        return fg, valid

    def forward(
        self,
        outputs: dict,
        target: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, dict]:
        mask_logits = outputs["mask_logits"].float()
        class_logits = outputs["class_logits"].float()
        mask_quality = outputs["mask_quality"].float()

        if mask_logits.ndim != 5:
            raise ValueError(f"mask_logits must be (B,N,D,H,W), got {tuple(mask_logits.shape)}")

        if class_logits.ndim != 3 or class_logits.shape[-1] != 2:
            raise ValueError(f"class_logits must be (B,N,2), got {tuple(class_logits.shape)}")

        if mask_quality.ndim != 3 or mask_quality.shape[-1] != 1:
            raise ValueError(f"mask_quality must be (B,N,1), got {tuple(mask_quality.shape)}")

        if not (
            torch.isfinite(mask_logits).all()
            and torch.isfinite(class_logits).all()
            and torch.isfinite(mask_quality).all()
        ):
            raise FloatingPointError("Non-finite query decoder output before QueryAuxLoss.")

        fg_bool, valid_bool = self._foreground_and_valid_mask(
            target,
            spatial_shape=tuple(mask_logits.shape[2:]),
        )

        fg = fg_bool[:, None].to(mask_logits.dtype)
        valid = valid_bool[:, None].to(mask_logits.dtype)

        valid_count = valid[:, 0].sum().clamp_min(1.0)

        p = torch.sigmoid(mask_logits)
        p_valid = p * valid
        fg_valid = fg * valid

        inter = (p_valid * fg_valid).flatten(2).sum(-1)
        union = (p_valid + fg_valid - p_valid * fg_valid).flatten(2).sum(-1)
        union = union.clamp_min(self.eps)

        iou = (inter / union).clamp(0.0, 1.0)
        iou_detached = iou.detach()

        tau = max(self.smooth_or_temperature, self.eps)
        n_queries = int(mask_logits.shape[1])

        union_logits = (
            tau * torch.logsumexp(mask_logits / tau, dim=1)
            - tau * math.log(max(n_queries, 1))
        )

        fg_flat = fg[:, 0]
        valid_flat = valid[:, 0]

        union_bce_raw = F.binary_cross_entropy_with_logits(
            union_logits,
            fg_flat,
            reduction="none",
        )
        union_bce = (union_bce_raw * valid_flat).sum() / valid_count

        union_prob = torch.sigmoid(union_logits) * valid_flat
        fg_dice = fg_flat * valid_flat

        dice_num = 2.0 * (union_prob * fg_dice).sum() + self.eps
        dice_den = (union_prob + fg_dice).sum() + self.eps
        union_dice = 1.0 - dice_num / dice_den

        has_fg = fg_dice.flatten(1).sum(-1) > 0
        if has_fg.any():
            best_iou = 1.0 - iou[has_fg].max(dim=1).values.mean()
        else:
            best_iou = mask_logits.sum() * 0.0

        cls_target = iou_detached.gt(self.pos_iou_thresh).long()
        cls_loss = F.cross_entropy(
            class_logits.reshape(-1, 2),
            cls_target.reshape(-1),
            reduction="mean",
        )

        qual_logits = mask_quality[..., 0]
        qual_loss = F.binary_cross_entropy_with_logits(
            qual_logits,
            iou_detached,
            reduction="mean",
        )

        loss = (
            self.w.union_bce * union_bce
            + self.w.union_dice * union_dice
            + self.w.best_iou * best_iou
            + self.w.cls * cls_loss
            + self.w.quality * qual_loss
        )

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite QueryAuxLoss.")

        comps = {
            "q_union_bce": union_bce.detach(),
            "q_union_dice": union_dice.detach(),
            "q_best_iou": best_iou.detach(),
            "q_cls": cls_loss.detach(),
            "q_quality": qual_loss.detach(),
        }

        return loss, comps


class nnUNetTrainerResSEWithQueryDecoder50(nnUNetTrainer):
    """
    Trainer for UNetResSE3DWithQueryDecoder.

    Trains for 50 epochs.

    Important:
    This class intentionally does NOT override __init__.
    Different nnU-Net versions have different __init__ signatures, and nnU-Net
    introspects trainer constructor signatures internally. Overriding __init__
    caused both:
    - KeyError: 'args'
    - TypeError: unexpected keyword argument 'unpack_dataset'

    Setting num_epochs in initialize() avoids those signature problems.
    """

    BACKBONE_CLASS_NAME: str = "nnunetv2.custom_nets.unet_full_simple.UNetResSE3D"
    QUERY_MODEL_CLASS_NAME: str = "nnunetv2.custom_nets.unet_full_simple.UNetResSE3DWithQueryDecoder"

    MAX_NUM_EPOCHS: int = 50

    N_QUERIES: int = 16
    N_HEADS: int = 8
    N_LAYERS: Optional[int] = None
    D_MODEL: Optional[int] = None
    DIM_FFN: Optional[int] = None
    DROPOUT: float = 0.0

    QUERY_LOSS_WEIGHT: float = 0.1
    QUERY_POS_IOU_THRESH: float = 0.10
    QUERY_LOSS_COMPONENT_WEIGHTS: QueryLossWeights = QueryLossWeights()
    QUERY_SMOOTH_OR_TEMPERATURE: float = 1.0

    GRAD_CLIP_MAX_NORM: float = 12.0
    VERIFY_PARAMETERS_AFTER_STEP: bool = True

    TRAIN_OUTPUT_KEYS: Tuple[str, ...] = (
        "loss",
        "loss_seg",
        "loss_query",
        "q_union_bce",
        "q_union_dice",
        "q_best_iou",
        "q_cls",
        "q_quality",
        "grad_norm",
        "skipped_step",
        "skipped_nonfinite_output",
        "skipped_floating_point_error",
        "skipped_nonfinite_loss",
        "skipped_nonfinite_grad",
    )

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        backbone = nnUNetTrainer.build_network_architecture(
            nnUNetTrainerResSEWithQueryDecoder50.BACKBONE_CLASS_NAME,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

        import pydoc

        query_cls = pydoc.locate(nnUNetTrainerResSEWithQueryDecoder50.QUERY_MODEL_CLASS_NAME)
        if query_cls is None:
            raise ImportError(
                f"Could not locate {nnUNetTrainerResSEWithQueryDecoder50.QUERY_MODEL_CLASS_NAME}. "
                f"Is it importable?"
            )

        query_model = query_cls(
            backbone=backbone,
            n_queries=int(nnUNetTrainerResSEWithQueryDecoder50.N_QUERIES),
            n_heads=int(nnUNetTrainerResSEWithQueryDecoder50.N_HEADS),
            n_layers=nnUNetTrainerResSEWithQueryDecoder50.N_LAYERS,
            d_model=nnUNetTrainerResSEWithQueryDecoder50.D_MODEL,
            dim_ffn=nnUNetTrainerResSEWithQueryDecoder50.DIM_FFN,
            dropout=float(nnUNetTrainerResSEWithQueryDecoder50.DROPOUT),
        )

        return _QueryDecoderWrapper(query_model)

    def initialize(self) -> None:
        super().initialize()
        self.num_epochs = int(self.MAX_NUM_EPOCHS)
        self._init_query_loss()

    def _init_query_loss(self) -> None:
        ignore = getattr(self.label_manager, "ignore_label", None)
        has_regions = bool(getattr(self.label_manager, "has_regions", False))

        self.query_aux_loss = QueryAuxLoss(
            ignore_label=ignore,
            has_regions=has_regions,
            pos_iou_thresh=float(self.QUERY_POS_IOU_THRESH),
            weights=self.QUERY_LOSS_COMPONENT_WEIGHTS,
            smooth_or_temperature=float(self.QUERY_SMOOTH_OR_TEMPERATURE),
        )

    def _unwrap_network(self) -> nn.Module:
        mod = self.network.module if self.is_ddp else self.network

        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        return mod

    def _forward_dict(self, x: torch.Tensor, **kwargs: Any) -> dict:
        out = self.network(x, return_dict=True, **kwargs)

        if not isinstance(out, dict):
            raise RuntimeError(
                "Network did not return a dict. "
                "Did build_network_architecture return _QueryDecoderWrapper?"
            )

        return out

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        mod = self._unwrap_network()

        if hasattr(mod, "query_model") and hasattr(mod.query_model, "backbone"):
            bb = mod.query_model.backbone
            if hasattr(bb, "deep_supervision"):
                bb.deep_supervision = bool(enabled)
                return

        if hasattr(mod, "decoder") and hasattr(mod.decoder, "deep_supervision"):
            mod.decoder.deep_supervision = bool(enabled)

    @staticmethod
    def _first_target(
        target: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        return target[0] if isinstance(target, (list, tuple)) else target

    @staticmethod
    def _seg_fullres_logits(
        seg_logits: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        return seg_logits[0] if isinstance(seg_logits, (list, tuple)) else seg_logits

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            if torch.is_tensor(x):
                x = x.detach().float()
                if x.numel() != 1:
                    x = x.mean()
                if not torch.isfinite(x):
                    return default
                return float(x.cpu())

            x = float(x)
            if not math.isfinite(x):
                return default
            return x

        except Exception:
            return default

    @classmethod
    def _blank_train_result(cls) -> dict:
        return {k: 0.0 for k in cls.TRAIN_OUTPUT_KEYS}

    @classmethod
    def _skipped_train_result(cls, reason: str) -> dict:
        out = cls._blank_train_result()
        out["skipped_step"] = 1.0

        key = f"skipped_{reason}"
        if key in out:
            out[key] = 1.0

        return out

    @staticmethod
    def _all_tensors_finite(obj: Any) -> bool:
        if torch.is_tensor(obj):
            return bool(torch.isfinite(obj).all().detach().cpu())

        if isinstance(obj, dict):
            return all(
                nnUNetTrainerResSEWithQueryDecoder50._all_tensors_finite(v)
                for v in obj.values()
            )

        if isinstance(obj, (list, tuple)):
            return all(
                nnUNetTrainerResSEWithQueryDecoder50._all_tensors_finite(v)
                for v in obj
            )

        return True

    def _parameters_are_finite(self) -> bool:
        mod = self._unwrap_network()

        for p in mod.parameters():
            if p is not None and not torch.isfinite(p).all():
                return False

        return True

    def _optimizer_step_with_finite_guard(self, loss: torch.Tensor) -> Tuple[bool, float]:
        if not torch.isfinite(loss.detach()):
            self.optimizer.zero_grad(set_to_none=True)
            return False, 0.0

        max_norm = float(self.GRAD_CLIP_MAX_NORM)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                max_norm=max_norm,
                error_if_nonfinite=False,
            )

            grad_norm_float = self._safe_float(grad_norm)

            if torch.isfinite(grad_norm):
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                return True, grad_norm_float

            self.optimizer.zero_grad(set_to_none=True)
            self.grad_scaler.update()
            return False, grad_norm_float

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            max_norm=max_norm,
            error_if_nonfinite=False,
        )

        grad_norm_float = self._safe_float(grad_norm)

        if torch.isfinite(grad_norm):
            self.optimizer.step()
            return True, grad_norm_float

        self.optimizer.zero_grad(set_to_none=True)
        return False, grad_norm_float

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

        use_amp = self.device.type == "cuda"

        try:
            with torch.autocast(self.device.type, enabled=True) if use_amp else dummy_context():
                out_dict = self._forward_dict(data)
                seg_logits = out_dict["seg_logits"]
                l_seg = self.loss(seg_logits, target)

            if not self._all_tensors_finite(out_dict):
                out = self._skipped_train_result("nonfinite_output")
                out["loss_seg"] = self._safe_float(l_seg)
                return out

            with torch.autocast(self.device.type, enabled=False) if use_amp else dummy_context():
                l_q, q_comps = self.query_aux_loss(out_dict, target)
                l = l_seg.float() + float(self.QUERY_LOSS_WEIGHT) * l_q.float()

        except FloatingPointError:
            self.optimizer.zero_grad(set_to_none=True)
            return self._skipped_train_result("floating_point_error")

        if not torch.isfinite(l.detach()):
            out = self._skipped_train_result("nonfinite_loss")
            out["loss_seg"] = self._safe_float(l_seg)
            out["loss_query"] = self._safe_float(l_q)
            if "q_comps" in locals():
                for k, v in q_comps.items():
                    if k in out:
                        out[k] = self._safe_float(v)
            return out

        stepped, grad_norm = self._optimizer_step_with_finite_guard(l)

        if not stepped:
            out = self._skipped_train_result("nonfinite_grad")
            out["loss_seg"] = self._safe_float(l_seg)
            out["loss_query"] = self._safe_float(l_q)
            out["grad_norm"] = self._safe_float(grad_norm)

            for k, v in q_comps.items():
                if k in out:
                    out[k] = self._safe_float(v)

            return out

        if self.VERIFY_PARAMETERS_AFTER_STEP and not self._parameters_are_finite():
            raise FloatingPointError(
                "Optimizer step produced non-finite parameters. "
                "Lower learning rate, reduce QUERY_LOSS_WEIGHT, or reduce GRAD_CLIP_MAX_NORM."
            )

        out = self._blank_train_result()

        out["loss"] = self._safe_float(l)
        out["loss_seg"] = self._safe_float(l_seg)
        out["loss_query"] = self._safe_float(l_q)
        out["grad_norm"] = self._safe_float(grad_norm)
        out["skipped_step"] = 0.0

        for k, v in q_comps.items():
            if k in out:
                out[k] = self._safe_float(v)

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

        use_amp = self.device.type == "cuda"

        with torch.autocast(self.device.type, enabled=True) if use_amp else dummy_context():
            out_dict = self._forward_dict(data)
            seg_logits = out_dict["seg_logits"]
            l_seg = self.loss(seg_logits, target)

        try:
            with torch.autocast(self.device.type, enabled=False) if use_amp else dummy_context():
                l_q, _ = self.query_aux_loss(out_dict, target)
                l = l_seg.float() + float(self.QUERY_LOSS_WEIGHT) * l_q.float()
        except FloatingPointError:
            l = l_seg.float()

        tp, fp, fn = self._compute_tp_fp_fn_hard(seg_logits, target)

        return {
            "loss": self._safe_float(l),
            "tp_hard": tp.numpy(),
            "fp_hard": fp.numpy(),
            "fn_hard": fn.numpy(),
        }

    def _compute_tp_fp_fn_hard(
        self,
        seg_logits: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
        target: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                valid = gt.ne(int(ignore))

            tp, fp, fn = [], [], []

            for c in range(1, n_cls):
                pred_c = pred.eq(c) & valid
                gt_c = gt.eq(c) & valid

                tp.append((pred_c & gt_c).sum())
                fp.append((pred_c & (~gt_c)).sum())
                fn.append(((~pred_c) & gt_c).sum())

            if len(tp) == 0:
                z = torch.zeros(0, dtype=torch.float32, device=logits.device)
                return z.cpu(), z.cpu(), z.cpu()

            return torch.stack(tp).cpu(), torch.stack(fp).cpu(), torch.stack(fn).cpu()

        if t.ndim < 2:
            raise ValueError("Region training expects target with channel dim")

        gt = t.gt(0.5)
        pred = torch.sigmoid(logits).gt(0.5)

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


__all__ = [
    "nnUNetTrainerResSEWithQueryDecoder",
    "QueryAuxLoss",
    "QueryLossWeights",
]