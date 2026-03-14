from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.custom_modules.res_se_block import ResidualSEBlock3D  
from nnunetv2.custom_modules.query_decoder import TransformerQueryDecoder, QueryDecoderOutput


# Utilities

def softmax_helper(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def as_3tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    x = tuple(int(i) for i in x)
    if len(x) != 3:
        raise ValueError(f"Expected 3 values, got {len(x)}")
    return x  


def expand_to_list(x: Union[int, Sequence[int]], n: int, name: str) -> List[int]:
    if isinstance(x, int):
        return [int(x)] * n
    x = [int(i) for i in x]
    if len(x) != n:
        raise ValueError(f"{name} must have length {n}, got {len(x)}")
    return x


def center_crop_or_pad(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Match x spatial shape to ref by center crop then symmetric pad."""
    if x.shape[2:] == ref.shape[2:]:
        return x

    # center crop if needed
    slices: List[slice] = [slice(None), slice(None)]
    for a, b in zip(x.shape[2:], ref.shape[2:]):
        if a > b:
            start = (a - b) // 2
            slices.append(slice(start, start + b))
        else:
            slices.append(slice(None))
    x = x[tuple(slices)]

    # symmetric pad if needed
    if x.shape[2:] != ref.shape[2:]:
        pad: List[int] = []
        for a, b in zip(reversed(x.shape[2:]), reversed(ref.shape[2:])):
            diff = max(0, b - a)
            left = diff // 2
            right = diff - left
            pad.extend([left, right])
        x = F.pad(x, pad)

    return x


def kernel_and_padding(
    kernel_size: Union[int, Sequence[int]],
) -> Tuple[Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]:
    k = as_3tuple(kernel_size)
    p = (k[0] // 2, k[1] // 2, k[2] // 2)
    k_arg: Union[int, Tuple[int, int, int]] = k[0] if k[0] == k[1] == k[2] else k
    p_arg: Union[int, Tuple[int, int, int]] = p[0] if p[0] == p[1] == p[2] else p
    return k_arg, p_arg


def nonlin_factory(
    nonlin: Optional[Callable[..., nn.Module]],
    nonlin_kwargs: Optional[dict],
) -> Optional[Callable[[], nn.Module]]:
    if nonlin is None:
        return None
    kw = {"inplace": True} if nonlin_kwargs is None else dict(nonlin_kwargs)

    def make() -> nn.Module:
        return nonlin(**kw)  

    return make


# Backbone: Residual-SE 3D U-Net

class EncoderStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        conv_op: type,
        kernel_size: Union[int, Sequence[int]],
        first_stride: Union[int, Sequence[int]],
        bias: bool,
        norm_op: Optional[type],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Callable[..., nn.Module]],
        nonlin_kwargs: Optional[dict],
        se_reduction: int,
    ) -> None:
        super().__init__()
        k, p = kernel_and_padding(kernel_size)
        norm = (lambda *_a, **_k: nn.Identity()) if norm_op is None else norm_op
        act = nonlin_factory(nonlin, nonlin_kwargs)
        nkwargs = {} if norm_op_kwargs is None else dict(norm_op_kwargs)

        blocks: List[nn.Module] = []
        for i in range(int(n_blocks)):
            stride = as_3tuple(first_stride) if i == 0 else 1
            blocks.append(
                ResidualSEBlock3D(
                    in_ch=in_ch if i == 0 else out_ch,
                    out_ch=out_ch,
                    stride=stride,
                    kernel_size=k,
                    padding=p,
                    bias=bias,
                    conv_op=conv_op,
                    norm_op=norm,
                    norm_kwargs=nkwargs,
                    nonlin=act,
                    se_reduction=se_reduction,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_ch_bottom: int,
        skip_ch: int,
        out_ch: int,
        n_blocks: int,
        conv_op: type,
        kernel_size: Union[int, Sequence[int]],
        up_stride: Union[int, Sequence[int]],
        bias: bool,
        norm_op: Optional[type],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Callable[..., nn.Module]],
        nonlin_kwargs: Optional[dict],
        se_reduction: int,
    ) -> None:
        super().__init__()
        s = as_3tuple(up_stride)
        self.up = nn.ConvTranspose3d(in_ch_bottom, out_ch, kernel_size=s, stride=s, bias=False)

        k, p = kernel_and_padding(kernel_size)
        norm = (lambda *_a, **_k: nn.Identity()) if norm_op is None else norm_op
        act = nonlin_factory(nonlin, nonlin_kwargs)
        nkwargs = {} if norm_op_kwargs is None else dict(norm_op_kwargs)

        blocks: List[nn.Module] = []
        for i in range(int(n_blocks)):
            blocks.append(
                ResidualSEBlock3D(
                    in_ch=(out_ch + skip_ch) if i == 0 else out_ch,
                    out_ch=out_ch,
                    stride=1,
                    kernel_size=k,
                    padding=p,
                    bias=bias,
                    conv_op=conv_op,
                    norm_op=norm,
                    norm_kwargs=nkwargs,
                    nonlin=act,
                    se_reduction=se_reduction,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = center_crop_or_pad(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.blocks(x)


class UNetResSE3D(nn.Module):
    """Residual-SE 3D U-Net backbone (nnU-Net v2 plan-friendly)."""

    inference_apply_nonlin: Callable[[torch.Tensor], torch.Tensor] = staticmethod(softmax_helper)

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, Sequence[int]],
        conv_op: type = nn.Conv3d,
        kernel_sizes: Union[Sequence[Union[int, Sequence[int]]], Union[int, Sequence[int]]] = 3,
        strides: Union[Sequence[Union[int, Sequence[int]]], Union[int, Sequence[int]]] = 1,
        n_conv_per_stage: Union[int, Sequence[int]] = 2,
        num_classes: int = 2,
        n_conv_per_stage_decoder: Union[int, Sequence[int]] = 2,
        conv_bias: bool = False,
        norm_op: Optional[type] = nn.InstanceNorm3d,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Callable[..., nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = False,
        se_reduction: int = 16,
        **_: Any,
    ) -> None:
        super().__init__()

        if int(n_stages) < 2:
            raise ValueError("n_stages must be >= 2")

        self.input_channels = int(input_channels)
        self.n_stages = int(n_stages)
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)

        if isinstance(features_per_stage, int):
            feats = [int(features_per_stage) * (2**i) for i in range(self.n_stages)]
        else:
            feats = expand_to_list(features_per_stage, self.n_stages, "features_per_stage")
        self.features_per_stage = feats

        ks = (
            [kernel_sizes] * self.n_stages
            if not isinstance(kernel_sizes, (list, tuple)) or len(kernel_sizes) != self.n_stages
            else list(kernel_sizes)
        )
        st = (
            [strides] * self.n_stages
            if not isinstance(strides, (list, tuple)) or len(strides) != self.n_stages
            else list(strides)
        )

        enc_blocks = expand_to_list(n_conv_per_stage, self.n_stages, "n_conv_per_stage")
        dec_blocks = (
            [int(n_conv_per_stage_decoder)] * (self.n_stages - 1)
            if isinstance(n_conv_per_stage_decoder, int)
            else expand_to_list(n_conv_per_stage_decoder, self.n_stages - 1, "n_conv_per_stage_decoder")
        )

        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = self.input_channels
        for s in range(self.n_stages):
            self.encoder.append(
                EncoderStage(
                    in_ch=in_ch,
                    out_ch=feats[s],
                    n_blocks=enc_blocks[s],
                    conv_op=conv_op,
                    kernel_size=ks[s],
                    first_stride=st[s],
                    bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    se_reduction=se_reduction,
                )
            )
            in_ch = feats[s]

        # Decoder
        self.decoder = nn.ModuleList()
        for s in range(self.n_stages - 2, -1, -1):
            self.decoder.append(
                DecoderStage(
                    in_ch_bottom=feats[s + 1],
                    skip_ch=feats[s],
                    out_ch=feats[s],
                    n_blocks=dec_blocks[s],
                    conv_op=conv_op,
                    kernel_size=ks[s],
                    up_stride=st[s + 1],
                    bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    se_reduction=se_reduction,
                )
            )

        # Segmentation heads per decoder resolution (excluding bottleneck)
        self.seg_heads = nn.ModuleList(
            [conv_op(feats[s], self.num_classes, kernel_size=1, stride=1, padding=0, bias=True) for s in range(self.n_stages - 1)]
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, a=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.InstanceNorm3d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_decoder_features: bool = False,
    ) -> Union[
        torch.Tensor,
        List[torch.Tensor],
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[List[torch.Tensor], List[torch.Tensor]],
    ]:
        skips: List[torch.Tensor] = []
        y = x
        for stage in self.encoder:
            y = stage(y)
            skips.append(y)

        y = skips[-1]
        seg_outputs: List[torch.Tensor] = []
        decoder_feats: List[torch.Tensor] = []

        for di, s in enumerate(range(self.n_stages - 2, -1, -1)):
            y = self.decoder[di](y, skips[s])
            decoder_feats.append(y)
            seg_outputs.append(self.seg_heads[s](y))

        seg_outputs.reverse()
        decoder_feats.reverse()

        seg_logits: Union[torch.Tensor, List[torch.Tensor]] = seg_outputs if self.deep_supervision else seg_outputs[0]
        return (seg_logits, decoder_feats) if return_decoder_features else seg_logits


# Query head wrapper + inference utility

@dataclass
class QueryInferenceResult:
    prob_map: torch.Tensor  # (B, D, H, W)
    keep_mask: torch.Tensor  # (B, N)
    scores: torch.Tensor  # (B, N)
    centroids: List[List[Tuple[float, float, float]]]  # per batch: list of (z,y,x)


def connected_components_centroids_3d(binary: np.ndarray) -> List[Tuple[float, float, float]]:
    from scipy import ndimage as ndi

    if binary.ndim != 3:
        raise ValueError(f"binary must be 3D, got {binary.shape}")

    structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
    labeled, n = ndi.label(binary.astype(np.uint8), structure=structure)
    if n == 0:
        return []

    com = ndi.center_of_mass(binary.astype(np.uint8), labeled, list(range(1, n + 1)))
    return [(float(z), float(y), float(x)) for (z, y, x) in com]


class UNetResSE3DWithQueryDecoder(nn.Module):
    """UNetResSE3D backbone + TransformerQueryDecoder head."""

    def __init__(
        self,
        backbone: UNetResSE3D,
        n_queries: int = 64,
        n_heads: int = 8,
        n_layers: Optional[int] = None,
        d_model: Optional[int] = None,
        dim_ffn: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone

        decoder_channels = list(backbone.features_per_stage[:-1])
        if d_model is None:
            d_model = decoder_channels[0]

        self.query_decoder = TransformerQueryDecoder(
            decoder_channels=decoder_channels,
            d_model=int(d_model),
            n_queries=int(n_queries),
            n_heads=int(n_heads),
            n_layers=n_layers,
            dim_ffn=dim_ffn,
            dropout=float(dropout),
        )

    @torch.no_grad()
    def inference_from_queries(
        self,
        class_logits: torch.Tensor,  # (B, N, 2)
        mask_logits: torch.Tensor,  # (B, N, D, H, W)
        mask_quality: torch.Tensor,  # (B, N, 1)
        score_thresh: float = 0.5,
        mask_thresh: float = 0.5,
    ) -> QueryInferenceResult:
        if class_logits.ndim != 3 or class_logits.shape[-1] != 2:
            raise ValueError("class_logits must have shape (B, N, 2)")
        if mask_logits.ndim != 5:
            raise ValueError("mask_logits must have shape (B, N, D, H, W)")
        if mask_quality.ndim != 3 or mask_quality.shape[-1] != 1:
            raise ValueError("mask_quality must have shape (B, N, 1)")

        cls_prob = torch.softmax(class_logits, dim=-1)[..., 1]
        qual = torch.sigmoid(mask_quality[..., 0])
        scores = cls_prob * qual
        keep = scores > float(score_thresh)

        mask_prob = torch.sigmoid(mask_logits)
        mask_prob = mask_prob * keep[:, :, None, None, None].to(mask_prob.dtype)
        prob_map = mask_prob.max(dim=1).values

        centroids: List[List[Tuple[float, float, float]]] = []
        prob_np = prob_map.detach().cpu().numpy()
        for b in range(prob_np.shape[0]):
            centroids.append(connected_components_centroids_3d(prob_np[b] > float(mask_thresh)))

        return QueryInferenceResult(prob_map=prob_map, keep_mask=keep, scores=scores, centroids=centroids)

    def forward(
        self,
        x: torch.Tensor,
        run_inference: bool = False,
        score_thresh: float = 0.5,
        mask_thresh: float = 0.5,
        iterate_coarse_to_fine: bool = True,
    ) -> dict:
        seg_logits, decoder_feats = self.backbone(x, return_decoder_features=True)
        final_feat = decoder_feats[0]

        qout: QueryDecoderOutput = self.query_decoder(
            decoder_feats=decoder_feats,
            final_feat=final_feat,
            iterate_coarse_to_fine=iterate_coarse_to_fine,
        )

        out = {
            "seg_logits": seg_logits,
            "class_logits": qout.class_logits,
            "mask_logits": qout.mask_logits,
            "mask_quality": qout.mask_quality,
            "queries": qout.queries,
        }

        if run_inference:
            inf = self.inference_from_queries(
                qout.class_logits,
                qout.mask_logits,
                qout.mask_quality,
                score_thresh=score_thresh,
                mask_thresh=mask_thresh,
            )
            out.update(
                {
                    "prob_map": inf.prob_map,
                    "scores": inf.scores,
                    "keep_mask": inf.keep_mask,
                    "centroids": inf.centroids,
                }
            )

        return out

