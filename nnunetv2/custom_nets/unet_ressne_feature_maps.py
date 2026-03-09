from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.custom_modules.res_se_block import ResidualSEBlock3D  # type: ignore


def _softmax_helper(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def _infer_dim_from_conv_op(conv_op: type) -> int:
    """Best-effort inference of spatial dimensionality from a convolution class."""
    # Prefer subclass checks (works for custom ops inheriting from torch convs)
    try:
        if issubclass(conv_op, nn.Conv3d):
            return 3
        if issubclass(conv_op, nn.Conv2d):
            return 2
        if issubclass(conv_op, nn.Conv1d):
            return 1
    except TypeError:
        # conv_op is not a class
        pass

    # Fallback to identity checks
    if conv_op is nn.Conv3d:
        return 3
    if conv_op is nn.Conv2d:
        return 2
    if conv_op is nn.Conv1d:
        return 1

    # Last resort: name heuristics
    name = getattr(conv_op, "__name__", str(conv_op))
    if "Conv3" in name or "conv3" in name:
        return 3
    if "Conv2" in name or "conv2" in name:
        return 2
    if "Conv1" in name or "conv1" in name:
        return 1
    raise ValueError(f"Could not infer spatial dim from conv_op={conv_op}")


def _expand_to_list(x: Union[int, Sequence[int]], n: int, name: str) -> List[int]:
    if isinstance(x, int):
        return [int(x)] * n
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f"{name} must have length {n}, got {len(x)}")
        return [int(i) for i in x]
    raise TypeError(f"{name} must be int or sequence, got {type(x)}")


def _as_3tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    """Convert int or len-3 sequence to a 3-tuple."""
    if isinstance(x, int):
        return (int(x), int(x), int(x))
    if isinstance(x, (list, tuple)):
        if len(x) != 3:
            raise ValueError(f"Expected sequence length 3, got {len(x)}")
        return (int(x[0]), int(x[1]), int(x[2]))
    raise TypeError(f"Expected int or sequence, got {type(x)}")


def _maybe_collapse_tuple_to_int(t: Tuple[int, int, int]) -> Union[int, Tuple[int, int, int]]:
    """If tuple is isotropic (a,a,a) return int a, else return the tuple."""
    if t[0] == t[1] == t[2]:
        return int(t[0])
    return t


def _kernel_and_padding_3d(
    kernel_size: Union[int, Sequence[int]],
) -> Tuple[Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]:
    """Return (kernel_size, padding) for Conv3d-compatible args."""
    k3 = _as_3tuple(kernel_size)
    p3 = (k3[0] // 2, k3[1] // 2, k3[2] // 2)
    k_arg = _maybe_collapse_tuple_to_int(k3)
    p_arg = _maybe_collapse_tuple_to_int(p3)
    return k_arg, p_arg


def _stride_3d(stride: Union[int, Sequence[int]]) -> Union[int, Tuple[int, int, int]]:
    s3 = _as_3tuple(stride)
    # Avoid stride=(1,1,1) because some block implementations check (stride != 1).
    if s3[0] == s3[1] == s3[2] == 1:
        return 1
    return _maybe_collapse_tuple_to_int(s3)


class _IdentityNorm(nn.Module):
    """A norm placeholder that accepts (num_features, **kwargs)."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make_nonlin_factory(
    nonlin: Optional[Callable[..., nn.Module]],
    nonlin_kwargs: Optional[dict],
) -> Optional[Callable[[], nn.Module]]:
    if nonlin is None:
        return None
    kwargs = {"inplace": True} if nonlin_kwargs is None else dict(nonlin_kwargs)

    def _factory() -> nn.Module:
        return nonlin(**kwargs)  # type: ignore[misc]

    return _factory


class _EncoderStage(nn.Module):
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
        k_arg, p_arg = _kernel_and_padding_3d(kernel_size)
        norm_op_cls = _IdentityNorm if norm_op is None else norm_op
        nonlin_factory = _make_nonlin_factory(nonlin, nonlin_kwargs)

        blocks: List[nn.Module] = []
        for b in range(n_blocks):
            stride_arg = _stride_3d(first_stride) if b == 0 else 1
            blocks.append(
                ResidualSEBlock3D(
                    in_ch=in_ch if b == 0 else out_ch,
                    out_ch=out_ch,
                    stride=stride_arg,
                    kernel_size=k_arg,
                    padding=p_arg,
                    bias=bias,
                    conv_op=conv_op,
                    norm_op=norm_op_cls,
                    norm_kwargs={} if norm_op_kwargs is None else dict(norm_op_kwargs),
                    nonlin=nonlin_factory,
                    se_reduction=se_reduction,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _DecoderStage(nn.Module):
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
        dim = _infer_dim_from_conv_op(conv_op)
        if dim != 3:
            raise ValueError(f"_DecoderStage expects Conv3d-like op, got dim={dim} ({conv_op})")

        s3 = _as_3tuple(up_stride)
        self.up = nn.ConvTranspose3d(in_ch_bottom, out_ch, kernel_size=s3, stride=s3, bias=False)

        k_arg, p_arg = _kernel_and_padding_3d(kernel_size)
        norm_op_cls = _IdentityNorm if norm_op is None else norm_op
        nonlin_factory = _make_nonlin_factory(nonlin, nonlin_kwargs)

        blocks: List[nn.Module] = []
        for b in range(n_blocks):
            blocks.append(
                ResidualSEBlock3D(
                    in_ch=(out_ch + skip_ch) if b == 0 else out_ch,
                    out_ch=out_ch,
                    stride=1,
                    kernel_size=k_arg,
                    padding=p_arg,
                    bias=bias,
                    conv_op=conv_op,
                    norm_op=norm_op_cls,
                    norm_kwargs={} if norm_op_kwargs is None else dict(norm_op_kwargs),
                    nonlin=nonlin_factory,
                    se_reduction=se_reduction,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Center-crop and/or symmetric pad x to match ref spatial size."""
        sx = x.shape[2:]
        sr = ref.shape[2:]
        if sx == sr:
            return x

        # center-crop if bigger
        slices = [slice(None), slice(None)]
        for a, b in zip(sx, sr):
            if a > b:
                start = (a - b) // 2
                slices.append(slice(start, start + b))
            else:
                slices.append(slice(None))
        x = x[tuple(slices)]

        # pad if smaller
        sx2 = x.shape[2:]
        if sx2 != sr:
            pad: List[int] = []
            for a, b in zip(reversed(sx2), reversed(sr)):
                if a < b:
                    diff = b - a
                    left = diff // 2
                    right = diff - left
                    pad.extend([left, right])
                else:
                    pad.extend([0, 0])
            x = F.pad(x, pad)
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._match_size(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.blocks(x)


class UNetResSE3D(nn.Module):
    """Residual-SE U-Net (3D), nnU-Net v2 plans-compatible."""

    inference_apply_nonlin: Callable[[torch.Tensor], torch.Tensor] = staticmethod(_softmax_helper)

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
        dropout_op: Optional[type] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Callable[..., nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = False,
        se_reduction: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        dim = _infer_dim_from_conv_op(conv_op)
        if dim != 3:
            raise ValueError(f"UNetResSE3D expects a 3D conv_op, got dim={dim} ({conv_op})")

        self.input_channels = int(input_channels)
        self.n_stages = int(n_stages)
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)

        if self.n_stages < 2:
            raise ValueError("UNetResSE3D requires n_stages >= 2 (needs at least one decoder stage)")

        # features per stage
        if isinstance(features_per_stage, int):
            feats = [int(features_per_stage * (2**i)) for i in range(self.n_stages)]
        else:
            feats = _expand_to_list(features_per_stage, self.n_stages, "features_per_stage")

        # kernel sizes per stage
        if not isinstance(kernel_sizes, (list, tuple)) or len(kernel_sizes) != self.n_stages:
            ks: List[Union[int, Sequence[int]]] = [kernel_sizes for _ in range(self.n_stages)]
        else:
            ks = list(kernel_sizes)  # type: ignore[list-item]
            if len(ks) != self.n_stages:
                raise ValueError(f"kernel_sizes must have length {self.n_stages}, got {len(ks)}")

        # strides per stage
        if not isinstance(strides, (list, tuple)) or len(strides) != self.n_stages:
            st: List[Union[int, Sequence[int]]] = [strides for _ in range(self.n_stages)]
        else:
            st = list(strides)  # type: ignore[list-item]
            if len(st) != self.n_stages:
                raise ValueError(f"strides must have length {self.n_stages}, got {len(st)}")

        enc_blocks = _expand_to_list(n_conv_per_stage, self.n_stages, "n_conv_per_stage")

        if isinstance(n_conv_per_stage_decoder, int):
            dec_blocks = [int(n_conv_per_stage_decoder)] * (self.n_stages - 1)
        else:
            dec_blocks = list(n_conv_per_stage_decoder)
            if len(dec_blocks) != self.n_stages - 1:
                raise ValueError(
                    f"n_conv_per_stage_decoder must have length {self.n_stages - 1}, got {len(dec_blocks)}"
                )

        self.encoder = nn.ModuleList()
        in_ch = self.input_channels
        for s in range(self.n_stages):
            self.encoder.append(
                _EncoderStage(
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

        self.decoder = nn.ModuleList()
        for s in range(self.n_stages - 2, -1, -1):
            self.decoder.append(
                _DecoderStage(
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

        # one seg head for each decoder output (stages 0..n_stages-2)
        self.seg_heads = nn.ModuleList(
            [
                conv_op(feats[s], self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                for s in range(self.n_stages - 1)
            ]
        )

        # accepted for plans-compatibility; unused
        _ = dropout_op, dropout_op_kwargs, kwargs

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(
            m,
            (
                nn.Conv3d,
                nn.Conv2d,
                nn.Conv1d,
                nn.ConvTranspose3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose1d,
            ),
        ):
            nn.init.kaiming_normal_(m.weight, a=0.01)
            if getattr(m, "bias", None) is not None and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(
            m,
            (
                nn.InstanceNorm3d,
                nn.InstanceNorm2d,
                nn.InstanceNorm1d,
                nn.BatchNorm3d,
                nn.BatchNorm2d,
                nn.BatchNorm1d,
                nn.GroupNorm,
            ),
        ):
            if getattr(m, "weight", None) is not None and m.weight is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None and m.bias is not None:
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
        """
        Args:
            x: Input tensor (B, C, D, H, W).
            return_decoder_features: If True, also return decoder feature maps at
                each decoder stage in full-res-first order.

        Returns:
            If return_decoder_features is False:
              - seg_logits: Tensor if deep_supervision=False, else List[Tensor]
            If return_decoder_features is True:
              - (seg_logits, decoder_feats) where decoder_feats is
                List[Tensor] ordered [F_largest, ..., F_smallest].
        """
        skips: List[torch.Tensor] = []
        out = x
        for s in range(self.n_stages):
            out = self.encoder[s](out)
            skips.append(out)

        out = skips[-1]
        seg_outputs: List[torch.Tensor] = []
        decoder_feats: List[torch.Tensor] = []

        # decoder stages correspond to encoder stages (n_stages-2 .. 0)
        # First iteration produces smallest decoder feature (closest to bottleneck)
        # Last iteration produces largest/full-res decoder feature
        for di, s in enumerate(range(self.n_stages - 2, -1, -1)):
            out = self.decoder[di](out, skips[s])
            if return_decoder_features:
                decoder_feats.append(out)
            seg_outputs.append(self.seg_heads[s](out))

        # seg_outputs collected coarse->fine; reverse to full-res first
        seg_outputs = list(reversed(seg_outputs))

        if return_decoder_features:
            # decoder_feats collected coarse->fine; reverse to full-res first
            decoder_feats = list(reversed(decoder_feats))

        seg_logits: Union[torch.Tensor, List[torch.Tensor]]
        if self.deep_supervision:
            seg_logits = seg_outputs
        else:
            seg_logits = seg_outputs[0]

        if return_decoder_features:
            return seg_logits, decoder_feats
        return seg_logits


def _run_self_tests() -> None:
    torch.manual_seed(0)

    # Test 1: basic forward shape
    m = UNetResSE3D(
        input_channels=1,
        n_stages=4,
        features_per_stage=12,
        kernel_sizes=3,
        strides=[1, 2, 2, 2],
        n_conv_per_stage=2,
        n_conv_per_stage_decoder=2,
        num_classes=3,
        deep_supervision=False,
    )
    x = torch.randn(2, 1, 33, 35, 31)
    y = m(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 3, 33, 35, 31)

    # Test 2: deep supervision shapes + decoder features
    m2 = UNetResSE3D(
        input_channels=2,
        n_stages=3,
        features_per_stage=[8, 16, 32],
        kernel_sizes=[(3, 3, 1), 3, 3],
        strides=[1, (2, 2, 1), 2],
        n_conv_per_stage=[1, 2, 2],
        n_conv_per_stage_decoder=[2, 1],
        num_classes=4,
        deep_supervision=True,
        norm_op=None,
    )
    x2 = torch.randn(1, 2, 29, 30, 17)
    seg, feats = m2(x2, return_decoder_features=True)
    assert isinstance(seg, list) and len(seg) == 2
    assert isinstance(feats, list) and len(feats) == 2
    assert feats[0].shape[2:] == x2.shape[2:]
    assert seg[0].shape[2:] == feats[0].shape[2:]
    assert seg[1].shape[2:] == feats[1].shape[2:]

    # Test 3: backward / grad sanity
    m3 = UNetResSE3D(
        input_channels=1,
        n_stages=2,
        features_per_stage=8,
        strides=[1, 2],
        num_classes=2,
        deep_supervision=False,
    )
    x3 = torch.randn(1, 1, 20, 21, 19, requires_grad=True)
    y3 = m3(x3)
    y3.mean().backward()
    assert x3.grad is not None and torch.isfinite(x3.grad).all()

    # Test 4: inference nonlin sums to 1 over channels
    probs = m3.inference_apply_nonlin(y3)
    s = probs.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5, rtol=0)

    print("Self-tests passed.")


__all__ = ["UNetResSE3D"]


if __name__ == "__main__":
    _run_self_tests()