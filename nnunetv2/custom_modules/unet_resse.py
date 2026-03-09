"""
UNetResSE3D – nnU-Net-style 3-D U-Net where every conv block is replaced
with a Residual + Squeeze-and-Excitation block (ResidualSEBlock3D).

The decoder saves a feature map at every upsampling stage so the
transformer query decoder can attend to multiple resolution levels.
"""

from typing import List, Tuple, Type, Optional, Callable
import torch
import torch.nn as nn
from nnunetv2.custom_modules.res_se_block import ResidualSEBlock3D


class EncoderBlock(nn.Module):
    """One encoder stage: ResidualSEBlock3D then 3-D max-pool."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True,
                 se_reduction: int = 8):
        super().__init__()
        self.block = ResidualSEBlock3D(in_ch, out_ch, se_reduction=se_reduction)
        self.pool  = nn.MaxPool3d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        return self.pool(x), x          # pooled output  +  skip connection


class DecoderBlock(nn.Module):
    """One decoder stage: transposed-conv upsample, concat skip, ResidualSEBlock3D."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 se_reduction: int = 8):
        super().__init__()
        self.up    = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.block = ResidualSEBlock3D(in_ch + skip_ch, out_ch,
                                       se_reduction=se_reduction)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class UNetResSE3D(nn.Module):
    """
    3-D U-Net with Residual SE blocks.

    Args
    ----
    in_channels     : number of input image channels (1 for CT/MRI)
    num_classes     : number of segmentation classes  (2 = bg + aneurysm)
    base_features   : feature-map width at first encoder stage (doubles each stage)
    depth           : number of encoder/decoder stages
    se_reduction    : squeeze-and-excitation bottleneck ratio
    return_decoder_features : if True, forward() returns (seg_logits, [F0,F1,...])
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_features: int = 32,
        depth: int = 4,
        se_reduction: int = 8,
        return_decoder_features: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.return_decoder_features = return_decoder_features

        # ── encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        ch = in_channels
        enc_channels = []          # track output channels at each encoder stage
        for i in range(depth):
            out_ch = base_features * (2 ** i)
            # last encoder stage: no pooling (bottleneck)
            self.encoders.append(
                EncoderBlock(ch, out_ch, pool=(i < depth - 1),
                             se_reduction=se_reduction)
            )
            enc_channels.append(out_ch)
            ch = out_ch

        # ── decoder ──────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        for i in range(depth - 1):
            skip_ch = enc_channels[depth - 2 - i]   # matching encoder skip
            out_ch  = skip_ch                        # output same as skip width
            self.decoders.append(
                DecoderBlock(ch, skip_ch, out_ch, se_reduction=se_reduction)
            )
            ch = out_ch

        # ── segmentation head ─────────────────────────────────────────────────
        self.seg_head = nn.Conv3d(ch, num_classes, kernel_size=1)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        # --- encoder ---
        skips = []
        for i, enc in enumerate(self.encoders):
            x, skip = enc(x)
            if i < self.depth - 1:          # don't store bottleneck as skip
                skips.append(skip)

        # x is now the bottleneck feature map

        # --- decoder  (collect feature maps at every stage) ---
        decoder_features = []
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]          # reverse order
            x = dec(x, skip)
            decoder_features.append(x)     # P0(coarse)→P3(fine)

        seg_logits = self.seg_head(x)

        if self.return_decoder_features:
            return seg_logits, decoder_features   # tuple
        return seg_logits