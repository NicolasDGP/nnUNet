from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class QueryDecoderOutput:
    class_logits: torch.Tensor   # (B, N, 2)
    mask_logits: torch.Tensor    # (B, N, D, H, W)
    mask_quality: torch.Tensor   # (B, N, 1)
    queries: torch.Tensor        # (B, N, d)


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ffn: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.ln_q1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln_q2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln_q3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # queries: (B, N, d)
        # kv: (B, S, d)
        q1 = self.ln_q1(queries)
        sa, _ = self.self_attn(q1, q1, q1, need_weights=False)
        queries = queries + sa

        q2 = self.ln_q2(queries)
        ca, _ = self.cross_attn(q2, kv, kv, need_weights=False)
        queries = queries + ca

        q3 = self.ln_q3(queries)
        queries = queries + self.ffn(q3)
        return queries


class TransformerQueryDecoder(nn.Module):
    """Transformer query decoder (TransUNet-style head).

    Implements the report logic:
      learnable queries -> self-attn -> cross-attn to CNN decoder feats per scale -> recompute masks via dot product

    Inputs:
      decoder_feats: list of nnU-Net decoder feature maps (B, C_t, D_t, H_t, W_t)
      final_feat: high-res decoder map F (B, C_f, D, H, W)

    Outputs:
      class_logits: (B, N, 2)
      mask_logits:  (B, N, D, H, W)
      mask_quality: (B, N, 1)

    Notes:
      - Features at each scale are projected to d_model via 1x1 conv.
      - Masks are always recomputed against the (projected) final high-res feature map.
    """

    def __init__(
        self,
        decoder_channels: List[int],
        d_model: int,
        n_queries: int = 64,
        n_heads: int = 8,
        n_layers: Optional[int] = None,
        dim_ffn: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_queries <= 0:
            raise ValueError("n_queries must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")

        self.d_model = int(d_model)
        self.n_queries = int(n_queries)

        self.query_embed = nn.Embedding(self.n_queries, self.d_model)

        # one projection per decoder scale
        self.feat_projs = nn.ModuleList()
        for c in decoder_channels:
            if int(c) == self.d_model:
                self.feat_projs.append(nn.Identity())
            else:
                self.feat_projs.append(nn.Conv3d(int(c), self.d_model, kernel_size=1, bias=False))

        # final feature projection (if needed)
        c_final = int(decoder_channels[0])  # typically full-res channels
        if c_final == self.d_model:
            self.final_proj = nn.Identity()
        else:
            self.final_proj = nn.Conv3d(c_final, self.d_model, kernel_size=1, bias=False)

        # layers
        if n_layers is None:
            n_layers = len(decoder_channels)
        self.n_layers = int(n_layers)
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")

        if dim_ffn is None:
            dim_ffn = 4 * self.d_model
        self.layers = nn.ModuleList(
            [_DecoderLayer(self.d_model, n_heads, int(dim_ffn), float(dropout)) for _ in range(self.n_layers)]
        )

        # heads
        self.class_head = nn.Linear(self.d_model, 2)
        self.quality_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, max(1, self.d_model // 2)),
            nn.GELU(),
            nn.Linear(max(1, self.d_model // 2), 1),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.query_embed.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _flatten_feat(feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, d, D, H, W) -> (B, S, d)
        b, d, dd, hh, ww = feat.shape
        return feat.view(b, d, dd * hh * ww).transpose(1, 2).contiguous()

    def forward(
        self,
        decoder_feats: List[torch.Tensor],
        final_feat: torch.Tensor,
        iterate_coarse_to_fine: bool = True,
    ) -> QueryDecoderOutput:
        if len(decoder_feats) == 0:
            raise ValueError("decoder_feats must be a non-empty list")

        b = final_feat.shape[0]

        # initialize learnable queries P0 (B, N, d)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)

        # project and flatten final high-res feature map for mask recomputation
        final_feat_p = self.final_proj(final_feat)  # (B, d, D, H, W)
        final_flat = self._flatten_feat(final_feat_p)  # (B, S, d)
        final_flat_t = final_flat.transpose(1, 2).contiguous()  # (B, d, S)

        # pick traversal order over scales
        feats = decoder_feats
        projs = self.feat_projs
        if iterate_coarse_to_fine:
            feats = list(reversed(decoder_feats))
            projs = nn.ModuleList(list(reversed(self.feat_projs)))

        # run T layers; if layers > scales, cycle over scales
        n_scales = len(feats)
        mask_logits_flat: torch.Tensor = torch.empty(0, device=final_feat.device)
        for t in range(self.n_layers):
            feat = feats[t % n_scales]
            proj = projs[t % n_scales]
            feat_p = proj(feat)  # (B, d, D_t, H_t, W_t)
            kv = self._flatten_feat(feat_p)  # (B, S_t, d)

            queries = self.layers[t](queries, kv)

            # recompute masks against final high-res features: Z_{t+1} = P_{t+1} @ F_flat^T
            mask_logits_flat = torch.bmm(queries, final_flat_t)  # (B, N, S)

        # reshape masks to volume
        _, _, dd, hh, ww = final_feat.shape
        mask_logits = mask_logits_flat.view(b, self.n_queries, dd, hh, ww)

        class_logits = self.class_head(queries)  # (B, N, 2)
        mask_quality = self.quality_head(queries)  # (B, N, 1)

        return QueryDecoderOutput(
            class_logits=class_logits,
            mask_logits=mask_logits,
            mask_quality=mask_quality,
            queries=queries,
        )


__all__ = ["TransformerQueryDecoder", "QueryDecoderOutput"]


def _run_self_tests() -> None:
    # Avoid potential BLAS/OpenMP deadlocks in constrained environments
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    torch.manual_seed(0)

    # Build synthetic multi-scale decoder features
    b = 2
    f0 = torch.randn(b, 16, 10, 12, 11)
    f1 = torch.randn(b, 32, 5, 6, 6)
    f2 = torch.randn(b, 64, 3, 3, 3)

    dec = TransformerQueryDecoder(decoder_channels=[16, 32, 64], d_model=16, n_queries=8, n_heads=4)
    out = dec([f0, f1, f2], final_feat=f0)
    assert out.class_logits.shape == (b, 8, 2)
    assert out.mask_quality.shape == (b, 8, 1)
    assert out.mask_logits.shape == (b, 8, 10, 12, 11)
    assert torch.isfinite(out.mask_logits).all()

    # Backward sanity
    loss = out.class_logits.mean() + out.mask_logits.mean() + out.mask_quality.mean()
    loss.backward()
    for n, p in dec.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {n}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {n}"

    print("Query decoder self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()