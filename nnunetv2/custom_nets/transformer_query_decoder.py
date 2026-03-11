"""
TransformerQueryDecoder  —  Section 2.2.4 / 2.2.5 of the project report.

Expects decoder_feats in FULL-RES-FIRST order:
    [F_finest, F_mid, ..., F_coarsest]
exactly as returned by UNetResSE3D.forward(x, return_decoder_features=True).

The loop processes coarse→fine (reversed internally) so queries first get
global context then progressively refine against finer features.
"""

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerQueryDecoder(nn.Module):
    """
    Parameters
    ----------
    feature_channels : list of ints
        Channel count at each decoder level, FULL-RES-FIRST order.
        e.g. for n_stages=4, features_per_stage=32:
             [32, 64, 128]   (n_stages-1 levels)
    num_queries      : number of learnable aneurysm candidate slots (N)
    embed_dim        : internal transformer width (d)
    num_heads        : attention heads (must divide embed_dim)
    ffn_dim          : feedforward hidden dimension
    num_classes      : 2 for background / aneurysm
    dropout          : attention dropout
    """

    def __init__(
        self,
        feature_channels: List[int],
        num_queries:  int   = 16,
        embed_dim:    int   = 256,
        num_heads:    int   = 8,
        ffn_dim:      int   = 512,
        num_classes:  int   = 2,
        dropout:      float = 0.0,
    ):
        super().__init__()

        self.num_queries    = num_queries
        self.embed_dim      = embed_dim
        self.num_scales     = len(feature_channels)
        self.feature_channels = feature_channels

        # ── learnable query embeddings  P_0  [N, d] ──────────────────────────
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        # ── per-scale linear projections: C_l → embed_dim ────────────────────
        # feature_channels is full-res-first; we project every level
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            for c in feature_channels          # one projector per scale
        ])

        # ── one transformer layer per scale ───────────────────────────────────
        # We iterate coarse→fine so index 0 = coarsest = feature_channels[-1]
        self.norm_sa   = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.num_scales)])
        self.norm_ca   = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.num_scales)])
        self.norm_ffn  = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.num_scales)])

        self.self_attn  = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(self.num_scales)
        ])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(self.num_scales)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_scales)
        ])

        # ── output heads  (Section 2.2.5) ─────────────────────────────────────
        self.class_head   = nn.Linear(embed_dim, num_classes)   # bg vs aneurysm
        self.quality_head = nn.Linear(embed_dim, 1)             # predicted IoU

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _flatten(feat: torch.Tensor) -> torch.Tensor:
        """[B, C, D, H, W] → [B, D*H*W, C]"""
        B, C = feat.shape[:2]
        return feat.view(B, C, -1).permute(0, 2, 1).contiguous()

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        decoder_feats: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        decoder_feats : list of tensors, FULL-RES-FIRST order
                        [F_finest[B,C0,D,H,W], ..., F_coarsest[B,Ck,d,h,w]]
                        as returned by UNetResSE3D

        Returns
        -------
        class_logits : [B, N, num_classes]
        mask_logits  : [B, N, D*H*W]   against the finest resolution
        mask_quality : [B, N, 1]
        """
        B = decoder_feats[0].shape[0]

        # ── initialise queries  [B, N, d] ─────────────────────────────────────
        P = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # ── finest feature for mask dot-product (full spatial res) ────────────
        # decoder_feats[0] is the finest (full-res-first order)
        F_fine_flat = self._flatten(decoder_feats[0])          # [B, DHW, C0]
        F_fine_proj = self.input_proj[0](F_fine_flat)          # [B, DHW, d]

        # initial mask Z_0 = P_0 @ F_fine^T  →  [B, N, DHW]
        Z = torch.bmm(P, F_fine_proj.transpose(1, 2))

        # ── loop coarse → fine (reverse the full-res-first list) ──────────────
        reversed_feats = list(reversed(decoder_feats))   # now coarse→fine
        # projectors in input_proj are indexed full-res-first, so reverse index
        n = self.num_scales

        for t, F_t in enumerate(reversed_feats):
            proj_idx = (n - 1) - t          # maps coarse→fine t to fine→coarse proj index

            # flatten + project this scale
            F_flat = self._flatten(F_t)                        # [B, tokens, C_t]
            F_proj = self.input_proj[proj_idx](F_flat)         # [B, tokens, d]

            # 1. Self-attention over queries
            P_norm = self.norm_sa[t](P)
            P_sa, _ = self.self_attn[t](P_norm, P_norm, P_norm)
            P = P + P_sa

            # 2. Cross-attention  Q=queries, K/V=CNN features at this scale
            P_norm = self.norm_ca[t](P)
            P_ca, _ = self.cross_attn[t](P_norm, F_proj, F_proj)
            P = P + P_ca

            # 3. FFN + residual
            P = P + self.ffn[t](self.norm_ffn[t](P))

            # 4. Recompute masks against finest resolution
            Z = torch.bmm(P, F_fine_proj.transpose(1, 2))     # [B, N, DHW]

        # ── heads ─────────────────────────────────────────────────────────────
        class_logits = self.class_head(P)                      # [B, N, num_classes]
        mask_quality = self.quality_head(P)                    # [B, N, 1]
        mask_logits  = Z                                       # [B, N, DHW]

        return class_logits, mask_logits, mask_quality


# ── inference helper ──────────────────────────────────────────────────────────

def get_final_probability_map(
    class_logits:  torch.Tensor,
    mask_logits:   torch.Tensor,
    mask_quality:  torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    threshold:     float = 0.5,
) -> torch.Tensor:
    """
    Collapse N query masks into one voxel probability map.

    Returns  prob_map : [B, D, H, W]  in [0, 1]
    """
    B = class_logits.shape[0]
    D, H, W = spatial_shape

    class_probs = torch.softmax(class_logits, dim=-1)[..., 1]  # [B, N]
    quality     = torch.sigmoid(mask_quality).squeeze(-1)       # [B, N]
    score       = class_probs * quality                         # [B, N]
    mask_probs  = torch.sigmoid(mask_logits)                    # [B, N, DHW]

    prob_map = torch.zeros(B, D * H * W, device=class_logits.device)
    for b in range(B):
        valid = score[b] > threshold
        if valid.any():
            prob_map[b] = mask_probs[b][valid].max(dim=0).values

    return prob_map.view(B, D, H, W)