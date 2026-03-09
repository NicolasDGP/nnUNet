"""
Transformer Query Decoder  (Section 2.2.4 – 2.2.5 of the project report)

Takes the list of decoder feature maps [F_coarse, ..., F_fine] from
UNetResSE3D and iteratively refines N learnable query embeddings via
self-attention and cross-attention at each scale.

Outputs
-------
class_logits  : [B, N, num_classes]   background vs aneurysm per query
mask_logits   : [B, N, D*H*W]         dot-product mask against finest feature
mask_quality  : [B, N, 1]             predicted IoU / quality score
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerQueryDecoder(nn.Module):

    def __init__(
        self,
        num_queries: int  = 16,
        feature_dim: int  = 32,    # must match UNetResSE3D base_features
        num_heads: int    = 4,
        num_scales: int   = 4,     # must match UNetResSE3D depth - 1
        num_classes: int  = 2,
        ffn_dim: int      = 128,
        dropout: float    = 0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        self.num_scales  = num_scales

        # ── learnable query embeddings  P_0  [N, d] ──────────────────────────
        self.query_embed = nn.Embedding(num_queries, feature_dim)

        # ── one transformer layer per decoder scale ───────────────────────────
        self.self_attn_layers  = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads,
                                  dropout=dropout, batch_first=True)
            for _ in range(num_scales)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads,
                                  dropout=dropout, batch_first=True)
            for _ in range(num_scales)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, feature_dim),
            )
            for _ in range(num_scales)
        ])
        self.norm_self  = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_scales)
        ])
        self.norm_cross = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_scales)
        ])
        self.norm_ffn   = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_scales)
        ])

        # ── project decoder features to feature_dim if channels differ ────────
        # We'll build these lazily on first forward so we don't have to hard-code
        # channel counts.  Or pass a list of encoder channel sizes if you prefer.
        self._proj_built = False
        self.feature_proj = None          # filled in _build_projections()

        # ── output heads (Section 2.2.5) ──────────────────────────────────────
        self.class_head        = nn.Linear(feature_dim, num_classes)
        self.mask_quality_head = nn.Linear(feature_dim, 1)

    # ── lazy projection builder ───────────────────────────────────────────────
    def _build_projections(self, decoder_features):
        """Build one Linear projection per scale if channel != feature_dim."""
        projs = []
        for f in decoder_features:
            c = f.shape[1]              # channel dim of this scale
            if c == self.feature_dim:
                projs.append(nn.Identity())
            else:
                projs.append(nn.Linear(c, self.feature_dim))
        self.feature_proj = nn.ModuleList(projs).to(decoder_features[0].device)
        self._proj_built = True

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _flatten(feat: torch.Tensor) -> torch.Tensor:
        """[B, C, D, H, W]  →  [B, D*H*W, C]"""
        B, C, *_ = feat.shape
        return feat.view(B, C, -1).permute(0, 2, 1)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, decoder_features, finest_feature=None):
        """
        Parameters
        ----------
        decoder_features : list of tensors from UNetResSE3D decoder,
                           ordered coarse → fine, each [B, C_l, D_l, H_l, W_l]
        finest_feature   : optional override for mask computation;
                           defaults to decoder_features[-1]
        """
        if finest_feature is None:
            finest_feature = decoder_features[-1]

        # build channel projections on first call
        if not self._proj_built:
            self._build_projections(decoder_features)

        B = decoder_features[0].shape[0]

        # ── initial queries  P_0 : [B, N, d] ─────────────────────────────────
        P = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)   # [B,N,d]

        # ── flatten finest feature for mask computation ───────────────────────
        F_fine = self._flatten(finest_feature)            # [B, DHW, d]
        # project if needed
        if finest_feature.shape[1] != self.feature_dim:
            F_fine = self.feature_proj[-1](F_fine)

        # ── initial mask  Z_0 = P_0 @ F_fine^T  →  [B, N, DHW] ──────────────
        Z = torch.bmm(P, F_fine.transpose(1, 2))

        # ── loop over scales (Section 2.2.4) ─────────────────────────────────
        for t, F_t in enumerate(decoder_features):

            # flatten + project this scale
            F_flat = self._flatten(F_t)                   # [B, D_t H_t W_t, C_t]
            F_flat = self.feature_proj[t](F_flat)         # [B, tokens, d]

            # 1. Self-attention over queries
            P_norm = self.norm_self[t](P)
            P_sa, _ = self.self_attn_layers[t](P_norm, P_norm, P_norm)
            P_tilde  = P + P_sa                           # residual

            # 2. Cross-attention  Q=queries, K/V=CNN decoder features
            P_norm2 = self.norm_cross[t](P_tilde)
            P_ca, _ = self.cross_attn_layers[t](P_norm2, F_flat, F_flat)
            P = P_tilde + P_ca                            # residual

            # 3. Feed-forward network + residual
            P = P + self.ffn_layers[t](self.norm_ffn[t](P))

            # 4. Recompute masks against finest resolution
            Z = torch.bmm(P, F_fine.transpose(1, 2))     # [B, N, DHW]

        # ── output heads ──────────────────────────────────────────────────────
        class_logits  = self.class_head(P)                # [B, N, num_classes]
        mask_quality  = self.mask_quality_head(P)         # [B, N, 1]
        mask_logits   = Z                                 # [B, N, DHW]

        return class_logits, mask_logits, mask_quality


# ── Inference helper ──────────────────────────────────────────────────────────

def get_final_probability_map(
    class_logits:  torch.Tensor,
    mask_logits:   torch.Tensor,
    mask_quality:  torch.Tensor,
    spatial_shape: Tuple,
    threshold:     float = 0.5,
) -> torch.Tensor:
    """
    Collapse N query masks into one voxel-wise probability map.

    Returns
    -------
    prob_map : [B, D, H, W]  values in [0, 1]
    """
    from typing import Tuple   # local import to keep top clean   ← DELETE THIS LINE
    B   = class_logits.shape[0]
    D, H, W = spatial_shape

    # aneurysm class probability
    class_probs    = torch.softmax(class_logits, dim=-1)[..., 1]   # [B, N]
    quality_scores = torch.sigmoid(mask_quality).squeeze(-1)       # [B, N]
    combined       = class_probs * quality_scores                  # [B, N]

    mask_probs = torch.sigmoid(mask_logits)                        # [B, N, DHW]

    prob_map = torch.zeros(B, D * H * W, device=class_logits.device)
    for b in range(B):
        valid = combined[b] > threshold
        if valid.any():
            prob_map[b] = mask_probs[b][valid].max(dim=0).values

    return prob_map.view(B, D, H, W)