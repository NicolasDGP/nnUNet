"""
End-to-end smoke test matching the real architecture.
Run with:  python test_full_pipeline.py
"""
import torch
import torch.nn.functional as F
from nnunetv2.custom_nets.unet_ressne_feature_maps import UNetResSE3D
from nnunetv2.custom_nets.transformer_query_decoder import TransformerQueryDecoder
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerResSE import (
    _ResSEWithTransformer,
    _soft_dice_loss,
    _focal_loss,
    _match_queries_to_gt,
    _query_mask_loss,
)

print("=" * 60)
print("Building UNetResSE3D (matching real nnUNet plan config)...")
print("=" * 60)

N_STAGES = 4
FEATURES = [32, 64, 128, 256]   # doubles each stage
STRIDES  = [1, 2, 2, 2]
PATCH    = (32, 32, 32)          # small for CPU test

unet = UNetResSE3D(
    input_channels        = 1,
    n_stages              = N_STAGES,
    features_per_stage    = FEATURES,
    kernel_sizes          = 3,
    strides               = STRIDES,
    n_conv_per_stage      = 2,
    n_conv_per_stage_decoder = 2,
    num_classes           = 2,
    norm_op               = __import__("torch").nn.InstanceNorm3d,
    nonlin                = __import__("torch").nn.LeakyReLU,
    nonlin_kwargs         = {"inplace": True},
    deep_supervision      = False,
    se_reduction          = 16,
)

x      = torch.randn(1, 1, *PATCH)
target = torch.zeros(1, 1, *PATCH, dtype=torch.long)
target[0, 0, 12:16, 12:16, 12:16] = 1   # fake aneurysm

print("Running UNet forward (with decoder features)...")
seg_logits, decoder_feats = unet(x, return_decoder_features=True)
print(f"  seg_logits : {seg_logits.shape}")
print(f"  decoder levels: {len(decoder_feats)}")
for i, f in enumerate(decoder_feats):
    print(f"    decoder_feats[{i}] : {f.shape}  (channels={f.shape[1]})")

# decoder_feats is full-res-first → channels = FEATURES[:N_STAGES-1]
decoder_channels = [f.shape[1] for f in decoder_feats]
print(f"\nDecoder channels (full-res-first): {decoder_channels}")

print("\nBuilding TransformerQueryDecoder...")
tqd = TransformerQueryDecoder(
    feature_channels = decoder_channels,
    num_queries      = 16,
    embed_dim        = 256,
    num_heads        = 8,
    ffn_dim          = 512,
    num_classes      = 2,
)

print("Running transformer forward...")
class_logits, mask_logits, mask_quality = tqd(decoder_feats)
print(f"  class_logits : {class_logits.shape}")    # [1, 16, 2]
print(f"  mask_logits  : {mask_logits.shape}")     # [1, 16, DHW]
print(f"  mask_quality : {mask_quality.shape}")    # [1, 16, 1]

print("\nBuilding full wrapped model...")
model = _ResSEWithTransformer(unet, tqd)
model.train()

print("Running full model forward (train mode)...")
out = model(x)
assert len(out) == 4, "Expected 4 outputs in train mode"
seg_logits, class_logits, mask_logits, mask_quality = out

print("\nComputing all losses...")
loss_seg  = _soft_dice_loss(seg_logits, target) + _focal_loss(seg_logits, target)
matched   = _match_queries_to_gt(mask_logits, target)
loss_mask = _query_mask_loss(mask_logits, target, matched)

B, N, _ = class_logits.shape
cls_tgt  = torch.zeros(B, N, dtype=torch.long)
for b in range(B):
    if matched[b] >= 0:
        cls_tgt[b, matched[b]] = 1
loss_cls = F.cross_entropy(class_logits.view(B*N, -1), cls_tgt.view(B*N))

total = loss_seg + 0.5*loss_cls + 1.0*loss_mask
print(f"  loss_seg  : {loss_seg.item():.4f}")
print(f"  loss_cls  : {loss_cls.item():.4f}")
print(f"  loss_mask : {loss_mask.item():.4f}")
print(f"  total     : {total.item():.4f}")

print("\nTesting backward pass...")
total.backward()
print("  Gradients flow correctly ✅")

print("\nTesting inference mode (eval)...")
model.eval()
with torch.no_grad():
    out_inf = model(x)
assert isinstance(out_inf, torch.Tensor), "Inference should return single tensor"
print(f"  Inference output: {out_inf.shape}  ✅")

print("\n" + "="*60)
print("✅ FULL PIPELINE PASSED — architecture correctly wired!")
print("="*60)