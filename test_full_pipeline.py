"""
End-to-end test: UNet + Transformer + all three losses.
Run with:  python test_full_pipeline.py
"""
import torch
import torch.nn.functional as F
from nnunetv2.custom_modules.unet_resse import UNetResSE3D
from nnunetv2.custom_modules.transformer_query_decoder import TransformerQueryDecoder
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerResSE import (
    _ResSEWithTransformer,
    soft_dice_loss,
    focal_loss,
    match_queries_to_gt,
    query_mask_loss,
)

print("Building full model...")
unet  = UNetResSE3D(in_channels=1, num_classes=2,
                    base_features=32, depth=4,
                    return_decoder_features=True)
tqd   = TransformerQueryDecoder(num_queries=16, feature_dim=32,
                                num_heads=4, num_scales=3, num_classes=2)
model = _ResSEWithTransformer(unet, tqd)

print("Creating dummy data + ground truth mask...")
x      = torch.randn(1, 1, 32, 32, 32)
# GT mask: mostly background (0), small aneurysm region (1)
target = torch.zeros(1, 1, 32, 32, 32, dtype=torch.long)
target[0, 0, 14:18, 14:18, 14:18] = 1   # 4x4x4 fake aneurysm

print("Running forward pass...")
seg_logits, class_logits, mask_logits, mask_quality = model(x)

print(f"  seg_logits   : {seg_logits.shape}")
print(f"  class_logits : {class_logits.shape}")
print(f"  mask_logits  : {mask_logits.shape}")
print(f"  mask_quality : {mask_quality.shape}")

print("Computing losses...")
loss_seg  = soft_dice_loss(seg_logits, target) + focal_loss(seg_logits, target)
matched   = match_queries_to_gt(mask_logits, target)
loss_mask = query_mask_loss(mask_logits, target, matched)

B, N, _ = class_logits.shape
cls_targets = torch.zeros(B, N, dtype=torch.long)
for b in range(B):
    if matched[b] >= 0:
        cls_targets[b, matched[b]] = 1
loss_cls = F.cross_entropy(class_logits.view(B*N, -1), cls_targets.view(B*N))

total = loss_seg + 0.5 * loss_cls + 1.0 * loss_mask

print(f"  loss_seg  : {loss_seg.item():.4f}")
print(f"  loss_cls  : {loss_cls.item():.4f}")
print(f"  loss_mask : {loss_mask.item():.4f}")
print(f"  total loss: {total.item():.4f}")

print("\nTesting backward pass...")
total.backward()
print("  Gradients flow correctly ✅")

print("\n✅ FULL PIPELINE PASSED — ready to train!")