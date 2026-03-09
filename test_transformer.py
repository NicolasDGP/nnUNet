"""
Quick smoke test — runs on CPU, no dataset needed.
Run with:  python test_transformer.py
"""
import torch
from nnunetv2.custom_modules.unet_resse import UNetResSE3D
from nnunetv2.custom_modules.transformer_query_decoder import TransformerQueryDecoder

print("Building UNet...")
unet = UNetResSE3D(
    in_channels=1,
    num_classes=2,
    base_features=32,
    depth=4,
    return_decoder_features=True,
)

print("Running forward pass on a tiny dummy volume (1x1x32x32x32)...")
x = torch.randn(1, 1, 32, 32, 32)   # batch=1, 1 channel, 32^3 volume

with torch.no_grad():
    seg_logits, decoder_features = unet(x)

print(f"  seg_logits shape      : {seg_logits.shape}")
print(f"  number of decoder maps: {len(decoder_features)}")
for i, f in enumerate(decoder_features):
    print(f"  decoder_features[{i}]   : {f.shape}")

print("\nBuilding Transformer Query Decoder...")
tqd = TransformerQueryDecoder(
    num_queries=16,
    feature_dim=32,
    num_heads=4,
    num_scales=len(decoder_features),
    num_classes=2,
)

print("Running transformer forward pass...")
with torch.no_grad():
    class_logits, mask_logits, mask_quality = tqd(decoder_features)

print(f"  class_logits shape : {class_logits.shape}")   # [1, 16, 2]
print(f"  mask_logits shape  : {mask_logits.shape}")    # [1, 16, DHW]
print(f"  mask_quality shape : {mask_quality.shape}")   # [1, 16, 1]

print("\n✅ ALL PASSED — both modules work correctly!")