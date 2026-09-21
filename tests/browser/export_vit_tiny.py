"""The weights the transformer inference comparison shares — timm's ViT-Tiny/16, twice.

    uv run --with torch --with timm --with onnx python tests/browser/export_vit_tiny.py

Writes to `borch-ts/test/out/` (never committed):

    vit_tiny.safetensors   for borch.ts — the names are timm's, which bimm-ts's
                           `vitTinyPatch16` keys the same
    vit_tiny.onnx          for ONNX Runtime Web — opset 17, dynamic batch
    vit_tiny.probe.json    a seeded batch-1 input (1 × 3 × 224 × 224) and torch's logits

The gate is the third file, as for the ResNet: a runtime whose logits differ from torch's
on that input has no speed to compare. The weights are timm's seed-0 draw — a logits gate
wants any weights, a top-1 wants trained ones, and this comparison is a clock: twelve
pre-norm blocks of attention over 197 tokens, layer norms, erf-GELU — the shapes a
transformer forward is made of, against a runtime that has kernels written for each.
"""
import json
import pathlib
import sys

import numpy as np
import timm
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_resnet18 import OUT, safetensors_bytes  # noqa: E402

MODEL = "vit_tiny_patch16_224"
SIDE = 224


def main():
    torch.manual_seed(0)
    model = timm.create_model(MODEL, pretrained=False).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vit_tiny.safetensors").write_bytes(safetensors_bytes(model.state_dict()))
    x = torch.zeros(1, 3, SIDE, SIDE)
    torch.onnx.export(model, x, str(OUT / "vit_tiny.onnx"), input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    # The probe input: xorshift32 from 12345, as the ResNet's, pixels in [-1, 1).
    s = 12345
    pix = np.empty(3 * SIDE * SIDE, dtype=np.float32)
    for i in range(pix.size):
        s ^= (s << 13) & 0xFFFFFFFF; s ^= s >> 17; s ^= (s << 5) & 0xFFFFFFFF
        pix[i] = s / 0x100000000 * 2 - 1
    with torch.no_grad():
        logits = model(torch.from_numpy(pix).reshape(1, 3, SIDE, SIDE)).numpy()[0]
    (OUT / "vit_tiny.probe.json").write_text(json.dumps(
        {"input": pix.tolist(), "shape": [1, 3, SIDE, SIDE], "logits": logits.tolist(),
         "torch": torch.__version__, "timm": timm.__version__, "model": MODEL}))
    params = sum(p.numel() for p in model.parameters())
    print(f"wrote {OUT}: vit_tiny safetensors ({params} params), onnx, probe (|logits| max {np.abs(logits).max():.4f})")


if __name__ == "__main__":
    main()
