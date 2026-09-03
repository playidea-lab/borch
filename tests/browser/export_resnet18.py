"""The weights the inference comparison shares — one ResNet-18 (CIFAR) from torch, twice.

    uv run --with torch python tests/browser/export_resnet18.py

Writes to `borch-ts/test/out/` (never committed):

    resnet18_cifar.safetensors   for borch.ts — the names are `bench.ts`'s ResNet18's
    resnet18_cifar.onnx          for ONNX Runtime Web — opset 17, dynamic batch
    resnet18_cifar.probe.json    a seeded batch-1 input and torch's logits for it

The last file is the gate: a runtime whose logits differ from torch's by more than 1e-3
on that input has no speed to compare. Weights are drawn with seed 0 and BatchNorm is
in eval mode with running statistics that are not the defaults, so the check sees the
normalisation as well as the convolutions.
"""
import json
import pathlib
import struct

import numpy as np
import torch
from torch import nn

OUT = pathlib.Path(__file__).resolve().parent.parent.parent / "borch-ts" / "test" / "out"


class Block(nn.Module):
    """`bench.ts`'s Block, with its attribute names — the state dict must key the same."""
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        shrinks = stride != 1 or cin != cout
        self.downConv = nn.Conv2d(cin, cout, 1, stride, 0, bias=False) if shrinks else None
        self.downBn = nn.BatchNorm2d(cout) if shrinks else None

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        side = self.downBn(self.downConv(x)) if self.downConv is not None else x
        return torch.relu(out + side)


class ResNet18(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.body = nn.Sequential(
            Block(64, 64, 1), Block(64, 64, 1), Block(64, 128, 2), Block(128, 128, 1),
            Block(128, 256, 2), Block(256, 256, 1), Block(256, 512, 2), Block(512, 512, 1))
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        h = torch.relu(self.bn(self.stem(x)))
        h = self.body(h).mean(dim=(2, 3))
        return self.fc(h)


def safetensors_bytes(tensors):
    """The format, written by hand: an 8-byte header length, a JSON header, the bytes."""
    header, blobs, offset = {}, [], 0
    for name, t in tensors.items():
        a = np.ascontiguousarray(t.detach().cpu().numpy())
        if a.dtype == np.int64:
            dtype = "I64"
        else:
            a = a.astype(np.float32); dtype = "F32"
        b = a.tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [offset, offset + len(b)]}
        blobs.append(b); offset += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h + b"".join(blobs)


def main():
    torch.manual_seed(0)
    model = ResNet18()
    # Running statistics that are not the defaults — otherwise eval-mode BatchNorm is an
    # identity and a runtime that ignores the buffers would pass the gate.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
                m.weight.normal_(1, 0.1)
                m.bias.normal_(0, 0.1)
    model.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    # `num_batches_tracked` is an int64 counter with no part in inference, and borch.ts's
    # safetensors reader takes F32 only — it is left out and the loader is told so.
    state = {k: v for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    (OUT / "resnet18_cifar.safetensors").write_bytes(safetensors_bytes(state))
    x = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(model, x, str(OUT / "resnet18_cifar.onnx"), input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    # The probe input: xorshift32 from 12345, as bench.ts draws, pixels in [-1, 1).
    s = 12345
    pix = np.empty(3 * 32 * 32, dtype=np.float32)
    for i in range(pix.size):
        s ^= (s << 13) & 0xFFFFFFFF; s ^= s >> 17; s ^= (s << 5) & 0xFFFFFFFF
        pix[i] = s / 0x100000000 * 2 - 1
    with torch.no_grad():
        logits = model(torch.from_numpy(pix).reshape(1, 3, 32, 32)).numpy()[0]
    (OUT / "resnet18_cifar.probe.json").write_text(json.dumps(
        {"input": pix.tolist(), "shape": [1, 3, 32, 32], "logits": logits.tolist(),
         "torch": torch.__version__}))
    print(f"wrote {OUT}: safetensors, onnx, probe (logits {np.round(logits, 4).tolist()})")


if __name__ == "__main__":
    main()
