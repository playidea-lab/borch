"""ORT's own reduced-precision forms of the comparison's ResNet-18 — f16 and int8 (QDQ).

    uv run --with onnx --with onnxruntime python tests/browser/export_ort_variants.py

The comparison's int8 rows (`docs/INT8.md`) stand beside ORT running the **f32** file, and
that is not the same question as "ORT at its best". This writes, beside each ResNet-18 ONNX
in `borch-ts/test/out/` (the seed-0 `resnet18_cifar.onnx`, and the trained
`resnet18_cifar_trained.onnx` where `train_resnet18_cifar.py` left one):

    <name>_f16.onnx    every float tensor and op in float16, the input and output kept f32
                       (Cast nodes at the ends) — onnxruntime's own converter
    <name>_int8.onnx   static int8 in the QDQ form, per-channel weights, u8 activations —
                       `onnxruntime.quantization.quantize_static`, calibrated on the same
                       images borch's static scales are (the second half of the test slice,
                       or the seeded pixels for the seed-0 network)

Which of those ORT Web's WebGPU provider actually runs on the GPU is not decided here — the
page times each against the wasm provider too, and a WebGPU time that is the wasm time is a
fallback, whatever the session said.
"""
import pathlib
import struct
import sys

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.transformers.float16 import convert_float_to_float16

OUT = pathlib.Path(__file__).resolve().parent.parent.parent / "borch-ts" / "test" / "out"
PIXELS = 3 * 32 * 32
CALIB_BATCH = 16


def seeded(n):
    """The bench's xorshift32 draw from 12345, pixels in [-1, 1) — the seed-0 network's calibration."""
    s = 12345
    pix = np.empty(n * PIXELS, dtype=np.float32)
    for i in range(pix.size):
        s ^= (s << 13) & 0xFFFFFFFF; s ^= s >> 17; s ^= (s << 5) & 0xFFFFFFFF
        pix[i] = s / 0x100000000 * 2 - 1
    return pix.reshape(n, 3, 32, 32)


def slice_second_half():
    """The second half of `cifar10_test.bin` — the images borch's static int8 calibrates on."""
    raw = (OUT / "cifar10_test.bin").read_bytes()
    n = struct.unpack("<I", raw[:4])[0]
    pix = np.frombuffer(raw, dtype=np.float32, count=n * PIXELS, offset=4).reshape(n, 3, 32, 32)
    return pix[n // 2:]


class Reader(CalibrationDataReader):
    def __init__(self, images):
        self.batches = iter(np.array_split(images, max(1, len(images) // CALIB_BATCH)))

    def get_next(self):
        b = next(self.batches, None)
        return None if b is None else {"input": np.ascontiguousarray(b, dtype=np.float32)}


def convert(name, images):
    src = OUT / f"{name}.onnx"
    f16 = convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(f16, str(OUT / f"{name}_f16.onnx"))
    pre = OUT / f"{name}_pre.onnx"
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(OUT / f"{name}_int8.onnx"), Reader(images), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    pre.unlink()
    q = onnx.load(str(OUT / f"{name}_int8.onnx"))
    kinds = sorted({n.op_type for n in q.graph.node})
    print(f"wrote {name}_f16.onnx and {name}_int8.onnx (calibrated on {len(images)} images; ops {' '.join(kinds)})")


def main():
    if not (OUT / "resnet18_cifar.onnx").exists():
        print("no resnet18_cifar.onnx — run tests/browser/export_resnet18.py first", file=sys.stderr)
        return 2
    convert("resnet18_cifar", seeded(64))
    if (OUT / "resnet18_cifar_trained.onnx").exists() and (OUT / "cifar10_test.bin").exists():
        convert("resnet18_cifar_trained", slice_second_half())
    else:
        print("no trained ResNet-18 ONNX beside the seed-0 one — the int8 accuracy row for ORT needs train_resnet18_cifar.py's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
