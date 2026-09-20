"""A **trained** ResNet-18 (CIFAR-10) for the accuracy gates — `docs/INT8.md` Step 4.

    uv run --with torch --with torchvision python tests/browser/train_resnet18_cifar.py [--epochs=20] [--slice=2000]

The comparison's weights (`export_resnet18.py`) are torch's seed-0 draw: right for a
logits gate, meaningless for a top-1. This trains the same network — the same class, so
the state dict keys the same names — on CIFAR-10 for a few minutes on a GPU, and writes
beside the seed-0 files (in `borch-ts/test/out/`, never committed):

    resnet18_cifar_trained.safetensors   the trained weights, for borch.ts
    resnet18_cifar_trained.onnx          the same for ONNX Runtime Web
    resnet18_cifar_trained.probe.json    the seeded probe input and torch's logits on it
    cifar10_test.bin                     a labelled slice of the test set, normalised as
                                         trained: u32 count · f32 pixels (NCHW) · u32 labels

and prints torch's top-1 on the slice and on the whole test set — the reference the
browser's f32 and int8 forwards are held to. CIFAR-10 is downloaded into `out/cifar`.
"""
import json
import pathlib
import struct
import sys
import time

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_resnet18 import OUT, ResNet18, safetensors_bytes  # noqa: E402

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def main(argv):
    epochs = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--epochs=")), "20"))
    slice_n = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--slice=")), "2000"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    root = OUT / "cifar"
    norm = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=500, shuffle=False, num_workers=4)

    model = ResNet18().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, epochs=epochs, steps_per_epoch=len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        seen, correct, total_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            seen += y.numel(); correct += (logits.argmax(1) == y).sum().item(); total_loss += loss.item() * y.numel()
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch + 1}/{epochs}  loss {total_loss / seen:.3f}  train {100 * correct / seen:.1f}%  test {100 * acc:.2f}%  {time.time() - t0:.0f}s  @test_top1={acc:.4f}", flush=True)

    model.eval().cpu()
    state = {k: v for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    (OUT / "resnet18_cifar_trained.safetensors").write_bytes(safetensors_bytes(state))
    x = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(model, x, str(OUT / "resnet18_cifar_trained.onnx"), input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    s = 12345
    pix = np.empty(3 * 32 * 32, dtype=np.float32)
    for i in range(pix.size):
        s ^= (s << 13) & 0xFFFFFFFF; s ^= s >> 17; s ^= (s << 5) & 0xFFFFFFFF
        pix[i] = s / 0x100000000 * 2 - 1
    with torch.no_grad():
        logits = model(torch.from_numpy(pix).reshape(1, 3, 32, 32)).numpy()[0]
    (OUT / "resnet18_cifar_trained.probe.json").write_text(json.dumps(
        {"input": pix.tolist(), "shape": [1, 3, 32, 32], "logits": logits.tolist(), "torch": torch.__version__}))
    # The labelled slice, normalised as trained, and torch's top-1 on it.
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x); ys.append(y)
        if sum(t.shape[0] for t in xs) >= slice_n: break
    xs = torch.cat(xs)[:slice_n]; ys = torch.cat(ys)[:slice_n]
    with torch.no_grad():
        slice_acc = (model(xs).argmax(1) == ys).float().mean().item()
    with (OUT / "cifar10_test.bin").open("wb") as f:
        f.write(struct.pack("<I", slice_n))
        f.write(np.ascontiguousarray(xs.numpy(), dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(ys.numpy(), dtype=np.uint32).tobytes())
    full_acc = evaluate(model.to(device), test_loader, device)
    print(f"wrote {OUT}: trained safetensors, onnx, probe, cifar10_test.bin ({slice_n} images)")
    print(f"torch top-1: slice {100 * slice_acc:.2f}%  full test set {100 * full_acc:.2f}%  @slice_top1={slice_acc:.4f} @full_top1={full_acc:.4f}")
    return 0


def evaluate(model, loader, device):
    model.eval()
    correct, seen = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); seen += y.numel()
    return correct / seen


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
