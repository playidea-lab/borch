"""Whether borch.ts's ResNet-18 is **the same model as the Python side's**, put to real
torch.

    npm run build:ts
    uv run --with playwright --with numpy --with torch python borch-ts/test/samemodel.py --headed

## Why it is needed

The ResNet-18 used by the bench and by the accuracy runner was carried into TypeScript by
**reading `tests/browser/bench.py` by eye**, so it sits outside the golden. Let the block
arrangement or a BN's position differ subtly and both the speed and the accuracy become
**a comparison of two different models** — and that parting is visible only in the values.

## How they are lined up

The naming rules differ between the two languages, so they are lined up **by position.**
The parameters are laid out in order and the list of shapes is weighed first — with the
same structure it is exactly equal, and with a different one it parts there first. Where
the shapes agree, those values go into the torch model as they are and the same input runs
forward and backward, and the output, the loss and the input gradient are weighed.
"""

import pathlib
import sys

import numpy as np

import run as runner
from launch import browser as browser_of, refuse_if_software

# The golden's tolerance. Bit equality is this project's explicit non-goal.
ATOL = 1e-4
RTOL = 1e-4


def _torch_model():
    """Stand the Python side's ResNet-18 up in real torch — the very function the bench
    uses."""
    import importlib.util

    import torch

    path = pathlib.Path(runner.ROOT) / "tests" / "browser" / "bench.py"
    spec = importlib.util.spec_from_file_location("bt_bench_src", path)
    src = spec.loader.get_source("bt_bench_src")
    # `bench.py` runs inside Pyodide, so it imports `js`. Only the part that builds the
    # model is needed here, so that one line is taken out — the model code is not changed
    # by a character.
    src = src.replace("import js\n", "").replace(
        "import borchvision as vision\n", "")
    module = importlib.util.module_from_spec(spec)
    exec(compile(src, str(path), "exec"), module.__dict__)  # noqa: S102
    return module.resnet18(torch)


def _close(name, got, want, bad):
    """Is it inside the tolerance. Where it parts, say at which position and by how
    much."""
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    if got.shape != want.shape:
        bad.append(f"{name}: {got.size} elements against {want.size}")
        return
    gap = np.abs(got - want)
    tol = ATOL + RTOL * np.abs(want)
    if np.any(gap > tol):
        at = int(np.argmax(gap - tol))
        bad.append(f"{name}: [{at}] {got[at]:.9g} ≠ {want[at]:.9g} "
                   f"(max diff {gap.max():.3e})")


def _piece_module(torch, name, shapes):
    """Stand one piece up in real torch. The partner of `dumpPieces` on the TypeScript
    side."""
    import torch.nn as tnn

    if name == "bn":
        return tnn.BatchNorm2d(3)
    if name == "avgpool":
        return tnn.AdaptiveAvgPool2d(1)
    if name == "relu":
        return tnn.ReLU()
    if name == "conv":
        return tnn.Conv2d(3, 4, 3, stride=1, padding=1, bias=False)
    if name in ("block", "blockDown"):
        cin, cout, stride = (3, 3, 1) if name == "block" else (3, 6, 2)

        class Block(tnn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = tnn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
                self.bn1 = tnn.BatchNorm2d(cout)
                self.conv2 = tnn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
                self.bn2 = tnn.BatchNorm2d(cout)
                self.shrinks = stride != 1 or cin != cout
                if self.shrinks:
                    self.dconv = tnn.Conv2d(cin, cout, 1, stride=stride, bias=False)
                    self.dbn = tnn.BatchNorm2d(cout)

            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                side = self.dbn(self.dconv(x)) if self.shrinks else x
                return torch.relu(out + side)

        return Block()
    raise ValueError(f"unknown piece: {name} (shapes {shapes})")


def _check_pieces(pieces):
    """Weigh the pieces one at a time. Folded **with a different weight per position**, so
    the correction terms do not cancel."""
    import torch

    if not pieces:
        return False
    print("\npiece by piece (backpropagated with a different weight per position)")
    failed = False
    for name, d in pieces.items():
        mod = _piece_module(torch, name, d["shapes"])
        mine = [tuple(s) for s in d["shapes"]]
        theirs = [tuple(p.shape) for p in mod.parameters()]
        if mine != theirs:
            print(f"  ✗ {name}: the parameter shapes differ — {mine} against {theirs}")
            failed = True
            continue
        with torch.no_grad():
            for p, v in zip(mod.parameters(), d["params"]):
                p.copy_(torch.tensor(v, dtype=torch.float32).reshape(p.shape))
        x = torch.tensor(d["input"], dtype=torch.float32).reshape(*d["inputShape"])
        x.requires_grad_(True)
        mod.train()
        y = mod(x)
        w = torch.tensor([((i % 7) + 1) * 0.3 for i in range(y.numel())],
                         dtype=torch.float32).reshape(y.shape)
        (y * w).sum().backward()

        bad = []
        _close("output", d["output"], y.detach().numpy(), bad)
        _close("input gradient", d["inputGrad"], x.grad.numpy(), bad)
        for i, (p, g) in enumerate(zip(mod.parameters(), d["paramGrads"])):
            if g is not None and p.grad is not None:
                _close(f"gradient of parameter {i}", g, p.grad.numpy(), bad)
        if bad:
            failed = True
            print(f"  ✗ {name}")
            for line in bad:
                print(f"      {line}")
        else:
            print(f"  ✓ {name}")
    return failed


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "bench.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}"))
            page.goto(f"http://127.0.0.1:{port}/borch-ts/test/samemodel.html")
            page.wait_for_function("window.__borchModel !== undefined",
                                   timeout=600_000)
            dump = page.evaluate("window.__borchModel")
    finally:
        stop()

    if "error" in dump:
        print(f"nothing could be dumped: {dump['error']}", file=sys.stderr)
        return 1

    import torch

    print(f"adapter: {dump['adapter']}")
    # **It does not judge on a software rasteriser.**
    #
    # A headless browser hands over SwiftShader. The piece-by-piece comparison passes
    # there too, and the whole model comes out at a largest difference of 3.9e-03 — which
    # is floating point accumulated across twenty layers rather than a defect (the same
    # code gives 4.6e-06 on Metal). Read that as a parting and a bug that does not exist
    # gets chased; widen the tolerance to cover it and a real parting is missed.
    if refuse_if_software(dump["adapter"], "the largest difference from torch"):
        return 1
    if _check_pieces(dump.get("pieces", {})):
        return 1
    model = _torch_model()
    mine = [tuple(s) for s in dump["shapes"]]
    theirs = [tuple(p.shape) for p in model.parameters()]

    if mine != theirs:
        print(f"**the structures differ** — {len(mine)} parameters against {len(theirs)}")
        for i, (a, b) in enumerate(zip(mine, theirs)):
            if a != b:
                print(f"  position {i}: borch.ts {a} · torch {b}")
                break
        if len(mine) != len(theirs):
            print(f"  the counts differ to begin with: {len(mine)} against {len(theirs)}")
        return 1
    print(f"the structures match — {len(mine)} parameters, the shapes equal down to their "
          "order")

    with torch.no_grad():
        for p, v in zip(model.parameters(), dump["params"]):
            p.copy_(torch.tensor(v, dtype=torch.float32).reshape(p.shape))

    x = torch.tensor(dump["input"], dtype=torch.float32).reshape(-1, 3, 32, 32)
    x.requires_grad_(True)
    y = torch.arange(x.shape[0]) % 10
    model.train()
    out = model(x)
    loss = torch.nn.CrossEntropyLoss()(out, y)
    loss.backward()

    bad = []

    def compare(name, got, want):
        got = np.asarray(got, dtype=np.float64).ravel()
        want = np.asarray(want, dtype=np.float64).ravel()
        if got.shape != want.shape:
            bad.append(f"{name}: {got.size} elements against {want.size}")
            return
        gap = np.abs(got - want)
        tol = ATOL + RTOL * np.abs(want)
        worst = int(np.argmax(gap - tol))
        if np.any(gap > tol):
            bad.append(f"{name}: [{worst}] {got[worst]} ≠ {want[worst]} "
                       f"(max diff {gap.max():.3e})")
        else:
            print(f"  {name} matches (largest difference {gap.max():.3e})")

    compare("the output", dump["output"], out.detach().numpy())
    compare("the loss", [dump["loss"]], [loss.item()])
    compare("the input gradient", dump["inputGrad"], x.grad.numpy())

    if bad:
        print("\nwhere it parted:")
        for line in bad:
            print(f"  ✗ {line}")
        return 1
    print("\nit is the same model — forward, loss and backward agree with real torch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
