"""borch.ts 의 ResNet-18 이 **파이썬 쪽과 같은 모델인지** 진짜 torch 로 맞춰본다.

    npm run build:ts
    uv run --with playwright --with numpy --with torch python borch-ts/test/samemodel.py --headed

## 왜 필요한가

벤치와 정확도에 쓴 ResNet-18 은 `tests/browser/bench.py` 를 **눈으로 읽어** TypeScript
로 옮긴 것이라 골든 밖에 있다. 블록 구성이나 BN 자리가 미묘하게 다르면 속도도
정확도도 **다른 모델끼리 비교한 것**이 되는데, 그 갈림은 값으로만 보인다.

## 어떻게 맞추는가

이름 짓는 규칙이 두 언어에서 다르므로 **자리로 맞춘다.** 파라미터를 순서대로 놓고
모양 목록을 먼저 견준다 — 구조가 같으면 정확히 같고, 다르면 거기서 먼저 갈린다.
모양이 맞으면 그 값을 torch 모델에 그대로 넣고 같은 입력으로 순방향·역방향을 돌려
출력·손실·입력기울기를 견준다.
"""

import pathlib
import sys

import numpy as np

import run as runner
from launch import browser as browser_of, refuse_if_software

# 골든과 같은 허용 오차. 비트 동등은 이 프로젝트의 명시적 비목표다.
ATOL = 1e-4
RTOL = 1e-4


def _torch_model():
    """파이썬 쪽 ResNet-18 을 진짜 torch 로 세운다 — 벤치가 쓰는 바로 그 함수다."""
    import importlib.util

    import torch

    path = pathlib.Path(runner.ROOT) / "tests" / "browser" / "bench.py"
    spec = importlib.util.spec_from_file_location("bt_bench_src", path)
    src = spec.loader.get_source("bt_bench_src")
    # `bench.py` 는 Pyodide 안에서 도는 파일이라 `js` 를 들여온다. 여기서는 모델을
    # 만드는 부분만 필요하므로 그 줄만 뺀다 — 모델 코드는 한 글자도 안 바꾼다.
    src = src.replace("import js\n", "").replace(
        "import borchvision as vision\n", "")
    module = importlib.util.module_from_spec(spec)
    exec(compile(src, str(path), "exec"), module.__dict__)  # noqa: S102
    return module.resnet18(torch)


def _close(name, got, want, bad):
    """허용 오차 안인가. 갈리면 어느 자리에서 얼마나 갈렸는지 적는다."""
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    if got.shape != want.shape:
        bad.append(f"{name}: 원소 수 {got.size} 대 {want.size}")
        return
    gap = np.abs(got - want)
    tol = ATOL + RTOL * np.abs(want)
    if np.any(gap > tol):
        at = int(np.argmax(gap - tol))
        bad.append(f"{name}: [{at}] {got[at]:.9g} ≠ {want[at]:.9g} "
                   f"(max diff {gap.max():.3e})")


def _piece_module(torch, name, shapes):
    """조각 하나를 진짜 torch 로 세운다. TypeScript 쪽 `dumpPieces` 와 짝이다."""
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
    raise ValueError(f"모르는 조각: {name} (모양 {shapes})")


def _check_pieces(pieces):
    """조각을 하나씩 견준다. **자리마다 다른 가중치로** 접었으므로 보정항이 안 상쇄된다."""
    import torch

    if not pieces:
        return False
    print("\n조각별 대조 (자리마다 다른 가중치로 역전파)")
    failed = False
    for name, d in pieces.items():
        mod = _piece_module(torch, name, d["shapes"])
        mine = [tuple(s) for s in d["shapes"]]
        theirs = [tuple(p.shape) for p in mod.parameters()]
        if mine != theirs:
            print(f"  ✗ {name}: 파라미터 모양이 다르다 — {mine} 대 {theirs}")
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
        _close("출력", d["output"], y.detach().numpy(), bad)
        _close("입력기울기", d["inputGrad"], x.grad.numpy(), bad)
        for i, (p, g) in enumerate(zip(mod.parameters(), d["paramGrads"])):
            if g is not None and p.grad is not None:
                _close(f"파라미터{i}기울기", g, p.grad.numpy(), bad)
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
        print(f"방출물이 없다: {dist}\n  먼저: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("pageerror", lambda e: print(f"  [브라우저 예외] {e}"))
            page.goto(f"http://127.0.0.1:{port}/borch-ts/test/samemodel.html")
            page.wait_for_function("window.__borchModel !== undefined",
                                   timeout=600_000)
            dump = page.evaluate("window.__borchModel")
    finally:
        stop()

    if "error" in dump:
        print(f"뽑지 못했다: {dump['error']}", file=sys.stderr)
        return 1

    import torch

    print(f"어댑터: {dump['adapter']}")
    # **소프트웨어 래스터라이저에서는 판정하지 않는다.**
    #
    # 헤드리스 브라우저는 SwiftShader 를 준다. 조각 대조는 거기서도 통과하는데 전체
    # 모델은 최대차 3.9e-03 이 나온다 — 스무 층을 지나며 쌓인 부동소수점 차이지 결함이
    # 아니다(같은 코드가 Metal 에서는 4.6e-06 이다). 그것을 갈림으로 읽으면 없는 버그를
    # 쫓게 되고, 반대로 허용 오차를 그만큼 늘리면 진짜 갈림을 놓친다.
    if refuse_if_software(dump["adapter"], "torch 와의 최대차"):
        return 1
    if _check_pieces(dump.get("pieces", {})):
        return 1
    model = _torch_model()
    mine = [tuple(s) for s in dump["shapes"]]
    theirs = [tuple(p.shape) for p in model.parameters()]

    if mine != theirs:
        print(f"**구조가 다르다** — 파라미터 {len(mine)}개 대 {len(theirs)}개")
        for i, (a, b) in enumerate(zip(mine, theirs)):
            if a != b:
                print(f"  자리 {i}: borch.ts {a} · torch {b}")
                break
        if len(mine) != len(theirs):
            print(f"  개수부터 다르다: {len(mine)} 대 {len(theirs)}")
        return 1
    print(f"구조 일치 — 파라미터 {len(mine)}개, 모양이 순서까지 같다")

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
            bad.append(f"{name}: 원소 수 {got.size} 대 {want.size}")
            return
        gap = np.abs(got - want)
        tol = ATOL + RTOL * np.abs(want)
        worst = int(np.argmax(gap - tol))
        if np.any(gap > tol):
            bad.append(f"{name}: [{worst}] {got[worst]} ≠ {want[worst]} "
                       f"(max diff {gap.max():.3e})")
        else:
            print(f"  {name} 일치 (최대차 {gap.max():.3e})")

    compare("출력", dump["output"], out.detach().numpy())
    compare("손실", [dump["loss"]], [loss.item()])
    compare("입력 기울기", dump["inputGrad"], x.grad.numpy())

    if bad:
        print("\n갈린 곳:")
        for line in bad:
            print(f"  ✗ {line}")
        return 1
    print("\n같은 모델이다 — 순방향·손실·역방향이 진짜 torch 와 맞는다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
