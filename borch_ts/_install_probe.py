"""`install("borch")` 뒤에 `import borch` 가 이쪽을 집는가.

브라우저 안에서만 확인할 수 있어서 여기 둔다 — `tests/browser/runner.html` 이
`?lib=borch_ts&install=borch` 로 부르면 이 파일이 돈다.

**묻는 것은 이름 하나다.** 값이 맞는지는 골든 792 건이 이미 본다. 여기서 보는 것은
`import borch as torch` 라고 쓴 코드가 손 안 대고 GPU 위에서 도는가이고, 그것이
`install` 의 존재 이유 전부다.
"""


def check():
    import sys

    import borch_ts

    borch_ts.install("borch")

    # 이 아래는 **사용자가 쓸 그대로**다. borch_ts 라는 이름이 안 나온다.
    import borch as torch
    from borch.optim.lr_scheduler import StepLR          # noqa: F401

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * 2).sum()
    y.backward()

    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU())
    out = model(torch.tensor([[1.0, 2.0, 3.0]]))

    return {
        "이름": sys.modules["borch"].__name__,
        # 하위 경로는 모듈이 아니라 이름 공간 객체다(코어의 `install` 도 그렇다) —
        # `__name__` 이 없다. 심겼는지는 `from … import` 가 통과한 것으로 이미 증명됐다.
        "하위 경로": type(sys.modules["borch.optim.lr_scheduler"]).__name__,
        "StepLR": StepLR.__name__ if hasattr(StepLR, "__name__") else "있다",
        "기울기": x.grad.tolist(),
        "층 출력 모양": list(out.shape),
        # **밑바닥이 무엇인지 말한다.** 이름이 `borch` 여도 도는 것은 WGSL 이다.
        #
        # `einsum` 같은 공통 이름으로는 못 가른다 — 코어에도 있다. 갈리는 것을
        # 물어야 한다: 코어는 float64 를 주고 이쪽은 거절한다(WebGPU 셰이더에
        # 배정도가 없다). 그 거절이 곧 신원이다.
        "밑바닥": _which(x),
    }


def _which(x):
    try:
        x.double()
    except RuntimeError:
        return "borch.ts (WGSL) — float64 를 거절한다"
    except AttributeError:
        return "알 수 없다 — double() 이 아예 없다"
    return "코어 (numpy) — float64 가 있다"
