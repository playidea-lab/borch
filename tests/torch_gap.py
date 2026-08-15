"""진짜 torch 가 가진 이름 중 **우리에게 없는 것**을 센다.

    uv run --with numpy --with torch python tests/torch_gap.py
    uv run --with numpy --with torch python tests/torch_gap.py --extra   # 반대 방향
    uv run --with numpy --with torch python tests/torch_gap.py --show nn # 한 자리만 전부

`tests/conformance.py` 는 **있는 것이 맞는가**를 묻는다. 이 파일은 **무엇이 없는가**를
묻는다. 둘은 다른 질문이고, 앞의 것만 재면 100% 가 나오면서 표면이 얼마나 좁은지는
안 보인다 — 실제로 그렇게 읽힐 뻔했다.

## 없는 것이 다 같은 종류가 아니다

셋으로 갈린다. 이 구분이 이 파일의 요점이다.

- **일부러 거절한다** — `cuda`·`compile`·분산·양자화. 브라우저에 존재할 수 없거나,
  그것을 배우려면 브라우저를 벗어나야 하는 것들이다. 흉내 내면 교훈이 사라진다.
- **커리큘럼 밖이다** — `torch.fft`·`torch.sparse`·`torch.onnx`. 넣을 수는 있지만
  입문 튜토리얼이 안 부른다. 표면이 늘면 조용히 틀릴 자리가 는다.
- **그냥 없다** — 있어야 하는데 아무도 안 물어서 없는 것. **이 목록이 값어치다.**
  실제로 `torch.sum` 이 여기 있었다(메서드로만 있고 모듈 함수가 없었다).

앞의 둘은 판단이고 마지막 하나는 결함이다. 기계는 셋을 못 가르므로 아래
`DELIBERATE` 에 앞의 둘을 적어 둔다 — 적히지 않은 것이 곧 검토 대상이다.
"""

import sys

import torch

import borch

# **일부러 안 하는 것.** 여기 적힌 앞머리로 시작하는 이름은 결함이 아니다.
# 각각 왜인지 적는다 — 안 적으면 다음 사람이 "그냥 안 한 것" 과 못 가른다.
DELIBERATE = {
    "cuda": "브라우저에 CUDA 가 없다. 흉내 내면 GPU 를 배우는 교훈이 사라진다",
    "mps": "같은 이유",
    "xpu": "같은 이유",
    "mtia": "같은 이유",
    "distributed": "한 탭 안이다. 분산을 배우려면 여러 기계로 나가야 한다",
    "compile": "TorchDynamo 는 CPython 바이트코드를 고쳐 쓴다. wasm 에 못 얹는다",
    "jit": "같은 이유",
    "export": "같은 이유",
    "fx": "같은 이유",
    "onnx": "내보내기는 배포의 일이고 여기는 문법 연습이다",
    "quantiz": "양자화는 실제 하드웨어 위에서만 뜻이 있다",
    "sparse": "커리큘럼 밖",
    "fft": "커리큘럼 밖",
    "special": "커리큘럼 밖",
    "futures": "커리큘럼 밖",
    "package": "커리굴럼 밖",
    "profiler": "커리큘럼 밖",
    "utils.tensorboard": "커리큘럼 밖",
    "backends": "고를 백엔드가 하나다",
    "multiprocessing": "브라우저에 프로세스가 없다",
}

# 볼 자리. (보이는 이름, torch 쪽, 우리 쪽)
def _spaces():
    got = [("torch", torch, borch),
           ("Tensor", torch.Tensor, borch.Tensor),
           ("nn", torch.nn, borch.nn),
           ("nn.functional", torch.nn.functional, borch.nn.functional),
           ("optim", torch.optim, borch.optim),
           ("optim.lr_scheduler", torch.optim.lr_scheduler, borch.optim.lr_scheduler),
           ("linalg", torch.linalg, borch.linalg),
           ("utils.data", torch.utils.data, borch.utils.data)]
    return [(name, a, b) for name, a, b in got if b is not None]


def _public(obj):
    """공개 이름만 — **부를 수 있는 것**으로 좁힌다.

    분모를 세 번 고쳤고 그 과정을 남긴다.

    1. `dir()` 그대로 → torch 표면 1,013 개. `Callable`·`Optional`(typing 임포트)과
       `AnyType`·`ArgumentSpec`(C 확장 내부 타입)이 API 로 세어졌다.
    2. `__all__` → 최상위에서 더 나빠졌다. torch 의 최상위 `__all__` 은 손으로 고른
       공개 목록이 아니라 C 내부 타입까지 담은 생성물이라 905 개가 나오고, 겹치는
       것이 3 개로 떨어졌다. **분모가 거짓이면 비율도 거짓이다.**
    3. 지금 — 부를 수 있는 것(함수·클래스)만, 그리고 남의 모듈에서 들어온 이름은 뺀다.

    그래도 완벽하지 않다. 이 수는 "표면의 몇 %" 라고 자랑할 값이 아니라 **어느 자리가
    비었는지 짚는 데** 쓰는 값이다.
    """
    out = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        thing = getattr(obj, name, None)
        if thing is None or not callable(thing):
            continue
        # 남의 모듈에서 들어온 이름(typing 등)은 그 자리의 API 가 아니다.
        home = getattr(thing, "__module__", "") or ""
        if home and not (home.startswith("torch") or home.startswith("borch")):
            continue
        out.add(name)
    return out


def _why(space, name):
    """`DELIBERATE` 에 걸리면 그 사유를, 아니면 None."""
    full = f"{space}.{name}" if space != "torch" else name
    for key, reason in DELIBERATE.items():
        if name.startswith(key) or full.startswith(key) or key in full:
            return reason
    return None


def main(argv):
    show = None
    if "--show" in argv:
        show = argv[argv.index("--show") + 1]
    extra = "--extra" in argv

    total_missing = total_have = 0
    for space, theirs, ours in _spaces():
        # `--show nn` 은 그 자리만 본다. 전부 찍으면 찾던 줄이 묻힌다.
        if show not in (None, "all") and space != show:
            continue
        a, b = _public(theirs), _public(ours)
        gap = sorted(a - b) if not extra else sorted(b - a)
        judged = [(n, _why(space, n)) for n in gap]
        unexplained = [n for n, why in judged if why is None]

        total_have += len(a & b)
        total_missing += len(unexplained)

        if extra:
            print(f"\n{space} — torch 에 없는데 우리에게 있는 것 {len(gap)}개")
            for n in gap:
                print(f"  + {n}")
            continue

        covered = len(a & b)
        print(f"\n{space} — torch {len(a)}개 중 {covered}개 있다 "
              f"({covered * 100 // max(1, len(a))}%)")
        print(f"  일부러 안 함 {len(judged) - len(unexplained)} · "
              f"**검토 대상 {len(unexplained)}**")
        if show in (space, "all"):
            for n in unexplained:
                print(f"    ? {n}")
        elif unexplained:
            head = ", ".join(unexplained[:8])
            more = f" … 외 {len(unexplained) - 8}" if len(unexplained) > 8 else ""
            print(f"    {head}{more}")

    if not extra:
        print(f"\n합계 — 겹치는 이름 {total_have}개, 설명 안 된 빈자리 {total_missing}개")
        print("  `--show <자리>` 로 그 자리를 전부 편다. `--show all` 은 전부.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
