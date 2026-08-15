"""borch_ts — **borch.ts 위에 얹은 얇은 결속(binding).**

`import borch_ts as torch` 로 쓰고, 밑에서는 TF.js 가 아니라 borch.ts(직접 쓴 WGSL)가
돈다. 자매(`borch_webgpu`)와 같은 자리를 노리되 밑바닥이 다르다.

## 왜 이것이 얇은가

자매는 5,296 줄이다. TF.js 가 주는 것이 원시 연산 104 개뿐이라 autograd 테이프와
`nn.Module` 과 옵티마이저를 **파이썬으로 다시 구현**해야 했다.

borch.ts 에는 그것이 이미 다 있고 골든 845 건이 붙잡고 있다. 그러니 파이썬이 할 일은
**이름을 바꿔 끼우는 것**뿐이다 — 그리고 그것마저 손으로 안 적는다. 모듈과 텐서의
`__getattr__` 이 `masked_select` 를 `maskedSelect` 로 바꿔 넘기고, 없으면 멈춘다.

## 동기로 도는 이유 — 이건 재봤다

WebGPU 에 동기 읽기가 없다. borch.ts 의 `toArray()` 는 약속을 돌려준다. 그대로
얹으면 `await loss.item()` 이 되고, 그러면 **"튜토리얼 코드를 임포트만 바꿔
돌린다"** 는 이 프로젝트의 유일한 주장이 깨진다.

Pyodide 의 `run_sync` 가 그 자리를 메운다(JSPI 위에 선다). 실측했다 —
`tests/browser/sync_probe.py` 가 `[2,4,6]` 을 낸다. **조건이 하나 있다:** 파이썬에
들어올 때 비동기 진입(`runPythonAsync`)이어야 한다. 동기 진입이면
`RuntimeError: No suspender` 로 멈추는데, 라이브러리의 한계가 아니라 그 스택에
중단할 자리가 없어서다. 러너는 이미 비동기로 들어간다.

## 브라우저 안에서만 돈다

`js.borch` 를 부른다. 네이티브 CPython 에서 임포트하면 바로 멈춘다 — 조용히 다른
것으로 폴백하면 "GPU 로 돌렸다" 고 착각하게 되고, 그건 이 프로젝트가 가장 싫어하는
종류다. 자매가 같은 이유로 같은 자리에서 멈춘다.

## 지금 어디까지인가

**전부가 아니다.** 이것은 자매를 borch.ts 위로 옮길 수 있는지를 재는 조각이고, 몇
건이 지나는지가 곧 진도다. 되는 것과 안 되는 것의 경계는 골든이 붙잡는다 —
`nn`·`optim` 은 아직 안 붙였고, 그 케이스들은 `AttributeError` 로 실패한다. 없는
것을 근사해서 초록을 만들지 않는다.
"""

try:                                                    # pragma: no cover
    import js as _js
except ImportError as exc:                              # pragma: no cover
    raise ImportError(
        "borch_ts 는 브라우저 안에서만 돈다 — Pyodide 가 아니면 `js` 가 없다.\n"
        "  네이티브에서는 `borch`(numpy)를 써라.") from exc

if getattr(_js, "borch", None) is None:                 # pragma: no cover
    raise ImportError(
        "`js.borch` 가 없다 — 페이지가 borch.ts 를 싣고 `await init()` 을 부른 뒤\n"
        "  전역에 두어야 한다. 조용히 다른 것으로 돌지 않으려고 여기서 멈춘다.")

from ._base import Tensor, tensor                        # noqa: E402,F401
# **이름 붙은 것을 먼저 들여온다.** 모듈의 `__getattr__` 은 여기 없는 이름만 받으므로,
# `linalg`·`einsum` 처럼 첫 인자가 텐서가 아닌 것들이 그쪽으로 새면 안 된다 —
# 실제로 `linalg` 가 함수로 잡혀서 `linalg.cholesky` 가 통째로 실패했다.
from ._ops import (                                      # noqa: E402,F401
    aminmax, arange, as_tensor, cat, chunk, clamp, clip, einsum, eye, flatten,
    flip,
    full, linalg, linspace, matrix_power, no_grad, ones, pow, quantile, rand,
    randn, repeat_interleave, split, squeeze, stack, where, zeros,
)
from ._ops import __getattr__                            # noqa: E402,F401
from . import _nn as nn, _optim as optim                 # noqa: E402,F401

# dtype 은 borch.ts 에서 float32 저장 위의 이름표다. 여기서도 이름으로 두되,
# 보이는 이름은 torch 의 것이다 — 골든이 `torch.float32` 를 답으로 굳혔다.
from ._base import _DType                                # noqa: E402

def install(name="borch_ts", modules=None):
    """`from <name>.nn import Linear` 이 통하게 하위 경로를 심는다.

    **기본은 심지 않는 것이다.** `import borch_ts as torch` 로 `torch.nn.Linear` 는
    그냥 닿는다 — 속성 접근이라 별칭이면 된다. 교재 코드의 대부분이 그 모양이다.

    안 닿는 것은 `from … import` 다. 그것은 `sys.modules` 에 등록된 **경로**를 보고,
    별칭은 그 파일 안의 이름 하나일 뿐이라 거기까지 못 간다. 경계는 재봤고
    `tests/test_alias.py` 가 값으로 붙잡고 있다.

    **이름은 기본이 자기 이름이다.** `torch` 로 심으면 남의 라이브러리가 하는
    `import torch` 까지 축소판을 받아서, 섞인 환경에서는 원인을 못 찾는 오류가 된다.
    자기 이름으로 심으면 하위 경로가 열리면서 남의 코드는 안 건드린다.

    코어의 `install()` 과 같은 모양이고 같은 이유다 — 하위 경로를 손으로 적으면
    어긋난다. 코어는 실제로 어긋나서 `from torch.optim.lr_scheduler import StepLR` 이
    교재 본문에서 멈춘 적이 있다.
    """
    import sys

    modules = sys.modules if modules is None else modules
    modules[name] = sys.modules[__name__]
    registered = [name]
    for path, mod in (("nn", nn), ("optim", optim), ("linalg", linalg),
                      ("nn.functional", nn.functional),
                      ("nn.utils", nn.utils), ("nn.utils.rnn", nn.utils.rnn),
                      ("optim.lr_scheduler", optim.lr_scheduler)):
        full = f"{name}.{path}"
        modules[full] = mod
        registered.append(full)
    return registered


float32 = _DType("float32")
int64 = _DType("int64")
# **`bool` 은 모듈 전역에 두지 않는다.** 파이썬 내장을 가려서 `isinstance(x, bool)` 이
# 깨지고, 그 자리가 `_DType` 의 `__repr__` 로 새어 나왔다. 골든은 `L.bool` 로 부르므로
# 모듈의 `__getattr__` 이 그 이름만 골라 준다 — `_ops.__getattr__` 이 그 일을 한다.
bool_ = _DType("bool")

