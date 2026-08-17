"""borch_webgpu — **borch.ts 위에 얹은 얇은 결속(binding).**

`import borch_webgpu as torch` 로 쓰고, 밑에서 직접 쓴 WGSL 커널이 돈다.

## 이 이름은 한 번 주인이 바뀌었다

전에는 같은 이름이 TF.js 판이었다. 그쪽은 **5,307 줄**이었는데, TF.js 가 주는 것이
원시 연산 104 개뿐이라 autograd 테이프와 `nn.Module` 과 옵티마이저를 파이썬으로
다시 구현해야 했기 때문이다. 이쪽은 **5,314 줄**이다 — borch.ts 에 그것이 이미 다
있어서 파이썬이 할 일이 이름을 바꿔 끼우는 것뿐이다.

`_data.py` 는 양쪽에서 거의 같다 — 저쪽에서 그대로 옮겨왔고 numpy 와 OPFS 위라
밑바닥과 상관이 없다. 줄어든 것은 전부 그 밖에서 줄었다.

이름은 사용자가 보는 것을 말한다: 브라우저에서, GPU 로. 그 뜻이 안 바뀌었으므로
이름도 안 바꿨고, 갈아탄 것은 밑바닥이다. 재보니 같은 벤치에서 배치 64 기준
154.7ms → 123.4ms 였고, 골든 859 건이 양쪽에 같은 질문을 했다.

이름 바꿔 끼우는 것마저 손으로 안 적는다. 모듈과 텐서의 `__getattr__` 이
`masked_select` 를 `maskedSelect` 로 바꿔 넘기고, 없으면 멈춘다. **넘기는 인자를
저쪽이 안 받으면 그것도 멈춘다** — JS 는 남는 인자를 조용히 버리므로, 그 확인이
없던 동안 `sum(dim=1)` 이 축을 무시한 전체 합을 값으로 냈다.

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
종류다. 코어(`borch`, numpy)가 네이티브 쪽 답이다.

## 지금 어디까지인가

골든 **1830 건 전부**를 지난다. 코어(numpy)는 그중 1777 건을 보는데, 나머지 53 건은
코어가 일부러 거절하는 것들(1·3 차원 합성곱, 랭크 7·8)이라 안 묻는다.

경계는 골든이 붙잡는다. 없는 것을 근사해서 초록을 만들지 않는다.
"""

try:                                                    # pragma: no cover
    import js as _js
except ImportError as exc:                              # pragma: no cover
    raise ImportError(
        "borch_webgpu 는 브라우저 안에서만 돈다 — Pyodide 가 아니면 `js` 가 없다.\n"
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
    aminmax, arange, argmax, argmin, as_tensor, cat, chunk, clamp, clip, einsum, eye,
    flatten, flip,
    full, linalg, linspace, matrix_power, memory, no_grad, norm, numel,
    ones, pow,
    quantile, rand, randn, repeat_interleave, scope, split, squeeze, stack,
    sum, swapdims, transpose, where, zeros,
    # torch 가 두 번째 이름으로 주는 것들 — 조합에 이름만 붙인다.
    add, adjoint, block_diag, broadcast_shapes, broadcast_tensors, broadcast_to,
    column_stack, concat, concatenate, div, divide, dstack, floor_divide, fmod,
    hstack, moveaxis, mul, multiply, remainder, row_stack, rsub, sub, subtract, t,
    true_divide, vstack,
    # 계산 자체가 없던 것들.
    cross, empty_like, float_power, fmax, fmin, inner, isclose, isin, isneginf,
    isposinf, isreal, kron, lerp, logical_xor, logspace, meshgrid, nan_to_num,
    rand_like, randint_like, randn_like, scalar_tensor, std_mean, var_mean, vdot,
    # 색인으로 쓰는 쪽.
    bucketize, index_add, index_copy, index_fill, scatter, scatter_add,
    searchsorted, take, take_along_dim,
    # 수치 계열. `lgamma`·`digamma`·`erfinv` 는 WGSL 쪽에 있어 `__getattr__` 이 넘긴다.
    cdist, corrcoef, cov, cumulative_trapezoid, tensordot, trapezoid,
    # 반사자 꼴 QR. `linalg.householder_product` 의 짝이라 최상위에도 있다.
    geqrf,
    # 창 함수. 텐서가 아니라 **개수**를 받으므로 `__getattr__` 이 못 넘긴다.
    bartlett_window, blackman_window, hamming_window, hann_window, kaiser_window,
    # 참거짓과 정수에서 하는 일이 다른 것, 그리고 제자리가 아닌 `fill`.
    bitwise_not, fill,
    # 모양·색인. **`as_strided` 는 torch 에서 뷰지만 우리는 사본이다.**
    cartesian_prod, chain_matmul, combinations, index_put, index_put_,
    split_with_sizes, tensor_split, tril_indices, triu_indices,
    unique_consecutive, unravel_index, vander,
    # **희소 전용이라 없다.** 이름은 두고 그 자리에서 멈춘다.
    sspaddmm,
    # 최상위 선형대수. **`linalg` 쪽과 이름이 겹치는 셋만 손으로 적혀 있다** —
    # 나머지는 `__getattr__` 이 첫 인자의 메서드로 넘긴다.
    lu, lu_solve, lu_unpack,
    # 통계. 난수 넷은 값이 아니라 **끝값**으로 굳는다. 뒤의 셋은 이름만 두고 거절한다.
    bernoulli, binomial, hash_tensor, histogramdd, istft, normal, poisson,
    stft, trapz,
    # 복소수의 이웃. `imag` 만 거절인데 **torch 자신이 그렇게 한다**(실측).
    angle, asarray, conj, conj_physical, conj_physical_, empty_permuted,
    empty_strided, frombuffer, imag, is_complex, is_conj, is_neg, real,
    resolve_conj, resolve_neg, range_top as range,
    # **최상위에만 있는 이름들.** `F` 쪽과 서명이 다른 것도 있어서 자리를 옮겨 준다.
    alpha_dropout_, batch_norm, ctc_loss, dropout_, feature_alpha_dropout_,
    feature_dropout, feature_dropout_, grid_sampler, max_pool1d_with_indices,
    nan_to_num_,
    # 기울기 모드.
    enable_grad, inference_mode, is_grad_enabled, is_inference,
    is_inference_mode_enabled, set_grad_enabled,
    # 난수 상태.
    get_rng_state, initial_seed, seed, set_rng_state,
    # 살펴보기.
    can_cast, finfo, get_default_dtype, iinfo, is_distributed, is_floating_point,
    is_nonzero, is_same_size, is_signed, is_storage, is_tensor, promote_types,
    set_default_dtype, typename,
)
from ._ops import (                                      # noqa: E402,F401
    Generator, manual_seed, randint, randperm,
)
from ._data import (                                     # noqa: E402,F401
    ConcatDataset, DataLoader, Dataset, RandomSampler, SequentialSampler, Subset,
    TensorDataset, WeightedRandomSampler, backend, cache_get, cache_put, cuda,
    decode_cifar10, fetch_cached, load, random_split, save, utils,
)
from ._ops import __getattr__                            # noqa: E402,F401
from . import _nn as nn, _optim as optim                 # noqa: E402,F401

# dtype 은 borch.ts 에서 float32 저장 위의 이름표다. 여기서도 이름으로 두되,
# 보이는 이름은 torch 의 것이다 — 골든이 `torch.float32` 를 답으로 굳혔다.
from ._base import _DType                                # noqa: E402

def install(name="borch_webgpu", modules=None):
    """`from <name>.nn import Linear` 이 통하게 하위 경로를 심는다.

    **기본은 심지 않는 것이다.** `import borch_webgpu as torch` 로 `torch.nn.Linear` 는
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

