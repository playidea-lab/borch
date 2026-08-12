"""적합성 하네스 — 표에서 대조 케이스를 뽑아낸다.

손으로 쓴 대조 테스트(`test_diff.py`)는 76개에서 멈춘다. 사람이 쓰는 속도가 한계이고,
쓴 사람이 생각한 경우만 들어간다. 이 파일은 대신 **명세를 표로 두고 케이스를 생성한다** —
연산 하나에 모양·dtype·인자 조합을 곱하면 수백 개가 나온다.

## 충실도 등급

    T1 값    allclose(1e-5) 로 같은 값·같은 모양      ← 지원 범위 전부가 목표
    T2 오류  같은 조건에서 같은 종류의 예외            ← 주요 오류
    T3 표현  print() 결과가 같음                     ← 흔한 것부터
    T4 비트  동일 비트                               ← 명시적 비목표

T4 를 목표에서 뺀 이유는 재보면 알 수 있다. float32 10만 개를 더하면 numpy 와 torch 가
1.4e-4 만큼 다르다 — float32 eps 의 1,000배다. 합산 **순서**가 다르기 때문이고,
torch 쪽 순서는 SIMD 폭과 스레드 수에 따라 또 달라진다. 쫓을 수 있는 목표가 아니다.

## 쓰는 법

    uv run --with pytest --with numpy --with torch pytest tests/conformance.py -q
    uv run --with numpy --with torch python tests/conformance.py      # 점수만
"""

import importlib.util
import itertools
import pathlib

import numpy as np
import torch as real

_spec = importlib.util.spec_from_file_location(
    "nanotorch", pathlib.Path(__file__).resolve().parent.parent / "nanotorch.py")
nano = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nano)

ATOL = 1e-5


# ---------------------------------------------------------------- 입력 명세

def make(shape, kind="float", seed=0):
    """같은 데이터를 양쪽 텐서로. 무작위는 씨앗을 고정해 재현 가능하게."""
    rng = np.random.default_rng(seed)
    if kind == "float":
        arr = rng.standard_normal(shape).astype(np.float32)
    elif kind == "positive":
        arr = np.abs(rng.standard_normal(shape)).astype(np.float32) + 0.1
    elif kind == "int":
        arr = rng.integers(0, 5, shape).astype(np.int64)
    elif kind == "ramp":
        arr = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    else:
        raise ValueError(kind)
    return arr


SHAPES_1D = [(3,), (7,)]
SHAPES_2D = [(2, 3), (4, 4), (1, 5)]
SHAPES_3D = [(2, 3, 4)]


# ---------------------------------------------------------------- 케이스 생성

class Case:
    def __init__(self, name, run_real, run_nano, tier="T1", grad=False):
        self.name = name
        self.run_real = run_real
        self.run_nano = run_nano
        self.tier = tier
        self.grad = grad


def unary(fn_name, shapes, kind="float", grad=True):
    """torch.fn(x) 형태. 값과 (원하면) 기울기를 함께 본다."""
    out = []
    for shape in shapes:
        arr = make(shape, kind)
        out.append(Case(
            f"{fn_name}{shape}",
            lambda a=arr, f=fn_name: getattr(real, f)(real.tensor(a)),
            lambda a=arr, f=fn_name: getattr(nano, f)(nano.tensor(a)),
            grad=grad,
        ))
    return out


def method(expr, shapes, kind="float", label=None, grad=False):
    """'t.sum(dim=0)' 처럼 메서드 호출식을 문자열로 받아 양쪽에 적용한다."""
    out = []
    for shape in shapes:
        arr = make(shape, kind)
        src = f"lambda t: t.{expr}"
        out.append(Case(
            f"{label or expr}{shape}",
            lambda a=arr, s=src: eval(s)(real.tensor(a)),          # noqa: S307
            lambda a=arr, s=src: eval(s)(nano.tensor(a)),          # noqa: S307
            grad=grad,
        ))
    return out


def binary(op, shape_pairs, kind="float"):
    out = []
    for sa, sb in shape_pairs:
        a, b = make(sa, kind, 0), make(sb, kind, 1)
        src = f"lambda x, y: x {op} y"
        out.append(Case(
            f"a {op} b {sa}·{sb}",
            lambda x=a, y=b, s=src: eval(s)(real.tensor(x), real.tensor(y)),   # noqa: S307
            lambda x=a, y=b, s=src: eval(s)(nano.tensor(x), nano.tensor(y)),   # noqa: S307
            grad=True,
        ))
    return out


BROADCAST_PAIRS = [
    ((2, 3), (2, 3)),
    ((2, 3), (3,)),
    ((2, 3), (2, 1)),
    ((4, 1, 3), (2, 3)),
    ((5,), ()),
]


def build_cases():
    cases = []

    # 원소별 함수
    for fn in ("sigmoid", "relu", "tanh", "exp"):
        cases += unary(fn, SHAPES_1D + SHAPES_2D)
    for fn in ("log", "sqrt"):
        cases += unary(fn, SHAPES_1D + SHAPES_2D, kind="positive")
    cases += unary("abs", SHAPES_1D + SHAPES_2D)

    # 산술과 브로드캐스팅
    for op in ("+", "-", "*"):
        cases += binary(op, BROADCAST_PAIRS)

    # 축약 — dim × keepdim 조합
    for fn, dims in [("sum", [None, 0, 1]), ("mean", [None, 0, 1])]:
        for dim, keep in itertools.product(dims, [False, True]):
            expr = f"{fn}()" if dim is None else f"{fn}(dim={dim}, keepdim={keep})"
            cases += method(expr, SHAPES_2D, grad=True)
    for fn in ("max", "min"):
        for dim in (0, 1):
            cases += method(f"{fn}(dim={dim}).values", SHAPES_2D, grad=True)
    for unbiased in (True, False):
        cases += method(f"std(unbiased={unbiased})", SHAPES_1D + SHAPES_2D)

    # 모양 조작
    cases += method("reshape(-1)", SHAPES_2D + SHAPES_3D, grad=True)
    cases += method("transpose(0, 1)", SHAPES_2D, grad=True)
    cases += method("unsqueeze(0)", SHAPES_1D + SHAPES_2D)
    cases += method("flatten(1)", SHAPES_3D)
    cases += method("permute(2, 0, 1)", SHAPES_3D)
    cases += method("t.T if False else t[0]", SHAPES_2D, label="idx[0]", grad=True)
    cases += method("[[0, 0, 1]]", SHAPES_2D, label="fancy-idx", grad=True)

    # softmax
    for dim in (-1, 0):
        cases.append(Case(
            f"softmax(dim={dim})(2,3)",
            lambda d=dim: real.nn.functional.softmax(real.tensor(make((2, 3))), dim=d),
            lambda d=dim: nano.softmax(nano.tensor(make((2, 3))), dim=d),
            grad=True,
        ))

    # dtype 승격 — numpy 규칙과 torch 규칙이 다르다. 여기가 자주 갈린다.
    for expr, label in [("tensor([1, 2, 3])", "int 리터럴"),
                        ("tensor([1.0, 2.0])", "float 리터럴"),
                        ("tensor([1, 2]) + 1.0", "int + float"),
                        ("tensor([1.0]) * 2", "float * int"),
                        ("tensor([True, False])", "bool 리터럴")]:
        cases.append(Case(
            f"dtype: {label}",
            lambda e=expr: str(eval(f"real.{e}").dtype),            # noqa: S307
            lambda e=expr: str(eval(f"nano.{e}").dtype),            # noqa: S307
            tier="T1",
        ))

    return cases


# ---------------------------------------------------------------- T2 오류

# 학습자가 실제로 만나는 실패들. 같은 조건에서 **같은 종류의 예외**가 나야 하고,
# 우리 메시지에는 torch 의 정규 영문 문구가 들어 있어야 한다 — 그래야 검색이 통한다.
ERROR_CASES = [
    ("행렬곱 모양 불일치", lambda t: t.randn(3, 4) @ t.randn(3, 2),
     "shapes cannot be multiplied"),
    ("브로드캐스트 불가", lambda t: t.randn(3, 4) + t.randn(3, 2),
     "must match the size of tensor"),
    ("reshape 원소수 불일치", lambda t: t.randn(2, 3).reshape(4, 2),
     "is invalid for input of size"),
    ("스칼라 아닌 backward", lambda t: t.randn(3, requires_grad=True).backward(),
     "grad can be implicitly created only for scalar outputs"),
    ("requires_grad 없이 backward", lambda t: t.randn(3).sum().backward(),
     "does not require grad"),
    ("정수 텐서에 requires_grad", lambda t: t.tensor([1, 2, 3], requires_grad=True), None),
    ("여러 원소에 item()", lambda t: t.randn(3).item(),
     "cannot be converted to Scalar"),
    ("Linear 입력 차원 불일치", lambda t: t.nn.Linear(4, 2)(t.randn(3, 5)),
     "shapes cannot be multiplied"),
    ("Conv2d 채널 불일치",
     lambda t: t.nn.functional.conv2d(t.randn(1, 3, 8, 8), t.randn(4, 1, 3, 3))
     if t is real else t.conv2d(t.randn(1, 3, 8, 8), t.randn(4, 1, 3, 3)), None),
    ("인덱스 범위 초과", lambda t: t.randn(3)[5], "out of bounds"),
    ("leaf 제자리 수정", lambda t: _inplace_leaf(t), None),
    ("backward 두 번", lambda t: _backward_twice(t),
     "backward through the graph a second time"),
]


def _inplace_leaf(t):
    x = t.randn(3, requires_grad=True)
    x += 1
    return x


def _backward_twice(t):
    x = t.randn(3, requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    y.backward()


def _raised(lib, fn):
    try:
        fn(lib)
        return None, ""
    except Exception as exc:                                        # noqa: BLE001
        return type(exc).__name__, str(exc)


def report_errors():
    same_kind = 0
    searchable = 0
    needs_phrase = 0
    problems = []

    for name, fn, phrase in ERROR_CASES:
        rk, _ = _raised(real, fn)
        nk, nm = _raised(nano, fn)

        if rk is None:
            problems.append(f"{name}: 진짜 torch 가 예외를 안 낸다 — 케이스가 틀렸다")
            continue
        if nk is None:
            problems.append(f"{name}: nanotorch 가 조용히 지나간다 (torch 는 {rk})")
            continue
        if nk != rk:
            problems.append(f"{name}: {rk} 여야 하는데 {nk}")
            continue
        same_kind += 1

        if phrase:
            needs_phrase += 1
            if phrase in nm:
                searchable += 1
            else:
                problems.append(f"{name}: 메시지에 검색용 문구가 없다 — \"{phrase}\"")

    total = len(ERROR_CASES)
    print(f"\n적합성 T2(오류) — 케이스 {total}개")
    print(f"  예외 종류 일치 {same_kind}/{total}"
          f" · 검색 가능한 메시지 {searchable}/{needs_phrase}")
    if problems:
        print("\n갈린 곳:")
        for why in problems:
            print(f"  ✗ {why}")
    return len(problems)


# ---------------------------------------------------------------- dtype 승격 전수

# torch 의 승격은 **범주**(bool < 정수 < 실수)로 가르고 그 범주 안에서만 올린다.
# numpy 는 다르게 올린다(float32 + int64 → float64). 한 칸이라도 물려받으면
# 학습자는 틀린 규칙을 배우므로 조합을 전부 훑는다.
DTYPES = ["float32", "float64", "int64", "bool"]
BIN_OPS = ["+", "-", "*", "/"]
PY_SCALARS = [("파이썬 int", 2), ("파이썬 float", 2.0), ("파이썬 bool", True)]


def _dt(lib, name):
    if name != "bool":
        return getattr(lib, name)
    return real.bool if lib is real else lib.bool_


def _mk(lib, name):
    return lib.tensor([1, 0] if name == "bool" else [1, 2], dtype=_dt(lib, name))


def _dtype_of(fn):
    try:
        return str(fn().dtype)
    except Exception as exc:                                        # noqa: BLE001
        return f"<{type(exc).__name__}>"


def report_dtypes():
    bad, total = [], 0
    for a in DTYPES:
        for b in DTYPES:
            for op in BIN_OPS:
                total += 1
                r = _dtype_of(lambda: eval(f"x {op} y", {},                      # noqa: S307
                                           {"x": _mk(real, a), "y": _mk(real, b)}))
                n = _dtype_of(lambda: eval(f"x {op} y", {},                      # noqa: S307
                                           {"x": _mk(nano, a), "y": _mk(nano, b)}))
                if r != n:
                    bad.append((f"{a} {op} {b}", r, n))
    for a in DTYPES:
        for label, value in PY_SCALARS:
            for op in BIN_OPS:
                total += 1
                r = _dtype_of(lambda: eval(f"x {op} s", {},                      # noqa: S307
                                           {"x": _mk(real, a), "s": value}))
                n = _dtype_of(lambda: eval(f"x {op} s", {},                      # noqa: S307
                                           {"x": _mk(nano, a), "s": value}))
                if r != n:
                    bad.append((f"{a} {op} {label}", r, n))

    print(f"\n적합성 dtype 승격 — 조합 {total}건")
    print(f"  일치 {total - len(bad)}/{total}")
    if bad:
        print("\n갈린 곳:")
        for name, r, n in bad[:20]:
            print(f"  ✗ {name:30} torch {r:<18} nano {n}")
        if len(bad) > 20:
            print(f"  … 외 {len(bad) - 20}건")
    return len(bad)


# ---------------------------------------------------------------- 저장소 공유

# torch 의 view·transpose·슬라이스는 **저장소를 공유한다.** 사본을 주면 편하지만,
# 실무에서 사고가 나는 바로 그 지점을 안 가르치게 된다.
VIEW_CASES = {
    "view 수정 → 원본": lambda t: _mutate(t.zeros(4), lambda a: a.view(2, 2), (0, 0), 9),
    "transpose 수정 → 원본": lambda t: _mutate(t.zeros(2, 2), lambda a: a.transpose(0, 1), (0, 1), 7),
    "인덱싱 수정 → 원본": lambda t: _mutate(t.zeros(2, 2), lambda a: a[0], 0, 5),
    "슬라이스 수정 → 원본": lambda t: _mutate(t.zeros(5), lambda a: a[1:3], 0, 9),
    "flatten 수정 → 원본": lambda t: _mutate(t.zeros(2, 2), lambda a: a.flatten(0), 0, 9),
    "squeeze 수정 → 원본": lambda t: _mutate(t.zeros(1, 3), lambda a: a.squeeze(), 0, 9),
    "unsqueeze 수정 → 원본": lambda t: _mutate(t.zeros(3), lambda a: a.unsqueeze(0), (0, 0), 9),
    "permute 수정 → 원본": lambda t: _mutate(t.zeros(2, 3), lambda a: a.permute(1, 0), (0, 1), 9),
    "clone 은 독립": lambda t: _mutate(t.zeros(2), lambda a: a.clone(), 0, 3),
    "detach 는 공유": lambda t: _mutate(t.zeros(2), lambda a: a.detach(), 0, 4),
    "팬시 인덱싱은 사본": lambda t: _mutate(t.zeros(3), lambda a: a[[0, 1]], 0, 9),
    "비연속 view 는 거부": lambda t: t.zeros(3, 4).transpose(0, 1).view(12).tolist(),
    "비연속 reshape 은 허용": lambda t: tuple(t.zeros(3, 4).transpose(0, 1).reshape(12).shape),
}


def _mutate(base, make_view, key, value):
    """뷰를 만들어 한 칸 고치고, **원본**이 어떻게 됐는지 돌려준다."""
    view = make_view(base)
    view[key] = value
    return base.tolist()


def report_views():
    same, problems = 0, []
    for name, fn in VIEW_CASES.items():
        r, n = _outcome(real, fn), _outcome(nano, fn)
        if r == n:
            same += 1
        else:
            problems.append(f"{name}: torch {r} · nano {n}")
    print(f"\n적합성 저장소 공유 — 케이스 {len(VIEW_CASES)}개")
    print(f"  일치 {same}/{len(VIEW_CASES)}")
    if problems:
        print("\n갈린 곳:")
        for why in problems:
            print(f"  ✗ {why}")
    return len(problems)


def _outcome(lib, fn):
    try:
        return str(fn(lib))
    except Exception as exc:                                        # noqa: BLE001
        return f"<{type(exc).__name__}>"


# ---------------------------------------------------------------- T3 표현

# 학습자가 가장 많이 하는 일이 print(tensor) 다. 교재의 예시와 화면이 다르면
# 그때마다 "내가 뭘 잘못했나" 를 의심하게 된다.
REPR_CASES = [
    ("스칼라", "tensor(3.14)"),
    ("정수값 float", "tensor([1.0, 2.0, 3.0])"),
    ("소수", "tensor([0.1, 0.25])"),
    ("음수 섞임", "tensor([-1.5, 2.0, -0.25])"),
    ("2차원", "tensor([[1.0, 2.0], [3.0, 4.0]])"),
    ("3차원", "zeros(2, 1, 3)"),
    ("정수", "tensor([1, 2, 3])"),
    ("불리언", "tensor([True, False])"),
    ("빈 텐서", "tensor([])"),
    ("큰 값·작은 값", "tensor([1e6, 2e-6])"),
    ("긴 1차원 줄바꿈", "arange(30).float()"),
    ("requires_grad", "tensor([1.0, 2.0], requires_grad=True)"),
    ("비잎 노드 grad_fn", "tensor([1.0], requires_grad=True) * 2"),
    ("합계 grad_fn", "tensor([1.0, 2.0], requires_grad=True).sum()"),
    ("relu grad_fn", "relu(tensor([-1.0, 2.0], requires_grad=True))"),
]


def report_repr():
    same, problems = 0, []
    for name, expr in REPR_CASES:
        try:
            r = repr(eval(expr, {"__builtins__": {}}, _ns(real)))     # noqa: S307
        except Exception as exc:                                      # noqa: BLE001
            problems.append(f"{name}: 진짜 torch 에서 실패 — {exc}")
            continue
        try:
            n = repr(eval(expr, {"__builtins__": {}}, _ns(nano)))     # noqa: S307
        except Exception as exc:                                      # noqa: BLE001
            problems.append(f"{name}: {type(exc).__name__} — {exc}")
            continue
        if r == n:
            same += 1
        else:
            problems.append(f"{name}:\n      torch {r!r}\n      nano  {n!r}")

    print(f"\n적합성 T3(표현) — 케이스 {len(REPR_CASES)}개")
    print(f"  일치 {same}/{len(REPR_CASES)}")
    if problems:
        print("\n갈린 곳:")
        for why in problems:
            print(f"  ✗ {why}")
    return len(problems)


def _ns(lib):
    """표현식에서 tensor(...) 처럼 쓰도록 이름을 펼친다."""
    return {n: getattr(lib, n) for n in
            ("tensor", "zeros", "ones", "arange", "randn", "relu", "sigmoid")
            if hasattr(lib, n)}


# ---------------------------------------------------------------- 판정

def compare(case):
    """(통과 여부, 사유). 예외는 사유에 담아 점수에 반영한다."""
    try:
        r = case.run_real()
    except Exception as exc:                                        # noqa: BLE001
        return None, f"진짜 torch 가 실패: {type(exc).__name__}"
    try:
        n = case.run_nano()
    except nano.NanoTorchError as exc:
        return False, f"미지원: {str(exc).splitlines()[0]}"
    except Exception as exc:                                        # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"

    if isinstance(r, str) or isinstance(n, str):
        return (r == n), f"{r!r} vs {n!r}"

    rn = r.detach().numpy()
    nn_ = n.data
    if rn.shape != nn_.shape:
        return False, f"모양 {rn.shape} vs {nn_.shape}"
    if not np.allclose(rn, nn_, atol=ATOL, rtol=ATOL):
        return False, f"값 최대차 {np.abs(rn - nn_).max():.3e}"

    if case.grad:
        try:
            ok, why = compare_grad(case)
            if not ok:
                return False, f"기울기: {why}"
        except nano.NanoTorchError as exc:
            return False, f"기울기 미지원: {str(exc).splitlines()[0]}"
        except Exception as exc:                                    # noqa: BLE001
            return False, f"기울기 {type(exc).__name__}: {exc}"
    return True, ""


def compare_grad(case):
    """같은 계산의 기울기를 본다. 스칼라로 접어 backward 를 부른다."""
    seen = {}

    def tap(lib, tensor_fn):
        original = tensor_fn

        def wrapped(data, *a, **k):
            t = original(np.asarray(data), *a, **k)
            if getattr(t, "dtype", None) is not None and "float" in str(t.dtype):
                t.requires_grad = True
                seen.setdefault(lib, []).append(t)
            return t
        return wrapped

    real_tensor, nano_tensor = real.tensor, nano.tensor
    real.tensor = tap("real", real_tensor)
    nano.tensor = tap("nano", nano_tensor)
    try:
        r_out, n_out = case.run_real(), case.run_nano()
        r_out.sum().backward()
        n_out.sum().backward()
    finally:
        real.tensor, nano.tensor = real_tensor, nano_tensor

    for rt, nt in zip(seen.get("real", []), seen.get("nano", [])):
        if rt.grad is None and nt.grad is None:
            continue
        if (rt.grad is None) != (nt.grad is None):
            return False, "한쪽만 기울기가 있다"
        if not np.allclose(rt.grad.numpy(), nt.grad.data, atol=ATOL, rtol=ATOL):
            return False, f"최대차 {np.abs(rt.grad.numpy() - nt.grad.data).max():.3e}"
    return True, ""


def report():
    cases = build_cases()
    passed, failed, skipped = [], [], []
    for case in cases:
        ok, why = compare(case)
        if ok is None:
            skipped.append((case, why))
        elif ok:
            passed.append(case)
        else:
            failed.append((case, why))

    total = len(passed) + len(failed)
    print(f"적합성 T1(값·기울기) — 케이스 {len(cases)}개")
    print(f"  통과 {len(passed)} · 실패 {len(failed)} · 건너뜀 {len(skipped)}")
    if total:
        print(f"  점수 {100 * len(passed) // total}%")
    if failed:
        print("\n갈린 곳:")
        for case, why in failed[:30]:
            print(f"  ✗ {case.name:34} {why}")
        if len(failed) > 30:
            print(f"  … 외 {len(failed) - 30}건")
    return len(failed)


if __name__ == "__main__":
    failed = report() + report_errors() + report_repr() + report_dtypes() + report_views()
    raise SystemExit(1 if failed else 0)
