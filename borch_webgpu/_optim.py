"""`torch.optim` 자리. borch.ts 의 옵티마이저를 그대로 부른다.

파라미터를 넘길 때 **손잡이로 바꿔서** 넘긴다 — 우리 `Tensor` 는 파이썬 껍데기라
JS 쪽이 그것을 모른다.
"""

import numpy as _np

import js as _js

from ._base import handle, tensor, wrap

_ts = _js.borch


class _Opt:
    __slots__ = ("_o",)

    def __init__(self, o):
        self._o = o

    def zero_grad(self):
        self._o.zeroGrad()

    def step(self):
        self._o.step()

    def state_dict(self):
        """**이어서 학습하기가 여기 걸려 있다.**

        전에는 이 자리가 통째로 없었다(`__slots__` 라 `AttributeError` 였다). 모델
        가중치는 `Module.state_dict` 로 저장되는데 옵티마이저는 저장할 길이 없었고,
        그래서 GPU 경로에서 이어 붙인 학습은 **모멘텀과 2 차 모먼트를 잃은 채로**
        다시 시작했다. 예외는 안 나고 손실 곡선만 한 번 튄다.

        **모양은 torch 와 다르다.** torch 는 `{"state": …, "param_groups": …}` 이고
        borch.ts 는 은행 구조(`{tensors, numbers}`)다. 파라미터마다 슬롯을 나누어
        torch 이름(`exp_avg` 따위)을 붙이려면 저쪽 설계를 바꿔야 하는데, 그것으로
        얻는 것은 **남의 torch 체크포인트를 읽는 일** 뿐이고 그 길은 어차피
        커널까지 막혀 있다. 우리 것을 저장했다 우리 것으로 되돌리는 일은 이 모양
        으로 된다 — 골든이 값으로 그것을 붙잡는다(`opt::*/이어서 학습하기`).

        `state_dict` 가 **사본이 아니라 지금 슬롯**을 준다는 것도 torch 와 같다.
        저장해 두고 계속 밟으면 저장해 둔 것도 같이 움직인다.
        """
        got = self._o.stateDict()
        return {
            "tensors": {str(k): wrap(getattr(got.tensors, k))
                        for k in _js.Object.keys(got.tensors)},
            "numbers": {str(k): getattr(got.numbers, k)
                        for k in _js.Object.keys(got.numbers)},
        }

    def load_state_dict(self, state):
        """`state_dict()` 가 준 것을 되돌린다. **같은 인자로 다시 세운 뒤** 부른다."""
        obj = _js.Object.new()
        tensors = _js.Object.new()
        for key, value in state["tensors"].items():
            setattr(tensors, key, handle(value))
        numbers = _js.Object.new()
        for key, value in state["numbers"].items():
            setattr(numbers, key, value)
        obj.tensors = tensors
        obj.numbers = numbers
        self._o.loadStateDict(obj)
        return self

    @property
    def param_groups(self):
        """**`opt.param_groups[0]["lr"]` 로 읽는다.** torch 코드가 그렇게 쓴다.

        `to_py()` 만으로는 안 된다 — JS 객체가 파이썬 딕셔너리가 아니라 프록시로
        남아서 `["lr"]` 이 안 먹는다. 그리고 학습률은 **읽을 때마다 지금 값**이어야
        하므로 한 번 베껴 두면 스케줄러가 바꾼 것을 못 본다.
        """
        return [_Group(g) for g in self._o.paramGroups]


class _Group:
    """파라미터 묶음 하나. 딕셔너리처럼 읽히되 값은 그때그때 JS 쪽에서 가져온다."""

    __slots__ = ("_g",)

    def __init__(self, g):
        self._g = g

    def __getitem__(self, key):
        return getattr(self._g, key)

    def __setitem__(self, key, value):
        setattr(self._g, key, value)

    def get(self, key, default=None):
        got = getattr(self._g, key, None)
        return default if got is None else got


def _params(ps):
    # 파라미터가 JS 배열로 올 수도 있다 — `model.parameters()` 를 그대로 넘기는 자리다.
    if hasattr(ps, "to_py"):
        ps = ps.to_py()
    return _js.Array.new(*[handle(p) for p in ps])


def SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0):
    return _Opt(_ts.optim.SGD.new(_params(params), lr, momentum, weight_decay))


def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.Adam.new(_params(params), lr, betas[0], betas[1],
                                   eps, weight_decay))


def RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.RMSprop.new(_params(params), lr, alpha, eps,
                                      weight_decay))


def Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0, eps=1e-10):
    return _Opt(_ts.optim.Adagrad.new(_params(params), lr, lr_decay, eps))


def Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
    return _Opt(_ts.optim.Adadelta.new(_params(params), lr, rho, eps))


def Adamax(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.Adamax.new(_params(params), lr, betas[0], betas[1], eps))


def NAdam(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
          momentum_decay=4e-3):
    return _Opt(_ts.optim.NAdam.new(_params(params), lr, betas[0], betas[1], eps,
                                    momentum_decay))


def RAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.RAdam.new(_params(params), lr, betas[0], betas[1], eps))


def ASGD(params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0):
    return _Opt(_ts.optim.ASGD.new(_params(params), lr, lambd, alpha, t0,
                                   weight_decay))


def Rprop(params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
    return _Opt(_ts.optim.Rprop.new(_params(params), lr, etas[0], etas[1],
                                    step_sizes[0], step_sizes[1]))


def Adafactor(params, lr=1e-2, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0,
              weight_decay=0.0):
    return _Opt(_ts.optim.Adafactor.new(_params(params), lr, beta2_decay,
                                        eps[0], eps[1], d, weight_decay))


class LBFGS:
    """준뉴턴법. **`step` 이 닫힘(closure)을 받는다** — 한 걸음 안에서 손실을 여러 번
    다시 재기 때문이다.

    ## 왜 여기에 있고 borch.ts 에 없는가

    이 알고리즘은 **제어 흐름이 값에 달려 있다** — `ys > 1e-10` 이냐, 기울기가 문턱
    아래냐, 손실이 더 안 줄었냐로 갈린다. borch.ts 는 동기 읽기가 없어서 GPU 위의
    수를 그 자리에서 못 본다. 여기는 `run_sync` 가 있으므로 이쪽이 그 자리다.

    파라미터는 GPU 에 그대로 두고 **평평한 기울기 벡터만** 오간다. LBFGS 는 원래
    작은 문제에 쓰는 것이라 그 값이 남는다.

    **코어에 같은 알고리즘이 한 벌 더 있다.** 두 꾸러미가 서로를 안 들여오기로 한
    구조라 그렇고, 골든이 같은 세 케이스를 양쪽에 물어 갈리면 잡는다.

    직선 탐색은 아직 없다 — `line_search_fn` 을 주면 시끄럽게 거절한다.
    """

    def __init__(self, params, lr=1.0, max_iter=20, max_eval=None,
                 tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100,
                 line_search_fn=None):
        got = params.to_py() if hasattr(params, "to_py") else list(params)
        self._ps = [wrap(handle(p)) for p in got]
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        self.param_groups = [dict(
            lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            history_size=history_size, line_search_fn=line_search_fn)]
        self._state = {}

    def zero_grad(self, set_to_none=True):
        for p in self._ps:
            p.grad = None

    def _flat_grad(self):
        parts = []
        for p in self._ps:
            g = p.grad
            parts.append(_np.zeros(int(handle(p).size), dtype=_np.float32)
                         if g is None else
                         _np.asarray(g.numpy(), dtype=_np.float32).reshape(-1))
        return _np.concatenate(parts)

    def _add_step(self, size, direction):
        from ._ops import no_grad as _no_grad
        at = 0
        # **`no_grad` 안에서 고친다.** 파라미터는 기울기가 켜진 잎이고, 그것을 제자리로
        # 고치는 것은 밖에서는 거절된다 — 옵티마이저만 해도 되는 일이라 그 자리를 연다.
        with _no_grad():
            for p in self._ps:
                h = handle(p)
                n = int(h.size)
                shape = [int(v) for v in h.shape]
                moved = (_np.asarray(p.numpy(), dtype=_np.float32).reshape(-1)
                         + size * direction[at:at + n])
                h.copyFrom(handle(tensor(moved.reshape(shape))))
                at += n

    def step(self, closure):
        group = self.param_groups[0]
        if group["line_search_fn"] is not None:
            raise RuntimeError(
                f"LBFGS(line_search_fn={group['line_search_fn']!r}) is not here yet.")
        lr, max_iter = group["lr"], group["max_iter"]
        max_eval = group["max_eval"]
        tol_grad, tol_change = group["tolerance_grad"], group["tolerance_change"]
        history = group["history_size"]
        st = self._state
        st.setdefault("n_iter", 0)

        orig = closure()
        loss = float(orig.item() if hasattr(orig, "item") else orig)
        evals = 1
        flat = self._flat_grad()
        if _np.abs(flat).max() <= tol_grad:
            return orig

        d, t = st.get("d"), st.get("t")
        old_dirs = st.get("old_dirs", [])
        old_stps = st.get("old_stps", [])
        ro = st.get("ro", [])
        h_diag = st.get("h_diag", 1.0)
        prev_flat, prev_loss = st.get("prev_flat"), st.get("prev_loss")

        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            st["n_iter"] += 1
            if st["n_iter"] == 1:
                d = -flat
                old_dirs, old_stps, ro, h_diag = [], [], [], 1.0
            else:
                y = flat - prev_flat
                s = d * t
                ys = float(y @ s)
                if ys > 1e-10:
                    if len(old_dirs) == history:
                        old_dirs.pop(0), old_stps.pop(0), ro.pop(0)
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)
                    h_diag = ys / float(y @ y)
                al = [0.0] * len(old_dirs)
                q = -flat
                for i in range(len(old_dirs) - 1, -1, -1):
                    al[i] = float(old_stps[i] @ q) * ro[i]
                    q = q - al[i] * old_dirs[i]
                r = q * h_diag
                for i in range(len(old_dirs)):
                    be = float(old_dirs[i] @ r) * ro[i]
                    r = r + old_stps[i] * (al[i] - be)
                d = r

            prev_flat, prev_loss = flat.copy(), loss
            t = min(1.0, 1.0 / _np.abs(flat).sum()) * lr if st["n_iter"] == 1 else lr
            if float(flat @ d) > -tol_change:
                break

            self._add_step(t, d)
            if n_iter != max_iter:
                got = closure()
                loss = float(got.item() if hasattr(got, "item") else got)
                flat = self._flat_grad()
                evals += 1
                if _np.abs(flat).max() <= tol_grad:
                    break
            if n_iter == max_iter or evals >= max_eval:
                break
            if _np.abs(d * t).max() <= tol_change:
                break
            if abs(loss - prev_loss) < tol_change:
                break

        st.update(d=d, t=t, old_dirs=old_dirs, old_stps=old_stps, ro=ro,
                  h_diag=h_diag, prev_flat=prev_flat, prev_loss=prev_loss)
        return orig


class _Sched:
    # `_keep` 은 JS 로 넘긴 파이썬 함수의 프록시를 붙잡아 둔다. 안 잡으면 호출이
    # 끝나는 순간 파괴되고, 스케줄러가 **나중에** 부를 때 "borrowed proxy was
    # automatically destroyed" 로 터진다 — 자매가 `tf.grad` 에서 겪은 것과 같다.
    __slots__ = ("_s", "_keep")

    def __init__(self, s, keep=None):
        self._s = s
        self._keep = keep

    def step(self, *args):
        self._s.step(*args)

    def get_last_lr(self):
        return list(self._s.getLastLr())

    def state_dict(self):
        """**옵티마이저만 되돌리면 학습률이 처음 값으로 돌아간다.**

        이 자리도 통째로 없었다. 반쯤 식혀 놓은 학습이 이어 붙이는 순간 다시
        뜨거워지고, 오류는 안 난다 — 손실 곡선만 한 번 올라갔다 내려온다.

        저쪽은 수 사전이라 그대로 넘긴다. 모양은 torch(`last_epoch`·`_step_count`)
        와 다르지만, 우리 것을 저장했다 우리 것으로 되돌리는 일은 이것으로 된다.
        """
        got = self._s.stateDict()
        return {str(k): getattr(got, k) for k in _js.Object.keys(got)}

    def load_state_dict(self, state):
        obj = _js.Object.new()
        for key, value in state.items():
            setattr(obj, key, value)
        self._s.loadStateDict(obj)
        return self


# 스케줄러도 이름 붙은 인자로 부른다 — `StepLR(o, step_size=2, gamma=0.5)`.
# JS 쪽은 자리로만 받으므로 여기서 편다.
_SCHED_ARGS = {
    "StepLR": ("step_size", "gamma"),
    "MultiStepLR": ("milestones", "gamma"),
    "ExponentialLR": ("gamma",),
    "CosineAnnealingLR": ("T_max", "eta_min"),
    "LambdaLR": ("lr_lambda",),
    # borch.ts 에는 `mode` 가 없다 — `rel` 하나뿐이라 자리도 없다.
    "ReduceLROnPlateau": ("factor", "patience", "threshold"),
    "ConstantLR": ("factor", "total_iters"),
    "LinearLR": ("start_factor", "end_factor", "total_iters"),
    "PolynomialLR": ("total_iters", "power"),
    "MultiplicativeLR": ("lr_lambda",),
    "CosineAnnealingWarmRestarts": ("T_0", "T_mult", "eta_min"),
    "OneCycleLR": ("max_lr", "total_steps", "pct_start", "div_factor",
                   "final_div_factor"),
    "CyclicLR": ("base_lr", "max_lr", "step_size_up", "step_size_down", "mode",
                 "gamma"),
}


def _sched(js_name):
    def make(opt, *args, **kw):
        from ._ops import _arg
        out = list(args)
        for i, key in enumerate(_SCHED_ARGS.get(js_name, ())):
            if key in kw:
                while len(out) <= i:
                    out.append(None)
                out[i] = kw[key]
        while out and out[-1] is None:
            out.pop()
        # 파이썬 함수는 프록시로 만들어 붙잡아 둔다 — `LambdaLR` 이 그것을 나중에 부른다.
        keep = None
        ready = []
        for a in out:
            if callable(a) and not isinstance(a, (int, float, str)):
                from pyodide.ffi import create_proxy
                keep = create_proxy(a)
                ready.append(keep)
            else:
                ready.append(_arg(a))
        s = getattr(_ts.optim, js_name).new(opt._o, *ready)
        # **세운 직후 0 번째 에폭을 적용한다.** torch 는 생성자가 하는 일인데
        # TypeScript 쪽은 하위 클래스 필드가 `super()` 뒤에 채워져서 거기서 못 한다.
        # `ReduceLROnPlateau` 만 이 계보 밖이라 그 자리가 없다 — 값을 받아 판단하는
        # 것이라 에폭이라는 개념 자체가 없다.
        if hasattr(s, "start"):
            s.start()
        return _Sched(s, keep)
    return make


# **`torch.optim.lr_scheduler` 는 이름 공간이다.** 골든이 그 경로로 부른다.
class _LRScheduler:
    StepLR = staticmethod(_sched("StepLR"))
    MultiStepLR = staticmethod(_sched("MultiStepLR"))
    ExponentialLR = staticmethod(_sched("ExponentialLR"))
    CosineAnnealingLR = staticmethod(_sched("CosineAnnealingLR"))
    LambdaLR = staticmethod(_sched("LambdaLR"))
    ReduceLROnPlateau = staticmethod(_sched("ReduceLROnPlateau"))
    ConstantLR = staticmethod(_sched("ConstantLR"))
    LinearLR = staticmethod(_sched("LinearLR"))
    PolynomialLR = staticmethod(_sched("PolynomialLR"))
    MultiplicativeLR = staticmethod(_sched("MultiplicativeLR"))
    CosineAnnealingWarmRestarts = staticmethod(_sched("CosineAnnealingWarmRestarts"))
    OneCycleLR = staticmethod(_sched("OneCycleLR"))
    CyclicLR = staticmethod(_sched("CyclicLR"))
    LRScheduler = _Sched

    @staticmethod
    def SequentialLR(optimizer, schedulers, milestones, last_epoch=-1):
        """스케줄러 목록과 이정표를 JS 배열로 넘긴다.

        **`Array.new` 를 쓰면 안 된다.** 인자가 수 하나면 `Array(3)` 이 되어 `[3]` 이
        아니라 **길이 3 짜리 빈 배열**이 나온다. 이정표가 보통 하나라 정확히 그 자리에
        걸렸고, 증상은 이정표를 지나도 학습률이 안 바뀌는 것이었다 — 이 저장소가 같은
        함정에 두 번째로 빠진 자리다.
        """
        return _Sched(_ts.optim.SequentialLR.new(
            optimizer._o, _js.Array.of(*[s._s for s in schedulers]),
            _js.Array.of(*[int(m) for m in milestones])))

    @staticmethod
    def ChainedScheduler(schedulers, optimizer=None):
        return _Sched(_ts.optim.ChainedScheduler.new(
            _js.Array.of(*[s._s for s in schedulers])))


lr_scheduler = _LRScheduler()

StepLR = _sched("StepLR")
MultiStepLR = _sched("MultiStepLR")
ExponentialLR = _sched("ExponentialLR")
CosineAnnealingLR = _sched("CosineAnnealingLR")
LambdaLR = _sched("LambdaLR")
