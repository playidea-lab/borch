"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np

from ._tensor import (
    Tensor,
)
from ._ops import (
    _Namespace, _unsupported,
)
from ._base import (
    _math, _np,
)

# ================================================================ optim

class Optimizer:
    """torch 의 옵티마이저는 `param_groups` 라는 목록을 들고 있다.

    학습률을 읽고 쓰는 표준 경로가 `opt.param_groups[0]["lr"]` 이고, 스케줄러도 그것을
    고친다. `opt.lr` 로 두면 짧지만 **남의 코드가 안 돌고, 스케줄러를 직접 못 쓴다.**
    """

    def __init__(self, params, defaults):
        params = list(params)
        if params and isinstance(params[0], dict):
            self.param_groups = [dict(defaults, **g, ) for g in params]
            for g in self.param_groups:
                g["params"] = list(g["params"])
        else:
            self.param_groups = [dict(defaults, params=params)]
        self.state = {}
        self.defaults = defaults

    @property
    def params(self):
        return [p for g in self.param_groups for p in g["params"]]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def _state(self, p):
        return self.state.setdefault(id(p), {})

    def state_dict(self):
        """torch 와 같은 모양 — 6장이 가르치는 "이어서 학습하기"가 이것에 걸려 있다.

        Adam 은 파라미터마다 보폭을 기억한다. 그 기억을 버리고 이어 학습하면
        손실이 한 번 튀었다가 다시 내려간다 — 오류는 안 나고 곡선만 이상해진다.
        """
        order = {id(p): i for i, p in enumerate(self.params)}
        return {
            "state": {order[k]: {n: (Tensor(v.copy()) if isinstance(v, _np.ndarray) else v)
                                 for n, v in st.items()}
                      for k, st in self.state.items() if k in order},
            "param_groups": [{k: v for k, v in g.items() if k != "params"}
                             | {"params": [order[id(p)] for p in g["params"]]}
                             for g in self.param_groups],
        }

    def load_state_dict(self, state):
        params = self.params
        self.state = {}
        for index, st in state.get("state", {}).items():
            self.state[id(params[int(index)])] = {
                n: (v.data.copy() if isinstance(v, Tensor) else v) for n, v in st.items()}
        for group, saved in zip(self.param_groups, state.get("param_groups", [])):
            for key, value in saved.items():
                if key != "params":
                    group[key] = value
        return self

    def step(self):
        raise NotImplementedError

    def __repr__(self):
        lr = self.param_groups[0].get("lr")
        return f"{type(self).__name__} (lr: {lr})"


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                if group["momentum"]:
                    st = self._state(p)
                    buf = st.get("momentum_buffer")
                    buf = g if buf is None else group["momentum"] * buf + g
                    st["momentum_buffer"] = buf
                    g = buf
                p._array = p.data - group["lr"] * g


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    decoupled = False          # AdamW 는 True — 감쇠를 기울기가 아니라 가중치에 직접 건다

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("exp_avg", _np.zeros_like(p.data))
                st.setdefault("exp_avg_sq", _np.zeros_like(p.data))
                st["step"] += 1
                g = p.grad.data
                if group["weight_decay"] and not self.decoupled:
                    g = g + group["weight_decay"] * p.data
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * (g * g)
                mh = st["exp_avg"] / (1 - b1 ** st["step"])
                vh = st["exp_avg_sq"] / (1 - b2 ** st["step"])
                new = p.data - group["lr"] * mh / (_np.sqrt(vh) + group["eps"])
                if group["weight_decay"] and self.decoupled:
                    new = new - group["lr"] * group["weight_decay"] * p.data
                p._array = new


class AdamW(Adam):
    """Adam 과 같은데 가중치 감쇠를 **기울기가 아니라 가중치에 직접** 건다."""

    decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("square_avg", _np.zeros_like(p.data))
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["square_avg"] = (group["alpha"] * st["square_avg"]
                                    + (1 - group["alpha"]) * g * g)
                p._array = p.data - group["lr"] * g / (_np.sqrt(st["square_avg"]) + group["eps"])




class Adagrad(Optimizer):
    """기울기 제곱을 **계속 더한다** — 줄기만 하고 안 는다.

    `RMSprop` 은 같은 자리에 지수이동평균을 두어 옛것을 잊는데, 이쪽은 안 잊는다.
    그래서 오래 돌리면 보폭이 0 으로 수렴한다 — 그것이 이 옵티마이저의 성질이고
    결함이 아니다.
    """

    def __init__(self, params, lr=0.01, lr_decay=0.0, weight_decay=0.0, eps=1e-10):
        super().__init__(params, dict(lr=lr, lr_decay=lr_decay,
                                      weight_decay=weight_decay, eps=eps))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("sum", _np.zeros_like(p.data))
                st["step"] += 1
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["sum"] = st["sum"] + g * g
                lr = group["lr"] / (1 + (st["step"] - 1) * group["lr_decay"])
                p._array = p.data - lr * g / (_np.sqrt(st["sum"]) + group["eps"])


class Adadelta(Optimizer):
    """**학습률이 거의 안 쓰인다.** 보폭을 갱신량의 이력에서 스스로 만든다.

    그래서 기본 `lr` 이 1.0 이다 — 다른 옵티마이저와 같은 값을 주면 엉뚱하게 크다.
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, rho=rho, eps=eps,
                                      weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            rho, eps = group["rho"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("square_avg", _np.zeros_like(p.data))
                st.setdefault("acc_delta", _np.zeros_like(p.data))
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["square_avg"] = rho * st["square_avg"] + (1 - rho) * g * g
                delta = (_np.sqrt(st["acc_delta"] + eps)
                         / _np.sqrt(st["square_avg"] + eps)) * g
                st["acc_delta"] = rho * st["acc_delta"] + (1 - rho) * delta * delta
                p._array = p.data - group["lr"] * delta


class Adamax(Optimizer):
    """Adam 의 2차 모멘트를 **제곱평균 대신 최댓값**으로 둔 것. 무한 노름 판이다."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("exp_avg", _np.zeros_like(p.data))
                st.setdefault("exp_inf", _np.zeros_like(p.data))
                st["step"] += 1
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_inf"] = _np.maximum(b2 * st["exp_inf"],
                                            _np.abs(g) + group["eps"])
                bias = 1 - b1 ** st["step"]
                p._array = p.data - (group["lr"] / bias) * st["exp_avg"] / st["exp_inf"]


class NAdam(Optimizer):
    """Adam 에 네스테로프의 앞보기를 붙인 것.

    **모멘텀 계수가 스텝마다 바뀐다** — `momentum_decay` 로 서서히 커지는 수열이고,
    그 수열의 **누적곱**을 들고 다녀야 한다. 상수로 두면 초반 몇 스텝이 조용히 갈린다.
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, momentum_decay=4e-3):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay,
                                      momentum_decay=momentum_decay))

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            psi = group["momentum_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("mu_product", 1.0)
                st.setdefault("exp_avg", _np.zeros_like(p.data))
                st.setdefault("exp_avg_sq", _np.zeros_like(p.data))
                st["step"] += 1
                t = st["step"]
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                mu = b1 * (1 - 0.5 * 0.96 ** (t * psi))
                mu_next = b1 * (1 - 0.5 * 0.96 ** ((t + 1) * psi))
                st["mu_product"] = st["mu_product"] * mu
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * g * g
                denom = _np.sqrt(st["exp_avg_sq"] / (1 - b2 ** t)) + group["eps"]
                new = p.data - group["lr"] * (1 - mu) / (1 - st["mu_product"]) * g / denom
                new = new - group["lr"] * mu_next / (
                    1 - st["mu_product"] * mu_next) * st["exp_avg"] / denom
                p._array = new


class RAdam(Optimizer):
    """Adam 인데 초반에는 **적응 보폭을 안 쓴다.**

    2차 모멘트의 표본이 적을 때 분산이 커서 초반 몇 스텝이 튀는 것이 Adam 의 알려진
    성질이고, 이쪽은 그 구간을 SGD 처럼 지나간다. 경계(`rho > 5`)를 빠뜨리면 값이
    Adam 과 같아지므로 골든이 다섯 스텝을 밟는다.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            rho_inf = 2.0 / (1 - b2) - 1
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("exp_avg", _np.zeros_like(p.data))
                st.setdefault("exp_avg_sq", _np.zeros_like(p.data))
                st["step"] += 1
                t = st["step"]
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * g * g
                mh = st["exp_avg"] / (1 - b1 ** t)
                rho = rho_inf - 2 * t * b2 ** t / (1 - b2 ** t)
                if rho > 5.0:
                    rect = _math.sqrt(((rho - 4) * (rho - 2) * rho_inf)
                                      / ((rho_inf - 4) * (rho_inf - 2) * rho))
                    denom = _np.sqrt(st["exp_avg_sq"] / (1 - b2 ** t)) + group["eps"]
                    p._array = p.data - group["lr"] * mh * rect / denom
                else:
                    # 적응 보폭을 안 쓴다 — 여기가 SGD 처럼 도는 구간이다.
                    p._array = p.data - group["lr"] * mh


class ASGD(Optimizer):
    """평균 내는 SGD. **걸음마다 학습률이 줄고**, 어느 시점부터 파라미터를 평균낸다.

    `eta` 는 `lr / (1 + lambd·lr·step)^alpha` 로 스스로 줄고, `mu` 가 평균의 무게다.
    기본 `t0` 이 100만이라 보통 학습에서는 `mu` 가 1 이고 `ax` 가 그냥 파라미터의
    사본이다 — 평균이 실제로 도는 것은 `t0` 을 낮춰야 보인다.

    **감쇠가 곱셈이다.** `param *= (1 - lambd·eta)` 를 먼저 하고 그다음에 기울기를
    뺀다. 기울기에 더하는 꼴(`weight_decay`)과 다른 자리이고, 둘 다 있으면 둘 다 건다.
    """

    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0,
                                      weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            lr, lambd = group["lr"], group["lambd"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("eta", lr)
                st.setdefault("mu", 1.0)
                st.setdefault("ax", _np.zeros_like(p.data))
                st["step"] += 1
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                eta, mu = st["eta"], st["mu"]
                p._array = p.data * (1 - lambd * eta) - eta * g
                # `mu` 가 1 이면 평균이 아니라 **사본**이다 — 더하면 두 배가 된다.
                st["ax"] = (p.data.copy() if mu == 1
                            else st["ax"] + (p.data - st["ax"]) * mu)
                step = st["step"]
                st["eta"] = lr / ((1 + lambd * lr * step) ** group["alpha"])
                st["mu"] = 1.0 / max(1.0, step - group["t0"])


class Rprop(Optimizer):
    """기울기의 **부호만** 본다. 크기는 안 쓰고 걸음 폭을 칸마다 따로 키우고 줄인다.

    부호가 그대로면 폭을 `etas[1]` 배로 키우고, 뒤집히면 `etas[0]` 배로 줄인다.
    **뒤집힌 칸은 그 걸음을 아예 안 간다** — 기울기를 0 으로 만들어 두고, 그래서 다음
    걸음의 "이전 기울기" 도 0 이 된다. 그 두 줄이 없으면 값이 그럴듯하게 다르고,
    부호가 안 바뀌는 입력으로는 영원히 안 걸린다.
    """

    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        super().__init__(params, dict(lr=lr, etas=etas, step_sizes=step_sizes))

    def step(self):
        for group in self.param_groups:
            eta_minus, eta_plus = group["etas"]
            low, high = group["step_sizes"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("prev", _np.zeros_like(p.data))
                st.setdefault("step_size", _np.full_like(p.data, group["lr"]))
                st["step"] += 1
                g = _np.array(p.grad.data, copy=True)
                sign = _np.sign(g * st["prev"])
                factor = _np.where(sign > 0, eta_plus,
                                   _np.where(sign < 0, eta_minus, 1.0))
                st["step_size"] = _np.clip(st["step_size"] * factor, low, high)
                g[sign < 0] = 0.0
                p._array = p.data - _np.sign(g) * st["step_size"]
                st["prev"] = g


class Adafactor(Optimizer):
    """Adam 인데 2차 모멘트를 **행과 열로 쪼개 든다.**

    Adam 은 파라미터마다 분산을 하나씩 들어서 기억이 가중치만큼 든다. 여기서는
    행 평균과 열 평균만 들고 그 바깥곱으로 되살린다 — `(R, C)` 자리에 `R + C` 만
    쓴다. 큰 언어모델을 메모리에 얹으려고 나온 방법이다.

    **1 차원 파라미터는 안 쪼갠다** — 쪼갤 축이 하나뿐이라 그냥 분산을 든다. 상태
    열쇠부터 갈린다(`variance` 대 `row_var`·`col_var`). 1 차원으로만 물으면 이
    최적화의 요점이 통째로 안 돌아간다.
    """

    def __init__(self, params, lr=1e-2, beta2_decay=-0.8, eps=(None, 1e-3),
                 d=1.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, beta2_decay=beta2_decay, eps=eps,
                                      d=d, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            lr, d = group["lr"], group["d"]
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                one = eps1 if eps1 is not None else _np.finfo(p.data.dtype).eps
                st = self._state(p)
                st.setdefault("step", 0)
                st["step"] += 1
                step = float(st["step"])
                blend = step ** group["beta2_decay"]
                rho = min(lr, 1.0 / _math.sqrt(step))
                alpha = max(eps2, _np.linalg.norm(p.data.reshape(-1))
                            / _math.sqrt(p.data.size)) * rho
                if group["weight_decay"]:
                    p._array = p.data * (1 - lr * group["weight_decay"])

                if g.ndim > 1:
                    st.setdefault("row_var", _np.zeros(g.shape[:-1] + (1,),
                                                       dtype=g.dtype))
                    st.setdefault("col_var", _np.zeros(g.shape[:-2] + (1, g.shape[-1]),
                                                       dtype=g.dtype))
                    row_mean = (g * g).mean(axis=-1, keepdims=True)
                    col_mean = (g * g).mean(axis=-2, keepdims=True)
                    st["row_var"] += (row_mean - st["row_var"]) * blend
                    st["col_var"] += (col_mean - st["col_var"]) * blend
                    var = st["row_var"] @ st["col_var"]
                    var = var / _np.maximum(st["row_var"].mean(axis=-2, keepdims=True),
                                            one)
                else:
                    st.setdefault("variance", _np.zeros_like(g))
                    st["variance"] += (g * g - st["variance"]) * blend
                    var = st["variance"].copy()

                update = g / _np.sqrt(_np.maximum(var, one * one))
                denom = max(1.0, _np.linalg.norm(update.reshape(-1))
                            / (_math.sqrt(update.size) * d))
                p._array = p.data - (alpha / denom) * update


class LBFGS(Optimizer):
    """준뉴턴법. **`step` 이 닫힘(closure)을 받는다** — 한 번 부르는 동안 손실을
    여러 번 다시 재기 때문이다.

    다른 옵티마이저는 기울기 한 벌로 한 걸음을 가는데, 이쪽은 안에서 `max_iter` 번
    돌면서 매번 손실과 기울기를 다시 묻는다. 그래서 학습 루프의 모양이 다르고,
    닫힘을 안 주면 아무것도 못 한다.

    **직선 탐색은 아직 없다.** `line_search_fn="strong_wolfe"` 는 시끄럽게 거절한다 —
    조용히 고정 보폭으로 가면 수렴이 다르게 나오고, 그 차이는 값이 아니라 곡선에서만
    보인다.
    """

    def __init__(self, params, lr=1.0, max_iter=20, max_eval=None,
                 tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100,
                 line_search_fn=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        super().__init__(params, dict(
            lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            history_size=history_size, line_search_fn=line_search_fn))
        if len(self.param_groups) != 1:
            raise ValueError("LBFGS 는 파라미터 묶음 하나만 받습니다.")
        self._global = {}

    def _flat_grad(self):
        return _np.concatenate([
            (_np.zeros(p.data.size, dtype=p.data.dtype) if p.grad is None
             else p.grad.data.reshape(-1)) for p in self.params])

    def _add_step(self, size, direction):
        at = 0
        for p in self.params:
            n = p.data.size
            p._array = p.data + size * direction[at:at + n].reshape(p.data.shape)
            at += n

    def step(self, closure):                                    # noqa: D102
        group = self.param_groups[0]
        if group["line_search_fn"] is not None:
            _unsupported(f"LBFGS(line_search_fn={group['line_search_fn']!r})")
        lr = group["lr"]
        max_iter, max_eval = group["max_iter"], group["max_eval"]
        tol_grad, tol_change = group["tolerance_grad"], group["tolerance_change"]
        history = group["history_size"]
        st = self._global
        st.setdefault("n_iter", 0)

        orig = closure()
        loss = float(orig)
        evals = 1
        flat = self._flat_grad()
        if _np.abs(flat).max() <= tol_grad:
            return orig

        d = st.get("d")
        t = st.get("t")
        old_dirs = st.get("old_dirs", [])
        old_stps = st.get("old_stps", [])
        ro = st.get("ro", [])
        h_diag = st.get("h_diag", 1.0)
        prev_flat = st.get("prev_flat")
        prev_loss = st.get("prev_loss")

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
                # 두 겹 되돌이 — 헤세 역행렬을 안 만들고 방향만 낸다.
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

            prev_flat = flat.copy()
            prev_loss = loss
            t = min(1.0, 1.0 / _np.abs(flat).sum()) * lr if st["n_iter"] == 1 else lr
            gtd = float(flat @ d)
            if gtd > -tol_change:
                break

            self._add_step(t, d)
            if n_iter != max_iter:
                # 마지막 되돌이에서는 다시 안 잰다 — torch 도 그렇다.
                loss = float(closure())
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


class _Scheduler:
    """스케줄러는 `optimizer.param_groups` 의 lr 을 고친다. 에폭마다 한 번 부른다."""

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        # **기준은 `initial_lr` 이지 지금 lr 이 아니다.** 옵티마이저에 한 번만 찍히고
        # (`setdefault`), 그 뒤에 세워지는 스케줄러들도 **같은** 기준을 본다.
        #
        # 처음에는 세울 때의 lr 을 기준으로 삼았다. 혼자 쓰면 둘이 같아서 안 걸리는데,
        # 스케줄러를 이어 붙이면 두 번째 것이 첫 번째가 이미 깎아 둔 값을 기준으로
        # 잡는다 — `SequentialLR` 이 이정표에서 0.2 로 돌아가야 하는 자리에서 0.05 로
        # 이어졌고, 최대차 1.5e-01 이었다.
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs = [g["initial_lr"] for g in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        """**옵티마이저는 안 담는다** — torch 도 그렇다. 담으면 서로를 물고 돈다.

        이어 붙인 학습에서 옵티마이저만 되돌리고 스케줄러를 새로 세우면 학습률이
        **처음 값으로 돌아간다.** 반쯤 식혀 놓은 학습이 다시 뜨거워지는 것이고,
        손실은 내려가던 것이 한 번 올라갔다 다시 내려온다 — 오류는 안 난다.
        골든이 이 자리를 `opt::StepLR/이어서 학습하기` 로 붙잡는다.

        하위 클래스가 자기 상태를 더 들어도 여기 적을 것이 없다. `__dict__` 에서
        옵티마이저만 빼고 통째로 담으므로 `T_cur` 같은 것이 저절로 따라온다 —
        이름을 따로 나열하면 스케줄러를 하나 더할 때마다 그 목록을 갱신해야 하고,
        잊으면 **그 스케줄러만 조용히 안 이어진다.**
        """
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state):
        self.__dict__.update(state)
        return self


class StepLR(_Scheduler):
    """step_size 에폭마다 gamma 를 곱한다. 4장의 "멀리서는 성큼, 가까이서는 조심".

    **재귀식이다** — 아래 `ExponentialLR` 이 적어 둔 것과 같은 이유다. 여기만
    `base * gamma ** (last_epoch // step_size)` 라는 닫힌 식이었다.

    혼자 처음부터 돌리면 두 방식이 **같은 수열**을 낸다. 그래서 `StepLR/자취` 가
    오래 초록이었다. 갈리는 것은 lr 이 이미 옮겨진 옵티마이저 위에 스케줄러를
    새로 세울 때 — 즉 **이어서 학습할 때**다. 닫힌 식은 그 순간 학습률을 처음
    값으로 되돌려 놓는다(0.05 를 0.2 로). 반쯤 식힌 학습이 다시 뜨거워지고,
    오류는 안 난다. `opt::StepLR/이어서 학습하기` 가 이 자리를 붙잡는다.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size, self.gamma = step_size, gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        now = [g["lr"] for g in self.optimizer.param_groups]
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return now
        return [lr * self.gamma for lr in now]


class MultiStepLR(_Scheduler):
    """이정표에서만 깎는다. **재귀식이다** — `StepLR` 과 같은 이유로 고쳤다.

    이정표가 겹쳐 적히면(`[3, 3]`) 그 자리에서 두 번 곱한다 — torch 가 그렇고,
    닫힌 식으로 세던 때도 그랬으므로 여기서도 개수를 센다.
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones, self.gamma = sorted(milestones), gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        now = [g["lr"] for g in self.optimizer.param_groups]
        hits = self.milestones.count(self.last_epoch)
        if hits == 0:
            return now
        return [lr * self.gamma ** hits for lr in now]


class ExponentialLR(_Scheduler):
    """**재귀식이다** — 지금 학습률에 곱한다. 원래 학습률에서 다시 세지 않는다.

    혼자 쓰면 두 방식이 같은 수열을 낸다. 갈리는 것은 **다른 스케줄러가 같은 lr 을
    함께 만질 때**다 — `ChainedScheduler` 로 둘을 겹치면 재귀식은 서로의 결과 위에
    쌓이고 닫힌 식은 남이 한 일을 덮어쓴다. torch 가 재귀식이고, 그래서 여기도 그렇다.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [g["lr"] for g in self.optimizer.param_groups]
        return [g["lr"] * self.gamma for g in self.optimizer.param_groups]


class CosineAnnealingLR(_Scheduler):
    """T_max 에폭에 걸쳐 코사인 곡선으로 내린다. 끝에서 부드럽게 멎는다."""

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base - self.eta_min)
                * (1 + _math.cos(_math.pi * self.last_epoch / self.T_max)) / 2
                for base in self.base_lrs]


class LambdaLR(_Scheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.lr_lambda(self.last_epoch) for base in self.base_lrs]


class ConstantLR(_Scheduler):
    """`total_iters` 까지 **깎아 두었다가 원래대로 돌아온다.** 워밍업의 가장 단순한 꼴."""

    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1):
        self.factor, self.total_iters = factor, total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 재귀식. 깎는 것도 되돌리는 것도 **그 순간 한 번씩만** 한다.
        groups = self.optimizer.param_groups
        if self.last_epoch == 0:
            return [g["lr"] * self.factor for g in groups]
        if self.last_epoch != self.total_iters:
            return [g["lr"] for g in groups]
        return [g["lr"] / self.factor for g in groups]


class LinearLR(_Scheduler):
    """시작 배율에서 끝 배율까지 **직선으로** 옮겨간다.

    `ConstantLR` 과 끝에서 만난다 — `total_iters` 를 지나면 둘 다 원래 학습률이다.
    그래서 마지막 값만 보면 둘을 못 가르고, 골든이 자취를 통째로 묻는다.
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0,
                 total_iters=5, last_epoch=-1):
        self.start_factor, self.end_factor = start_factor, end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = min(self.last_epoch, self.total_iters)
        scale = self.start_factor + (self.end_factor - self.start_factor) * (
            t / self.total_iters if self.total_iters else 1.0)
        return [base * scale for base in self.base_lrs]


class PolynomialLR(_Scheduler):
    """`(1 - t/T)^power` 로 내린다. `power=1` 이면 직선이다."""

    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1):
        self.total_iters, self.power = total_iters, power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        groups = self.optimizer.param_groups
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [g["lr"] for g in groups]
        # 재귀식이라 **한 스텝의 비율**을 곱한다. `t == total_iters` 에서 0 이 된다.
        decay = ((1.0 - self.last_epoch / self.total_iters)
                 / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [g["lr"] * decay for g in groups]


class MultiplicativeLR(_Scheduler):
    """**곱해 나간다** — `LambdaLR` 처럼 배율을 받지만 기준이 원래 학습률이 아니라
    지금 학습률이다. 그 차이 때문에 같은 람다를 줘도 결과가 다르다."""

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return list(self.base_lrs)
        return [g["lr"] * self.lr_lambda(self.last_epoch)
                for g in self.optimizer.param_groups]


class CosineAnnealingWarmRestarts(_Scheduler):
    """코사인으로 내리다가 **처음으로 되돌린다.** 주기가 `T_mult` 배씩 길어진다."""

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
        self.T_0, self.T_mult, self.eta_min = T_0, T_mult, eta_min
        self.T_i, self.T_cur = T_0, last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base - self.eta_min)
                * (1 + _math.cos(_math.pi * self.T_cur / self.T_i)) / 2
                for base in self.base_lrs]

    def step(self):
        self.last_epoch += 1
        self.T_cur = self.T_cur + 1
        # 주기를 다 쓰면 되돌리고 다음 주기를 늘린다.
        while self.T_cur >= self.T_i:
            self.T_cur -= self.T_i
            self.T_i *= self.T_mult
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr


class OneCycleLR(_Scheduler):
    """올렸다가 내린다. **현대 학습 레시피의 기본값에 가깝다.**

    torch 의 기본은 코사인 모양이고 올라가는 구간이 전체의 30% 다. 초기 학습률은
    `max_lr/div_factor` 이고 끝은 `초기/final_div_factor` 라, **옵티마이저에 준
    학습률이 아예 안 쓰인다** — 세우는 순간 덮어쓴다.
    """

    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 div_factor=25.0, final_div_factor=1e4, last_epoch=-1):
        self.max_lr, self.total_steps, self.pct_start = max_lr, total_steps, pct_start
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        # **torch 의 셈을 그대로 쓴다.** `pct_start × total_steps − 1` 이지
        # `pct_start × (total_steps − 1)` 이 아니다 — 뒤엣것으로 적었더니 정점이
        # 반 스텝쯤 밀려 최대차 7.6e-02 가 났다.
        self.up = float(pct_start * total_steps) - 1
        self.down = total_steps - self.up - 1
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = min(self.last_epoch, self.total_steps - 1)
        if t <= self.up:
            frac = t / self.up if self.up else 1.0
            lo, hi = self.initial_lr, self.max_lr
        else:
            frac = (t - self.up) / max(1e-12, self.down)
            lo, hi = self.max_lr, self.min_lr
        # 코사인 보간 — 양 끝에서 기울기가 0 이다.
        scale = (1 - _math.cos(_math.pi * frac)) / 2
        return [lo + (hi - lo) * scale for _ in self.base_lrs]


class CyclicLR(_Scheduler):
    """학습률을 **오르내리게** 한다. 안장점을 빠져나오라고 일부러 흔드는 방식이다.

    `step_size_up` 만큼 올라갔다가 `step_size_down` 만큼 내려온다. 안 주면 올라간
    만큼 내려온다 — **오르내림이 같으면 그 인자가 있는지도 안 보인다.**

    `mode` 셋:
      `triangular`  — 봉우리 높이가 늘 같다
      `triangular2` — 한 주기마다 높이가 절반이 된다
      `exp_range`   — 높이에 `gamma^걸음` 을 곱한다 (주기가 아니라 **걸음**이다)

    마지막 것의 기준이 걸음이라는 게 갈리는 자리다. `scale_mode` 가 `cycle` 이면
    주기 번호를 넣고 `iterations` 면 걸음 수를 넣는데, `exp_range` 만 후자다.
    """

    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000,
                 step_size_down=None, mode="triangular", gamma=1.0,
                 scale_fn=None, scale_mode="cycle", cycle_momentum=True,
                 base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
        self.base_lr, self.max_lr = base_lr, max_lr
        self.up = step_size_up
        self.down = step_size_up if step_size_down is None else step_size_down
        self.mode, self.gamma = mode, gamma
        self.cycle_momentum = cycle_momentum
        self.base_momentum, self.max_momentum = base_momentum, max_momentum
        if scale_fn is None:
            if mode == "triangular":
                scale_fn, scale_mode = (lambda _c: 1.0), "cycle"
            elif mode == "triangular2":
                scale_fn, scale_mode = (lambda c: 1 / (2.0 ** (c - 1))), "cycle"
            elif mode == "exp_range":
                scale_fn, scale_mode = (lambda i: gamma ** i), "iterations"
            else:
                raise ValueError(f"CyclicLR: 모르는 mode {mode!r}")
        self.scale_fn, self.scale_mode = scale_fn, scale_mode
        super().__init__(optimizer, last_epoch)

    def _shape(self):
        total = self.up + self.down
        ratio = self.up / total
        cycle = _math.floor(1 + self.last_epoch / total)
        x = 1 + self.last_epoch / total - cycle
        # 올라가는 구간과 내려오는 구간의 기울기가 다르다 — 위 함정이 이것이다.
        rise = x / ratio if x <= ratio else (x - 1) / (ratio - 1)
        scale = self.scale_fn(cycle if self.scale_mode == "cycle"
                              else self.last_epoch)
        return rise * scale

    def get_lr(self):
        height = (self.max_lr - self.base_lr) * self._shape()
        lr = self.base_lr + height
        if self.cycle_momentum:
            # **모멘텀은 반대로 간다** — 학습률이 높을 때 낮다.
            span = (self.max_momentum - self.base_momentum) * self._shape()
            for group in self.optimizer.param_groups:
                if "momentum" in group:
                    group["momentum"] = self.max_momentum - span
        return [lr for _ in self.optimizer.param_groups]


class SequentialLR:
    """스케줄러를 **이어 붙인다.** 이정표에 닿으면 다음 것으로 넘어간다.

    `_Scheduler` 를 안 물려받는다 — 자기 `get_lr` 이 없고 남의 것을 골라 부르는
    일이라, 물려받으면 생성자가 `step()` 을 한 번 부르며 순서가 어긋난다.
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        self.optimizer = optimizer
        self.schedulers = list(schedulers)
        self.milestones = list(milestones)
        self.last_epoch = 0
        # **세우는 순간 첫 스케줄러의 값으로 되돌린다.** 각 스케줄러가 만들어질 때
        # 한 번씩 lr 을 고쳐 놓았으므로, 그대로 두면 마지막 것의 값에서 시작한다.
        for group, base in zip(optimizer.param_groups, self.schedulers[0].base_lrs):
            group["lr"] = base
        self.schedulers[0].last_epoch = -1
        self.schedulers[0].step()

    def _which(self):
        idx = sum(1 for m in self.milestones if m <= self.last_epoch)
        return idx, (self.milestones[idx - 1] if idx else 0)

    def step(self):
        self.last_epoch += 1
        idx, start = self._which()
        sch = self.schedulers[min(idx, len(self.schedulers) - 1)]
        if self.last_epoch == start:
            # **넘어가는 순간 기준 학습률로 되돌리고** 새 스케줄러를 처음부터 밟는다.
            # 앞 스케줄러가 깎아 둔 값에서 이어지지 않는다 — torch 가 그 자리에서
            # 닫힌 식을 쓰기 때문이고, 닫힌 식의 기준은 `initial_lr` 이다.
            for group, base in zip(self.optimizer.param_groups, sch.base_lrs):
                group["lr"] = base
            sch.last_epoch = -1
        sch.step()

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class ChainedScheduler:
    """여럿을 **동시에** 건다. 각자의 배율이 곱해진다."""

    def __init__(self, schedulers):
        self.schedulers = list(schedulers)
        self.optimizer = self.schedulers[0].optimizer

    def step(self):
        for sch in self.schedulers:
            sch.step()

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class ReduceLROnPlateau:
    """**좋아지지 않을 때** 내린다. 다른 것들과 달리 `step(metric)` 으로 값을 받는다 —
    6장의 조기 종료와 같은 발상이고, 멈추는 대신 보폭을 줄인다."""

    def __init__(self, optimizer, mode="min", factor=0.1, patience=10,
                 threshold=1e-4, min_lr=0.0):
        self.optimizer = optimizer
        self.mode, self.factor, self.patience = mode, factor, patience
        self.threshold, self.min_lr = threshold, min_lr
        self.best = None
        self.num_bad_epochs = 0

    def _better(self, value):
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best * (1 - self.threshold)
        return value > self.best * (1 + self.threshold)

    def step(self, metric):
        metric = float(metric.item() if isinstance(metric, Tensor) else metric)
        if self._better(metric):
            self.best, self.num_bad_epochs = metric, 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                for group in self.optimizer.param_groups:
                    group["lr"] = max(group["lr"] * self.factor, self.min_lr)
                self.num_bad_epochs = 0

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class _LRScheduler(_Namespace):
    StepLR = StepLR
    MultiStepLR = MultiStepLR
    ExponentialLR = ExponentialLR
    CosineAnnealingLR = CosineAnnealingLR
    LambdaLR = LambdaLR
    ReduceLROnPlateau = ReduceLROnPlateau
    ConstantLR = ConstantLR
    LinearLR = LinearLR
    PolynomialLR = PolynomialLR
    MultiplicativeLR = MultiplicativeLR
    CosineAnnealingWarmRestarts = CosineAnnealingWarmRestarts
    OneCycleLR = OneCycleLR
    CyclicLR = CyclicLR
    SequentialLR = SequentialLR
    ChainedScheduler = ChainedScheduler
    # torch 가 기반 클래스를 이 이름으로 내놓는다. 상속해서 자기 스케줄러를 만드는
    # 코드가 이것을 부른다.
    LRScheduler = _Scheduler


class _Optim(_Namespace):
    SGD = SGD
    Adam = Adam
    AdamW = AdamW
    RMSprop = RMSprop
    Adagrad = Adagrad
    Adadelta = Adadelta
    Adamax = Adamax
    NAdam = NAdam
    RAdam = RAdam
    ASGD = ASGD
    Rprop = Rprop
    Adafactor = Adafactor
    LBFGS = LBFGS
    Optimizer = Optimizer
    lr_scheduler = _LRScheduler()


optim = _Optim()


