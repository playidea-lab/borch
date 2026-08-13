"""대조 케이스 표 — **torch 를 임포트하지 않는다.**

골든 2단계는 브라우저에서 도는데 거기에는 진짜 torch 가 없다. 케이스 표가 torch 를
끌고 오면 그쪽에서는 임포트조차 안 된다. 그래서 표만 여기 떼어 두고, 어느 라이브러리를
넣을지는 부르는 쪽이 정한다 — 케이스는 전부 `lambda L: ...` 로 라이브러리를 인자로 받는다.

`conformance.py` 와 `golden.py` 가 **같은 표**를 본다. 두 벌로 두면 언젠가 갈리고,
그때 갈린 쪽이 어느 쪽인지 아무도 모른다.
"""

import hashlib

import numpy as np


def golden_inputs():
    """케이스가 쓰는 입력. 뽑는 **순서**가 값을 정하므로 건드리지 않는다."""
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal(6).astype(np.float32)
    xp = np.abs(x1) + 0.2
    x2 = rng.standard_normal((3, 4)).astype(np.float32)
    img = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
    # dtype 을 반드시 적는다. numpy 의 기본 정수는 C 의 `long` 을 따르는데, 그게
    # 64비트 맥·리눅스에서는 int64 지만 **wasm32(Pyodide)에서는 int32** 다.
    # 안 적으면 브라우저와 네이티브가 다른 입력을 만들고, 골든 대조가 대조가 아니게 된다.
    # (실측으로 걸렸다 — 입력 지문 검사가 잡아준 첫 건이다.)
    idx2 = np.array([[0, 2], [1, 3], [2, 0]], dtype=np.int64)
    # erf·gelu 의 꼬리. xp 는 양수 0.2 이상만 보고 x1 은 대략 [-2, 2] 라,
    # 자릿수가 날아가는 두 자리(원점 근처와 큰 |x|)를 아무도 안 보고 있었다.
    tail = np.array([-8., -6., -4., -1., -1e-3, 0., 1e-3, 1., 4., 6., 8.], dtype=np.float32)

    # 학습 케이스용. 가중치를 고정해 넣어야 세 라이브러리가 **같은 자리에서 출발**한다 —
    # 각자 초기화하면 무엇이 갈렸는지가 아니라 초기화가 갈렸는지를 보게 된다.
    train_x = rng.standard_normal((24, 6)).astype(np.float32)
    train_y = rng.integers(0, 3, 24).astype(np.int64)
    w0 = (rng.standard_normal((8, 6)) * 0.3).astype(np.float32)
    b0 = (rng.standard_normal(8) * 0.1).astype(np.float32)
    w1 = (rng.standard_normal((3, 8)) * 0.3).astype(np.float32)
    b1 = (rng.standard_normal(3) * 0.1).astype(np.float32)

    # 합성곱용. img 는 (2,3,4,4) 라 3채널 → 4채널 3×3 필터가 맞는다.
    cw = (rng.standard_normal((4, 3, 3, 3)) * 0.3).astype(np.float32)
    cb = (rng.standard_normal(4) * 0.1).astype(np.float32)

    # CNN 학습용 — (8,1,8,8) → conv → pool → flatten(64) → 3
    cnn_x = rng.standard_normal((8, 1, 8, 8)).astype(np.float32)
    cnn_y = rng.integers(0, 3, 8).astype(np.int64)
    ck = (rng.standard_normal((4, 1, 3, 3)) * 0.3).astype(np.float32)
    ckb = (rng.standard_normal(4) * 0.1).astype(np.float32)
    fw = (rng.standard_normal((3, 64)) * 0.2).astype(np.float32)
    fb = (rng.standard_normal(3) * 0.1).astype(np.float32)

    return {"x1": x1, "xp": xp, "x2": x2, "img": img, "idx2": idx2, "tail": tail,
            "train_x": train_x, "train_y": train_y,
            "w0": w0, "b0": b0, "w1": w1, "b1": b1,
            "cw": cw, "cb": cb,
            "cnn_x": cnn_x, "cnn_y": cnn_y, "ck": ck, "ckb": ckb, "fw": fw, "fb": fb}


def wide_cases(inp=None):
    """교재 범위 밖이지만 튜토리얼·실무에서 흔한 것들.

    이름만 있고 값이 다르면 그것도 거짓이라, 있는 것은 전부 값으로 대조한다.
    """
    inp = golden_inputs() if inp is None else inp
    x1, xp, x2 = inp["x1"], inp["xp"], inp["x2"]
    img, idx2, tail = inp["img"], inp["idx2"], inp["tail"]

    cases = []
    for fn in ("log2 log10 rsqrt square reciprocal tan sinh cosh erf sign floor ceil round "
               "sqrt exp abs sin cos").split():
        cases.append((fn, lambda L, f=fn: getattr(L, f)(L.tensor(xp))))
    for fn in ("prod count_nonzero",).__getitem__(0).split():
        cases.append((fn, lambda L, f=fn: getattr(L, f)(L.tensor(x1))))
    cases += [
        ("median", lambda L: L.median(L.tensor(x1))),
        ("median(dim)", lambda L: L.median(L.tensor(x2), dim=1).values),
        ("cumsum", lambda L: L.cumsum(L.tensor(x1), 0)),
        ("cumprod", lambda L: L.cumprod(L.tensor(x1), 0)),
        ("norm", lambda L: L.norm(L.tensor(x2))),
        ("topk", lambda L: L.topk(L.tensor(x1), 3).values),
        ("sort", lambda L: L.sort(L.tensor(x1)).values),
        ("unique", lambda L: L.unique(L.tensor(np.array([1., 1., 2., 3.], dtype=np.float32)))),
        ("gather", lambda L: L.gather(L.tensor(x2), 1, L.tensor(idx2))),
        ("flip", lambda L: L.flip(L.tensor(x2), [0])),
        ("roll", lambda L: L.roll(L.tensor(x1), 2)),
        ("index_select",
         lambda L: L.index_select(L.tensor(x2), 0, L.tensor(np.array([2, 0], dtype=np.int64)))),
        ("masked_select", lambda L: L.masked_select(L.tensor(x1), L.tensor(x1) > 0)),
        ("narrow", lambda L: L.narrow(L.tensor(x2), 1, 1, 2)),
        ("split", lambda L: L.split(L.tensor(x1), 2)[1]),
        ("chunk", lambda L: L.chunk(L.tensor(x1), 3)[2]),
        ("unbind", lambda L: L.unbind(L.tensor(x2))[1]),
        ("maximum", lambda L: L.maximum(L.tensor(x1), L.tensor(-x1))),
        ("minimum", lambda L: L.minimum(L.tensor(x1), L.tensor(-x1))),
        ("clamp", lambda L: L.clamp(L.tensor(x1), min=-0.5, max=0.5)),
        ("mm", lambda L: L.mm(L.tensor(x2), L.tensor(x2.T))),
        ("dot", lambda L: L.dot(L.tensor(x1), L.tensor(x1))),
        ("outer", lambda L: L.outer(L.tensor(x1[:2]), L.tensor(x1[:3]))),
        ("diag", lambda L: L.diag(L.tensor(x2[:3, :3]))),
        ("trace", lambda L: L.trace(L.tensor(x2[:3, :3]))),
        ("F.gelu", lambda L: L.nn.functional.gelu(L.tensor(x1))),
        ("F.silu", lambda L: L.nn.functional.silu(L.tensor(x1))),
        ("F.leaky_relu", lambda L: L.nn.functional.leaky_relu(L.tensor(x1), 0.1)),
        ("F.elu", lambda L: L.nn.functional.elu(L.tensor(x1))),
        ("F.log_softmax", lambda L: L.nn.functional.log_softmax(L.tensor(x2), dim=-1)),
        ("F.avg_pool2d", lambda L: L.nn.functional.avg_pool2d(L.tensor(img), 2)),
        ("F.l1_loss", lambda L: L.nn.functional.l1_loss(L.tensor(x1), L.tensor(-x1))),
        ("F.smooth_l1_loss",
         lambda L: L.nn.functional.smooth_l1_loss(L.tensor(x1), L.tensor(-x1))),
        ("F.nll_loss", lambda L: L.nn.functional.nll_loss(
            L.nn.functional.log_softmax(L.tensor(x2), dim=-1),
            L.tensor(np.array([0, 1, 2], dtype=np.int64)))),
        ("F.pad", lambda L: L.nn.functional.pad(L.tensor(x2), (1, 1))),
        ("F.normalize", lambda L: L.nn.functional.normalize(L.tensor(x2), dim=1)),
        ("F.cosine_similarity",
         lambda L: L.nn.functional.cosine_similarity(L.tensor(x2), L.tensor(x2 * 2))),
        ("F.one_hot",
         lambda L: L.nn.functional.one_hot(L.tensor(np.array([0, 2], dtype=np.int64)), 3)),
        ("erf(꼬리)", lambda L: L.erf(L.tensor(tail))),
        ("F.gelu(꼬리)", lambda L: L.nn.functional.gelu(L.tensor(tail))),
    ]

    # 합성곱·풀링·정규화 — S3 가 더한 것들. 스트라이드 2 를 같이 보는 것은 의도다.
    # 역방향에서 기울기 사이에 0 을 끼우는 경로가 거기서만 돈다.
    cw, cb = inp["cw"], inp["cb"]
    cases += [
        ("F.conv2d", lambda L: L.nn.functional.conv2d(
            L.tensor(img), L.tensor(cw), L.tensor(cb), 1, 1)),
        ("F.conv2d(패딩0)", lambda L: L.nn.functional.conv2d(
            L.tensor(img), L.tensor(cw), None, 1, 0)),
        ("F.conv2d(스트라이드2)", lambda L: L.nn.functional.conv2d(
            L.tensor(img), L.tensor(cw), L.tensor(cb), 2, 1)),
        ("F.max_pool2d", lambda L: L.nn.functional.max_pool2d(L.tensor(img), 2)),
        ("BatchNorm2d(학습)", lambda L: L.nn.BatchNorm2d(3)(L.tensor(img))),
        # **저장·복원 뒤의 평가 모드.** running 통계가 state_dict 에서 빠지면 여기서만
        # 갈린다 — 학습은 멀쩡해 보이고 추론만 틀리는, 코어가 겪은 그 결함이다.
        ("BatchNorm2d(저장→복원→eval)", lambda L: _bn_roundtrip(L, img)),
        ("median(dim).indices", lambda L: L.median(L.tensor(x2), dim=1).indices),
    ]
    return cases


def _bn_roundtrip(L, img):
    trained = L.nn.BatchNorm2d(3)
    trained(L.tensor(img))                      # running 통계가 갱신된다
    fresh = L.nn.BatchNorm2d(3)
    fresh.load_state_dict(trained.state_dict())
    fresh.eval()
    return fresh(L.tensor(img))


def _grad_of(leaf, name):
    """잎에 기울기가 **실제로 도착했는지** 확인하고 꺼낸다.

    안 왔으면(None) 그래프가 끊긴 것이다. 그냥 두면 대조 단계에서 엉뚱한 오류가 나고,
    코어가 겪은 그대로 "학습은 도는데 가중치가 안 움직이는" 상태를 못 잡는다.
    """
    if leaf.grad is None:
        raise RuntimeError(f"{name}: 기울기가 잎에 도착하지 않았다 — 그래프가 끊겼다")
    return leaf.grad


def grad_cases(inp=None):
    """**기울기**를 대조한다.

    순방향만 맞고 역방향이 틀리면 "학습은 돌아가고 손실도 내려가는데 값이 다른" 상태가
    된다. 코어가 BatchNorm 으로 오래 겪은 종류이고, 값 대조만으로는 안 잡힌다.

    각 케이스는 잎을 만들어 스칼라로 접고 `backward()` 를 부른 뒤 **잎의 기울기**를 준다.
    """
    inp = golden_inputs() if inp is None else inp
    x1, xp, x2 = inp["x1"], inp["xp"], inp["x2"]

    cases = []

    def unary(name, fn, arr=x1):
        def run(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            f(L, x).sum().backward()
            return _grad_of(x, n)
        cases.append((f"grad::{name}", run))

    def binary(name, fn, which, a1=x1, a2=x1):
        def run(L, f=fn, w=which, p=a1, q=a2, n=name):
            a = L.tensor(p, requires_grad=True)
            b = L.tensor(q, requires_grad=True)
            f(L, a, b).sum().backward()
            return _grad_of(a if w == "a" else b, n)
        cases.append((f"grad::{name}/{which}", run))

    # 원소별 — 양수만 받는 것은 xp 로 준다
    for name, fn in [("exp", lambda L, x: L.exp(x)), ("abs", lambda L, x: L.abs(x)),
                     ("sin", lambda L, x: L.sin(x)), ("cos", lambda L, x: L.cos(x)),
                     ("tan", lambda L, x: L.tan(x)), ("sinh", lambda L, x: L.sinh(x)),
                     ("cosh", lambda L, x: L.cosh(x)), ("tanh", lambda L, x: L.tanh(x)),
                     ("erf", lambda L, x: L.erf(x)), ("square", lambda L, x: L.square(x))]:
        unary(name, fn)
    for name, fn in [("log", lambda L, x: L.log(x)), ("log2", lambda L, x: L.log2(x)),
                     ("log10", lambda L, x: L.log10(x)), ("sqrt", lambda L, x: L.sqrt(x)),
                     ("rsqrt", lambda L, x: L.rsqrt(x)),
                     ("reciprocal", lambda L, x: L.reciprocal(x))]:
        unary(name, fn, xp)

    # 활성 — 학습 경로가 실제로 지나는 곳
    for name, fn in [("relu", lambda L, x: L.relu(x)),
                     ("sigmoid", lambda L, x: L.sigmoid(x)),
                     ("gelu", lambda L, x: L.nn.functional.gelu(x)),
                     ("silu", lambda L, x: L.nn.functional.silu(x)),
                     ("leaky_relu", lambda L, x: L.nn.functional.leaky_relu(x, 0.1)),
                     ("elu", lambda L, x: L.nn.functional.elu(x)),
                     ("pow2", lambda L, x: x ** 2), ("neg", lambda L, x: -x)]:
        unary(name, fn)

    # 축약·모양
    unary("sum", lambda L, x: x.sum(), x2)
    unary("sum(dim)", lambda L, x: x.sum(dim=1), x2)
    unary("mean", lambda L, x: x.mean(), x2)
    unary("mean(dim)", lambda L, x: x.mean(dim=0), x2)
    unary("softmax", lambda L, x: L.nn.functional.softmax(x, dim=-1), x2)
    unary("log_softmax", lambda L, x: L.nn.functional.log_softmax(x, dim=-1), x2)
    unary("cumsum", lambda L, x: L.cumsum(x, 0))
    unary("flip", lambda L, x: L.flip(x, [0]))
    unary("clamp", lambda L, x: L.clamp(x, min=-0.5, max=0.5))
    unary("norm", lambda L, x: L.norm(x), x2)
    unary("normalize", lambda L, x: L.nn.functional.normalize(x, dim=1), x2)

    # 뽑기·손실 — **여기가 그래프를 끊기 쉬운 자리다.** 값만 떼어 돌려주면 뽑은 자리로
    # 기울기가 안 가고, 분류 손실이 통째로 미분 불가가 된다. 실제로 그렇게 났다.
    idx2, targets = inp["idx2"], np.array([0, 1, 2], dtype=np.int64)
    unary("gather", lambda L, x: L.gather(x, 1, L.tensor(idx2)), x2)
    unary("nll_loss", lambda L, x: L.nn.functional.nll_loss(
        L.nn.functional.log_softmax(x, dim=-1), L.tensor(targets)), x2)
    unary("cross_entropy",
          lambda L, x: L.nn.functional.cross_entropy(x, L.tensor(targets)), x2)

    # 뽑기·자르기 계열 — **여기가 그래프를 끊기 가장 쉬운 자리다.** 값만 떼어 돌려주면
    # 뽑은 자리로 기울기가 안 가고 학습이 조용히 멈춘다. 코어가 ROADMAP 11번에서
    # topk·sort 로 겪었고, 이 라이브러리도 리뷰 전까지 같은 상태였다.
    unary("topk", lambda L, x: L.topk(x, 3).values)
    unary("sort", lambda L, x: L.sort(x).values)
    unary("sort(내림차순)", lambda L, x: L.sort(x, descending=True).values)
    unary("narrow", lambda L, x: L.narrow(x, 0, 1, 3))
    unary("split", lambda L, x: L.split(x, 2)[1])
    unary("chunk", lambda L, x: L.chunk(x, 3)[2])
    unary("unbind", lambda L, x: L.unbind(x)[1], x2)
    unary("index_select",
          lambda L, x: L.index_select(x, 0, L.tensor(np.array([2, 0], dtype=np.int64))), x2)
    unary("pad", lambda L, x: L.nn.functional.pad(x, (1, 1)), x2)
    unary("prod", lambda L, x: L.prod(x), xp)

    # 인덱싱 — torch 코드가 가장 자주 하는 일이고, 자르기와 같은 이유로 그래프를 잇는다.
    unary("idx[0]", lambda L, x: x[0], x2)
    unary("idx[-1]", lambda L, x: x[-1], x2)
    unary("idx[1:3]", lambda L, x: x[1:3])
    unary("idx[:, 1]", lambda L, x: x[:, 1], x2)
    unary("idx[1, 2]", lambda L, x: x[1, 2], x2)
    unary("idx[0:2, 1:3]", lambda L, x: x[0:2, 1:3], x2)
    unary("idx[목록]", lambda L, x: x[[2, 0]], x2)

    # 이항 — 양쪽 잎 모두 본다. 한쪽만 보면 반대쪽 끊김을 못 잡는다.
    for which in ("a", "b"):
        binary("add", lambda L, a, b: a + b, which)
        binary("sub", lambda L, a, b: a - b, which)
        binary("mul", lambda L, a, b: a * b, which)
        binary("div", lambda L, a, b: a / b, which, xp, xp)
        binary("maximum", lambda L, a, b: L.maximum(a, b), which, x1, -x1)
        binary("minimum", lambda L, a, b: L.minimum(a, b), which, x1, -x1)
        binary("matmul", lambda L, a, b: a @ b, which, x2, x2.T.copy())
        binary("l1_loss", lambda L, a, b: L.nn.functional.l1_loss(a, b), which)
        binary("mse_loss", lambda L, a, b: L.nn.functional.mse_loss(a, b), which)
        binary("smooth_l1_loss",
               lambda L, a, b: L.nn.functional.smooth_l1_loss(a, b), which)
        binary("cosine_similarity",
               lambda L, a, b: L.nn.functional.cosine_similarity(a, b), which, x2, x2 * 2)

    # 합성곱 — **역방향을 직접 짠 자리다.** 입력·가중치·편향 셋 다 본다.
    # 스트라이드 2 를 같이 보는 것은 의도다 — 기울기 사이에 0 을 끼우는 경로가 거기서만 돈다.
    img, cw, cb = inp["img"], inp["cw"], inp["cb"]

    def conv_grad(label, which, stride, padding, use_bias):
        def run(L, w=which, s=stride, p=padding, ub=use_bias, n=label):
            x = L.tensor(img, requires_grad=True)
            k = L.tensor(cw, requires_grad=True)
            b = L.tensor(cb, requires_grad=True) if ub else None
            L.nn.functional.conv2d(x, k, b, s, p).sum().backward()
            return _grad_of({"x": x, "w": k, "b": b}[w], n)
        cases.append((f"grad::{label}/{which}", run))

    for which in ("x", "w", "b"):
        conv_grad("conv2d", which, 1, 1, True)
    for which in ("x", "w"):
        conv_grad("conv2d(패딩0)", which, 1, 0, False)
        conv_grad("conv2d(스트라이드2)", which, 2, 1, False)

    unary("max_pool2d", lambda L, x: L.nn.functional.max_pool2d(x, 2), img)

    # BatchNorm — 평균·분산이 그래프 안에 있어야 한다. 밖으로 빼면 입력 기울기가
    # 어긋나고 weight 에는 **아예 안 온다**(None). 그래서 둘 다 본다.
    def bn_grad(which):
        def run(L, w=which):
            x = L.tensor(img, requires_grad=True)
            bn = L.nn.BatchNorm2d(3)
            bn(x).sum().backward()
            return _grad_of(x if w == "x" else bn.weight, f"BatchNorm2d/{w}")
        cases.append((f"grad::BatchNorm2d/{which}", run))

    for which in ("x", "weight"):
        bn_grad(which)

    return cases


_TRAIN_STEPS = 5


def train_cases(inp=None):
    """**학습이 도는가** — 조각이 엮였을 때를 본다.

    단위 대조는 연산 하나씩만 본다. 모듈·손실·옵티마이저가 엮여야만 갈리는 것이 있고,
    코어가 통합 시나리오에서 잡은 결함 셋은 전부 그 자리에서 나왔다.

    스텝을 적게(5) 두는 것은 의도다. 학습은 차이를 증폭시키므로, 길게 돌리면 무엇이
    틀렸는지가 아니라 float32 가 갈라진 것을 보게 된다 — T4 는 비목표다.
    """
    inp = golden_inputs() if inp is None else inp
    xin, yin = inp["train_x"], inp["train_y"]
    weights = {"0.weight": inp["w0"], "0.bias": inp["b0"],
               "2.weight": inp["w1"], "2.bias": inp["b1"]}

    # 모멘텀을 **따로 본다.** 모멘텀 없는 SGD 만 보고 있었더니, 버퍼가 첫 스텝에
    # grad 의 손잡이를 물고 있다가 두 번째 스텝에서 터지는 것을 못 잡았다.
    _OPTS = {"SGD": {}, "SGD(모멘텀)": {"momentum": 0.9}, "Adam": {}}

    def trained(L, opt_name):
        model = L.nn.Sequential(L.nn.Linear(6, 8), L.nn.ReLU(), L.nn.Linear(8, 3))
        model.load_state_dict({k: L.tensor(v) for k, v in weights.items()})
        kind = opt_name.split("(")[0]
        opt = getattr(L.optim, kind)(model.parameters(), lr=0.05, **_OPTS[opt_name])
        crit = L.nn.CrossEntropyLoss()
        x, y = L.tensor(xin), L.tensor(yin)
        for _ in range(_TRAIN_STEPS):
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        return model

    def loss_of(L, model):
        return L.nn.CrossEntropyLoss()(model(L.tensor(xin)), L.tensor(yin))

    cases = []
    for opt_name in ("SGD", "SGD(모멘텀)", "Adam"):
        cases.append((f"train::{opt_name}/손실",
                      lambda L, o=opt_name: loss_of(L, trained(L, o))))
        # 가중치까지 본다. 손실만 보면 **파라미터가 안 움직여도** 비슷해 보일 수 있다.
        cases.append((f"train::{opt_name}/0.weight",
                      lambda L, o=opt_name: dict(trained(L, o).named_parameters())["0.weight"]))

    # CNN — 합성곱·풀링이 학습 루프 안에서 엮였을 때. 코어가 잡은 결함 셋이 전부
    # 이렇게 엮인 자리에서 나왔고, 단위 대조는 그것을 못 봤다.
    cnn_x, cnn_y = inp["cnn_x"], inp["cnn_y"]
    cnn_w = {"0.weight": inp["ck"], "0.bias": inp["ckb"],
             "4.weight": inp["fw"], "4.bias": inp["fb"]}

    def cnn_trained(L):
        model = L.nn.Sequential(
            L.nn.Conv2d(1, 4, 3, padding=1), L.nn.ReLU(), L.nn.MaxPool2d(2),
            L.nn.Flatten(), L.nn.Linear(4 * 4 * 4, 3))
        model.load_state_dict({k: L.tensor(v) for k, v in cnn_w.items()}, strict=False)
        opt = L.optim.SGD(model.parameters(), lr=0.05)
        crit = L.nn.CrossEntropyLoss()
        x, y = L.tensor(cnn_x), L.tensor(cnn_y)
        for _ in range(_TRAIN_STEPS):
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        return model

    cases.append(("train::CNN/손실", lambda L: L.nn.CrossEntropyLoss()(
        cnn_trained(L)(L.tensor(cnn_x)), L.tensor(cnn_y))))
    cases.append(("train::CNN/conv.weight",
                  lambda L: dict(cnn_trained(L).named_parameters())["0.weight"]))
    return cases


def golden_cases(inp=None):
    """골든이 다루는 전부 — 값·기울기·학습·dtype·표현."""
    inp = golden_inputs() if inp is None else inp
    return (wide_cases(inp) + grad_cases(inp) + train_cases(inp)
            + dtype_cases(inp) + repr_cases(inp) + error_cases(inp))


_DTYPES = ["float32", "int64", "bool"]
_BIN_OPS = ["+", "-", "*", "/"]
_PY_SCALARS = [("파이썬 int", 2), ("파이썬 float", 2.0), ("파이썬 bool", True)]


def _dtype_tensor(L, name):
    """양쪽에서 **같은 뜻의** dtype 을 고른다. 이름이 갈리는 것은 bool 뿐이다."""
    kind = getattr(L, "bool", None) if name == "bool" else getattr(L, name)
    if name == "bool":
        kind = getattr(L, "bool_", None) or L.bool
    return L.tensor([1, 0] if name == "bool" else [1, 2], dtype=kind)


def dtype_cases(inp=None):
    """dtype 승격 — torch 는 **범주**(bool < 정수 < 실수)로 가르고 그 안에서만 올린다.

    numpy 규칙을 물려받으면 `float32 + int64` 가 float64 가 되고, 학습자는 틀린 규칙을
    배운다. 코어는 이 표를 112건으로 덮는데, 여기는 **float64 가 빠진 3종**이다 —
    TF.js 에 배정도가 없어서 있을 수가 없다.

    결과는 dtype 이름 문자열이다. 값이 아니라 **어떤 형이 나오는가**를 묻는다.
    거부하는 조합(불리언 뺄셈)은 **예외 종류**를 답으로 적는다 — 거부하는 것도 명세다.
    """

    def outcome(fn):
        try:
            return str(fn().dtype)
        except Exception as exc:                                    # noqa: BLE001
            return f"<{type(exc).__name__}>"

    cases = []
    for a in _DTYPES:
        for b in _DTYPES:
            for op in _BIN_OPS:
                cases.append((
                    f"dtype::{a} {op} {b}",
                    lambda L, x=a, y=b, o=op: outcome(
                        lambda: eval(f"p {o} q", {},                 # noqa: S307
                                     {"p": _dtype_tensor(L, x), "q": _dtype_tensor(L, y)}))))
    for a in _DTYPES:
        for label, value in _PY_SCALARS:
            for op in _BIN_OPS:
                cases.append((
                    f"dtype::{a} {op} {label}",
                    lambda L, x=a, v=value, o=op: outcome(
                        lambda: eval(f"p {o} s", {},                 # noqa: S307
                                     {"p": _dtype_tensor(L, x), "s": v}))))
    return cases


def _inplace_leaf(L):
    x = L.randn(3, requires_grad=True)
    x += 1
    return x


def _backward_twice(L):
    x = L.tensor([1.0, 2.0], requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    y.backward()


# 학습자가 실제로 만나는 실패들. 같은 조건에서 **같은 종류의 예외**가 나야 하고,
# 메시지에는 torch 의 정규 영문 문구가 들어 있어야 한다 — 그래야 검색이 통한다.
#
# 코어의 12건 중 10건이다. 빠진 둘(인덱스 범위 초과 · leaf 제자리 수정)은 메시지가
# 아니라 **기능이 없어서** 못 덮는다. 없는 기능은 AttributeError 로 시끄럽게 죽는다.
_ERROR_CASES = [
    ("행렬곱 모양 불일치", lambda L: L.randn(3, 4) @ L.randn(3, 2),
     "shapes cannot be multiplied"),
    ("브로드캐스트 불가", lambda L: L.randn(3, 4) + L.randn(3, 2),
     "must match the size of tensor"),
    ("reshape 원소수 불일치", lambda L: L.randn(2, 3).reshape(4, 2),
     "is invalid for input of size"),
    ("스칼라 아닌 backward", lambda L: L.randn(3, requires_grad=True).backward(),
     "grad can be implicitly created only for scalar outputs"),
    ("requires_grad 없이 backward", lambda L: L.randn(3).sum().backward(),
     "does not require grad"),
    ("정수 텐서에 requires_grad", lambda L: L.tensor([1, 2, 3], requires_grad=True), None),
    ("여러 원소에 item()", lambda L: L.randn(3).item(),
     "cannot be converted to Scalar"),
    ("Linear 입력 차원 불일치", lambda L: L.nn.Linear(4, 2)(L.randn(3, 5)),
     "shapes cannot be multiplied"),
    ("Conv2d 채널 불일치",
     lambda L: L.nn.functional.conv2d(L.randn(1, 3, 8, 8), L.randn(4, 1, 3, 3)), None),
    ("backward 두 번", _backward_twice,
     "backward through the graph a second time"),
    ("인덱스 범위 초과", lambda L: L.randn(3)[5], "out of bounds"),
    ("leaf 제자리 수정", _inplace_leaf, None),
]


def error_cases(inp=None):
    """T2 — 같은 조건에서 같은 예외가, 검색 가능한 문구와 함께 나는가."""

    def outcome(L, fn, phrase):
        try:
            fn(L)
            return "예외가 안 났다"
        except Exception as exc:                                    # noqa: BLE001
            found = (phrase in str(exc)) if phrase else True
            return f"{type(exc).__name__}|문구={found}"

    return [(f"error::{name}", lambda L, f=fn, p=phrase: outcome(L, f, p))
            for name, fn, phrase in _ERROR_CASES]


_REPR_CASES = [
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
    ("Size", "tensor([[1.0, 2.0], [3.0, 4.0]]).shape"),
]


def repr_cases(inp=None):
    """T3 — `print(t)` 가 진짜와 같은가.

    학습자가 가장 많이 하는 일이 print(tensor) 다. 다르게 찍히면 교재의 예시와 화면이
    안 맞고, 그때마다 "내가 뭘 잘못했나" 를 의심하게 된다. 값이 아니라 **글자**를 본다.
    """

    def ns(L):
        names = ("tensor", "zeros", "ones", "arange", "relu", "sigmoid")
        return {n: getattr(L, n) for n in names if hasattr(L, n)}

    return [(f"repr::{name}",
             lambda L, e=expr: repr(eval(e, {"__builtins__": {}}, ns(L))))   # noqa: S307
            for name, expr in _REPR_CASES]


def to_numpy(t):
    """어느 라이브러리의 텐서든 numpy 로.

    진짜 torch 와 browsertorch 는 둘 다 `.detach().numpy()` 가 통한다. GPU 백엔드도
    그 두 이름만 맞추면 하네스를 안 고쳐도 된다 — `numpy()` 안에서 읽어오면 된다.
    """
    return np.asarray(t.detach().numpy())


def manifest_hash(cases):
    """케이스 이름 목록의 해시. 표가 바뀐 뒤 낡은 골든으로 대조하는 것을 막는다."""
    h = hashlib.sha256()
    for name, _ in cases:
        h.update(name.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def input_fingerprints(inp):
    """입력 배열의 **키별** 지문. dtype·모양·바이트를 전부 문다.

    통짜 해시 하나만 두면 "입력이 다르다"까지만 알고 **어느 것이** 다른지는 모른다.
    그러면 갈렸을 때 사람이 처음부터 뒤져야 한다 — 실제로 한 번 그랬다.
    """
    out = {}
    for key in sorted(inp):
        arr = np.ascontiguousarray(inp[key])
        h = hashlib.sha256()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
        out[key] = h.hexdigest()
    return out


def input_fingerprint(inp):
    """입력 전체의 지문.

    numpy 의 `default_rng` 는 버전이 달라도 같은 수를 주기로 되어 있다. 그 약속에
    검사를 안 걸어두면, 어긋났을 때 **다른 입력끼리 조용히 비교**하게 된다 —
    그러면 하네스가 통과 도장을 찍는데 아무것도 대조하지 않은 셈이 된다.
    """
    h = hashlib.sha256()
    for key, digest in sorted(input_fingerprints(inp).items()):
        h.update(key.encode("utf-8"))
        h.update(digest.encode("utf-8"))
    return h.hexdigest()
