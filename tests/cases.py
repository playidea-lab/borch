"""대조 케이스 표 — **torch 를 임포트하지 않는다.**

골든 2단계는 브라우저에서 도는데 거기에는 진짜 torch 가 없다. 케이스 표가 torch 를
끌고 오면 그쪽에서는 임포트조차 안 된다. 그래서 표만 여기 떼어 두고, 어느 라이브러리를
넣을지는 부르는 쪽이 정한다 — 케이스는 전부 `lambda L: ...` 로 라이브러리를 인자로 받는다.

`conformance.py` 와 `golden.py` 가 **같은 표**를 본다. 두 벌로 두면 언젠가 갈리고,
그때 갈린 쪽이 어느 쪽인지 아무도 모른다.
"""

import collections
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

    # 순환·어텐션용. (T=5, N=2, I=3) 과 (B=2, T=5, E=4).
    seq_x = rng.standard_normal((5, 2, 3)).astype(np.float32)
    attn_x = rng.standard_normal((2, 5, 4)).astype(np.float32)

    # ── 여기부터는 원래 케이스 함수 **안에서** 만들던 것들이다 ──────────────
    #
    # 옮긴 이유: 케이스 안에서 만들면 `golden.json` 에 안 실리고, 그러면 파이썬이
    # 아닌 구현은 **기대값은 있는데 입력이 없는** 상태가 된다. 실제로 borch.ts 가
    # 그 이유 하나로 87건에서 막혔다 — numpy 의 난수기를 다시 만들지 않는 한 방법이
    # 없고, 그것을 다시 만들면 틀렸을 때 조용히 틀린다.
    #
    # **각자 자기 시드를 그대로 쓰고 맨 뒤에 붙인다.** 위의 `rng` 를 더 소비하면
    # x1 이하가 통째로 갈리기 때문이다 — 이 함수의 docstring 이 경고하는 그것이다.
    # 값이 한 자리도 안 바뀌므로 기존 기대값도 안 바뀐다.
    ck1 = (np.random.default_rng(13).standard_normal((4, 3, 3)) * 0.3).astype(np.float32)
    vol5 = np.random.default_rng(17).standard_normal((1, 2, 4, 4, 4)).astype(np.float32)
    ck3 = (np.random.default_rng(19).standard_normal((3, 2, 3, 3, 3)) * 0.3).astype(np.float32)

    # 고랭크 배터리. 축이 뒤바뀌면 값보다 **모양**에서 먼저 걸리도록 한 축만 3 이다.
    high = {}
    for r in (6, 7, 8):
        shape = [2] * r
        shape[r // 2] = 3
        high[f"rank{r}"] = np.random.default_rng(100 + r).standard_normal(shape).astype(np.float32)
    v7 = np.random.default_rng(107).standard_normal([2] * 7).astype(np.float32)
    v8 = np.random.default_rng(108).standard_normal([2] * 8).astype(np.float32)

    # 1·3차원 계열. **한 rng 를 순서대로 쓴다** — 순서가 값을 정하므로 그대로 옮긴다.
    nd = np.random.default_rng(41)
    nd_seq = nd.standard_normal((2, 3, 8)).astype(np.float32)
    nd_k1 = (nd.standard_normal((4, 3, 3)) * 0.3).astype(np.float32)
    nd_vol = nd.standard_normal((1, 2, 4, 4, 4)).astype(np.float32)
    nd_k3 = (nd.standard_normal((3, 2, 3, 3, 3)) * 0.3).astype(np.float32)
    nd_img = nd.standard_normal((2, 3, 4, 4)).astype(np.float32)

    # 변환용 이미지. uint8 과 실수를 **둘 다** 둔다 — ToTensor 가 uint8 일 때만
    # 255 로 나누는 것이 요점이라, 한쪽만 있으면 그 규칙을 안 보게 된다.
    # 순환·어텐션의 고정 가중치.
    #
    # 원래는 `mod.named_parameters()` 를 돌면서 그 자리에서 뽑았다. 그러면 torch 가
    # 있어야 모양을 알 수 있어서 이 함수(numpy 만 쓰는 1단계)에 못 들어온다. 그래서
    # **모양을 여기 적는다** — 틀리면 `load_state_dict` 가 굳히기 단계에서 시끄럽게
    # 죽으므로 조용히 틀리지 않는다.
    #
    # 뽑는 **순서**가 값을 정한다. torch 의 `named_parameters()` 순서 그대로다:
    # weight_ih, weight_hh, bias_ih, bias_hh.
    def _fixed(seed, shapes):
        r = np.random.default_rng(seed)
        return [(r.standard_normal(s) * 0.2).astype(np.float32) for s in shapes]

    # RNN(3,4)·LSTM(3,4)·GRU(3,4) 는 게이트 수만 다르다 — 1·4·3 배다.
    rnn_w = {}
    for kind, gates in (("RNN", 1), ("LSTM", 4), ("GRU", 3)):
        h = 4 * gates
        parts = _fixed(7, [(h, 3), (h, 4), (h,), (h,)])
        for name, arr in zip(("wih", "whh", "bih", "bhh"), parts):
            rnn_w[f"{kind.lower()}_{name}"] = arr
    # MultiheadAttention(4, 2): in_proj_weight, in_proj_bias, out_proj.weight, out_proj.bias
    mha = _fixed(11, [(12, 4), (12,), (4, 4), (4,)])

    # 활성함수가 **꺾이는 자리.** 난수로는 절대 안 나오는 값들이라 손으로 적는다 —
    # `hardtanh` 의 ±1, `relu6` 의 0·6, `hardsigmoid`·`hardswish` 의 ±3,
    # shrink 계열의 ±0.5. 꺾이는 점을 안 넣으면 그 점에서 규칙이 갈려도 안 걸린다.
    kinks = np.array([-6., -3., -1., -0.5, -1e-3, 0., 1e-3, 0.5, 1., 3., 6.],
                     dtype=np.float32)

    # 전치 합성곱의 가중치. **`conv2d` 와 축 순서가 다르다** — `(입력, 출력, …)` 이다.
    # 그 순서를 뒤집으면 모양은 맞는데 값이 통째로 달라지고, 그것이 이 층에서 가장
    # 흔한 실수다. 입력 채널은 각각 `nd_seq`(3)·`img`(3)·`nd_vol`(2) 에 맞춘다.
    tc = np.random.default_rng(53)
    tw1 = (tc.standard_normal((3, 4, 3)) * 0.3).astype(np.float32)
    tw2 = (tc.standard_normal((3, 4, 3, 3)) * 0.3).astype(np.float32)
    tw3 = (tc.standard_normal((2, 3, 3, 3, 3)) * 0.3).astype(np.float32)
    tb = (tc.standard_normal(4) * 0.1).astype(np.float32)
    tb3 = (tc.standard_normal(3) * 0.1).astype(np.float32)

    vis = np.random.default_rng(31)
    vis_u8 = vis.integers(0, 256, (5, 4, 3), dtype=np.uint8)
    vis_f = vis.random((5, 4, 3)).astype(np.float32)
    vis_gray = vis.integers(0, 256, (5, 4), dtype=np.uint8)

    return {"x1": x1, "xp": xp, "x2": x2, "img": img, "idx2": idx2, "tail": tail,
            "seq_x": seq_x, "attn_x": attn_x,
            "train_x": train_x, "train_y": train_y,
            "w0": w0, "b0": b0, "w1": w1, "b1": b1,
            "cw": cw, "cb": cb,
            "cnn_x": cnn_x, "cnn_y": cnn_y, "ck": ck, "ckb": ckb, "fw": fw, "fb": fb,
            "ck1": ck1, "vol5": vol5, "ck3": ck3,
            **high, "rank7_unbind": v7, "rank8_unbind": v8,
            "nd_seq": nd_seq, "nd_k1": nd_k1, "nd_vol": nd_vol, "nd_k3": nd_k3,
            "nd_img": nd_img,
            **rnn_w,
            "mha_in_w": mha[0], "mha_in_b": mha[1],
            "mha_out_w": mha[2], "mha_out_b": mha[3],
            "vis_u8": vis_u8, "vis_f": vis_f, "vis_gray": vis_gray,
            "kinks": kinks,
            "tw1": tw1, "tw2": tw2, "tw3": tw3, "tb": tb, "tb3": tb3}


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
        # 3단계에서 더한 수학·모양·비교. 값만 보는 것들이라 여기 둔다.
        ("eye", lambda L: L.eye(3)),
        ("full", lambda L: L.full((2, 3), 2.5)),
        ("zeros_like", lambda L: L.zeros_like(L.tensor(x2))),
        ("ones_like", lambda L: L.ones_like(L.tensor(x2))),
        ("linspace", lambda L: L.linspace(0, 1, 5)),
        ("tril", lambda L: L.tril(L.tensor(x2[:3, :3]))),
        ("triu", lambda L: L.triu(L.tensor(x2[:3, :3]), 1)),
        ("argmax", lambda L: L.tensor(x2).argmax(dim=1)),
        ("argmin", lambda L: L.tensor(x2).argmin(dim=1)),
        ("argsort", lambda L: L.argsort(L.tensor(x1))),
        ("bincount", lambda L: L.bincount(L.tensor(np.array([0, 1, 1, 3], dtype=np.int64)))),
        ("eq", lambda L: L.eq(L.tensor(x1), L.tensor(x1))),
        ("gt", lambda L: L.gt(L.tensor(x1), L.tensor(-x1))),
        ("logical_and", lambda L: L.logical_and(L.tensor(x1) > 0, L.tensor(-x1) > 0)),
        ("logical_not", lambda L: L.logical_not(L.tensor(x1) > 0)),
        ("isnan", lambda L: L.isnan(L.tensor(x1))),
        ("isfinite", lambda L: L.isfinite(L.tensor(x1))),
        ("all", lambda L: (L.tensor(x1) > -99).all()),
        ("any", lambda L: (L.tensor(x1) > 99).any()),
        ("bmm", lambda L: L.bmm(L.tensor(x2.reshape(1, 3, 4)),
                                L.tensor(x2.T.copy().reshape(1, 4, 3)))),
        ("einsum", lambda L: L.einsum("ij,kj->ik", L.tensor(x2), L.tensor(x2))),
        ("repeat_interleave", lambda L: L.repeat_interleave(L.tensor(x1), 2)),
        ("tile", lambda L: L.tile(L.tensor(x1), (2,))),
        ("movedim", lambda L: L.movedim(L.tensor(x2), 0, 1)),
        ("as_tensor", lambda L: L.as_tensor(x1)),
        # 길이가 다른 것을 한 배치에 담는 자리. 교재 ch05 가 이 경로를 그대로 쓴다.
        ("pad_sequence", lambda L: _pad(L)),
        ("pad_sequence(batch_first)", lambda L: _pad(L, batch_first=True)),
        ("pad_sequence(채움값)", lambda L: _pad(L, batch_first=True, padding_value=-1.0)),
        ("pad_sequence(2차원)", lambda L: L.nn.utils.rnn.pad_sequence(
            [L.tensor(x2[:3]), L.tensor(x2[:1])], batch_first=True)),
    ]
    return cases


def _pad(L, **kwargs):
    """길이 3·1·2 짜리 셋을 쌓는다. 채운 자리가 어디인지 눈으로 보이는 최소 크기."""
    parts = [L.tensor(np.array(v, dtype=np.float32))
             for v in ([1., 2., 3.], [4.], [5., 6.])]
    return L.nn.utils.rnn.pad_sequence(parts, **kwargs)


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

    # 형 변환. **여기에 조용히 틀린 자리가 있었다.**
    #
    # `.float()` 과 `.double()` 이 결과에 `requires_grad=True` 를 붙여놓고 부모를 안 달았다.
    # 그래서 `backward()` 는 예외 없이 잘 돌고 원래 텐서의 `.grad` 만 `None` 으로 남았다 —
    # 경고도 예외도 없이. `x.float()` 는 튜토리얼에 흔하니 조용히 학습이 안 되는 자리였다.
    # 다른 열두 군데(tril·diag·einsum·cumprod 등)도 기울기가 없지만 그쪽은 결과가
    # `requires_grad=False` 라 `backward()` 가 **거절한다** — 없는 것과 틀린 것의 차이다.
    unary("float()", lambda L, x: x.float())

    # `.double()` 은 같은 `_cast` 를 지나므로 산수는 위에서 이미 대조된다. 여기서 남은
    # 질문은 **브라우저 쪽이 이것을 거절하는가**다. 배정도가 없는 것이 문서화된
    # 한계이고(TF.js 도 그랬고 WGSL 의 f32 도 그렇다), 그 한계가 조용히 넓어지지
    # 않는지를 붙잡는다.
    def double_grad(L):
        x = L.tensor(x1, requires_grad=True)
        x.double().sum().backward()
        return _grad_of(x, "double()")

    cases.append(("grad::double()=브라우저는거절", _as_expected(double_grad)))

    # torch 는 흘리는데 코어가 안 흘리던 열두 자리. 전부 결과가 `requires_grad=False`
    # 라 `backward()` 가 거절했으므로 **조용히 틀리지는 않았지만**, 없는 것과 있는 것의
    # 차이는 남는다. 이제 흘린다.
    #
    # 자리마다 다른 가중치를 곱해서 받는다. 그냥 `sum()` 이면 기울기가 전부 1 이라
    # `movedim` 이 축을 뒤바꿔도, `tile` 이 조각을 엉뚱하게 겹쳐도 통과한다.
    mat = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    mat2 = np.array([[2., 0., 1.], [1., 3., 2.], [0., 1., 4.]], dtype=np.float32)
    vec = np.array([1., 2., 3., 4.], dtype=np.float32)
    zeroed = np.array([2., 0., 3., 4.], dtype=np.float32)
    short = np.array([1., 5., 2.], dtype=np.float32)

    def flows(name, fn, *arrays, which=0):
        def run(L, f=fn, a=arrays, n=name, w=which):
            leaves = [L.tensor(x, requires_grad=True) for x in a]
            out = f(L, *leaves)
            if out.shape:
                out = out * L.arange(out.numel()).reshape(out.shape).float()
            out.sum().backward()
            return _grad_of(leaves[w], n)
        cases.append((f"grad::{name}", run))

    flows("tril", lambda L, x: L.tril(x), mat)
    flows("triu(k=1)", lambda L, x: L.triu(x, 1), mat)
    flows("diag(2차원)", lambda L, x: L.diag(x), mat)
    flows("diag(1차원)", lambda L, x: L.diag(x), short)
    flows("trace", lambda L, x: L.trace(x), mat)
    flows("einsum(ij->i)", lambda L, x: L.einsum("ij->i", x), mat)
    for who in (0, 1):
        flows(f"einsum(ij,jk->ik)/{'ab'[who]}",
              lambda L, a, b: L.einsum("ij,jk->ik", a, b), mat, mat2, which=who)
    flows("cumprod", lambda L, x: L.cumprod(x, 0), vec)
    # 0 이 섞인 입력. 흔한 유도는 여기서 나눗셈이 터져 조용히 nan 을 흘린다.
    flows("cumprod(0포함)", lambda L, x: L.cumprod(x, 0), zeroed)
    flows("cumprod(2차원)", lambda L, x: L.cumprod(x, 1), mat)
    flows("tile", lambda L, x: L.tile(x, (2,)), vec)
    flows("tile(2차원)", lambda L, x: L.tile(x, (2, 3)), mat)
    flows("movedim", lambda L, x: L.movedim(x, 0, 1), mat)
    flows("repeat_interleave", lambda L, x: L.repeat_interleave(x, 3), vec)
    flows("repeat_interleave(dim)", lambda L, x: L.repeat_interleave(x, 2, 0), mat)
    # **꺾이는 자리에서 흘리는가.**
    #
    # torch 의 relu 는 입력이 정확히 0 이면 기울기를 0 으로 준다 — `x > 0` 이지
    # `x >= 0` 이 아니다. 위의 `x1` 은 무작위 정규분포라 0 이 한 번도 안 나오고,
    # 그래서 이 자리를 골든 798 건 중 아무도 안 보고 있었다. borch.ts 가 거기서 1 을
    # 흘리고 있었는데 ResNet 을 진짜 torch 와 맞춰보다 드러났다(입력 기울기 최대차 1.5e-2).
    #
    # **가중치를 자리마다 다르게 주는 것이 조건이다.** 그냥 `sum()` 이면 0 자리의
    # 기울기가 1 이든 0 이든 합계만 달라져서 다른 자리에 묻힌다 — `flows` 가 자리마다
    # 다른 가중치를 곱하는 이유가 이것이고, 여기서는 그것이 검사 자체다.
    edge = np.array([-1., 0., 1., 0.], dtype=np.float32)
    flows("relu(0에서)", lambda L, x: L.relu(x), edge)

    flows("median()", lambda L, x: L.median(x), vec)
    flows("median(dim)", lambda L, x: L.median(x, dim=1).values, mat)
    flows("fmod(%)", lambda L, x: x % 2, vec)
    for who in (0, 1):
        flows(f"pad_sequence/{'ab'[who]}",
              lambda L, a, b: L.nn.utils.rnn.pad_sequence([a, b]), vec, short, which=who)

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

    # 이어 붙이기·쌓기 — DataLoader 의 collate 가 이것 위에 선다
    unary("cat", lambda L, x: L.cat([x, x * 2]))
    unary("cat(dim=1)", lambda L, x: L.cat([x, x * 2], 1), x2)
    unary("stack", lambda L, x: L.stack([x, x * 3]))
    unary("stack(dim=1)", lambda L, x: L.stack([x, x * 3], 1), x2)

    # 메서드 형태 — torch 코드는 `x.exp()` 와 `torch.exp(x)` 를 섞어 쓴다
    unary("메서드 x.abs()", lambda L, x: x.abs())
    unary("메서드 x.exp()", lambda L, x: x.exp())
    unary("메서드 x.sqrt()", lambda L, x: x.sqrt(), xp)

    # 층 래퍼 — functional 판이 이미 있던 것들을 감쌌다. 값과 기울기 둘 다 본다.
    unary("LayerNorm", lambda L, x: L.nn.LayerNorm(4)(x), x2)
    unary("F.layer_norm", lambda L, x: L.nn.functional.layer_norm(x, (4,)), x2)
    unary("BatchNorm1d", lambda L, x: L.nn.BatchNorm1d(4)(x), x2)
    unary("F.linear", lambda L, x: L.nn.functional.linear(x, L.tensor(x2)), x2)
    unary("Softmax(층)", lambda L, x: L.nn.Softmax(dim=-1)(x), x2)
    unary("LogSoftmax(층)", lambda L, x: L.nn.LogSoftmax(dim=-1)(x), x2)
    unary("LeakyReLU(층)", lambda L, x: L.nn.LeakyReLU(0.1)(x))
    unary("ELU(층)", lambda L, x: L.nn.ELU()(x))
    unary("SiLU(층)", lambda L, x: L.nn.SiLU()(x))
    unary("Identity", lambda L, x: L.nn.Identity()(x))
    unary("Unflatten", lambda L, x: L.nn.Unflatten(0, (3, 2))(x))
    binary("L1Loss(층)", lambda L, a, b: L.nn.L1Loss()(a, b), "a")
    binary("SmoothL1Loss(층)", lambda L, a, b: L.nn.SmoothL1Loss()(a, b), "a")
    binary("BCEWithLogitsLoss", lambda L, a, b: L.nn.BCEWithLogitsLoss()(a, b), "a")

    # Embedding — 같은 번호가 여러 번 나오면 그 행에 기울기가 **쌓여야** 한다.
    def emb_grad(L):
        w = L.tensor(inp["w0"][:5], requires_grad=True)      # (5, 6)
        idx = L.tensor(np.array([0, 2, 0, 4], dtype=np.int64))
        L.nn.functional.embedding(idx, w).sum().backward()
        return _grad_of(w, "embedding")

    cases.append(("grad::embedding(중복 번호)", emb_grad))

    # 수학·모양 — 각각 TF.js 대응은 있지만 역전파를 붙여야 했던 것들
    unary("where", lambda L, x: L.where(x > 0, x, x * 0.1))
    unary("masked_fill", lambda L, x: x.masked_fill(x > 0, -1.0))
    unary("clone", lambda L, x: x.clone())
    unary("permute", lambda L, x: x.permute(1, 0), x2)
    unary("squeeze", lambda L, x: x.unsqueeze(0).squeeze())
    unary("max(dim)", lambda L, x: x.max(dim=1).values, x2)
    unary("min(dim)", lambda L, x: x.min(dim=1).values, x2)
    unary("var", lambda L, x: x.var())
    unary("std", lambda L, x: x.std())

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

    # **평균 풀링의 역방향을 아무도 안 묻고 있었다.**
    #
    # `F.avg_pool2d` 는 순방향만 표에 있었다. 그런데 borch.ts 에서 평균 풀링의 역방향이
    # **아예 안 도는** 것을 통합 시험에서 잡았다 — 쓰지 않는 바인딩이 레이아웃에서
    # 빠지면서 커맨드 버퍼가 통째로 무효가 됐는데, WebGPU 는 그걸 예외로 안 던진다.
    # 손실은 ln 10 에 앉아 있었고 ms/step 은 계속 나왔다. 표가 이것을 물었다면
    # 통합까지 갈 일이 아니었다.
    #
    # 최댓값 풀링과 갈리는 지점이 핵심이다. max 는 이긴 자리 하나에만 흘리고 avg 는
    # 창의 모든 자리에 1/n 씩 나눈다. 둘을 바꿔 구현해도 순방향은 멀쩡하다.
    def pool_grad(name, fn, arr=img):
        def run(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            # 자리마다 다른 가중치. 균일하게 접으면 avg 든 max 든 **입력 기울기의 합이
            # 같아서** 나누는 방식이 틀려도 통과한다.
            out = out * L.arange(out.numel()).reshape(out.shape).float()
            out.sum().backward()
            return _grad_of(x, n)
        cases.append((f"grad::{name}", run))

    pool_grad("avg_pool2d", lambda L, x: L.nn.functional.avg_pool2d(x, 2))
    pool_grad("avg_pool2d(스트라이드1)",
              lambda L, x: L.nn.functional.avg_pool2d(x, 2, 1))
    pool_grad("adaptive_avg_pool2d",
              lambda L, x: L.nn.functional.adaptive_avg_pool2d(x, 1))
    pool_grad("max_pool2d(가중치)", lambda L, x: L.nn.functional.max_pool2d(x, 2))

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

    # **위의 `sum()` 이 BatchNorm 역방향의 절반을 가린다.**
    #
    # 입력 기울기는 세 항으로 되어 있다. 곧바로 오는 항 하나와, 평균·분산이 입력에
    # 의존하기 때문에 생기는 보정항 둘이다. 그런데 상류 기울기가 **전부 1** 이면
    # 그 보정항 둘이 정확히 상쇄된다 — 위 케이스의 기대값이 4.7e-10 인 것이 그
    # 상쇄의 흔적이고, 즉 저 케이스는 보정항을 아예 안 묻고 있다.
    #
    # 보정항을 빼먹은 구현(평균·분산을 상수 취급하는 흔한 실수)은 위를 통과하고
    # 아래에서 걸린다. 자리마다 다른 가중치로 접으면 상쇄가 깨진다.
    def bn_grad_weighted(which):
        def run(L, w=which):
            x = L.tensor(img, requires_grad=True)
            bn = L.nn.BatchNorm2d(3)
            out = bn(x)
            out = out * L.arange(out.numel()).reshape(out.shape).float()
            out.sum().backward()
            return _grad_of(x if w == "x" else bn.weight,
                            f"BatchNorm2d(가중치)/{w}")
        cases.append((f"grad::BatchNorm2d(가중치)/{which}", run))

    for which in ("x", "weight"):
        bn_grad_weighted(which)

    # ── 씨앗을 직접 주는 역방향 (야코비안-벡터 곱) ────────────────────────
    #
    # `y.backward(v)` 는 `(∂y/∂x)ᵀ v` 를 준다. 위의 케이스들은 전부 `.sum()` 으로
    # 접고 부르는데, 그것은 **v 가 전부 1 인 특수한 경우**다.
    #
    # **씨앗을 균일하게 두면 이 케이스가 아무것도 안 잰다.** `backward(ones)` 는
    # `sum().backward()` 와 같은 답이라, 씨앗을 통째로 무시하는 구현도 통과한다.
    # 그래서 여기서는 자리마다 다른 값을 준다.
    #
    # borch.ts 만 이것을 못 받고 있었다 — 첫 인자가 `retain_graph` 였다. 코어는
    # 처음부터 받았고 골든이 안 물어서 그 갈림이 안 보였다.
    def seeded_backward(name, fn, arr, shape_of, which="x", arr2=None):
        def run(L, f=fn, a=arr, s=shape_of, w=which, b=arr2, n=name):
            x = L.tensor(a, requires_grad=True)
            args = [x] if b is None else [x, L.tensor(b, requires_grad=True)]
            y = f(L, *args)
            # 1, 2, 3 … 을 출력 모양으로 놓는다. 대칭이 없어야 씨앗이 실제로 쓰인다.
            seed = np.arange(1, int(np.prod(s)) + 1, dtype=np.float32).reshape(s)
            y.backward(L.tensor(seed))
            return _grad_of(args[0] if w == "x" else args[1], n)
        cases.append((f"grad::vjp::{name}", run))

    seeded_backward("exp", lambda L, x: L.exp(x), x1, (6,))
    seeded_backward("square", lambda L, x: x * x, x1, (6,))
    seeded_backward("mul/a", lambda L, a, b: a * b, x1, (6,), "x", x1)
    seeded_backward("mul/b", lambda L, a, b: a * b, x1, (6,), "b", x1)
    # 출력 모양이 입력과 다른 자리. 씨앗의 모양을 출력에서 가져오는지가 여기서 갈린다.
    seeded_backward("matmul", lambda L, x: L.matmul(x, x.t()), x2, (3, 3))
    seeded_backward("reshape", lambda L, x: x.reshape(2, 3), x1, (2, 3))
    # 스칼라에 씨앗을 주는 것도 torch 가 받는다 — 값이 그만큼 배가 된다.
    seeded_backward("scalar", lambda L, x: (x * x).sum(), x1, ())

    # ── 거절하는 자리 — **셋이 같은 문구로** ──────────────────────────────
    #
    # `_as_expected` 와 다르다. 저쪽은 브라우저가 **일부러** torch 와 다른 자리이고,
    # 이쪽은 셋 다 torch 와 같아야 하는 자리다. 값을 굳힐 수 없으니 문구의 조각을
    # 굳힌다 — 통과해 버리면 "안 던졌다" 가 답이 되어 갈린다.
    def refuses(name, fragment, fn):
        def run(L, f=fn, frag=fragment):
            try:
                f(L)
            except Exception as exc:                            # noqa: BLE001
                return frag if frag in str(exc) else f"다른 문구 <{exc}>"
            return "안 던졌다"
        cases.append((f"grad::거절::{name}", run))

    # **차례가 있다.** 비스칼라이면서 requires_grad 가 아니면 torch 는 "스칼라가
    # 아니다" 가 아니라 이쪽으로 거절한다 — 실측한 값이다. borch.ts 만 반대 차례였고
    # 골든이 그 조합을 안 물어서 안 보였다.
    refuses("requires_grad 를 먼저 본다",
            "does not require grad",
            lambda L: L.tensor(x1).backward())
    refuses("씨앗 없는 비스칼라",
            "grad can be implicitly created only for scalar outputs",
            lambda L: L.tensor(x1, requires_grad=True).backward())
    # 어긋난 씨앗을 브로드캐스팅으로 맞춰 주면 안 된다. 맞춰 주면 값이 그럴듯한 채로
    # 틀린 기울기가 나오고, 그것은 학습이 안 되는 것으로만 드러난다. 코어는 여기를
    # 안 보고 있어서 numpy 의 `ValueError` 가 원인에서 먼 자리에 떴다.
    def bad_seed(L):
        y = L.tensor(x1, requires_grad=True) * 2
        y.backward(L.tensor(np.ones(7, dtype=np.float32)))

    refuses("씨앗 모양이 어긋남", "Mismatch in shape", bad_seed)

    return cases


ACT_PREFIX = "act::"

# `(함수 이름, 층 이름)` — 인자 없이 기본값으로 부를 수 있는 것들.
# 함수 꼴과 층 꼴이 **같은 값**을 내야 한다. 층이 다른 함수를 부르는 실수는 값으로만
# 잡히고, 한 줄짜리 감싸개일수록 그 실수를 눈으로 못 본다.
_ACTS = [
    ("celu", "CELU"),
    ("hardshrink", "Hardshrink"),
    ("hardsigmoid", "Hardsigmoid"),
    ("hardswish", "Hardswish"),
    ("hardtanh", "Hardtanh"),
    ("logsigmoid", "LogSigmoid"),
    ("mish", "Mish"),
    ("relu6", "ReLU6"),
    ("selu", "SELU"),
    ("softplus", "Softplus"),
    ("softshrink", "Softshrink"),
    ("softsign", "Softsign"),
    ("tanhshrink", "Tanhshrink"),
]


def act_cases(inp=None):
    """활성함수 열일곱. **꺾이는 자리에서 묻는다.**

    이 저장소가 `relu` 로 배운 것이 여기 그대로 적용된다 — 난수 입력은 특별한 값을
    안 준다. 정확히 0, 정확히 ±1, 정확히 ±3, 정확히 6 은 뽑히지 않는데 활성함수는
    바로 그 점에서 꺾인다. 그래서 입력을 손으로 적었다(`kinks`).

    함수 꼴과 층 꼴을 **둘 다** 묻는다. 층은 함수를 감싼 한 줄이라 틀릴 데가 없어
    보이지만, 틀리는 방식이 하나 있다 — 다른 함수를 부르는 것. 그것은 값으로만 잡힌다.
    """
    inp = golden_inputs() if inp is None else inp
    k = inp["kinks"]
    x1, x2 = inp["x1"], inp["x2"]
    cases = []

    def add(name, fn, arr=k):
        """값과 기울기를 짝으로 단다. 활성함수는 **기울기가 본체다.**"""
        cases.append((ACT_PREFIX + name, lambda L, f=fn, a=arr: f(L, L.tensor(a))))

        def grad(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            # 자리마다 다른 가중치를 곱해 접는다 — 그냥 `sum()` 이면 기울기가 전부
            # 1 이라 어느 자리가 틀렸는지 값에 안 남는다.
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(x, n)
        cases.append((ACT_PREFIX + f"grad::{name}", grad))

    # 인자 없는 것들 — 함수 꼴은 값·기울기, 층 꼴은 값.
    for fname, cls in _ACTS:
        add(f"F.{fname}", lambda L, x, f=fname: getattr(L.nn.functional, f)(x))
        cases.append((ACT_PREFIX + f"nn.{cls}",
                      lambda L, c=cls, a=k: getattr(L.nn, c)()(L.tensor(a))))

    # 인자를 받는 것들. **기본값만 물으면 그 인자가 아예 안 쓰여도 통과한다.**
    add("F.hardtanh(범위)", lambda L, x: L.nn.functional.hardtanh(x, -0.5, 0.5))
    add("F.softplus(beta)", lambda L, x: L.nn.functional.softplus(x, beta=2.0))
    add("F.celu(alpha)", lambda L, x: L.nn.functional.celu(x, alpha=0.5))
    add("F.hardshrink(람다)", lambda L, x: L.nn.functional.hardshrink(x, lambd=1.0))
    add("F.softshrink(람다)", lambda L, x: L.nn.functional.softshrink(x, lambd=1.0))
    add("F.threshold", lambda L, x: L.nn.functional.threshold(x, 0.5, -1.0))
    cases.append((ACT_PREFIX + "nn.Threshold",
                  lambda L: L.nn.Threshold(0.5, -1.0)(L.tensor(k))))
    cases.append((ACT_PREFIX + "nn.Hardtanh(범위)",
                  lambda L: L.nn.Hardtanh(-0.5, 0.5)(L.tensor(k))))

    # `softmin` 은 `softmax(-x)` 다. **부호를 빠뜨리면 softmax 와 같아지고**, 그것은
    # 이름만 다른 같은 함수라 값으로만 갈린다.
    add("F.softmin", lambda L, x: L.nn.functional.softmin(x, dim=-1), x2)
    cases.append((ACT_PREFIX + "nn.Softmin",
                  lambda L: L.nn.Softmin(dim=-1)(L.tensor(x2))))

    # `glu` 는 축을 반으로 갈라 한쪽을 게이트로 쓴다 — 원소별이 아닌 유일한 자리다.
    add("F.glu", lambda L, x: L.nn.functional.glu(x, dim=-1), x1)
    cases.append((ACT_PREFIX + "nn.GLU", lambda L: L.nn.GLU(dim=-1)(L.tensor(x1))))

    # `prelu` 는 **학습되는 기울기**를 갖는다. 층 꼴에서 그것이 파라미터로 잡혀야
    # 하는데, 그 자리가 방금 컨테이너에서 본 등록 문제와 같은 기계다.
    add("F.prelu",
        lambda L, x: L.nn.functional.prelu(
            x, L.tensor(np.array([0.25], dtype=np.float32))))
    cases.append((ACT_PREFIX + "nn.PReLU", lambda L: L.nn.PReLU()(L.tensor(k))))
    cases.append((ACT_PREFIX + "nn.PReLU/파라미터 이름",
                  lambda L: " ".join(n for n, _ in L.nn.PReLU().named_parameters())))
    return cases


NUM_PREFIX = "num::"


def numeric_cases(inp=None):
    """수치 계열. **조합되는 것과 급수로 세는 것이 섞여 있다.**

    앞의 묶음들은 있는 연산을 엮으면 끝났다. 여기 `lgamma`·`digamma`·`erfinv` 는
    닫힌 식이 없어서 근사식을 적어야 하고, 그러면 **얼마나 맞는가**가 곧 답이다 —
    이 저장소의 허용 오차(1e-4)를 지나야 하고 그것이 이 케이스들의 값어치다.

    `cdist`·`corrcoef`·`cov` 는 통계 쪽에서 늘 부르는 것들이고 전부 조합이다.
    """
    inp = golden_inputs() if inp is None else inp
    x1, x2 = inp["x1"], inp["xp"]                       # xp 는 양수만
    mat = inp["x2"]                                     # (3, 4)
    other = (mat * 0.5 + 1.0).astype(np.float32)
    # 감마 계열은 **양수에서만** 본다. 음의 정수에서 발산하는 것이 정의라, 그 자리를
    # 케이스로 두면 무한대 비교가 되고 근사식의 품질과 상관없는 것을 묻게 된다.
    gam = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.5], dtype=np.float32)
    unit = np.array([-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((NUM_PREFIX + name, fn))

    def with_grad(name, fn, arr):
        add(name, lambda L, f=fn, a=arr: f(L, L.tensor(a)))

        def grad(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(x, n)
        cases.append((NUM_PREFIX + f"grad::{name}", grad))

    # ── 조합되는 것들. ─────────────────────────────────────────────────────
    add("cdist", lambda L: L.cdist(L.tensor(mat), L.tensor(other)))
    add("corrcoef", lambda L: L.corrcoef(L.tensor(mat)))
    add("cov", lambda L: L.cov(L.tensor(mat)))
    add("tensordot",
        lambda L: L.tensordot(L.tensor(mat), L.tensor(other), dims=([1], [1])))
    add("trapezoid", lambda L: L.trapezoid(L.tensor(x2)))
    add("trapezoid(dx)", lambda L: L.trapezoid(L.tensor(x2), dx=0.5))
    add("cumulative_trapezoid", lambda L: L.cumulative_trapezoid(L.tensor(x2)))

    # ── 급수로 세는 것들. **여기서는 얼마나 맞는가가 답이다.** ───────────────
    with_grad("lgamma", lambda L, x: L.lgamma(x), gam)
    with_grad("digamma", lambda L, x: L.digamma(x), gam)
    with_grad("erfinv", lambda L, x: L.erfinv(x), unit)
    return cases


BIT_PREFIX = "bit::"


def bit_cases(inp=None):
    """비트·정수 수학·창 함수.

    ## 음수가 이 묶음의 전부다

    비트 연산을 양수로만 물으면 세 벌의 구현이 전부 통과하고도 서로 다를 수 있다.

    - **오른쪽 시프트는 산술이다.** `-3 >> 5` 가 `-1` 이지 큰 양수가 아니다(실측).
      논리 시프트로 적으면 음수에서만 갈린다.
    - **`gcd` 는 늘 0 이상이다.** `gcd(-3, 5)` 가 `1` 이다 — 부호를 안 버리면 음수가
      나온다.
    - **0 이 섞여야 한다.** `lcm(0, 7)` 은 0 이고, 그 자리를 안 물으면 `|a·b|/gcd` 가
      0 으로 나누는 것을 아무도 못 본다.
    - **참거짓은 다른 계산이다.** `~True` 는 `-2` 가 아니라 `False` 다. 정수로만
      물으면 이 갈래가 통째로 안 돌아간다.

    ## 창 함수는 `periodic` 이 요점이다

    기본값이 참이고, 그것이 **길이를 하나 늘린다** — `N+1` 짜리 대칭 창을 만들어
    마지막을 버린다(실측). 거짓으로만 재면 두 갈래가 같은 함수처럼 보인다.
    `n == 1` 도 따로 묻는다. 나누는 자리가 0 이 되는 유일한 크기다.

    ## `frexp` 와 `fill` 이 여기 있는 이유

    `frexp` 는 지수를 **int32** 로 내고(실측), `fill` 은 이름이 한 글자 다른 `fill_`
    과 달리 **제자리가 아니다**. 둘 다 값만 보면 그럴듯해서, 원본이 그대로인지·형이
    무엇인지를 따로 물어야 드러난다.
    """
    ints = np.array([12, 10, -3, 0], dtype=np.int64)
    rhs = np.array([10, 3, 5, 7], dtype=np.int64)
    flags = np.array([True, True, False, False])
    reals = np.array([-2.5, 0.5, 1.5, 3.0], dtype=np.float32)
    grid = np.array([[1.0, 2.0, -1.0], [3.0, 4.0, 0.5]], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((BIT_PREFIX + name, fn))

    for name in ("bitwise_and", "bitwise_or", "bitwise_xor",
                 "bitwise_left_shift", "bitwise_right_shift", "gcd", "lcm"):
        add(name, lambda L, n=name: getattr(L, n)(L.tensor(ints), L.tensor(rhs)))
        add(f"메서드::{name}",
            lambda L, n=name: getattr(L.tensor(ints), n)(L.tensor(rhs)))

    add("bitwise_not", lambda L: L.bitwise_not(L.tensor(ints)))
    # 참거짓 갈래. 여기서만 논리 연산으로 돈다.
    add("bitwise_and(참거짓)",
        lambda L: L.bitwise_and(L.tensor(flags), L.tensor(~flags)))
    add("bitwise_or(참거짓)",
        lambda L: L.bitwise_or(L.tensor(flags), L.tensor(~flags)))
    add("bitwise_not(참거짓)", lambda L: L.bitwise_not(L.tensor(flags)))

    # 제자리 판. **같은 텐서를 돌려줘야** 이어 부르는 코드가 원본을 고친다.
    for name in ("gcd_", "lcm_"):
        def run(L, n=name):
            x = L.tensor(ints.copy())
            getattr(L, n)(x, L.tensor(rhs))
            return x

        add(f"제자리::{name}", run)

        def is_self(L, n=name):
            x = L.tensor(ints.copy())
            return str(getattr(L, n)(x, L.tensor(rhs)) is x)

        add(f"제자리::{name}(같은 텐서)", is_self)

    for name, arg in (("clamp_max", 1.0), ("clamp_min", 0.0)):
        add(name, lambda L, n=name, a=arg: getattr(L, n)(L.tensor(reals), a))

        def run(L, n=f"{name}_", a=arg):
            x = L.tensor(reals.copy())
            getattr(L, n)(x, a)
            return x

        add(f"제자리::{name}_", run)

    add("arctan2",
        lambda L: L.arctan2(L.tensor(reals), L.tensor(reals + 1.0)))
    add("i0", lambda L: L.i0(L.tensor(reals)))
    # 큰 쪽 갈래. 급수가 3.75 에서 바뀌므로 그 너머를 따로 묻는다.
    add("i0(큰 값)",
        lambda L: L.i0(L.tensor(np.array([4.0, 8.0, 12.0], dtype=np.float32))))
    for p in (2, 3):
        add(f"mvlgamma(p={p})",
            lambda L, q=p: L.mvlgamma(
                L.tensor(np.array([2.0, 3.0, 4.5], dtype=np.float32)), q))

    m, e = "가수", "지수"
    add(f"frexp({m})",
        lambda L: L.frexp(L.tensor(np.array([1.0, 0.5, 8.0, -3.0],
                                            dtype=np.float32)))[0])
    add(f"frexp({e})",
        lambda L: L.frexp(L.tensor(np.array([1.0, 0.5, 8.0, -3.0],
                                            dtype=np.float32)))[1])

    add("nextafter",
        lambda L: L.nextafter(L.tensor(np.array([1.0, 2.0], dtype=np.float32)),
                              L.tensor(np.array([2.0, 1.0], dtype=np.float32))))

    # **축을 가로로도 세로로도 묻는다.** 정사각이 아닌 것으로 물어야 축이 바뀌었을 때
    # 모양에서 먼저 걸린다.
    for dim in (0, 1):
        add(f"logcumsumexp(dim={dim})",
            lambda L, d=dim: L.logcumsumexp(L.tensor(grid), d))

    def logcumsumexp_grad(L):
        x = L.tensor(grid, requires_grad=True)
        out = L.logcumsumexp(x, 1)
        # **고르지 않은 무게로 센다.** 전부 1 이면 누적의 순서가 상쇄되어
        # 뒤에서부터 쌓이는 규칙이 안 드러난다.
        (out * L.tensor(np.array([[1.0, 2.0, 0.5], [0.5, 3.0, 1.5]],
                                 dtype=np.float32))).sum().backward()
        return _grad_of(x, "logcumsumexp")

    cases.append((BIT_PREFIX + "grad::logcumsumexp", logcumsumexp_grad))

    add("fill", lambda L: L.fill(L.tensor(reals), 7.0))

    def fill_leaves_source(L):
        """**`fill` 은 제자리가 아니다.** 원본을 돌려주므로, 제자리로 잘못 짓면
        7 로 채워진 것이 나온다."""
        x = L.tensor(reals.copy())
        L.fill(x, 7.0)
        return x

    add("fill(원본은 그대로)", fill_leaves_source)

    def detach_in_place(L):
        """**같은 텐서**에서 그래프를 끊는다. `detach()` 로 잘못 지으면 `is` 가
        거짓이 되고, 원본은 여전히 위쪽에 붙어 있어 역전파가 계속 흐른다."""
        x = L.tensor(reals, requires_grad=True)
        y = x * 2
        z = L.detach_(y)
        return f"{z is y} {bool(y.requires_grad)}"

    add("detach_", detach_in_place)

    windows = ("bartlett_window", "blackman_window", "hamming_window",
               "hann_window", "kaiser_window")
    for name in windows:
        for periodic in (True, False):
            add(f"{name}(6, periodic={periodic})",
                lambda L, n=name, p=periodic: getattr(L, n)(6, p))
        # 나누는 자리가 0 이 되는 유일한 크기.
        add(f"{name}(1)", lambda L, n=name: getattr(L, n)(1))
    add("hamming_window(alpha, beta)",
        lambda L: L.hamming_window(6, True, 0.5, 0.5))
    add("kaiser_window(beta=8)", lambda L: L.kaiser_window(6, True, 8.0))
    return cases


SPOT_PREFIX = "spot::"


def shape_index_cases(inp=None):
    """모양·색인. 저장소의 **어느 칸을 볼 것인가**를 묻는 이름들.

    ## `as_strided` 는 torch 에서 뷰이고 우리는 사본이다

    torch 는 저장소 하나를 여러 틀로 보므로 그 결과에 쓰면 원본이 바뀐다. borch.ts 의
    텐서는 GPU 버퍼를 하나씩 가져서 그 뷰가 표현이 안 되고, 코어만 진짜 뷰를 내면 세
    구현이 갈린다 — **값으로는 안 보이고 쓸 때만 보이는** 갈림이라 제일 나쁘다. 셋 다
    사본으로 맞췄고, 그래서 여기 케이스는 **읽기만** 묻는다. 쓰는 쪽은
    `as_strided_scatter` 가 맡는다.

    ## 무엇을 물어야 갈리는가

    - **겹치는 걸음**을 물어야 기울기가 쌓이는지 보인다. 안 겹치는 걸음으로만 재면
      한 칸에 한 번씩 오므로 쌓기를 빼먹어도 통과한다.
    - **`step`** 이 1 이 아니어야 `slice_scatter` 가 건너뛰는 자리를 안 건드리는지
      드러난다.
    - **`offset`** 이 0 이 아니어야 `diagonal_scatter`·`diag_embed` 의 밀림이 보인다.
    - **배치 축**이 있어야 대각선이 맨 뒤로 가는 규약이 드러난다. 2차원으로만 재면
      남는 축이 없어 순서를 못 묻는다.
    - **나눠떨어지지 않는 크기**여야 `tensor_split` 이 나머지를 앞에서부터 나눠 갖는
      것이 보인다. 떨어지는 크기로 재면 `chunk` 와 같은 함수처럼 보인다.
    - **겹치는 번호**여야 `index_put`·`put` 의 `accumulate` 두 갈래가 갈린다.
    - **항등원이 아닌 밑판**이어야 `include_self` 가 보인다. 1 로 채운 판에 곱하기를
      하면 켜나 끄나 같은 답이다(실측).
    - **이미 작은 조각**이 섞여야 `renorm` 이 안 건드리는 조건이 드러난다. 그리고
      **깎인 조각의 기울기**를 물어야 배율 안에 `x` 가 들어 있다는 것이 보인다 —
      순방향만으로는 `g·s` 로 적은 틀린 역방향이 통과한다.
    """
    grid = np.arange(12, dtype=np.float32).reshape(3, 4)
    line = np.arange(10, dtype=np.float32)
    trio = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    duo = np.array([4.0, 5.0], dtype=np.float32)
    # **고르지 않은 무게.** 전부 1 이면 자리마다 다른 몫이 상쇄되어 안 보인다.
    weight = np.array([[1.0, 2.0, 0.5, 3.0], [2.0, 0.5, 1.5, 1.0],
                       [0.25, 3.0, 2.0, 0.75]], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((SPOT_PREFIX + name, fn))

    def grad(name, fn, w=weight):
        def run(L, f=fn, n=name):
            x = L.tensor(grid.copy(), requires_grad=True)
            out = f(L, x)
            (out * L.tensor(np.asarray(w, dtype=np.float32).reshape(
                tuple(out.shape)))).sum().backward()
            return _grad_of(x, n)

        cases.append((SPOT_PREFIX + f"grad::{name}", run))

    # ── 걸음 ────────────────────────────────────────────────────────────
    add("as_strided", lambda L: L.as_strided(L.tensor(grid), (2, 2), (1, 2)))
    add("as_strided(offset)",
        lambda L: L.as_strided(L.tensor(grid), (2, 2), (1, 2), 3))
    add("as_strided(겹침)",
        lambda L: L.as_strided(L.tensor(grid), (3, 3), (1, 1)))
    grad("as_strided", lambda L, x: L.as_strided(x, (3, 4), (1, 3)))
    # 겹치는 걸음의 기울기 — 한 칸으로 여러 번 온다.
    grad("as_strided(겹침)", lambda L, x: L.as_strided(x, (3, 3), (1, 1)),
         np.arange(1, 10, dtype=np.float32).reshape(3, 3))

    def as_strided_in_place(L):
        """**모양까지 따라가야 한다.** 값만 옮기면 정사각으로 물었을 때만 통과한다."""
        x = L.tensor(grid.copy())
        got = L.as_strided_(x, (2, 3), (1, 2))
        return f"{got is x} {tuple(x.shape)}"

    add("제자리::as_strided_", as_strided_in_place)

    add("as_strided_scatter",
        lambda L: L.as_strided_scatter(L.tensor(grid), L.zeros(2, 2),
                                       (2, 2), (1, 2), 3))

    # ── 갈아끼우기 ──────────────────────────────────────────────────────
    add("select_scatter",
        lambda L: L.select_scatter(L.tensor(grid), L.zeros(4), 0, 1))
    add("slice_scatter",
        lambda L: L.slice_scatter(L.tensor(grid), L.zeros(3, 2), 1, 1, 3))
    add("slice_scatter(step=2)",
        lambda L: L.slice_scatter(L.tensor(grid), L.zeros(3, 2), 1, 0, 4, 2))
    # **길이가 offset 을 따라 변한다.** (3,4) 에서 0·1 은 세 칸이고 -1 은 두 칸이다 —
    # 셋 다 세 칸으로 주면 torch 가 그 자리에서 거절한다.
    for off, k in ((-1, 2), (0, 3), (1, 3)):
        add(f"diagonal_scatter(offset={off})",
            lambda L, o=off, n=k: L.diagonal_scatter(L.tensor(grid),
                                                     L.zeros(n), o))
    grad("select_scatter", lambda L, x: L.select_scatter(x, L.zeros(4), 0, 1))
    grad("slice_scatter",
         lambda L, x: L.slice_scatter(x, L.zeros(3, 2), 1, 0, 4, 2))
    grad("diagonal_scatter",
         lambda L, x: L.diagonal_scatter(x, L.zeros(3), 1))

    def scatter_src_grad(name, fn, src, w=weight):
        """넣은 값 쪽 기울기. **넣은 자리로만** 흘러야 한다."""
        def run(L, f=fn, s=src, n=name):
            v = L.tensor(np.asarray(s, dtype=np.float32), requires_grad=True)
            out = f(L, L.tensor(grid), v)
            (out * L.tensor(np.asarray(w, dtype=np.float32).reshape(
                tuple(out.shape)))).sum().backward()
            return _grad_of(v, n)

        cases.append((SPOT_PREFIX + f"grad(넣는 값)::{name}", run))

    scatter_src_grad("select_scatter",
                     lambda L, t, v: L.select_scatter(t, v, 0, 1), np.ones(4))
    scatter_src_grad("diagonal_scatter",
                     lambda L, t, v: L.diagonal_scatter(t, v, 1), np.ones(3))
    scatter_src_grad("as_strided_scatter",
                     lambda L, t, v: L.as_strided_scatter(t, v, (2, 2),
                                                          (1, 2), 3),
                     np.ones((2, 2)))

    # ── diag_embed ─────────────────────────────────────────────────────
    for off in (-1, 0, 1):
        add(f"diag_embed(1차, offset={off})",
            lambda L, o=off: L.diag_embed(L.tensor(trio), o))
    # **배치 축이 있어야** 대각선 축이 맨 뒤로 가는 규약이 드러난다.
    add("diag_embed(2차)", lambda L: L.diag_embed(L.tensor(grid)))
    add("diag_embed(dim1=0, dim2=1)",
        lambda L: L.diag_embed(L.tensor(grid), 0, 0, 1))
    grad("diag_embed", lambda L, x: L.diag_embed(x),
         np.arange(1, 49, dtype=np.float32).reshape(3, 4, 4))

    # ── 쪼개기 ──────────────────────────────────────────────────────────
    for k in (3, 4, 5):
        # 10 을 4 로 쪼개면 3·3·2·2 다 — 나머지를 **앞에서부터** 나눠 갖는다.
        add(f"tensor_split({k})",
            lambda L, n=k: L.cat(list(L.tensor_split(L.tensor(line), n))))
        # **조각 크기 자체를 묻는다.** 이어 붙이면 나머지를 어떻게 나눴는지가
        # 사라진다 — 3·3·2·2 든 2·2·3·3 이든 이어 붙인 값은 같다.
        add(f"tensor_split({k}, 조각 크기)",
            lambda L, n=k: L.tensor(np.asarray(
                [float(p.shape[0]) for p in L.tensor_split(L.tensor(line), n)],
                dtype=np.float32)))
    add("tensor_split(자리 목록)",
        lambda L: L.cat(list(L.tensor_split(L.tensor(line), (2, 5)))))
    add("tensor_split(dim=1)",
        lambda L: L.tensor_split(L.tensor(grid), 3, dim=1)[1])
    add("split_with_sizes",
        lambda L: L.split_with_sizes(L.tensor(line), [2, 3, 5])[1])
    # (3,4) 를 3 으로 쪼개면 2·1·1 이라 가운데 조각이 (3,1) 이다.
    grad("tensor_split", lambda L, x: L.tensor_split(x, 3, dim=1)[1],
         np.array([[1.0], [3.0], [5.0]], dtype=np.float32))

    # ── 번호 풀기·이어진 중복 ────────────────────────────────────────────
    add("unravel_index",
        lambda L: L.cat(L.unravel_index(
            L.tensor(np.array([0, 5, 11], dtype=np.int64)), (3, 4))))
    runs = np.array([1, 1, 2, 2, 2, 1, 3], dtype=np.int64)
    add("unique_consecutive",
        lambda L: L.unique_consecutive(L.tensor(runs)))
    add("unique_consecutive(inverse)",
        lambda L: L.unique_consecutive(L.tensor(runs), return_inverse=True)[1])
    add("unique_consecutive(counts)",
        lambda L: L.unique_consecutive(L.tensor(runs), return_counts=True)[1])
    rows = np.array([[1, 1], [1, 1], [1, 2], [3, 3]], dtype=np.int64)
    add("unique_consecutive(dim=0)",
        lambda L: L.unique_consecutive(L.tensor(rows), dim=0))
    add("unique_consecutive(dim=0, counts)",
        lambda L: L.unique_consecutive(L.tensor(rows), return_counts=True,
                                       dim=0)[1])

    # ── 가면·평평한 넣기 ────────────────────────────────────────────────
    mask = np.array([[True, False, True, False],
                     [False, True, False, True],
                     [True, True, False, False]])
    feed = np.arange(100, 112, dtype=np.float32)
    add("masked_scatter",
        lambda L: L.masked_scatter(L.tensor(grid), L.tensor(mask),
                                   L.tensor(feed)))
    grad("masked_scatter",
         lambda L, x: L.masked_scatter(x, L.tensor(mask), L.tensor(feed)))
    scatter_src_grad("masked_scatter",
                     lambda L, t, v: L.masked_scatter(t, L.tensor(mask), v),
                     feed)

    def masked_scatter_in_place(L):
        """**메서드로만 있다** — `torch.masked_scatter_` 라는 최상위 이름은 없다."""
        x = L.tensor(grid.copy())
        got = x.masked_scatter_(L.tensor(mask), L.tensor(feed))
        return f"{got is x} {float(x[0, 0].item())}"

    add("제자리::masked_scatter_", masked_scatter_in_place)

    # **번호가 겹친다** — 0 이 두 번 나온다. 여기서만 두 갈래가 갈린다.
    flat_idx = np.array([0, 0, 5], dtype=np.int64)
    flat_val = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    for acc in (False, True):
        add(f"put(accumulate={acc})",
            lambda L, a=acc: L.put(L.tensor(grid), L.tensor(flat_idx),
                                   L.tensor(flat_val), a))
    grad("put", lambda L, x: L.put(x, L.tensor(flat_idx), L.tensor(flat_val)))

    rowsi = np.array([0, 1, 0], dtype=np.int64)
    colsi = np.array([1, 2, 1], dtype=np.int64)
    vals = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    for acc in (False, True):
        add(f"index_put(accumulate={acc})",
            lambda L, a=acc: L.index_put(
                L.tensor(grid), (L.tensor(rowsi), L.tensor(colsi)),
                L.tensor(vals), a))
    grad("index_put",
         lambda L, x: L.index_put(x, (L.tensor(rowsi), L.tensor(colsi)),
                                  L.tensor(vals)))

    def index_put_in_place(L):
        x = L.tensor(grid.copy())
        got = L.index_put_(x, (L.tensor(rowsi), L.tensor(colsi)), L.tensor(vals))
        return f"{got is x} {float(x[0, 1].item())}"

    add("제자리::index_put_", index_put_in_place)

    # ── 줄이며 넣기 ─────────────────────────────────────────────────────
    #
    # **밑판이 2.5 다.** 1 이면 곱하기에서 항등원이라 `include_self` 가 안 보이고,
    # 0 이면 더하기에서 같은 일이 난다.
    base = np.full((3, 4), 2.5, dtype=np.float32)
    lines = np.array([0, 0, 2], dtype=np.int64)
    # `index_reduce` 에 `sum` 은 없다 — 그 자리는 `index_add` 다(실측).
    for red in ("prod", "mean", "amax", "amin"):
        for inc in (True, False):
            add(f"index_reduce({red}, include_self={inc})",
                lambda L, r=red, s=inc: L.index_reduce(
                    L.tensor(base), 0, L.tensor(lines), L.tensor(grid), r,
                    include_self=s))
    dup = np.array([[0, 0, 1, 2], [1, 1, 2, 3], [2, 2, 3, 0]], dtype=np.int64)
    for red in ("sum", "prod", "amax", "amin", "mean"):
        for inc in (True, False):
            add(f"scatter_reduce({red}, include_self={inc})",
                lambda L, r=red, s=inc: L.scatter_reduce(
                    L.tensor(base), 1, L.tensor(dup), L.tensor(grid), r,
                    include_self=s))

    # ── renorm ─────────────────────────────────────────────────────────
    #
    # 첫 줄은 이미 작아서 **안 건드려야** 한다. 나머지 둘은 깎인다.
    tall = np.array([[3.0, 4.0], [6.0, 8.0], [30.0, 40.0]], dtype=np.float32)
    for p in (1, 2, 3):
        add(f"renorm(p={p})",
            lambda L, q=p: L.renorm(L.tensor(tall), q, 0, 5.0))
    add("renorm(dim=1)", lambda L: L.renorm(L.tensor(tall), 2, 1, 5.0))
    # **깎인 줄의 기울기.** 배율 안에 x 가 있어서 `g·s` 로 적으면 여기서 갈린다.
    grad("renorm", lambda L, x: L.renorm(x, 2, 0, 5.0))

    # ── 조합·만들기 ─────────────────────────────────────────────────────
    add("cartesian_prod(둘)",
        lambda L: L.cartesian_prod(L.tensor(trio), L.tensor(duo)))
    add("cartesian_prod(하나)", lambda L: L.cartesian_prod(L.tensor(trio)))
    add("cartesian_prod(셋)",
        lambda L: L.cartesian_prod(L.tensor(trio), L.tensor(duo),
                                   L.tensor(duo)))
    for r in (1, 2, 3):
        add(f"combinations(r={r})",
            lambda L, k=r: L.combinations(L.tensor(trio), k))
    add("combinations(중복 허용)",
        lambda L: L.combinations(L.tensor(trio), 2, True))
    for off in (-1, 0, 1):
        add(f"tril_indices(offset={off})",
            lambda L, o=off: L.tril_indices(3, 4, o))
        add(f"triu_indices(offset={off})",
            lambda L, o=off: L.triu_indices(3, 4, o))
    # **음수가 섞인 입력이다.** WGSL 의 `pow` 는 밑이 음수면 답이 없어서, 거듭제곱으로
    # 짜면 여기서 NaN 이 된다 — 양수로만 재면 그 갈래가 안 보인다.
    add("vander", lambda L: L.vander(L.tensor(trio)))
    add("vander(N=2)", lambda L: L.vander(L.tensor(trio), 2))
    add("vander(increasing)", lambda L: L.vander(L.tensor(trio), None, True))
    add("vander(N=5)", lambda L: L.vander(L.tensor(trio), 5))

    # ── 행렬 ────────────────────────────────────────────────────────────
    m1 = np.arange(6, dtype=np.float32).reshape(2, 3)
    m2 = np.arange(12, dtype=np.float32).reshape(3, 4)
    m3 = np.arange(8, dtype=np.float32).reshape(4, 2)
    add("chain_matmul",
        lambda L: L.chain_matmul(L.tensor(m1), L.tensor(m2), L.tensor(m3)))
    add("ger", lambda L: L.ger(L.tensor(trio), L.tensor(duo)))
    add("mv", lambda L: L.mv(L.tensor(grid),
                             L.tensor(np.array([1., 0., 0., 2.],
                                               dtype=np.float32))))
    # `mv` 는 1차원이 낀 행렬곱이다 — 그 역방향이 코어에서 축 하나를 놓치고 있었다.
    grad("mv", lambda L, x: L.mv(x, L.tensor(
        np.array([1., 0., 0., 2.], dtype=np.float32))),
        np.array([1.0, 2.0, 0.5], dtype=np.float32))
    return cases


CPLX_PREFIX = "cplx::"
FFT_PREFIX = "fft::"

# **코어만 보는 케이스.** `WEBGPU_PREFIX` 와 정확히 반대 방향이다.
#
# 복소수가 셋 다에 생기면서 한 번 비었고, **`fft` 가 그 자리를 도로 채웠다** —
# 예상한 대로다. 코어 → borch.ts → 결속 순으로 좁혀지는 동안 이 목록이 "어디까지
# 왔는가" 를 수로 보여 준다. 비면 `startswith(())` 가 언제나 거짓이라 아무것도
# 안 건너뛴다.
CORE_ONLY_PREFIXES = (FFT_PREFIX,)


def complex_cases(inp=None):
    """복소수 — **1 단계는 코어(numpy)만이다.**

    자매(borch.ts)에는 아직 저장이 없어서 이 케이스들은 지금 브라우저에서 안 돈다.
    그래서 `webgpu::` 와 반대 방향의 자리다 — 이쪽은 **코어만 보는** 표이고,
    borch.ts 가 인터리브 저장을 갖추면 그때 같이 초록이 된다.

    ## 규약이 실측으로 못 박혀 있다

    **torch 는 복소 손실에 `backward()` 를 거절한다**(실측). 손실이 늘 실수라면
    Wirtinger 규약이 이것으로 정리된다 —

        z.grad = ∂L/∂re + i·∂L/∂im

    실측이 그것을 받친다(z = 1+2j): `z.real → 1+0j`, **`z.imag → 0+1j`**(−1j 가
    아니다), `|z|² → 2+4j`, `(z·z̄).real → 2+4j`.

    ## 켤레가 붙는 자리와 안 붙는 자리

    이 규약에서 **정칙 함수의 역방향은 `conj(f'(z))·g`** 다. 곱셈·나눗셈이 그 자리이고,
    실수에서는 켤레가 항등이라 **실수 입력으로는 있는지 없는지 알 수 없다.**

    반대로 `abs` 는 실수를 내므로 정칙이 아니고 켤레가 **안** 붙는다 — `z/|z|` 다.
    `conj` 자신은 `conj(g)` 다. 셋을 한 표에서 물어야 어느 규칙이 어디에 붙는지가 갈린다.

    ## 기울기를 실수 잎에서 받는다

    복소수 잎을 직접 만들지 않고 `complex(re, im)` 으로 엮은 뒤 **실수 잎 둘의
    기울기**를 본다. 그것이 규약의 반대 방향이고, 값이 `(∂L/∂re, ∂L/∂im)` 로 나뉘어
    나와서 **어느 쪽이 틀렸는지가 보인다** — 복소수 하나로 받으면 둘이 섞인다.
    """
    re = np.array([1.0, -3.0], dtype=np.float32)
    im = np.array([2.0, 0.5], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((CPLX_PREFIX + name, fn))

    def z(L):
        return L.complex(L.tensor(re), L.tensor(im))

    def grad(name, fn):
        """`re`·`im` 두 잎의 기울기를 이어 붙여 하나로 굳힌다."""
        def run(L, f=fn, n=name):
            r = L.tensor(re.copy(), requires_grad=True)
            i = L.tensor(im.copy(), requires_grad=True)
            f(L, L.complex(r, i)).sum().backward()
            return L.cat([_grad_of(r, n), _grad_of(i, n)])

        cases.append((CPLX_PREFIX + f"grad::{name}", run))

    # ── 만들기·꺼내기 ───────────────────────────────────────────────────
    add("complex(re, im)", lambda L: L.view_as_real(z(L)))
    add("complex 의 형", lambda L: str(z(L).dtype))
    add("polar",
        lambda L: L.view_as_real(L.polar(
            L.tensor(np.array([1.0, 2.0], dtype=np.float32)),
            L.tensor(np.array([0.0, 1.5708], dtype=np.float32)))))
    add("view_as_complex 왕복",
        lambda L: L.view_as_real(L.view_as_complex(L.view_as_real(z(L)))))
    add("real", lambda L: L.real(z(L)))
    add("imag", lambda L: L.imag(z(L)))
    # **`conj_physical` 로 묻는다.** torch 의 `conj` 는 **게으르다** — 켤레 비트만
    # 세워 두고 값을 안 뒤집어서, `view_as_real` 이 "풀지 않은 켤레" 라며 거절한다
    # (실측). 우리 것은 즉시 뒤집으므로 그 상태가 없다. 그 갈림은 README 의
    # "torch 와 갈리는 자리" 에 적었고, 여기서는 **값**을 물으려는 것이므로 둘 다
    # 값을 내는 이름으로 묻는다.
    add("conj_physical", lambda L: L.view_as_real(L.conj_physical(z(L))))
    add("angle", lambda L: L.angle(z(L)))
    add("abs", lambda L: z(L).abs())
    add("abs 의 형", lambda L: str(z(L).abs().dtype))
    add("is_complex", lambda L: str(L.is_complex(z(L))))

    # ── 산술 ────────────────────────────────────────────────────────────
    add("z * z", lambda L: L.view_as_real(z(L) * z(L)))
    add("z + z", lambda L: L.view_as_real(z(L) + z(L)))
    add("z / z", lambda L: L.view_as_real(z(L) / z(L)))
    add("z * 실수", lambda L: L.view_as_real(z(L) * L.tensor(re)))
    for other, tag in ((np.float32, "float32"), (np.int64, "int64")):
        add(f"complex64 + {tag} 의 형",
            lambda L, k=other: str(
                (z(L) + L.tensor(np.array([1], dtype=k))).dtype))

    # ── 기울기 — **켤레가 붙는 자리와 안 붙는 자리** ────────────────────
    grad("z.real", lambda L, w: L.real(w))
    # **`0+1j` 다.** `−1j` 로 적으면 부호만 뒤집힌 채 그럴듯하게 돈다.
    grad("z.imag", lambda L, w: L.imag(w))
    grad("abs(z)", lambda L, w: w.abs())
    grad("abs(z) 제곱", lambda L, w: w.abs() * w.abs())
    # 곱셈·나눗셈이 **켤레가 붙는** 자리다.
    grad("(z*z).real", lambda L, w: L.real(w * w))
    grad("(z*conj(z)).real", lambda L, w: L.real(w * L.conj_physical(w)))
    grad("view_as_real 합", lambda L, w: L.view_as_real(w))

    # ── 찍기 ────────────────────────────────────────────────────────────
    #
    # **글자가 명세다.** 복소수의 `repr` 은 실수의 것을 조금 고친 게 아니라 규칙이
    # 하나 더 있다 — **실수부와 허수부를 따로 잰다.** `[1+2j, -0.5-1j]` 에서 실수부는
    # 소수 네 자리를 요구하고 허수부는 정수라, torch 가 `1.0000+2.j` 를 찍는다.
    # 한 형식으로 재면 `1.0000+2.0000j` 가 되는데, 값은 전부 맞는 채로 글자만 갈린다.
    #
    # 세 줄로 그 규칙을 가른다: 실수부만 소수인 것, 허수부만 소수인 것, 그리고 형
    # 이름이 붙는 자리(빈 텐서에는 `j` 라는 단서가 없어서 torch 가 형을 찍는다).
    def shown(fn):
        return lambda L, f=fn: repr(f(L))

    def cx(L, values):
        return L.tensor(np.array(values, dtype=np.complex64))

    add("repr::실수부만 소수", shown(lambda L: cx(L, [1 + 2j, -0.5 - 1j])))
    add("repr::허수부만 소수", shown(lambda L: cx(L, [1 + 2j, -3 + 0.5j])))
    add("repr::둘 다 정수", shown(lambda L: cx(L, [1 + 2j, -3 - 1j])))
    add("repr::2 차원", shown(lambda L: cx(L, [[1 + 2j, -0.5 - 1j],
                                              [3 + 0j, 0 + 4j]])))
    # **음의 0 은 부호가 산다** — 허수부를 절댓값으로 찍으면 여기서만 갈린다.
    add("repr::음의 0 허수부", shown(lambda L: cx(L, [complex(1.0, -0.0)])))
    # **부호를 옮기는 길 자체를 묻는다** — repr 을 안 거친다.
    #
    # 위의 `repr::음의 0 허수부` 가 이 결함을 처음 잡았는데, 그 케이스가 빨개지면
    # 화면에는 "글자가 다르다" 로 보인다. 원인은 두 칸 떨어져 있었다 — 결속의
    # `_read` 가 `np.asarray(JsProxy, dtype=float32)` 로 값을 받으면서 **음의 0 을
    # 0 으로 만들고** 있었다(JS 쪽에서는 `Object.is(x, -0)` 이 참인 채로 왔다).
    #
    # **값 대조로는 영영 못 잡는다** — `-0.0 == 0.0` 이기 때문이다. 그래서 부호
    # 비트를 답으로 굳힌다. 회귀하면 "변환이 부호를 잃었다" 로 곧장 읽힌다.
    #
    # 복소수가 아니라 **실수** 텐서로 묻는다. 그 결함은 복소수가 생기기 전부터 거기
    # 있었고 복소수 repr 이 우연히 드러냈을 뿐이라, 실수로 물어야 자리가 맞는다.
    # 답을 **글자로** 낸다. 하네스는 답에 `.detach()` 를 부르므로 배열을 그냥 주면
    # 굳히는 자리에서 멈추고, 부호는 어차피 두 값뿐이라 글자가 더 읽힌다.
    def signbits(L):
        bits = np.signbit(to_numpy(L.tensor([-0.0, 0.0, -1.5, 1.5])))
        return "".join("-" if b else "+" for b in bits)

    add("read::음의 0 이 변환을 건넌다", signbits)

    # 빈 것에는 `j` 가 없어서 형 이름이 붙는다(실측). 값이 있으면 안 붙는다.
    add("repr::빈 것", shown(lambda L: cx(L, [])))
    # 한 줄에 몇 개가 들어가는지도 글자다. torch 는 글자 수가 아니라 **폭**으로 센다.
    add("repr::줄바꿈",
        shown(lambda L: cx(L, [complex(k, 0.5) for k in range(12)])))
    add("repr::grad_fn 이 붙는다",
        shown(lambda L: L.complex(L.tensor(re, requires_grad=True),
                                  L.tensor(im, requires_grad=True))))
    # **실수 쪽에서 같이 드러난 자리.** 정수 판의 서식이 `nan` 에도 점을 붙여서
    # `tensor([nan., 1.])` 이 나왔다 — 소수 판은 `f"{nan:.4f}"` 가 이미 `nan` 이라
    # 안 갈렸고, 그래서 nan 이 낀 **정수** 텐서에서만 났다.
    add("repr::nan 낀 정수 텐서",
        shown(lambda L: L.tensor(np.array([float("nan"), 1.0], dtype=np.float32))))

    # ── 거절 ────────────────────────────────────────────────────────────
    def refuses_complex_loss(L):
        """**복소 손실은 거절해야 한다.** 위 규약 전체가 이것 위에 서 있다.

        손실이 실수라는 것이 `z.grad = ∂L/∂re + i·∂L/∂im` 의 전제다. 거절을 안 하면
        규약이 정의되지 않은 자리에 그럴듯한 숫자가 들어가고, 값 케이스는 전부
        초록인 채로 남는다 — **전제를 케이스로 묻지 않으면 전제가 아니라 희망이다.**
        """
        r = L.tensor(re.copy(), requires_grad=True)
        i = L.tensor(im.copy(), requires_grad=True)
        try:
            (L.complex(r, i) * L.complex(r, i)).sum().backward()
            return "예외가 안 났다"
        except Exception as exc:                                    # noqa: BLE001
            return type(exc).__name__

    add("복소 손실의 backward 는 거절", refuses_complex_loss)
    return cases


def fft_cases(inp=None):
    """푸리에 변환 — `torch.fft` 와 `stft`.

    **복소수 위에 선다.** 이 이름들은 오래 거절이었고 거절문에 "복소수 규약을 안
    정했다" 고 적혀 있었다. 그 이유가 정확했기 때문에 규약이 정해진 날 문이 열렸다 —
    "저장이 없다" 로 적어 두었으면 저장이 생긴 뒤에도 아무도 다시 안 물었을 것이다.

    ## 값보다 기울기가 요점이다

    변환은 선형이라 순방향은 맞히기 쉽다. 어려운 자리는 **어느 쪽 반쪽을 세는가** 다 —

    * `rfft` 의 역방향은 저장된 반쪽에만 기울기가 온다. 켤레 짝을 더하면 두 배가 된다.
    * `irfft` 의 역방향은 **가장자리만 한 번, 가운데는 두 번** 세야 한다. 되살린
      켤레 짝이 같은 저장 칸에서 왔기 때문이다.

    둘은 서로 반대 방향의 실수이고, **둘 다 순방향 값은 멀쩡하다.**

    ## `abs` 의 칼날을 피한 입력

    `stft(…).abs()` 케이스의 신호가 고르지 않은 수인 데는 이유가 있다. 경사 신호
    (`arange/8 − 1`)는 나이퀴스트 칸이 **정확히 0** 이 되는데, 거기서 `abs` 는
    미분 불가능하고 부호가 구현의 반올림에 달린다 — 우리는 float64 로 누산해서
    +1 을, torch 는 float32 FFT 라 0 을 골랐다. **규칙이 갈린 것이 아니라 케이스가
    칼날 위에 서 있던 것**이고, 그런 케이스를 굳히면 골든이 부동소수 우연을 명세로
    박제한다. 0 인 칸이 없는 신호로 바꾸니 열여섯 자리가 전부 맞았다.
    """
    sig = np.array([0.3, -1.2, 0.7, 2.1, -0.4, 1.5, -2.3, 0.9,
                    1.1, -0.6, 0.25, -1.7, 2.4, 0.05, -0.8, 1.35],
                   dtype=np.float32)
    xs = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 0.25], dtype=np.float32)
    ys = np.array([0.5, 1.0, -1.5, 0.25, 2.0, -0.5], dtype=np.float32)
    mat = np.arange(12, dtype=np.float32).reshape(3, 4)
    cases = []

    def add(name, fn):
        cases.append((FFT_PREFIX + name, fn))

    def x(L):
        return L.tensor(xs.copy())

    def z(L):
        return L.complex(L.tensor(xs.copy()), L.tensor(ys.copy()))

    def pair(fn):
        """복소수 답은 실수 짝으로 묻는다 — 골든 파일이 실수만 담는다."""
        return lambda L, f=fn: L.view_as_real(f(L))

    # ── 값 ──────────────────────────────────────────────────────────────
    add("fft(실수)", pair(lambda L: L.fft.fft(x(L))))
    add("fft(복소)", pair(lambda L: L.fft.fft(z(L))))
    add("fft 의 형", lambda L: str(L.fft.fft(x(L)).dtype))
    add("ifft(fft)", pair(lambda L: L.fft.ifft(L.fft.fft(x(L)))))
    add("ifft(복소)", pair(lambda L: L.fft.ifft(z(L))))
    add("rfft", pair(lambda L: L.fft.rfft(x(L))))
    add("irfft(rfft)", lambda L: L.fft.irfft(L.fft.rfft(x(L))))
    add("irfft 의 형", lambda L: str(L.fft.irfft(L.fft.rfft(x(L))).dtype))
    # **홀수 길이를 따로 묻는다.** `irfft` 는 n 을 안 주면 `2*(m-1)` 이라 짝수만
    # 나온다 — 홀수는 n 을 줘야만 나오고, 되살리는 켤레 짝의 개수가 거기서 갈린다.
    add("irfft(n=5)", lambda L: L.fft.irfft(L.fft.rfft(x(L)), n=5))
    add("irfft(n=7)", lambda L: L.fft.irfft(L.fft.rfft(x(L)), n=7))
    for norm in ("forward", "backward", "ortho"):
        add(f"fft norm={norm}", pair(lambda L, m=norm: L.fft.fft(x(L), norm=m)))
        add(f"ifft norm={norm}", pair(lambda L, m=norm: L.fft.ifft(z(L), norm=m)))
    for n in (4, 8):
        add(f"fft(n={n})", pair(lambda L, k=n: L.fft.fft(x(L), n=k)))
        add(f"rfft(n={n})", pair(lambda L, k=n: L.fft.rfft(x(L), n=k)))
    add("fft(dim=0)", pair(lambda L: L.fft.fft(L.tensor(mat.copy()), dim=0)))
    add("rfft(dim=0)", pair(lambda L: L.fft.rfft(L.tensor(mat.copy()), dim=0)))

    for n in (5, 6):
        add(f"fftfreq({n})", lambda L, k=n: L.fft.fftfreq(k))
        add(f"rfftfreq({n})", lambda L, k=n: L.fft.rfftfreq(k))
        # **홀수에서 갈린다.** `fftshift` 는 `n//2` 만큼 미는데 되돌리려면
        # `(n+1)//2` 여야 한다 — 짝수만 물으면 둘이 같아서 안 보인다.
        add(f"fftshift({n})", lambda L, k=n: L.fft.fftshift(L.fft.fftfreq(k)))
        add(f"ifftshift(fftshift({n}))",
            lambda L, k=n: L.fft.ifftshift(L.fft.fftshift(L.fft.fftfreq(k))))
    add("fftfreq(6, d=0.5)", lambda L: L.fft.fftfreq(6, 0.5))

    # ── 기울기 ──────────────────────────────────────────────────────────
    def grad(name, fn):
        def run(L, f=fn, n=name):
            leaf = L.tensor(xs.copy(), requires_grad=True)
            f(L, leaf).sum().backward()
            return _grad_of(leaf, n)

        add(f"grad::{name}", run)

    grad("fft 실수부", lambda L, t: L.real(L.fft.fft(t)))
    grad("fft 크기", lambda L, t: L.fft.fft(t).abs())
    # **켤레 짝을 더하면 두 배가 된다.** 실측이 `[4, 0, 1, 0, 1, 0]` 이다.
    grad("rfft 실수부", lambda L, t: L.real(L.fft.rfft(t)))
    grad("rfft 허수부", lambda L, t: L.imag(L.fft.rfft(t)))
    grad("rfft 크기", lambda L, t: L.fft.rfft(t).abs())
    # **가장자리를 두 번 세면 여기서 갈린다.**
    grad("irfft(rfft)", lambda L, t: L.fft.irfft(L.fft.rfft(t)))
    grad("irfft 가중", lambda L, t: L.fft.irfft(L.fft.rfft(t))
         * L.tensor(np.arange(6, dtype=np.float32)))
    grad("ifft(fft) 실수부", lambda L, t: L.real(L.fft.ifft(L.fft.fft(t))))
    grad("fftshift(rfft) 크기",
         lambda L, t: L.fft.fftshift(L.fft.rfft(t)).abs())

    # ── stft ────────────────────────────────────────────────────────────
    def s(L):
        return L.tensor(sig.copy())

    def hann(L, n=8):
        return L.hann_window(n)

    for center in (True, False):
        for hop in (2, 4):
            add(f"stft center={center} hop={hop}",
                pair(lambda L, c=center, h=hop: L.stft(
                    s(L), 8, h, window=hann(L), center=c, return_complex=True)))
    add("stft 기본 hop",
        pair(lambda L: L.stft(s(L), 8, window=hann(L), return_complex=True)))
    add("stft 창 없이", pair(lambda L: L.stft(s(L), 8, 4, return_complex=True)))
    # 창이 짧으면 **가운데에 놓고 양쪽을 0 으로 채운다**(실측). 왼쪽 정렬이면 갈린다.
    add("stft win_length=6",
        pair(lambda L: L.stft(s(L), 8, 4, 6, hann(L, 6), return_complex=True)))
    add("stft onesided=False",
        pair(lambda L: L.stft(s(L), 8, 4, window=hann(L), onesided=False,
                              return_complex=True)))
    add("stft normalized",
        pair(lambda L: L.stft(s(L), 8, 4, window=hann(L), normalized=True,
                              return_complex=True)))
    for mode in ("reflect", "constant", "replicate"):
        add(f"stft pad_mode={mode}",
            pair(lambda L, m=mode: L.stft(
                L.tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
                4, 2, window=L.ones([4]), pad_mode=m, return_complex=True)))
    add("stft 배치",
        pair(lambda L: L.stft(L.tensor(sig.reshape(1, 16).copy()), 8, 4,
                              window=hann(L), return_complex=True)))
    add("istft(length=16)",
        lambda L: L.istft(L.stft(s(L), 8, 4, window=hann(L),
                                 return_complex=True),
                          8, 4, window=hann(L), length=16))
    add("istft 길이 없이",
        lambda L: L.istft(L.stft(s(L), 8, 4, window=hann(L),
                                 return_complex=True),
                          8, 4, window=hann(L)))

    def sgrad(name, fn):
        def run(L, f=fn, n=name):
            leaf = L.tensor(sig.copy(), requires_grad=True)
            f(L, leaf).sum().backward()
            return _grad_of(leaf, n)

        add(f"grad::{name}", run)

    sgrad("stft 크기",
          lambda L, t: L.stft(t, 8, 4, window=hann(L),
                              return_complex=True).abs())
    sgrad("stft center=False 크기",
          lambda L, t: L.stft(t, 8, 4, window=hann(L), center=False,
                              return_complex=True).abs())
    sgrad("istft(stft)",
          lambda L, t: L.istft(L.stft(t, 8, 4, window=hann(L),
                                      return_complex=True),
                               8, 4, window=hann(L), length=16))

    # ── 거절 ────────────────────────────────────────────────────────────
    def refuses(name, body):
        def run(L, f=body):
            try:
                f(L)
                return "예외가 안 났다"
            except Exception as exc:                            # noqa: BLE001
                return type(exc).__name__

        add(name, run)

    refuses("rfft(복소)는 거절", lambda L: L.fft.rfft(z(L)))
    # **`return_complex` 를 안 주면 torch 가 멈춘다**(실측). 기본값을 정해 주면
    # 곧 폐기될 모양(실수 `(…, 2)`)을 가르치게 된다.
    refuses("stft 는 return_complex 를 요구",
            lambda L: L.stft(s(L), 8, 4, window=hann(L)))
    refuses("복소 스펙트럼의 backward 는 거절",
            lambda L: L.fft.fft(
                L.tensor(xs.copy(), requires_grad=True)).sum().backward())
    return cases


MAKE_PREFIX = "make::"


def make_cases(inp=None):
    """**복소수가 없어도 답이 있는 이름들**, 그리고 생성 몇.

    ## 답할 수 있는 것과 없는 것은 다르다

    복소수 규약을 안 정했다고 `is_complex` 까지 없으면, 그것을 분기에 쓰는 교재 코드가
    `AttributeError` 로 멈춘다. 실수 텐서에서 이 이름들은 전부 답이 있다 —
    `real`·`conj`·`resolve_conj` 는 **항등**이고, 판정 셋은 **전부 거짓**이며,
    `angle` 은 음수에서 π 다.

    `imag` 만 거절인데, 그것은 **torch 자신이 실수에서 거절하기 때문**이다(실측:
    "imag is not implemented for tensors with non-complex dtypes"). 우리 한계가
    아니라 torch 를 그대로 옮긴 것이다.

    ## 무엇을 물어야 갈리는가

    - **형을 셋으로 묻는다.** `real(bool)` 은 `bool` 이다(실측). 항등을 `positive` 의
      단항 커널로 보내면 형이 float32 로 떨어지는데, float32 입력으로만 재면 그것이
      안 보인다 — dtype 이름표 건에서 겪은 자리와 같다.
    - **`angle` 은 그 반대다.** 정수를 넣어도 **float32** 가 나온다(실측). 각도는 정수
      칸에 안 들어가므로 그것이 맞고, 실수만 넣으면 규칙이 안 드러난다.
    - **`asarray` 는 안 베끼는 것이 기본이다.** `copy=True` 여야 사본이다(실측).
    - **`frombuffer` 의 `offset` 은 바이트다** — 원소 수로 읽으면 값이 밀린다.
    - **`range` 는 끝을 포함한다.** `arange` 는 뺀다 — `range(0, 4)` 가 다섯 개다.
      조용히 `arange` 로 넘기면 원소가 하나 모자라고, 그것이 torch 가 이 이름을
      폐기하는 사유이기도 하다.

    ## 둘은 거절한다

    `empty_strided`·`empty_permuted` 는 **걸음(stride) 자체가 유일한 답**인데(값은
    쓰레기다) 우리 텐서에 걸음이라는 것이 없다. `as_strided` 와 다른 자리다 — 그쪽은
    값이 답이라 사본으로도 같은 답을 낸다.
    """
    plain = np.array([[-1.5, 0.0, 2.0], [3.0, -4.0, 0.5]], dtype=np.float32)
    ints = np.array([1, -2, 3], dtype=np.int64)
    flags = np.array([True, False, True])
    kinds = ((plain, "float32"), (ints, "int64"), (flags, "bool"))
    cases = []

    def add(name, fn):
        cases.append((MAKE_PREFIX + name, fn))

    # ── 항등 다섯 — **형까지 지켜야 한다** ──────────────────────────────
    for name in ("real", "conj", "conj_physical", "resolve_conj", "resolve_neg"):
        for src, tag in kinds:
            add(f"{name}({tag})",
                lambda L, n=name, s=src: getattr(L, n)(L.tensor(s)))
            add(f"{name}({tag}) 형",
                lambda L, n=name, s=src: str(getattr(L, n)(L.tensor(s)).dtype))

    # ── angle — 형이 **언제나 float32** ────────────────────────────────
    for src, tag in kinds:
        add(f"angle({tag})", lambda L, s=src: L.angle(L.tensor(s)))
        add(f"angle({tag}) 형", lambda L, s=src: str(L.angle(L.tensor(s)).dtype))

    # ── 판정 셋 — 전부 거짓 ────────────────────────────────────────────
    for name in ("is_complex", "is_conj", "is_neg"):
        add(name, lambda L, n=name: " ".join(
            str(getattr(L, n)(L.tensor(s))) for s, _ in kinds))

    # ── 생성 ───────────────────────────────────────────────────────────
    add("asarray(list)", lambda L: L.asarray([1.0, 2.0]))
    add("asarray(ndarray) 형", lambda L: str(L.asarray(ints).dtype))
    raw = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    add("frombuffer",
        lambda L: L.frombuffer(bytearray(raw), dtype=L.float32))
    add("frombuffer(count=2)",
        lambda L: L.frombuffer(bytearray(raw), dtype=L.float32, count=2))
    # **`offset` 은 바이트다** — 원소 수로 읽으면 여기서 갈린다.
    add("frombuffer(offset=4)",
        lambda L: L.frombuffer(bytearray(raw), dtype=L.float32, offset=4))
    # **끝을 포함한다** — `arange` 와 한 칸 다르다.
    add("range(0, 4)", lambda L: L.range(0, 4))
    add("range(1, 7, 2)", lambda L: L.range(1, 7, 2))
    add("range(0, 1, 0.25)", lambda L: L.range(0, 1, 0.25))
    add("range 와 arange 의 개수",
        lambda L: f"{L.range(0, 4).numel()} {L.arange(0, 4).numel()}")
    return cases


STAT_PREFIX = "stat::"


def stat_cases(inp=None):
    """통계. **난수의 값은 못 굳히지만 끝값은 결정적이다.**

    ## 난수 넷을 어떻게 묻는가

    `normal`·`bernoulli`·`poisson`·`binomial` 의 값은 골든이 못 굳힌다 — torch 의 난수
    줄기와 우리 것이 다르고, 같게 만들 방법도 없다. 그래서 **결정적인 구석**을 묻는다:

    - `std=0` 이면 평균 그대로 (실측)
    - `p=0` 이면 전부 0, `p=1` 이면 전부 1
    - `poisson(0)` 은 전부 0
    - 나머지는 모양만

    "난수라 못 묻는다" 와 "안 묻는다" 는 다르다. 끝값을 안 물으면 `bernoulli` 가
    확률을 아예 안 보고 있어도 통과한다.

    ## 나머지가 갈리는 자리

    - **범위 밖을 버린다.** `histc`·`histogram` 은 `min`/`max` 밖의 값을 양끝 칸으로
      몰아넣지 않는다(실측). 전부 범위 안인 자료로 재면 그 규칙이 안 드러난다.
    - **`min == max` 면 자료의 범위를 쓴다.** 기본값이 `0, 0` 이라 그 갈래가 기본이다.
    - **마지막 칸은 오른쪽이 닫혀 있다.** 최댓값이 마지막 칸에 들어간다.
    - **`mode` 는 같은 횟수면 작은 값이 이기고, 자리는 그 값의 마지막이다**(실측:
      `[4,4,5,5]` 가 값 4 · 자리 1). 비긴 자리가 없으면 그 규칙이 안 드러난다.
    - **`nanmedian` 은 짝수 개에서 아래를 고른다** — 평균을 내지 않는다. 그리고
      `median` 은 NaN 이 하나만 있어도 NaN 을 낸다 — 둘을 나란히 물어야 갈린다.
    - **`gradient` 의 `edge_order`.** 1 이면 양끝이 한쪽 차분이고 2 면 이차식이다.
      `x²` 을 넣으면 2 에서만 정확한 도함수가 나온다.
    - **`histogram(density)` 는 칸 너비로 나눈다** — 경계를 직접 준 경우 칸마다
      너비가 달라서, 균등한 칸으로만 재면 그 나눗셈이 안 드러난다.

    ## 셋은 이름만 두고 거절한다

    `stft`·`istft` 는 **복소수 dtype 이 없다**. torch 의 기본이 이제 복소수이고, 실수
    `(…, 2)` 로 내는 길은 폐기 예정이라 그 꼴로 흉내 내면 곧 사라질 모양을 가르친다.
    `hash_tensor` 는 uint64 도 없고 어떤 해시인지 규격도 없다 — 값을 맞출 수 없는
    것에 이름만 놓으면 그 값을 믿는 코드가 생긴다.
    """
    x = np.array([0.5, 2.0, 2.0, 3.5, 1.0, 4.0, 2.0], dtype=np.float32)
    w = np.array([1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0], dtype=np.float32)
    # **비긴 자리가 있다** — 없으면 `mode` 의 규칙이 안 드러난다.
    tie = np.array([[1.0, 2.0, 2.0, 3.0], [4.0, 4.0, 5.0, 5.0]], dtype=np.float32)
    holes = np.array([[1.0, np.nan, 3.0, 5.0], [2.0, 4.0, np.nan, np.nan]],
                     dtype=np.float32)
    # `x²` 이다 — `edge_order=2` 가 정확해지는 자리.
    line = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    mat = np.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0], [64.0, 128.0, 256.0]],
                   dtype=np.float32)
    pts = np.array([[0.5, 1.0], [1.5, 1.5], [2.5, 0.5], [0.2, 2.5]],
                   dtype=np.float32)
    sparse = np.array([0.0, 3.0, 0.0, 5.0, 0.0], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((STAT_PREFIX + name, fn))

    # ── 히스토그램 ──────────────────────────────────────────────────────
    add("histc(bins=4)", lambda L: L.histc(L.tensor(x), bins=4))
    add("histc(min/max)",
        lambda L: L.histc(L.tensor(x), bins=4, min=0.0, max=4.0))
    add("histc(범위 밖은 버림)",
        lambda L: L.histc(L.tensor(x), bins=2, min=1.0, max=3.0))
    add("histogram 의 hist", lambda L: L.histogram(L.tensor(x), bins=4).hist)
    add("histogram 의 edges",
        lambda L: L.histogram(L.tensor(x), bins=4).bin_edges)
    add("histogram(weight)",
        lambda L: L.histogram(L.tensor(x), bins=4, weight=L.tensor(w)).hist)
    add("histogram(density)",
        lambda L: L.histogram(L.tensor(x), bins=4, density=True).hist)
    add("histogram(range)",
        lambda L: L.histogram(L.tensor(x), bins=4, range=(0.0, 4.0)).hist)
    # **칸 너비가 다르다** — `density` 가 칸마다 다른 값으로 나누는지 여기서만 보인다.
    add("histogram(경계를 직접)",
        lambda L: L.histogram(
            L.tensor(x),
            bins=L.tensor(np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32))).hist)
    add("histogramdd 의 hist",
        lambda L: L.histogramdd(L.tensor(pts), bins=[2, 2]).hist)
    add("histogramdd 의 edges",
        lambda L: L.cat(list(L.histogramdd(L.tensor(pts), bins=[2, 2]).bin_edges)))

    # ── mode · nanmedian ────────────────────────────────────────────────
    for dim in (0, 1):
        add(f"mode(dim={dim}) 값",
            lambda L, d=dim: L.mode(L.tensor(tie), dim=d).values)
        add(f"mode(dim={dim}) 자리",
            lambda L, d=dim: L.mode(L.tensor(tie), dim=d).indices)
    add("mode(keepdim) 모양",
        lambda L: str(tuple(L.mode(L.tensor(tie), dim=1, keepdim=True).values.shape)))
    add("nanmedian(전체)", lambda L: L.nanmedian(L.tensor(holes)))
    add("nanmedian(dim=1) 값",
        lambda L: L.nanmedian(L.tensor(holes), dim=1).values)
    add("nanmedian(dim=1) 자리",
        lambda L: L.nanmedian(L.tensor(holes), dim=1).indices)
    # **짝수 개면 아래를 고른다** — 평균을 내면 여기서 갈린다.
    add("nanmedian(짝수 개)",
        lambda L: L.nanmedian(L.tensor(np.array([1.0, 2.0, 3.0, 4.0],
                                                dtype=np.float32))))
    # `median` 은 NaN 하나에도 NaN 이다 — 나란히 둬야 `nanmedian` 이 무엇인지 보인다.
    #
    # **값이 아니라 판정을 굳힌다.** 골든의 대조는 `allclose` 인데 그것이 `equal_nan`
    # 없이 돌아서 **NaN 은 자기 자신과도 다르다** — 답이 NaN 인 케이스는 이 하네스가
    # 통째로 못 굳힌다. 그 자리를 "NaN 인가" 로 바꾸면 문자열 비교가 되고, 묻고 싶던
    # 것(둘이 다르다)은 그대로 남는다.
    add("median(NaN 이 섞이면 NaN 이다)",
        lambda L: " ".join(
            str(bool(v))
            for v in L.isnan(L.median(L.tensor(holes), dim=1).values).tolist()))

    # ── gradient · trapz ────────────────────────────────────────────────
    add("gradient(기본)", lambda L: L.cat(list(L.gradient(L.tensor(line)))))
    add("gradient(spacing=2)",
        lambda L: L.cat(list(L.gradient(L.tensor(line), spacing=2.0))))
    add("gradient(edge_order=2)",
        lambda L: L.cat(list(L.gradient(L.tensor(line), edge_order=2))))
    for axis in (0, 1):
        add(f"gradient(2차)[{axis}]",
            lambda L, a=axis: L.gradient(L.tensor(mat))[a])
    add("gradient(dim=1)", lambda L: L.gradient(L.tensor(mat), dim=1)[0])
    add("trapz(y)", lambda L: L.trapz(L.tensor(line)))
    add("trapz(dx=2)", lambda L: L.trapz(L.tensor(line), dx=2.0))
    add("trapz(y, x)",
        lambda L: L.trapz(L.tensor(line),
                          L.tensor(np.array([0.0, 1.0, 3.0, 6.0, 10.0],
                                            dtype=np.float32))))

    # ── nonzero_static ──────────────────────────────────────────────────
    #
    # **모자라면 채우고 넘치면 자른다.** 딱 맞는 크기로만 재면 두 갈래가 안 드러난다.
    for size in (1, 2, 5):
        add(f"nonzero_static(size={size})",
            lambda L, n=size: L.nonzero_static(L.tensor(sparse), size=n))
    add("nonzero_static(fill=-9)",
        lambda L: L.nonzero_static(L.tensor(sparse), size=5, fill_value=-9))

    # ── 난수 넷 — 결정적인 끝값만 ───────────────────────────────────────
    add("bernoulli(p=0)", lambda L: L.bernoulli(L.zeros(4)))
    add("bernoulli(p=1)", lambda L: L.bernoulli(L.ones(4)))
    add("poisson(0)", lambda L: L.poisson(L.zeros(4)))
    ten = np.array([10.0, 10.0], dtype=np.float32)
    add("binomial(p=0)",
        lambda L: L.binomial(L.tensor(ten),
                             L.tensor(np.zeros(2, dtype=np.float32))))
    add("binomial(p=1)",
        lambda L: L.binomial(L.tensor(ten),
                             L.tensor(np.ones(2, dtype=np.float32))))
    add("normal(std=0)",
        lambda L: L.normal(L.tensor(np.array([1.0, 100.0], dtype=np.float32)),
                           L.tensor(np.zeros(2, dtype=np.float32))))
    # 값은 못 묻지만 **모양은 묻는다** — 그것마저 안 물으면 이름만 있는 것과 같다.
    add("normal(size) 모양",
        lambda L: str(tuple(L.normal(0.0, 1.0, (2, 3)).shape)))
    add("bernoulli 모양", lambda L: str(tuple(L.bernoulli(L.zeros(2, 3)).shape)))
    return cases


TOPLIN_PREFIX = "toplin::"


def top_linalg_cases(inp=None):
    """최상위 선형대수. `linalg` 쪽과 **같은 계산인데 부르는 법이 다른** 것들.

    ## 인자 순서가 뒤집혀 있다

    torch 는 옛 이름들을 최상위에 남겨 뒀고, 그것들은 대개 **오른쪽 변을 먼저** 받는다
    — `lu_solve(b, LU, piv)` 대 `linalg.lu_solve(LU, piv, b)`. `triangular_solve` 도
    `b` 가 먼저이고 **기본 `upper` 가 참이다**(`linalg` 쪽은 필수 인자다). 자리를
    잘못 옮기면 다른 삼각을 풀고도 값이 그럴듯하게 나온다.

    ## 무엇을 물어야 갈리는가

    - **`orgqr` 과 `ormqr` 은 다른 Q 를 쓴다.** 앞은 `m×k` 로 자른 Q 이고 뒤는 자르지
      않은 `m×m` 이다 — 반사자가 `Rᵐ` 위의 사상이라 그렇다. **세로로 긴 행렬**로
      물어야 보인다. 정사각으로 재면 둘이 같다(실측으로 걸렸다).
    - **`unitriangular` 은 대각을 안 보고 1 로 친다.** 대각이 1 인 행렬로 재면 그
      깃발이 아무 일도 안 한다.
    - **`lu_unpack` 은 끄면 빈 텐서를 준다** — `None` 이 아니다(실측). 모양을 물어야
      드러난다.
    - **`lobpcg` 의 `largest` 가 순서까지 정한다** — 참이면 큰 것부터, 거짓이면 작은
      것부터다(실측). `k=1` 로만 재면 순서가 없다.
    - **`svd_lowrank` 는 정확히 저계수인 입력에서만 답이 굳는다.** torch 는 무작위로
      사영하는데, 계수가 `q` 를 넘으면 씨앗에 따라 특이값이 0.5 씩 움직인다(실측:
      씨앗 둘의 차가 0.54). 계수가 `q` 이하면 7e-7 안이다 — 골든이 물을 수 있는
      자리는 그쪽뿐이라, 입력을 `(8,3)@(3,5)` 로 **정확히 계수 3** 으로 만든다.
    - **`pca_lowrank(center=False)` 는 `svd_lowrank` 와 같은 것이다**(실측). 가운데
      맞추기가 차이 전부라 참으로만 재면 그 갈래가 안 보인다.

    ## 고유벡터는 안 묻는다

    부호가 임의다 — 같은 고유쌍인데 `-v` 가 나올 수 있고, 그것은 갈림이 아니다.
    고윳값만 굳힌다.
    """
    spd = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]],
                   dtype=np.float32)
    gen = np.array([[4.0, 3.0, 2.0], [1.0, 5.0, 3.0], [2.0, 1.0, 6.0]],
                   dtype=np.float32)
    # **대각이 1 이 아니다** — `unitriangular` 이 실제로 무엇을 하는지 보려면 그래야 한다.
    tri = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [4.0, 2.0, 5.0]],
                   dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    # **세로로 길다** — `orgqr` 과 `ormqr` 이 여기서만 갈린다.
    tall = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    side = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    weight = np.array([[1.0, 2.0], [0.5, 3.0], [2.0, 1.0]], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((TOPLIN_PREFIX + name, fn))

    def chol(L, upper):
        low = L.linalg.cholesky(L.tensor(spd))
        return low.transpose(0, 1) if upper else low

    # ── 촐레스키 ────────────────────────────────────────────────────────
    for upper in (False, True):
        add(f"cholesky_solve(upper={upper})",
            lambda L, u=upper: L.cholesky_solve(L.tensor(rhs), chol(L, u),
                                                upper=u))
        add(f"cholesky_inverse(upper={upper})",
            lambda L, u=upper: L.cholesky_inverse(chol(L, u), upper=u))

    def cholesky_solve_grad(L):
        """**인수 쪽으로도 흘러야 한다.** `b` 로만 흘리면 순방향은 맞고 여기서 갈린다."""
        x = L.tensor(spd.copy(), requires_grad=True)
        out = L.cholesky_solve(L.tensor(rhs), L.linalg.cholesky(x))
        (out * L.tensor(weight)).sum().backward()
        return _grad_of(x, "cholesky_solve")

    cases.append((TOPLIN_PREFIX + "grad::cholesky_solve", cholesky_solve_grad))

    # ── 삼각 ────────────────────────────────────────────────────────────
    for upper in (False, True):
        for trans in (False, True):
            for unit in (False, True):
                add(f"triangular_solve(u={upper},t={trans},unit={unit})",
                    lambda L, u=upper, t=trans, n=unit: L.triangular_solve(
                        L.tensor(rhs), L.tensor(tri), upper=u, transpose=t,
                        unitriangular=n).solution)
    # **둘째 자리는 계수의 사본이다** — 쓸모가 없어 보여도 torch 가 그렇게 준다.
    add("triangular_solve(둘째 자리)",
        lambda L: L.triangular_solve(L.tensor(rhs), L.tensor(tri),
                                     upper=False).cloned_coefficient)

    def triangular_solve_grad(L):
        x = L.tensor(spd.copy(), requires_grad=True)
        out = L.triangular_solve(L.tensor(rhs), L.tril(x),
                                 upper=False).solution
        (out * L.tensor(weight)).sum().backward()
        return _grad_of(x, "triangular_solve")

    cases.append((TOPLIN_PREFIX + "grad::triangular_solve",
                  triangular_solve_grad))

    # ── LU ──────────────────────────────────────────────────────────────
    add("lu 의 LU", lambda L: L.lu(L.tensor(gen))[0])
    add("lu 의 pivots", lambda L: L.lu(L.tensor(gen))[1])
    add("lu(get_infos=True) 의 info",
        lambda L: L.lu(L.tensor(gen), True, True)[2])
    add("lu_solve", lambda L: L.lu_solve(L.tensor(rhs), *L.lu(L.tensor(gen))))
    for data_flag in (True, False):
        for piv_flag in (True, False):
            for slot, name in enumerate(("P", "L", "U")):
                add(f"lu_unpack(data={data_flag}, piv={piv_flag}) 의 {name}",
                    lambda L, d=data_flag, p=piv_flag, s=slot: L.lu_unpack(
                        *L.lu(L.tensor(gen)), unpack_data=d,
                        unpack_pivots=p)[s])
                # **끄면 빈 텐서다.** 모양을 안 물으면 그것이 안 드러난다.
                add(f"lu_unpack(data={data_flag}, piv={piv_flag}) 의 {name} 모양",
                    lambda L, d=data_flag, p=piv_flag, s=slot: str(tuple(
                        L.lu_unpack(*L.lu(L.tensor(gen)), unpack_data=d,
                                    unpack_pivots=p)[s].shape)))

    # ── 반사자 ──────────────────────────────────────────────────────────
    add("orgqr", lambda L: L.orgqr(*L.geqrf(L.tensor(tall))))
    add("orgqr 은 자른 Q 다 (linalg.qr 의 Q 와 같다)",
        lambda L: L.linalg.qr(L.tensor(tall))[0])
    for left in (True, False):
        for trans in (True, False):
            add(f"ormqr(left={left}, transpose={trans})",
                lambda L, lf=left, tr=trans: L.ormqr(
                    *L.geqrf(L.tensor(tall)),
                    L.tensor(side if lf else side.T), lf, tr))

    # ── 고유쌍·저계수 ────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    spread = np.array([8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.2])
    big = (basis @ np.diag(spread) @ basis.T).astype(np.float32)
    big = ((big + big.T) / 2).astype(np.float32)
    for k in (1, 2, 3):
        for largest in (True, False):
            add(f"lobpcg(k={k}, largest={largest}) 고윳값",
                lambda L, kk=k, lg=largest: L.lobpcg(L.tensor(big), k=kk,
                                                     largest=lg)[0])
    # **정확히 계수 3 이다** — 넘치면 torch 가 씨앗에 흔들려 굳힐 수가 없다.
    low = (rng.standard_normal((8, 3))
           @ rng.standard_normal((3, 5))).astype(np.float32)
    add("svd_lowrank 의 S", lambda L: L.svd_lowrank(L.tensor(low), q=3)[1])
    add("svd_lowrank 의 모양",
        lambda L: " ".join(str(tuple(t.shape))
                           for t in L.svd_lowrank(L.tensor(low), q=3)))
    add("pca_lowrank 의 S", lambda L: L.pca_lowrank(L.tensor(low), q=3)[1])
    add("pca_lowrank(center=False) 의 S",
        lambda L: L.pca_lowrank(L.tensor(low), q=3, center=False)[1])
    return cases


CACHE_PREFIX = "cache::"


def scalar_cache_cases(inp=None):
    """**크기 1 짜리 파라미터가 전역 상수를 더럽히지 않는가.**

    ## 왜 이 자리가 있는가

    자매(borch.ts)는 `Tensor.full` 이 원소 하나짜리를 **값으로 캐시한다.** 같은 값을
    두 번 물으면 같은 버퍼가 나오고, `zeros`·`ones` 도 그 문을 지난다. 빠르지만 —
    **제자리로 고칠 것이 그 버퍼를 물려받으면 전역 상수가 통째로 바뀐다.**

    옵티마이저 상태에서 실제로 그렇게 났고(자매 세션이 잡았다), 층 쪽에도 같은 문이
    있다: `nn.PReLU()` 의 기본 가중치는 크기 1 의 `0.25` 이고, `BatchNorm(1)` 의 이동
    통계는 크기 1 의 `0`·`1` 이다. 셋 다 학습이 **제자리로** 고치는 것들이다.

    ## 케이스가 실제로 그 자리를 밟게 하려면

    파라미터 값만 보면 오염돼도 그 자리는 맞게 나온다 — 고쳐 쓴 값이 곧 답이니까.
    **한 스텝 밟은 뒤에 새로 만든 상수**를 봐야 한다. 그래서 순서가 이렇다:

    1. 크기 1 파라미터를 만든다 (여기서 캐시된 버퍼를 물려받는다)
    2. 한 스텝 학습시킨다 (제자리로 고친다 — 오염이 있다면 여기서 난다)
    3. **그 뒤에** `zeros`·`ones`·`full(0.25)` 를 새로 만들어 값을 본다

    3번이 없으면 이 케이스는 초록인 채로 아무것도 안 잰다. 옵티마이저 쪽 케이스를
    두 번 고쳐 쓴 이유가 정확히 이것이었다.

    torch 에는 이런 캐시가 없으므로 답은 언제나 깨끗한 상수다 — 그것이 기대값이다.
    """
    cases = []

    def add(name, fn):
        cases.append((CACHE_PREFIX + name, fn))

    def fresh(L):
        """**새로** 만든 전역 상수 셋. 오염됐으면 여기서 다른 값이 나온다."""
        return L.cat([L.zeros(1), L.ones(1), L.full((1,), 0.25)])

    def prelu_then_constants(L):
        m = L.nn.PReLU()
        opt = L.optim.SGD(m.parameters(), lr=0.5)
        opt.zero_grad()
        # **음수 자리가 있어야 기울기가 흐른다** — PReLU 의 가중치는 음수 쪽에만 붙는다.
        m(L.tensor(np.array([[-2.0, 1.0]], dtype=np.float32))).sum().backward()
        opt.step()
        return fresh(L)

    add("PReLU 한 스텝 뒤의 상수", prelu_then_constants)
    # 파라미터 자체도 굳힌다 — 학습이 정말 움직였는지가 여기서만 보인다. 안 움직였으면
    # 위의 케이스는 밟지도 않은 자리를 통과시킨 것이다.
    add("PReLU 가 실제로 움직였다", lambda L: _prelu_stepped(L))

    def batchnorm_then_constants(L):
        bn = L.nn.BatchNorm1d(1)
        bn(L.tensor(np.array([[1.0], [3.0]], dtype=np.float32)))
        return fresh(L)

    add("BatchNorm(1) 한 번 지난 뒤의 상수", batchnorm_then_constants)
    add("BatchNorm(1) 의 이동 통계가 움직였다",
        lambda L: _batchnorm_stats(L))
    return cases


def _prelu_stepped(L):
    m = L.nn.PReLU()
    opt = L.optim.SGD(m.parameters(), lr=0.5)
    opt.zero_grad()
    m(L.tensor(np.array([[-2.0, 1.0]], dtype=np.float32))).sum().backward()
    opt.step()
    return list(m.parameters())[0]


def _batchnorm_stats(L):
    bn = L.nn.BatchNorm1d(1)
    bn(L.tensor(np.array([[1.0], [3.0]], dtype=np.float32)))
    return L.cat([bn.running_mean, bn.running_var])


BLEND_PREFIX = "blend::"


def blend_cases(inp=None):
    """addmm 계열. 여덟이 전부 `β·input + α·(무슨 곱)` 한 꼴이다.

    ## `beta=0` 이 이 묶음의 요점이다

    **값은 안 보고 그래프에는 남는다.** 둘 다여야 하고 요구가 반대 방향이다 —

    - `input * 0` 으로 적으면 NaN 을 넣었을 때 결과가 NaN 이 된다. torch 는 멀쩡하다.
    - 그렇다고 그래프에서 빼면 `input.grad` 가 0 이 아니라 **없다.** torch 는 0 을
      준다(실측). 빼 두면 `backward()` 가 "requires_grad 가 아니다" 로 멈춘다.

    평범한 입력으로는 **어느 쪽도** 안 보인다 — NaN 을 넣어야 첫째가, 기울기를 물어야
    둘째가 드러난다. 그래서 둘 다 묻는다.

    ## 나머지가 갈리는 자리

    - **`beta` 와 `alpha` 를 둘 다 1 이 아니게** 해야 어느 쪽이 어디에 곱해지는지가
      드러난다. 둘 다 1 이면 자리를 바꿔 적어도 같은 답이다.
    - **배치를 둘 이상**으로 둬야 `addbmm`(합친다)과 `baddbmm`(지킨다)이 갈린다.
      배치가 1 이면 두 함수가 같아 보인다.
    - **`input` 이 결과보다 작아야** 퍼지는 것이 보인다. torch 는 `(4,)` 도 스칼라도
      받는다(실측).
    - `addcmul`·`addcdiv` 에는 **`beta` 가 없다** — `input` 의 계수가 늘 1 이다.
      `value` 만 있고, 그것은 곱 쪽에 붙는다.
    """
    m1 = np.arange(6, dtype=np.float32).reshape(2, 3)
    m2 = np.arange(12, dtype=np.float32).reshape(3, 4)
    base = np.full((2, 4), 10.0, dtype=np.float32)
    b1 = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    b2 = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    deep = np.full((2, 2, 4), 10.0, dtype=np.float32)
    vec = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    v1 = np.array([1.0, 2.0], dtype=np.float32)
    v2 = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    t0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t1 = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    t2 = np.array([[5.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    # **고르지 않은 무게.** 전부 1 이면 자리마다 다른 몫이 상쇄되어 안 보인다.
    weight = np.array([[1.0, 2.0, 0.5, 3.0], [2.0, 0.5, 1.5, 1.0]],
                      dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((BLEND_PREFIX + name, fn))

    def grad(name, fn, src=base, w=weight):
        def run(L, f=fn, s=src, ww=w, n=name):
            x = L.tensor(np.asarray(s, dtype=np.float32).copy(),
                         requires_grad=True)
            out = f(L, x)
            (out * L.tensor(np.asarray(ww, dtype=np.float32).reshape(
                tuple(out.shape)))).sum().backward()
            return _grad_of(x, n)

        cases.append((BLEND_PREFIX + f"grad::{name}", run))

    # ── addmm ───────────────────────────────────────────────────────────
    for beta, alpha in ((1, 1), (2, 3), (0, 1), (1, 0), (-1, 0.5)):
        add(f"addmm(beta={beta}, alpha={alpha})",
            lambda L, b=beta, a=alpha: L.addmm(L.tensor(base), L.tensor(m1),
                                               L.tensor(m2), beta=b, alpha=a))
    # **NaN 을 넣어야 `input * 0` 으로 적은 것이 드러난다.**
    nan_base = np.full((2, 4), np.nan, dtype=np.float32)
    add("addmm(beta=0, input=NaN)",
        lambda L: L.addmm(L.tensor(nan_base), L.tensor(m1), L.tensor(m2),
                          beta=0))
    add("addmm(input 이 (4,))",
        lambda L: L.addmm(L.ones(4), L.tensor(m1), L.tensor(m2)))
    add("addmm(input 이 스칼라)",
        lambda L: L.addmm(L.ones(()), L.tensor(m1), L.tensor(m2)))
    grad("addmm(beta=2, alpha=3)",
         lambda L, x: L.addmm(x, L.tensor(m1), L.tensor(m2), beta=2, alpha=3))
    # **기울기를 물어야 그래프에서 뺀 것이 드러난다.** 빼 두면 여기서 멈춘다.
    grad("addmm(beta=0)",
         lambda L, x: L.addmm(x, L.tensor(m1), L.tensor(m2), beta=0))
    grad("addmm(퍼지는 input)",
         lambda L, x: L.addmm(x, L.tensor(m1), L.tensor(m2)),
         src=np.ones(4, dtype=np.float32))
    grad("addmm(mat1)",
         lambda L, x: L.addmm(L.tensor(base), x, L.tensor(m2), alpha=3), m1)

    # ── addbmm · baddbmm ────────────────────────────────────────────────
    for beta, alpha in ((1, 1), (2, 3), (0, 1)):
        add(f"addbmm(beta={beta}, alpha={alpha})",
            lambda L, b=beta, a=alpha: L.addbmm(L.tensor(base), L.tensor(b1),
                                                L.tensor(b2), beta=b, alpha=a))
        add(f"baddbmm(beta={beta}, alpha={alpha})",
            lambda L, b=beta, a=alpha: L.baddbmm(L.tensor(deep), L.tensor(b1),
                                                 L.tensor(b2), beta=b, alpha=a))
    add("baddbmm(input 이 (2,4))",
        lambda L: L.baddbmm(L.tensor(base), L.tensor(b1), L.tensor(b2)))
    grad("addbmm", lambda L, x: L.addbmm(x, L.tensor(b1), L.tensor(b2)))
    grad("addbmm(batch1)",
         lambda L, x: L.addbmm(L.tensor(base), x, L.tensor(b2), alpha=2), b1)
    grad("baddbmm", lambda L, x: L.baddbmm(x, L.tensor(b1), L.tensor(b2)),
         src=deep,
         w=np.arange(1, 17, dtype=np.float32).reshape(2, 2, 4))

    # ── addmv · addr ────────────────────────────────────────────────────
    for beta, alpha in ((1, 1), (2, 3), (0, 1)):
        add(f"addmv(beta={beta}, alpha={alpha})",
            lambda L, b=beta, a=alpha: L.addmv(L.ones(2), L.tensor(m1),
                                               L.tensor(vec), beta=b, alpha=a))
        add(f"addr(beta={beta}, alpha={alpha})",
            lambda L, b=beta, a=alpha: L.addr(L.ones(2, 3), L.tensor(v1),
                                              L.tensor(v2), beta=b, alpha=a))
    grad("addmv(mat)",
         lambda L, x: L.addmv(L.ones(2), x, L.tensor(vec), alpha=2), m1,
         w=np.array([1.0, 2.0], dtype=np.float32))
    grad("addr(vec1)",
         lambda L, x: L.addr(L.ones(2, 3), x, L.tensor(v2), alpha=2), v1,
         w=np.arange(1, 7, dtype=np.float32).reshape(2, 3))

    # ── addcmul · addcdiv ───────────────────────────────────────────────
    for value in (1, 2, -1, 0):
        add(f"addcmul(value={value})",
            lambda L, v=value: L.addcmul(L.tensor(t0), L.tensor(t1),
                                         L.tensor(t2), value=v))
        add(f"addcdiv(value={value})",
            lambda L, v=value: L.addcdiv(L.tensor(t0), L.tensor(t1),
                                         L.tensor(t2), value=v))
    add("addcmul(브로드캐스트)",
        lambda L: L.addcmul(L.tensor(t0),
                            L.tensor(np.array([1.0, 10.0], dtype=np.float32)),
                            L.tensor(t2)))
    grad("addcdiv",
         lambda L, x: L.addcdiv(x, L.tensor(t1), L.tensor(t2), value=2),
         src=t0, w=np.array([[1.0, 2.0], [0.5, 3.0]], dtype=np.float32))

    # ── 제자리 ──────────────────────────────────────────────────────────
    #
    # **메서드로만 있다** — `torch.addmm_` 이라는 최상위 이름이 없다(실측).
    # 예외가 `addmv_` 하나인데, 그것도 메서드 꼴로 함께 묻는다.
    inplace = (
        ("addmm_", base, lambda L, t: t.addmm_(L.tensor(m1), L.tensor(m2))),
        ("addbmm_", base, lambda L, t: t.addbmm_(L.tensor(b1), L.tensor(b2))),
        ("baddbmm_", deep, lambda L, t: t.baddbmm_(L.tensor(b1), L.tensor(b2))),
        ("addmv_", np.ones(2, dtype=np.float32),
         lambda L, t: t.addmv_(L.tensor(m1), L.tensor(vec))),
        ("addr_", np.ones((2, 3), dtype=np.float32),
         lambda L, t: t.addr_(L.tensor(v1), L.tensor(v2))),
        ("addcmul_", t0, lambda L, t: t.addcmul_(L.tensor(t1), L.tensor(t2))),
        ("addcdiv_", t0, lambda L, t: t.addcdiv_(L.tensor(t1), L.tensor(t2))),
    )
    for name, src, run in inplace:
        def value(L, s=src, r=run):
            x = L.tensor(np.asarray(s, dtype=np.float32).copy())
            r(L, x)
            return x

        add(f"제자리::{name}", value)

        def is_self(L, s=src, r=run):
            x = L.tensor(np.asarray(s, dtype=np.float32).copy())
            return str(r(L, x) is x)

        add(f"제자리::{name}(같은 텐서)", is_self)
    return cases


INDEX_PREFIX = "index::"


def index_cases(inp=None):
    """색인으로 **쓰는** 쪽. 읽는 쪽(`gather`·`index_select`)은 이미 있었다.

    `gather` 는 있는데 그 반대인 `scatter` 가 없었다. 한쪽만 있으면 "꺼낼 수는 있는데
    되돌려 넣을 수가 없는" 상태이고, 그 자리는 임베딩이나 원-핫을 손으로 만드는
    코드가 바로 만난다.

    **번호가 겹칠 때가 요점이다.** `scatter` 는 마지막에 쓴 것이 남고 `scatter_add`
    는 더한다 — 겹치지 않는 번호로만 재면 둘이 같은 함수처럼 보인다.
    """
    inp = golden_inputs() if inp is None else inp
    x2 = inp["x2"]                                       # (3, 4)
    src = (x2 * 10).astype(np.float32)
    # 겹치는 번호. 0 이 두 번 나온다 — `scatter` 와 `scatter_add` 가 여기서 갈린다.
    dup = np.array([[0, 0, 1, 2], [1, 1, 2, 3], [2, 2, 3, 0]], dtype=np.int64)
    flat_idx = np.array([0, 2, 2, 5], dtype=np.int64)
    cases = []

    def add(name, fn):
        cases.append((INDEX_PREFIX + name, fn))

    add("scatter(겹치는 번호)",
        lambda L: L.zeros(3, 4).scatter(1, L.tensor(dup), L.tensor(src)))
    add("scatter_add(겹치는 번호)",
        lambda L: L.zeros(3, 4).scatter_add(1, L.tensor(dup), L.tensor(src)))
    add("scatter(스칼라)",
        lambda L: L.zeros(3, 4).scatter(1, L.tensor(dup), 7.0))

    # 기울기도 본다. **쓰인 자리로만 흘러야 한다.**
    def scatter_grad(L):
        s = L.tensor(src, requires_grad=True)
        out = L.zeros(3, 4).scatter_add(1, L.tensor(dup), s)
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(s, "scatter_add")

    cases.append((INDEX_PREFIX + "grad::scatter_add", scatter_grad))

    add("index_add",
        lambda L: L.zeros(3, 4).index_add(
            0, L.tensor(np.array([0, 0, 2], dtype=np.int64)), L.tensor(x2)))
    add("index_copy",
        lambda L: L.zeros(3, 4).index_copy(
            0, L.tensor(np.array([2, 1, 0], dtype=np.int64)), L.tensor(x2)))
    add("index_fill",
        lambda L: L.tensor(x2).index_fill(
            1, L.tensor(np.array([0, 2], dtype=np.int64)), -1.0))

    # `take` 는 **평평하게 펴서** 뽑는다 — 축이라는 개념이 없다.
    add("take", lambda L: L.take(L.tensor(x2), L.tensor(flat_idx)))
    add("take_along_dim",
        lambda L: L.take_along_dim(L.tensor(x2), L.tensor(dup), dim=1))

    # 정렬된 것 안에서 자리를 찾는다. **`right` 가 동점의 어느 쪽인지를 정한다.**
    line = np.array([1., 3., 5., 7.], dtype=np.float32)
    want = np.array([0., 3., 6., 9.], dtype=np.float32)
    add("searchsorted", lambda L: L.searchsorted(L.tensor(line), L.tensor(want)))
    add("searchsorted(right)",
        lambda L: L.searchsorted(L.tensor(line), L.tensor(want), right=True))
    add("bucketize", lambda L: L.bucketize(L.tensor(want), L.tensor(line)))
    return cases


NEWFN_PREFIX = "newfn::"


def new_function_cases(inp=None):
    """torch 에 있고 여기 없던 **진짜 새 기능** 한 묶음.

    앞선 두 묶음은 이름만 없던 것들이었다 — 연산자가 이미 하는 일에 철자를 붙였다.
    여기는 계산 자체가 없던 자리다.

    고른 기준은 **교재가 부르는가**다. `torch.igammac` 이 없어서 멈추는 코드보다
    `torch.meshgrid` 나 `torch.randn_like` 가 없어서 멈추는 코드가 훨씬 많다.
    """
    inp = golden_inputs() if inp is None else inp
    x1, x2 = inp["x1"], inp["x2"]
    pos = inp["xp"]
    withnan = np.array([1., np.nan, -np.inf, np.inf, 3.], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((NEWFN_PREFIX + name, fn))

    # ── `*_like` — 모양만 빌린다. **값이 아니라 모양을 답으로 굳힌다.** ──────
    #
    # 난수 계열은 값이 같을 수 없으므로 모양을 문자열로 낸다. `zeros_like` 처럼
    # 값이 정해진 것은 값으로 묻는다.
    for name in ("empty_like", "rand_like", "randn_like"):
        add(f"{name}/모양",
            lambda L, n=name: " ".join(str(int(v)) for v in
                                       getattr(L, n)(L.tensor(x2)).shape))
    add("randint_like/모양",
        lambda L: " ".join(str(int(v)) for v in L.randint_like(L.tensor(x2), 5).shape))

    add("logspace", lambda L: L.logspace(0.0, 2.0, 5))
    add("scalar_tensor", lambda L: L.scalar_tensor(2.5))

    # ── meshgrid. **`indexing` 을 안 주면 torch 가 경고하고 `ij` 로 간다.** ──
    add("meshgrid/0", lambda L: L.meshgrid(L.tensor(x1[:3]), L.tensor(x1[:2]),
                                           indexing="ij")[0])
    add("meshgrid/1", lambda L: L.meshgrid(L.tensor(x1[:3]), L.tensor(x1[:2]),
                                           indexing="ij")[1])
    add("meshgrid(xy)", lambda L: L.meshgrid(L.tensor(x1[:3]), L.tensor(x1[:2]),
                                             indexing="xy")[0])

    # ── 원소별. ────────────────────────────────────────────────────────────
    add("lerp", lambda L: L.lerp(L.tensor(x1), L.tensor(x1 * 2), 0.25))
    add("nan_to_num", lambda L: L.nan_to_num(L.tensor(withnan)))
    add("nan_to_num(값 지정)",
        lambda L: L.nan_to_num(L.tensor(withnan), nan=0.5, posinf=9.0, neginf=-9.0))
    add("isclose", lambda L: L.isclose(L.tensor(x1), L.tensor(x1 + 1e-9)))
    add("isreal", lambda L: L.isreal(L.tensor(withnan)))
    add("isposinf", lambda L: L.isposinf(L.tensor(withnan)))
    add("isneginf", lambda L: L.isneginf(L.tensor(withnan)))
    # **`fmax`·`fmin` 은 NaN 을 건너뛴다** — `maximum` 은 NaN 을 물고 나온다.
    add("fmax(NaN 건너뜀)",
        lambda L: L.fmax(L.tensor(withnan), L.tensor(np.zeros(5, dtype=np.float32))))
    add("fmin(NaN 건너뜀)",
        lambda L: L.fmin(L.tensor(withnan), L.tensor(np.zeros(5, dtype=np.float32))))
    add("float_power", lambda L: L.float_power(L.tensor(pos), 2.0))
    add("logical_xor",
        lambda L: L.logical_xor(L.tensor(np.array([1., 0., 1., 0.], dtype=np.float32)),
                                L.tensor(np.array([1., 1., 0., 0.], dtype=np.float32))))

    # `isin` — 원소가 그 목록에 있는가. 브로드캐스팅 하나로 풀린다.
    add("isin", lambda L: L.isin(L.tensor(np.array([1., 2., 3., 4.], dtype=np.float32)),
                                 L.tensor(np.array([2., 4.], dtype=np.float32))))

    # ── 짝을 내는 축약. **하나만 물으면 다른 하나가 틀려도 통과한다.** ───────
    add("var_mean/분산", lambda L: L.var_mean(L.tensor(x2))[0])
    add("var_mean/평균", lambda L: L.var_mean(L.tensor(x2))[1])
    add("std_mean/표준편차", lambda L: L.std_mean(L.tensor(x2))[0])

    # ── 곱셈 계열. ─────────────────────────────────────────────────────────
    add("inner", lambda L: L.inner(L.tensor(x2), L.tensor(x2)))
    add("vdot", lambda L: L.vdot(L.tensor(x1), L.tensor(x1)))
    add("kron", lambda L: L.kron(L.tensor(x1[:2]), L.tensor(x1[2:4])))
    add("cross", lambda L: L.cross(L.tensor(x1[:3].reshape(1, 3)),
                                   L.tensor(x1[3:6].reshape(1, 3)), dim=1))
    return cases


POOL_PREFIX = "pool::"


def pool_cases(inp=None):
    """풀링의 나머지 차원과 나머지 종류.

    `max_pool1d/2d/3d`·`avg_pool2d`·`adaptive_avg_pool2d` 만 있었다. 차원이 하나
    있으면 나머지 둘도 있을 것이라고 읽히는 자리이고, 그 기대가 어긋나면 1 차원
    신호나 3 차원 부피를 다루는 코드가 중간에 멈춘다.

    **적응형은 창 크기를 입력에서 거꾸로 푼다.** 나누어떨어지지 않을 때 어느 자리를
    어떻게 나눌지가 규칙이고, 그 규칙이 torch 와 갈리면 값이 조용히 다르다 — 그래서
    나누어떨어지는 경우와 안 떨어지는 경우를 둘 다 묻는다.
    """
    inp = golden_inputs() if inp is None else inp
    seq, img, vol = inp["nd_seq"], inp["img"], inp["nd_vol"]
    cases = []

    def add(name, fn, arr):
        cases.append((POOL_PREFIX + name, lambda L, f=fn, a=arr: f(L, L.tensor(a))))

        def grad(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(x, n)
        cases.append((POOL_PREFIX + f"grad::{name}", grad))

    # ── 평균 풀링의 1·3 차원. 2 차원만 있었다. ─────────────────────────────
    add("F.avg_pool1d", lambda L, x: L.nn.functional.avg_pool1d(x, 2), seq)
    add("F.avg_pool3d", lambda L, x: L.nn.functional.avg_pool3d(x, 2), vol)
    cases.append((POOL_PREFIX + "nn.AvgPool1d",
                  lambda L: L.nn.AvgPool1d(2)(L.tensor(seq))))
    cases.append((POOL_PREFIX + "nn.AvgPool3d",
                  lambda L: L.nn.AvgPool3d(2)(L.tensor(vol))))

    # ── 적응형 평균. **나누어떨어질 때와 아닐 때를 둘 다 본다.** ────────────
    add("F.adaptive_avg_pool1d(4)",
        lambda L, x: L.nn.functional.adaptive_avg_pool1d(x, 4), seq)
    add("F.adaptive_avg_pool1d(3)",                       # 8 을 3 으로 — 안 떨어진다
        lambda L, x: L.nn.functional.adaptive_avg_pool1d(x, 3), seq)
    add("F.adaptive_avg_pool3d",
        lambda L, x: L.nn.functional.adaptive_avg_pool3d(x, 2), vol)
    cases.append((POOL_PREFIX + "nn.AdaptiveAvgPool1d",
                  lambda L: L.nn.AdaptiveAvgPool1d(4)(L.tensor(seq))))
    cases.append((POOL_PREFIX + "nn.AdaptiveAvgPool3d",
                  lambda L: L.nn.AdaptiveAvgPool3d(2)(L.tensor(vol))))

    # ── 적응형 최대. 평균과 **동점 규칙이 다르다** — 하나만 고른다. ─────────
    add("F.adaptive_max_pool1d",
        lambda L, x: L.nn.functional.adaptive_max_pool1d(x, 4), seq)
    add("F.adaptive_max_pool2d",
        lambda L, x: L.nn.functional.adaptive_max_pool2d(x, 2), img)
    add("F.adaptive_max_pool2d(안 떨어짐)",
        lambda L, x: L.nn.functional.adaptive_max_pool2d(x, 3), img)
    add("F.adaptive_max_pool3d",
        lambda L, x: L.nn.functional.adaptive_max_pool3d(x, 2), vol)
    for nd, arr, size in (("1d", seq, 4), ("2d", img, 2), ("3d", vol, 2)):
        cases.append((POOL_PREFIX + f"nn.AdaptiveMaxPool{nd}",
                      lambda L, n=nd, a=arr, s=size:
                      getattr(L.nn, f"AdaptiveMaxPool{n}")(s)(L.tensor(a))))

    # ── LP 풀링. `p` 승 평균의 `p` 제곱근이다 — p=1 은 합, p=∞ 는 최대에 가깝다. ─
    add("F.lp_pool1d(p=2)", lambda L, x: L.nn.functional.lp_pool1d(x, 2, 2), seq)
    add("F.lp_pool2d(p=2)", lambda L, x: L.nn.functional.lp_pool2d(x, 2, 2), img)
    add("F.lp_pool2d(p=1)", lambda L, x: L.nn.functional.lp_pool2d(x, 1, 2), img)
    cases.append((POOL_PREFIX + "nn.LPPool2d",
                  lambda L: L.nn.LPPool2d(2, 2)(L.tensor(img))))
    return cases


MODFN_PREFIX = "modfn::"


def module_function_cases(inp=None):
    """`torch.sum(x)` 처럼 **모듈 함수로 부르는 꼴.**

    torch 는 거의 모든 것을 두 이름으로 준다 — `x.sum()` 과 `torch.sum(x)`. 이 표는
    오래 메서드 꼴로만 물었고, 그래서 모듈 함수가 통째로 빠져 있는 것을 못 봤다.
    `reduce::sum(dim)` 케이스를 쓰다가 `module 'borch' has no attribute 'sum'` 로
    걸렸고, 그때 세어 보니 그런 이름이 **쉰 개**였다.

    여기서 묻는 것은 값이 맞는가가 아니라 — 그건 메서드 쪽 케이스가 이미 묻는다 —
    **그 이름이 그 자리에 있는가**다. 그래서 한 줄씩 값으로 확인한다.
    """
    inp = golden_inputs() if inp is None else inp
    x2, x1 = inp["x2"], inp["x1"]
    cases = []

    def add(name, fn):
        cases.append((MODFN_PREFIX + name, fn))

    add("sum", lambda L: L.sum(L.tensor(x2)))
    add("sum(dim)", lambda L: L.sum(L.tensor(x2), dim=1))
    add("mean", lambda L: L.mean(L.tensor(x2)))
    add("mean(dim)", lambda L: L.mean(L.tensor(x2), dim=0))
    add("std", lambda L: L.std(L.tensor(x2)))
    add("var", lambda L: L.var(L.tensor(x2)))
    add("numel", lambda L: L.tensor(np.int64(L.numel(L.tensor(x2)))))
    add("argmax", lambda L: L.argmax(L.tensor(x2)))
    add("argmin(dim)", lambda L: L.argmin(L.tensor(x2), dim=1))
    add("clone", lambda L: L.clone(L.tensor(x2)))
    add("detach", lambda L: L.detach(L.tensor(x2)))
    add("flatten", lambda L: L.flatten(L.tensor(x2)))
    # **모듈 꼴은 축을 튜플로 받는다.** 메서드는 흩어서도 받는데 여기는 아니다 —
    # `torch.permute(x, 1, 0)` 은 `TypeError` 다.
    add("permute", lambda L: L.permute(L.tensor(x2), (1, 0)))
    add("transpose", lambda L: L.transpose(L.tensor(x2), 0, 1))
    add("squeeze", lambda L: L.squeeze(L.tensor(x1).reshape(1, 6, 1)))

    # **`max`·`min` 은 축을 주면 짝을 낸다.** 자리로 꺼내면 양쪽 이름이 달라도 통한다.
    add("max", lambda L: L.max(L.tensor(x2)))
    add("max(dim)/값", lambda L: L.max(L.tensor(x2), dim=1)[0])
    add("min(dim)/번호", lambda L: L.min(L.tensor(x2), dim=1)[1])

    # 제자리 연산도 모듈 이름이 있다. **원본이 바뀌는지**를 본다.
    def inplace(L):
        t = L.tensor(x1.copy())
        L.relu_(t)
        return t

    add("relu_(원본이 바뀐다)", inplace)

    # ── torch 가 **두 번째 이름**으로 주는 것들. ────────────────────────────
    #
    # `a + b` 는 되는데 `torch.add(a, b)` 가 없었다. 계산이 아니라 이름이 없어서
    # 안 도는 자리이고, 그런 자리는 값 대조로만 있는지 없는지가 드러난다.
    a2 = x2
    b2 = (x2 * 0.5 + 1.0).astype(np.float32)
    add("add", lambda L: L.add(L.tensor(a2), L.tensor(b2)))
    # **`alpha` 는 연산자에 없다** — 별칭으로 두면 이 자리가 조용히 빠진다.
    add("add(alpha)", lambda L: L.add(L.tensor(a2), L.tensor(b2), alpha=2.0))
    add("sub", lambda L: L.sub(L.tensor(a2), L.tensor(b2)))
    add("mul", lambda L: L.mul(L.tensor(a2), L.tensor(b2)))
    add("div", lambda L: L.div(L.tensor(a2), L.tensor(b2)))
    add("div(floor)",
        lambda L: L.div(L.tensor(a2), L.tensor(b2), rounding_mode="floor"))
    add("rsub", lambda L: L.rsub(L.tensor(a2), L.tensor(b2)))
    # **`remainder` 와 `fmod` 는 음수에서 갈린다** — 부호가 반대쪽을 따른다.
    neg = np.array([[-5., -3., 3., 5.]], dtype=np.float32)
    add("remainder(음수)",
        lambda L: L.remainder(L.tensor(neg), L.tensor(np.float32(3.0))))
    add("fmod(음수)", lambda L: L.fmod(L.tensor(neg), L.tensor(np.float32(3.0))))
    add("floor_divide(음수)",
        lambda L: L.floor_divide(L.tensor(neg), L.tensor(np.float32(3.0))))

    for name in ("greater", "greater_equal", "less", "less_equal", "not_equal"):
        add(name, lambda L, n=name: getattr(L, n)(L.tensor(a2), L.tensor(b2)))

    # 쌓기 넷. **1 차원과 2 차원에서 규칙이 갈린다.**
    line = x1[:4]
    add("hstack(1차원)", lambda L: L.hstack([L.tensor(line), L.tensor(line)]))
    add("hstack(2차원)", lambda L: L.hstack([L.tensor(a2), L.tensor(b2)]))
    add("vstack(1차원)", lambda L: L.vstack([L.tensor(line), L.tensor(line)]))
    add("column_stack(1차원)",
        lambda L: L.column_stack([L.tensor(line), L.tensor(line)]))
    add("dstack", lambda L: L.dstack([L.tensor(a2), L.tensor(b2)]))
    add("concat", lambda L: L.concat([L.tensor(a2), L.tensor(b2)], 0))
    add("block_diag", lambda L: L.block_diag(L.tensor(a2), L.tensor(b2[:1])))

    add("t(2차원)", lambda L: L.t(L.tensor(a2)))
    add("t(1차원은 그대로)", lambda L: L.t(L.tensor(line)))
    add("adjoint", lambda L: L.adjoint(L.tensor(a2)))
    add("moveaxis", lambda L: L.moveaxis(L.tensor(a2), 0, 1))
    add("broadcast_to", lambda L: L.broadcast_to(L.tensor(line).reshape(1, 4), (3, 4)))
    add("broadcast_tensors",
        lambda L: L.broadcast_tensors(L.tensor(line).reshape(1, 4),
                                      L.tensor(x1[:3]).reshape(3, 1))[0])
    return cases


SDPA_PREFIX = "sdpa::"


def sdpa_cases(inp=None):
    """`scaled_dot_product_attention`. **요즘 트랜스포머 코드가 직접 부르는 이름이다.**

    `MultiheadAttention` 은 이미 있는데 그 밑의 함수가 없었다. 층을 안 쓰고 어텐션을
    손으로 짜는 코드가 늘었고, 그런 코드는 이 이름을 부른다.

    묻는 것은 넷이다 — 맨 것, 더하는 가림막, 인과 가림막, 그리고 셋 다의 기울기.
    **가림막이 곱셈이 아니라 덧셈이라는 것**이 가장 흔한 오해다. `-inf` 를 더해
    softmax 가 0 을 내게 하는 것이지 0 을 곱하는 것이 아니다 — 곱하면 softmax 가
    이미 정규화한 뒤라 나머지가 1 로 안 돌아간다.
    """
    inp = golden_inputs() if inp is None else inp
    a = inp["attn_x"]                                  # (2, 5, 4)
    # 더하는 가림막. 0 은 통과, 큰 음수는 막는다.
    add_mask = np.zeros((5, 5), dtype=np.float32)
    add_mask[:, 3:] = -1e9
    cases = []

    def add(name, fn):
        cases.append((SDPA_PREFIX + name, lambda L, f=fn: f(L)))

        def grad(L, f=fn, n=name):
            q = L.tensor(a, requires_grad=True)
            out = f(L, q)
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(q, n)
        cases.append((SDPA_PREFIX + f"grad::{name}", grad))

    def plain(L, q=None):
        x = L.tensor(a) if q is None else q
        return L.nn.functional.scaled_dot_product_attention(x, x, x)

    def masked(L, q=None):
        x = L.tensor(a) if q is None else q
        return L.nn.functional.scaled_dot_product_attention(
            x, x, x, attn_mask=L.tensor(add_mask))

    def causal(L, q=None):
        x = L.tensor(a) if q is None else q
        return L.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True)

    add("맨 것", plain)
    add("더하는 가림막", masked)
    add("인과", causal)

    # **q·k·v 가 다른 자리도 본다.** 셋을 같은 것으로만 주면 세 인자를 뒤바꿔 써도
    # 값이 같아서 안 걸린다.
    def three(L):
        q = L.tensor(a)
        k = L.tensor(a * 0.5 + 0.1)
        v = L.tensor(a[::-1].copy())
        return L.nn.functional.scaled_dot_product_attention(q, k, v)

    cases.append((SDPA_PREFIX + "q·k·v 가 다를 때", three))
    return cases


DROPOUT_PREFIX = "dropout::"


def dropout_cases(inp=None):
    """Dropout. **값이 아니라 성질을 묻는다.**

    이 표의 다른 케이스는 전부 진짜 torch 의 **값**을 답으로 굳힌다. 여기서는 그럴
    수가 없다 — 답이 난수기에 달려 있고, 우리 난수기가 torch 의 것과 같을 이유가
    없기 때문이다. 그렇다고 안 물으면 층 하나가 통째로 검사 밖에 남는다.

    그래서 **torch 와 우리가 똑같이 답할 수 있는 것**만 묻는다:

    - 평가 모드는 항등이다 (값으로 묻는다 — 난수가 안 낀다)
    - `p=0` 도 항등이고 `p=1` 은 전부 0 이다
    - 살아남은 값은 정확히 `x/(1-p)` 배다 (**보정을 빠뜨리는 것이 가장 흔한 실수**이고,
      그러면 학습과 추론의 크기가 안 맞는다)
    - 떨구는 비율이 대략 `p` 다
    - 기울기는 살아남은 자리로만 흐른다

    답이 "그런가/아닌가" 이므로 난수기가 달라도 양쪽이 같은 답을 낸다. **값을 못
    묻는다고 안 묻는 것과, 물을 수 있는 것을 골라 묻는 것은 다르다.**
    """
    inp = golden_inputs() if inp is None else inp
    # 비율을 재려면 표본이 많아야 한다. 작은 배열로 재면 난수의 흔들림이 답을 흔든다.
    big = np.tile(inp["train_x"], (40, 1)).astype(np.float32)     # 960 × 6
    x2 = inp["x2"]
    cases = []

    def verdict(name, fn):
        cases.append((DROPOUT_PREFIX + name, lambda L, f=fn: f(L)))

    # ── 난수가 안 끼는 자리는 값으로 묻는다. ────────────────────────────────
    cases.append((DROPOUT_PREFIX + "eval 은 항등",
                  lambda L: L.nn.functional.dropout(L.tensor(x2), 0.5, training=False)))
    cases.append((DROPOUT_PREFIX + "p=0 은 항등",
                  lambda L: L.nn.functional.dropout(L.tensor(x2), 0.0, training=True)))
    cases.append((DROPOUT_PREFIX + "p=1 은 전부 0",
                  lambda L: L.nn.functional.dropout(L.tensor(x2), 1.0, training=True)))
    cases.append((DROPOUT_PREFIX + "nn.Dropout(eval) 은 항등",
                  lambda L: L.nn.Dropout(0.5).eval()(L.tensor(x2))))

    # ── 난수가 끼는 자리는 성질로 묻는다. ───────────────────────────────────
    def scaled(L):
        """살아남은 값이 정확히 `1/(1-p)` 배인가. **보정을 빼먹으면 여기서 걸린다.**"""
        p = 0.5
        x = L.tensor(big)
        out = to_numpy(L.nn.functional.dropout(x, p, training=True))
        src = np.asarray(big)
        kept = out != 0
        if not kept.any():
            return "아무것도 안 남았다"
        got = out[kept] / src[kept]
        return "맞다" if np.allclose(got, 1 / (1 - p), atol=1e-4) else \
            f"배율이 {float(np.abs(got - 1 / (1 - p)).max()):.3g} 만큼 어긋난다"

    def ratio(L):
        """떨구는 비율이 대략 `p` 인가. 표본 5,760 개에서 ±5%p 면 넉넉하다."""
        p = 0.5
        out = to_numpy(L.nn.functional.dropout(L.tensor(big), p, training=True))
        dropped = float((out == 0).mean())
        return "대략 맞다" if abs(dropped - p) < 0.05 else f"{dropped:.3f} 이 떨어졌다"

    def flows(L):
        """기울기가 **살아남은 자리로만** 흐르는가. 떨군 자리에 0 이 아닌 것이 오면 틀렸다."""
        x = L.tensor(big, requires_grad=True)
        out = L.nn.functional.dropout(x, 0.5, training=True)
        out.sum().backward()
        got = to_numpy(x.grad)
        made = to_numpy(out)
        stray = int(((made == 0) & (got != 0)).sum())
        return "살아남은 자리로만" if stray == 0 else f"떨군 자리 {stray} 곳에 흘렀다"

    def differs(L):
        """두 번 부르면 **다른 자리**를 떨구는가. 한 번 뽑아 캐시하면 여기서 걸린다."""
        x = L.tensor(big)
        a = to_numpy(L.nn.functional.dropout(x, 0.5, training=True))
        b = to_numpy(L.nn.functional.dropout(x, 0.5, training=True))
        return "다르다" if not np.array_equal(a == 0, b == 0) else "두 번이 같다"

    verdict("살아남은 값은 1/(1-p) 배", scaled)
    verdict("대략 p 만큼 떨군다", ratio)
    verdict("기울기는 살아남은 자리로만", flows)
    verdict("두 번 부르면 다른 자리", differs)
    return cases


OPT_PREFIX = "opt::"

# `(이름, 인자)`. **하나씩은 안 된다** — 옵티마이저는 상태를 쌓으므로 첫 스텝에서는
# 대부분 서로 비슷하게 굴고, 갈리는 것은 그 뒤다.
_OPTIMIZERS = [
    ("Adagrad", {"lr": 0.1}),
    ("Adadelta", {"lr": 0.5}),
    ("Adamax", {"lr": 0.05}),
    ("NAdam", {"lr": 0.05}),
    ("RAdam", {"lr": 0.05}),
    ("ASGD", {"lr": 0.05}),
    ("Rprop", {"lr": 0.05}),
    # **2 차원 가중치라야 Adafactor 의 요점이 돈다** — 여기 모델의 `0.weight` 가
    # (8, 6) 이라 행·열로 쪼개는 길을 지난다. 1 차원만 물으면 그 길이 통째로 안 돈다.
    ("Adafactor", {"lr": 0.05}),
]

# `(이름, 만드는 인자, 몇 번 밟을까)`. 학습률의 **자취**를 묻는다.
_SCHEDULERS = [
    ("ConstantLR", {"factor": 0.5, "total_iters": 3}, 8),
    ("LinearLR", {"start_factor": 0.5, "end_factor": 1.0, "total_iters": 4}, 8),
    ("PolynomialLR", {"total_iters": 5, "power": 2.0}, 8),
    ("MultiplicativeLR", {}, 6),
    ("CosineAnnealingWarmRestarts", {"T_0": 3, "T_mult": 2}, 10),
    ("OneCycleLR", {"max_lr": 0.4, "total_steps": 10}, 10),
    ("CyclicLR", {"base_lr": 0.01, "max_lr": 0.1, "step_size_up": 3}, 14),
    # **오르내림을 다르게 준다** — 같으면 `step_size_down` 이 있는지도 안 보인다.
    ("CyclicLR(위아래 다름)", {"base_lr": 0.01, "max_lr": 0.1, "step_size_up": 2,
                          "step_size_down": 4}, 14),
    # 두 번째 주기부터 봉우리가 절반이 된다 — 한 주기만 밟으면 안 갈린다.
    ("CyclicLR(triangular2)", {"base_lr": 0.01, "max_lr": 0.1, "step_size_up": 3,
                               "mode": "triangular2"}, 14),
    # **`exp_range` 의 기준은 주기가 아니라 걸음이다.** 그 하나가 갈리는 자리다.
    ("CyclicLR(exp_range)", {"base_lr": 0.01, "max_lr": 0.1, "step_size_up": 3,
                             "mode": "exp_range", "gamma": 0.9}, 14),
]


def opt_cases(inp=None):
    """옵티마이저 다섯과 스케줄러 여섯. **여러 스텝 뒤를 묻는다.**

    ## 왜 한 스텝으로는 안 되는가

    옵티마이저는 상태를 쌓는다. 첫 스텝에서 `Adam` 과 `NAdam` 과 `RAdam` 은 거의
    같은 값을 내고, `Adagrad` 와 `SGD` 도 학습률만 다른 정도다. 갈리는 것은 그
    누적이 자리를 잡은 뒤이므로, 한 스텝만 재면 다섯 개를 전부 `SGD` 로 구현해도
    통과한다.

    ## 스케줄러는 자취로 묻는다

    스케줄러가 하는 일은 **학습률의 수열**을 만드는 것이라, 그 수열을 통째로 답으로
    굳힌다. 마지막 값만 보면 가는 길이 달라도 통과하고, 실제로 `LinearLR` 과
    `ConstantLR` 은 끝에서 만난다 — `total_iters` 를 지나면 둘 다 원래 학습률이다.

    `MultiplicativeLR` 은 람다를 받는데, 골든이 담는 것은 답이지 함수가 아니므로
    부르는 쪽에서 같은 식을 쓴다(0.9 를 곱한다).
    """
    inp = golden_inputs() if inp is None else inp
    xin, yin = inp["train_x"], inp["train_y"]
    weights = {"0.weight": inp["w0"], "0.bias": inp["b0"],
               "2.weight": inp["w1"], "2.bias": inp["b1"]}
    cases = []

    def model_of(L):
        m = L.nn.Sequential(L.nn.Linear(6, 8), L.nn.ReLU(), L.nn.Linear(8, 3))
        m.load_state_dict({k: L.tensor(v) for k, v in weights.items()})
        return m

    def trained(L, name, args):
        m = model_of(L)
        opt = getattr(L.optim, name)(m.parameters(), **args)
        crit = L.nn.CrossEntropyLoss()
        x, y = L.tensor(xin), L.tensor(yin)
        for _ in range(5):
            opt.zero_grad()
            crit(m(x), y).backward()
            opt.step()
        return m

    for name, args in _OPTIMIZERS:
        cases.append((OPT_PREFIX + f"{name}/0.weight",
                      lambda L, n=name, a=args:
                      dict(trained(L, n, a).named_parameters())["0.weight"]))
        cases.append((OPT_PREFIX + f"{name}/손실",
                      lambda L, n=name, a=args: L.nn.CrossEntropyLoss()(
                          trained(L, n, a)(L.tensor(xin)), L.tensor(yin))))

    def lr_trace(L, name, args, steps):
        """학습률의 자취. **옵티마이저를 실제로 밟는다** — 순서가 값을 정한다."""
        m = model_of(L)
        opt = L.optim.SGD(m.parameters(), lr=0.2)
        # 이름에 괄호로 갈래를 적어 두었다 — 클래스 이름은 그 앞까지다.
        kind = name.split("(")[0]
        if kind == "MultiplicativeLR":
            sch = L.optim.lr_scheduler.MultiplicativeLR(opt, lambda epoch: 0.9)
        else:
            sch = getattr(L.optim.lr_scheduler, kind)(opt, **args)
        seen = []
        for _ in range(steps):
            seen.append(round(float(opt.param_groups[0]["lr"]), 6))
            opt.step()
            sch.step()
        # **라이브러리의 텐서로 돌려준다.** 하네스가 값을 받을 때 `detach` 를 부르고,
        # 맨 numpy 배열은 그것을 모른다. 수로 비교되므로 허용 오차도 그대로 적용된다.
        return L.tensor(np.array(seen, dtype=np.float32))

    for name, args, steps in _SCHEDULERS:
        cases.append((OPT_PREFIX + f"{name}/자취",
                      lambda L, n=name, a=args, s=steps: lr_trace(L, n, a, s)))

    # **이어 붙이는 둘.** 스케줄러를 조합하는 자리이고, 조합이 틀리면 개별 스케줄러가
    # 전부 맞아도 값이 갈린다.
    def sequential(L):
        m = model_of(L)
        opt = L.optim.SGD(m.parameters(), lr=0.2)
        a = L.optim.lr_scheduler.ConstantLR(opt, factor=0.25, total_iters=3)
        b = L.optim.lr_scheduler.ExponentialLR(opt, gamma=0.8)
        sch = L.optim.lr_scheduler.SequentialLR(opt, [a, b], milestones=[3])
        seen = []
        for _ in range(8):
            seen.append(round(float(opt.param_groups[0]["lr"]), 6))
            opt.step()
            sch.step()
        # **라이브러리의 텐서로 돌려준다.** 하네스가 값을 받을 때 `detach` 를 부르고,
        # 맨 numpy 배열은 그것을 모른다. 수로 비교되므로 허용 오차도 그대로 적용된다.
        return L.tensor(np.array(seen, dtype=np.float32))

    def chained(L):
        m = model_of(L)
        opt = L.optim.SGD(m.parameters(), lr=0.2)
        a = L.optim.lr_scheduler.ConstantLR(opt, factor=0.5, total_iters=2)
        b = L.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
        sch = L.optim.lr_scheduler.ChainedScheduler([a, b])
        seen = []
        for _ in range(6):
            seen.append(round(float(opt.param_groups[0]["lr"]), 6))
            opt.step()
            sch.step()
        # **라이브러리의 텐서로 돌려준다.** 하네스가 값을 받을 때 `detach` 를 부르고,
        # 맨 numpy 배열은 그것을 모른다. 수로 비교되므로 허용 오차도 그대로 적용된다.
        return L.tensor(np.array(seen, dtype=np.float32))

    cases.append((OPT_PREFIX + "SequentialLR/자취", sequential))
    cases.append((OPT_PREFIX + "ChainedScheduler/자취", chained))

    # ── 갈래를 좁혀 묻는 자리 ────────────────────────────────────────────
    #
    # 위의 모델 학습은 옵티마이저가 **대충 맞으면** 지난다. 갈래를 정하는 인자들은
    # 파라미터 하나에 기울기를 손으로 먹여야 드러난다.
    start = np.array([1.0, -2.0, 0.5], dtype=np.float32)

    def walk(L, name, grads, **args):
        p = L.tensor(start.copy(), requires_grad=True)
        opt = getattr(L.optim, name)([p], **args)
        seen = []
        for g in grads:
            opt.zero_grad()
            p.grad = L.tensor(g)
            opt.step()
            # **세 구현이 다 아는 길로 읽는다** — 하네스의 `to_numpy` 와 같은 길이다.
            # `p.data` 는 결속에서 저쪽 속성으로 새어 모양이 어긋난다.
            seen.append(np.asarray(p.detach().numpy(), dtype=np.float32).copy())
        return L.tensor(np.stack(seen))

    ramp = [np.array([0.1, -0.3, 0.2], dtype=np.float32) * (i + 1)
            for i in range(4)]
    # **부호가 뒤집히는 기울기.** Rprop 의 `etas` 와 "뒤집힌 칸은 안 간다" 규칙이
    # 여기서만 보인다 — 부호가 그대로면 폭이 커지기만 해서 한쪽 갈래만 돈다.
    flip = [np.array([0.1, -0.3, 0.2], dtype=np.float32),
            np.array([-0.1, -0.3, 0.2], dtype=np.float32),
            np.array([-0.2, -0.3, 0.2], dtype=np.float32),
            np.array([-0.2, 0.3, 0.2], dtype=np.float32)]

    narrow = [
        ("ASGD/기본값", "ASGD", ramp, {}),
        ("ASGD/lambd", "ASGD", ramp, {"lr": 0.1, "lambd": 0.01}),
        ("ASGD/alpha", "ASGD", ramp, {"lr": 0.1, "alpha": 0.5}),
        # **`t0` 을 낮춰야 평균이 실제로 돈다** — 기본값 100만에서 `mu` 는 늘 1 이고
        # `ax` 는 파라미터의 사본이다. 평균 갈래가 통째로 안 돌아간다.
        ("ASGD/t0(평균이 도는 자리)", "ASGD", ramp, {"lr": 0.1, "t0": 2}),
        ("ASGD/weight_decay", "ASGD", ramp, {"lr": 0.1, "weight_decay": 0.1}),
        ("Rprop/기본값", "Rprop", ramp, {}),
        ("Rprop/부호 바뀜", "Rprop", flip, {"lr": 0.1}),
        ("Rprop/etas", "Rprop", flip, {"lr": 0.1, "etas": (0.4, 1.5)}),
        ("Rprop/step_sizes 상한", "Rprop", ramp,
         {"lr": 0.1, "step_sizes": (1e-6, 0.11)}),
        ("Adafactor/기본값", "Adafactor", ramp, {}),
        ("Adafactor/weight_decay", "Adafactor", ramp,
         {"lr": 0.1, "weight_decay": 0.1}),
        ("Adafactor/d", "Adafactor", ramp, {"lr": 0.1, "d": 2.0}),
    ]
    for label, name, grads, args in narrow:
        cases.append((OPT_PREFIX + label,
                      lambda L, n=name, g=grads, a=args: walk(L, n, g, **a)))

    def adafactor_matrix(L, shape, **args):
        """**2 차원부터 행·열로 쪼갠다.** 1 차원은 상태 열쇠부터 다르다(`variance` 대
        `row_var`·`col_var`) — 이 최적화의 요점이 거기 있어서, 벡터로만 물으면 그 길이
        한 번도 안 돌아간다."""
        n = int(np.prod(shape))
        p = L.tensor((np.arange(n, dtype=np.float32).reshape(shape) / 4 - 0.5),
                     requires_grad=True)
        opt = L.optim.Adafactor([p], lr=0.1, **args)
        base = np.arange(n, dtype=np.float32).reshape(shape) / 8 - 0.2
        for i in range(3):
            opt.zero_grad()
            p.grad = L.tensor(base * (i + 1))
            opt.step()
        return p

    cases.append((OPT_PREFIX + "Adafactor/2차원",
                  lambda L: adafactor_matrix(L, (3, 4))))
    cases.append((OPT_PREFIX + "Adafactor/3차원",
                  lambda L: adafactor_matrix(L, (2, 3, 4))))

    def lbfgs(L, steps=3, **args):
        """**`step` 이 닫힘을 받는다** — 한 걸음 안에서 손실을 여러 번 다시 잰다.

        그 모양이 다른 옵티마이저와 달라서, 학습 루프를 그대로 쓰면 아무것도 안 한다.
        """
        p = L.tensor(start.copy(), requires_grad=True)
        opt = L.optim.LBFGS([p], **args)
        seen = []
        for i in range(steps):
            def closure(pp=p, k=i):
                pp.grad = L.tensor(np.array([0.1, -0.3, 0.2], dtype=np.float32))
                return (pp * pp).sum()
            opt.step(closure)
            seen.append(np.asarray(p.detach().numpy(), dtype=np.float32).copy())
        return L.tensor(np.stack(seen))

    cases.append((OPT_PREFIX + "LBFGS/기본값", lambda L: lbfgs(L, lr=0.1)))
    cases.append((OPT_PREFIX + "LBFGS/max_iter",
                  lambda L: lbfgs(L, lr=0.1, max_iter=3)))
    cases.append((OPT_PREFIX + "LBFGS/history_size",
                  lambda L: lbfgs(L, lr=0.5, max_iter=5, history_size=2)))

    def scalar_param_keeps_constants(L):
        """**원소가 하나인 파라미터를 학습해도 상수가 안 변해야 한다.**

        GPU 판은 원소 하나짜리 텐서를 **값으로 캐시해** 돌려준다 — 학습 루프에서
        `x * 0.05` 가 매 스텝 같은 상수를 만드는 것을 막으려는 것이고, 아무도 그
        버퍼에 안 쓰는 한 옳다. 옵티마이저 상태는 그 버퍼에 **제자리로 쓴다.**
        크기 1 파라미터에서 그 둘이 만나면 프로그램 전체가 공유하는 상수가 조용히
        덮어써진다 — 예외도 경고도 없고, 그때부터 아주 먼 자리에서 값이 틀린다.

        그래서 학습을 시킨 **뒤에** 같은 상수로 곱해 본다. 진짜 torch 에는 그런 캐시가
        없으니 답이 자명하고, 그 자명한 답이 이 결함을 잡는다. 가중치를 밖에서 넣는
        보통의 옵티마이저 케이스는 상태 은행을 안 지나가서 이것을 못 본다.

        **한 걸음으로는 못 잡는다.** Rprop 의 첫 걸음은 폭을 안 바꾸므로 같은 값을
        덮어써서 표가 안 난다 — 처음에 그렇게 적었고 안 걸렸다. 여러 걸음을 밟아
        상태가 실제로 움직인 뒤에 묻는다. 상수도 하나만 보면 안 된다: 상태 은행은
        0 에서 시작하므로 **0 과 1 이 먼저 더럽혀진다.**

        **옵티마이저를 골고루 밟아야 한다.** `SGD`·`Adam`·`RMSprop` 은 전용 커널을
        써서 나머지와 다른 밑동에 있다 — 처음에 공통 밑동만 고쳤더니 그 셋이 안 닿은
        채로 남았고, 그 셋을 안 물었으면 고친 줄 알고 넘어갔다.
        """
        seen = []
        for name, kw in (("Rprop", {"lr": 0.05}), ("Adafactor", {"lr": 0.05}),
                         ("ASGD", {"lr": 0.05}), ("Adagrad", {"lr": 0.05}),
                         ("SGD", {"lr": 0.05, "momentum": 0.9}),
                         ("Adam", {"lr": 0.05}), ("RMSprop", {"lr": 0.05})):
            p = L.tensor(np.array([0.5], dtype=np.float32), requires_grad=True)
            opt = getattr(L.optim, name)([p], **kw)
            for i in range(3):
                opt.zero_grad()
                p.grad = L.tensor(np.array([0.1 * (i + 1)], dtype=np.float32))
                opt.step()
        # 학습 뒤에 그 상수들이 아직 그 값인가.
        probe = L.tensor(np.array([1.0, 2.0], dtype=np.float32))
        for k in (0.0, 1.0, 0.05):
            seen.append(probe * k)
        return L.cat(seen)

    cases.append((OPT_PREFIX + "크기 1 파라미터가 상수를 안 더럽힌다",
                  scalar_param_keeps_constants))
    return cases


FNAME_PREFIX = "fname::"


def functional_name_cases(inp=None):
    """`nn.functional` 의 남은 이름들 — **제자리 활성**과 `interpolate` 의 옛 이름.

    ## 제자리 활성

    `F.relu_(x)` 는 `x` 를 제 버퍼에서 고친다. 학습 루프가 중간 텐서를 안 만들려고
    쓰는 자리다. 계산은 밑줄 없는 쪽이 하고 여기서는 되쓰기만 하므로, 물어야 할 것이
    셋이다 — **값이 같은가**, **같은 텐서를 돌려주는가**, **기울기 켜진 잎을 거절하는가**.
    가운데 것을 안 물으면 새 텐서를 돌려주는 구현이 값 케이스를 전부 지난다.

    ## `upsample` 세 이름

    torch 가 폐기 경고를 내면서도 계속 받는다. **`upsample_bilinear` 만
    `align_corners=True`** 이고 `interpolate(mode='bilinear')` 의 기본값은 거짓이다 —
    이름만 보고 별명으로 두면 가장자리가 어긋나는데 안쪽은 비슷해서 눈으로는 안 갈린다.
    """
    cases = []

    def add(name, fn):
        cases.append((FNAME_PREFIX + name, fn))

    def F(L):
        return L.nn.functional

    line = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]], dtype=np.float32)

    inplace = (
        ("relu_", lambda f, t: f.relu_(t)),
        ("celu_", lambda f, t: f.celu_(t)),
        ("celu_(alpha=0.5)", lambda f, t: f.celu_(t, 0.5)),
        ("elu_", lambda f, t: f.elu_(t)),
        ("selu_", lambda f, t: f.selu_(t)),
        ("hardtanh_", lambda f, t: f.hardtanh_(t)),
        ("hardtanh_(-1,1)", lambda f, t: f.hardtanh_(t, -1.0, 1.0)),
        ("leaky_relu_", lambda f, t: f.leaky_relu_(t)),
        ("leaky_relu_(0.3)", lambda f, t: f.leaky_relu_(t, 0.3)),
        ("threshold_", lambda f, t: f.threshold_(t, 0.5, -1.0)),
        # 무작위가 끼는 것은 **평가 모드**에서만 답이 정해진다.
        ("rrelu_(평가)", lambda f, t: f.rrelu_(t, 0.1, 0.3, False)),
    )
    for name, run in inplace:
        def value(L, r=run):
            x = L.tensor(line.copy())
            r(F(L), x)
            return x

        add(f"제자리::{name}", value)

    def same_tensor(L):
        """**같은 텐서를 돌려줘야 한다** — 새것을 내면 위 값 케이스는 전부 지난다."""
        x = L.tensor(line.copy())
        got = [F(L).relu_(x) is x, F(L).elu_(x) is x, F(L).selu_(x) is x]
        return " ".join(str(v) for v in got)

    add("제자리::같은 텐서인가", same_tensor)

    def refuses_leaf(L):
        x = L.tensor(line.copy(), requires_grad=True)
        try:
            F(L).relu_(x)
            return "예외가 안 났다"
        except Exception as exc:                                    # noqa: BLE001
            return type(exc).__name__

    add("제자리::기울기 켜진 잎은 거절", refuses_leaf)

    def same_as_plain(L):
        """제자리와 제자리 아닌 것이 **같은 답**이어야 한다. 두 벌이면 갈린다."""
        x = L.tensor(line.copy())
        F(L).leaky_relu_(x, 0.3)
        return x - F(L).leaky_relu(L.tensor(line.copy()), 0.3)

    add("제자리::제자리 아닌 것과 같다", same_as_plain)

    img = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    add("upsample(scale)",
        lambda L: F(L).upsample(L.tensor(img), scale_factor=2))
    add("upsample_nearest",
        lambda L: F(L).upsample_nearest(L.tensor(img), scale_factor=2))
    # **여기만 `align_corners=True`** 다.
    add("upsample_bilinear",
        lambda L: F(L).upsample_bilinear(L.tensor(img), scale_factor=2))
    add("upsample(size, bilinear)",
        lambda L: F(L).upsample(L.tensor(img), size=(8, 8), mode="bilinear"))
    add("upsample_bilinear(size=6)",
        lambda L: F(L).upsample_bilinear(L.tensor(img), size=(6, 6)))

    def upsample_corners_differ(L):
        """`upsample_bilinear` 과 `interpolate(bilinear)` 은 **같으면 안 된다.**

        전자는 `align_corners=True`, 후자의 기본값은 거짓이다. 별명으로 두면 여기서만
        드러난다 — 값 케이스는 각각 제 답을 내므로 통과한다.
        """
        x = L.tensor(img)
        a = F(L).upsample_bilinear(x, scale_factor=2)
        b = F(L).interpolate(x, scale_factor=2, mode="bilinear")
        return float((a - b).abs().sum().item()) > 1e-6

    add("upsample_bilinear 은 별명이 아니다",
        lambda L: str(upsample_corners_differ(L)))

    # ── `max`·`min` 의 세 얼굴 ──────────────────────────────────────────
    #
    # torch 는 인자에 따라 **다른 것을 낸다**: `max(x)` 는 전부의 최댓값 하나,
    # `max(x, dim)` 은 `(값, 번호)` 쌍, `max(x, other)` 는 칸마다의 최댓값.
    #
    # 이 세 갈래를 안 물었더니 결속에서 `x.max()` 가 **축 0 만 줄인 쌍**을 냈다 —
    # 저쪽의 `max(dim=0)` 으로 그냥 넘어갔기 때문이다. 스칼라로 바꿀 때만 시끄럽고,
    # 비교에 쓰면 칸마다 비교가 되어 조용히 다른 답이다.
    grid2 = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]], dtype=np.float32)
    other2 = np.array([[2.0, 2.0, 2.0], [7.0, 0.0, 7.0]], dtype=np.float32)

    add("max::전부", lambda L: L.tensor(grid2).max())
    add("min::전부", lambda L: L.tensor(grid2).min())
    add("max::전부(모양)", lambda L: str(tuple(L.tensor(grid2).max().shape)))
    add("max::축 하나의 값", lambda L: L.tensor(grid2).max(dim=1).values)
    add("max::축 하나의 번호",
        lambda L: L.tensor(grid2).max(dim=1).indices.float())
    add("min::축 하나의 값", lambda L: L.tensor(grid2).min(dim=0).values)
    add("max::칸마다",
        lambda L: L.max(L.tensor(grid2), L.tensor(other2)))
    add("min::칸마다",
        lambda L: L.min(L.tensor(grid2), L.tensor(other2)))

    # ── batch_norm ─────────────────────────────────────────────────────
    #
    # 층의 함수 꼴. **학습이면 running 통계를 제자리에서 고친다** — 넘긴 텐서가
    # 갱신되어 돌아온다. 새것을 돌려주는 구현은 출력 케이스를 전부 지나고 평가
    # 모드의 값만 틀리므로, 갱신된 통계 자체를 답으로 굳힌다.
    bn_x = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10) - 1
    bn_rm = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    bn_rv = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    bn_w = np.array([1.5, 0.5, 2.0], dtype=np.float32)
    bn_b = np.array([0.1, -0.1, 0.2], dtype=np.float32)

    def bn(L, **kw):
        return F(L).batch_norm(
            L.tensor(bn_x), L.tensor(bn_rm.copy()), L.tensor(bn_rv.copy()),
            L.tensor(bn_w), L.tensor(bn_b), **kw)

    add("batch_norm::평가", lambda L: bn(L, training=False))
    add("batch_norm::eps=0.1", lambda L: bn(L, training=False, eps=0.1))
    add("batch_norm::학습", lambda L: bn(L, training=True))
    add("batch_norm::가중치 없이",
        lambda L: F(L).batch_norm(L.tensor(bn_x), L.tensor(bn_rm.copy()),
                                  L.tensor(bn_rv.copy()), training=False))
    # 통계를 안 줘도 학습 모드는 이번 배치로 센다.
    add("batch_norm::통계 없이 학습",
        lambda L: F(L).batch_norm(L.tensor(bn_x), None, None, L.tensor(bn_w),
                                  L.tensor(bn_b), training=True))

    def bn_updates(L, momentum=0.1):
        """**갱신된 통계가 답이다.** 정규화는 편향 분산, 갱신은 비편향 분산을 쓴다 —
        둘 다 편향으로 두면 여기서만 2.6% 어긋난다."""
        rm, rv = L.tensor(bn_rm.copy()), L.tensor(bn_rv.copy())
        F(L).batch_norm(L.tensor(bn_x), rm, rv, training=True, momentum=momentum)
        return L.cat([rm, rv])

    add("batch_norm::갱신된 통계", bn_updates)
    add("batch_norm::갱신된 통계(momentum=0.5)",
        lambda L: bn_updates(L, 0.5))

    def bn_layer_matches(L):
        """**층과 함수가 같은 답이어야 한다** — 두 벌로 두면 언젠가 갈린다."""
        img = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2) / 10
        layer = L.nn.BatchNorm2d(3)
        layer.eval()
        got = layer(L.tensor(img))
        same = F(L).batch_norm(
            L.tensor(img), layer.running_mean, layer.running_var,
            layer.weight, layer.bias, training=False, eps=1e-5)
        return got - same

    add("batch_norm::층과 같은 답", bn_layer_matches)

    def bn_grad(L):
        x = L.tensor(bn_x, requires_grad=True)
        F(L).batch_norm(x, L.tensor(bn_rm.copy()), L.tensor(bn_rv.copy()),
                        L.tensor(bn_w), L.tensor(bn_b), training=True).sum().backward()
        return x.grad

    add("batch_norm::grad", bn_grad)

    # ── embedding_bag ──────────────────────────────────────────────────
    eb_table = np.arange(20, dtype=np.float32).reshape(5, 4) / 10
    eb_idx = np.array([[0, 2], [1, 4]], dtype=np.int64)
    for mode in ("mean", "sum", "max"):
        add(f"embedding_bag::{mode}",
            lambda L, m=mode: F(L).embedding_bag(
                L.tensor(eb_idx), L.tensor(eb_table), mode=m))
    # **1 차원 번호 줄 + `offsets`** — 가방 길이가 제각각인 자리다.
    add("embedding_bag::offsets",
        lambda L: F(L).embedding_bag(
            L.tensor(np.array([0, 2, 1, 4, 3], dtype=np.int64)),
            L.tensor(eb_table), L.tensor(np.array([0, 2], dtype=np.int64)),
            mode="sum"))
    add("embedding_bag::per_sample_weights",
        lambda L: F(L).embedding_bag(
            L.tensor(eb_idx), L.tensor(eb_table), mode="sum",
            per_sample_weights=L.tensor(
                np.array([[1.0, 2.0], [0.5, 0.5]], dtype=np.float32))))

    def eb_grad(L):
        table = L.tensor(eb_table, requires_grad=True)
        F(L).embedding_bag(L.tensor(eb_idx), table, mode="sum").sum().backward()
        return table.grad

    add("embedding_bag::grad", eb_grad)

    # ── gumbel_softmax ─────────────────────────────────────────────────
    #
    # 무작위라 값을 못 묻는다. **성질을 묻는다** — 행의 합이 1 이고, `hard` 는 0/1
    # 뿐이며 여전히 합이 1 이다. 그것들은 어떤 뽑기에서도 참이어야 한다.
    gs_logits = np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]], dtype=np.float32)

    def gs_soft(L):
        out = F(L).gumbel_softmax(L.tensor(gs_logits))
        rows = out.sum(dim=1)
        return f"{tuple(out.shape)} 합={[round(float(v), 4) for v in rows.tolist()]}"

    add("gumbel_softmax::행 합이 1", gs_soft)

    def gs_hard(L):
        out = F(L).gumbel_softmax(L.tensor(gs_logits), hard=True)
        vals = sorted({round(float(v), 6) for v in out.reshape(-1).tolist()})
        return f"값={vals} 합={[round(float(v), 4) for v in out.sum(dim=1).tolist()]}"

    add("gumbel_softmax::hard 는 0/1", gs_hard)

    def gs_grad(L):
        """**`hard` 여도 기울기가 흐른다** — 값은 0/1 이고 미분은 부드러운 쪽 것이다.

        그 갈라 둠이 이 함수의 요점이라, 기울기가 아예 안 오는 구현은 여기서만 걸린다.
        골고루 더하면 softmax 의 성질 때문에 0 이 나오므로 **한쪽에 무게를 준다.**
        """
        x = L.tensor(gs_logits, requires_grad=True)
        weights = L.tensor(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                                    dtype=np.float32))
        (F(L).gumbel_softmax(x, hard=True) * weights).sum().backward()
        return "기울기 있음" if x.grad is not None else "기울기가 안 왔다"

    add("gumbel_softmax::hard 에도 기울기가 흐른다", gs_grad)

    # ── 공간 변환기 ────────────────────────────────────────────────────
    #
    # `affine_grid` 가 "출력의 이 칸은 입력의 어디를 보는가" 를 적고 `grid_sample` 이
    # 그 자리에서 값을 떠 온다. 사이의 `theta` 가 학습되는 것이 요점이라, 사슬 전체로
    # 기울기가 가는지를 물어야 한다.
    #
    # ## 케이스 모양 함정 둘
    #
    # 1. **정사각으로만 물으면 `(x, y)` 순서를 못 본다.** 격자의 마지막 축은 `(x, y)`
    #    인데 모양은 `(H, W)` 라 뒤집혀 있다. 3×3 에서는 뒤집어 적어도 답이 같다.
    # 2. **기울기는 칸 안쪽에서 물어야 한다.** 90° 회전은 격자가 칸 경계에 정확히
    #    떨어지는데, 거기서는 `floor` 가 6e-8 차이에 뒤집혀 기울기가 통째로 달라진다
    #    (실측: `tests/probe_grid5.py`). 값은 그 자리에서도 안정하다 — 무게가 0 이라
    #    어느 쪽 모서리를 골라도 같은 값이 나온다. 그래서 **값은 회전으로, 기울기는
    #    비스듬한 `theta` 로** 묻는다. 경계에서 답이 갈리는 것은 결함이 아니라 그
    #    자리에 답이 없는 것이다.
    eye3 = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
    shift3 = np.array([[[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]], dtype=np.float32)
    flip3 = np.array([[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]], dtype=np.float32)
    rot3 = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    tilt3 = np.array([[[0.8, 0.2, 0.05], [-0.15, 0.9, -0.1]]], dtype=np.float32)
    img3 = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    rect24 = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)

    for name, th, size in (("항등", eye3, (1, 1, 3, 3)),
                           ("이동", shift3, (1, 1, 2, 2)),
                           ("뒤집기", flip3, (1, 1, 3, 3)),
                           ("회전", rot3, (1, 1, 3, 3)),
                           ("직사각 2x4", eye3, (1, 1, 2, 4))):
        for ac in (False, True):
            add(f"affine_grid::{name}(align={ac})",
                lambda L, t=th, s=size, a=ac: F(L).affine_grid(
                    L.tensor(t), s, align_corners=a))

    for ac in (False, True):
        for mode in ("bilinear", "nearest"):
            add(f"grid_sample::항등({mode}, align={ac})",
                lambda L, m=mode, a=ac: F(L).grid_sample(
                    L.tensor(img3),
                    F(L).affine_grid(L.tensor(eye3), (1, 1, 3, 3), align_corners=a),
                    mode=m, align_corners=a))
        add(f"grid_sample::뒤집기(align={ac})",
            lambda L, a=ac: F(L).grid_sample(
                L.tensor(img3),
                F(L).affine_grid(L.tensor(flip3), (1, 1, 3, 3), align_corners=a),
                align_corners=a))

    # 범위 밖을 가리키는 격자 — 세 가지 채우기가 여기서만 갈린다.
    out_grid = np.array([[[[-2.0, -2.0], [2.0, 2.0]],
                          [[0.0, 0.0], [-1.0, 1.0]]]], dtype=np.float32)
    for pad in ("zeros", "border", "reflection"):
        for ac in (False, True):
            add(f"grid_sample::padding={pad}(align={ac})",
                lambda L, p=pad, a=ac: F(L).grid_sample(
                    L.tensor(img3), L.tensor(out_grid), padding_mode=p,
                    align_corners=a))

    # 반 칸 어긋난 자리 — 겹선형이 실제로 섞는지는 여기서만 보인다.
    half_grid = np.array([[[[0.25, -0.3], [-0.6, 0.4]]]], dtype=np.float32)
    add("grid_sample::반 칸",
        lambda L: F(L).grid_sample(L.tensor(img3), L.tensor(half_grid),
                                   align_corners=False))
    add("grid_sample::직사각 입력",
        lambda L: F(L).grid_sample(L.tensor(rect24), L.tensor(half_grid),
                                   align_corners=False))

    planes33 = np.arange(2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3)
    add("grid_sample::여러 평면",
        lambda L: F(L).grid_sample(
            L.tensor(planes33), L.tensor(np.tile(out_grid, (2, 1, 1, 1))),
            align_corners=False))

    def grid_grad_input(L):
        x = L.tensor(img3, requires_grad=True)
        F(L).grid_sample(x, L.tensor(half_grid), align_corners=False).sum().backward()
        return x.grad

    add("grid_sample::grad(입력)", grid_grad_input)

    def grid_grad_grid(L):
        g = L.tensor(half_grid, requires_grad=True)
        F(L).grid_sample(L.tensor(img3), g, align_corners=False).sum().backward()
        return g.grad

    add("grid_sample::grad(격자)", grid_grad_grid)

    def grid_grad_theta(L):
        """**사슬 전체.** 공간 변환기가 `theta` 를 배우는 길이 이것이다.

        비스듬한 `theta` 를 쓴다 — 회전은 격자가 칸 경계에 떨어져 답이 불안정하다.
        """
        t = L.tensor(tilt3, requires_grad=True)
        F(L).grid_sample(
            L.tensor(img3),
            F(L).affine_grid(t, (1, 1, 3, 3), align_corners=False),
            align_corners=False).sum().backward()
        return t.grad

    add("grid_sample::grad(theta 까지)", grid_grad_theta)

    # ── multi_head_attention_forward ───────────────────────────────────
    #
    # `MultiheadAttention` 이 안에서 하는 계산을 이름으로 낸 것. torch 의 그 층도
    # 이 함수를 부른다.
    #
    # **입력이 `(L, N, E)` 다 — 길이가 앞이다.** 층은 `batch_first` 를 받지만 이
    # 함수는 늘 길이가 앞이라, 배치를 앞에 두고 부르면 조용히 다른 축을 섞는다.
    # 그래서 `L != N` 인 모양으로 묻는다 — 같으면 축을 바꿔도 안 걸린다.
    mha_L, mha_N, mha_E, mha_H, mha_S = 3, 2, 4, 2, 3

    def mha_w(shape, spin=0.0):
        """**난수가 아니다.** TypeScript 쪽 케이스에도 같은 값을 적어야 하는데 난수
        생성기는 언어를 못 건넌다. 세는 값이라 양쪽이 같은 것을 만든다."""
        n = int(np.prod(shape))
        return (np.sin(np.arange(n, dtype=np.float64) + spin) * 0.5
                ).astype(np.float32).reshape(shape)

    mha_q = mha_w((mha_L, mha_N, mha_E), 0.0)
    mha_k = mha_w((mha_S, mha_N, mha_E), 0.7)
    mha_v = mha_w((mha_S, mha_N, mha_E), 1.3)
    mha_inw = mha_w((3 * mha_E, mha_E), 2.1)
    mha_inb = mha_w((3 * mha_E,), 0.4)
    mha_ow = mha_w((mha_E, mha_E), 1.9)
    mha_ob = mha_w((mha_E,), 2.6)

    def mha(L, **kw):
        return F(L).multi_head_attention_forward(
            L.tensor(mha_q), L.tensor(mha_k), L.tensor(mha_v), mha_E, mha_H,
            L.tensor(mha_inw), L.tensor(mha_inb), None, None, False, 0.0,
            L.tensor(mha_ow), L.tensor(mha_ob), True, **kw)

    add("mha::출력", lambda L: mha(L)[0])
    add("mha::가중치(머리 평균)", lambda L: mha(L)[1])
    # **머리마다** 돌려주는 갈래 — 평균만 물으면 머리를 섞어도 안 보인다.
    add("mha::가중치(머리마다)",
        lambda L: mha(L, average_attn_weights=False)[1])
    add("mha::need_weights=False",
        lambda L: mha(L, need_weights=False)[0])
    add("mha::가중치가 None 인가",
        lambda L: str(mha(L, need_weights=False)[1] is None))

    causal = np.triu(np.ones((mha_L, mha_S), dtype=bool), k=1)
    add("mha::불리언 가림막",
        lambda L: mha(L, attn_mask=L.tensor(causal))[0])
    # **실수 가림막은 더하는 것이다** — 0 이 아니면 가림으로 뭉뚱그리면 인과 마스크만
    # 우연히 맞는다.
    add("mha::실수 가림막",
        lambda L: mha(L, attn_mask=L.tensor(
            np.where(causal, -np.inf, 0.0).astype(np.float32)))[0])
    pad_mask = np.array([[False, False, True], [False, True, True]])
    add("mha::key_padding_mask",
        lambda L: mha(L, key_padding_mask=L.tensor(pad_mask))[0])
    add("mha::key_padding_mask 가중치",
        lambda L: mha(L, key_padding_mask=L.tensor(pad_mask))[1])
    add("mha::is_causal",
        lambda L: mha(L, attn_mask=L.tensor(causal))[0])

    def mha_layer_matches(L):
        """**층과 함수가 같은 답이어야 한다.** 층이 이 함수를 부르는지가 여기서 보인다."""
        layer = L.nn.MultiheadAttention(mha_E, mha_H)
        layer.load_state_dict({
            "in_proj_weight": L.tensor(mha_inw), "in_proj_bias": L.tensor(mha_inb),
            "out_proj.weight": L.tensor(mha_ow), "out_proj.bias": L.tensor(mha_ob)})
        got = layer(L.tensor(mha_q), L.tensor(mha_k), L.tensor(mha_v))[0]
        return got - mha(L)[0]

    add("mha::층과 같은 답", mha_layer_matches)

    def mha_grad(L):
        q = L.tensor(mha_q, requires_grad=True)
        F(L).multi_head_attention_forward(
            q, L.tensor(mha_k), L.tensor(mha_v), mha_E, mha_H,
            L.tensor(mha_inw), L.tensor(mha_inb), None, None, False, 0.0,
            L.tensor(mha_ow), L.tensor(mha_ob))[0].sum().backward()
        return q.grad

    add("mha::grad(query)", mha_grad)

    def mha_refuses(L):
        """**안 하는 갈래는 시끄럽게 거절한다** — 조용히 무시하면 값이 그럴듯하게 다르다."""
        try:
            F(L).multi_head_attention_forward(
                L.tensor(mha_q), L.tensor(mha_k), L.tensor(mha_v), mha_E, mha_H,
                L.tensor(mha_inw), L.tensor(mha_inb),
                L.tensor(np.zeros((1, 1, mha_E), dtype=np.float32)), None,
                False, 0.0, L.tensor(mha_ow), L.tensor(mha_ob))
            return "통과했다"
        except Exception:                                           # noqa: BLE001
            return "거절"

    add("mha::bias_k 는 거절", mha_refuses)
    return cases


TOP_PREFIX = "top::"


def top_level_cases(inp=None):
    """torch **최상위**에만 있는 이름들과, 살펴보는 것들.

    ## 최상위는 `F` 와 서명이 같지 않다

    torch 는 `nn.functional` 의 것을 최상위에도 두는데, 그쪽은 날 ATen 연산이라
    **인자 순서가 다르고 열거형이 정수다.**

    - `torch.batch_norm` 은 가중치가 이동 통계보다 **앞**이다. 그대로 넘기면
      가중치를 평균으로 쓴다 — 예외가 아니라 그럴듯하게 다른 값이다.
    - `torch.grid_sampler` 는 `mode` 가 0·1, 채우기가 0·1·2 다.
    - `torch.ctc_loss` 는 `reduction` 이 0·1·2 이고 **기본이 1(mean)** 이다.

    같은 계산인데 부르는 법이 다른 것이라, 계산은 한 벌만 두고 자리만 옮긴다.
    그 옮김이 맞는지는 값으로만 확인된다.

    ## 살펴보는 것들은 값이 아니라 판정이다

    `is_floating_point`·`can_cast`·`typename` 은 교재 코드가 **분기에 쓰는** 자리다.
    없으면 계산이 다 맞아도 그 줄에서 멈추고, 틀리면 다른 가지로 간다.
    """
    cases = []

    def add(name, fn):
        cases.append((TOP_PREFIX + name, fn))

    holes = np.array([[-1.0, 0.5, np.nan], [0.25, np.inf, 1.0]], dtype=np.float32)
    img = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    plain = np.array([[-1.0, 0.5, 2.0], [0.25, -3.0, 1.0]], dtype=np.float32)

    # ── 최상위 전용 제자리들 ────────────────────────────────────────────
    inplace = (("nan_to_num_", holes, ()),
               ("dropout_", img, (0.0, True)),
               ("feature_dropout_", img, (0.0, True)),
               ("alpha_dropout_", img, (0.0, False)),
               ("feature_alpha_dropout_", img, (0.0, False)))
    for name, src, args in inplace:
        def run(L, n=name, s=src, a=args):
            x = L.tensor(s.copy())
            getattr(L, n)(x, *a)
            return x

        add(f"제자리::{name}", run)

        def is_self(L, n=name, s=src, a=args):
            x = L.tensor(s.copy())
            return str(getattr(L, n)(x, *a) is x)

        add(f"제자리::{name}(같은 텐서)", is_self)

    # `feature_dropout` 은 **채널째** 떨군다 — `dropout2d` 와 같은 계산이다.
    add("feature_dropout(p=0)",
        lambda L: L.feature_dropout(L.tensor(img), 0.0, True))

    # ── 날 ATen 서명 ────────────────────────────────────────────────────
    bn_y = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    add("batch_norm(최상위 서명)",
        lambda L: L.batch_norm(
            L.tensor(bn_y), L.tensor(np.array([1.5, 0.5], dtype=np.float32)),
            L.tensor(np.array([0.1, -0.1], dtype=np.float32)),
            L.tensor(np.array([0.2, 0.3], dtype=np.float32)),
            L.tensor(np.array([1.0, 2.0], dtype=np.float32)),
            False, 0.1, 1e-5, False))

    line = np.arange(8, dtype=np.float32).reshape(1, 1, 8)
    add("max_pool1d_with_indices(값)",
        lambda L: L.max_pool1d_with_indices(L.tensor(line), 2, 2, 0, 1, False)[0])
    add("max_pool1d_with_indices(자리)",
        lambda L: L.max_pool1d_with_indices(L.tensor(line), 2, 2, 0, 1, False)[1])

    eye = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
    quad = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    for pad, tag in ((0, "zeros"), (1, "border")):
        add(f"grid_sampler(정수 열거형, {tag})",
            lambda L, p=pad: L.grid_sampler(
                L.tensor(quad),
                L.nn.functional.affine_grid(L.tensor(eye), (1, 1, 2, 2),
                                            align_corners=False),
                0, p, False))

    ctc_lp = np.arange(20, dtype=np.float32).reshape(5, 1, 4)
    for red, tag in ((0, "none"), (1, "mean"), (2, "sum")):
        add(f"ctc_loss(정수 reduction={tag})",
            lambda L, r=red: L.ctc_loss(
                L.nn.functional.log_softmax(L.tensor(ctc_lp), dim=2),
                L.tensor(np.array([[1, 2]], dtype=np.int64)),
                L.tensor(np.array([5], dtype=np.int64)),
                L.tensor(np.array([2], dtype=np.int64)), 0, r, False))

    # ── 기울기 모드 ─────────────────────────────────────────────────────
    def grad_modes(L):
        """**중첩이 되어야 한다** — `no_grad` 안에서 `enable_grad` 가 다시 켠다.
        나갈 때 원래 값으로 돌아가는지도 여기서만 보인다."""
        seen = [L.is_grad_enabled()]
        with L.no_grad():
            seen.append(L.is_grad_enabled())
            with L.enable_grad():
                seen.append(L.is_grad_enabled())
            seen.append(L.is_grad_enabled())
        with L.set_grad_enabled(False):
            seen.append(L.is_grad_enabled())
        seen.append(L.is_grad_enabled())
        return " ".join(str(v) for v in seen)

    add("기울기 모드::중첩", grad_modes)
    add("기울기 모드::is_inference_mode_enabled",
        lambda L: str(L.is_inference_mode_enabled()))

    def inference(L):
        with L.inference_mode():
            return str(L.is_grad_enabled())

    add("기울기 모드::inference_mode 안", inference)

    # ── 살펴보기 ───────────────────────────────────────────────────────
    ints = np.array([1, 2], dtype=np.int64)
    checks = (
        ("is_tensor", lambda L: (L.is_tensor(L.tensor(plain)), L.is_tensor(3))),
        ("is_floating_point", lambda L: (L.is_floating_point(L.tensor(plain)),
                                         L.is_floating_point(L.tensor(ints)))),
        ("is_nonzero", lambda L: (L.is_nonzero(L.tensor(np.float32(3))),
                                  L.is_nonzero(L.tensor(np.float32(0))))),
        ("is_same_size", lambda L: (L.is_same_size(L.tensor(plain),
                                                   L.tensor(plain)),
                                    L.is_same_size(L.tensor(plain),
                                                   L.tensor(img)))),
        ("is_signed", lambda L: (L.is_signed(L.tensor(plain)),)),
        ("is_storage", lambda L: (L.is_storage(L.tensor(plain)),)),
        # **한 방향만 참이다** — 좁아지는 쪽은 거짓이다.
        ("can_cast", lambda L: (L.can_cast(L.int64, L.float32),
                                L.can_cast(L.float32, L.int64))),
    )
    for name, fn in checks:
        add(f"살펴보기::{name}", lambda L, f=fn: " ".join(str(v) for v in f(L)))

    add("살펴보기::typename",
        lambda L: f"{L.typename(L.tensor(plain))} {L.typename(3)}")
    add("살펴보기::promote_types",
        lambda L: str(L.promote_types(L.int64, L.float32)))
    add("살펴보기::get_default_dtype", lambda L: str(L.get_default_dtype()))
    add("살펴보기::finfo",
        lambda L: (f"{L.finfo(L.float32).eps:.9g} {L.finfo(L.float32).max:.9g} "
                   f"{L.finfo(L.float32).bits}"))
    add("살펴보기::iinfo",
        lambda L: f"{L.iinfo(L.int64).max} {L.iinfo(L.int64).min}")

    def rng_round_trip(L):
        """**상태를 되돌리면 같은 수가 나와야 한다.** 그 왕복이 이어서 학습하기다.

        torch 는 상태를 바이트 텐서로 주고 우리는 우리 생성기의 상태를 준다 —
        모양이 다르므로 **값이 아니라 왕복이 되는가**를 묻는다. 답할 수 없는 질문을
        표에 넣으면 그 표가 무엇을 통과했는지 못 말한다.
        """
        L.manual_seed(7)
        state = L.get_rng_state()
        first = L.randn(3)
        L.set_rng_state(state)
        second = L.randn(3)
        return f"왕복={bool(L.allclose(first, second))} 씨앗={L.initial_seed()}"

    add("난수::상태 왕복", rng_round_trip)
    return cases


UNPOOL_PREFIX = "unpool::"


def unpool_cases(inp=None):
    """이긴 자리를 함께 내는 풀링과, 그 자리로 되돌리는 짝.

    ## 왜 자리표가 따로 필요한가

    최대 풀링은 창마다 하나만 남기고 나머지를 버린다. 값만 보면 **어느 칸이 이겼는지가
    없다** — 그래서 `MaxUnpool` 은 값만으로는 못 돌아간다. torch 는 풀링에게 자리표를
    같이 내게 하고(`return_indices=True`) 그것을 되돌리기에 넘긴다. 자동 부호기에서
    흔한 짝이다.

    ## 자리 번호의 규약

    **평면 안의 평평한 번호**다 — 2차원이면 `h*W + w`, 배치와 채널마다 0 부터 다시
    센다. 재봤다(`tests/probe_pool.py`). 이것을 전체 텐서 기준으로 착각하면 배치가
    하나일 때만 맞는다.

    ## 여기서 묻는 것

    자리표는 값이 아니라 **정수 표**라, 값 대조로는 근처만 맞아도 통과할 수가 없다 —
    한 칸만 어긋나도 정수가 달라진다. 그래서 자리표 자체를 답으로 굳힌다.

    같은 값이 둘일 때 누가 이기는가도 여기 있다. torch 는 평평한 번호가 작은 쪽,
    즉 **행 우선으로 먼저 나오는 자리**를 고른다. 축을 앞에서부터 접으면 열 우선
    첫째가 나오는데 **값은 같으므로 아무 값 케이스에도 안 걸린다.**
    """
    cases = []

    def add(name, fn):
        cases.append((UNPOOL_PREFIX + name, fn))

    def F(L):
        return L.nn.functional

    def grid(*shape):
        return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)

    plane = grid(1, 1, 4, 4)
    planes = grid(2, 2, 4, 4)
    line = grid(1, 1, 8)
    cube = grid(1, 1, 4, 4, 4)
    odd = grid(1, 1, 3, 3)

    # ── 자리표 ──────────────────────────────────────────────────────────
    pools = (
        ("max_pool1d", line, lambda L, x: F(L).max_pool1d(x, 2, return_indices=True)),
        ("max_pool2d", plane, lambda L, x: F(L).max_pool2d(x, 2, return_indices=True)),
        ("max_pool2d(stride=1)", plane,
         lambda L, x: F(L).max_pool2d(x, 2, stride=1, return_indices=True)),
        ("여러 평면", planes,
         lambda L, x: F(L).max_pool2d(x, 2, return_indices=True)),
        ("max_pool3d", cube, lambda L, x: F(L).max_pool3d(x, 2, return_indices=True)),
        ("적응형", plane,
         lambda L, x: F(L).adaptive_max_pool2d(x, 2, return_indices=True)),
        # **나누어떨어지지 않는 적응형** — 창 크기가 자리마다 다르다.
        ("적응형(3→2)", odd,
         lambda L, x: F(L).adaptive_max_pool2d(x, 2, return_indices=True)),
        ("적응형 1차원", line,
         lambda L, x: F(L).adaptive_max_pool1d(x, 4, return_indices=True)),
        ("적응형 3차원", cube,
         lambda L, x: F(L).adaptive_max_pool3d(x, 2, return_indices=True)),
    )
    for name, src, call in pools:
        add(f"자리::{name}",
            lambda L, s=src, c=call: c(L, L.tensor(s))[1])
        add(f"값::{name}",
            lambda L, s=src, c=call: c(L, L.tensor(s))[0])

    # **이름이 둘인 같은 계산.** `return_indices=True` 와 `*_with_indices` 다.
    def two_names(L):
        x = L.tensor(plane)
        a = F(L).max_pool2d(x, 2, return_indices=True)
        b = F(L).max_pool2d_with_indices(x, 2)
        c = F(L).adaptive_max_pool2d_with_indices(x, 2)
        return L.cat([a[0].reshape(-1), b[0].reshape(-1), c[0].reshape(-1),
                      a[1].reshape(-1).float(), b[1].reshape(-1).float()])

    add("이름이 둘인 같은 계산", two_names)

    # **자리를 켜도 값은 그대로여야 한다.** 두 경로가 갈리면 여기서만 보인다.
    def same_value(L):
        x = L.tensor(plane)
        return F(L).max_pool2d(x, 2) - F(L).max_pool2d(x, 2, return_indices=True)[0]

    add("자리를 켜도 값은 같다", same_value)

    # ── 되돌리기 ────────────────────────────────────────────────────────
    def unpool(L, src, dim, **kw):
        pool = getattr(F(L), f"max_pool{dim}d")
        out, idx = pool(L.tensor(src), 2, return_indices=True)
        return getattr(F(L), f"max_unpool{dim}d")(out, idx, 2, **kw)

    add("되돌리기::1차원", lambda L: unpool(L, line, 1))
    add("되돌리기::2차원", lambda L: unpool(L, plane, 2))
    add("되돌리기::3차원", lambda L: unpool(L, cube, 3))
    add("되돌리기::여러 평면", lambda L: unpool(L, planes, 2))
    # 풀링이 버린 자투리는 되살릴 수 없어서 torch 가 크기를 직접 주는 길을 연다.
    add("되돌리기::output_size",
        lambda L: unpool(L, plane, 2, output_size=(5, 5)))

    def unpool_stride(L):
        """창이 겹치면 되돌린 자리도 겹친다 — 나중 것이 이긴다(더하기가 아니다)."""
        x = L.tensor(plane)
        out, idx = F(L).max_pool2d(x, 2, stride=1, return_indices=True)
        return F(L).max_unpool2d(out, idx, 2, stride=1)

    add("되돌리기::겹치는 창", unpool_stride)

    # ── 층 ─────────────────────────────────────────────────────────────
    def layer_pair(L):
        pool = L.nn.MaxPool2d(2, return_indices=True)
        unpool_layer = L.nn.MaxUnpool2d(2)
        out, idx = pool(L.tensor(plane))
        return unpool_layer(out, idx)

    add("층::MaxPool2d → MaxUnpool2d", layer_pair)

    def layer_adaptive(L):
        pool = L.nn.AdaptiveMaxPool2d(2, return_indices=True)
        return pool(L.tensor(plane))[1]

    add("층::AdaptiveMaxPool2d 자리", layer_adaptive)

    for dim in (1, 2, 3):
        add(f"층::repr::MaxUnpool{dim}d",
            lambda L, d=dim: repr(getattr(L.nn, f"MaxUnpool{d}d")(2)))

    # ── 기울기 ──────────────────────────────────────────────────────────
    def grad_pool(L):
        x = L.tensor(plane, requires_grad=True)
        out, _ = F(L).max_pool2d(x, 2, return_indices=True)
        (out * 2).sum().backward()
        return x.grad

    add("grad::자리 판의 풀링", grad_pool)

    def grad_unpool(L):
        pooled = L.tensor(grid(1, 1, 2, 2), requires_grad=True)
        _, idx = F(L).max_pool2d(L.tensor(plane), 2, return_indices=True)
        F(L).max_unpool2d(pooled, idx, 2).sum().backward()
        return pooled.grad

    add("grad::되돌리기", grad_unpool)

    # ── LPPool3d ───────────────────────────────────────────────────────
    small = grid(1, 1, 4, 4, 4) / 8
    add("lp_pool3d", lambda L: F(L).lp_pool3d(L.tensor(small), 2, 2))
    add("lp_pool3d(p=1)", lambda L: F(L).lp_pool3d(L.tensor(small), 1, 2))
    add("층::LPPool3d", lambda L: L.nn.LPPool3d(2, 2)(L.tensor(small)))

    # ── 분수 최대 풀링 ──────────────────────────────────────────────────
    #
    # 창의 시작 자리를 표본이 흔든다. 케이스 모양에 함정이 둘 있고 **둘 다 밟았다.**
    #
    # 1. **나누어떨어지면 표본이 아무 일도 안 한다.** 6→3 창 2 면 α 가 정확히 2 라
    #    무엇을 넣어도 같은 답이 나온다 — 무작위 부분이 통째로 안 보이는 모양이다.
    #    그래서 7→3 으로 묻는다.
    # 2. **두 축에 같은 표본을 주면 축 순서가 안 보인다.** ATen 은 2차원판에서
    #    표본을 (너비, 높이) 로 읽고 3차원판에서는 (깊이, 높이, 너비) 로 읽는다 —
    #    두 함수가 서로 어긋나 있다. 축마다 다른 표본을 줘야만 드러난다.
    frac = np.arange(49, dtype=np.float32).reshape(1, 1, 7, 7)
    frac3 = np.arange(343, dtype=np.float32).reshape(1, 1, 7, 7, 7)
    planes2 = np.arange(2 * 2 * 7 * 7, dtype=np.float32).reshape(2, 2, 7, 7)

    def frac2(L, samples, src=None, **kw):
        return F(L).fractional_max_pool2d(
            L.tensor(frac if src is None else src), 2,
            return_indices=True, _random_samples=L.tensor(samples), **kw)

    for u in (0.0, 0.25, 0.5, 0.75, 0.99):
        s = np.array([[[u, u]]], dtype=np.float32)
        add(f"분수::값(u={u})",
            lambda L, s=s: frac2(L, s, output_size=(3, 3))[0])
        add(f"분수::자리(u={u})",
            lambda L, s=s: frac2(L, s, output_size=(3, 3))[1])

    # **축마다 다른 표본** — 순서를 뒤집으면 여기서만 갈린다.
    axis_split = np.array([[[0.0, 0.75]]], dtype=np.float32)
    add("분수::축마다 다른 표본",
        lambda L: frac2(L, axis_split, output_size=(3, 3))[1])

    # 평면마다 다른 표본 — 표본이 `(N, C, 축)` 이라 창이 평면마다 갈린다.
    per_plane = np.array([[[0.0, 0.0], [0.3, 0.7]],
                          [[0.9, 0.1], [0.5, 0.5]]], dtype=np.float32)
    add("분수::평면마다 다른 표본",
        lambda L: frac2(L, per_plane, src=planes2, output_size=(3, 3))[0])
    add("분수::평면마다 다른 표본 자리",
        lambda L: frac2(L, per_plane, src=planes2, output_size=(3, 3))[1])

    zero2 = np.zeros((1, 1, 2), dtype=np.float32)
    add("분수::output_ratio",
        lambda L: frac2(L, zero2, output_ratio=(0.5, 0.5))[0])

    def frac_overlap(L):
        """창 3 에 출력 3 이면 창이 겹친다 — 겹쳐도 같은 규칙이다."""
        return F(L).fractional_max_pool2d(
            L.tensor(frac), 3, output_size=(3, 3), return_indices=True,
            _random_samples=L.tensor(zero2))[1]

    add("분수::겹치는 창", frac_overlap)

    # 3차원 — 세 축이 서로 다른 답을 내는 표본이라야 순서를 가른다.
    s3 = np.array([[[0.2, 0.0, 0.25]]], dtype=np.float32)
    for which, label in ((0, "값"), (1, "자리")):
        add(f"분수::3차원 {label}",
            lambda L, w=which: F(L).fractional_max_pool3d(
                L.tensor(frac3), 2, output_size=(3, 3, 3), return_indices=True,
                _random_samples=L.tensor(s3))[w])

    add("분수::이름이 둘인 같은 계산",
        lambda L: F(L).fractional_max_pool2d_with_indices(
            L.tensor(frac), 2, output_size=(3, 3),
            _random_samples=L.tensor(axis_split))[0])

    def frac_grad(L):
        x = L.tensor(frac, requires_grad=True)
        s = L.tensor(np.array([[[0.25, 0.75]]], dtype=np.float32))
        F(L).fractional_max_pool2d(x, 2, output_size=(3, 3),
                                   _random_samples=s).sum().backward()
        return x.grad

    add("분수::grad", frac_grad)

    def frac_layer(L):
        layer = L.nn.FractionalMaxPool2d(2, output_size=(3, 3),
                                         _random_samples=L.tensor(axis_split))
        return layer(L.tensor(frac))

    add("층::FractionalMaxPool2d", frac_layer)
    # **`repr` 이 비어 있다** — torch 의 `extra_repr` 가 아무것도 안 낸다.
    add("층::repr::FractionalMaxPool2d",
        lambda L: repr(L.nn.FractionalMaxPool2d(2, output_size=(3, 3))))
    add("층::repr::FractionalMaxPool3d",
        lambda L: repr(L.nn.FractionalMaxPool3d(2, output_size=(3, 3, 3))))

    def frac_random_shape(L):
        """표본을 안 주면 무작위다 — **값은 못 묻고 모양과 범위를 묻는다.**

        어느 칸이 이기든 그 값은 제 창 안에 있고, 창은 반드시 입력 안이다. 그래서
        "모양이 맞고 값이 전부 입력에 있던 수" 는 무작위와 상관없이 참이어야 한다.
        """
        out = L.nn.FractionalMaxPool2d(2, output_size=(3, 3))(L.tensor(frac))
        inside = ((out >= 0).float() * (out <= 48).float()).sum()
        return f"{tuple(out.shape)} 안에 있는 것={int(inside.item())}"

    add("분수::표본 없이(모양과 범위)", frac_random_shape)

    # ── CTC ────────────────────────────────────────────────────────────
    #
    # 소리와 글자를 자리를 맞추지 않고 잇는 손실. 가능한 정렬을 전부 더한다.
    #
    # `reduction="mean"` 이 예사롭지 않다 — 표본마다 **제 표적 길이로 나눈 뒤**
    # 평균한다. 표적 길이가 다 같은 케이스로 물으면 그냥 평균과 답이 같아서 그
    # 나눗셈이 안 보인다. 그래서 길이를 2 와 1 로 어긋나게 준다.
    ctc_T, ctc_N, ctc_C = 5, 2, 4
    ctc_logits = (np.arange(ctc_T * ctc_N * ctc_C, dtype=np.float32)
                  .reshape(ctc_T, ctc_N, ctc_C) / 10)
    ctc_targets = np.array([[1, 2], [3, 0]], dtype=np.int64)
    ctc_in = np.array([5, 5], dtype=np.int64)
    ctc_tgt = np.array([2, 1], dtype=np.int64)

    def logp(L, grad=False):
        return F(L).log_softmax(L.tensor(ctc_logits, requires_grad=grad), dim=2)

    def ctc(L, targets=None, il=None, tl=None, **kw):
        return F(L).ctc_loss(
            logp(L),
            L.tensor(ctc_targets if targets is None else targets),
            L.tensor(ctc_in if il is None else il),
            L.tensor(ctc_tgt if tl is None else tl), **kw)

    for red in ("mean", "sum", "none"):
        add(f"ctc::reduction={red}", lambda L, r=red: ctc(L, reduction=r))

    # 표적은 `(N, S)` 로도 오고 이어붙인 1차원으로도 온다 — torch 가 둘 다 받는다.
    add("ctc::1차원 표적",
        lambda L: ctc(L, targets=np.array([1, 2, 3], dtype=np.int64),
                      reduction="none"))
    # 공백이 0 이 아닐 수도 있다.
    add("ctc::blank=3",
        lambda L: ctc(L, targets=np.array([[1, 2], [0, 0]], dtype=np.int64),
                      blank=3, reduction="none"))
    add("ctc::입력 길이가 다를 때",
        lambda L: ctc(L, il=np.array([5, 3], dtype=np.int64), reduction="none"))
    # **같은 글자가 이어지면 사이에 공백이 반드시 든다** — 안 그러면 한 글자로 접힌다.
    add("ctc::반복 글자",
        lambda L: ctc(L, targets=np.array([[1, 1], [1, 1]], dtype=np.int64),
                      tl=np.array([2, 2], dtype=np.int64), reduction="none"))

    long_target = np.array([[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]], dtype=np.int64)
    short_in = np.array([2, 2], dtype=np.int64)
    six = np.array([6, 6], dtype=np.int64)
    # 정렬이 하나도 없으면 확률이 0 이고 손실이 `inf` 다. 문턱값이 아니라 실제 조건이다.
    add("ctc::표적이 입력보다 길 때",
        lambda L: ctc(L, targets=long_target, il=short_in, tl=six, reduction="none"))
    add("ctc::zero_infinity",
        lambda L: ctc(L, targets=long_target, il=short_in, tl=six,
                      reduction="none", zero_infinity=True))

    def ctc_grad_logits(L):
        """로짓까지 흘린 기울기 — 진짜 코드가 하는 모양이다."""
        x = L.tensor(ctc_logits, requires_grad=True)
        F(L).ctc_loss(F(L).log_softmax(x, dim=2), L.tensor(ctc_targets),
                      L.tensor(ctc_in), L.tensor(ctc_tgt),
                      reduction="sum").backward()
        return x.grad

    add("ctc::grad(로짓까지)", ctc_grad_logits)

    def ctc_grad_logp(L):
        """**`log_probs` 를 바로 잎으로 둔 자리.**

        여기서 torch 는 참도함수가 아닌 값을 낸다 — 유한차분은 `-γ` 인데 torch 는
        `exp(log_probs) - γ` 다. 위 케이스는 `log_softmax` 를 지나므로 둘이 같은
        답이 되어 **그 차이를 못 본다.** 이 케이스만 그것을 본다.
        """
        base = np.log(np.exp(ctc_logits)
                      / np.exp(ctc_logits).sum(axis=2, keepdims=True))
        x = L.tensor(base.astype(np.float32), requires_grad=True)
        F(L).ctc_loss(x, L.tensor(ctc_targets), L.tensor(ctc_in),
                      L.tensor(ctc_tgt), reduction="sum").backward()
        return x.grad

    add("ctc::grad(log_probs 까지)", ctc_grad_logp)

    def ctc_layer(L):
        layer = L.nn.CTCLoss()
        return layer(logp(L), L.tensor(ctc_targets), L.tensor(ctc_in),
                     L.tensor(ctc_tgt))

    add("층::CTCLoss", ctc_layer)
    add("층::CTCLoss(blank=3, sum)",
        lambda L: L.nn.CTCLoss(blank=3, reduction="sum")(
            logp(L), L.tensor(np.array([[1, 2], [0, 0]], dtype=np.int64)),
            L.tensor(ctc_in), L.tensor(ctc_tgt)))
    add("층::repr::CTCLoss", lambda L: repr(L.nn.CTCLoss()))
    add("층::repr::CTCLoss(인자 있음)",
        lambda L: repr(L.nn.CTCLoss(blank=2, reduction="sum", zero_infinity=True)))

    # ── AdaptiveLogSoftmaxWithLoss ─────────────────────────────────────
    #
    # 어휘가 클 때의 softmax. 자주 나오는 글자는 머리에서 바로 내고, 드문 것은
    # 뭉치를 고른 확률에 뭉치 안의 확률을 **곱해서**(로그에서는 더해서) 낸다.
    #
    # 기본값이 함정이다 — `div_value=4.0`·`head_bias=False`. 2.0 으로 알고 물으면
    # 꼬리 층의 모양이 통째로 달라진다. 중간 차원이 **0 으로 떨어지는 것도** 정상이라
    # 골든이 그 모양까지 묻는다.
    asm_N, asm_D, asm_C = 6, 4, 12
    asm_x = (np.arange(asm_N * asm_D, dtype=np.float32).reshape(asm_N, asm_D) / 10) - 1
    asm_y = np.array([0, 1, 5, 7, 10, 11], dtype=np.int64)
    def asm_w(shape):
        """**난수가 아니다.** 이 가중치는 TypeScript 쪽 케이스에도 똑같이 적어야 하는데
        난수 생성기는 언어를 못 건넌다. 셀 수 있는 값이라 양쪽이 같은 것을 적는다."""
        n = int(np.prod(shape))
        return ((np.arange(n, dtype=np.float32) / n) - 0.5).reshape(shape)

    asm_weights = {
        "head.weight": asm_w((5, 4)),
        "tail.0.0.weight": asm_w((2, 4)),
        "tail.0.1.weight": asm_w((4, 2)),
        "tail.1.0.weight": asm_w((1, 4)),
        "tail.1.1.weight": asm_w((5, 1)),
    }

    def asm(L, **kw):
        """**가중치를 심는다** — 초기화가 갈리면 값 대조가 대조가 아니다."""
        model = L.nn.AdaptiveLogSoftmaxWithLoss(asm_D, asm_C, [3, 7],
                                                div_value=2.0, **kw)
        model.load_state_dict({k: L.tensor(v) for k, v in asm_weights.items()})
        return model

    add("적응형softmax::log_prob", lambda L: asm(L).log_prob(L.tensor(asm_x)))
    # **행마다 확률의 합이 1 이어야 한다** — 뭉치를 고른 확률을 안 더하면 여기서 깨진다.
    add("적응형softmax::행 합이 1",
        lambda L: asm(L).log_prob(L.tensor(asm_x)).exp().sum(dim=1))
    add("적응형softmax::output",
        lambda L: asm(L)(L.tensor(asm_x), L.tensor(asm_y)).output)
    add("적응형softmax::loss",
        lambda L: asm(L)(L.tensor(asm_x), L.tensor(asm_y)).loss)
    add("적응형softmax::predict",
        lambda L: asm(L).predict(L.tensor(asm_x)))

    def asm_output_is_gathered(L):
        """`output` 은 `log_prob` 에서 정답 자리를 고른 것과 같아야 한다.

        torch 는 필요한 뭉치만 골라 더 싸게 내는데, 두 길이 갈리면 학습만 조금씩
        어긋난다. 값으로 묶어 둔다.
        """
        model = asm(L)
        x = L.tensor(asm_x)
        lp = model.log_prob(x)
        picked = lp.gather(1, L.tensor(asm_y.reshape(-1, 1)))
        return model(x, L.tensor(asm_y)).output - picked.reshape(-1)

    add("적응형softmax::output 은 고른 것과 같다", asm_output_is_gathered)

    def asm_grad(L):
        x = L.tensor(asm_x, requires_grad=True)
        asm(L)(x, L.tensor(asm_y)).loss.backward()
        return x.grad

    add("적응형softmax::grad", asm_grad)

    def asm_shapes(L, **kw):
        """**기본값 `div_value=4.0`** 에서는 꼬리 차원이 0 으로도 떨어진다.

        torch 는 그 자리에서 빈 층을 만들고 넘어간다. 막지 않는 것이 흉내다 —
        코어는 √0 으로 나누다 멈춘 적이 있다.
        """
        model = L.nn.AdaptiveLogSoftmaxWithLoss(asm_D, asm_C, [3, 7], **kw)
        sd = model.state_dict()
        return " ".join(f"{k}{tuple(sd[k].shape)}" for k in sorted(sd))

    add("적응형softmax::기본값의 모양", asm_shapes)
    add("적응형softmax::div_value=2 의 모양",
        lambda L: asm_shapes(L, div_value=2.0))
    add("적응형softmax::head_bias 의 열쇠",
        lambda L: asm_shapes(L, head_bias=True))
    add("층::repr::AdaptiveLogSoftmaxWithLoss", lambda L: repr(asm(L)))
    return cases


DATACONV_PREFIX = "dataconv::"


def default_convert_cases(inp=None):
    """`utils.data.default_convert` — **바뀌는 것과 안 바뀌는 것**.

    `default_collate` 의 짝이고 이름도 비슷한데 규칙이 다르다. 함정이 둘이고, 둘 다
    값으로 물어서는 안 보인다.

    - **튜플이 리스트가 된다.** torch 자신이 남긴 하위 호환이다. 값만 보면 원소가
      같으니 통과하고, 그 뒤에 `a, b = ...` 로 푸는 코드도 리스트에서 똑같이 돌아서
      한참 뒤에야 갈린다.
    - **파이썬 수는 안 바뀐다.** `3` 은 `3` 으로 나온다. 이름이 비슷한
      `default_collate` 는 수를 텐서로 접으므로, 같을 것 같은데 아니다.

    그래서 값이 아니라 **무엇이 되어 나왔는가**를 답으로 굳힌다.
    """
    cases = []

    def add(name, fn):
        cases.append((DATACONV_PREFIX + name, fn))

    Point = collections.namedtuple("Point", "x y")
    one = np.array([1., 2.], dtype=np.float32)
    two = np.array([3., 4.], dtype=np.float32)

    def kinds(L):
        conv = L.utils.data.default_convert
        got = (conv(one), conv((one, two)), conv([one]), conv({"a": one}),
               conv(Point(one, two)), conv(3), conv(2.5), conv("안녕"),
               conv(L.tensor(one)))
        return " ".join(type(g).__name__ for g in got)

    add("무엇이 되어 나오는가", kinds)

    def values(L):
        """값도 옮겨져야 한다 — 껍데기만 바꾸고 숫자를 흘리면 위 케이스는 통과한다."""
        conv = L.utils.data.default_convert
        pair = conv((one, two))
        inside = conv({"a": one})["a"]
        return L.stack([pair[0], pair[1], inside])

    add("값이 그대로 온다", values)

    def worker(L):
        """**주 프로세스에서는 `None`.** 여기는 일꾼이 없으므로 언제나 그렇다."""
        return str(L.utils.data.get_worker_info())

    add("get_worker_info", worker)
    return cases


METHOD2_PREFIX = "method2::"


def method_name_cases(inp=None):
    """**같은 계산에 이름이 둘이다** — `torch.add(x, y)` 와 `x.add(y)`.

    이 저장소는 한 방향 고리만 갖고 있었다: 텐서 메서드를 훑어 모듈 함수를 만드는
    것. 반대 방향이 없어서 **계산은 다 해 놓고 이름이 한쪽에서만 닿았다** —
    `borch.matrix_exp(x)` 는 되고 `x.matrix_exp()` 는 안 됐다. 교재 코드에서
    `x.add(y)` 는 아주 흔한 꼴이고, 그때 나는 것은 `AttributeError` 다.

    ## 제자리 연산도 같은 이야기다

    `abs` 가 있는데 `abs_` 만 없는 자리가 마흔일곱이었다. 계산은 밑줄 없는 쪽이
    하고 여기서는 **제 버퍼에 되쓰는 것**만 한다 — 같은 식을 두 벌로 두면 갈린다.

    ## 여기서는 **둘이 같은 답인지**를 묻는다

    이름이 닿는지만 보면 껍데기만 있는 메서드도 통과한다. 값을 물어야 그 이름이
    진짜 그 계산에 닿았는지가 드러난다. 이름 목록이 진짜 torch 의 것인지는
    `tests/test_tensor_api.py` 가 따로 본다 — 없는 이름을 만들면 그것에 기대어 짠
    코드가 진짜 torch 에서 안 돈다.
    """
    cases = []

    def add(name, fn):
        cases.append((METHOD2_PREFIX + name, fn))

    a = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    b = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    sym = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    neg = np.array([[-1., 2.], [-3., 0.5]], dtype=np.float32)

    # ── 메서드로 부른 답 ────────────────────────────────────────────────
    pairs = (
        ("add", lambda L, x: x.add(L.tensor(b))),
        ("sub", lambda L, x: x.sub(L.tensor(b))),
        ("mul", lambda L, x: x.mul(L.tensor(b))),
        ("div", lambda L, x: x.div(L.tensor(b))),
        ("multiply", lambda L, x: x.multiply(L.tensor(b))),
        ("true_divide", lambda L, x: x.true_divide(L.tensor(b))),
        ("floor_divide", lambda L, x: x.floor_divide(L.tensor(b))),
        ("remainder", lambda L, x: x.remainder(L.tensor(b))),
        ("fmod", lambda L, x: x.fmod(2.0)),
        ("lerp", lambda L, x: x.lerp(L.tensor(b), 0.5)),
        ("greater", lambda L, x: x.greater(L.tensor(b))),
        ("less_equal", lambda L, x: x.less_equal(L.tensor(b))),
        ("logical_and", lambda L, x: x.logical_and(L.tensor(b))),
        ("logical_not", lambda L, x: x.logical_not()),
        ("isclose", lambda L, x: x.isclose(L.tensor(b))),
        ("nan_to_num", lambda L, x: x.nan_to_num()),
        ("fmax", lambda L, x: x.fmax(L.tensor(b))),
        ("inner", lambda L, x: x.inner(L.tensor(b))),
        # kron 은 축소판이 1 차원만 한다 — 아래에서 따로 묻는다.
        ("count_nonzero", lambda L, x: x.count_nonzero()),
        ("adjoint", lambda L, x: x.adjoint()),
        ("moveaxis", lambda L, x: x.moveaxis(0, 1)),
        ("t", lambda L, x: x.t()),
        ("det", lambda L, x: x.det()),
        ("inverse", lambda L, x: x.inverse()),
        ("matrix_exp", lambda L, x: x.matrix_exp()),
        ("matrix_power", lambda L, x: x.matrix_power(2)),
        ("pinverse", lambda L, x: x.pinverse()),
        ("qr", lambda L, x: x.qr()[1]),
        ("svd", lambda L, x: x.svd()[1]),
        ("lgamma", lambda L, x: x.lgamma()),
        ("digamma", lambda L, x: x.digamma()),
        ("log_softmax", lambda L, x: x.log_softmax(1)),
        ("hardshrink", lambda L, x: x.hardshrink()),
    )
    for name, fn in pairs:
        src = sym if name in ("cholesky",) else a
        add(name, lambda L, f=fn, s=src: f(L, L.tensor(s)))

    add("cholesky", lambda L: L.tensor(sym).cholesky())
    add("slogdet", lambda L: L.tensor(a).slogdet()[1])
    add("logdet", lambda L: L.tensor(sym).logdet())
    add("corrcoef", lambda L: L.tensor(a).corrcoef())
    add("cov", lambda L: L.tensor(a).cov())
    add("cross",
        lambda L: L.tensor(np.array([1., 2., 3.], dtype=np.float32)).cross(
            L.tensor(np.array([4., 5., 6.], dtype=np.float32))))
    add("vdot",
        lambda L: L.tensor(np.array([1., 2., 3.], dtype=np.float32)).vdot(
            L.tensor(np.array([4., 5., 6.], dtype=np.float32))))
    add("kron",
        lambda L: L.tensor(np.array([1., 2., 3.], dtype=np.float32)).kron(
            L.tensor(np.array([4., 5.], dtype=np.float32))))
    add("broadcast_to",
        lambda L: L.tensor(np.array([1., 2.], dtype=np.float32)).broadcast_to((3, 2)))
    add("prelu",
        lambda L: L.tensor(neg).prelu(
            L.tensor(np.array([0.25], dtype=np.float32))))

    # **함수로 부른 것과 같아야 한다.** 이름만 닿고 다른 계산이면 여기서 갈린다.
    def same_as_function(L):
        x, y = L.tensor(a), L.tensor(b)
        checks = (x.add(y) - L.add(x, y), x.mul(y) - L.mul(x, y),
                  x.det().reshape(1) - L.det(x).reshape(1),
                  (x.matrix_exp() - L.matrix_exp(x)).reshape(-1))
        return L.cat([c.reshape(-1) for c in checks])

    add("함수와 같은 답", same_as_function)

    # ── 제자리 연산 ─────────────────────────────────────────────────────
    # **케이스마다 배열을 새로 만든다.** 코어의 `tensor(ndarray)` 는 그 버퍼를 **공유**하고
    # (torch 로 치면 `from_numpy` 쪽), torch 의 `tensor()` 는 복사한다. 여기 쓰는 배열을
    # 밖에 한 벌 두었더니 첫 제자리 연산이 그것을 고쳐서, 뒤 케이스들이 torch 가 본 것과
    # **다른 입력**을 받았다 — 열세 건이 한꺼번에 갈렸는데 원인은 제자리 연산이 아니었다.
    def _small():
        return np.array([[0.25, 0.5], [0.75, -0.5]], dtype=np.float32)

    def _square():
        return np.array([[1., 2.], [3., 4.]], dtype=np.float32)

    inplace = ("absolute", "acosh", "arctan", "arctanh", "asinh", "atanh",
               "deg2rad", "erfc", "exp2", "fix", "negative", "rad2deg",
               "sgn", "sinc", "logit")
    for name in inplace:

        def run(L, n=name):
            # acosh 는 1 이상, logit 은 0..1 안에서만 답이 있다 — 밖은 양쪽 다 NaN 이고
            # NaN 은 서로 같다고 못 하므로 물어봐야 아무것도 안 드러난다.
            s = _small()
            if n == "acosh":
                s = np.abs(s) + 1.0
            elif n == "logit":
                s = np.abs(s) * 0.8 + 0.1
            x = L.tensor(s)
            getattr(x, n + "_")()
            return x

        add(f"제자리::{name}_", run)

    def inplace_args(L):
        """인자를 받는 제자리 연산 — 자리 수만 다르고 나머지는 같다."""
        x = L.tensor(_square())
        x.transpose_(0, 1)
        y = L.tensor(_square())
        y.tril_()
        z = L.tensor(_square())
        z.cumsum_(1)
        return L.cat([x.reshape(-1), y.reshape(-1), z.reshape(-1)])

    add("제자리::인자를 받는 것", inplace_args)

    def _rect():
        return np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)

    def inplace_changes_shape(L):
        """**모양을 바꾸는 제자리 연산이 있다** — 값이 아니라 보는 틀을 고친다.

        이것을 정사각으로 물으면 **모양이 안 바뀌어도 통과한다.** 실제로 2×2 로
        물었을 때 `transpose_` 가 아무 일도 안 하고 초록이었다. 여기서는 직사각을
        주고 모양 자체를 답으로 낸다 — 값이 아니라 그 틀이 물음이기 때문이다.
        """
        x = L.tensor(_rect())
        x.transpose_(0, 1)
        y = L.tensor(_rect())
        y.unsqueeze_(0)
        z = L.tensor(_rect()).reshape(1, 2, 3)
        z.squeeze_(0)
        return " ".join(str(tuple(t.shape)) for t in (x, y, z))

    add("제자리::모양이 바뀐다", inplace_changes_shape)

    def inplace_transpose_values(L):
        """모양뿐 아니라 **값도 옮겨 앉아야** 한다."""
        x = L.tensor(_rect())
        x.transpose_(0, 1)
        return x

    add("제자리::transpose_ 의 값", inplace_transpose_values)

    def inplace_is_same_object(L):
        """**제자리는 같은 텐서를 고친다.** 새것을 돌려주면 뜻이 없다."""
        x = L.tensor(_small())
        return "같은 것=" + str(x.absolute_() is x)

    add("제자리::같은 텐서인가", inplace_is_same_object)

    def inplace_refuses_leaf(L):
        x = L.tensor(_small(), requires_grad=True)
        try:
            x.absolute_()
            return "예외가 안 났다"
        except Exception as exc:                                    # noqa: BLE001
            return type(exc).__name__

    add("제자리::기울기 켜진 잎은 거절", inplace_refuses_leaf)
    return cases


CELL_PREFIX = "cell::"


def cell_cases(inp=None):
    """RNN 셀 셋 — 되풀이의 **한 걸음**.

    `RNN`·`LSTM`·`GRU` 는 시간 축을 통째로 받는다. 셀은 한 걸음만 떼는 것이라, 시간
    루프를 손으로 적고 싶은 코드(스케줄링·강제 교사·빔서치)가 이 이름을 부른다.

    ## 이름이 층 쪽과 다르다

    층은 `weight_ih_l0` 처럼 층 번호를 붙이고 셀은 `weight_ih` 다 — 셀에는 층이
    없기 때문이다. `state_dict` 열쇠가 그 이름이므로 **틀리면 체크포인트가 안 맞는다.**

    ## `LSTMCell` 만 둘을 돌려준다

    `(h, c)` 다. `RNNCell`·`GRUCell` 은 `h` 하나다 — 셋을 한 모양으로 두면 LSTM 의
    기억 칸이 사라지고, 그러면 값은 나오는데 학습이 안 된다.

    ## 게이트 순서가 값의 전부다

    `weight_ih` 가 `(3H, in)`·`(4H, in)` 인데 그 안의 순서가 규약이다 — GRU 는
    `r, z, n` 이고 LSTM 은 `i, f, g, o` 다. 순서를 바꾸면 모양이 같고 값만 갈린다.
    가중치를 못 박고 값을 묻는 이유가 그것이다.
    """
    cases = []

    def add(name, fn):
        cases.append((CELL_PREFIX + name, fn))

    x = np.array([[1., 2.]], dtype=np.float32)
    h = np.array([[0.5, 0.5]], dtype=np.float32)
    c0 = np.array([[0.2, 0.3]], dtype=np.float32)
    eye = np.eye(2, dtype=np.float32)

    def weights(L, gates, scale=0.5):
        return {
            "weight_ih": L.tensor(np.tile(eye, (gates, 1))),
            "weight_hh": L.tensor(np.tile(eye, (gates, 1)) * scale),
            "bias_ih": L.tensor(np.zeros(gates * 2, dtype=np.float32)),
            "bias_hh": L.tensor(np.zeros(gates * 2, dtype=np.float32)),
        }

    def rnn_cell(L, nonlinearity="tanh"):
        cell = L.nn.RNNCell(2, 2, nonlinearity=nonlinearity)
        cell.load_state_dict(weights(L, 1))
        return cell(L.tensor(x), L.tensor(h))

    add("RNNCell", lambda L: rnn_cell(L))
    add("RNNCell(relu)", lambda L: rnn_cell(L, "relu"))

    def rnn_cell_no_hidden(L):
        """**숨은 상태를 안 주면 0 에서 시작한다.** 첫 걸음이 그 모양이다."""
        cell = L.nn.RNNCell(2, 2)
        cell.load_state_dict(weights(L, 1))
        return cell(L.tensor(x))

    add("RNNCell(상태 없이)", rnn_cell_no_hidden)

    def gru_cell(L):
        cell = L.nn.GRUCell(2, 2)
        cell.load_state_dict(weights(L, 3))
        return cell(L.tensor(x), L.tensor(h))

    add("GRUCell", gru_cell)

    def lstm_cell(L, which):
        cell = L.nn.LSTMCell(2, 2)
        cell.load_state_dict(weights(L, 4))
        out = cell(L.tensor(x), (L.tensor(h), L.tensor(c0)))
        return out[0] if which == "h" else out[1]

    add("LSTMCell/h", lambda L: lstm_cell(L, "h"))
    add("LSTMCell/c", lambda L: lstm_cell(L, "c"))

    def lstm_cell_no_state(L):
        cell = L.nn.LSTMCell(2, 2)
        cell.load_state_dict(weights(L, 4))
        return cell(L.tensor(x))[0]

    add("LSTMCell(상태 없이)", lstm_cell_no_state)

    # ── 이름과 글자 ─────────────────────────────────────────────────────
    add("state_dict 열쇠",
        lambda L: ",".join(L.nn.RNNCell(3, 2).state_dict()))
    add("state_dict 열쇠(bias 없이)",
        lambda L: ",".join(L.nn.RNNCell(3, 2, bias=False).state_dict()))
    for name in ("RNNCell", "GRUCell", "LSTMCell"):
        add(f"repr::{name}", lambda L, c=name: repr(getattr(L.nn, c)(3, 2)))
    add("repr::RNNCell(relu)",
        lambda L: repr(L.nn.RNNCell(3, 2, nonlinearity="relu")))
    add("repr::RNNCell(bias 없이)",
        lambda L: repr(L.nn.RNNCell(3, 2, bias=False)))
    for name, gates in (("RNNCell", 1), ("GRUCell", 3), ("LSTMCell", 4)):
        add(f"모양::{name}",
            lambda L, c=name: str(tuple(getattr(L.nn, c)(3, 2).weight_ih.shape)))

    # ── 기울기 ──────────────────────────────────────────────────────────
    def cell_grad(L, name, gates):
        # **상태의 모양이 셀마다 다르다.** `LSTMCell` 만 `(h, c)` 짝을 받고 나머지는
        # `h` 하나다 — 하나로 뭉뚱그리면 torch 가 그 자리에서 거절한다.
        cell = getattr(L.nn, name)(2, 2)
        cell.load_state_dict(weights(L, gates))
        inp = L.tensor(x, requires_grad=True)
        state = ((L.tensor(h), L.tensor(c0)) if name == "LSTMCell"
                 else L.tensor(h))
        out = cell(inp, state)
        out = out[0] if name == "LSTMCell" else out
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(inp, name)

    for name, gates in (("RNNCell", 1), ("GRUCell", 3), ("LSTMCell", 4)):
        add(f"grad::{name}", lambda L, c=name, g=gates: cell_grad(L, c, g))
    return cases


MISC_PREFIX = "misc::"


def misc_cases(inp=None):
    """남은 층 아홉 — 창을 펴는 둘과 나머지.

    ## `Unfold` 와 `Fold` 는 서로의 역이 아니다

    `Unfold` 는 창을 열로 펴고 `Fold` 는 그것을 되접는데, **겹친 자리를 더한다.**
    4×4 를 2×2 창으로 펴서 그대로 되접으면 가운데 값이 네 번 세어져 원본이 안 나온다
    (실측: `[[0,2,4,3],[8,20,24,14],…]`). 되돌리기로 읽으면 조용히 틀리는 자리다.

    합치는 것이 규약이라 **역방향이 저절로 맞는다** — `Unfold` 의 역방향이 곧 `Fold`
    이고 그 반대도 같다. 색인 하나로 적으면 둘이 한 기계가 된다.

    ## `LocalResponseNorm` 의 창은 한쪽으로 치우쳐 있다

    채널 `c` 의 창이 `[c − n//2, c + n − 1 − n//2]` 다. `size=2` 면 `{c−1, c}` 이지
    `{c, c+1}` 이 아니다 — 재서 확인했다. 가운데를 잡으면 값이 한 칸씩 밀리는데,
    크기가 같아서 모양으로는 안 보인다.

    ## `RReLU` 는 평가 모드에서 기울기가 정해진다

    학습 때는 `[lower, upper]` 에서 뽑고 평가 때는 그 **가운데**를 쓴다 —
    기본값이면 `(1/8 + 1/3)/2 = 0.2292` 다. 난수가 안 끼는 쪽만 값으로 묻는다.

    ## `UpsamplingBilinear2d` 는 `align_corners=True` 다

    `Upsample(mode='bilinear')` 의 기본값은 `False` 라 값이 다르다. 이름만 보고
    별명으로 두면 가장자리가 어긋나는데, 안쪽은 비슷해서 눈으로는 안 갈린다.
    """
    cases = []

    def add(name, fn):
        cases.append((MISC_PREFIX + name, fn))

    def F(L):
        return L.nn.functional

    img = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    img3 = np.arange(3 * 16, dtype=np.float32).reshape(1, 3, 4, 4)
    small = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    chans = (np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1) + 1)
    cube = np.arange(3 * 4, dtype=np.float32).reshape(1, 3, 2, 2)

    # ── 창을 펴고 되접기 ────────────────────────────────────────────────
    add("unfold", lambda L: F(L).unfold(L.tensor(img), 2))
    add("unfold(stride=2)", lambda L: F(L).unfold(L.tensor(img), 2, stride=2))
    add("unfold(padding=1)", lambda L: F(L).unfold(L.tensor(img), 2, padding=1))
    add("unfold(채널 셋)", lambda L: F(L).unfold(L.tensor(img3), 2))
    # **되접으면 겹친 자리가 더해진다.** 원본이 안 나오는 것이 규약이다.
    add("fold(겹친 자리는 더한다)",
        lambda L: F(L).fold(F(L).unfold(L.tensor(img), 2), (4, 4), 2))
    add("fold(stride=2 면 안 겹친다)",
        lambda L: F(L).fold(F(L).unfold(L.tensor(img), 2, stride=2), (4, 4), 2,
                            stride=2))
    add("층::Unfold", lambda L: L.nn.Unfold(2)(L.tensor(img)))
    add("층::Fold",
        lambda L: L.nn.Fold((4, 4), 2)(L.nn.Unfold(2)(L.tensor(img))))

    def unfold_grad(L):
        x = L.tensor(img, requires_grad=True)
        out = F(L).unfold(x, 2)
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(x, "unfold")

    add("grad::unfold", unfold_grad)

    # ── Bilinear ────────────────────────────────────────────────────────
    w = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 10
    bias = np.array([0.5, -0.25], dtype=np.float32)
    a1 = np.array([[1., 2., 3.]], dtype=np.float32)
    a2 = np.array([[1., 1., 1., 1.]], dtype=np.float32)
    add("bilinear",
        lambda L: F(L).bilinear(L.tensor(a1), L.tensor(a2), L.tensor(w),
                                L.tensor(bias)))
    add("bilinear(편향 없이)",
        lambda L: F(L).bilinear(L.tensor(a1), L.tensor(a2), L.tensor(w)))

    def bilinear_layer(L):
        # **`load_state_dict` 로 넣는다.** `.data =` 는 라이브러리마다 뜻이 달라서
        # (torch 는 텐서를, 우리 코어는 numpy 배열을 준다) 케이스 본문이 한쪽 편을
        # 들게 된다 — 이 표가 이미 그 이유로 그쪽을 안 쓴다.
        layer = L.nn.Bilinear(3, 4, 2)
        layer.load_state_dict({"weight": L.tensor(w), "bias": L.tensor(bias)})
        return layer(L.tensor(a1), L.tensor(a2))

    add("층::Bilinear", bilinear_layer)
    add("repr::Bilinear", lambda L: repr(L.nn.Bilinear(3, 4, 2)))

    # ── LocalResponseNorm ───────────────────────────────────────────────
    add("local_response_norm",
        lambda L: F(L).local_response_norm(L.tensor(chans), 2))
    # 기본값은 차이가 아주 작아서 창의 자리를 못 가른다. 알파를 키워 묻는다.
    add("local_response_norm(alpha=1)",
        lambda L: F(L).local_response_norm(L.tensor(chans), 2, alpha=1.0,
                                           beta=1.0, k=1.0))
    add("local_response_norm(size=3)",
        lambda L: F(L).local_response_norm(L.tensor(chans), 3, alpha=1.0,
                                           beta=1.0, k=1.0))
    add("층::LocalResponseNorm",
        lambda L: L.nn.LocalResponseNorm(2)(L.tensor(chans)))
    add("repr::LocalResponseNorm", lambda L: repr(L.nn.LocalResponseNorm(2)))

    # ── Softmax2d ───────────────────────────────────────────────────────
    add("층::Softmax2d", lambda L: L.nn.Softmax2d()(L.tensor(cube)))
    add("Softmax2d 는 softmax(dim=1)",
        lambda L: L.nn.Softmax2d()(L.tensor(cube)) - F(L).softmax(L.tensor(cube),
                                                                  dim=1))
    add("repr::Softmax2d", lambda L: repr(L.nn.Softmax2d()))

    # ── RReLU — 난수가 안 끼는 쪽만 ─────────────────────────────────────
    neg = np.array([[-1., -2., 1.]], dtype=np.float32)
    add("rrelu(eval)", lambda L: F(L).rrelu(L.tensor(neg), training=False))
    add("층::RReLU(eval)", lambda L: L.nn.RReLU().eval()(L.tensor(neg)))
    add("rrelu(eval, 범위 지정)",
        lambda L: F(L).rrelu(L.tensor(neg), lower=0.2, upper=0.4,
                             training=False))
    add("repr::RReLU", lambda L: repr(L.nn.RReLU()))

    # ── Upsampling — 옛 이름 둘 ─────────────────────────────────────────
    add("층::UpsamplingNearest2d",
        lambda L: L.nn.UpsamplingNearest2d(scale_factor=2)(L.tensor(small)))
    add("층::UpsamplingBilinear2d",
        lambda L: L.nn.UpsamplingBilinear2d(scale_factor=2)(L.tensor(small)))
    # **`align_corners=True` 다.** `Upsample` 의 기본값과 다르다는 것이 요점이라
    # 그 차이를 값으로 묻는다.
    add("UpsamplingBilinear2d 는 align_corners=True",
        lambda L: L.nn.UpsamplingBilinear2d(scale_factor=2)(L.tensor(small))
        - F(L).interpolate(L.tensor(small), scale_factor=2, mode="bilinear",
                           align_corners=True))

    # ── EmbeddingBag ────────────────────────────────────────────────────
    table = np.arange(15, dtype=np.float32).reshape(5, 3)
    bags = np.array([[0, 1], [2, 3]], dtype=np.int64)

    def bag(L, mode):
        layer = L.nn.EmbeddingBag(5, 3, mode=mode)
        layer.load_state_dict({"weight": L.tensor(table)})
        return layer(L.tensor(bags))

    for mode in ("sum", "mean", "max"):
        add(f"층::EmbeddingBag({mode})", lambda L, m=mode: bag(L, m))
    add("repr::EmbeddingBag", lambda L: repr(L.nn.EmbeddingBag(5, 3)))

    def bag_offsets(L):
        layer = L.nn.EmbeddingBag(5, 3, mode="sum")
        layer.load_state_dict({"weight": L.tensor(table)})
        return layer(L.tensor(np.array([0, 1, 2, 3], dtype=np.int64)),
                     L.tensor(np.array([0, 2], dtype=np.int64)))

    add("층::EmbeddingBag(offsets)", bag_offsets)
    return cases


SHUFFLE_PREFIX = "shuffle::"


def shuffle_cases(inp=None):
    """자리를 옮기는 층 셋과 **채널째 떨구는** dropout 다섯.

    ## `Dropout2d` 는 원소가 아니라 채널을 떨군다

    이름이 `Dropout` 옆에 있어서 "2 차원용" 으로 읽기 쉬운데, 하는 일이 다르다 —
    한 채널을 **통째로** 0 으로 만들거나 통째로 남긴다. 그래서 채널 안이 섞인 답이
    나오면 그것은 원소별 dropout 이고, 값 하나만 봐서는 구분이 안 된다. 골든이
    "채널 안이 전부 같은가" 를 묻는 이유다.

    ## `AlphaDropout` 은 0 을 안 넣는다

    SELU 와 함께 쓰라고 만든 것이라, 떨군 자리에 **음의 상수**를 넣고 전체에 아핀
    변환을 걸어 평균과 분산을 지킨다. 입력이 전부 1 일 때 답이 `-0.779` 와 `1.666`
    두 값이었다(실측). 0 을 넣으면 SELU 의 자기정규화가 깨지는데, 값이 그럴듯해서
    학습이 도는 동안은 안 보인다.

    ## 자리 옮기기는 값으로 묻는다

    `PixelShuffle`·`PixelUnshuffle`·`ChannelShuffle` 은 난수가 안 끼므로 값을 그대로
    묻는다. 입력을 `arange` 로 두면 **어느 자리가 어디로 갔는지**가 답에 그대로 나온다.
    """
    cases = []

    def add(name, fn):
        cases.append((SHUFFLE_PREFIX + name, fn))

    def F(L):
        return L.nn.functional

    # ── 자리 옮기기 — 값으로 ────────────────────────────────────────────
    pix = np.arange(8 * 2 * 2, dtype=np.float32).reshape(1, 8, 2, 2)
    flat = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    chan = np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1)
    chan6 = np.arange(6 * 2, dtype=np.float32).reshape(1, 6, 2, 1)

    add("pixel_shuffle", lambda L: F(L).pixel_shuffle(L.tensor(pix), 2))
    add("pixel_unshuffle", lambda L: F(L).pixel_unshuffle(L.tensor(flat), 2))
    # 되돌리면 그대로여야 한다 — 둘이 서로의 역이라는 것이 규약이다.
    add("pixel 왕복",
        lambda L: F(L).pixel_unshuffle(F(L).pixel_shuffle(L.tensor(pix), 2), 2))
    add("channel_shuffle(2)", lambda L: F(L).channel_shuffle(L.tensor(chan), 2))
    add("channel_shuffle(3)", lambda L: F(L).channel_shuffle(L.tensor(chan6), 3))
    add("층::PixelShuffle", lambda L: L.nn.PixelShuffle(2)(L.tensor(pix)))
    add("층::PixelUnshuffle", lambda L: L.nn.PixelUnshuffle(2)(L.tensor(flat)))
    add("층::ChannelShuffle", lambda L: L.nn.ChannelShuffle(2)(L.tensor(chan)))
    for name, layer in (("PixelShuffle", "PixelShuffle(2)"),
                        ("PixelUnshuffle", "PixelUnshuffle(2)"),
                        ("ChannelShuffle", "ChannelShuffle(2)")):
        add(f"repr::{name}",
            lambda L, c=name: repr(getattr(L.nn, c)(2)))

    def shuffle_grad(L):
        x = L.tensor(pix, requires_grad=True)
        out = F(L).pixel_shuffle(x, 2)
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(x, "pixel_shuffle")

    add("grad::pixel_shuffle", shuffle_grad)

    # ── 난수가 안 끼는 자리는 값으로 ────────────────────────────────────
    img = np.arange(1 * 4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2)
    for name in ("dropout1d", "dropout2d", "dropout3d",
                 "alpha_dropout", "feature_alpha_dropout"):
        arr = (np.arange(1 * 4 * 3, dtype=np.float32).reshape(1, 4, 3)
               if name == "dropout1d" else
               np.arange(1 * 3 * 2 * 1 * 1, dtype=np.float32).reshape(1, 3, 2, 1, 1)
               if name == "dropout3d" else img)
        add(f"{name}::eval 은 항등",
            lambda L, n=name, a=arr: getattr(F(L), n)(L.tensor(a), 0.5,
                                                      training=False))
        add(f"{name}::p=0 은 항등",
            lambda L, n=name, a=arr: getattr(F(L), n)(L.tensor(a), 0.0,
                                                      training=True))

    # ── 난수가 끼는 자리는 성질로 ───────────────────────────────────────
    #
    # 답이 난수기에 달렸고 우리 난수기는 torch 의 것과 다르다. 그래도 **양쪽이 똑같이
    # 답할 수 있는 것**은 있다 — 채널 안이 한 덩어리인가, 살아남은 값이 몇 배인가.
    big = np.ones((200, 8, 2, 2), dtype=np.float32)

    def whole_channels(L):
        """채널 안이 **전부 같은가.** 원소별로 떨구면 여기서 갈린다."""
        out = to_numpy(F(L).dropout2d(L.tensor(big), 0.5, training=True))
        flat_ch = out.reshape(out.shape[0], out.shape[1], -1)
        uniform = np.all(flat_ch == flat_ch[:, :, :1], axis=2)
        return f"채널마다 한 덩어리={bool(uniform.all())}"

    add("dropout2d::채널째 떨군다", whole_channels)

    def channel_scale(L):
        """살아남은 채널은 정확히 `1/(1-p)` 배다.

        **자릿수를 못 박는다.** `round()` 로 두었더니 파이썬이 `2.0` 을 내고 JS 가
        `2` 를 내서 값이 아니라 글자에서 갈렸다 — 답이 문자열인 케이스는 그 문자열이
        곧 계약이라 만드는 쪽에서 모양을 정해야 한다.
        """
        out = to_numpy(F(L).dropout2d(L.tensor(big), 0.5, training=True))
        kept = out[out != 0]
        return f"배율={float(kept.mean()):.3f}" if kept.size else "배율=none"

    add("dropout2d::살아남은 배율", channel_scale)

    def channel_rate(L):
        """떨구는 비율이 대략 `p` 다. 채널 단위로 센다."""
        out = to_numpy(F(L).dropout2d(L.tensor(big), 0.5, training=True))
        per = out.reshape(out.shape[0], out.shape[1], -1)[:, :, 0]
        return f"대략 절반={bool(0.4 < float((per == 0).mean()) < 0.6)}"

    add("dropout2d::떨구는 비율", channel_rate)

    def alpha_values(L):
        """**떨군 자리가 0 이 아니다.** 답에 나오는 서로 다른 값이 둘뿐이어야 한다."""
        ones = np.ones((400, 8), dtype=np.float32)
        out = to_numpy(F(L).alpha_dropout(L.tensor(ones), 0.5, training=True))
        vals = np.unique(np.round(out, 4))
        lo, hi = float(vals.min()), float(vals.max())
        return (f"값이 둘={len(vals) == 2} 낮은쪽={round(lo, 3)} "
                f"높은쪽={round(hi, 3)}")

    add("alpha_dropout::떨군 자리가 0 이 아니다", alpha_values)

    def feature_alpha_whole(L):
        out = to_numpy(F(L).feature_alpha_dropout(L.tensor(big), 0.5,
                                                  training=True))
        flat_ch = out.reshape(out.shape[0], out.shape[1], -1)
        uniform = np.all(flat_ch == flat_ch[:, :, :1], axis=2)
        return f"채널마다 한 덩어리={bool(uniform.all())}"

    add("feature_alpha_dropout::채널째 떨군다", feature_alpha_whole)

    # 층으로도 닿아야 한다. **랭크가 층마다 다르다** — `Dropout1d` 는 4 차원을
    # 거절한다("2D 나 3D 를 달라"). 공간 축의 수가 이름에 들어 있으니 당연한데,
    # 같은 입력을 다섯에 돌려 쓰려다 걸렸다.
    ranks = {
        "Dropout1d": np.arange(4 * 3, dtype=np.float32).reshape(1, 4, 3),
        "Dropout2d": img,
        "Dropout3d": np.arange(3 * 2, dtype=np.float32).reshape(1, 3, 2, 1, 1),
        "AlphaDropout": img,
        "FeatureAlphaDropout": img,
    }
    for name, arr in ranks.items():
        add(f"층::{name}(eval)",
            lambda L, c=name, a=arr: getattr(L.nn, c)(0.5).eval()(L.tensor(a)))
        add(f"repr::{name}", lambda L, c=name: repr(getattr(L.nn, c)(0.5)))
    return cases


LAZY_PREFIX = "lazy::"


def lazy_cases(inp=None):
    """모양을 **첫 forward 에서 알아내는** 층들.

    `nn.LazyLinear(3)` 은 `in_features` 를 안 받는다. 처음 지나가는 값을 보고 정한다 —
    합성곱 뒤에 몇 채널이 나오는지를 손으로 세는 일이 사라지므로 실제 코드가 자주 쓴다.

    ## 클래스가 바뀐다

    이것이 규약의 핵심이고 짐작으로는 안 나온다. 첫 forward 뒤에 그 물건은 **더 이상
    `LazyLinear` 가 아니라 `Linear` 다** — `type(m).__name__` 이 바뀌고
    `isinstance(m, nn.LazyLinear)` 이 거짓이 되며 `has_uninitialized_params` 라는
    메서드 자체가 사라진다(실측). 깃발 하나로 처리하면 이름이 안 바뀌고, 그러면
    `repr` 도 `isinstance` 도 갈린다.

    ## 여기서는 **셋 다 답할 수 있는 것만** 묻는다

    torch 는 굳기 전에도 파라미터를 둘 내놓고(`<UninitializedParameter>`), 모양을
    물으면 던지고, 그런데도 옵티마이저에는 넣게 해 준다. 그 기계는 코어에만 있다 —
    브라우저 쪽 층은 굳기 전에 텐서가 아예 없다.

    그래서 그쪽은 `tests/test_lazy.py` 가 **코어와 진짜 torch 를 직접 견준다.**
    골든에 넣으면 브라우저 둘이 답할 수 없는 질문이 되고, 답할 수 없는 질문을 표에
    두면 그 표가 "무엇이 통과했는가" 를 못 말하게 된다.

    ## 값은 못 묻는다

    가중치 초기값은 난수기에서 나오고 우리 난수기는 torch 와 다르다. 대신 **성질**을
    묻는다: 같은 씨앗에서 `LazyLinear` 가 굳은 것과 `Linear` 를 바로 세운 것이 같은가.
    torch 에서 참이고(실측), 우리에게서도 참이어야 한다 — 게으른 쪽이 다른 초기화를
    쓰면 학습이 미묘하게 갈린다.
    """
    cases = []

    def add(name, fn):
        cases.append((LAZY_PREFIX + name, fn))

    x2d = np.arange(10, dtype=np.float32).reshape(2, 5)
    img = np.arange(2 * 2 * 8 * 8, dtype=np.float32).reshape(2, 2, 8, 8) / 100

    # ── 굳으면 딴 것이 된다 ────────────────────────────────────────────
    #
    # 사용자가 실제로 보는 것은 `print(model)` 이다. 그 글자가 바뀌는 것이 이 규약의
    # 관찰 가능한 알맹이라, 이름과 `isinstance` 대신 그것을 묻는다 — 결속의 층은
    # 파이썬 쪽에서 전부 한 클래스라 이름으로는 셋이 못 맞춘다.
    add("굳기전::repr", lambda L: repr(L.nn.LazyLinear(3)))

    def repr_after(L):
        m = L.nn.LazyLinear(3)
        m(L.tensor(x2d))
        return repr(m)

    add("굳은뒤::repr", repr_after)

    def method_gone(L):
        m = L.nn.LazyLinear(3)
        before = hasattr(m, "has_uninitialized_params")
        m(L.tensor(x2d))
        return f"전 {before} 후 {hasattr(m, 'has_uninitialized_params')}"

    add("has_uninitialized_params 가 사라진다", method_gone)

    # ── 굳은 뒤의 모양 ─────────────────────────────────────────────────
    shapes = (
        ("LazyLinear", lambda L: L.nn.LazyLinear(3), x2d),
        ("LazyConv2d", lambda L: L.nn.LazyConv2d(4, 3), img),
        ("LazyBatchNorm2d", lambda L: L.nn.LazyBatchNorm2d(), img),
        ("LazyInstanceNorm2d", lambda L: L.nn.LazyInstanceNorm2d(), img),
        ("LazyConvTranspose2d", lambda L: L.nn.LazyConvTranspose2d(4, 3), img),
    )
    # **모양만 묻는다.** 굳은 뒤의 클래스 이름은 결속 쪽이 못 답한다 — 거기서는 층이
    # 전부 `Module` 한 클래스다. 이름이 바뀐다는 것은 `repr` 과 `test_lazy.py` 가
    # 이미 붙잡고 있으므로, 여기서는 셋 다 답할 수 있는 것만 묻는다.
    for name, make, arr in shapes:
        def run(L, m=make, a=arr):
            return str(tuple(m(L)(L.tensor(a)).shape))
        add(f"굳은뒤::{name}", run)

    def weight_shape(L):
        m = L.nn.LazyLinear(3)
        m(L.tensor(x2d))
        return str(tuple(m.weight.shape))

    add("굳은뒤::가중치 모양", weight_shape)

    # ── 성질: 게으른 쪽과 바로 세운 쪽이 같은 초기화를 쓰는가 ──────────
    def same_init(L):
        L.manual_seed(0)
        lazy = L.nn.LazyLinear(3)
        got = lazy(L.tensor(x2d))
        L.manual_seed(0)
        eager = L.nn.Linear(5, 3)
        want = eager(L.tensor(x2d))
        same = bool(np.allclose(np.asarray(got.detach().numpy()),
                                np.asarray(want.detach().numpy()), atol=1e-5))
        return f"같다={same}"

    add("성질::같은 씨앗이면 같은 초기화", same_init)

    # 학습이 실제로 돈다 — 굳은 뒤 옵티마이저가 그 파라미터를 움직이는가.
    def trains(L):
        L.manual_seed(0)
        m = L.nn.LazyLinear(2)
        opt = L.optim.SGD(m.parameters(), lr=0.1)
        target = L.tensor(np.zeros((2, 2), dtype=np.float32))
        first = None
        for _ in range(3):
            loss = ((m(L.tensor(x2d)) - target) ** 2).mean()
            first = float(loss.item()) if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
        last = float(((m(L.tensor(x2d)) - target) ** 2).mean().item())
        return f"손실이 내려갔다={last < first}"

    add("성질::굳은 뒤 학습이 돈다", trains)

    # ── 씨앗이 층 초기화에도 닿는가 ────────────────────────────────────
    #
    # **게으른 층이 끌어낸 결함이다.** 코어의 `manual_seed` 가 모듈 전역 이름을 다시
    # 묶기만 해서, 임포트 때 그 생성기를 붙잡아 간 `_nn` 쪽은 옛것을 계속 썼다 —
    # `randn` 만 재현되고 **층 초기화와 dropout 은 씨앗을 안 따랐다.**
    #
    # 골든이 오래 못 본 이유는 케이스마다 가중치를 밖에서 넣어 주기 때문이다. 게으른
    # 층이 초기화를 스스로 하면서 처음으로 그 자리가 물어졌다. 이 셋이 남는다.
    def reproducible(L, make):
        L.manual_seed(0)
        first = np.asarray(make().detach().numpy()).copy()
        L.manual_seed(0)
        again = np.asarray(make().detach().numpy()).copy()
        return f"재현된다={bool(np.array_equal(first, again))}"

    add("씨앗::Linear 초기화",
        lambda L: reproducible(L, lambda: L.nn.Linear(4, 3).weight))
    add("씨앗::Conv2d 초기화",
        lambda L: reproducible(L, lambda: L.nn.Conv2d(2, 3, 3).weight))
    add("씨앗::dropout 마스크",
        lambda L: reproducible(
            L, lambda: L.nn.Dropout(0.5)(L.ones(8))))
    return cases


LOSS_PREFIX = "loss::"

# 손실용 입력. **값이 0 으로 뭉개지지 않게 골랐다** — 처음 쓴 삼중항 입력은 여백이
# 한 번도 안 걸려서 네 갈래가 전부 0.0 이었고, 그러면 무엇을 바꿔도 통과한다.
_LOSS_X = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]], dtype=np.float32)
_LOSS_Y = np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 0.25]], dtype=np.float32)
_LOSS_ANC = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
_LOSS_POS = np.array([[2.0, 0.5], [1.5, 1.0]], dtype=np.float32)
_LOSS_NEG = np.array([[1.1, 0.1], [0.2, 0.9]], dtype=np.float32)
_LOSS_A = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=np.float32)
_LOSS_B = np.array([[0.5, 1.5], [1.0, -0.5]], dtype=np.float32)
_LOSS_SIGN = np.array([1.0, -1.0], dtype=np.float32)
_LOSS_HINGE = np.array([[0.5, 1.5], [2.0, 0.25]], dtype=np.float32)
_LOSS_HTGT = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
_LOSS_COUNT = np.array([[1.0, 2.0, 0.0], [3.0, 0.5, 1.0]], dtype=np.float32)
_LOSS_VAR = np.array([[1.0, 0.5, 2.0], [0.25, 1.5, 1.0]], dtype=np.float32)
_LOSS_MM = np.array([[0.1, 0.2, 0.4], [0.8, 0.3, 0.1]], dtype=np.float32)


def loss_cases(inp=None):
    """손실 열셋과 거리 셋.

    ## 기본값이 차이를 덮는 자리가 둘

    `HuberLoss(δ)` 와 `SmoothL1Loss(β)` 는 **δ=1 에서만 같다.** 실제 관계는
    `huber(δ) = δ · smooth_l1(β=δ)` 라 δ=0.5 면 두 배, δ=2 면 절반이다. 기본값으로만
    물으면 둘을 같은 함수로 두고도 통과한다 — 그래서 δ 를 바꿔 묻는다.

    `KLDivLoss` 는 reduction 이 **넷**이다. `mean` 은 원소 수로 나누고 `batchmean` 은
    배치 크기로 나눈다 — 여기서는 6 으로 나누느냐 2 로 나누느냐이고, torch 자신도
    "다음 주 버전에서 `mean` 을 `batchmean` 처럼 바꾸겠다" 고 경고한다. 둘 다 묻는다.

    ## 켜고 끄는 항이 조건부인 것 둘

    `PoissonNLLLoss(full=True)` 의 스털링 보정은 **`target > 1` 일 때만** 더해진다
    (실측: target 이 0·0.5·1 이면 차이가 0 이고 2 에서 0.6518 이 붙는다). 조건 없이
    늘 더하면 target 이 작은 자리에서만 틀린다.

    `GaussianNLLLoss` 는 분산을 `eps` 로 자른다 — `var=1e-9` 에 기본 `eps=1e-6` 이면
    `1e-6` 으로 잘려 124993 이 나오고, `eps=1e-2` 로 주면 10.2 다. 안 자르면 0 으로
    나눠 무한대가 된다.

    ## `pairwise_distance` 의 `eps` 는 **차에** 더한다

    결과에 더하는 것이 아니다. `p=1` 로 차가 정확히 1.0 인 자리를 물으면 1.0000020 이
    나온다(=1 + 2·1e-6). 결과에 더한다고 읽으면 1.000001 이 되어 자릿수 하나가 갈린다.
    """
    x, y = _LOSS_X, _LOSS_Y
    a, b, sign = _LOSS_A, _LOSS_B, _LOSS_SIGN
    anc, pos, neg = _LOSS_ANC, _LOSS_POS, _LOSS_NEG

    def F(L):
        return L.nn.functional

    cases = []

    def add(name, fn):
        cases.append((LOSS_PREFIX + name, fn))

    # ── Huber — 기본값이 SmoothL1 과 겹치는 자리 ─────────────────────────
    for tag, delta in (("기본", None), ("δ=0.5", 0.5), ("δ=2", 2.0)):
        add(f"huber({tag})",
            lambda L, d=delta: F(L).huber_loss(L.tensor(x), L.tensor(y))
            if d is None else F(L).huber_loss(L.tensor(x), L.tensor(y), delta=d))
    add("huber(none)",
        lambda L: F(L).huber_loss(L.tensor(x), L.tensor(y), reduction="none"))
    add("huber(sum)",
        lambda L: F(L).huber_loss(L.tensor(x), L.tensor(y), reduction="sum"))
    # **같은 δ 로 물으면 둘의 관계가 값에 드러난다.**
    add("huber(δ=0.5)/smooth_l1(β=0.5)",
        lambda L: F(L).huber_loss(L.tensor(x), L.tensor(y), delta=0.5)
        / F(L).smooth_l1_loss(L.tensor(x), L.tensor(y), beta=0.5))

    # ── KL — reduction 이 넷 ─────────────────────────────────────────────
    def kl(L, red, log_target=False):
        logp = F(L).log_softmax(L.tensor(x), dim=1)
        tgt = F(L).softmax(L.tensor(y), dim=1)
        if log_target:
            tgt = tgt.log()
        return F(L).kl_div(logp, tgt, reduction=red, log_target=log_target)

    for red in ("none", "mean", "sum", "batchmean"):
        add(f"kl_div({red})", lambda L, r=red: kl(L, r))
    add("kl_div(log_target)", lambda L: kl(L, "mean", log_target=True))

    # ── 포아송·가우스 ───────────────────────────────────────────────────
    for log_input in (True, False):
        for full in (False, True):
            add(f"poisson(log_input={log_input},full={full})",
                lambda L, li=log_input, f=full: F(L).poisson_nll_loss(
                    L.tensor(np.abs(x) + 0.5), L.tensor(_LOSS_COUNT),
                    log_input=li, full=f))
    add("poisson(none)",
        lambda L: F(L).poisson_nll_loss(L.tensor(np.abs(x) + 0.5),
                                        L.tensor(_LOSS_COUNT), reduction="none"))
    for full in (False, True):
        add(f"gaussian(full={full})",
            lambda L, f=full: F(L).gaussian_nll_loss(
                L.tensor(x), L.tensor(y), L.tensor(_LOSS_VAR), full=f))
    # 분산이 `eps` 아래로 내려가는 자리 — 안 자르면 여기서 무한대가 된다.
    tiny = np.array([[1e-9, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    add("gaussian(var<eps)",
        lambda L: F(L).gaussian_nll_loss(L.tensor(x), L.tensor(y), L.tensor(tiny),
                                         reduction="none"))
    add("gaussian(eps=1e-2)",
        lambda L: F(L).gaussian_nll_loss(L.tensor(x), L.tensor(y), L.tensor(tiny),
                                         eps=1e-2, reduction="none"))

    # ── 여백 계열 ───────────────────────────────────────────────────────
    add("margin_ranking",
        lambda L: F(L).margin_ranking_loss(
            L.tensor(np.array([1., 2.], dtype=np.float32)),
            L.tensor(np.array([2., 1.], dtype=np.float32)),
            L.tensor(sign), margin=0.5))
    add("margin_ranking(none)",
        lambda L: F(L).margin_ranking_loss(
            L.tensor(np.array([1., 2.], dtype=np.float32)),
            L.tensor(np.array([2., 1.], dtype=np.float32)),
            L.tensor(sign), margin=0.5, reduction="none"))
    for margin in (0.0, 0.5):
        add(f"cosine_embedding(margin={margin})",
            lambda L, m=margin: F(L).cosine_embedding_loss(
                L.tensor(a), L.tensor(b), L.tensor(sign), margin=m,
                reduction="none"))
    for margin in (1.0, 2.0):
        add(f"hinge_embedding(margin={margin})",
            lambda L, m=margin: F(L).hinge_embedding_loss(
                L.tensor(_LOSS_HINGE), L.tensor(_LOSS_HTGT), margin=m,
                reduction="none"))
    # **표적이 ±1 이 아닌 자리.** torch 는 두 항을 갈라 고르는 것이 아니라 **더한다** —
    # `y ≠ 1` 에 여백 항, `y ≠ −1` 에 `x` 를 놓고 합하므로 `y=0` 에서는 둘 다 켜진다.
    # ±1 만 물으면 한쪽씩만 켜져서 그 차이가 안 드러나고, `sign()` 은 0 을 만든다.
    add("hinge_embedding(y=0)",
        lambda L: F(L).hinge_embedding_loss(
            L.tensor(np.array([[-1., 0.5, 2.]], dtype=np.float32)),
            L.tensor(np.array([[0., 0., 0.]], dtype=np.float32)), reduction="none"))
    add("soft_margin",
        lambda L: F(L).soft_margin_loss(L.tensor(x), L.tensor(np.sign(y))))
    add("soft_margin(none)",
        lambda L: F(L).soft_margin_loss(L.tensor(x), L.tensor(np.sign(y)),
                                        reduction="none"))

    # ── 삼중항 ──────────────────────────────────────────────────────────
    triplets = (("기본", {}), ("margin=2", {"margin": 2.0}), ("p=1", {"p": 1}),
                ("swap", {"swap": True}))
    for tag, kw in triplets:
        add(f"triplet({tag})",
            lambda L, k=kw: F(L).triplet_margin_loss(
                L.tensor(anc), L.tensor(pos), L.tensor(neg), **k))
    add("triplet(none)",
        lambda L: F(L).triplet_margin_loss(L.tensor(anc), L.tensor(pos),
                                           L.tensor(neg), reduction="none"))
    for tag, margin in (("기본", 1.0), ("margin=2", 2.0)):
        add(f"triplet_with_distance({tag})",
            lambda L, m=margin: F(L).triplet_margin_with_distance_loss(
                L.tensor(anc), L.tensor(pos), L.tensor(neg), margin=m))

    # ── 여러 라벨 ───────────────────────────────────────────────────────
    add("multilabel_soft_margin",
        lambda L: F(L).multilabel_soft_margin_loss(
            L.tensor(np.array([[0.5, -1.0, 2.0]], dtype=np.float32)),
            L.tensor(np.array([[1.0, 0.0, 1.0]], dtype=np.float32))))
    for tag, kw in (("기본", {}), ("margin=0.5", {"margin": 0.5}), ("p=2", {"p": 2})):
        add(f"multi_margin({tag})",
            lambda L, k=kw: F(L).multi_margin_loss(
                L.tensor(_LOSS_MM),
                L.tensor(np.array([2, 0], dtype=np.int64)), reduction="none", **k))
    add("multi_margin(weight)",
        lambda L: F(L).multi_margin_loss(
            L.tensor(_LOSS_MM), L.tensor(np.array([2, 0], dtype=np.int64)),
            weight=L.tensor(np.array([1., 2., 0.5], dtype=np.float32))))
    # **표적이 목록이고 −1 이 끝을 뜻한다.** 그 규약을 안 지키면 −1 을 반의 하나로 센다.
    add("multilabel_margin",
        lambda L: F(L).multilabel_margin_loss(
            L.tensor(np.array([[0.1, 0.2, 0.4, 0.8]], dtype=np.float32)),
            L.tensor(np.array([[3, 0, -1, 1]], dtype=np.int64))))

    # ── 거리 ────────────────────────────────────────────────────────────
    add("pairwise_distance",
        lambda L: F(L).pairwise_distance(L.tensor(a), L.tensor(b)))
    # `eps` 를 결과가 아니라 **차에** 더한다 — p=1 에서 자릿수로 드러난다.
    add("pairwise_distance(p=1)",
        lambda L: F(L).pairwise_distance(L.tensor(a), L.tensor(b), p=1))
    add("pairwise_distance(eps=0)",
        lambda L: F(L).pairwise_distance(L.tensor(a), L.tensor(b), p=1, eps=0))
    add("pairwise_distance(keepdim)",
        lambda L: F(L).pairwise_distance(L.tensor(a), L.tensor(b), keepdim=True))
    add("pdist",
        lambda L: F(L).pdist(L.tensor(np.array([[0., 0.], [3., 4.], [1., 1.]],
                                               dtype=np.float32))))

    # **원소가 하나인 것을 접기.** 접을 것이 없다는 뜻이라 그냥 값이 나와야 하는데,
    # GPU 쪽이 그 자리에서 0 을 냈다 — 명령을 모아 두었다가 보내는데 이 길만 자기
    # 인코더를 만들어 **먼저** 제출해서, 아직 계산 안 된 버퍼를 복사했다. 예외도
    # NaN 도 아닌 0 이라 손실이 조용히 사라지는 자리였고, 손실 하나가 배치 1 로
    # 물어보기 전까지 골든 1,399 건이 초록이었다.
    add("원소 하나를 mean",
        lambda L: (L.tensor(np.array([1., 2., 3.], dtype=np.float32)).sum()
                   * 1.0).reshape(1).mean())
    add("원소 하나를 sum",
        lambda L: (L.tensor(np.array([1., 2., 3.], dtype=np.float32)).sum()
                   * 1.0).reshape(1).sum())

    # ── 층으로도 닿아야 한다 ────────────────────────────────────────────
    layers = (
        ("HuberLoss", lambda L: L.nn.HuberLoss(delta=0.5)(L.tensor(x), L.tensor(y))),
        ("KLDivLoss", lambda L: L.nn.KLDivLoss(reduction="batchmean")(
            F(L).log_softmax(L.tensor(x), dim=1), F(L).softmax(L.tensor(y), dim=1))),
        ("PoissonNLLLoss", lambda L: L.nn.PoissonNLLLoss()(
            L.tensor(np.abs(x) + 0.5), L.tensor(_LOSS_COUNT))),
        ("GaussianNLLLoss", lambda L: L.nn.GaussianNLLLoss()(
            L.tensor(x), L.tensor(y), L.tensor(_LOSS_VAR))),
        ("MarginRankingLoss", lambda L: L.nn.MarginRankingLoss(margin=0.5)(
            L.tensor(np.array([1., 2.], dtype=np.float32)),
            L.tensor(np.array([2., 1.], dtype=np.float32)), L.tensor(sign))),
        ("CosineEmbeddingLoss", lambda L: L.nn.CosineEmbeddingLoss()(
            L.tensor(a), L.tensor(b), L.tensor(sign))),
        ("HingeEmbeddingLoss", lambda L: L.nn.HingeEmbeddingLoss()(
            L.tensor(_LOSS_HINGE), L.tensor(_LOSS_HTGT))),
        ("SoftMarginLoss", lambda L: L.nn.SoftMarginLoss()(
            L.tensor(x), L.tensor(np.sign(y)))),
        ("TripletMarginLoss", lambda L: L.nn.TripletMarginLoss()(
            L.tensor(anc), L.tensor(pos), L.tensor(neg))),
        ("TripletMarginWithDistanceLoss",
         lambda L: L.nn.TripletMarginWithDistanceLoss()(
             L.tensor(anc), L.tensor(pos), L.tensor(neg))),
        ("MultiLabelSoftMarginLoss", lambda L: L.nn.MultiLabelSoftMarginLoss()(
            L.tensor(np.array([[0.5, -1.0, 2.0]], dtype=np.float32)),
            L.tensor(np.array([[1.0, 0.0, 1.0]], dtype=np.float32)))),
        ("MultiMarginLoss", lambda L: L.nn.MultiMarginLoss()(
            L.tensor(_LOSS_MM), L.tensor(np.array([2, 0], dtype=np.int64)))),
        ("MultiLabelMarginLoss", lambda L: L.nn.MultiLabelMarginLoss()(
            L.tensor(np.array([[0.1, 0.2, 0.4, 0.8]], dtype=np.float32)),
            L.tensor(np.array([[3, 0, -1, 1]], dtype=np.int64)))),
        ("PairwiseDistance", lambda L: L.nn.PairwiseDistance()(
            L.tensor(a), L.tensor(b))),
        ("CosineSimilarity", lambda L: L.nn.CosineSimilarity(dim=1)(
            L.tensor(a), L.tensor(b))),
    )
    for name, fn in layers:
        add(f"층::{name}", fn)

    # ── 최상위로도 닿아야 한다 ──────────────────────────────────────────
    #
    # 빈자리 목록을 정리하자 "`F` 에 있는데 최상위에 없는" 이름 아홉이 드러났다.
    # 그런데 재 보니 **일곱은 최상위 쪽이 다른 함수였다** — 날 ATen 연산이라 기본
    # reduction 이 `none` 이고 `reduction` 이 문자열이 아니라 정수다.
    # `torch.kl_div(a, b)` 는 `[2,2]` 를 내고 `F.kl_div(a, b)` 는 스칼라를 낸다.
    #
    # 친절한 별명으로 걸었으면 **모양부터 갈렸을 것이다.** 글자 그대로 같은 함수인
    # 것은 이 둘뿐이고(`torch.pdist is F.pdist` 가 참이다), 그래서 둘만 낸다.
    add("최상위::pairwise_distance",
        lambda L: L.pairwise_distance(L.tensor(a), L.tensor(b)))
    add("최상위::pdist",
        lambda L: L.pdist(L.tensor(np.array([[0., 0.], [3., 4.], [1., 1.]],
                                            dtype=np.float32))))

    # ── 기울기 ──────────────────────────────────────────────────────────
    #
    # **손실은 기울기가 전부다.** 값이 맞고 기울기가 틀리면 학습이 조용히 다른 데로
    # 간다 — 이 저장소가 BatchNorm 으로 오래 겪은 종류다.
    grads = (
        ("huber", lambda L, p: F(L).huber_loss(p, L.tensor(y), delta=0.5)),
        ("kl_div", lambda L, p: F(L).kl_div(F(L).log_softmax(p, dim=1),
                                            F(L).softmax(L.tensor(y), dim=1))),
        ("poisson", lambda L, p: F(L).poisson_nll_loss(p, L.tensor(_LOSS_COUNT))),
        ("gaussian", lambda L, p: F(L).gaussian_nll_loss(
            p, L.tensor(y), L.tensor(_LOSS_VAR))),
        ("soft_margin", lambda L, p: F(L).soft_margin_loss(p, L.tensor(np.sign(y)))),
        ("hinge_embedding", lambda L, p: F(L).hinge_embedding_loss(
            p, L.tensor(np.sign(y)))),
        ("multilabel_soft_margin", lambda L, p: F(L).multilabel_soft_margin_loss(
            p, L.tensor((y > 0).astype(np.float32)))),
    )
    for name, fn in grads:
        def run(L, f=fn, n=name):
            p = L.tensor(x, requires_grad=True)
            f(L, p).backward()
            return _grad_of(p, n)
        cases.append((LOSS_PREFIX + f"grad::{name}", run))

    def triplet_grad(L):
        p = L.tensor(anc, requires_grad=True)
        F(L).triplet_margin_loss(p, L.tensor(pos), L.tensor(neg)).backward()
        return _grad_of(p, "triplet")

    cases.append((LOSS_PREFIX + "grad::triplet", triplet_grad))

    def cosine_grad(L):
        p = L.tensor(a, requires_grad=True)
        F(L).cosine_embedding_loss(p, L.tensor(b), L.tensor(sign)).backward()
        return _grad_of(p, "cosine_embedding")

    cases.append((LOSS_PREFIX + "grad::cosine_embedding", cosine_grad))
    return cases


PAD_PREFIX = "pad::"

# 패딩용 입력. 값이 자리 번호라 **어디서 온 값인지 답만 보고 알 수 있다** —
# 거울인지 되풀이인지 가장자리인지가 값에 그대로 드러난다.
_PAD_1D = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
_PAD_2D = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)
_PAD_3D = np.arange(24, dtype=np.float32).reshape(1, 1, 2, 3, 4)

_PAD_MODES = ("constant", "reflect", "replicate", "circular")


def pad_cases(inp=None):
    """패딩 — **네 가지 모드와 층 열다섯 개.**

    지금까지 `F.pad` 는 상수만 했다. 나머지 셋(`reflect`·`replicate`·`circular`)이
    없으면 그 위에 얹힌 층 열다섯 개가 통째로 없는 것이고, 그것이 `nn` 의 빈자리
    여든넷 중 가장 큰 덩어리였다.

    ## 값이 자리 번호다

    입력을 `arange` 로 두면 답만 보고 **어디서 온 값인지** 알 수 있다. 3 칸짜리
    `[0,1,2]` 를 앞뒤로 늘리면 모드마다 다음이 나온다(진짜 torch 에 물어 확인):

        constant   9 9 [0 1 2] 9      ← 채운다
        reflect    2 1 [0 1 2] 1      ← 가장자리를 거울로, 가장자리는 안 겹친다
        replicate  0 0 [0 1 2] 2      ← 가장자리를 늘인다
        circular   1 2 [0 1 2] 0      ← 반대편에서 가져온다

    구현이 갈릴 자리는 거울의 기준점(0 을 겹치는가)과 감는 방향인데, 값에 그대로
    나오므로 이 네 줄이 그 둘을 다 붙잡는다.

    ## 짝의 개수와 랭크가 맞물린다

    `F.pad(4차원, (1,1), mode='reflect')` 는 **거절이다** — torch 가
    `NotImplementedError` 를 낸다. 짝이 하나면 2·3 차원, 둘이면 3·4 차원, 셋이면
    4·5 차원이라야 한다. 아무 랭크나 받으면 축을 잘못 잡고도 통과한다.

    ## `reflect` 만 크기를 따진다

    거울로 접으려면 접을 것이 있어야 하므로 패딩이 그 축의 크기보다 작아야 한다.
    `replicate` 는 다섯 칸을 늘려도 된다 — 늘일 값이 늘 있기 때문이다.
    """
    cases = []
    shapes = (("1d", _PAD_1D, (2, 1)), ("2d", _PAD_2D, (1, 1, 1, 1)),
              ("3d", _PAD_3D, (1, 1, 1, 1, 1, 1)))
    for tag, arr, pads in shapes:
        for mode in _PAD_MODES:
            def run(L, a=arr, p=pads, m=mode):
                kw = {"value": 9.0} if m == "constant" else {}
                return L.nn.functional.pad(L.tensor(a), p, mode=m, **kw)
            cases.append((PAD_PREFIX + f"{tag}::{mode}", run))

            def grad(L, a=arr, p=pads, m=mode):
                x = L.tensor(a, requires_grad=True)
                out = L.nn.functional.pad(x, p, mode=m)
                (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
                return _grad_of(x, f"pad {m}")
            cases.append((PAD_PREFIX + f"grad::{tag}::{mode}", grad))

    # 비대칭 — 앞뒤가 다른 자리에서 축을 뒤집어 잡으면 여기서 드러난다.
    cases += [
        (PAD_PREFIX + "비대칭::reflect",
         lambda L: L.nn.functional.pad(L.tensor(_PAD_2D), (1, 2, 0, 1), mode="reflect")),
        (PAD_PREFIX + "비대칭::circular",
         lambda L: L.nn.functional.pad(L.tensor(_PAD_2D), (2, 1, 1, 0), mode="circular")),
        # `replicate` 는 크기를 안 따진다 — 늘일 값이 늘 있다.
        (PAD_PREFIX + "replicate(크게)",
         lambda L: L.nn.functional.pad(L.tensor(_PAD_1D), (5, 0), mode="replicate")),
        # 배치 없는 입력도 받는다.
        (PAD_PREFIX + "2차원 입력::reflect",
         lambda L: L.nn.functional.pad(
             L.tensor(np.arange(6, dtype=np.float32).reshape(2, 3)), (1, 1),
             mode="reflect")),
    ]

    # ── 층 열다섯 개 ────────────────────────────────────────────────────
    layers = (
        ("ReflectionPad1d", 2, _PAD_1D), ("ReflectionPad2d", 1, _PAD_2D),
        ("ReflectionPad2d(비대칭)", (1, 2, 0, 1), _PAD_2D),
        ("ReflectionPad3d", 1, _PAD_3D),
        ("ReplicationPad1d", 2, _PAD_1D), ("ReplicationPad2d", 1, _PAD_2D),
        ("ReplicationPad3d", 1, _PAD_3D),
        ("ZeroPad1d", 2, _PAD_1D), ("ZeroPad2d", 1, _PAD_2D),
        ("ZeroPad3d", 1, _PAD_3D),
        ("CircularPad1d", 2, _PAD_1D), ("CircularPad2d", 1, _PAD_2D),
        ("CircularPad3d", 1, _PAD_3D),
    )
    for name, arg, arr in layers:
        cls = name.split("(")[0]

        def run(L, c=cls, a=arg, t=arr):
            return getattr(L.nn, c)(a)(L.tensor(t))
        cases.append((PAD_PREFIX + f"층::{name}", run))
        cases.append((PAD_PREFIX + f"repr::{name}",
                      lambda L, c=cls, a=arg: repr(getattr(L.nn, c)(a))))

    for name, arg, arr in (("ConstantPad1d", 2, _PAD_1D), ("ConstantPad2d", 1, _PAD_2D),
                           ("ConstantPad3d", 1, _PAD_3D)):
        cases.append((PAD_PREFIX + f"층::{name}",
                      lambda L, c=name, a=arg, t=arr:
                      getattr(L.nn, c)(a, 7.0)(L.tensor(t))))
        # **`ConstantPad` 만 이름을 붙여 찍는다** — 나머지는 짝만 찍는다.
        cases.append((PAD_PREFIX + f"repr::{name}",
                      lambda L, c=name, a=arg: repr(getattr(L.nn, c)(a, 7.0))))

    # 층에도 기울기가 흘러야 한다 — 함수만 이어 놓고 층을 안 이으면 여기서 끊긴다.
    def layer_grad(L):
        x = L.tensor(_PAD_2D, requires_grad=True)
        out = L.nn.ReflectionPad2d(1)(x)
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(x, "ReflectionPad2d")

    cases.append((PAD_PREFIX + "grad::층::ReflectionPad2d", layer_grad))

    # ── 거절 ────────────────────────────────────────────────────────────
    def refuses(L, fn, what):
        try:
            fn(L)
            return "예외가 안 났다"
        except Exception as exc:                                    # noqa: BLE001
            return f"{type(exc).__name__}"

    bad = (
        ("reflect(크기 초과)",
         lambda L: L.nn.functional.pad(L.tensor(_PAD_1D), (3, 0), mode="reflect")),
        ("짝 개수가 랭크와 안 맞음",
         lambda L: L.nn.functional.pad(L.tensor(_PAD_2D), (1, 1), mode="reflect")),
    )
    for name, fn in bad:
        cases.append((PAD_PREFIX + f"거절::{name}",
                      lambda L, f=fn, n=name: refuses(L, f, n)))
    return cases


NORM_PREFIX = "norm::"


def norm_cases(inp=None):
    """정규화 세 가지와 전치 합성곱. **모양이 맞아도 값이 틀리는 자리들이다.**

    ## 정규화 — 무엇을 묶어 평균 내는가

    `LayerNorm`·`GroupNorm`·`InstanceNorm`·`BatchNorm` 은 식이 같고 **묶는 축만**
    다르다. 축을 잘못 고르면 모양은 그대로이고 값만 갈리는데, 학습은 그래도 돌아서
    한참 뒤에야 이상하다는 것을 안다.

    그래서 같은 입력에 `GroupNorm(1)`·`GroupNorm(3)`·`InstanceNorm2d` 를 나란히
    묻는다. 셋은 서로의 특수한 경우다 — 묶는 규칙이 틀리면 셋 중 둘이 같아진다.

    ## 전치 합성곱 — 가중치 축이 뒤집혀 있다

    `conv2d` 의 가중치는 `(출력, 입력, kh, kw)` 인데 `conv_transpose2d` 는
    `(입력, 출력, kh, kw)` 다. 뒤집어 놓아도 정사각 커널이면 **모양이 그대로 맞는다** —
    값으로만 갈린다. 이 층에서 가장 흔한 실수이고 그래서 값으로 묻는다.
    """
    inp = golden_inputs() if inp is None else inp
    img, seq, vol = inp["img"], inp["nd_seq"], inp["nd_vol"]
    tw1, tw2, tw3, tb, tb3 = (inp["tw1"], inp["tw2"], inp["tw3"], inp["tb"], inp["tb3"])
    cases = []

    def add(name, fn, arr):
        cases.append((NORM_PREFIX + name, lambda L, f=fn, a=arr: f(L, L.tensor(a))))

        def grad(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(x, n)
        cases.append((NORM_PREFIX + f"grad::{name}", grad))

    # ── GroupNorm. 그룹 수가 1 이면 LayerNorm, 채널 수면 InstanceNorm 이다. ──
    add("F.group_norm(1)", lambda L, x: L.nn.functional.group_norm(x, 1), img)
    add("F.group_norm(3)", lambda L, x: L.nn.functional.group_norm(x, 3), img)
    cases.append((NORM_PREFIX + "nn.GroupNorm(1,3)",
                  lambda L: L.nn.GroupNorm(1, 3)(L.tensor(img))))
    cases.append((NORM_PREFIX + "nn.GroupNorm(3,3)",
                  lambda L: L.nn.GroupNorm(3, 3)(L.tensor(img))))
    # **가중치가 붙으면 파라미터가 잡혀야 한다.** 이름이 곧 state_dict 열쇠다.
    cases.append((NORM_PREFIX + "nn.GroupNorm/파라미터 이름",
                  lambda L: " ".join(n for n, _ in L.nn.GroupNorm(3, 3).named_parameters())))

    # ── InstanceNorm. 표본마다·채널마다 따로 정규화한다. ────────────────────
    add("F.instance_norm", lambda L, x: L.nn.functional.instance_norm(x), img)
    for nd, arr in (("1d", seq), ("2d", img), ("3d", vol)):
        chan = arr.shape[1]
        cases.append((NORM_PREFIX + f"nn.InstanceNorm{nd}",
                      lambda L, n=nd, c=chan, a=arr:
                      getattr(L.nn, f"InstanceNorm{n}")(c)(L.tensor(a))))

    # ── RMSNorm. 평균을 안 뺀다 — 그것이 LayerNorm 과의 유일한 차이다. ──────
    add("F.rms_norm", lambda L, x: L.nn.functional.rms_norm(x, (4,)), img)
    cases.append((NORM_PREFIX + "nn.RMSNorm",
                  lambda L: L.nn.RMSNorm(4)(L.tensor(img))))

    # ── 전치 합성곱. ───────────────────────────────────────────────────────
    add("F.conv_transpose1d",
        lambda L, x: L.nn.functional.conv_transpose1d(x, L.tensor(tw1)), seq)
    add("F.conv_transpose2d",
        lambda L, x: L.nn.functional.conv_transpose2d(x, L.tensor(tw2)), img)
    add("F.conv_transpose2d(스트라이드2)",
        lambda L, x: L.nn.functional.conv_transpose2d(x, L.tensor(tw2), stride=2), img)
    add("F.conv_transpose2d(패딩1)",
        lambda L, x: L.nn.functional.conv_transpose2d(x, L.tensor(tw2), padding=1), img)
    add("F.conv_transpose2d(편향)",
        lambda L, x: L.nn.functional.conv_transpose2d(x, L.tensor(tw2), L.tensor(tb)),
        img)
    add("F.conv_transpose3d",
        lambda L, x: L.nn.functional.conv_transpose3d(x, L.tensor(tw3)), vol)

    # 가중치 쪽 기울기도 본다. **입력 쪽만 보면 축이 뒤집힌 것을 놓친다.**
    def weight_grad(L):
        w = L.tensor(tw2, requires_grad=True)
        out = L.nn.functional.conv_transpose2d(L.tensor(img), w)
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(w, "conv_transpose2d 가중치")

    cases.append((NORM_PREFIX + "grad::conv_transpose2d/가중치", weight_grad))

    for nd, arr, w, b in (("1d", seq, tw1, tb), ("2d", img, tw2, tb),
                          ("3d", vol, tw3, tb3)):
        def run(L, n=nd, a=arr, ww=w, bb=b):
            layer = getattr(L.nn, f"ConvTranspose{n}")(
                ww.shape[0], ww.shape[1], ww.shape[2])
            layer.load_state_dict({"weight": L.tensor(ww), "bias": L.tensor(bb)})
            return layer(L.tensor(a))
        cases.append((NORM_PREFIX + f"nn.ConvTranspose{nd}", run))
    return cases


CONTAINER_PREFIX = "container::"


def container_cases(inp=None):
    """합성 구조를 뚫고 **파라미터가 보이는가.**

    이 표에서 가장 늦게 생긴 자리이고, 늦은 이유가 이 케이스들의 값어치다.

    ## 왜 이것이 다른 케이스와 종류가 다른가

    나머지 케이스는 **값**을 묻는다 — `exp` 가 틀리면 숫자가 다르고 바로 보인다.
    여기서 묻는 것은 **순회**다. `parameters()` 가 어떤 파라미터를 안 내놓으면
    옵티마이저가 그것을 못 보고, 못 보면 안 갱신하고, 안 갱신해도 **손실은 내려간다**
    (남은 파라미터가 대신 맞춘다). 예외도 경고도 없다. 이 저장소가 반복해서 잡아온
    결함의 모양 그대로인데, 그 자리를 여태 아무도 안 물었다.

    ## 어떻게 묻는가 — 이름과 값을 **둘 다**

    이름만 물으면 등록은 됐는데 갱신이 안 되는 경우를 놓친다. 값만 물으면 이름이
    `layers.0.weight` 가 아니라 `0.weight` 로 나와도 통과한다 — 그러면
    `load_state_dict` 가 남의 체크포인트를 못 읽는다.

    그래서 자리마다 둘을 짝으로 둔다: `named_parameters` 의 **이름 목록**과, SGD 를
    몇 스텝 돌린 뒤의 **파라미터 값**. 등록이 빠지면 값이 출발점 그대로 남아 갈린다.

    ## 왜 여기가 비어 있었나

    표의 모든 모델이 `nn.Sequential` 로 세워져 있었다. 그것 하나만 물으면 torch 코드가
    가장 흔히 하는 일 — `nn.Module` 을 상속하고 층을 속성으로 붙이는 것 — 이 한 번도
    안 걸린다. 실제로 벤치가 진짜 ResNet 을 세우다 `Module.__init__() missing 1
    required positional argument` 로 멈춰서야 알았다.
    """
    inp = golden_inputs() if inp is None else inp
    xin, yin = inp["train_x"], inp["train_y"]
    w0, b0, w1, b1 = inp["w0"], inp["b0"], inp["w1"], inp["b1"]
    # 손으로 만드는 선형층용 — `(6, 8)` 로 눕혀 `x @ w` 가 되게 한다. 전치를 케이스
    # 안에서 하면 그 전치가 틀렸는지 순회가 틀렸는지 못 가른다.
    flat_w = w0.T.copy()

    cases = []

    def add(name, build, load, forward, want_names):
        """자리 하나에 **이름·값** 두 케이스를 단다.

        `build(L)` 이 모델을 세우고, `load(L, m)` 이 고정 가중치를 넣고,
        `forward(L, m, x)` 가 출력을 낸다. 셋을 나눈 것은 컨테이너마다 값을 넣는
        방법이 다르기 때문이다(`load_state_dict` 가 닿는 자리와 안 닿는 자리).
        """
        def names(L):
            m = build(L)
            return " ".join(n for n, _ in m.named_parameters())

        def trained(L):
            m = build(L)
            load(L, m)
            opt = L.optim.SGD(m.parameters(), lr=0.05)
            x, y = L.tensor(xin), L.tensor(yin)
            for _ in range(3):
                opt.zero_grad()
                out = forward(L, m, x)
                # 출력 모양이 자리마다 달라서 손실을 하나로 못 쓴다. 자리마다 다른
                # 가중치를 곱해 접는다 — 그냥 `sum()` 이면 기울기가 전부 1 이라
                # 어느 자리가 안 움직였는지가 값에 안 남는다.
                w = L.arange(out.numel()).reshape(out.shape).float()
                (out * w).sum().backward()
                opt.step()
            return m

        cases.append((CONTAINER_PREFIX + f"{name}/이름", names))
        cases.append((CONTAINER_PREFIX + f"{name}/학습",
                      lambda L: dict(trained(L).named_parameters())[want_names]))

    # ── 상속한 Module. **torch 코드가 가장 흔히 하는 일이다.** ────────────────
    def build_subclass(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = L.nn.Linear(6, 8)
                self.fc2 = L.nn.Linear(8, 3)
        return Net()

    add("상속", build_subclass,
        lambda L, m: m.load_state_dict({
            "fc1.weight": L.tensor(w0), "fc1.bias": L.tensor(b0),
            "fc2.weight": L.tensor(w1), "fc2.bias": L.tensor(b1)}),
        lambda L, m, x: m.fc2(L.relu(m.fc1(x))),
        "fc1.weight")

    # ── ModuleList. 층 수가 변하는 모델이 전부 이것을 쓴다. ──────────────────
    def build_list(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = L.nn.ModuleList([L.nn.Linear(6, 8), L.nn.Linear(8, 3)])
        return Net()

    add("ModuleList", build_list,
        lambda L, m: m.load_state_dict({
            "layers.0.weight": L.tensor(w0), "layers.0.bias": L.tensor(b0),
            "layers.1.weight": L.tensor(w1), "layers.1.bias": L.tensor(b1)}),
        lambda L, m, x: m.layers[1](L.relu(m.layers[0](x))),
        "layers.0.weight")

    # **`append` 로 세운 것도 같은 이름이 나와야 한다.** 층 수가 정해지지 않은
    # 모델을 쓰는 법이 이것뿐이고, 생성자로 넣은 것과 갈리면 그 자리에서 갈린다.
    def build_appended(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = L.nn.ModuleList()
                self.layers.append(L.nn.Linear(6, 8))
                self.layers.append(L.nn.Linear(8, 3))
        return Net()

    add("ModuleList(append)", build_appended,
        lambda L, m: m.load_state_dict({
            "layers.0.weight": L.tensor(w0), "layers.0.bias": L.tensor(b0),
            "layers.1.weight": L.tensor(w1), "layers.1.bias": L.tensor(b1)}),
        lambda L, m, x: m.layers[1](L.relu(m.layers[0](x))),
        "layers.1.weight")

    # ── ModuleDict. 이름으로 갈래를 고르는 모델이 쓴다. ─────────────────────
    def build_dict(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = L.nn.ModuleDict({
                    "down": L.nn.Linear(6, 8), "up": L.nn.Linear(8, 3)})
        return Net()

    add("ModuleDict", build_dict,
        lambda L, m: m.load_state_dict({
            "blocks.down.weight": L.tensor(w0), "blocks.down.bias": L.tensor(b0),
            "blocks.up.weight": L.tensor(w1), "blocks.up.bias": L.tensor(b1)}),
        lambda L, m, x: m.blocks["up"](L.relu(m.blocks["down"](x))),
        "blocks.down.weight")

    # ── ParameterList. **여기가 조용히 틀리는 자리다.** ──────────────────────
    #
    # 맨 리스트에 `Parameter` 를 담아 속성으로 붙이면 `__setattr__` 이 그것을
    # `Parameter` 로도 `Module` 로도 못 알아본다 — 어느 목록에도 안 들어가고,
    # `parameters()` 가 안 내놓고, 옵티마이저가 못 본다. torch 도 똑같이 못 알아보고
    # **그래서 `ParameterList` 가 존재한다.** 그것이 없으면 이 자리에 올바른 방법이 없다.
    def build_plist(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.ws = L.nn.ParameterList(
                    [L.nn.Parameter(L.tensor(flat_w)), L.nn.Parameter(L.tensor(b0))])
        return Net()

    add("ParameterList", build_plist,
        lambda L, m: None,                       # 세울 때 이미 고정값이 들어갔다
        lambda L, m, x: x @ m.ws[0] + m.ws[1],
        "ws.0")

    def build_pdict(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.ws = L.nn.ParameterDict({
                    "w": L.nn.Parameter(L.tensor(flat_w)),
                    "b": L.nn.Parameter(L.tensor(b0))})
        return Net()

    add("ParameterDict", build_pdict,
        lambda L, m: None,
        lambda L, m, x: x @ m.ws["w"] + m.ws["b"],
        "ws.w")

    # ── `state_dict` 의 열쇠. 이름이 갈리면 남의 체크포인트를 못 읽는다. ─────
    cases.append((CONTAINER_PREFIX + "상속/state_dict 열쇠",
                  lambda L: " ".join(sorted(build_subclass(L).state_dict()))))
    cases.append((CONTAINER_PREFIX + "ModuleDict/state_dict 열쇠",
                  lambda L: " ".join(sorted(build_dict(L).state_dict()))))

    # ── `eval()` 이 컨테이너를 **뚫고** 내려가는가. ─────────────────────────
    #
    # 안 내려가면 학습은 멀쩡해 보이고 **추론만 틀린다.** 가장 늦게 발견되는 종류다.
    #
    # BatchNorm 으로 묻는다. 갓 세운 것은 `running_mean=0`·`running_var=1` 이라
    # 평가 모드의 출력이 입력과 거의 같고, 학습 모드는 배치 통계로 정규화해서
    # 눈에 띄게 다른 값이 된다 — `eval()` 이 안 내려가면 그 차이가 값에 남는다.
    #
    # 원래 `Dropout` 으로 쓰려 했는데 borch.ts 에 없다(난수 커널이 없다). 여기서
    # 그것을 우회한 것이 아니라, 이 케이스가 묻는 것이 순회이지 Dropout 이 아니다.
    def eval_through(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = L.nn.ModuleList([L.nn.Linear(6, 8), L.nn.BatchNorm1d(8)])
        m = Net()
        m.load_state_dict({"layers.0.weight": L.tensor(w0), "layers.0.bias": L.tensor(b0)},
                          strict=False)
        m.eval()
        return m.layers[1](m.layers[0](L.tensor(xin)))

    cases.append((CONTAINER_PREFIX + "eval 이 컨테이너를 뚫는다", eval_through))
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
    _OPTS = {"SGD": {}, "SGD(모멘텀)": {"momentum": 0.9}, "Adam": {}, "RMSprop": {}}

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

    # 스케줄러는 **파이썬 실수 연산뿐**이라 torch 와 값이 그대로 같아야 한다.
    # 한 값이 아니라 **궤적 전체**를 본다 — 코어가 그렇게 하다가 StepLR 의 차이를 잡았다.
    def lr_traj(L, make, steps=6):
        p = L.tensor([1.0], requires_grad=True)
        opt = L.optim.SGD([p], lr=1.0)
        sch = make(L, opt)
        seen = [opt.param_groups[0]["lr"]]
        for _ in range(steps):
            sch.step()
            seen.append(opt.param_groups[0]["lr"])
        return L.tensor(seen)

    schedules = {
        "StepLR": lambda L, o: L.optim.lr_scheduler.StepLR(o, step_size=2, gamma=0.5),
        "MultiStepLR": lambda L, o: L.optim.lr_scheduler.MultiStepLR(o, [2, 4], gamma=0.5),
        "ExponentialLR": lambda L, o: L.optim.lr_scheduler.ExponentialLR(o, gamma=0.9),
        "CosineAnnealingLR": lambda L, o: L.optim.lr_scheduler.CosineAnnealingLR(o, T_max=6),
        "LambdaLR": lambda L, o: L.optim.lr_scheduler.LambdaLR(o, lambda e: 1.0 / (1 + e)),
    }
    for name, make in schedules.items():
        cases.append((f"sched::{name}", lambda L, m=make: lr_traj(L, m)))

    def plateau(L):
        p = L.tensor([1.0], requires_grad=True)
        opt = L.optim.SGD([p], lr=1.0)
        sch = L.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1, factor=0.5)
        seen = []
        for metric in [1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0]:
            sch.step(metric)
            seen.append(opt.param_groups[0]["lr"])
        return L.tensor(seen)

    cases.append(("sched::ReduceLROnPlateau", plateau))

    # 순환·트랜스포머 — 가중치를 고정해 넣어야 세 라이브러리가 같은 자리에서 출발한다.
    # 파라미터 **이름**이 torch 와 같아야 state_dict 로 넣을 수 있다는 것도 여기서 걸린다.
    seq_x = inp["seq_x"]

    def recurrent(L, kind, batch_first=False):
        mod = getattr(L.nn, kind)(3, 4, batch_first=batch_first)
        # 가중치가 `golden_inputs()` 에서 온다 — 그래야 JSON 으로 나가고 파이썬이
        # 아닌 구현도 **같은 자리에서 출발**할 수 있다.
        low = kind.lower()
        fixed = {f"{n}_l0": L.tensor(inp[f"{low}_{k}"])
                 for n, k in (("weight_ih", "wih"), ("weight_hh", "whh"),
                              ("bias_ih", "bih"), ("bias_hh", "bhh"))}
        mod.load_state_dict(fixed, strict=False)
        return mod

    for kind in ("RNN", "LSTM", "GRU"):
        cases.append((f"seq::{kind}/출력", lambda L, k=kind: recurrent(L, k)(L.tensor(seq_x))[0]))
        cases.append((f"seq::{kind}/마지막상태", lambda L, k=kind: (
            recurrent(L, k)(L.tensor(seq_x))[1][0] if k == "LSTM"
            else recurrent(L, k)(L.tensor(seq_x))[1])))

    def attention(L, mask=None):
        mod = L.nn.MultiheadAttention(4, 2, batch_first=True)
        fixed = {"in_proj_weight": L.tensor(inp["mha_in_w"]),
                 "in_proj_bias": L.tensor(inp["mha_in_b"]),
                 "out_proj.weight": L.tensor(inp["mha_out_w"]),
                 "out_proj.bias": L.tensor(inp["mha_out_b"])}
        mod.load_state_dict(fixed, strict=False)
        x = L.tensor(inp["attn_x"])
        return mod(x, x, x, attn_mask=mask(L) if mask else None, need_weights=False)[0]

    cases.append(("seq::MultiheadAttention", lambda L: attention(L)))
    # 인과 마스크는 **실수**다(0/-inf). "0 이 아니면 가림" 으로 뭉뚱그리면 여기서 갈린다.
    cases.append(("seq::MultiheadAttention(인과 마스크)",
                  lambda L: attention(L, lambda LL: LL.nn.Transformer
                                      .generate_square_subsequent_mask(5))))

    # RMSprop — 옵티마이저 중 유일하게 골든이 안 보던 것
    cases.append(("train::RMSprop/0.weight", lambda L: dict(
        trained(L, "RMSprop").named_parameters())["0.weight"]))
    return cases


WEBGPU_PREFIX = "webgpu::"


def webgpu_cases(inp=None):
    """**브라우저 구현에만 있는 것들.**

    코어는 이것들을 일부러 거절한다 — 커리큘럼이 안 쓰고, 표면이 늘면 조용히 틀릴
    자리가 늘기 때문이다. 브라우저 쪽은 헌장이 달라서(성능·실전 모델) 넣는다.

    기대값은 **진짜 torch** 로 굳힌다. 코어는 이 케이스들을 건너뛴다 — 두 구현의
    범위가 갈리고, 하네스가 그것을 표현해야 한다. 하네스는 `hasattr(lib, "backend")`
    로 브라우저 쪽을 알아본다.
    """
    inp = golden_inputs() if inp is None else inp
    seq = inp["seq_x"].transpose(1, 2, 0).copy()          # (N=2, C=3, L=5)
    img = inp["img"]
    ck1 = inp["ck1"]

    def conv1d_grad(which):
        def run(L, w=which):
            x = L.tensor(seq, requires_grad=True)
            k = L.tensor(ck1, requires_grad=True)
            L.nn.functional.conv1d(x, k, None, 1, 1).sum().backward()
            return _grad_of(x if w == "x" else k, f"conv1d/{w}")
        return run

    cases = [
        (WEBGPU_PREFIX + "F.conv1d",
         lambda L: L.nn.functional.conv1d(L.tensor(seq), L.tensor(ck1), None, 1, 1)),
        (WEBGPU_PREFIX + "F.conv1d(스트라이드2)",
         lambda L: L.nn.functional.conv1d(L.tensor(seq), L.tensor(ck1), None, 2, 1)),
        (WEBGPU_PREFIX + "grad::conv1d/x", conv1d_grad("x")),
        (WEBGPU_PREFIX + "grad::conv1d/w", conv1d_grad("w")),
        (WEBGPU_PREFIX + "F.max_pool1d",
         lambda L: L.nn.functional.max_pool1d(L.tensor(seq), 2)),
        (WEBGPU_PREFIX + "Upsample(최근접)",
         lambda L: L.nn.Upsample(scale_factor=2)(L.tensor(img))),
    ]

    def upsample_grad(L):
        x = L.tensor(img, requires_grad=True)
        L.nn.Upsample(scale_factor=2)(x).sum().backward()
        return _grad_of(x, "Upsample")

    cases.append((WEBGPU_PREFIX + "grad::Upsample", upsample_grad))

    # 3차원 계열. conv3d 의 역방향만 `tf.grad` 를 타서 느리지만, **느린 것은 틀린 것이
    # 아니다** — 값이 맞는지는 여기서 붙잡고, 느리다는 사실은 코드가 경고로 알린다.
    vol = inp["vol5"]
    ck3 = inp["ck3"]

    def conv3d_grad(which):
        def run(L, w=which):
            x = L.tensor(vol, requires_grad=True)
            k = L.tensor(ck3, requires_grad=True)
            L.nn.functional.conv3d(x, k, None, 1, 1).sum().backward()
            return _grad_of(x if w == "x" else k, f"conv3d/{w}")
        return run

    def bn3d_grad(L):
        x = L.tensor(vol, requires_grad=True)
        L.nn.BatchNorm3d(2)(x).sum().backward()
        return _grad_of(x, "BatchNorm3d")

    cases += [
        (WEBGPU_PREFIX + "F.conv3d",
         lambda L: L.nn.functional.conv3d(L.tensor(vol), L.tensor(ck3), None, 1, 1)),
        (WEBGPU_PREFIX + "grad::conv3d/x", conv3d_grad("x")),
        (WEBGPU_PREFIX + "grad::conv3d/w", conv3d_grad("w")),
        (WEBGPU_PREFIX + "F.max_pool3d",
         lambda L: L.nn.functional.max_pool3d(L.tensor(vol), 2)),
        (WEBGPU_PREFIX + "grad::max_pool3d",
         lambda L: _grad_of(_pool3d_leaf(L, vol), "max_pool3d")),
        (WEBGPU_PREFIX + "BatchNorm3d(학습)", lambda L: L.nn.BatchNorm3d(2)(L.tensor(vol))),
        (WEBGPU_PREFIX + "grad::BatchNorm3d", bn3d_grad),
    ]

    # conv3d 를 굳히다가 `tf.pad` 가 랭크 5 에서 **모양만 맞고 값이 깨지는** 것을 잡았다.
    # 예외를 안 던지므로 부르는 쪽은 아무것도 모른다. conv3d 는 고쳤지만, 같은 함수를
    # 부르는 자리가 거기만은 아니다 — 자르기의 역방향이 잘려나간 자리를 0 으로 도로
    # 메울 때 `pad` 를 쓴다. 그 입력이 랭크 5 면 **틀린 기울기가 조용히 나온다.**
    # 그러니 눈으로 훑어 "없더라" 하지 말고, 걸릴 자리를 세워 두고 물어본다.
    def slice5_grad(kind):
        def run(L, k=kind):
            x = L.tensor(vol, requires_grad=True)
            if k == "narrow":
                out = L.narrow(x, 2, 1, 2)
            elif k == "unbind":
                out = L.unbind(x, 2)[1]
            else:
                out = L.split(x, 2, dim=3)[0]
            # 가중치를 다르게 줘야 어느 자리가 0 이어야 하는지가 값으로 드러난다 —
            # 그냥 sum() 이면 기울기가 전부 1 이라 자리가 뒤바뀌어도 안 걸린다.
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(x, f"랭크5 {k}")
        return run

    for kind in ("narrow", "unbind", "split"):
        cases.append((WEBGPU_PREFIX + f"grad::랭크5 {kind}", slice5_grad(kind)))

    # `pad_sequence` 의 **기울기**. 값은 코어와 자매가 같아서 공용 케이스 4건이 보지만,
    # 기울기는 자매에만 있다. 코어 쪽은 numpy 로 자리를 메워 맨 텐서를 돌려주므로 그래프가
    # 끊긴다 — 실측하면 `backward()` 가 "requires_grad 가 아닌 텐서" 라며 거절한다.
    # 진짜 torch 는 미분되므로 그건 코어의 구멍이고, 이 케이스를 자매 전용으로 둔 것은
    # 그 구멍을 덮으려는 것이 아니라 **자매가 안 끊는다는 사실을 붙잡아 두려는** 것이다.
    # (조용히 틀리는 것이 아니라 시끄럽게 거절하는 쪽이라 급한 불은 아니다.)
    def pad_sequence_grad(L):
        a = L.tensor(np.array([[1., 2.], [3., 4.]], dtype=np.float32), requires_grad=True)
        b = L.tensor(np.array([[5., 6.]], dtype=np.float32), requires_grad=True)
        out = L.nn.utils.rnn.pad_sequence([a, b])
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(a, "pad_sequence")

    cases.append((WEBGPU_PREFIX + "grad::pad_sequence", pad_sequence_grad))
    cases += _highrank_battery(HIGH_RANKS, inp)
    cases += _rank_ceiling_cases(CEILING_RANKS, inp)

    # `F.pad` 는 공개 API 로 `tf.pad` 에 닿는 **또 하나의 문**이다. 자르기의 역방향만
    # 고치고 이쪽을 안 봤더니, 같은 버그가 사용자가 직접 부르는 경로에 그대로 남아 있었다.
    # 랭크 6 이상은 배터리가 본다. 여기는 그 아래 둘이다.
    for rank, src in (("랭크4", img), ("랭크5", vol)):
        cases.append((WEBGPU_PREFIX + f"F.pad({rank})",
                      lambda L, s=src: L.nn.functional.pad(L.tensor(s), (1, 2))))
        # 0 이 아닌 값으로도 채운다. 고랭크에서는 0 과 다른 코드가 도므로
        # (`zeros` 대신 `fill`), 0 만 재면 그쪽은 아무도 안 지나간 채 남는다.
        cases.append((WEBGPU_PREFIX + f"F.pad({rank}, 값)",
                      lambda L, s=src: L.nn.functional.pad(
                          L.tensor(s), (2, 1, 1, 0), value=-1.5)))
    return cases


# 자매가 **torch 와 값이 같다고 주장하는** 랭크. 배터리 전부를 통과해야 여기 든다.
HIGH_RANKS = (6,)
# 그 위. TF.js 커널이 없어 **일부 연산만 거절되는** 구간이라 따로 본다 — 자세한 사정은
# `_rank_ceiling_cases` 에 적었다.
CEILING_RANKS = (7, 8)


def _as_expected(fn):
    """**torch 는 되고 브라우저 구현은 거절하는** 자리를 골든에 담는 방법.

    골든은 진짜 torch 로 굳는데, 이 자리는 브라우저 쪽이 torch 와 **일부러 다르다.**
    그래서 값을 물으면 영원히 갈린 채로 남는다. 값 대신 "문서에 적은 대로 굴었는가"를
    묻는다 — torch 는 성공이, 브라우저 쪽은 거절이 정답이라 양쪽 다 제대로면 같은
    답이 나온다.

    브라우저 쪽이 어느 날 조용히 값을 돌려주기 시작하면 (그 값이 맞든 틀리든) 여기서
    갈린다. **그 일이 실제로 났다** — TF.js 판을 걷어내니 랭크 7·8 을 거절하던 일곱
    자리가 전부 값을 냈고, 이 장치가 그것을 일곱 건의 "뜻밖의 성공" 으로 보고했다.
    그때 한 일은 한계를 **의도적으로** 다시 적는 것이었지 케이스를 지우는 것이 아니었다.
    저절로 넓어지면 안 된다는 것이 이 함수가 있는 이유다.
    """
    def run(L):
        # 밑바닥이 GPU 버퍼라 뷰로 나눠 갖지 않는다. 배정도도 없다. 두 이유가
        # 여기 모이는 몇 자리를 만든다.
        must_reject = hasattr(L, "backend")
        try:
            fn(L)
        except Exception as exc:                                # noqa: BLE001
            return "기대대로" if must_reject else f"뜻밖의 거절 <{type(exc).__name__}>"
        return "뜻밖의 성공" if must_reject else "기대대로"
    return run


def _rank_ceiling_cases(ranks, inp):
    """랭크 7 이상 — **되는 것과 안 되는 것이 갈리는 구간.**

    **이 표는 한 번 갈아엎었다.** 원래는 TF.js 의 천장을 못 박는 자리였다 — 재보니
    랭크 7 부터 `GPU for rank 7 is not yet supported` 를 던지는데 전부가 아니라
    일부여서(원소별·permute·reshape 는 돌고 축 축약과 `fill` 이 없다), "랭크 7 은
    된다"도 "안 된다"도 둘 다 거짓이었다. 그래서 되는 쪽은 값으로, 안 되는 쪽은
    `_as_expected` 로 거절 자체를 굳혔다.

    TF.js 를 걷어내면서 그 천장이 사라졌다. 손으로 쓴 WGSL 에는 랭크 한계가 없고,
    거절하던 일곱 자리가 **전부 값을 낸다.** 그래서 전부 값으로 다시 적었다 —
    `_as_expected` 의 주석이 미리 적어둔 그대로다: "TF.js 가 나중에 고랭크 커널을
    채워도 갈린다. 그때는 한계를 **의도적으로** 다시 적으라는 뜻이지 저절로 넓어지면
    안 된다."

    **거절이 값으로 바뀌는 것은 공짜가 아니다.** 안 던진다는 것과 맞는 값이라는 것은
    다른 말이고, 예전 표는 앞의 것만 물을 수 있었다. 이제 진짜 torch 의 값과 맞춘다.
    """
    cases = []
    for r in ranks:
        shape = [2] * r
        axis = r // 2
        shape[axis] = 3
        count = int(np.prod(shape))
        v = inp[f"rank{r}"]
        tag = f"랭크{r}"

        cases += [
            (WEBGPU_PREFIX + f"{tag} 원소별", lambda L, a=v: L.tensor(a) * 2.0 + 1.0),
            (WEBGPU_PREFIX + f"{tag} permute",
             lambda L, a=v, p=tuple(reversed(range(r))): L.tensor(a).permute(*p)),
            (WEBGPU_PREFIX + f"{tag} reshape(내림)",
             lambda L, a=v, s=tuple(shape[:-2]) + (shape[-2] * shape[-1],):
             L.tensor(a).reshape(*s)),
            (WEBGPU_PREFIX + f"{tag} reshape(올림)",
             lambda L, n=count, s=tuple(shape): L.arange(n).float().reshape(*s)),
            (WEBGPU_PREFIX + f"F.pad({tag})",
             lambda L, a=v: L.nn.functional.pad(L.tensor(a), (1, 2))),
        ]

        def elemwise_grad(L, a=v):
            x = L.tensor(a, requires_grad=True)
            (x * x + x).sum().backward()
            return x.grad

        # TF.js 가 거절하던 셋. 지금은 값을 낸다.
        cases += [
            (WEBGPU_PREFIX + f"{tag} 합(축)",
             lambda L, a=v, ax=axis: L.tensor(a).sum(dim=ax)),
            (WEBGPU_PREFIX + f"F.pad({tag}, 값)",
             lambda L, a=v: L.nn.functional.pad(L.tensor(a), (2, 1, 1, 0), value=-1.5)),
            (WEBGPU_PREFIX + f"grad::{tag} 원소별", elemwise_grad),
        ]

    # **여기 적힌 이력을 지우지 않는다.** TF.js 시절 경계는 연산 이름에도 입력 랭크에도
    # 깔끔하게 안 걸렸다 — 랭크 7 은 순방향도 기울기도 되고 랭크 8 은 값은 나오는데
    # 기울기가 없었다. 그래서 넷을 따로 적었고, 그 자리가 지금은 넷 다 값을 낸다.
    #
    # 처음에는 "랭크 8 을 unbind 하면 결과가 랭크 7 이라 거절될 것"이라고 적었다가
    # 물어보니 순방향이 통과했다. 앞서 본 실패는 순방향이 아니라 **기울기**의 것이었고,
    # 나는 실패 이름에 붙은 `grad::` 를 안 읽고 원인을 지어냈다. 짐작으로 경계를 적으면
    # 그 짐작이 문서가 된다는 것이 이 네 줄이 남은 이유다.
    v7 = inp["rank7_unbind"]
    v8 = inp["rank8_unbind"]

    def unbind_grad(arr):
        def run(L):
            x = L.tensor(arr, requires_grad=True)
            out = L.unbind(x, 0)[1]
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(x, "unbind 기울기")
        return run

    cases += [
        (WEBGPU_PREFIX + "랭크7 unbind(순방향)", lambda L: L.unbind(L.tensor(v7), 0)[1]),
        (WEBGPU_PREFIX + "랭크8 unbind(순방향)", lambda L: L.unbind(L.tensor(v8), 0)[1]),
        (WEBGPU_PREFIX + "grad::랭크7 unbind", unbind_grad(v7)),
        (WEBGPU_PREFIX + "grad::랭크8 unbind", unbind_grad(v8)),
    ]
    return cases


VISION_PREFIX = "vision::"
_BT_VISION = None


def _is_real_torch(L):
    return getattr(L, "__name__", "") == "torch"


def _vision(L):
    """`L` 에 짝지어지는 torchvision 을 준다 — 진짜 torch 면 **진짜 torchvision** 이다.

    이것이 이 표의 값어치다. 우리 변환을 우리 기대값에 대조하면 아무것도 증명 못 한다.
    """
    if _is_real_torch(L):
        from torchvision import transforms as real
        return real
    global _BT_VISION
    if _BT_VISION is None:
        try:                                    # 브라우저에서는 /work 가 경로에 있다
            import borch_vision as mod
        except ImportError:                     # 네이티브에서는 저장소 루트를 짚는다
            import importlib.util
            import pathlib
            path = pathlib.Path(__file__).resolve().parent.parent / "borch_vision.py"
            spec = importlib.util.spec_from_file_location("borch_vision", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        _BT_VISION = mod
    _BT_VISION.use(L)                           # 어느 라이브러리의 텐서를 만들지는 여기서
    return _BT_VISION.transforms


def _pil_position(L, arr):
    """torchvision 에서 이 자리에 오는 것은 **PIL 이미지**이고, 우리에게는 PIL 이 없어서
    (H,W,C) 배열이 그 자리를 대신한다. 같은 그림을 각자의 형식으로 준다 — 형식을 맞추지
    않고 비교하면 그건 대조가 아니라 우연이다."""
    if _is_real_torch(L):
        from PIL import Image
        return Image.fromarray(arr)
    return arr


def _as_tensor(L, arr):
    """골든은 `.detach().numpy()` 로 값을 꺼낸다. PIL·배열로 나온 것을 그 규격에 맞춘다."""
    return L.tensor(np.ascontiguousarray(np.asarray(arr, dtype=np.float32)))


def vision_cases(inp=None):
    """`borch_vision` — torchvision 의 `transforms` 만.

    **무작위 변환은 뽑기를 대조할 수 없다.** torch 의 난수기를 우리가 못 쓰기 때문이다.
    그래서 확률을 0 이나 1 로 못 박거나, 자를 자리가 하나뿐이게 만들어 **결정적인
    자리만** 묻는다. 뽑기 자체가 제대로 도는지는 pytest 가 분포로 본다 —
    여기서 "무작위니까 대조 못 한다"고 넘기면 그게 안 본 것을 봤다고 적는 짓이다.
    """
    inp = golden_inputs() if inp is None else inp
    img_u8 = inp["vis_u8"]                                       # (H,W,C)
    img_f = inp["vis_f"]
    gray = inp["vis_gray"]
    mean, std = (0.5, 0.4, 0.3), (0.2, 0.3, 0.4)

    def compose(L):
        T = _vision(L)
        return T.Compose([T.ToTensor(), T.Normalize(mean, std)])(img_u8)

    def normalize(L):
        T = _vision(L)
        return T.Normalize(mean, std)(T.ToTensor()(img_u8))

    def flip(L, p):
        T = _vision(L)
        out = T.RandomHorizontalFlip(p=p)(_pil_position(L, img_u8))
        return _as_tensor(L, out)

    def crop(L, size, padding):
        # 자를 자리가 **하나뿐**이 되게 크기를 맞춘다. 그래야 뽑기와 무관하게 결정적이다.
        T = _vision(L)
        out = T.RandomCrop(size, padding=padding)(_pil_position(L, img_u8))
        return _as_tensor(L, out)

    cases = [
        # ToTensor 의 핵심은 **uint8 일 때만 255 로 나눈다**는 것이다. 실수를 한 번 더
        # 나누면 예외 없이 255 배 어두워지고 학습만 조용히 안 된다.
        (VISION_PREFIX + "ToTensor(uint8)", lambda L: _vision(L).ToTensor()(img_u8)),
        (VISION_PREFIX + "ToTensor(실수)", lambda L: _vision(L).ToTensor()(img_f)),
        (VISION_PREFIX + "ToTensor(2차원)", lambda L: _vision(L).ToTensor()(gray)),
        (VISION_PREFIX + "Normalize", normalize),
        (VISION_PREFIX + "Compose", compose),
        (VISION_PREFIX + "Flip(p=1)", lambda L: flip(L, 1.0)),
        (VISION_PREFIX + "Flip(p=0)", lambda L: flip(L, 0.0)),
        (VISION_PREFIX + "Crop(패딩없음)", lambda L: crop(L, (5, 4), 0)),
        # 패딩을 준 뒤 크기를 딱 맞추면 자를 자리가 하나다 — 패딩 자체가 대조된다.
        (VISION_PREFIX + "Crop(패딩1)", lambda L: crop(L, (7, 6), 1)),
    ]

    # 표현(T3). 이 프로젝트는 `repr` 도 명세로 본다 — 튜토리얼이 `print(transform)` 을
    # 하고, 거기서 다르면 학습자는 다른 것을 배운다.
    reprs = (
        ("ToTensor", lambda T: T.ToTensor()),
        ("Normalize", lambda T: T.Normalize(mean, std)),
        ("RandomHorizontalFlip", lambda T: T.RandomHorizontalFlip(p=0.5)),
        ("RandomCrop", lambda T: T.RandomCrop(32, padding=4)),
        ("Compose", lambda T: T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])),
    )
    for name, build in reprs:
        cases.append((VISION_PREFIX + f"repr::{name}",
                      lambda L, b=build: repr(b(_vision(L)))))
    return cases


def _highrank_battery(ranks, inp):
    """랭크가 올라갈 때 **어디서 무너지는지** 본다.

    처음에는 랭크 6 만 손으로 적었다. 그런데 랭크 5 에서 셋, 랭크 6 에서 하나가
    나왔고 전부 "고쳤다"고 생각한 다음에 나왔다. 그러니 랭크마다 케이스를 베껴 적는
    방식은 다음 랭크를 물을 때 또 베끼게 된다 — 랭크를 인자로 받게 만들어 둔다.

    무엇을 묻는가: 값이 통째로 나오는 것부터 묻는다. 스칼라로 줄이면 자리가 뒤바뀌어도
    합이 같아 통과하기 때문이다. 기울기도 `sum()` 대신 자리마다 다른 가중치를 곱해
    받는다 — 그냥 `sum()` 이면 기울기가 전부 1 이라 0 이 엉뚱한 자리에 박혀도 안 걸린다.

    TF.js 가 정말 못 하는 랭크가 있으면 답은 조용히 틀린 값이 아니라 **거절**이어야
    하고, 그것도 여기서 드러난다 — 골든은 진짜 torch 로 굳으므로 torch 가 내는 값과
    다르면 그것이 예외든 틀린 수든 갈린 것으로 잡힌다.
    """
    cases = []
    for r in ranks:
        # 축이 뒤바뀌면 값보다 **모양**에서 먼저 걸리도록 한 축만 3 으로 둔다.
        shape = [2] * r
        axis = r // 2
        shape[axis] = 3
        count = int(np.prod(shape))
        v = inp[f"rank{r}"]
        tag = f"랭크{r}"

        def slice_grad(kind, arr=v, ax=axis, t=tag):
            def run(L):
                x = L.tensor(arr, requires_grad=True)
                if kind == "narrow":
                    out = L.narrow(x, ax, 1, 2)
                elif kind == "unbind":
                    out = L.unbind(x, ax)[1]
                else:
                    out = L.split(x, 1, dim=ax)[0]
                (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
                return _grad_of(x, f"{t} {kind}")
            return run

        def elemwise_grad(L, arr=v, t=tag):
            x = L.tensor(arr, requires_grad=True)
            (x * x + x).sum().backward()
            return _grad_of(x, f"{t} 원소별")

        cases += [
            (WEBGPU_PREFIX + f"{tag} 원소별", lambda L, a=v: L.tensor(a) * 2.0 + 1.0),
            (WEBGPU_PREFIX + f"{tag} 합(축)",
             lambda L, a=v, ax=axis: L.tensor(a).sum(dim=ax)),
            (WEBGPU_PREFIX + f"{tag} permute",
             lambda L, a=v, p=tuple(reversed(range(r))): L.tensor(a).permute(*p)),
            # 랭크를 내리는 reshape 과 올리는 reshape 을 둘 다 본다. `_dilate` 때
            # 걸린 것이 reshape 과 pad 가 붙은 자리였으므로 따로 떼어 묻는다.
            (WEBGPU_PREFIX + f"{tag} reshape(내림)",
             lambda L, a=v, s=tuple(shape[:-2]) + (shape[-2] * shape[-1],):
             L.tensor(a).reshape(*s)),
            (WEBGPU_PREFIX + f"{tag} reshape(올림)",
             lambda L, n=count, s=tuple(shape): L.arange(n).float().reshape(*s)),
            (WEBGPU_PREFIX + f"grad::{tag} 원소별", elemwise_grad),
            (WEBGPU_PREFIX + f"F.pad({tag})",
             lambda L, a=v: L.nn.functional.pad(L.tensor(a), (1, 2))),
            (WEBGPU_PREFIX + f"F.pad({tag}, 값)",
             lambda L, a=v: L.nn.functional.pad(L.tensor(a), (2, 1, 1, 0), value=-1.5)),
        ]
        for kind in ("narrow", "unbind", "split"):
            cases.append((WEBGPU_PREFIX + f"grad::{tag} {kind}", slice_grad(kind)))
    return cases


def _pool3d_leaf(L, vol):
    x = L.tensor(vol, requires_grad=True)
    L.nn.functional.max_pool3d(x, 2).sum().backward()
    return x


FLOW_PREFIX = "flow::"

# **기울기가 흐르는지만 묻는 표.**
#
# 값만 대조하는 검사는 그래프가 끊긴 것을 못 본다 — 값은 맞기 때문이다. 실제로
# 자매의 `roll` 과 `masked_select` 가 그렇게 조용히 끊겨 있었고, 골든 746건이 전부
# 초록인 채였다. 두 라이브러리에 같은 것을 물어 갈리는지 보는 것이 그것을 잡는 길이다.
#
# 여기 담긴 것은 **셋 다 흘려야 하는 것들**이다. 흐르지 않는 것(`nonzero`·`quantile`·
# `argsort`·`signbit` 등)은 값이 모양에 달렸거나 계단이라 torch 도 안 흘리거나,
# 우리가 일부러 안 넣은 것이라 여기 없다.
_FLOW_OPS = (
    "exp", "log", "sqrt", "abs", "sin", "tanh", "sigmoid", "relu", "erf", "erfc",
    "sinc", "sum", "mean", "prod", "norm", "amax", "amin", "nansum", "nanmean",
    "logsumexp", "cumsum", "cumprod", "median", "msort", "diff", "flip", "roll",
    "tile", "repeat_interleave", "narrow", "index_select", "masked_select",
    "masked_fill", "unbind", "ravel", "clamp", "softmax", "einsum", "diag", "trace",
    "tril", "diagonal", "diagflat", "rot90", "select", "swapaxes", "movedim",
    "det", "logdet", "inverse", "cholesky", "matrix_power", "gather",
)


def flow_cases(inp=None):
    """각 연산이 **기울기를 흘리는가**. 값이 아니라 그 사실을 굳힌다.

    `requires_grad` 하나를 문자열로 답한다 — 값을 묻는 다른 표와 겹치지 않게 하고,
    "흐른다고 했는데 안 흐른다"가 딱 하나의 이유로 갈리게 하기 위해서다.
    """
    vec = np.array([0.5, 2.0, 1.5, 3.0], dtype=np.float32)
    mat = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    pair = np.arange(1., 7., dtype=np.float32).reshape(2, 3)
    sym = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    idx2 = np.array([[0, 2], [1, 0]], dtype=np.int64)
    idx1 = np.array([1, 0], dtype=np.int64)
    mask = np.array([True, False, True, False])

    calls = {
        "exp": (lambda L, x: L.exp(x), vec), "log": (lambda L, x: L.log(x), vec),
        "sqrt": (lambda L, x: L.sqrt(x), vec), "abs": (lambda L, x: L.abs(x), vec),
        "sin": (lambda L, x: L.sin(x), vec), "tanh": (lambda L, x: L.tanh(x), vec),
        "sigmoid": (lambda L, x: L.sigmoid(x), vec),
        "relu": (lambda L, x: L.relu(x), vec), "erf": (lambda L, x: L.erf(x), vec),
        "erfc": (lambda L, x: L.erfc(x), vec), "sinc": (lambda L, x: L.sinc(x), vec),
        "sum": (lambda L, x: x.sum(), vec), "mean": (lambda L, x: x.mean(), vec),
        "prod": (lambda L, x: L.prod(x), vec), "norm": (lambda L, x: L.norm(x), vec),
        "amax": (lambda L, x: L.amax(x), vec), "amin": (lambda L, x: L.amin(x), vec),
        "nansum": (lambda L, x: L.nansum(x), vec),
        "nanmean": (lambda L, x: L.nanmean(x), vec),
        "logsumexp": (lambda L, x: L.logsumexp(x, 0), vec),
        "cumsum": (lambda L, x: L.cumsum(x, 0), vec),
        "cumprod": (lambda L, x: L.cumprod(x, 0), vec),
        "median": (lambda L, x: L.median(x), vec),
        "msort": (lambda L, x: L.msort(x), vec),
        "diff": (lambda L, x: L.diff(x), vec),
        "flip": (lambda L, x: L.flip(x, (0,)), vec),
        "roll": (lambda L, x: L.roll(x, 1), vec),
        "tile": (lambda L, x: L.tile(x, (2,)), vec),
        "repeat_interleave": (lambda L, x: L.repeat_interleave(x, 2), vec),
        "narrow": (lambda L, x: L.narrow(x, 0, 0, 2), vec),
        "index_select": (lambda L, x: L.index_select(x, 0, L.tensor(idx1)), vec),
        "masked_select": (lambda L, x: L.masked_select(x, L.tensor(mask)), vec),
        "masked_fill": (lambda L, x: L.masked_fill(x, L.tensor(mask), 0.0), vec),
        "unbind": (lambda L, x: L.unbind(x, 0)[1], vec),
        "ravel": (lambda L, x: L.ravel(x), vec),
        "clamp": (lambda L, x: L.clamp(x, 1.0, 2.0), vec),
        "softmax": (lambda L, x: L.softmax(x, 0), vec),
        "einsum": (lambda L, x: L.einsum("ij->i", x), mat),
        "diag": (lambda L, x: L.diag(x), mat), "trace": (lambda L, x: L.trace(x), mat),
        "tril": (lambda L, x: L.tril(x), mat),
        "diagonal": (lambda L, x: L.diagonal(x), mat),
        "diagflat": (lambda L, x: L.diagflat(x), vec),
        "rot90": (lambda L, x: L.rot90(x, 1, (0, 1)), mat),
        "select": (lambda L, x: L.select(x, 0, 1), mat),
        "swapaxes": (lambda L, x: L.swapaxes(x, 0, 1), mat),
        "movedim": (lambda L, x: L.movedim(x, 0, 1), mat),
        "det": (lambda L, x: L.det(x), sym), "logdet": (lambda L, x: L.logdet(x), sym),
        "inverse": (lambda L, x: L.inverse(x), sym),
        "cholesky": (lambda L, x: L.linalg.cholesky(x), sym),
        "matrix_power": (lambda L, x: L.matrix_power(x, 2), sym),
        "gather": (lambda L, x: L.gather(x, 1, L.tensor(idx2)), pair),
    }

    def asks(name):
        fn, arr = calls[name]

        def run(L):
            """**두 가지를 함께 답한다.** `requires_grad` 만 물으면 부족하다 —
            `.float()` 은 True 라고 말해놓고 `.grad` 를 `None` 으로 남겼고, 그 검사만
            있었으면 통과했을 것이다. 되짚어서 기울기가 실제로 생겼는지까지 본다.
            """
            x = L.tensor(arr, requires_grad=True)
            out = fn(L, x)
            flow = "흐름" if bool(getattr(out, "requires_grad", False)) else "안흐름"
            try:
                out.sum().backward()
            except Exception:                                       # noqa: BLE001
                return f"{flow}/역전파거절"
            return f"{flow}/{'기울기있음' if x.grad is not None else '조용히None'}"
        return run

    return [(FLOW_PREFIX + name, asks(name)) for name in _FLOW_OPS]


NDIM_PREFIX = "ndim::"


def ndim_cases(inp=None):
    """1차원·3차원 계열. **자매에는 있고 코어에는 거절 stub 이던 자리다.**

    비대칭이 이 프로젝트의 약속과 정면으로 어긋났다 — "임포트만 바꾸면 같은 코드"인데,
    자매에서 돌던 `nn.Conv1d` 가 코어에서 `BrowserTorchError` 로 멈췄다. 이 표가 그
    비대칭이 다시 벌어지는 것을 막는다.
    """
    inp = golden_inputs() if inp is None else inp
    seq, k1 = inp["nd_seq"], inp["nd_k1"]
    vol, k3 = inp["nd_vol"], inp["nd_k3"]
    img = inp["nd_img"]

    calls = (
        ("F.conv1d", lambda L: L.nn.functional.conv1d(
            L.tensor(seq), L.tensor(k1), None, 1, 1), seq),
        ("F.conv1d(걸음2)", lambda L: L.nn.functional.conv1d(
            L.tensor(seq), L.tensor(k1), None, 2, 1), seq),
        ("F.conv1d(채움0)", lambda L: L.nn.functional.conv1d(
            L.tensor(seq), L.tensor(k1), None, 1, 0), seq),
        ("F.conv3d", lambda L: L.nn.functional.conv3d(
            L.tensor(vol), L.tensor(k3), None, 1, 1), vol),
        ("F.conv3d(채움0)", lambda L: L.nn.functional.conv3d(
            L.tensor(vol), L.tensor(k3), None, 1, 0), vol),
        ("F.max_pool1d", lambda L: L.nn.functional.max_pool1d(L.tensor(seq), 2), seq),
        ("F.max_pool3d", lambda L: L.nn.functional.max_pool3d(L.tensor(vol), 2), vol),
        ("F.interpolate", lambda L: L.nn.functional.interpolate(
            L.tensor(img), scale_factor=2), img),
        ("F.adaptive_avg_pool2d", lambda L: L.nn.functional.adaptive_avg_pool2d(
            L.tensor(img), 2), img),
    )
    cases = [(NDIM_PREFIX + name, fn) for name, fn, _ in calls]

    # 모듈 쪽. 가중치는 밖에서 넣어 **양쪽이 같은 자리에서 출발**하게 한다 —
    # 각자 초기화하면 무엇이 갈렸는지가 아니라 초기화가 갈렸는지를 보게 된다.
    def conv1d_module(L):
        m = L.nn.Conv1d(3, 4, 3, padding=1)
        # **`load_state_dict` 로 넣는다.** `m.weight.data[...] = ndarray` 는 torch 가
        # 거절한다(`can't assign a numpy.ndarray to a torch.FloatTensor`) — 셋 다
        # 통하는 길은 이것뿐이고, 마침 이 라이브러리가 맞춰둔 표면이기도 하다.
        m.load_state_dict({"weight": L.tensor(k1),
                           "bias": L.tensor(np.zeros(4, dtype=np.float32))})
        return m(L.tensor(seq))

    cases += [
        (NDIM_PREFIX + "nn.Conv1d", conv1d_module),
        (NDIM_PREFIX + "nn.MaxPool1d", lambda L: L.nn.MaxPool1d(2)(L.tensor(seq))),
        (NDIM_PREFIX + "nn.MaxPool3d", lambda L: L.nn.MaxPool3d(2)(L.tensor(vol))),
        (NDIM_PREFIX + "nn.BatchNorm3d", lambda L: L.nn.BatchNorm3d(2)(L.tensor(vol))),
        (NDIM_PREFIX + "nn.Upsample",
         lambda L: L.nn.Upsample(scale_factor=2)(L.tensor(img))),
    ]

    # 메서드로만 있던 것들의 **함수 형태**. torch 는 둘 다 주는데 코어는 한쪽만
    # 있었고, 자매와 맞춰보다 드러났다.
    mask = np.array([[True, False, True, False]] * 2)
    flat = np.arange(8, dtype=np.float32).reshape(2, 4)
    cases += [
        (NDIM_PREFIX + "torch.matmul",
         lambda L: L.matmul(L.tensor(flat), L.tensor(flat.T.copy()))),
        # **모양을 튜플 하나로 받는다.** `torch.reshape(x, 4, 2)` 는 torch 가 거절한다 —
        # 메서드(`x.reshape(4, 2)`)와 함수의 서명이 다르다.
        (NDIM_PREFIX + "torch.reshape", lambda L: L.reshape(L.tensor(flat), (4, 2))),
        (NDIM_PREFIX + "torch.unsqueeze", lambda L: L.unsqueeze(L.tensor(flat), 1)),
        (NDIM_PREFIX + "torch.masked_fill",
         lambda L: L.masked_fill(L.tensor(flat), L.tensor(mask), -1.0)),
        (NDIM_PREFIX + "x.masked_fill",
         lambda L: L.tensor(flat).masked_fill(L.tensor(mask), -1.0)),
        (NDIM_PREFIX + "x.index_select",
         lambda L: L.tensor(flat).index_select(0, L.tensor(np.array([1, 0], dtype=np.int64)))),
        (NDIM_PREFIX + "x.masked_select",
         lambda L: L.tensor(flat).masked_select(L.tensor(mask))),
        (NDIM_PREFIX + "x.repeat_interleave",
         lambda L: L.tensor(flat).repeat_interleave(2)),
    ]

    grads = (
        ("conv1d", lambda L, x: L.nn.functional.conv1d(x, L.tensor(k1), None, 1, 1), seq),
        ("conv3d", lambda L, x: L.nn.functional.conv3d(x, L.tensor(k3), None, 1, 1), vol),
        ("max_pool1d", lambda L, x: L.nn.functional.max_pool1d(x, 2), seq),
        ("max_pool3d", lambda L, x: L.nn.functional.max_pool3d(x, 2), vol),
        ("interpolate", lambda L, x: L.nn.functional.interpolate(x, scale_factor=2), img),
        ("BatchNorm3d", lambda L, x: L.nn.BatchNorm3d(2)(x), vol),
    )
    for name, fn, arr in grads:
        def run(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(x, n)
        cases.append((NDIM_PREFIX + f"grad::{name}", run))
    return cases


LINALG_PREFIX = "linalg::"


def linalg_cases(inp=None):
    """선형대수 분해. **양쪽 다 넣었다** — 자매는 CPU 로 읽어와 numpy 로 계산하고,
    처음 부를 때 한 번 느리다고 경고한다.

    기울기는 닫힌 꼴이 있는 것만 있다(`det`·`logdet`·`inverse`·`solve`·`cholesky`·
    `matrix_power`). `qr`·`svd`·`pinverse`·`lstsq` 는 값만 준다 — torch 는 미분하는데
    우리는 안 한다. 유도가 까다롭고 틀리면 조용히 틀리므로, 없는 것을 시끄럽게 둔다.
    """
    mat = np.array([[4., 1.], [2., 3.]], dtype=np.float32)
    sym = np.array([[4., 1.], [1., 3.]], dtype=np.float32)      # 대칭 양정부호
    vec = np.array([1., 2.], dtype=np.float32)

    cases = [
        (LINALG_PREFIX + "det", lambda L: L.det(L.tensor(mat))),
        (LINALG_PREFIX + "logdet", lambda L: L.logdet(L.tensor(sym))),
        (LINALG_PREFIX + "slogdet/부호", lambda L: L.slogdet(L.tensor(mat))[0]),
        (LINALG_PREFIX + "slogdet/로그", lambda L: L.slogdet(L.tensor(mat))[1]),
        (LINALG_PREFIX + "inverse", lambda L: L.inverse(L.tensor(mat))),
        (LINALG_PREFIX + "pinverse", lambda L: L.pinverse(L.tensor(mat))),
        (LINALG_PREFIX + "matrix_power", lambda L: L.matrix_power(L.tensor(mat), 3)),
        (LINALG_PREFIX + "matrix_power(음수)",
         lambda L: L.matrix_power(L.tensor(mat), -1)),
        (LINALG_PREFIX + "cholesky", lambda L: L.linalg.cholesky(L.tensor(sym))),
        (LINALG_PREFIX + "solve", lambda L: L.linalg.solve(L.tensor(mat), L.tensor(vec))),
        (LINALG_PREFIX + "matrix_rank",
         lambda L: L.linalg.matrix_rank(L.tensor(mat))),
        # torch 의 `linalg.lstsq` 는 해 말고도 잔차·랭크를 같이 준다. `.solution` 으로
        # 물어야 셋이 같은 것을 비교한다 — 우리 것도 그 이름을 갖게 했다.
        (LINALG_PREFIX + "lstsq",
         lambda L: L.linalg.lstsq(L.tensor(mat), L.tensor(vec)).solution),
        (LINALG_PREFIX + "eigh/고윳값", lambda L: L.linalg.eigh(L.tensor(sym))[0]),
        # `torch.linalg` 이름으로도 닿아야 한다 — 튜토리얼이 그쪽을 쓴다.
        (LINALG_PREFIX + "linalg.det", lambda L: L.linalg.det(L.tensor(mat))),
        (LINALG_PREFIX + "linalg.inv", lambda L: L.linalg.inv(L.tensor(mat))),
        (LINALG_PREFIX + "qr/R", lambda L: L.linalg.qr(L.tensor(mat))[1]),
    ]

    # **부호 규약이 구현마다 다르다.** QR 의 Q 와 SVD 의 U·Vh 는 열 부호를 뒤집어도
    # 같은 분해이므로 절댓값으로 묻는다 — 부호까지 맞추라고 하면 numpy 와 LAPACK 의
    # 규약 차이를 재는 것이지 우리 구현을 재는 것이 아니다.
    cases += [
        (LINALG_PREFIX + "qr/|Q|", lambda L: L.linalg.qr(L.tensor(mat))[0].abs()),
        (LINALG_PREFIX + "svd/|U|", lambda L: L.linalg.svd(L.tensor(mat))[0].abs()),
        (LINALG_PREFIX + "svd/S", lambda L: L.linalg.svd(L.tensor(mat))[1]),
        (LINALG_PREFIX + "svd/|Vh|", lambda L: L.linalg.svd(L.tensor(mat))[2].abs()),
    ]

    grads = (
        ("det", lambda L, x: L.det(x), mat),
        ("logdet", lambda L, x: L.logdet(x), sym),
        ("slogdet", lambda L, x: L.slogdet(x)[1], mat),
        ("inverse", lambda L, x: L.inverse(x), mat),
        ("cholesky", lambda L, x: L.linalg.cholesky(x), sym),
        ("matrix_power", lambda L, x: L.matrix_power(x, 3), mat),
    )
    for name, fn, arr in grads:
        def run(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            if out.shape:
                out = out * L.arange(out.numel()).reshape(out.shape).float()
            out.sum().backward()
            return _grad_of(x, n)
        cases.append((LINALG_PREFIX + f"grad::{name}", run))

    def solve_grad(which):
        def run(L, w=which):
            a = L.tensor(mat, requires_grad=True)
            b = L.tensor(vec, requires_grad=True)
            out = L.linalg.solve(a, b)
            (out * L.tensor(np.array([1., 2.], dtype=np.float32))).sum().backward()
            return _grad_of(a if w == "a" else b, f"solve/{w}")
        return run

    for who in ("a", "b"):
        cases.append((LINALG_PREFIX + f"grad::solve/{who}", solve_grad(who)))
    return cases


# 배치·직사각용 입력. **손으로 적는다** — 난수는 특이행렬도 행 교환도 안 만들고,
# 여기서 재려는 것이 바로 그 두 자리다.
_LA_BATCH = np.array([[[4., 1.], [2., 3.]],
                      [[2., 0.], [1., 5.]],
                      [[3., -1.], [1., 2.]]], dtype=np.float32)
_LA_BATCH_SYM = np.array([[[4., 1.], [1., 3.]],
                          [[9., 2.], [2., 5.]],
                          [[2., 0.5], [0.5, 1.]]], dtype=np.float32)
_LA_BATCH_VEC = np.array([[1., 2.], [3., 1.], [0., 4.]], dtype=np.float32)
_LA_BATCH_RHS = np.array([[[1., 0.], [2., 1.]],
                          [[0., 3.], [1., 1.]],
                          [[2., 2.], [0., 1.]]], dtype=np.float32)
_LA_RECT = np.array([[1., 2.], [3., 4.], [5., 7.]], dtype=np.float32)
_LA_SYM3 = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]], dtype=np.float32)
# 첫 원소가 두 번째 행보다 작다 → 부분 피벗이 행을 바꾼다. 안 바꾸는 행렬만 물으면
# pivots 가 항등이라 **1 부터 세는지 0 부터 세는지 구분이 안 된다.**
_LA_PIVOT = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
_LA_SINGULAR = np.array([[1., 2.], [2., 4.]], dtype=np.float32)


def linalg_struct_cases(inp=None):
    """`linalg` 의 **구조** — 배치·직사각·이름·`_ex`·LU.

    앞의 `linalg_cases` 는 2 차원 정사각 하나만 묻는다. 그런데 torch 의 `linalg` 는
    **전부 배치**다: `det((3,2,2))` 이 `(3,)` 을 내고 `inv`·`solve`·`cholesky`·
    `slogdet`·`matrix_rank` 가 다 그렇다. 한 장만 묻는 골든은 배치를 못 본다.

    ## 이름으로도 물어야 한다

    torch 는 이 결과들을 이름 붙은 튜플로 준다 — `slogdet(A).logabsdet`,
    `qr(A).Q`, `lu_factor(A).pivots`, `inv_ex(A).info`. 자리로만 맞춰 놓으면 값이
    맞는데도 교재 코드가 속성 접근에서 멈춘다. `lstsq` 가 `.solution` 으로 이미
    같은 자리를 겪었다.

    ## `_ex` 는 안 던지는 쪽이다

    `inv` 는 특이행렬에서 `LinAlgError` 를 던지고 `inv_ex` 는 `info` 에 0 이 아닌
    수를 담아 조용히 돌아온다. **둘 다 있어야 한다** — 던지는 쪽이 없으면 특이행렬이
    NaN 으로 새고, 안 던지는 쪽이 없으면 배치에서 한 장이 특이일 때 전부가 죽는다.

    ## 피벗은 1 부터 센다

    `lu_factor` 의 `pivots` 가 LAPACK 규약이라 **1 부터 시작한다.** 교환이 없는
    행렬에서는 `[1, 2]` 이지 `[0, 1]` 이 아니다. 이것을 0 부터 세면 `lu_solve` 가
    조용히 다른 답을 낸다 — 그래서 행을 실제로 바꾸는 행렬로도 묻는다.
    """
    bat, sym, vec = _LA_BATCH, _LA_BATCH_SYM, _LA_BATCH_VEC
    rhs, rect, sym3 = _LA_BATCH_RHS, _LA_RECT, _LA_SYM3

    cases = [
        # ── 배치 ────────────────────────────────────────────────────────
        (LINALG_PREFIX + "batch::det", lambda L: L.linalg.det(L.tensor(bat))),
        (LINALG_PREFIX + "batch::inv", lambda L: L.linalg.inv(L.tensor(bat))),
        (LINALG_PREFIX + "batch::solve(벡터)",
         lambda L: L.linalg.solve(L.tensor(bat), L.tensor(vec))),
        (LINALG_PREFIX + "batch::solve(행렬)",
         lambda L: L.linalg.solve(L.tensor(bat), L.tensor(rhs))),
        (LINALG_PREFIX + "batch::cholesky",
         lambda L: L.linalg.cholesky(L.tensor(sym))),
        (LINALG_PREFIX + "batch::slogdet/부호",
         lambda L: L.linalg.slogdet(L.tensor(bat))[0]),
        (LINALG_PREFIX + "batch::slogdet/로그",
         lambda L: L.linalg.slogdet(L.tensor(bat))[1]),
        (LINALG_PREFIX + "batch::matrix_rank",
         lambda L: L.linalg.matrix_rank(L.tensor(bat))),
        (LINALG_PREFIX + "batch::matrix_power",
         lambda L: L.linalg.matrix_power(L.tensor(bat), 3)),
        (LINALG_PREFIX + "batch::qr/R", lambda L: L.linalg.qr(L.tensor(bat))[1]),
        (LINALG_PREFIX + "batch::svd/S", lambda L: L.linalg.svd(L.tensor(bat))[1]),
        (LINALG_PREFIX + "batch::eigh/값",
         lambda L: L.linalg.eigh(L.tensor(sym))[0]),
        (LINALG_PREFIX + "batch::pinv", lambda L: L.linalg.pinv(L.tensor(bat))),
        (LINALG_PREFIX + "batch::logdet", lambda L: L.logdet(L.tensor(sym))),
        # 3×3 도 묻는다 — 2×2 는 야코비 회전이 한 번뿐이라 쓸어담기 반복을 안 지난다.
        (LINALG_PREFIX + "3x3::eigh/값", lambda L: L.linalg.eigh(L.tensor(sym3))[0]),
        (LINALG_PREFIX + "3x3::svd/S", lambda L: L.linalg.svd(L.tensor(sym3))[1]),
        (LINALG_PREFIX + "3x3::det", lambda L: L.linalg.det(L.tensor(sym3))),
        (LINALG_PREFIX + "3x3::inv", lambda L: L.linalg.inv(L.tensor(sym3))),

        # ── 직사각 ──────────────────────────────────────────────────────
        # 정사각만 되면 `qr`·`svd`·`pinv` 는 쓸 자리의 절반을 못 받는다 —
        # 최소제곱이 바로 그 절반이다.
        (LINALG_PREFIX + "rect::qr/R", lambda L: L.linalg.qr(L.tensor(rect))[1]),
        (LINALG_PREFIX + "rect::qr/|Q|",
         lambda L: L.linalg.qr(L.tensor(rect))[0].abs()),
        (LINALG_PREFIX + "rect::qr(complete)/|Q|",
         lambda L: L.linalg.qr(L.tensor(rect), mode="complete")[0].abs()),
        (LINALG_PREFIX + "rect::svd/S", lambda L: L.linalg.svd(L.tensor(rect))[1]),
        (LINALG_PREFIX + "rect::svd/|U|",
         lambda L: L.linalg.svd(L.tensor(rect))[0].abs()),
        (LINALG_PREFIX + "rect::svd(축소)/|U|",
         lambda L: L.linalg.svd(L.tensor(rect), full_matrices=False)[0].abs()),
        (LINALG_PREFIX + "rect::pinv", lambda L: L.linalg.pinv(L.tensor(rect))),
        (LINALG_PREFIX + "rect::matrix_rank",
         lambda L: L.linalg.matrix_rank(L.tensor(rect))),
        (LINALG_PREFIX + "rect::lstsq",
         lambda L: L.linalg.lstsq(L.tensor(rect),
                                  L.tensor(np.array([1., 2., 3.], dtype=np.float32))
                                  ).solution),
    ]

    # ── 이름으로 묻기 ───────────────────────────────────────────────────
    named = (
        ("slogdet.sign", lambda L: L.linalg.slogdet(L.tensor(bat)).sign),
        ("slogdet.logabsdet", lambda L: L.linalg.slogdet(L.tensor(bat)).logabsdet),
        ("qr.R", lambda L: L.linalg.qr(L.tensor(rect)).R),
        ("qr.|Q|", lambda L: L.linalg.qr(L.tensor(rect)).Q.abs()),
        ("svd.S", lambda L: L.linalg.svd(L.tensor(rect)).S),
        ("svd.|Vh|", lambda L: L.linalg.svd(L.tensor(rect)).Vh.abs()),
        ("eigh.eigenvalues", lambda L: L.linalg.eigh(L.tensor(sym3)).eigenvalues),
        ("eigh.|eigenvectors|",
         lambda L: L.linalg.eigh(L.tensor(sym3)).eigenvectors.abs()),
    )
    cases += [(LINALG_PREFIX + f"name::{n}", f) for n, f in named]

    # ── `_ex` — 던지는 대신 info 를 준다 ────────────────────────────────
    mat2 = np.array([[4., 1.], [2., 3.]], dtype=np.float32)
    ex = (
        ("inv_ex/값", lambda L: L.linalg.inv_ex(L.tensor(mat2)).inverse),
        ("inv_ex/info", lambda L: L.linalg.inv_ex(L.tensor(mat2)).info),
        ("inv_ex(특이)/info",
         lambda L: L.linalg.inv_ex(L.tensor(_LA_SINGULAR)).info),
        ("cholesky_ex/L",
         lambda L: L.linalg.cholesky_ex(L.tensor(_LA_BATCH_SYM[0])).L),
        ("cholesky_ex(비양정)/info",
         lambda L: L.linalg.cholesky_ex(L.tensor(_LA_SINGULAR)).info),
        ("solve_ex/값",
         lambda L: L.linalg.solve_ex(L.tensor(mat2),
                                     L.tensor(np.array([1., 2.], dtype=np.float32))
                                     ).result),
        ("solve_ex/info",
         lambda L: L.linalg.solve_ex(L.tensor(mat2),
                                     L.tensor(np.array([1., 2.], dtype=np.float32))
                                     ).info),
    )
    cases += [(LINALG_PREFIX + f"ex::{n}", f) for n, f in ex]

    def catches(L):
        """**`except torch.linalg.LinAlgError` 가 잡는가.**

        예외의 클래스 이름은 양쪽이 다를 수 있다(진짜 torch 는 `_LinAlgError` 다).
        이름을 맞추라고 하면 남의 사정을 재는 것이므로, 사용자가 실제로 쓰는 것 —
        그 이름으로 잡히는지 — 를 묻는다.
        """
        try:
            L.linalg.inv(L.tensor(_LA_SINGULAR))
        except L.linalg.LinAlgError:
            return "LinAlgError 로 잡힌다"
        except Exception as exc:                                    # noqa: BLE001
            return f"다른 것이 났다: {type(exc).__name__}"
        return "예외가 안 났다"

    cases.append((LINALG_PREFIX + "ex::inv(특이)가 던지는 것", catches))

    # ── LU — 이미 안에서 구하던 것을 밖으로 ─────────────────────────────
    lu_inputs = (("교환없음", mat2), ("교환", _LA_PIVOT))
    for tag, arr in lu_inputs:
        cases += [
            (LINALG_PREFIX + f"lu::lu_factor/{tag}/LU",
             lambda L, a=arr: L.linalg.lu_factor(L.tensor(a)).LU),
            (LINALG_PREFIX + f"lu::lu_factor/{tag}/pivots",
             lambda L, a=arr: L.linalg.lu_factor(L.tensor(a)).pivots),
            (LINALG_PREFIX + f"lu::lu/{tag}/P",
             lambda L, a=arr: L.linalg.lu(L.tensor(a)).P),
            (LINALG_PREFIX + f"lu::lu/{tag}/L",
             lambda L, a=arr: L.linalg.lu(L.tensor(a)).L),
            (LINALG_PREFIX + f"lu::lu/{tag}/U",
             lambda L, a=arr: L.linalg.lu(L.tensor(a)).U),
        ]

    def lu_solve(L):
        a = L.tensor(_LA_PIVOT)
        f = L.linalg.lu_factor(a)
        return L.linalg.lu_solve(f.LU, f.pivots,
                                 L.tensor(np.array([[1.], [2.]], dtype=np.float32)))

    cases.append((LINALG_PREFIX + "lu::lu_solve(교환)", lu_solve))

    # ── `_ex`·LDL·반사자 ────────────────────────────────────────────────
    #
    # `lu_factor_ex` 는 던지는 대신 **`info` 로 알린다.** 0 이면 잘 됐고, `k` 면
    # `k` 번째 피벗이 0 이다(1 부터 센다). 특이행렬로도 물어야 그 수가 드러난다.
    lin4 = np.array([[2.0, 1.0, 0.5, -1.0], [1.0, 3.0, -0.5, 0.25],
                     [0.5, -0.5, 2.5, 0.75], [-1.0, 0.25, 0.75, 4.0]],
                    dtype=np.float32)
    singular2 = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    rect53 = (np.arange(15, dtype=np.float32).reshape(5, 3) / 4 - 1.5
              + np.eye(5, 3, dtype=np.float32) * 3)

    for tag, arr in (("정사각", _LA_PIVOT), ("직사각", rect53)):
        for field in ("LU", "pivots", "info"):
            cases.append((
                LINALG_PREFIX + f"ex::lu_factor_ex/{tag}/{field}",
                lambda L, a=arr, f=field: getattr(
                    L.linalg.lu_factor_ex(L.tensor(a)), f)))
    cases.append((LINALG_PREFIX + "ex::lu_factor_ex/특이행렬 info",
                  lambda L: L.linalg.lu_factor_ex(L.tensor(singular2)).info))

    # LDL 은 **대칭**에만 뜻이 있다. `lin4` 는 대칭으로 지었다.
    for field in ("LD", "pivots"):
        cases.append((LINALG_PREFIX + f"ex::ldl_factor/{field}",
                      lambda L, f=field: getattr(
                          L.linalg.ldl_factor(L.tensor(lin4)), f)))
    cases.append((LINALG_PREFIX + "ex::ldl_factor_ex/info",
                  lambda L: L.linalg.ldl_factor_ex(L.tensor(lin4)).info))

    def ldl_solve(L):
        f = L.linalg.ldl_factor(L.tensor(lin4))
        b = np.array([[1.0, -2.0], [0.5, 0.25], [-1.5, 3.0], [2.0, 0.5]],
                     dtype=np.float32)
        return L.linalg.ldl_solve(f.LD, f.pivots, L.tensor(b))

    cases.append((LINALG_PREFIX + "ex::ldl_solve", ldl_solve))

    # **반사자 꼴 QR.** `geqrf` 가 담고 `householder_product` 가 `Q` 로 편다.
    #
    # **정사각으로도 물어야 한다.** LAPACK 은 대각 아래가 전부 0 이면 반사를 안 하고
    # (`tau = 0`) 값을 그대로 두는데, 정사각의 마지막 열이 늘 그 자리다. 직사각으로만
    # 물으면 그 열이 안 나와서, 거기서 부호를 뒤집어도 안 걸린다 — 실제로 그랬다.
    for tag, arr in (("정사각", lin4), ("직사각", rect53)):
        cases.append((LINALG_PREFIX + f"ex::geqrf/{tag}/a",
                      lambda L, a=arr: L.geqrf(L.tensor(a))[0]))
        cases.append((LINALG_PREFIX + f"ex::geqrf/{tag}/tau",
                      lambda L, a=arr: L.geqrf(L.tensor(a))[1]))

        def product(L, a=arr):
            got = L.geqrf(L.tensor(a))
            return L.linalg.householder_product(got[0], got[1])

        cases.append((LINALG_PREFIX + f"ex::householder_product/{tag}", product))

    # ── 배치의 기울기 ───────────────────────────────────────────────────
    # **값이 맞는데 기울기가 안 맞는 자리가 여기다.** 역방향 식이 `.T` 로 적혀 있으면
    # 2 차원에서는 맞고 배치에서는 축을 통째로 뒤집어 조용히 틀린다.
    bgrads = (
        ("det", lambda L, x: L.linalg.det(x), bat),
        ("logdet", lambda L, x: L.logdet(x), sym),
        ("slogdet", lambda L, x: L.linalg.slogdet(x)[1], bat),
        ("inv", lambda L, x: L.linalg.inv(x), bat),
        ("cholesky", lambda L, x: L.linalg.cholesky(x), sym),
        ("matrix_power", lambda L, x: L.linalg.matrix_power(x, 3), bat),
        ("3x3/inv", lambda L, x: L.linalg.inv(x), sym3),
        ("3x3/cholesky", lambda L, x: L.linalg.cholesky(x), sym3),
    )
    for name, fn, arr in bgrads:
        def run(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            if out.shape:
                out = out * L.arange(out.numel()).reshape(out.shape).float()
            out.sum().backward()
            return _grad_of(x, n)
        cases.append((LINALG_PREFIX + f"batch::grad::{name}", run))

    def batch_solve_grad(which, rhs_arr, tag):
        def run(L, w=which, r=rhs_arr, g=tag):
            a = L.tensor(bat, requires_grad=True)
            b = L.tensor(r, requires_grad=True)
            out = L.linalg.solve(a, b)
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(a if w == "a" else b, f"batch solve {g}/{w}")
        return run

    for tag, arr in (("벡터", vec), ("행렬", rhs)):
        for who in ("a", "b"):
            cases.append((LINALG_PREFIX + f"batch::grad::solve({tag})/{who}",
                          batch_solve_grad(who, arr, tag)))
    return cases


_LA_MAT = np.array([[4., 1.], [2., 3.]], dtype=np.float32)
_LA_VEC3 = np.array([3., -4., 0.], dtype=np.float32)
_LA_UPPER = np.array([[2., 1.], [0., 3.]], dtype=np.float32)
_LA_CUBE = np.arange(24, dtype=np.float32).reshape(2, 3, 4)


def linalg_name_cases(inp=None):
    """`linalg` 의 **조합층** — 있는 것에 이름을 붙이는 자리와, 갈래가 있는 노름.

    앞의 둘이 구조였다면 이쪽은 표면이다. 대부분 이미 있는 것의 조합이라 계산이
    새로 필요한 것은 `matrix_exp` 하나뿐이다. 그래도 **조합이 자명하지 않은 자리**가
    셋 있다.

    ## 노름은 갈래가 값을 바꾼다

    `matrix_norm` 의 기본은 프로베니우스인데 `ord=2` 는 최대 특잇값, `nuc` 는 특잇값의
    합, `1` 은 열 절댓값 합의 최대, `inf` 는 행 쪽이다 — **전부 다른 수**다. 골든이
    rank 2 인 행렬로 묻는 이유가 그것이다. rank 1 을 주면 프로베니우스·2·핵 노름이
    우연히 같아져서 셋을 구분 못 한다.

    ## `linalg.diagonal` 은 `torch.diagonal` 과 **기본 축이 다르다**

    `linalg` 쪽은 마지막 두 축을 보고(`dim1=-2, dim2=-1`) `torch` 쪽은 앞의 두 축을
    본다. 3 차원을 주면 `(2,3,4)` 가 각각 `(2,3)` 과 `(4,2)` 로 갈린다 — 이름이 비슷해
    같은 것으로 읽기 쉬운데 모양부터 다르다.

    ## `eigh` 는 **아래 삼각만 읽는다**

    `[[4,99],[1,3]]` 과 `[[4,1],[1,3]]` 의 답이 같다(실측). 대칭이 아닌 것을 주면
    위쪽은 무시하고 아래쪽을 거울로 삼는다. 행렬 전체를 보는 구현은 여기서 갈리는데,
    **대칭을 주는 한 안 드러난다** — 그래서 일부러 안 대칭인 것을 묻는다.
    """
    mat, rect, sym3 = _LA_MAT, _LA_RECT, _LA_SYM3
    vec3, upper, cube = _LA_VEC3, _LA_UPPER, _LA_CUBE
    sym = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    skew = np.array([[4., 99.], [1., 3.]], dtype=np.float32)   # 아래 삼각만 읽어야

    cases = [
        # ── 이미 있는 것에 이름 붙이기 ──────────────────────────────────
        (LINALG_PREFIX + "name2::matmul",
         lambda L: L.linalg.matmul(L.tensor(mat), L.tensor(mat))),
        (LINALG_PREFIX + "name2::vecdot",
         lambda L: L.linalg.vecdot(L.tensor(mat), L.tensor(mat))),
        (LINALG_PREFIX + "name2::cross",
         lambda L: L.linalg.cross(L.tensor(np.array([1., 2., 3.], dtype=np.float32)),
                                  L.tensor(np.array([4., 5., 6.], dtype=np.float32)))),
        (LINALG_PREFIX + "name2::svdvals",
         lambda L: L.linalg.svdvals(L.tensor(mat))),
        (LINALG_PREFIX + "name2::svdvals(직사각)",
         lambda L: L.linalg.svdvals(L.tensor(rect))),
        (LINALG_PREFIX + "name2::eigvalsh",
         lambda L: L.linalg.eigvalsh(L.tensor(sym))),
        (LINALG_PREFIX + "name2::eigvalsh(3x3)",
         lambda L: L.linalg.eigvalsh(L.tensor(sym3))),
        # **아래 삼각만 읽는가.** 위쪽에 99 를 넣어도 답이 안 바뀌어야 한다.
        (LINALG_PREFIX + "name2::eigvalsh(아래삼각만)",
         lambda L: L.linalg.eigvalsh(L.tensor(skew))),
        (LINALG_PREFIX + "name2::eigh(아래삼각만)/값",
         lambda L: L.linalg.eigh(L.tensor(skew))[0]),

        # ── 축이 갈리는 자리 ────────────────────────────────────────────
        (LINALG_PREFIX + "name2::linalg.diagonal",
         lambda L: L.linalg.diagonal(L.tensor(cube))),
        (LINALG_PREFIX + "name2::torch.diagonal(다른 축)",
         lambda L: L.diagonal(L.tensor(cube))),
        (LINALG_PREFIX + "name2::linalg.diagonal(offset)",
         lambda L: L.linalg.diagonal(L.tensor(mat), offset=1)),

        # ── 노름의 갈래 ────────────────────────────────────────────────
        (LINALG_PREFIX + "name2::vector_norm",
         lambda L: L.linalg.vector_norm(L.tensor(vec3))),
        (LINALG_PREFIX + "name2::vector_norm(행렬을 통째로)",
         lambda L: L.linalg.vector_norm(L.tensor(mat))),
        (LINALG_PREFIX + "name2::vector_norm(dim)",
         lambda L: L.linalg.vector_norm(L.tensor(mat), dim=1)),
    ]
    for tag, ordv in (("1", 1), ("inf", float("inf")), ("-inf", float("-inf")),
                      ("0", 0), ("3", 3)):
        cases.append((LINALG_PREFIX + f"name2::vector_norm(ord={tag})",
                      lambda L, o=ordv: L.linalg.vector_norm(L.tensor(vec3), ord=o)))
    for tag, ordv in (("fro", "fro"), ("nuc", "nuc"), ("2", 2), ("-2", -2),
                      ("1", 1), ("-1", -1), ("inf", float("inf"))):
        cases.append((LINALG_PREFIX + f"name2::matrix_norm(ord={tag})",
                      lambda L, o=ordv: L.linalg.matrix_norm(L.tensor(mat), ord=o)))
    cases.append((LINALG_PREFIX + "name2::matrix_norm(기본)",
                  lambda L: L.linalg.matrix_norm(L.tensor(mat))))
    cases.append((LINALG_PREFIX + "name2::matrix_norm(배치)",
                  lambda L: L.linalg.matrix_norm(L.tensor(_LA_BATCH))))
    for tag, pv in (("기본", None), ("fro", "fro"), ("nuc", "nuc"), ("2", 2),
                    ("-2", -2), ("1", 1), ("inf", float("inf"))):
        cases.append((LINALG_PREFIX + f"name2::cond(p={tag})",
                      lambda L, p=pv: L.linalg.cond(L.tensor(mat), p)))

    cases += [
        # ── 조합이 한 줄이 아닌 것들 ────────────────────────────────────
        (LINALG_PREFIX + "name2::multi_dot",
         lambda L: L.linalg.multi_dot([L.tensor(mat), L.tensor(mat), L.tensor(mat)])),
        (LINALG_PREFIX + "name2::multi_dot(둘)",
         lambda L: L.linalg.multi_dot([L.tensor(mat), L.tensor(mat)])),
        (LINALG_PREFIX + "name2::vander",
         lambda L: L.linalg.vander(L.tensor(np.array([1., 2., 3.], dtype=np.float32)))),
        (LINALG_PREFIX + "name2::vander(N)",
         lambda L: L.linalg.vander(L.tensor(np.array([2., 3.], dtype=np.float32)), N=4)),
        (LINALG_PREFIX + "name2::solve_triangular(위)",
         lambda L: L.linalg.solve_triangular(
             L.tensor(upper), L.tensor(np.array([[1.], [3.]], dtype=np.float32)),
             upper=True)),
        (LINALG_PREFIX + "name2::solve_triangular(아래)",
         lambda L: L.linalg.solve_triangular(
             L.tensor(np.array([[2., 0.], [1., 3.]], dtype=np.float32)),
             L.tensor(np.array([[1.], [2.]], dtype=np.float32)), upper=False)),
        # **대각을 안 본다.** 안 지키면 값이 조용히 달라지는 갈래다.
        (LINALG_PREFIX + "name2::solve_triangular(단위대각)",
         lambda L: L.linalg.solve_triangular(
             L.tensor(upper), L.tensor(np.array([[1.], [3.]], dtype=np.float32)),
             upper=True, unitriangular=True)),
        (LINALG_PREFIX + "name2::tensorsolve",
         lambda L: L.linalg.tensorsolve(
             L.tensor(np.eye(4, dtype=np.float32).reshape(2, 2, 2, 2)),
             L.tensor(np.array([[1., 2.], [3., 4.]], dtype=np.float32)))),
        (LINALG_PREFIX + "name2::tensorinv",
         lambda L: L.linalg.tensorinv(
             L.tensor(np.eye(4, dtype=np.float32).reshape(2, 2, 2, 2)), ind=2)),

        # ── 닫힌 식이 없는 것 하나 ──────────────────────────────────────
        # **스케일링과 제곱이 필요하다.** 테일러만으로는 큰 행렬에서 안 모인다 —
        # `A*5` 의 답이 4.8e+10 이라 항이 커지는 쪽이 먼저 넘친다.
        (LINALG_PREFIX + "name2::matrix_exp(멱영)",
         lambda L: L.linalg.matrix_exp(
             L.tensor(np.array([[0., 1.], [0., 0.]], dtype=np.float32)))),
        (LINALG_PREFIX + "name2::matrix_exp",
         lambda L: L.linalg.matrix_exp(L.tensor(mat))),
        (LINALG_PREFIX + "name2::matrix_exp(큰 값)",
         lambda L: L.linalg.matrix_exp(L.tensor(mat * 5))),
        (LINALG_PREFIX + "name2::matrix_exp(3x3)",
         lambda L: L.linalg.matrix_exp(L.tensor(sym3))),
        (LINALG_PREFIX + "name2::matrix_exp(배치)",
         lambda L: L.linalg.matrix_exp(L.tensor(_LA_BATCH))),
        (LINALG_PREFIX + "name2::torch.matrix_exp",
         lambda L: L.matrix_exp(L.tensor(mat))),
    ]
    return cases


def linalg_grad_cases(inp=None):
    """**분해의 기울기.** torch 는 이것들을 전부 미분하는데 우리는 값만 냈다.

    오래 안 넣은 이유가 있었다 — 유도가 까다롭고 틀리면 조용히 틀린다. 값은 맞고
    학습만 미묘하게 갈리는 종류다. 그래서 골든이 있는 것이고, 여기서는 골든이
    바로 그 일을 한다: **진짜 torch 가 낸 수와 자리마다 맞춘다.**

    ## 안전한 둘과 미묘한 셋

    특잇값과 고윳값은 각각 `U diag(ḡ) Vᵀ` 와 `V diag(ḡ) Vᵀ` 로 끝난다 — 겹침 문제가
    없고 유도가 한 줄이다.

    나머지 셋은 다르다. **고유벡터**는 `1/(λᵢ-λⱼ)` 가 들어가서 고윳값이 겹치면
    터진다(torch 도 같이 터진다 — 흉내가 아니라 같은 한계다). **QR** 은 아래삼각을
    거울로 접는 자리의 규약이 갈리기 쉽고, **유사역행렬**은 항이 셋이라 하나를
    빠뜨려도 정사각에서는 맞고 직사각에서만 틀린다. 그래서 셋 다 **직사각으로도**
    묻는다.

    ## 행렬 지수는 자기 자신으로 미분한다

    `e^A` 의 프레셰 도함수는 블록 행렬 하나로 나온다 — `expm([[Aᵀ, Ḡ],[0, Aᵀ]])` 의
    오른쪽 위가 답이다. 근사가 아니라 항등식이라, 순방향에 쓴 급수를 그대로 다시
    쓰면 기울기가 따라온다.
    """
    mat, rect, sym3 = _LA_MAT, _LA_RECT, _LA_SYM3
    sym = np.array([[4., 1.], [1., 3.]], dtype=np.float32)

    grads = (
        # 안전한 둘.
        ("svdvals", lambda L, x: L.linalg.svdvals(x), mat),
        ("svd/S", lambda L, x: L.linalg.svd(x)[1], mat),
        ("svd/S(직사각)",
         lambda L, x: L.linalg.svd(x, full_matrices=False)[1], rect),
        ("eigvalsh", lambda L, x: L.linalg.eigvalsh(x), sym),
        ("eigh/값", lambda L, x: L.linalg.eigh(x)[0], sym),
        ("eigh/값(3x3)", lambda L, x: L.linalg.eigh(x)[0], sym3),
        # 미묘한 셋.
        #
        # **고유벡터는 제곱해서 묻는다.** 열 부호를 뒤집어도 같은 고유분해라 어느
        # 쪽을 고를지는 구현이 정하고, 야코비 회전과 LAPACK 이 실제로 다르게 고른다.
        # 값 케이스는 절댓값으로 물어서 그 차이를 덮고 있었는데, 기울기는 부호에
        # 민감해서 그대로 드러났다(2×2 에서 정확히 부호만 뒤집혀 나왔다).
        #
        # `V∘V` 는 부호를 뒤집어도 그대로다. 그래서 이 손실의 기울기는 **부호 규약과
        # 무관하게 정해진다** — 양쪽이 같은 답을 낼 수 있는 질문으로 바꾼 것이지,
        # 어려워서 피한 것이 아니다. dropout 을 성질로 물은 것과 같은 자리다.
        ("eigh/벡터²", lambda L, x: L.linalg.eigh(x)[1] ** 2, sym),
        ("eigh/벡터²(3x3)", lambda L, x: L.linalg.eigh(x)[1] ** 2, sym3),
        ("qr/R", lambda L, x: L.linalg.qr(x)[1], mat),
        ("qr/Q", lambda L, x: L.linalg.qr(x)[0], mat),
        ("qr/R(직사각)", lambda L, x: L.linalg.qr(x)[1], rect),
        ("qr/Q(직사각)", lambda L, x: L.linalg.qr(x)[0], rect),
        ("pinv", lambda L, x: L.linalg.pinv(x), mat),
        # **직사각이 진짜 시험이다.** 정사각에서는 빠뜨린 항이 0 이 되어 안 드러난다.
        ("pinv(직사각)", lambda L, x: L.linalg.pinv(x), rect),
        ("pinv(3x3)", lambda L, x: L.linalg.pinv(x), sym3),
        # 자기 자신으로 미분하는 것.
        ("matrix_exp", lambda L, x: L.linalg.matrix_exp(x), mat),
        ("matrix_exp(3x3)", lambda L, x: L.linalg.matrix_exp(x), sym3),
        ("matrix_exp(작은 값)", lambda L, x: L.linalg.matrix_exp(x), mat * 0.1),
    )
    cases = []
    for name, fn, arr in grads:
        def run(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            if out.shape:
                out = out * L.arange(out.numel()).reshape(out.shape).float()
            out.sum().backward()
            return _grad_of(x, n)
        cases.append((LINALG_PREFIX + f"grad2::{name}", run))

    # 값과 기울기를 **한 번에** 묻는 자리 하나. 분해를 쓰고 그 위에 손실을 얹는
    # 모양이 실제 코드가 하는 일이고, 조각으로 물으면 이어 붙는지를 못 본다.
    def chained(L):
        x = L.tensor(mat, requires_grad=True)
        s = L.linalg.svdvals(x)
        loss = (s * s).sum() + L.linalg.matrix_norm(x, ord="nuc")
        loss.backward()
        return _grad_of(x, "svdvals→노름")

    cases.append((LINALG_PREFIX + "grad2::이어 붙이기", chained))
    return cases


INPLACE_PREFIX = "inplace::"

_INPLACE_UNARY = ("abs_", "sqrt_", "exp_", "log_", "sin_", "cos_", "tan_", "tanh_",
                  "sigmoid_", "relu_", "erf_", "floor_", "ceil_", "round_", "sign_",
                  "reciprocal_", "square_", "trunc_", "frac_", "neg_", "rsqrt_",
                  "log2_", "log10_", "expm1_", "log1p_", "sinh_", "cosh_")


def inplace_cases(inp=None):
    """제자리 연산. **두 구현이 일부러 갈리는 자리다.**

    갈리는 것은 하나뿐이다 — 뷰를 통한 전파. torch 와 코어는 저장소를 공유해서
    `b = a.view(2,2); b.add_(10)` 이 `a` 까지 바꾸고, 브라우저 쪽은 GPU 버퍼를 뷰로
    나눠 갖지 않아 그럴 수 없어 거절한다. (TF.js 판일 때도 결론이 같았는데 이유는
    달랐다 — 그쪽은 텐서가 불변이었다.) 나머지(자기 자신 고치기)는 양쪽이 같다.

    그 갈림을 여기서 못 박는다. 브라우저 쪽이 어느 날 조용히 값을 돌려주기 시작하면
    (그 값이 맞든 틀리든) 빌드가 깨진다.
    """
    plain = np.array([1., 4., 9., 2.], dtype=np.float32)
    small = np.array([0.5, 0.8, 0.3, 0.9], dtype=np.float32)   # 정의역이 좁은 것들용

    def run(name, call, arr):
        def go(L, f=call, a=arr):
            x = L.tensor(a.copy())
            f(x)
            return x               # **되돌려받은 것이 아니라 원본을 본다**
        return (INPLACE_PREFIX + name, go)

    cases = [
        run("add_", lambda x: x.add_(1), plain),
        run("add_(alpha)", lambda x: x.add_(1, alpha=2), plain),
        run("sub_", lambda x: x.sub_(1), plain),
        run("mul_", lambda x: x.mul_(2), plain),
        run("div_", lambda x: x.div_(2), plain),
        run("pow_", lambda x: x.pow_(2), plain),
        # `neg_` 는 인자가 없어서 아래 `_INPLACE_UNARY` 루프가 만든다. 여기 한 번 더
        # 있었고, 이름이 같아 루프 것이 이것을 덮고 있었다.
        run("zero_", lambda x: x.zero_(), plain),
        run("fill_", lambda x: x.fill_(7), plain),
        run("clamp_", lambda x: x.clamp_(2, 5), plain),
        run("clip_", lambda x: x.clip_(2, 5), plain),
        # **이어 부르기가 진짜 시험이다.** 되돌려준 것이 자기 자신이어야 이어진다.
        run("이어 부르기", lambda x: x.mul_(2).add_(1).clamp_(0, 10), plain),
    ]
    for name in _INPLACE_UNARY:
        arr = small if name in ("asin_", "acos_", "atan_", "log_", "log2_", "log10_",
                                "sqrt_", "rsqrt_", "log1p_") else plain
        cases.append(run(name, lambda x, n=name: getattr(x, n)(), arr))

    # 뷰 전파 — **브라우저 쪽만 거절한다.**
    def view_propagates(L):
        a = L.arange(4).float()
        a.view(2, 2).add_(10)
        return a

    cases.append((INPLACE_PREFIX + "뷰 전파=브라우저는거절",
                  _as_expected(view_propagates)))

    # 잎에 기울기가 켜져 있으면 **양쪽 다** 거절한다.
    def leaf_refuses(L):
        x = L.tensor(plain, requires_grad=True)
        try:
            x.add_(1)
        except Exception:                                       # noqa: BLE001
            return "기대대로 거절"
        return "뜻밖의 성공"

    cases.append((INPLACE_PREFIX + "잎 제자리 수정=거절", leaf_refuses))

    # `no_grad` 안에서는 잎도 고칠 수 있다 — 옵티마이저가 실제로 그렇게 한다.
    def under_no_grad(L):
        x = L.tensor(plain, requires_grad=True)
        with L.no_grad():
            x.add_(1)
        return x

    cases.append((INPLACE_PREFIX + "no_grad 안에서는 된다", under_no_grad))
    return cases


SHAPE_PREFIX = "shape::"


def shape_cases(inp=None):
    """모양 바꾸기의 나머지.

    **메서드로 부른다.** `expand`·`repeat`·`ravel`·`select`·`unfold`·`expand_as` 는
    torch 에 모듈 함수가 없고 메서드로만 있다 — 부르는 법이 하나뿐이라 그쪽으로 묻는다.
    """
    mat = np.arange(6, dtype=np.float32).reshape(2, 3)
    square = np.arange(9, dtype=np.float32).reshape(3, 3)
    line = np.arange(5, dtype=np.float32)
    col = mat[:, :1].copy()
    flat6 = np.arange(6, dtype=np.float32)
    pair = np.array([1., 2.], dtype=np.float32)
    # **랭크 3.** 축을 바꾸는 것을 2차원으로만 물으면 `(0,1)` 밖의 자리를 못 본다 —
    # 2차원에서는 어느 두 축을 골라도 답이 하나뿐이라, 축 인자를 통째로 버리는
    # 구현도 통과한다. 축 길이를 전부 다르게 두어 모양에서 먼저 걸리게 한다.
    cube = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    calls = (
        ("expand", lambda t: t.expand(2, 3), col),
        ("expand(-1)", lambda t: t.expand(-1, 3), col),
        ("expand(앞에 축 추가)", lambda t: t.expand(2, 2, 3), mat),
        ("repeat", lambda t: t.repeat(2, 1), mat),
        ("repeat(둘 다)", lambda t: t.repeat(2, 3), mat),
        ("ravel", lambda t: t.ravel(), mat),
        ("swapaxes", lambda t: t.swapaxes(0, 1), mat),
        ("swapdims", lambda t: t.swapdims(0, 1), mat),
        ("transpose(랭크3)", lambda t: t.transpose(1, 2), cube),
        ("transpose(랭크3, 0과2)", lambda t: t.transpose(0, 2), cube),
        ("transpose(랭크3, 음수축)", lambda t: t.transpose(-1, -3), cube),
        ("swapdims(랭크3)", lambda t: t.swapdims(0, 1), cube),
        ("select", lambda t: t.select(0, 1), mat),
        ("select(dim1)", lambda t: t.select(1, 2), mat),
        ("diagonal", lambda t: t.diagonal(), square),
        ("diagonal(위로 1)", lambda t: t.diagonal(offset=1), square),
        ("diagonal(아래로 1)", lambda t: t.diagonal(offset=-1), square),
        ("diagflat", lambda t: t.diagflat(), pair),
        ("rot90", lambda t: t.rot90(1, (0, 1)), mat),
        ("rot90(두 번)", lambda t: t.rot90(2, (0, 1)), mat),
        ("unfold", lambda t: t.unfold(0, 3, 1), line),
        ("unfold(걸음2)", lambda t: t.unfold(0, 2, 2), line),
        ("unflatten", lambda t: t.unflatten(0, (2, 3)), flat6),
        ("fliplr", lambda t: t.fliplr(), mat),
        ("flipud", lambda t: t.flipud(), mat),
    )
    cases = [(SHAPE_PREFIX + name, lambda L, f=fn, a=arr: f(L.tensor(a)))
             for name, fn, arr in calls]

    # 나누는 것들 — 조각마다 이름을 붙인다. 하나만 보면 나머지가 안 걸린다.
    for name, parts in (("hsplit", 3), ("vsplit", 2)):
        for i in range(parts):
            cases.append((SHAPE_PREFIX + f"{name}[{i}]",
                          lambda L, n=name, p=parts, k=i: getattr(L.tensor(mat), n)(p)[k]))

    cases.append((SHAPE_PREFIX + "atleast_2d",
                  lambda L: L.atleast_2d(L.tensor(np.float32(1.)))))

    # 기울기. **`expand` 와 `unfold` 가 여기서 갈린다** — expand 는 늘린 축을 도로
    # 합치고, unfold 는 겹친 창만큼 쌓는다(실측: 길이 5 를 3·1 로 펴면 [1,2,3,2,1]).
    grads = (
        ("expand", lambda t: t.expand(2, 3), col),
        ("repeat", lambda t: t.repeat(2, 1), mat),
        ("diagonal", lambda t: t.diagonal(), square),
        ("diagonal(위로 1)", lambda t: t.diagonal(offset=1), square),
        ("diagflat", lambda t: t.diagflat(), pair),
        ("rot90", lambda t: t.rot90(1, (0, 1)), mat),
        ("unfold(겹침)", lambda t: t.unfold(0, 3, 1), line),
        ("select", lambda t: t.select(0, 1), mat),
        ("swapaxes", lambda t: t.swapaxes(0, 1), mat),
    )
    for name, fn, arr in grads:
        def run(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(x)
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(x, n)
        cases.append((SHAPE_PREFIX + f"grad::{name}", run))
    return cases


REDUCE_PREFIX = "reduce::"


def reduce_cases(inp=None):
    """축약의 나머지 — `amax`·`nansum`·`logsumexp`·`cummax`·`kthvalue` 등.

    **동점을 일부러 넣는다.** `amax` 는 동점일 때 기울기를 고르게 나누고(실측:
    [1,3,3,2] → [0,.5,.5,0]) `cummax` 는 나중 자리를 준다. 동점 없는 입력으로 재면
    그 규칙을 하나도 안 보게 되고, 값은 맞는데 학습이 미묘하게 갈리는 자리가 남는다.
    """
    tie = np.array([1., 3., 3., 2.], dtype=np.float32)          # 동점이 있다
    mat = np.array([[1., 5., 3.], [4., 2., 6.]], dtype=np.float32)
    withnan = np.array([1., np.nan, 3., 5.], dtype=np.float32)
    zeros_in = np.array([0., 1., 0., 2.], dtype=np.float32)
    weights = np.arange(1, 5, dtype=np.float32)

    def values_of(got):
        return got if hasattr(got, "numpy") else got.values

    cases = []

    def add(name, fn, grad_of=None):
        cases.append((REDUCE_PREFIX + name, lambda L, f=fn: values_of(f(L))))
        if grad_of is not None:
            def run(L, f=fn, arr=grad_of):
                x = L.tensor(arr, requires_grad=True)
                out = values_of(f(L, x))
                w = L.arange(out.numel()).reshape(out.shape).float() if out.shape else None
                (out * w if w is not None else out).sum().backward()
                return _grad_of(x, name)
            cases.append((REDUCE_PREFIX + f"grad::{name}", run))

    add("amax", lambda L, x=None: L.amax(x if x is not None else L.tensor(tie)), tie)
    add("amin", lambda L, x=None: L.amin(x if x is not None else L.tensor(tie)), tie)
    add("amax(dim)", lambda L, x=None: L.amax(x if x is not None else L.tensor(mat), dim=1), mat)
    add("amin(keepdim)",
        lambda L, x=None: L.amin(x if x is not None else L.tensor(mat), dim=1, keepdim=True), mat)
    add("nansum", lambda L, x=None: L.nansum(x if x is not None else L.tensor(withnan)), withnan)
    add("nanmean", lambda L, x=None: L.nanmean(x if x is not None else L.tensor(withnan)), withnan)
    add("logsumexp", lambda L, x=None: L.logsumexp(x if x is not None else L.tensor(tie), 0), tie)
    add("logsumexp(dim1)",
        lambda L, x=None: L.logsumexp(x if x is not None else L.tensor(mat), 1), mat)
    add("cummax", lambda L, x=None: L.cummax(x if x is not None else L.tensor(tie), 0), tie)
    add("cummin", lambda L, x=None: L.cummin(x if x is not None else L.tensor(tie), 0), tie)
    add("kthvalue", lambda L, x=None: L.kthvalue(x if x is not None else L.tensor(tie), 2), tie)
    add("msort", lambda L, x=None: L.msort(x if x is not None else L.tensor(mat)))
    add("diff", lambda L, x=None: L.diff(x if x is not None else L.tensor(tie)), tie)
    add("diff(n=2)", lambda L, x=None: L.diff(x if x is not None else L.tensor(tie), n=2), tie)
    add("dist", lambda L, x=None: L.dist(x if x is not None else L.tensor(tie),
                                         L.tensor(np.zeros(4, dtype=np.float32))), tie)

    # **축을 받는 것은 값으로 물어야 한다.** 기울기로만 물으면 축을 통째로 무시해도
    # 통과한다 — `sum(dim=1).sum()` 과 `sum().sum()` 의 기울기가 둘 다 전부 1 이라
    # 답이 같기 때문이다. `grad::sum(dim)` 이라는 이름의 케이스가 이미 있었고, 그
    # 이름 때문에 아무도 다시 안 봤다.
    #
    # 그 사이 `borch_webgpu` 는 `sum(dim=1)` 에 **축을 무시한 스칼라**를 내고 있었다.
    # borch.ts 가 전체 합과 축 합을 다른 이름으로 두는데 JS 가 남는 인자를 조용히
    # 버려서다. 랭크 6 케이스 하나가 모양으로 걸릴 때까지 792 건이 전부 초록이었다.
    #
    # **메서드로 묻는다.** 모듈 함수 `L.sum` 은 코어에도 자매에도 없다 — torch 에는
    # 있으므로 그것대로 구멍이지만 이 케이스가 잡으려는 것과 다른 이야기다.
    add("sum(dim)", lambda L, x=None: (x if x is not None else L.tensor(mat)).sum(dim=1), mat)
    add("sum(dim0)", lambda L, x=None: (x if x is not None else L.tensor(mat)).sum(dim=0), mat)
    add("sum(dim,keepdim)",
        lambda L, x=None: (x if x is not None else L.tensor(mat)).sum(dim=1, keepdim=True), mat)
    add("norm(dim)", lambda L, x=None: (x if x is not None else L.tensor(mat)).norm(dim=1), mat)
    add("norm(p=1,dim)",
        lambda L, x=None: (x if x is not None else L.tensor(mat)).norm(p=1, dim=0), mat)

    # 번호를 주는 것들 — 값만 보면 번호가 틀려도 통과한다.
    for name, fn in (("cummax", lambda L: L.cummax(L.tensor(tie), 0)),
                     ("cummin", lambda L: L.cummin(L.tensor(tie), 0)),
                     ("kthvalue", lambda L: L.kthvalue(L.tensor(tie), 2))):
        cases.append((REDUCE_PREFIX + f"{name} 번호", lambda L, f=fn: f(L).indices))

    # 기울기가 없는 것들. 값만 굳힌다.
    cases += [
        (REDUCE_PREFIX + "quantile", lambda L: L.quantile(L.tensor(tie), 0.5)),
        (REDUCE_PREFIX + "quantile(여럿)",
         lambda L: L.quantile(L.tensor(tie), L.tensor(np.array([0.25, 0.75], dtype=np.float32)))),
        (REDUCE_PREFIX + "nanquantile",
         lambda L: L.nanquantile(L.tensor(withnan), 0.5)),
        (REDUCE_PREFIX + "nonzero", lambda L: L.nonzero(L.tensor(zeros_in))),
        (REDUCE_PREFIX + "argwhere", lambda L: L.argwhere(L.tensor(zeros_in))),
        # **색인으로 묻는다.** torch 의 `aminmax` 는 `.min`·`.max` 로 부르고 우리 것은
        # `.values`·`.indices` 라 이름이 안 맞는다 — 자리로 물으면 양쪽 다 통한다.
        (REDUCE_PREFIX + "aminmax/최소", lambda L: L.aminmax(L.tensor(tie))[0]),
        (REDUCE_PREFIX + "aminmax/최대", lambda L: L.aminmax(L.tensor(tie))[1]),
    ]
    return cases


MATH_PREFIX = "math::"

# 새로 붙인 수학 함수들. **입력 범위가 함수마다 다르다** — `acos` 는 [-1,1] 밖에서
# NaN 이고, NaN 은 자기 자신과도 달라서 대조가 통과할 수가 없다. 그래서 정의역 안에서만
# 묻는다. 밖에서 무엇을 하는지는 별개 질문이고, 여기서 섞으면 둘 다 못 본다.
_MATH_DOMAIN = {
    "acos": "unit", "asin": "unit", "atanh": "unit", "logit": "unit",
    "arccos": "unit", "arcsin": "unit", "arctanh": "unit",
    "acosh": "big", "arccosh": "big",
    "log1p": "pos",
}
_MATH_UNARY = (
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "expm1", "log1p", "exp2",
    "deg2rad", "rad2deg", "trunc", "frac", "positive", "erfc", "sinc", "logit",
    "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh", "fix", "absolute",
)
_MATH_BINARY = ("atan2", "hypot", "copysign", "logaddexp", "logaddexp2")


def math_cases(inp=None):
    """삼각·지수·로그의 나머지. 값과 기울기를 **둘 다** 묻는다.

    기울기를 같이 묻는 이유가 있다. 값만 맞고 그래프가 끊긴 함수는 값 검사를 통과하고,
    이 저장소는 그런 것을 이미 열넷 찾았다.
    """
    plain = np.array([0.5, 2.0, -1.5, 3.0], dtype=np.float32)
    unit = np.array([0.2, 0.6, -0.9, 0.45], dtype=np.float32)      # (-1, 1) 안
    big = np.array([1.5, 2.5, 3.0, 1.2], dtype=np.float32)          # > 1
    pos = np.array([0.5, 2.0, 1.5, 3.0], dtype=np.float32)
    other = np.array([1.0, 2.0, -3.0, 0.5], dtype=np.float32)
    logit_in = np.array([0.2, 0.6, 0.35, 0.45], dtype=np.float32)   # (0, 1) 안
    weights = np.arange(1, 5, dtype=np.float32)                     # 자리마다 다른 가중치

    def pick(name):
        kind = _MATH_DOMAIN.get(name)
        if name in ("logit",):
            return logit_in
        return {"unit": unit, "big": big, "pos": pos}.get(kind, plain)

    cases = []
    for name in _MATH_UNARY:
        cases.append((MATH_PREFIX + name,
                      lambda L, n=name: getattr(L, n)(L.tensor(pick(n)))))

        def grad(L, n=name):
            x = L.tensor(pick(n), requires_grad=True)
            (getattr(L, n)(x) * L.tensor(weights)).sum().backward()
            return _grad_of(x, n)

        cases.append((MATH_PREFIX + f"grad::{name}", grad))

    for name in _MATH_BINARY:
        cases.append((MATH_PREFIX + name,
                      lambda L, n=name: getattr(L, n)(L.tensor(plain), L.tensor(other))))
        for who in (0, 1):
            def bgrad(L, n=name, w=who):
                leaves = [L.tensor(plain, requires_grad=True),
                          L.tensor(other, requires_grad=True)]
                (getattr(L, n)(*leaves) * L.tensor(weights)).sum().backward()
                return _grad_of(leaves[w], f"{n}/{'ab'[w]}")

            cases.append((MATH_PREFIX + f"grad::{name}/{'ab'[who]}", bgrad))

    # `x·log(y)` — **x 가 0 인 자리가 있어야 이 함수를 시험하는 것이다.**
    zeros_in = np.array([0.0, 2.0, 0.0, 3.0], dtype=np.float32)
    ypos = np.array([1.0, 2.0, 0.5, 4.0], dtype=np.float32)
    cases.append((MATH_PREFIX + "xlogy(x에 0 포함)",
                  lambda L: L.xlogy(L.tensor(zeros_in), L.tensor(ypos))))

    # 계단 함수는 **0 을 흘린다.** 전에는 그래프를 끊어 `backward()` 가 거절했는데
    # torch 는 0 을 준다 — 없는 것과 0 인 것은 다르다.
    for name in ("sign", "floor", "ceil", "round", "trunc", "fix"):
        def zgrad(L, n=name):
            x = L.tensor(plain, requires_grad=True)
            (getattr(L, n)(x) * L.tensor(weights)).sum().backward()
            return _grad_of(x, n)

        cases.append((MATH_PREFIX + f"grad::{name}(0이어야)", zgrad))

    # 값만 있고 기울기가 없는 것들 — 참·거짓이거나 계단이다.
    cases.append((MATH_PREFIX + "signbit", lambda L: L.signbit(L.tensor(plain))))
    cases.append((MATH_PREFIX + "heaviside",
                  lambda L: L.heaviside(L.tensor(np.array([-1., 0., 1., 0.], dtype=np.float32)),
                                        L.tensor(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)))))
    cases.append((MATH_PREFIX + "ldexp",
                  lambda L: L.ldexp(L.tensor(plain),
                                    L.tensor(np.array([1., 2., 0., -1.], dtype=np.float32)))))
    cases.append((MATH_PREFIX + "sgn", lambda L: L.sgn(L.tensor(plain))))

    def sgn_grad(L):
        x = L.tensor(plain, requires_grad=True)
        L.sgn(x).sum().backward()
        # **`clone()` 이 필요하다.** torch 의 sgn 기울기는 ZeroTensor(게으른 0 텐서)라
        # `.numpy()` 가 거절한다 — 곱셈으로는 안 풀리고(`* 1.0` 도 ZeroTensor 다),
        # 복제해야 진짜 버퍼가 생긴다. 값은 0 이다.
        #
        # 이 자리를 처음에는 "torch 가 sgn 역전파를 거절한다"고 읽고 거절 케이스로
        # 굳혔다. 틀렸다 — 예외는 `backward()` 가 아니라 결과를 찍던 쪽에서 났다.
        return x.grad.detach().clone()

    cases.append((MATH_PREFIX + "grad::sgn(0이어야)", sgn_grad))
    return cases


METHOD_PREFIX = "method::"

# `x.f(...)` 로 부를 수 있어야 하는 것들과, 그때 줄 인자.
#
# **표면이 늘면 조용히 틀릴 자리가 는다** — 이 저장소가 그것을 네 번 겪었다. 그래서
# 이름을 붙이는 것과 그 이름에 케이스를 세우는 것을 한 번에 한다. 케이스 없는 표면은
# 안 넣는다는 것이 이번에 기능을 늘리기로 하면서 붙인 유일한 조건이다.
#
# `x.f(...)` 와 `torch.f(x, ...)` 가 같은지도 torch 에게 물어보고 담았다. 하나가
# 달랐다 — `where` 는 인자 순서가 뒤집혀서, 그냥 붙였으면 `x` 가 조건 자리로 갔다.
_METHOD_ARGS = {
    "ceil": (), "cos": (), "cosh": (), "erf": (), "floor": (), "isfinite": (),
    "isinf": (), "isnan": (), "neg": (), "reciprocal": (), "relu": (), "round": (),
    "sigmoid": (), "sign": (), "sin": (), "sinh": (), "square": (), "tan": (),
    "tanh": (), "prod": (), "norm": (), "argsort": (), "unique": (),
    "clamp": (0.0, 1.0), "pow": (2,), "roll": (1,), "cumsum": (0,), "cumprod": (0,),
    "softmax": (0,), "narrow": (0, 0, 2), "flip": ((0,),),
    "tile": ((2,),), "topk": (2,), "sort": (), "median": (),
}
# 양수만 받는 것 — 음수를 주면 NaN 이 나오고 NaN 은 자기 자신과도 다르다.
_METHOD_ARGS_POS = {"log2": (), "log10": (), "rsqrt": ()}
# 짝이 필요한 것. 상대는 같은 모양의 다른 벡터다.
_METHOD_ARGS_PAIR = {"eq": (), "ne": (), "lt": (), "le": (), "gt": (), "ge": (),
                     "maximum": (), "minimum": (), "dot": (), "outer": ()}
# 행렬이어야 하는 것.
_METHOD_ARGS_MAT = {"diag": (), "trace": (), "tril": (), "triu": ()}


def method_cases(inp=None):
    """모듈 함수를 **메서드로도** 부를 수 있는가. 값까지 대조한다.

    이름만 있고 값이 다르면 그것도 거짓이다 — 이 표는 `hasattr` 을 묻지 않는다.
    """
    pos = np.array([0.5, 2.0, 1.5, 3.0], dtype=np.float32)
    vec = np.array([0.5, 2.0, -1.5, 3.0], dtype=np.float32)
    other = np.array([1.0, 2.0, -3.0, 0.5], dtype=np.float32)
    mat = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    mask = np.array([True, False, True, False])

    def values_of(got):
        """(값, 번호) 를 주는 것들은 값 쪽만 본다 — 번호는 동점에서 갈릴 수 있다.

        `getattr(got, "values", got)` 로 쓰면 안 된다. **진짜 torch 텐서에는 `.values`
        가 메서드로 있어서**(희소 텐서용) 텐서 대신 그 메서드가 나온다 — 굳히기가
        `'builtin_function_or_method' object has no attribute 'detach'` 로 터지며
        알려줬다. 텐서인지를 먼저 묻는다.
        """
        return got if hasattr(got, "numpy") else got.values

    def call(name, args, arr, extra=()):
        def run(L, n=name, a=args, base=arr, ex=extra):
            return values_of(
                getattr(L.tensor(base), n)(*[L.tensor(e) for e in ex], *a))
        return run

    cases = []
    for name, args in _METHOD_ARGS.items():
        cases.append((METHOD_PREFIX + name, call(name, args, vec)))
    for name, args in _METHOD_ARGS_POS.items():
        cases.append((METHOD_PREFIX + name, call(name, args, pos)))
    for name, args in _METHOD_ARGS_PAIR.items():
        cases.append((METHOD_PREFIX + name, call(name, args, vec, extra=(other,))))
    for name, args in _METHOD_ARGS_MAT.items():
        cases.append((METHOD_PREFIX + name, call(name, args, mat)))

    # 여럿을 돌려주는 것들은 조각마다 이름을 붙인다 — 하나만 보면 나머지가 안 걸린다.
    for name, args in (("chunk", (2,)), ("split", (2,)), ("unbind", ())):
        for piece in (0, 1):
            cases.append((
                METHOD_PREFIX + f"{name}[{piece}]",
                lambda L, n=name, a=args, p=piece: getattr(L.tensor(vec), n)(*a)[p]))

    # `where` — **인자 순서가 함수와 뒤집힌 유일한 자리다.**
    cases.append((METHOD_PREFIX + "where",
                  lambda L: L.tensor(vec).where(L.tensor(mask), L.tensor(other))))
    # 참·거짓을 돌려주는 것들. 값이 아니라 **판정**을 굳힌다.
    cases.append((METHOD_PREFIX + "equal",
                  lambda L: str(bool(L.tensor(vec).equal(L.tensor(vec))))))
    cases.append((METHOD_PREFIX + "equal(다른 것)",
                  lambda L: str(bool(L.tensor(vec).equal(L.tensor(other))))))
    cases.append((METHOD_PREFIX + "allclose",
                  lambda L: str(bool(L.tensor(vec).allclose(L.tensor(vec))))))

    # 행렬곱 계열은 모양이 달라 따로 준다.
    # `movedim` 은 **네 조합을 다 묻는다.** 전에는 `(0, 0)` 하나였는데 그건 항등이라
    # 아무것도 안 물은 것과 같았고, 그 뒤에 숨어 있던 자매의 `movedim(0, -1)` 이
    # 조용히 항등으로 굴고 있었다(`list.insert(-1, …)` 은 맨 뒤가 아니다).
    for src, dst in ((0, -1), (-1, 0), (0, 1), (1, 0)):
        cases.append((METHOD_PREFIX + f"movedim({src},{dst})",
                      lambda L, s=src, d=dst: L.tensor(mat).movedim(s, d)))

    cases.append((METHOD_PREFIX + "mm", lambda L: L.tensor(mat).mm(L.tensor(mat))))
    cases.append((METHOD_PREFIX + "gather",
                  lambda L: L.tensor(mat).gather(1, L.tensor(
                      np.array([[0, 2], [1, 0], [2, 1]], dtype=np.int64)))))
    # 기울기도 본다. 메서드로 불렀다고 그래프가 끊기면 값만 보는 검사는 통과한다.
    def method_grad(L):
        x = L.tensor(vec, requires_grad=True)
        (x.square() * L.arange(4).float()).sum().backward()
        return _grad_of(x, "method::square")

    cases.append((METHOD_PREFIX + "grad::square", method_grad))
    return cases


EDGE_PREFIX = "edge::"


def edge_cases(inp=None):
    """**꺾이는 자리.** 나머지 표가 구조적으로 못 보는 곳을 모아 둔다.

    다른 표의 입력은 거의 다 `default_rng` 가 뽑은 정규분포다. 그것은 좋은 기본값이지만
    한 가지를 못 한다 — **특별한 값이 한 번도 안 나온다.** 정확히 0, 정확히 같은 두 수,
    정확히 경계값, 정확히 .5. 함수가 꺾이는 자리가 전부 거기에 있다.

    `relu` 가 그래서 뚫렸다. 입력이 정확히 0 일 때 torch 는 기울기를 0 으로 주는데
    (`x > 0` 이지 `x >= 0` 이 아니다) borch.ts 는 1 을 흘렸고, 골든 798 건이 전부
    통과했다. relu 케이스의 입력에 0 이 없었기 때문이다. 그 하나를 고치는 것으로는
    부족하다 — 같은 이유로 안 보이는 자리가 이만큼 더 있다.

    **여기서는 답을 추측하지 않는다.** 동점에서 torch 가 기울기를 나눠 주는지 한쪽에만
    주는지, `round(0.5)` 가 0 인지 1 인지는 우리가 정할 것이 아니다. 진짜 torch 가
    무엇을 하든 그것이 답이고, 이 표는 그 답을 묻기만 한다.
    """
    cases = []

    def value(name, fn):
        cases.append((EDGE_PREFIX + name, fn))

    def grad(name, fn, arr, which=0):
        """자리마다 다른 가중치로 접는다 — 균일하게 접으면 꺾인 자리가 묻힌다."""
        def run(L, f=fn, a=arr, n=name, w=which):
            leaves = [L.tensor(x.copy(), requires_grad=True) for x in a]
            out = f(L, *leaves)
            if out.shape:
                out = out * L.arange(out.numel()).reshape(out.shape).float()
            out.sum().backward()
            return _grad_of(leaves[w], n)
        cases.append((EDGE_PREFIX + "grad::" + name, run))

    # 정확히 0 을 품은 입력. 이 표의 거의 모든 케이스가 이것을 쓴다.
    z = np.array([-2., -1., 0., 1., 2., 0.], dtype=np.float32)
    # 정확히 같은 값이 겹친 짝. 동점에서 기울기가 어디로 가는가를 묻는다.
    ta = np.array([1., 2., 3., 2.], dtype=np.float32)
    tb = np.array([1., 5., 3., 0.], dtype=np.float32)      # 자리 0·2 가 동점
    # .5 로 끝나는 값들 — 반올림 규칙(짝수로 붙이기)이 여기서만 드러난다.
    half = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    # 부호가 섞인 나눗셈. `%` 의 부호 규칙이 언어마다 갈리는 자리다.
    neg = np.array([-7., -3., 3., 7.], dtype=np.float32)

    # ── 0 에서 꺾이는 것들 ─────────────────────────────────────────────────
    for name, fn in (
        ("abs", lambda L, x: L.abs(x)),
        ("sign", lambda L, x: L.sign(x)),
        ("relu", lambda L, x: L.nn.functional.relu(x)),
        ("F.leaky_relu", lambda L, x: L.nn.functional.leaky_relu(x, 0.1)),
        ("F.elu", lambda L, x: L.nn.functional.elu(x)),
        ("F.gelu", lambda L, x: L.nn.functional.gelu(x)),
        ("F.silu", lambda L, x: L.nn.functional.silu(x)),
    ):
        value(f"{name}(0포함)", lambda L, f=fn, a=z: f(L, L.tensor(a)))
        grad(f"{name}(0포함)", fn, (z,))

    # ── 경계에 정확히 닿는 clamp ───────────────────────────────────────────
    # 입력에 -1 과 1 이 그대로 있다. 자르는 쪽과 흘리는 쪽의 경계가 `<` 인지 `<=` 인지가
    # 여기서만 갈린다.
    value("clamp(경계에서)", lambda L: L.clamp(L.tensor(z), min=-1., max=1.))
    grad("clamp(경계에서)", lambda L, x: L.clamp(x, min=-1., max=1.), (z,))
    grad("clamp(위만)", lambda L, x: L.clamp(x, max=1.), (z,))
    grad("clamp(아래만)", lambda L, x: L.clamp(x, min=-1.), (z,))

    # ── 동점 ───────────────────────────────────────────────────────────────
    # **torch 는 동점에서 기울기를 나눠 준다.** maximum 의 두 입력이 같으면 각각 절반씩
    # 받는다. 한쪽에 몰아주는 구현은 순방향이 완벽히 같으므로 값 대조로는 절대 안 잡힌다.
    value("maximum(동점)", lambda L: L.maximum(L.tensor(ta), L.tensor(tb)))
    value("minimum(동점)", lambda L: L.minimum(L.tensor(ta), L.tensor(tb)))
    for who in (0, 1):
        grad(f"maximum(동점)/{'ab'[who]}",
             lambda L, a, b: L.maximum(a, b), (ta, tb), which=who)
        grad(f"minimum(동점)/{'ab'[who]}",
             lambda L, a, b: L.minimum(a, b), (ta, tb), which=who)

    # 접는 쪽의 동점 — 최댓값이 두 자리에 있을 때 어디로 흘리는가.
    # (`max`·`min`·`argmax` 는 두 라이브러리에 **메서드로만** 있다. torch 에는 모듈
    #  함수도 있으니 그 자체가 갈림이지만, 여기서 묻는 것은 동점이므로 메서드로 쓴다.)
    dup = np.array([1., 3., 2., 3.], dtype=np.float32)
    value("max(동점).indices", lambda L: L.tensor(dup).max(dim=0).indices)
    value("min(동점).indices", lambda L: L.tensor(-dup).min(dim=0).indices)
    value("argmax(동점)", lambda L: L.tensor(dup).argmax())
    grad("max(동점)", lambda L, x: x.max(dim=0).values.reshape(1), (dup,))

    # 정렬의 동점 — 같은 값끼리의 **순서**가 안정적인가. 답이 갈리면 indices 가 갈린다.
    value("sort(동점).values", lambda L: L.sort(L.tensor(dup)).values)
    value("sort(동점).indices", lambda L: L.sort(L.tensor(dup)).indices)
    value("topk(동점).indices", lambda L: L.topk(L.tensor(dup), 3).indices)

    # 창 안에 같은 값이 둘 있는 풀링. **`maximum` 과 답이 다르다** — torch 의 풀링은
    # 이긴 자리 하나를 골라 거기로만 흘리고 나누지 않는다. 풀링을 `maximum` 위에
    # 얹어 구현하면(세 라이브러리 중 둘이 그랬다) 여기서만 갈린다.
    tied_img = np.array([[[[1., 1., 2., 0.],
                           [1., 0., 2., 2.],
                           [3., 3., 0., 1.],
                           [0., 3., 1., 1.]]]], dtype=np.float32)
    value("max_pool2d(동점)", lambda L: L.nn.functional.max_pool2d(L.tensor(tied_img), 2))

    def pooled_tie(L):
        x = L.tensor(tied_img.copy(), requires_grad=True)
        out = L.nn.functional.max_pool2d(x, 2)
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(x, "max_pool2d(동점)")

    cases.append((EDGE_PREFIX + "grad::max_pool2d(동점)", pooled_tie))

    # ── 반올림 규칙 ────────────────────────────────────────────────────────
    # **torch 는 .5 를 짝수로 붙인다** — round(0.5)=0, round(1.5)=2, round(2.5)=2.
    # 흔한 구현(`floor(x+0.5)`)은 전부 위로 올려서 조용히 갈린다.
    value("round(.5에서)", lambda L: L.round(L.tensor(half)))
    value("floor(정수에서)", lambda L: L.floor(L.tensor(z)))
    value("ceil(정수에서)", lambda L: L.ceil(L.tensor(z)))
    value("trunc(음수)", lambda L: L.trunc(L.tensor(half)))
    value("frac(음수)", lambda L: L.frac(L.tensor(half)))

    # ── 나머지의 부호 ──────────────────────────────────────────────────────
    # **torch 의 `%` 는 나누는 수의 부호를 따른다** — `-7 % 3` 이 2 이지 -1 이 아니다.
    # JS 의 `%` 는 반대로 나뉘는 수의 부호를 따르므로(-1), 그것을 그대로 쓰면 음수에서만
    # 갈린다. 양수 입력으로는 절대 안 드러나고, 두 규칙 다 "나머지"라고 불린다.
    value("%(음수)", lambda L: L.tensor(neg) % 3.)
    value("%(음수로 나누기)", lambda L: L.tensor(neg) % -3.)

    return cases


def golden_cases(inp=None):
    """골든이 다루는 전부 — 값·기울기·학습·dtype·표현."""
    inp = golden_inputs() if inp is None else inp
    return (wide_cases(inp) + grad_cases(inp) + train_cases(inp)
            + dtype_cases(inp) + repr_cases(inp) + error_cases(inp)
            + vision_cases(inp) + method_cases(inp) + math_cases(inp)
            + reduce_cases(inp) + shape_cases(inp) + inplace_cases(inp)
            + linalg_cases(inp) + linalg_struct_cases(inp) + linalg_name_cases(inp)
            + linalg_grad_cases(inp) + ndim_cases(inp) + flow_cases(inp)
            + container_cases(inp) + act_cases(inp) + norm_cases(inp)
            + pad_cases(inp) + loss_cases(inp) + lazy_cases(inp)
            + shuffle_cases(inp) + misc_cases(inp) + cell_cases(inp)
            + method_name_cases(inp) + default_convert_cases(inp)
            + unpool_cases(inp) + functional_name_cases(inp)
            + top_level_cases(inp)
            + opt_cases(inp) + dropout_cases(inp) + sdpa_cases(inp)
            + module_function_cases(inp) + pool_cases(inp)
            + new_function_cases(inp) + index_cases(inp) + numeric_cases(inp)
            + bit_cases(inp) + shape_index_cases(inp) + blend_cases(inp)
            + scalar_cache_cases(inp) + top_linalg_cases(inp) + stat_cases(inp)
            + make_cases(inp) + complex_cases(inp) + fft_cases(inp)
            + webgpu_cases(inp) + edge_cases(inp))


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

    # ── 자리만 옮기는 것은 **형을 지킨다** ──────────────────────────────
    #
    # 이 표가 없어서 결함이 하나 살아 있었다. 자매의 데이터셋이 `int64` 라벨에서 한
    # 표본을 꺼내니 `float32` 가 나왔는데, **값이 맞아서** 골든 어디에도 안 걸렸다 —
    # 모양 연산 뒤의 형을 묻는 케이스가 하나도 없었기 때문이다.
    #
    # 경계는 **값을 만드는가**다. 자리만 옮기는 것(고르기·자르기·이어붙이기·
    # 갈아끼우기)은 원래 형이 그대로 나오고, 셈을 하는 것은 승격 규칙을 따른다.
    # 축약(`sum`·`amax`·`cumsum`)은 torch 에서도 형을 지키지만 **여기 표에 없다** —
    # `bool` 의 합은 `int64` 라 규칙이 하나 더 있고, 그것은 따로 잴 자리다.
    ints = np.arange(12, dtype=np.int64).reshape(3, 4)
    flags = (ints % 2 == 0)
    pick = np.array([0, 2], dtype=np.int64)
    spread = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]], dtype=np.int64)
    moves = (
        ("reshape", lambda L, t: t.reshape(4, 3)),
        ("ravel", lambda L, t: t.ravel()),
        ("squeeze", lambda L, t: t.reshape(1, 12).squeeze(0)),
        ("unsqueeze", lambda L, t: t.unsqueeze(0)),
        ("transpose", lambda L, t: t.transpose(0, 1)),
        ("t", lambda L, t: t.t()),
        ("permute", lambda L, t: t.permute(1, 0)),
        ("flip", lambda L, t: t.flip(0)),
        ("select", lambda L, t: t.select(0, 1)),
        ("narrow", lambda L, t: t.narrow(1, 1, 2)),
        ("diagonal", lambda L, t: t.diagonal()),
        ("chunk[0]", lambda L, t: t.chunk(2, 0)[0]),
        ("unbind[0]", lambda L, t: t.unbind(0)[0]),
        ("tensor_split[0]", lambda L, t: t.tensor_split(3, 1)[0]),
        ("index_select", lambda L, t: t.index_select(0, L.tensor(pick))),
        ("gather", lambda L, t: t.gather(1, L.tensor(spread))),
        ("take", lambda L, t: t.take(L.tensor(pick))),
        ("masked_select", lambda L, t: t.masked_select(t > 5)),
        ("cat", lambda L, t: L.cat([t, t], 0)),
        ("stack", lambda L, t: L.stack([t, t], 0)),
        ("repeat", lambda L, t: t.repeat(2, 1)),
        ("roll", lambda L, t: t.roll(1, 0)),
        ("tril", lambda L, t: t.tril()),
        ("triu", lambda L, t: t.triu()),
        ("pad", lambda L, t: F(L).pad(t, (1, 1))),
        ("as_strided", lambda L, t: L.as_strided(t, (2, 2), (1, 2))),
        ("diag_embed", lambda L, t: L.diag_embed(t)),
        ("slice_scatter",
         lambda L, t: L.slice_scatter(t, L.zeros(3, 2).long(), 1, 0, 2)),
        ("select_scatter",
         lambda L, t: L.select_scatter(t, L.zeros(4).long(), 0, 1)),
        ("scatter", lambda L, t: t.scatter(1, L.tensor(spread), t)),
        ("sort[0]", lambda L, t: t.sort(1)[0]),
    )
    for name, fn in moves:
        # **정수와 참거짓 둘 다 묻는다.** 하나만 물으면 "float32 로 떨어뜨리는" 결함이
        # 나머지 하나에서만 살아남을 수 있다.
        cases.append((f"dtype::자리만::{name}(int64)",
                      lambda L, f=fn: outcome(lambda: f(L, L.tensor(ints)))))
        cases.append((f"dtype::자리만::{name}(bool)",
                      lambda L, f=fn: outcome(lambda: f(L, L.tensor(flags)))))
    # **`topk` 는 정수만 묻는다.** 참거짓에서는 torch 가 거절하는데 우리와 **예외
    # 종류**가 다르고(RuntimeError vs TypeError), 그것은 형 보존이 아니라 거절 문구의
    # 이야기다. 한 표에 두 질문을 섞으면 어느 쪽이 빨간지 못 읽는다.
    cases.append(("dtype::자리만::topk[0](int64)",
                  lambda L: outcome(lambda: L.tensor(ints).topk(2, 1)[0])))
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

    진짜 torch 와 borch 는 둘 다 `.detach().numpy()` 가 통한다. GPU 백엔드도
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
