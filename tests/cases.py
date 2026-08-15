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
]

# `(이름, 만드는 인자, 몇 번 밟을까)`. 학습률의 **자취**를 묻는다.
_SCHEDULERS = [
    ("ConstantLR", {"factor": 0.5, "total_iters": 3}, 8),
    ("LinearLR", {"start_factor": 0.5, "end_factor": 1.0, "total_iters": 4}, 8),
    ("PolynomialLR", {"total_iters": 5, "power": 2.0}, 8),
    ("MultiplicativeLR", {}, 6),
    ("CosineAnnealingWarmRestarts", {"T_0": 3, "T_mult": 2}, 10),
    ("OneCycleLR", {"max_lr": 0.4, "total_steps": 10}, 10),
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
        if name == "MultiplicativeLR":
            sch = L.optim.lr_scheduler.MultiplicativeLR(opt, lambda epoch: 0.9)
        else:
            sch = getattr(L.optim.lr_scheduler, name)(opt, **args)
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
            + pad_cases(inp)
            + opt_cases(inp) + dropout_cases(inp) + sdpa_cases(inp)
            + module_function_cases(inp) + pool_cases(inp)
            + new_function_cases(inp) + index_cases(inp) + numeric_cases(inp)
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
