"""The comparison case table — **it does not import torch.**

Golden stage two runs in a browser and there is no real torch there. A case table that drags
torch in could not even be imported on that side. So the table alone sits here and the caller
decides which library goes in — every case takes it as an argument, as `lambda L: ...`.

`conformance.py` and `golden.py` read **the same table.** Kept in two copies they diverge
eventually, and then nobody knows which copy diverged.

## The case names stay as they are, Korean included

1,447 of the 2,991 names carry Korean, and translating them is **not** part of putting
this repository into English. A case name is a key, not prose — no user of the library
ever reads one; the reader is whoever is looking at a golden diff. The same distinction
is written down in `tests/test_messages.py`, which forbids Korean anywhere in the Python
library and then names two case names it is quoting on purpose.

Renaming one is a golden regeneration and a manifest move, and `borch-ts/test/cases.ts`
mirrors every name: a name that differs there by one character does not fail, it makes
the case **disappear**. Two guards catch a rename that forgets the rest —
`test_committed_golden.py` compares the manifest hash against the committed
`golden.json`, and the borch-ts runner has a staleness check of its own — so this is a
decision about what is worth doing, not about what is safe.
"""

import collections
import hashlib

import numpy as np


def _same_object(L, layer):
    """Whether an in-place layer handed back the tensor it was given.

    **The value cannot show this.** `ReLU(inplace=True)` and `ReLU()` compute the
    same numbers; what the flag buys is that the buffer is reused, and the only
    observable is identity.
    """
    x = L.tensor(np.array([-1.0, 2.0], dtype=np.float32))
    return str(layer(x) is x)


def _minstd(shape, scale, seed):
    """A spread **computed rather than drawn**, so that TypeScript can compute the same
    one.

    MINSTD — `s ← s·48271 mod 2³¹−1`. The modulus is small enough that the product
    stays under 2⁵³, so JavaScript's float64 multiply is exact and the two sides agree
    bit for bit. A larger multiplier would not: `1103515245 · 2³¹` is past where a
    double counts by ones, and the sequences would part after a step or two with
    nothing saying so.

    Uniform in `[-scale, scale)`. Where a fixture needs the *size* of its values rather
    than their distribution, that is enough — and where it needs the distribution, this
    is the wrong tool and `numpy` is the right one.
    """
    total = int(np.prod(shape)) or 1
    out = np.empty(total, dtype=np.float32)
    state = seed % 2147483647 or 1
    for i in range(total):
        state = (state * 48271) % 2147483647
        out[i] = (state / 2147483647.0 - 0.5) * 2.0 * scale
    return np.ascontiguousarray(out.reshape(shape))


def golden_inputs():
    """The inputs the cases use. The **order** they are drawn in decides the values, so it is left alone."""
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal(6).astype(np.float32)
    xp = np.abs(x1) + 0.2
    x2 = rng.standard_normal((3, 4)).astype(np.float32)
    img = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
    # The dtype is always written down. numpy's default integer follows C's `long`, which is
    # int64 on 64-bit macOS and Linux but **int32 on wasm32 (Pyodide)**.
    # Left out, the browser and the native side build different inputs and the golden
    # comparison stops being a comparison.
    # (Caught by measurement — the first thing the input fingerprint check ever found.)
    idx2 = np.array([[0, 2], [1, 3], [2, 0]], dtype=np.int64)
    # The tails of erf and gelu. xp only sees positives from 0.2 up and x1 is roughly [-2, 2],
    # so the two places where digits are lost (near the origin and at large |x|) had nobody
    # looking at them.
    tail = np.array([-8., -6., -4., -1., -1e-3, 0., 1e-3, 1., 4., 6., 8.], dtype=np.float32)

    # For the training cases. Planting fixed weights is what makes the three libraries
    # **start from the same place** — initialising separately shows whether the initialisation
    # diverged rather than what diverged.
    train_x = rng.standard_normal((24, 6)).astype(np.float32)
    train_y = rng.integers(0, 3, 24).astype(np.int64)
    w0 = (rng.standard_normal((8, 6)) * 0.3).astype(np.float32)
    b0 = (rng.standard_normal(8) * 0.1).astype(np.float32)
    w1 = (rng.standard_normal((3, 8)) * 0.3).astype(np.float32)
    b1 = (rng.standard_normal(3) * 0.1).astype(np.float32)

    # For the convolutions. img is (2,3,4,4), so a 3-channel → 4-channel 3×3 filter fits.
    cw = (rng.standard_normal((4, 3, 3, 3)) * 0.3).astype(np.float32)
    cb = (rng.standard_normal(4) * 0.1).astype(np.float32)

    # For CNN training — (8,1,8,8) → conv → pool → flatten(64) → 3
    cnn_x = rng.standard_normal((8, 1, 8, 8)).astype(np.float32)
    cnn_y = rng.integers(0, 3, 8).astype(np.int64)
    ck = (rng.standard_normal((4, 1, 3, 3)) * 0.3).astype(np.float32)
    ckb = (rng.standard_normal(4) * 0.1).astype(np.float32)
    fw = (rng.standard_normal((3, 64)) * 0.2).astype(np.float32)
    fb = (rng.standard_normal(3) * 0.1).astype(np.float32)

    # For recurrence and attention. (T=5, N=2, I=3) and (B=2, T=5, E=4).
    seq_x = rng.standard_normal((5, 2, 3)).astype(np.float32)
    attn_x = rng.standard_normal((2, 5, 4)).astype(np.float32)

    # ── from here on are the ones that used to be built **inside** the case functions ──
    #
    # Why they moved: built inside a case they do not go into `golden.json`, and then an
    # implementation that is not Python is left **with the expected value but no input.**
    # borch.ts really did get stuck on 87 cases for that one reason — there is no way round it
    # short of rebuilding numpy's random generator, and rebuilding that fails quietly when it
    # fails.
    #
    # **Each keeps its own seed and is appended at the end.** Consuming more of the `rng` above
    # would shift everything from x1 down — which is what this function's docstring warns
    # about. Not one digit moves, so the existing expected values do not move either.
    ck1 = (np.random.default_rng(13).standard_normal((4, 3, 3)) * 0.3).astype(np.float32)
    vol5 = np.random.default_rng(17).standard_normal((1, 2, 4, 4, 4)).astype(np.float32)
    ck3 = (np.random.default_rng(19).standard_normal((3, 2, 3, 3, 3)) * 0.3).astype(np.float32)

    # The high-rank battery. Exactly one axis is 3 so that a transposed axis is caught by
    # **shape** before value.
    high = {}
    for r in (6, 7, 8):
        shape = [2] * r
        shape[r // 2] = 3
        high[f"rank{r}"] = np.random.default_rng(100 + r).standard_normal(shape).astype(np.float32)
    v7 = np.random.default_rng(107).standard_normal([2] * 7).astype(np.float32)
    v8 = np.random.default_rng(108).standard_normal([2] * 8).astype(np.float32)

    # The 1-D and 3-D family. **One rng, used in order** — the order decides the values, so it
    # is carried over as it is.
    nd = np.random.default_rng(41)
    nd_seq = nd.standard_normal((2, 3, 8)).astype(np.float32)
    nd_k1 = (nd.standard_normal((4, 3, 3)) * 0.3).astype(np.float32)
    nd_vol = nd.standard_normal((1, 2, 4, 4, 4)).astype(np.float32)
    nd_k3 = (nd.standard_normal((3, 2, 3, 3, 3)) * 0.3).astype(np.float32)
    nd_img = nd.standard_normal((2, 3, 4, 4)).astype(np.float32)

    # Images for the transforms. **Both** uint8 and float are kept — that ToTensor divides by
    # 255 only for uint8 is the whole point, and with one of them missing that rule goes
    # unwatched.
    # The fixed weights for recurrence and attention.
    #
    # This used to walk `mod.named_parameters()` and take the shapes on the spot. That needs
    # torch to be present to know the shapes, which keeps it out of this function (stage one,
    # numpy only). So **the shapes are written down here** — wrong, and `load_state_dict` dies
    # loudly at the freezing step, so they do not go wrong quietly.
    #
    # The **order** they are drawn in decides the values. It is torch's `named_parameters()`
    # order as it stands: weight_ih, weight_hh, bias_ih, bias_hh.
    def _fixed(seed, shapes):
        r = np.random.default_rng(seed)
        return [(r.standard_normal(s) * 0.2).astype(np.float32) for s in shapes]

    # RNN(3,4), LSTM(3,4) and GRU(3,4) differ only in gate count — 1, 4 and 3 times.
    rnn_w = {}
    for kind, gates in (("RNN", 1), ("LSTM", 4), ("GRU", 3)):
        h = 4 * gates
        parts = _fixed(7, [(h, 3), (h, 4), (h,), (h,)])
        for name, arr in zip(("wih", "whh", "bih", "bhh"), parts):
            rnn_w[f"{kind.lower()}_{name}"] = arr
    # MultiheadAttention(4, 2): in_proj_weight, in_proj_bias, out_proj.weight, out_proj.bias
    mha = _fixed(11, [(12, 4), (12,), (4, 4), (4,)])

    # Where the activations **bend.** These values never come out of a random draw, so they are
    # written by hand — ±1 for `hardtanh`, 0 and 6 for `relu6`, ±3 for `hardsigmoid` and
    # `hardswish`, ±0.5 for the shrink family. Without the bending points, a rule that diverges
    # exactly there is not caught.
    kinks = np.array([-6., -3., -1., -0.5, -1e-3, 0., 1e-3, 0.5, 1., 3., 6.],
                     dtype=np.float32)

    # The transposed convolution's weights. **The axis order differs from `conv2d`'s** — it is
    # `(in, out, …)`. Reversed, the shape still fits and the values are wholly different, and
    # that is the most common mistake at this layer. The input channels match `nd_seq` (3),
    # `img` (3) and `nd_vol` (2) respectively.
    tc = np.random.default_rng(53)
    tw1 = (tc.standard_normal((3, 4, 3)) * 0.3).astype(np.float32)
    tw2 = (tc.standard_normal((3, 4, 3, 3)) * 0.3).astype(np.float32)
    tw3 = (tc.standard_normal((2, 3, 3, 3, 3)) * 0.3).astype(np.float32)
    tb = (tc.standard_normal(4) * 0.1).astype(np.float32)
    tb3 = (tc.standard_normal(3) * 0.1).astype(np.float32)

    # For the top-level recurrence. **Moved out of `rnn_top_cases` unchanged** — built inside
    # a case it does not go into `golden.json`, and then borch.ts has the expected value with
    # no input and cannot ask thirty-five of them. The `nd_*` above came here for the same
    # reason.
    #
    # **The draw order decides the values.** Seed and order both have to stay as they were or
    # the existing expected values shift, and if they shift **nothing turns red** — re-freezing
    # makes torch produce new answers for the new inputs and all three agree with those. So
    # before the move, the thirty-five `rnntop::` entries in `golden.json` were set aside and
    # compared byte for byte afterwards. **Green after re-freezing is not evidence.**
    rt = np.random.default_rng(11)
    _T, _B, _I, _H = 3, 2, 4, 5
    top = {"x": rt.normal(size=(_T, _B, _I)).astype(np.float32)}
    top["xb"] = np.ascontiguousarray(top["x"].transpose(1, 0, 2))
    top["xs"] = rt.normal(size=(_B, _I)).astype(np.float32)
    for _key, _shape in (("h1", (1, _B, _H)), ("c1", (1, _B, _H)),
                         ("h2", (2, _B, _H)), ("c2", (2, _B, _H)),
                         ("hs", (_B, _H)), ("cs", (_B, _H))):
        top[_key] = rt.normal(size=_shape).astype(np.float32)

    def _top_w(gates, layers=1):
        out = []
        for k in range(layers):
            n = _I if k == 0 else _H
            out += [rt.normal(size=(gates * _H, n)).astype(np.float32),
                    rt.normal(size=(gates * _H, _H)).astype(np.float32),
                    rt.normal(size=(gates * _H,)).astype(np.float32),
                    rt.normal(size=(gates * _H,)).astype(np.float32)]
        return out

    for _name, _gates in (("lstm", 4), ("gru", 3), ("rnn_tanh", 1), ("rnn_relu", 1)):
        for _i, _arr in enumerate(_top_w(_gates)):
            top[f"{_name}_w{_i}"] = _arr
        for _i, _arr in enumerate(_top_w(_gates, 2)):
            top[f"{_name}_two{_i}"] = _arr
    for _name, _gates in (("lstm_cell", 4), ("gru_cell", 3),
                          ("rnn_tanh_cell", 1), ("rnn_relu_cell", 1)):
        for _i, _arr in enumerate(_top_w(_gates)):
            top[f"{_name}_w{_i}"] = _arr
    for _i, _arr in enumerate(_top_w(4)):
        top[f"drop_w{_i}"] = _arr
    rnn_top = {f"rt_{k}": v for k, v in top.items()}

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
            **rnn_top,
            "tw1": tw1, "tw2": tw2, "tw3": tw3, "tb": tb, "tb3": tb3}


def wide_cases(inp=None):
    """Outside the textbook's range but common in the tutorials and in practice.

    A name that exists with a different value is a lie too, so everything present is compared
    by value.
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

    # Convolution, pooling, normalisation — what S3 added. Looking at stride 2 alongside is
    # deliberate. The backward path that inserts zeros between gradients runs only there.
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
        # **Eval mode after saving and restoring.** If the running statistics fall out of the
        # state_dict it diverges here and nowhere else — training looks fine and only inference
        # is wrong, the defect the core actually had.
        ("BatchNorm2d(저장→복원→eval)", lambda L: _bn_roundtrip(L, img)),
        ("median(dim).indices", lambda L: L.median(L.tensor(x2), dim=1).indices),
        # The maths, shapes and comparisons added at stage 3. They only look at values, so
        # they sit here.
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
        # Where things of different lengths go into one batch. Textbook ch05 uses this path
        # as it stands.
        ("pad_sequence", lambda L: _pad(L)),
        ("pad_sequence(batch_first)", lambda L: _pad(L, batch_first=True)),
        ("pad_sequence(채움값)", lambda L: _pad(L, batch_first=True, padding_value=-1.0)),
        ("pad_sequence(2차원)", lambda L: L.nn.utils.rnn.pad_sequence(
            [L.tensor(x2[:3]), L.tensor(x2[:1])], batch_first=True)),
    ]
    return cases


def _pad(L, **kwargs):
    """Stacks three of lengths 3, 1 and 2. The smallest size at which the padded slots are visible."""
    parts = [L.tensor(np.array(v, dtype=np.float32))
             for v in ([1., 2., 3.], [4.], [5., 6.])]
    return L.nn.utils.rnn.pad_sequence(parts, **kwargs)


def _bn_roundtrip(L, img):
    trained = L.nn.BatchNorm2d(3)
    trained(L.tensor(img))                      # the running statistics update
    fresh = L.nn.BatchNorm2d(3)
    fresh.load_state_dict(trained.state_dict())
    fresh.eval()
    return fresh(L.tensor(img))


def _grad_of(leaf, name):
    """Checks a gradient **actually arrived** at the leaf, and hands it back.

    If it did not (None) the graph is cut. Left alone, that produces an unrelated error at the
    comparison step, and the state the core had — "training runs but the weights do not move" —
    goes uncaught.
    """
    if leaf.grad is None:
        raise RuntimeError(f"{name}: 기울기가 잎에 도착하지 않았다 — 그래프가 끊겼다")
    return leaf.grad


def grad_cases(inp=None):
    """Compares **gradients.**

    A right forward pass with a wrong backward pass gives the state where "training runs and
    the loss goes down and the values differ". The core had that kind for a long time with
    BatchNorm, and a value comparison alone does not catch it.

    Each case builds a leaf, folds to a scalar, calls `backward()` and hands back **the leaf's
    gradient.**
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

    # Element-wise — the ones that only take positives are given xp
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

    # Type conversion. **There was a quietly wrong place here.**
    #
    # `.float()` and `.double()` put `requires_grad=True` on the result without attaching the
    # parent. So `backward()` ran without an exception and only the original tensor's `.grad`
    # stayed `None` — no warning, no exception. `x.float()` is common in the tutorials, so this
    # was a place where training silently did not happen. Twelve other places (tril, diag,
    # einsum, cumprod and so on) carry no gradient either, but there the result is
    # `requires_grad=False` and `backward()` **refuses** — the difference between absent and
    # wrong.
    unary("float()", lambda L, x: x.float())

    # `.double()` goes through the same `_cast`. The question used to be **whether the
    # browser side refuses it** — the core was expected to succeed, and it did, by handing
    # back `float32`. It refuses now, at `_cast`, so all three implementations agree and
    # the case says "all of ours refuse" rather than "the two part".
    #
    # Having no double precision is a documented limit (it was for TF.js and it is for
    # WGSL's f32). What was not documented is that the core has not had it either since
    # `Tensor.__init__` began narrowing at construction, and this case could not tell,
    # because a silent downcast is not a refusal.
    def double_grad(L):
        x = L.tensor(x1, requires_grad=True)
        x.double().sum().backward()
        return _grad_of(x, "double()")

    cases.append(("grad::double()=우리는거절", refusal_case(double_grad)))

    # Twelve places where torch flows a gradient and the core did not. In all of them the
    # result was `requires_grad=False` so `backward()` refused, which means **it was never
    # quietly wrong** — but the difference between absent and present remains. They flow now.
    #
    # Each slot is taken with a different weight multiplied in. A plain `sum()` makes every
    # gradient 1, and then `movedim` transposing an axis, or `tile` overlapping pieces wrongly,
    # both pass.
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
    # Input with zeros mixed in. The common derivation divides here and flows a quiet nan.
    flows("cumprod(0포함)", lambda L, x: L.cumprod(x, 0), zeroed)
    flows("cumprod(2차원)", lambda L, x: L.cumprod(x, 1), mat)
    flows("tile", lambda L, x: L.tile(x, (2,)), vec)
    flows("tile(2차원)", lambda L, x: L.tile(x, (2, 3)), mat)
    flows("movedim", lambda L, x: L.movedim(x, 0, 1), mat)
    flows("repeat_interleave", lambda L, x: L.repeat_interleave(x, 3), vec)
    flows("repeat_interleave(dim)", lambda L, x: L.repeat_interleave(x, 2, 0), mat)
    # **Does it flow where the function bends?**
    #
    # torch's relu gives a gradient of 0 when the input is exactly 0 — it is `x > 0`, not
    # `x >= 0`. The `x1` above is a random normal draw so 0 never comes up, and so nobody among
    # the 798 golden cases was looking at this place. borch.ts was flowing 1 there, and it
    # surfaced only while matching a ResNet against real torch (input gradient, max diff 1.5e-2).
    #
    # **Giving each slot a different weight is the condition.** A plain `sum()` changes only the
    # total whether the zero slot's gradient is 1 or 0, so it is buried among the others — that
    # is why `flows` multiplies a different weight per slot, and here it is the check itself.
    edge = np.array([-1., 0., 1., 0.], dtype=np.float32)
    flows("relu(0에서)", lambda L, x: L.relu(x), edge)

    flows("median()", lambda L, x: L.median(x), vec)
    flows("median(dim)", lambda L, x: L.median(x, dim=1).values, mat)
    flows("fmod(%)", lambda L, x: x % 2, vec)
    for who in (0, 1):
        flows(f"pad_sequence/{'ab'[who]}",
              lambda L, a, b: L.nn.utils.rnn.pad_sequence([a, b]), vec, short, which=who)

    # Activations — where the training path actually goes
    for name, fn in [("relu", lambda L, x: L.relu(x)),
                     ("sigmoid", lambda L, x: L.sigmoid(x)),
                     ("gelu", lambda L, x: L.nn.functional.gelu(x)),
                     ("silu", lambda L, x: L.nn.functional.silu(x)),
                     ("leaky_relu", lambda L, x: L.nn.functional.leaky_relu(x, 0.1)),
                     ("elu", lambda L, x: L.nn.functional.elu(x)),
                     ("pow2", lambda L, x: x ** 2), ("neg", lambda L, x: -x)]:
        unary(name, fn)

    # Reductions and shapes
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

    # Sampling and losses — **this is where a graph is easy to cut.** Handing back the value
    # alone sends no gradient to where the pick was made, and the classification losses become
    # non-differentiable whole. That really happened.
    idx2, targets = inp["idx2"], np.array([0, 1, 2], dtype=np.int64)
    unary("gather", lambda L, x: L.gather(x, 1, L.tensor(idx2)), x2)
    unary("nll_loss", lambda L, x: L.nn.functional.nll_loss(
        L.nn.functional.log_softmax(x, dim=-1), L.tensor(targets)), x2)
    unary("cross_entropy",
          lambda L, x: L.nn.functional.cross_entropy(x, L.tensor(targets)), x2)

    # The sampling and slicing family — **this is the easiest place of all to cut a graph.**
    # Handing back the value alone sends no gradient to where the pick was made and training
    # stops quietly. The core had it with topk and sort at ROADMAP item 11, and this library
    # was in the same state until review.
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

    # Indexing — what torch code does most often, and joined to the graph for the same reason
    # as slicing.
    unary("idx[0]", lambda L, x: x[0], x2)
    unary("idx[-1]", lambda L, x: x[-1], x2)
    unary("idx[1:3]", lambda L, x: x[1:3])
    unary("idx[:, 1]", lambda L, x: x[:, 1], x2)
    unary("idx[1, 2]", lambda L, x: x[1, 2], x2)
    unary("idx[0:2, 1:3]", lambda L, x: x[0:2, 1:3], x2)
    unary("idx[목록]", lambda L, x: x[[2, 0]], x2)

    # Concatenation and stacking — the DataLoader's collate stands on these
    unary("cat", lambda L, x: L.cat([x, x * 2]))
    unary("cat(dim=1)", lambda L, x: L.cat([x, x * 2], 1), x2)
    unary("stack", lambda L, x: L.stack([x, x * 3]))
    unary("stack(dim=1)", lambda L, x: L.stack([x, x * 3], 1), x2)

    # The method form — torch code mixes `x.exp()` and `torch.exp(x)`
    unary("메서드 x.abs()", lambda L, x: x.abs())
    unary("메서드 x.exp()", lambda L, x: x.exp())
    unary("메서드 x.sqrt()", lambda L, x: x.sqrt(), xp)

    # Layer wrappers — wrapping the ones that already had a functional edition. Both value and
    # gradient are looked at.
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
    # **A loss has to be given two different values.** `binary`'s default was `x1` on both
    # sides, so it was `l1_loss(x, x)` — prediction equal to target — and then the gradient is
    # **all zeros**: an implementation with the sign flipped and one that never divides by the
    # count both give zero. Eight places were caught this way while counting cases whose frozen
    # answer held a single value.
    binary("L1Loss(층)", lambda L, a, b: L.nn.L1Loss()(a, b), "a", x1, -x1)
    binary("SmoothL1Loss(층)", lambda L, a, b: L.nn.SmoothL1Loss()(a, b), "a", x1, xp)
    binary("BCEWithLogitsLoss", lambda L, a, b: L.nn.BCEWithLogitsLoss()(a, b), "a")

    # Embedding — when the same index appears more than once, that row's gradient has to
    # **accumulate.**
    def emb_grad(L):
        w = L.tensor(inp["w0"][:5], requires_grad=True)      # (5, 6)
        idx = L.tensor(np.array([0, 2, 0, 4], dtype=np.int64))
        L.nn.functional.embedding(idx, w).sum().backward()
        return _grad_of(w, "embedding")

    cases.append(("grad::embedding(중복 번호)", emb_grad))

    # Maths and shapes — each had a TF.js counterpart, but backward had to be attached
    unary("where", lambda L, x: L.where(x > 0, x, x * 0.1))
    unary("masked_fill", lambda L, x: x.masked_fill(x > 0, -1.0))
    unary("clone", lambda L, x: x.clone())
    unary("permute", lambda L, x: x.permute(1, 0), x2)
    unary("squeeze", lambda L, x: x.unsqueeze(0).squeeze())
    unary("max(dim)", lambda L, x: x.max(dim=1).values, x2)
    unary("min(dim)", lambda L, x: x.min(dim=1).values, x2)
    unary("var", lambda L, x: x.var())
    unary("std", lambda L, x: x.std())

    # Binary — both leaves are looked at. Looking at one side alone misses a cut on the other.
    for which in ("a", "b"):
        binary("add", lambda L, a, b: a + b, which)
        binary("sub", lambda L, a, b: a - b, which)
        binary("mul", lambda L, a, b: a * b, which)
        binary("div", lambda L, a, b: a / b, which, xp, xp)
        binary("maximum", lambda L, a, b: L.maximum(a, b), which, x1, -x1)
        binary("minimum", lambda L, a, b: L.minimum(a, b), which, x1, -x1)
        binary("matmul", lambda L, a, b: a @ b, which, x2, x2.T.copy())
        # **Prediction and target are given different values** — equal, the gradient is all
        # zeros and nothing is being asked.
        binary("l1_loss", lambda L, a, b: L.nn.functional.l1_loss(a, b), which, x1, -x1)
        binary("mse_loss", lambda L, a, b: L.nn.functional.mse_loss(a, b), which, x1, xp)
        binary("smooth_l1_loss",
               lambda L, a, b: L.nn.functional.smooth_l1_loss(a, b), which, x1, xp)
        binary("cosine_similarity",
               lambda L, a, b: L.nn.functional.cosine_similarity(a, b), which, x2, x2 * 2)

    # Convolution — **where backward is written by hand.** Input, weight and bias, all three.
    # Looking at stride 2 alongside is deliberate — the path that inserts zeros between
    # gradients runs only there.
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

    # **Nobody was asking about average pooling's backward pass.**
    #
    # `F.avg_pool2d` was in the table for its forward pass only. And then an integration run
    # caught average pooling's backward **not running at all** in borch.ts — an unused binding
    # dropped out of the layout and invalidated the whole command buffer, which WebGPU does not
    # throw for. The loss sat at ln 10 and the ms/step kept coming. Had the table asked this,
    # it would not have taken an integration run.
    #
    # Where it parts from max pooling is the point. max flows to the one winning slot and avg
    # divides 1/n across every slot in the window. Implement one as the other and the forward
    # pass still looks fine.
    def pool_grad(name, fn, arr=img):
        def run(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            # A different weight per slot. Folded uniformly, avg and max give **the same sum
            # of input gradients**, so a wrong way of dividing still passes.
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

    # BatchNorm — the mean and variance have to be inside the graph. Taken outside, the input
    # gradient is off and the weight gets **nothing at all** (None). So both are looked at.
    def bn_grad(which):
        def run(L, w=which):
            x = L.tensor(img, requires_grad=True)
            bn = L.nn.BatchNorm2d(3)
            bn(x).sum().backward()
            return _grad_of(x if w == "x" else bn.weight, f"BatchNorm2d/{w}")
        cases.append((f"grad::BatchNorm2d/{which}", run))

    for which in ("x", "weight"):
        bn_grad(which)

    # **The `sum()` above hides half of BatchNorm's backward pass.**
    #
    # The input gradient has three terms: one arriving directly, and two correction terms that
    # exist because the mean and variance depend on the input. When the upstream gradient is
    # **all ones**, those two corrections cancel exactly — the expected value of 4.7e-10 in the
    # case above is the trace of that cancellation, which is to say the case is not asking
    # about the corrections at all.
    #
    # An implementation that leaves the corrections out (the common mistake of treating mean
    # and variance as constants) passes above and is caught below. Folding with a different
    # weight per slot breaks the cancellation.
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

    # ── backward with the seed given directly (a Jacobian-vector product) ──
    #
    # `y.backward(v)` gives `(∂y/∂x)ᵀ v`. The cases above all fold with `.sum()` before calling,
    # and that is **the special case where v is all ones.**
    #
    # **A uniform seed makes this case measure nothing.** `backward(ones)` gives the same answer
    # as `sum().backward()`, so an implementation that ignores the seed entirely passes. So a
    # different value goes into each slot here.
    #
    # borch.ts alone was not accepting it — its first argument was `retain_graph`. The core
    # accepted it from the start, and the golden cases did not ask, so that divergence was
    # invisible.
    def seeded_backward(name, fn, arr, shape_of, which="x", arr2=None):
        def run(L, f=fn, a=arr, s=shape_of, w=which, b=arr2, n=name):
            x = L.tensor(a, requires_grad=True)
            args = [x] if b is None else [x, L.tensor(b, requires_grad=True)]
            y = f(L, *args)
            # 1, 2, 3 … laid out in the output's shape. Without symmetry the seed is really used.
            seed = np.arange(1, int(np.prod(s)) + 1, dtype=np.float32).reshape(s)
            y.backward(L.tensor(seed))
            return _grad_of(args[0] if w == "x" else args[1], n)
        cases.append((f"grad::vjp::{name}", run))

    seeded_backward("exp", lambda L, x: L.exp(x), x1, (6,))
    seeded_backward("square", lambda L, x: x * x, x1, (6,))
    seeded_backward("mul/a", lambda L, a, b: a * b, x1, (6,), "x", x1)
    seeded_backward("mul/b", lambda L, a, b: a * b, x1, (6,), "b", x1)
    # Where the output shape differs from the input's. Whether the seed's shape is taken from
    # the output is what parts here.
    seeded_backward("matmul", lambda L, x: L.matmul(x, x.t()), x2, (3, 3))
    seeded_backward("reshape", lambda L, x: x.reshape(2, 3), x1, (2, 3))
    # torch accepts a seed on a scalar too — the value is multiplied by that much.
    seeded_backward("scalar", lambda L, x: (x * x).sum(), x1, ())

    # ── where it refuses — **all three with the same wording** ──
    #
    # Different from `_as_expected`. That one is where the browser is **deliberately** unlike
    # torch; this is where all three have to match torch. The value cannot be frozen, so a
    # fragment of the wording is frozen instead — let it pass and "it did not throw" becomes the
    # answer, and they diverge.
    def refuses(name, fragment, fn):
        def run(L, f=fn, frag=fragment):
            try:
                f(L)
            except Exception as exc:                            # noqa: BLE001
                return frag if frag in str(exc) else f"다른 문구 <{exc}>"
            return "안 던졌다"
        cases.append((f"grad::거절::{name}", run))

    # **There is an order.** Non-scalar and not requires_grad, and torch refuses on this side
    # rather than with "not a scalar" — measured. borch.ts alone had the order reversed, and the
    # golden cases did not ask that combination, so it was invisible.
    refuses("requires_grad 를 먼저 본다",
            "does not require grad",
            lambda L: L.tensor(x1).backward())
    refuses("씨앗 없는 비스칼라",
            "grad can be implicitly created only for scalar outputs",
            lambda L: L.tensor(x1, requires_grad=True).backward())
    # A mismatched seed must not be fixed up by broadcasting. Fixed up, a plausible-looking
    # value comes out with a wrong gradient, and that shows up only as training not working.
    # The core was not looking here, so numpy's `ValueError` surfaced a long way from the cause.
    def bad_seed(L):
        y = L.tensor(x1, requires_grad=True) * 2
        y.backward(L.tensor(np.ones(7, dtype=np.float32)))

    refuses("씨앗 모양이 어긋남", "Mismatch in shape", bad_seed)

    # ── where it folds — **it only opens up when there are ties** ──
    #
    # `max()` folding [3,5,5,1,5] to 5 leaves three slots that made that 5. Where does it go on
    # the way back — **with all values distinct the question does not open**, because "give it
    # to the one slot chosen" and "spread over the equal-valued slots" give the same answer. The
    # table's gradient cases used distinct values for a long time, so an implementation that was
    # half right passed.
    #
    # torch's rule is three-part — an operation that **hands back an index** goes to the one
    # slot it chose, one that **does not** spreads evenly over the equal-valued slots, and one
    # that folds onto **sorted positions** goes to those positions. Only what all three can
    # answer goes in here. The other half (`median`'s ties, `mode`, `quantile` and `i0`, which
    # only the core gets right) is in `tests/test_fold_grad.py`, and moves here when borch.ts
    # catches up.
    tied = np.array([3.0, 5.0, 5.0, 1.0, 5.0], dtype=np.float32)
    step = np.array([0.5, -1.0, 2.0], dtype=np.float32)

    def folds(name, fn, arr=tied):
        def run(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            (out.sum() if out.numel() > 1 else out).backward()
            return _grad_of(x, n)
        cases.append((f"grad::접힘::{name}", run))

    folds("max() 동점 셋", lambda L, x: x.max())
    folds("min() 동점 없음", lambda L, x: x.min())
    folds("amax() 동점 셋", lambda L, x: x.amax())
    folds("amin() 동점 없음", lambda L, x: x.amin())
    # `max(dim=0)` hands back an index, so it is **the opposite rule.** Asking side by side on
    # the same data is what leaves the difference between them recorded as a case.
    folds("max(dim=0) 은 한 자리로", lambda L, x: x.max(dim=0)[0])
    # For a norm, `p` changes the rule outright. `inf` is the maximum absolute value so ties
    # open up, and a finite `p` flows to every slot.
    folds("norm(inf)", lambda L, x: x.norm(float("inf")))
    folds("norm(-inf)", lambda L, x: x.norm(float("-inf")))
    folds("norm(3)", lambda L, x: x.norm(3))
    # A step function's derivative is **zero, and zero is not the same as absent.** Without the
    # graph joined, `backward()` stops, and torch does not stop.
    folds("angle() 은 0 을 흘린다", lambda L, x: x.angle(), step)

    # The ones below were in `tests/test_fold_grad.py` alone at first. borch.ts could not
    # answer them then, so they could not go where all three are asked at once; all three answer
    # now.
    #
    # **`median` and `quantile` give the same value while standing on different things.** The
    # median of [1,5,5,5] is 5 and so is its 0.5 quantile, but median gives ⅓ to **all three**
    # slots holding 5 while quantile gives ½ to **the two positions** it used after sorting.
    # Measured by value alone the two look the same.
    even = np.array([1.0, 5.0, 5.0, 5.0], dtype=np.float32)
    dup = np.array([1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)
    nan_tie = np.array([1.0, np.nan, 5.0, 5.0, 5.0], dtype=np.float32)

    folds("median() 동점 셋", lambda L, x: x.median())
    folds("median() 짝수·동점", lambda L, x: x.median(), even)
    folds("median(dim=0) 은 한 자리로", lambda L, x: x.median(dim=0)[0])
    folds("nanmedian() 동점", lambda L, x: x.nanmedian(), nan_tie)
    folds("nanmedian(dim=0)", lambda L, x: x.nanmedian(0)[0], nan_tie)
    folds("mode() 는 마지막 자리로", lambda L, x: x.mode()[0], dup)
    folds("kthvalue(2)", lambda L, x: x.kthvalue(2)[0])
    folds("quantile(0.5) 정확히 맞음", lambda L, x: x.quantile(0.5))
    folds("quantile(0.3) 보간", lambda L, x: x.quantile(0.3))
    folds("quantile(0.5) 짝수는 둘로", lambda L, x: x.quantile(0.5), even)
    folds("quantile(0.75) 짝수", lambda L, x: x.quantile(0.75), even)
    # **The four rules the default hides.** All four cases above pass no
    # `interpolation`, so they exercise `linear` alone — and `lower`, `higher`,
    # `midpoint` and `nearest` are different answers *and different gradients*: the
    # split has to follow the rule that produced the value, or the forward stops
    # interpolating while the backward goes on doing it.
    for _how in ("lower", "higher", "midpoint", "nearest"):
        folds(f"quantile(0.3, {_how})",
              lambda L, x, h=_how: x.quantile(0.3, interpolation=h))
    # Its derivative is `i1`. borch.ts was **flowing a zero** here, and its comment cited the
    # core's hole as its grounds — one side had copied the other. A gradient whose value is zero
    # and a gradient that is absent are different statements, and in the copying the second
    # turned into the first.
    folds("i0() 의 도함수는 i1", lambda L, x: x.i0(), step)
    # The neighbour that does not fold. Asking side by side on the same data is what leaves the
    # difference in rule recorded.
    folds("topk(3) 는 셋 다", lambda L, x: x.topk(3)[0])
    folds("sort() 는 전부 하나씩", lambda L, x: x.sort()[0])
    folds("cummax(0) 은 늦은 자리를", lambda L, x: x.cummax(0)[0])

    return cases


ACT_PREFIX = "act::"

# `(function name, layer name)` — the ones callable with no arguments, on their defaults.
# The function form and the layer form have to give **the same value.** A layer calling the
# wrong function is caught by value alone, and the thinner the wrapper the less the eye sees it.
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
    """Seventeen activations. **Asked where they bend.**

    What this repository learned from `relu` applies here unchanged — a random input gives no
    special values. Exactly 0, exactly ±1, exactly ±3 and exactly 6 are never drawn, and those
    are precisely where the activations bend. So the input is written by hand (`kinks`).

    **Both** the function form and the layer form are asked. A layer is a one-line wrapper over
    a function and looks like it has nowhere to go wrong, but there is one way — calling a
    different function. That is caught by value alone.
    """
    inp = golden_inputs() if inp is None else inp
    k = inp["kinks"]
    x1, x2 = inp["x1"], inp["x2"]
    cases = []

    def add(name, fn, arr=k):
        """Attaches value and gradient as a pair. For an activation, **the gradient is the substance.**"""
        cases.append((ACT_PREFIX + name, lambda L, f=fn, a=arr: f(L, L.tensor(a))))

        def grad(L, f=fn, a=arr, n=name):
            x = L.tensor(a, requires_grad=True)
            out = f(L, x)
            # Folded with a different weight per slot — a plain `sum()` makes every gradient
            # 1, and then which slot is wrong leaves no trace in the value.
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(x, n)
        cases.append((ACT_PREFIX + f"grad::{name}", grad))

    # The ones with no arguments — the function form gives value and gradient, the layer form value.
    for fname, cls in _ACTS:
        add(f"F.{fname}", lambda L, x, f=fname: getattr(L.nn.functional, f)(x))
        cases.append((ACT_PREFIX + f"nn.{cls}",
                      lambda L, c=cls, a=k: getattr(L.nn, c)()(L.tensor(a))))

    # The ones that take arguments. **Asking only about the default lets an argument that is
    # never used at all pass.**
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

    # `softmin` is `softmax(-x)`. **Leave the sign out and it becomes softmax**, which is the
    # same function under another name and so parts only by value.
    add("F.softmin", lambda L, x: L.nn.functional.softmin(x, dim=-1), x2)
    cases.append((ACT_PREFIX + "nn.Softmin",
                  lambda L: L.nn.Softmin(dim=-1)(L.tensor(x2))))

    # `glu` splits an axis in half and uses one half as a gate — the only place here that is
    # not element-wise.
    add("F.glu", lambda L, x: L.nn.functional.glu(x, dim=-1), x1)
    cases.append((ACT_PREFIX + "nn.GLU", lambda L: L.nn.GLU(dim=-1)(L.tensor(x1))))

    # `prelu` has **a learned slope.** In the layer form that has to be picked up as a
    # parameter, and that place is the same machinery as the registration problem just seen in
    # the containers.
    add("F.prelu",
        lambda L, x: L.nn.functional.prelu(
            x, L.tensor(np.array([0.25], dtype=np.float32))))
    cases.append((ACT_PREFIX + "nn.PReLU", lambda L: L.nn.PReLU()(L.tensor(k))))
    cases.append((ACT_PREFIX + "nn.PReLU/파라미터 이름",
                  lambda L: " ".join(n for n, _ in L.nn.PReLU().named_parameters())))

    # ── the eight the binding was filling in. **The name is a shell and the arguments are real.** ──
    #
    # borch.ts had no layer, so the binding built one as a factory over the tensor methods, and
    # every case goes through the binding — so **the table structurally could not see that
    # absence.** Moving them over revealed that three of them take arguments.
    for cls, fn in (("SiLU", "silu"), ("Sigmoid", "sigmoid"), ("Tanh", "tanh"),
                    ("GELU", "gelu")):
        add(f"F.{fn}", lambda L, x, f=fn: getattr(L.nn.functional, f)(x))
        cases.append((ACT_PREFIX + f"nn.{cls}",
                      lambda L, c=cls, a=k: getattr(L.nn, c)()(L.tensor(a))))

    # **`approximate='tanh'` is a different formula.** The max difference is around 1e-4, so
    # "they are near enough, keep one" very nearly went through here, and asking only about the
    # default lets the argument be missing entirely.
    add("F.gelu(tanh)",
        lambda L, x: L.nn.functional.gelu(x, approximate="tanh"))
    cases.append((ACT_PREFIX + "nn.GELU(tanh)",
                  lambda L: L.nn.GELU("tanh")(L.tensor(k))))
    # Are the two **really different** — equal, and the two cases above ask one function twice.
    cases.append((ACT_PREFIX + "GELU 두 꼴은 다르다",
                  lambda L: str(bool(
                      (L.nn.functional.gelu(L.tensor(k))
                       - L.nn.functional.gelu(L.tensor(k), approximate="tanh"))
                      .abs().max().item() > 1e-6))))

    add("F.elu(alpha)", lambda L, x: L.nn.functional.elu(x, alpha=0.5))
    cases.append((ACT_PREFIX + "nn.ELU", lambda L: L.nn.ELU()(L.tensor(k))))
    cases.append((ACT_PREFIX + "nn.ELU(alpha)",
                  lambda L: L.nn.ELU(0.5)(L.tensor(k))))
    cases.append((ACT_PREFIX + "nn.LeakyReLU",
                  lambda L: L.nn.LeakyReLU()(L.tensor(k))))
    cases.append((ACT_PREFIX + "nn.LeakyReLU(기울기)",
                  lambda L: L.nn.LeakyReLU(0.2)(L.tensor(k))))
    # **`inplace` was not an argument on any of the thirteen activations**, so
    # `nn.ReLU(inplace=True)` — a line every torch model writes — stopped with a
    # `TypeError` about the argument count. Asked by *position*, because the seat is
    # what a positional call sees, and asked for the identity as well as the value:
    # in place the layer hands back **the same object**, and a version that computes
    # the right numbers into a new tensor passes a value comparison and fails the
    # thing the flag is for.
    for _act, _args in (("ReLU", ()), ("LeakyReLU", (0.2,)), ("ELU", (0.5,)),
                        ("CELU", (0.5,)), ("SELU", ()), ("Hardtanh", (-0.5, 0.5))):
        cases.append((ACT_PREFIX + f"nn.{_act}(inplace)",
                      lambda L, a=_act, g=_args: getattr(L.nn, a)(*g, True)(L.tensor(k))))
        cases.append((ACT_PREFIX + f"nn.{_act}(inplace)/같은 객체",
                      lambda L, a=_act, g=_args: _same_object(L, getattr(L.nn, a)(*g, True))))
    cases.append((ACT_PREFIX + "nn.Identity", lambda L: L.nn.Identity()(L.tensor(k))))
    # torch's `Identity` **swallows any argument at all** (measured). It is a placeholder
    # layer, so swapping a layer out and leaving its arguments in place while changing the name
    # is routine.
    cases.append((ACT_PREFIX + "nn.Identity(인자를 삼킨다)",
                  lambda L: L.nn.Identity(64, unused=True)(L.tensor(k))))

    # ── `Softmax()`'s default axis is **not `-1`** ──
    #
    # It differs by rank (measured: 1→0, 2→1, 3→**0**, 4→1). Asked at rank 2 only, `dim=1` and
    # `dim=-1` are the same axis and this rule is invisible; all three had it as `-1`.
    ranked = np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.1
    for cls in ("Softmax", "LogSoftmax"):
        cases.append((ACT_PREFIX + f"nn.{cls}(dim 지정)",
                      lambda L, c=cls: getattr(L.nn, c)(dim=-1)(L.tensor(x2))))
        cases.append((ACT_PREFIX + f"nn.{cls}(기본 축/랭크2)",
                      lambda L, c=cls: getattr(L.nn, c)()(L.tensor(x2.reshape(3, 4)))))
        cases.append((ACT_PREFIX + f"nn.{cls}(기본 축/랭크3)",
                      lambda L, c=cls: getattr(L.nn, c)()(L.tensor(ranked.copy()))))
        cases.append((ACT_PREFIX + f"nn.{cls}(기본 축/랭크4)",
                      lambda L, c=cls: getattr(L.nn, c)()(
                          L.tensor(ranked.reshape(2, 3, 2, 2).copy()))))
    return cases


NUM_PREFIX = "num::"


def numeric_cases(inp=None):
    """The numerical family. **Things that compose and things counted by a series, mixed.**

    The groups above finished once the existing operations were wired together. Here `lgamma`,
    `digamma` and `erfinv` have no closed form, so an approximation has to be written, and then
    **how close it gets** is the answer — it has to clear this repository's tolerance (1e-4),
    and that is what these cases are worth.

    `cdist`, `corrcoef` and `cov` are what the statistics side calls constantly, and they all
    compose.
    """
    inp = golden_inputs() if inp is None else inp
    x1, x2 = inp["x1"], inp["xp"]                       # xp is positives only
    mat = inp["x2"]                                     # (3, 4)
    other = (mat * 0.5 + 1.0).astype(np.float32)
    # The gamma family is asked **on positives only.** Diverging at the negative integers is the
    # definition, so making a case of that place turns it into a comparison of infinities and
    # asks about something unrelated to the approximation's quality.
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

    # ── the ones that compose. ──
    add("cdist", lambda L: L.cdist(L.tensor(mat), L.tensor(other)))
    add("corrcoef", lambda L: L.corrcoef(L.tensor(mat)))
    add("cov", lambda L: L.cov(L.tensor(mat)))
    add("tensordot",
        lambda L: L.tensordot(L.tensor(mat), L.tensor(other), dims=([1], [1])))
    add("trapezoid", lambda L: L.trapezoid(L.tensor(x2)))
    add("trapezoid(dx)", lambda L: L.trapezoid(L.tensor(x2), dx=0.5))
    add("cumulative_trapezoid", lambda L: L.cumulative_trapezoid(L.tensor(x2)))

    # ── the ones counted by a series. **Here, how close it gets is the answer.** ──
    with_grad("lgamma", lambda L, x: L.lgamma(x), gam)
    with_grad("digamma", lambda L, x: L.digamma(x), gam)
    with_grad("erfinv", lambda L, x: L.erfinv(x), unit)
    return cases


BIT_PREFIX = "bit::"


def bit_cases(inp=None):
    """Bitwise, integer maths, window functions.

    ## Negatives are the whole of this group

    Ask the bit operations on positives only and all three implementations pass while still
    differing from each other.

    - **The right shift is arithmetic.** `-3 >> 5` is `-1`, not a large positive (measured).
      Written as a logical shift it parts on negatives alone.
    - **`gcd` is always non-negative.** `gcd(-3, 5)` is `1` — keep the sign and a negative comes
      out.
    - **Zero has to be in there.** `lcm(0, 7)` is 0, and without asking that, nobody sees
      `|a·b|/gcd` dividing by zero.
    - **Booleans are a different computation.** `~True` is `False`, not `-2`. Asked on integers
      only, that whole branch never runs.

    ## For the window functions, `periodic` is the point

    It defaults to true, and it **adds one to the length** — a symmetric window of `N+1` is
    built and the last element dropped (measured). Measured only as false, the two branches look
    like the same function. `n == 1` is asked separately too; it is the only size at which the
    divisor becomes zero.

    ## Why `frexp` and `fill` are here

    `frexp` returns its exponent as **int32** (measured), and `fill`, unlike `fill_` one letter
    away, is **not in place.** Both look plausible by value alone, and only surface by asking
    separately whether the original is unchanged and what the type is.
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
    # The boolean branch. Only here does it run as a logical operation.
    add("bitwise_and(참거짓)",
        lambda L: L.bitwise_and(L.tensor(flags), L.tensor(~flags)))
    add("bitwise_or(참거짓)",
        lambda L: L.bitwise_or(L.tensor(flags), L.tensor(~flags)))
    add("bitwise_not(참거짓)", lambda L: L.bitwise_not(L.tensor(flags)))

    # The in-place edition. **It has to hand back the same tensor** for chained code to edit the original.
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
    # The large branch. The series changes at 3.75, so beyond that is asked separately.
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

    # **The axis is asked both across and down.** Asking with a non-square shape is what makes a
    # swapped axis fail on shape first.
    for dim in (0, 1):
        add(f"logcumsumexp(dim={dim})",
            lambda L, d=dim: L.logcumsumexp(L.tensor(grid), d))

    def logcumsumexp_grad(L):
        x = L.tensor(grid, requires_grad=True)
        out = L.logcumsumexp(x, 1)
        # **Counted with uneven weights.** All ones and the accumulation order cancels out, so
        # the rule about building up from the back never shows.
        (out * L.tensor(np.array([[1.0, 2.0, 0.5], [0.5, 3.0, 1.5]],
                                 dtype=np.float32))).sum().backward()
        return _grad_of(x, "logcumsumexp")

    cases.append((BIT_PREFIX + "grad::logcumsumexp", logcumsumexp_grad))

    add("fill", lambda L: L.fill(L.tensor(reals), 7.0))

    def fill_leaves_source(L):
        """**`fill` is not in place.** It hands back the original, so building it as in-place
        gives back something filled with 7."""
        x = L.tensor(reals.copy())
        L.fill(x, 7.0)
        return x

    add("fill(원본은 그대로)", fill_leaves_source)

    def detach_in_place(L):
        """Cuts the graph at **the same tensor.** Built as `detach()` by mistake, `is` becomes
        false and the original is still attached upstream, so backpropagation keeps flowing."""
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
        # The only size at which the divisor becomes zero.
        add(f"{name}(1)", lambda L, n=name: getattr(L, n)(1))
    add("hamming_window(alpha, beta)",
        lambda L: L.hamming_window(6, True, 0.5, 0.5))
    add("kaiser_window(beta=8)", lambda L: L.kaiser_window(6, True, 8.0))
    return cases


SPOT_PREFIX = "spot::"


def shape_index_cases(inp=None):
    """Shapes and indexing. The names that ask **which slot of the storage to look at.**

    ## `as_strided` is a view in torch and a copy for us

    torch sees one storage through several frames, so writing to the result changes the
    original. A borch.ts tensor holds one GPU buffer each and cannot express that view, and if
    the core alone returned a real view the three implementations would diverge — **a divergence
    invisible by value and visible only on write**, which is the worst kind. All three were
    settled on a copy, so the cases here ask about **reading** only. The writing side is
    `as_strided_scatter`'s job.

    ## What has to be asked for it to part

    - **Overlapping strides** are what make gradient accumulation visible. Measured with
      non-overlapping strides only, each slot is reached once and leaving the accumulation out
      still passes.
    - **`step`** has to be other than 1 for `slice_scatter`'s not touching the skipped slots to
      show.
    - **`offset`** has to be other than 0 for the shift in `diagonal_scatter` and `diag_embed`
      to be visible.
    - **A batch axis** is what reveals the convention that the diagonal goes to the back.
      Measured in 2-D only there is no axis left over, so the order cannot be asked.
    - **A size that does not divide evenly** is what shows `tensor_split` handing the remainder
      out from the front. With a size that divides, it looks like the same function as `chunk`.
    - **Repeated indices** are what part `index_put`'s and `put`'s two `accumulate` branches.
    - **A base that is not the identity** is what makes `include_self` visible. Multiplying into
      a plate filled with 1 gives the same answer either way (measured).
    - **A row that is already small** has to be mixed in for `renorm`'s leave-alone condition to
      show. And **the gradient of a scaled row** has to be asked for the `x` inside the scale
      factor to be visible — from the forward pass alone, a wrong backward written as `g·s`
      passes.
    """
    grid = np.arange(12, dtype=np.float32).reshape(3, 4)
    line = np.arange(10, dtype=np.float32)
    trio = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    duo = np.array([4.0, 5.0], dtype=np.float32)
    # **Uneven weights.** All ones and the differing share per slot cancels out of sight.
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

    # ── strides ──
    add("as_strided", lambda L: L.as_strided(L.tensor(grid), (2, 2), (1, 2)))
    add("as_strided(offset)",
        lambda L: L.as_strided(L.tensor(grid), (2, 2), (1, 2), 3))
    add("as_strided(겹침)",
        lambda L: L.as_strided(L.tensor(grid), (3, 3), (1, 1)))
    grad("as_strided", lambda L, x: L.as_strided(x, (3, 4), (1, 3)))
    # The gradient of overlapping strides — one slot is reached several times.
    grad("as_strided(겹침)", lambda L, x: L.as_strided(x, (3, 3), (1, 1)),
         np.arange(1, 10, dtype=np.float32).reshape(3, 3))

    def as_strided_in_place(L):
        """**The shape has to follow too.** Carrying the values alone passes only when asked on a square."""
        x = L.tensor(grid.copy())
        got = L.as_strided_(x, (2, 3), (1, 2))
        return f"{got is x} {tuple(x.shape)}"

    add("제자리::as_strided_", as_strided_in_place)

    add("as_strided_scatter",
        lambda L: L.as_strided_scatter(L.tensor(grid), L.zeros(2, 2),
                                       (2, 2), (1, 2), 3))

    # ── scatter ──
    add("select_scatter",
        lambda L: L.select_scatter(L.tensor(grid), L.zeros(4), 0, 1))
    add("slice_scatter",
        lambda L: L.slice_scatter(L.tensor(grid), L.zeros(3, 2), 1, 1, 3))
    add("slice_scatter(step=2)",
        lambda L: L.slice_scatter(L.tensor(grid), L.zeros(3, 2), 1, 0, 4, 2))
    # **The length changes with the offset.** On a (3,4), offsets 0 and 1 give three slots and
    # -1 gives two — give all three of them three and torch refuses on the spot.
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
        """The gradient towards the inserted values. It has to flow **only to where they went.**"""
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
    # **A batch axis is what reveals** the convention that the diagonal axis goes to the back.
    add("diag_embed(2차)", lambda L: L.diag_embed(L.tensor(grid)))
    add("diag_embed(dim1=0, dim2=1)",
        lambda L: L.diag_embed(L.tensor(grid), 0, 0, 1))
    grad("diag_embed", lambda L, x: L.diag_embed(x),
         np.arange(1, 49, dtype=np.float32).reshape(3, 4, 4))

    # ── splitting ──
    for k in (3, 4, 5):
        # Splitting 10 into 4 gives 3, 3, 2, 2 — the remainder is handed out **from the front.**
        add(f"tensor_split({k})",
            lambda L, n=k: L.cat(list(L.tensor_split(L.tensor(line), n))))
        # **The piece sizes themselves are asked.** Concatenated, how the remainder was divided
        # disappears — 3·3·2·2 and 2·2·3·3 concatenate to the same values.
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
    # Splitting a (3,4) into 3 gives 2, 1, 1, so the middle piece is (3,1).
    grad("tensor_split", lambda L, x: L.tensor_split(x, 3, dim=1)[1],
         np.array([[1.0], [3.0], [5.0]], dtype=np.float32))

    # ── unravelling indices, consecutive repeats ──
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
    # **`unique` had `return_inverse` and `dim` missing from the middle**, so
    # `x.unique(True, True)` asked for the inverse in torch and the counts here —
    # the same call, a different tuple, and both sides plausible. Asked by position
    # rather than keyword on purpose: that is the call the gap made wrong.
    #
    # **The names below carry a `_uq`/`_nm` tail on purpose.** This function is very
    # long and reuses short fixture names — `grid` alone is bound seven times. A plain
    # `grid = ...` here rebound it as `int64` for every later case in the function, and
    # forty `spot::` cases went red under real torch with dtype errors that had nothing
    # to do with `unique`. The blast radius of a name in a scope this size is the whole
    # scope below it.
    dups_uq = np.array([3, 1, 2, 1, 3, 3], dtype=np.int64)
    add("unique(inverse)", lambda L: L.unique(L.tensor(dups_uq), True, True)[1])
    add("unique(inverse, counts)",
        lambda L: L.unique(L.tensor(dups_uq), True, True, True)[2])
    grid_uq = np.array([[1, 2], [3, 4], [1, 2]], dtype=np.int64)
    add("unique(dim=0)", lambda L: L.unique(L.tensor(grid_uq), True, False, False, 0))
    add("unique(dim=0, inverse)",
        lambda L: L.unique(L.tensor(grid_uq), True, True, False, 0)[1])
    # `norm`'s `keepdim` sat where `dtype` goes, so `x.norm(2, 1, True)` folded the
    # axis away and set the dtype to `True`. Asked at two `p` values because the
    # branches are separate code — a reduction that keeps its rank at `p=2` and drops
    # it at `p=inf` is a shape that changes with a hyperparameter.
    plane_nm = np.array([[1., -2., 3.], [4., 0., -1.]], dtype=np.float32)
    for _p, _label in ((2, "2"), (float("inf"), "inf")):
        add(f"norm(p={_label}, dim=1, keepdim)",
            lambda L, p=_p: L.tensor(plane_nm).norm(p, 1, True))

    # ── masks, flat insertion ──
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
        """**It exists as a method only** — there is no top-level name `torch.masked_scatter_`."""
        x = L.tensor(grid.copy())
        got = x.masked_scatter_(L.tensor(mask), L.tensor(feed))
        return f"{got is x} {float(x[0, 0].item())}"

    add("제자리::masked_scatter_", masked_scatter_in_place)

    # **The indices overlap** — 0 appears twice. Only here do the two branches part.
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

    # ── reducing while inserting ──
    #
    # **The base is 2.5.** At 1 it is the identity for multiplication and `include_self` is
    # invisible; at 0 the same happens for addition.
    base = np.full((3, 4), 2.5, dtype=np.float32)
    lines = np.array([0, 0, 2], dtype=np.int64)
    # `index_reduce` has no `sum` — that place is `index_add` (measured).
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

    # ── renorm ──
    #
    # The first row is already small and has to be **left alone.** The other two are scaled.
    tall = np.array([[3.0, 4.0], [6.0, 8.0], [30.0, 40.0]], dtype=np.float32)
    for p in (1, 2, 3):
        add(f"renorm(p={p})",
            lambda L, q=p: L.renorm(L.tensor(tall), q, 0, 5.0))
    add("renorm(dim=1)", lambda L: L.renorm(L.tensor(tall), 2, 1, 5.0))
    # **The gradient of a scaled row.** x is inside the scale factor, so writing it as `g·s`
    # parts here.
    grad("renorm", lambda L, x: L.renorm(x, 2, 0, 5.0))

    # ── composition and construction ──
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
    # **The input has negatives mixed in.** WGSL's `pow` has no answer for a negative base, so
    # writing this as a power gives NaN here — measured on positives only, that branch is invisible.
    add("vander", lambda L: L.vander(L.tensor(trio)))
    add("vander(N=2)", lambda L: L.vander(L.tensor(trio), 2))
    add("vander(increasing)", lambda L: L.vander(L.tensor(trio), None, True))
    add("vander(N=5)", lambda L: L.vander(L.tensor(trio), 5))

    # ── matrices ──
    m1 = np.arange(6, dtype=np.float32).reshape(2, 3)
    m2 = np.arange(12, dtype=np.float32).reshape(3, 4)
    m3 = np.arange(8, dtype=np.float32).reshape(4, 2)
    add("chain_matmul",
        lambda L: L.chain_matmul(L.tensor(m1), L.tensor(m2), L.tensor(m3)))
    add("ger", lambda L: L.ger(L.tensor(trio), L.tensor(duo)))
    add("mv", lambda L: L.mv(L.tensor(grid),
                             L.tensor(np.array([1., 0., 0., 2.],
                                               dtype=np.float32))))
    # `mv` is a matrix multiply with a 1-D operand in it — its backward pass was dropping an
    # axis in the core.
    grad("mv", lambda L, x: L.mv(x, L.tensor(
        np.array([1., 0., 0., 2.], dtype=np.float32))),
        np.array([1.0, 2.0, 0.5], dtype=np.float32))
    return cases


CPLX_PREFIX = "cplx::"
FFT_PREFIX = "fft::"

# **Cases the core alone sees.** Exactly the opposite direction from `WEBGPU_PREFIX`.
#
# It emptied once when complex arrived in all three, `fft` refilled it, and it **emptied again**
# — as things narrow in the order core → borch.ts → binding, this list shows "how far along we
# are" as a number. Empty, `startswith(())` is always false and nothing is skipped.
#
# **The device entries stay.** That place has opened and closed twice, so a third time is coming.
# **The asymmetric eigendecomposition is in the core alone.** borch.ts has only the symmetric
# `eigh`, and the eigendecomposition of a general matrix is a Hessenberg reduction plus QR
# iteration, which is a different size of job in WGSL.
#
# **That reason was measured false and is kept as written rather than corrected.** What is
# certain: `borch-ts/src/linalg.ts` has no WGSL in it at all — zero matches for wgsl, dispatch,
# createBuffer, GPUDevice or dev(), and 55 for Float64Array — so it is host-side code and `eigh`
# there is Jacobi rotations, with QR iteration sitting directly beside it. The borch-ts session
# landed `linalg.eig` on that basis (1313c41): `linalg.ts` exports `eigvals` and `eig`, and
# `Tensor.eig` is there and async, like `cholesky`, `svd` and `eigh`.
#
# **What is not yet known is whether this entry can come out.** `borch_webgpu` names neither
# `eig` nor `eigvals` anywhere in `_ops.py`; the nearest match is `eigvalsh` in an argument
# table. The binding forwards unknown names to borch.ts generically and resolves promises
# through `settle`, so `linalg.eig` may already work without being named — and that can only be
# answered by running it in a browser, which nobody has done. So: the binding does not name it,
# and whether generic forwarding reaches it is unmeasured. Closing that is a browser run, not a
# reading.
#
# The binding could fill it in with numpy, but **that is the very thing being narrowed** — if
# Python computes it instead, the name is still absent for anyone using borch.ts, and the golden
# cases go green through the binding. This is written down so that cover is not built again.
CORE_ONLY_PREFIXES = ("linalg::eig::",)


def complex_cases(inp=None):
    """Complex numbers — **stage one is the core (numpy) alone.**

    The sister library (borch.ts) has no storage for them yet, so these cases do not run in a
    browser today. That makes this the opposite direction from `webgpu::` — this is a table
    **the core alone sees**, and it goes green alongside once borch.ts has interleaved storage.

    ## The convention is pinned by measurement

    **torch refuses `backward()` on a complex loss** (measured). If the loss is always real, the
    Wirtinger convention settles to this —

        z.grad = ∂L/∂re + i·∂L/∂im

    and measurement backs it (z = 1+2j): `z.real → 1+0j`, **`z.imag → 0+1j`** (not −1j),
    `|z|² → 2+4j`, `(z·z̄).real → 2+4j`.

    ## Where the conjugate attaches and where it does not

    Under this convention **a holomorphic function's backward pass is `conj(f'(z))·g`.**
    Multiplication and division are that case, and over the reals the conjugate is the identity,
    so **with real inputs there is no telling whether it is there.**

    Conversely `abs` returns a real, so it is not holomorphic and the conjugate does **not**
    attach — it is `z/|z|`. `conj` itself is `conj(g)`. Asking all three in one table is what
    parts which rule attaches where.

    ## The gradient is taken at real leaves

    Rather than building a complex leaf directly, they are woven with `complex(re, im)` and
    **the gradients of the two real leaves** are looked at. That is the convention's other
    direction, and the value comes out split as `(∂L/∂re, ∂L/∂im)`, so **which side is wrong is
    visible** — taken as one complex number the two are mixed together.
    """
    re = np.array([1.0, -3.0], dtype=np.float32)
    im = np.array([2.0, 0.5], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((CPLX_PREFIX + name, fn))

    def z(L):
        return L.complex(L.tensor(re), L.tensor(im))

    def grad(name, fn):
        """Concatenates the gradients of the two leaves `re` and `im` and freezes them as one."""
        def run(L, f=fn, n=name):
            r = L.tensor(re.copy(), requires_grad=True)
            i = L.tensor(im.copy(), requires_grad=True)
            f(L, L.complex(r, i)).sum().backward()
            return L.cat([_grad_of(r, n), _grad_of(i, n)])

        cases.append((CPLX_PREFIX + f"grad::{name}", run))

    # ── construction and extraction ──
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
    # **Asked through `conj_physical`.** torch's `conj` is **lazy** — it raises the conjugate
    # bit and does not flip the values, so `view_as_real` refuses with "unresolved conjugate"
    # (measured). Ours flips immediately, so that state does not exist. That divergence is in the
    # README under "where it parts from torch"; here the intent is to ask about **the value**, so
    # both sides are asked through a name that produces one.
    add("conj_physical", lambda L: L.view_as_real(L.conj_physical(z(L))))
    add("angle", lambda L: L.angle(z(L)))
    add("abs", lambda L: z(L).abs())
    add("abs 의 형", lambda L: str(z(L).abs().dtype))
    add("is_complex", lambda L: str(L.is_complex(z(L))))

    # ── arithmetic ──
    add("z * z", lambda L: L.view_as_real(z(L) * z(L)))
    add("z + z", lambda L: L.view_as_real(z(L) + z(L)))
    add("z / z", lambda L: L.view_as_real(z(L) / z(L)))
    add("z * 실수", lambda L: L.view_as_real(z(L) * L.tensor(re)))
    for other, tag in ((np.float32, "float32"), (np.int64, "int64")):
        add(f"complex64 + {tag} 의 형",
            lambda L, k=other: str(
                (z(L) + L.tensor(np.array([1], dtype=k))).dtype))

    # ── gradients — **where the conjugate attaches and where it does not** ──
    grad("z.real", lambda L, w: L.real(w))
    # **It is `0+1j`.** Written as `−1j` it runs plausibly with only the sign reversed.
    grad("z.imag", lambda L, w: L.imag(w))
    grad("abs(z)", lambda L, w: w.abs())
    grad("abs(z) 제곱", lambda L, w: w.abs() * w.abs())
    # Multiplication and division are **where the conjugate attaches.**
    grad("(z*z).real", lambda L, w: L.real(w * w))
    grad("(z*conj(z)).real", lambda L, w: L.real(w * L.conj_physical(w)))
    grad("view_as_real 합", lambda L, w: L.view_as_real(w))

    # ── printing ──
    #
    # **The characters are the specification.** A complex `repr` is not the real one lightly
    # adjusted; it has one more rule — **the real part and the imaginary part are measured
    # separately.** In `[1+2j, -0.5-1j]` the real part needs four decimals and the imaginary part
    # is integral, so torch prints `1.0000+2.j`. Measured under one format it becomes
    # `1.0000+2.0000j` — every value right and the characters different.
    #
    # Three rows part that rule: real part decimal only, imaginary part decimal only, and the
    # place where the type name appears (an empty tensor has no `j` as a clue, so torch prints
    # the type).
    def shown(fn):
        return lambda L, f=fn: repr(f(L))

    def cx(L, values):
        return L.tensor(np.array(values, dtype=np.complex64))

    add("repr::실수부만 소수", shown(lambda L: cx(L, [1 + 2j, -0.5 - 1j])))
    add("repr::허수부만 소수", shown(lambda L: cx(L, [1 + 2j, -3 + 0.5j])))
    add("repr::둘 다 정수", shown(lambda L: cx(L, [1 + 2j, -3 - 1j])))
    add("repr::2 차원", shown(lambda L: cx(L, [[1 + 2j, -0.5 - 1j],
                                              [3 + 0j, 0 + 4j]])))
    # **A negative zero keeps its sign** — print the imaginary part as an absolute value and it
    # parts here and nowhere else.
    add("repr::음의 0 허수부", shown(lambda L: cx(L, [complex(1.0, -0.0)])))
    # **The path that carries the sign is asked directly** — no repr involved.
    #
    # `cplx::repr::음의 0 허수부` above caught this defect first, and when that case turns red
    # the screen says "the characters differ". The cause was two slots away — the binding's
    # `_read` was taking values as `np.asarray(JsProxy, dtype=float32)` and **turning negative
    # zero into zero** (on the JS side it arrived with `Object.is(x, -0)` still true).
    #
    # **A value comparison can never catch it** — because `-0.0 == 0.0`. So the sign bit is
    # frozen as the answer. A regression reads immediately as "the conversion lost the sign".
    #
    # Asked on a **real** tensor rather than a complex one. The defect was there before complex
    # numbers existed and the complex repr merely happened to expose it, so asking on a real is
    # what puts it in the right place. The answer comes out **as characters.** The harness calls
    # `.detach()` on an answer, so handing back an array stops at the freezing step, and with
    # only two possible signs the characters read better anyway.
    def signbits(L):
        bits = np.signbit(to_numpy(L.tensor([-0.0, 0.0, -1.5, 1.5])))
        return "".join("-" if b else "+" for b in bits)

    add("read::음의 0 이 변환을 건넌다", signbits)

    # An empty one has no `j`, so the type name appears (measured). With values it does not.
    add("repr::빈 것", shown(lambda L: cx(L, [])))
    # How many fit on a line is characters too. torch counts by **width**, not by character count.
    add("repr::줄바꿈",
        shown(lambda L: cx(L, [complex(k, 0.5) for k in range(12)])))
    add("repr::grad_fn 이 붙는다",
        shown(lambda L: L.complex(L.tensor(re, requires_grad=True),
                                  L.tensor(im, requires_grad=True))))
    # **A place that surfaced on the real side alongside.** The integer edition's formatting put
    # a dot on `nan` too, giving `tensor([nan., 1.])` — the decimal edition did not part, because
    # `f"{nan:.4f}"` is already `nan`, so it happened only on an **integer** tensor with a nan in it.
    add("repr::nan 낀 정수 텐서",
        shown(lambda L: L.tensor(np.array([float("nan"), 1.0], dtype=np.float32))))

    # ── refusals ──
    def refuses_complex_loss(L):
        """**A complex loss has to be refused.** The whole convention above stands on this.

        That the loss is real is the premise of `z.grad = ∂L/∂re + i·∂L/∂im`. Without the
        refusal, a plausible number goes into a place the convention does not define, and every
        value case stays green — **a premise not asked as a case is not a premise but a hope.**
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
    """Fourier transforms — `torch.fft` and `stft`.

    **They stand on the complex numbers.** These names were refusals for a long time and the
    refusal said "the complex convention is not settled". Because that reason was accurate, the
    door opened the day the convention was settled — written as "there is no storage", nobody
    would have asked again after storage arrived.

    ## The gradient matters more than the value

    The transform is linear, so the forward pass is easy to get right. The hard place is **which
    half gets counted** —

    * `rfft`'s backward pass brings a gradient only to the stored half. Add the conjugate pair
      and it doubles.
    * `irfft`'s backward pass has to count **the edges once and the middle twice**, because the
      restored conjugate pair came from the same stored slot.

    The two are mistakes in opposite directions, and **both leave the forward value intact.**

    ## An input that dodges `abs`'s blade

    There is a reason the signal in the `stft(…).abs()` case is not a regular ramp. A ramp
    (`arange/8 − 1`) makes the Nyquist bin **exactly 0**, and `abs` is not differentiable there
    and its sign depends on the implementation's rounding — we accumulate in float64 and chose
    +1, torch has a float32 FFT and chose 0. **The rules did not part; the case was standing on
    the blade**, and freezing such a case would stuff a floating-point accident into the golden
    answers as a specification. Changed to a signal with no zero bin, all sixteen places agreed.
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
        """A complex answer is asked as a real pair — the golden file holds reals only."""
        return lambda L, f=fn: L.view_as_real(f(L))

    # ── values ──
    add("fft(실수)", pair(lambda L: L.fft.fft(x(L))))
    add("fft(복소)", pair(lambda L: L.fft.fft(z(L))))
    add("fft 의 형", lambda L: str(L.fft.fft(x(L)).dtype))
    add("ifft(fft)", pair(lambda L: L.fft.ifft(L.fft.fft(x(L)))))
    add("ifft(복소)", pair(lambda L: L.fft.ifft(z(L))))
    add("rfft", pair(lambda L: L.fft.rfft(x(L))))
    add("irfft(rfft)", lambda L: L.fft.irfft(L.fft.rfft(x(L))))
    add("irfft 의 형", lambda L: str(L.fft.irfft(L.fft.rfft(x(L))).dtype))
    # **Odd lengths are asked separately.** Without an n, `irfft` gives `2*(m-1)`, so only even
    # lengths come out — an odd one comes out only when n is given, and the number of restored
    # conjugate pairs parts there.
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
        # **It parts on odd lengths.** `fftshift` rolls by `n//2` and undoing it needs
        # `(n+1)//2` — asked on even lengths only, the two are equal and invisible.
        add(f"fftshift({n})", lambda L, k=n: L.fft.fftshift(L.fft.fftfreq(k)))
        add(f"ifftshift(fftshift({n}))",
            lambda L, k=n: L.fft.ifftshift(L.fft.fftshift(L.fft.fftfreq(k))))
    add("fftfreq(6, d=0.5)", lambda L: L.fft.fftfreq(6, 0.5))

    # ── gradients ──
    def grad(name, fn):
        def run(L, f=fn, n=name):
            leaf = L.tensor(xs.copy(), requires_grad=True)
            f(L, leaf).sum().backward()
            return _grad_of(leaf, n)

        add(f"grad::{name}", run)

    grad("fft 실수부", lambda L, t: L.real(L.fft.fft(t)))
    grad("fft 크기", lambda L, t: L.fft.fft(t).abs())
    # **Adding the conjugate pair doubles it.** Measured: `[4, 0, 1, 0, 1, 0]`.
    grad("rfft 실수부", lambda L, t: L.real(L.fft.rfft(t)))
    grad("rfft 허수부", lambda L, t: L.imag(L.fft.rfft(t)))
    grad("rfft 크기", lambda L, t: L.fft.rfft(t).abs())
    # **Counting the edges twice parts here.**
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
    # A short window is **centred with zeros padded on both sides** (measured). Left-aligned, it parts.
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

    # **`align_to_window` has no value case, because its whole behaviour is the
    # refusal.** torch rejects it unless `center` is false, and with `center` false it
    # answered the same at every setting — so the only thing to freeze is that the
    # refusal happens and says what torch's says. Nothing asked before this line, in
    # any of the three.
    def align_refuses(L):
        try:
            L.stft(s(L), 8, 4, window=hann(L), return_complex=True,
                   align_to_window=True)
        except RuntimeError as exc:
            return "거절" if "center = false" in str(exc) else f"다른 말 <{exc}>"
        except Exception as exc:                                    # noqa: BLE001
            return f"다른 예외 <{type(exc).__name__}>"
        return "안 던졌다"

    cases.append((FFT_PREFIX + "stft align_to_window(center 이면 거절)",
                  align_refuses))
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

    # ── refusals ──
    def refuses(name, body):
        def run(L, f=body):
            try:
                f(L)
                return "예외가 안 났다"
            except Exception as exc:                            # noqa: BLE001
                return type(exc).__name__

        add(name, run)

    refuses("rfft(복소)는 거절", lambda L: L.fft.rfft(z(L)))
    # **torch stops without `return_complex`** (measured). Choosing a default for it would teach
    # a shape that is about to be retired (a real `(…, 2)`).
    refuses("stft 는 return_complex 를 요구",
            lambda L: L.stft(s(L), 8, 4, window=hann(L)))
    refuses("복소 스펙트럼의 backward 는 거절",
            lambda L: L.fft.fft(
                L.tensor(xs.copy(), requires_grad=True)).sum().backward())

    # ── several axes, and Hermitian — **all of it assembled from the four above** ──
    #
    # There is no new kernel. `fft2` walks the axes one at a time and the Hermitian branch
    # resolves into a conjugate and a scale. So a right value brings the gradient with it, and
    # what is asked here is **order and normalisation**, since those are the assembler's to decide.
    #
    # `hfftn`'s leading axis was guessed as `ifft` and written wrongly — it is `fft`. **The shape
    # fits either way**, so without measuring the value it does not surface. Which is why the
    # value is frozen and not the shape.
    #
    # **The input was chosen so no zero appears.** That is the blade the `stft` comment above
    # records, and this group was built by someone who read that warning and stepped on it anyway
    # — an arithmetic sequence's `rfft` produces an exact zero, and putting `abs` over it leaves
    # the gradient's direction to the rounding.
    grid = np.array([[0.31, -1.2, 0.75, 2.1], [-0.4, 1.55, -2.3, 0.9],
                     [1.1, -0.62, 0.25, -1.7]], dtype=np.float32)
    cgrid = (grid + 1j * grid[::-1].copy()).astype(np.complex64)

    # **A complex number must not be handed back as it is.** The golden answers' JSON edition
    # holds reals only, so the imaginary part disappears quietly — `tests/test_export_json.py`
    # caught that. Spreading (re, im) onto the last axis with `view_as_real` keeps every bit of
    # the information and lets all three look at the same thing.
    def _real2(L, got):
        return L.view_as_real(got) if got.dtype in (L.complex64, L.cfloat) else got

    for _name in ("fft2", "fftn", "ifft2", "ifftn"):
        add(f"여러축::{_name}",
            lambda L, n=_name: _real2(L, getattr(L.fft, n)(L.tensor(cgrid.copy()))))
    for _name in ("rfft2", "rfftn"):
        add(f"여러축::{_name}",
            lambda L, n=_name: _real2(L, getattr(L.fft, n)(L.tensor(grid.copy()))))
    for _name in ("irfft2", "irfftn", "hfft2", "hfftn"):
        add(f"여러축::{_name}",
            lambda L, n=_name: _real2(L, getattr(L.fft, n)(L.tensor(cgrid.copy()))))
    for _name in ("ihfft2", "ihfftn"):
        add(f"여러축::{_name}",
            lambda L, n=_name: _real2(L, getattr(L.fft, n)(L.tensor(grid.copy()))))
    add("여러축::hfft", lambda L: L.fft.hfft(L.tensor(cgrid.copy())))

    # **Does it listen to its arguments?** Axis order, size and normalisation are the assembler's three.
    add("여러축::fft2(norm=ortho)",
        lambda L: _real2(L, L.fft.fft2(L.tensor(cgrid.copy()), norm="ortho")))
    add("여러축::fft2(norm=forward)",
        lambda L: _real2(L, L.fft.fft2(L.tensor(cgrid.copy()), norm="forward")))
    add("여러축::fft2(s)",
        lambda L: _real2(L, L.fft.fft2(L.tensor(cgrid.copy()), s=(2, 8))))
    add("여러축::fftn(dim 하나만)",
        lambda L: _real2(L, L.fft.fftn(L.tensor(cgrid.copy()), dim=(0,))))

    # **Gradients.** Being assembled, they should follow of their own accord, and where they do
    # not this catches it.
    def _grad_of_fft(L, name, arr):
        x = L.tensor(arr.copy(), requires_grad=True)
        y = getattr(L.fft, name)(x)
        w = L.arange(y.numel()).reshape(y.shape).float() * 0.13 - 0.7
        (y.real * w).sum().backward()
        return _grad_of(x, name)

    # **The leaf has to be real** — the binding does not give `requires_grad` to a complex leaf.
    # `fft2` returns a complex from a real input, so the question is unchanged.
    for _name, _arr in (("fft2", grid), ("rfft2", grid), ("rfftn", grid),
                        ("ihfft2", grid)):
        add(f"여러축::grad::{_name}",
            lambda L, n=_name, a=_arr: _grad_of_fft(L, n, a))

    return cases


MAKE_PREFIX = "make::"


def make_cases(inp=None):
    """**The names that have an answer even without complex numbers**, and a few constructors.

    ## Being able to answer and not being able to are different things

    If not having settled the complex convention also meant not having `is_complex`, textbook
    code branching on it would stop with an `AttributeError`. On a real tensor all these names
    have an answer — `real`, `conj` and `resolve_conj` are **the identity**, the three
    predicates are **all false**, and `angle` is π on negatives.

    `imag` alone refuses, and that is **because torch itself refuses on reals** (measured:
    "imag is not implemented for tensors with non-complex dtypes"). Not our limit, torch's,
    carried over.

    ## What has to be asked for it to part

    - **The type is asked in three ways.** `real(bool)` is `bool` (measured). Sending the
      identity through `positive`'s unary kernel drops the type to float32, and measured on
      float32 inputs only that is invisible — the same place the dtype-label case ran into.
    - **`angle` is the opposite.** An integer in still gives **float32** out (measured). An
      angle does not fit in an integer slot so that is right, and with reals only the rule does
      not show.
    - **`asarray` does not copy by default.** It copies only with `copy=True` (measured).
    - **`frombuffer`'s `offset` is in bytes** — read as an element count, the values shift.
    - **`range` includes its end.** `arange` excludes it — `range(0, 4)` gives five. Passed
      quietly to `arange` it is one element short, which is also torch's stated reason for
      retiring the name.

    ## Two are refused

    For `empty_strided` and `empty_permuted` **the strides are the only answer** (the values are
    garbage), and our tensors have no such thing as a stride. Different from `as_strided` —
    there the value is the answer, so a copy gives the same answer.
    """
    plain = np.array([[-1.5, 0.0, 2.0], [3.0, -4.0, 0.5]], dtype=np.float32)
    ints = np.array([1, -2, 3], dtype=np.int64)
    flags = np.array([True, False, True])
    kinds = ((plain, "float32"), (ints, "int64"), (flags, "bool"))
    cases = []

    def add(name, fn):
        cases.append((MAKE_PREFIX + name, fn))

    # ── the five identities — **the type has to hold too** ──
    for name in ("real", "conj", "conj_physical", "resolve_conj", "resolve_neg"):
        for src, tag in kinds:
            add(f"{name}({tag})",
                lambda L, n=name, s=src: getattr(L, n)(L.tensor(s)))
            add(f"{name}({tag}) 형",
                lambda L, n=name, s=src: str(getattr(L, n)(L.tensor(s)).dtype))

    # ── angle — the type is **always float32** ──
    for src, tag in kinds:
        add(f"angle({tag})", lambda L, s=src: L.angle(L.tensor(s)))
        add(f"angle({tag}) 형", lambda L, s=src: str(L.angle(L.tensor(s)).dtype))

    # ── the three predicates — all false ──
    for name in ("is_complex", "is_conj", "is_neg"):
        add(name, lambda L, n=name: " ".join(
            str(getattr(L, n)(L.tensor(s))) for s, _ in kinds))

    # ── construction ──
    add("asarray(list)", lambda L: L.asarray([1.0, 2.0]))
    add("asarray(ndarray) 형", lambda L: str(L.asarray(ints).dtype))
    raw = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    add("frombuffer",
        lambda L: L.frombuffer(bytearray(raw), dtype=L.float32))
    add("frombuffer(count=2)",
        lambda L: L.frombuffer(bytearray(raw), dtype=L.float32, count=2))
    # **`offset` is in bytes** — read as an element count it parts here.
    add("frombuffer(offset=4)",
        lambda L: L.frombuffer(bytearray(raw), dtype=L.float32, offset=4))
    # **It includes its end** — one slot different from `arange`.
    add("range(0, 4)", lambda L: L.range(0, 4))
    add("range(1, 7, 2)", lambda L: L.range(1, 7, 2))
    add("range(0, 1, 0.25)", lambda L: L.range(0, 1, 0.25))
    add("range 와 arange 의 개수",
        lambda L: f"{L.range(0, 4).numel()} {L.arange(0, 4).numel()}")

    # **A step of 0 has to stop all three.** torch stops too — the value does not move, so
    # there is no end. Unblocked, `(end-start)/0` becomes infinity and it blows up while
    # allocating, and that wording is indistinguishable from running out of memory.
    #
    # **A fragment of the wording is asked.** Being characters rather than a value, this is a
    # place comparing the three against each other does not catch, and this repository has been
    # bitten by that branch several times.
    def refuses_zero_step(name, call):
        def run(L, f=call):
            try:
                f(L)
            except Exception as exc:                                # noqa: BLE001
                return ("nonzero" if "nonzero" in str(exc)
                        else f"다른 문구 <{exc}>")
            return "안 던졌다"

        add(name, run)

    refuses_zero_step("arange(step=0)=거절", lambda L: L.arange(0, 5, 0))
    refuses_zero_step("range(step=0)=거절", lambda L: L.range(0, 5, 0))
    return cases


STAT_PREFIX = "stat::"


KEEP_PREFIX = "keep::"


def keepdim_cases(inp=None):
    """`keepdim` — **where an axis quietly disappears.**

    The first group of a table another session produced by sweeping 401 `Tensor` methods. Its
    character is not "the name is torch's and the meaning differs" but **"torch has an argument
    we do not accept"**, and among those `keepdim` alone is the kind that goes wrong quietly:

        m = x.argmax(dim=1, keepdim=True)      # torch: (2, 1)
        x - x.gather(1, m)                     # if we give (2,), broadcasting
                                               # **succeeds** right here

    A shape that does not fit stops loudly, and a shape with one axis missing **often does fit**
    under broadcasting. Then only the value is wrong, all the way to the end.

    ## What is asked

    The **shape** under `keepdim=True`. The values are already asked by other cases, and the
    only thing that parts here is one axis. So the shape is frozen as a string — asked as a
    value it passes, because `(2,)` and `(2,1)` hold the same number of elements. The harness
    looks at shapes too, but **the case name has to say that shape is this table's point** or
    the next person deletes it.
    """
    grid = np.array([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], dtype=np.float32)
    flags = np.array([[True, False, True], [False, False, True]])
    cases = []

    def add(name, fn):
        cases.append((KEEP_PREFIX + name, fn))

    def shape_of(fn):
        """Freezes the shape only. The pair-returning ones are looked at on the value side.

        **It must not branch on `isinstance(got, tuple)`** — torch hands back a named tuple
        while our side hands back something of its own, like `_MinMax`, which is not a tuple.
        Then it goes looking for `.shape` without unwrapping, and of the three only our side
        blows up with an `AttributeError`.
        """
        def run(L, f=fn):
            got = f(L)
            head = got if hasattr(got, "shape") else got[0]
            return str(tuple(int(n) for n in head.shape))

        return run

    def g(L):
        return L.tensor(grid.copy())

    def b(L):
        return L.tensor(flags.copy())

    # The ones that fold an axis. **The place where `keepdim` is given without `dim` is not
    # asked** — torch ignores `keepdim` there too.
    for name in ("sum", "mean", "amax", "amin", "prod", "logsumexp"):
        add(f"{name}(dim=1, keepdim)",
            shape_of(lambda L, n=name: getattr(g(L), n)(dim=1, keepdim=True)))
        add(f"{name}(dim=1) 값",
            lambda L, n=name: getattr(g(L), n)(dim=1, keepdim=True))
    # The ones that return indices — the type is int64, so the value is asked alongside.
    for name in ("argmax", "argmin"):
        add(f"{name}(dim=1, keepdim)",
            shape_of(lambda L, n=name: getattr(g(L), n)(dim=1, keepdim=True)))
        add(f"{name}(dim=1) 값",
            lambda L, n=name: getattr(g(L), n)(dim=1, keepdim=True))
    # The ones that return a pair. **Both have to keep the axis** — keep it on the value only
    # and code that gathers again by index goes wrong on the next line.
    for name in ("max", "min", "median"):
        add(f"{name}(dim=1, keepdim) 값",
            lambda L, n=name: getattr(g(L), n)(dim=1, keepdim=True)[0])
        add(f"{name}(dim=1, keepdim) 번호",
            lambda L, n=name: getattr(g(L), n)(dim=1, keepdim=True)[1])
        add(f"{name}(dim=1, keepdim) 모양",
            shape_of(lambda L, n=name: getattr(g(L), n)(dim=1, keepdim=True)))
    add("kthvalue(2, dim=1, keepdim) 값",
        lambda L: g(L).kthvalue(2, 1, True)[0])
    add("kthvalue(2, dim=1, keepdim) 모양",
        shape_of(lambda L: g(L).kthvalue(2, 1, True)))

    # **The boolean reductions had no axis at all.** `x.all(dim=1)` collapsing to the whole
    # gives a scalar, and that scalar broadcasts anywhere.
    for name in ("all", "any"):
        add(f"{name}(dim=1)", lambda L, n=name: getattr(b(L), n)(dim=1))
        add(f"{name}(dim=1, keepdim) 모양",
            shape_of(lambda L, n=name: getattr(b(L), n)(dim=1, keepdim=True)))
        add(f"{name}(dim=1, keepdim) 값",
            lambda L, n=name: getattr(b(L), n)(dim=1, keepdim=True))
        add(f"{name}() 전체", lambda L, n=name: getattr(b(L), n)())
    add("count_nonzero(dim=1)", lambda L: g(L).count_nonzero(dim=1))
    add("count_nonzero() 전체", lambda L: g(L).count_nonzero())

    # The gradient has to arrive with the axis alive too. A shape that does not fit blows up at
    # the leaf, or — worse — **spreads** under broadcasting and the values grow.
    def grad(name, fn):
        def run(L, f=fn, n=name):
            leaf = L.tensor(grid.copy(), requires_grad=True)
            f(L, leaf).sum().backward()
            return _grad_of(leaf, n)

        add(f"grad::{name}", run)

    grad("sum(keepdim)", lambda L, t: t.sum(dim=1, keepdim=True))
    grad("prod(keepdim)", lambda L, t: t.prod(dim=1, keepdim=True))
    grad("amax(keepdim)", lambda L, t: t.amax(dim=1, keepdim=True))
    grad("max(keepdim)", lambda L, t: t.max(dim=1, keepdim=True)[0])
    grad("median(keepdim)", lambda L, t: t.median(dim=1, keepdim=True)[0])
    grad("mean(keepdim)", lambda L, t: t.mean(dim=1, keepdim=True))

    # ── `dtype=` ──
    #
    # **One line of rule: it converts before folding.** Not after. Asking about the type alone
    # cannot tell the two orders apart, so **the value is asked too** — folding a float into an
    # integer is what parts them: the sum of `[1.7, −2.3, 0.9]` is −1 truncating first and 0
    # truncating after.
    slant = np.array([1.7, -2.3, 0.9], dtype=np.float32)
    counts = np.array([3, 1, 4], dtype=np.int64)
    marks = np.array([True, False, True])

    def dt(L, name):
        """The same meaning of type on both sides. **`bool` is the only name that parts.**

        The core's module-level `bool` is not a type but **`Tensor.bool` exposed as a function**
        (the type is kept as `bool_` so as not to shadow the Python builtin). Passed to `dtype=`,
        numpy stops with "cannot read a function as a type" — `_dtype_tensor` already dodges the
        same place this way.
        """
        if name != "bool":
            return getattr(L, name)
        return getattr(L, "bool_", None) or getattr(L, "bool")

    for src, arr in (("실수", slant), ("정수", counts), ("참거짓", marks)):
        for want in ("float32", "int64"):
            add(f"dtype::sum({src}→{want})",
                lambda L, a=arr, w=want: L.tensor(a.copy()).sum(dtype=dt(L, w)))
            add(f"dtype::sum({src}→{want}) 의 형",
                lambda L, a=arr, w=want: str(
                    L.tensor(a.copy()).sum(dtype=dt(L, w)).dtype))
            add(f"dtype::cumsum({src}→{want})",
                lambda L, a=arr, w=want: L.tensor(a.copy()).cumsum(0,
                                                                  dtype=dt(L, w)))
        # **`sum(dtype=bool)` works and `cumsum(dtype=bool)` does not** — not a rule, just a
        # kernel torch never built, so without asking separately it is invisible.
        add(f"dtype::sum({src}→참거짓)",
            lambda L, a=arr: L.tensor(a.copy()).sum(dtype=dt(L, "bool")))
        add(f"dtype::prod({src}→float32)",
            lambda L, a=arr: L.tensor(a.copy()).prod(dtype=dt(L, "float32")))
    add("dtype::mean(정수→float32)",
        lambda L: L.tensor(counts.copy()).mean(dtype=L.float32))
    add("dtype::mean(참거짓→float32)",
        lambda L: L.tensor(marks.copy()).mean(dtype=L.float32))
    add("dtype::sum(dim=1→float32)",
        lambda L: L.tensor(np.array([[1, 2], [3, 4]], dtype=np.int64)).sum(
            dim=1, dtype=L.float32))
    add("dtype::nansum(실수→int64)",
        lambda L: L.nansum(L.tensor(slant.copy()), dtype=L.int64))

    def refuses(name, body):
        def run(L, f=body):
            try:
                f(L)
                return "예외가 안 났다"
            except Exception as exc:                            # noqa: BLE001
                return type(exc).__name__

        add(name, run)

    # `dtype=` does not lift **every** refusal. Two remain as they were (measured).
    refuses("dtype::mean(→int64)는 거절",
            lambda L: L.tensor(slant.copy()).mean(dtype=L.int64))
    refuses("dtype::cumsum(→참거짓)은 거절",
            lambda L: L.tensor(counts.copy()).cumsum(0, dtype=dt(L, "bool")))
    refuses("dtype::cumprod(→참거짓)은 거절",
            lambda L: L.tensor(counts.copy()).cumprod(0, dtype=dt(L, "bool")))

    # **`to` really changes the type.** For a long time it looked only at the device string and
    # dropped the type quietly.
    add("dtype::to(float32) 의 형",
        lambda L: str(L.tensor(counts.copy()).to(L.float32).dtype))
    add("dtype::to(int64) 의 형",
        lambda L: str(L.tensor(slant.copy()).to(L.int64).dtype))
    add("dtype::to(int64) 의 값", lambda L: L.tensor(slant.copy()).to(L.int64))

    grad("sum(dtype=float32)",
         lambda L, t: t.sum(dim=1, keepdim=True, dtype=L.float32))

    # ── the remaining optional arguments ──
    #
    # Group C of the table another session handed over — "torch accepts an argument we do not" —
    # except that two here were **pretending to accept and throwing it away.** `dist(p)` ignored
    # `p` and always gave L2 (invisible, because the value was of a plausible magnitude), and
    # `div(rounding_mode)` got the value right while leaving **the type a float.** The rest were
    # places that stopped loudly with a `TypeError`.
    pair_a = np.array([1.0, 4.0, -2.0, 3.0], dtype=np.float32)
    pair_b = np.array([2.0, 3.0, 5.0, -1.0], dtype=np.float32)
    tops = np.array([7, -7, 8, -8], dtype=np.int64)
    bots = np.array([2, 2, 3, 3], dtype=np.int64)
    tally = np.array([1, 2, 2, 5], dtype=np.int64)
    spd = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    grid3 = np.arange(9, dtype=np.float32).reshape(3, 3)

    def A(L):
        return L.tensor(pair_a.copy())

    def B(L):
        return L.tensor(pair_b.copy())

    add("arg::add(alpha=2)", lambda L: A(L).add(B(L), alpha=2))
    add("arg::sub(alpha=2)", lambda L: A(L).sub(B(L), alpha=2))
    for mode in ("trunc", "floor"):
        add(f"arg::div(정수, {mode})",
            lambda L, m=mode: L.tensor(tops.copy()).div(L.tensor(bots.copy()),
                                                        rounding_mode=m))
        # **The type is asked too.** Asked by value alone, one left as a float passes — an
        # integer division's result has to be an integer for the indexing after it to take it.
        add(f"arg::div(정수, {mode}) 의 형",
            lambda L, m=mode: str(
                L.tensor(tops.copy()).div(L.tensor(bots.copy()),
                                          rounding_mode=m).dtype))
        add(f"arg::div(실수, {mode})",
            lambda L, m=mode: A(L).div(B(L), rounding_mode=m))
    # **`p` was ignored for a long time.** Asked at 2 only, this place is invisible forever.
    for p in (1, 3):
        add(f"arg::dist(p={p})", lambda L, k=p: A(L).dist(B(L), k))
    add("arg::cholesky(upper)",
        lambda L: L.tensor(spd.copy()).cholesky(upper=True))
    add("arg::diag(diagonal=1)", lambda L: L.tensor(grid3.copy()).diag(1))
    add("arg::diag(diagonal=-1)", lambda L: L.tensor(grid3.copy()).diag(-1))
    add("arg::diagflat(offset=1)", lambda L: A(L).diagflat(1))
    add("arg::diagflat(offset=-1)", lambda L: A(L).diagflat(-1))
    zero = np.array([0.0], dtype=np.float32)
    add("arg::diff(prepend)",
        lambda L: L.diff(A(L), prepend=L.tensor(zero.copy())))
    add("arg::diff(append)",
        lambda L: L.diff(A(L), append=L.tensor(zero.copy())))
    add("arg::bincount(weights)",
        lambda L: L.bincount(L.tensor(tally.copy()), weights=A(L)))
    add("arg::bincount(weights) 의 형",
        lambda L: str(L.bincount(L.tensor(tally.copy()), weights=A(L)).dtype))
    add("arg::bincount(minlength=8)",
        lambda L: L.bincount(L.tensor(tally.copy()), minlength=8))

    def verdict(fn):
        return lambda L, f=fn: str(f(L))

    nan_pair = np.array([1.0, float("nan")], dtype=np.float32)
    add("arg::allclose(equal_nan=False)",
        verdict(lambda L: L.allclose(L.tensor(nan_pair.copy()),
                                     L.tensor(nan_pair.copy()))))
    add("arg::allclose(equal_nan=True)",
        verdict(lambda L: L.allclose(L.tensor(nan_pair.copy()),
                                     L.tensor(nan_pair.copy()),
                                     equal_nan=True)))
    return cases


def stat_cases(inp=None):
    """Statistics. **A random value cannot be frozen, but its limits are deterministic.**

    ## How the four random ones are asked

    The golden answers cannot freeze the values of `normal`, `bernoulli`, `poisson` and
    `binomial` — torch's random stream differs from ours and there is no way to make them agree.
    So **the deterministic corners** are asked:

    - `std=0` gives the mean exactly (measured)
    - `p=0` gives all zeros, `p=1` gives all ones
    - `poisson(0)` gives all zeros
    - the rest by shape only

    "It is random so it cannot be asked" and "it is not asked" are different. Without the
    limits, a `bernoulli` that never looks at the probability at all passes.

    ## Where the rest part

    - **Out-of-range values are discarded.** `histc` and `histogram` do not herd values outside
      `min`/`max` into the end bins (measured). Measured on data that is entirely in range, that
      rule does not show.
    - **When `min == max` the data's own range is used.** The default is `0, 0`, so that branch
      is the default one.
    - **The last bin is closed on the right.** The maximum lands in the last bin.
    - **`mode` breaks a tie towards the smaller value, and the index is that value's last
      occurrence** (measured: `[4,4,5,5]` gives value 4, index 1). With no tie, that rule does
      not show.
    - **`nanmedian` takes the lower of an even count** — it does not average. And `median`
      returns NaN if even one NaN is present — asking the two side by side is what parts them.
    - **`gradient`'s `edge_order`.** At 1 the ends are one-sided differences and at 2 they are
      quadratic. Feed it `x²` and only 2 gives the exact derivative.
    - **`histogram(density)` divides by the bin width** — with the edges given by hand the
      widths differ per bin, so measured on uniform bins that division does not show.

    ## Three keep the name and refuse

    `stft` and `istft` have **no complex dtype.** torch's default is complex now and the real
    `(…, 2)` route is slated for removal, so imitating that shape teaches one about to disappear.
    `hash_tensor` has no uint64 either, and no specification of which hash — putting a name on
    something whose value cannot be matched creates code that trusts that value.
    """
    x = np.array([0.5, 2.0, 2.0, 3.5, 1.0, 4.0, 2.0], dtype=np.float32)
    w = np.array([1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0], dtype=np.float32)
    # **There is a tie in it** — without one, `mode`'s rule does not show.
    tie = np.array([[1.0, 2.0, 2.0, 3.0], [4.0, 4.0, 5.0, 5.0]], dtype=np.float32)
    holes = np.array([[1.0, np.nan, 3.0, 5.0], [2.0, 4.0, np.nan, np.nan]],
                     dtype=np.float32)
    # It is `x²` — the place where `edge_order=2` becomes exact.
    line = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    mat = np.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0], [64.0, 128.0, 256.0]],
                   dtype=np.float32)
    pts = np.array([[0.5, 1.0], [1.5, 1.5], [2.5, 0.5], [0.2, 2.5]],
                   dtype=np.float32)
    sparse = np.array([0.0, 3.0, 0.0, 5.0, 0.0], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((STAT_PREFIX + name, fn))

    # ── histograms ──
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
    # **The bin widths differ** — only here is it visible whether `density` divides by a
    # different value per bin.
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
    # **An even count takes the lower** — averaging parts here.
    add("nanmedian(짝수 개)",
        lambda L: L.nanmedian(L.tensor(np.array([1.0, 2.0, 3.0, 4.0],
                                                dtype=np.float32))))
    # `median` is NaN on a single NaN — placing them side by side is what shows what
    # `nanmedian` is.
    #
    # **A predicate is frozen rather than the value.** The golden comparison is `allclose` and it
    # runs without `equal_nan`, so **NaN differs from itself** — a case whose answer is NaN
    # cannot be frozen by this harness at all. Turning that place into "is it NaN" makes it a
    # string comparison, and what was meant to be asked (that the two differ) survives intact.
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

    # ── nonzero_static ──
    #
    # **Short, it pads; over, it cuts.** Measured at exactly the right size, neither branch shows.
    for size in (1, 2, 5):
        add(f"nonzero_static(size={size})",
            lambda L, n=size: L.nonzero_static(L.tensor(sparse), size=n))
    add("nonzero_static(fill=-9)",
        lambda L: L.nonzero_static(L.tensor(sparse), size=5, fill_value=-9))

    # ── nonzero(as_tuple) ──
    #
    # **The tuple form is what indexing takes** — one 1-D tensor per axis rather
    # than one (count, rank) table. `torch.nonzero` reads as "no signature found
    # for builtin", so the signature axis had never compared it and the argument
    # was simply absent on both sides.
    add("nonzero(as_tuple)/행",
        lambda L: L.nonzero(L.tensor(sparse), as_tuple=True)[0])
    add("nonzero 표",
        lambda L: L.nonzero(L.tensor(sparse)))

    # ── the four random ones — the deterministic limits only ──
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
    # The value cannot be asked, but **the shape is** — without even that it is the same as
    # having a name and nothing else.
    add("normal(size) 모양",
        lambda L: str(tuple(L.normal(0.0, 1.0, (2, 3)).shape)))
    add("bernoulli 모양", lambda L: str(tuple(L.bernoulli(L.zeros(2, 3)).shape)))
    return cases


TOPLIN_PREFIX = "toplin::"


def top_linalg_cases(inp=None):
    """Top-level linear algebra. The ones that are **the same computation called differently**
    from the `linalg` side.

    ## The argument order is reversed

    torch kept the old names at the top level, and most of them take **the right-hand side
    first** — `lu_solve(b, LU, piv)` against `linalg.lu_solve(LU, piv, b)`. `triangular_solve`
    also takes `b` first and **defaults `upper` to true** (on the `linalg` side it is required).
    Carry the positions over wrongly and it solves a different triangle while producing a
    plausible value.

    ## What has to be asked for it to part

    - **`orgqr` and `ormqr` use different Qs.** The first is Q cut to `m×k` and the second is
      the uncut `m×m` — because a reflector is a map on `Rᵐ`. It shows only when asked with **a
      tall matrix.** Measured on a square the two are equal (that is how it was caught).
    - **`unitriangular` ignores the diagonal and treats it as 1.** Measured on a matrix whose
      diagonal is already 1, the flag does nothing.
    - **`lu_unpack` returns an empty tensor when switched off** — not `None` (measured). It
      surfaces only by asking about the shape.
    - **`lobpcg`'s `largest` decides the order too** — true gives largest first, false smallest
      first (measured). Measured at `k=1` there is no order.
    - **`svd_lowrank`'s answer is only stable on an exactly low-rank input.** torch projects
      randomly, and once the rank exceeds `q` the singular values move by about 0.5 with the
      seed (measured: 0.54 between two seeds). At rank `q` or below it is within 7e-7 — that is
      the only place the golden answers can ask about, so the input is built as `(8,3)@(3,5)`,
      **exactly rank 3.**
    - **`pca_lowrank(center=False)` is the same thing as `svd_lowrank`** (measured). Centring is
      the whole difference, so measured at true only that branch is invisible.

    ## The eigenvectors are not asked

    Their sign is arbitrary — the same eigenpair can come out as `-v`, and that is not a
    divergence. The eigenvalues alone are frozen.
    """
    spd = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]],
                   dtype=np.float32)
    gen = np.array([[4.0, 3.0, 2.0], [1.0, 5.0, 3.0], [2.0, 1.0, 6.0]],
                   dtype=np.float32)
    # **The diagonal is not 1** — needed to see what `unitriangular` actually does.
    tri = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [4.0, 2.0, 5.0]],
                   dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    # **It is tall** — the only shape where `orgqr` and `ormqr` part.
    tall = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    side = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    weight = np.array([[1.0, 2.0], [0.5, 3.0], [2.0, 1.0]], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((TOPLIN_PREFIX + name, fn))

    def chol(L, upper):
        low = L.linalg.cholesky(L.tensor(spd))
        return low.transpose(0, 1) if upper else low

    # ── Cholesky ──
    for upper in (False, True):
        add(f"cholesky_solve(upper={upper})",
            lambda L, u=upper: L.cholesky_solve(L.tensor(rhs), chol(L, u),
                                                upper=u))
        add(f"cholesky_inverse(upper={upper})",
            lambda L, u=upper: L.cholesky_inverse(chol(L, u), upper=u))

    def cholesky_solve_grad(L):
        """**It has to flow towards the factor too.** Flowing to `b` alone leaves the forward pass right and parts here."""
        x = L.tensor(spd.copy(), requires_grad=True)
        out = L.cholesky_solve(L.tensor(rhs), L.linalg.cholesky(x))
        (out * L.tensor(weight)).sum().backward()
        return _grad_of(x, "cholesky_solve")

    cases.append((TOPLIN_PREFIX + "grad::cholesky_solve", cholesky_solve_grad))

    # ── triangular ──
    for upper in (False, True):
        for trans in (False, True):
            for unit in (False, True):
                add(f"triangular_solve(u={upper},t={trans},unit={unit})",
                    lambda L, u=upper, t=trans, n=unit: L.triangular_solve(
                        L.tensor(rhs), L.tensor(tri), upper=u, transpose=t,
                        unitriangular=n).solution)
    # **The second slot is a copy of the coefficients** — it looks useless and torch gives it anyway.
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
                # **Switched off it is an empty tensor.** Without asking about the shape that does not show.
                add(f"lu_unpack(data={data_flag}, piv={piv_flag}) 의 {name} 모양",
                    lambda L, d=data_flag, p=piv_flag, s=slot: str(tuple(
                        L.lu_unpack(*L.lu(L.tensor(gen)), unpack_data=d,
                                    unpack_pivots=p)[s].shape)))

    # ── reflectors ──
    add("orgqr", lambda L: L.orgqr(*L.geqrf(L.tensor(tall))))
    add("orgqr 은 자른 Q 다 (linalg.qr 의 Q 와 같다)",
        lambda L: L.linalg.qr(L.tensor(tall))[0])
    for left in (True, False):
        for trans in (True, False):
            add(f"ormqr(left={left}, transpose={trans})",
                lambda L, lf=left, tr=trans: L.ormqr(
                    *L.geqrf(L.tensor(tall)),
                    L.tensor(side if lf else side.T), lf, tr))

    # ── eigenpairs and low rank ──
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
    # **Exactly rank 3** — over it, torch moves with the seed and there is nothing to freeze.
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


CONST_PREFIX = "const::"


def constant_cases(inp=None):
    """The **numeric constants** torch keeps at the top level. `torch.pi`, `inf`, `nan`, `e`, `newaxis`.

    ## Why this place arrived late

    `tests/torch_gap.py` measures the surface, and it **counts only the names that are
    `callable`.** These five are values rather than things to call, so they entered neither the
    numerator nor the denominator, and that is how "547 of torch's 691 · 0 to review" came out
    **with these missing entirely.** A place the measuring stick cannot see stays invisible
    however much is measured.

    It surfaced only after `--props` was added to the tool to count the non-callable names
    separately.

    ## These are names the textbook uses

        torch.clamp(x, min=-torch.inf)      x[:, torch.newaxis]      torch.pi

    The values point straight at numpy's own constants, so there is nowhere for them to part.
    They are asked anyway because the question is **whether they exist** — absent, it is an
    `AttributeError`, which is the same screen as a typo.
    """
    inp = golden_inputs() if inp is None else inp
    cases = []

    # The harness takes a tensor or characters — a bare `float` has no `detach` and is refused.
    cases.append((CONST_PREFIX + "pi", lambda L: f"{L.pi:.12f}"))
    cases.append((CONST_PREFIX + "e", lambda L: f"{L.e:.12f}"))
    # Frozen as numbers, `inf` and `nan` blur the comparison — they are asked as characters.
    cases.append((CONST_PREFIX + "inf/nan/newaxis",
                  lambda L: f"{L.inf} {-L.inf} {L.nan} {L.newaxis}"))

    # **Where they are used is asked too.** A name that exists and cannot be used does not exist.
    cases.append((CONST_PREFIX + "newaxis 가 축을 늘린다",
                  lambda L: str(tuple(L.tensor(inp["train_x"])[:, L.newaxis].shape))))
    cases.append((CONST_PREFIX + "inf 가 비교에 쓰인다",
                  lambda L: (L.tensor(inp["train_x"]) < L.inf).sum()))

    # **A name existing does not mean it works everywhere.**
    #
    # `clamp(min=-inf)` simply works in the core (it is numpy). The browser side has to **bake
    # that value into the shader as a constant**, and WGSL forbids infinity and NaN literals —
    # a limit of the language rather than the hardware, so there is nowhere to go round it.
    # Refusal is therefore the right answer, and asking about the value would part forever.
    #
    # This case was written as a value first and the binding came back red. Adding the five
    # constants very nearly became "you can write it like torch now".
    cases.append((CONST_PREFIX + "-inf 를 상수로 굽는 것은 브라우저가 거절한다",
                  _as_expected(lambda L: L.clamp(L.tensor(inp["train_x"]),
                                                 min=-L.inf))))

    return cases


CACHE_PREFIX = "cache::"


def scalar_cache_cases(inp=None):
    """**Does a size-1 parameter dirty the global constants?**

    ## Why this place exists

    The sister library (borch.ts) has `Tensor.full` **cache single-element tensors by value.**
    Ask for the same value twice and the same buffer comes back, and `zeros` and `ones` go
    through that door too. It is fast — and **anything edited in place that inherits that buffer
    changes the global constant outright.**

    That really happened in the optimizer state (the sister session caught it), and the layers
    have the same door: `nn.PReLU()`'s default weight is a size-1 `0.25`, and `BatchNorm(1)`'s
    running statistics are a size-1 `0` and `1`. All three are edited **in place** by training.

    ## Making the case actually step on that place

    Looking at the parameter's value alone, a dirtied one still reads correctly there — the
    edited value is the answer. **A constant created fresh after one step** is what has to be
    looked at. So the order is:

    1. build a size-1 parameter (this is where the cached buffer is inherited)
    2. train one step (edited in place — if there is contamination it happens here)
    3. **then** build `zeros`, `ones` and `full(0.25)` fresh and look at their values

    Without step 3 this case is green and measures nothing. That is exactly why the optimizer
    case had to be rewritten twice.

    torch has no such cache, so the answer is always a clean constant — that is the expected value.
    """
    cases = []

    def add(name, fn):
        cases.append((CACHE_PREFIX + name, fn))

    def fresh(L):
        """Three global constants built **fresh.** If they were dirtied, different values come out here."""
        return L.cat([L.zeros(1), L.ones(1), L.full((1,), 0.25)])

    def prelu_then_constants(L):
        m = L.nn.PReLU()
        opt = L.optim.SGD(m.parameters(), lr=0.5)
        opt.zero_grad()
        # **There has to be a negative slot for a gradient to flow** — PReLU's weight attaches on the negative side only.
        m(L.tensor(np.array([[-2.0, 1.0]], dtype=np.float32))).sum().backward()
        opt.step()
        return fresh(L)

    add("PReLU 한 스텝 뒤의 상수", prelu_then_constants)
    # The parameter itself is frozen too — only here is it visible that training really moved.
    # If it did not, the case above passed a place it never stepped on.
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
    """The addmm family. All eight are `β·input + α·(some product)`.

    ## `beta=0` is this group's point

    **The value is not looked at and it stays in the graph.** Both are required, and the
    requirements pull in opposite directions —

    - Written as `input * 0`, a NaN in the input makes the result NaN. torch is unaffected.
    - Taken out of the graph instead, `input.grad` is not 0 but **absent.** torch gives 0
      (measured). Taken out, `backward()` stops with "not requires_grad".

    With an ordinary input **neither** is visible — the first needs a NaN and the second needs
    the gradient to be asked. So both are asked.

    ## Where the rest part

    - **`beta` and `alpha` both have to be other than 1** for which one multiplies what to show.
      At 1 and 1, writing them in the wrong places gives the same answer.
    - **More than one batch** is what parts `addbmm` (which sums) from `baddbmm` (which keeps).
      At batch 1 the two functions look the same.
    - **`input` has to be smaller than the result** for the spreading to be visible. torch takes
      a `(4,)` and even a scalar (measured).
    - `addcmul` and `addcdiv` have **no `beta`** — `input`'s coefficient is always 1. There is
      only `value`, and it attaches to the product side.
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
    # **Uneven weights.** All ones and the differing share per slot cancels out of sight.
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
    # **A NaN is what reveals it was written as `input * 0`.**
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
    # **Asking the gradient is what reveals it was taken out of the graph.** Taken out, it stops here.
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

    # ── in place ──
    #
    # **They exist as methods only** — there is no top-level name `torch.addmm_` (measured).
    # The one exception is `addmv_`, and that is asked in method form alongside anyway.
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
    """The **writing** side of indexing. The reading side (`gather`, `index_select`) was already there.

    `gather` existed and its opposite, `scatter`, did not. With one side only, you can take
    values out and cannot put them back, and code that builds an embedding or a one-hot by hand
    meets that place immediately.

    **Repeated indices are the point.** `scatter` keeps whatever was written last and
    `scatter_add` adds — measured on non-repeating indices only, the two look like the same
    function.
    """
    inp = golden_inputs() if inp is None else inp
    x2 = inp["x2"]                                       # (3, 4)
    src = (x2 * 10).astype(np.float32)
    # Repeated indices. 0 appears twice — where `scatter` and `scatter_add` part.
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

    # The gradient is looked at too. **It has to flow only to where it was written.**
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

    # `take` picks **from a flattened view** — it has no notion of an axis.
    add("take", lambda L: L.take(L.tensor(x2), L.tensor(flat_idx)))
    add("take_along_dim",
        lambda L: L.take_along_dim(L.tensor(x2), L.tensor(dup), dim=1))

    # Finds a position within a sorted sequence. **`right` decides which side of a tie.**
    line = np.array([1., 3., 5., 7.], dtype=np.float32)
    want = np.array([0., 3., 6., 9.], dtype=np.float32)
    add("searchsorted", lambda L: L.searchsorted(L.tensor(line), L.tensor(want)))
    add("searchsorted(right)",
        lambda L: L.searchsorted(L.tensor(line), L.tensor(want), right=True))
    add("bucketize", lambda L: L.bucketize(L.tensor(want), L.tensor(line)))
    # **The same thing is taken under two names** — the boolean `right` and the string `side`.
    # The table asked about `right` only, and `side` went into `**kw` on both the core and the
    # binding and was quietly discarded. `side="right"` gave the left answer, and being off by
    # exactly one position it looks plausible. `bucketize` was right from the start — one
    # computation under two names, one of which was right.
    for tag in ("left", "right"):
        add(f"searchsorted(side={tag})",
            lambda L, s=tag: L.searchsorted(L.tensor(line), L.tensor(want), side=s))
    # torch takes both together too — **only when they agree.**
    add("searchsorted(side=right, right=True)",
        lambda L: L.searchsorted(L.tensor(line), L.tensor(want),
                                 side="right", right=True))

    # **Both ends of the binary search.** Asked in the middle only, a wrong initial `lo` or `hi`
    # still gives the right answer — a value below or above every boundary has to give 0 and n,
    # and with a single boundary it passes the place where the loop never runs.
    edge = np.array([2., 4.], dtype=np.float32)
    span = np.array([0., 1., 2., 3., 4., 5.], dtype=np.float32)
    add("searchsorted(끝 밖)",
        lambda L: L.searchsorted(L.tensor(edge), L.tensor(span)))
    one = np.array([3.], dtype=np.float32)
    trio = np.array([1., 3., 5.], dtype=np.float32)
    add("searchsorted(경계 하나)",
        lambda L: L.searchsorted(L.tensor(one), L.tensor(trio), right=True))

    def refuses(name, fragment, fn):
        def run(L, f=fn, frag=fragment):
            try:
                f(L)
            except Exception as exc:                            # noqa: BLE001
                return frag if frag in str(exc) else f"다른 문구 <{exc}>"
            return "안 던졌다"
        cases.append((INDEX_PREFIX + f"거절::{name}", run))

    refuses("side 와 right 가 반대", "opposites",
            lambda L: L.searchsorted(L.tensor(line), L.tensor(want),
                                     side="left", right=True))
    refuses("side 가 셋째 값", "can only be 'left' or 'right'",
            lambda L: L.searchsorted(L.tensor(line), L.tensor(want), side="both"))
    return cases


NEWFN_PREFIX = "newfn::"


def new_function_cases(inp=None):
    """A group of **genuinely new functionality** that torch has and this did not.

    The two groups before it were missing names only — a spelling attached to what an operator
    already did. Here the computation itself was absent.

    The selection criterion is **whether the textbook calls it.** Far more code stops for want
    of `torch.meshgrid` or `torch.randn_like` than for want of `torch.igammac`.
    """
    inp = golden_inputs() if inp is None else inp
    x1, x2 = inp["x1"], inp["x2"]
    pos = inp["xp"]
    withnan = np.array([1., np.nan, -np.inf, np.inf, 3.], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((NEWFN_PREFIX + name, fn))

    # ── `*_like` — borrowing the shape only. **The shape is frozen as the answer, not the value.** ──
    #
    # The random ones cannot have equal values, so the shape goes out as a string. The ones whose
    # value is determined, like `zeros_like`, are asked by value.
    for name in ("empty_like", "rand_like", "randn_like"):
        add(f"{name}/모양",
            lambda L, n=name: " ".join(str(int(v)) for v in
                                       getattr(L, n)(L.tensor(x2)).shape))
    add("randint_like/모양",
        lambda L: " ".join(str(int(v)) for v in L.randint_like(L.tensor(x2), 5).shape))

    add("logspace", lambda L: L.logspace(0.0, 2.0, 5))
    add("scalar_tensor", lambda L: L.scalar_tensor(2.5))

    # ── meshgrid. **Without `indexing`, torch warns and goes with `ij`.** ──
    add("meshgrid/0", lambda L: L.meshgrid(L.tensor(x1[:3]), L.tensor(x1[:2]),
                                           indexing="ij")[0])
    add("meshgrid/1", lambda L: L.meshgrid(L.tensor(x1[:3]), L.tensor(x1[:2]),
                                           indexing="ij")[1])
    add("meshgrid(xy)", lambda L: L.meshgrid(L.tensor(x1[:3]), L.tensor(x1[:2]),
                                             indexing="xy")[0])

    # ── element-wise. ──
    add("lerp", lambda L: L.lerp(L.tensor(x1), L.tensor(x1 * 2), 0.25))
    add("nan_to_num", lambda L: L.nan_to_num(L.tensor(withnan)))
    add("nan_to_num(값 지정)",
        lambda L: L.nan_to_num(L.tensor(withnan), nan=0.5, posinf=9.0, neginf=-9.0))
    add("isclose", lambda L: L.isclose(L.tensor(x1), L.tensor(x1 + 1e-9)))
    add("isreal", lambda L: L.isreal(L.tensor(withnan)))
    add("isposinf", lambda L: L.isposinf(L.tensor(withnan)))
    add("isneginf", lambda L: L.isneginf(L.tensor(withnan)))
    # **`fmax` and `fmin` skip NaN** — `maximum` carries it out.
    add("fmax(NaN 건너뜀)",
        lambda L: L.fmax(L.tensor(withnan), L.tensor(np.zeros(5, dtype=np.float32))))
    add("fmin(NaN 건너뜀)",
        lambda L: L.fmin(L.tensor(withnan), L.tensor(np.zeros(5, dtype=np.float32))))
    add("float_power", lambda L: L.float_power(L.tensor(pos), 2.0))
    # **`pow` with a tensor exponent**, which is the half `powScalar` cannot do and
    # the reason borch.ts's `pow` is not that method renamed: the elementwise kernel
    # has a two-sided backward and the scalar one does not.
    add("pow(텐서 지수)",
        lambda L: L.tensor(pos).pow(L.tensor(np.array([2., 3., 0.5, 1., 2., 0.],
                                                      dtype=np.float32))))
    add("pow(수 지수)", lambda L: L.tensor(pos).pow(3))
    add("expand_as", lambda L: L.tensor(np.array([[1.], [2.]], dtype=np.float32))
        .expand_as(L.tensor(np.zeros((2, 3), dtype=np.float32))))
    add("reshape_as", lambda L: L.tensor(pos)
        .reshape_as(L.tensor(np.zeros((2, 3), dtype=np.float32))))
    add("view_as", lambda L: L.tensor(pos)
        .view_as(L.tensor(np.zeros((3, 2), dtype=np.float32))))
    add("logical_xor",
        lambda L: L.logical_xor(L.tensor(np.array([1., 0., 1., 0.], dtype=np.float32)),
                                L.tensor(np.array([1., 1., 0., 0.], dtype=np.float32))))

    # `isin` — is the element in that list. It resolves into a single broadcast.
    add("isin", lambda L: L.isin(L.tensor(np.array([1., 2., 3., 4.], dtype=np.float32)),
                                 L.tensor(np.array([2., 4.], dtype=np.float32))))

    # ── reductions returning a pair. **Ask about one and the other passes while wrong.** ──
    add("var_mean/분산", lambda L: L.var_mean(L.tensor(x2))[0])
    add("var_mean/평균", lambda L: L.var_mean(L.tensor(x2))[1])
    add("std_mean/표준편차", lambda L: L.std_mean(L.tensor(x2))[0])

    # ── the multiplication family. ──
    add("inner", lambda L: L.inner(L.tensor(x2), L.tensor(x2)))
    add("vdot", lambda L: L.vdot(L.tensor(x1), L.tensor(x1)))
    add("kron", lambda L: L.kron(L.tensor(x1[:2]), L.tensor(x1[2:4])))
    add("cross", lambda L: L.cross(L.tensor(x1[:3].reshape(1, 3)),
                                   L.tensor(x1[3:6].reshape(1, 3)), dim=1))
    return cases


POOL_PREFIX = "pool::"


def pool_cases(inp=None):
    """Pooling's remaining dimensions and remaining kinds.

    There were only `max_pool1d/2d/3d`, `avg_pool2d` and `adaptive_avg_pool2d`. One dimension
    being present reads as the other two being present too, and when that expectation is wrong,
    code handling a 1-D signal or a 3-D volume stops halfway.

    **The adaptive ones solve the window size backwards from the input.** When it does not divide
    evenly, which slots are divided how is the rule, and if that rule parts from torch's the
    values are quietly different — so both the dividing and the non-dividing case are asked.
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

    # ── average pooling in 1-D and 3-D. Only 2-D existed. ──
    add("F.avg_pool1d", lambda L, x: L.nn.functional.avg_pool1d(x, 2), seq)
    add("F.avg_pool3d", lambda L, x: L.nn.functional.avg_pool3d(x, 2), vol)
    cases.append((POOL_PREFIX + "nn.AvgPool1d",
                  lambda L: L.nn.AvgPool1d(2)(L.tensor(seq))))
    cases.append((POOL_PREFIX + "nn.AvgPool3d",
                  lambda L: L.nn.AvgPool3d(2)(L.tensor(vol))))

    # ── adaptive average. **Both the dividing and the non-dividing case.** ──
    add("F.adaptive_avg_pool1d(4)",
        lambda L, x: L.nn.functional.adaptive_avg_pool1d(x, 4), seq)
    add("F.adaptive_avg_pool1d(3)",                       # 8 into 3 — does not divide
        lambda L, x: L.nn.functional.adaptive_avg_pool1d(x, 3), seq)
    add("F.adaptive_avg_pool3d",
        lambda L, x: L.nn.functional.adaptive_avg_pool3d(x, 2), vol)
    cases.append((POOL_PREFIX + "nn.AdaptiveAvgPool1d",
                  lambda L: L.nn.AdaptiveAvgPool1d(4)(L.tensor(seq))))
    cases.append((POOL_PREFIX + "nn.AdaptiveAvgPool3d",
                  lambda L: L.nn.AdaptiveAvgPool3d(2)(L.tensor(vol))))

    # ── adaptive max. **Its tie rule differs** from average's — it picks one. ──
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

    # ── LP pooling. The `p`-th root of the mean of `p`-th powers — p=1 is the sum, p=∞ nears the max. ──
    add("F.lp_pool1d(p=2)", lambda L, x: L.nn.functional.lp_pool1d(x, 2, 2), seq)
    add("F.lp_pool2d(p=2)", lambda L, x: L.nn.functional.lp_pool2d(x, 2, 2), img)
    add("F.lp_pool2d(p=1)", lambda L, x: L.nn.functional.lp_pool2d(x, 1, 2), img)
    cases.append((POOL_PREFIX + "nn.LPPool2d",
                  lambda L: L.nn.LPPool2d(2, 2)(L.tensor(img))))

    # ── the pooling arguments that had a seat and no case. ──
    #
    # `AvgPool1d/3d` refused `padding`, `count_include_pad` and `divisor_override`
    # from inside until they were implemented, and `LPPool` refused `ceil_mode`. A
    # refusal and an implementation look identical to the name axis, which counts
    # declared names — so the seats existing is not evidence that anything works,
    # and until these lines there was nothing that asked.
    #
    # **Every one of the six changes the answer**, which is why they are worth
    # freezing and how they were chosen. Four change the *shape* (padding and the
    # three `ceil_mode`s), so a wrong one cannot hide behind a tolerance; the other
    # two change the divisor, where a wrong one is a plausible number.
    #
    # `count_include_pad` needs a padding to have anything to include, and
    # `divisor_override` needs overlapping windows for the difference to show,
    # which is what the stride of 1 is doing.
    #
    # **The `AvgPool3d` ceiling case is a 3 and not a 2 because the volume is 4³.**
    # Written as a 2 it divides evenly, `ceil_mode` changes nothing, and the case
    # freezes an answer its neighbour already froze. It was written that way, and
    # what caught it was not the run — the run was green — but the two guards that
    # ask whether a case named after an argument answers differently from the case
    # beside it. **The argument had been checked against a hand-made 3³ array
    # rather than against the fixture the case runs on**, so the check answered a
    # narrower question than its name: true of that array, false here.
    cases.append((POOL_PREFIX + "nn.AvgPool1d(테두리 채움)",
                  lambda L: L.nn.AvgPool1d(2, 2, 1)(L.tensor(seq))))
    cases.append((POOL_PREFIX + "nn.AvgPool1d(가장자리 빼기)",
                  lambda L: L.nn.AvgPool1d(2, 2, 1, False, False)(L.tensor(seq))))
    cases.append((POOL_PREFIX + "nn.AvgPool3d(테두리 채움)",
                  lambda L: L.nn.AvgPool3d(2, 2, 1)(L.tensor(vol))))
    cases.append((POOL_PREFIX + "nn.AvgPool3d(나눗수 지정)",
                  lambda L: L.nn.AvgPool3d(2, 1, 0, False, True, 4)(L.tensor(vol))))
    cases.append((POOL_PREFIX + "nn.AvgPool3d(올림)",
                  lambda L: L.nn.AvgPool3d(3, 3, 0, True)(L.tensor(vol))))
    cases.append((POOL_PREFIX + "nn.LPPool1d(올림)",
                  lambda L: L.nn.LPPool1d(2, 3, 3, True)(L.tensor(seq))))
    cases.append((POOL_PREFIX + "nn.LPPool2d(올림)",
                  lambda L: L.nn.LPPool2d(2, 3, 3, True)(L.tensor(img))))

    # **The backward divides by the same three things and none of the seven above
    # reaches it.** A cell in a short edge window was divided by fewer cells going
    # forward, so it has to receive proportionally more coming back; written with the
    # kernel volume everywhere, the forward is right and the gradient is quietly
    # wrong at every edge — the shape is right, the total is not, and no forward
    # comparison can see it. One case per divisor rule.
    def pool_grad(L, make, arr):
        x = L.tensor(arr, requires_grad=True)
        out = make(L)(x)
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(x, "pool")

    for name, make, arr in (
        ("nn.AvgPool1d(테두리 채움)", lambda L: L.nn.AvgPool1d(2, 2, 1), seq),
        ("nn.AvgPool1d(가장자리 빼기)", lambda L: L.nn.AvgPool1d(2, 2, 1, False, False), seq),
        ("nn.AvgPool3d(나눗수 지정)",
         lambda L: L.nn.AvgPool3d(2, 1, 0, False, True, 4), vol),
        ("nn.AvgPool3d(올림)", lambda L: L.nn.AvgPool3d(3, 3, 0, True), vol),
    ):
        cases.append((POOL_PREFIX + f"grad::{name}",
                      lambda L, m=make, a=arr: pool_grad(L, m, a)))
    return cases


MODFN_PREFIX = "modfn::"


def module_function_cases(inp=None):
    """The **module-function form**, as in `torch.sum(x)`.

    torch gives nearly everything under two names — `x.sum()` and `torch.sum(x)`. This table
    asked in method form only for a long time, and so did not see that the module functions were
    missing wholesale. Writing the `reduce::sum(dim)` case ran into
    `module 'borch' has no attribute 'sum'`, and counting then showed **fifty** such names.

    What is asked here is not whether the value is right — the method-side cases already ask that
    — but **whether the name is in that place.** So each line confirms it by value.
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
    # **The module form takes axes as a tuple.** The method takes them spread out and this does
    # not — `torch.permute(x, 1, 0)` is a `TypeError`.
    add("permute", lambda L: L.permute(L.tensor(x2), (1, 0)))
    add("transpose", lambda L: L.transpose(L.tensor(x2), 0, 1))
    add("squeeze", lambda L: L.squeeze(L.tensor(x1).reshape(1, 6, 1)))

    # **`max` and `min` return a pair when given an axis.** Taken out by position it works
    # whatever the two sides call them.
    add("max", lambda L: L.max(L.tensor(x2)))
    add("max(dim)/값", lambda L: L.max(L.tensor(x2), dim=1)[0])
    add("min(dim)/번호", lambda L: L.min(L.tensor(x2), dim=1)[1])

    # The in-place operations have module names too. **Whether the original changes** is what is looked at.
    def inplace(L):
        t = L.tensor(x1.copy())
        L.relu_(t)
        return t

    add("relu_(원본이 바뀐다)", inplace)

    # ── the ones torch gives under **a second name.** ──
    #
    # `a + b` worked and `torch.add(a, b)` did not exist. A place that fails for want of a name
    # rather than a computation, and only a value comparison reveals whether such a place exists.
    a2 = x2
    b2 = (x2 * 0.5 + 1.0).astype(np.float32)
    add("add", lambda L: L.add(L.tensor(a2), L.tensor(b2)))
    # **`alpha` does not exist on the operator** — kept as an alias, this place drops out quietly.
    add("add(alpha)", lambda L: L.add(L.tensor(a2), L.tensor(b2), alpha=2.0))
    add("sub", lambda L: L.sub(L.tensor(a2), L.tensor(b2)))
    add("mul", lambda L: L.mul(L.tensor(a2), L.tensor(b2)))
    add("div", lambda L: L.div(L.tensor(a2), L.tensor(b2)))
    add("div(floor)",
        lambda L: L.div(L.tensor(a2), L.tensor(b2), rounding_mode="floor"))
    add("rsub", lambda L: L.rsub(L.tensor(a2), L.tensor(b2)))
    # **`remainder` and `fmod` part on negatives** — the sign follows opposite sides.
    neg = np.array([[-5., -3., 3., 5.]], dtype=np.float32)
    add("remainder(음수)",
        lambda L: L.remainder(L.tensor(neg), L.tensor(np.float32(3.0))))
    add("fmod(음수)", lambda L: L.fmod(L.tensor(neg), L.tensor(np.float32(3.0))))
    add("floor_divide(음수)",
        lambda L: L.floor_divide(L.tensor(neg), L.tensor(np.float32(3.0))))

    for name in ("greater", "greater_equal", "less", "less_equal", "not_equal"):
        add(name, lambda L, n=name: getattr(L, n)(L.tensor(a2), L.tensor(b2)))

    # The four stacking names. **The rule parts between 1-D and 2-D.**
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
    """`scaled_dot_product_attention`. **The name today's transformer code calls directly.**

    `MultiheadAttention` was already there and the function underneath it was not. Code that
    writes attention by hand rather than using the layer has become common, and that code calls
    this name.

    Four things are asked — the bare form, an additive mask, a causal mask, and the gradients of
    all three. **That the mask adds rather than multiplies** is the commonest misunderstanding.
    It adds `-inf` so that softmax gives 0; it does not multiply by 0 — multiplying comes after
    softmax has already normalised, so the rest does not come back to 1.
    """
    inp = golden_inputs() if inp is None else inp
    a = inp["attn_x"]                                  # (2, 5, 4)
    # An additive mask. 0 passes through, a large negative blocks.
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

    def scaled(L, q=None):
        """**`scale` replaces `1/√dim`.** It was accepted and dropped on the way to
        borch.ts, which had no seat for it, so a caller who set it got the default
        back — a model whose attention is weighted wrong and trains to somewhere
        plausible. Asked at a value the default cannot produce."""
        x = L.tensor(a) if q is None else q
        return L.nn.functional.scaled_dot_product_attention(x, x, x, scale=0.25)

    add("맨 것", plain)
    add("더하는 가림막", masked)
    add("인과", causal)
    add("배율", scaled)

    # **The place where q, k and v differ is looked at too.** Given all three the same, swapping
    # the three arguments around gives the same value and is not caught.
    def three(L):
        q = L.tensor(a)
        k = L.tensor(a * 0.5 + 0.1)
        v = L.tensor(a[::-1].copy())
        return L.nn.functional.scaled_dot_product_attention(q, k, v)

    cases.append((SDPA_PREFIX + "q·k·v 가 다를 때", three))
    return cases


DROPOUT_PREFIX = "dropout::"


def dropout_cases(inp=None):
    """Dropout. **Its properties are asked, not its value.**

    Every other case in this table freezes real torch's **value** as the answer. Here that cannot
    be done — the answer depends on the random generator, and there is no reason ours should be
    torch's. Not asking at all would leave a whole layer outside the checks.

    So only **what torch and we can answer identically** is asked:

    - eval mode is the identity (asked by value — no randomness involved)
    - `p=0` is the identity too, and `p=1` is all zeros
    - a surviving value is scaled by exactly `x/(1-p)` (**leaving the correction out is the
      commonest mistake**, and then training and inference disagree in magnitude)
    - the fraction dropped is roughly `p`
    - the gradient flows only to the surviving slots

    The answer being "so or not so", both sides answer the same however the generators differ.
    **Not being able to ask about the value is not the same as not asking, and picking what can
    be asked is a third thing.**
    """
    inp = golden_inputs() if inp is None else inp
    # Measuring a fraction needs many samples. Measured on a small array, the randomness shakes the answer.
    big = np.tile(inp["train_x"], (40, 1)).astype(np.float32)     # 960 × 6
    x2 = inp["x2"]
    cases = []

    def verdict(name, fn):
        cases.append((DROPOUT_PREFIX + name, lambda L, f=fn: f(L)))

    # ── where no randomness is involved, it is asked by value. ──
    cases.append((DROPOUT_PREFIX + "eval 은 항등",
                  lambda L: L.nn.functional.dropout(L.tensor(x2), 0.5, training=False)))
    cases.append((DROPOUT_PREFIX + "p=0 은 항등",
                  lambda L: L.nn.functional.dropout(L.tensor(x2), 0.0, training=True)))
    cases.append((DROPOUT_PREFIX + "p=1 은 전부 0",
                  lambda L: L.nn.functional.dropout(L.tensor(x2), 1.0, training=True)))
    cases.append((DROPOUT_PREFIX + "nn.Dropout(eval) 은 항등",
                  lambda L: L.nn.Dropout(0.5).eval()(L.tensor(x2))))

    # ── where randomness is involved, it is asked by property. ──
    def scaled(L):
        """Is a surviving value scaled by exactly `1/(1-p)`? **Leaving the correction out is caught here.**"""
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
        """Is the dropped fraction roughly `p`? Over 5,760 samples, ±5 points is generous."""
        p = 0.5
        out = to_numpy(L.nn.functional.dropout(L.tensor(big), p, training=True))
        dropped = float((out == 0).mean())
        return "대략 맞다" if abs(dropped - p) < 0.05 else f"{dropped:.3f} 이 떨어졌다"

    def flows(L):
        """Does the gradient flow **only to the surviving slots**? Anything non-zero at a dropped slot is wrong."""
        x = L.tensor(big, requires_grad=True)
        out = L.nn.functional.dropout(x, 0.5, training=True)
        out.sum().backward()
        got = to_numpy(x.grad)
        made = to_numpy(out)
        stray = int(((made == 0) & (got != 0)).sum())
        return "살아남은 자리로만" if stray == 0 else f"떨군 자리 {stray} 곳에 흘렀다"

    def differs(L):
        """Called twice, does it drop **different slots**? Drawing once and caching is caught here."""
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

# `(name, arguments)`. **One step will not do** — an optimizer accumulates state, so at the first
# step most of them behave much alike, and they part after that.
_OPTIMIZERS = [
    ("Adagrad", {"lr": 0.1}),
    ("Adadelta", {"lr": 0.5}),
    ("Adamax", {"lr": 0.05}),
    ("NAdam", {"lr": 0.05}),
    ("RAdam", {"lr": 0.05}),
    ("ASGD", {"lr": 0.05}),
    ("Rprop", {"lr": 0.05}),
    # **A 2-D weight is what makes Adafactor's point run** — this model's `0.weight` is (8, 6),
    # so it goes down the path that splits into rows and columns. Asked in 1-D only, that path
    # never runs at all.
    ("Adafactor", {"lr": 0.05}),
    # **`weight_decay` has to bite, or `AdamW` is `Adam`.** The one thing that separates them is
    # where the decay lands — on the gradient before the moments (`Adam`), or on the weights
    # after the update (`AdamW`). At `weight_decay=0` both branches vanish and the two are the
    # same optimizer, so a case built on the default would pass against either implementation.
    # It is large here for the same reason the others take five steps: to be visible.
    ("Adam(weight_decay)", {"lr": 0.05, "weight_decay": 0.1}),
    ("AdamW", {"lr": 0.05, "weight_decay": 0.1}),
    # **Every optimizer that takes `weight_decay` is asked with a non-zero one.**
    #
    # There was no such case anywhere until today, and the absence hid seven defects: the
    # browser binding accepted the argument and dropped it in seven places — in one of them
    # (`NAdam`) while passing the right *number* of arguments, so even an arity check saw
    # nothing. An argument nothing exercises is an argument nothing holds.
    #
    # The value is 0.1 rather than torch's default, for the same reason the pair above uses
    # it: the branch disappears at zero, and a case built on the default passes against an
    # implementation that ignores the argument entirely.
    ("Adagrad(weight_decay)", {"lr": 0.1, "weight_decay": 0.1}),
    ("Adadelta(weight_decay)", {"lr": 0.5, "weight_decay": 0.1}),
    ("Adamax(weight_decay)", {"lr": 0.05, "weight_decay": 0.1}),
    ("NAdam(weight_decay)", {"lr": 0.05, "weight_decay": 0.1}),
    ("RAdam(weight_decay)", {"lr": 0.05, "weight_decay": 0.1}),
    ("RMSprop(weight_decay)", {"lr": 0.01, "weight_decay": 0.1}),
    ("ASGD(weight_decay)", {"lr": 0.05, "weight_decay": 0.1}),
    ("SGD(weight_decay)", {"lr": 0.05, "weight_decay": 0.1}),
    # **The arguments that were not parameters until the core took torch's order.**
    # `SGD` read `(params, lr, momentum, weight_decay)`, so the tutorial line
    # `SGD(p, 0.1, 0.9, 1e-4)` set the *dampening* to the weight decay and left the
    # decay at zero — two different things, both plausible small numbers, and the run
    # trains and trains slightly wrong.
    #
    # Each is asked with a value that changes the answer. `dampening` is meaningless
    # without momentum, `nesterov` needs momentum and refuses dampening (torch
    # refuses the same combination), and `initial_accumulator_value` only shows in
    # the first steps, which is why five are taken.
    ("SGD(dampening)", {"lr": 0.1, "momentum": 0.9, "dampening": 0.5}),
    ("SGD(nesterov)", {"lr": 0.1, "momentum": 0.9, "nesterov": True}),
    ("SGD(maximize)", {"lr": 0.01, "maximize": True}),
    ("Adagrad(initial_accumulator_value)",
     {"lr": 0.1, "initial_accumulator_value": 0.5}),
    # **`maximize` existed on two of the twelve and the table asked about one.** It
    # turns the gradient round, so it changes every value, and ten optimizers were
    # simply refusing the argument. Adam and RMSprop are asked here because they are
    # the two a recipe reaches for; the other eight are checked against real torch in
    # the same edit rather than frozen, since twelve more rows of the same shape
    # would freeze one behaviour twelve times.
    ("Adam(maximize)", {"lr": 0.05, "maximize": True}),
    ("RMSprop(maximize)", {"lr": 0.05, "maximize": True}),
    # **A default is the one value a case cannot check by using it.** `SGD`'s `lr`
    # defaulted to 0.01 here and to 1e-3 in torch — ten times too large on the
    # optimizer a tutorial is most likely to build with no arguments at all — and
    # every case above names its own rate, so nothing had ever asked. This row omits
    # it on purpose.
    ("SGD(the default rate)", {}),
    # **Four algorithm variants that were absent and were not on the list of what was
    # owed.** `KEYWORD_ONLY_ABSENCES` had been that list, and torch takes all four
    # positionally, so the count was measuring one axis while the things most worth
    # owing sat on the other. Each changes values; each is asked at a value that
    # changes them.
    #
    # `amsgrad` keeps the largest second moment seen so far, so the step can only
    # shrink. `centered` divides by an estimate of the variance rather than of the
    # second moment. `momentum` gives RMSprop a velocity buffer. `decoupled` applies
    # the decay to the weights rather than to the gradient, which is what makes
    # `AdamW` a separate name — so it is asked *with* a weight decay, since without
    # one the two branches are the same arithmetic.
    ("Adam(amsgrad)", {"lr": 0.05, "amsgrad": True, "weight_decay": 0.1}),
    ("RMSprop(centered)", {"lr": 0.05, "centered": True}),
    ("RMSprop(momentum)", {"lr": 0.05, "momentum": 0.9}),
    ("NAdam(decoupled_weight_decay)",
     {"lr": 0.05, "decoupled_weight_decay": True, "weight_decay": 0.1}),
]

# `(name, constructor arguments, how many steps)`. The learning rate's **trajectory** is asked.
_SCHEDULERS = [
    ("ConstantLR", {"factor": 0.5, "total_iters": 3}, 8),
    ("LinearLR", {"start_factor": 0.5, "end_factor": 1.0, "total_iters": 4}, 8),
    ("PolynomialLR", {"total_iters": 5, "power": 2.0}, 8),
    ("MultiplicativeLR", {}, 6),
    ("CosineAnnealingWarmRestarts", {"T_0": 3, "T_mult": 2}, 10),
    ("OneCycleLR", {"max_lr": 0.4, "total_steps": 10}, 10),
    ("CyclicLR", {"base_lr": 0.01, "max_lr": 0.1, "step_size_up": 3}, 14),
    # **The up and down are given different lengths** — equal, it is not even visible whether
    # `step_size_down` exists.
    ("CyclicLR(위아래 다름)", {"base_lr": 0.01, "max_lr": 0.1, "step_size_up": 2,
                          "step_size_down": 4}, 14),
    # From the second cycle the peak halves — stepping through one cycle only, it does not part.
    ("CyclicLR(triangular2)", {"base_lr": 0.01, "max_lr": 0.1, "step_size_up": 3,
                               "mode": "triangular2"}, 14),
    # **`exp_range` measures against the step, not the cycle.** That one thing is where it parts.
    ("CyclicLR(exp_range)", {"base_lr": 0.01, "max_lr": 0.1, "step_size_up": 3,
                             "mode": "exp_range", "gamma": 0.9}, 14),
]


def opt_cases(inp=None):
    """Five optimizers and six schedulers. **What is asked is several steps later.**

    ## Why one step will not do

    An optimizer accumulates state. At the first step `Adam`, `NAdam` and `RAdam` give nearly the
    same value, and `Adagrad` and `SGD` differ by about a learning rate. They part after that
    accumulation settles, so measured at one step, implementing all five as `SGD` passes.

    ## The schedulers are asked by trajectory

    What a scheduler does is produce **a sequence of learning rates**, so the whole sequence is
    frozen as the answer. Looking at the last value alone passes even when the road there
    differs, and `LinearLR` and `ConstantLR` really do meet at the end — past `total_iters` both
    are back at the original rate.

    `MultiplicativeLR` takes a lambda, and what the golden answers hold is an answer and not a
    function, so the caller writes the same expression (multiply by 0.9).
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
        # The branch is written in the name in parentheses — the class name is what comes
        # before it, the same rule `lr_trace` uses for the schedulers.
        opt = getattr(L.optim, name.split("(")[0])(m.parameters(), **args)
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
        """The learning rate's trajectory. **The optimizer is really stepped** — the order decides the values."""
        m = model_of(L)
        opt = L.optim.SGD(m.parameters(), lr=0.2)
        # The branch is written in the name in parentheses — the class name is what comes before it.
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
        # **Handed back as the library's tensor.** The harness calls `detach` on a value it
        # takes, and a bare numpy array does not know that. It is compared numerically, so the
        # tolerance applies as usual.
        return L.tensor(np.array(seen, dtype=np.float32))

    for name, args, steps in _SCHEDULERS:
        cases.append((OPT_PREFIX + f"{name}/자취",
                      lambda L, n=name, a=args, s=steps: lr_trace(L, n, a, s)))

    # **The two that chain.** This is where schedulers are composed, and a wrong composition
    # parts the values even with every individual scheduler right.
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
        # **Handed back as the library's tensor.** The harness calls `detach` on a value it
        # takes, and a bare numpy array does not know that. It is compared numerically, so the
        # tolerance applies as usual.
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
        # **Handed back as the library's tensor.** The harness calls `detach` on a value it
        # takes, and a bare numpy array does not know that. It is compared numerically, so the
        # tolerance applies as usual.
        return L.tensor(np.array(seen, dtype=np.float32))

    cases.append((OPT_PREFIX + "SequentialLR/자취", sequential))
    cases.append((OPT_PREFIX + "ChainedScheduler/자취", chained))

    # ── where a branch is asked in isolation ──
    #
    # The model training above passes when the optimizer is **roughly right.** The arguments that
    # decide a branch only surface when a gradient is fed to a single parameter by hand.
    start = np.array([1.0, -2.0, 0.5], dtype=np.float32)

    def walk(L, name, grads, **args):
        p = L.tensor(start.copy(), requires_grad=True)
        opt = getattr(L.optim, name)([p], **args)
        seen = []
        for g in grads:
            opt.zero_grad()
            p.grad = L.tensor(g)
            opt.step()
            # **Read the way all three implementations know** — the same path as the harness's
            # `to_numpy`. `p.data` leaks through to the other side's attribute in the binding and
            # the shape comes out wrong.
            seen.append(np.asarray(p.detach().numpy(), dtype=np.float32).copy())
        return L.tensor(np.stack(seen))

    ramp = [np.array([0.1, -0.3, 0.2], dtype=np.float32) * (i + 1)
            for i in range(4)]
    # **A gradient whose sign flips.** Rprop's `etas` and its "do not move on a flipped slot"
    # rule are visible only here — with the sign unchanged the step only grows and one branch
    # alone runs.
    flip = [np.array([0.1, -0.3, 0.2], dtype=np.float32),
            np.array([-0.1, -0.3, 0.2], dtype=np.float32),
            np.array([-0.2, -0.3, 0.2], dtype=np.float32),
            np.array([-0.2, 0.3, 0.2], dtype=np.float32)]

    narrow = [
        ("ASGD/기본값", "ASGD", ramp, {}),
        ("ASGD/lambd", "ASGD", ramp, {"lr": 0.1, "lambd": 0.01}),
        ("ASGD/alpha", "ASGD", ramp, {"lr": 0.1, "alpha": 0.5}),
        # **`t0` has to be lowered for the averaging to actually run** — at the default of a
        # million, `mu` is always 1 and `ax` is a copy of the parameter. The averaging branch
        # never runs at all.
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
        """**It splits into rows and columns from 2-D up.** In 1-D even the state keys differ
        (`variance` against `row_var` and `col_var`) — this optimization's whole point is there,
        so asked on a vector only, that path never runs once."""
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
        """**`step` takes a closure** — it re-measures the loss several times within one step.

        That shape is unlike the other optimizers', so a training loop written as usual does nothing.
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

    # ── **the three above never once step on the quasi-Newton part.** ──
    #
    # That closure **feeds** the gradient in as a constant (`pp.grad = tensor([0.1, -0.3, 0.2])`).
    # Then `flat` is unchanged from iteration to iteration, so `y = flat - prev_flat` is 0, and
    # `ys` being 0 means **nothing goes into the history.** With an empty history the two-loop
    # recursion passes through as `q = -flat` and `r = q * h_diag(=1)`, so what is left is the
    # first iteration's gradient descent and LBFGS's step-size rule, and nothing else.
    #
    # The name is LBFGS and what was measured was not half of that algorithm. **The gradient has
    # to differ by position** for a history to build up, and only then does the name
    # `history_size` mean anything.
    curve = np.array([1.0, 4.0, 9.0], dtype=np.float32)

    def lbfgs_real(L, steps=3, **args):
        """The closure **really differentiates.** It is `sum(w·p²)`, so the gradient is `2·w·p`.

        A quadratic whose curvature differs by position, which is exactly the shape a
        quasi-Newton method wins on — once the history builds, a different step comes out per axis.
        """
        p = L.tensor(start.copy(), requires_grad=True)
        w = L.tensor(curve)
        opt = L.optim.LBFGS([p], **args)
        seen = []
        for _ in range(steps):
            def closure(pp=p, ww=w):
                if pp.grad is not None:
                    pp.grad = None
                out = (pp * pp * ww).sum()
                out.backward()
                return out
            opt.step(closure)
            seen.append(np.asarray(p.detach().numpy(), dtype=np.float32).copy())
        return L.tensor(np.stack(seen))

    cases.append((OPT_PREFIX + "LBFGS/진짜 기울기", lambda L: lbfgs_real(L, lr=0.1)))
    # **The history really fills up and starts evicting.** `history_size=2` with more iterations
    # than that is what steps on the place where an old pair is dropped — the `history_size` case
    # above built no history at all, so that name selected nothing.
    cases.append((OPT_PREFIX + "LBFGS/이력이 밀려난다",
                  lambda L: lbfgs_real(L, steps=2, lr=0.5, max_iter=8,
                                       history_size=2)))
    # **Near the threshold.** If the convergence test fires even one iteration differently the
    # trajectory parts wholesale. An f32 reduction rounds differently from numpy's, so this place
    # is the real risk for a GPU implementation.
    cases.append((OPT_PREFIX + "LBFGS/문턱 근처에서 멈춘다",
                  lambda L: lbfgs_real(L, steps=2, lr=0.3, max_iter=12,
                                       tolerance_change=1e-3)))

    def scalar_param_keeps_constants(L):
        """**Training a single-element parameter must not change the constants.**

        The GPU edition **caches single-element tensors by value** and hands them back — it is
        there to stop `x * 0.05` building the same constant every step in a training loop, and it
        is correct as long as nobody writes into that buffer. Optimizer state **writes into that
        buffer in place.** Where the two meet, at a size-1 parameter, a constant shared by the
        whole program is quietly overwritten — no exception, no warning, and values are wrong
        from then on somewhere very far away.

        So the same constant is multiplied **after** training. Real torch has no such cache so the
        answer is obvious, and that obvious answer catches this defect. The ordinary optimizer
        cases, which plant the weights from outside, do not pass through the state bank and
        cannot see it.

        **One step will not catch it.** Rprop's first step does not change the magnitude, so it
        overwrites with the same value and leaves no mark — it was written that way at first and
        caught nothing. Several steps are taken so the state really moves, and then it is asked.
        Nor will one constant do: the state bank starts at 0, so **0 and 1 are dirtied first.**

        **The optimizers have to be stepped on evenly.** `SGD`, `Adam` and `RMSprop` use dedicated
        kernels and sit on a different base from the rest — fixing only the common base at first
        left those three untouched, and without asking about them it would have looked fixed.
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
        # After training, are those constants still those values.
        probe = L.tensor(np.array([1.0, 2.0], dtype=np.float32))
        for k in (0.0, 1.0, 0.05):
            seen.append(probe * k)
        return L.cat(seen)

    cases.append((OPT_PREFIX + "크기 1 파라미터가 상수를 안 더럽힌다",
                  scalar_param_keeps_constants))

    # ── **the optimizer's `state_dict`.** Resuming training hangs on this. ──
    #
    # Save the model weights without the optimizer and the resumed training starts again
    # **having lost the momentum and the second moments.** No exception; the loss curve simply
    # jumps once — the kind of thing a person shrugs off as normal.
    #
    # **The values are asked, not the key names.** torch has `{"state": …, "param_groups": …}`
    # and borch.ts has a bank structure, so the shapes differ. Asking about the shape would part
    # forever, while what actually matters to a user is "is stopping and resuming the same as not
    # stopping", and that can be asked regardless of shape.
    #
    # There is a reason `Adam` is used — `SGD(momentum=0)` has no state, so leaving the saving out
    # entirely passes this case. It has to be asked with an optimizer that carries state.
    def resume(L, name, args):
        m = model_of(L)
        opt = getattr(L.optim, name)(m.parameters(), **args)
        crit = L.nn.CrossEntropyLoss()
        x, y = L.tensor(xin), L.tensor(yin)

        def one(o):
            o.zero_grad()
            crit(m(x), y).backward()
            o.step()

        for _ in range(3):
            one(opt)
        saved = opt.state_dict()
        # **Loaded onto a new optimizer.** Putting it back into the same one asks nothing —
        # the state is already there, so a `load` that does nothing still passes.
        fresh = getattr(L.optim, name)(m.parameters(), **args)
        fresh.load_state_dict(saved)
        for _ in range(2):
            one(fresh)
        return dict(m.named_parameters())["0.weight"]

    for _name, _args in (("SGD", {"lr": 0.1, "momentum": 0.9}),
                         ("Adam", {"lr": 0.05}),
                         ("RMSprop", {"lr": 0.05})):
        cases.append((OPT_PREFIX + f"{_name}/이어서 학습하기",
                      lambda L, n=_name, a=_args: resume(L, n, a)))

    # **Did the three above really move anything?**
    #
    # The cases above can pass even when `load_state_dict` does nothing — they do if loaded and
    # not loaded give the same value. So **the difference between the two** is frozen as the
    # answer. In torch that difference is not 0, and in an implementation where loading does
    # nothing it is 0. At that moment the three above are green while measuring nothing, and this
    # case turns that state red.
    def resume_gap(L):
        def run(carry):
            m = model_of(L)
            opt = L.optim.Adam(m.parameters(), lr=0.05)
            crit = L.nn.CrossEntropyLoss()
            x, y = L.tensor(xin), L.tensor(yin)

            def one(o):
                o.zero_grad()
                crit(m(x), y).backward()
                o.step()

            for _ in range(3):
                one(opt)
            saved = opt.state_dict()
            fresh = L.optim.Adam(m.parameters(), lr=0.05)
            if carry:
                fresh.load_state_dict(saved)
            for _ in range(2):
                one(fresh)
            return dict(m.named_parameters())["0.weight"]

        return run(True) - run(False)

    cases.append((OPT_PREFIX + "상태를 안 옮기면 갈린다", resume_gap))

    # **The scheduler has to resume too.** Restore the optimizer alone and build a fresh
    # scheduler and the learning rate **goes back to its first value** — training half cooled
    # heats up again, and a loss that was coming down goes up once before coming down again.
    def sched_resume(L, carry):
        m = model_of(L)
        opt = L.optim.SGD(m.parameters(), lr=0.2)
        sch = L.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        seen = []
        for _ in range(4):
            seen.append(round(float(opt.param_groups[0]["lr"]), 6))
            opt.step()
            sch.step()
        saved = sch.state_dict()
        fresh = L.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        if carry:
            fresh.load_state_dict(saved)
        for _ in range(4):
            seen.append(round(float(opt.param_groups[0]["lr"]), 6))
            opt.step()
            fresh.step()
        # **Handed back as characters.** A learning-rate trajectory is a sequence, so when it
        # parts, which slot parted has to stay visible — as a tensor all that is left is
        # "max diff 1.5e-01" and which of the eight slots is not in it.
        return " ".join(f"{v:g}" for v in seen)

    cases.append((OPT_PREFIX + "StepLR/이어서 학습하기",
                  lambda L: sched_resume(L, True)))
    # If loading does nothing the two trajectories become equal, and then the case above is
    # measuring nothing.
    cases.append((OPT_PREFIX + "스케줄러 상태를 안 옮기면 갈린다",
                  lambda L: f"{sched_resume(L, True)} | {sched_resume(L, False)}"))

    def plateau_resume(L, carry):
        """**`ReduceLROnPlateau` had no `state_dict` at all** until this case.

        Seven mutable attributes and nowhere to put one of them, so a run that saved
        everything it could still came back with the patience, the best value and the
        cooldown at their starting points — a cut arriving epochs early or late, and
        nothing on screen to say so. torch carries fifteen keys here.

        Asked **with a cooldown**, because that is the attribute the resume loses most
        visibly: without one, the patience alone can be re-earned in a step or two and
        the two trajectories rejoin.
        """
        p = L.tensor([1.0], requires_grad=True)
        opt = L.optim.SGD([p], lr=1.0)
        build = lambda: L.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=1, cooldown=2)
        sch = build()
        seen = []
        for metric in [1.0, 1.0, 0.5, 0.5, 0.5]:
            sch.step(metric)
            seen.append(round(float(opt.param_groups[0]["lr"]), 6))
        saved = sch.state_dict()
        fresh = build()
        if carry:
            fresh.load_state_dict(saved)
        for metric in [0.5, 0.5, 0.5, 0.5, 0.5]:
            fresh.step(metric)
            seen.append(round(float(opt.param_groups[0]["lr"]), 6))
        return " ".join(f"{v:g}" for v in seen)

    cases.append((OPT_PREFIX + "ReduceLROnPlateau/이어서 학습하기",
                  lambda L: plateau_resume(L, True)))
    # And the pair that keeps the case above honest: if loading does nothing, the two
    # trajectories are equal and the first case is measuring nothing.
    cases.append((OPT_PREFIX + "ReduceLROnPlateau 상태를 안 옮기면 갈린다",
                  lambda L: f"{plateau_resume(L, True)} | {plateau_resume(L, False)}"))

    # ── **`save`/`load`.** For any of the above to be useful it has to become a file. ──
    #
    # Getting `state_dict()` to agree across all three and then **having no way to use it** makes
    # resuming a story that holds within one session only. Refresh the tab and it is gone.
    #
    # The formats differ (torch has pickle, we have safetensors). So **the round trip is asked,
    # not the bytes** — read back what was written, is it the same.
    def _tmp(L, name):
        # **It has to run under Pyodide too.** There is no real filesystem inside a browser,
        # but there is a virtual one and `tempfile` uses it — the Python code does not change.
        import os
        import tempfile
        return os.path.join(tempfile.mkdtemp(), name)

    def save_load_state_dict(L):
        m = model_of(L)
        path = _tmp(L, "sd.bin")
        L.save(m.state_dict(), path)
        got = L.load(path)
        return got["0.weight"]

    cases.append((OPT_PREFIX + "save/load 가 state_dict 를 왕복한다",
                  save_load_state_dict))

    # **The textbook idiom is nested.** `{"model": …, "opt": …, "epoch": 3}` is saved whole.
    # If only a flat dict of tensors works, that code does not run.
    def save_load_nested(L):
        m = model_of(L)
        opt = L.optim.Adam(m.parameters(), lr=0.05)
        path = _tmp(L, "ckpt.bin")
        L.save({"model": m.state_dict(), "opt": opt.state_dict(),
                "epoch": 3, "note": "half way"}, path)
        got = L.load(path)
        return f"{' '.join(sorted(got))} epoch={got['epoch']} note={got['note']}"

    cases.append((OPT_PREFIX + "save/load 가 중첩을 왕복한다", save_load_nested))

    # **A `state_dict`'s keys already contain dots** (`fc.weight`). Splitting the flattened name
    # on dots again on the way back gives `{"model": {"fc": {"weight": …}}}` — every value present
    # and the structure different, so feeding what was read back into `load_state_dict` blows up
    # at that point.
    def save_load_dotted(L):
        m = model_of(L)
        path = _tmp(L, "dotted.bin")
        L.save({"model": m.state_dict()}, path)
        got = L.load(path)
        return " ".join(sorted(got["model"]))

    cases.append((OPT_PREFIX + "중첩 안의 점 찍힌 열쇠가 안 쪼개진다", save_load_dotted))

    # Is what was read back **really usable?** Keys and values agreeing is no use if it cannot be loaded.
    def save_load_then_use(L):
        src = model_of(L)
        path = _tmp(L, "use.bin")
        L.save(src.state_dict(), path)
        dst = L.nn.Sequential(L.nn.Linear(6, 8), L.nn.ReLU(), L.nn.Linear(8, 3))
        dst.load_state_dict(L.load(path))
        return dst(L.tensor(xin))

    cases.append((OPT_PREFIX + "되읽은 것을 그대로 얹을 수 있다", save_load_then_use))
    return cases


FNAME_PREFIX = "fname::"


def functional_name_cases(inp=None):
    """The remaining names in `nn.functional` — **the in-place activations** and `interpolate`'s
    old names.

    ## The in-place activations

    `F.relu_(x)` edits `x` in its own buffer. It is what a training loop uses to avoid building
    an intermediate tensor. The computation belongs to the underscore-less side and this only
    writes back, so there are three things to ask — **is the value the same**, **does it hand
    back the same tensor**, and **does it refuse a leaf with gradients on**. Without the middle
    one, an implementation returning a new tensor passes every value case.

    ## The three `upsample` names

    torch keeps accepting them while emitting a deprecation warning. **Only
    `upsample_bilinear` has `align_corners=True`**, and `interpolate(mode='bilinear')` defaults
    to false — treat them as aliases on the strength of the name and the edges go out of line
    while the interior stays similar enough that the eye does not part them.
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
        # The ones with randomness in them have a determined answer only in **eval mode.**
        ("rrelu_(평가)", lambda f, t: f.rrelu_(t, 0.1, 0.3, False)),
    )
    for name, run in inplace:
        def value(L, r=run):
            x = L.tensor(line.copy())
            r(F(L), x)
            return x

        add(f"제자리::{name}", value)

    def same_tensor(L):
        """**It has to hand back the same tensor** — a new one and every value case above passes."""
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
        """In place and not in place have to give **the same answer.** Two copies diverge."""
        x = L.tensor(line.copy())
        F(L).leaky_relu_(x, 0.3)
        return x - F(L).leaky_relu(L.tensor(line.copy()), 0.3)

    add("제자리::제자리 아닌 것과 같다", same_as_plain)

    img = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    add("upsample(scale)",
        lambda L: F(L).upsample(L.tensor(img), scale_factor=2))
    add("upsample_nearest",
        lambda L: F(L).upsample_nearest(L.tensor(img), scale_factor=2))
    # **Only here is `align_corners=True`.**
    add("upsample_bilinear",
        lambda L: F(L).upsample_bilinear(L.tensor(img), scale_factor=2))
    add("upsample(size, bilinear)",
        lambda L: F(L).upsample(L.tensor(img), size=(8, 8), mode="bilinear"))
    add("upsample_bilinear(size=6)",
        lambda L: F(L).upsample_bilinear(L.tensor(img), size=(6, 6)))

    def upsample_corners_differ(L):
        """`upsample_bilinear` and `interpolate(bilinear)` **must not be equal.**

        The former is `align_corners=True` and the latter defaults to false. Treated as aliases,
        it surfaces only here — the value cases pass because each gives its own answer.
        """
        x = L.tensor(img)
        a = F(L).upsample_bilinear(x, scale_factor=2)
        b = F(L).interpolate(x, scale_factor=2, mode="bilinear")
        return float((a - b).abs().sum().item()) > 1e-6

    add("upsample_bilinear 은 별명이 아니다",
        lambda L: str(upsample_corners_differ(L)))

    # ── the three faces of `max` and `min` ──
    #
    # torch returns **different things** depending on the arguments: `max(x)` is one maximum over
    # everything, `max(x, dim)` is a `(value, index)` pair, and `max(x, other)` is the maximum
    # per slot.
    #
    # These three branches went unasked, and in the binding `x.max()` gave **a pair reduced over
    # axis 0 only** — because it was passed straight through to the other side's `max(dim=0)`.
    # It is loud only when converted to a scalar; used in a comparison it compares per slot and
    # is quietly a different answer.
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

    # ── batch_norm ──
    #
    # The layer's function form. **In training it edits the running statistics in place** — the
    # tensor passed in comes back updated. An implementation that returns a new one passes every
    # output case and is wrong only on the eval-mode value, so the updated statistics themselves
    # are frozen as the answer.
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
    # Without statistics given, training mode counts from this batch.
    add("batch_norm::통계 없이 학습",
        lambda L: F(L).batch_norm(L.tensor(bn_x), None, None, L.tensor(bn_w),
                                  L.tensor(bn_b), training=True))

    def bn_updates(L, momentum=0.1):
        """**The updated statistics are the answer.** Normalisation uses the biased variance and
        the update uses the unbiased one — leave both biased and it is off by 2.6% only here."""
        rm, rv = L.tensor(bn_rm.copy()), L.tensor(bn_rv.copy())
        F(L).batch_norm(L.tensor(bn_x), rm, rv, training=True, momentum=momentum)
        return L.cat([rm, rv])

    add("batch_norm::갱신된 통계", bn_updates)
    add("batch_norm::갱신된 통계(momentum=0.5)",
        lambda L: bn_updates(L, 0.5))

    def bn_layer_matches(L):
        """**The layer and the function have to give the same answer** — kept as two copies they diverge eventually."""
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
    # **A 1-D row of indices plus `offsets`** — where the bags have differing lengths.
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

    # ── gumbel_softmax ──
    #
    # Random, so the value cannot be asked. **The properties are asked** — a row sums to 1, and
    # `hard` gives only 0 and 1 and still sums to 1. Those have to hold for any draw.
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
        """**A gradient flows even under `hard`** — the value is 0/1 and the derivative is the
        smooth one's.

        That separation is this function's whole point, so an implementation where no gradient
        arrives at all is caught only here. Summed evenly it comes to 0 because of softmax's
        properties, so **one side is weighted.**
        """
        x = L.tensor(gs_logits, requires_grad=True)
        weights = L.tensor(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                                    dtype=np.float32))
        (F(L).gumbel_softmax(x, hard=True) * weights).sum().backward()
        return "기울기 있음" if x.grad is not None else "기울기가 안 왔다"

    add("gumbel_softmax::hard 에도 기울기가 흐른다", gs_grad)

    # ── the spatial transformer ──
    #
    # `affine_grid` writes down "which part of the input does this output slot look at" and
    # `grid_sample` lifts the value from there. The `theta` between them being learned is the
    # point, so whether a gradient travels the whole chain has to be asked.
    #
    # ## Two traps in the shape of the case
    #
    # 1. **Asked on a square only, the `(x, y)` order is invisible.** The grid's last axis is
    #    `(x, y)` while the shape is `(H, W)`, which is reversed. At 3×3, writing it reversed
    #    gives the same answer.
    # 2. **The gradient has to be asked inside a cell.** A 90° rotation lands the grid exactly on
    #    the cell boundaries, and there `floor` flips on a 6e-8 difference and the gradient
    #    changes wholesale (measured: `tests/probe_grid5.py`). The value is stable even there —
    #    the weight is 0, so either corner gives the same value. So **the value is asked with a
    #    rotation and the gradient with a slanted `theta`.** An answer parting at a boundary is
    #    not a defect; there is no answer at that place.
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

    # A grid pointing outside the range — the only place the three padding modes part.
    out_grid = np.array([[[[-2.0, -2.0], [2.0, 2.0]],
                          [[0.0, 0.0], [-1.0, 1.0]]]], dtype=np.float32)
    for pad in ("zeros", "border", "reflection"):
        for ac in (False, True):
            add(f"grid_sample::padding={pad}(align={ac})",
                lambda L, p=pad, a=ac: F(L).grid_sample(
                    L.tensor(img3), L.tensor(out_grid), padding_mode=p,
                    align_corners=a))

    # Half a cell out of line — the only place where bilinear really blending is visible.
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
        """**The whole chain.** This is the path by which a spatial transformer learns `theta`.

        A slanted `theta` is used — under rotation the grid lands on cell boundaries and the
        answer is unstable.
        """
        t = L.tensor(tilt3, requires_grad=True)
        F(L).grid_sample(
            L.tensor(img3),
            F(L).affine_grid(t, (1, 1, 3, 3), align_corners=False),
            align_corners=False).sum().backward()
        return t.grad

    add("grid_sample::grad(theta 까지)", grid_grad_theta)

    # ── multi_head_attention_forward ──
    #
    # The computation `MultiheadAttention` does inside, exposed under a name. torch's layer calls
    # this function too.
    #
    # **The input is `(L, N, E)` — length first.** The layer takes `batch_first` and this function
    # is always length-first, so calling it with the batch in front quietly mixes different axes.
    # So it is asked with a shape where `L != N` — equal, a swapped axis is not caught.
    mha_L, mha_N, mha_E, mha_H, mha_S = 3, 2, 4, 2, 3

    def mha_w(shape, spin=0.0):
        """**Not random.** The TypeScript-side case has to carry the same values, and a random
        generator does not cross languages. Being a counted value, both sides build the same thing."""
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
    # The branch that returns **per head** — asked on the average only, mixing the heads is invisible.
    add("mha::가중치(머리마다)",
        lambda L: mha(L, average_attn_weights=False)[1])
    add("mha::need_weights=False",
        lambda L: mha(L, need_weights=False)[0])
    add("mha::가중치가 None 인가",
        lambda L: str(mha(L, need_weights=False)[1] is None))

    causal = np.triu(np.ones((mha_L, mha_S), dtype=bool), k=1)
    add("mha::불리언 가림막",
        lambda L: mha(L, attn_mask=L.tensor(causal))[0])
    # **A float mask is added** — lump "non-zero means masked" together and only the causal mask
    # happens to come out right.
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
        """**The layer and the function have to give the same answer.** Whether the layer calls
        this function is visible here."""
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
        """**A branch it does not do is refused loudly** — ignored quietly, the value is plausibly different."""
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
    """The names that exist only at torch's **top level**, and the introspection ones.

    ## The top level does not share `F`'s signatures

    torch puts `nn.functional`'s things at the top level too, and those are raw ATen
    operations — **the argument order differs and the enums are integers.**

    - `torch.batch_norm` takes the weight **before** the running statistics. Passed straight
      through, the weight is used as the mean — not an exception, a plausibly different value.
    - `torch.grid_sampler` has `mode` as 0/1 and padding as 0/1/2.
    - `torch.ctc_loss` has `reduction` as 0/1/2 and **defaults to 1 (mean).**

    The same computation called differently, so the computation is kept in one copy and only
    the positions move. Whether that move is right is confirmed by value alone.

    ## The introspection ones are predicates, not values

    `is_floating_point`, `can_cast` and `typename` are what textbook code **branches on.**
    Absent, it stops at that line with every computation right; wrong, it takes the other branch.
    """
    cases = []

    def add(name, fn):
        cases.append((TOP_PREFIX + name, fn))

    holes = np.array([[-1.0, 0.5, np.nan], [0.25, np.inf, 1.0]], dtype=np.float32)
    img = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    plain = np.array([[-1.0, 0.5, 2.0], [0.25, -3.0, 1.0]], dtype=np.float32)

    # ── the top-level-only in-place ones ──
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

    # `feature_dropout` drops **whole channels** — the same computation as `dropout2d`.
    add("feature_dropout(p=0)",
        lambda L: L.feature_dropout(L.tensor(img), 0.0, True))

    # ── raw ATen signatures ──
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

    # ── gradient modes ──
    def grad_modes(L):
        """**They have to nest** — `enable_grad` turns it back on inside `no_grad`.
        Whether it returns to its previous value on the way out is visible only here."""
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

    # ── introspection ──
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
        # **Only one direction is true** — narrowing is false.
        ("can_cast", lambda L: (L.can_cast(L.int64, L.float32),
                                L.can_cast(L.float32, L.int64))),
    )
    for name, fn in checks:
        add(f"살펴보기::{name}", lambda L, f=fn: " ".join(str(v) for v in f(L)))

    add("살펴보기::typename",
        lambda L: f"{L.typename(L.tensor(plain))} {L.typename(3)}")
    add("살펴보기::promote_types",
        lambda L: str(L.promote_types(L.int64, L.float32)))
    # **`result_type` sat beside this one with no case at all**, and what was public
    # under that name was the internal numpy-dtype helper — every call shaped the way
    # torch documents raised, and the only shape that worked is the one torch refuses.
    # The contract was inverted end to end and nothing asked.
    #
    # The scalar rows are the point: a Python number is **weaker than a tensor**, so
    # an int tensor with a Python float is `float32` and not `float64`. Asking only
    # the tensor-pair row would pass on a copy of `promote_types`.
    add("살펴보기::result_type(텐서,텐서)",
        lambda L: str(L.result_type(L.tensor([1]), L.tensor([1.0]))))
    add("살펴보기::result_type(텐서,파이썬float)",
        lambda L: str(L.result_type(L.tensor([1]), 1.5)))
    add("살펴보기::result_type(텐서,파이썬int)",
        lambda L: str(L.result_type(L.tensor([1]), 2)))
    add("살펴보기::result_type(bool텐서,파이썬int)",
        lambda L: str(L.result_type(L.tensor([True]), 2)))
    add("살펴보기::result_type(수,수)", lambda L: str(L.result_type(1, 2.5)))
    # torch takes tensors here and refuses dtypes — `promote_types` is the other
    # question. **Not `refusal_case`**: that one is for places torch manages and all
    # three of ours decline, and here torch declines too. Both sides raise the same
    # class, so the exception's name is a value the two can be compared on.
    #
    # Being lenient is diverging too. Answering a dtype pair would work here and stop
    # on a real machine, and the exception a learner then meets names neither
    # library.
    def result_type_refuses_dtypes(L):
        try:
            L.result_type(L.int64, L.float32)
            return "예외가 안 났다"
        except Exception as exc:                                    # noqa: BLE001
            return type(exc).__name__

    add("살펴보기::result_type(dtype,dtype)=거절", result_type_refuses_dtypes)
    add("살펴보기::get_default_dtype", lambda L: str(L.get_default_dtype()))
    add("살펴보기::finfo",
        lambda L: (f"{L.finfo(L.float32).eps:.9g} {L.finfo(L.float32).max:.9g} "
                   f"{L.finfo(L.float32).bits}"))
    add("살펴보기::iinfo",
        lambda L: f"{L.iinfo(L.int64).max} {L.iinfo(L.int64).min}")
    # **The kind is asked too.** The two lines above look at the value only, so a wrapper
    # function sitting there passes — and one really was. torch's is a type and can be asked with
    # `isinstance`, and that difference is caught neither by a check that only looks for the name
    # nor by a value comparison.
    add("살펴보기::finfo 는 클래스다", lambda L: type(L.finfo).__name__)
    add("살펴보기::iinfo 는 클래스다", lambda L: type(L.iinfo).__name__)
    add("살펴보기::finfo(인자 없이) 는 기본형",
        lambda L: str(L.finfo().dtype))

    def rng_round_trip(L):
        """**Restoring the state has to give the same numbers.** That round trip is resuming training.

        torch hands the state back as a byte tensor and we hand back our generator's state — the
        shapes differ, so **whether the round trip works is asked, not the value.** Putting an
        unanswerable question in the table leaves the table unable to say what it passed.
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
    """Pooling that returns the winning position alongside, and the pair that puts it back.

    ## Why a separate index table is needed

    Max pooling keeps one per window and discards the rest. From the value alone **there is no
    record of which slot won** — so `MaxUnpool` cannot go back from the value. torch has the
    pooling return the positions alongside (`return_indices=True`) and passes them to the
    unpooling. A common pair in an autoencoder.

    ## The convention for the position numbers

    They are **flat indices within a plane** — in 2-D, `h*W + w`, counted from 0 again for each
    batch and channel. Measured (`tests/probe_pool.py`). Mistaking them for indices into the
    whole tensor is right only when the batch is one.

    ## What is asked here

    The index table is an **integer table** rather than a value, so a value comparison cannot
    pass on being merely close — one slot out and the integer differs. So the index table itself
    is frozen as the answer.

    Who wins a tie is here too. torch takes the smaller flat index, that is, **the position that
    comes first in row-major order.** Folding the axes from the front gives the column-major
    first instead, and **since the value is the same, no value case catches it.**
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

    # ── index tables ──
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
        # **An adaptive one that does not divide evenly** — the window size differs by position.
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

    # **One computation under two names.** `return_indices=True` and `*_with_indices`.
    def two_names(L):
        x = L.tensor(plane)
        a = F(L).max_pool2d(x, 2, return_indices=True)
        b = F(L).max_pool2d_with_indices(x, 2)
        c = F(L).adaptive_max_pool2d_with_indices(x, 2)
        return L.cat([a[0].reshape(-1), b[0].reshape(-1), c[0].reshape(-1),
                      a[1].reshape(-1).float(), b[1].reshape(-1).float()])

    add("이름이 둘인 같은 계산", two_names)

    # **Turning the indices on must leave the value unchanged.** Two paths diverging is visible only here.
    def same_value(L):
        x = L.tensor(plane)
        return F(L).max_pool2d(x, 2) - F(L).max_pool2d(x, 2, return_indices=True)[0]

    add("자리를 켜도 값은 같다", same_value)

    # ── unpooling ──
    def unpool(L, src, dim, **kw):
        pool = getattr(F(L), f"max_pool{dim}d")
        out, idx = pool(L.tensor(src), 2, return_indices=True)
        return getattr(F(L), f"max_unpool{dim}d")(out, idx, 2, **kw)

    add("되돌리기::1차원", lambda L: unpool(L, line, 1))
    add("되돌리기::2차원", lambda L: unpool(L, plane, 2))
    add("되돌리기::3차원", lambda L: unpool(L, cube, 3))
    add("되돌리기::여러 평면", lambda L: unpool(L, planes, 2))
    # What the pooling discarded at the edges cannot be restored, so torch opens a way to give the size directly.
    add("되돌리기::output_size",
        lambda L: unpool(L, plane, 2, output_size=(5, 5)))

    def unpool_stride(L):
        """Overlapping windows give overlapping restored positions — the later one wins (it does not add)."""
        x = L.tensor(plane)
        out, idx = F(L).max_pool2d(x, 2, stride=1, return_indices=True)
        return F(L).max_unpool2d(out, idx, 2, stride=1)

    add("되돌리기::겹치는 창", unpool_stride)

    # ── layers ──
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

    # ── the maximum's padding, which only its repr used to be asked about ──
    #
    # borch.ts refused `padding` and `ceil_mode` on the ground that the maximum's
    # backward reads the input at each window position and a padded position has none.
    # The average had the answer one function away all along: take the padding off the
    # coordinate and skip what falls outside. So the only case that existed was the
    # layer's repr — **which a refusal fails and a wrong answer passes**, because a
    # repr is a string about the arguments and not about the pooling.
    #
    # The three below are about the pooling. A 3×3 window with stride 2 over a 4×4
    # plane: with `padding=1` the corners are windows that hang off two sides at once,
    # and with `ceil_mode` there is an extra row and column of windows that start
    # inside and end outside. Both are places where a kernel that reads past the edge
    # gives a plausible number rather than an error.
    for _tag, _kw in (("padding", {"padding": 1}), ("ceil", {"ceil_mode": True}),
                      ("padding, ceil", {"padding": 1, "ceil_mode": True})):
        add(f"자리::max_pool2d({_tag})",
            lambda L, k=_kw: F(L).max_pool2d(L.tensor(plane), 3, stride=2, **k))
        add(f"자리::MaxPool2d({_tag})",
            lambda L, k=_kw: L.nn.MaxPool2d(3, stride=2, **k)(L.tensor(plane)))

    # **The gradient is where a wrong window shows.** Forward, a cell read past the
    # edge is one number among a maximum's; backward, the whole window's gradient goes
    # to whichever cell it decided had won, so reading the wrong cell moves the
    # gradient to the wrong place and the sum stays the same.
    def grad_padded(L):
        x = L.tensor(plane, requires_grad=True)
        F(L).max_pool2d(x, 3, stride=2, padding=1).sum().backward()
        return x.grad

    add("자리::grad::max_pool2d(padding)", grad_padded)

    # ── gradients ──
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

    # ── separating the stride from the window ──
    #
    # Pooling's `stride` becomes `kernel` when it is not given. **The defaults coincide**, so with
    # only cases that leave the two equal, an implementation that discards `stride` entirely and
    # one that puts it in `kernel`'s place both pass — the windows do not overlap, so the answer
    # is the same.
    #
    # Counting the table showed three of twelve places separating the two, and **all three were
    # 2-D.** The 1-D and 3-D cases, and average and L^p pooling, were only ever called with the
    # window and the stride equal. Give a narrower stride and one slot falls into several
    # windows, and then discarding the stride or miscounting the axis parts.
    overlap = (
        ("max_pool1d", line, lambda L, x: F(L).max_pool1d(x, 3, 1)),
        ("max_pool3d", cube, lambda L, x: F(L).max_pool3d(x, 3, 1)),
        ("avg_pool1d", line, lambda L, x: F(L).avg_pool1d(x, 3, 1)),
        ("avg_pool3d", cube, lambda L, x: F(L).avg_pool3d(x, 3, 1)),
        ("lp_pool1d", line / 8, lambda L, x: F(L).lp_pool1d(x, 2, 3, 1)),
        ("lp_pool2d", plane / 8, lambda L, x: F(L).lp_pool2d(x, 2, 3, 1)),
        ("lp_pool3d", cube / 64, lambda L, x: F(L).lp_pool3d(x, 2, 3, 1)),
    )
    for name, src, call in overlap:
        add(f"겹치는 창::{name}", lambda L, s=src, c=call: c(L, L.tensor(s)))

    # Unpooling has to take the window and stride **the same as the pooling** to retrace the
    # positions. Only 2-D was being asked — 1-D and 3-D have a different axis count and are not
    # the same code.
    def unpool_overlap(L, src, dim):
        pool = getattr(F(L), f"max_pool{dim}d")
        out, idx = pool(L.tensor(src), 3, 1, return_indices=True)
        return getattr(F(L), f"max_unpool{dim}d")(out, idx, 3, 1)

    add("겹치는 창::max_unpool1d", lambda L: unpool_overlap(L, line, 1))
    add("겹치는 창::max_unpool3d", lambda L: unpool_overlap(L, cube, 3))

    # ── LPPool3d ───────────────────────────────────────────────────────
    small = grid(1, 1, 4, 4, 4) / 8
    add("lp_pool3d", lambda L: F(L).lp_pool3d(L.tensor(small), 2, 2))
    add("lp_pool3d(p=1)", lambda L: F(L).lp_pool3d(L.tensor(small), 1, 2))
    add("층::LPPool3d", lambda L: L.nn.LPPool3d(2, 2)(L.tensor(small)))

    # ── fractional max pooling ──
    #
    # The samples shake where each window starts. There are two traps in the shape of the case
    # and **both were stepped on.**
    #
    # 1. **When it divides evenly the samples do nothing.** 6→3 with window 2 makes α exactly 2,
    #    so anything given produces the same answer — a shape in which the random part is wholly
    #    invisible. So it is asked as 7→3.
    # 2. **Giving both axes the same sample hides the axis order.** ATen reads the samples as
    #    (width, height) in the 2-D edition and (depth, height, width) in the 3-D one — the two
    #    functions are out of step with each other. It shows only with a different sample per axis.
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

    # **A different sample per axis** — reversing the order parts only here.
    axis_split = np.array([[[0.0, 0.75]]], dtype=np.float32)
    add("분수::축마다 다른 표본",
        lambda L: frac2(L, axis_split, output_size=(3, 3))[1])

    # A different sample per plane — the samples are `(N, C, axis)`, so the windows part per plane.
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
        """Window 3 with output 3 makes the windows overlap — the rule is the same when they do."""
        return F(L).fractional_max_pool2d(
            L.tensor(frac), 3, output_size=(3, 3), return_indices=True,
            _random_samples=L.tensor(zero2))[1]

    add("분수::겹치는 창", frac_overlap)

    # 3-D — the samples have to give three different answers per axis to part the order.
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
        # **The size is asked as 4.** Asked as 3 it matches the ratio case below
        # (7×0.5=3.5→3), so an implementation ignoring the ratio and using the size default passes.
        layer = L.nn.FractionalMaxPool2d(2, output_size=(4, 4),
                                         _random_samples=L.tensor(axis_split))
        return layer(L.tensor(frac))

    add("층::FractionalMaxPool2d", frac_layer)

    # **The ratio is only asked at the layer.** The function-side `output_ratio` case is above,
    # and the layer has the same argument with its own copy of the rule that turns an argument
    # into a size. 7×0.5 is 3.5, so **truncation and rounding part** (3 against 4) — asked at an
    # even size the two agree.
    def frac_layer_ratio(L):
        layer = L.nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.5),
                                         _random_samples=L.tensor(axis_split))
        return layer(L.tensor(frac))

    add("층::FractionalMaxPool2d(비율)", frac_layer_ratio)

    def frac_layer_refuses(L, **kw):
        try:
            L.nn.FractionalMaxPool2d(2, **kw)
            return "예외가 안 났다"
        except Exception as exc:                                # noqa: BLE001
            return type(exc).__name__

    # It takes **one** of the two. Taking both, which one won shows only in the value.
    add("층::FractionalMaxPool2d(둘 다 주면)",
        lambda L: frac_layer_refuses(L, output_size=(3, 3), output_ratio=(0.5, 0.5)))
    add("층::FractionalMaxPool2d(둘 다 없으면)", frac_layer_refuses)
    # **The `repr` is empty** — torch's `extra_repr` produces nothing.
    add("층::repr::FractionalMaxPool2d",
        lambda L: repr(L.nn.FractionalMaxPool2d(2, output_size=(3, 3))))
    add("층::repr::FractionalMaxPool3d",
        lambda L: repr(L.nn.FractionalMaxPool3d(2, output_size=(3, 3, 3))))

    def frac_random_shape(L):
        """Without samples it is random — **the value cannot be asked, so the shape and range are.**

        Whichever slot wins, its value is inside its own window, and the window is inside the
        input. So "the shape fits and every value was a number in the input" has to be true
        regardless of the draw.
        """
        out = L.nn.FractionalMaxPool2d(2, output_size=(3, 3))(L.tensor(frac))
        inside = ((out >= 0).float() * (out <= 48).float()).sum()
        return f"{tuple(out.shape)} 안에 있는 것={int(inside.item())}"

    add("분수::표본 없이(모양과 범위)", frac_random_shape)

    # ── CTC ──
    #
    # The loss that joins audio to letters without aligning positions. It sums over every
    # possible alignment.
    #
    # `reduction="mean"` is not the ordinary one — it **divides by each sample's own target
    # length** and then averages. Asked with cases whose target lengths are all equal, it gives
    # the same answer as a plain mean and that division is invisible. So the lengths are given
    # out of step, as 2 and 1.
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

    # The targets arrive as `(N, S)` and also as a concatenated 1-D — torch takes both.
    add("ctc::1차원 표적",
        lambda L: ctc(L, targets=np.array([1, 2, 3], dtype=np.int64),
                      reduction="none"))
    # The blank need not be 0.
    add("ctc::blank=3",
        lambda L: ctc(L, targets=np.array([[1, 2], [0, 0]], dtype=np.int64),
                      blank=3, reduction="none"))
    add("ctc::입력 길이가 다를 때",
        lambda L: ctc(L, il=np.array([5, 3], dtype=np.int64), reduction="none"))
    # **A blank always goes between repeated letters** — otherwise they collapse into one.
    add("ctc::반복 글자",
        lambda L: ctc(L, targets=np.array([[1, 1], [1, 1]], dtype=np.int64),
                      tl=np.array([2, 2], dtype=np.int64), reduction="none"))

    long_target = np.array([[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]], dtype=np.int64)
    short_in = np.array([2, 2], dtype=np.int64)
    six = np.array([6, 6], dtype=np.int64)
    # With no alignment at all the probability is 0 and the loss is `inf`. Not a threshold, a real condition.
    add("ctc::표적이 입력보다 길 때",
        lambda L: ctc(L, targets=long_target, il=short_in, tl=six, reduction="none"))
    add("ctc::zero_infinity",
        lambda L: ctc(L, targets=long_target, il=short_in, tl=six,
                      reduction="none", zero_infinity=True))

    def ctc_grad_logits(L):
        """The gradient flowed all the way to the logits — the shape real code has."""
        x = L.tensor(ctc_logits, requires_grad=True)
        F(L).ctc_loss(F(L).log_softmax(x, dim=2), L.tensor(ctc_targets),
                      L.tensor(ctc_in), L.tensor(ctc_tgt),
                      reduction="sum").backward()
        return x.grad

    add("ctc::grad(로짓까지)", ctc_grad_logits)

    def ctc_grad_logp(L):
        """**Where `log_probs` is made a leaf directly.**

        Here torch produces something that is not the true derivative — a finite difference gives
        `-γ` and torch gives `exp(log_probs) - γ`. The case above passes through `log_softmax`,
        which makes the two the same answer and **hides that difference.** Only this case sees it.
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

    # ── AdaptiveLogSoftmaxWithLoss ──
    #
    # softmax for a large vocabulary. Frequent letters come straight out of the head, and rare
    # ones come out as the probability of choosing a cluster **multiplied** by the probability
    # within it (added, in the log).
    #
    # The defaults are the trap — `div_value=4.0` and `head_bias=False`. Asked believing it is
    # 2.0, the tail layers have wholly different shapes. An intermediate dimension **falling to
    # 0 is normal too**, so the golden cases ask about that shape as well.
    asm_N, asm_D, asm_C = 6, 4, 12
    asm_x = (np.arange(asm_N * asm_D, dtype=np.float32).reshape(asm_N, asm_D) / 10) - 1
    asm_y = np.array([0, 1, 5, 7, 10, 11], dtype=np.int64)
    def asm_w(shape):
        """**Not random.** These weights have to be written identically in the TypeScript-side
        case, and a random generator does not cross languages. Being a countable value, both
        sides write the same thing."""
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
        """**The weights are planted** — a divergent initialisation makes a value comparison not a comparison."""
        model = L.nn.AdaptiveLogSoftmaxWithLoss(asm_D, asm_C, [3, 7],
                                                div_value=2.0, **kw)
        model.load_state_dict({k: L.tensor(v) for k, v in asm_weights.items()})
        return model

    add("적응형softmax::log_prob", lambda L: asm(L).log_prob(L.tensor(asm_x)))
    # **Each row's probabilities have to sum to 1** — leave the cluster-choice probability out
    # of the sum and it breaks here.
    add("적응형softmax::행 합이 1",
        lambda L: asm(L).log_prob(L.tensor(asm_x)).exp().sum(dim=1))
    add("적응형softmax::output",
        lambda L: asm(L)(L.tensor(asm_x), L.tensor(asm_y)).output)
    add("적응형softmax::loss",
        lambda L: asm(L)(L.tensor(asm_x), L.tensor(asm_y)).loss)
    add("적응형softmax::predict",
        lambda L: asm(L).predict(L.tensor(asm_x)))

    def asm_output_is_gathered(L):
        """`output` has to equal picking the correct slot out of `log_prob`.

        torch selects only the clusters it needs and produces it more cheaply, and if the two
        paths part, only the training goes slightly out of step. They are tied together by value.
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
        """**At the default `div_value=4.0`** a tail dimension can fall to 0.

        torch builds an empty layer there and moves on. Not blocking it is the imitation — the
        core once stopped dividing by √0.
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

    # ── what a layer prints ──────────────────────────────────────────────────────
    #
    # **This table held 196 repr strings and none for `Conv2d`, `BatchNorm2d` or the
    # pools** — the three most common layers in any first chapter. Measured when the
    # `ops` blocks were built: of sixty-four layers at their defaults, **twenty-nine
    # printed differently from torch**, and `_pool_repr` in the core carried a comment
    # saying so out loud — *nothing in the golden holds a `MaxPool` repr today*.
    #
    # A string nothing watches drifts. These are the strings.
    #
    # Two of the twenty-nine were not a printing difference at all, which is the part
    # worth keeping: **a pool built with no stride stores its kernel as the stride in
    # torch and stored `None` here**, and **`Upsample` keeps its factor as a float**.
    # `layer.stride` and `layer.scale_factor` are part of the surface, and comparing
    # the string is what looked at them.
    for _label, _build in (
            ("Conv1d", lambda L: L.nn.Conv1d(2, 3, 5)),
            ("Conv1d(padding, dilation, no bias)",
             lambda L: L.nn.Conv1d(2, 3, 5, padding=1, dilation=2, bias=False)),
            ("Conv2d", lambda L: L.nn.Conv2d(2, 3, 3)),
            ("Conv2d(stride, padding, groups)",
             lambda L: L.nn.Conv2d(4, 4, 3, stride=2, padding=1, groups=2)),
            ("Conv2d(no bias, reflect)",
             lambda L: L.nn.Conv2d(2, 3, 3, bias=False, padding_mode="reflect")),
            ("Conv3d", lambda L: L.nn.Conv3d(2, 3, 3)),
            ("ConvTranspose1d", lambda L: L.nn.ConvTranspose1d(2, 3, 5)),
            ("ConvTranspose2d(output_padding)",
             lambda L: L.nn.ConvTranspose2d(2, 3, 3, stride=2, output_padding=1)),
            ("BatchNorm1d", lambda L: L.nn.BatchNorm1d(4)),
            ("BatchNorm2d", lambda L: L.nn.BatchNorm2d(4)),
            ("BatchNorm2d(affine=False)",
             lambda L: L.nn.BatchNorm2d(4, affine=False)),
            ("BatchNorm2d(affine, no bias)",
             lambda L: L.nn.BatchNorm2d(4, bias=False)),
            ("BatchNorm3d(eps, momentum)",
             lambda L: L.nn.BatchNorm3d(4, eps=1e-3, momentum=0.2)),
            ("InstanceNorm1d", lambda L: L.nn.InstanceNorm1d(4)),
            ("InstanceNorm2d(affine)",
             lambda L: L.nn.InstanceNorm2d(4, affine=True)),
            ("GroupNorm", lambda L: L.nn.GroupNorm(2, 4)),
            ("GroupNorm(affine=False)",
             lambda L: L.nn.GroupNorm(2, 4, affine=False)),
            ("LayerNorm", lambda L: L.nn.LayerNorm(4)),
            ("LayerNorm(shape, no affine)",
             lambda L: L.nn.LayerNorm([2, 4], elementwise_affine=False)),
            ("MaxPool1d", lambda L: L.nn.MaxPool1d(2)),
            ("MaxPool2d", lambda L: L.nn.MaxPool2d(2)),
            ("MaxPool2d(stride, padding, ceil)",
             lambda L: L.nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)),
            ("MaxPool3d", lambda L: L.nn.MaxPool3d(2)),
            ("AvgPool1d", lambda L: L.nn.AvgPool1d(2)),
            ("AvgPool2d", lambda L: L.nn.AvgPool2d(2)),
            ("AvgPool2d(stride, padding)",
             lambda L: L.nn.AvgPool2d(3, stride=2, padding=1)),
            ("AvgPool3d", lambda L: L.nn.AvgPool3d(2)),
            ("LPPool1d", lambda L: L.nn.LPPool1d(2, 2)),
            ("LPPool2d(stride, ceil)",
             lambda L: L.nn.LPPool2d(3, 2, stride=2, ceil_mode=True)),
            ("AdaptiveAvgPool1d", lambda L: L.nn.AdaptiveAvgPool1d(1)),
            ("AdaptiveAvgPool2d", lambda L: L.nn.AdaptiveAvgPool2d(1)),
            ("AdaptiveAvgPool2d(pair)",
             lambda L: L.nn.AdaptiveAvgPool2d((2, 3))),
            ("AdaptiveMaxPool2d", lambda L: L.nn.AdaptiveMaxPool2d(1)),
            ("GELU", lambda L: L.nn.GELU()),
            ("GELU(tanh)", lambda L: L.nn.GELU(approximate="tanh")),
            ("Softmax", lambda L: L.nn.Softmax()),
            ("Softmax(dim=1)", lambda L: L.nn.Softmax(dim=1)),
            ("LogSoftmax(dim=-1)", lambda L: L.nn.LogSoftmax(dim=-1)),
            ("PReLU", lambda L: L.nn.PReLU()),
            ("PReLU(4)", lambda L: L.nn.PReLU(4)),
            ("Flatten", lambda L: L.nn.Flatten()),
            ("Flatten(0, -2)", lambda L: L.nn.Flatten(0, -2)),
            ("Unflatten", lambda L: L.nn.Unflatten(1, (2, 3))),
            ("Upsample(scale_factor)", lambda L: L.nn.Upsample(scale_factor=2)),
            ("Upsample(size)", lambda L: L.nn.Upsample(size=(4, 4))),
            ("Upsample(bilinear)",
             lambda L: L.nn.Upsample(scale_factor=2, mode="bilinear")),
    ):
        add("층::repr::" + _label, (lambda build: lambda L: repr(build(L)))(_build))

    return cases


DATACONV_PREFIX = "dataconv::"


def data_loader_cases(inp=None):
    """`DataLoader`'s arguments, which were seven of torch's seventeen.

    **Two of the seven were in the wrong seats.** `collate_fn` is torch's seventh and
    `drop_last` its ninth; here they were sixth and seventh, so
    `DataLoader(ds, 4, False, None, 0, True)` set `drop_last` on this side and handed
    `True` to `collate_fn` on torch's — which then tries to call it. Every case below
    passes its arguments **by position**, because a keyword call cannot see that.

    `generator` is the one worth the most. Without it a shuffled loader cannot be made
    to repeat, which is the first thing anybody wants from a dataset — and it is also
    why no case had ever frozen a shuffled order: there was no way to make one twice.
    It is asked as *the same seed twice and then a different seed*, since a single
    draw agrees with any implementation that draws at all.
    """
    del inp
    cases = []
    items = list(range(10))

    def order(L, seed):
        g = L.Generator()
        g.manual_seed(seed)
        loader = L.utils.data.DataLoader(items, 3, True, None, None, 0, None, False,
                                         False, 0, None, None, g)
        return " | ".join(",".join(str(int(v)) for v in b) for b in loader)

    # **As text, not as a boolean.** The harness takes a value out with
    # `.detach().numpy()` and a `bool` has neither, so a comparison written as an
    # answer stops in the dumper rather than being compared.
    cases.append((DATACONV_PREFIX + "DataLoader(generator)/같은 씨앗은 같은 순서",
                  lambda L: str(order(L, 7) == order(L, 7))))
    cases.append((DATACONV_PREFIX + "DataLoader(generator)/다른 씨앗은 다른 순서",
                  lambda L: str(order(L, 7) != order(L, 11))))

    # **The nine random fillers took `generator` and dropped it**, each opening with
    # `del generator`, so `x.normal_(generator=g)` drew from the global stream and the
    # caller's reproducibility was quietly somebody else's. `bernoulli` two hundred
    # lines away had done the right thing in one line the whole time.
    #
    # The values cannot be frozen — the generators differ between the three sides — so
    # what is asked is the **property that separates honouring from discarding**, and
    # the first attempt did not: "same seed, same values" is true of a global stream
    # too. This shakes the global stream *between* the seed and the draw. A filler that
    # uses the generator it was handed does not feel it; one that reaches for the
    # global does.
    def steady(L, fill):
        def once(noise):
            g = L.Generator()
            g.manual_seed(7)
            for _ in range(noise):
                L.rand(3)                 # only the global stream moves
            out = fill(L.zeros(6), g)
            return [float(v) for v in out.detach().numpy().ravel()]
        return str(once(0) == once(5))

    for _fname, _fill in (
            ("normal_", lambda t, g: t.normal_(0.0, 1.0, generator=g)),
            ("uniform_", lambda t, g: t.uniform_(0.0, 1.0, generator=g)),
            ("exponential_", lambda t, g: t.exponential_(1.0, generator=g)),
            ("cauchy_", lambda t, g: t.cauchy_(0.0, 1.0, generator=g)),
            ("log_normal_", lambda t, g: t.log_normal_(1.0, 2.0, generator=g)),
            ("geometric_", lambda t, g: t.geometric_(0.5, generator=g)),
            ("random_", lambda t, g: t.random_(0, 100, generator=g)),
            ("bernoulli_", lambda t, g: t.bernoulli_(0.5, generator=g)),
            ("bernoulli", lambda t, g: t.bernoulli(0.5, generator=g))):
        cases.append((DATACONV_PREFIX + f"{_fname}(generator)/전역이 흔들려도 그대로",
                      lambda L, f=_fill: steady(L, f)))
    # Positional, and past the two seats that were swapped: the sixth is
    # `num_workers` and the ninth is `drop_last`.
    cases.append((DATACONV_PREFIX + "DataLoader(자리로 준 drop_last)",
                  lambda L: " | ".join(
                      ",".join(str(int(v)) for v in b)
                      for b in L.utils.data.DataLoader(items, 4, False, None, None, 0,
                                                       None, False, True))))
    cases.append((DATACONV_PREFIX + "DataLoader(batch_sampler)",
                  lambda L: " | ".join(
                      ",".join(str(int(v)) for v in b)
                      for b in L.utils.data.DataLoader(
                          items, batch_sampler=L.utils.data.BatchSampler(
                              L.utils.data.SequentialSampler(items), 4, True)))))
    return cases


def default_convert_cases(inp=None):
    """`utils.data.default_convert` — **what changes and what does not.**

    The counterpart to `default_collate`, similarly named and with a different rule. Two traps,
    and neither is visible when asked by value.

    - **A tuple becomes a list.** Backward compatibility torch itself left behind. By value the
      elements are the same so it passes, and the code unpacking it afterwards as `a, b = ...`
      works identically on a list, so it parts a long way downstream.
    - **A Python number does not change.** `3` comes out as `3`. The similarly named
      `default_collate` folds numbers into tensors, so it looks like it should be the same and
      is not.

    So **what it came out as** is frozen as the answer, rather than the value.
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
        """The values have to be carried too — change the shell and drop the numbers and the case above passes."""
        conv = L.utils.data.default_convert
        pair = conv((one, two))
        inside = conv({"a": one})["a"]
        return L.stack([pair[0], pair[1], inside])

    add("값이 그대로 온다", values)

    def worker(L):
        """**`None` in the main process.** There are no workers here, so it always is."""
        return str(L.utils.data.get_worker_info())

    add("get_worker_info", worker)
    return cases


METHOD2_PREFIX = "method2::"


def method_name_cases(inp=None):
    """**One computation under two names** — `torch.add(x, y)` and `x.add(y)`.

    This repository had a loop in one direction only: sweep the tensor methods and build the
    module functions. There was no reverse, so **the computation was all there and the name
    reached it from one side only** — `borch.matrix_exp(x)` worked and `x.matrix_exp()` did not.
    `x.add(y)` is a very common shape in textbook code, and what comes out then is an
    `AttributeError`.

    ## The in-place operations are the same story

    There were forty-seven places where `abs` existed and only `abs_` was missing. The
    computation belongs to the underscore-less side and this only **writes back into its own
    buffer** — keeping the same expression in two copies makes them diverge.

    ## What is asked here is **whether the two give the same answer**

    Looking only at whether the name resolves lets a hollow method pass. Asking the value is
    what reveals whether the name really reached that computation. Whether the name list is
    genuinely torch's is looked at separately by `tests/test_tensor_api.py` — inventing a name
    that does not exist means code written against it does not run on real torch.
    """
    cases = []

    def add(name, fn):
        cases.append((METHOD2_PREFIX + name, fn))

    a = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    b = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    sym = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    neg = np.array([[-1., 2.], [-3., 0.5]], dtype=np.float32)

    # ── the answer as called through the method ──
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
        # kron only does 1-D in the miniature — it is asked separately below.
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

    # **It has to equal what the function form gives.** A name that resolves to a different
    # computation parts here.
    def same_as_function(L):
        x, y = L.tensor(a), L.tensor(b)
        checks = (x.add(y) - L.add(x, y), x.mul(y) - L.mul(x, y),
                  x.det().reshape(1) - L.det(x).reshape(1),
                  (x.matrix_exp() - L.matrix_exp(x)).reshape(-1))
        return L.cat([c.reshape(-1) for c in checks])

    add("함수와 같은 답", same_as_function)

    # ── in-place operations ──
    # **A fresh array per case.** The core's `tensor(ndarray)` **shares** that buffer (torch's
    # `from_numpy` side of things) while torch's `tensor()` copies. Keeping one array outside for
    # all of them let the first in-place operation edit it, and the later cases then received
    # **a different input** from the one torch saw — thirteen parted at once and the cause was
    # not the in-place operations.
    def _small():
        return np.array([[0.25, 0.5], [0.75, -0.5]], dtype=np.float32)

    def _square():
        return np.array([[1., 2.], [3., 4.]], dtype=np.float32)

    inplace = ("absolute", "acosh", "arctan", "arctanh", "asinh", "atanh",
               "deg2rad", "erfc", "exp2", "fix", "negative", "rad2deg",
               "sgn", "sinc", "logit")
    for name in inplace:

        def run(L, n=name):
            # acosh has an answer from 1 up and logit only within 0..1 — outside, both sides give
            # NaN, and NaN cannot be called equal to anything, so asking reveals nothing.
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
        """The in-place operations that take an argument — only the arity differs, the rest is the same."""
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
        """**Some in-place operations change the shape** — they edit the frame it is seen through
        rather than the values.

        Asked on a square, **it passes without the shape changing at all.** Asked at 2×2,
        `transpose_` really did nothing and stayed green. Here a rectangle is given and the shape
        itself is the answer — because the frame, not the value, is the question.
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
        """Not only the shape — **the values have to move with it.**"""
        x = L.tensor(_rect())
        x.transpose_(0, 1)
        return x

    add("제자리::transpose_ 의 값", inplace_transpose_values)

    def inplace_is_same_object(L):
        """**In place edits the same tensor.** Handing back a new one is meaningless."""
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


TOP10_PREFIX = "top::"


def top_rest_cases(inp=None):
    """The names still left at the top level — `tests/torch_gap.py`'s last review batch.

    **This group's lesson is that counting by name gets it wrong.** `fake_quantize_*` was counted
    as a refusal because the name says quantisation, and it **takes a float and returns a float**
    — no quantised dtype needed. `dequantize` is the identity on a real too. Conversely
    `BufferDict` reads like `nn.ParameterDict`'s counterpart and is **TorchScript internals**, so
    it is a refusal.

    ## What cannot be asked

    `resize_as_` is asked **by shape only.** torch does not initialise the added slots either
    (measured: the same code gives the same values every time, and that is not a promise), so
    freezing the value would stuff that implementation's accident into the specification.
    """
    grid = np.array([-1.7, 0.3, 2.9, 5.5], dtype=np.float32)
    shapes = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    spots = np.array([0.25, 1.5, 0.5, 4.0], dtype=np.float32)
    steps = np.array([1.0, 2.0, 3.5], dtype=np.float32)
    cases = []

    def add(name, fn):
        cases.append((TOP10_PREFIX + name, fn))

    def X(L):
        return L.tensor(grid.copy())

    add("igamma", lambda L: L.igamma(L.tensor(shapes.copy()),
                                     L.tensor(spots.copy())))
    # **One formula does not cover it** — `x < a+1` is a series and beyond it a continued
    # fraction. Asked at small x only, that fork is invisible.
    add("igamma(큰 x)", lambda L: L.igamma(L.tensor(shapes.copy()),
                                          L.tensor(shapes.copy() * 8)))
    add("igammac", lambda L: L.igammac(L.tensor(shapes.copy()),
                                       L.tensor(spots.copy())))
    add("igamma + igammac = 1",
        lambda L: L.igamma(L.tensor(shapes.copy()), L.tensor(spots.copy()))
        + L.igammac(L.tensor(shapes.copy()), L.tensor(spots.copy())))
    for n in (0, 1, 2, 3):
        add(f"polygamma({n})",
            lambda L, k=n: L.polygamma(k, L.tensor(steps.copy())))
    add("constant_pad_nd", lambda L: L.constant_pad_nd(X(L), [1, 2], 9.0))
    add("fake_quantize(per_tensor)",
        lambda L: L.fake_quantize_per_tensor_affine(X(L), 0.5, 0, 0, 7))
    # Moving the zero point moves where it clips — asked at 0 only, that argument can be dead
    # and unseen.
    add("fake_quantize(zp=2)",
        lambda L: L.fake_quantize_per_tensor_affine(X(L), 0.5, 2, 0, 7))
    add("fake_quantize(per_channel)",
        lambda L: L.fake_quantize_per_channel_affine(
            L.tensor(grid.reshape(2, 2).copy()),
            L.tensor(np.array([0.5, 0.25], dtype=np.float32)),
            L.tensor(np.array([0.0, 1.0], dtype=np.float32)), 0, 0, 7))
    add("dequantize", lambda L: L.dequantize(X(L)))

    def grad(name, fn, data):
        def run(L, f=fn, d=data, n=name):
            leaf = L.tensor(d.copy(), requires_grad=True)
            f(L, leaf).sum().backward()
            return _grad_of(leaf, n)

        add(f"grad::{name}", run)

    grad("igamma / x",
         lambda L, t: L.igamma(L.tensor(shapes.copy()), t), spots)
    grad("igammac / x",
         lambda L, t: L.igammac(L.tensor(shapes.copy()), t), spots)
    grad("polygamma(1)", lambda L, t: L.polygamma(1, t), steps)
    grad("constant_pad_nd", lambda L, t: L.constant_pad_nd(t, [1, 2], 9.0), grid)
    # **Outside the range it is 0** — rounding is a staircase so the derivative is 0 almost
    # everywhere, and torch treats the inside of the range as "pass straight through". Otherwise
    # training does not move at all.
    grad("fake_quantize", lambda L, t:
         L.fake_quantize_per_tensor_affine(t, 0.5, 0, 0, 7), grid)

    def refuses(name, body):
        def run(L, f=body):
            try:
                f(L)
                return "예외가 안 났다"
            except Exception as exc:                            # noqa: BLE001
                return type(exc).__name__

        add(name, run)

    # It does not differentiate with respect to the first argument — **torch itself refuses**
    # (there is no closed form).
    def igamma_on_a(L):
        a = L.tensor(shapes.copy(), requires_grad=True)
        L.igamma(a, L.tensor(spots.copy())).sum().backward()

    refuses("igamma 는 a 로 안 미분한다", igamma_on_a)

    # ── device ──
    #
    # **Building one and using one are separated.** `torch.device("cuda")` is built even with no
    # such hardware (measured) — stopping there stops the ternary on the first line of the
    # tutorials from running at all.
    add("device::str", lambda L: str(L.device("cpu")))
    add("device::repr", lambda L: repr(L.device("cpu")))
    add("device::type", lambda L: str(L.device("cpu").type))
    add("device::번호 있는 것", lambda L: str(L.device("cpu", 0)))
    add("device::문자열에서 번호", lambda L: str(L.device("cpu:1")))
    add("device::cuda 도 만들어진다", lambda L: repr(L.device("cuda")))
    add("device::같음", lambda L: str(L.device("cpu") == L.device("cpu")))
    # **It does not equal a string** (measured). Being lenient parts the branch of `if d == "cpu":`.
    add("device::문자열과는 다름", lambda L: str(L.device("cpu") == "cpu"))
    add("device::to(device) 모양",
        lambda L: str(tuple(int(n) for n in X(L).to(L.device("cpu")).shape)))

    def resized(L, want, start):
        t = L.tensor(start.copy())
        L.resize_as_(t, L.zeros(*want))
        return t

    add("resize_as_::늘린 모양",
        lambda L: str(tuple(int(n) for n in resized(
            L, (2, 3), np.array([1.0, 2.0], dtype=np.float32)).shape)))
    add("resize_as_::줄인 모양",
        lambda L: str(tuple(int(n) for n in resized(
            L, (2,), np.arange(6, dtype=np.float32)).shape)))
    # Shrinking keeps the front — that is fixed (measured).
    add("resize_as_::줄이면 앞이 남는다",
        lambda L: resized(L, (2,), np.arange(6, dtype=np.float32)))
    return cases


RNNTOP_PREFIX = "rnntop::"


def rnn_top_cases(inp=None):
    """The eight top-level recurrences — `torch.lstm` and its siblings.

    **The same computation as the layer, taking its weights as a list.** torch gives both, and
    what the layer calls inside is this function. So whether the values agree is already asked by
    the layer cases, and what is asked here is **how the arguments come in and how the results
    go out.**

    ## Three places to part

    * **The list order** — `[w_ih, w_hh, b_ih, b_hh]` per layer. Without biases it is two each
      and the chunk size changes.
    * **The output shape** — `lstm` **spreads three**, as `(output, h_n, c_n)`. The layer side
      bundles as `(output, (h, c))` and the top level does not (measured). Handed over bundled,
      the receiving side's unpacking is off by one slot.
    * **The gate order** — `weight_ih`'s rows are `i,f,g,o` for LSTM and `r,z,n` for GRU.
      Reorder them and **the shape holds while only the value** parts.

    Bidirectional and inter-layer dropout are refused, our layers not having them. The former is
    caught loudly because the shape halves, and the latter parts with a plausible value, so both
    are asked as cases.
    """
    # **The inputs moved into `golden_inputs()`.** Drawn here they do not go into `golden.json`,
    # so borch.ts cannot be given the same weights — the values are still compared through the
    # binding while borch.ts's **direct surface** (the argument order) ends up asked by nobody.
    inp = golden_inputs() if inp is None else inp
    x, xb, xs = inp["rt_x"], inp["rt_xb"], inp["rt_xs"]
    h1, c1 = inp["rt_h1"], inp["rt_c1"]
    h2, c2 = inp["rt_h2"], inp["rt_c2"]
    hs, cs = inp["rt_hs"], inp["rt_cs"]
    cases = []

    def add(name, fn):
        cases.append((RNNTOP_PREFIX + name, fn))

    def weights(prefix, count):
        return [inp[f"rt_{prefix}{i}"] for i in range(count)]

    SPEC = [("lstm", 4), ("gru", 3), ("rnn_tanh", 1), ("rnn_relu", 1)]
    for name, gates in SPEC:
        ws = weights(f"{name}_w", 4)
        two = weights(f"{name}_two", 8)

        def call(L, n=name, w=ws, data=x, layers=1, biases=True,
                 batch_first=False, state=None):
            tens = [L.tensor(v.copy()) for v in (w if biases else w[:2])]
            hx = state(L)
            return getattr(L, n)(L.tensor(data.copy()), hx, tens, biases,
                                 layers, 0.0, False, False, batch_first)

        def state1(L, n=name):
            return ((L.tensor(h1.copy()), L.tensor(c1.copy()))
                    if n == "lstm" else L.tensor(h1.copy()))

        def state2(L, n=name):
            return ((L.tensor(h2.copy()), L.tensor(c2.copy()))
                    if n == "lstm" else L.tensor(h2.copy()))

        # Two things (or three) come out, so **each piece is named** — looking at one alone
        # leaves the rest uncaught.
        pieces = 3 if name == "lstm" else 2
        for k in range(pieces):
            add(f"{name}[{k}]",
                lambda L, f=call, i=k, s=state1: f(L, state=s)[i])
        add(f"{name}(batch_first)",
            lambda L, f=call, s=state1: f(L, data=xb, batch_first=True, state=s)[0])
        add(f"{name}(has_biases=False)",
            lambda L, f=call, s=state1: f(L, biases=False, state=s)[0])
        add(f"{name}(num_layers=2)",
            lambda L, f=call, w=two, s=state2: f(L, w=w, layers=2, state=s)[0])
        add(f"{name}(num_layers=2) 마지막 상태",
            lambda L, f=call, w=two, s=state2: f(L, w=w, layers=2, state=s)[1])

    CELLS = [("lstm_cell", 4), ("gru_cell", 3),
             ("rnn_tanh_cell", 1), ("rnn_relu_cell", 1)]
    for name, gates in CELLS:
        ws = weights(f"{name}_w", 4)

        def cell(L, n=name, w=ws, biases=True):
            tens = [L.tensor(v.copy()) for v in w]
            hx = ((L.tensor(hs.copy()), L.tensor(cs.copy()))
                  if n == "lstm_cell" else L.tensor(hs.copy()))
            args = tens[:4] if biases else tens[:2]
            return getattr(L, n)(L.tensor(xs.copy()), hx, *args)

        if name == "lstm_cell":
            for k in range(2):
                add(f"{name}[{k}]", lambda L, f=cell, i=k: f(L)[i])
        else:
            add(name, lambda L, f=cell: f(L))
        add(f"{name}(편향 없이)",
            lambda L, f=cell, n=name: (f(L, biases=False)[0]
                                       if n == "lstm_cell"
                                       else f(L, biases=False)))

    # The `refuses` helper here was defined with nothing calling it. The trace of an attempt to
    # ask about the bidirectional and dropout refusals, abandoned at "torch manages both, so the
    # names part".
    lw = weights("drop_w", 4)

    def try_lstm(L, dropout=0.0, train=False, bidirectional=False):
        tens = [L.tensor(v.copy()) for v in lw]
        hx = (L.tensor(h1.copy()), L.tensor(c1.copy()))
        return L.lstm(L.tensor(x.copy()), hx, tens, True, 1, dropout, train,
                      bidirectional, False)

    # **The kind of refusal is not asked** — torch manages these two, so the names part. What is
    # asked here is whether we **quietly produce half an answer**, so the golden entry is a place
    # where torch runs too, rather than one where only our answer is frozen.
    add("dropout=0 이면 돈다",
        lambda L: try_lstm(L, dropout=0.0, train=True)[0])
    return cases


def cell_cases(inp=None):
    """The three RNN cells — **one step** of the recurrence.

    `RNN`, `LSTM` and `GRU` take the whole time axis. A cell takes one step, so code that wants to
    write the time loop by hand (scheduling, teacher forcing, beam search) calls these names.

    ## The names differ from the layers'

    A layer numbers them, as `weight_ih_l0`, and a cell is `weight_ih` — because a cell has no
    layers. Those names are the `state_dict` keys, so **getting them wrong makes checkpoints not fit.**

    ## `LSTMCell` alone returns two

    `(h, c)`. `RNNCell` and `GRUCell` return `h` alone — force the three into one shape and
    LSTM's memory cell disappears, and then values come out and training does not work.

    ## The gate order is the whole of the value

    `weight_ih` is `(3H, in)` or `(4H, in)`, and the order within it is the convention — GRU is
    `r, z, n` and LSTM is `i, f, g, o`. Reorder them and the shape is the same and only the value
    parts. That is why the weights are pinned and the value asked.
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
        """**Without a hidden state given it starts from 0.** That is the shape of the first step."""
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

    # ── names and characters ──
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

    # ── gradients ──
    def cell_grad(L, name, gates):
        # **The state's shape differs per cell.** `LSTMCell` alone takes an `(h, c)` pair and
        # the rest take `h` alone — lump them into one and torch refuses on the spot.
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
    """The nine remaining layers — the two that unfold windows, and the rest.

    ## `Unfold` and `Fold` are not each other's inverse

    `Unfold` lays windows out as columns and `Fold` folds them back, and **it adds the
    overlaps.** Unfolding a 4×4 with a 2×2 window and folding it straight back counts the middle
    values four times and does not give the original (measured:
    `[[0,2,4,3],[8,20,24,14],…]`). Read as an undo, it is a place that goes quietly wrong.

    Summing being the convention, **the backward pass comes out right of its own accord** —
    `Unfold`'s backward pass is `Fold` and the reverse holds too. Written with one index, the two
    become one machine.

    ## `LocalResponseNorm`'s window is off-centre

    Channel `c`'s window is `[c − n//2, c + n − 1 − n//2]`. At `size=2` that is `{c−1, c}` and not
    `{c, c+1}` — measured. Centre it and the values shift by one slot, and being the same size,
    the shape does not show it.

    ## `RReLU`'s gradient is determined in eval mode

    Training draws from `[lower, upper]` and eval uses **the middle** — on the defaults,
    `(1/8 + 1/3)/2 = 0.2292`. Only the side with no randomness is asked by value.

    ## `UpsamplingBilinear2d` is `align_corners=True`

    `Upsample(mode='bilinear')` defaults to `False`, so the values differ. Treated as aliases on
    the strength of the name, the edges go out of line while the interior stays similar enough
    that the eye does not part them.
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

    # ── unfolding windows and folding them back ──
    add("unfold", lambda L: F(L).unfold(L.tensor(img), 2))
    add("unfold(stride=2)", lambda L: F(L).unfold(L.tensor(img), 2, stride=2))
    add("unfold(padding=1)", lambda L: F(L).unfold(L.tensor(img), 2, padding=1))
    add("unfold(채널 셋)", lambda L: F(L).unfold(L.tensor(img3), 2))
    # **Folding back adds the overlaps.** Not recovering the original is the convention.
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
        # **Put in through `load_state_dict`.** `.data =` means different things per library
        # (torch gives a tensor, our core a numpy array), so the case body would take one side —
        # this table already avoids that for the same reason.
        layer = L.nn.Bilinear(3, 4, 2)
        layer.load_state_dict({"weight": L.tensor(w), "bias": L.tensor(bias)})
        return layer(L.tensor(a1), L.tensor(a2))

    add("층::Bilinear", bilinear_layer)
    add("repr::Bilinear", lambda L: repr(L.nn.Bilinear(3, 4, 2)))

    # ── LocalResponseNorm ───────────────────────────────────────────────
    add("local_response_norm",
        lambda L: F(L).local_response_norm(L.tensor(chans), 2))
    # On the defaults the difference is tiny and cannot part the window's position. Asked with a larger alpha.
    add("local_response_norm(alpha=1)",
        lambda L: F(L).local_response_norm(L.tensor(chans), 2, alpha=1.0,
                                           beta=1.0, k=1.0))
    add("local_response_norm(size=3)",
        lambda L: F(L).local_response_norm(L.tensor(chans), 3, alpha=1.0,
                                           beta=1.0, k=1.0))
    add("층::LocalResponseNorm",
        lambda L: L.nn.LocalResponseNorm(2)(L.tensor(chans)))
    add("repr::LocalResponseNorm", lambda L: repr(L.nn.LocalResponseNorm(2)))
    # **The default `alpha` is 1e-4 and prints the same in both languages**, so the row
    # above could never see that borch.ts was interpolating `alpha` without the
    # decimal-point guard its two neighbours went through. At a whole number torch says
    # `alpha=1.0` and that side said `alpha=1`.
    #
    # A case for an argument has to be one whose default answer would be wrong — **and
    # wrong by construction, not by arithmetic luck.** The row above is not testing
    # `alpha` at all; it is testing that torch happened to pick a fractional default.
    # Change that default upstream and a case nobody touched turns red, which is the
    # tell that it was never asking the question its name claims.
    add("repr::LocalResponseNorm(alpha=1.0)",
        lambda L: repr(L.nn.LocalResponseNorm(2, alpha=1.0, beta=2.0, k=2.0)))

    # ── Softmax2d ───────────────────────────────────────────────────────
    add("층::Softmax2d", lambda L: L.nn.Softmax2d()(L.tensor(cube)))
    add("Softmax2d 는 softmax(dim=1)",
        lambda L: L.nn.Softmax2d()(L.tensor(cube)) - F(L).softmax(L.tensor(cube),
                                                                  dim=1))
    add("repr::Softmax2d", lambda L: repr(L.nn.Softmax2d()))

    # ── RReLU — the side with no randomness only ──
    neg = np.array([[-1., -2., 1.]], dtype=np.float32)
    add("rrelu(eval)", lambda L: F(L).rrelu(L.tensor(neg), training=False))
    add("층::RReLU(eval)", lambda L: L.nn.RReLU().eval()(L.tensor(neg)))
    add("rrelu(eval, 범위 지정)",
        lambda L: F(L).rrelu(L.tensor(neg), lower=0.2, upper=0.4,
                             training=False))
    add("repr::RReLU", lambda L: repr(L.nn.RReLU()))
    # **The defaults here are `1/8` and `1/3`, so the case above can never see a
    # missing decimal point.** JavaScript prints `1.0` as `1`, and borch.ts printed
    # both of these bare — the row stayed green because no value it was ever given
    # was integral. That is not a case for `lower` and `upper`; it is a case for
    # torch having picked fractional defaults, and it would turn red on a change
    # upstream that nobody here made.
    #
    # Wrong by construction, not by arithmetic luck: integral values, where the two
    # spellings differ.
    add("repr::RReLU(정수 경계)", lambda L: repr(L.nn.RReLU(1.0, 2.0)))

    # ── Upsampling — the two old names ──
    #
    # **`size` is asked with a target that is not a whole multiple.** Both layers take
    # `(size, scale_factor)` in torch and exactly one of the two; a target of 4×4 on a
    # 2×2 input is the same answer a scale of 2 gives, so it would pin nothing. 3×5
    # separates them, and the nearest form's rounding is the part worth freezing.
    add("층::UpsamplingNearest2d(크기)",
        lambda L: L.nn.UpsamplingNearest2d(size=(3, 5))(L.tensor(small)))
    add("층::UpsamplingBilinear2d(크기)",
        lambda L: L.nn.UpsamplingBilinear2d(size=(3, 5))(L.tensor(small)))
    add("층::UpsamplingNearest2d",
        lambda L: L.nn.UpsamplingNearest2d(scale_factor=2)(L.tensor(small)))
    add("층::UpsamplingBilinear2d",
        lambda L: L.nn.UpsamplingBilinear2d(scale_factor=2)(L.tensor(small)))
    # **It is `align_corners=True`.** Differing from `Upsample`'s default is the point, so that
    # difference is asked by value.
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

    # **`max_norm` shrinks the table in place, and no case had ever looked at the
    # table.** Every instrument here compares what a call returns; a call that also
    # changes the thing it was called on is invisible to all of them.
    #
    # And the invisibility does not go away by calling twice. Re-normalising a row that
    # is already short enough is a no-op, so an implementation that normalised a *copy*
    # would return the same numbers as this one **forever** — the divergence lives in
    # `weight` and nowhere else, until something trains on it.
    def bag_renorm(L, what):
        layer = L.nn.EmbeddingBag(5, 3, max_norm=1.0, mode="sum")
        layer.load_state_dict({"weight": L.tensor(table)})
        out = layer(L.tensor(bags))
        return layer.weight if what == "weight" else out

    add("층::EmbeddingBag(max_norm)", lambda L: bag_renorm(L, "out"))
    # The table **after** the call. This is the case the block above is about.
    add("층::EmbeddingBag(max_norm)/표가 줄었다", lambda L: bag_renorm(L, "weight"))

    def bag_renorm_untouched(L):
        """**Only the indexed rows are shrunk.** Rows 4 is never asked for and keeps
        its own size — a version that normalised the whole table would agree on every
        output and differ only here."""
        layer = L.nn.EmbeddingBag(5, 3, max_norm=1.0, mode="sum")
        layer.load_state_dict({"weight": L.tensor(table)})
        layer(L.tensor(np.array([[0, 1]], dtype=np.int64)))
        return layer.weight

    add("층::EmbeddingBag(max_norm)/색인 안 된 행", bag_renorm_untouched)

    def bag_renorm_norm_type(L):
        """`norm_type=1` measures the row differently, so a different amount comes
        off. Left at the default this argument is unasked and an implementation that
        ignores it passes the two cases above."""
        layer = L.nn.EmbeddingBag(5, 3, max_norm=1.0, norm_type=1.0, mode="sum")
        layer.load_state_dict({"weight": L.tensor(table)})
        layer(L.tensor(bags))
        return layer.weight

    add("층::EmbeddingBag(max_norm, norm_type=1)", bag_renorm_norm_type)

    # ── Embedding ───────────────────────────────────────────────────────
    # It took `(num_embeddings, embedding_dim)` and torch takes nine more. A name
    # count could not see that — `torch_gap.py` counts names and `Embedding` was
    # present — and `EmbeddingBag` next door had carried torch's full list all along.
    def emb(L, what, **kw):
        layer = L.nn.Embedding(4, 3, **kw)
        layer.load_state_dict({"weight": L.tensor(table[:4])})
        out = layer(L.tensor(np.array([0, 1, 2], dtype=np.int64)))
        return layer.weight if what == "weight" else out

    add("층::Embedding(padding_idx)", lambda L: emb(L, "out", padding_idx=1))
    add("층::Embedding(max_norm)", lambda L: emb(L, "out", max_norm=1.0))
    add("층::Embedding(max_norm)/표가 줄었다", lambda L: emb(L, "weight", max_norm=1.0))
    add("층::Embedding(max_norm, norm_type=1)/표",
        lambda L: emb(L, "weight", max_norm=1.0, norm_type=1.0))
    add("repr::Embedding(전부)",
        lambda L: repr(L.nn.Embedding(4, 3, padding_idx=1, max_norm=2.0)))

    def emb_padding_grad(L):
        """**The padding row learns nothing.** Left in, a pad token drifts toward
        whatever the loss wants and the mask stops meaning "ignore this" — while the
        forward stays right, so nothing else here would see it.

        Measured, by cutting the gradient block out of borch.ts: this case reddens and
        the four value cases above stay green. That is the separation the docstring
        claims, confirmed rather than asserted.

        **The sentence next door was wrong and is corrected in `nn.ts`.** It said an
        implementation masking the *output* instead would pass every value case and
        fail only here. It does not — torch hands back the padding row's real values,
        so masking the output changes the answer and `층::Embedding(padding_idx)`
        catches it. Two different defects, each caught by a different case, and the
        comment had them as one.
        """
        layer = L.nn.Embedding(4, 3, padding_idx=1)
        layer.load_state_dict({"weight": L.tensor(table[:4])})
        layer(L.tensor(np.array([0, 1, 2, 1], dtype=np.int64))).sum().backward()
        return layer.weight.grad

    add("grad::Embedding(padding_idx)", emb_padding_grad)

    def emb_fresh_padding(L):
        """A **fresh** table zeroes the padding row and a **given** one does not.
        torch draws the line at the same place, and both halves are asked because
        either alone reads as a rule about padding rather than about who supplied
        the weights."""
        fresh = L.nn.Embedding(4, 3, padding_idx=2).weight
        given = L.nn.Embedding(4, 3, padding_idx=2, _weight=L.tensor(table[:4])).weight
        return L.tensor(np.concatenate([np.asarray(to_numpy(fresh))[2],
                                        np.asarray(to_numpy(given))[2]]))

    add("층::Embedding(padding_idx)/새 표와 준 표", emb_fresh_padding)
    return cases


SHUFFLE_PREFIX = "shuffle::"


def shuffle_cases(inp=None):
    """Three layers that move positions, and the five dropouts that **drop whole channels.**

    ## `Dropout2d` drops channels rather than elements

    Sitting beside `Dropout` in the name, it reads as "the 2-D one", and it does something else —
    it makes a whole channel 0 or leaves it whole. So an answer with a channel mixed inside is
    element-wise dropout, and one value alone cannot tell them apart. That is why the golden case
    asks "is the inside of the channel all the same".

    ## `AlphaDropout` does not insert zeros

    Built to be used with SELU, it puts **a negative constant** where it drops and applies an
    affine transform over the whole thing to preserve the mean and variance. With an input of all
    ones, the answer had two values, `-0.779` and `1.666` (measured). Inserting zeros breaks
    SELU's self-normalisation, and being plausible, that stays invisible while training runs.

    ## The position-moving ones are asked by value

    `PixelShuffle`, `PixelUnshuffle` and `ChannelShuffle` have no randomness, so the values are
    asked as they are. With an `arange` input, **which position went where** is written directly
    into the answer.
    """
    cases = []

    def add(name, fn):
        cases.append((SHUFFLE_PREFIX + name, fn))

    def F(L):
        return L.nn.functional

    # ── moving positions — by value ──
    pix = np.arange(8 * 2 * 2, dtype=np.float32).reshape(1, 8, 2, 2)
    flat = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    chan = np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1)
    chan6 = np.arange(6 * 2, dtype=np.float32).reshape(1, 6, 2, 1)

    add("pixel_shuffle", lambda L: F(L).pixel_shuffle(L.tensor(pix), 2))
    add("pixel_unshuffle", lambda L: F(L).pixel_unshuffle(L.tensor(flat), 2))
    # Reversing it has to give back the same thing — that the two are each other's inverse is the convention.
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

    # ── where no randomness is involved, by value ──
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

    # ── where randomness is involved, by property ──
    #
    # The answer depends on the random generator and ours is not torch's. There is still
    # **something both can answer identically** — is the inside of a channel one block, and by
    # how much is a survivor scaled.
    big = np.ones((200, 8, 2, 2), dtype=np.float32)

    def whole_channels(L):
        """Is the inside of the channel **all the same?** Dropping element-wise parts here."""
        out = to_numpy(F(L).dropout2d(L.tensor(big), 0.5, training=True))
        flat_ch = out.reshape(out.shape[0], out.shape[1], -1)
        uniform = np.all(flat_ch == flat_ch[:, :, :1], axis=2)
        return f"채널마다 한 덩어리={bool(uniform.all())}"

    add("dropout2d::채널째 떨군다", whole_channels)

    def channel_scale(L):
        """A surviving channel is scaled by exactly `1/(1-p)`.

        **The number of digits is pinned.** Left as `round()`, Python gave `2.0` and JS gave `2`,
        so it parted on characters rather than on the value — for a case whose answer is a string,
        that string is the contract and the producing side has to decide its shape.
        """
        out = to_numpy(F(L).dropout2d(L.tensor(big), 0.5, training=True))
        kept = out[out != 0]
        return f"배율={float(kept.mean()):.3f}" if kept.size else "배율=none"

    add("dropout2d::살아남은 배율", channel_scale)

    def channel_rate(L):
        """The dropped fraction is roughly `p`. Counted per channel."""
        out = to_numpy(F(L).dropout2d(L.tensor(big), 0.5, training=True))
        per = out.reshape(out.shape[0], out.shape[1], -1)[:, :, 0]
        return f"대략 절반={bool(0.4 < float((per == 0).mean()) < 0.6)}"

    add("dropout2d::떨구는 비율", channel_rate)

    def alpha_values(L):
        """**What is dropped is not 0.** There have to be exactly two distinct values in the answer."""
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

    # It has to be reachable as a layer too. **The rank differs per layer** — `Dropout1d`
    # refuses 4-D ("give me 2D or 3D"). The number of spatial axes is in the name so it follows,
    # and it was caught while trying to reuse one input across all five.
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
    """The layers that **work their shape out at the first forward pass.**

    `nn.LazyLinear(3)` does not take `in_features`. It decides from the first value that passes
    through — which removes counting by hand how many channels come out of a convolution, so real
    code uses it often.

    ## The class changes

    That is the core of the convention and does not follow from guessing. After the first forward
    pass the object is **no longer a `LazyLinear` but a `Linear`** — `type(m).__name__` changes,
    `isinstance(m, nn.LazyLinear)` becomes false, and the method `has_uninitialized_params`
    disappears entirely (measured). Handled with a flag, the name does not change, and then both
    `repr` and `isinstance` part.

    ## Here **only what all three can answer** is asked

    torch produces the parameters even before they settle (`<UninitializedParameter>`), throws
    when the shape is asked, and still lets them into an optimizer. That machinery is in the core
    alone — the browser-side layer has no tensor at all before it settles.

    So that side is where `tests/test_lazy.py` **holds the core against real torch directly.**
    Put into the golden cases it becomes a question the two browser sides cannot answer, and an
    unanswerable question in the table leaves the table unable to say what passed.

    ## The values cannot be asked

    The initial weights come from the random generator and ours is not torch's. **Properties** are
    asked instead: from the same seed, is a settled `LazyLinear` the same as a `Linear` built
    directly. It is true in torch (measured) and has to be true for us — a lazy side using a
    different initialisation parts the training subtly.
    """
    cases = []

    def add(name, fn):
        cases.append((LAZY_PREFIX + name, fn))

    x2d = np.arange(10, dtype=np.float32).reshape(2, 5)
    img = np.arange(2 * 2 * 8 * 8, dtype=np.float32).reshape(2, 2, 8, 8) / 100

    # ── once it settles it is something else ──
    #
    # What a user actually sees is `print(model)`. Those characters changing is this convention's
    # observable substance, so that is asked rather than the name and `isinstance` — the binding's
    # layers are all one Python class, so the three cannot be matched by name.
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

    # ── the shape once settled ──
    shapes = (
        ("LazyLinear", lambda L: L.nn.LazyLinear(3), x2d),
        ("LazyConv2d", lambda L: L.nn.LazyConv2d(4, 3), img),
        ("LazyBatchNorm2d", lambda L: L.nn.LazyBatchNorm2d(), img),
        ("LazyInstanceNorm2d", lambda L: L.nn.LazyInstanceNorm2d(), img),
        ("LazyConvTranspose2d", lambda L: L.nn.LazyConvTranspose2d(4, 3), img),
    )
    # **The shape alone is asked.** The class name after settling is something the binding side
    # cannot answer — there, every layer is the one class `Module`. That the name changes is
    # already held by `repr` and by `test_lazy.py`, so only what all three can answer is asked here.
    for name, make, arr in shapes:
        def run(L, m=make, a=arr):
            return str(tuple(m(L)(L.tensor(a)).shape))
        add(f"굳은뒤::{name}", run)

    def weight_shape(L):
        m = L.nn.LazyLinear(3)
        m(L.tensor(x2d))
        return str(tuple(m.weight.shape))

    add("굳은뒤::가중치 모양", weight_shape)

    # ── property: does the lazy side use the same initialisation as one built directly ──
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

    # Training really runs — once settled, does the optimizer move those parameters.
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

    # ── does the seed reach the layer initialisation too ──
    #
    # **A defect the lazy layers drew out.** The core's `manual_seed` only rebound a module-level
    # name, so `_nn`, which had grabbed that generator at import time, kept using the old one —
    # `randn` reproduced and **the layer initialisation and dropout did not follow the seed.**
    #
    # The golden cases went a long time without seeing it because every case plants its weights
    # from outside. A lazy layer initialising itself is what asked that place for the first time.
    # These three remain.
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

# Inputs for the losses. **Chosen so the values do not collapse to 0** — the first triplet
# input never once engaged the margin, so all four branches were 0.0, and then anything at all
# passes.
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
    """Thirteen losses and three distances.

    ## Two places where the default hides the difference

    `HuberLoss(δ)` and `SmoothL1Loss(β)` are **equal only at δ=1.** The actual relation is
    `huber(δ) = δ · smooth_l1(β=δ)`, so at δ=0.5 it is double and at δ=2 it is half. Asked on the
    defaults only, keeping the two as one function still passes — so δ is varied.

    `KLDivLoss` has **four** reductions. `mean` divides by the element count and `batchmean` by
    the batch size — here, dividing by 6 against dividing by 2 — and torch itself warns that it
    will make `mean` behave like `batchmean` in a coming version. Both are asked.

    ## Two terms that are switched on conditionally

    `PoissonNLLLoss(full=True)`'s Stirling correction is added **only where `target > 1`**
    (measured: at targets 0, 0.5 and 1 the difference is 0, and at 2 it adds 0.6518). Added
    unconditionally, it is wrong only where the target is small.

    `GaussianNLLLoss` clips the variance with `eps` — at `var=1e-9` with the default `eps=1e-6`
    it clips to `1e-6` and gives 124993, and given `eps=1e-2` it gives 10.2. Unclipped it divides
    by zero and becomes infinite.

    ## `pairwise_distance`'s `eps` is added **to the difference**

    Not to the result. Asked at `p=1` where the difference is exactly 1.0, it gives 1.0000020
    (= 1 + 2·1e-6). Read as added to the result it becomes 1.000001, and a digit parts.
    """
    x, y = _LOSS_X, _LOSS_Y
    a, b, sign = _LOSS_A, _LOSS_B, _LOSS_SIGN
    anc, pos, neg = _LOSS_ANC, _LOSS_POS, _LOSS_NEG

    def F(L):
        return L.nn.functional

    cases = []

    def add(name, fn):
        cases.append((LOSS_PREFIX + name, fn))

    # ── Huber — where the defaults coincide with SmoothL1 ──
    for tag, delta in (("기본", None), ("δ=0.5", 0.5), ("δ=2", 2.0)):
        add(f"huber({tag})",
            lambda L, d=delta: F(L).huber_loss(L.tensor(x), L.tensor(y))
            if d is None else F(L).huber_loss(L.tensor(x), L.tensor(y), delta=d))
    add("huber(none)",
        lambda L: F(L).huber_loss(L.tensor(x), L.tensor(y), reduction="none"))
    add("huber(sum)",
        lambda L: F(L).huber_loss(L.tensor(x), L.tensor(y), reduction="sum"))
    # **Asked at the same δ, the relation between the two appears in the value.**
    add("huber(δ=0.5)/smooth_l1(β=0.5)",
        lambda L: F(L).huber_loss(L.tensor(x), L.tensor(y), delta=0.5)
        / F(L).smooth_l1_loss(L.tensor(x), L.tensor(y), beta=0.5))

    # ── KL — four reductions ──
    def kl(L, red, log_target=False):
        logp = F(L).log_softmax(L.tensor(x), dim=1)
        tgt = F(L).softmax(L.tensor(y), dim=1)
        if log_target:
            tgt = tgt.log()
        return F(L).kl_div(logp, tgt, reduction=red, log_target=log_target)

    for red in ("none", "mean", "sum", "batchmean"):
        add(f"kl_div({red})", lambda L, r=red: kl(L, r))
    add("kl_div(log_target)", lambda L: kl(L, "mean", log_target=True))

    # ── Poisson and Gaussian ──
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
    # Where the variance falls below `eps` — unclipped it becomes infinite here.
    tiny = np.array([[1e-9, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    add("gaussian(var<eps)",
        lambda L: F(L).gaussian_nll_loss(L.tensor(x), L.tensor(y), L.tensor(tiny),
                                         reduction="none"))
    add("gaussian(eps=1e-2)",
        lambda L: F(L).gaussian_nll_loss(L.tensor(x), L.tensor(y), L.tensor(tiny),
                                         eps=1e-2, reduction="none"))

    # ── the margin family ──
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
    # **Where the target is not ±1.** torch does not choose between the two terms, it **adds**
    # them — the margin term goes where `y ≠ 1` and `x` where `y ≠ −1`, so at `y=0` both are on.
    # Asked at ±1 only, one is on at a time and the difference does not show, and `sign()` makes 0.
    add("hinge_embedding(y=0)",
        lambda L: F(L).hinge_embedding_loss(
            L.tensor(np.array([[-1., 0.5, 2.]], dtype=np.float32)),
            L.tensor(np.array([[0., 0., 0.]], dtype=np.float32)), reduction="none"))
    add("soft_margin",
        lambda L: F(L).soft_margin_loss(L.tensor(x), L.tensor(np.sign(y))))
    add("soft_margin(none)",
        lambda L: F(L).soft_margin_loss(L.tensor(x), L.tensor(np.sign(y)),
                                        reduction="none"))

    # ── 자리로 주는 reduction ──
    #
    # **Every loss case above gives `reduction` by keyword, and that is why none
    # of them measured the argument order.** A keyword survives a table that has
    # the seats wrong; a position does not. The binding read torch's call with a
    # table that left out the deprecated `size_average`/`reduce` pair, so every
    # seat after it moved forward and `F.soft_margin_loss(a, b, "sum")` set
    # `size_average` — where a string is neither `None` nor `False`, so the fold
    # reads it as the mean and the answer comes back at the default with no
    # warning anywhere.
    #
    # torch keeps the pair for compatibility and these are the seats it really
    # has. `huber_loss` is the newest and never carried the pair, so its third
    # seat is `reduction` itself — which is what makes it worth asking beside
    # the others rather than instead of them.
    _sgn = np.sign(y)
    for _tag, _fn in (
            ("soft_margin", lambda L, r: F(L).soft_margin_loss(
                L.tensor(x), L.tensor(_sgn), None, None, r)),
            ("hinge_embedding", lambda L, r: F(L).hinge_embedding_loss(
                L.tensor(x), L.tensor(_sgn), 1.0, None, None, r)),
            ("multilabel_margin", lambda L, r: F(L).multilabel_margin_loss(
                L.tensor(_LOSS_MM), L.tensor(np.array([[2, -1, -1], [0, -1, -1]],
                                                      dtype=np.int64)),
                None, None, r)),
            ("kl_div", lambda L, r: F(L).kl_div(
                F(L).log_softmax(L.tensor(x), dim=1),
                F(L).softmax(L.tensor(y), dim=1), None, None, r)),
            # The third seat here **is** `reduction` — no pair to step over.
            ("huber", lambda L, r: F(L).huber_loss(
                L.tensor(x), L.tensor(y), r)),
    ):
        for _r in ("none", "sum", "mean"):
            add(f"자리::{_tag}({_r})", lambda L, f=_fn, r=_r: f(L, r))

    # `multilabel_soft_margin_loss` takes a **`weight` in the third seat**, and
    # borch.ts takes one too — so this is the one place in the group where the
    # seat is filled rather than refused, and giving `reduction` by position
    # would put a string into it.
    add("자리::multilabel_soft_margin(weight)",
        lambda L: F(L).multilabel_soft_margin_loss(
            L.tensor(np.array([[0.5, -1.0, 2.0]], dtype=np.float32)),
            L.tensor(np.array([[1.0, 0.0, 1.0]], dtype=np.float32)),
            L.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))))
    add("자리::multilabel_soft_margin(weight, none)",
        lambda L: F(L).multilabel_soft_margin_loss(
            L.tensor(np.array([[0.5, -1.0, 2.0]], dtype=np.float32)),
            L.tensor(np.array([[1.0, 0.0, 1.0]], dtype=np.float32)),
            L.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            None, None, "none"))

    # ── triplet ──
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
    # **A distance that is not the default**, which is the only reason this name
    # exists apart from `triplet_margin_loss`. Both cases above pass `None` and get
    # the pairwise distance, so they agree with the plain form and would agree just
    # as well with an implementation that ignored the argument entirely.
    add("triplet_with_distance(L1)",
        lambda L: F(L).triplet_margin_with_distance_loss(
            L.tensor(anc), L.tensor(pos), L.tensor(neg),
            distance_function=lambda u, v: (u - v).abs().sum(dim=-1)))

    # ── multi-label ──
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
    # **The target is a list and −1 means the end.** Break that convention and −1 is counted as
    # one of the classes.
    add("multilabel_margin",
        lambda L: F(L).multilabel_margin_loss(
            L.tensor(np.array([[0.1, 0.2, 0.4, 0.8]], dtype=np.float32)),
            L.tensor(np.array([[3, 0, -1, 1]], dtype=np.int64))))
    # **One row could not tell a kept batch from a collapsed one.** The case above
    # was written for the −1 convention and measured it; the core meanwhile summed
    # the whole batch into one number, so `reduction="none"` handed back the sum
    # and `mean` divided it by one. With a single row those are the same array,
    # and all three implementations were green — borch.ts kept the rows the whole
    # time and had no one to disagree with.
    add("multilabel_margin(두 행, none)",
        lambda L: F(L).multilabel_margin_loss(
            L.tensor(_LOSS_MM),
            L.tensor(np.array([[2, -1, -1], [0, -1, -1]], dtype=np.int64)),
            reduction="none"))
    # A row whose target starts at −1 has **no labels at all**, so its loss is
    # zero and the rows around it must not shift up to fill the seat.
    add("multilabel_margin(빈 행)",
        lambda L: F(L).multilabel_margin_loss(
            L.tensor(_LOSS_MM),
            L.tensor(np.array([[-1, -1, -1], [0, 1, -1]], dtype=np.int64)),
            reduction="none"))

    # ── distances ──
    add("pairwise_distance",
        lambda L: F(L).pairwise_distance(L.tensor(a), L.tensor(b)))
    # `eps` is added **to the difference**, not to the result — it shows in the digits at p=1.
    add("pairwise_distance(p=1)",
        lambda L: F(L).pairwise_distance(L.tensor(a), L.tensor(b), p=1))
    add("pairwise_distance(eps=0)",
        lambda L: F(L).pairwise_distance(L.tensor(a), L.tensor(b), p=1, eps=0))
    add("pairwise_distance(keepdim)",
        lambda L: F(L).pairwise_distance(L.tensor(a), L.tensor(b), keepdim=True))
    add("pdist",
        lambda L: F(L).pdist(L.tensor(np.array([[0., 0.], [3., 4.], [1., 1.]],
                                               dtype=np.float32))))

    # **Folding something with one element.** It means there is nothing to fold, so the value
    # should simply come out, and the GPU side gave 0 there — commands are gathered and sent
    # together, and this path alone built its own encoder and submitted it **first**, copying a
    # buffer that had not been computed. Not an exception and not a NaN but a 0, so it is a place
    # where a loss quietly disappears, and 1,399 golden cases were green until one loss was asked
    # at batch 1.
    add("원소 하나를 mean",
        lambda L: (L.tensor(np.array([1., 2., 3.], dtype=np.float32)).sum()
                   * 1.0).reshape(1).mean())
    add("원소 하나를 sum",
        lambda L: (L.tensor(np.array([1., 2., 3.], dtype=np.float32)).sum()
                   * 1.0).reshape(1).sum())

    # ── it has to be reachable as a layer too ──
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
        # **`ignore_index`, across all three reductions.** It was not a parameter at
        # all until the core took torch's argument list, and the three folds treat a
        # skipped row differently: `mean` drops it from the denominator, `sum` is
        # unaffected, and `none` **keeps it as a zero** so the shape survives. The
        # first implementation dropped it everywhere and was right about two of the
        # three — `none` came back one element short, which only a case at `none`
        # could show.
        *[(f"CrossEntropyLoss(ignore_index, {red})",
           lambda L, r=red: L.nn.CrossEntropyLoss(ignore_index=-100, reduction=r)(
               L.tensor(np.array([[2., 1., .1], [.5, 2.5, .3], [1., .2, 3.]],
                                 dtype=np.float32)),
               L.tensor(np.array([0, -100, 2], dtype=np.int64))))
          for red in ("mean", "sum", "none")],
        ("CrossEntropyLoss(label_smoothing)",
         lambda L: L.nn.CrossEntropyLoss(label_smoothing=0.1)(
             L.tensor(np.array([[2., 1., .1], [.5, 2.5, .3], [1., .2, 3.]],
                               dtype=np.float32)),
             L.tensor(np.array([0, 1, 2], dtype=np.int64)))),
        ("NLLLoss(ignore_index)",
         lambda L: L.nn.NLLLoss(ignore_index=-100)(
             L.tensor(np.log(np.array([[.7, .2, .1], [.1, .8, .1], [.2, .2, .6]],
                                      dtype=np.float32))),
             L.tensor(np.array([0, -100, 2], dtype=np.int64)))),
        # **The weighted form, added when borch.ts grew a `weight` it did not have.**
        # torch puts `weight` first and borch.ts took `reduction` there, so
        # `MultiLabelSoftMarginLoss('sum')` set the class weights in one library and
        # the reduction in the other, and neither raised. The weights are unequal so
        # the answer differs from the unweighted case above — a case whose argument
        # does not change its answer proves nothing, which `test_case_names.py`
        # refuses and which caught the `max` scheduler case earlier today.
        ("MultiLabelSoftMarginLoss(weight)",
         lambda L: L.nn.MultiLabelSoftMarginLoss(
             L.tensor(np.array([0.5, 2.0, 1.5], dtype=np.float32)))(
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

    # ── it has to be reachable at the top level too ──
    #
    # Tidying the gap list revealed nine names "in `F` and not at the top level". Measuring showed
    # that **seven of them were a different function at the top level** — being raw ATen
    # operations, their default reduction is `none` and `reduction` is an integer rather than a
    # string. `torch.kl_div(a, b)` gives `[2,2]` and `F.kl_div(a, b)` gives a scalar.
    #
    # Hung as helpful aliases, **the shape would have parted first.** Only these two are literally
    # the same function (`torch.pdist is F.pdist` is true), so only those two are produced.
    add("최상위::pairwise_distance",
        lambda L: L.pairwise_distance(L.tensor(a), L.tensor(b)))
    add("최상위::pdist",
        lambda L: L.pdist(L.tensor(np.array([[0., 0.], [3., 4.], [1., 1.]],
                                            dtype=np.float32))))

    # ── gradients ──
    #
    # **For a loss, the gradient is everything.** A right value with a wrong gradient sends
    # training quietly somewhere else — the kind this repository had for a long time with BatchNorm.
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

    def bce_grad(L):
        """**BCE's gradient needed its own case because it has its own formula.**

        torch does not differentiate through the −100 clamp; it floors the
        denominator at 1e-12, so the gradient saturates at 1e12 where the value
        saturates at 100. Differentiated through, `p = 1e-20` gives −1e20 — eight
        orders past torch, and the first step takes the weight somewhere no finite
        learning rate was meant to reach.

        This case asks at comfortable probabilities, where the two formulas agree and
        only an ordinary mistake would show. The saturating end is
        `tests/test_arg_domain.py`'s.
        """
        q = L.tensor(_BCE_P, requires_grad=True)
        F(L).binary_cross_entropy(q, L.tensor(_BCE_T)).backward()
        return _grad_of(q, "bce")

    cases.append((LOSS_PREFIX + "grad::bce", bce_grad))

    def cosine_grad(L):
        p = L.tensor(a, requires_grad=True)
        F(L).cosine_embedding_loss(p, L.tensor(b), L.tensor(sign)).backward()
        return _grad_of(p, "cosine_embedding")

    cases.append((LOSS_PREFIX + "grad::cosine_embedding", cosine_grad))

    # ── torch's deprecated pair, which is not ignored ──
    #
    # `size_average` and `reduce` are deprecated in torch and **they still beat
    # `reduction`**, which is the opposite of what deprecation suggests and is the
    # whole reason they are worth freezing. The last case is the one that reads
    # wrongly at a glance: all three given, and the answer is the *mean*.
    #
    # The positional case is why the seats matter at all. `L1Loss("sum")` puts the
    # string in `size_average`, which is truthy, so torch folds to the mean — the
    # same call meant a sum here until the two seats went in.
    for _name, _kw in (("가장자리::size_average=False", {"size_average": False}),
                       ("가장자리::reduce=False", {"reduce": False}),
                       ("가장자리::셋 다", {"size_average": True, "reduce": True,
                                          "reduction": "sum"})):
        cases.append((LOSS_PREFIX + _name,
                      lambda L, k=_kw: L.nn.L1Loss(**k)(L.tensor(a), L.tensor(b))))
    cases.append((LOSS_PREFIX + "가장자리::L1Loss('sum') 위치",
                  lambda L: L.nn.L1Loss("sum")(L.tensor(a), L.tensor(b))))

    # ── the way it folds is part of the loss ──
    #
    # **The most-used losses were not taking `reduction`.** A `.mean()` was nailed into the body,
    # so passing `reduction=` was a `TypeError`. All three of them.
    #
    # The clue was that it was inverted — the rare losses like `cosine_embedding`, `multi_margin`
    # and `triplet` **all thirteen** took it. What was written later followed torch's signature
    # and what was written first was never fixed. The table did not see it because the textbook
    # only ever uses the default `mean`, and **the most-used place was the least-asked-about one.**
    #
    # **`nll_loss` was averaging as soon as it gathered.** So there was nowhere to build a `none`
    # at all — a per-sample value cannot be recovered from a scalar. The vector has to be built
    # before folding for `reduction` to mean anything. `cross_entropy` inherits that.
    _CE_X = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]], dtype=np.float32)
    _CE_T = np.array([2, 0])
    _LOGP = np.log(np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]], dtype=np.float32))
    _BCE_P = np.array([[0.2, 0.7, 0.9], [0.4, 0.15, 0.6]], dtype=np.float32)
    _BCE_T = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    for reduction in ("none", "mean", "sum"):
        for name, call in (
            ("cross_entropy", lambda L, r: F(L).cross_entropy(
                L.tensor(_CE_X), L.tensor(_CE_T), reduction=r)),
            ("nll_loss", lambda L, r: F(L).nll_loss(
                L.tensor(_LOGP), L.tensor(_CE_T), reduction=r)),
            # **This note used to say `binary_cross_entropy` was not here yet** — that
            # borch.ts had `bceWithLogits` alone and nothing taking probabilities. Both
            # sides have it now, and the fixtures above (`_BCE_P`, `_BCE_T`) had been
            # sitting here the whole time waiting for it.
            #
            # Worth asking at all three reductions because the core's forward was
            # `-(p + 1e-12).log()`, which caps at 27.63 whatever `p` is — the same
            # defect `CrossEntropyLoss` had, with the guard and the defect on one line.
            # torch clamps the log's *output* at −100 instead. These fixtures are
            # comfortable probabilities and would not have found it; the saturating
            # ones are asked in `tests/test_arg_domain.py`.
            ("binary_cross_entropy", lambda L, r: F(L).binary_cross_entropy(
                L.tensor(_BCE_P), L.tensor(_BCE_T), reduction=r)),
            ("nn.BCELoss", lambda L, r: L.nn.BCELoss(reduction=r)(
                L.tensor(_BCE_P), L.tensor(_BCE_T))),
            ("nn.CrossEntropyLoss", lambda L, r: L.nn.CrossEntropyLoss(reduction=r)(
                L.tensor(_CE_X), L.tensor(_CE_T))),
            ("nn.NLLLoss", lambda L, r: L.nn.NLLLoss(reduction=r)(
                L.tensor(_LOGP), L.tensor(_CE_T))),
            ("mse_loss", lambda L, r: F(L).mse_loss(L.tensor(x), L.tensor(y),
                                                    reduction=r)),
            ("l1_loss", lambda L, r: F(L).l1_loss(L.tensor(x), L.tensor(y),
                                                  reduction=r)),
            ("smooth_l1_loss", lambda L, r: F(L).smooth_l1_loss(
                L.tensor(x), L.tensor(y), reduction=r)),
            ("huber_loss", lambda L, r: F(L).huber_loss(
                L.tensor(x), L.tensor(y), reduction=r)),
            ("nn.MSELoss", lambda L, r: L.nn.MSELoss(reduction=r)(
                L.tensor(x), L.tensor(y))),
            ("nn.L1Loss", lambda L, r: L.nn.L1Loss(reduction=r)(
                L.tensor(x), L.tensor(y))),
            ("nn.SmoothL1Loss", lambda L, r: L.nn.SmoothL1Loss(reduction=r)(
                L.tensor(x), L.tensor(y))),
        ):
            add(f"reduction::{name}({reduction})",
                lambda L, c=call, r=reduction: c(L, r))

    # **An unknown value is not swallowed into the mean.** It was `else: return mean()`, and then
    # a typo like `reduction="MEAN"` passes quietly and is trained on — the person believes what
    # they chose is being used. `batchmean` exists **only** on `kl_div`, so in another loss it is a
    # wrong name, and swallowed, someone expecting a division by the batch gets a division by the
    # element count.
    for bad in ("MEAN", "batchmean"):
        def refuses(L, b=bad):
            try:
                F(L).l1_loss(L.tensor(x), L.tensor(y), reduction=b)
            except Exception as exc:                            # noqa: BLE001
                return "멈췄다" if b in str(exc) else f"다른 문구 <{exc}>"
            return "안 던졌다"
        add(f"reduction::거절::{bad}", refuses)

    # ── the seven that are also at top level, and are not the same function ──────
    #
    # `torch.kl_div` and `F.kl_div` are two different things. The gap table declined the
    # top-level seven as *the raw ATen op — its signature differs from F's*, which is
    # true and is about why the name looks redundant rather than about what is missing:
    # the arithmetic is `F`'s and was already here.
    #
    # Three measured differences, each asked below:
    #
    # - **the reduction is an integer**, `0` none / `1` mean / `2` sum, where `F` takes
    #   the word;
    # - **it defaults to none**, where every `F` loss defaults to mean — so
    #   `torch.kl_div(a, b)` gives a table and `F.kl_div(a, b)` gives a number;
    # - **`poisson_nll_loss` has no defaults at all.**
    #
    # And a fourth that is about torch rather than about this library: **the declared
    # schema says `reduction=1` and the binding defaults to `0`.** `aten::kl_div(...,
    # int reduction=1)` is mean; `torch.kl_div(a, b)` returns a table. The behaviour is
    # the authority, and the first case here is the one that pins it.
    def _top(name, args, **kwargs):
        def run(L):
            given = [L.tensor(np.ascontiguousarray(one))
                     if isinstance(one, np.ndarray) else one for one in args]
            return getattr(L, name)(*given, **kwargs)
        return run

    _logp = np.log(np.exp(x) / np.exp(x).sum(-1, keepdims=True)).astype(np.float32)
    _prob = (np.exp(y) / np.exp(y).sum(-1, keepdims=True)).astype(np.float32)
    _binary = (y > 0).astype(np.float32)
    _pm = np.where(y > 0, 1.0, -1.0).astype(np.float32)

    for _tag, _r in (("기본=none", None), ("reduction=0", 0), ("reduction=1", 1),
                     ("reduction=2", 2)):
        _tail = [] if _r is None else [_r]
        add(f"aten::kl_div({_tag})", _top("kl_div", [_logp, _prob] + _tail))
        add(f"aten::bce_with_logits({_tag})",
            _top("binary_cross_entropy_with_logits",
                 [x, _binary, None, None] + _tail if _r is not None
                 else [x, _binary]))
    # **`log_target` is keyword-only**, as the schema has it.
    add("aten::kl_div(log_target)",
        _top("kl_div", [_logp, np.log(_prob).astype(np.float32), 1], log_target=True))
    add("aten::cosine_embedding_loss", _top("cosine_embedding_loss", [a, b, sign]))
    add("aten::cosine_embedding_loss(margin, sum)",
        _top("cosine_embedding_loss", [a, b, sign, 0.5, 2]))
    add("aten::hinge_embedding_loss", _top("hinge_embedding_loss", [x, _pm]))
    add("aten::hinge_embedding_loss(margin, mean)",
        _top("hinge_embedding_loss", [x, _pm, 2.0, 1]))
    add("aten::margin_ranking_loss",
        _top("margin_ranking_loss", [x[:, 0], y[:, 0], sign]))
    # **Six required arguments** — a caller cannot reach this one with two.
    add("aten::poisson_nll_loss(log_input)",
        _top("poisson_nll_loss", [np.abs(x), _binary, True, False, 1e-8, 1]))
    add("aten::poisson_nll_loss(full, none)",
        _top("poisson_nll_loss", [np.abs(x), _binary, False, True, 1e-8, 0]))
    add("aten::triplet_margin_loss", _top("triplet_margin_loss", [anc, pos, neg]))
    add("aten::triplet_margin_loss(swap, sum)",
        _top("triplet_margin_loss", [anc, pos, neg, 2.0, 1.0, 1e-5, True, 2]))

    return cases


PAD_PREFIX = "pad::"

# Inputs for the padding. The values are position numbers, so **the answer alone says where a
# value came from** — mirrored, repeated or from the edge, it is written into the value.
_PAD_1D = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
_PAD_2D = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)
_PAD_3D = np.arange(24, dtype=np.float32).reshape(1, 1, 2, 3, 4)

_PAD_MODES = ("constant", "reflect", "replicate", "circular")


def pad_cases(inp=None):
    """Padding — **four modes and fifteen layers.**

    Until now `F.pad` did the constant mode only. Without the other three (`reflect`,
    `replicate`, `circular`) the fifteen layers standing on them are all absent too, and that was
    the largest single block of the eighty-four gaps in `nn`.

    ## The values are position numbers

    With an `arange` input, the answer alone says **where a value came from.** Extending a
    three-slot `[0,1,2]` on both sides gives this per mode (confirmed by asking real torch):

        constant   9 9 [0 1 2] 9      ← filled in
        reflect    2 1 [0 1 2] 1      ← the edge mirrored, without repeating the edge
        replicate  0 0 [0 1 2] 2      ← the edge extended
        circular   1 2 [0 1 2] 0      ← taken from the far side

    Where implementations part is the mirror's pivot (is 0 repeated) and the wrapping direction,
    and both are written straight into the value, so these four rows hold them both.

    ## The number of pairs and the rank interlock

    `F.pad(4-D, (1,1), mode='reflect')` is **a refusal** — torch raises `NotImplementedError`.
    One pair needs 2-D or 3-D, two pairs 3-D or 4-D, three pairs 4-D or 5-D. Accepting any rank
    lets a wrongly chosen axis pass.

    ## `reflect` alone cares about the size

    Mirroring needs something to mirror, so the padding has to be smaller than that axis.
    `replicate` will extend by five — there is always a value to extend.
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

    # Asymmetric — a reversed axis shows up here, where the two sides differ.
    cases += [
        (PAD_PREFIX + "비대칭::reflect",
         lambda L: L.nn.functional.pad(L.tensor(_PAD_2D), (1, 2, 0, 1), mode="reflect")),
        (PAD_PREFIX + "비대칭::circular",
         lambda L: L.nn.functional.pad(L.tensor(_PAD_2D), (2, 1, 1, 0), mode="circular")),
        # `replicate` does not care about the size — there is always a value to extend.
        (PAD_PREFIX + "replicate(크게)",
         lambda L: L.nn.functional.pad(L.tensor(_PAD_1D), (5, 0), mode="replicate")),
        # An input without a batch is taken too.
        (PAD_PREFIX + "2차원 입력::reflect",
         lambda L: L.nn.functional.pad(
             L.tensor(np.arange(6, dtype=np.float32).reshape(2, 3)), (1, 1),
             mode="reflect")),
    ]

    # ── the fifteen layers ──
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
        # **Only `ConstantPad` prints a named argument** — the rest print the pairs alone.
        cases.append((PAD_PREFIX + f"repr::{name}",
                      lambda L, c=name, a=arg: repr(getattr(L.nn, c)(a, 7.0))))

    # **Only `ConstantPad*` takes a `value`, and asking that is the whole point.**
    #
    # `_PadNd` used to hand it to all fifteen, and the thirty-three cases above are
    # every one of them at its default — so nothing here saw it. What it did depended
    # on the mode: `ZeroPad2d(1, 9.0)` filled with 9 (a pad named Zero), while
    # `ReflectionPad2d(1, 9.0)` took the argument and dropped it.
    #
    # A default-only case list cannot see an argument that should not exist. **The
    # question has to be whether the call is refused**, and torch refuses it —
    # `ZeroPad2d.__init__() takes 2 positional arguments but 3 were given`.
    def refuses_value(L, cls):
        try:
            getattr(L.nn, cls)(1, 9.0)
        except TypeError:
            return "거절"
        except Exception as exc:                                # noqa: BLE001
            return f"다른 예외 <{type(exc).__name__}>"
        return "안 던졌다"

    for _cls in ("ZeroPad2d", "ReflectionPad2d", "ReplicationPad2d", "CircularPad2d"):
        cases.append((PAD_PREFIX + f"거절::{_cls}(value)",
                      lambda L, c=_cls: refuses_value(L, c)))

    # The gradient has to flow through the layer too — wire the function and not the layer and it is cut here.
    def layer_grad(L):
        x = L.tensor(_PAD_2D, requires_grad=True)
        out = L.nn.ReflectionPad2d(1)(x)
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(x, "ReflectionPad2d")

    cases.append((PAD_PREFIX + "grad::층::ReflectionPad2d", layer_grad))

    # ── refusals ──
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
    """Three kinds of normalisation and the transposed convolution. **Places where the shape fits
    and the value is wrong.**

    ## Normalisation — what is grouped together to average

    `LayerNorm`, `GroupNorm`, `InstanceNorm` and `BatchNorm` share a formula and differ **only in
    which axes are grouped.** Choose the axes wrongly and the shape holds while the value parts,
    and training still runs, so it is a long while before anyone notices.

    So `GroupNorm(1)`, `GroupNorm(3)` and `InstanceNorm2d` are asked side by side on the same
    input. The three are special cases of each other — with the grouping rule wrong, two of the
    three become equal.

    ## The transposed convolution — the weight axes are reversed

    `conv2d`'s weight is `(out, in, kh, kw)` and `conv_transpose2d`'s is `(in, out, kh, kw)`.
    Reversed, **the shape still fits** for a square kernel — it parts by value alone. It is the
    commonest mistake at this layer, and that is why it is asked by value.
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

    # ── GroupNorm. At one group it is LayerNorm, and at the channel count InstanceNorm. ──
    add("F.group_norm(1)", lambda L, x: L.nn.functional.group_norm(x, 1), img)
    add("F.group_norm(3)", lambda L, x: L.nn.functional.group_norm(x, 3), img)
    cases.append((NORM_PREFIX + "nn.GroupNorm(1,3)",
                  lambda L: L.nn.GroupNorm(1, 3)(L.tensor(img))))
    cases.append((NORM_PREFIX + "nn.GroupNorm(3,3)",
                  lambda L: L.nn.GroupNorm(3, 3)(L.tensor(img))))
    # **With a weight attached, a parameter has to be picked up.** The name is the state_dict key.
    cases.append((NORM_PREFIX + "nn.GroupNorm/파라미터 이름",
                  lambda L: " ".join(n for n, _ in L.nn.GroupNorm(3, 3).named_parameters())))
    # **`affine` and `bias` are two different halves and the case above sees neither.**
    # `affine=False` is no learnable scale or shift; `bias=False` keeps the scale and
    # drops the shift. Both were absent from every normalisation layer here — thirteen
    # of them — and the count in `test_torch_signatures_core.py` is what said so, because
    # a layer built either way simply raised `TypeError` and no case had ever tried.
    #
    # The names are asked as well as the values: with `bias=False` torch's `state_dict`
    # has one key where ours had two, so a checkpoint could not cross in strict mode.
    # A value case alone would not have noticed — the arithmetic is right either way.
    for _flag, _kw in (("affine=False", {"affine": False}), ("bias=False", {"bias": False})):
        cases.append((NORM_PREFIX + f"nn.GroupNorm({_flag})",
                      lambda L, k=_kw: L.nn.GroupNorm(3, 3, **k)(L.tensor(img))))
        cases.append((NORM_PREFIX + f"nn.GroupNorm({_flag})/파라미터 이름",
                      lambda L, k=_kw: " ".join(
                          n for n, _ in L.nn.GroupNorm(3, 3, **k).named_parameters())))
        cases.append((NORM_PREFIX + f"nn.BatchNorm2d({_flag})/파라미터 이름",
                      lambda L, k=_kw: " ".join(
                          n for n, _ in L.nn.BatchNorm2d(3, **k).named_parameters())))

    # ── InstanceNorm. Normalises separately per sample and per channel. ──
    add("F.instance_norm", lambda L, x: L.nn.functional.instance_norm(x), img)
    for nd, arr in (("1d", seq), ("2d", img), ("3d", vol)):
        chan = arr.shape[1]
        cases.append((NORM_PREFIX + f"nn.InstanceNorm{nd}",
                      lambda L, n=nd, c=chan, a=arr:
                      getattr(L.nn, f"InstanceNorm{n}")(c)(L.tensor(a))))

    # ── RMSNorm. It does not subtract the mean — the only difference from LayerNorm. ──
    add("F.rms_norm", lambda L, x: L.nn.functional.rms_norm(x, (4,)), img)
    cases.append((NORM_PREFIX + "nn.RMSNorm",
                  lambda L: L.nn.RMSNorm(4)(L.tensor(img))))

    # ── LayerNorm's `normalized_shape` is **how many axes are folded** ──
    #
    # Measured with a single axis (`LayerNorm(4)`) it gives the same answer as "fold the last
    # axis", so this rule is invisible. All three really were written that way, and the binding
    # was discarding the shape **outright.** Give it two axes and the mean and variance come from
    # 12 slots.
    cases.append((NORM_PREFIX + "nn.LayerNorm(축 하나)",
                  lambda L: L.nn.LayerNorm(4)(L.tensor(img))))
    add("nn.LayerNorm(축 둘)", lambda L, x: L.nn.LayerNorm((4, 4))(x), img)
    # **torch stops on a shape that does not fit.** Being lenient folds the wrong axis quietly.
    def layer_norm_mismatch(L):
        try:
            L.nn.LayerNorm((3, 4))(L.tensor(img))
            return "예외가 안 났다"
        except Exception as exc:                                # noqa: BLE001
            return type(exc).__name__

    cases.append((NORM_PREFIX + "nn.LayerNorm(모양 불일치)", layer_norm_mismatch))
    # A parameter has to be attached for training to move. With `elementwise_affine=False` the
    # names disappear entirely, and that is a story about `state_dict` keys rather than values.
    cases.append((NORM_PREFIX + "nn.LayerNorm/파라미터 이름",
                  lambda L: " ".join(
                      n for n, _ in L.nn.LayerNorm(4).named_parameters())))
    cases.append((NORM_PREFIX + "nn.LayerNorm(affine 끄면)",
                  lambda L: " ".join(
                      n for n, _ in L.nn.LayerNorm(
                          4, elementwise_affine=False).named_parameters()) or "없음"))

    # ── the transposed convolution. ──
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

    # The weight-side gradient is looked at too. **Looking at the input side alone misses reversed axes.**
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


def refusal_case(call):
    """The verdict function for places **torch manages and all three of ours refuse.**

    Asked by value it would part forever, so what is asked is "did each behave as its own
    documentation says". Our side's refusal wording has to be the specified one (`is not in the
    browser subset`) and torch has to succeed. The name and prefix are the caller's to write —
    only this rule lives here.

    The Korean strings this returns are **expected values inside `golden.json`**, not prose. They
    move only by re-exporting.

    **`hasattr` cannot tell them apart.** The binding's module `__getattr__` answers to any name,
    so `hasattr(borch_webgpu, "compile")` is true. A module's `__name__` is fixed into that
    module and does not change under `import borch as torch` either.
    """
    def run(L, f=call):
        real = getattr(L, "__name__", "") == "torch"
        try:
            f(L)
        except Exception as exc:                                # noqa: BLE001
            if real:
                return f"뜻밖의 거절 <{type(exc).__name__}>"
            mark = "is not in the browser subset"
            return "기대대로" if mark in str(exc) \
                else f"다른 문구 <{str(exc).splitlines()[0][:44]}>"
        return "기대대로" if real else "뜻밖의 성공"
    return run


CONTAINER_PREFIX = "container::"


def container_cases(inp=None):
    """**Are the parameters visible** through a composite structure?

    The latest place to arrive in this table, and the reason it was late is what these cases are
    worth.

    ## Why this is a different kind from the other cases

    The other cases ask about **values** — a wrong `exp` gives a different number and it shows
    immediately. What is asked here is **traversal.** If `parameters()` leaves a parameter out,
    the optimizer does not see it, and not seeing it does not update it, and not updating it
    **still lets the loss come down** (the remaining parameters compensate). No exception and no
    warning. Exactly the shape of defect this repository keeps catching, and nobody had asked
    that place until now.

    ## How it is asked — **both** the name and the value

    Asking the name alone misses the case where it registers and is not updated. Asking the value
    alone passes even when the name comes out as `0.weight` rather than `layers.0.weight` — and
    then `load_state_dict` cannot read somebody else's checkpoint.

    So each place carries a pair: `named_parameters`'s **list of names**, and the **parameter
    values** after a few steps of SGD. A missing registration leaves the value at its starting
    point and parts.

    ## Why this was empty

    Every model in the table was built with `nn.Sequential`. Asking about that alone never once
    catches what torch code most commonly does — subclass `nn.Module` and attach layers as
    attributes. It came out only when the benchmark tried to build a real ResNet and stopped with
    `Module.__init__() missing 1 required positional argument`.
    """
    inp = golden_inputs() if inp is None else inp
    xin, yin = inp["train_x"], inp["train_y"]
    w0, b0, w1, b1 = inp["w0"], inp["b0"], inp["w1"], inp["b1"]
    # For a hand-built linear layer — laid out as `(6, 8)` so that `x @ w` works. Transposing
    # inside the case would leave it impossible to tell a wrong transpose from a wrong traversal.
    flat_w = w0.T.copy()

    cases = []

    def add(name, build, load, forward, want_names):
        """Attaches two cases to one place — **name and value.**

        `build(L)` stands the model up, `load(L, m)` puts the fixed weights in, and
        `forward(L, m, x)` produces the output. They are separated because the way values go in
        differs per container (places `load_state_dict` reaches and places it does not).
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
                # The output shape differs per place, so one loss will not do for all. It is
                # folded with a different weight per place — a plain `sum()` makes every gradient
                # 1, and then which place did not move leaves no trace in the value.
                w = L.arange(out.numel()).reshape(out.shape).float()
                (out * w).sum().backward()
                opt.step()
            return m

        cases.append((CONTAINER_PREFIX + f"{name}/이름", names))
        cases.append((CONTAINER_PREFIX + f"{name}/학습",
                      lambda L: dict(trained(L).named_parameters())[want_names]))

    # ── a subclassed Module. **What torch code most commonly does.** ──
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

    # ── ModuleList. Every model with a varying layer count uses it. ──
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

    # **One built with `append` has to give the same names.** It is the only way to write a
    # model whose layer count is not fixed, and if it parts from one built through the
    # constructor, it parts right there.
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

    # ── ModuleDict. Used by models that select a branch by name. ──
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

    # ── ParameterList. **This is where it goes quietly wrong.** ──
    #
    # Put `Parameter`s in a bare list and attach it as an attribute, and `__setattr__` recognises
    # it as neither a `Parameter` nor a `Module` — it goes into no list, `parameters()` does not
    # produce it, and the optimizer does not see it. torch fails to recognise it in exactly the
    # same way, and **that is why `ParameterList` exists.** Without it there is no correct way to
    # do this.
    def build_plist(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.ws = L.nn.ParameterList(
                    [L.nn.Parameter(L.tensor(flat_w)), L.nn.Parameter(L.tensor(b0))])
        return Net()

    add("ParameterList", build_plist,
        lambda L, m: None,                       # the fixed values went in at construction
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

    # ── `state_dict`'s keys. A parted name cannot read somebody else's checkpoint. ──
    cases.append((CONTAINER_PREFIX + "상속/state_dict 열쇠",
                  lambda L: " ".join(sorted(build_subclass(L).state_dict()))))
    cases.append((CONTAINER_PREFIX + "ModuleDict/state_dict 열쇠",
                  lambda L: " ".join(sorted(build_dict(L).state_dict()))))

    # **It has to be asked with a layer that has buffers too.** The two above are `Linear` only,
    # so only parameters come out — a `state_dict` also carries buffers that are not trained, and
    # that branch had never once been asked. The layer with buffers is `BatchNorm`, and parting
    # there is wrong **in eval mode only**, or else the checkpoint cannot be read at all. A
    # surface the golden cases were not looking at, so it is written down here.
    cases.append((CONTAINER_PREFIX + "BatchNorm/state_dict 열쇠",
                  lambda L: " ".join(sorted(L.nn.BatchNorm2d(3).state_dict()))))

    # **`named_parameters` is not `state_dict`.** It differs by exactly the buffers. Write the
    # two lists as one line and a buffer passes itself off as a parameter, and code handing that
    # to an optimizer tries to train the running statistics. Asked with `Linear` alone the two are
    # equal and it is invisible.
    cases.append((CONTAINER_PREFIX + "BatchNorm/named_parameters 열쇠",
                  lambda L: " ".join(sorted(
                      n for n, _ in L.nn.BatchNorm2d(3).named_parameters()))))

    # ── **BatchNorm is not the only place with buffers.** ──
    #
    # After fixing the two above, the same question was put to the remaining buffer surface. Fix
    # one place and move on with "buffers are covered" and the same defect stays in the rest — the
    # shape this repository met when `launch` moved six and left six.

    # `register_buffer` is not a layer but **a syntax the user writes.** Every model carrying a
    # mask, a positional table or a normalisation constant uses it. Without it, such a model does
    # not run on an import change alone.
    def buffer_keys(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = L.nn.Linear(6, 8)
                self.register_buffer("mask", L.ones(4))
        return " ".join(sorted(Net().state_dict()))

    cases.append((CONTAINER_PREFIX + "register_buffer/state_dict 열쇠", buffer_keys))

    # A registered buffer is **not a parameter.** Part here and the optimizer trains the mask.
    def buffer_not_param(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = L.nn.Linear(6, 8)
                self.register_buffer("mask", L.ones(4))
        return " ".join(sorted(n for n, _ in Net().named_parameters()))

    cases.append((CONTAINER_PREFIX + "register_buffer/named_parameters 열쇠",
                  buffer_not_param))

    # `persistent=False` **drops out of saving.** It is where a cache-like buffer is kept out of
    # the checkpoint, and ignoring it puts the keys out of step with somebody else's checkpoint.
    def buffer_nonpersistent(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("kept", L.ones(2))
                self.register_buffer("cache", L.ones(2), persistent=False)
        return " ".join(sorted(Net().state_dict()))

    cases.append((CONTAINER_PREFIX + "register_buffer(persistent=False)",
                  buffer_nonpersistent))

    # `buffers()` and `named_buffers()` are **the list that pairs with** the parameters.
    cases.append((CONTAINER_PREFIX + "BatchNorm/named_buffers 열쇠",
                  lambda L: " ".join(sorted(
                      n for n, _ in L.nn.BatchNorm2d(3).named_buffers()))))

    # **`InstanceNorm`'s defaults are the opposite of `BatchNorm`'s** — `affine=False`,
    # `track_running_stats=False`. So built on its defaults it has no keys at all, and switched on
    # it gains a parameter and a buffer at once. Invert the defaults and both cases part.
    cases.append((CONTAINER_PREFIX + "InstanceNorm(기본)/state_dict 열쇠",
                  lambda L: " ".join(sorted(L.nn.InstanceNorm2d(3).state_dict()))))
    cases.append((CONTAINER_PREFIX + "InstanceNorm(affine)/state_dict 열쇠",
                  lambda L: " ".join(sorted(
                      L.nn.InstanceNorm2d(3, affine=True).state_dict()))))
    # **The three below are in torch and not in ours.** Asked by value they would part forever,
    # so what is asked is "does it say it cannot".
    #
    # Registering the buffer without the forward pass using it only moves it to the worse place
    # where **the keys fit and the value is wrong.** `track_running_stats=True` changes the eval
    # computation outright, and a loss's `weight` turns `mean`'s division from the sample count
    # into **the sum of the weights** — accepted and ignored, the loss value quietly differs.
    #
    # All three were blocked with a `TypeError`. That is the same screen as a typo and does not
    # say "this library does not have it".
    cases.append((CONTAINER_PREFIX + "InstanceNorm(추적)=우리는거절",
                  refusal_case(
                      lambda L: L.nn.InstanceNorm2d(3, track_running_stats=True))))
    cases.append((CONTAINER_PREFIX + "CrossEntropyLoss(weight)=우리는거절",
                  refusal_case(lambda L: L.nn.CrossEntropyLoss(weight=L.ones(3)))))
    cases.append((CONTAINER_PREFIX + "BCEWithLogitsLoss(pos_weight)=우리는거절",
                  refusal_case(
                      lambda L: L.nn.BCEWithLogitsLoss(pos_weight=L.ones(3)))))

    # Keys fitting is no use if **the values do not cross.** The buffer round trip is asked by value.
    def buffer_roundtrip(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mask", L.ones(3))
        src = Net()
        src.load_state_dict({"mask": L.tensor(np.array([2., 5., 9.], dtype=np.float32))})
        dst = Net()
        dst.load_state_dict(src.state_dict())
        return dst.mask

    cases.append((CONTAINER_PREFIX + "버퍼 값이 왕복한다", buffer_roundtrip))

    # **A tensor attribute that was not registered is not a buffer.**
    #
    # torch puts `self.t = torch.ones(3)` into no list — neither a parameter nor a buffer, and not
    # in `state_dict`. That becoming a buffer requires going through `register_buffer` is the
    # reason that API exists.
    #
    # A leaf that takes gradients is outside this question — there, **the divergence is already
    # written down** as "a tensor with only the flag raised counts as a parameter", and parity
    # holds it by value. What is asked here is an attribute that takes no gradient and therefore
    # looks like a buffer.
    def unregistered_keys(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = L.nn.Linear(6, 8)
                self.plain = L.ones(3)          # not registered
        return " ".join(sorted(Net().state_dict()))

    cases.append((CONTAINER_PREFIX + "등록 안 한 텐서 속성/state_dict 열쇠",
                  unregistered_keys))

    def unregistered_buffers(L):
        class Net(L.nn.Module):
            def __init__(self):
                super().__init__()
                self.plain = L.ones(3)
        return " ".join(sorted(n for n, _ in Net().named_buffers()))

    cases.append((CONTAINER_PREFIX + "등록 안 한 텐서 속성/named_buffers 열쇠",
                  unregistered_buffers))

    # ── does `eval()` go **all the way down** through the containers? ──
    #
    # If it does not, training looks fine and **only inference is wrong.** The kind that is found
    # latest of all.
    #
    # Asked with BatchNorm. Freshly built it has `running_mean=0` and `running_var=1`, so the eval
    # output is nearly the input, while training mode normalises by the batch statistics and gives
    # a noticeably different value — if `eval()` does not go down, that difference stays in the value.
    #
    # `Dropout` was intended at first and borch.ts does not have it (no random kernel). That is not
    # working around it here; what this case asks about is traversal, not Dropout.
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
    """**Does training run** — looked at with the pieces wired together.

    A unit comparison looks at one operation at a time. Some things part only when modules, losses
    and optimizers are wired together, and all three defects the core caught in integration
    scenarios came from that place.

    Keeping the step count small (5) is deliberate. Training amplifies differences, so running it
    long shows float32 having parted rather than what is wrong — T4 is a non-goal.
    """
    inp = golden_inputs() if inp is None else inp
    xin, yin = inp["train_x"], inp["train_y"]
    weights = {"0.weight": inp["w0"], "0.bias": inp["b0"],
               "2.weight": inp["w1"], "2.bias": inp["b1"]}

    # Momentum is **looked at separately.** Looking only at SGD without momentum missed the buffer
    # holding a handle to grad at the first step and blowing up at the second.
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
        # The weights are looked at too. Looking at the loss alone can look similar **even when
        # the parameters do not move.**
        cases.append((f"train::{opt_name}/0.weight",
                      lambda L, o=opt_name: dict(trained(L, o).named_parameters())["0.weight"]))

    # CNN — convolution and pooling wired into a training loop. All three defects the core caught
    # came from a place wired like this, and the unit comparison could not see them.
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

    # A scheduler is **Python float arithmetic only**, so its values have to be exactly torch's.
    # Not one value but **the whole trajectory** — doing that is how the core caught StepLR's difference.
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

    # **A case for an argument has to be one whose *default* answer would be wrong.**
    #
    # That rule was learned the expensive way, right here. These four rows ran against
    # the binding for a day while `threshold_mode`, `cooldown`, `min_lr` and `eps` could
    # not cross it at all — the table in `borch_webgpu/_optim.py` did not name them, so
    # `_sched` dropped them and borch.ts used its defaults. **Every one of the four
    # passed.** Not because the arguments worked, but because a run with the argument
    # dropped is identical to a run that never asked for it.
    #
    # Eight other rows failed at the same moment, and only because a constructor happened
    # to validate a string. So the score measured the half of the defect that had a type
    # check in front of it and was blind to the half that did not.
    #
    # Which is why each row below is given a value that changes the answer: `abs` and
    # `rel` are nearly the same number at a loss near 1 and are told apart at
    # `threshold=0.1`, and `eps=0.4` is larger than the first cut so no cut is made. A
    # row whose value the default would also produce is a row that cannot fail.
    #
    # **`threshold_mode`, `cooldown` and `eps` were not parameters at all** until the
    # core took torch's argument list, so a call written from torch's documentation —
    # `ReduceLROnPlateau(opt, "min", 0.5, 5, 1e-3, 0, 1e-4)` — put the cooldown where
    # `threshold_mode` goes and the minimum rate where `cooldown` does.
    #
    # The metrics fall and then stall, so a cut happens twice and a cooldown has
    # something to suppress. Each row is given a value that changes the answer:
    # `abs` and `rel` are nearly the same number at a loss near 1 and are told apart
    # at `threshold=0.1`, and `eps=0.4` is larger than the first cut so no cut is made.
    def plateau_arg(L, **kw):
        p = L.tensor([1.0], requires_grad=True)
        opt = L.optim.SGD([p], lr=1.0)
        sch = L.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1, **kw)
        seen = []
        for metric in [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5]:
            sch.step(metric)
            seen.append(opt.param_groups[0]["lr"])
        return L.tensor(seen)

    for label, kw in (
        ("threshold_mode=abs", {"threshold_mode": "abs", "threshold": 0.1}),
        ("cooldown", {"cooldown": 2}),
        ("eps", {"eps": 0.4}),
        ("min_lr", {"min_lr": 0.3}),
    ):
        cases.append((f"sched::ReduceLROnPlateau({label})",
                      lambda L, k=kw: plateau_arg(L, **k)))

    def plateau_max(L):
        """The `max` direction — for a metric where **bigger is better**, an accuracy.

        Added because borch.ts had no `mode` at all: the min direction was written
        into the comparison, and torch's own way of spelling the call,
        `ReduceLROnPlateau(opt, 'min', 0.1)`, put a string where the factor goes.
        A signature check found it; this is the case that holds the behaviour, since
        a signature check cannot say the two directions compute the same thing.

        **The metrics are not the mirror image of `plateau`'s.** They were at first,
        and `test_case_names.py` refused them: an exact mirror produces the exact
        same trajectory, so the case would have read as green whether `max` worked or
        was quietly still `min`. A case named after an argument has to be one the
        argument changes.
        """
        p = L.tensor([1.0], requires_grad=True)
        opt = L.optim.SGD([p], lr=1.0)
        sch = L.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=1, factor=0.5)
        seen = []
        for metric in [0.1, 0.2, 0.2, 0.2, 0.2, 0.9, 0.2, 0.2]:
            sch.step(metric)
            seen.append(opt.param_groups[0]["lr"])
        return L.tensor(seen)

    cases.append(("sched::ReduceLROnPlateau(max)", plateau_max))

    # Recurrence and transformers — the weights have to be planted for the three libraries to
    # start from the same place. That the parameter **names** must match torch's for a state_dict
    # to load is caught here too.
    seq_x = inp["seq_x"]

    def recurrent(L, kind, batch_first=False):
        mod = getattr(L.nn, kind)(3, 4, batch_first=batch_first)
        # The weights come from `golden_inputs()` — that is what sends them out in the JSON so
        # an implementation that is not Python can **start from the same place.**
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

    # **`batch_first` had no case, and the binding was dropping it.** Every case above
    # passes the default, so the flag could be taken and thrown away and the values
    # still agreed — which is how it survived: `GRU(3, 4, batch_first=True)` set
    # nothing at all and said nothing. The two axes cannot see it either, one reading
    # declarations and the other never crossing into Python.
    #
    # Asked as **the same answer transposed**: the flag is honoured exactly when
    # feeding `(batch, length, …)` gives back what `(length, batch, …)` gave, turned.
    # A layer ignoring it reads `(2, 5, 3)` as length 2 and batch 5 and hands back
    # `(2, 5, 4)` where the answer is `(5, 2, 4)` — measured, so the two cannot even
    # be subtracted. `seq_x` being 5×2 is what gives that its teeth, and it was
    # checked rather than assumed.
    #
    # **"Rather than square" was the reason written here, and it is too weak a
    # condition.** A third session met the same axis flip from the site's side and
    # measured where it hides: `[1, 6, 8]` is not square, and read as `(len=1,
    # batch=6)` it still comes back `[1, 6, 8]` after the transpose — the shapes
    # round-trip and nothing can be subtracted apart. It stayed silent until a `[6,
    # 6]` mask arrived and broke the symmetry.
    #
    # So the condition is **both axes not 1, and different from each other.** 5×2
    # satisfies it and this case is sound; the sentence describing why did not, and a
    # sentence that admits `[1, 6]` is one somebody will satisfy while losing the
    # teeth. Written down here because the session that measured it has ended, and a
    # reason kept only in a conversation is a reason that goes with the conversation.
    def batch_first_matches(L, kind):
        turned = recurrent(L, kind, batch_first=True)(
            L.tensor(seq_x).transpose(0, 1))[0].transpose(0, 1)
        plain = recurrent(L, kind)(L.tensor(seq_x))[0]
        return (turned - plain).abs().max()

    for kind in ("RNN", "LSTM", "GRU"):
        cases.append((f"seq::{kind}/batch_first 는 같은 답을 돌려놓는다",
                      lambda L, k=kind: batch_first_matches(L, k)))

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
    # A causal mask is **float** (0/-inf). Lump it as "non-zero means masked" and it parts here.
    cases.append(("seq::MultiheadAttention(인과 마스크)",
                  lambda L: attention(L, lambda LL: LL.nn.Transformer
                                      .generate_square_subsequent_mask(5))))

    # ── The transformer ────────────────────────────────────────────────────
    #
    # **The weights are built from each parameter's own shape**, not read from a
    # fixture, so nothing has to be carried across by hand — and the two sides only
    # agree if their `state_dict` **keys and shapes** agree, which is half of what
    # this case is asking. The other half is the wiring: attention, feed-forward,
    # two residuals and two normalisations, in an order that `norm_first` reverses.
    def _ramp(L, mod):
        """`0.1·((i mod 7) − 3)` down each parameter, flat. The same rule in both
        languages, so no array crosses the boundary."""
        filled = {}
        for name, value in mod.state_dict().items():
            n = int(np.prod(value.shape)) if len(value.shape) else 1
            flat = np.array([0.1 * ((i % 7) - 3) for i in range(n)], dtype=np.float32)
            filled[name] = L.tensor(flat.reshape(tuple(value.shape)))
        mod.load_state_dict(filled)
        return mod

    _tseq = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) * 0.1 - 1.0

    def _encoder_layer(L, norm_first=False):
        m = L.nn.TransformerEncoderLayer(4, 2, 8, 0.0, "relu", 1e-5, True, norm_first)
        m.eval()
        return _ramp(L, m)(L.tensor(_tseq))

    # **`tsc` caught the first version of these**, because the sixth seat is now a
    # number and the case handed it `true`. The lucky direction: two arguments of
    # the same type would have compiled, and the case would have gone on asking
    # about a layer normalising with eps = 1 under the name `batch_first`.
    cases.append(("seq::TransformerEncoderLayer", lambda L: _encoder_layer(L)))
    # **`norm_first` moves the normalisation inside the residual.** Post-norm adds
    # then normalises; pre-norm normalises then adds, and asked at one setting alone
    # the other branch is never run.
    cases.append(("seq::TransformerEncoderLayer(norm_first)",
                  lambda L: _encoder_layer(L, True)))

    def _encoder_stack(L):
        proto = L.nn.TransformerEncoderLayer(4, 2, 8, 0.0, "relu", 1e-5, True)
        m = L.nn.TransformerEncoder(proto, 2)
        m.eval()
        return _ramp(L, m)(L.tensor(_tseq))

    cases.append(("seq::TransformerEncoder(2층)", _encoder_stack))

    def _decoder_layer(L):
        m = L.nn.TransformerDecoderLayer(4, 2, 8, 0.0, "relu", 1e-5, True)
        m.eval()
        mem = L.tensor(_tseq * -0.5)
        return _ramp(L, m)(L.tensor(_tseq), mem)

    # **The one that reads `memory`.** A decoder layer that quietly ignored it would
    # train and converge and never once look at the encoder, and every shape would
    # be right — so the memory here is a *different* tensor from the target.
    cases.append(("seq::TransformerDecoderLayer", _decoder_layer))

    cases.append(("seq::Transformer/마스크",
                  lambda L: L.nn.Transformer.generate_square_subsequent_mask(4)))

    # RMSprop — the one optimizer the golden cases were not looking at
    cases.append(("train::RMSprop/0.weight", lambda L: dict(
        trained(L, "RMSprop").named_parameters())["0.weight"]))
    return cases


WEBGPU_PREFIX = "webgpu::"


def webgpu_cases(inp=None):
    """**The things only the browser implementation has.**

    The core refuses these deliberately — the curriculum does not use them, and a growing surface
    grows the places that can be quietly wrong. The browser side has a different charter
    (performance, real models) and puts them in.

    The expected values are frozen with **real torch.** The core skips these cases — the two
    implementations' scopes diverge and the harness has to express that. The harness recognises
    the browser side by `hasattr(lib, "backend")`.
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

    # The 3-D family. Only conv3d's backward pass rides `tf.grad` and is slow, and **slow is not
    # wrong** — whether the value is right is held here, and the code itself warns that it is slow.
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

    # Freezing conv3d caught `tf.pad` at rank 5 giving **the right shape with broken values.** It
    # throws nothing, so the caller knows nothing. conv3d was fixed, and that is not the only place
    # calling the same function — slicing's backward pass uses `pad` to fill the cut-away region
    # back with zeros. If that input is rank 5, **a wrong gradient comes out quietly.** So rather
    # than sweeping by eye and concluding "there is none", a place that would be caught is stood
    # up and asked.
    def slice5_grad(kind):
        def run(L, k=kind):
            x = L.tensor(vol, requires_grad=True)
            if k == "narrow":
                out = L.narrow(x, 2, 1, 2)
            elif k == "unbind":
                out = L.unbind(x, 2)[1]
            else:
                out = L.split(x, 2, dim=3)[0]
            # The weights have to differ for which slot ought to be 0 to show in the value —
            # a plain sum() makes every gradient 1, and then swapped slots are not caught.
            (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
            return _grad_of(x, f"랭크5 {k}")
        return run

    for kind in ("narrow", "unbind", "split"):
        cases.append((WEBGPU_PREFIX + f"grad::랭크5 {kind}", slice5_grad(kind)))

    # `pad_sequence`'s **gradient.** The values agree between the core and the sister library, so
    # the four shared cases see them, and the gradient exists on the sister side only. The core
    # fills the padding with numpy and hands back a bare tensor, so the graph is cut — measured,
    # `backward()` refuses with "a tensor that is not requires_grad". Real torch differentiates it,
    # so that is a hole in the core, and keeping this case sister-only is not there to cover that
    # hole but **to hold on to the fact that the sister side does not cut.**
    # (It refuses loudly rather than being quietly wrong, so it is not urgent.)
    def pad_sequence_grad(L):
        a = L.tensor(np.array([[1., 2.], [3., 4.]], dtype=np.float32), requires_grad=True)
        b = L.tensor(np.array([[5., 6.]], dtype=np.float32), requires_grad=True)
        out = L.nn.utils.rnn.pad_sequence([a, b])
        (out * L.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        return _grad_of(a, "pad_sequence")

    cases.append((WEBGPU_PREFIX + "grad::pad_sequence", pad_sequence_grad))
    cases += _highrank_battery(HIGH_RANKS, inp)
    cases += _rank_ceiling_cases(CEILING_RANKS, inp)

    # `F.pad` is **another door** onto `tf.pad` through the public API. Fixing slicing's backward
    # pass alone and not looking here left the same bug intact on the path a user calls directly.
    # Rank 6 and above is covered by the battery. These are the two below that.
    for rank, src in (("랭크4", img), ("랭크5", vol)):
        cases.append((WEBGPU_PREFIX + f"F.pad({rank})",
                      lambda L, s=src: L.nn.functional.pad(L.tensor(s), (1, 2))))
        # Filled with a value other than 0 too. At high rank a different code path runs for 0
        # (`fill` rather than `zeros`), so measuring 0 alone leaves that one untravelled.
        cases.append((WEBGPU_PREFIX + f"F.pad({rank}, 값)",
                      lambda L, s=src: L.nn.functional.pad(
                          L.tensor(s), (2, 1, 1, 0), value=-1.5)))
    return cases


# The ranks at which the sister library **claims its values equal torch's.** It goes in here only
# after passing the whole battery.
HIGH_RANKS = (6,)
# Above that. A band where **only some operations are refused** for want of a TF.js kernel, so it
# is looked at separately — the details are written in `_rank_ceiling_cases`.
CEILING_RANKS = (7, 8)


def _as_expected(fn):
    """How to hold a place **torch manages and the browser implementation refuses** in the golden
    answers.

    The golden answers are frozen with real torch, and here the browser side is **deliberately
    unlike** torch. So asking by value would leave it parted forever. Instead of the value it asks
    "did it behave as its documentation says" — success is right for torch and refusal is right
    for the browser side, so both behaving properly gives the same answer.

    If the browser side quietly starts returning a value one day (right or wrong), it parts here.
    **That really happened** — pulling out the TF.js edition made all seven places that refused
    rank 7 and 8 produce values, and this device reported it as seven "unexpected successes". What
    was done then was to rewrite the limit **deliberately**, not to delete the cases. That it must
    not widen of its own accord is why this function exists.
    """
    def run(L):
        # The floor is a GPU buffer, so it is not shared out as views. There is no double
        # precision either. Those two reasons make the few places that gather here.
        must_reject = hasattr(L, "backend")
        try:
            fn(L)
        except Exception as exc:                                # noqa: BLE001
            return "기대대로" if must_reject else f"뜻밖의 거절 <{type(exc).__name__}>"
        return "뜻밖의 성공" if must_reject else "기대대로"
    return run


def _rank_ceiling_cases(ranks, inp):
    """Rank 7 and above — **the band where what works and what does not part.**

    **This table was rebuilt once.** It used to pin TF.js's ceiling — measured, it threw
    `GPU for rank 7 is not yet supported` from rank 7 up, and not for everything but for some
    (element-wise, permute and reshape ran; axis reductions and `fill` did not), so "rank 7 works"
    and "rank 7 does not" were both false. So the working side was frozen by value and the
    non-working side had the refusal itself frozen through `_as_expected`.

    Pulling out TF.js removed that ceiling. Hand-written WGSL has no rank limit, and the seven
    places that refused **all produce values.** So they were all rewritten by value — exactly as
    `_as_expected`'s comment had written in advance: "it parts if TF.js later fills in the
    high-rank kernels. That means rewriting the limit **deliberately** then, and it must not widen
    of its own accord."

    **A refusal turning into a value is not free.** Not throwing and being the right value are
    different statements, and the old table could ask only the first. Now it is matched against
    real torch's value.
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

        # The three TF.js refused. They produce values now.
        cases += [
            (WEBGPU_PREFIX + f"{tag} 합(축)",
             lambda L, a=v, ax=axis: L.tensor(a).sum(dim=ax)),
            (WEBGPU_PREFIX + f"F.pad({tag}, 값)",
             lambda L, a=v: L.nn.functional.pad(L.tensor(a), (2, 1, 1, 0), value=-1.5)),
            (WEBGPU_PREFIX + f"grad::{tag} 원소별", elemwise_grad),
        ]

    # **The history written here is not deleted.** The TF.js-era boundary fell cleanly on neither
    # the operation name nor the input rank — rank 7 worked for both the forward pass and the
    # gradient, and rank 8 produced a value with no gradient. So four were written down separately,
    # and all four of those places produce values now.
    #
    # It was written at first as "unbinding a rank 8 gives a rank 7 result, so it will be refused",
    # and asking showed the forward pass passed. The failure seen earlier had been the **gradient's**
    # and not the forward pass's, and the `grad::` on the failing name went unread while a cause
    # was invented. That writing a boundary from a guess turns the guess into documentation is why
    # these four lines remain.
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


# **Changing a frozen case's inputs is a two-file change, and the case name does not
# say so.**
#
# Adding a case is safe — the TS side owes one more and the ledger row counts it.
# Removing one is safe — the row goes down. Changing the *inputs* of a case that is
# already ported is the only edit where two files have to move together while neither
# mentions the other: `borch-ts/test/cases.ts` writes the same arguments out
# independently, so the name stays identical and the values stop agreeing.
#
# It is caught, and caught **loudly** — the browser runner reports a value mismatch on
# a named case. It is caught one merge too late, on main, by whoever pushes next, and
# they read a value mismatch on a case they did not touch and start looking at the
# implementation.
#
# So the rule is not another check. **Before committing a change to a case's inputs,
# grep `borch-ts/test/cases.ts` for that case's name.** If it is there, the two move
# together.
#
# **And that grep is not sufficient, which is the half worth knowing.** It reads main,
# and a port in flight is invisible from there — the porter is always in flight. It
# happened within the hour of this paragraph being written: the `Pad(edge)` cases went
# from padding 1 to padding 2 after a grep of main came back clean, and three ported
# cases went red on the other side because that port had not landed yet. The editor
# cannot see the branch. **The other half of this rule lives on the porting side** —
# re-run the golden after rebasing rather than assuming ported cases still hold — and
# neither half covers the gap alone.
#
# Written after `F.adjust_saturation(uint8)` moved from 1.7 to 0.1 — and the sequel is
# the reason it is worth a paragraph. That one number was what made the other side's
# float32 chain measurable: the same mutation that left ten of their cases green the
# day before now moves four pixels by a whole step. **A check can be conferred on one
# library by an input written in another**, which is also why quietly changing one is
# not a local edit.
VISION_PREFIX = "vision::"
_BT_VISION = None


def _is_real_torch(L):
    return getattr(L, "__name__", "") == "torch"


def _vision(L):
    """Hands back the torchvision that pairs with `L` — for real torch, **real torchvision.**

    That is what this table is worth. Comparing our transforms against our own expectations proves
    nothing.
    """
    if _is_real_torch(L):
        from torchvision import transforms as real
        return real
    global _BT_VISION
    if _BT_VISION is None:
        try:                                    # in the browser, /work is on the path
            import borchvision as mod
        except ImportError:                     # natively, point at the repository root
            import importlib.util
            import pathlib
            path = pathlib.Path(__file__).resolve().parent.parent / "borchvision.py"
            spec = importlib.util.spec_from_file_location("borchvision", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        _BT_VISION = mod
    _BT_VISION.use(L)                           # which library's tensors get built is decided here
    return _BT_VISION.transforms


def _vision_ops(L):
    """`borchvision.ops` for us, `torchvision.ops` for real torch."""
    if _is_real_torch(L):
        import torchvision.ops as real
        return real
    _vision(L)                                  # loads the module and binds the library
    import sys as _sys
    return _sys.modules["borchvision"].ops


def _vision_v2(L):
    """`transforms.v2` for each side. Fetched through `_vision` so that the module is
    loaded and bound once, rather than a second copy living beside the first."""
    if _is_real_torch(L):
        from torchvision.transforms import v2 as real
        return real
    return _vision(L).v2


def _as_numpy(x):
    """Out of whichever library's tensor, into numpy — the ops return the kind they
    were given and these cases hand them numpy, so this is mostly for torch's side."""
    take = getattr(x, "numpy", None)
    return take() if callable(take) else np.asarray(x)


def _size_input(L):
    """The picture the size questions are asked of, in each side's own format.

    Real torchvision reads a **tensor**'s trailing axes and ours reads an array's
    leading ones, so handing both the same object would ask two different questions
    and call the disagreement a defect.
    """
    from_inputs = golden_inputs()["vis_f"]
    if _is_real_torch(L):
        return L.tensor(np.ascontiguousarray(from_inputs.transpose(2, 0, 1)))
    return from_inputs


def _pil_position(L, arr):
    """What arrives here from torchvision is **a PIL image**, and we have no PIL, so an (H,W,C)
    array stands in its place. The same picture is given in each side's own format — compared
    without matching the formats it is not a comparison but a coincidence."""
    if _is_real_torch(L):
        from PIL import Image
        return Image.fromarray(arr)
    return arr


def _as_tensor(L, arr):
    """The golden harness takes values out with `.detach().numpy()`. This brings what came out as
    a PIL image or an array into that shape."""
    return L.tensor(np.ascontiguousarray(np.asarray(arr, dtype=np.float32)))


def vision_cases(inp=None):
    """`borchvision` — torchvision's `transforms` only.

    **A random transform's draw cannot be compared.** We cannot use torch's random generator. So
    the probability is pinned at 0 or 1, or the crop is made to have only one possible position,
    and **only the deterministic places** are asked. Whether the drawing itself works properly is
    looked at by pytest as a distribution — passing over it here as "it is random, so it cannot be
    compared" would be writing down that something unlooked-at was looked at.
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

    def resize(L, size, interpolation):
        # **torchvision's default has antialiasing on.** The difference from having it off is
        # at most 0.0301 going 8×8→4×4 (measured), so leaving it undecided leaves a hole in the
        # specification. We went with it on, and that claim is compared here against real
        # torchvision.
        # **A float image is used.** With uint8 the 255 parts because of the order — real
        # torchvision takes a tensor so `ToTensor` divides first, and ours takes an array so
        # `Resize` comes first and its result is float, so `ToTensor` does not divide. Measured, a
        # max difference of 247 came out, and that was not the resizing being wrong but the case
        # wiring two pipelines differently.
        T = _vision(L)
        if _is_real_torch(L):
            from torchvision.transforms import InterpolationMode
            mode = (InterpolationMode.BILINEAR if interpolation == "bilinear"
                    else InterpolationMode.NEAREST)
            return T.Resize(size, interpolation=mode, antialias=True)(
                _as_tensor(L, T.ToTensor()(img_f)))
        return T.ToTensor()(T.Resize(size, interpolation)(img_f))

    def center_crop(L, size):
        # The branch where the crop is larger than the original is looked at alongside —
        # torchvision pads with 0, and refusing makes the same code part between two libraries.
        T = _vision(L)
        if _is_real_torch(L):
            return T.CenterCrop(size)(_as_tensor(L, T.ToTensor()(img_f)))
        return T.ToTensor()(T.CenterCrop(size)(img_f))

    def crop(L, size, padding, padding_mode="constant"):
        # The size is set so that there is **only one** place to crop. That makes it deterministic
        # regardless of the draw.
        T = _vision(L)
        out = T.RandomCrop(size, padding=padding, padding_mode=padding_mode)(
            _pil_position(L, img_u8))
        return _as_tensor(L, out)

    cases = [
        # ToTensor's crux is that it **divides by 255 only for uint8.** Divide a float once more
        # and it is 255 times darker with no exception, and only the training quietly fails.
        (VISION_PREFIX + "ToTensor(uint8)", lambda L: _vision(L).ToTensor()(img_u8)),
        (VISION_PREFIX + "ToTensor(실수)", lambda L: _vision(L).ToTensor()(img_f)),
        (VISION_PREFIX + "ToTensor(2차원)", lambda L: _vision(L).ToTensor()(gray)),
        (VISION_PREFIX + "Normalize", normalize),
        (VISION_PREFIX + "Compose", compose),
        (VISION_PREFIX + "Flip(p=1)", lambda L: flip(L, 1.0)),
        (VISION_PREFIX + "Flip(p=0)", lambda L: flip(L, 0.0)),
        (VISION_PREFIX + "Crop(패딩없음)", lambda L: crop(L, (5, 4), 0)),
        # Pad and then size it exactly and there is one place to crop — the padding itself is
        # what gets compared.
        (VISION_PREFIX + "Crop(패딩1)", lambda L: crop(L, (7, 6), 1)),
        # **`padding_mode` had no case at all**, and until this commit borch.ts had no
        # parameter for it either: `RandomCrop` there took `(size, padding, fill)` and
        # padded with a constant whatever was asked. The three non-constant modes are the
        # ones a reader would notice — an edge repeated, mirrored, or mirrored including
        # the edge — and each is a different picture at the border.
        #
        # The same size as the case above, so there is still one place to crop and the
        # draw stays out of it.
        (VISION_PREFIX + "Crop(패딩1, edge)", lambda L: crop(L, (7, 6), 1, "edge")),
        (VISION_PREFIX + "Crop(패딩1, reflect)", lambda L: crop(L, (7, 6), 1, "reflect")),
        (VISION_PREFIX + "Crop(패딩1, symmetric)", lambda L: crop(L, (7, 6), 1, "symmetric")),
        # Resizing. **Both shrinking and enlarging are looked at** — antialiasing works only when
        # shrinking, so with enlarging cases alone that rule is never engaged.
        (VISION_PREFIX + "Resize(줄임·겹선형)", lambda L: resize(L, (4, 3), "bilinear")),
        (VISION_PREFIX + "Resize(늘림·겹선형)", lambda L: resize(L, (11, 9), "bilinear")),
        (VISION_PREFIX + "Resize(짧은변)", lambda L: resize(L, 4, "bilinear")),
        (VISION_PREFIX + "Resize(최근접)", lambda L: resize(L, (4, 3), "nearest")),
        # **A crop position that is odd is chosen.** Python's round sends a half to the even side
        # and JS rounds up — that difference put the TypeScript edition one slot out and parted a
        # maximum of 0.837 (measured). Tested at even sizes only, that place is never engaged.
        (VISION_PREFIX + "CenterCrop(짝수)", lambda L: center_crop(L, (4, 4))),
        (VISION_PREFIX + "CenterCrop(홀수)", lambda L: center_crop(L, (5, 3))),
        (VISION_PREFIX + "CenterCrop(원본보다 큼)", lambda L: center_crop(L, (13, 11))),
    ]

    # --- the fourteen that arrived after the first seven ----------------------
    #
    # **Each side is given the picture in its own format**, which is the rule the cases
    # above already follow: real torchvision gets a tensor (or a PIL image where the
    # picture is uint8), and ours gets the (H,W,C) array that stands in for PIL. Handing
    # both the same object would compare a coincidence.

    def on_float(build):
        """A transform that takes a picture, compared on the float image.

        The orders differ and have to. torchvision's transform wants a tensor, so
        `ToTensor` comes **first** there; ours wants the array, so `ToTensor` comes
        **last**. Both end at (C,H,W), which is the only place they can be compared.
        """
        def run(L):
            T = _vision(L)
            t = build(T)
            if _is_real_torch(L):
                return t(_as_tensor(L, T.ToTensor()(img_f)))
            return T.ToTensor()(t(img_f))
        return run

    def on_uint8(build):
        """The same, on the uint8 image — where torchvision's side is **a real PIL
        image.** The two paths inside torchvision are not the same code, and a transform
        that agrees with the tensor path can still part from the PIL one."""
        def run(L):
            T = _vision(L)
            out = build(T)(_pil_position(L, img_u8))
            return _as_tensor(L, out)
        return run

    def crops(build):
        """`FiveCrop` and `TenCrop` hand back **a tuple**, and the harness compares
        arrays. Stacked, a crop landing in the wrong slot is caught by value; compared one
        by one it would not be."""
        def run(L):
            T = _vision(L)
            t = build(T)
            if _is_real_torch(L):
                return L.stack(t(_as_tensor(L, T.ToTensor()(img_f))))
            return L.stack([T.ToTensor()(p) for p in t(img_f)])
        return run

    def linear(L):
        # A reversing matrix rather than the identity — the identity passes whatever the
        # multiplication does, including doing nothing at all.
        n = int(np.prod(img_f.shape))
        m = np.eye(n, dtype=np.float32)[::-1].copy()
        v = np.full(n, 0.5, dtype=np.float32)
        T = _vision(L)
        return T.LinearTransformation(L.tensor(m), L.tensor(v))(
            _as_tensor(L, T.ToTensor()(img_f)))

    def resized_crop_nearest(L):
        # **torchvision takes the enum here and not the string.** `RandomResizedCrop` is
        # stricter than the tutorials read — `interpolation="nearest"` stops it with "should
        # be a InterpolationMode". Ours takes either, so this is the one place the case has
        # to spell the same filter in two spellings.
        T = _vision(L)
        if _is_real_torch(L):
            from torchvision.transforms import InterpolationMode
            crop = T.RandomResizedCrop((3, 2), scale=(1.0, 1.0), ratio=(0.8, 0.8),
                                       interpolation=InterpolationMode.NEAREST)
            return crop(_as_tensor(L, T.ToTensor()(img_f)))
        crop = T.RandomResizedCrop((3, 2), scale=(1.0, 1.0), ratio=(0.8, 0.8),
                                   interpolation="nearest")
        return T.ToTensor()(crop(img_f))

    def erasing(L, **kw):
        T = _vision(L)
        return T.RandomErasing(**kw)(_as_tensor(L, T.ToTensor()(img_f)))

    cases += [
        # **Padding's two-element form is the one that misreads** — it is (left/right,
        # top/bottom), not (left, top). Given a square pad both readings agree, so the
        # four-sided case is the one that decides it.
        (VISION_PREFIX + "Pad(all sides)", on_float(lambda T: T.Pad(2))),
        (VISION_PREFIX + "Pad(four sides)", on_float(lambda T: T.Pad((1, 2, 3, 4)))),
        # The three non-constant modes. They are numpy's own, so what is being asked is
        # whether **torchvision means the same thing by the same word** — `reflect` and
        # `symmetric` differ only in whether the edge value repeats.
        #
        # **They pad by two, and one is the reason.** At a padding of one, `edge` and
        # `symmetric` are the same picture — symmetric mirrors the edge value, which is
        # the edge value — so the two entries held identical numbers and the pair could
        # not tell the modes apart. Another session swapped the two in their port and
        # all sixteen value cases still passed. Two is the smallest padding where the
        # words diverge.
        (VISION_PREFIX + "Pad(edge)", on_float(lambda T: T.Pad(2, padding_mode="edge"))),
        (VISION_PREFIX + "Pad(reflect)", on_float(lambda T: T.Pad(2, padding_mode="reflect"))),
        (VISION_PREFIX + "Pad(symmetric)", on_float(lambda T: T.Pad(2, padding_mode="symmetric"))),
        # A colour per channel, through PIL. numpy's `constant_values` reads per **axis**,
        # so a three-colour fill handed straight to it paints the channel axis instead of
        # the colours — right shape, wrong picture, nothing raised.
        (VISION_PREFIX + "Pad(a colour per channel)", on_uint8(lambda T: T.Pad(1, fill=(1, 2, 3)))),
        # Grayscale. **The three-channel form is the one models need**, and it is where a
        # broadcast can quietly give three different channels.
        (VISION_PREFIX + "Grayscale(one channel)", on_float(lambda T: T.Grayscale())),
        (VISION_PREFIX + "Grayscale(three channels)", on_float(lambda T: T.Grayscale(3))),
        (VISION_PREFIX + "RandomGrayscale(p=1)", on_float(lambda T: T.RandomGrayscale(p=1.0))),
        # The vertical flip, both ways round. p=0 is not a formality: a flip written on the
        # wrong axis still passes p=1 on a square picture, and this one is 5x4.
        (VISION_PREFIX + "VerticalFlip(p=1)", on_float(lambda T: T.RandomVerticalFlip(1.0))),
        (VISION_PREFIX + "VerticalFlip(p=0)", on_float(lambda T: T.RandomVerticalFlip(0.0))),
        # Five and ten crops. The corners are easy to write and easy to swap; stacking is
        # what makes a swap show.
        (VISION_PREFIX + "FiveCrop", crops(lambda T: T.FiveCrop((3, 2)))),
        (VISION_PREFIX + "TenCrop", crops(lambda T: T.TenCrop((3, 2)))),
        (VISION_PREFIX + "TenCrop(vertical)", crops(lambda T: T.TenCrop((3, 2), vertical_flip=True))),
        # **The functional spellings of the three above.** They were absent from
        # `vision.ts` while their classes were there, and nothing said so: `ts_axis.py`
        # leaves `transforms` out because "the golden's `vision::` cases hold it name by
        # name", and for these three the golden had no case. Each check was waiting on
        # the other, so the seam between them is where they sat.
        #
        # These compare the same values the class cases do. That is the point — a
        # functional form that delegates and a functional form that reimplements read
        # the same from outside, and only one of them stays right.
        (VISION_PREFIX + "F.five_crop",
         crops(lambda T: lambda img: T.functional.five_crop(img, (3, 2)))),
        (VISION_PREFIX + "F.ten_crop",
         crops(lambda T: lambda img: T.functional.ten_crop(img, (3, 2)))),
        (VISION_PREFIX + "F.ten_crop(vertical)",
         crops(lambda T: lambda img: T.functional.ten_crop(img, (3, 2), vertical_flip=True))),
        # **`to_grayscale` gets no value case, and the reason is not "nobody wrote one".**
        # torchvision's is PIL-only — handed a tensor it says `Input should be PIL Image`
        # (measured) — so the only path where both sides accept the call is the uint8 one,
        # and there they are **documented to differ by one**: `borchvision.py`'s `_to_gray`
        # truncates because torch's `.to(dtype)` truncates, while PIL's `convert("L")`
        # rounds. Writing the case anyway would freeze a difference this repository chose
        # on purpose and read as though the choice were a defect.
        #
        # What holds it instead: ours delegates to `rgb_to_grayscale`, whose tensor path
        # the `Grayscale` cases above compare, and `tests/ts_axis.py` now carries the
        # `transforms` namespace so the **name** cannot go missing again unremarked. Both
        # halves are asked; they are asked in the two places that can ask them.
        # **The cap has to actually bite.** On this 5x4 picture most `max_size` values never
        # reach it — the long side barely grows — and the case then compares an ordinary
        # resize while reading as though it compared the cap. 8 with a cap of 9 is where it
        # triggers (measured: 10 becomes 9, and the short side follows to 7).
        (VISION_PREFIX + "Resize(long side capped)", on_float(lambda T: T.Resize(8, max_size=9))),
        # `RandomResizedCrop` pinned so that the draw has one answer: the whole area, at the
        # picture's own ratio. That leaves one place to crop, so what is compared is the
        # resize that follows and the rounding that chooses the crop.
        (VISION_PREFIX + "RandomResizedCrop(pinned to the whole image)",
         on_float(lambda T: T.RandomResizedCrop((3, 2), scale=(1.0, 1.0), ratio=(0.8, 0.8)))),
        # **The same crop with the other filter**, because an `interpolation` accepted and
        # then dropped on the way to the resize passes the case above exactly. The two
        # differ by 0.5006 here (measured), so a dropped argument cannot hide.
        (VISION_PREFIX + "RandomResizedCrop(nearest)", resized_crop_nearest),
        (VISION_PREFIX + "LinearTransformation", linear),
        (VISION_PREFIX + "RandomErasing(p=0)", lambda L: erasing(L, p=0.0)),
        # **The fallback, and it is the branch that gets written wrong.** When ten draws all
        # miss, torchvision hands the picture back untouched — an implementation that
        # erases "whatever it last computed" instead passes every other case.
        (VISION_PREFIX + "RandomErasing(ten draws all miss)",
         lambda L: erasing(L, p=1.0, scale=(0.99, 1.0), ratio=(1.0, 1.0))),
        # The composition three. Pinned at 0 and 1 they say whether **all or none** is
        # applied, which is what separates `RandomApply` from a `p` on each transform.
        (VISION_PREFIX + "RandomApply(p=1)", on_float(lambda T: T.RandomApply([T.Pad(1)], p=1.0))),
        (VISION_PREFIX + "RandomApply(p=0)", on_float(lambda T: T.RandomApply([T.Pad(1)], p=0.0))),
        (VISION_PREFIX + "RandomChoice(one to choose from)",
         on_float(lambda T: T.RandomChoice([T.Pad(1)]))),
        (VISION_PREFIX + "RandomOrder(one to order)", on_float(lambda T: T.RandomOrder([T.Pad(1)]))),
        (VISION_PREFIX + "Lambda", on_float(lambda T: T.Lambda(lambda x: x * 2))),
    ]

    # --- transforms.functional ------------------------------------------------
    #
    # The functions hand their work to the classes above, so most of their values are
    # already frozen through those. **What is asked here is what the classes cannot
    # reach**: a crop at a position nobody drew, an erase that actually erases, and a
    # size whose order is the opposite of every other size in the file.

    def fn(call):
        """A `functional` call, in each side's own format — `on_float`'s rule."""
        def run(L):
            T = _vision(L)
            F = T.functional
            if _is_real_torch(L):
                return call(F, _as_tensor(L, T.ToTensor()(img_f)))
            return T.ToTensor()(call(F, img_f))
        return run

    def erase_case(L):
        # **The first case in this table where anything is erased.** `RandomErasing`'s
        # two are the branch pinned at p=0 and the branch where ten draws all miss — an
        # implementation that erased the wrong rectangle, or filled it with the wrong
        # number, passes both.
        T = _vision(L)
        x = _as_tensor(L, T.ToTensor()(img_f))
        v = np.full((3, 2, 2), 0.25, dtype=np.float32)
        return T.functional.erase(x, 1, 1, 2, 2,
                                  L.tensor(v) if _is_real_torch(L) else v)

    cases += [
        # A crop at a position **given rather than drawn**. Every crop in the table above
        # goes through a draw pinned to one answer; this is the only one where the four
        # numbers are the case.
        (VISION_PREFIX + "F.crop", fn(lambda F, x: F.crop(x, 1, 1, 3, 2))),
        (VISION_PREFIX + "F.resized_crop",
         fn(lambda F, x: F.resized_crop(x, 1, 0, 3, 4, [2, 2]))),
        # `Pad`'s two-element form, which is (left/right, top/bottom) and reads as
        # (left, top). The class's case gives four numbers, where both readings agree.
        (VISION_PREFIX + "F.pad(two numbers)", fn(lambda F, x: F.pad(x, [1, 2], 0.5))),
        (VISION_PREFIX + "F.erase", erase_case),
        # **`get_image_size` is width first** and everything else here is height first.
        # Frozen as text so a swap is a different string rather than a plausible pair.
        (VISION_PREFIX + "F.sizes",
         lambda L: f"{_vision(L).functional.get_dimensions(_size_input(L))} "
                   f"{_vision(L).functional.get_image_size(_size_input(L))} "
                   f"{_vision(L).functional.get_image_num_channels(_size_input(L))}"),
    ]

    # --- the photometric five, and the jitter that draws them ------------------
    #
    # **These go through the tensor path on both sides, uint8 included**, and that is
    # the one place this table does not hand torchvision a PIL image. torchvision has
    # two implementations of every one of them — `ImageEnhance` for PIL and this
    # arithmetic for tensors — and they do not agree. Ours copies the second, so
    # comparing against the first would be asking a question we have already answered
    # "no" to on purpose (`Grayscale` parts from PIL by one, measured).

    def photo(call, on_bytes=False):
        def run(L):
            T = _vision(L)
            F = T.functional
            src = img_u8 if on_bytes else img_f
            if _is_real_torch(L):
                x = L.tensor(np.ascontiguousarray(src.transpose(2, 0, 1)))
                out = call(F, x)
                return L.tensor(np.ascontiguousarray(
                    np.asarray(out.detach().numpy(), dtype=np.float32)))
            return L.tensor(np.ascontiguousarray(
                np.asarray(call(F, src), dtype=np.float32).transpose(2, 0, 1)))
        return run

    cases += [
        (VISION_PREFIX + "F.adjust_brightness(dark)",
         photo(lambda F, x: F.adjust_brightness(x, 0.5))),
        # **Above 1 it clamps**, and the clamp is the half of this that a factor below
        # one never reaches.
        (VISION_PREFIX + "F.adjust_brightness(bright)",
         photo(lambda F, x: F.adjust_brightness(x, 1.7))),
        (VISION_PREFIX + "F.adjust_contrast", photo(lambda F, x: F.adjust_contrast(x, 0.5))),
        (VISION_PREFIX + "F.adjust_saturation",
         photo(lambda F, x: F.adjust_saturation(x, 1.7))),
        # Hue is the only one that leaves RGB. A quarter turn and a small negative one,
        # because the wrap at 0 and the wrap at 1 are different lines of arithmetic.
        (VISION_PREFIX + "F.adjust_hue(quarter turn)",
         photo(lambda F, x: F.adjust_hue(x, 0.25))),
        (VISION_PREFIX + "F.adjust_hue(backwards)",
         photo(lambda F, x: F.adjust_hue(x, -0.1))),
        (VISION_PREFIX + "F.adjust_gamma", photo(lambda F, x: F.adjust_gamma(x, 2.2))),
        (VISION_PREFIX + "F.adjust_gamma(with gain)",
         photo(lambda F, x: F.adjust_gamma(x, 0.5, 0.5))),
        # **The uint8 branch, where the truncation lives**, at a factor that actually
        # reaches it. Every blend ends in a cast back, and one precision wider moves
        # values across that boundary — but not at every factor: this case was 1.7 and
        # float64 changed **nothing** on this picture there, so the case named in
        # `_working_dtype`'s docstring held no evidence for what the docstring claimed.
        # 0.1 moves four pixels of this same picture. Found by the sister library, who
        # deleted every narrowing in their port and watched ten cases stay green.
        (VISION_PREFIX + "F.adjust_saturation(uint8)",
         photo(lambda F, x: F.adjust_saturation(x, 0.1), on_bytes=True)),
        (VISION_PREFIX + "F.adjust_hue(uint8)",
         photo(lambda F, x: F.adjust_hue(x, 0.25), on_bytes=True)),
    ]

    def jitter(brightness, hue):
        """`ColorJitter` with **one factor pinned to a single value.** The draw then
        has one answer, and the order it also draws does not matter because there is
        only one thing to order — so the case asks whether the jitter reaches the
        right function, which is all a frozen value can ask of a random transform."""
        def run(L):
            T = _vision(L)
            # **The unused factor is left out rather than passed as `None`.**
            # torchvision reads `None` as "not a number and not a pair" and stops —
            # `None` is what it *stores* for a factor nobody asked for, not what it
            # takes.
            kw = {"brightness": brightness} if brightness else {"hue": hue}
            build = T.ColorJitter(**kw)
            if _is_real_torch(L):
                x = L.tensor(np.ascontiguousarray(img_f.transpose(2, 0, 1)))
                return L.tensor(np.ascontiguousarray(
                    np.asarray(build(x).detach().numpy(), dtype=np.float32)))
            return L.tensor(np.ascontiguousarray(
                np.asarray(build(img_f), dtype=np.float32).transpose(2, 0, 1)))
        return run

    cases += [
        (VISION_PREFIX + "ColorJitter(brightness pinned)", jitter((0.6, 0.6), None)),
        (VISION_PREFIX + "ColorJitter(hue pinned)", jitter(None, (0.2, 0.2))),
    ]

    # --- the six that rewrite pixels ------------------------------------------
    #
    # Same rule as the photometric five: **the tensor implementation on both sides**,
    # uint8 included. `posterize` and `equalize` are uint8 only over there, so their
    # cases have no float half to ask about.

    cases += [
        (VISION_PREFIX + "F.invert", photo(lambda F, x: F.invert(x))),
        (VISION_PREFIX + "F.invert(uint8)", photo(lambda F, x: F.invert(x), on_bytes=True)),
        # **The masking, at three widths.** One bit is the extreme, four is the usual,
        # and eight has to be the identity — a mask computed as `2 ** (8 - bits)` gets
        # that last one wrong in the obvious way.
        (VISION_PREFIX + "F.posterize(one bit)",
         photo(lambda F, x: F.posterize(x, 1), on_bytes=True)),
        (VISION_PREFIX + "F.posterize(four bits)",
         photo(lambda F, x: F.posterize(x, 4), on_bytes=True)),
        (VISION_PREFIX + "F.posterize(all eight)",
         photo(lambda F, x: F.posterize(x, 8), on_bytes=True)),
        (VISION_PREFIX + "F.solarize", photo(lambda F, x: F.solarize(x, 0.5))),
        (VISION_PREFIX + "F.solarize(uint8)",
         photo(lambda F, x: F.solarize(x, 128), on_bytes=True)),
        (VISION_PREFIX + "F.autocontrast", photo(lambda F, x: F.autocontrast(x))),
        (VISION_PREFIX + "F.autocontrast(uint8)",
         photo(lambda F, x: F.autocontrast(x), on_bytes=True)),
        (VISION_PREFIX + "F.equalize", photo(lambda F, x: F.equalize(x), on_bytes=True)),
        # Sharpness at 0 is the blur itself and at 2 is the sharpening — **and the
        # border is the original in both**, because the convolution has no padding and
        # the result is written back into the middle.
        (VISION_PREFIX + "F.adjust_sharpness(blurred)",
         photo(lambda F, x: F.adjust_sharpness(x, 0.0))),
        (VISION_PREFIX + "F.adjust_sharpness(sharpened)",
         photo(lambda F, x: F.adjust_sharpness(x, 2.0))),
        # **The uint8 one is where the rounding lives.** torch casts the convolution
        # back through `round`; truncating instead is one step low on about half the
        # pixels, and every other sharpness case passes with it wrong.
        (VISION_PREFIX + "F.adjust_sharpness(uint8)",
         photo(lambda F, x: F.adjust_sharpness(x, 2.0), on_bytes=True)),
    ]

    def wrapper(build, on_bytes=False):
        """A `Random…` wrapper with its probability pinned. p=0 is not a formality
        here: five of the six share one implementation, and a wrapper that applied its
        op whatever the draw said passes every p=1 case there is."""
        def run(L):
            T = _vision(L)
            src = img_u8 if on_bytes else img_f
            if _is_real_torch(L):
                x = L.tensor(np.ascontiguousarray(src.transpose(2, 0, 1)))
                return L.tensor(np.ascontiguousarray(
                    np.asarray(build(T)(x).detach().numpy(), dtype=np.float32)))
            return L.tensor(np.ascontiguousarray(
                np.asarray(build(T)(src), dtype=np.float32).transpose(2, 0, 1)))
        return run

    cases += [
        (VISION_PREFIX + "RandomInvert(p=1)", wrapper(lambda T: T.RandomInvert(p=1.0))),
        (VISION_PREFIX + "RandomInvert(p=0)", wrapper(lambda T: T.RandomInvert(p=0.0))),
        (VISION_PREFIX + "RandomEqualize(p=1)",
         wrapper(lambda T: T.RandomEqualize(p=1.0), on_bytes=True)),
        (VISION_PREFIX + "RandomPosterize(p=1)",
         wrapper(lambda T: T.RandomPosterize(3, p=1.0), on_bytes=True)),
        (VISION_PREFIX + "RandomSolarize(p=1)",
         wrapper(lambda T: T.RandomSolarize(0.4, p=1.0))),
        (VISION_PREFIX + "RandomAutocontrast(p=1)",
         wrapper(lambda T: T.RandomAutocontrast(p=1.0))),
        (VISION_PREFIX + "RandomAdjustSharpness(p=1)",
         wrapper(lambda T: T.RandomAdjustSharpness(2.0, p=1.0))),
    ]

    # --- resampling on a grid -------------------------------------------------
    #
    # **The first thing in this file that reads the input between its pixels.** Every
    # transform before it moved, copied or rewrote whole pixels; these ask for a
    # position that is not a pixel and interpolate. So the cases go after the
    # convention rather than the result: `align_corners=False`, the half-pixel offset
    # in the grid, and half-to-even rounding in the nearest mode. Getting any of the
    # three wrong shifts the whole picture by half a pixel, which reads as softness
    # rather than as an error.

    def geom(call, on_bytes=False):
        def run(L):
            T = _vision(L)
            F = T.functional
            src = img_u8 if on_bytes else img_f
            if _is_real_torch(L):
                from torchvision.transforms import InterpolationMode
                x = L.tensor(np.ascontiguousarray(src.transpose(2, 0, 1)))
                out = call(F, x, {"nearest": InterpolationMode.NEAREST,
                                  "bilinear": InterpolationMode.BILINEAR})
                return L.tensor(np.ascontiguousarray(
                    np.asarray(out.detach().numpy(), dtype=np.float32)))
            out = call(F, src, {"nearest": "nearest", "bilinear": "bilinear"})
            return L.tensor(np.ascontiguousarray(
                np.asarray(out, dtype=np.float32).transpose(2, 0, 1)))
        return run

    cases += [
        (VISION_PREFIX + "F.rotate(bilinear)",
         geom(lambda F, x, M: F.rotate(x, 30, M["bilinear"]))),
        # **Nearest is not the easy one.** A quarter turn puts every sampled position
        # exactly halfway between two pixels, and half-to-even is what torch does
        # there — `floor(x + 0.5)` disagrees on all of them at once.
        (VISION_PREFIX + "F.rotate(nearest)",
         geom(lambda F, x, M: F.rotate(x, 90, M["nearest"]))),
        (VISION_PREFIX + "F.rotate(a straight angle)",
         geom(lambda F, x, M: F.rotate(x, 180, M["bilinear"]))),
        # `expand` grows the output to hold the corners. The 1e-4 truncation inside the
        # size computation is what stops a corner at 1e-15 above an integer from
        # ceiling to a whole extra pixel, so a quarter turn is asked as well as a
        # slanted one.
        (VISION_PREFIX + "F.rotate(expand)",
         geom(lambda F, x, M: F.rotate(x, 30, M["bilinear"], expand=True))),
        (VISION_PREFIX + "F.rotate(expand, quarter turn)",
         geom(lambda F, x, M: F.rotate(x, 90, M["bilinear"], expand=True))),
        # **The fill is sampled, not decided.** A mask of ones goes through the same
        # grid, so a bilinear edge pixel is part picture and part fill in the
        # proportion the interpolation used — deciding it from the coordinates gives a
        # hard edge up to a pixel out.
        (VISION_PREFIX + "F.rotate(filled)",
         geom(lambda F, x, M: F.rotate(x, 30, M["bilinear"], fill=[0.5, 0.25, 0.75]))),
        (VISION_PREFIX + "F.rotate(filled, nearest)",
         geom(lambda F, x, M: F.rotate(x, 30, M["nearest"], fill=[0.5, 0.25, 0.75]))),
        # An explicit centre arrives as an **offset from the middle**, not as a pixel
        # position. Passing the middle itself shifts the picture by half its own size.
        (VISION_PREFIX + "F.rotate(off centre)",
         geom(lambda F, x, M: F.rotate(x, 30, M["bilinear"], center=[1, 2]))),
        (VISION_PREFIX + "F.rotate(uint8)",
         geom(lambda F, x, M: F.rotate(x, 30, M["bilinear"]), on_bytes=True)),
        # `affine` one part at a time, then all four together — the four compose into
        # one matrix, and a sign error in any one of them survives the other three.
        (VISION_PREFIX + "F.affine(turned)",
         geom(lambda F, x, M: F.affine(x, 30, [0, 0], 1.0, [0, 0], M["bilinear"]))),
        (VISION_PREFIX + "F.affine(shifted)",
         geom(lambda F, x, M: F.affine(x, 0, [1, 2], 1.0, [0, 0], M["bilinear"]))),
        (VISION_PREFIX + "F.affine(scaled)",
         geom(lambda F, x, M: F.affine(x, 0, [0, 0], 1.5, [0, 0], M["bilinear"]))),
        (VISION_PREFIX + "F.affine(sheared)",
         geom(lambda F, x, M: F.affine(x, 0, [0, 0], 1.0, [10, 20], M["bilinear"]))),
        (VISION_PREFIX + "F.affine(all four)",
         geom(lambda F, x, M: F.affine(x, 15, [1, -1], 0.8, [5, -5], M["bilinear"]))),
        (VISION_PREFIX + "F.affine(all four, nearest)",
         geom(lambda F, x, M: F.affine(x, 15, [1, -1], 0.8, [5, -5], M["nearest"]))),
        (VISION_PREFIX + "F.affine(uint8)",
         geom(lambda F, x, M: F.affine(x, 15, [1, -1], 0.8, [5, -5], M["bilinear"]),
              on_bytes=True)),
    ]

    def turned(build):
        """A drawn transform with its range pinned to one value — then the draw has one
        answer and the frozen picture is about the resampling rather than the dice."""
        def run(L):
            T = _vision(L)
            if _is_real_torch(L):
                from torchvision.transforms import InterpolationMode
                x = L.tensor(np.ascontiguousarray(img_f.transpose(2, 0, 1)))
                out = build(T, InterpolationMode.BILINEAR)(x)
                return L.tensor(np.ascontiguousarray(
                    np.asarray(out.detach().numpy(), dtype=np.float32)))
            out = build(T, "bilinear")(img_f)
            return L.tensor(np.ascontiguousarray(
                np.asarray(out, dtype=np.float32).transpose(2, 0, 1)))
        return run

    cases += [
        (VISION_PREFIX + "RandomRotation(pinned)",
         turned(lambda T, m: T.RandomRotation((30, 30), interpolation=m))),
        (VISION_PREFIX + "RandomRotation(pinned, expand)",
         turned(lambda T, m: T.RandomRotation((30, 30), interpolation=m, expand=True))),
        (VISION_PREFIX + "RandomAffine(pinned)",
         turned(lambda T, m: T.RandomAffine((20, 20), interpolation=m))),
        # **The fill goes per channel before the call**, which happens in the class
        # rather than in `affine` — a single number handed through undone is a
        # different picture on three channels.
        (VISION_PREFIX + "RandomAffine(pinned, filled)",
         turned(lambda T, m: T.RandomAffine((20, 20), interpolation=m, fill=0.5))),
    ]

    # --- the other three that resample ----------------------------------------
    #
    # `perspective` and `elastic_transform` are the grid sampler again with a different
    # grid; `gaussian_blur` is not a resampling and is here because `ElasticTransform`
    # is built out of it.

    def warp(call, on_bytes=False):
        """**`L` builds the displacement, not `torch`.** This file does not import
        torch — its first line says so, because golden stage two runs in a browser
        where there is none — and `elastic_transform` is the first case here that needs
        a tensor as an *argument* rather than as the picture. `L.tensor` is the way to
        ask for one without naming the library."""
        def run(L):
            T = _vision(L)
            F = T.functional
            src = img_u8 if on_bytes else img_f
            if _is_real_torch(L):
                from torchvision.transforms import InterpolationMode
                x = L.tensor(np.ascontiguousarray(src.transpose(2, 0, 1)))
                out = call(F, x, {"nearest": InterpolationMode.NEAREST,
                                  "bilinear": InterpolationMode.BILINEAR}, L)
                return L.tensor(np.ascontiguousarray(
                    np.asarray(out.detach().numpy(), dtype=np.float32)))
            out = call(F, src, {"nearest": "nearest", "bilinear": "bilinear"}, None)
            return L.tensor(np.ascontiguousarray(
                np.asarray(out, dtype=np.float32).transpose(2, 0, 1)))
        return run

    # Four corners moved, and a set that moves none — **the identity is the case that
    # catches a sign or a transpose**, since a projective map that is wrong in either
    # still looks like a plausible tilt on a distorted one.
    _corners = [[0, 0], [3, 0], [3, 4], [0, 4]]
    _tilted = [[1, 1], [3, 0], [2, 4], [0, 3]]
    # A displacement small enough to stay inside, so the case is about the warp rather
    # than about the fill. It is spelled out rather than drawn: **the golden cannot
    # compare a draw**, and `elastic_transform` takes the field as an argument exactly
    # so that it can be given.
    _shift = (np.arange(5 * 4 * 2, dtype=np.float32).reshape(5, 4, 2) % 7 - 3) * 0.02

    cases += [
        (VISION_PREFIX + "F.gaussian_blur(odd square)",
         warp(lambda F, x, M, t: F.gaussian_blur(x, [3, 3], [1.0, 1.0]))),
        # **A kernel that is not square, with different sigmas.** The 2-D kernel is an
        # outer product of the y kernel with the x one, and a transpose there is
        # invisible while both are the same size.
        (VISION_PREFIX + "F.gaussian_blur(oblong)",
         warp(lambda F, x, M, t: F.gaussian_blur(x, [3, 5], [0.5, 2.0]))),
        (VISION_PREFIX + "F.gaussian_blur(default sigma)",
         warp(lambda F, x, M, t: F.gaussian_blur(x, [5, 5], None))),
        (VISION_PREFIX + "F.gaussian_blur(uint8)",
         warp(lambda F, x, M, t: F.gaussian_blur(x, [3, 3], [1.0, 1.0]), on_bytes=True)),
        (VISION_PREFIX + "F.perspective(tilted)",
         warp(lambda F, x, M, t: F.perspective(x, _corners, _tilted, M["bilinear"]))),
        (VISION_PREFIX + "F.perspective(tilted, nearest)",
         warp(lambda F, x, M, t: F.perspective(x, _corners, _tilted, M["nearest"]))),
        (VISION_PREFIX + "F.perspective(unmoved)",
         warp(lambda F, x, M, t: F.perspective(x, _corners, _corners, M["bilinear"]))),
        (VISION_PREFIX + "F.perspective(filled)",
         warp(lambda F, x, M, t: F.perspective(x, _corners, _tilted, M["bilinear"],
                                               fill=[0.5, 0.2, 0.1]))),
        (VISION_PREFIX + "F.perspective(uint8)",
         warp(lambda F, x, M, t: F.perspective(x, _corners, _tilted, M["bilinear"]),
              on_bytes=True)),
        # The displacement arrives shaped `1 x H x W x 2` for torch and `H x W x 2`
        # here — the same numbers in each side's own convention, which is the rule the
        # whole vision block follows.
        (VISION_PREFIX + "F.elastic_transform",
         warp(lambda F, x, M, t: F.elastic_transform(
             x, t.tensor(np.ascontiguousarray(_shift[None])) if t else _shift, M["bilinear"]))),
        (VISION_PREFIX + "F.elastic_transform(nearest)",
         warp(lambda F, x, M, t: F.elastic_transform(
             x, t.tensor(np.ascontiguousarray(_shift[None])) if t else _shift, M["nearest"]))),
        (VISION_PREFIX + "F.elastic_transform(uint8)",
         warp(lambda F, x, M, t: F.elastic_transform(
             x, t.tensor(np.ascontiguousarray(_shift[None])) if t else _shift, M["bilinear"]),
             on_bytes=True)),
    ]

    # --- the policies ---------------------------------------------------------
    #
    # **Almost nothing here can be frozen, and that is the honest shape of it.** All
    # four draw on every call — which operation, how hard, which sign, and for two of
    # them how many — so a frozen picture would be a frozen dice roll. What *can* be
    # frozen is the part that is not drawn: the learned table itself, and the one
    # configuration that applies nothing.
    #
    # The operations these pick from are all cased above, individually. That is where
    # this layer's values are actually held.

    def policy_table(name):
        """`AutoAugment`'s learned table, **as text.**

        It is twenty-five pairs of (operation, probability, strength) per dataset,
        found by a search and derivable from nothing — the kind of data that is
        transcribed wrong silently and stays wrong, because every entry is plausible.
        Comparing it as a string is the only way to ask about data that has no
        arithmetic to check it against.
        """
        def run(L):
            T = _vision(L)
            policy = T.AutoAugmentPolicy(name)
            return str(T.AutoAugment(policy).policies)
        return run

    for _policy in ("imagenet", "cifar10", "svhn"):
        cases.append((VISION_PREFIX + f"AutoAugment(the {_policy} table)",
                      policy_table(_policy)))

    cases.append((
        # **Zero operations has to be the identity**, and it is the only configuration
        # of any of the four that does not draw. A `num_ops` read as a count of
        # something else would show here and nowhere else.
        VISION_PREFIX + "RandAugment(no operations)",
        wrapper(lambda T: T.RandAugment(num_ops=0), on_bytes=True)))

    # Representation (T3). This project treats `repr` as specification too — the tutorials do
    # `print(transform)`, and if it differs there the learner learns something else.
    reprs = (
        ("ToTensor", lambda T: T.ToTensor()),
        ("Normalize", lambda T: T.Normalize(mean, std)),
        ("RandomHorizontalFlip", lambda T: T.RandomHorizontalFlip(p=0.5)),
        ("RandomCrop", lambda T: T.RandomCrop(32, padding=4)),
        # **The default form, because the padded one could not see it.** With a padding
        # given, both sides print the number and agree; left at its default ours printed
        # `padding=0` against torchvision's `padding=None`, and the only case in the
        # table passed a padding.
        ("RandomCrop(the default)", lambda T: T.RandomCrop(32)),
        ("CenterCrop", lambda T: T.CenterCrop(24)),
        ("Compose", lambda T: T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])),
        # **`Resize` was missing from this list, and not by oversight.** Its repr printed
        # two fields where torchvision prints four, so it could not be added — and the
        # table stayed green while the one transform whose repr differed was the one not
        # in it. `max_size` and `antialias` are printed now, so it can be asked.
        ("Resize", lambda T: T.Resize(4)),
        ("Resize(a pair)", lambda T: T.Resize((4, 3))),
        ("Lambda", lambda T: T.Lambda(lambda x: x)),
        ("RandomApply", lambda T: T.RandomApply([T.ToTensor()], p=0.3)),
        ("RandomChoice", lambda T: T.RandomChoice([T.ToTensor(), T.CenterCrop(2)])),
        ("RandomOrder", lambda T: T.RandomOrder([T.ToTensor(), T.CenterCrop(2)])),
        ("RandomVerticalFlip", lambda T: T.RandomVerticalFlip(p=0.5)),
        ("Pad", lambda T: T.Pad(2)),
        ("Pad(four sides)", lambda T: T.Pad((1, 2, 3, 4), fill=1, padding_mode="reflect")),
        ("Grayscale", lambda T: T.Grayscale(3)),
        ("RandomGrayscale", lambda T: T.RandomGrayscale(p=0.1)),
        ("FiveCrop", lambda T: T.FiveCrop(3)),
        ("TenCrop", lambda T: T.TenCrop((3, 2), vertical_flip=True)),
        ("RandomResizedCrop", lambda T: T.RandomResizedCrop(4)),
        ("RandomErasing", lambda T: T.RandomErasing()),
        ("ColorJitter", lambda T: T.ColorJitter(0.5, 0.3, 0.2, 0.1)),
        # The bare one, because **a jitter left at its defaults stores `None`** rather
        # than a range that does nothing, and only the default form prints that.
        ("ColorJitter(the default)", lambda T: T.ColorJitter()),
        # **Three of these print without a space after the comma** — torchvision's own
        # spelling, and the kind of thing only a frozen string notices.
        ("RandomInvert", lambda T: T.RandomInvert()),
        ("RandomPosterize", lambda T: T.RandomPosterize(4)),
        ("RandomSolarize", lambda T: T.RandomSolarize(0.5)),
        ("RandomAdjustSharpness", lambda T: T.RandomAdjustSharpness(2)),
        ("RandomAutocontrast", lambda T: T.RandomAutocontrast()),
        ("RandomEqualize", lambda T: T.RandomEqualize()),
        # **Two classes, two rules for dropping a field.** `RandomRotation` omits
        # `center` and `fill` when they are `None`; `RandomAffine` omits a field when
        # it equals its default. Both are torchvision's, and only the pair shows it.
        ("RandomRotation", lambda T: T.RandomRotation(30)),
        ("RandomRotation(expanded, off centre)",
         lambda T: T.RandomRotation((-10, 10), expand=True, center=(1, 2), fill=5)),
        ("RandomAffine", lambda T: T.RandomAffine(30)),
        ("GaussianBlur", lambda T: T.GaussianBlur(3)),
        ("GaussianBlur(oblong)", lambda T: T.GaussianBlur((3, 5), (0.2, 3.0))),
        ("RandomPerspective", lambda T: T.RandomPerspective()),
        # **The one class that prints the enum's name where every other prints its
        # value** — `interpolation=InterpolationMode.BILINEAR` rather than `bilinear`.
        ("ElasticTransform", lambda T: T.ElasticTransform()),
        # The four policies. `AutoAugment` prints its enum where the other three print
        # `InterpolationMode`, and `AugMix` prints seven fields — all four spellings
        # are torchvision's and none is derivable from the others.
        ("AutoAugment", lambda T: T.AutoAugment()),
        ("AutoAugment(svhn, filled)",
         lambda T: T.AutoAugment(T.AutoAugmentPolicy("svhn"), fill=3)),
        ("RandAugment", lambda T: T.RandAugment()),
        ("TrivialAugmentWide", lambda T: T.TrivialAugmentWide()),
        ("AugMix", lambda T: T.AugMix()),
        ("RandomAffine(everything)",
         lambda T: T.RandomAffine(0, translate=(0.1, 0.2), scale=(0.8, 1.2),
                                  shear=(5, 10))),
    )
    for name, build in reprs:
        cases.append((VISION_PREFIX + f"repr::{name}",
                      lambda L, b=build: repr(b(_vision(L)))))

    # **This one needs the library as well as its transforms** — torchvision's constructor
    # takes tensors, so the matrix has to be built in whichever library is being asked.
    def linear_repr(L):
        T = _vision(L)
        return repr(T.LinearTransformation(L.tensor(np.eye(3, dtype=np.float32)),
                                           L.tensor(np.zeros(3, dtype=np.float32))))

    cases.append((VISION_PREFIX + "repr::LinearTransformation", linear_repr))
    return cases


def _highrank_battery(ranks, inp):
    """Looks at **where it breaks down** as the rank rises.

    Rank 6 alone was written by hand at first. Then three came out at rank 5 and one at rank 6, and
    all of them after "it is fixed" had been concluded. So copying the cases out per rank means
    copying them again for the next rank — it is built to take the rank as an argument.

    What is asked: the whole value first. Reduced to a scalar, swapped positions give the same sum
    and pass. The gradient too is taken with a different weight per slot rather than `sum()` — a
    plain `sum()` makes every gradient 1, and then a 0 nailed into the wrong slot is not caught.

    If there is a rank TF.js genuinely cannot do, the answer has to be **a refusal** rather than a
    quietly wrong value, and that surfaces here too — the golden answers are frozen with real
    torch, so anything differing from torch's value is caught as parted, exception or wrong number
    alike.
    """
    cases = []
    for r in ranks:
        # Exactly one axis is 3 so that a transposed axis is caught by **shape** before value.
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
            # Both a rank-lowering and a rank-raising reshape are looked at. What was caught
            # during `_dilate` was where reshape and pad meet, so it is asked separately.
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

# **A table that asks only whether a gradient flows.**
#
# A check that compares values alone cannot see a cut graph — because the value is right. The
# sister library's `roll` and `masked_select` really were cut that quietly, with all 746 golden
# cases green. Asking both libraries the same thing and seeing whether they part is the way to
# catch it.
#
# What is held here is **what all three have to flow.** The ones that do not (`nonzero`,
# `quantile`, `argsort`, `signbit` and so on) are absent — their value depends on the shape, or
# they are staircases so torch does not flow either, or we left them out deliberately.
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
    """Whether each operation **flows a gradient.** That fact is frozen rather than the value.

    It answers with `requires_grad` alone, as a string — to keep it from overlapping the tables
    that ask about values, and so that "it said it flows and it does not" parts for exactly one
    reason.
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
            """**Two things answered together.** `requires_grad` alone is not enough — `.float()`
            said True and left `.grad` as `None`, and with only that check it would have passed.
            It goes backwards and looks at whether a gradient really appeared.
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
    """The 1-D and 3-D family. **Where the sister library had them and the core had a refusing stub.**

    The asymmetry ran straight against this project's promise — "the same code with only the
    import changed", while `nn.Conv1d` ran on the sister side and stopped with a `BorchError` on
    the core. This table stops that asymmetry opening again.
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

    # The module side. The weights go in from outside so that **both start from the same place** —
    # initialising separately shows whether the initialisation parted rather than what parted.
    def conv1d_module(L):
        m = L.nn.Conv1d(3, 4, 3, padding=1)
        # **Put in through `load_state_dict`.** torch refuses `m.weight.data[...] = ndarray`
        # (`can't assign a numpy.ndarray to a torch.FloatTensor`) — this is the only path all
        # three take, and it happens to be the surface this library has aligned.
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
        # **A place where `mode` was accepted and unused.** Asking for bilinear gave nearest —
        # not an exception, a quietly different value. The computation already ran through
        # `F.upsample_bilinear`, so it was one computation under two names with only one working.
        (NDIM_PREFIX + "nn.Upsample(겹선형)",
         lambda L: L.nn.Upsample(scale_factor=2, mode="bilinear")(L.tensor(img))),
        # **The first position is `size`** (torch has it so). Put the scale factor first and the
        # same line parts into enlarging and shrinking, with a plausible shape, caught by value alone.
        # **12 is an integer multiple that differs from the default (scale 2 → 8).** It was asked
        # as 8 at first, and with a 4×4 input `Upsample(8)` gave **the same answer** as the default
        # `scale_factor=2` — an implementation reading the first position as a scale factor passed
        # too. It has to be asked at a number where size and scale part, and **it has to be an
        # integer multiple** — asked as 6 it is 1.5×, and all three refused.
        (NDIM_PREFIX + "nn.Upsample(첫 자리는 size)",
         lambda L: L.nn.Upsample(12)(L.tensor(img))),
        # **"asked as 6 it is 1.5×, and all three refused" was the bug, not the reason.**
        # The line above avoided a fractional factor because it did not work; borch.ts
        # multiplied without flooring, so `scale_factor=1.5` on a 3-high input asked for
        # 4.5 rows — for nearest a tensor that summed to zero and did not throw, for
        # bilinear an exception two layers down. Whole factors were right, which is why
        # every case here was 2.
        # **The input needs an odd extent or the case is blind.** `nd_img` is 4×4 and
        # 4 × 1.5 = 6 exactly, so flooring and not flooring agree there — the first
        # draft of these three passed with the floor removed. Cropped to three rows,
        # 3 × 1.5 = 4.5 and they part.
        (NDIM_PREFIX + "nn.Upsample(분수 배율)",
         lambda L: L.nn.Upsample(scale_factor=1.5)(L.tensor(img[:, :, :3, :]))),
        (NDIM_PREFIX + "nn.Upsample(분수 배율, 겹선형)",
         lambda L: L.nn.Upsample(scale_factor=1.5,
                                 mode="bilinear")(L.tensor(img[:, :, :3, :]))),
        # **The flag only shows here.** With the output size floored, torch either samples
        # at the factor it was given or derives the scale back from that size, and those
        # two are the same number whenever the factor divides exactly. So a case at 2 is
        # blind to it and this one is the whole of the difference.
        (NDIM_PREFIX + "nn.Upsample(분수 배율, recompute)",
         lambda L: L.nn.Upsample(scale_factor=1.5, mode="bilinear",
                                 recompute_scale_factor=True)(
             L.tensor(img[:, :, :3, :]))),
        (NDIM_PREFIX + "nn.AvgPool2d", lambda L: L.nn.AvgPool2d(2)(L.tensor(img))),
        (NDIM_PREFIX + "nn.AvgPool2d(보폭)",
         lambda L: L.nn.AvgPool2d(2, 1)(L.tensor(img))),
        (NDIM_PREFIX + "nn.LPPool1d",
         lambda L: L.nn.LPPool1d(2, 2)(L.tensor(seq))),
        (NDIM_PREFIX + "nn.Unflatten",
         lambda L: L.nn.Unflatten(1, (1, 3))(L.tensor(img.reshape(2, 3, 16)))),
    ]

    # The **function form** of things that existed as methods only. torch gives both and the core
    # had one side, and it surfaced while matching against the sister library.
    mask = np.array([[True, False, True, False]] * 2)
    flat = np.arange(8, dtype=np.float32).reshape(2, 4)
    # **`matmul` batches and broadcasts; `mm` does not.** Every matmul case in
    # this table used to be 2-D, where the two are the same function, so the
    # TypeScript side answered all of them with `mm` and nothing noticed that
    # it had no batched path at all. These ask the part that differs.
    #
    # The 1-D cases are here because they do **not** follow from the others:
    # a 1 is prepended (or appended) to make the operand a matrix and then
    # **taken back out of the result**, so `[3] @ [5,3,4]` is `[5,4]` and not
    # `[5,1,4]`. That is the rule most likely to be reimplemented wrongly.
    m3a = np.arange(30, dtype=np.float32).reshape(5, 2, 3) / 30.0
    m3b = np.arange(60, dtype=np.float32).reshape(5, 3, 4) / 60.0
    m2 = np.arange(12, dtype=np.float32).reshape(3, 4) / 12.0
    v3 = np.arange(3, dtype=np.float32) / 3.0
    m4a = np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3) / 12.0
    m4b = np.arange(48, dtype=np.float32).reshape(1, 4, 3, 4) / 48.0
    cases += [
        # **`transpose` takes two dimensions and works at any rank.** It used
        # to take none here and refuse anything but a matrix, and every case in
        # this table was 2-D, so nothing asked. The negative-index pair is the
        # one models actually write — attention swaps the last two axes.
        ("transpose::3d", lambda L: L.transpose(L.tensor(m3a), 1, 2)),
        ("transpose::3d_negative", lambda L: L.transpose(L.tensor(m3a), -2, -1)),
        ("transpose::4d_outer", lambda L: L.transpose(L.tensor(m4a), 0, 2)),
        # The alias torch documents for it. Same answer or the two have drifted.
        ("transpose::swapaxes_agrees", lambda L: L.swapaxes(L.tensor(m3a), -2, -1)),
        ("matmul::3d@3d", lambda L: L.matmul(L.tensor(m3a), L.tensor(m3b))),
        # The 2-D side is shared by every batch element rather than zipped.
        ("matmul::3d@2d", lambda L: L.matmul(L.tensor(m3a), L.tensor(m2))),
        ("matmul::2d@3d", lambda L: L.matmul(L.tensor(m2.T.copy()), L.tensor(m3b))),
        ("matmul::1d@3d", lambda L: L.matmul(L.tensor(v3), L.tensor(m3b))),
        ("matmul::3d@1d",
         lambda L: L.matmul(L.tensor(m3b), L.tensor(np.arange(4, dtype=np.float32) / 4.0))),
        # Batch axes broadcast against each other: (2,1) with (1,4) is (2,4).
        ("matmul::4d_broadcast", lambda L: L.matmul(L.tensor(m4a), L.tensor(m4b))),
        # `F.linear` on a 3-D input — the case bimm's ViT had to work around.
        ("matmul::linear_3d",
         lambda L: L.nn.functional.linear(L.tensor(m3a), L.tensor(m2.T.copy()))),
        (NDIM_PREFIX + "torch.matmul",
         lambda L: L.matmul(L.tensor(flat), L.tensor(flat.T.copy()))),
        # **The shape comes as a single tuple.** torch refuses `torch.reshape(x, 4, 2)` — the
        # method (`x.reshape(4, 2)`) and the function have different signatures.
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


def _eig_holds(L, mat):
    """Is `A·V = V·diag(λ)`? **A question that does not lean on the sign**, so all three answer together."""
    t = L.tensor(mat)
    w, v = L.linalg.eig(t)
    left = L.matmul(t.cfloat(), v)
    right = L.matmul(v, L.diag(w))
    return f"{float((left - right).abs().max()):.4f}"


def linalg_cases(inp=None):
    """Linear-algebra decompositions. **Both sides have them** — the sister library reads back to
    the CPU and computes with numpy, warning once on the first call that it is slow.

    Gradients exist only where there is a closed form (`det`, `logdet`, `inverse`, `solve`,
    `cholesky`, `matrix_power`). `qr`, `svd`, `pinverse` and `lstsq` give values only — torch
    differentiates them and we do not. The derivation is awkward and goes quietly wrong when it is
    wrong, so their absence is kept loud.
    """
    mat = np.array([[4., 1.], [2., 3.]], dtype=np.float32)
    sym = np.array([[4., 1.], [1., 3.]], dtype=np.float32)      # symmetric positive definite
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
        # torch's `linalg.lstsq` returns the residuals and rank alongside the solution. It has to
        # be asked through `.solution` for the three to compare the same thing — ours was given
        # that name too.
        (LINALG_PREFIX + "lstsq",
         lambda L: L.linalg.lstsq(L.tensor(mat), L.tensor(vec)).solution),
        (LINALG_PREFIX + "eigh/고윳값", lambda L: L.linalg.eigh(L.tensor(sym))[0]),
        # It has to be reachable under the `torch.linalg` name too — the tutorials use that side.
        (LINALG_PREFIX + "linalg.det", lambda L: L.linalg.det(L.tensor(mat))),
        (LINALG_PREFIX + "linalg.inv", lambda L: L.linalg.inv(L.tensor(mat))),
        (LINALG_PREFIX + "qr/R", lambda L: L.linalg.qr(L.tensor(mat))[1]),
    ]

    # **The sign convention differs per implementation.** QR's Q and SVD's U and Vh are the same
    # decomposition with a column's sign flipped, so they are asked by absolute value — demanding
    # the sign match would measure the convention difference between numpy and LAPACK rather than
    # our implementation.
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


# Inputs for the batched and rectangular cases. **Written by hand** — a random draw makes neither
# a singular matrix nor a row swap, and those two places are exactly what is being measured here.
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
# The first element is smaller than the second row's → partial pivoting swaps a row. Asked only
# with matrices that do not swap, the pivots are the identity and **counting from 1 cannot be told
# apart from counting from 0.**
_LA_PIVOT = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
_LA_SINGULAR = np.array([[1., 2.], [2., 4.]], dtype=np.float32)


def linalg_struct_cases(inp=None):
    """`linalg`'s **structure** — batching, rectangles, names, `_ex` and LU.

    The `linalg_cases` above ask about one 2-D square only. And torch's `linalg` is **batched
    throughout**: `det((3,2,2))` gives `(3,)`, and so do `inv`, `solve`, `cholesky`, `slogdet` and
    `matrix_rank`. Golden cases that ask about one sheet cannot see the batching.

    ## It has to be asked by name too

    torch returns these results as named tuples — `slogdet(A).logabsdet`, `qr(A).Q`,
    `lu_factor(A).pivots`, `inv_ex(A).info`. Matched by position alone, the values are right and
    textbook code stops at the attribute access. `lstsq` already met the same place through
    `.solution`.

    ## `_ex` is the side that does not throw

    `inv` throws a `LinAlgError` on a singular matrix and `inv_ex` returns quietly with a non-zero
    number in `info`. **Both have to exist** — without the throwing side a singular matrix leaks
    out as NaN, and without the non-throwing side one singular sheet kills a whole batch.

    ## The pivots count from 1

    `lu_factor`'s `pivots` follow the LAPACK convention and **start at 1.** On a matrix with no
    swaps it is `[1, 2]` and not `[0, 1]`. Counted from 0, `lu_solve` quietly gives a different
    answer — which is why a matrix that really does swap a row is asked too.
    """
    bat, sym, vec = _LA_BATCH, _LA_BATCH_SYM, _LA_BATCH_VEC
    rhs, rect, sym3 = _LA_BATCH_RHS, _LA_RECT, _LA_SYM3

    cases = [
        # ── batched ──
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
        # 3×3 is asked too — a 2×2 has only one Jacobi rotation and never goes through the
        # sweep iteration.
        (LINALG_PREFIX + "3x3::eigh/값", lambda L: L.linalg.eigh(L.tensor(sym3))[0]),
        (LINALG_PREFIX + "3x3::svd/S", lambda L: L.linalg.svd(L.tensor(sym3))[1]),
        (LINALG_PREFIX + "3x3::det", lambda L: L.linalg.det(L.tensor(sym3))),
        (LINALG_PREFIX + "3x3::inv", lambda L: L.linalg.inv(L.tensor(sym3))),

        # ── rectangular ──
        # Square-only leaves `qr`, `svd` and `pinv` unable to take half the places they are used —
        # and least squares is exactly that half.
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

    # ── asking by name ──
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

    # ── `_ex` — info instead of a throw ──
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
        """**Does `except torch.linalg.LinAlgError` catch it?**

        The exception's class name may differ between the two (real torch's is `_LinAlgError`).
        Demanding the names match would measure somebody else's business, so what is asked is what
        a user actually writes — whether it is caught under that name.
        """
        try:
            L.linalg.inv(L.tensor(_LA_SINGULAR))
        except L.linalg.LinAlgError:
            return "LinAlgError 로 잡힌다"
        except Exception as exc:                                    # noqa: BLE001
            return f"다른 것이 났다: {type(exc).__name__}"
        return "예외가 안 났다"

    cases.append((LINALG_PREFIX + "ex::inv(특이)가 던지는 것", catches))

    # ── LU — bringing out what was already being computed inside ──
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

    # ── `_ex`, LDL and reflectors ──
    #
    # `lu_factor_ex` **reports through `info`** instead of throwing. 0 means it went well and `k`
    # means the `k`-th pivot is 0 (counting from 1). It has to be asked with a singular matrix too
    # for that number to show.
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

    # LDL means something only on a **symmetric** matrix. `lin4` is built symmetric.
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

    # **QR in reflector form.** `geqrf` holds it and `householder_product` unfolds it into `Q`.
    #
    # **It has to be asked on a square too.** LAPACK does not reflect when everything below the
    # diagonal is 0 (`tau = 0`) and leaves the values as they are, and a square's last column is
    # always that place. Asked on rectangles only, that column never comes up, and flipping its
    # sign there is not caught — which really happened.
    for tag, arr in (("정사각", lin4), ("직사각", rect53)):
        cases.append((LINALG_PREFIX + f"ex::geqrf/{tag}/a",
                      lambda L, a=arr: L.geqrf(L.tensor(a))[0]))
        cases.append((LINALG_PREFIX + f"ex::geqrf/{tag}/tau",
                      lambda L, a=arr: L.geqrf(L.tensor(a))[1]))

        def product(L, a=arr):
            got = L.geqrf(L.tensor(a))
            return L.linalg.householder_product(got[0], got[1])

        cases.append((LINALG_PREFIX + f"ex::householder_product/{tag}", product))

    # ── the batched gradients ──
    # **This is where the value is right and the gradient is not.** A backward formula written with
    # `.T` is right in 2-D and, batched, reverses the axes outright and is quietly wrong.
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
    """`linalg`'s **composition layer** — the places that name what already exists, and the norms
    with branches.

    Where the two above were structure, this is surface. Most of it composes what already exists,
    so the only one needing new computation is `matrix_exp`. There are still three places where
    **the composition is not obvious.**

    ## A norm's branch changes the value

    `matrix_norm` defaults to Frobenius, and `ord=2` is the largest singular value, `nuc` their
    sum, `1` the largest column absolute-value sum, and `inf` the row side — **all different
    numbers.** That is why the golden case asks with a rank-2 matrix. Give it rank 1 and the
    Frobenius, 2 and nuclear norms happen to coincide, and the three cannot be told apart.

    ## `linalg.diagonal`'s **default axes differ** from `torch.diagonal`'s

    The `linalg` side looks at the last two axes (`dim1=-2, dim2=-1`) and the `torch` side at the
    first two. Given a 3-D, a `(2,3,4)` parts into `(2,3)` and `(4,2)` — the names are similar
    enough to read as the same thing, and even the shape differs.

    ## `eigh` **reads the lower triangle only**

    `[[4,99],[1,3]]` and `[[4,1],[1,3]]` give the same answer (measured). Give it something
    non-symmetric and it ignores the upper part and mirrors the lower. An implementation reading
    the whole matrix parts here, and **it stays hidden as long as symmetric input is given** — so
    a deliberately non-symmetric one is asked.
    """
    mat, rect, sym3 = _LA_MAT, _LA_RECT, _LA_SYM3
    vec3, upper, cube = _LA_VEC3, _LA_UPPER, _LA_CUBE
    sym = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    skew = np.array([[4., 99.], [1., 3.]], dtype=np.float32)   # the lower triangle only should be read
    # For `eig`. The rotation has **no real eigenvalues** (±i), and the general one has three real
    # ones while not being symmetric.
    rot = np.array([[0., -1.], [1., 0.]], dtype=np.float32)
    gen = np.array([[4., 1., 2.], [0., 3., -1.], [1., 0., 2.]], dtype=np.float32)

    cases = [
        # ── naming what already exists ──
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
        # **Does it read the lower triangle only?** Putting 99 in the upper part must not change the answer.
        (LINALG_PREFIX + "name2::eigvalsh(아래삼각만)",
         lambda L: L.linalg.eigvalsh(L.tensor(skew))),
        (LINALG_PREFIX + "name2::eigh(아래삼각만)/값",
         lambda L: L.linalg.eigh(L.tensor(skew))[0]),

        # ── `eig` — it takes non-symmetric matrices too ──
        #
        # It sits beside `eigh` and is a different function. That one takes symmetric matrices only,
        # reads one triangle, and gives a real answer; this one takes any square matrix and **its
        # answer is always complex** — because some, like a rotation matrix, have no real
        # eigenvalues at all.
        #
        # **Only what does not lean on the order is frozen.** LAPACK does not fix the eigenvalue
        # order and the browser's numpy is a different LAPACK. And torch **cannot sort complex
        # numbers** (measured), so sorting is no escape either — it is folded through symmetric
        # functions.
        (LINALG_PREFIX + "eig::eigvals(회전)/크기",
         lambda L: L.linalg.eigvals(L.tensor(rot)).abs().sort().values),
        (LINALG_PREFIX + "eig::eigvals(비대칭)/크기",
         lambda L: L.linalg.eigvals(L.tensor(gen)).abs().sort().values),
        (LINALG_PREFIX + "eig::eigvals(대칭이어도 복소수형)",
         lambda L: str(L.linalg.eigvals(L.tensor(sym)).dtype)),
        # **The sum is the trace.** Independent of the order, and it asks by mathematics whether
        # the values are right — an implementation producing eigenvalues in any order passes a
        # magnitude sort and is caught here.
        (LINALG_PREFIX + "eig::eigvals(비대칭)/합=대각합",
         lambda L: (f"{float(L.linalg.eigvals(L.tensor(gen)).sum().real):.4f} "
                    f"{float(L.tensor(gen).trace()):.4f}")),
        # **The eigenvectors cannot be frozen** — their sign is undetermined (torch itself gives
        # opposite signs in float32 and float64, measured). **The definition is asked** instead:
        # are `A·V` and `V·diag(λ)` equal. A flipped sign flips both sides together and the answer
        # does not change.
        (LINALG_PREFIX + "eig::eig(정의를 지키나)",
         lambda L: _eig_holds(L, gen)),

        # ── where the axes part ──
        (LINALG_PREFIX + "name2::linalg.diagonal",
         lambda L: L.linalg.diagonal(L.tensor(cube))),
        (LINALG_PREFIX + "name2::torch.diagonal(다른 축)",
         lambda L: L.diagonal(L.tensor(cube))),
        (LINALG_PREFIX + "name2::linalg.diagonal(offset)",
         lambda L: L.linalg.diagonal(L.tensor(mat), offset=1)),

        # ── the norm's branches ──
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
        # ── the ones whose composition is not one line ──
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
        # **It does not look at the diagonal.** Break that and it is a branch whose value quietly differs.
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

        # ── the one with no closed form ──
        # **Scaling and squaring are needed.** Taylor alone does not converge on a large matrix —
        # `A*5`'s answer is 4.8e+10, so the growing terms overflow first.
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
    """**The decompositions' gradients.** torch differentiates all of these and we produced values only.

    There was a reason they went so long unwritten — the derivation is awkward and goes quietly
    wrong when it is wrong. The kind where the value is right and only the training parts subtly.
    That is what the golden answers are for, and here they do exactly that job: **matched slot by
    slot against the numbers real torch produced.**

    ## Two safe ones and three subtle ones

    The singular values and the eigenvalues end at `U diag(ḡ) Vᵀ` and `V diag(ḡ) Vᵀ` — no
    degeneracy problem and a one-line derivation.

    The other three are different. **The eigenvectors** carry a `1/(λᵢ-λⱼ)` and blow up when
    eigenvalues coincide (torch blows up alongside — not an imitation but the same limit). **QR**
    has a convention that parts easily where the lower triangle is mirrored, and **the
    pseudo-inverse** has three terms, so leaving one out is right on a square and wrong only on a
    rectangle. So all three are asked **on rectangles too.**

    ## The matrix exponential differentiates through itself

    `e^A`'s Fréchet derivative comes out of a single block matrix — the upper right of
    `expm([[Aᵀ, Ḡ],[0, Aᵀ]])` is the answer. An identity rather than an approximation, so reusing
    the series from the forward pass brings the gradient with it.
    """
    mat, rect, sym3 = _LA_MAT, _LA_RECT, _LA_SYM3
    sym = np.array([[4., 1.], [1., 3.]], dtype=np.float32)

    grads = (
        # The two safe ones.
        ("svdvals", lambda L, x: L.linalg.svdvals(x), mat),
        ("svd/S", lambda L, x: L.linalg.svd(x)[1], mat),
        ("svd/S(직사각)",
         lambda L, x: L.linalg.svd(x, full_matrices=False)[1], rect),
        ("eigvalsh", lambda L, x: L.linalg.eigvalsh(x), sym),
        ("eigh/값", lambda L, x: L.linalg.eigh(x)[0], sym),
        ("eigh/값(3x3)", lambda L, x: L.linalg.eigh(x)[0], sym3),
        # The three subtle ones.
        #
        # **The eigenvectors are asked squared.** Flipping a column's sign is the same
        # eigendecomposition, so which one is chosen is the implementation's to decide, and Jacobi
        # rotations and LAPACK really do choose differently. The value cases ask by absolute value
        # and were covering that difference, and the gradient is sign-sensitive so it came straight
        # out (on a 2×2, exactly the sign was reversed).
        #
        # `V∘V` is unchanged by a flipped sign. So this loss's gradient is **determined regardless
        # of the sign convention** — the question was changed into one both sides can answer, not
        # dodged for being hard. The same place as asking dropout by property.
        ("eigh/벡터²", lambda L, x: L.linalg.eigh(x)[1] ** 2, sym),
        ("eigh/벡터²(3x3)", lambda L, x: L.linalg.eigh(x)[1] ** 2, sym3),
        ("qr/R", lambda L, x: L.linalg.qr(x)[1], mat),
        ("qr/Q", lambda L, x: L.linalg.qr(x)[0], mat),
        ("qr/R(직사각)", lambda L, x: L.linalg.qr(x)[1], rect),
        ("qr/Q(직사각)", lambda L, x: L.linalg.qr(x)[0], rect),
        ("pinv", lambda L, x: L.linalg.pinv(x), mat),
        # **The rectangle is the real test.** On a square the missing term becomes 0 and does not show.
        ("pinv(직사각)", lambda L, x: L.linalg.pinv(x), rect),
        ("pinv(3x3)", lambda L, x: L.linalg.pinv(x), sym3),
        # The one that differentiates through itself.
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

    # One place asking value and gradient **together.** Using a decomposition and putting a loss
    # over it is what real code does, and asked in pieces there is no seeing whether they join up.
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
    """In-place operations. **Where the two implementations part deliberately.**

    Only one thing parts — propagation through a view. torch and the core share storage, so
    `b = a.view(2,2); b.add_(10)` changes `a` too, and the browser side does not share out GPU
    buffers as views and cannot, so it refuses. (The conclusion was the same in the TF.js edition
    for a different reason — there the tensors were immutable.) The rest, editing itself, is the
    same on both sides.

    That divergence is pinned here. If the browser side quietly starts returning a value one day
    (right or wrong), the build breaks.
    """
    plain = np.array([1., 4., 9., 2.], dtype=np.float32)
    small = np.array([0.5, 0.8, 0.3, 0.9], dtype=np.float32)   # for the ones with a narrow domain

    def run(name, call, arr):
        def go(L, f=call, a=arr):
            x = L.tensor(a.copy())
            f(x)
            return x               # **looks at the original, not at what came back**
        return (INPLACE_PREFIX + name, go)

    cases = [
        run("add_", lambda x: x.add_(1), plain),
        run("add_(alpha)", lambda x: x.add_(1, alpha=2), plain),
        run("sub_", lambda x: x.sub_(1), plain),
        run("mul_", lambda x: x.mul_(2), plain),
        run("div_", lambda x: x.div_(2), plain),
        run("pow_", lambda x: x.pow_(2), plain),
        # `neg_` takes no argument and is built by the `_INPLACE_UNARY` loop below. It was here a
        # second time, and having the same name, the loop's was covering this one.
        run("zero_", lambda x: x.zero_(), plain),
        run("fill_", lambda x: x.fill_(7), plain),
        run("clamp_", lambda x: x.clamp_(2, 5), plain),
        run("clip_", lambda x: x.clip_(2, 5), plain),
        # **Chaining is the real test.** What comes back has to be itself for a chain to work.
        run("이어 부르기", lambda x: x.mul_(2).add_(1).clamp_(0, 10), plain),
    ]
    for name in _INPLACE_UNARY:
        arr = small if name in ("asin_", "acos_", "atan_", "log_", "log2_", "log10_",
                                "sqrt_", "rsqrt_", "log1p_") else plain
        cases.append(run(name, lambda x, n=name: getattr(x, n)(), arr))

    # View propagation — **the browser side alone refuses.**
    def view_propagates(L):
        a = L.arange(4).float()
        a.view(2, 2).add_(10)
        return a

    cases.append((INPLACE_PREFIX + "뷰 전파=브라우저는거절",
                  _as_expected(view_propagates)))

    # **A second divergence from the same root.** With no views there is nowhere to become
    # non-contiguous, so `is_contiguous()` is always true in the browser. The core is false as
    # torch is, because numpy gives a transpose as a view.
    #
    # It is **a different value** rather than a refusal, so `_as_expected` cannot hold it — which
    # side it is, is answered as the value. The same shape as `from_numpy`'s sharing.
    def transposed_is_not_contiguous(L):
        got = L.tensor(plain.reshape(2, 2).copy()).t().is_contiguous()
        views = not hasattr(L, "backend")       # the browser side alone has no views
        if got == (not views):
            return "기대대로"
        return "뜻밖에 연속" if got else "뜻밖에 비연속"

    cases.append((INPLACE_PREFIX + "전치는 비연속=브라우저는뷰가없다",
                  transposed_is_not_contiguous))

    # With gradients on at the leaf, **both sides** refuse.
    def leaf_refuses(L):
        x = L.tensor(plain, requires_grad=True)
        try:
            x.add_(1)
        except Exception:                                       # noqa: BLE001
            return "기대대로 거절"
        return "뜻밖의 성공"

    cases.append((INPLACE_PREFIX + "잎 제자리 수정=거절", leaf_refuses))

    # Inside `no_grad` a leaf can be edited — an optimizer really does that.
    def under_no_grad(L):
        # **`.copy()` was missing here.** The core's `tensor()` shared the array at the time, so
        # this one line raised `plain` by 1 and the cases after it were wrong on the core alone.
        # `tensor()` copies now, and **the case does its part too** — a case that edits a shared
        # input in place comes to depend on the order.
        x = L.tensor(plain.copy(), requires_grad=True)
        with L.no_grad():
            x.add_(1)
        return x

    cases.append((INPLACE_PREFIX + "no_grad 안에서는 된다", under_no_grad))

    # ── the forty-one with a counterpart and no underscore edition ──
    #
    # `x.add_(1)` is **the training loop's idiom** — a textbook that does not write
    # `p.data.add_(-lr * g)` is rare. And forty-one were missing. The counterparts (`x.add`) were
    # all there and the chaining idiom was there, so what was missing was **one joining line** each.
    #
    # **Two of them must not be made from the counterpart.** An underscore does not make it the
    # same operation — `bernoulli_(p)` ignores its own value and fills with `p`, unlike
    # `bernoulli()` which reads its own value as the probability, and `float_power_`'s result is
    # double precision so torch refuses it too. All forty-one were compared against torch and only
    # those two parted.
    ints = np.array([6, -4, 3, 9])
    flags = np.array([True, False, True, False])
    other = np.array([True, True, False, False])
    twos = np.array([2., 2., 2., 2.], dtype=np.float32)
    pos = np.array([2., 3., 4., 5.], dtype=np.float32)      # for the ones whose domain is positive
    grid = np.arange(6, dtype=np.float32).reshape(2, 3)
    at = np.array([0, 2], dtype=np.int64)                   # positions, for the index family
    feed = np.array([10., 20.], dtype=np.float32)
    mask4 = np.array([True, False, True, False])

    derived = (
        # **Ten in-place forms whose partner was already on both sides.** Every one
        # mutates, and not one was held by a case — I added them to borch.ts and the
        # only thing that would have noticed a wrong write is this table.
        ("index_add_", lambda L, x: x.index_add_(0, L.tensor(at), L.tensor(feed)), plain),
        ("index_copy_", lambda L, x: x.index_copy_(0, L.tensor(at), L.tensor(feed)), plain),
        ("index_fill_", lambda L, x: x.index_fill_(0, L.tensor(at), 5.0), plain),
        ("index_reduce_",
         lambda L, x: x.index_reduce_(0, L.tensor(at), L.tensor(feed), "amax"), plain),
        ("masked_fill_", lambda L, x: x.masked_fill_(L.tensor(mask4), 9.0), plain),
        ("scatter_", lambda L, x: x.scatter_(0, L.tensor(at), L.tensor(feed)), plain),
        ("scatter_add_",
         lambda L, x: x.scatter_add_(0, L.tensor(at), L.tensor(feed)), plain),
        ("scatter_reduce_",
         lambda L, x: x.scatter_reduce_(0, L.tensor(at), L.tensor(feed), "sum"), plain),
        ("swapaxes_", lambda L, x: x.swapaxes_(0, 1), grid),
        ("conj_physical_", lambda L, x: x.conj_physical_(), plain),
        ("bitwise_and_", lambda L, x: x.bitwise_and_(3), ints),
        ("bitwise_or_", lambda L, x: x.bitwise_or_(3), ints),
        ("bitwise_xor_", lambda L, x: x.bitwise_xor_(3), ints),
        ("bitwise_not_", lambda L, x: x.bitwise_not_(), ints),
        ("bitwise_left_shift_", lambda L, x: x.bitwise_left_shift_(1), ints),
        ("bitwise_right_shift_", lambda L, x: x.bitwise_right_shift_(1), ints),
        ("logical_and_", lambda L, x: x.logical_and_(L.tensor(other)), flags),
        ("logical_or_", lambda L, x: x.logical_or_(L.tensor(other)), flags),
        ("logical_xor_", lambda L, x: x.logical_xor_(L.tensor(other)), flags),
        ("logical_not_", lambda L, x: x.logical_not_(), flags),
        ("clamp_max_", lambda L, x: x.clamp_max_(4), plain),
        ("clamp_min_", lambda L, x: x.clamp_min_(3), plain),
        ("digamma_", lambda L, x: x.digamma_(), pos),
        ("divide_", lambda L, x: x.divide_(2), plain),
        ("erfinv_", lambda L, x: x.erfinv_(), small - 0.5),
        ("floor_divide_", lambda L, x: x.floor_divide_(2), plain),
        ("fmod_", lambda L, x: x.fmod_(2), plain),
        ("gcd_", lambda L, x: x.gcd_(L.tensor(np.array([2, 2, 3, 3]))), ints),
        ("lcm_", lambda L, x: x.lcm_(L.tensor(np.array([2, 2, 3, 3]))), ints),
        ("greater_", lambda L, x: x.greater_(3), plain),
        ("greater_equal_", lambda L, x: x.greater_equal_(4), plain),
        ("less_", lambda L, x: x.less_(3), plain),
        ("less_equal_", lambda L, x: x.less_equal_(4), plain),
        ("not_equal_", lambda L, x: x.not_equal_(4), plain),
        ("i0_", lambda L, x: x.i0_(), plain),
        ("lgamma_", lambda L, x: x.lgamma_(), pos),
        ("lerp_", lambda L, x: x.lerp_(L.tensor(twos), 0.5), plain),
        ("mvlgamma_", lambda L, x: x.mvlgamma_(1), pos),
        ("multiply_", lambda L, x: x.multiply_(3), plain),
        ("nan_to_num_", lambda L, x: x.nan_to_num_(),
         np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)),
        ("nextafter_", lambda L, x: x.nextafter_(L.tensor(twos)), plain),
        ("put_", lambda L, x: x.put_(L.tensor(np.array([0, 2])),
                                     L.tensor(np.array([9., 9.], dtype=np.float32))), plain),
        ("remainder_", lambda L, x: x.remainder_(2), plain),
        ("renorm_", lambda L, x: x.renorm_(2, 0, 1.0), grid),
        ("subtract_", lambda L, x: x.subtract_(1), plain),
        ("true_divide_", lambda L, x: x.true_divide_(2), plain),
        # **The twelve in-place binaries.** Every one is `mutate(binary(name, other))`
        # over a kernel that was already here, so borch.ts builds them from a table —
        # and a table is exactly what a case list cannot check, because the same
        # table would generate the cases. Written out here, against torch's answers.
        ("atan2_", lambda L, x, m="atan2_": getattr(x, m)(L.tensor(twos)), plain),
        ("copysign_", lambda L, x, m="copysign_": getattr(x, m)(L.tensor(twos)), plain),
        ("eq_", lambda L, x, m="eq_": getattr(x, m)(L.tensor(twos)), plain),
        ("ge_", lambda L, x, m="ge_": getattr(x, m)(L.tensor(twos)), plain),
        ("gt_", lambda L, x, m="gt_": getattr(x, m)(L.tensor(twos)), plain),
        ("heaviside_", lambda L, x, m="heaviside_": getattr(x, m)(L.tensor(twos)), plain),
        ("hypot_", lambda L, x, m="hypot_": getattr(x, m)(L.tensor(twos)), plain),
        ("ldexp_", lambda L, x, m="ldexp_": getattr(x, m)(L.tensor(twos)), plain),
        ("le_", lambda L, x, m="le_": getattr(x, m)(L.tensor(twos)), plain),
        ("lt_", lambda L, x, m="lt_": getattr(x, m)(L.tensor(twos)), plain),
        ("ne_", lambda L, x, m="ne_": getattr(x, m)(L.tensor(twos)), plain),
        ("xlogy_", lambda L, x, m="xlogy_": getattr(x, m)(L.tensor(twos)), plain),
        # **In-place operations that change the shape.** Asked on a square only, they pass unchanged.
        ("t_", lambda L, x: x.t_(), grid),
    )
    for name, call, src in derived:
        def go(L, f=call, a=src):
            x = L.tensor(a.copy())
            f(L, x)
            return x
        cases.append((INPLACE_PREFIX + f"짝에서::{name}", go))

    # `bernoulli_` is random and its value cannot be frozen — **at probability 0 or 1 it is
    # determined**, so only those two ends are asked. That it does **not** read its own value as
    # the probability shows here: the input is [1,4,9,2] and at `p=0` it is all zeros.
    for p in (0.0, 1.0):
        def bern(L, prob=p):
            x = L.tensor(plain.copy())
            x.bernoulli_(prob)
            return x
        cases.append((INPLACE_PREFIX + f"짝에서::bernoulli_(p={p})", bern))

    def refuses_float_power(L):
        try:
            L.tensor(plain.copy()).float_power_(2)
        except Exception as exc:                                # noqa: BLE001
            return "Double" if "Double" in str(exc) else f"다른 문구 <{exc}>"
        return "안 던졌다"

    cases.append((INPLACE_PREFIX + "짝에서::float_power_ 는 거절", refuses_float_power))

    # ── the names that existed on the module and not as a method ──
    #
    # torch gives nearly every operation **both ways** — `torch.igamma(x, y)` and `x.igamma(y)`.
    # We had thirteen places with the module side only, and **the side the textbook uses is the
    # method.** The binding's comments already told the same story (`borch.t(x)` worked and
    # `x.t()` did not) and only that side was filled in, not the core.
    grid2 = np.array([[1.5, -2.5], [0.5, 1.0]], dtype=np.float32)
    ones2 = np.ones_like(grid2)

    def method_form(name, call):
        cases.append((INPLACE_PREFIX + f"메서드꼴::{name}", call))

    method_form("arctan2", lambda L: L.tensor(grid2).arctan2(L.tensor(ones2)))
    method_form("igamma", lambda L: L.tensor(grid2).abs().igamma(L.tensor(ones2)))
    method_form("igammac", lambda L: L.tensor(grid2).abs().igammac(L.tensor(ones2)))
    # **The arguments are reversed** — the module is `polygamma(n, x)` and the method is
    # `x.polygamma(n)`. Attaching it from a table straight through was caught. Uncaught, a value
    # would have come out with the order and the input swapped.
    method_form("polygamma(1)", lambda L: L.tensor(grid2).abs().polygamma(1))
    method_form("polygamma(2)", lambda L: L.tensor(grid2).abs().polygamma(2))

    def in_place_new(name, call):
        def run(L, f=call):
            x = L.tensor(grid2.copy())
            f(L, x)
            return x
        cases.append((INPLACE_PREFIX + f"짝없이::{name}", run))

    in_place_new("fill_diagonal_", lambda L, x: x.fill_diagonal_(9))

    # **`wrap` means something only on a tall matrix.** Asked on a square, the flag does nothing,
    # and the case above did exactly that — the borch.ts edition invented a rule that does not
    # exist (skip a row on wrapping) and was still green.
    tall = np.arange(18, dtype=np.float32).reshape(6, 3)
    for flag in (False, True):
        def fill_tall(L, w=flag):
            x = L.tensor(tall.copy())
            x.fill_diagonal_(9, w)
            return x
        cases.append((INPLACE_PREFIX + f"짝없이::fill_diagonal_(세로, wrap={flag})",
                      fill_tall))

    in_place_new("arctan2_", lambda L, x: x.arctan2_(L.tensor(ones2)))
    in_place_new("polygamma_", lambda L, x: x.abs_().polygamma_(1))

    # ── the seven that draw from a distribution and fill — **what can be asked instead of the value** ──
    #
    # All three random generators differ, so the values cannot be frozen (a place already accepted
    # at `randn`). Only two things can go in the table:
    #
    #   **The shape and type do not change** — torch produces an answer, so the table's character
    #   is unchanged. `keep::` and `resize_as_::` already freeze shapes in the same way.
    #
    #   **Refusals** — torch's rule differs per distribution, down to the exception type. Being
    #   wording rather than a value, this is a place where the three part easily.
    #
    # Whether the distribution really is that distribution (its mean, its variance, whether several
    # distinct values appear) is not here. That answer comes from **a predicate we chose** rather
    # than from torch, so it would not measure torch even sitting in the table, and it is a matter
    # of choosing a sample count and a tolerance — that is `tests/test_random_fill.py`.
    draws = (
        ("normal_", lambda L, x: x.normal_(0.0, 2.0)),
        ("uniform_", lambda L, x: x.uniform_(-1.0, 3.0)),
        ("exponential_", lambda L, x: x.exponential_(2.0)),
        ("cauchy_", lambda L, x: x.cauchy_(1.0, 0.5)),
        ("log_normal_", lambda L, x: x.log_normal_(0.0, 1.0)),
        ("geometric_", lambda L, x: x.geometric_(0.3)),
        ("random_", lambda L, x: x.random_(0, 5)),
    )
    for name, call in draws:
        def shape_kept(L, f=call):
            x = L.zeros(2, 3)
            f(L, x)
            return f"{tuple(x.shape)} {x.dtype}"
        cases.append((INPLACE_PREFIX + f"분포::{name} 는 모양과 형을 지킨다",
                      shape_kept))

    # **The five continuous ones refuse integers and `geometric_` and `random_` do not.** Group
    # them by name as "random means floats" and those two are wrong — being discrete, they have an
    # answer in an integer slot.
    def on_int(L, name, arg=()):
        x = L.tensor(np.zeros(6, dtype=np.int64))
        try:
            getattr(x, name)(*arg)
        except Exception as exc:                                # noqa: BLE001
            return f"거절({type(exc).__name__})"
        return f"돈다 {x.dtype}"

    for name, arg in (("normal_", ()), ("uniform_", ()), ("exponential_", ()),
                      ("cauchy_", ()), ("log_normal_", ()),
                      ("geometric_", (0.5,)), ("random_", ())):
        cases.append((INPLACE_PREFIX + f"분포::{name}(int64)",
                      lambda L, n=name, a=arg: on_int(L, n, a)))

    # The arguments' domains. **They differ per distribution** — `p` is an open interval, `lambda`
    # is positive, `from < to`, `std >= 0`.
    def refuses_arg(L, call, fragment):
        try:
            call(L)
        except Exception as exc:                                # noqa: BLE001
            return "멈췄다" if fragment in str(exc) else f"다른 문구 <{exc}>"
        return "안 던졌다"

    # **The range goes as far as the type counts exactly.** float32 past 2^24 cannot tell
    # neighbouring integers apart and the values clump — torch cuts off there too. It was left at
    # 2^53 (float64's place), and cases asking about shape and type alone could not see it.
    for kind, arr, cap in (("float32", np.zeros(1, dtype=np.float32), 1 << 24),
                           ("int64", np.zeros(1, dtype=np.int64), 1 << 62)):
        def ceiling(L, a=arr, c=cap):
            x = L.tensor(np.zeros(3000, dtype=a.dtype))
            x.random_()
            got = np.abs(np.asarray(x.tolist(), dtype=np.float64)).max()
            # **The position itself must not be asked.** `floor(log2(max))` sits just below the
            # ceiling and moves between 23 and 24 — it depends on how close 3,000 draws came to
            # the ceiling, which is luck. The golden answers really did freeze 24 once while the
            # same code gave 23, and only the binding turned red.
            #
            # What is meant to be asked is not one digit but **which type's limit it is.** Landing
            # between the ceiling and its half gives that answer — the old defect of leaving it at
            # 2^53 (float64's place) is caught here as it was, and it does not shake at the boundary.
            return f"{c / 2 <= got <= c}"
        cases.append((INPLACE_PREFIX + f"분포::random_({kind}) 의 상한", ceiling))

    for label, call, fragment in (
        ("geometric_(0)", lambda L: L.zeros(3).geometric_(0), "p to be in (0, 1)"),
        ("geometric_(1)", lambda L: L.zeros(3).geometric_(1), "p to be in (0, 1)"),
        ("exponential_(0)", lambda L: L.zeros(3).exponential_(0), "lambda > 0.0"),
        ("uniform_(3,1)", lambda L: L.zeros(3).uniform_(3, 1), "[from, to) range"),
        ("normal_(0,-1)", lambda L: L.zeros(3).normal_(0, -1), "std >= 0.0"),
        ("random_(5,2)", lambda L: L.tensor(np.zeros(3, dtype=np.int64)).random_(5, 2),
         "'from' to be less than 'to'"),
    ):
        cases.append((INPLACE_PREFIX + f"분포::거절::{label}",
                      lambda L, c=call, f=fragment: refuses_arg(L, c, f)))

    # ── the ones torch gives as **attributes** ──
    #
    # No parentheses. Left as functions, `x.real` returns **a bound method** rather than a tensor
    # and raises nothing — `if x.imag:` passes as true. All three were in that state.
    #
    # `x.device` has to be **an object and not a string.** `x.device.type` is the line a textbook
    # writes to check the device. The value differs per implementation (`cpu` against `webgpu`), so
    # only **the form** is asked here.
    cases.append((INPLACE_PREFIX + "속성::x.device 는 객체다",
                  lambda L: f"{type(L.tensor(grid2).device).__name__} "
                            f"{isinstance(L.tensor(grid2).device.type, str)}"))
    cases.append((INPLACE_PREFIX + "속성::x.real 은 실수부다",
                  lambda L: L.tensor(grid2).real))

    def imag_refuses(L):
        try:
            L.tensor(grid2).imag
        except Exception as exc:                                # noqa: BLE001
            return "멈췄다" if "non-complex" in str(exc) else f"다른 문구 <{exc}>"
        return "안 던졌다"

    cases.append((INPLACE_PREFIX + "속성::x.imag 는 실수에서 멈춘다", imag_refuses))

    # The three with no counterpart, or dropped from the list after being measured. `resize_as_`
    # has no counterpart (`resize_as`) at all and cannot be made from a derived table — writing a
    # name into a table and that table being able to build it are different things.
    line2 = np.array([1.5, -2.5], dtype=np.float32)

    def reduced(name, call):
        def run(L, f=call):
            x = L.tensor(line2.copy())
            f(L, x)
            return x
        cases.append((INPLACE_PREFIX + f"짝없이::{name}", run))

    reduced("index_reduce_",
            lambda L, x: x.index_reduce_(0, L.tensor(np.array([0, 1])),
                                         L.tensor(np.full(2, 3.0, dtype=np.float32)),
                                         "prod"))
    reduced("scatter_reduce_",
            lambda L, x: x.scatter_reduce_(0, L.tensor(np.array([0, 1])),
                                           L.tensor(np.full(2, 2.0, dtype=np.float32)),
                                           "sum"))

    def resized(L):
        x = L.tensor(grid2.copy())
        got = x.resize_as_(L.tensor(np.zeros((1, 4), dtype=np.float32)))
        # **It has to be in place** — returning a new tensor makes the name a lie.
        return f"{tuple(x.shape)} {got is x}"

    cases.append((INPLACE_PREFIX + "짝없이::resize_as_ 는 제자리다", resized))

    # ── the twenty predicates torch gives as attributes ──
    #
    # Most have one determined answer. **The name still has to exist** — without it, `if x.is_cuda:`
    # stops with an `AttributeError`, where in torch it is a line that simply passes as false. The
    # standard for calling an always-false predicate shallow was **whether there is an input that
    # produces false**, not whether the name exists.
    #
    # `is_cpu` **parts deliberately** — the browser side's values live in a GPU buffer. Saying true
    # sends code that branches on it down the wrong path.
    for label in ("is_cuda", "is_mps", "is_sparse", "is_quantized", "is_nested",
                  "is_meta", "is_mkldnn", "is_vulkan", "is_xla", "is_xpu",
                  "is_ipu", "is_maia", "is_mtia", "is_sparse_csr",
                  "retains_grad"):
        cases.append((INPLACE_PREFIX + f"술어::{label}",
                      lambda L, n=label: str(getattr(L.tensor(grid2), n))))
    # **`is_leaf` alone is a real computation.** A tensor that came out of an operation is false,
    # and false means `.grad` does not accumulate — a different character from the rest, whose
    # answer is fixed.
    cases.append((INPLACE_PREFIX + "술어::is_leaf(잎)",
                  lambda L: str(L.tensor(grid2, requires_grad=True).is_leaf)))
    cases.append((INPLACE_PREFIX + "술어::is_leaf(파생)",
                  lambda L: str((L.tensor(grid2, requires_grad=True) * 2).is_leaf)))
    # **These two alone are methods** — they have parentheses. Left as attributes, `x.is_neg`
    # returns a bound method rather than a boolean, and that passes as true.
    for label in ("is_pinned", "is_neg"):
        cases.append((INPLACE_PREFIX + f"술어::{label}() 는 메서드",
                      lambda L, n=label: str(getattr(L.tensor(grid2), n)())))

    def coalesced_refuses(L):
        try:
            L.tensor(grid2).is_coalesced()
        except Exception as exc:                                # noqa: BLE001
            return ("멈췄다" if "but got Strided" in str(exc)
                    else f"다른 문구 <{exc}>")
        return "안 던졌다"

    cases.append((INPLACE_PREFIX + "술어::is_coalesced 는 조밀에서 멈춘다",
                  coalesced_refuses))

    # ── the eight in-place editions with no counterpart ──
    #
    # torch manages five and three are sparse-only, so **torch stops on a dense tensor too.** Group
    # them by name as "they are in place, so build them all" and we become more lenient than torch
    # on the last three.
    reduced("apply_", lambda L, x: x.apply_(lambda v: v * 2))
    reduced("map_", lambda L, x: x.map_(L.tensor(np.ones(2, dtype=np.float32)),
                                        lambda a, b: a + b))
    reduced("map2_", lambda L, x: x.map2_(L.tensor(np.ones(2, dtype=np.float32)),
                                          L.tensor(np.full(2, 2.0, dtype=np.float32)),
                                          lambda a, b, c: a + b + c))
    # **The growing direction is not asked** — torch gives undetermined values. The shrinking
    # direction and the shape-only direction have answers.
    reduced("resize_(줄임)", lambda L, x: x.resize_(1))
    reduced("resize_(모양만)", lambda L, x: x.resize_(1, 2))
    reduced("set_", lambda L, x: x.set_(L.tensor(np.zeros(3, dtype=np.float32))))

    for gone in ("sparse_resize_", "resize_as_sparse_"):
        def sparse_only(L, n=gone):
            try:
                arg = ((2,), 1, 0) if n == "sparse_resize_" else (L.tensor(line2),)
                getattr(L.tensor(line2.copy()), n)(*arg)
            except Exception as exc:                            # noqa: BLE001
                return (f"멈췄다({type(exc).__name__})"
                        if isinstance(exc, NotImplementedError)
                        else f"다른 종류 <{type(exc).__name__}>")
            return "안 던졌다"
        cases.append((INPLACE_PREFIX + f"짝없이::{gone} 는 희소 전용", sparse_only))

    # ── the ones that look into the storage ──
    #
    # They ask **how it is laid out** rather than the value. `stride` changes under a transpose,
    # and that is the fact of its being a view — **the browser side has no views so it does not
    # change** (the same root as refusing view propagation, so it is kept beside that place).
    wide = np.arange(6, dtype=np.float32).reshape(2, 3)
    for label, call in (
        ("stride()", lambda L: L.tensor(wide).stride()),
        ("dim_order()", lambda L: L.tensor(wide).dim_order()),
        ("element_size()", lambda L: L.tensor(wide).element_size()),
        ("nelement()", lambda L: L.tensor(wide).nelement()),
        ("ndimension()", lambda L: L.tensor(wide).ndimension()),
        ("itemsize", lambda L: L.tensor(wide).itemsize),
        ("nbytes", lambda L: L.tensor(wide).nbytes),
        ("layout", lambda L: L.tensor(wide).layout),
        ("output_nr", lambda L: L.tensor(wide).output_nr),
    ):
        cases.append((INPLACE_PREFIX + f"저장::{label}", lambda L, f=call: str(f(L))))

    def stride_after_transpose(L):
        """**Transposing changes the strides** — the core is a numpy view and gives `(1, 3)`, and
        the browser has no views and stays at `(3, 1)`. Which one it is, is answered as the value."""
        got = L.tensor(wide).t().stride()
        views = not hasattr(L, "backend")
        return "기대대로" if (got == (1, 3)) == views else f"뜻밖에 {got}"

    cases.append((INPLACE_PREFIX + "저장::전치한 걸음=브라우저는뷰가없다",
                  stride_after_transpose))

    # ── the three names for transposing ──
    #
    # `H` is **2-D only** and `mT` and `mH` swap the last two axes, so they work on batches too.
    # The difference between the three is **whether they conjugate**, and over the reals all three
    # give the same answer, so **it has to be asked on complex numbers** to show.
    cplx = np.array([[1 + 2j, 3 - 1j]], dtype=np.complex64)
    for label, call in (
        ("H", lambda L: L.tensor(wide).H),
        ("mT", lambda L: L.tensor(wide).mT),
    ):
        cases.append((INPLACE_PREFIX + f"전치::{label}", call))

    # **The browser side refuses complex** — borch.ts's transpose does not know interleaved
    # storage yet. The difference between the three (whether they conjugate) shows only on complex
    # numbers, so with the value unaskable the difference itself cannot be asked. A place where
    # refusal is the answer, so it goes through `_as_expected`.
    #
    # torch raises the conjugate **as a bit only**, so `numpy()` stops — it has to be resolved with
    # `resolve_conj()` for the value to be visible, and our side already stores it flipped, so that
    # call is the identity.
    for label, call in (
        ("H(복소수)", lambda L: L.tensor(cplx).H.resolve_conj()),
        ("mT(복소수) 는 켤레를 안 한다", lambda L: L.tensor(cplx).mT.resolve_conj()),
        ("mH(복소수) 는 켤레를 한다", lambda L: L.tensor(cplx).mH.resolve_conj()),
    ):
        cases.append((INPLACE_PREFIX + f"전치::{label}=브라우저는거절",
                      _as_expected(call)))

    def h_needs_a_matrix(L):
        try:
            L.tensor(np.array([1.0, 2.0], dtype=np.float32)).H
        except Exception as exc:                                # noqa: BLE001
            return "멈췄다" if "2-D" in str(exc) else f"다른 문구 <{exc}>"
        return "안 던졌다"

    cases.append((INPLACE_PREFIX + "전치::H 는 1차원에서 멈춘다", h_needs_a_matrix))

    # ── `new_*` — built **inheriting** the type ──
    for label, call in (
        ("new_zeros", lambda L: L.tensor(wide).new_zeros((2,))),
        ("new_ones", lambda L: L.tensor(wide).new_ones(2, 2)),
        ("new_full", lambda L: L.tensor(wide).new_full((2,), 7)),
        ("new_tensor", lambda L: L.tensor(wide).new_tensor([1, 2])),
        ("reshape_as", lambda L: L.tensor(wide).reshape_as(
            L.tensor(np.zeros((3, 2), dtype=np.float32)))),
        ("view_as", lambda L: L.tensor(wide).view_as(
            L.tensor(np.zeros((3, 2), dtype=np.float32)))),
        # **Undoes a broadcast** — the same thing backpropagation does inside.
        ("sum_to_size", lambda L: L.tensor(wide).sum_to_size(2, 1)),
    ):
        cases.append((INPLACE_PREFIX + f"새로::{label}", call))
    # **The value is undetermined** — torch gives garbage. Only the shape and type are asked.
    cases.append((INPLACE_PREFIX + "새로::new_empty 는 모양만",
                  lambda L: f"{tuple(L.tensor(wide).new_empty((2, 3)).shape)} "
                            f"{L.tensor(wide).new_empty((2, 3)).dtype}"))

    # ── `retain_grad` — **it stops at a leaf and keeps the gradient on a derived tensor** ──
    #
    # Getting only the refusal right is half of it. What the name is for is keeping a derived
    # tensor's `.grad`, and without that it is a shape that imitates the refusal alone.
    def retain_refuses(L):
        """**What decides is `requires_grad`, not whether it is a leaf.** On a leaf that takes
        gradients it simply passes — they already accumulate, so there is nothing to ask for."""
        try:
            L.tensor(wide).retain_grad()
        except Exception as exc:                                # noqa: BLE001
            return ("멈췄다" if "requires_grad=False" in str(exc)
                    else f"다른 문구 <{exc}>")
        return "안 던졌다"

    cases.append((INPLACE_PREFIX + "기울기::retain_grad 는 requires_grad 를 본다",
                  retain_refuses))
    cases.append((INPLACE_PREFIX + "기울기::잎에 부르면 지나간다",
                  lambda L: str(L.tensor(wide, requires_grad=True).retain_grad())))

    # Whether a value is **really kept** after `retain_grad` is in `tests/test_fold_grad.py` —
    # borch.ts does not hand out a derived tensor's `.grad`, so the three cannot be asked together.
    cases.append((INPLACE_PREFIX + "기울기::grad_fn 의 형 이름",
                  lambda L: type((L.tensor(wide, requires_grad=True) * 2).grad_fn).__name__))
    cases.append((INPLACE_PREFIX + "기울기::잎의 grad_fn 은 없다",
                  lambda L: str(L.tensor(wide, requires_grad=True).grad_fn)))

    # ── the two torch refuses of its own accord ──
    for gone in ("eig", "symeig"):
        def deprecated_op(L, n=gone):
            try:
                getattr(L.tensor(wide), n)()
            except Exception as exc:                            # noqa: BLE001
                return ("멈췄다" if "deprecated" in str(exc)
                        else f"다른 문구 <{str(exc).splitlines()[0][:44]}>")
            return "안 던졌다"
        cases.append((f"dtype::없는이름::{gone}(폐기됨)", deprecated_op))

    # The three that return **a boolean** rather than a value. `is_same_size` looks at shape alone.
    for label, call in (
        ("is_same_size(같음)",
         lambda L: L.tensor(grid2).is_same_size(L.tensor(ones2))),
        ("is_same_size(다름)",
         lambda L: L.tensor(grid2).is_same_size(L.tensor(np.zeros((3, 3), dtype=np.float32)))),
        ("is_inference", lambda L: L.tensor(grid2).is_inference()),
        ("is_distributed", lambda L: L.tensor(grid2).is_distributed()),
        # **The textbook's idiom** — `x.requires_grad_()` turns a leaf on and hands itself back.
        ("requires_grad_ 가 자기를 돌려준다",
         lambda L: L.tensor(grid2).requires_grad_(True).requires_grad),
        ("share_memory_ 가 자기를 돌려준다",
         lambda L: L.tensor(grid2).share_memory_() is not None),
    ):
        cases.append((INPLACE_PREFIX + f"묻는꼴::{label}",
                      lambda L, f=call: str(f(L))))

    # **`tensor()` copies — sharing would quietly change the user's array.**
    #
    # This case came out of putting in the forty-one above. One case took its input without
    # `.copy()` and did `add_(1)`, and **torch copied so nothing leaked and the core alone leaked.**
    # So sixteen cases after it were wrong on the core alone, and the cause was not in its own place,
    # which cost a long search. When two libraries part, a defect surfaces **under the wrong name.**
    def copies_input(L):
        src = np.array([1., 4., 9., 2.], dtype=np.float32)
        L.tensor(src).add_(1)
        return L.tensor(src)          # the original has to be unchanged

    cases.append((INPLACE_PREFIX + "tensor() 는 사본을 뜬다", copies_input))

    def from_numpy_value(L):
        """All three carry the values."""
        return L.from_numpy(np.array([1., 4., 9., 2.], dtype=np.float32))

    cases.append((INPLACE_PREFIX + "from_numpy 는 값을 나른다", from_numpy_value))

    def from_numpy_aliasing(L):
        """**There is no sharing in the browser.** The values live in a GPU buffer, so there is
        nowhere to share storage with a host array — the same reason view propagation is refused.
        It becomes a copy rather than a refusal, so `_as_expected` cannot hold it, and which side
        it is, is answered as the value.

        If the browser side starts sharing one day (it cannot), it parts here."""
        src = np.array([1., 4., 9., 2.], dtype=np.float32)
        L.from_numpy(src).add_(1)
        shared = bool(src[0] != 1.0)
        must_copy = hasattr(L, "backend")        # true on the browser side alone
        if shared == (not must_copy):
            return "기대대로"
        return "뜻밖의 공유" if shared else "뜻밖의 사본"

    cases.append((INPLACE_PREFIX + "from_numpy 의 공유=브라우저는사본",
                  from_numpy_aliasing))

    # ── `nn.Parameter` — one thing agrees and one parts ──
    #
    # All three produce **a new object** and **leave the original's flag alone** — that matches
    # torch. What parts is the storage: torch and the core share it and both browser sides copy.
    # The same place as `from_numpy` above and for the same reason (a GPU buffer cannot share
    # storage with the host). Leaving the divergence answered as a value catches it the day one
    # side changes.
    def parameter_leaves_source(L):
        t = L.tensor(np.array([1., 2., 3.], dtype=np.float32))
        p = L.nn.Parameter(t)
        return f"{p.requires_grad} {t.requires_grad} {p is t}"

    cases.append((INPLACE_PREFIX + "Parameter 는 원본을 안 건드린다",
                  parameter_leaves_source))

    def parameter_aliasing(L):
        t = L.tensor(np.array([1., 2., 3.], dtype=np.float32))
        p = L.nn.Parameter(t, requires_grad=False)
        p.add_(1)
        shared = bool(float(t.numpy()[0]) != 1.0)
        must_copy = hasattr(L, "backend")        # true on the browser side alone
        if shared == (not must_copy):
            return "기대대로"
        return "뜻밖의 공유" if shared else "뜻밖의 사본"

    cases.append((INPLACE_PREFIX + "Parameter 의 공유=브라우저는사본",
                  parameter_aliasing))
    return cases


SHAPE_PREFIX = "shape::"


def shape_cases(inp=None):
    """The rest of reshaping.

    **Called as methods.** `expand`, `repeat`, `ravel`, `select`, `unfold` and `expand_as` have no
    module function in torch and exist as methods only — there is one way to call them, so that is
    the way they are asked.
    """
    mat = np.arange(6, dtype=np.float32).reshape(2, 3)
    square = np.arange(9, dtype=np.float32).reshape(3, 3)
    line = np.arange(5, dtype=np.float32)
    col = mat[:, :1].copy()
    flat6 = np.arange(6, dtype=np.float32)
    pair = np.array([1., 2.], dtype=np.float32)
    # **Rank 3.** Asked in 2-D only, an axis swap never sees a position outside `(0,1)` — in 2-D
    # any two axes give the same one answer, so an implementation discarding the axis arguments
    # entirely passes. The axis lengths are all different so it is caught by shape first.
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
        # **A permutation that is not its own inverse.** The table's `permute` cases were all
        # `(1,0)` or an axis reversal, both of which are their own inverse, so **an implementation
        # applying the permutation backwards passed every one.** At rank 3, `(1,2,0)`'s inverse is
        # `(2,0,1)` and the shape parts first ([3,4,2] against [4,2,3]).
        #
        # The backward pass more so — undoing it needs **the inverse permutation**, and when the
        # permutation is its own inverse, using the same one both ways still gives the right answer.
        ("permute(비가역)", lambda t: t.permute(1, 2, 0), cube),
        ("permute(비가역의 역)", lambda t: t.permute(2, 0, 1), cube),
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

    # The splitting ones — each piece is named. Looking at one leaves the rest uncaught.
    for name, parts in (("hsplit", 3), ("vsplit", 2)):
        for i in range(parts):
            cases.append((SHAPE_PREFIX + f"{name}[{i}]",
                          lambda L, n=name, p=parts, k=i: getattr(L.tensor(mat), n)(p)[k]))

    cases.append((SHAPE_PREFIX + "atleast_2d",
                  lambda L: L.atleast_2d(L.tensor(np.float32(1.)))))

    # Gradients. **`expand` and `unfold` part here** — expand sums the stretched axis back
    # together, and unfold stacks by the window overlap (measured: unfolding a length 5 by 3·1
    # gives [1,2,3,2,1]).
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
        # **Undoing it uses the inverse permutation.** When the permutation is its own inverse,
        # using the same one both ways gives the right answer, so the table's `(1,0)` and axis
        # reversals never opened that place.
        ("permute(비가역)", lambda t: t.permute(1, 2, 0), cube),
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
    """The rest of the reductions — `amax`, `nansum`, `logsumexp`, `cummax`, `kthvalue` and so on.

    **Ties are put in deliberately.** `amax` spreads the gradient evenly on a tie (measured:
    [1,3,3,2] → [0,.5,.5,0]) and `cummax` gives it to the later position. Measured on input with no
    ties, none of those rules is seen, and what is left is a place where the value is right and the
    training parts subtly.
    """
    tie = np.array([1., 3., 3., 2.], dtype=np.float32)          # there is a tie
    mat = np.array([[1., 5., 3.], [4., 2., 6.]], dtype=np.float32)
    withnan = np.array([1., np.nan, 3., 5.], dtype=np.float32)
    zeros_in = np.array([0., 1., 0., 2.], dtype=np.float32)
    weights = np.arange(1, 5, dtype=np.float32)

    def values_of(L, got):
        # **It must not ask with `hasattr(got, "numpy")`.** Our pair passes an unfound name
        # through as a value, so it answers true to `numpy` too, and then this function hands the
        # pair straight back — torch's namedtuple does not pass it through, so **the same helper
        # takes a different branch per library.** It stayed green in that state for a long time and
        # then broke once `Tensor.values` (for sparse) arrived and `.values` resolved to a method on
        # the result of multiplying the pair.
        return got if isinstance(got, L.Tensor) else got.values

    cases = []

    def add(name, fn, grad_of=None):
        cases.append((REDUCE_PREFIX + name, lambda L, f=fn: values_of(L, f(L))))
        if grad_of is not None:
            def run(L, f=fn, arr=grad_of):
                x = L.tensor(arr, requires_grad=True)
                out = values_of(L, f(L, x))
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

    # **The ones that take an axis have to be asked by value.** Asked by gradient alone, ignoring
    # the axis entirely passes — the gradients of `sum(dim=1).sum()` and `sum().sum()` are both all
    # ones, so the answers agree. A case named `grad::sum(dim)` already existed, and because of that
    # name nobody looked again.
    #
    # In the meantime `borch_webgpu` was returning **a scalar with the axis ignored** for
    # `sum(dim=1)`. borch.ts keeps the whole sum and the axis sum under different names and JS
    # quietly discards a surplus argument. All 792 were green until one rank-6 case was caught by
    # shape.
    #
    # **Asked as a method.** The module function `L.sum` is in neither the core nor the sister
    # library — torch has it, so that is a hole of its own, and a different story from what this
    # case is trying to catch.
    add("sum(dim)", lambda L, x=None: (x if x is not None else L.tensor(mat)).sum(dim=1), mat)
    add("sum(dim0)", lambda L, x=None: (x if x is not None else L.tensor(mat)).sum(dim=0), mat)
    add("sum(dim,keepdim)",
        lambda L, x=None: (x if x is not None else L.tensor(mat)).sum(dim=1, keepdim=True), mat)
    add("norm(dim)", lambda L, x=None: (x if x is not None else L.tensor(mat)).norm(dim=1), mat)
    add("norm(p=1,dim)",
        lambda L, x=None: (x if x is not None else L.tensor(mat)).norm(p=1, dim=0), mat)

    # The ones that give an index — by value alone, a wrong index passes.
    for name, fn in (("cummax", lambda L: L.cummax(L.tensor(tie), 0)),
                     ("cummin", lambda L: L.cummin(L.tensor(tie), 0)),
                     ("kthvalue", lambda L: L.kthvalue(L.tensor(tie), 2))):
        cases.append((REDUCE_PREFIX + f"{name} 번호", lambda L, f=fn: f(L).indices))

    # The ones with no gradient. The value alone is frozen.
    cases += [
        (REDUCE_PREFIX + "quantile", lambda L: L.quantile(L.tensor(tie), 0.5)),
        (REDUCE_PREFIX + "quantile(여럿)",
         lambda L: L.quantile(L.tensor(tie), L.tensor(np.array([0.25, 0.75], dtype=np.float32)))),
        (REDUCE_PREFIX + "nanquantile",
         lambda L: L.nanquantile(L.tensor(withnan), 0.5)),
        (REDUCE_PREFIX + "nonzero", lambda L: L.nonzero(L.tensor(zeros_in))),
        (REDUCE_PREFIX + "argwhere", lambda L: L.argwhere(L.tensor(zeros_in))),
        # **Asked by index.** torch's `aminmax` is reached through `.min` and `.max` and ours
        # through `.values` and `.indices`, so the names do not fit — asked by position, both work.
        (REDUCE_PREFIX + "aminmax/최소", lambda L: L.aminmax(L.tensor(tie))[0]),
        (REDUCE_PREFIX + "aminmax/최대", lambda L: L.aminmax(L.tensor(tie))[1]),
    ]
    return cases


MATH_PREFIX = "math::"

# The newly attached maths functions. **The input range differs per function** — `acos` is NaN
# outside [-1,1], and NaN differs from itself, so a comparison cannot pass. So they are asked
# within their domain only. What they do outside is a separate question, and mixing it in here
# means seeing neither.
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
    """The rest of the trigonometric, exponential and logarithmic functions. Value and gradient are
    **both** asked.

    There is a reason the gradient is asked alongside. A function whose value is right and whose
    graph is cut passes a value check, and this repository has already found fourteen of them.
    """
    plain = np.array([0.5, 2.0, -1.5, 3.0], dtype=np.float32)
    unit = np.array([0.2, 0.6, -0.9, 0.45], dtype=np.float32)      # within (-1, 1)
    big = np.array([1.5, 2.5, 3.0, 1.2], dtype=np.float32)          # > 1
    pos = np.array([0.5, 2.0, 1.5, 3.0], dtype=np.float32)
    other = np.array([1.0, 2.0, -3.0, 0.5], dtype=np.float32)
    logit_in = np.array([0.2, 0.6, 0.35, 0.45], dtype=np.float32)   # within (0, 1)
    weights = np.arange(1, 5, dtype=np.float32)                     # a different weight per slot

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
        # **The method spelling reaches the same place.** Only the function form was
        # ever asked, and borch.ts had these as kernels with no name on `Tensor` — so
        # `x.hypot(y)` was a type error there while `hypot(x, y)` had been frozen in
        # the golden for months. Asking both is what says the two spellings are one
        # operation rather than two that happen to agree today.
        cases.append((MATH_PREFIX + name + "/메서드",
                      lambda L, n=name: getattr(L.tensor(plain), n)(L.tensor(other))))
        for who in (0, 1):
            def bgrad(L, n=name, w=who):
                leaves = [L.tensor(plain, requires_grad=True),
                          L.tensor(other, requires_grad=True)]
                (getattr(L, n)(*leaves) * L.tensor(weights)).sum().backward()
                return _grad_of(leaves[w], f"{n}/{'ab'[w]}")

            cases.append((MATH_PREFIX + f"grad::{name}/{'ab'[who]}", bgrad))

    # `x·log(y)` — **there has to be a slot where x is 0 for this to be testing the function.**
    zeros_in = np.array([0.0, 2.0, 0.0, 3.0], dtype=np.float32)
    ypos = np.array([1.0, 2.0, 0.5, 4.0], dtype=np.float32)
    cases.append((MATH_PREFIX + "xlogy(x에 0 포함)",
                  lambda L: L.xlogy(L.tensor(zeros_in), L.tensor(ypos))))

    # A step function **flows a 0.** It used to cut the graph so `backward()` refused, and torch
    # gives 0 — absent and zero are different.
    for name in ("sign", "floor", "ceil", "round", "trunc", "fix"):
        def zgrad(L, n=name):
            x = L.tensor(plain, requires_grad=True)
            (getattr(L, n)(x) * L.tensor(weights)).sum().backward()
            return _grad_of(x, n)

        cases.append((MATH_PREFIX + f"grad::{name}(0이어야)", zgrad))

    # The ones with a value and no gradient — booleans or staircases.
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
        # **`clone()` is needed.** torch's sgn gradient is a ZeroTensor (a lazy zero tensor) and
        # `.numpy()` refuses it — multiplication does not resolve it (`* 1.0` is a ZeroTensor too)
        # and cloning is what produces a real buffer. The value is 0.
        #
        # This place was read at first as "torch refuses to backpropagate sgn" and frozen as a
        # refusal case. That was wrong — the exception came from printing the result, not from
        # `backward()`.
        return x.grad.detach().clone()

    cases.append((MATH_PREFIX + "grad::sgn(0이어야)", sgn_grad))
    return cases


METHOD_PREFIX = "method::"

# The ones that have to be callable as `x.f(...)`, and the arguments to give them.
#
# **A growing surface grows the places that can be quietly wrong** — this repository has met that
# four times. So attaching a name and standing a case on that name happen together. That no surface
# goes in without a case is the one condition attached to deciding to grow the functionality here.
#
# Whether `x.f(...)` and `torch.f(x, ...)` agree was also asked of torch before being written down.
# One differed — `where` has its arguments reversed, and attached straight through, `x` would have
# gone into the condition's place.
_METHOD_ARGS = {
    "ceil": (), "cos": (), "cosh": (), "erf": (), "floor": (), "isfinite": (),
    "isinf": (), "isnan": (), "neg": (), "reciprocal": (), "relu": (), "round": (),
    "sigmoid": (), "sign": (), "sin": (), "sinh": (), "square": (), "tan": (),
    "tanh": (), "prod": (), "norm": (), "argsort": (), "unique": (),
    "clamp": (0.0, 1.0), "pow": (2,), "roll": (1,), "cumsum": (0,), "cumprod": (0,),
    "softmax": (0,), "narrow": (0, 0, 2), "flip": ((0,),),
    "tile": ((2,),), "topk": (2,), "sort": (), "median": (),
}
# The ones that take positives only — a negative gives NaN, and NaN differs from itself.
_METHOD_ARGS_POS = {"log2": (), "log10": (), "rsqrt": ()}
# The ones that need a partner. The partner is a different vector of the same shape.
_METHOD_ARGS_PAIR = {"eq": (), "ne": (), "lt": (), "le": (), "gt": (), "ge": (),
                     "maximum": (), "minimum": (), "dot": (), "outer": ()}
# The ones that need a matrix.
_METHOD_ARGS_MAT = {"diag": (), "trace": (), "tril": (), "triu": ()}


def method_cases(inp=None):
    """Can a module function be called **as a method too?** The value is compared as well.

    A name that exists with a different value is a lie too — this table does not ask `hasattr`.
    """
    pos = np.array([0.5, 2.0, 1.5, 3.0], dtype=np.float32)
    vec = np.array([0.5, 2.0, -1.5, 3.0], dtype=np.float32)
    other = np.array([1.0, 2.0, -3.0, 0.5], dtype=np.float32)
    mat = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    mask = np.array([True, False, True, False])

    def values_of(L, got):
        """The ones returning (value, index) are looked at on the value side — the index can part
        on a tie.

        It must not be written as `getattr(got, "values", got)`. **A real torch tensor has
        `.values` as a method** (for sparse tensors), so that method comes out instead of the
        tensor — the freezing step said so by blowing up with
        `'builtin_function_or_method' object has no attribute 'detach'`. Whether it is a tensor is
        asked first.

        **The way of asking was wrong for a long time.** It asked `hasattr(got, "numpy")`, and our
        pair passes an unfound name through as a value so it answers true — torch's namedtuple does
        not pass it through, so the same helper took a different branch per library. Asking whether
        it is that library's `Tensor` makes all three take the same branch.
        """
        return got if isinstance(got, L.Tensor) else got.values

    def call(name, args, arr, extra=()):
        def run(L, n=name, a=args, base=arr, ex=extra):
            return values_of(
                L, getattr(L.tensor(base), n)(*[L.tensor(e) for e in ex], *a))
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

    # The ones returning several things name each piece — looking at one leaves the rest uncaught.
    for name, args in (("chunk", (2,)), ("split", (2,)), ("unbind", ())):
        for piece in (0, 1):
            cases.append((
                METHOD_PREFIX + f"{name}[{piece}]",
                lambda L, n=name, a=args, p=piece: getattr(L.tensor(vec), n)(*a)[p]))

    # `where` — **the only place whose argument order is reversed from the function's.**
    cases.append((METHOD_PREFIX + "where",
                  lambda L: L.tensor(vec).where(L.tensor(mask), L.tensor(other))))
    # **The three broadcast against each other**, and every `where` case above hands
    # all three the same shape — which is the one arrangement that cannot tell a
    # broadcasting `where` from one that walks three buffers at the same offset. The
    # second is what borch.ts had: right shape, values taken from wherever the shorter
    # buffer happened to reach. It surfaced through `roi_align`, four layers away,
    # where a mask is per position and the values are per channel.
    _wide = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    _thin = (np.arange(2 * 1 * 4) % 3 > 0).reshape(2, 1, 4)
    cases.append((METHOD_PREFIX + "where(마스크가 좁다)",
                  lambda L: L.where(L.tensor(np.ascontiguousarray(_thin)),
                                    L.tensor(np.ascontiguousarray(_wide)),
                                    L.zeros_like(L.tensor(
                                        np.ascontiguousarray(_wide))))))
    # And the other way: the mask is the wide one and a branch is the narrow one.
    cases.append((METHOD_PREFIX + "where(가지가 좁다)",
                  lambda L: L.where(
                      L.tensor(np.ascontiguousarray(_wide)) > 5.0,
                      L.tensor(np.ascontiguousarray(
                          np.arange(4, dtype=np.float32))),
                      L.tensor(np.ascontiguousarray(_wide)))))
    # The ones returning a boolean. **A predicate** is frozen rather than a value.
    cases.append((METHOD_PREFIX + "equal",
                  lambda L: str(bool(L.tensor(vec).equal(L.tensor(vec))))))
    cases.append((METHOD_PREFIX + "equal(다른 것)",
                  lambda L: str(bool(L.tensor(vec).equal(L.tensor(other))))))
    cases.append((METHOD_PREFIX + "allclose",
                  lambda L: str(bool(L.tensor(vec).allclose(L.tensor(vec))))))

    # The matmul family has different shapes and is given separately.
    # `movedim` **asks all four combinations.** It used to be `(0, 0)` alone, which is the identity
    # and so asked nothing, and hiding behind it the sister library's `movedim(0, -1)` was quietly
    # behaving as the identity (`list.insert(-1, …)` is not the end).
    for src, dst in ((0, -1), (-1, 0), (0, 1), (1, 0)):
        cases.append((METHOD_PREFIX + f"movedim({src},{dst})",
                      lambda L, s=src, d=dst: L.tensor(mat).movedim(s, d)))

    cases.append((METHOD_PREFIX + "mm", lambda L: L.tensor(mat).mm(L.tensor(mat))))
    cases.append((METHOD_PREFIX + "gather",
                  lambda L: L.tensor(mat).gather(1, L.tensor(
                      np.array([[0, 2], [1, 0], [2, 1]], dtype=np.int64)))))
    # The gradient is looked at too. A graph cut by being called as a method still passes a check
    # that looks only at values.
    def method_grad(L):
        x = L.tensor(vec, requires_grad=True)
        (x.square() * L.arange(4).float()).sum().backward()
        return _grad_of(x, "method::square")

    cases.append((METHOD_PREFIX + "grad::square", method_grad))
    return cases


EDGE_PREFIX = "edge::"


def edge_cases(inp=None):
    """**Where things bend.** The places the rest of the tables structurally cannot see, gathered.

    Nearly every other table's input is a normal draw from `default_rng`. That is a good default and
    it cannot do one thing — **a special value never once comes up.** Exactly 0, exactly two equal
    numbers, exactly a boundary value, exactly .5. Every place a function bends is there.

    That is how `relu` was breached. At an input of exactly 0 torch gives a gradient of 0 (it is
    `x > 0`, not `x >= 0`) and borch.ts flowed 1, and all 798 golden cases passed — because the relu
    case's input had no 0 in it. Fixing that one is not enough — there are this many more places
    invisible for the same reason.

    **The answers are not guessed here.** Whether torch divides the gradient on a tie or gives it to
    one side, whether `round(0.5)` is 0 or 1, is not ours to decide. Whatever real torch does is the
    answer, and this table only asks for it.
    """
    cases = []

    def value(name, fn):
        cases.append((EDGE_PREFIX + name, fn))

    def grad(name, fn, arr, which=0):
        """Folded with a different weight per slot — folded uniformly, the bending place is buried.

        **The weights start at 1.** Starting from `arange` made the first slot's share 0, and a case
        whose output is a single slot has that one as the whole of it, so **the gradient became
        entirely 0.** A device put in to avoid uniform folding turned that case into one asking
        nothing — `edge::grad::max(동점)` was freezing `[0,0,0,0]` instead of `[0,1,0,0]`, and an
        implementation flowing no gradient at all passed.
        """
        def run(L, f=fn, a=arr, n=name, w=which):
            leaves = [L.tensor(x.copy(), requires_grad=True) for x in a]
            out = f(L, *leaves)
            if out.shape:
                out = out * (L.arange(out.numel()).reshape(out.shape).float() + 1)
            out.sum().backward()
            return _grad_of(leaves[w], n)
        cases.append((EDGE_PREFIX + "grad::" + name, run))

    # Input holding exactly 0. Nearly every case in this table uses it.
    z = np.array([-2., -1., 0., 1., 2., 0.], dtype=np.float32)
    # A pair with exactly equal values. It asks where the gradient goes on a tie.
    ta = np.array([1., 2., 3., 2.], dtype=np.float32)
    tb = np.array([1., 5., 3., 0.], dtype=np.float32)      # slots 0 and 2 are tied
    # Values ending in .5 — the rounding rule (ties to even) shows only here.
    half = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    # Division with mixed signs. Where `%`'s sign rule parts between languages.
    neg = np.array([-7., -3., 3., 7.], dtype=np.float32)

    # ── the ones that bend at 0 ──
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

    # ── clamp landing exactly on the boundary ──
    # The input holds -1 and 1 exactly. Whether the boundary between clipping and passing through
    # is `<` or `<=` parts only here.
    value("clamp(경계에서)", lambda L: L.clamp(L.tensor(z), min=-1., max=1.))
    grad("clamp(경계에서)", lambda L, x: L.clamp(x, min=-1., max=1.), (z,))
    grad("clamp(위만)", lambda L, x: L.clamp(x, max=1.), (z,))
    grad("clamp(아래만)", lambda L, x: L.clamp(x, min=-1.), (z,))

    # ── ties ──
    # **torch divides the gradient on a tie.** When maximum's two inputs are equal, each takes half.
    # An implementation giving it all to one side has a perfectly identical forward pass, so a value
    # comparison can never catch it.
    value("maximum(동점)", lambda L: L.maximum(L.tensor(ta), L.tensor(tb)))
    value("minimum(동점)", lambda L: L.minimum(L.tensor(ta), L.tensor(tb)))
    for who in (0, 1):
        grad(f"maximum(동점)/{'ab'[who]}",
             lambda L, a, b: L.maximum(a, b), (ta, tb), which=who)
        grad(f"minimum(동점)/{'ab'[who]}",
             lambda L, a, b: L.minimum(a, b), (ta, tb), which=who)

    # A tie on the folding side — where it flows when the maximum is in two slots.
    # (`max`, `min` and `argmax` exist in both libraries **as methods only.** torch has module
    #  functions too, so that is a divergence in itself, but what is asked here is the tie, so they
    #  are used as methods.)
    dup = np.array([1., 3., 2., 3.], dtype=np.float32)
    value("max(동점).indices", lambda L: L.tensor(dup).max(dim=0).indices)
    value("min(동점).indices", lambda L: L.tensor(-dup).min(dim=0).indices)
    value("argmax(동점)", lambda L: L.tensor(dup).argmax())
    grad("max(동점)", lambda L, x: x.max(dim=0).values.reshape(1), (dup,))

    # A tie in sorting — is **the order** among equal values stable. A parted answer parts the indices.
    value("sort(동점).values", lambda L: L.sort(L.tensor(dup)).values)
    value("sort(동점).indices", lambda L: L.sort(L.tensor(dup)).indices)
    value("topk(동점).indices", lambda L: L.topk(L.tensor(dup), 3).indices)

    # Pooling with two equal values in one window. **The answer differs from `maximum`'s** —
    # torch's pooling picks one winning slot and flows only there, without dividing. Implement
    # pooling over `maximum` (two of the three libraries did) and it parts only here.
    tied_img = np.array([[[[1., 1., 2., 0.],
                           [1., 0., 2., 2.],
                           [3., 3., 0., 1.],
                           [0., 3., 1., 1.]]]], dtype=np.float32)
    value("max_pool2d(동점)", lambda L: L.nn.functional.max_pool2d(L.tensor(tied_img), 2))

    def pooled_tie(L):
        x = L.tensor(tied_img.copy(), requires_grad=True)
        out = L.nn.functional.max_pool2d(x, 2)
        (out * (L.arange(out.numel()).reshape(out.shape).float() + 1)).sum().backward()
        return _grad_of(x, "max_pool2d(동점)")

    cases.append((EDGE_PREFIX + "grad::max_pool2d(동점)", pooled_tie))

    # ── the rounding rule ──
    # **torch ties .5 to even** — round(0.5)=0, round(1.5)=2, round(2.5)=2. The common
    # implementation (`floor(x+0.5)`) rounds every one up and parts quietly.
    value("round(.5에서)", lambda L: L.round(L.tensor(half)))
    value("floor(정수에서)", lambda L: L.floor(L.tensor(z)))
    value("ceil(정수에서)", lambda L: L.ceil(L.tensor(z)))
    value("trunc(음수)", lambda L: L.trunc(L.tensor(half)))
    value("frac(음수)", lambda L: L.frac(L.tensor(half)))

    # ── the remainder's sign ──
    # **torch's `%` follows the divisor's sign** — `-7 % 3` is 2, not -1. JS's `%` follows the
    # dividend's instead (-1), so using it as it stands parts on negatives alone. It never shows on
    # positive input, and both rules are called "the remainder".
    value("%(음수)", lambda L: L.tensor(neg) % 3.)
    value("%(음수로 나누기)", lambda L: L.tensor(neg) % -3.)

    return cases


def _no_duplicate_names(cases):
    """A duplicated name stops it. **Duplicated, one is quietly eaten.**

    `dump` is a list so it runs both and the later one overwrites the earlier's answer, while
    `export_json` is a dictionary so the case count drops by one — that mismatch is how it was first
    noticed (2724 against 2723). The runner counts "a name not in the golden answers" and cannot
    count "a name in the golden answers twice".

    The case that loses by being overwritten is **run by nobody.** Carelessly duplicating a name
    while writing one really happened (a value case and the value case `add` builds), and the table
    was green at the time.
    """
    seen, dup = set(), []
    for name, _ in cases:
        (dup.append(name) if name in seen else seen.add(name))
    if dup:
        raise AssertionError(
            "케이스 이름이 겹친다 — 겹치면 하나가 조용히 안 돌아간다:\n  "
            + "\n  ".join(sorted(set(dup))))
    return cases


def golden_cases(inp=None):
    """Everything the golden answers cover — values, gradients, training, dtypes, representations."""
    inp = golden_inputs() if inp is None else inp
    return _no_duplicate_names(
           wide_cases(inp) + grad_cases(inp) + train_cases(inp)
            + dtype_cases(inp) + repr_cases(inp) + error_cases(inp)
            + vision_cases(inp) + ops_cases(inp) + v2_cases(inp) + v2_functional_cases(inp) + dataset_cases(inp) + dataset_text_cases(inp) + dataset_last_three_cases(inp) + method_cases(inp) + math_cases(inp)
            + reduce_cases(inp) + shape_cases(inp) + inplace_cases(inp)
            + linalg_cases(inp) + linalg_struct_cases(inp) + linalg_name_cases(inp)
            + linalg_grad_cases(inp) + ndim_cases(inp) + flow_cases(inp)
            + container_cases(inp) + act_cases(inp) + norm_cases(inp)
            + pad_cases(inp) + loss_cases(inp) + lazy_cases(inp)
            + shuffle_cases(inp) + misc_cases(inp) + cell_cases(inp)
            + method_name_cases(inp) + default_convert_cases(inp)
            + data_loader_cases(inp)
            + unpool_cases(inp) + functional_name_cases(inp)
            + top_level_cases(inp)
            + opt_cases(inp) + dropout_cases(inp) + sdpa_cases(inp)
            + module_function_cases(inp) + pool_cases(inp)
            + new_function_cases(inp) + index_cases(inp) + numeric_cases(inp)
            + bit_cases(inp) + shape_index_cases(inp) + blend_cases(inp)
            + scalar_cache_cases(inp) + constant_cases(inp)
            + top_linalg_cases(inp) + stat_cases(inp)
            + make_cases(inp) + complex_cases(inp) + fft_cases(inp)
            + keepdim_cases(inp) + rnn_top_cases(inp) + top_rest_cases(inp)
            + webgpu_cases(inp) + edge_cases(inp))


_DTYPES = ["float32", "int64", "bool"]
_BIN_OPS = ["+", "-", "*", "/"]
_PY_SCALARS = [("파이썬 int", 2), ("파이썬 float", 2.0), ("파이썬 bool", True)]


def _dtype_tensor(L, name):
    """Picks a dtype **meaning the same thing** on both sides. bool is the only name that parts."""
    kind = getattr(L, "bool", None) if name == "bool" else getattr(L, name)
    if name == "bool":
        kind = getattr(L, "bool_", None) or L.bool
    return L.tensor([1, 0] if name == "bool" else [1, 2], dtype=kind)


def dtype_cases(inp=None):
    """dtype promotion — torch sorts by **category** (bool < integer < float) and promotes within it.

    Inherit numpy's rule and `float32 + int64` becomes float64, and the learner learns the wrong
    rule. The core covers this table with 112 cases, and here are **the 3 kinds with float64 taken
    out** — TF.js has no double precision, so they cannot exist.

    The result is the dtype's name as a string. It asks **which type comes out** rather than the
    value. A combination that is refused (boolean subtraction) has **the exception type** written
    down as the answer — refusing is specification too.
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

    # ── the ones that only move positions **keep the type** ──
    #
    # A defect survived for want of this table. Taking one sample out of the sister library's
    # dataset at an `int64` label gave `float32`, and **the value was right** so nothing in the
    # golden answers caught it — there was not one case asking about the type after a shape
    # operation.
    #
    # The boundary is **whether it produces a value.** The ones that only move positions (selecting,
    # slicing, concatenating, scattering) give back the original type, and the ones that compute
    # follow the promotion rules. Reductions (`sum`, `amax`, `cumsum`) keep the type in torch too and
    # are **not in this table** — a sum of `bool` is `int64`, which is one more rule, and that is a
    # place to measure separately.
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
        # **This line was dead for a long time.** `F` is a local helper inside five other
        # functions, so here it raised a `NameError`, and `outcome` caught it and produced
        # `<NameError>`. The golden answers froze that value as the right one, and **all three go
        # through the same Python line so all three produced it and the table was green** — `pad`
        # was never once called.
        #
        # That comparing the three against each other did not catch it is because the case body is
        # **one and the same thing** for all three. Writing the borch.ts-side body separately in
        # TypeScript exposed it on the spot — the body becoming two is what this table is worth.
        ("pad", lambda L, t: L.nn.functional.pad(t, (1, 1))),
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
        # **Both integer and boolean are asked.** Ask one and a defect that "drops to float32"
        # can survive in the other alone.
        cases.append((f"dtype::자리만::{name}(int64)",
                      lambda L, f=fn: outcome(lambda: f(L, L.tensor(ints)))))
        cases.append((f"dtype::자리만::{name}(bool)",
                      lambda L, f=fn: outcome(lambda: f(L, L.tensor(flags)))))
    # **`topk` is asked on integers only.** On booleans torch refuses and **the exception type**
    # differs from ours (RuntimeError against TypeError), and that is a story about refusal wording
    # rather than type preservation. Mix two questions into one table and there is no reading which
    # of them is red.
    cases.append(("dtype::자리만::topk[0](int64)",
                  lambda L: outcome(lambda: L.tensor(ints).topk(2, 1)[0])))

    # ── the fourteen type-changing names — **what is absent has to refuse in the same words** ──
    #
    # A place asking **how the refusal comes out** rather than the value. Such places are not caught
    # by the three comparing each other — unasked, each says something different, and the learner
    # reads that as **something that differs per implementation.** It really was so: the core gave
    # `AttributeError: 'Tensor' object has no attribute 'half'` (indistinguishable from a typo) and
    # the binding gave "borch.ts 텐서에 `half` 이 없다". Those are the messages as they stood then;
    # both have since been rewritten, and the quote is kept as the record of what was measured.
    #
    # `half` and `bfloat16` are **lines actually typed** in the tutorials' mixed-precision section.
    # `.int()` was worse — both **silently gave int64.** torch gives int32, and when we do not have
    # that slot it is better to stop than to hand over a different slot instead.
    floats = np.array([1.5, -2.5, 3.0], dtype=np.float32)

    def casts(name, call):
        """A type that exists freezes **its name**, and one that does not freezes **a fragment of the refusal wording.**"""
        def run(L, f=call):
            try:
                return str(f(L).dtype)
            except Exception as exc:                            # noqa: BLE001
                text = str(exc)
                mark = "is not in the browser subset"
                return f"거절({mark})" if mark in text \
                    else f"거절(다른 문구: {text.splitlines()[0][:40]})"
        cases.append((f"dtype::형바꾸기::{name}", run))

    def we_refuse(name, call, group="형바꾸기"):
        """The refusal cases at the type-changing places. One `refusal_case` makes the verdict —
        two copies of the same rule leave one of them fixed, and this repository has stepped on that
        place several times.

        `group` is the name of the place. A type alias (`dtype=torch.int`) follows the same rule
        that **the core refuses too**, so the verdict is not written twice; only the name differs.
        """
        cases.append((f"dtype::{group}::{name}=우리는거절", refusal_case(call)))

    for name in ("float", "long", "bool", "cfloat"):
        casts(name, lambda L, n=name: getattr(L.tensor(floats), n)())
    casts("type_as", lambda L: L.tensor(floats).type_as(L.tensor(ints)))
    for name in ("half", "bfloat16", "chalf", "cdouble", "byte", "char",
                 "short", "int"):
        we_refuse(name, lambda L, n=name: getattr(L.tensor(floats), n)())

    # ── the three asked — **an input producing false was measured first** ──
    #
    # An always-true predicate is not being asked even when it has a case. Three of the four have an
    # answer in the type and the value and really produce false in torch. The fourth
    # (`is_contiguous`) parts because the browser has no views, so it is kept beside `inplace::`'s
    # view place.
    for label, call in (
        ("is_floating_point(float32)", lambda L: L.tensor(floats).is_floating_point()),
        ("is_floating_point(int64)", lambda L: L.tensor(ints).is_floating_point()),
        ("is_floating_point(bool)", lambda L: L.tensor(flags).is_floating_point()),
        ("is_signed(float32)", lambda L: L.tensor(floats).is_signed()),
        ("is_signed(int64)", lambda L: L.tensor(ints).is_signed()),
        ("is_signed(bool)", lambda L: L.tensor(flags).is_signed()),
        ("is_nonzero(0)", lambda L: L.tensor(np.zeros(1, dtype=np.float32)).is_nonzero()),
        ("is_nonzero(3)", lambda L: L.tensor(np.full(1, 3.0, dtype=np.float32)).is_nonzero()),
    ):
        cases.append((f"dtype::묻는것::{label}",
                      lambda L, f=call: str(f(L))))

    # With several it stops — the place that stops `if tensor:` quietly looking at the first element.
    def nonzero_many(L):
        try:
            L.tensor(floats).is_nonzero()
        except Exception as exc:                                # noqa: BLE001
            return "멈췄다" if "ambiguous" in str(exc) else f"다른 문구 <{exc}>"
        return "안 던졌다"

    cases.append(("dtype::묻는것::is_nonzero(여럿)은 멈춘다", nonzero_many))

    # ── the six that read as "rightly absent" by name and have an answer on a dense tensor ──
    #
    # Sparse, device and quantisation, all counted as refusals, and measuring showed torch simply
    # managing **fourteen** of thirty-three on a dense tensor. Six of those need no sparse or
    # quantisation machinery but have an answer — **"this tensor is dense" and "it is on the CPU".**
    #
    # Counting by name and freezing a refusal case would **pin down a defect that is not there**,
    # and later, when somebody implements that name, a green case turns red — the check becomes a
    # place that blocks the functionality.
    for label, call in (
        ("dense_dim", lambda L: L.tensor(floats).dense_dim()),
        ("sparse_dim", lambda L: L.tensor(floats).sparse_dim()),
        ("storage_offset", lambda L: L.tensor(floats).storage_offset()),
        ("get_device(CPU 는 -1)", lambda L: L.tensor(floats).get_device()),
    ):
        cases.append((f"dtype::조밀에도답::{label}", lambda L, f=call: str(f(L))))
    for label, call in (
        ("to_dense 는 항등", lambda L: L.tensor(floats).to_dense()),
        ("dequantize 는 항등", lambda L: L.tensor(floats).dequantize()),
    ):
        cases.append((f"dtype::조밀에도답::{label}", call))

    # ── an absent name stops in **the same words** across the three ──
    #
    # The other twenty-seven (sparse, storage, quantisation) really are absent. And the refusal
    # wording had parted — the core used Python's standard wording and the binding used
    # "borch.ts 텐서에 X 이 없다". Real torch uses the standard wording too, so it was matched to
    # that. The quoted message is as it stood then; it has since been rewritten.
    #
    # **The twenty-seven are not frozen one by one.** That would turn that many red when somebody
    # implements sparse. Three representatives are asked, to see **whether the fallback path says
    # the same thing.**
    def missing_name(L, name):
        """**If torch has the name, whatever it does is as expected.**

        It sorted at first on "torch manages it", and `coalesce` stops on a dense tensor with a
        `RuntimeError` and `int_repr` with a `NotImplementedError` — **the name exists and it merely
        does not work on that input.** What we are asking is "when a name torch has and we do not is
        asked, do our three say the same thing", so on torch's side only an `AttributeError` is
        unexpected.
        """
        real = getattr(L, "__name__", "") == "torch"
        try:
            getattr(L.tensor(floats), name)()
        except AttributeError as exc:
            if real:
                return "뜻밖에 torch 에도 없다"
            return ("기대대로" if "has no attribute" in str(exc)
                    else f"다른 문구 <{str(exc).splitlines()[0][:44]}>")
        except Exception as exc:                                # noqa: BLE001
            return "기대대로" if real else f"다른 종류 <{type(exc).__name__}>"
        return "기대대로" if real else "뜻밖의 성공"

    # **The two torch removed in 1.9.** The names remain and calling them stops. Counted as "names
    # torch has" they look like something to implement, and implementing them makes **us more
    # lenient** — that code breaks on real torch.
    for gone in ("lstsq", "solve"):
        def deprecated(L, n=gone):
            try:
                getattr(L.tensor(floats.reshape(1, 3)), n)(L.tensor(floats.reshape(3, 1)))
            except Exception as exc:                            # noqa: BLE001
                return ("멈췄다" if "deprecated since version 1.9" in str(exc)
                        else f"다른 문구 <{str(exc).splitlines()[0][:44]}>")
            return "안 던졌다"
        cases.append((f"dtype::없는이름::{gone}(폐기됨)", deprecated))

    # **`coalesce`, `untyped_storage` and `int_repr` are no longer here.** All three were changed to
    # have the name and refuse with a reason — this check has to look at places where **the name
    # really is absent.** It uses the two `NOT_API` records as not being public API.
    for name in ("narrow_copy", "unsafe_chunk"):
        cases.append((f"dtype::없는이름::{name}",
                      lambda L, n=name: missing_name(L, n)))

    square2 = np.array([[1.5, 0.0], [0.0, -2.5]], dtype=np.float32)

    # ── the ones whose name exists and **does not fit this tensor** ──
    #
    # It gave `'Tensor' object has no attribute 'coalesce'` for a long time, and that is
    # **indistinguishable from a typo.** torch says "expected a sparse layout and got Strided" —
    # meaning the name exists and does not fit this tensor.
    #
    # It splits three ways: **the sparse accessors** blame the layout, and **the sparse constructors**
    # and **storage and quantisation** say the functionality is not here. torch refuses only the
    # first and manages the rest — the places we part are different ones.
    def refuses_with(L, name, fragment, arg=()):
        try:
            getattr(L.tensor(square2), name)(*arg)
        except Exception as exc:                                # noqa: BLE001
            return ("멈췄다" if fragment in str(exc)
                    else f"다른 문구 <{str(exc).splitlines()[0][:44]}>")
        return "안 던졌다"

    for name in ("coalesce", "indices", "values", "crow_indices", "row_indices"):
        cases.append((INPLACE_PREFIX + f"희소::{name} 는 배치를 탓한다",
                      lambda L, n=name: refuses_with(L, n, "but got Strided")))

    # Below are the places **torch manages.** They part because we have no sparse tensors and no
    # storage object, and that is not the layout's fault but **absent functionality.**
    for name in ("to_sparse", "to_sparse_csr", "sparse_mask", "untyped_storage"):
        def absent_here(L, n=name):
            """**torch manages it and we stop.** Asked by value it would part forever, so what is
            asked is "did each behave as its own documentation says" — written at first to hand back
            different strings on each side, which showed that the golden answers then froze torch's
            answer and left us red forever."""
            real = getattr(L, "__name__", "") == "torch"
            try:
                getattr(L.tensor(square2), n)(
                    *((L.tensor(square2).to_sparse(),) if n == "sparse_mask" else ()))
            except Exception as exc:                            # noqa: BLE001
                return "기대대로" if not real else f"뜻밖의 거절 <{type(exc).__name__}>"
            return "기대대로" if real else "뜻밖의 성공"
        cases.append((INPLACE_PREFIX + f"없는기능::{name}=우리는거절", absent_here))

    # `int_repr` and `cuda` **stop torch too** — a different branch from the four above. Kept in one
    # group, "does torch manage it" differs per name while the verdict is single, and it does not fit.
    for name in ("int_repr", "cuda"):
        def both_refuse(L, n=name):
            try:
                getattr(L.tensor(square2), n)()
            except Exception:                                   # noqa: BLE001
                return "멈췄다"
            return "안 던졌다"
        cases.append((INPLACE_PREFIX + f"없는기능::{name} 는 양쪽 다 멈춘다",
                      both_refuse))

    # `is_set_to` is **not always false** — a view is true and a copy is false, so this predicate is
    # also a name for asking whether `tensor()` copies.
    cases.append((INPLACE_PREFIX + "술어::is_set_to(자기)",
                  lambda L: str(L.tensor(square2).is_set_to(L.tensor(square2)))))

    def set_to_itself(L):
        x = L.tensor(square2)
        return str(x.is_set_to(x))

    cases.append((INPLACE_PREFIX + "술어::is_set_to(같은 객체)", set_to_itself))
    cases.append((INPLACE_PREFIX + "술어::is_shared()",
                  lambda L: str(L.tensor(square2).is_shared())))

    # `double` **parts deliberately** — the browser side has no double precision in a WebGPU
    # shader, so refusal is the answer there and `_as_expected` holds the place.
    #
    # **The sentence beside this used to say "the core has float64", and the core does not.**
    # `Tensor.__init__` narrows `float64` to `float32` at construction, deliberately: its own
    # comment calls having no such dtype "this library's first design decision". The claim was
    # true once and stopped being true when that narrowing went in, and nothing said so —
    # because `_as_expected` asks **"did it refuse?"** and not what came back. A silent
    # downcast is not a refusal, so the case passed and read as "the core has float64".
    we_refuse("double", lambda L: L.tensor(floats).double())

    # **Two more spellings of the same request**, so that the gate is asked where it sits
    # rather than only through `.double()`. All three arrive at `_cast`, and the reason it
    # is one gate and not three is that three copies of a rule leave one behind — a thing
    # this file has stepped on before.
    we_refuse(".to(float64)", lambda L: L.tensor(floats).to(L.float64))
    we_refuse(".type(float64)", lambda L: L.tensor(floats).type(L.float64))

    # ── a type **alias** is a type, not a function ──
    #
    # `torch.float`, `torch.double`, `torch.int` and `torch.bool` are dtypes. And those four are also
    # names of Tensor methods, so the loop that exposes methods as module functions was filling the
    # same names **with functions.** So `zeros(2, dtype=torch.float)`, common in the textbook,
    # stopped with `'function' object has no attribute 'np'` — the type it names was perfectly
    # present and only the name was covered.
    #
    # **Asking for the name alone does not catch it.** A check that looks only at existence passes
    # with a function sitting there (the coverage table really did). So **it is used** — a tensor is
    # built with that type and the resulting type's name is frozen. With a function sitting there it
    # stops on the spot.
    for _alias in ("float", "bool"):
        cases.append((f"dtype::별칭::dtype={_alias} 로 만든다",
                      lambda L, a=_alias: str(L.zeros(2, dtype=getattr(L, a)).dtype)))

    # **`torch.int` is int32** (`long` is int64). The integer slots were gathered into int64 alone,
    # so that type is absent — the name still has to **point at int32.** With no name at all,
    # `dtype=torch.int` stops with the same wording as a typo. Building one parts across all three
    # (torch manages it and our two refuse), so only **what it points at** is frozen here.
    cases.append(("dtype::별칭::int 은 int32 를 가리킨다",
                  lambda L: str(L.int)))
    # The constructing side. A place **the core refuses too**, so it is `we_refuse` rather than `_as_expected`.
    we_refuse("dtype=int", lambda L: L.zeros(2, dtype=L.int), group="별칭")
    # **This was `_as_expected` and it froze a silent downcast as the right answer.**
    # That helper asks only *did it raise*, and the core did not — it handed back a
    # `float32` where torch hands back a `float64`. Right by the question asked and
    # wrong by the question's name, which is the shape this repository keeps finding.
    #
    # `.double()`, `.to(float64)` and `.type(float64)` had already been changed to
    # refuse, in `Tensor._cast`, whose comment names this exact failure: *the request
    # was granted in name and answered in another cell.* `dtype=` was the fourth
    # spelling of that request and the only one still answered that way — measured
    # across thirty-seven factories. Now all four refuse, and the core refuses too,
    # so the verdict is `we_refuse` and not a browser-only divergence.
    we_refuse("dtype=double", lambda L: L.zeros(2, dtype=L.double), group="별칭")

    # ── does a factory function **actually use** `dtype=` ──
    #
    # It surfaced while pinning the aliases above: the binding's `zeros`, `ones`, `full`, `eye` and
    # `linspace` were discarding `**kw` outright. `zeros(2, dtype=int64)` gave float32, and **the
    # value being 0 it was right**, so a value comparison did not catch it. Because the golden
    # answers had not one case of the form `zeros(..., dtype=)` — what is not asked is not right.
    #
    # `ones` is asked alongside to see whether the five go through **one door.** Asking one alone
    # lets an edition that fixed only that one pass.
    cases.append(("dtype::별칭::zeros(dtype=int64)",
                  lambda L: str(L.zeros(2, dtype=L.int64).dtype)))
    cases.append(("dtype::별칭::ones(dtype=int64)",
                  lambda L: str(L.ones(2, dtype=L.int64).dtype)))
    # `requires_grad` was being discarded into the same `**kw` alongside.
    cases.append(("dtype::별칭::zeros(requires_grad=True)",
                  lambda L: str(L.zeros(2, requires_grad=True).requires_grad)))

    # ── do the fourteen factories listen to **the same two arguments** ──
    #
    # After pinning the above, an exhaustive measurement showed the answer differing per factory.
    # `zeros` listened to both, `zeros_like` **accepted `dtype=` and did not use it** (the value
    # right and the type wrong), and `rand` did not accept `requires_grad=` at all, so
    # `rand(3, requires_grad=True)` stopped. **With some working and some not, a learner cannot
    # form a rule.**
    #
    # The `requires_grad` side is worse. A wrong type is noticed eventually, and a leaf with no
    # gradient attached leaves **that parameter alone quietly unmoving while the loss comes down**
    # — with no error.
    _like = np.array([1.0, 2.0], dtype=np.float32)
    _factories = (
        ("zeros", lambda L, k: L.zeros(2, **k)),
        ("ones", lambda L, k: L.ones(2, **k)),
        ("empty", lambda L, k: L.empty(2, **k)),
        ("full", lambda L, k: L.full((2,), 3.0, **k)),
        ("eye", lambda L, k: L.eye(2, **k)),
        # **An integer tensor cannot take gradients** (torch refuses). `arange(4)` is int64, so it
        # has to be asked as a float — the value of asking type and gradient from one input.
        ("arange", lambda L, k: L.arange(4.0, **k)),
        ("linspace", lambda L, k: L.linspace(0.0, 1.0, 3, **k)),
        ("logspace", lambda L, k: L.logspace(0.0, 1.0, 3, **k)),
        ("scalar_tensor", lambda L, k: L.scalar_tensor(1.0, **k)),
        ("zeros_like", lambda L, k: L.zeros_like(L.tensor(_like), **k)),
        ("ones_like", lambda L, k: L.ones_like(L.tensor(_like), **k)),
        ("empty_like", lambda L, k: L.empty_like(L.tensor(_like), **k)),
        ("full_like", lambda L, k: L.full_like(L.tensor(_like), 2.0, **k)),
        # **`randn_like`'s type cannot be asked.** A normal draw cannot go into an integer slot
        # (torch refuses), so what is left is float64, and the browser side has no double precision
        # — a place where the three part, so it cannot be frozen by value. The gradient alone is asked.
        ("randn_like", lambda L, k: L.randn_like(L.tensor(_like), **k), None),
    )
    for _spec in _factories:
        _name, _call = _spec[0], _spec[1]
        _dt = _spec[2] if len(_spec) > 2 else "int64"
        if _dt is not None:
            cases.append((f"dtype::공장::{_name}(dtype={_dt})",
                          lambda L, f=_call, d=_dt: str(f(L, {"dtype": getattr(L, d)}).dtype)))
        cases.append((f"dtype::공장::{_name}(requires_grad=True)",
                      lambda L, f=_call: str(f(L, {"requires_grad": True}).requires_grad)))

    # **The shape is asked alongside.** Asking the type alone passes even with `eye(n, m)`'s second argument missing.
    cases.append(("dtype::공장::eye(2, 3) 은 직사각이다",
                  lambda L: str(tuple(L.eye(2, 3).shape))))

    # ── **the factories left outside** the fourteen above ──
    #
    # After gathering the fourteen behind one door, the same yardstick was held to the rest. **What
    # was fixed was a list and not a class of defect** — the shape of swallowing arguments into
    # `**kw` was still there in the five window functions and the four that build indices.
    #
    # The window functions are worse. If `hann_window(8, requires_grad=True)` quietly gives a leaf
    # with no gradient, code training it has no error and **that window alone does not move.**
    # Exactly the branch the comment above calls "worse".
    _rest = (
        ("hann_window", lambda L, k: L.hann_window(8, **k), None),
        ("hamming_window", lambda L, k: L.hamming_window(8, **k), None),
        ("blackman_window", lambda L, k: L.blackman_window(8, **k), None),
        ("bartlett_window", lambda L, k: L.bartlett_window(8, **k), None),
        ("kaiser_window", lambda L, k: L.kaiser_window(8, **k), None),
    )
    for _name, _call, _dt in _rest:
        cases.append((f"dtype::공장::{_name}(requires_grad=True)",
                      lambda L, f=_call: str(f(L, {"requires_grad": True}).requires_grad)))

    # The four that build indices. **The gradient cannot be asked** (they are integers and torch
    # refuses) — the type alone is asked. Swallow `dtype=` and it parts here.
    _int_makers = (
        ("randint", lambda L, k: L.randint(0, 5, (4,), **k)),
        ("randperm", lambda L, k: L.randperm(4, **k)),
        ("tril_indices", lambda L, k: L.tril_indices(3, 3, **k)),
        ("triu_indices", lambda L, k: L.triu_indices(3, 3, **k)),
    )
    for _name, _call in _int_makers:
        cases.append((f"dtype::공장::{_name}(dtype=int64)",
                      lambda L, f=_call: str(f(L, {"dtype": L.int64}).dtype)))

    # ── a list built with grep **has things in it that do not belong** ──
    #
    # Sweeping for functions carrying `**kw` produced four candidates, and reading torch's
    # signatures one by one left two. `empty_strided` already refuses (there is no such thing as a
    # stride), and `bernoulli` and `poisson` **do not take those two arguments in torch itself** —
    # there is nothing to swallow to begin with. Fixing things in a batch because the branch looks
    # the same is what this table prevents.
    #
    # `normal`'s **`dtype=` cannot be asked.** A normal draw cannot go into an integer slot and what
    # is left is float64, and the browser side has no double precision — the same situation as
    # `randn_like`.
    cases.append(("dtype::공장::normal(requires_grad=True)",
                  lambda L: str(L.normal(0.0, 1.0, (2,), requires_grad=True).requires_grad)))

    def _buf():
        # **It has to be a writable buffer** — torch warns on read-only `bytes`.
        return bytearray(np.array([1.5, -2.5, 3.0], dtype=np.float32).tobytes())

    cases.append(("dtype::공장::frombuffer(requires_grad=True)",
                  lambda L: str(L.frombuffer(_buf(), dtype=L.float32,
                                             requires_grad=True).requires_grad)))

    # **`hasattr` swallows the refusal.** `_AbsentDtype.np` is a door held open to stop on, and
    # `hasattr(dtype, "np")` eats that exception and answers false — and then it falls into the
    # numpy branch and numpy's `TypeError` comes out instead of the wording we designed. It does
    # stop, so the value is not wrong, and it is the place where **absent and a typo become the same
    # screen.** `_np_of` had written that very trap down in a comment, and only `frombuffer` was
    # outside the list.
    cases.append(("dtype::없는형::frombuffer(dtype=half)=우리는거절",
                  refusal_case(lambda L: L.frombuffer(_buf(), dtype=L.half))))

    # **Among the reductions, only `norm` was not listening to `dtype=`.** `sum`, `mean` and `prod`
    # listen, and with one not listening there might as well be no rule. torch **converts the type
    # first and then computes** — converting after computing has already lost precision and the
    # value differs.
    # The only type that can be asked is float64 — asked on the default (float32) the answers agree
    # and the case asks nothing. And the browser side has no double precision, so **the three part.**
    # Instead of the value, "did each behave as its own documentation says" is asked.
    # **And here too the core stopped succeeding**, for the reason written beside
    # `dtype=double` above: `norm(dtype=)` converts before computing, so the dtype
    # is named by the caller and a named request is not an accident. It answered in
    # `float32` and said nothing.
    we_refuse("x.norm(dtype=float64)",
              lambda L: L.tensor(np.float32([1.0, 2.0, 3.0])).norm(dtype=L.float64),
              group="공장")

    # ── `out=` — writing into a tensor made in advance ──
    #
    # **The convention was taken down from real torch** (not guessed). Five things are observed: it
    # writes into the destination and returns **the same object** · a differing shape is not an
    # error but **a resize** · the type follows `can_cast` (category only, precision free) · it
    # stops when **either the input or the destination requires gradients** · a reduction cannot
    # take it without `dim`.
    #
    # The saving does not happen for us — we compute and then move. **That the destination changes
    # and that what comes back is the same object are still facts**, so those two are kept.
    _o_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    _o_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    def _wrote(L, fn, shape=(3,), dt=None):
        """Looks at **whether the same object comes back** and **whether the destination changed**,
        together.

        The value is rounded — `exp` parts in the last digit between GPU and CPU, and what is asked
        here is not precision but whether `out=` wrote into the destination.
        """
        dst = L.zeros(shape) if dt is None else L.zeros(shape, dtype=getattr(L, dt))
        got = fn(L, dst)
        flat = np.asarray(dst.tolist(), dtype=np.float64).round(4).ravel().tolist()
        return f"{got is dst} {flat}"

    for _name, _fn in (
        ("add", lambda L, o: L.add(L.tensor(_o_a), L.tensor(_o_b), out=o)),
        ("mul", lambda L, o: L.mul(L.tensor(_o_a), L.tensor(_o_b), out=o)),
        ("exp", lambda L, o: L.exp(L.tensor(_o_a), out=o)),
        ("clamp", lambda L, o: L.clamp(L.tensor(_o_a), 1.5, out=o)),
        ("logspace", lambda L, o: L.logspace(0.0, 1.0, 3, out=o)),
    ):
        cases.append((f"dtype::out::{_name}(out=) 이 쓰고 같은 것을 준다",
                      lambda L, f=_fn: _wrote(L, f)))

    cases.append(("dtype::out::matmul(out=)",
                  lambda L: _wrote(L, lambda L2, o: L2.matmul(
                      L2.tensor(np.eye(3, dtype=np.float32)),
                      L2.tensor(np.eye(3, dtype=np.float32)), out=o), (3, 3))))
    # **The ones that produce indices need an integer destination too** — put into a float slot, the type rule blocks it.
    cases.append(("dtype::out::searchsorted(out=)",
                  lambda L: _wrote(L, lambda L2, o: L2.searchsorted(
                      L2.tensor(_o_a), L2.tensor(np.float32([2.5])), out=o),
                      (1,), "int64")))

    # **An integer result into a float slot is allowed** — the widening direction of the category.
    cases.append(("dtype::out::int 결과를 float 칸에",
                  lambda L: _wrote(L, lambda L2, o: L2.add(
                      L2.tensor(np.array([1, 2, 3])), L2.tensor(np.array([1, 1, 1])), out=o))))

    # Three refusals. Freezing the value would mean matching the wording too, so **the kind alone** is looked at.
    def _refused(L, fn):
        try:
            fn(L)
        except Exception as exc:                            # noqa: BLE001
            return type(exc).__name__
        return "안 멈췄다"

    cases.append(("dtype::out::형이 좁아지면 멈춘다",
                  lambda L: _refused(L, lambda L2: L2.add(
                      L2.tensor(_o_a), L2.tensor(_o_b),
                      out=L2.zeros(3, dtype=L2.int64)))))
    cases.append(("dtype::out::입력이 기울기를 요구하면 멈춘다",
                  lambda L: _refused(L, lambda L2: L2.add(
                      L2.tensor(_o_a, requires_grad=True), L2.tensor(_o_b),
                      out=L2.zeros(3)))))
    cases.append(("dtype::out::목적지가 기울기를 요구하면 멈춘다",
                  lambda L: _refused(L, lambda L2: L2.add(
                      L2.tensor(_o_a), L2.tensor(_o_b),
                      out=L2.zeros(3, requires_grad=True)))))

    # **A differing shape is resized** — not an error. The warning's wording is long, so only the shape is looked at.
    def _resized(L):
        dst = L.zeros(7)
        L.add(L.tensor(_o_a), L.tensor(_o_b), out=dst)
        return str(tuple(dst.shape))

    cases.append(("dtype::out::모양이 다르면 다시 잡는다", _resized))

    # ── the four types whose **name exists and slot does not** ──
    #
    # `torch.half`, `bfloat16`, `short` and `chalf` are dtypes, and on our side they were
    # **functions** derived from Tensor methods. `dtype=torch.half` stopped with
    # `'function' object has no attribute 'np'`, which is the same wording as a typo. The type
    # itself is absent, so it is not an alias but a name that **says what it points at and stops
    # when used.**
    for _name in ("half", "bfloat16", "short", "chalf"):
        cases.append((f"dtype::별칭::{_name} 이 가리키는 형",
                      lambda L, n=_name: str(getattr(L, n))))
        we_refuse(f"dtype={_name}",
                  lambda L, n=_name: L.zeros(2, dtype=getattr(L, n)), group="별칭")

    # ── an absent name has to be absent to `hasattr` too ──
    #
    # The reason `we_refuse` above could not sort by `hasattr` is itself **the user's defect.** If
    # the binding's module `__getattr__` hands out a function for any name, `hasattr(torch,
    # "compile")` is true and code branching on existence **goes down the absent path** — the error
    # comes much later, at the call site. What blocked the test apparatus blocks the user too.
    #
    # A place where the three part, so the value cannot be frozen as it is. Real torch true and our
    # two false is **the answer**, so it folds into "did each behave as its own documentation says"
    # — the same shape as `we_refuse`.
    def absent(name):
        def run(L, n=name):
            real = getattr(L, "__name__", "") == "torch"
            has = hasattr(L, n)
            want = real
            return "묻는 대로" if has == want else (
                f"뜻밖에 있다({n})" if has else f"뜻밖에 없다({n})")
        cases.append((f"dtype::없는이름::hasattr({name})", run))

    # Chosen as names torch has and our two do not (measured). `cuda` was taken out because we have
    # it too — asked with a name that exists, this check asks nothing.
    for name in ("compile", "vmap", "autocast", "jit", "sparse_coo_tensor"):
        absent(name)
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


# The failures a learner actually meets. **The same kind of exception** has to come out under the
# same conditions, and the message has to carry torch's canonical English phrase — that is what makes
# a search work.
#
# Ten of the core's twelve. The two missing (index out of range, in-place edit of a leaf) cannot be
# covered **for want of the functionality** rather than the message. Absent functionality dies loudly
# as an AttributeError.
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
    """T2 — does the same exception come out under the same conditions, with a searchable phrase."""

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
    """T3 — is `print(t)` the same as the real thing?

    What a learner does most often is print(tensor). Printed differently, the screen does not match
    the textbook's example, and they doubt what they did wrong every time. **The characters** are
    looked at, not the value.
    """

    def ns(L):
        names = ("tensor", "zeros", "ones", "arange", "relu", "sigmoid")
        return {n: getattr(L, n) for n in names if hasattr(L, n)}

    return [(f"repr::{name}",
             lambda L, e=expr: repr(eval(e, {"__builtins__": {}}, ns(L))))   # noqa: S307
            for name, expr in _REPR_CASES]


def to_numpy(t):
    """Any library's tensor into numpy.

    Real torch and borch both take `.detach().numpy()`. A GPU backend needs only those two names to
    fit and the harness need not be edited — it can read back inside `numpy()`.
    """
    return np.asarray(t.detach().numpy())


def manifest_hash(cases):
    """A hash of the case-name list. Stops a comparison against stale golden answers after the table changes."""
    h = hashlib.sha256()
    for name, _ in cases:
        h.update(name.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def input_fingerprints(inp):
    """A **per-key** fingerprint of the input arrays. It bites on dtype, shape and bytes alike.

    With one fingerprint over the whole thing, you know only "the inputs differ" and not **which**
    one. Then a person has to search from the beginning when it parts — which really happened once.
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
    """A fingerprint of all the inputs.

    numpy's `default_rng` promises the same numbers across versions. Without a check on that
    promise, breaking it leaves **different inputs quietly compared** — and then the harness stamps
    a pass having compared nothing.
    """
    h = hashlib.sha256()
    for key, digest in sorted(input_fingerprints(inp).items()):
        h.update(key.encode("utf-8"))
        h.update(digest.encode("utf-8"))
    return h.hexdigest()


OPS_PREFIX = "ops::"


def ops_cases(inp=None):
    """`borchvision.ops` — **box geometry, and only that.**

    Eleven of torchvision's thirty-nine; the other twenty-eight need a model's feature
    maps or predictions and there is no detector in the catalogue. These eleven need
    nothing but four numbers a box, so unlike most of the vision block **every one of
    them is deterministic** and the whole namespace can be frozen. That is unusual
    enough here to be worth saying: there is no distribution half to this one.

    The boxes are written out rather than drawn. Overlaps have to be **arranged** —
    random boxes in a large enough field mostly miss each other, and an IoU table of
    zeros passes against an implementation that computes the wrong thing.
    """
    del inp

    # Three that overlap in different amounts, one far away, and **a duplicate** —
    # the duplicate is what makes `nms` at a low threshold have something to do.
    _boxes = np.array([[0.0, 0.0, 10.0, 10.0],
                       [1.0, 1.0, 11.0, 11.0],
                       [5.0, 5.0, 15.0, 15.0],
                       [30.0, 30.0, 40.0, 40.0],
                       [0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    _others = np.array([[2.0, 2.0, 8.0, 8.0],
                        [12.0, 0.0, 22.0, 10.0],
                        [30.0, 31.0, 41.0, 39.0]], dtype=np.float32)
    _scores = np.array([0.9, 0.75, 0.6, 0.95, 0.5], dtype=np.float32)
    # Pairs for the losses, **chosen so the aspect-ratio term is not zero**. The flat
    # box against the tall one of equal area is the pair that separates the complete
    # loss from the distance loss; the last pair is identical to itself, which is the
    # only input for which every one of these losses must return exactly 0.
    _shapes = np.array([[0.0, 0.0, 10.0, 10.0],
                        [5.0, 5.0, 15.0, 15.0],
                        [0.0, 0.0, 20.0, 4.0],
                        [0.0, 0.0, 4.0, 20.0],
                        [1.0, 1.0, 3.0, 3.0]], dtype=np.float32)
    _targets = np.array([[1.0, 1.0, 11.0, 11.0],
                         [20.0, 20.0, 30.0, 30.0],
                         [0.0, 0.0, 4.0, 20.0],
                         [0.0, 0.0, 20.0, 4.0],
                         [1.0, 1.0, 3.0, 3.0]], dtype=np.float32)
    # Logits, not probabilities. ±40 is past where a naive sigmoid or a naive
    # `log(1 + exp(-x))` stops being finite in float32.
    _logits = np.array([[-1.0, 2.0, 0.0], [0.5, -3.0, 8.0], [-40.0, 40.0, -0.25]],
                       dtype=np.float32)
    _hits = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                     dtype=np.float32)
    _labels = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    # A mask with one blank plane — **the blank is the case**: torchvision answers
    # zeros rather than raising, and that is what lets a batch with one still stack.
    _masks = np.zeros((3, 6, 8), dtype=np.uint8)
    _masks[0, 1:4, 2:6] = 1
    _masks[1, 0:2, 0:1] = 1

    def op(call):
        """**Each side gets the boxes in its own kind.** torchvision's `batched_nms`
        and `masks_to_boxes` call `.numel()` on what they are given, so they need real
        tensors; ours take either. `to` is what makes the same case body ask both."""
        def run(L):
            O = _vision_ops(L)
            to = ((lambda a: L.tensor(np.ascontiguousarray(a))) if _is_real_torch(L)
                  else (lambda a: a))
            got = np.asarray(_as_numpy(call(O, to)), dtype=np.float32)
            # **`ascontiguousarray` turns a 0-d array into a 1-d one**, and that is not a
            # contiguity fix, it is a reshape. Every case here returned a table until the
            # losses arrived, so a reduced one was the first scalar to pass through and
            # the golden froze `[1]` for a value torchvision returns as `()`.
            #
            # Nothing on the Python side could see it — both libraries go through this
            # same wrapper, so both got `[1]` and agreed. It took borch.ts, which does
            # not, to disagree: *ours scalar, golden [1]*. **A defect in the harness reads
            # as a defect in whichever implementation did not go through it.**
            #
            # **`got.shape == ()` and not `got.ndim == 0`**, which says the same thing.
            # `test_binding_fills_in.py` parses *this file* for every attribute name a
            # case reaches for, and a numpy array's `.ndim` here is indistinguishable
            # from a tensor's — it reported the binding filling in for borch.ts on a
            # name no case asks about. That file's own docstring calls editing this one
            # "the fourth time a check changed what it greps out of it"; this is the
            # fifth, and `shape` is already asked everywhere, so it costs nothing.
            return L.tensor(got if got.shape == () else np.ascontiguousarray(got))
        return run

    cases = [
        (OPS_PREFIX + "box_area", op(lambda O, to: O.box_area(to(_boxes)))),
        # The same boxes read three ways. **`fmt` is a claim about four numbers that
        # look identical either way**, so a wrong one is a wrong answer with nothing
        # raised — and the round trip is what pins the pair of conversions together.
        (OPS_PREFIX + "box_convert(xyxy to xywh)",
         op(lambda O, to: O.box_convert(to(_boxes), "xyxy", "xywh"))),
        (OPS_PREFIX + "box_convert(xyxy to cxcywh)",
         op(lambda O, to: O.box_convert(to(_boxes), "xyxy", "cxcywh"))),
        (OPS_PREFIX + "box_convert(cxcywh back to xyxy)",
         op(lambda O, to: O.box_convert(O.box_convert(to(_boxes), "xyxy", "cxcywh"),
                                    "cxcywh", "xyxy"))),
        (OPS_PREFIX + "box_area(cxcywh)",
         op(lambda O, to: O.box_area(O.box_convert(to(_boxes), "xyxy", "cxcywh"), "cxcywh"))),
        # **N by M and not a paired list.** Five boxes against three gives fifteen
        # numbers, and an implementation that pairs them off returns three.
        (OPS_PREFIX + "box_iou", op(lambda O, to: O.box_iou(to(_boxes), to(_others)))),
        # The three penalised IoUs. They agree with plain IoU wherever the boxes
        # overlap and part where they do not, which is why `_others` has one box that
        # misses everything.
        (OPS_PREFIX + "generalized_box_iou",
         op(lambda O, to: O.generalized_box_iou(to(_boxes), to(_others)))),
        (OPS_PREFIX + "distance_box_iou",
         op(lambda O, to: O.distance_box_iou(to(_boxes), to(_others)))),
        (OPS_PREFIX + "complete_box_iou",
         op(lambda O, to: O.complete_box_iou(to(_boxes), to(_others)))),
        # **The losses are paired, where the IoUs above are N by M.** Five boxes against
        # five, one answer each — an implementation that reached for the matrix and took
        # its diagonal would agree here and cost `N²` to do it, so the shape is the case
        # as much as the values are.
        #
        # `_shapes` is deliberately not `_boxes`: the pairs there have **matched aspect
        # ratios**, and `complete_box_iou_loss`'s extra term is exactly zero whenever
        # they do. Measured on that pair alone it equals the distance loss to the last
        # bit, so the case would pass while asking nothing about the one thing that
        # function adds. On these it differs by 0.217.
        (OPS_PREFIX + "generalized_box_iou_loss",
         op(lambda O, to: O.generalized_box_iou_loss(to(_shapes), to(_targets)))),
        (OPS_PREFIX + "distance_box_iou_loss",
         op(lambda O, to: O.distance_box_iou_loss(to(_shapes), to(_targets)))),
        (OPS_PREFIX + "complete_box_iou_loss",
         op(lambda O, to: O.complete_box_iou_loss(to(_shapes), to(_targets)))),
        (OPS_PREFIX + "generalized_box_iou_loss(mean)",
         op(lambda O, to: O.generalized_box_iou_loss(to(_shapes), to(_targets),
                                                     reduction="mean"))),
        (OPS_PREFIX + "complete_box_iou_loss(sum)",
         op(lambda O, to: O.complete_box_iou_loss(to(_shapes), to(_targets),
                                                  reduction="sum"))),
        # Focal loss takes **logits**, so the values run either side of zero, and one
        # pair sits at ±40 — far enough out that a naive `1 / (1 + exp(-x))` reaches
        # numpy's overflow and a naive `log(1 + exp(-x))` returns `inf` where the loss
        # is finite. Both are written around that, and this is what asks.
        (OPS_PREFIX + "sigmoid_focal_loss",
         op(lambda O, to: O.sigmoid_focal_loss(to(_logits), to(_hits)))),
        # **`alpha=-1` is a switch, not a weight** — it turns the class balancing off
        # rather than weighting by a negative number.
        (OPS_PREFIX + "sigmoid_focal_loss(alpha off)",
         op(lambda O, to: O.sigmoid_focal_loss(to(_logits), to(_hits), -1))),
        # `gamma=0` removes the focusing term, leaving plain weighted cross-entropy —
        # the value this whole loss is a modulation of.
        (OPS_PREFIX + "sigmoid_focal_loss(gamma 0)",
         op(lambda O, to: O.sigmoid_focal_loss(to(_logits), to(_hits), 0.25, 0))),
        (OPS_PREFIX + "sigmoid_focal_loss(mean)",
         op(lambda O, to: O.sigmoid_focal_loss(to(_logits), to(_hits), 0.25, 2, "mean"))),
        (OPS_PREFIX + "clip_boxes_to_image",
         op(lambda O, to: O.clip_boxes_to_image(to(_boxes), (20, 25)))),
        (OPS_PREFIX + "remove_small_boxes",
         op(lambda O, to: O.remove_small_boxes(to(_boxes), 10.5))),
        # **`> threshold` and not `>=`.** At zero, boxes that merely touch both
        # survive; the duplicate does not. Both ends are asked.
        (OPS_PREFIX + "nms(nothing may overlap)",
         op(lambda O, to: O.nms(to(_boxes), to(_scores), 0.0))),
        (OPS_PREFIX + "nms(half)", op(lambda O, to: O.nms(to(_boxes), to(_scores), 0.5))),
        (OPS_PREFIX + "nms(everything survives)",
         op(lambda O, to: O.nms(to(_boxes), to(_scores), 1.0))),
        # Per class, by moving each class out of the others' reach.
        #
        # **This row does not test the offset**, and the comment that used to sit here
        # said it did — *the duplicate is in the same class as the box it duplicates, so
        # the offset trick has something to prove.* The premise is true and the
        # conclusion does not follow: a duplicate is suppressed whether or not the
        # classes were separated first. Measured by deleting the offset from an
        # implementation — this case stayed green.
        #
        # What the offset changes is a pair in **different** classes that overlap enough
        # to suppress each other, and with these labels the largest such overlap is 0.22
        # against a threshold of 0.5. So it is kept, because a frozen value is worth
        # keeping, and the row below is what actually asks the question.
        (OPS_PREFIX + "batched_nms",
         op(lambda O, to: O.batched_nms(to(_boxes), to(_scores), to(_labels), 0.5))),
        # `_labels` again with the second box moved to the other class. Now box 1 is
        # class 1 while box 0 — which overlaps it at 0.68 — is class 0, so a single pass
        # over the unshifted boxes drops box 1 and the per-class answer keeps it. The
        # two spellings differ in **length**, not only in content.
        (OPS_PREFIX + "batched_nms(classes that would suppress each other)",
         op(lambda O, to: O.batched_nms(to(_boxes), to(_scores),
                                        to(np.array([0, 1, 1, 1, 0], dtype=np.int64)),
                                        0.5))),
        (OPS_PREFIX + "masks_to_boxes", op(lambda O, to: O.masks_to_boxes(to(_masks)))),
    ]
    def _fill(L, module):
        """**The same numbers on both sides.** The two libraries initialise weights
        from different generators, so a layer compared as built compares two draws;
        every parameter and buffer is written here from one arange instead.

        `running_var` is made positive because a variance is, and a negative one takes
        `rsqrt` into `nan` — which would be a case that fails for a reason about the
        fixture rather than about the layer.
        """
        seen = dict(module.named_parameters())
        seen.update(dict(module.named_buffers()))
        for turn, name in enumerate(sorted(seen)):
            target = seen[name]
            if "num_batches" in name:
                continue
            shape = tuple(target.shape)
            block = ((np.arange(int(np.prod(shape)) or 1, dtype=np.float32)
                      * 0.037 + turn * 0.11) % 1.7 - 0.6).reshape(shape)
            if "running_var" in name:
                # **One channel's variance is very nearly zero**, which is the only
                # place epsilon shows. With every variance comfortably positive,
                # `(v + eps).rsqrt()` and `v.rsqrt()` agree to five decimals and a
                # reader that added epsilon afterwards passes — measured: that mutation
                # survived until this line was written.
                block = np.abs(block) + 0.5
                block.reshape(-1)[0] = 1e-9
            if hasattr(target, "copy_"):
                with L.no_grad():
                    target.copy_(L.tensor(np.ascontiguousarray(block)))
            else:                       # a buffer, handed back as a plain array
                np.asarray(target)[...] = block
        return module

    def _layer_values(build, shape):
        """One layer's output, with the weights written rather than drawn."""
        def run(L):
            module = _fill(L, build(L))
            module.eval()
            x = ((np.arange(int(np.prod(shape)), dtype=np.float32) * 0.013)
                 % 1.9 - 0.9).reshape(shape)
            with L.no_grad():
                out = module(L.tensor(np.ascontiguousarray(x)))
            return L.tensor(np.ascontiguousarray(
                np.asarray(out).reshape(-1).astype(np.float32)))
        return run

    def _ops_of(L):
        if _is_real_torch(L):
            import torchvision.ops as real
            return real
        _vision(L)
        import sys as _sys
        return _sys.modules["borchvision"].ops

    # **These twenty-eight were declined for what they are for**, not for what they
    # need — *it needs a model to be a block of*, *a feature map comes from a model*.
    # Five of them are a convolution, a norm and an activation; linear layers and
    # dropout; an affine transform over four buffers. The ingredients were all in the
    # core the whole time.
    cases += [
        # **`padding=None` keeps the size**, and zero padding is a network that trains
        # on an input that shrinks by two per layer.
        (OPS_PREFIX + "Conv2dNormActivation(3→4, k3)",
         _layer_values(lambda L: _ops_of(L).Conv2dNormActivation(3, 4), (2, 3, 6, 7))),
        # **`bias=None` means "only when there is no norm"** — with the norm dropped
        # the convolution grows a bias, and the parameter list changes shape.
        (OPS_PREFIX + "Conv2dNormActivation(k5, dilation 2, no norm)",
         _layer_values(lambda L: _ops_of(L).Conv2dNormActivation(
             3, 4, 5, dilation=2, norm_layer=None), (2, 3, 9, 9))),
        # **The two activations are not the same one**: the outer squashes to (0, 1)
        # because its output is a gain.
        (OPS_PREFIX + "SqueezeExcitation(6, 2)",
         _layer_values(lambda L: _ops_of(L).SqueezeExcitation(6, 2), (2, 6, 5, 4))),
        # **The last layer gets no activation and still gets dropout.**
        (OPS_PREFIX + "MLP(4 → [6, 3])",
         _layer_values(lambda L: _ops_of(L).MLP(4, [6, 3]), (5, 4))),
        (OPS_PREFIX + "MLP(with a norm between)",
         _layer_values(lambda L: _ops_of(L).MLP(
             4, [6, 3], norm_layer=L.nn.BatchNorm1d), (5, 4))),
        # **Epsilon is added before the reciprocal square root.**
        (OPS_PREFIX + "FrozenBatchNorm2d(3)",
         _layer_values(lambda L: _ops_of(L).FrozenBatchNorm2d(3), (2, 3, 4, 5))),
        (OPS_PREFIX + "FrozenBatchNorm2d(3)=repr",
         lambda L: repr(_ops_of(L).FrozenBatchNorm2d(3))),
        (OPS_PREFIX + "Permute([0, 2, 3, 1])",
         _layer_values(lambda L: _ops_of(L).Permute([0, 2, 3, 1]), (2, 3, 4, 5))),
    ]

    # ── the structured dropouts ─────────────────────────────────────────────────
    #
    # **These six were declined for being a backbone's** — *structured dropout for
    # convolutional backbones*, *it drops whole residual blocks — it needs blocks*.
    # True of what they are for, and neither about whether they can exist.
    #
    # **Only the deterministic half can live here.** The two libraries draw from
    # different generators, so what is frozen is the settings where a dropout is the
    # identity; what they distribute is checked in `tests/test_ops_dropouts.py`, which
    # is also where the finding lives that torchvision does not drop `p` either.

    def _passthrough(name, shape, **kwargs):
        def run(L):
            x = ((np.arange(int(np.prod(shape)), dtype=np.float32) * 0.017)
                 % 1.7 - 0.8).reshape(shape)
            out = getattr(_ops_of(L), name)(
                L.tensor(np.ascontiguousarray(x)), **kwargs)
            return L.tensor(np.ascontiguousarray(
                np.asarray(out, dtype=np.float32).reshape(-1)))
        return run

    cases += [
        (OPS_PREFIX + "stochastic_depth(training=False)",
         _passthrough("stochastic_depth", (2, 3, 4, 5), p=0.5, mode="row",
                      training=False)),
        (OPS_PREFIX + "stochastic_depth(p=0)",
         _passthrough("stochastic_depth", (2, 3, 4, 5), p=0.0, mode="batch")),
        (OPS_PREFIX + "drop_block2d(training=False)",
         _passthrough("drop_block2d", (2, 3, 5, 5), p=0.3, block_size=3,
                      training=False)),
        (OPS_PREFIX + "drop_block3d(training=False)",
         _passthrough("drop_block3d", (2, 3, 5, 5, 5), p=0.3, block_size=3,
                      training=False)),
        (OPS_PREFIX + "DropBlock2d(eval)",
         _layer_values(lambda L: _ops_of(L).DropBlock2d(0.3, 3), (2, 3, 5, 5))),
        (OPS_PREFIX + "DropBlock2d(0.3, 3)=repr",
         lambda L: repr(_ops_of(L).DropBlock2d(0.3, 3))),
        (OPS_PREFIX + "DropBlock3d(0.2, 5, inplace)=repr",
         lambda L: repr(_ops_of(L).DropBlock3d(0.2, 5, True))),
        # **The mode prints unquoted** — torchvision's own repr, copied rather than
        # tidied, because the string is the answer.
        (OPS_PREFIX + "StochasticDepth(0.5, row)=repr",
         lambda L: repr(_ops_of(L).StochasticDepth(0.5, "row"))),
    ]

    # ── sampling a feature map at coordinates that are not integers ─────────────
    #
    # **Eleven names waited on one piece of arithmetic**, and their rows said *a feature
    # map comes from a model* — true of where the tensor came from, not of what the
    # function needs. Eight of the eleven are here; the other three sit on top of these
    # and are still to write.
    #
    # The boxes are arranged, and three of the five are the cases the boundary rules
    # turn on: **one that fits, one hanging off the top-left, one off the bottom-right,
    # one smaller than a cell, and one in the second image of the batch.**
    #
    # Written without the outside ones, this table passes against a reader that clamps
    # samples to the edge instead of zeroing them — which is what torchvision's *own
    # Python reference* does, and what the first version of `_bilinear_sample` did. The
    # half-outside box was exactly twice torchvision's answer.
    _roi_x = ((np.arange(2 * 3 * 7 * 9, dtype=np.float32).reshape(2, 3, 7, 9) % 17)
              * 0.3)
    _rois = np.array([[0.0, 1.0, 1.0, 6.0, 5.0],
                      [0.0, -3.0, -2.0, 4.0, 3.0],
                      [0.0, 6.0, 4.0, 20.0, 20.0],
                      [0.0, 2.0, 2.0, 2.4, 2.4],
                      [1.0, 0.0, 0.0, 8.0, 6.0]], dtype=np.float32)
    # Eight channels so that `pooled=2` leaves two out — the position-sensitive pair
    # needs `out * ph * pw`, and `out=1` would hide the channel ordering entirely.
    _ps_x = ((np.arange(1 * 8 * 7 * 9, dtype=np.float32).reshape(1, 8, 7, 9) % 13)
             * 0.4)
    _ps_rois = _rois[_rois[:, 0] == 0]

    def _roi_case(name, picture, boxes, size, **kwargs):
        def run(L):
            out = getattr(_ops_of(L), name)(
                L.tensor(np.ascontiguousarray(picture)),
                L.tensor(np.ascontiguousarray(boxes)), size, **kwargs)
            return L.tensor(np.ascontiguousarray(
                np.asarray(_as_numpy(out), dtype=np.float32).reshape(-1)))
        return run

    def _roi_module(name, args, kwargs, picture, boxes):
        def run(L):
            out = getattr(_ops_of(L), name)(*args, **kwargs)(
                L.tensor(np.ascontiguousarray(picture)),
                L.tensor(np.ascontiguousarray(boxes)))
            return L.tensor(np.ascontiguousarray(
                np.asarray(_as_numpy(out), dtype=np.float32).reshape(-1)))
        return run

    cases += [
        # **`aligned` moves every value**, so both settings are asked. The default is
        # `False` because that is what the released weights were trained with.
        (OPS_PREFIX + "roi_align(aligned=False)",
         _roi_case("roi_align", _roi_x, _rois, (3, 2))),
        (OPS_PREFIX + "roi_align(aligned=True)",
         _roi_case("roi_align", _roi_x, _rois, (3, 2), aligned=True)),
        # **`sampling_ratio=-1` is per box** — as many samples as the bin is wide —
        # while a fixed one over-samples the small boxes and under-samples the large.
        (OPS_PREFIX + "roi_align(sampling_ratio=2)",
         _roi_case("roi_align", _roi_x, _rois, (3, 2), sampling_ratio=2)),
        (OPS_PREFIX + "roi_align(sampling_ratio=1)",
         _roi_case("roi_align", _roi_x, _rois, (3, 2), sampling_ratio=1)),
        # **`spatial_scale` is how much smaller the map is than the image.**
        (OPS_PREFIX + "roi_align(spatial_scale=0.5)",
         _roi_case("roi_align", _roi_x, _rois, (3, 2), spatial_scale=0.5)),
        (OPS_PREFIX + "roi_align(output_size=1)",
         _roi_case("roi_align", _roi_x, _rois, 1)),
        # **The rounding is C's, not Python's.** `spatial_scale=0.5` puts every odd
        # coordinate on a half, where Python rounds to even and C rounds away from zero.
        (OPS_PREFIX + "roi_pool", _roi_case("roi_pool", _roi_x, _rois, (3, 2))),
        (OPS_PREFIX + "roi_pool(spatial_scale=0.5, halves)",
         _roi_case("roi_pool", _roi_x, _rois, (3, 2), spatial_scale=0.5)),
        # **The output channel is the slow axis and the bin is the fast one.** The other
        # order gives the right shape from the wrong channels.
        (OPS_PREFIX + "ps_roi_align", _roi_case("ps_roi_align", _ps_x, _ps_rois, 2)),
        (OPS_PREFIX + "ps_roi_align(sampling_ratio=2)",
         _roi_case("ps_roi_align", _ps_x, _ps_rois, 2, sampling_ratio=2)),
        # **An average, not a maximum** — the one thing here that reads like a typo.
        (OPS_PREFIX + "ps_roi_pool", _roi_case("ps_roi_pool", _ps_x, _ps_rois, 2)),
        (OPS_PREFIX + "ps_roi_pool(spatial_scale=0.5)",
         _roi_case("ps_roi_pool", _ps_x, _ps_rois, 2, spatial_scale=0.5)),
        (OPS_PREFIX + "RoIAlign(aligned)",
         _roi_module("RoIAlign", (2, 1.0, 2), {"aligned": True}, _roi_x, _rois)),
        (OPS_PREFIX + "RoIAlign(2, 1.0, 2, aligned)=repr",
         lambda L: repr(_ops_of(L).RoIAlign(2, 1.0, 2, aligned=True))),
        (OPS_PREFIX + "RoIPool", _roi_module("RoIPool", (2, 1.0), {}, _roi_x, _rois)),
        (OPS_PREFIX + "RoIPool(2, 0.5)=repr",
         lambda L: repr(_ops_of(L).RoIPool(2, 0.5))),
        (OPS_PREFIX + "PSRoIAlign",
         _roi_module("PSRoIAlign", (2, 1.0, 2), {}, _ps_x, _ps_rois)),
        (OPS_PREFIX + "PSRoIAlign(2, 1.0, 2)=repr",
         lambda L: repr(_ops_of(L).PSRoIAlign(2, 1.0, 2))),
        (OPS_PREFIX + "PSRoIPool",
         _roi_module("PSRoIPool", (2, 1.0), {}, _ps_x, _ps_rois)),
        (OPS_PREFIX + "PSRoIPool(2, 1.0)=repr",
         lambda L: repr(_ops_of(L).PSRoIPool(2, 1.0))),
    ]

    # ── the two that sit on the sampler ─────────────────────────────────────────
    #
    # **The pyramid and the level-picker.** Their rows read *the detector's neck.
    # Nothing feeds it here* — which is about a catalogue rather than a library: a
    # pyramid is a handful of convolutions over a dict of tensors, and it does not meet
    # the backbone that filled the dict.
    #
    # The maps are **three different sizes and three different widths**, because the
    # 1×1 convolutions exist to make the widths agree and a fixture where they already
    # agree cannot see them do it.
    _fpn_shapes = ((1, 4, 16, 16), (1, 6, 8, 8), (1, 8, 4, 4))

    def _fpn_input(L, widths=None):
        import collections
        out = collections.OrderedDict()
        for i, shape in enumerate(_fpn_shapes):
            shape = shape if widths is None else (1, widths) + shape[2:]
            block = ((np.arange(int(np.prod(shape))).reshape(shape) % 11)
                     * 0.2).astype(np.float32)
            out[f"feat{i}"] = L.tensor(np.ascontiguousarray(block))
        return out

    def fpn_case():
        """`FeaturePyramidNetwork`, weights written rather than drawn.

        **The maps arrive coarsest last** and the addition runs top down, so reversing
        them adds the fine into the coarse — which runs, learns, and is the opposite of
        the paper. The three sizes here are 16, 8 and 4, so a reader that took a fixed
        factor of two instead of the lateral's own size would agree on this fixture and
        not on an odd one; that half is a pytest.
        """
        def run(L):
            model = _fill(L, _ops_of(L).FeaturePyramidNetwork([4, 6, 8], 5))
            model.eval()
            with L.no_grad():
                out = model(_fpn_input(L))
            parts = [np.asarray([len(out)], dtype=np.float32)]
            for _name, value in out.items():
                parts.append(np.asarray(_as_numpy(value)).reshape(-1)
                             .astype(np.float32))
            return L.tensor(np.ascontiguousarray(np.concatenate(parts)))
        return run

    def multiscale_case(sampling_ratio):
        """`MultiScaleRoIAlign`. **The level comes from the box's size**, not from the
        map: a 224-pixel box goes to level 4 and moves one level per doubling.

        The three boxes here are 10, 200 and 56 pixels a side against a 64-pixel image,
        which land on three different levels — a fixture whose boxes were all one size
        would pass against a reader that always took the first map.
        """
        def run(L):
            model = _ops_of(L).MultiScaleRoIAlign(
                ["feat0", "feat1", "feat2"], 3, sampling_ratio)
            boxes = np.array([[0.0, 0.0, 10.0, 10.0],
                              [0.0, 0.0, 200.0, 200.0],
                              [4.0, 4.0, 60.0, 60.0]], dtype=np.float32)
            with L.no_grad():
                out = model(_fpn_input(L, widths=5),
                            [L.tensor(np.ascontiguousarray(boxes))], [(64, 64)])
            return L.tensor(np.ascontiguousarray(
                np.asarray(_as_numpy(out)).reshape(-1).astype(np.float32)))
        return run

    cases += [
        (OPS_PREFIX + "FeaturePyramidNetwork(three widths, three sizes)", fpn_case()),
        (OPS_PREFIX + "MultiScaleRoIAlign(a level per box)", multiscale_case(2)),
        (OPS_PREFIX + "MultiScaleRoIAlign(sampling_ratio=-1)", multiscale_case(-1)),
        (OPS_PREFIX + "MultiScaleRoIAlign(names, 3, 2)=repr",
         lambda L: repr(_ops_of(L).MultiScaleRoIAlign(["feat0", "feat1"], 3, 2))),
    ]

    # ── the convolution whose sampling positions are learned ────────────────────
    #
    # **The last of the eleven**, and the only one that needed something the RoI work
    # did not build: the coordinates come from a tensor the network produced, so the
    # gradient runs back through the *positions* as well as the values.
    #
    # The offsets here are large enough — over a pixel — that most of them leave the
    # kernel's own footprint, and some leave the map. **Small offsets hide both edge
    # rules**: a displacement of a tenth of a pixel never crosses a border, and the
    # whole thing then agrees with an implementation that clamps.
    def _deform_input(seed, batch, in_c, out_c, kh, kw, h, w, wgroups, ogroups,
                      use_mask, stride, padding, dilation):
        """**Computed, not drawn.** `default_rng` is numpy's and TypeScript has none of
        it, so a fixture built from it can be asked of one implementation and not of
        the other — which is the whole of why these eight cases were never ported.

        `_minstd` gives both sides the same bytes from the same three numbers. What was
        lost is that the values were normal; what the comment above actually asks for is
        their **size**, and that is checked below rather than assumed.
        """
        out_h = (h + 2 * padding[0] - (dilation[0] * (kh - 1) + 1)) // stride[0] + 1
        out_w = (w + 2 * padding[1] - (dilation[1] * (kw - 1) + 1)) // stride[1] + 1
        offset = _minstd((batch, ogroups * 2 * kh * kw, out_h, out_w), 2.4, seed + 2)
        # **The claim the comment makes, asked rather than trusted.** Small offsets hide
        # both edge rules: a displacement of a tenth of a pixel never crosses a border,
        # and the whole thing then agrees with an implementation that clamps.
        assert (np.abs(offset) > 1.0).mean() > 0.4, "the offsets stay inside the kernel"
        return {
            "input": _minstd((batch, in_c, h, w), 2.0, seed),
            "weight": _minstd((out_c, in_c // wgroups, kh, kw), 0.6, seed + 1),
            "offset": offset,
            "mask": (_minstd((batch, ogroups * kh * kw, out_h, out_w), 0.5, seed + 3)
                     + 0.5 if use_mask else None),
            "bias": _minstd((out_c,), 2.0, seed + 4),
        }

    def deform_case(label, **shape):
        given = _deform_input(**shape)
        settings = {k: shape[k] for k in ("stride", "padding", "dilation")}
        del label

        def run(L):
            out = _ops_of(L).deform_conv2d(
                L.tensor(np.ascontiguousarray(given["input"])),
                L.tensor(np.ascontiguousarray(given["offset"])),
                L.tensor(np.ascontiguousarray(given["weight"])),
                L.tensor(np.ascontiguousarray(given["bias"])),
                mask=(None if given["mask"] is None
                      else L.tensor(np.ascontiguousarray(given["mask"]))),
                **settings)
            return L.tensor(np.ascontiguousarray(
                np.asarray(_as_numpy(out), dtype=np.float32).reshape(-1)))
        return run

    def deform_module_case():
        """The module, with the weight written rather than drawn — and **the offsets
        still come from outside.** That is the shape of the layer: it holds a weight and
        takes a displacement field, because the field is produced by another convolution
        the caller writes."""
        given = _deform_input(seed=11, batch=2, in_c=4, out_c=6, kh=3, kw=3, h=6, w=6,
                              wgroups=1, ogroups=1, use_mask=False, stride=(1, 1),
                              padding=(1, 1), dilation=(1, 1))

        def run(L):
            layer = _fill(L, _ops_of(L).DeformConv2d(4, 6, 3, padding=1))
            layer.eval()
            with L.no_grad():
                out = layer(L.tensor(np.ascontiguousarray(given["input"])),
                            L.tensor(np.ascontiguousarray(given["offset"])))
            return L.tensor(np.ascontiguousarray(
                np.asarray(_as_numpy(out), dtype=np.float32).reshape(-1)))
        return run

    _DEFORM = dict(seed=3, batch=2, in_c=4, out_c=6, kh=3, kw=3, h=7, w=8,
                   wgroups=1, ogroups=1, use_mask=False,
                   stride=(1, 1), padding=(0, 0), dilation=(1, 1))

    cases += [
        (OPS_PREFIX + "deform_conv2d", deform_case("plain", **_DEFORM)),
        # **v2 adds a learned weight per position**, not only a learned place.
        (OPS_PREFIX + "deform_conv2d(mask, v2)",
         deform_case("mask", **{**_DEFORM, "use_mask": True})),
        # **Two kinds of group, and they may differ** — one offset field can steer
        # several groups of filters. The mask is indexed by the offset group too, which
        # a fixture with one group cannot show.
        (OPS_PREFIX + "deform_conv2d(weight groups)",
         deform_case("wg", **{**_DEFORM, "wgroups": 2})),
        (OPS_PREFIX + "deform_conv2d(offset groups)",
         deform_case("og", **{**_DEFORM, "ogroups": 2})),
        (OPS_PREFIX + "deform_conv2d(both groups, mask)",
         deform_case("both", **{**_DEFORM, "wgroups": 2, "ogroups": 2,
                                "use_mask": True})),
        # **The kernel's dilation is applied before the offset**, so a zero displacement
        # leaves an ordinary dilated convolution.
        (OPS_PREFIX + "deform_conv2d(stride, padding, dilation)",
         deform_case("spd", **{**_DEFORM, "h": 9, "w": 9, "stride": (2, 2),
                               "padding": (1, 1), "dilation": (2, 2)})),
        (OPS_PREFIX + "deform_conv2d(1x1 kernel, mask)",
         deform_case("1x1", **{**_DEFORM, "kh": 1, "kw": 1, "h": 5, "w": 5,
                               "use_mask": True})),
        (OPS_PREFIX + "deform_conv2d(non-square kernel)",
         deform_case("3x1", **{**_DEFORM, "kw": 1, "h": 6, "w": 6})),
        (OPS_PREFIX + "DeformConv2d(4, 6, 3, padding=1)", deform_module_case()),
        (OPS_PREFIX + "DeformConv2d(groups, no bias)=repr",
         lambda L: repr(_ops_of(L).DeformConv2d(4, 6, 3, groups=2, bias=False,
                                                dilation=2))),
        (OPS_PREFIX + "DeformConv2d(4, 6, 3, padding=1)=repr",
         lambda L: repr(_ops_of(L).DeformConv2d(4, 6, 3, padding=1))),
    ]

    return cases


V2_PREFIX = "v2::"


def _v2_named(x):
    """`Lambda`'s repr prints the function's name, so it has to have one. A lambda
    would print `<lambda>` on both sides and prove nothing about the name being read."""
    return x


def v2_cases(inp=None):
    """`transforms.v2` — **fifty-two reprs and the arithmetic underneath them.**

    v2's transforms take the same arguments and compute the same numbers as v1's; what
    differs is what they print. That is not a cosmetic difference here, because a
    tutorial's `print(transform)` is how a reader checks that the thing they built is
    the thing they meant. So the repr is the larger half of this block: fifty-two of
    them, one per name, and they are frozen as strings against real torchvision's.

    The reprs are worth freezing precisely because they came out **wrong four times**
    before they came out right — `Resize(5)` stores `[5]` and not `5`, `RandomErasing`
    turns its `value` into a list, `RandomChoice` fills in uniform probabilities that
    were never passed, and `ElasticTransform`'s printed fill cannot be read back off
    the finished object. Every one of those was found by comparing, not by reading.

    The value half is smaller and asks two things. That the v2 names inherit the v1
    arithmetic rather than a copy of it — three transforms are run through the v2
    spelling and have to land on the v1 answer — and that the twelve v2 adds compute
    what torchvision computes. Of those twelve, six draw and so are pinned to the
    settings where they do not: `p=0`, or `sigma=0`, or an argument that leaves the
    picture alone. Whether the drawing itself is right is pytest's question.

    `MixUp` and `CutMix` are here as reprs only. Their weight is drawn from a Beta
    distribution and there is no argument that pins it — an `alpha` large enough to
    concentrate it still does not fix it — so their values are pytest's, where the
    properties that matter (the pairing is a roll by one, the label follows the area
    actually pasted) can be asked without a shared generator.
    """
    inp = golden_inputs() if inp is None else inp

    def v2_repr(build):
        """Both sides get their own `v2` module and their own dtypes; what is compared
        is the string that comes back."""
        return lambda L: repr(build(_vision_v2(L), L))

    # Fifty recipes. The arguments are chosen to make the repr **say something** —
    # a default-everything constructor prints the same text no matter what the
    # constructor did with what it was given.
    _reprs = (
        ("Resize", lambda m, L: m.Resize((4, 3))),
        ("Resize(one number)", lambda m, L: m.Resize(5)),
        ("CenterCrop", lambda m, L: m.CenterCrop(4)),
        ("RandomCrop", lambda m, L: m.RandomCrop(4)),
        ("RandomResizedCrop", lambda m, L: m.RandomResizedCrop(4)),
        ("FiveCrop", lambda m, L: m.FiveCrop(3)),
        ("TenCrop", lambda m, L: m.TenCrop(3)),
        ("Pad", lambda m, L: m.Pad(2)),
        ("RandomHorizontalFlip", lambda m, L: m.RandomHorizontalFlip()),
        ("RandomVerticalFlip", lambda m, L: m.RandomVerticalFlip()),
        ("Grayscale", lambda m, L: m.Grayscale(3)),
        ("RandomGrayscale", lambda m, L: m.RandomGrayscale()),
        ("Normalize", lambda m, L: m.Normalize([0.5], [0.5])),
        ("RandomErasing", lambda m, L: m.RandomErasing()),
        ("ColorJitter", lambda m, L: m.ColorJitter(0.5)),
        ("ColorJitter(all four)", lambda m, L: m.ColorJitter(0.5, 0.3, 0.2, 0.1)),
        ("RandomInvert", lambda m, L: m.RandomInvert()),
        ("RandomPosterize", lambda m, L: m.RandomPosterize(4)),
        ("RandomSolarize", lambda m, L: m.RandomSolarize(0.5)),
        ("RandomAutocontrast", lambda m, L: m.RandomAutocontrast()),
        ("RandomEqualize", lambda m, L: m.RandomEqualize()),
        ("RandomAdjustSharpness", lambda m, L: m.RandomAdjustSharpness(2)),
        ("RandomRotation", lambda m, L: m.RandomRotation(30)),
        ("RandomAffine", lambda m, L: m.RandomAffine(30)),
        ("RandomPerspective", lambda m, L: m.RandomPerspective()),
        ("ElasticTransform", lambda m, L: m.ElasticTransform()),
        ("GaussianBlur", lambda m, L: m.GaussianBlur(3)),
        ("AutoAugment", lambda m, L: m.AutoAugment()),
        ("RandAugment", lambda m, L: m.RandAugment()),
        ("TrivialAugmentWide", lambda m, L: m.TrivialAugmentWide()),
        ("AugMix", lambda m, L: m.AugMix()),
        ("RandomOrder", lambda m, L: m.RandomOrder([m.Identity(), m.RGB()])),
        ("RandomChoice", lambda m, L: m.RandomChoice([m.Identity(), m.RGB()])),
        # One child and two children print differently — torch's `nn.Module` puts one
        # on the same line and breaks two across lines. Both spellings are asked.
        ("Compose(one)", lambda m, L: m.Compose([m.Identity()])),
        ("Compose(two)", lambda m, L: m.Compose([m.Identity(), m.RGB()])),
        ("RandomApply", lambda m, L: m.RandomApply([m.Identity()], p=0.3)),
        ("Lambda", lambda m, L: m.Lambda(_v2_named, int, float)),
        ("Identity", lambda m, L: m.Identity()),
        ("RGB", lambda m, L: m.RGB()),
        ("ToImage", lambda m, L: m.ToImage()),
        ("ToPureTensor", lambda m, L: m.ToPureTensor()),
        ("ToDtype", lambda m, L: m.ToDtype(L.float32, scale=True)),
        ("GaussianNoise", lambda m, L: m.GaussianNoise()),
        ("GaussianNoise(three arguments)", lambda m, L: m.GaussianNoise(0.1, 0.5, False)),
        ("RandomChannelPermutation", lambda m, L: m.RandomChannelPermutation()),
        ("RandomPhotometricDistort", lambda m, L: m.RandomPhotometricDistort()),
        ("RandomResize", lambda m, L: m.RandomResize(8, 16)),
        ("RandomShortestSize", lambda m, L: m.RandomShortestSize(8, 20)),
        ("RandomZoomOut", lambda m, L: m.RandomZoomOut()),
        ("ScaleJitter", lambda m, L: m.ScaleJitter((8, 8))),
        ("MixUp", lambda m, L: m.MixUp(num_classes=4)),
        ("CutMix", lambda m, L: m.CutMix(alpha=0.5, num_classes=3)),
    )

    cases = [(V2_PREFIX + "repr " + name, v2_repr(build)) for name, build in _reprs]

    img_f = inp["vis_f"]
    gray = np.ascontiguousarray(img_f[:, :, :1])
    img_u8 = inp["vis_u8"]

    def on(picture, build):
        """The same picture in each side's own format, out as a float tensor.

        Ours takes `(H,W,C)` arrays and torchvision's v2 takes `(C,H,W)` tensors —
        handing both the same object would ask two different questions.
        """
        def run(L):
            m = _vision_v2(L)
            if _is_real_torch(L):
                given = L.tensor(np.ascontiguousarray(picture.transpose(2, 0, 1)))
                out = _as_numpy(build(m, L)(given).detach())
                # `len(out.shape)` and not `out.ndim`: `test_binding_fills_in` parses
                # this file for attribute references and cannot tell a numpy array's
                # `.ndim` from a tensor's, so writing it here reports a name the
                # golden cases ask about when no case asks about it.
                out = out.transpose(1, 2, 0) if len(out.shape) == 3 else out
            else:
                out = _as_numpy(build(m, L)(picture))
            return L.tensor(np.ascontiguousarray(np.asarray(out, dtype=np.float32)))
        return run

    cases += [
        # The three that are v1's arithmetic reached through a v2 name. If the twin
        # subclass ever grows a body of its own these stop matching v1's own cases.
        (V2_PREFIX + "Resize(inherited)", on(img_f, lambda m, L: m.Resize((4, 3)))),
        (V2_PREFIX + "CenterCrop(inherited)", on(img_f, lambda m, L: m.CenterCrop(4))),
        (V2_PREFIX + "Pad(inherited)", on(img_f, lambda m, L: m.Pad(2))),

        (V2_PREFIX + "Identity", on(img_f, lambda m, L: m.Identity())),
        (V2_PREFIX + "ToPureTensor", on(img_f, lambda m, L: m.ToPureTensor())),
        # `RGB` on a colour picture is the identity; on a grey one it is the case.
        (V2_PREFIX + "RGB(three channels)", on(img_f, lambda m, L: m.RGB())),
        (V2_PREFIX + "RGB(one channel)", on(gray, lambda m, L: m.RGB())),

        # `ToImage` moves the axes and **does not scale**; `ToDtype(scale=True)` is the
        # other half. The pair is what v2 tells you to write instead of `ToTensor`, so
        # the pair is asked as well as each part.
        (V2_PREFIX + "ToImage", lambda L: _v2_from_picture(L, img_u8,
                                                           lambda m, L2: m.ToImage())),
        (V2_PREFIX + "ToDtype(scaling)", on(img_u8, lambda m, L: m.ToDtype(L.float32, scale=True))),
        (V2_PREFIX + "ToDtype(not scaling)", on(img_u8, lambda m, L: m.ToDtype(L.float32))),
        (V2_PREFIX + "ToImage then ToDtype", lambda L: _v2_from_picture(
            L, img_u8, lambda m, L2: m.Compose([m.ToImage(),
                                                m.ToDtype(L2.float32, scale=True)]))),

        # Six that draw, pinned where they do not draw. `sigma=0` leaves the mean,
        # which is what makes the clip case a clip case rather than a noise case.
        (V2_PREFIX + "GaussianNoise(sigma=0)",
         on(img_f, lambda m, L: m.GaussianNoise(0.0, 0.0))),
        (V2_PREFIX + "GaussianNoise(clipping)",
         on(img_f, lambda m, L: m.GaussianNoise(5.0, 0.0, True))),
        (V2_PREFIX + "GaussianNoise(not clipping)",
         on(img_f, lambda m, L: m.GaussianNoise(5.0, 0.0, False))),
        (V2_PREFIX + "RandomZoomOut(p=0)", on(img_f, lambda m, L: m.RandomZoomOut(p=0.0))),
        (V2_PREFIX + "RandomPhotometricDistort(p=0)",
         on(img_f, lambda m, L: m.RandomPhotometricDistort(p=0.0))),
        # A one-wide range has one answer, so the draw is a draw with nothing to draw.
        (V2_PREFIX + "RandomResize(one size)", on(img_f, lambda m, L: m.RandomResize(4, 5))),
        (V2_PREFIX + "RandomShortestSize(one size)",
         on(img_f, lambda m, L: m.RandomShortestSize(4, 40))),
        (V2_PREFIX + "ScaleJitter(one factor)",
         on(img_f, lambda m, L: m.ScaleJitter((8, 8), (1.0, 1.0)))),
    ]

    # ── the type system, and the thirteen names that are it ─────────────────────
    #
    # **Thirteen of `v2`'s twenty absences were the type system rather than users of
    # it.** The rows read *boxes travelling with the picture — the point of v2's type
    # system*, and the point of a thing is not a reason it cannot exist.
    #
    # The boxes below are arranged, and three of the four are why: one sits inside the
    # canvas, one hangs off the top-left into negative coordinates, one off the
    # bottom-right past both edges, and one is degenerate. Clamping is the whole subject
    # here and a box that is already inside cannot show it.
    _BOXES = np.array([[1.0, 2.0, 5.0, 8.0],
                       [-3.0, -1.0, 4.0, 3.0],
                       [6.0, 4.0, 20.0, 20.0],
                       [2.0, 2.0, 2.0, 6.0]], dtype=np.float32)
    _CANVAS = (8, 10)

    def _tv(L):
        """`tv_tensors` for each side. **Not `_vision(L).tv_tensors`** — that helper
        hands back the `transforms` module, and the types live one level up beside it."""
        if _is_real_torch(L):
            import torchvision.tv_tensors as real
            return real
        _vision(L)
        import sys as _sys
        return _sys.modules["borchvision"].tv_tensors

    def _boxes(L, fmt="XYXY"):
        made = _tv(L)
        if _is_real_torch(L):
            return made.BoundingBoxes(L.tensor(np.ascontiguousarray(_BOXES)),
                                      format=fmt, canvas_size=_CANVAS)
        return made.BoundingBoxes(np.ascontiguousarray(_BOXES), format=fmt,
                                  canvas_size=_CANVAS)

    def _values(L, out):
        return L.tensor(np.ascontiguousarray(
            np.asarray(_as_numpy(out), dtype=np.float32).reshape(-1)))

    def convert_case(target):
        """**The label goes with the numbers.** A conversion that returned a plain
        tensor would leave the next transform reading the same four numbers under the
        old rule — which is the failure the whole type system exists to prevent, and it
        is invisible in the values."""
        def run(L):
            return _values(L, _vision_v2(L).functional.convert_bounding_box_format(
                _boxes(L), new_format=target))
        return run

    def clamp_case(mode, fmt="XYXY"):
        """**`soft` and `hard` are the same thing for a box that is not rotated**, which
        is every box here — torchvision says so in a comment, and both are asked so that
        a reader who invented a difference between them fails.

        `None` returns the boxes untouched, which the two boxes hanging off the canvas
        make visible and an inside-only fixture would not.
        """
        def run(L):
            given = _boxes(L, fmt)
            if fmt != "XYXY":
                given = _vision_v2(L).functional.convert_bounding_box_format(
                    _boxes(L), new_format=fmt)
            return _values(L, _vision_v2(L).functional.clamp_bounding_boxes(
                given, clamping_mode=mode))
        return run

    def transform_case(build):
        def run(L):
            return _values(L, build(_vision_v2(L))(_boxes(L)))
        return run

    def query_case(what):
        """`query_size` asks **the boxes' canvas, not their shape.** A `(4, 4)` box
        tensor is four boxes and not a four-by-four picture, which is why this cannot be
        `shape[-2:]` on everything — and the fixture has exactly four boxes so that a
        reader who did it that way agrees with the picture by accident."""
        def run(L):
            made = _tv(L)
            picture = (made.Image(L.tensor(np.zeros((3, 8, 10), np.float32)))
                       if _is_real_torch(L)
                       else made.Image(np.zeros((3, 8, 10), np.float32)))
            flat = [picture, _boxes(L)]
            answer = getattr(_vision_v2(L), what)(flat)
            return L.tensor(np.asarray(answer, dtype=np.float32))
        return run

    def nest_case(kind):
        """**The shape of the sample comes back**, and the parts this transform does not
        own come back untouched. A dict keeps its keys and their order; a tuple stays a
        tuple; a string in the middle passes through.

        Written against one tensor, every one of those is invisible.
        """
        def run(L):
            v2 = _vision_v2(L)
            made = _tv(L)

            class Double(v2.Transform):
                _transformed_types = (made.Image,)

                def transform(self, inpt, params):
                    return inpt * 2

            picture = (made.Image(L.tensor(np.ones((1, 2, 2), np.float32)))
                       if _is_real_torch(L)
                       else made.Image(np.ones((1, 2, 2), np.float32)))
            sample = ({"img": picture, "boxes": _boxes(L)} if kind == "dict"
                      else (picture, _boxes(L)))
            out = Double()(sample)
            picked = out["img"] if kind == "dict" else out[0]
            boxes = out["boxes"] if kind == "dict" else out[1]
            marker = 1.0 if isinstance(out, dict) == (kind == "dict") else 0.0
            kept = 1.0 if type(boxes).__name__ == "BoundingBoxes" else 0.0
            return L.tensor(np.ascontiguousarray(np.concatenate([
                np.asarray(_as_numpy(picked), dtype=np.float32).reshape(-1),
                np.asarray([marker, kept], dtype=np.float32)])))
        return run

    cases += [
        (V2_PREFIX + "convert_bounding_box_format(XYWH)", convert_case("XYWH")),
        (V2_PREFIX + "convert_bounding_box_format(CXCYWH)", convert_case("CXCYWH")),
        (V2_PREFIX + "clamp_bounding_boxes(soft)", clamp_case("soft")),
        (V2_PREFIX + "clamp_bounding_boxes(hard, the same for these)",
         clamp_case("hard")),
        (V2_PREFIX + "clamp_bounding_boxes(None, untouched)", clamp_case(None)),
        # **The clamp goes through XYXY and comes back**, so a format that is not the
        # corners is where a one-way conversion shows.
        (V2_PREFIX + "clamp_bounding_boxes(XYWH)", clamp_case("soft", "XYWH")),
        (V2_PREFIX + "clamp_bounding_boxes(CXCYWH)", clamp_case("soft", "CXCYWH")),
        (V2_PREFIX + "ClampBoundingBoxes",
         transform_case(lambda m: m.ClampBoundingBoxes())),
        (V2_PREFIX + "ConvertBoundingBoxFormat(XYWH)",
         transform_case(lambda m: m.ConvertBoundingBoxFormat("XYWH"))),
        (V2_PREFIX + "query_size(the boxes' canvas)", query_case("query_size")),
        (V2_PREFIX + "query_chw(images only)", query_case("query_chw")),
        (V2_PREFIX + "Transform(a dict comes back a dict)", nest_case("dict")),
        (V2_PREFIX + "Transform(a tuple comes back a tuple)", nest_case("tuple")),
    ]
    return cases


def _v2_from_picture(L, picture, build):
    """`ToImage` is asked of **an (H,W,C) byte picture on both sides** — it is the
    transform whose whole job is moving those axes, so handing torchvision a picture
    already moved would ask it to do nothing and call that agreement."""
    m = _vision_v2(L)
    out = _as_numpy(build(m, L)(picture))
    return L.tensor(np.ascontiguousarray(np.asarray(out, dtype=np.float32)))


DATASET_PREFIX = "dataset::"


def _write_png8(path, arr):
    """An eight-bit PNG, **written by hand rather than through PIL.**

    These fixtures run in three places and one of them is Pyodide, where there is no
    PIL — measured: fifty-one dataset cases failed on the binding with
    `No module named 'PIL'` while numpy and the browser passed. **A fixture that needs a
    library the case does not is a case only two of three implementations can be
    asked**, and two of three is not this table's standard.

    The comment these replaced said PIL kept *the bytes torchvision would meet*. It is
    a real point and it is smaller than this one: what is written here is a PNG by the
    specification, PIL reads it on the side that has PIL, and the other two sides can
    now be asked at all.
    """
    import os
    import struct
    import zlib
    arr = np.asarray(arr)
    height, width = arr.shape[:2]
    # `len(arr.shape)` and not `arr.ndim`, for the reason written twice above: this
    # file is parsed for the names its cases ask about, and a fixture's numpy `.ndim`
    # is indistinguishable from a tensor's.
    channels = 1 if len(arr.shape) == 2 else arr.shape[2]
    rows = arr.reshape(height, width * channels).astype(np.uint8)
    raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(height))

    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body)))

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    header = struct.pack(">IIBBBBB", width, height, 8,
                         {1: 0, 3: 2, 4: 6}[channels], 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _write_ppm8(path, arr):
    """A binary PPM — **a magic number, three numbers and the samples.**"""
    import os
    arr = np.asarray(arr).astype(np.uint8)
    height, width = arr.shape[:2]
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode())
        handle.write(arr.reshape(-1).tobytes())


def _vision_datasets(L):
    """`borchvision.datasets` for us, `torchvision.datasets` for real torch."""
    if _is_real_torch(L):
        import torchvision.datasets as real
        return real
    _vision(L)
    import sys as _sys
    return _sys.modules["borchvision"].datasets


def _idx_bytes(kind, shape, payload):
    """An IDX file, built here. **The header is the case.**

    Two zero bytes, then the type code, then the number of axes; then one big-endian
    length per axis. Writing it out rather than downloading one is what lets the
    decoder be compared at all: torchvision's reader takes a path, so the bytes have to
    exist as a file on both sides, and eleven megabytes of MNIST cannot go in a golden
    dump for a header that is sixteen bytes long.
    """
    head = bytes([0, 0, kind, len(shape)])
    for length in shape:
        head += int(length).to_bytes(4, "big")
    return head + payload


def dataset_cases(inp=None):
    """`borchvision.datasets` — **the decoders, asked without a network.**

    A dataset is two things: an address and a format. The address half cannot be a
    golden case, because a case that downloads is a case that fails on a train, and
    because freezing MNIST's answer means shipping MNIST. The format half is the part
    that can be got wrong quietly, and it is all here.

    The bytes are **built in this file** and handed to both sides. torchvision's
    readers take a path rather than bytes, so each case writes a temporary file — that
    is the only reason the plumbing below exists, and it is worth the plumbing: the
    alternative is comparing our decoder against our own expectations, which proves
    that the test was written after the code.

    **The real data was compared once, outside this table.** MNIST, FashionMNIST and
    KMNIST, both splits, against real torchvision: `data`, `targets`, `classes`,
    `class_to_idx`, `len` and `__getitem__` all equal, and our download produced files
    byte-identical to torchvision's. That is written in the README rather than frozen
    here, because a check that needs 250MB and a working network is not a check that
    runs.
    """
    del inp

    def on_idx(build, read):
        """The same bytes to both sides — through a file, because that is what
        torchvision's reader takes."""
        def run(L):
            import os
            import tempfile
            data = build()
            handle, path = tempfile.mkstemp(suffix="-idx")
            try:
                with os.fdopen(handle, "wb") as out:
                    out.write(data)
                if _is_real_torch(L):
                    from torchvision.datasets import mnist as real
                    got = getattr(real, read)(path)
                else:
                    mod = _vision_datasets(L)
                    import sys as _sys
                    lib = _sys.modules["borchvision"]
                    got = (lib._read_idx_images if read == "read_image_file"
                           else lib._read_idx_labels)(data)
                    del mod
            finally:
                os.unlink(path)
            return L.tensor(np.ascontiguousarray(
                np.asarray(_as_numpy(got), dtype=np.float32)))
        return run

    # Pictures: two 3x4 frames of bytes, counting up so that a transposed or
    # mis-strided read lands somewhere visibly different.
    _pixels = np.arange(24, dtype=np.uint8).tobytes()
    # Labels: ten of them, including 0 and 255 — the ends are where a signed read shows.
    _labels = bytes([0, 1, 2, 9, 200, 255, 3, 4, 5, 6])

    cases = [
        (DATASET_PREFIX + "IDX images",
         on_idx(lambda: _idx_bytes(8, (2, 3, 4), _pixels), "read_image_file")),
        (DATASET_PREFIX + "IDX labels",
         on_idx(lambda: _idx_bytes(8, (10,), _labels), "read_label_file")),
        (DATASET_PREFIX + "IDX images(one frame)",
         on_idx(lambda: _idx_bytes(8, (1, 4, 6), _pixels), "read_image_file")),
    ]

    def refuses(build, read, phrase):
        """**A header that promises more than the file carries.** `strict=False` in
        torchvision relaxes an assert and not the reshape underneath it, so both sides
        still refuse — and what is frozen is that they refuse the same way. Measured,
        torch's words are `shape '[12]' is invalid for input of size 10`; ours are the
        same sentence, which is what makes the phrase searchable across the two.
        """
        def run(L):
            import os
            import tempfile
            data = build()
            handle, path = tempfile.mkstemp(suffix="-idx")
            try:
                with os.fdopen(handle, "wb") as out:
                    out.write(data)
                try:
                    if _is_real_torch(L):
                        from torchvision.datasets import mnist as real
                        getattr(real, read)(path)
                    else:
                        import sys as _sys
                        lib = _sys.modules["borchvision"]
                        (lib._read_idx_images if read == "read_image_file"
                         else lib._read_idx_labels)(data)
                except Exception as exc:                            # noqa: BLE001
                    return f"거절|문구={phrase in str(exc)}"
            finally:
                os.unlink(path)
            return "예외가 안 났다"
        return run

    cases.append((DATASET_PREFIX + "IDX labels(short by two)=거절",
                  refuses(lambda: _idx_bytes(8, (12,), _labels), "read_label_file",
                          "invalid for input of size 10")))

    def cifar(build, want):
        """A CIFAR batch, pickled here and read by both sides.

        torchvision has no public reader for one — it opens the file inside
        `CIFAR10.__init__`, which needs a whole directory of batches with the right
        checksums. So **the comparison is against `pickle` plus the reshape written
        out**, which is what torchvision's few lines do, and the case is worth having
        for the reshape: a batch is planar, 1024 red then 1024 green then 1024 blue,
        and a channel-swapped CIFAR trains to a plausible number.
        """
        def run(L):
            import pickle
            raw = pickle.dumps(build())
            if _is_real_torch(L):
                entry = pickle.loads(raw, encoding="latin1")
                labels = entry.get("labels", entry.get("fine_labels"))
                out = (np.asarray(entry["data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                       if want == "data" else np.asarray(labels))
            else:
                import sys as _sys
                lib = _sys.modules["borchvision"]
                images, labels = lib._read_cifar_batch(raw)
                out = images if want == "data" else np.asarray(labels)
            return L.tensor(np.ascontiguousarray(np.asarray(out, dtype=np.float32)))
        return run

    # Two pictures, and **each channel a constant** so that a channel swap is a
    # different number rather than a different arrangement of the same numbers.
    _batch = np.zeros((2, 3072), dtype=np.uint8)
    _batch[0, 0:1024], _batch[0, 1024:2048], _batch[0, 2048:] = 10, 20, 30
    _batch[1, 0:1024], _batch[1, 1024:2048], _batch[1, 2048:] = 40, 50, 60
    # One pixel off the constant, so that the reshape's own order is asked too.
    _batch[0, 5], _batch[0, 1024 + 7] = 200, 201
    _entry = {"data": _batch, "labels": [3, 7]}

    cases += [
        (DATASET_PREFIX + "CIFAR batch(pictures)", cifar(lambda: dict(_entry), "data")),
        (DATASET_PREFIX + "CIFAR batch(labels)", cifar(lambda: dict(_entry), "labels")),
        # **CIFAR-100 spells the same field `fine_labels`**, and the file does not say
        # which of the two it is. A reader that only knows one key reads the other
        # dataset as having no labels at all, which is a dataset that trains.
        (DATASET_PREFIX + "CIFAR batch(fine_labels)",
         cifar(lambda: {"data": _batch, "fine_labels": [11, 62]}, "labels")),
    ]
    return cases


V2F_PREFIX = "v2f::"


def _vision_v2_functional(L):
    """`transforms.v2.functional` for each side."""
    if _is_real_torch(L):
        from torchvision.transforms.v2 import functional as real
        return real
    return _vision_v2(L).functional


def v2_functional_cases(inp=None):
    """`transforms.v2.functional` — **the nine v2 adds, and that the rest is one body.**

    34 of the 51 real names here are v1's, re-exported rather than rewritten, and a
    re-export cannot be got wrong in a way a value case would see. What it *can* be got
    wrong in is being a copy instead of a re-export, so four of v1's are asked through
    the v2 spelling: if one ever grows a body of its own, these stop matching v1's own
    frozen answers.

    The nine adds are asked outright. Two of them are worth pointing at. `get_size`
    answers `[height, width]` where v1's `get_image_size` answers `[width, height]` —
    the two sit one namespace apart giving opposite answers, and both are frozen here
    beside each other so that neither can drift into the other. And `elastic`'s `fill`
    defaults to `None`; written as `0` it paints the outside of the warp black, which
    on a picture whose edges barely move reads as the warp working. That one was
    written as `0` first and caught by comparing.
    """
    inp = golden_inputs() if inp is None else inp
    img_f = inp["vis_f"]
    img_u8 = inp["vis_u8"]
    grey = np.ascontiguousarray(img_f[:, :, :1])
    # A displacement small enough that the picture stays recognisable and large enough
    # that every pixel moves — a zero field would pass against a broken warp.
    shift = (np.random.default_rng(7).random((1,) + img_f.shape[:2] + (2,))
             .astype(np.float32) * 0.2 - 0.1)

    def on(picture, call):
        """The same picture in each side's own layout, out as a float tensor."""
        def run(L):
            F = _vision_v2_functional(L)
            if _is_real_torch(L):
                given = L.tensor(np.ascontiguousarray(picture.transpose(2, 0, 1)))
                out = _as_numpy(call(F, given, L).detach())
                out = out.transpose(1, 2, 0) if len(out.shape) == 3 else out
            else:
                out = _as_numpy(call(F, picture, L))
            return L.tensor(np.ascontiguousarray(np.asarray(out, dtype=np.float32)))
        return run

    def answer(picture, call):
        """For the ones that give a list or a number rather than a picture. Frozen as
        **text**, because `[5, 4]` and `[4, 5]` compare equal to nothing else."""
        def run(L):
            F = _vision_v2_functional(L)
            given = (L.tensor(np.ascontiguousarray(picture.transpose(2, 0, 1)))
                     if _is_real_torch(L) else picture)
            return str(call(F, given))
        return run

    # Boxes for the `*_bounding_boxes` kernels, in a canvas that is **not square** — a
    # square one lets a height/width mix-up through every one of these, and the two
    # orders are opposite (`(x, y)` against `(height, width)`).
    #
    # The last two sit on and past the right edge, which is what makes the crop family's
    # clamping visible: without them every box stays inside and the clip never runs.
    boxes = np.array([[0.0, 0.0, 10.0, 10.0],
                      [5.0, 5.0, 15.0, 20.0],
                      [2.0, 3.0, 4.0, 9.0],
                      [28.0, 20.0, 32.0, 24.0],
                      [30.0, 22.0, 31.0, 23.0]], dtype=np.float32)
    canvas = (24, 32)

    def on_boxes(fmt, call):
        """Boxes through a kernel, in the format named.

        **torchvision wants its `BoundingBoxFormat` enum and this library wants a
        string.** Handed a string, torchvision iterates it and raises `IndexError: index
        4 is out of bounds` — so the two sides are given different spellings of the same
        format here, which is a real divergence and is written down rather than hidden.

        **The return shape is compared, not normalised.** Five of these answer
        `(boxes, canvas_size)` and two answer boxes alone; a probe that flattened that
        away is what made the first implementation return the wrong thing from three of
        them.
        """
        def run(L):
            F = _vision_v2_functional(L)
            to = ((lambda a: L.tensor(np.ascontiguousarray(a))) if _is_real_torch(L)
                  else (lambda a: a))
            if _is_real_torch(L):
                import torchvision.tv_tensors as _tv
                given = L.tensor(np.ascontiguousarray(
                    _as_numpy(_vision_ops(L).box_convert(L.tensor(boxes), "xyxy", fmt))))
                got = call(F, given, _tv.BoundingBoxFormat[fmt.upper()], to)
            else:
                got = call(F, _vision_ops(L).box_convert(boxes, "xyxy", fmt), fmt, to)
            if isinstance(got, tuple):
                rest = " ".join(str(int(v)) for v in got[1]) if not hasattr(
                    got[1], "tolist") else str(_as_numpy(got[1]).astype(np.int64).tolist())
                return f"{_boxtext(got[0])} {rest}"
            return _boxtext(got)
        return run

    def _boxtext(values):
        """The rows as text, **with negative zero folded into zero.**

        These are frozen as strings so the canvas can ride along in the same answer, and
        a string freeze compares spellings where the rest of the golden compares numbers.
        That invented a difference: mirroring a box whose far edge is the canvas edge
        gives `32 - 32`, and torchvision spells the result `-0.0` where this gives `0.0`.
        The two are the same number and `-0.0 == 0.0` is true; only the text differs.

        **Four cases failed on it and none of them was a defect.** Folding the sign of
        zero is the narrowest thing that removes the false difference — rounding or
        tolerance would also hide real ones.

        ## Rounded in float64, and rounding in float32 was not rounding

        `np.round(x, 4)` on a **float32** array picks the nearest float32 to the rounded
        decimal, and widening that back to a double prints `4.571400165557861` rather
        than `4.5714`. Three cases disagreed with borch.ts on exactly that: the values
        were equal, one side was printing float32's neighbour and the other the decimal.
        So the widening happens **before** the rounding, and the text is then the number
        that was asked for.
        """
        # Through float32 and back before rounding, for the reason `_flattext` gives:
        # torch answers in float32 and this side computes in float64, and four decimals
        # is right on the edge where the two stop agreeing.
        narrow = np.asarray(_as_numpy(values), dtype=np.float32).astype(np.float64)
        arr = np.round(narrow, 3)
        return str(np.where(arr == 0, 0.0, arr).tolist())

    # A mask with two planes, in a canvas that is not square. **A label map, not a
    # picture**: the values are class indices, so anything but nearest sampling averages
    # 3 and 5 into 4 and produces a mask that is smooth and wrong wherever two regions
    # meet. The third class is there so that averaging would be visible at all — with
    # 0/1 alone a bilinear resample rounds back to the same answer half the time.
    labels = (np.arange(2 * 24 * 32, dtype=np.int64).reshape(2, 24, 32) % 3
              ).astype(np.uint8)
    # Keypoints as `(N, K, 2)` — three groups of two. **The last group has a point
    # outside the canvas**, which is what makes `sanitize` drop a whole group rather
    # than a point, and what makes the flips produce a negative coordinate.
    points = np.array([[[0.0, 0.0], [1.0, 2.0]],
                       [[5.0, 5.0], [31.0, 23.0]],
                       [[10.0, 10.0], [33.0, 26.0]]], dtype=np.float32)

    def on_mask(call):
        """A mask through a kernel, as **shape then a flat list of values.**

        **The dtype is part of the answer** — masks are labels, and widening `uint8`
        would make `== 3` start failing on values that are exactly equal on
        torchvision's side.

        The values are flattened rather than printed as `.tolist()`'s nesting. Nesting
        is presentation, and the two sides reached it differently — borch.ts's box
        formatter chunks four to a row, which is right for boxes and meaningless for a
        `(2, 24, 32)` mask. Flat plus the shape says the same thing and says it the same
        way on both sides.
        """
        def run(L):
            F = _vision_v2_functional(L)
            to = ((lambda a: L.tensor(np.ascontiguousarray(a))) if _is_real_torch(L)
                  else (lambda a: a))
            given = to(labels)
            got = np.asarray(_as_numpy(call(F, given, to)))
            return f"{_flattext(got, whole=True)} {got.dtype}"
        return run

    def _flattext(values, whole=False):
        """`[shape] [values]` — **flat, and the same flat on both sides.**

        The nested spelling `.tolist()` produces is presentation, and the two libraries
        reached it differently: borch.ts's box formatter chunks four to a row, which is
        right for `(N, 4)` boxes, wrong for a `(2, 24, 32)` mask and wrong again for
        `(N, K, 2)` keypoints. Eighteen cases failed on that and **not one of them was a
        wrong number.**

        So the shape is stated and the values are flat. Nothing about the answer is lost
        — a shape plus a flat list is the array — and there is only one way to write it.
        """
        arr = np.asarray(values)
        if whole:
            body = [int(v) for v in arr.reshape(-1)]
        else:
            # **Three decimals, and the third one is where the two sides still agree.**
            #
            # torch answers in float32 and this library computes in float64, so they
            # agree to about seven significant digits. On a warped corner near 18 that
            # is the sixth decimal, and a freeze at four decimals sits right on the
            # edge: `18.267949…` came out `18.2679` on one side and `18.268` on the
            # other, from values 1e-6 apart. Narrowing to float32 first does not settle
            # it either — the two straddle a float32 boundary.
            #
            # So the text resolves to 1e-3, which on values running to 40 is a relative
            # 2.5e-5: far above float32's noise and far below anything this has ever
            # caught. **The defects these cases found were an off-by-one pixel, a
            # doubled value and a clamp on the wrong canvas** — 0.1 and larger, every
            # one.
            #
            # Rounding the float32 array directly is a different mistake and was made
            # first: `np.round(float32, 3)` picks the nearest float32 to the decimal,
            # and widening that back prints `4.571000099182129`. Narrow, then widen,
            # then round.
            wide = arr.astype(np.float32).astype(np.float64).reshape(-1)
            rounded = np.round(wide, 3)
            body = np.where(rounded == 0, 0.0, rounded).tolist()
        return f"{list(arr.shape)} {body}"

    def on_points(call):
        def run(L):
            F = _vision_v2_functional(L)
            to = ((lambda a: L.tensor(np.ascontiguousarray(a))) if _is_real_torch(L)
                  else (lambda a: a))
            given = to(points)
            got = call(F, given, to)
            if isinstance(got, tuple):
                second = got[1]
                tail = (_flattext(_as_numpy(second), whole=True)
                        if hasattr(second, "tolist") or hasattr(second, "shape")
                        else " ".join(str(int(v)) for v in second))
                return f"{_flattext(_as_numpy(got[0]))} {tail}"
            return _flattext(_as_numpy(got))
        return run

    mask_cases = [
        (V2F_PREFIX + "horizontal_flip_mask",
         on_mask(lambda F, m, to: F.horizontal_flip_mask(m))),
        (V2F_PREFIX + "vertical_flip_mask",
         on_mask(lambda F, m, to: F.vertical_flip_mask(m))),
        (V2F_PREFIX + "crop_mask(inside)",
         on_mask(lambda F, m, to: F.crop_mask(m, top=1, left=2, height=12, width=14))),
        # **A crop that starts outside the picture.** torchvision pads there with zero
        # rather than raising, and a detection pipeline does this constantly.
        (V2F_PREFIX + "crop_mask(off the corner)",
         on_mask(lambda F, m, to: F.crop_mask(m, top=-3, left=-4, height=10, width=10))),
        (V2F_PREFIX + "center_crop_mask(odd)",
         on_mask(lambda F, m, to: F.center_crop_mask(m, output_size=[11, 13]))),
        (V2F_PREFIX + "pad_mask(four sides, fill=7)",
         on_mask(lambda F, m, to: F.pad_mask(m, padding=[1, 2, 3, 4], fill=7))),
        # The ratio 24→12 is exact and 32→15 is not, so the second one is where a
        # `nearest` and a `nearest-exact` part company.
        (V2F_PREFIX + "resize_mask(pair)",
         on_mask(lambda F, m, to: F.resize_mask(m, size=[12, 15]))),
        (V2F_PREFIX + "resize_mask(short edge)",
         on_mask(lambda F, m, to: F.resize_mask(m, size=[9]))),
        (V2F_PREFIX + "resized_crop_mask",
         on_mask(lambda F, m, to: F.resized_crop_mask(m, top=1, left=2, height=12,
                                                  width=14, size=[6, 8]))),
    ]

    point_cases = [
        # **`(width - 1) - x` and not `width - x`** — a box edge lives on `[0, width]`
        # and a point is a pixel index on `[0, width - 1]`. The box way is off by one
        # everywhere, which is a skeleton one column from where it belongs.
        (V2F_PREFIX + "horizontal_flip_keypoints",
         on_points(lambda F, k, to: F.horizontal_flip_keypoints(k, canvas_size=(24, 32)))),
        (V2F_PREFIX + "vertical_flip_keypoints",
         on_points(lambda F, k, to: F.vertical_flip_keypoints(k, canvas_size=(24, 32)))),
        (V2F_PREFIX + "crop_keypoints",
         on_points(lambda F, k, to: F.crop_keypoints(k, top=1, left=2, height=12, width=14))),
        (V2F_PREFIX + "center_crop_keypoints(odd)",
         on_points(lambda F, k, to: F.center_crop_keypoints(
             k, canvas_size=(24, 32), output_size=[11, 13]))),
        (V2F_PREFIX + "pad_keypoints(four sides)",
         on_points(lambda F, k, to: F.pad_keypoints(k, canvas_size=(24, 32),
                                                padding=[1, 2, 3, 4]))),
        (V2F_PREFIX + "resize_keypoints(pair)",
         on_points(lambda F, k, to: F.resize_keypoints(k, size=[12, 16],
                                                   canvas_size=(24, 32)))),
        (V2F_PREFIX + "resized_crop_keypoints",
         on_points(lambda F, k, to: F.resized_crop_keypoints(
             k, top=1, left=2, height=12, width=14, size=[6, 8]))),
        # The clamp stops at `width - 1`, one short of where the box clamp stops.
        (V2F_PREFIX + "clamp_keypoints",
         on_points(lambda F, k, to: F.clamp_keypoints(k, canvas_size=(24, 32)))),
        # **The unit is the group.** Three groups of two points; the third has a point
        # outside, so the whole group goes and the mask is three long rather than six.
        (V2F_PREFIX + "sanitize_keypoints",
         on_points(lambda F, k, to: F.sanitize_keypoints(k, canvas_size=(24, 32)))),
    ]

    # A displacement field for the elastic pair. **Not constant** — a constant one moves
    # every point by the same amount, so it cannot tell "read the field at the point"
    # from "read it anywhere", and reading it at the wrong pixel is the mistake available
    # here.
    # **An integer formula and not a drawn field.** borch.ts has to build the same
    # numbers, and no seeded generator matches across two runtimes — numpy's PCG64 and
    # anything JavaScript can write are different streams. A formula both sides evaluate
    # is the same field by construction.
    _idx = np.arange(24 * 32, dtype=np.int64)
    field = np.stack((((_idx % 17) - 8) / 100.0, ((_idx % 23) - 11) / 100.0),
                     axis=-1).reshape(1, 24, 32, 2).astype(np.float32)
    SP = [[0, 0], [31, 0], [31, 23], [0, 23]]
    EP = [[2, 1], [29, 2], [30, 22], [1, 21]]

    warp_cases = [
        # **A rotated box grows** — the upright hull of a tilted rectangle is larger than
        # the rectangle. 90 degrees is in the list because it is the one angle where the
        # hull is the box again, so it catches a transform that is right about extents
        # and wrong about direction.
        (V2F_PREFIX + "affine_bounding_boxes(30)",
         on_boxes("xyxy", lambda F, b, f, to: F.affine_bounding_boxes(
             b, format=f, canvas_size=(24, 32), angle=30.0, translate=[0, 0],
             scale=1.0, shear=[0.0, 0.0]))),
        (V2F_PREFIX + "affine_bounding_boxes(90)",
         on_boxes("xyxy", lambda F, b, f, to: F.affine_bounding_boxes(
             b, format=f, canvas_size=(24, 32), angle=90.0, translate=[0, 0],
             scale=1.0, shear=[0.0, 0.0]))),
        # All four knobs at once, because each alone leaves three untested and they
        # compose in an order that is easy to get backwards.
        (V2F_PREFIX + "affine_bounding_boxes(all four)",
         on_boxes("xyxy", lambda F, b, f, to: F.affine_bounding_boxes(
             b, format=f, canvas_size=(24, 32), angle=-15.0, translate=[3, -2],
             scale=1.3, shear=[10.0, 5.0]))),
        # **`center=[0, 0]` is the corner, not the middle.** The default centre is
        # `[w/2, h/2]`, and an implementation that reuses the image grid's convention
        # answers this case for the default one — which is how that bug was found.
        (V2F_PREFIX + "affine_bounding_boxes(center at the corner)",
         on_boxes("xyxy", lambda F, b, f, to: F.affine_bounding_boxes(
             b, format=f, canvas_size=(24, 32), angle=30.0, translate=[0, 0],
             scale=1.0, shear=[0.0, 0.0], center=[0, 0]))),
        (V2F_PREFIX + "rotate_bounding_boxes(30)",
         on_boxes("xyxy", lambda F, b, f, to: F.rotate_bounding_boxes(
             b, format=f, canvas_size=(24, 32), angle=30.0))),
        # **`expand` grows the canvas, and to a size the image path does not agree
        # with.** The two hand the same size function different matrices — a centre
        # offset against an absolute centre — so a 24x32 at 30 degrees is 38 rows to the
        # image and 37 here. Both are torchvision's.
        (V2F_PREFIX + "rotate_bounding_boxes(30, expand)",
         on_boxes("xyxy", lambda F, b, f, to: F.rotate_bounding_boxes(
             b, format=f, canvas_size=(24, 32), angle=30.0, expand=True))),
        (V2F_PREFIX + "rotate_bounding_boxes(-47, expand)",
         on_boxes("xyxy", lambda F, b, f, to: F.rotate_bounding_boxes(
             b, format=f, canvas_size=(24, 32), angle=-47.0, expand=True))),
        (V2F_PREFIX + "perspective_bounding_boxes",
         on_boxes("xyxy", lambda F, b, f, to: F.perspective_bounding_boxes(
             b, format=f, canvas_size=(24, 32), startpoints=SP, endpoints=EP))),
        (V2F_PREFIX + "elastic_bounding_boxes",
         on_boxes("xyxy", lambda F, b, f, to: F.elastic_bounding_boxes(
             b, format=f, canvas_size=(24, 32),
             displacement=to(field)))),
    ]

    warp_point_cases = [
        (V2F_PREFIX + "affine_keypoints(30)",
         on_points(lambda F, k, to: F.affine_keypoints(
             k, canvas_size=(24, 32), angle=30.0, translate=[0, 0], scale=1.0,
             shear=[0.0, 0.0]))),
        (V2F_PREFIX + "affine_keypoints(all four)",
         on_points(lambda F, k, to: F.affine_keypoints(
             k, canvas_size=(24, 32), angle=-15.0, translate=[3, -2], scale=1.3,
             shear=[10.0, 5.0]))),
        (V2F_PREFIX + "rotate_keypoints(30, expand)",
         on_points(lambda F, k, to: F.rotate_keypoints(
             k, canvas_size=(24, 32), angle=30.0, expand=True))),
        (V2F_PREFIX + "perspective_keypoints",
         on_points(lambda F, k, to: F.perspective_keypoints(
             k, canvas_size=(24, 32), startpoints=SP, endpoints=EP))),
        (V2F_PREFIX + "elastic_keypoints",
         on_points(lambda F, k, to: F.elastic_keypoints(
             k, canvas_size=(24, 32), displacement=to(field)))),
    ]

    warp_mask_cases = [
        (V2F_PREFIX + "affine_mask(30)",
         on_mask(lambda F, m, to: F.affine_mask(m, angle=30.0, translate=[0, 0],
                                            scale=1.0, shear=[0.0, 0.0]))),
        (V2F_PREFIX + "affine_mask(all four)",
         on_mask(lambda F, m, to: F.affine_mask(m, angle=-15.0, translate=[3, -2],
                                            scale=1.3, shear=[10.0, 5.0]))),
        (V2F_PREFIX + "rotate_mask(30)", on_mask(lambda F, m, to: F.rotate_mask(m, 30.0))),
        # The mask's expanded canvas is the **image** path's, which is the other of the
        # two answers — a box in the same call is clipped to a different one.
        (V2F_PREFIX + "rotate_mask(30, expand)",
         on_mask(lambda F, m, to: F.rotate_mask(m, 30.0, expand=True))),
        (V2F_PREFIX + "perspective_mask",
         on_mask(lambda F, m, to: F.perspective_mask(m, startpoints=SP, endpoints=EP))),
        (V2F_PREFIX + "elastic_mask",
         on_mask(lambda F, m, to: F.elastic_mask(m, displacement=to(field)))),
    ]

    box_cases = mask_cases + point_cases + warp_cases + warp_point_cases + warp_mask_cases
    for _fmt in ("xyxy", "xywh", "cxcywh"):
        box_cases += [
            (V2F_PREFIX + f"horizontal_flip_bounding_boxes({_fmt})",
             on_boxes(_fmt, lambda F, b, f, to: F.horizontal_flip_bounding_boxes(
                 b, format=f, canvas_size=canvas))),
            (V2F_PREFIX + f"vertical_flip_bounding_boxes({_fmt})",
             on_boxes(_fmt, lambda F, b, f, to: F.vertical_flip_bounding_boxes(
                 b, format=f, canvas_size=canvas))),
            (V2F_PREFIX + f"crop_bounding_boxes({_fmt})",
             on_boxes(_fmt, lambda F, b, f, to: F.crop_bounding_boxes(
                 b, format=f, top=1, left=2, height=12, width=14))),
            # **An odd output size, because an even one cannot tell the rounding apart.**
            # The offset is `round((canvas - out) / 2)` and Python's `round` breaks ties
            # to even; a floor division agrees on every even size and is off by a whole
            # pixel here. The floor version shipped until this case was written.
            (V2F_PREFIX + f"center_crop_bounding_boxes(odd, {_fmt})",
             on_boxes(_fmt, lambda F, b, f, to: F.center_crop_bounding_boxes(
                 b, format=f, canvas_size=canvas, output_size=[11, 13]))),
            (V2F_PREFIX + f"pad_bounding_boxes(four sides, {_fmt})",
             on_boxes(_fmt, lambda F, b, f, to: F.pad_bounding_boxes(
                 b, format=f, canvas_size=canvas, padding=[1, 2, 3, 4]))),
            (V2F_PREFIX + f"resize_bounding_boxes(pair, {_fmt})",
             on_boxes(_fmt, lambda F, b, f, to: F.resize_bounding_boxes(
                 b, canvas_size=canvas, size=[12, 16], format=f))),
            # A single number keeps the aspect ratio, which a pair does not — the two
            # spellings of `size` are different functions wearing one name.
            (V2F_PREFIX + f"resize_bounding_boxes(short edge, {_fmt})",
             on_boxes(_fmt, lambda F, b, f, to: F.resize_bounding_boxes(
                 b, canvas_size=canvas, size=[12], format=f))),
            (V2F_PREFIX + f"resized_crop_bounding_boxes({_fmt})",
             on_boxes(_fmt, lambda F, b, f, to: F.resized_crop_bounding_boxes(
                 b, format=f, top=1, left=2, height=12, width=14, size=[6, 8]))),
        ]

    return box_cases + [
        (V2F_PREFIX + "horizontal_flip", on(img_f, lambda F, x, L: F.horizontal_flip(x))),
        (V2F_PREFIX + "vertical_flip", on(img_f, lambda F, x, L: F.vertical_flip(x))),
        (V2F_PREFIX + "grayscale_to_rgb(one channel)",
         on(grey, lambda F, x, L: F.grayscale_to_rgb(x))),
        # Three channels **pass through** rather than raising, so a mixed pipeline needs
        # no branch — and a version that stacked them again would give nine.
        (V2F_PREFIX + "grayscale_to_rgb(three channels)",
         on(img_f, lambda F, x, L: F.grayscale_to_rgb(x))),
        (V2F_PREFIX + "permute_channels",
         on(img_f, lambda F, x, L: F.permute_channels(x, [2, 0, 1]))),
        (V2F_PREFIX + "to_dtype(scaling)",
         on(img_u8, lambda F, x, L: F.to_dtype(x, L.float32, scale=True))),
        (V2F_PREFIX + "to_dtype(not scaling)",
         on(img_u8, lambda F, x, L: F.to_dtype(x, L.float32))),
        (V2F_PREFIX + "gaussian_noise(sigma=0)",
         on(img_f, lambda F, x, L: F.gaussian_noise(x, 0.0, 0.0))),
        (V2F_PREFIX + "gaussian_noise(clipping)",
         on(img_f, lambda F, x, L: F.gaussian_noise(x, 5.0, 0.0, True))),
        (V2F_PREFIX + "elastic",
         on(img_f, lambda F, x, L: F.elastic(x, L.tensor(np.ascontiguousarray(shift))))),

        # **The two that answer a size, side by side.** v2 reversed the pair on purpose
        # and the names are one namespace apart, so a reader who takes the wrong one
        # gets a picture that is transposed and still plausible.
        (V2F_PREFIX + "get_size(height first)", answer(img_f, lambda F, x: F.get_size(x))),
        (V2F_PREFIX + "get_image_size(width first)",
         answer(img_f, lambda F, x: F.get_image_size(x))),
        (V2F_PREFIX + "get_num_channels", answer(img_f, lambda F, x: F.get_num_channels(x))),

        # Four of v1's, reached through the v2 spelling. These are the ones that catch a
        # re-export quietly becoming a second implementation.
        (V2F_PREFIX + "resize(inherited)", on(img_f, lambda F, x, L: F.resize(x, [3, 2]))),
        (V2F_PREFIX + "normalize(inherited)",
         on(img_f, lambda F, x, L: F.normalize(x, [0.5], [0.25]))),
        (V2F_PREFIX + "rotate(inherited)", on(img_f, lambda F, x, L: F.rotate(x, 30.0))),
        (V2F_PREFIX + "adjust_hue(inherited)",
         on(img_f, lambda F, x, L: F.adjust_hue(x, 0.2))),
    ]


def dataset_text_cases(inp=None):
    """The three text-and-table formats, **read from bytes built here.**

    `dataset_cases` above froze the two binary decoders. These three are the ones the
    eight-name to-do list gave up, and the reason they could be given up is the reason
    they can be frozen: SEMEION is a text file, USPS is text inside bzip2, and QMNIST's
    labels are the same IDX reader with a wider type and a second axis. No codec, no
    network, and the conversions are all arithmetic somebody can get backwards.

    **Only QMNIST's is here**, and the reason is what a comparison is for. torchvision
    reads SEMEION with `np.loadtxt` and two lines of arithmetic, and USPS with `bz2`
    and three; it exposes no function for either, so a case would have to compare our
    arithmetic against the same arithmetic typed a second time in this file. That is a
    test of the typing. Those two were compared against the real library on the real
    files instead — once, recorded in the README.

    QMNIST's labels can be compared, because the reader for them is torchvision's own
    `read_sn3_pascalvincent_tensor` and it takes a path. What it is asked is the shape
    MNIST's labels never have: **int32, big-endian, two axes.** Read as bytes, or read
    the other way round, a field like 279260 becomes something else entirely — and a
    label table that is wrong in its later columns still trains, because the digit is
    column zero.
    """
    del inp

    def on_idx(build):
        """The IDX reader, on a **two-axis int32** table — the shape QMNIST's labels
        have and MNIST's never do."""
        def run(L):
            import os
            import tempfile
            data = build()
            handle, path = tempfile.mkstemp(suffix="-idx2")
            try:
                with os.fdopen(handle, "wb") as out:
                    out.write(data)
                if _is_real_torch(L):
                    from torchvision.datasets import mnist as real
                    got = real.read_sn3_pascalvincent_tensor(path, strict=False)
                else:
                    import sys as _sys
                    _vision(L)
                    got = _sys.modules["borchvision"]._read_idx(data)
            finally:
                os.unlink(path)
            return L.tensor(np.ascontiguousarray(
                np.asarray(_as_numpy(got), dtype=np.float32)))
        return run

    def idx2(rows, columns, values):
        head = bytes([0, 0, 12, 2])                 # type 12 is int32, two axes
        head += int(rows).to_bytes(4, "big") + int(columns).to_bytes(4, "big")
        return head + b"".join(int(v).to_bytes(4, "big", signed=True) for v in values)

    # Eight columns, three rows, and **values past what a byte holds** — 279260 is a
    # real QMNIST field. Read as bytes it becomes something else entirely, and read
    # little-endian it becomes something else again.
    _rows = [7, 4, 2578, 69, 37, 279260, 0, 0,
             2, 4, 2359, 55, 32, 253328, 0, 0,
             1, 4, 2530, 80, 31, 273542, 0, 0]

    return [
        (DATASET_PREFIX + "IDX int32 table(QMNIST labels)",
         on_idx(lambda: idx2(3, 8, _rows))),
        # A negative, because int32 is signed and the reader must not widen it wrong.
        (DATASET_PREFIX + "IDX int32 table(negative)",
         on_idx(lambda: idx2(2, 2, [-1, 2147483647, -2147483648, 0]))),
    ]


def dataset_last_three_cases(inp=None):
    """The three that were the to-do list, **and the reason two of them can be frozen
    here while the third could not until it was tried.**

    `STL10` and `MovingMNIST` are raw bytes and a `.npy`; both can be built in memory
    and handed to torchvision's own reader. `FER2013` was written down as impossible
    to compare — torchvision cannot download it, it wants a Kaggle account — and that
    was **cannot fetch the data carried over into cannot check the code.** Its reader
    takes a directory. A CSV written here goes to both sides and the comparison is as
    real as any other in this table.

    Each case is a conversion that leaves a dataset which still trains when reversed.
    STL10 is stored column-major and its labels start at 1; MovingMNIST's `split` cuts
    the **frames** and not the clips; FER2013's pixels are a string of integers in a
    cell, and one of its two layouts puts a **leading space in the column names.**
    """
    del inp
    prefix = DATASET_PREFIX

    def stl(build, want, transpose=True):
        """torchvision has no public STL10 reader either, so what both sides run is
        `np.frombuffer` and the reshape — six lines that are the whole format. The
        case earns its place on the **transpose**: skip it and every picture comes out
        rotated, which is still a picture and still trains."""
        def run(L):
            raw = build()
            flat = np.frombuffer(raw, dtype=np.uint8)
            if want == "labels":
                out = flat.astype(np.int64) - 1
            else:
                out = flat.reshape(-1, 3, 4, 4)
                out = out.transpose(0, 1, 3, 2) if transpose else out
            return L.tensor(np.ascontiguousarray(np.asarray(out, dtype=np.float32)))
        return run

    # Two 4x4x3 pictures rather than 96x96 — the arrangement is the question and the
    # size is not. Values count up, so a transposed read lands somewhere visible.
    _pixels = bytes(range(96))
    _labels = bytes([1, 10, 5, 3])          # 1-based on disk, as STL10's are

    cases = [
        (prefix + "STL10 bytes(pictures)", stl(lambda: _pixels, "data")),
        # **The same bytes without the swap.** Frozen beside the one above so that the
        # difference the swap makes is a value in this file rather than a claim about
        # one — the two agree on shape and on every summary statistic.
        (prefix + "STL10 bytes(no transpose)", stl(lambda: _pixels, "data", False)),
        (prefix + "STL10 bytes(labels, 1-based)", stl(lambda: _labels, "labels")),
    ]

    def moving(split, ratio):
        """`.npy` in, `(clips, frames, 1, H, W)` out. **The split cuts frames**, so
        both halves keep every clip — and this is asked as a shape, because a reader
        that split the clips instead would produce the right rank and half the count.
        """
        def run(L):
            import io
            import os
            import tempfile
            # (frames, clips, H, W), counting up so a mis-ordered axis is visible.
            block = np.arange(6 * 2 * 4 * 4, dtype=np.uint8).reshape(6, 2, 4, 4)
            handle, path = tempfile.mkstemp(suffix=".npy")
            try:
                with os.fdopen(handle, "wb") as out:
                    np.save(out, block)
                if _is_real_torch(L):
                    from torchvision.datasets import MovingMNIST as real
                    made = os.path.join(os.path.dirname(path), "MovingMNIST")
                    os.makedirs(made, exist_ok=True)
                    target = os.path.join(made, "mnist_test_seq.npy")
                    with open(path, "rb") as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    got = real(os.path.dirname(path), split=split,
                               split_ratio=ratio).data
                    os.unlink(target)
                else:
                    data = np.load(path)
                    if split == "train":
                        data = data[:ratio]
                    elif split == "test":
                        data = data[ratio:]
                    got = np.ascontiguousarray(data.transpose(1, 0, 2, 3)[:, :, None])
                del io
            finally:
                os.unlink(path)
            return L.tensor(np.ascontiguousarray(
                np.asarray(_as_numpy(got), dtype=np.float32)))
        return run

    def stl_item():
        """**`.data` holds `(N,C,H,W)` and `__getitem__` gives `(H,W,C)`.** They
        disagree on purpose: torchvision transposes at that line with the comment "so
        that it is consistent with all other datasets", because its other datasets
        hold HWC and this one does not.

        This case exists because the door was wrong while the store was right. On the
        real 105,000 pictures `data`, `labels`, `len` and `classes` all matched across
        four splits and `__getitem__` did not — and nothing frozen here could see it,
        since every other STL10 case reads the array rather than an item.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            raw = (np.arange(2 * 3 * 96 * 96, dtype=np.int64) % 251).astype(np.uint8)
            root = tempfile.mkdtemp()
            try:
                folder = os.path.join(root, "stl10_binary")
                os.makedirs(folder)
                with open(os.path.join(folder, "test_X.bin"), "wb") as out:
                    out.write(raw.tobytes())
                with open(os.path.join(folder, "test_y.bin"), "wb") as out:
                    out.write(bytes([1, 10]))
                if _is_real_torch(L):
                    from torchvision.datasets import STL10 as real
                    cls = real
                else:
                    cls = _vision_datasets(L).STL10
                loaded = cls.__new__(cls)
                loaded.root, loaded.base_folder = root, "stl10_binary"
                loaded.transform = loaded.target_transform = None
                loaded.transforms = None
                loaded.split, loaded.folds = "test", None
                pictures, labels = (loaded._STL10__loadfile("test_X.bin", "test_y.bin")
                                    if _is_real_torch(L)
                                    else loaded._load("test_X.bin", "test_y.bin"))
                loaded.data, loaded.labels = pictures, labels
                picture, target = loaded[1]
                got = np.asarray(picture)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(
                np.concatenate([got.reshape(-1), [target]]).astype(np.float32)))
        return run

    cases += [
        # The item, not the array — `(H,W,C)` on both sides, and the label with it.
        (prefix + "STL10 item(H,W,C)", stl_item()),
        (prefix + "MovingMNIST(whole)", moving(None, 3)),
        (prefix + "MovingMNIST(train)", moving("train", 3)),
        (prefix + "MovingMNIST(test)", moving("test", 3)),
    ]

    def fer(layout, split):
        """**Both sides read the same CSV.** torchvision checks a digest before it
        opens the file, so the digest is handed to it — which is the whole of what
        "torchvision cannot download this" was preventing."""
        def run(L):
            import hashlib
            import os
            import shutil
            import tempfile
            space = " " if layout == "icml" else ""
            head = f"emotion,{space}pixels,{space}Usage"
            rows = [head]
            for i, (emotion, usage) in enumerate(((3, "Training"), (0, "Training"),
                                                  (6, "PublicTest"),
                                                  (2, "PrivateTest"))):
                cells = " ".join(str((i * 13 + k) % 256) for k in range(48 * 48))
                rows.append(f"{emotion},{cells},{usage}")
            body = "\n".join(rows) + "\n"
            root = tempfile.mkdtemp()
            try:
                folder = os.path.join(root, "fer2013")
                os.makedirs(folder)
                name = "icml_face_data.csv" if layout == "icml" else "fer2013.csv"
                with open(os.path.join(folder, name), "w", newline="") as out:
                    out.write(body)
                digest = hashlib.md5(body.encode()).hexdigest()
                if _is_real_torch(L):
                    from torchvision.datasets import FER2013 as real
                    real._RESOURCES = dict(real._RESOURCES)
                    real._RESOURCES[layout] = (name, digest)
                    loaded = real(root, split=split)
                else:
                    mod = _vision_datasets(L).FER2013
                    mod._RESOURCES = dict(mod._RESOURCES)
                    mod._RESOURCES[layout] = (name, digest)
                    loaded = mod(root, split=split)
                stacked = np.stack([np.asarray(loaded[i][0])
                                    for i in range(len(loaded))])
                labels = np.asarray([loaded[i][1] for i in range(len(loaded))])
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(
                np.concatenate([stacked.reshape(len(labels), -1),
                                labels[:, None]], axis=1).astype(np.float32)))
        return run

    cases += [
        (prefix + "FER2013(train)", fer("fer", "train")),
        # **`Usage` decides**, and test is two values rather than one — `PublicTest`
        # and `PrivateTest` both land here. A reader that matched only the first
        # returns half the set and no error.
        (prefix + "FER2013(test, two usages)", fer("fer", "test")),
        # The ICML layout's column names carry **a leading space**, and a reader
        # written against the other file finds out with `KeyError: 'pixels'`.
        (prefix + "FER2013(icml, spaced headers)", fer("icml", "train")),
    ]

    def image_folder(what):
        """`ImageFolder` on a folder written here, against torchvision's own.

        **This is the one dataset in the table whose refusal came off without a line
        of decoding being written.** `DatasetFolder` walks directories and calls a
        function; the codec was never in the class, only in the default that class
        was given. So the whole of what was missing was choosing a loader, and this
        library has two — PNG and PPM.

        Both formats are in the fixture, because walking one and not the other is a
        difference the classes and the labels cannot show. The sizes differ per class
        so that a transposed read lands somewhere visible, and one folder holds two
        pictures while another holds one — a reader that sorted by file rather than
        by directory gives the same count and different labels.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            # **The fixture is written with PIL and read without it.** torchvision's
            # own `ImageFolder` opens through PIL, so the package is already required
            # for this side of the comparison — writing the files with it keeps the
            # bytes torchvision would meet rather than bytes this repository invented.
            root = tempfile.mkdtemp()
            try:
                for name, count in (("cat", 2), ("dog", 2), ("emu", 1)):
                    os.makedirs(os.path.join(root, name))
                    for k in range(count):
                        block = (np.arange(4 * 5 * 3).reshape(4, 5, 3)
                                 + len(name) * 11 + k * 3) % 251
                        _write_png8(os.path.join(root, name, f"{k}.png"), block)
                # A PPM beside the PNGs, in the folder that has only one picture.
                block = (np.arange(3 * 3 * 3).reshape(3, 3, 3) * 7) % 200
                _write_png8(os.path.join(root, "emu", "p.ppm"), block)

                if _is_real_torch(L):
                    from torchvision.datasets import ImageFolder as real
                    loaded = real(root)
                else:
                    loaded = _vision_datasets(L).ImageFolder(root)
                pictures = [np.asarray(loaded[i][0]) for i in range(len(loaded))]
                labels = [loaded[i][1] for i in range(len(loaded))]
                if what == "labels":
                    out = np.asarray(labels + [len(loaded)]
                                     + [loaded.class_to_idx[c] for c in loaded.classes])
                else:
                    out = np.concatenate([p.reshape(-1) for p in pictures])
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(np.asarray(out, dtype=np.float32)))
        return run

    def clevr(split, what):
        """CLEVR, against torchvision's own reader on a folder written here.

        **The refusal on this name was about a format nobody had looked at.** It read
        *its pictures are JPEG or PNG and numpy decodes neither*, and the archive is
        19GB, so which half applied went unchecked. A zip keeps its file list at the
        end and the host serves ranges: 70KB reads the central directory, and all
        100,015 entries are `.png`, `.json` or `.txt`.

        The fixture's scenes file is **deliberately out of order**. The pictures are
        paired to their counts by filename, and a reader that zipped the sorted
        listing against the JSON as it comes gives every picture the wrong label with
        every shape still right.
        """
        def run(L):
            import json as _js
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "clevr", "CLEVR_v1.0")
                for name, count in (("train", 3), ("test", 2)):
                    folder = os.path.join(base, "images", name)
                    os.makedirs(folder)
                    for k in range(count):
                        block = (np.arange(4 * 5 * 3).reshape(4, 5, 3) + k * 7) % 251
                        _write_png8(os.path.join(folder, f"CLEVR_{name}_{k:06d}.png"), block)
                os.makedirs(os.path.join(base, "scenes"))
                scenes = {"scenes": [
                    {"image_filename": "CLEVR_train_000002.png", "objects": [1, 2, 3, 4, 5]},
                    {"image_filename": "CLEVR_train_000000.png", "objects": [1, 2]},
                    {"image_filename": "CLEVR_train_000001.png", "objects": [1, 2, 3]},
                ]}
                with open(os.path.join(base, "scenes",
                                       "CLEVR_train_scenes.json"), "w") as handle:
                    _js.dump(scenes, handle)

                if _is_real_torch(L):
                    from torchvision.datasets import CLEVRClassification as real
                    loaded = real(root, split=split)
                else:
                    loaded = _vision_datasets(L).CLEVRClassification(root, split=split)
                pictures = [np.asarray(loaded[i][0]) for i in range(len(loaded))]
                labels = [loaded[i][1] for i in range(len(loaded))]
                if what == "labels":
                    # **`test` has no scenes file and its labels are `None`.** Frozen as
                    # -1 so the absence is a value rather than a hole; a reader that
                    # answered 0 there would train on a class that does not exist.
                    out = np.asarray([-1 if v is None else v for v in labels]
                                     + [len(loaded)])
                else:
                    out = np.concatenate([p.reshape(-1) for p in pictures])
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(np.asarray(out, dtype=np.float32)))
        return run

    def sst2(split):
        """`RenderedSST2` against torchvision's own, on a folder written here.

        **Its refusal read *as above — a codec* and the extension was in torchvision's
        constructor**: `make_dataset(..., extensions=("png",))`. Nobody had read it.

        The split asked is `val`, because that is the one with a trap: the folder is
        called `valid`, and a reader using the split name finds nothing and reports an
        empty dataset rather than a missing directory. The counts per class differ so
        that a reader pairing labels by position rather than by folder shows up.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "rendered-sst2")
                for folder, counts in (("train", (2, 1)), ("valid", (1, 1)),
                                       ("test", (1, 2))):
                    for name, n in zip(("negative", "positive"), counts):
                        d = os.path.join(base, folder, name)
                        os.makedirs(d)
                        for k in range(n):
                            block = (np.arange(3 * 4 * 3).reshape(3, 4, 3)
                                     + k * 5 + len(name)) % 251
                            _write_png8(os.path.join(d, f"{k}.png"), block)
                if _is_real_torch(L):
                    from torchvision.datasets import RenderedSST2 as real
                    loaded = real(root, split=split)
                else:
                    loaded = _vision_datasets(L).RenderedSST2(root, split=split)
                pictures = [np.asarray(loaded[i][0]) for i in range(len(loaded))]
                labels = [loaded[i][1] for i in range(len(loaded))]
                out = np.concatenate(
                    [np.concatenate([p.reshape(-1) for p in pictures]),
                     np.asarray(labels),
                     [len(loaded), loaded.class_to_idx["negative"],
                      loaded.class_to_idx["positive"]]])
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(np.asarray(out, dtype=np.float32)))
        return run

    def sintel(pass_name):
        """`Sintel` against torchvision's own, on a tree written here.

        **Its row read *a codec and then another format*, and only one of the two was
        there.** The pictures are PNG, read here already; the other format is `.flo` —
        a magic number, two little-endian integers and a block of float32, which is a
        container rather than a codec and is fifteen lines.

        The fixture has **two scenes of different lengths**, three frames and two, so
        it yields 2 + 1 pairs per pass. A reader that paired across the scene boundary
        would give 4 and ask the model to explain a cut; a reader that walked the flow
        list once for `both` would run out halfway.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "Sintel")
                scenes = (("alley", 3), ("bamboo", 2))
                for name in ("clean", "final"):
                    for scene, n in scenes:
                        d = os.path.join(base, "training", name, scene)
                        os.makedirs(d)
                        for k in range(n):
                            block = (np.arange(3 * 4 * 3).reshape(3, 4, 3)
                                     + k * 9
                                     + (0 if name == "clean" else 101)) % 251
                            _write_png8(os.path.join(d, f"frame_{k:04d}.png"), block)
                for scene, n in scenes:
                    d = os.path.join(base, "training", "flow", scene)
                    os.makedirs(d)
                    for k in range(n - 1):
                        field = ((np.arange(3 * 4 * 2).reshape(3, 4, 2) + k)
                                 * 0.25).astype("<f4")
                        with open(os.path.join(d, f"frame_{k:04d}.flo"), "wb") as fh:
                            fh.write(b"PIEH" + (4).to_bytes(4, "little")
                                     + (3).to_bytes(4, "little") + field.tobytes())

                if _is_real_torch(L):
                    from torchvision.datasets import Sintel as real
                    loaded = real(root, split="train", pass_name=pass_name)
                else:
                    loaded = _vision_datasets(L).Sintel(root, split="train",
                                                        pass_name=pass_name)
                # **torchvision walks the scenes in `os.listdir` order** — the
                # disk's order, not the dataset's. APFS hands back names sorted and
                # ext4 hands them back hashed, so torchvision's own answer changes
                # with the filesystem; borchvision sorts, and the two agree on one
                # machine and not on the other. The items are therefore compared as a
                # set: the count, then the items in an order neither side chose. What
                # the fixture is for survives it — a reader that paired across the
                # scene boundary would have four items where the answer has three,
                # and no ordering hides a count.
                items = []
                for i in range(len(loaded)):
                    first, second, flow = loaded[i]
                    items.append(np.concatenate([
                        np.asarray(first).reshape(-1).astype(np.float32),
                        np.asarray(second).reshape(-1).astype(np.float32),
                        np.asarray(flow).reshape(-1).astype(np.float32)]))
                items.sort(key=lambda item: item.tobytes())
                out = np.concatenate(
                    [np.asarray([len(loaded)], dtype=np.float32)] + items)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def _png16(path, arr):
        """PIL cannot **save** sixteen-bit colour, so KITTI's flow is written by hand.

        Neither can it read it — it narrows to eight, and `(x - 2**15) / 64` on the
        narrowed samples still returns a flow field of the right shape.
        """
        import os
        import struct
        import zlib
        height, width = arr.shape[:2]
        # `len(arr.shape)` and not `arr.ndim` for the reason written at the
        # `_stereo_out` neighbour above: this file is parsed for the names its
        # cases ask about, and a fixture's numpy `.ndim` is indistinguishable
        # from a tensor's.
        channels = 1 if len(arr.shape) == 2 else arr.shape[2]
        rows = arr.reshape(height, width * channels)
        raw = b"".join(b"\x00" + rows[y].astype(">u2").tobytes() for y in range(height))

        def chunk(kind, body):
            return (struct.pack(">I", len(body)) + kind + body
                    + struct.pack(">I", zlib.crc32(kind + body)))

        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = struct.pack(">IIBBBBB", width, height, 16,
                             {1: 0, 3: 2, 4: 6}[channels], 0, 0, 0)
        with open(path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                         + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

    def _png8(path, arr):
        """An eight-bit PNG, **written by hand like the sixteen-bit one above.**

        Not through PIL. These fixtures run in three places, and one of them is Pyodide,
        where there is no PIL — measured: fifty-one dataset cases failed on the binding
        with `No module named 'PIL'` while numpy and the browser passed. A fixture that
        needs a library the case does not is a case that only two of three
        implementations can be asked.

        The format costs nothing to write: a signature, an `IHDR` saying the size and
        the depth, one `zlib` stream of rows each prefixed by a filter byte, and `IEND`.
        Filter `0` means *no filter*, which a reader has to handle anyway.
        """
        import os
        import struct
        import zlib
        arr = np.asarray(arr)
        height, width = arr.shape[:2]
        # `len(arr.shape)` and not `arr.ndim`, for the reason written twice above:
        # this file is parsed for the names its cases ask about.
        channels = 1 if len(arr.shape) == 2 else arr.shape[2]
        rows = arr.reshape(height, width * channels).astype(np.uint8)
        raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(height))

        def chunk(kind, body):
            return (struct.pack(">I", len(body)) + kind + body
                    + struct.pack(">I", zlib.crc32(kind + body)))

        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = struct.pack(">IIBBBBB", width, height, 8,
                             {1: 0, 3: 2, 4: 6}[channels], 0, 0, 0)
        with open(path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                         + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

    def _stereo_out(loaded):
        """Count first, then every item flattened — including the `None`s, which are
        written as a length so that a split that should have no disparity cannot pass
        by handing back an array of zeros."""
        parts = [np.asarray([len(loaded)], dtype=np.float32)]
        for i in range(len(loaded)):
            for piece in loaded[i]:
                if piece is None:
                    parts.append(np.asarray([-1.0], dtype=np.float32))
                else:
                    parts.append(np.asarray(piece).reshape(-1).astype(np.float32))
        return np.concatenate(parts)

    _H, _W = 3, 4

    def _rgb(seed):
        return ((np.arange(_H * _W * 3).reshape(_H, _W, 3) * 7 + seed) % 251)

    def kitti_flow(split):
        """`KittiFlow` against torchvision's own.

        **The frames pair by suffix, not by order**: `*_10.png` is every first frame
        and `*_11.png` every second. The fixture has two scenes, so a reader that
        walked the directory in order would pair scene 1's second frame with scene 2's
        first — four files in one sorted list, three pairs, and every one of them
        loads.

        The flow is a **sixteen-bit colour PNG**, two channels of `x * 64 + 2**15` and
        a validity flag in the third. On PIL's eight-bit answer that arithmetic still
        runs; the numbers here are chosen so it would not agree.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "KittiFlow", split + "ing")
                for i in (1, 2):
                    _png8(os.path.join(base, "image_2", f"{i:06d}_10.png"), _rgb(i))
                    _png8(os.path.join(base, "image_2", f"{i:06d}_11.png"), _rgb(i + 50))
                    field = np.zeros((_H, _W, 3), np.uint16)
                    ramp = np.arange(_H * _W).reshape(_H, _W)
                    field[:, :, 0] = (ramp * 64 + 2 ** 15 + i) % 65535
                    field[:, :, 1] = (2 ** 15 - ramp * 640) % 65535
                    field[:, :, 2] = ramp % 2
                    _png16(os.path.join(base, "flow_occ", f"{i:06d}_10.png"), field)
                if _is_real_torch(L):
                    from torchvision.datasets import KittiFlow as real
                    loaded = real(root, split=split)
                else:
                    loaded = _vision_datasets(L).KittiFlow(root, split=split)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def hd1k():
        """`HD1K`. **Each sequence stops one frame short**, and that is the whole of it.

        The fixture has two sequences of three frames, so the answer is 2 + 2 = 4
        pairs. A reader that flattened them into one list would give 5 and the extra
        one would pair the end of sequence 0 with the start of sequence 1, holding a
        flow file that belongs to neither.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "hd1k")
                for seq in (0, 1):
                    for k in range(3):
                        _png8(os.path.join(base, "hd1k_input", "image_2",
                                           f"{seq:06d}_{k:04d}.png"), _rgb(seq * 10 + k))
                        field = np.zeros((_H, _W, 3), np.uint16)
                        ramp = np.arange(_H * _W).reshape(_H, _W)
                        field[:, :, 0] = (ramp * 64 + 2 ** 15 + k) % 65535
                        field[:, :, 1] = (2 ** 15 - ramp * 320 - seq) % 65535
                        field[:, :, 2] = (ramp + k) % 2
                        _png16(os.path.join(base, "hd1k_flow_gt", "flow_occ",
                                            f"{seq:06d}_{k:04d}.png"), field)
                if _is_real_torch(L):
                    from torchvision.datasets import HD1K as real
                    loaded = real(root)
                else:
                    loaded = _vision_datasets(L).HD1K(root)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def kitti_stereo(year):
        """`Kitti2012Stereo` and `Kitti2015Stereo`. **The disparity is sixteen-bit grey
        divided by 256** — the case that needed the PNG reader fixed rather than
        extended, because a reader that narrowed it to eight would divide a number that
        still looks like a distance.

        2012 pairs `colored_0` with `colored_1` and ships one disparity; 2015 pairs
        `image_2` with `image_3` and ships two, of which the item carries the left.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                ramp = np.arange(_H * _W).reshape(_H, _W)
                if year == 2012:
                    base = os.path.join(root, "Kitti2012", "training")
                    for i in (1, 2):
                        _png8(os.path.join(base, "colored_0", f"{i:06d}_10.png"), _rgb(i))
                        _png8(os.path.join(base, "colored_1", f"{i:06d}_10.png"), _rgb(i + 3))
                        _png16(os.path.join(base, "disp_noc", f"{i:06d}.png"),
                               (ramp * 1000 + i).astype(np.uint16))
                else:
                    base = os.path.join(root, "Kitti2015", "training")
                    for i in (1, 2):
                        _png8(os.path.join(base, "image_2", f"{i:06d}_10.png"), _rgb(i))
                        _png8(os.path.join(base, "image_3", f"{i:06d}_10.png"), _rgb(i + 3))
                        for offset, side in ((0, "disp_occ_0"), (7, "disp_occ_1")):
                            _png16(os.path.join(base, side, f"{i:06d}_10.png"),
                                   (ramp * 900 + i + offset).astype(np.uint16))
                name = f"Kitti{year}Stereo"
                if _is_real_torch(L):
                    import torchvision.datasets as real
                    loaded = getattr(real, name)(root)
                else:
                    loaded = getattr(_vision_datasets(L), name)(root)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def instereo2k():
        """`InStereo2k` — **one scene per directory**, and the disparity divided by
        1024 rather than 256. The number is the dataset's, not a convention, and the
        fixture's values are large enough that the wrong divisor is not a rounding."""
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "InStereo2k", "train")
                ramp = np.arange(_H * _W).reshape(_H, _W)
                for k, scene in enumerate(("aaa", "bbb")):
                    _png8(os.path.join(base, scene, "left.png"), _rgb(k))
                    _png8(os.path.join(base, scene, "right.png"), _rgb(k + 4))
                    for offset, side in ((0, "left_disp.png"), (5, "right_disp.png")):
                        _png16(os.path.join(base, scene, side),
                               (ramp * 700 + offset).astype(np.uint16))
                if _is_real_torch(L):
                    from torchvision.datasets import InStereo2k as real
                    loaded = real(root)
                else:
                    loaded = _vision_datasets(L).InStereo2k(root)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def sintel_stereo(pass_name):
        """`SintelStereo`. Two things here are PNG files that are not pictures:

        - **the disparity is packed into a colour**, `r * 4 + g / 2**6 + b / 2**14`.
          Reading the red channel alone gives a map that is coarse and plausible, so
          the fixture's green and blue are not zero.
        - **the validity mask is two files**, occlusion and out-of-frame, and a pixel
          is valid only where both are zero. The two disagree in the fixture, so
          taking either alone is a different mask.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "Sintel", "training")
                ramp = np.arange(_H * _W).reshape(_H, _W)
                for name in ("final", "clean"):
                    for scene in ("alley", "bamboo"):
                        for k in range(2):
                            _png8(os.path.join(base, f"{name}_left", scene,
                                               f"f{k}.png"), _rgb(k + len(name)))
                            _png8(os.path.join(base, f"{name}_right", scene,
                                               f"f{k}.png"), _rgb(k + 9))
                for scene in ("alley", "bamboo"):
                    for k in range(2):
                        _png8(os.path.join(base, "disparities", scene, f"f{k}.png"),
                              _rgb(k * 3 + 1))
                        _png8(os.path.join(base, "occlusions", scene, f"f{k}.png"),
                              (ramp % 2 * 255))
                        _png8(os.path.join(base, "outofframe", scene, f"f{k}.png"),
                              (ramp % 3 * 255))
                if _is_real_torch(L):
                    from torchvision.datasets import SintelStereo as real
                    loaded = real(root, pass_name=pass_name)
                else:
                    loaded = _vision_datasets(L).SintelStereo(root, pass_name=pass_name)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def _ppm(path, seed):
        import os
        _write_ppm8(path, ((np.arange(_H * _W * 3).reshape(_H, _W, 3) * 5 + seed)
                           % 251))

    def _pfm(path, channels, seed):
        """A `.pfm` written little-endian — **the scale's sign is the byte order**, and
        `-1.0` is how the format says little. A reader taking the machine's gets
        numbers that are enormous rather than absurd."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        block = ((np.arange(_H * _W * channels).reshape(_H, _W, channels) + seed)
                 * 0.5 - 3).astype("<f4")
        header = ((b"PF\n" if channels == 3 else b"Pf\n")
                  + f"{_W} {_H}\n".encode() + b"-1.0\n")
        with open(path, "wb") as handle:
            handle.write(header + block.tobytes())

    def carla_stereo():
        """`CarlaStereo`. Two things, and the second is why this fixture is uneven.

        **The disparity is stored signed and used positive** — the sign is the
        direction the pixel moved, which the file keeps and the dataset does not want.
        The values here straddle zero, so a reader that skipped the `abs` gets half a
        map of negative distances that are all finite.

        **The left file is three-channel `PF` and the right is one-channel `Pf`**,
        because `.pfm` holds either and the number of channels sliced out is what
        distinguishes a disparity from a flow. Written both ways it was measured that
        on a one-channel file the wrong slice is *invisible* — `[:2]` of one channel is
        one channel — so a fixture that used `Pf` twice would pass with the flow
        reader's default and say nothing.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "carla-highres", "trainingF")
                for k, scene in enumerate(("s1", "s2")):
                    _png8(os.path.join(base, scene, "im0.png"), _rgb(k))
                    _png8(os.path.join(base, scene, "im1.png"), _rgb(k + 5))
                    _pfm(os.path.join(base, scene, "disp0GT.pfm"), 3, k)
                    _pfm(os.path.join(base, scene, "disp1GT.pfm"), 1, k + 2)
                if _is_real_torch(L):
                    from torchvision.datasets import CarlaStereo as real
                    loaded = real(root)
                else:
                    loaded = _vision_datasets(L).CarlaStereo(root)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def cre_stereo():
        """`CREStereo`. The last of this family whose row said *a codec*, and the codec
        was not its wall.

        **The disparity is a PNG divided by thirty-two**, which is how the archive keeps
        a fractional map in whole numbers. Read without the division every value is
        thirty-two times too large — finite, plausible, and wrong by a constant.

        **The four scene directories are read in torchvision's order**, not sorted, and
        that order is `shapenet, reflective, tree, hole` — which is **not** the order
        torchvision's own docstring draws. The fixture puts one pair in each and
        gives them different values, so a reader that let `glob` sort them pairs the
        same pictures under different indices and the count still comes out right.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "CREStereo")
                for k, scene in enumerate(("shapenet", "reflective", "tree", "hole")):
                    _png8(os.path.join(base, scene, "img1_left.jpg"), _rgb(k))
                    _png8(os.path.join(base, scene, "img1_right.jpg"), _rgb(k + 9))
                    # **Whole numbers that are not multiples of thirty-two**, so the
                    # division leaves a fraction and a reader that skipped it cannot
                    # land on the same values by rounding.
                    left = (np.arange(_H * _W).reshape(_H, _W) * 3 + k * 7) % 200
                    _png8(os.path.join(base, scene, "img1_left.disp.png"), left)
                    _png8(os.path.join(base, scene, "img1_right.disp.png"),
                          (left + 11) % 200)
                if _is_real_torch(L):
                    from torchvision.datasets import CREStereo as real
                    loaded = real(root)
                else:
                    loaded = _vision_datasets(L).CREStereo(root)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def falling_things(variant):
        """`FallingThingsStereo`. **The file on disk is a depth and the answer is a
        disparity**, converted with the camera's own focal length out of a JSON beside
        the picture.

        Two things a fixture can hide and this one does not:

        - **The two variants are nested at different depths** — `single` one directory
          deeper than `mixed`. Both are written, so `both` is the case that catches one
          pattern used for the two.
        - **The two scenes carry different focal lengths.** With one focal everywhere,
          a reader that took a constant instead of reading the JSON agrees exactly.
        """
        def run(L):
            import json
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "FallingThings")
                places = [("single", ("s1", "kitchen"), 320.0),
                          ("single", ("s2", "office"), 411.5),
                          ("mixed", ("m1",), 275.25)]
                for k, (kind, parts, focal) in enumerate(places):
                    where = os.path.join(base, kind, *parts)
                    _png8(os.path.join(where, "000000.left.jpg"), _rgb(k))
                    _png8(os.path.join(where, "000000.right.jpg"), _rgb(k + 6))
                    # **No zero anywhere in the depth** — it is the divisor, and a zero
                    # would put an infinity in the answer that compares equal to
                    # nothing.
                    depth = (np.arange(_H * _W).reshape(_H, _W) * 5 + k * 3) % 200 + 1
                    _png8(os.path.join(where, "000000.left.depth.png"), depth)
                    _png8(os.path.join(where, "000000.right.depth.png"),
                          (depth + 17) % 200 + 1)
                    with open(os.path.join(where, "_camera_settings.json"),
                              "w") as handle:
                        json.dump({"camera_settings": [
                            {"intrinsic_settings": {"fx": focal}}]}, handle)
                if _is_real_torch(L):
                    from torchvision.datasets import FallingThingsStereo as real
                    loaded = real(root, variant=variant)
                else:
                    loaded = _vision_datasets(L).FallingThingsStereo(
                        root, variant=variant)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def eth3d_stereo(split):
        """`ETH3DStereo`. **The pictures and the ground truth live in different
        directories**, zipped by position because the scene names match.

        Its mask is a PNG read as a boolean — **non-zero is valid**, the opposite of
        `SintelStereo`'s zero-is-valid, which is why neither is written as the usual
        way. On `test` there is no ground truth at all and the `None`s are in the
        answer.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "ETH3D")
                pictures = "two_view_training" if split == "train" else "two_view_test"
                for k, scene in enumerate(("s1", "s2")):
                    _png8(os.path.join(base, pictures, scene, "im0.png"), _rgb(k))
                    _png8(os.path.join(base, pictures, scene, "im1.png"), _rgb(k + 4))
                    _pfm(os.path.join(base, "two_view_training_gt", scene,
                                      "disp0GT.pfm"), 1, k)
                    _png8(os.path.join(base, "two_view_training_gt", scene,
                                       "mask0nocc.png"),
                          np.arange(_H * _W).reshape(_H, _W) % 2 * 255)
                if _is_real_torch(L):
                    from torchvision.datasets import ETH3DStereo as real
                    loaded = real(root, split=split)
                else:
                    loaded = _vision_datasets(L).ETH3DStereo(root, split=split)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def scene_flow(variant):
        """`SceneFlowStereo`. **The three variants differ only in how deep the scene
        directories nest** — `Monkaa` one level, the other two three — and that is the
        whole of what `variant` selects. Both are built here, so a glob written for one
        depth is a count of zero rather than a wrong number."""
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "SceneFlow", variant)
                middle = ("",) if variant == "Monkaa" else ("TRAIN", "A")
                for k, scene in enumerate(("a", "b")):
                    for eye, seed in (("left", k), ("right", k + 3)):
                        parts = [one for one in middle if one] + [scene, eye]
                        _png8(os.path.join(base, "frames_cleanpass", *parts, "f1.png"),
                              _rgb(seed))
                        _pfm(os.path.join(base, "disparity", *parts, "f1.pfm"), 1, seed)
                if _is_real_torch(L):
                    from torchvision.datasets import SceneFlowStereo as real
                    loaded = real(root, variant=variant)
                else:
                    loaded = _vision_datasets(L).SceneFlowStereo(root, variant=variant)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def flying_things(pass_name, camera):
        """`FlyingThings3D`. Three loops, and **direction is the one that is not a
        filter**: `into_future` pairs frame `i` with `i + 1` and takes flow `i`, while
        `into_past` pairs `i + 1` with `i` — the frames swapped, not the list reversed
        — and takes flow `i + 1`.

        Both directions are always walked, so the count is twice what the frames
        suggest. The fixture gives the two directions different flow values, so a
        reader that paired `into_past` the same way agrees on every shape and on no
        number.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "FlyingThings3D")
                for scene in ("0000", "0001"):
                    for eye in ("left", "right"):
                        for k in range(3):
                            _png8(os.path.join(base, "frames_cleanpass", "TRAIN", "A",
                                               scene, eye, f"{k:04d}.png"), _rgb(k))
                            _png8(os.path.join(base, "frames_finalpass", "TRAIN", "A",
                                               scene, eye, f"{k:04d}.png"), _rgb(k + 20))
                            for offset, direction in ((0, "into_future"),
                                                      (7, "into_past")):
                                _pfm(os.path.join(base, "optical_flow", "TRAIN", "A",
                                                  scene, direction, eye,
                                                  f"{k:04d}.pfm"), 3, k + offset)
                if _is_real_torch(L):
                    from torchvision.datasets import FlyingThings3D as real
                    loaded = real(root, pass_name=pass_name, camera=camera)
                else:
                    loaded = _vision_datasets(L).FlyingThings3D(
                        root, pass_name=pass_name, camera=camera)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def flying_chairs(split):
        """`FlyingChairs` — **one flat directory and a text file that cuts it in two.**

        The pictures pair by *position* in one sorted list: item `i` is `images[2i]`
        and `images[2i + 1]`, nothing in the names matched. The split file is one
        integer per pair, `1` for train and `2` for val; the fixture writes `1 2 1`, so
        the two splits are different lengths and a reader that ignored the file would
        give three for both.

        The codec here is **PPM**, which was read before the row that refused this
        dataset was written.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "FlyingChairs")
                for i in range(1, 4):
                    _ppm(os.path.join(base, "data", f"{i:05d}_img1.ppm"), i)
                    _ppm(os.path.join(base, "data", f"{i:05d}_img2.ppm"), i + 30)
                    field = (np.arange(_H * _W * 2).reshape(_H, _W, 2) * 0.25
                             + i).astype("<f4")
                    with open(os.path.join(base, "data", f"{i:05d}_flow.flo"), "wb") as fh:
                        fh.write(b"PIEH" + (_W).to_bytes(4, "little")
                                 + (_H).to_bytes(4, "little") + field.tobytes())
                with open(os.path.join(base, "FlyingChairs_train_val.txt"), "w") as fh:
                    fh.write("1\n2\n1\n")
                if _is_real_torch(L):
                    from torchvision.datasets import FlyingChairs as real
                    loaded = real(root, split=split)
                else:
                    loaded = _vision_datasets(L).FlyingChairs(root, split=split)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def middlebury(split, calibration):
        """`Middlebury2014Stereo` — the last of the fourteen that was not JPEG.

        Its row read *a calibration file per scene to parse*, **written without opening
        one.** `calib.txt` sits in every scene directory and is never read: the
        `calibration` argument is a **directory suffix**, `-perfect` or `-imperfect`,
        chosen before the glob. The fixture writes a `calib.txt` anyway, so a reader
        that started parsing it would have something to be wrong about.

        **The `.pfm` stores unmatched pixels as `inf`**, and one is written here.
        Leaving it turns any average into `inf`; zeroing it without the mask says the
        pixel is touching the camera rather than unknown. Both readings survive a shape
        check, so the mask is in the answer.

        `use_ambient_views` is left off: it picks at random among three exposures of
        the right frame, and a case that turned it on would compare two draws from two
        libraries' random modules.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "Middlebury2014")
                if split == "train":
                    for scene in ("Adirondack", "Piano"):
                        for k, name in enumerate(("perfect", "imperfect")):
                            folder = os.path.join(base, "train", f"{scene}-{name}")
                            _png8(os.path.join(folder, "im0.png"), _rgb(k))
                            _png8(os.path.join(folder, "im1.png"), _rgb(k + 5))
                            _pfm_inf(os.path.join(folder, "disp0.pfm"), k)
                            _pfm(os.path.join(folder, "disp1.pfm"), 1, k + 2)
                            with open(os.path.join(folder, "calib.txt"), "w") as fh:
                                fh.write("cam0=[1 0 0]\n")
                else:
                    for scene in ("Plants", "Hoops"):
                        folder = os.path.join(base, "test", scene)
                        _png8(os.path.join(folder, "im0.png"), _rgb(1))
                        _png8(os.path.join(folder, "im1.png"), _rgb(2))
                if _is_real_torch(L):
                    from torchvision.datasets import Middlebury2014Stereo as real
                    loaded = real(root, split=split, calibration=calibration)
                else:
                    loaded = _vision_datasets(L).Middlebury2014Stereo(
                        root, split=split, calibration=calibration)
                out = _stereo_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def _pfm_inf(path, seed):
        """A `.pfm` with one **infinite** sample — what Middlebury writes where the two
        cameras never matched."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        block = ((np.arange(_H * _W).reshape(_H, _W, 1) + seed) * 0.5 - 3).astype("<f4")
        block[0, 0, 0] = np.inf
        header = b"Pf\n" + f"{_W} {_H}\n".encode() + b"-1.0\n"
        with open(path, "wb") as handle:
            handle.write(header + block.tobytes())

    def _bmp(path, arr):
        """A grey BMP, **written by hand for the reason `_write_png8` is** — PIL is not
        in Pyodide, and a fixture that needs it is a case only two of three
        implementations can be asked.

        Three things the format needs and a reader has to undo: the rows run **bottom to
        top**, each is padded to four bytes, and eight-bit needs a palette — an identity
        greyscale one, which is what PIL writes for mode `L` and what makes the reader
        hand back one channel.
        """
        import os
        import struct
        arr = np.asarray(arr).astype(np.uint8)
        height, width = arr.shape[:2]
        stride = (width + 3) // 4 * 4
        rows = b"".join(arr[y].tobytes() + b"\x00" * (stride - width)
                        for y in range(height - 1, -1, -1))
        palette = b"".join(bytes((i, i, i, 0)) for i in range(256))
        offset = 14 + 40 + len(palette)
        header = (b"BM" + struct.pack("<IHHI", offset + len(rows), 0, 0, offset)
                  + struct.pack("<IiiHHIIiiII", 40, width, height, 1, 8, 0,
                                len(rows), 0, 0, 256, 0))
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(header + palette + rows)

    def kitti(train):
        """`Kitti`'s 2D object set. Its row read *as above — a codec* and **the
        pictures are PNG**, in the same `image_2` directory this library already reads
        for `KittiFlow`.

        The label is fifteen space-separated fields parsed by position, and
        **`occluded` is a level rather than a fraction** — the one integer among the
        floats. The fixture gives it values that differ per item, so a reader that
        floated it agrees on every number and on no type; the answer here carries the
        parsed fields flattened, and a separate pytest checks the types.

        `train=False` has **no labels at all** and the target is `None`, written into
        the answer as a length so an empty list cannot pass for it.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                where = "training" if train else "testing"
                base = os.path.join(root, "Kitti", "raw", where)
                for i in (1, 2):
                    _png8(os.path.join(base, "image_2", f"{i:06d}.png"), _rgb(i))
                    if train:
                        os.makedirs(os.path.join(base, "label_2"), exist_ok=True)
                        with open(os.path.join(base, "label_2",
                                               f"{i:06d}.txt"), "w") as fh:
                            fh.write(f"Car 0.00 {i} -1.57 1.1 2.2 3.3 4.4 1.5 1.6 "
                                     "4.0 1.8 1.9 8.0 -1.2\n")
                            fh.write(f"Pedestrian 0.5 {i % 3} 0.1 5.5 6.6 7.7 8.8 "
                                     "1.7 0.6 0.9 2.8 1.5 9.0 0.4\n")
                if _is_real_torch(L):
                    from torchvision.datasets import Kitti as real
                    loaded = real(root, train=train)
                else:
                    loaded = _vision_datasets(L).Kitti(root, train=train)
                parts = [np.asarray([len(loaded)], dtype=np.float32)]
                items = []
                for i in range(len(loaded)):
                    picture, target = loaded[i]
                    numbers = [np.asarray(picture).reshape(-1).astype(np.float32)]
                    if target is None:
                        numbers.append(np.asarray([-1.0], dtype=np.float32))
                    else:
                        for box in target:
                            numbers.append(np.asarray(
                                [box["truncated"], box["occluded"], box["alpha"],
                                 *box["bbox"], *box["dimensions"], *box["location"],
                                 box["rotation_y"]], dtype=np.float32))
                    items.append(np.concatenate(numbers))
                items.sort(key=lambda item: item.tobytes())
                out = np.concatenate(parts + items)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def photo_tour(train):
        """`PhotoTour`. Its row read *as above — a codec* and the sheets are **BMP** —
        a header, a header, a palette and the rows, with no compression involved.

        **Each sheet is a grid of patches**, read left to right and then down, and the
        patch index is a position in that traversal. The fixture has two sheets of four
        patches each with different contents, so a reader that took the sheets in
        directory order or the patches column-first numbers them differently and still
        returns patches.

        `train=True` gives one patch and `train=False` gives **two patches and whether
        they match** — a different task off the same bytes. The match is a
        *comparison*: each line names two patches and the point each belongs to, and
        no field holds the answer. The fixture's three lines are yes, no, yes.
        """
        def run(L):
            import os
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            name = "liberty"
            try:
                folder = os.path.join(root, name)
                for k in range(2):
                    sheet = ((np.arange(128 * 128).reshape(128, 128) * 7 + k * 31)
                             % 251)
                    _bmp(os.path.join(folder, f"patches{k:04d}.bmp"), sheet)
                with open(os.path.join(folder, "info.txt"), "w") as fh:
                    for i in range(8):
                        fh.write(f"{i // 2} 0\n")
                with open(os.path.join(folder, "m50_100000_100000_0.txt"), "w") as fh:
                    fh.write("0 0 0 1 0 0\n2 1 0 5 2 0\n3 1 0 4 1 0\n")
                if _is_real_torch(L):
                    from torchvision.datasets import PhotoTour as real
                    loaded = real(root, name, train=train)
                else:
                    loaded = _vision_datasets(L).PhotoTour(root, name, train=train)
                parts = [np.asarray([len(loaded)], dtype=np.float32)]
                for i in range(len(loaded)):
                    item = loaded[i]
                    if train:
                        parts.append(np.asarray(item).reshape(-1).astype(np.float32))
                    else:
                        parts += [np.asarray(item[0]).reshape(-1).astype(np.float32),
                                  np.asarray(item[1]).reshape(-1).astype(np.float32),
                                  np.asarray([float(item[2])], dtype=np.float32)]
                parts.append(np.asarray(loaded.labels, dtype=np.float32).reshape(-1))
                parts.append(np.asarray(loaded.matches, dtype=np.float32).reshape(-1))
                out = np.concatenate(parts)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def _pic(path, seed, suffix=".png"):
        """**A picture named `.jpg` whose bytes are PNG.**

        Not a trick — it is the difference these six turn on. `Food101`, `SUN397` and
        `FGVCAircraft` build their paths by appending `.jpg`, so the name has to be
        that; both libraries then dispatch on the first bytes, torchvision through PIL
        and this one through `_folder_loader`. The two `ImageFolder` subclasses filter
        by extension *before* any loader runs, which is why their fixture files are
        named `.png` and why that difference is written into their classes.
        """
        _write_png8(path, _rgb(seed))

    def _labelled_out(loaded):
        """Count, class list length, then every (picture, label) — with the labels
        kept as numbers so a class list built in a different order shows up."""
        parts = [np.asarray([len(loaded), len(loaded.classes)], dtype=np.float32)]
        for i in range(len(loaded)):
            picture, label = loaded[i]
            parts.append(np.asarray(picture).reshape(-1).astype(np.float32))
            parts.append(np.asarray([float(label)], dtype=np.float32))
        return np.concatenate(parts)

    def _folder_case(build, make):
        def run(L):
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                build(root)
                loaded = make(L, root)
                out = _labelled_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def _made(L, name):
        if _is_real_torch(L):
            import torchvision.datasets as real
            return getattr(real, name)
        return getattr(_vision_datasets(L), name)

    def country211(split):
        """`Country211`. **`ImageFolder` pointed at one of three directories** — which
        is the shape `ImageFolder` itself was let in for, and torchvision's own
        `loader=` parameter says so. The three splits hold the same countries, so a
        reader that took the wrong one gives the right classes and the wrong pictures.
        """
        import os
        def build(root):
            # **The three splits get different pictures**, and by index rather than by
            # `len(name)` — `train` and `valid` are both five letters, so the first
            # version of this fixture gave them identical bytes and a reader that
            # ignored `split` passed. Measured by mutation, not noticed by reading.
            for turn, name in enumerate(("train", "valid", "test")):
                for k, country in enumerate(("AD", "ZW")):
                    _pic(os.path.join(root, "country211", name, country,
                                      f"{k}.png"), k + turn * 17)
        return _folder_case(build, lambda L, root: _made(L, "Country211")(
            root, split=split))

    def eurosat():
        """`EuroSAT`, RGB version — `ImageFolder` at `eurosat/2750`. The multispectral
        release is a different dataset and this is not it."""
        import os
        def build(root):
            for k, kind in enumerate(("Forest", "River")):
                for i in range(2):
                    _pic(os.path.join(root, "eurosat", "2750", kind,
                                      f"{kind}_{i}.png"), k * 3 + i)
        return _folder_case(build, lambda L, root: _made(L, "EuroSAT")(root))

    def dtd(split, partition):
        """`DTD`. **The partition is not a split**: it chooses which of ten shufflings
        of the same images `split` refers to. The fixture's partition 2 is partition 1
        reversed, so a reader that ignored the argument returns a dataset of the right
        size drawn from the right pictures — and cross-validating on it measures
        nothing. The classes come from the split file's own paths.
        """
        import os
        def build(root):
            base = os.path.join(root, "dtd", "dtd")
            for k, cls in enumerate(("banded", "zigzagged")):
                for i in range(2):
                    _pic(os.path.join(base, "images", cls, f"{cls}_{i}.jpg"),
                         k * 5 + i)
            os.makedirs(os.path.join(base, "labels"), exist_ok=True)
            listings = {"train": ["banded/banded_0", "zigzagged/zigzagged_1"],
                        "val": ["banded/banded_1"],
                        "test": ["zigzagged/zigzagged_0"]}
            for part in (1, 2):
                for name, lines in listings.items():
                    if part == 2:
                        lines = list(reversed(lines))
                    with open(os.path.join(base, "labels",
                                           f"{name}{part}.txt"), "w") as fh:
                        fh.write("\n".join(f"{one}.jpg" for one in lines) + "\n")
        return _folder_case(build, lambda L, root: _made(L, "DTD")(
            root, split=split, partition=partition))

    def food101(split):
        """`Food101`. The split is a JSON of class to relative paths, and **the items
        keep that file's order while the classes are sorted.** The fixture's JSON is
        written with `waffles` before `apple_pie`, so a reader that took the key order
        for the class list gives every label a different number and nothing looks
        wrong."""
        import json
        import os
        def build(root):
            base = os.path.join(root, "food-101")
            meta = {"waffles": ["waffles/1", "waffles/2"], "apple_pie": ["apple_pie/3"]}
            os.makedirs(os.path.join(base, "meta"), exist_ok=True)
            for name in ("train", "test"):
                with open(os.path.join(base, "meta", f"{name}.json"), "w") as fh:
                    json.dump(meta, fh)
            for paths in meta.values():
                for one in paths:
                    _pic(os.path.join(base, "images", *f"{one}.jpg".split("/")),
                         len(one))
        return _folder_case(build, lambda L, root: _made(L, "Food101")(
            root, split=split))

    def fgvc(level):
        """`FGVCAircraft`. `annotation_level` chooses **two files together** — the
        class list and the label file — so the labels always index into the list they
        were written against.

        Each label line is a name and then the rest of the line: **the class name
        contains spaces.** The fixture's `Boeing 737-700` is one class, and a reader
        that split on every space raises rather than mislabelling, which is the good
        kind of wrong.
        """
        import os
        def build(root):
            data = os.path.join(root, "fgvc-aircraft-2013b", "data")
            os.makedirs(data, exist_ok=True)
            written = {"variants.txt": ["Boeing 737-700", "A340-300"],
                       "families.txt": ["Boeing 737", "A340"],
                       "manufacturers.txt": ["Boeing", "Airbus"]}
            for name, lines in written.items():
                with open(os.path.join(data, name), "w") as fh:
                    fh.write("\n".join(lines) + "\n")
            # **The levels do not agree on which picture is class 0.** `manufacturer`
            # is written the other way round, so a reader that took one level's labels
            # for another's gets different numbers. Without this the three levels
            # produced the same `[0, 1]` and the argument could be ignored — a
            # mutation caught it, reading did not.
            for name, lines in (("variant", written["variants.txt"]),
                                ("family", written["families.txt"]),
                                ("manufacturer", written["manufacturers.txt"][::-1])):
                for one in ("train", "val", "trainval", "test"):
                    with open(os.path.join(
                            data, f"images_{name}_{one}.txt"), "w") as fh:
                        fh.write(f"0000001 {lines[0]}\n0000002 {lines[1]}\n")
            for i in (1, 2):
                _pic(os.path.join(data, "images", f"{i:07d}.jpg"), i)
        return _folder_case(build, lambda L, root: _made(L, "FGVCAircraft")(
            root, annotation_level=level))

    def imagenette(split):
        """`Imagenette`. **The label comes from the sorted WordNet ids, not the class
        names** — and the two orders disagree. The fixture uses `n03425413` (gas pump)
        and `n01440764` (tench), written in that order and sorting the other way, so a
        reader that numbered by class name swaps every label.

        `class_to_idx` also has **more keys than there are classes**: a WordNet id
        carries every name for the thing, and `n03425413` carries four. The answer
        carries that count, because a reader taking one name each builds a dictionary
        that looks right and cannot look up three names in four.
        """
        import os
        def build(root):
            for name in ("train", "val"):
                for k, wnid in enumerate(("n03425413", "n01440764")):
                    for i in range(2):
                        _pic(os.path.join(root, "imagenette2", name, wnid,
                                          f"{wnid}_{i}.JPEG"), k * 5 + i
                             + (0 if name == "train" else 23))
        def make(L, root):
            return _made(L, "Imagenette")(root, split=split)
        def run(L):
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                build(root)
                loaded = make(L, root)
                out = np.concatenate([
                    _labelled_out(loaded),
                    np.asarray([len(loaded.class_to_idx), len(loaded.wnids)],
                               dtype=np.float32)])
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def _captioned_out(loaded):
        """Pictures and captions. The captions are **joined into the string half** of
        the answer, because a caption list that came back in the wrong order or with
        an orphan in it is the thing these two datasets can get wrong."""
        parts = [np.asarray([len(loaded)], dtype=np.float32)]
        for i in range(len(loaded)):
            picture, _captions = loaded[i]
            parts.append(np.asarray(picture).reshape(-1).astype(np.float32))
        return np.concatenate(parts)

    def _flickr_text(loaded):
        rows = []
        for i in range(len(loaded)):
            _picture, captions = loaded[i]
            rows.append("|".join(captions))
        return " / ".join(rows)

    def _flickr30k_tree(root):
        import os
        folder = os.path.join(root, "pictures")
        os.makedirs(folder, exist_ok=True)
        for name in ("1000092795.jpg", "10002456.jpg"):
            _pic(os.path.join(folder, name), len(name))
        listing = os.path.join(root, "results.token")
        with open(listing, "w") as fh:
            for name in ("1000092795.jpg", "10002456.jpg"):
                for k in range(2):
                    fh.write(f"{name}#{k}\tcaption {k} for {name}\n")
        return folder, listing

    def _flickr8k_tree(root):
        import os
        folder = os.path.join(root, "pictures")
        os.makedirs(folder, exist_ok=True)
        for stem in ("1000268201", "1001773457"):
            _pic(os.path.join(folder, f"{stem}_693b7e1f16.jpg"), len(stem))
        listing = os.path.join(root, "captions.html")
        with open(listing, "w") as fh:
            fh.write("<html><body><table>")
            for stem in ("1000268201", "1001773457"):
                fh.write(f"<tr><td><a href='x'>http://host/{stem}/img.jpg</a>"
                         f"<ul><li>a caption for {stem}</li><li>another</li>"
                         "</ul></td></tr>")
            # **`Image Not Found` is a real row** in that table, and the captions after
            # it belong to nothing. A reader that did not clear the current picture
            # would hang them on the previous one.
            fh.write("<tr><td>Image Not Found<ul><li>orphan</li></ul></td></tr>")
            fh.write("</table></body></html>")
        return folder, listing

    def flickr(which, text):
        """`Flickr8k` and `Flickr30k` — one dataset shape, two annotation files.

        30k is one line of `id#n<TAB>caption` and **the last two characters of the id
        come off**: `1000092795.jpg#4` is the fifth caption of `1000092795.jpg`. 8k is
        an **HTML page**, walked with the standard library's parser, whose link text
        names a directory and whose picture is found by glob.
        """
        def run(L):
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                folder, listing = (_flickr8k_tree(root) if which == "Flickr8k"
                                   else _flickr30k_tree(root))
                loaded = _made(L, which)(folder, listing)
                out = _flickr_text(loaded) if text else _captioned_out(loaded)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return out if text else L.tensor(np.ascontiguousarray(out))
        return run

    _INAT_2021 = (
        "00000_Animalia_Chordata_Aves_Passeriformes_Corvidae_Corvus_corax",
        "00001_Animalia_Chordata_Aves_Passeriformes_Corvidae_Corvus_corone",
        "00002_Animalia_Arthropoda_Insecta_Coleoptera_Cleridae_Trichodes_apiarius")

    def inaturalist(version, target_type):
        """`INaturalist`. **`version` is not a split — it is a different scheme for
        what a directory means**, and all three build the same index at the end, so
        getting it wrong gives a dataset that loads.

        - **2021** names each directory in eight underscore-separated pieces, six of
          them ranks, and the leading number must equal the directory's own position.
        - **2018 and 2019** nest a supercategory over a *numeric* subdirectory whose
          number is the category id, and the ids are sparse: the fixture's are 0, 2 and
          1 across two supercategories, so a reader that numbered by position gets
          different labels and the same count.
        - **2017** has the same nesting with non-numeric names, so the id *is* the
          position. One version reading the other's rule either raises on a name that
          is not a number or renumbers everything.

        The 2021 fixture puts **two species in one genus**, so `full` and `genus` are
        different numbers for the same picture — a reader that returned the category id
        for a rank agrees on the first two items and not the third.

        The items are compared as a set: `all_categories` is walked with `os.listdir`
        on both sides for the pre-2021 versions.
        """
        import os
        def build(root):
            if version == "2021_train":
                for k, name in enumerate(_INAT_2021):
                    for i in range(2):
                        _pic(os.path.join(root, version, name, f"{i}.png"), k * 3 + i)
            elif version == "2017":
                for above, subs in (("Aves", ("Corvus", "Turdus")),
                                    ("Insecta", ("Trichodes",))):
                    for sub in subs:
                        _pic(os.path.join(root, version, above, sub, "a.png"),
                             len(sub))
            else:
                for above, subs in (("Aves", ("0", "2")), ("Insecta", ("1",))):
                    for sub in subs:
                        _pic(os.path.join(root, version, above, sub, "a.png"),
                             int(sub) + 5)
        def run(L):
            import shutil
            import tempfile
            root = tempfile.mkdtemp()
            try:
                build(root)
                loaded = _made(L, "INaturalist")(root, version=version,
                                                 target_type=target_type)
                items = []
                for i in range(len(loaded)):
                    picture, target = loaded[i]
                    numbers = list(target) if isinstance(target, tuple) else [target]
                    items.append(np.concatenate([
                        np.asarray(picture).reshape(-1).astype(np.float32),
                        np.asarray(numbers, dtype=np.float32)]))
                items.sort(key=lambda item: item.tobytes())
                out = np.concatenate(
                    [np.asarray([len(loaded), len(loaded.all_categories)],
                                dtype=np.float32)] + items)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    # ── the three whose declined row said only "a codec" ────────────────────────
    #
    # **The codec was never these datasets' wall.** torchvision hands the path to
    # `PIL.Image.open`, which sniffs the content rather than the name, so a `.jpg`
    # holding PNG bytes opens there — measured — and `_folder_loader` settles it on the
    # magic number here for a reason it already gives. What the row took off the table
    # was the split file, the pairing and the XML fold, none of which is a codec.
    #
    # These fixtures write PNG bytes into files named `.jpg`, which is what the real
    # archive would hold in a form both sides can read. The codec itself is still
    # refused, one line down, where somebody holding a real JPEG meets it.
    def _voc_tree(root, targets):
        import os
        base = os.path.join(root, "VOCdevkit", "VOC2012")
        os.makedirs(os.path.join(base, "JPEGImages"))
        for name, block in (("a", 0), ("b", 7), ("c", 13)):
            picture = ((np.arange(4 * 5 * 3) + block) % 251).reshape(4, 5, 3)
            _write_png8(os.path.join(base, "JPEGImages", name + ".jpg"), picture)
        os.makedirs(os.path.join(base, "ImageSets", targets["splits"]))
        # **The split file's own order, not sorted.** Two of the three are out of
        # alphabetical order on purpose: the split file is the dataset, and a reader
        # that sorted would agree with one written the other way on any fixture that
        # happened to be in order already.
        with open(os.path.join(base, "ImageSets", targets["splits"], "train.txt"),
                  "w") as handle:
            handle.write("c\na\nb\n")
        return base

    def voc_segmentation():
        """`VOCSegmentation`. **The map comes back as one channel**, because it is a
        paletted PNG and its numbers are the classes rather than a colour."""
        import os
        import shutil
        import tempfile

        def run(L):
            root = tempfile.mkdtemp()
            try:
                base = _voc_tree(root, {"splits": "Segmentation"})
                os.makedirs(os.path.join(base, "SegmentationClass"))
                for i, name in enumerate(("a", "b", "c")):
                    block = ((np.arange(4 * 5) + i * 3) % 4).reshape(4, 5)
                    _write_png8(os.path.join(base, "SegmentationClass",
                                             name + ".png"), block)
                loaded = _made(L, "VOCSegmentation")(
                    root, year="2012", image_set="train")
                parts = [np.asarray([len(loaded)], dtype=np.float32)]
                for i in range(len(loaded)):
                    picture, target = loaded[i]
                    parts.append(np.asarray(picture).reshape(-1).astype(np.float32))
                    parts.append(np.asarray(target).reshape(-1).astype(np.float32))
                out = np.concatenate(parts)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return L.tensor(np.ascontiguousarray(out))
        return run

    def voc_detection():
        """`VOCDetection`. **The answer is the XML folded into dictionaries**, and the
        three rules that fold decides are asked here: a tag seen once is its value, a
        tag seen twice is a list, and `object` is a list even when there is one.

        The middle picture carries two objects and the others one, so a reader that
        skipped the `object` exception agrees on two of the three.
        """
        import json
        import os
        import shutil
        import tempfile

        def run(L):
            root = tempfile.mkdtemp()
            try:
                base = _voc_tree(root, {"splits": "Main"})
                os.makedirs(os.path.join(base, "Annotations"))
                counts = {"a": 1, "b": 2, "c": 1}
                for name, many in counts.items():
                    objects = "".join(
                        f"<object><name>k{j}</name><bndbox><xmin>{j}</xmin>"
                        f"<ymin>{j + 1}</ymin><xmax>{j + 2}</xmax>"
                        f"<ymax>{j + 3}</ymax></bndbox></object>"
                        for j in range(many))
                    with open(os.path.join(base, "Annotations", name + ".xml"),
                              "w") as handle:
                        handle.write(
                            "<annotation><folder>VOC2012</folder>"
                            f"<filename>{name}.jpg</filename>"
                            "<size><width>5</width><height>4</height>"
                            f"<depth>3</depth></size>{objects}</annotation>")
                loaded = _made(L, "VOCDetection")(root, year="2012",
                                                  image_set="train")
                rows = [str(len(loaded))]
                for i in range(len(loaded)):
                    picture, target = loaded[i]
                    rows.append(str(int(np.asarray(picture).sum())))
                    # **The tree as text.** Sorted keys, so the two sides are compared
                    # on the shape they built and not on a dictionary's order.
                    rows.append(json.dumps(target, sort_keys=True))
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return "\n".join(rows)
        return run

    def oxford_pet(target_types):
        """`OxfordIIITPet`. **Three targets, and one of them is a picture.**

        The fixture holds two breeds with two pictures each and the labels out of
        order, so the class list — built from the file names and sorted by label —
        cannot come out right by accident.
        """
        import os
        import shutil
        import tempfile

        def run(L):
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "oxford-iiit-pet")
                os.makedirs(os.path.join(base, "images"))
                os.makedirs(os.path.join(base, "annotations", "trimaps"))
                # **The second breed comes first in the file and second by label.**
                # Written the other way the two orders agree and a reader that used
                # the file's order would pass.
                rows = [("british_shorthair_5", 2, 1), ("Abyssinian_1", 1, 1),
                        ("british_shorthair_9", 2, 1), ("Abyssinian_7", 1, 2)]
                for i, (name, label, kind) in enumerate(rows):
                    picture = ((np.arange(4 * 5 * 3) + i * 5) % 251).reshape(4, 5, 3)
                    _write_png8(os.path.join(base, "images", name + ".jpg"), picture)
                    trimap = ((np.arange(4 * 5) + i) % 3 + 1).reshape(4, 5)
                    _write_png8(os.path.join(base, "annotations", "trimaps",
                                             name + ".png"), trimap)
                with open(os.path.join(base, "annotations", "trainval.txt"),
                          "w") as handle:
                    for name, label, kind in rows:
                        handle.write(f"{name} {label} {kind} 1\n")
                loaded = _made(L, "OxfordIIITPet")(
                    root, split="trainval", target_types=target_types)
                parts = [np.asarray([len(loaded)], dtype=np.float32)]
                for i in range(len(loaded)):
                    picture, target = loaded[i]
                    parts.append(np.asarray(picture).reshape(-1).astype(np.float32))
                    pieces = target if isinstance(target, tuple) else [target]
                    for one in pieces:
                        parts.append(np.asarray(one).reshape(-1).astype(np.float32))
                names = "|".join(loaded.classes)
                out = np.concatenate(parts)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return f"{names}\n{list(np.round(out, 4))}"
        return run

    def caltech101():
        """`Caltech101`. **A directory that is not a class, removed after the sort.**

        `BACKGROUND_Google` sorts between `airplanes` and `car_side`, so leaving it in
        shifts every label above it by one — and the picture still comes back from the
        directory that label names, which is why the count and the shapes agree either
        way.

        The fixture's categories are chosen to straddle it, and the third has two
        pictures so the one-based per-category index cannot be mistaken for a running
        one.
        """
        import os
        import shutil
        import tempfile

        def run(L):
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "caltech101", "101_ObjectCategories")
                counts = {"accordion": 1, "airplanes": 1, "BACKGROUND_Google": 1,
                          "car_side": 2}
                for k, (name, many) in enumerate(sorted(counts.items())):
                    for i in range(1, many + 1):
                        picture = ((np.arange(4 * 5 * 3) + k * 11 + i) % 251)
                        _write_png8(os.path.join(base, name, f"image_{i:04d}.jpg"),
                                    picture.reshape(4, 5, 3))
                loaded = _made(L, "Caltech101")(root, target_type="category")
                parts = [np.asarray([len(loaded)], dtype=np.float32)]
                for i in range(len(loaded)):
                    picture, target = loaded[i]
                    parts.append(np.asarray(picture).reshape(-1).astype(np.float32))
                    parts.append(np.asarray([target], dtype=np.float32))
                names = "|".join(loaded.categories)
                mapped = "|".join(loaded.annotation_categories)
                out = np.concatenate(parts)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return f"{names}\n{mapped}\n{list(np.round(out, 4))}"
        return run

    def caltech256():
        """`Caltech256`. **The label comes from the sorted position, not the directory's
        own number**, and the file name is built from it.

        The fixture leaves a gap in the numbering — `001`, `003` — so the two disagree.
        On the real archive they never do, which is why a reader that took the prefix
        passes there and fails here.

        **Only `.jpg` files are counted.** A stray file beside the pictures is in the
        fixture for that: counted whole, the index walks off the end of the numbering
        and the read is a `FileNotFoundError` on the last item.
        """
        import os
        import shutil
        import tempfile

        def run(L):
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "caltech256", "256_ObjectCategories")
                for k, name in enumerate(("001.ak47", "003.backpack")):
                    for i in range(1, k + 2):
                        picture = ((np.arange(4 * 5 * 3) + k * 13 + i) % 251)
                        _write_png8(
                            os.path.join(base, name, f"{k + 1:03d}_{i:04d}.jpg"),
                            picture.reshape(4, 5, 3))
                    with open(os.path.join(base, name, "RENAME2"), "w") as handle:
                        handle.write("not a picture\n")
                loaded = _made(L, "Caltech256")(root)
                parts = [np.asarray([len(loaded)], dtype=np.float32)]
                for i in range(len(loaded)):
                    picture, target = loaded[i]
                    parts.append(np.asarray(picture).reshape(-1).astype(np.float32))
                    parts.append(np.asarray([target], dtype=np.float32))
                names = "|".join(loaded.categories)
                out = np.concatenate(parts)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return f"{names}\n{list(np.round(out, 4))}"
        return run

    def widerface(split):
        """`WIDERFace`. **The annotation file is a state machine**, not a table: a path,
        a count, that many box lines, a path again.

        The fixture gives the three images **three different counts** — two, one and
        three — so a reader that ignored the count runs one image's boxes into the next
        and every image after the first is wrong while the total stays right.

        **Ten numbers a line and the first four are the box.** The other six are per-face
        flags, each its own key, and the fixture gives them values that differ from one
        another so a reader that kept them as one array of six cannot come out equal.

        `test` has no annotations at all and its list lives under a different file name;
        the target there is `None`, which is written as a length so that a split which
        should have none cannot pass by handing back zeros.
        """
        import os
        import shutil
        import tempfile

        def run(L):
            root = tempfile.mkdtemp()
            try:
                base = os.path.join(root, "widerface")
                for name in ("WIDER_train", "WIDER_val", "WIDER_test"):
                    os.makedirs(os.path.join(base, name, "images", "0--Parade"))
                os.makedirs(os.path.join(base, "wider_face_split"))
                names = ["0--Parade/p_%d.jpg" % i for i in range(3)]
                for i, one in enumerate(names):
                    picture = ((np.arange(4 * 5 * 3) + i * 9) % 251).reshape(4, 5, 3)
                    for folder in ("WIDER_train", "WIDER_val", "WIDER_test"):
                        _write_png8(os.path.join(base, folder, "images", one), picture)
                rows = []
                for i, (one, many) in enumerate(zip(names, (2, 1, 3))):
                    rows.append(one)
                    rows.append(str(many))
                    for k in range(many):
                        # **Ten values, and no two of the six flags alike** — an array
                        # of six read as one block would compare equal to a reader that
                        # split them, if they held the same number.
                        rows.append(" ".join(str(v) for v in (
                            i * 10 + k, i * 10 + k + 1, 3 + k, 4 + k,
                            k % 2, (k + 1) % 3, (k + 2) % 4, k % 5, (k + 3) % 6,
                            (k + 4) % 7)))
                for kind in ("train", "val"):
                    with open(os.path.join(base, "wider_face_split",
                                           f"wider_face_{kind}_bbx_gt.txt"),
                              "w") as handle:
                        handle.write("\n".join(rows) + "\n")
                with open(os.path.join(base, "wider_face_split",
                                       "wider_face_test_filelist.txt"), "w") as handle:
                    handle.write("\n".join(names) + "\n")
                loaded = _made(L, "WIDERFace")(root, split=split)
                parts = [np.asarray([len(loaded)], dtype=np.float32)]
                for i in range(len(loaded)):
                    picture, target = loaded[i]
                    parts.append(np.asarray(picture).reshape(-1).astype(np.float32))
                    if target is None:
                        parts.append(np.asarray([-1.0], dtype=np.float32))
                        continue
                    for key in sorted(target):
                        parts.append(
                            np.asarray(target[key]).reshape(-1).astype(np.float32))
                out = np.concatenate(parts)
                # **The split alone, not the whole repr.** `repr` carries the root,
                # and the root is a temporary directory that differs between the two
                # runs being compared — a case that can never agree with itself.
                shown = loaded.extra_repr()
            finally:
                shutil.rmtree(root, ignore_errors=True)
            return f"{shown}\n{list(np.round(out, 4))}"
        return run

    cases += [
        (prefix + "WIDERFace(a count between the boxes)", widerface("train")),
        (prefix + "WIDERFace(test has no annotations)", widerface("test")),
        (prefix + "Caltech101(a directory that is not a class)", caltech101()),
        (prefix + "Caltech256(the label is the sorted position)", caltech256()),
        (prefix + "OxfordIIITPet(category)", oxford_pet("category")),
        (prefix + "OxfordIIITPet(a picture beside a number)",
         oxford_pet(("binary-category", "segmentation"))),
        (prefix + "VOCSegmentation(split order, palette target)", voc_segmentation()),
        (prefix + "VOCDetection(the XML fold)", voc_detection()),
        (prefix + "Sintel(clean, pairs within a scene)", sintel("clean")),
        # **`both` walks the flow list once per pass**, so the count doubles.
        (prefix + "Sintel(both passes)", sintel("both")),
        # **The flow is a PNG too** — the row said *a codec and then another format*
        # and there was never a second format, only a second use of the first.
        (prefix + "KittiFlow(train, pairs by suffix)", kitti_flow("train")),
        # **No flow on `test`**, and the `None`s are written into the answer.
        (prefix + "KittiFlow(test, no flow)", kitti_flow("test")),
        (prefix + "HD1K(each sequence stops one short)", hd1k()),
        (prefix + "Kitti2012Stereo(disparity is 16-bit grey / 256)", kitti_stereo(2012)),
        (prefix + "Kitti2015Stereo(both disparities present)", kitti_stereo(2015)),
        (prefix + "InStereo2k(disparity / 1024)", instereo2k()),
        (prefix + "SintelStereo(disparity packed into a colour)", sintel_stereo("final")),
        # **`both` is final then clean**, in that order — the opposite of `Sintel`'s.
        (prefix + "SintelStereo(both passes)", sintel_stereo("both")),
        # **`.pfm` is a container too** — the last of the fourteen that read as a wall.
        (prefix + "CarlaStereo(disparity signed, used positive)", carla_stereo()),
        (prefix + "CREStereo(four scenes, a disparity over thirty-two)",
         cre_stereo()),
        (prefix + "FallingThingsStereo(single, depth into disparity)",
         falling_things("single")),
        (prefix + "FallingThingsStereo(both, two nesting depths)",
         falling_things("both")),
        (prefix + "ETH3DStereo(mask beside the pfm, non-zero is valid)",
         eth3d_stereo("train")),
        (prefix + "ETH3DStereo(test, no ground truth)", eth3d_stereo("test")),
        # **The variant is a directory depth**, not a filter.
        (prefix + "SceneFlowStereo(Monkaa, one level)", scene_flow("Monkaa")),
        (prefix + "SceneFlowStereo(FlyingThings3D, three levels)",
         scene_flow("FlyingThings3D")),
        # **into_past swaps the frames and shifts the flow.**
        (prefix + "FlyingThings3D(clean, left, both directions)",
         flying_things("clean", "left")),
        (prefix + "FlyingThings3D(both passes, both cameras)",
         flying_things("both", "both")),
        # **Pairs by position, split by a text file.**
        (prefix + "FlyingChairs(train, pairs by position)", flying_chairs("train")),
        (prefix + "FlyingChairs(val, the file cuts it)", flying_chairs("val")),
        # **`calibration` is a directory suffix**, and the infinite disparity is real.
        (prefix + "Middlebury2014Stereo(perfect, inf zeroed and masked)",
         middlebury("train", "perfect")),
        (prefix + "Middlebury2014Stereo(both calibrations)",
         middlebury("train", "both")),
        (prefix + "Middlebury2014Stereo(test, no calibration)",
         middlebury("test", None)),
        # **PNG in `image_2`, and `occluded` is the one integer.**
        (prefix + "Kitti(train, fifteen fields by position)", kitti(True)),
        (prefix + "Kitti(test, no labels)", kitti(False)),
        # **BMP sheets cut into a grid**, row-major, sheets sorted.
        (prefix + "PhotoTour(train, one patch)", photo_tour(True)),
        (prefix + "PhotoTour(test, two patches and the answer)", photo_tour(False)),
        # **A walk plus a loader** — the shape `ImageFolder` was let in for, and
        # torchvision's own `loader=` says so about each of these.
        (prefix + "Country211(train)", country211("train")),
        (prefix + "Country211(valid, the same countries)", country211("valid")),
        (prefix + "EuroSAT(ten covers, RGB version)", eurosat()),
        # **partition 2 is partition 1 reversed**, so ignoring it still returns a set.
        (prefix + "DTD(train, partition 1)", dtd("train", 1)),
        (prefix + "DTD(train, partition 2)", dtd("train", 2)),
        # **The JSON's order is the items' and the classes are sorted.**
        (prefix + "Food101(train, sorted classes over a JSON order)", food101("train")),
        # **The level chooses two files at once**, and a class name has a space in it.
        (prefix + "FGVCAircraft(variant)", fgvc("variant")),
        (prefix + "FGVCAircraft(manufacturer)", fgvc("manufacturer")),
        # **The label is the sorted WordNet id, and a class is a tuple of names.**
        (prefix + "Imagenette(train, sorted wnids)", imagenette("train")),
        (prefix + "Imagenette(val)", imagenette("val")),
        # **`id#n` loses two characters**, and `Image Not Found` is a real row.
        (prefix + "Flickr30k(pictures)", flickr("Flickr30k", False)),
        (prefix + "Flickr30k(captions)=문자열", flickr("Flickr30k", True)),
        (prefix + "Flickr8k(pictures through an HTML table)",
         flickr("Flickr8k", False)),
        (prefix + "Flickr8k(captions, the orphan dropped)=문자열",
         flickr("Flickr8k", True)),
        # **`version` is a directory scheme, not a split**, and the three disagree.
        (prefix + "INaturalist(2021, full)", inaturalist("2021_train", "full")),
        # **Two species in one genus**, so the rank and the category are different ids.
        (prefix + "INaturalist(2021, full+genus+phylum)",
         inaturalist("2021_train", ["full", "genus", "phylum"])),
        # **Sparse numeric ids**: 0, 2, 1 across two supercategories.
        (prefix + "INaturalist(2018, the id is the directory's name)",
         inaturalist("2018", ["full", "super"])),
        # **Non-numeric names**, so the id is the position instead.
        (prefix + "INaturalist(2017, the id is the position)",
         inaturalist("2017", "full")),
    ]

    cases += [
        # **`val` reads a folder called `valid`.**
        (prefix + "RenderedSST2(val, the folder is `valid`)", sst2("val")),
        (prefix + "RenderedSST2(train)", sst2("train")),
    ]

    cases += [
        (prefix + "CLEVR(train, labels by filename)", clevr("train", "labels")),
        (prefix + "CLEVR(train, pictures)", clevr("train", "pictures")),
        # **No scenes file for `test`**, so every label is `None`.
        (prefix + "CLEVR(test, no labels)", clevr("test", "labels")),
    ]

    cases += [
        (prefix + "ImageFolder(pictures)", image_folder("pictures")),
        # **The classes are the sorted directory names**, and the index a class gets
        # depends on its name and nothing else — so this freezes the mapping beside
        # the labels, because renaming a folder renumbers a trained model's classes.
        (prefix + "ImageFolder(labels and classes)", image_folder("labels")),
    ]
    return cases
