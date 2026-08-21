"""Compares borch against real torch by value.

This file is borch's reason to exist. If the imitation behaves even slightly differently,
whoever learns from it learns something false. "It runs" is not enough — **the same numbers
have to come out.**

The same input goes into both sides and values and shapes are compared. Layers initialised at
random get one side's weights copied in so that a comparison is possible.

    uv run --with pytest --with numpy --with torch pytest tests/
"""

import pathlib
import sys

import numpy as np
import pytest
import torch as real

# It used to pick up `borch.py` by path. Once it became a package, that stopped working.
_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import borch as bt                                            # noqa: E402

TOL = 1e-4


def same(a, b, tol=TOL, what=""):
    """Are both tensors equal in value and in shape?"""
    an = a.detach().numpy() if isinstance(a, real.Tensor) else a.data
    bn = b.detach().numpy() if isinstance(b, real.Tensor) else b.data
    assert an.shape == bn.shape, f"{what} shape differs: {an.shape} vs {bn.shape}"
    assert np.allclose(an, bn, atol=tol, rtol=tol), (
        f"{what} value differs\n  torch: {an}\n  borch: {bn}"
    )


def pair(data, requires_grad=False):
    """Makes both tensors from the same data."""
    arr = np.asarray(data, dtype=np.float32)
    return (real.tensor(arr, requires_grad=requires_grad),
            bt.tensor(arr, requires_grad=requires_grad))


def grads_match(fn_real, fn_mini, data, what=""):
    """Are the gradients of the same computation equal? `fn` takes a tensor and returns a scalar."""
    r, m = pair(data, requires_grad=True)
    fn_real(r).backward()
    fn_mini(m).backward()
    same(r.grad, m.grad, what=f"{what} gradient")


# ------------------------------------------------------------- reductions

@pytest.mark.parametrize("dim", [None, 0, 1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum(dim, keepdim):
    r, m = pair([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    kw = {} if dim is None else {"dim": dim, "keepdim": keepdim}
    same(r.sum(**kw), m.sum(**kw), what=f"sum(dim={dim}, keepdim={keepdim})")


@pytest.mark.parametrize("dim", [None, 0, 1])
def test_mean(dim):
    r, m = pair([[1.0, 2.0], [3.0, 4.0]])
    kw = {} if dim is None else {"dim": dim}
    same(r.mean(**kw), m.mean(**kw), what=f"mean(dim={dim})")


@pytest.mark.parametrize("dim", [0, 1])
def test_max_values_and_indices(dim):
    r, m = pair([[1.0, 7.0, 3.0], [9.0, 2.0, 5.0]])
    same(r.max(dim=dim).values, m.max(dim=dim).values, what=f"max(dim={dim}).values")
    assert r.max(dim=dim).indices.tolist() == m.max(dim=dim).indices.tolist()


@pytest.mark.parametrize("dim", [0, 1])
def test_min_values(dim):
    r, m = pair([[1.0, 7.0, 3.0], [9.0, 2.0, 5.0]])
    same(r.min(dim=dim).values, m.min(dim=dim).values, what=f"min(dim={dim})")


def test_sum_backward():
    grads_match(lambda t: t.sum(), lambda t: t.sum(), [[1.0, 2.0], [3.0, 4.0]], "sum")


def test_sum_dim_backward():
    grads_match(lambda t: (t.sum(dim=0) * 2).sum(),
                lambda t: (t.sum(dim=0) * 2).sum(), [[1.0, 2.0], [3.0, 4.0]], "sum(dim=0)")


def test_mean_backward():
    grads_match(lambda t: t.mean(), lambda t: t.mean(), [[1.0, 2.0], [3.0, 4.0]], "mean")


def test_max_backward_goes_to_one_place():
    """The gradient goes only to the maximum's slot — shared out, training differs subtly."""
    grads_match(lambda t: t.max(dim=1).values.sum(),
                lambda t: t.max(dim=1).values.sum(), [[1.0, 7.0], [9.0, 2.0]], "max")


def test_std_unbiased_both_ways():
    r, m = pair([1.0, 2.0, 3.0, 4.0])
    same(r.std(unbiased=False), m.std(unbiased=False), what="std(unbiased=False)")
    same(r.std(unbiased=True), m.std(unbiased=True), what="std(unbiased=True)")


# --------------------------------------------------------- shape, indexing

def test_indexing_backward():
    grads_match(lambda t: t[0].sum(), lambda t: t[0].sum(), [[1.0, 2.0], [3.0, 4.0]], "t[0]")


def test_fancy_indexing_backward():
    """Picking the same slot twice has to add the gradients."""
    idx = [0, 0, 1]
    grads_match(lambda t: t[idx].sum(), lambda t: t[idx].sum(),
                [[1.0, 2.0], [3.0, 4.0]], "t[[0,0,1]]")


def test_reshape_backward():
    grads_match(lambda t: (t.reshape(4) * torch_ramp()).sum(),
                lambda t: (t.reshape(4) * mini_ramp()).sum(),
                [[1.0, 2.0], [3.0, 4.0]], "reshape")


def torch_ramp():
    return real.tensor([1.0, 2.0, 3.0, 4.0])


def mini_ramp():
    return bt.tensor([1.0, 2.0, 3.0, 4.0])


def test_transpose_and_matmul_backward():
    r, m = pair([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    rb = real.tensor([[1.0, 0.0], [0.5, 2.0]])
    mb = bt.tensor([[1.0, 0.0], [0.5, 2.0]])
    (r @ rb.T).sum().backward()
    (m @ mb.T).sum().backward()
    same(r.grad, m.grad, what="matmul+transpose gradient")


def test_batched_matmul():
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    b = np.arange(8, dtype=np.float32).reshape(2, 4, 1)
    same(real.tensor(a) @ real.tensor(b), bt.tensor(a) @ bt.tensor(b), what="batched matmul")


def test_squeeze_unsqueeze():
    r, m = pair([[1.0], [2.0]])
    same(r.squeeze(-1), m.squeeze(-1), what="squeeze")
    same(r.unsqueeze(0), m.unsqueeze(0), what="unsqueeze")


def test_permute():
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    same(real.tensor(a).permute(2, 0, 1), bt.tensor(a).permute(2, 0, 1), what="permute")


# ------------------------------------------------------------ broadcasting

def test_broadcast_bias_backward():
    """The gradient of (N,D) + (D,) has to fold back down to (D,)."""
    rb = real.tensor([1.0, 1.0], requires_grad=True)
    mb = bt.tensor([1.0, 1.0], requires_grad=True)
    (real.tensor([[1.0, 2.0], [3.0, 4.0]]) + rb).sum().backward()
    (bt.tensor([[1.0, 2.0], [3.0, 4.0]]) + mb).sum().backward()
    same(rb.grad, mb.grad, what="broadcast bias gradient")


def test_broadcast_column_backward():
    rb = real.tensor([[1.0], [2.0]], requires_grad=True)
    mb = bt.tensor([[1.0], [2.0]], requires_grad=True)
    (rb * real.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    (mb * bt.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    same(rb.grad, mb.grad, what="column broadcast gradient")


# --------------------------------------------------------------- functions

@pytest.mark.parametrize("fn", ["sigmoid", "relu", "tanh", "exp", "log", "sqrt", "abs"])
def test_elementwise(fn):
    data = [0.5, 1.5, 2.5]
    same(getattr(real, fn)(real.tensor(data)), getattr(bt, fn)(bt.tensor(data)), what=fn)


@pytest.mark.parametrize("fn", ["sigmoid", "relu", "tanh", "exp"])
def test_elementwise_backward(fn):
    grads_match(lambda t: getattr(real, fn)(t).sum(),
                lambda t: getattr(bt, fn)(t).sum(), [0.5, 1.5, -0.5], fn)


def test_sigmoid_extreme():
    """Does it blow up at large negatives and positives?"""
    data = [-1000.0, -50.0, 0.0, 50.0, 1000.0]
    same(real.sigmoid(real.tensor(data)), bt.sigmoid(bt.tensor(data)), what="sigmoid at the extremes")


def test_softmax_and_backward():
    data = [[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]
    same(real.nn.functional.softmax(real.tensor(data), dim=-1),
         bt.softmax(bt.tensor(data), dim=-1), what="softmax")
    grads_match(lambda t: (real.nn.functional.softmax(t, dim=-1) * torch_ramp()[:3]).sum(),
                lambda t: (bt.softmax(t, dim=-1) * mini_ramp()[:3]).sum(),
                [1.0, 2.0, 3.0], "softmax")


def test_masked_fill_backward():
    mask_r = real.tensor([[1, 0], [1, 1]])
    mask_m = bt.tensor([[1, 0], [1, 1]])
    grads_match(lambda t: t.masked_fill(mask_r == 0, 0.0).sum(),
                lambda t: t.masked_fill(mask_m == 0, 0.0).sum(),
                [[1.0, 2.0], [3.0, 4.0]], "masked_fill")


def test_where():
    cond_r, cond_m = real.tensor([True, False]), bt.tensor([True, False])
    same(real.where(cond_r, real.tensor([1.0, 2.0]), real.tensor([3.0, 4.0])),
         bt.where(cond_m, bt.tensor([1.0, 2.0]), bt.tensor([3.0, 4.0])), what="where")


def test_tril_triu():
    a = np.ones((3, 3), dtype=np.float32)
    same(real.tril(real.tensor(a)), bt.tril(bt.tensor(a)), what="tril")
    same(real.triu(real.tensor(a), diagonal=1), bt.triu(bt.tensor(a), diagonal=1), what="triu")


# ---------------------------------------------------------------- dtype

def test_dtype_rules():
    assert str(real.tensor([1, 2, 3]).dtype) == str(bt.tensor([1, 2, 3]).dtype)
    assert str(real.tensor([1.0, 2.0]).dtype) == str(bt.tensor([1.0, 2.0]).dtype)


def test_int_tensor_refuses_grad():
    with pytest.raises(RuntimeError):
        real.tensor([1, 2, 3], requires_grad=True)
    with pytest.raises(RuntimeError):
        bt.tensor([1, 2, 3], requires_grad=True)


# ------------------------------------------------------------------ layers

def copy_linear(src, dst):
    dst.weight.data = bt.tensor(src.weight.detach().numpy().copy())
    dst.bias.data = bt.tensor(src.bias.detach().numpy().copy())


def test_linear_forward_and_backward():
    rl, ml = real.nn.Linear(3, 2), bt.nn.Linear(3, 2)
    copy_linear(rl, ml)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ro, mo = rl(real.tensor(x)), ml(bt.tensor(x))
    same(ro, mo, what="Linear output")
    ro.sum().backward()
    mo.sum().backward()
    same(rl.weight.grad, ml.weight.grad, what="Linear weight gradient")
    same(rl.bias.grad, ml.bias.grad, what="Linear bias gradient")


def test_embedding_forward_and_backward():
    rl, ml = real.nn.Embedding(5, 3), bt.nn.Embedding(5, 3)
    ml.weight.data = bt.tensor(rl.weight.detach().numpy().copy())
    ids = [[0, 2, 2]]
    ro = rl(real.tensor(ids))
    mo = ml(bt.tensor(ids))
    same(ro, mo, what="Embedding output")
    ro.sum().backward()
    mo.sum().backward()
    same(rl.weight.grad, ml.weight.grad, what="Embedding gradient (the same index has to add up)")


def test_layernorm():
    rl, ml = real.nn.LayerNorm(4), bt.nn.LayerNorm(4)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, -5.0, 3.0]], dtype=np.float32)
    same(rl(real.tensor(x)), ml(bt.tensor(x)), what="LayerNorm")


def test_batchnorm_backward():
    """The forward pass was right and the backward pass wrong for a long time — the mean and
    variance were computed outside the graph.

    Training runs and the loss goes down; only the values differ. Comparing the forward pass
    alone does not catch it.
    """
    rl, ml = real.nn.BatchNorm2d(2), bt.nn.BatchNorm2d(2)
    copy_state(rl, ml)
    x = np.random.default_rng(0).standard_normal((4, 2, 3, 3)).astype(np.float32)
    rx, mx = real.tensor(x, requires_grad=True), bt.tensor(x, requires_grad=True)
    rl(rx).sum().backward()
    ml(mx).sum().backward()
    same(rx.grad, mx.grad, tol=1e-4, what="BatchNorm input gradient")
    same(rl.weight.grad, ml.weight.grad, tol=1e-4, what="BatchNorm weight gradient")
    same(rl.bias.grad, ml.bias.grad, tol=1e-4, what="BatchNorm bias gradient")
    assert np.allclose(rl.running_mean.numpy(), ml.running_mean, atol=1e-5)
    assert np.allclose(rl.running_var.numpy(), ml.running_var, atol=1e-4)


@pytest.mark.parametrize("layer,shape", [
    (lambda L: L.nn.Linear(3, 2), (4, 3)),
    (lambda L: L.nn.LayerNorm(3), (4, 3)),
    (lambda L: L.nn.BatchNorm2d(2), (4, 2, 3, 3)),
    (lambda L: L.nn.Conv2d(2, 3, 3, padding=1), (2, 2, 5, 5)),
    (lambda L: L.nn.Embedding(5, 3), None),
])
def test_every_layer_passes_gradient_to_its_weights(layer, shape):
    """Per layer, **does a gradient actually arrive at the weight?** `None` means the graph is cut."""
    rl, ml = layer(real), layer(bt)
    copy_state(rl, ml)
    if shape is None:
        rx, mx = real.tensor([[0, 1]]), bt.tensor([[0, 1]])
    else:
        x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        rx, mx = real.tensor(x), bt.tensor(x)
    rl(rx).sum().backward()
    ml(mx).sum().backward()
    for (name, rp), (_, mp) in zip(rl.named_parameters(), ml.named_parameters()):
        assert rp.grad is not None, f"real torch has no {name} gradient either — the case is wrong"
        assert mp.grad is not None, f"no gradient arrived at {name} — the graph is cut"
        same(rp.grad, mp.grad, tol=1e-4, what=f"{name} gradient")


def test_data_assignment_rejects_ndarray():
    """torch refuses `p.data = ndarray`. Accepting it breaks code that worked in the browser
    on the person's own machine — being lenient is a divergence too."""
    for lib in (real, bt):
        p = lib.nn.Linear(2, 2).weight
        with pytest.raises(TypeError):
            p.data = np.zeros((2, 2), dtype=np.float32)


def test_batchnorm_train_and_eval():
    rl, ml = real.nn.BatchNorm2d(2), bt.nn.BatchNorm2d(2)
    x = np.arange(2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3)
    same(rl(real.tensor(x)), ml(bt.tensor(x)), tol=1e-3, what="BatchNorm2d (train mode)")
    rl.eval(); ml.eval()
    same(rl(real.tensor(x)), ml(bt.tensor(x)), tol=1e-3, what="BatchNorm2d (eval mode)")


def test_dropout_eval_is_identity():
    x = np.ones((4, 4), dtype=np.float32)
    rd, md = real.nn.Dropout(0.5), bt.nn.Dropout(0.5)
    rd.eval(); md.eval()
    same(rd(real.tensor(x)), md(bt.tensor(x)), what="Dropout(eval)")


@pytest.mark.parametrize("padding,stride", [(0, 1), (1, 1), (1, 2)])
def test_conv2d(padding, stride):
    x = np.arange(1 * 2 * 5 * 5, dtype=np.float32).reshape(1, 2, 5, 5)
    w = np.arange(3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3) / 10
    same(real.nn.functional.conv2d(real.tensor(x), real.tensor(w), stride=stride, padding=padding),
         bt.conv2d(bt.tensor(x), bt.tensor(w), stride=stride, padding=padding),
         tol=1e-3, what=f"conv2d(pad={padding}, stride={stride})")


def test_conv2d_backward():
    x = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
    w = np.ones((1, 1, 3, 3), dtype=np.float32)
    rx, mx = real.tensor(x, requires_grad=True), bt.tensor(x, requires_grad=True)
    rw, mw = real.tensor(w, requires_grad=True), bt.tensor(w, requires_grad=True)
    real.nn.functional.conv2d(rx, rw).sum().backward()
    bt.conv2d(mx, mw).sum().backward()
    same(rx.grad, mx.grad, tol=1e-3, what="conv2d input gradient")
    same(rw.grad, mw.grad, tol=1e-3, what="conv2d filter gradient")


def test_max_pool2d_and_backward():
    x = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
    rx, mx = real.tensor(x, requires_grad=True), bt.tensor(x, requires_grad=True)
    ro = real.nn.functional.max_pool2d(rx, 2)
    mo = bt.max_pool2d(mx, 2)
    same(ro, mo, what="max_pool2d")
    ro.sum().backward()
    mo.sum().backward()
    same(rx.grad, mx.grad, what="max_pool2d gradient")


def copy_rnn(src, dst):
    for key, value in src.state_dict().items():
        getattr(dst, key).data = bt.tensor(value.detach().numpy().copy())


@pytest.mark.parametrize("kwargs", [
    {}, {"num_layers": 2}, {"batch_first": True},
    {"nonlinearity": "relu"}, {"bias": False},
])
def test_rnn_forward(kwargs):
    rr, nr = real.nn.RNN(4, 6, **kwargs), bt.nn.RNN(4, 6, **kwargs)
    copy_rnn(rr, nr)
    shape = (5, 3, 4) if kwargs.get("batch_first") else (3, 5, 4)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    ro, rh = rr(real.tensor(x))
    no, nh = nr(bt.tensor(x))
    same(ro, no, what=f"RNN output {kwargs}")
    same(rh, nh, what=f"RNN h_n {kwargs}")


def test_rnn_state_dict_keys():
    rr, nr = real.nn.RNN(3, 4, num_layers=2), bt.nn.RNN(3, 4, num_layers=2)
    assert list(rr.state_dict().keys()) == list(nr.state_dict().keys())


def test_rnn_backward():
    rr, nr = real.nn.RNN(3, 4), bt.nn.RNN(3, 4)
    copy_rnn(rr, nr)
    x = np.random.default_rng(1).standard_normal((5, 2, 3)).astype(np.float32)
    rx, nx = real.tensor(x, requires_grad=True), bt.tensor(x, requires_grad=True)
    rr(rx)[0].sum().backward()
    nr(nx)[0].sum().backward()
    same(rx.grad, nx.grad, tol=1e-4, what="RNN input gradient")
    same(rr.weight_hh_l0.grad, nr.weight_hh_l0.grad, tol=1e-4, what="RNN weight_hh gradient")


def test_rnn_initial_hidden():
    """Given `h_0` directly, it has to start from there."""
    rr, nr = real.nn.RNN(3, 4), bt.nn.RNN(3, 4)
    copy_rnn(rr, nr)
    x = np.zeros((2, 1, 3), dtype=np.float32)
    h0 = np.ones((1, 1, 4), dtype=np.float32) * 0.5
    same(rr(real.tensor(x), real.tensor(h0))[0], nr(bt.tensor(x), bt.tensor(h0))[0],
         what="RNN initial hidden state")


def test_stack_and_cat_backward():
    a = bt.tensor([1.0, 2.0], requires_grad=True)
    b = bt.tensor([3.0, 4.0], requires_grad=True)
    ra = real.tensor([1.0, 2.0], requires_grad=True)
    rb = real.tensor([3.0, 4.0], requires_grad=True)
    (bt.stack([a, b]) * bt.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    (real.stack([ra, rb]) * real.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    same(ra.grad, a.grad, what="stack gradient")

    c = bt.tensor([1.0], requires_grad=True)
    rc = real.tensor([1.0], requires_grad=True)
    bt.cat([c, c * 2]).sum().backward()
    real.cat([rc, rc * 2]).sum().backward()
    same(rc.grad, c.grad, what="cat gradient")


def copy_state(src, dst):
    """Plants torch's state as it is. Values can only be compared if the initial values match.

    Not just the parameters but **the buffers too** — BatchNorm's `running_mean` is one, and
    leaving it out makes the values diverge in eval mode only.
    """
    assert list(src.state_dict().keys()) == list(dst.state_dict().keys()), (
        f"state_dict keys differ\n  torch {list(src.state_dict())}\n  borch {list(dst.state_dict())}")
    own = dict(dst.named_parameters())
    buffers = dict(dst.named_buffers())
    for key, value in src.state_dict().items():
        array = value.detach().numpy().copy()
        if key in own:
            own[key].data = bt.tensor(array)
        elif key in buffers:
            dst.load_state_dict({key: bt.tensor(array)}, strict=False)


@pytest.mark.parametrize("cls", ["RNN", "LSTM", "GRU"])
@pytest.mark.parametrize("kwargs", [
    {}, {"num_layers": 2}, {"batch_first": True}, {"bias": False},
])
def test_recurrent_forward(cls, kwargs):
    rr = getattr(real.nn, cls)(4, 6, **kwargs)
    nr = getattr(bt.nn, cls)(4, 6, **kwargs)
    copy_state(rr, nr)
    shape = (5, 3, 4) if kwargs.get("batch_first") else (3, 5, 4)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    ro, rs = rr(real.tensor(x))
    no, ns = nr(bt.tensor(x))
    same(ro, no, what=f"{cls} output {kwargs}")
    if cls == "LSTM":
        same(rs[0], ns[0], what="LSTM h_n")
        same(rs[1], ns[1], what="LSTM c_n")
    else:
        same(rs, ns, what=f"{cls} final state")


@pytest.mark.parametrize("cls", ["RNN", "LSTM", "GRU"])
def test_recurrent_backward(cls):
    rr, nr = getattr(real.nn, cls)(3, 4), getattr(bt.nn, cls)(3, 4)
    copy_state(rr, nr)
    x = np.random.default_rng(1).standard_normal((5, 2, 3)).astype(np.float32)
    rx, nx = real.tensor(x, requires_grad=True), bt.tensor(x, requires_grad=True)
    rr(rx)[0].sum().backward()
    nr(nx)[0].sum().backward()
    same(rx.grad, nx.grad, tol=1e-4, what=f"{cls} input gradient")
    same(rr.weight_hh_l0.grad, nr.weight_hh_l0.grad, tol=1e-4, what=f"{cls} weight_hh gradient")


def test_lstm_gate_order():
    """With the gate order (i, f, g, o) wrong the values look plausible and training does not work."""
    rr, nr = real.nn.LSTM(2, 3), bt.nn.LSTM(2, 3)
    copy_state(rr, nr)
    x = np.ones((1, 1, 2), dtype=np.float32)
    same(rr(real.tensor(x))[1][1], nr(bt.tensor(x))[1][1], what="LSTM c_n (gate order)")


def test_gru_bias_inside_reset_gate():
    """In the n gate, r multiplies the hidden term including its bias. Left outside, it drifts slightly."""
    rr, nr = real.nn.GRU(2, 3), bt.nn.GRU(2, 3)
    copy_state(rr, nr)
    x = np.random.default_rng(2).standard_normal((4, 1, 2)).astype(np.float32)
    same(rr(real.tensor(x))[0], nr(bt.tensor(x))[0], what="GRU output")


@pytest.mark.parametrize("batch_first", [True, False])
def test_multihead_attention(batch_first):
    rm = real.nn.MultiheadAttention(8, 2, batch_first=batch_first)
    nm = bt.nn.MultiheadAttention(8, 2, batch_first=batch_first)
    copy_state(rm, nm)
    x = np.random.default_rng(0).standard_normal(
        (2, 5, 8) if batch_first else (5, 2, 8)).astype(np.float32)
    ro, rw = rm(real.tensor(x), real.tensor(x), real.tensor(x))
    no, nw = nm(bt.tensor(x), bt.tensor(x), bt.tensor(x))
    same(ro, no, what="MHA output")
    same(rw, nw, what="MHA weights (averaged over heads)")


def test_multihead_attention_mask():
    rm = real.nn.MultiheadAttention(8, 2, batch_first=True)
    nm = bt.nn.MultiheadAttention(8, 2, batch_first=True)
    copy_state(rm, nm)
    x = np.random.default_rng(0).standard_normal((2, 5, 8)).astype(np.float32)
    mask = np.triu(np.ones((5, 5), dtype=bool), 1)
    _, rw = rm(real.tensor(x), real.tensor(x), real.tensor(x), attn_mask=real.tensor(mask))
    _, nw = nm(bt.tensor(x), bt.tensor(x), bt.tensor(x), attn_mask=bt.tensor(mask))
    same(rw, nw, what="MHA causal mask")
    upper = np.triu(np.ones((5, 5)), 1).astype(bool)
    assert np.abs(nw.data[0][upper]).max() < 1e-6, "a masked slot has to be 0"


@pytest.mark.parametrize("kwargs", [
    {}, {"norm_first": True}, {"activation": "gelu"},
])
def test_encoder_layer(kwargs):
    rl = real.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0,
                                         batch_first=True, **kwargs)
    nl = bt.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0,
                                         batch_first=True, **kwargs)
    copy_state(rl, nl)
    rl.eval()
    nl.eval()
    x = np.random.default_rng(0).standard_normal((2, 5, 8)).astype(np.float32)
    same(rl(real.tensor(x)), nl(bt.tensor(x)), tol=1e-4, what=f"EncoderLayer {kwargs}")


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_encoder_layer_mask(mode):
    rl = real.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    nl = bt.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    copy_state(rl, nl)
    getattr(rl, mode)()
    getattr(nl, mode)()
    x = np.random.default_rng(0).standard_normal((2, 5, 8)).astype(np.float32)
    mask = np.triu(np.ones((5, 5), dtype=bool), 1)
    same(rl(real.tensor(x), src_mask=real.tensor(mask)),
         nl(bt.tensor(x), src_mask=bt.tensor(mask)), tol=1e-4,
         what=f"EncoderLayer mask ({mode})")


def test_transformer_encoder_stack():
    rl = real.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    nl = bt.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    re_, ne = real.nn.TransformerEncoder(rl, 3), bt.nn.TransformerEncoder(nl, 3)
    copy_state(re_, ne)
    re_.eval()
    ne.eval()
    x = np.random.default_rng(0).standard_normal((2, 5, 8)).astype(np.float32)
    same(re_(real.tensor(x)), ne(bt.tensor(x)), tol=1e-4, what="TransformerEncoder, 3 layers")


def test_encoder_backward():
    rl = real.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    nl = bt.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    copy_state(rl, nl)
    rl.eval()
    nl.eval()
    x = np.random.default_rng(3).standard_normal((2, 5, 8)).astype(np.float32)
    rx, nx = real.tensor(x, requires_grad=True), bt.tensor(x, requires_grad=True)
    rl(rx).sum().backward()
    nl(nx).sum().backward()
    same(rx.grad, nx.grad, tol=1e-4, what="EncoderLayer input gradient")
    same(rl.self_attn.in_proj_weight.grad, nl.self_attn.in_proj_weight.grad, tol=1e-4,
         what="in_proj_weight gradient")


@pytest.mark.parametrize("kwargs", [{}, {"norm_first": True}])
def test_decoder_layer(kwargs):
    rl = real.nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0,
                                         batch_first=True, **kwargs)
    nl = bt.nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0,
                                         batch_first=True, **kwargs)
    copy_state(rl, nl)
    rl.eval()
    nl.eval()
    rng = np.random.default_rng(0)
    tgt = rng.standard_normal((2, 4, 8)).astype(np.float32)
    mem = rng.standard_normal((2, 6, 8)).astype(np.float32)
    same(rl(real.tensor(tgt), real.tensor(mem)),
         nl(bt.tensor(tgt), bt.tensor(mem)), tol=1e-4, what=f"DecoderLayer {kwargs}")


def test_square_subsequent_mask():
    r = real.nn.Transformer.generate_square_subsequent_mask(4).numpy()
    n = bt.nn.Transformer.generate_square_subsequent_mask(4).data
    assert np.array_equal(np.isneginf(r), np.isneginf(n)), "the masked slots have to be the same"
    assert np.array_equal(np.nan_to_num(r, neginf=0.0), np.nan_to_num(n, neginf=0.0))


def test_float_mask_is_added_not_thresholded():
    """A float mask is **added** to the scores. Lumping "non-zero means masked" together diverges here."""
    rm = real.nn.MultiheadAttention(8, 2, batch_first=True)
    nm = bt.nn.MultiheadAttention(8, 2, batch_first=True)
    copy_state(rm, nm)
    x = np.random.default_rng(0).standard_normal((1, 4, 8)).astype(np.float32)
    bias = np.zeros((4, 4), dtype=np.float32)
    bias[0, 1] = -2.0                    # a mask that **lowers** rather than hides
    _, rw = rm(real.tensor(x), real.tensor(x), real.tensor(x), attn_mask=real.tensor(bias))
    _, nw = nm(bt.tensor(x), bt.tensor(x), bt.tensor(x), attn_mask=bt.tensor(bias))
    same(rw, nw, what="float mask (weight adjustment)")


def test_decoder_layer_causal_mask():
    rl = real.nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    nl = bt.nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    copy_state(rl, nl)
    rl.eval()
    nl.eval()
    rng = np.random.default_rng(0)
    tgt = rng.standard_normal((2, 4, 8)).astype(np.float32)
    mem = rng.standard_normal((2, 6, 8)).astype(np.float32)
    same(rl(real.tensor(tgt), real.tensor(mem),
            tgt_mask=real.nn.Transformer.generate_square_subsequent_mask(4)),
         nl(bt.tensor(tgt), bt.tensor(mem),
            tgt_mask=bt.nn.Transformer.generate_square_subsequent_mask(4)),
         tol=1e-4, what="DecoderLayer causal mask")


def test_full_transformer():
    kw = dict(d_model=8, nhead=2, num_encoder_layers=2, num_decoder_layers=2,
              dim_feedforward=16, dropout=0.0, batch_first=True)
    rt, nt = real.nn.Transformer(**kw), bt.nn.Transformer(**kw)
    copy_state(rt, nt)
    rt.eval()
    nt.eval()
    rng = np.random.default_rng(0)
    src = rng.standard_normal((2, 6, 8)).astype(np.float32)
    tgt = rng.standard_normal((2, 4, 8)).astype(np.float32)
    same(rt(real.tensor(src), real.tensor(tgt)),
         nt(bt.tensor(src), bt.tensor(tgt)), tol=1e-4, what="nn.Transformer")


def test_transformer_backward():
    kw = dict(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
              dim_feedforward=16, dropout=0.0, batch_first=True)
    rt, nt = real.nn.Transformer(**kw), bt.nn.Transformer(**kw)
    copy_state(rt, nt)
    rt.eval()
    nt.eval()
    rng = np.random.default_rng(4)
    src = rng.standard_normal((1, 3, 8)).astype(np.float32)
    tgt = rng.standard_normal((1, 2, 8)).astype(np.float32)
    rs, ns = real.tensor(src, requires_grad=True), bt.tensor(src, requires_grad=True)
    rt(rs, real.tensor(tgt)).sum().backward()
    nt(ns, bt.tensor(tgt)).sum().backward()
    same(rs.grad, ns.grad, tol=1e-4, what="Transformer input gradient")


# Measuring coverage showed the layers below had **never once been exercised.** They all
# passed, but there was no evidence they were right — only that nothing asked. BatchNorm2d
# was wrong for that long in exactly this way.

@pytest.mark.parametrize("mode", ["train", "eval"])
def test_batchnorm1d(mode):
    rl, ml = real.nn.BatchNorm1d(4), bt.nn.BatchNorm1d(4)
    copy_state(rl, ml)
    getattr(rl, mode)()
    getattr(ml, mode)()
    x = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    same(rl(real.tensor(x)), ml(bt.tensor(x)), tol=1e-4, what=f"BatchNorm1d {mode}")


def test_batchnorm1d_backward_and_running():
    rl, ml = real.nn.BatchNorm1d(4), bt.nn.BatchNorm1d(4)
    copy_state(rl, ml)
    x = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    for _ in range(3):
        rl(real.tensor(x))
        ml(bt.tensor(x))
    assert np.allclose(rl.running_mean.numpy(), ml.running_mean, atol=1e-5)
    rx, mx = real.tensor(x, requires_grad=True), bt.tensor(x, requires_grad=True)
    rl(rx).sum().backward()
    ml(mx).sum().backward()
    same(rx.grad, mx.grad, tol=1e-4, what="BatchNorm1d gradient")


@pytest.mark.parametrize("build,shape", [
    (lambda L: L.nn.Softmax(dim=-1), (3, 4)),
    (lambda L: L.nn.LogSoftmax(dim=-1), (3, 4)),
    (lambda L: L.nn.GELU(), (6,)),
    (lambda L: L.nn.SiLU(), (6,)),
    (lambda L: L.nn.LeakyReLU(0.1), (6,)),
    (lambda L: L.nn.ELU(), (6,)),
    (lambda L: L.nn.AvgPool2d(2), (1, 1, 4, 4)),
    (lambda L: L.nn.AdaptiveAvgPool2d(1), (1, 2, 3, 4)),
    (lambda L: L.nn.Unflatten(1, (2, 3)), (2, 6)),
])
def test_stateless_layers(build, shape):
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    same(build(real)(real.tensor(x)), build(bt)(bt.tensor(x)), tol=1e-4,
         what=f"{type(build(bt)).__name__}")


@pytest.mark.parametrize("name", ["L1Loss", "SmoothL1Loss"])
def test_extra_losses(name):
    x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    same(getattr(real.nn, name)()(real.tensor(x), real.tensor(-x)),
         getattr(bt.nn, name)()(bt.tensor(x), bt.tensor(-x)), what=name)


def test_nll_loss_layer():
    x = np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32)
    target = np.array([0, 1, 2, 3])
    same(real.nn.NLLLoss()(real.nn.LogSoftmax(dim=-1)(real.tensor(x)), real.tensor(target)),
         bt.nn.NLLLoss()(bt.nn.LogSoftmax(dim=-1)(bt.tensor(x)), bt.tensor(target)),
         what="NLLLoss")


@pytest.mark.parametrize("name,build,data", [
    ("topk", lambda L, t: L.topk(t, 3).values, np.array([3., 1., 4., 1., 5., 9.], dtype=np.float32)),
    ("sort", lambda L, t: L.sort(t).values, np.array([3., 1., 4., 1., 5., 9.], dtype=np.float32)),
])
def test_selection_keeps_gradient(name, build, data):
    """Picking and then cutting the graph stops training silently — top-k sampling is that place."""
    weights = np.arange(1.0, 4.0, dtype=np.float32)
    rt, mt = real.tensor(data, requires_grad=True), bt.tensor(data, requires_grad=True)
    rv, mv = build(real, rt), build(bt, mt)
    w = weights if rv.shape[0] == 3 else np.arange(1.0, rv.shape[0] + 1.0, dtype=np.float32)
    (rv * real.tensor(w)).sum().backward()
    (mv * bt.tensor(w)).sum().backward()
    same(rt.grad, mt.grad, what=f"{name} gradient")


def test_no_grad_does_not_disable_leaves():
    """`no_grad` only keeps **the result of an operation** from having a graph.
    Turning off hand-made leaves too drops parameters built inside it out of training, silently."""
    for lib in (real, bt):
        with lib.no_grad():
            leaf = lib.tensor([1.0], requires_grad=True)
            derived = lib.tensor([2.0], requires_grad=True) * 2
        assert leaf.requires_grad is True, f"{lib.__name__}: a leaf has to stay on"
        assert derived.requires_grad is False, f"{lib.__name__}: the result of an operation has to be off"


def test_weighted_sampler_actually_weights():
    """The heavier-weighted side really has to be drawn more often. That it runs is no evidence."""
    weights = [1.0, 1.0, 8.0]
    for lib in (real, bt):
        picks = list(lib.utils.data.WeightedRandomSampler(weights, 2000))
        share = picks.count(2) / len(picks)
        assert 0.6 < share < 0.9, f"{lib.__name__}: the third was drawn at a share of {share:.2f}"


def test_concat_dataset():
    for lib in (real, bt):
        a = lib.utils.data.TensorDataset(lib.zeros(2, 3))
        b = lib.utils.data.TensorDataset(lib.ones(3, 3))
        joined = lib.utils.data.ConcatDataset([a, b])
        assert len(joined) == 5
        assert float(joined[0][0].sum()) == 0.0
        assert float(joined[4][0].sum()) == 3.0


def test_minmax_result_unpacks_both_ways():
    for lib in (real, bt):
        result = lib.tensor([[1.0, 3.0], [4.0, 2.0]]).max(dim=1)
        values, indices = result
        assert values.tolist() == result[0].tolist() == result.values.tolist()
        assert indices.tolist() == result[1].tolist() == result.indices.tolist()


# ------------------------------------------------------------------ losses

def test_mse_loss():
    p, t = [[1.0], [2.0]], [[1.5], [1.0]]
    same(real.nn.MSELoss()(real.tensor(p), real.tensor(t)),
         bt.nn.MSELoss()(bt.tensor(p), bt.tensor(t)), what="MSELoss")


def test_bce_with_logits():
    p, t = [[0.5], [-2.0], [8.0]], [[1.0], [0.0], [1.0]]
    same(real.nn.BCEWithLogitsLoss()(real.tensor(p), real.tensor(t)),
         bt.nn.BCEWithLogitsLoss()(bt.tensor(p), bt.tensor(t)), what="BCEWithLogitsLoss")


def test_bce_with_logits_backward():
    t_real, t_mini = real.tensor([[1.0], [0.0]]), bt.tensor([[1.0], [0.0]])
    grads_match(lambda x: real.nn.BCEWithLogitsLoss()(x, t_real),
                lambda x: bt.nn.BCEWithLogitsLoss()(x, t_mini),
                [[0.5], [-1.0]], "BCEWithLogitsLoss")


def test_cross_entropy():
    logits = [[1.0, 2.0, 0.5], [0.1, 0.2, 3.0]]
    target = [1, 2]
    same(real.nn.CrossEntropyLoss()(real.tensor(logits), real.tensor(target)),
         bt.nn.CrossEntropyLoss()(bt.tensor(logits), bt.tensor(target)),
         what="CrossEntropyLoss")


# ---------------------------------------------------------------- optimizer

def _train(module, tensor_fn, optimizer, steps=5):
    x = tensor_fn([[1.0, 2.0], [3.0, 4.0]])
    y = tensor_fn([[1.0], [0.0]])
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((module(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    return module


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_sgd_multi_step(momentum):
    rl, ml = real.nn.Linear(2, 1), bt.nn.Linear(2, 1)
    copy_linear(rl, ml)
    _train(rl, real.tensor, real.optim.SGD(rl.parameters(), lr=0.05, momentum=momentum))
    _train(ml, bt.tensor, bt.optim.SGD(ml.parameters(), lr=0.05, momentum=momentum))
    same(rl.weight, ml.weight, what=f"weights after five steps of SGD(momentum={momentum})")


def test_adam_multi_step():
    rl, ml = real.nn.Linear(2, 1), bt.nn.Linear(2, 1)
    copy_linear(rl, ml)
    _train(rl, real.tensor, real.optim.Adam(rl.parameters(), lr=0.01), steps=10)
    _train(ml, bt.tensor, bt.optim.Adam(ml.parameters(), lr=0.01), steps=10)
    same(rl.weight, ml.weight, what="weights after ten steps of Adam")


def _lr(opt):
    """The standard path for reading the learning rate. Read the same way on both sides or a
    difference stays hidden — torch used to be read through `param_groups` and borch through
    `.lr`, and **the test was covering the difference up.**"""
    return opt.param_groups[0]["lr"]


@pytest.mark.parametrize("make,steps", [
    (lambda L, o: L.optim.lr_scheduler.StepLR(o, step_size=2, gamma=0.5), 6),
    (lambda L, o: L.optim.lr_scheduler.MultiStepLR(o, milestones=[2, 4], gamma=0.5), 6),
    (lambda L, o: L.optim.lr_scheduler.ExponentialLR(o, gamma=0.9), 5),
    (lambda L, o: L.optim.lr_scheduler.CosineAnnealingLR(o, T_max=5), 5),
    (lambda L, o: L.optim.lr_scheduler.LambdaLR(o, lambda e: 1 / (1 + e)), 5),
])
def test_scheduler_trajectory(make, steps):
    """Not one value but **the whole trajectory.** The last one can match while the middle differs."""
    ro = real.optim.SGD(real.nn.Linear(2, 1).parameters(), lr=1.0)
    mo = bt.optim.SGD(bt.nn.Linear(2, 1).parameters(), lr=1.0)
    rs, ms = make(real, ro), make(bt, mo)
    for epoch in range(steps):
        assert abs(_lr(ro) - _lr(mo)) < 1e-9, f"diverged at epoch {epoch}: {_lr(ro)} vs {_lr(mo)}"
        ro.step()          # torch warns that the optimizer must be called first
        mo.step()
        rs.step()
        ms.step()


def test_reduce_on_plateau():
    ro = real.optim.SGD(real.nn.Linear(2, 1).parameters(), lr=1.0)
    mo = bt.optim.SGD(bt.nn.Linear(2, 1).parameters(), lr=1.0)
    rs = real.optim.lr_scheduler.ReduceLROnPlateau(ro, patience=1, factor=0.5)
    ms = bt.optim.lr_scheduler.ReduceLROnPlateau(mo, patience=1, factor=0.5)
    for metric in [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]:
        rs.step(metric)
        ms.step(metric)
        assert abs(_lr(ro) - _lr(mo)) < 1e-9, f"diverged at metric={metric}"


@pytest.mark.parametrize("name,kwargs", [
    ("SGD", {"lr": 0.05}), ("SGD", {"lr": 0.05, "momentum": 0.9}),
    ("SGD", {"lr": 0.05, "weight_decay": 0.01}),
    ("Adam", {"lr": 0.01}), ("Adam", {"lr": 0.01, "weight_decay": 0.01}),
    ("AdamW", {"lr": 0.01, "weight_decay": 0.05}),
    ("RMSprop", {"lr": 0.01}),
])
def test_optimizer_trajectory(name, kwargs):
    rl, ml = real.nn.Linear(2, 1), bt.nn.Linear(2, 1)
    copy_linear(rl, ml)
    ro = getattr(real.optim, name)(rl.parameters(), **kwargs)
    mo = getattr(bt.optim, name)(ml.parameters(), **kwargs)
    _train(rl, real.tensor, ro, steps=8)
    _train(ml, bt.tensor, mo, steps=8)
    same(rl.weight, ml.weight, tol=1e-5, what=f"{name}{kwargs}, eight steps")


def test_optimizer_state_dict_roundtrip():
    """Adam remembers a step size per parameter. Throwing that memory away and continuing to
    train makes the loss jump once — no error, just a curve that looks odd (chapter 6)."""
    ml = bt.nn.Linear(2, 1)
    mo = bt.optim.Adam(ml.parameters(), lr=0.01)
    _train(ml, bt.tensor, mo, steps=5)
    saved = mo.state_dict()
    before = [p.data.copy() for p in ml.parameters()]

    fresh_opt = bt.optim.Adam(ml.parameters(), lr=0.01)
    fresh_opt.load_state_dict(saved)
    _train(ml, bt.tensor, fresh_opt, steps=1)
    after_restored = [p.data.copy() for p in ml.parameters()]

    for p, b in zip(ml.parameters(), before):
        p.data = bt.tensor(b)
    _train(ml, bt.tensor, mo, steps=1)
    for restored, continued in zip(after_restored, [p.data for p in ml.parameters()]):
        assert np.allclose(restored, continued, atol=1e-6), (
            "the loaded optimizer does not go on taking the same step")


# ---------------------------------------------------------- save and load

def test_state_dict_roundtrip(tmp_path):
    m = bt.nn.Sequential(bt.nn.Linear(3, 2), bt.nn.ReLU(), bt.nn.Linear(2, 1))
    x = bt.tensor([[1.0, 2.0, 3.0]])
    before = m(x)

    path = tmp_path / "m.pt"
    bt.save(m.state_dict(), str(path))

    fresh = bt.nn.Sequential(bt.nn.Linear(3, 2), bt.nn.ReLU(), bt.nn.Linear(2, 1))
    fresh.load_state_dict(bt.load(str(path)))
    same(before, fresh(x), what="output of a saved and loaded model")


def test_state_dict_keys_match_real():
    rm = real.nn.Sequential(real.nn.Linear(3, 2), real.nn.ReLU(), real.nn.Linear(2, 1))
    mm = bt.nn.Sequential(bt.nn.Linear(3, 2), bt.nn.ReLU(), bt.nn.Linear(2, 1))
    assert list(rm.state_dict().keys()) == list(mm.state_dict().keys())


def test_load_state_dict_rejects_wrong_shape():
    m = bt.nn.Linear(3, 2)
    bad = {"weight": bt.zeros(5, 5), "bias": bt.zeros(2)}
    with pytest.raises(RuntimeError):
        m.load_state_dict(bad)


# -------------------------------------------------------------------- data

def test_dataloader_batches_match():
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.float32)

    rd = real.utils.data.DataLoader(
        real.utils.data.TensorDataset(real.tensor(x), real.tensor(y)), batch_size=3, shuffle=False)
    md = bt.utils.data.DataLoader(
        bt.utils.data.TensorDataset(bt.tensor(x), bt.tensor(y)), batch_size=3, shuffle=False)

    rb = list(rd)
    mb = list(md)
    assert len(rb) == len(mb) == 4, "ten split by three is four batches, counting the remainder"
    for (rx, ry), (mx, my) in zip(rb, mb):
        same(rx, mx, what="batch x")
        same(ry, my, what="batch y")


def test_dataloader_len():
    ds = bt.utils.data.TensorDataset(bt.zeros(10, 2), bt.zeros(10))
    assert len(bt.utils.data.DataLoader(ds, batch_size=3)) == 4
    assert len(bt.utils.data.DataLoader(ds, batch_size=3, drop_last=True)) == 3


def test_sampler_and_shuffle_conflict():
    ds = bt.utils.data.TensorDataset(bt.zeros(4, 2), bt.zeros(4))
    sampler = bt.utils.data.SequentialSampler(ds)
    with pytest.raises(ValueError):
        bt.utils.data.DataLoader(ds, sampler=sampler, shuffle=True)


# ------------------------------------------------------------- refusals

@pytest.mark.parametrize("call", [
    lambda: bt.zeros(2).to("cuda"),
    lambda: bt.zeros(2).to(device="cuda"),
    lambda: bt.cuda.synchronize(),
    lambda: bt.nn.Module().to("mps"),
])
def test_unsupported_raises_loudly(call):
    """What is absent stops rather than approximating. Better to end here than to give a
    different value quietly.

    This list shrank every time the supported range grew, and twice it broke by being left
    stale. So now it keeps **only what cannot exist in a browser** — entries that will never
    grow.
    """
    with pytest.raises(bt.BorchError):
        call()


def test_sspaddmm_refuses_because_there_is_no_sparse():
    """**Refused because there are no sparse tensors.**

    Not put in the list above — that place holds only "what cannot exist in a browser", and
    sparse can exist. If a sparse arrangement is ever built, this check will turn red then and
    say to implement `sspaddmm` for real.

    Imitating it with a dense tensor would hand over something **whose shape is right and
    whose storage is different**, and whoever learns from that has the wrong idea of what
    sparse is.
    """
    with pytest.raises(bt.BorchError):
        bt.sspaddmm(bt.zeros(2, 4), bt.zeros(2, 3), bt.zeros(3, 4))


def test_cuda_reports_unavailable():
    assert bt.cuda.is_available() is False


def test_no_grad_blocks_graph():
    r, m = pair([1.0, 2.0], requires_grad=True)
    with real.no_grad():
        assert (r * 2).requires_grad is False
    with bt.no_grad():
        assert (m * 2).requires_grad is False


def test_inplace_on_grad_tensor_raises():
    r, m = pair([1.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        r += 1
    with pytest.raises(RuntimeError):
        m += 1


def test_the_three_transposes_differ_only_on_complex():
    """`H`, `mT` and `mH` differ by **whether they conjugate** — over the reals all three give
    the same answer.

    In the golden cases these three go in as `_as_expected`, because borch.ts still refuses a
    complex transpose and the value cannot be asked of all three together. **The moment they
    are wrapped that way the core's value check disappears** — `mH` returning `mT` leaves "did
    the browser refuse?" true all the same. Building `mH` without the conjugate really did
    leave the golden cases green. That missing half is here.
    """
    raw = np.array([[1 + 2j, 3 - 1j]], dtype=np.complex64)
    for name in ("H", "mT", "mH"):
        want = getattr(real.tensor(raw), name).resolve_conj().numpy()
        got = np.asarray(getattr(bt.tensor(raw), name).resolve_conj().tolist())
        assert np.array_equal(got, want), f"{name}: {got} != {want}"
    # Pins **that the three differ from each other** too — this is not asking one function thrice.
    x = bt.tensor(raw)
    assert x.mT.tolist() != x.mH.resolve_conj().tolist(), "mT and mH are the same"
