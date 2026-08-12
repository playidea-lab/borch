"""nanotorch 와 진짜 torch 를 값으로 대조한다.

이 파일이 nanotorch 의 존재 근거다. 흉내가 조금이라도 다르게 동작하면 그것을 배우는
사람은 거짓을 배운다. "돌아간다"로는 부족하고 **같은 숫자가 나와야** 한다.

같은 입력을 양쪽에 넣고 값과 모양을 비교한다. 무작위로 초기화되는 층은 한쪽의 가중치를
복사해 심어 비교 가능하게 만든다.

    uv run --with pytest --with numpy --with torch pytest tests/
"""

import importlib.util
import pathlib

import numpy as np
import pytest
import torch as real

_spec = importlib.util.spec_from_file_location(
    "nanotorch", pathlib.Path(__file__).resolve().parent.parent / "nanotorch.py")
mini = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mini)

TOL = 1e-4


def same(a, b, tol=TOL, what=""):
    """양쪽 텐서의 값과 모양이 같은가."""
    an = a.detach().numpy() if isinstance(a, real.Tensor) else a.data
    bn = b.detach().numpy() if isinstance(b, real.Tensor) else b.data
    assert an.shape == bn.shape, f"{what} 모양이 다르다: {an.shape} vs {bn.shape}"
    assert np.allclose(an, bn, atol=tol, rtol=tol), (
        f"{what} 값이 다르다\n  진짜: {an}\n  축소판: {bn}"
    )


def pair(data, requires_grad=False):
    """같은 데이터로 양쪽 텐서를 만든다."""
    arr = np.asarray(data, dtype=np.float32)
    return (real.tensor(arr, requires_grad=requires_grad),
            mini.tensor(arr, requires_grad=requires_grad))


def grads_match(fn_real, fn_mini, data, what=""):
    """같은 계산의 기울기가 같은가. fn 은 텐서를 받아 스칼라를 돌려준다."""
    r, m = pair(data, requires_grad=True)
    fn_real(r).backward()
    fn_mini(m).backward()
    same(r.grad, m.grad, what=f"{what} 기울기")


# ---------------------------------------------------------------- 축약

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
    """최댓값 자리에만 기울기가 간다 — 나눠 가지면 학습이 미묘하게 달라진다."""
    grads_match(lambda t: t.max(dim=1).values.sum(),
                lambda t: t.max(dim=1).values.sum(), [[1.0, 7.0], [9.0, 2.0]], "max")


def test_std_unbiased_both_ways():
    r, m = pair([1.0, 2.0, 3.0, 4.0])
    same(r.std(unbiased=False), m.std(unbiased=False), what="std(unbiased=False)")
    same(r.std(unbiased=True), m.std(unbiased=True), what="std(unbiased=True)")


# ---------------------------------------------------------------- 모양·인덱싱

def test_indexing_backward():
    grads_match(lambda t: t[0].sum(), lambda t: t[0].sum(), [[1.0, 2.0], [3.0, 4.0]], "t[0]")


def test_fancy_indexing_backward():
    """같은 자리를 두 번 고르면 기울기가 더해져야 한다."""
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
    return mini.tensor([1.0, 2.0, 3.0, 4.0])


def test_transpose_and_matmul_backward():
    r, m = pair([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    rb = real.tensor([[1.0, 0.0], [0.5, 2.0]])
    mb = mini.tensor([[1.0, 0.0], [0.5, 2.0]])
    (r @ rb.T).sum().backward()
    (m @ mb.T).sum().backward()
    same(r.grad, m.grad, what="matmul+transpose 기울기")


def test_batched_matmul():
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    b = np.arange(8, dtype=np.float32).reshape(2, 4, 1)
    same(real.tensor(a) @ real.tensor(b), mini.tensor(a) @ mini.tensor(b), what="배치 matmul")


def test_squeeze_unsqueeze():
    r, m = pair([[1.0], [2.0]])
    same(r.squeeze(-1), m.squeeze(-1), what="squeeze")
    same(r.unsqueeze(0), m.unsqueeze(0), what="unsqueeze")


def test_permute():
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    same(real.tensor(a).permute(2, 0, 1), mini.tensor(a).permute(2, 0, 1), what="permute")


# ---------------------------------------------------------------- 브로드캐스팅

def test_broadcast_bias_backward():
    """(N,D) + (D,) 의 기울기는 (D,) 로 접혀야 한다."""
    rb = real.tensor([1.0, 1.0], requires_grad=True)
    mb = mini.tensor([1.0, 1.0], requires_grad=True)
    (real.tensor([[1.0, 2.0], [3.0, 4.0]]) + rb).sum().backward()
    (mini.tensor([[1.0, 2.0], [3.0, 4.0]]) + mb).sum().backward()
    same(rb.grad, mb.grad, what="브로드캐스트 bias 기울기")


def test_broadcast_column_backward():
    rb = real.tensor([[1.0], [2.0]], requires_grad=True)
    mb = mini.tensor([[1.0], [2.0]], requires_grad=True)
    (rb * real.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    (mb * mini.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    same(rb.grad, mb.grad, what="열 브로드캐스트 기울기")


# ---------------------------------------------------------------- 함수

@pytest.mark.parametrize("fn", ["sigmoid", "relu", "tanh", "exp", "log", "sqrt", "abs"])
def test_elementwise(fn):
    data = [0.5, 1.5, 2.5]
    same(getattr(real, fn)(real.tensor(data)), getattr(mini, fn)(mini.tensor(data)), what=fn)


@pytest.mark.parametrize("fn", ["sigmoid", "relu", "tanh", "exp"])
def test_elementwise_backward(fn):
    grads_match(lambda t: getattr(real, fn)(t).sum(),
                lambda t: getattr(mini, fn)(t).sum(), [0.5, 1.5, -0.5], fn)


def test_sigmoid_extreme():
    """큰 음수·양수에서 터지지 않는가."""
    data = [-1000.0, -50.0, 0.0, 50.0, 1000.0]
    same(real.sigmoid(real.tensor(data)), mini.sigmoid(mini.tensor(data)), what="sigmoid 극단값")


def test_softmax_and_backward():
    data = [[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]
    same(real.nn.functional.softmax(real.tensor(data), dim=-1),
         mini.softmax(mini.tensor(data), dim=-1), what="softmax")
    grads_match(lambda t: (real.nn.functional.softmax(t, dim=-1) * torch_ramp()[:3]).sum(),
                lambda t: (mini.softmax(t, dim=-1) * mini_ramp()[:3]).sum(),
                [1.0, 2.0, 3.0], "softmax")


def test_masked_fill_backward():
    mask_r = real.tensor([[1, 0], [1, 1]])
    mask_m = mini.tensor([[1, 0], [1, 1]])
    grads_match(lambda t: t.masked_fill(mask_r == 0, 0.0).sum(),
                lambda t: t.masked_fill(mask_m == 0, 0.0).sum(),
                [[1.0, 2.0], [3.0, 4.0]], "masked_fill")


def test_where():
    cond_r, cond_m = real.tensor([True, False]), mini.tensor([True, False])
    same(real.where(cond_r, real.tensor([1.0, 2.0]), real.tensor([3.0, 4.0])),
         mini.where(cond_m, mini.tensor([1.0, 2.0]), mini.tensor([3.0, 4.0])), what="where")


def test_tril_triu():
    a = np.ones((3, 3), dtype=np.float32)
    same(real.tril(real.tensor(a)), mini.tril(mini.tensor(a)), what="tril")
    same(real.triu(real.tensor(a), diagonal=1), mini.triu(mini.tensor(a), diagonal=1), what="triu")


# ---------------------------------------------------------------- dtype

def test_dtype_rules():
    assert str(real.tensor([1, 2, 3]).dtype) == str(mini.tensor([1, 2, 3]).dtype)
    assert str(real.tensor([1.0, 2.0]).dtype) == str(mini.tensor([1.0, 2.0]).dtype)


def test_int_tensor_refuses_grad():
    with pytest.raises(RuntimeError):
        real.tensor([1, 2, 3], requires_grad=True)
    with pytest.raises(RuntimeError):
        mini.tensor([1, 2, 3], requires_grad=True)


# ---------------------------------------------------------------- 층

def copy_linear(src, dst):
    dst.weight.data = mini.tensor(src.weight.detach().numpy().copy())
    dst.bias.data = mini.tensor(src.bias.detach().numpy().copy())


def test_linear_forward_and_backward():
    rl, ml = real.nn.Linear(3, 2), mini.nn.Linear(3, 2)
    copy_linear(rl, ml)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ro, mo = rl(real.tensor(x)), ml(mini.tensor(x))
    same(ro, mo, what="Linear 출력")
    ro.sum().backward()
    mo.sum().backward()
    same(rl.weight.grad, ml.weight.grad, what="Linear weight 기울기")
    same(rl.bias.grad, ml.bias.grad, what="Linear bias 기울기")


def test_embedding_forward_and_backward():
    rl, ml = real.nn.Embedding(5, 3), mini.nn.Embedding(5, 3)
    ml.weight.data = mini.tensor(rl.weight.detach().numpy().copy())
    ids = [[0, 2, 2]]
    ro = rl(real.tensor(ids))
    mo = ml(mini.tensor(ids))
    same(ro, mo, what="Embedding 출력")
    ro.sum().backward()
    mo.sum().backward()
    same(rl.weight.grad, ml.weight.grad, what="Embedding 기울기 (같은 번호는 더해져야 한다)")


def test_layernorm():
    rl, ml = real.nn.LayerNorm(4), mini.nn.LayerNorm(4)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, -5.0, 3.0]], dtype=np.float32)
    same(rl(real.tensor(x)), ml(mini.tensor(x)), what="LayerNorm")


def test_batchnorm_backward():
    """순방향만 맞고 역방향이 틀린 채로 오래 있었다 — 평균·분산을 그래프 밖에서 계산했다.

    학습은 돌아가고 손실도 내려가는데 값만 다르다. 순방향만 대조하면 안 잡힌다.
    """
    rl, ml = real.nn.BatchNorm2d(2), mini.nn.BatchNorm2d(2)
    copy_state(rl, ml)
    x = np.random.default_rng(0).standard_normal((4, 2, 3, 3)).astype(np.float32)
    rx, mx = real.tensor(x, requires_grad=True), mini.tensor(x, requires_grad=True)
    rl(rx).sum().backward()
    ml(mx).sum().backward()
    same(rx.grad, mx.grad, tol=1e-4, what="BatchNorm 입력 기울기")
    same(rl.weight.grad, ml.weight.grad, tol=1e-4, what="BatchNorm weight 기울기")
    same(rl.bias.grad, ml.bias.grad, tol=1e-4, what="BatchNorm bias 기울기")
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
    """층마다 **가중치에 기울기가 실제로 도착하는가.** None 이면 그래프가 끊긴 것이다."""
    rl, ml = layer(real), layer(mini)
    copy_state(rl, ml)
    if shape is None:
        rx, mx = real.tensor([[0, 1]]), mini.tensor([[0, 1]])
    else:
        x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        rx, mx = real.tensor(x), mini.tensor(x)
    rl(rx).sum().backward()
    ml(mx).sum().backward()
    for (name, rp), (_, mp) in zip(rl.named_parameters(), ml.named_parameters()):
        assert rp.grad is not None, f"진짜 torch 도 {name} 기울기가 없다 — 케이스가 틀렸다"
        assert mp.grad is not None, f"{name} 에 기울기가 안 왔다 — 그래프가 끊겼다"
        same(rp.grad, mp.grad, tol=1e-4, what=f"{name} 기울기")


def test_data_assignment_rejects_ndarray():
    """torch 는 `p.data = ndarray` 를 거부한다. 받아주면 브라우저에서 되던 코드가
    자기 컴퓨터에서 깨진다 — 관대한 것도 갈리는 것이다."""
    for lib in (real, mini):
        p = lib.nn.Linear(2, 2).weight
        with pytest.raises(TypeError):
            p.data = np.zeros((2, 2), dtype=np.float32)


def test_batchnorm_train_and_eval():
    rl, ml = real.nn.BatchNorm2d(2), mini.nn.BatchNorm2d(2)
    x = np.arange(2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3)
    same(rl(real.tensor(x)), ml(mini.tensor(x)), tol=1e-3, what="BatchNorm2d (학습 모드)")
    rl.eval(); ml.eval()
    same(rl(real.tensor(x)), ml(mini.tensor(x)), tol=1e-3, what="BatchNorm2d (평가 모드)")


def test_dropout_eval_is_identity():
    x = np.ones((4, 4), dtype=np.float32)
    rd, md = real.nn.Dropout(0.5), mini.nn.Dropout(0.5)
    rd.eval(); md.eval()
    same(rd(real.tensor(x)), md(mini.tensor(x)), what="Dropout(eval)")


@pytest.mark.parametrize("padding,stride", [(0, 1), (1, 1), (1, 2)])
def test_conv2d(padding, stride):
    x = np.arange(1 * 2 * 5 * 5, dtype=np.float32).reshape(1, 2, 5, 5)
    w = np.arange(3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3) / 10
    same(real.nn.functional.conv2d(real.tensor(x), real.tensor(w), stride=stride, padding=padding),
         mini.conv2d(mini.tensor(x), mini.tensor(w), stride=stride, padding=padding),
         tol=1e-3, what=f"conv2d(pad={padding}, stride={stride})")


def test_conv2d_backward():
    x = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
    w = np.ones((1, 1, 3, 3), dtype=np.float32)
    rx, mx = real.tensor(x, requires_grad=True), mini.tensor(x, requires_grad=True)
    rw, mw = real.tensor(w, requires_grad=True), mini.tensor(w, requires_grad=True)
    real.nn.functional.conv2d(rx, rw).sum().backward()
    mini.conv2d(mx, mw).sum().backward()
    same(rx.grad, mx.grad, tol=1e-3, what="conv2d 입력 기울기")
    same(rw.grad, mw.grad, tol=1e-3, what="conv2d 필터 기울기")


def test_max_pool2d_and_backward():
    x = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
    rx, mx = real.tensor(x, requires_grad=True), mini.tensor(x, requires_grad=True)
    ro = real.nn.functional.max_pool2d(rx, 2)
    mo = mini.max_pool2d(mx, 2)
    same(ro, mo, what="max_pool2d")
    ro.sum().backward()
    mo.sum().backward()
    same(rx.grad, mx.grad, what="max_pool2d 기울기")


def copy_rnn(src, dst):
    for key, value in src.state_dict().items():
        getattr(dst, key).data = mini.tensor(value.detach().numpy().copy())


@pytest.mark.parametrize("kwargs", [
    {}, {"num_layers": 2}, {"batch_first": True},
    {"nonlinearity": "relu"}, {"bias": False},
])
def test_rnn_forward(kwargs):
    rr, nr = real.nn.RNN(4, 6, **kwargs), mini.nn.RNN(4, 6, **kwargs)
    copy_rnn(rr, nr)
    shape = (5, 3, 4) if kwargs.get("batch_first") else (3, 5, 4)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    ro, rh = rr(real.tensor(x))
    no, nh = nr(mini.tensor(x))
    same(ro, no, what=f"RNN 출력 {kwargs}")
    same(rh, nh, what=f"RNN h_n {kwargs}")


def test_rnn_state_dict_keys():
    rr, nr = real.nn.RNN(3, 4, num_layers=2), mini.nn.RNN(3, 4, num_layers=2)
    assert list(rr.state_dict().keys()) == list(nr.state_dict().keys())


def test_rnn_backward():
    rr, nr = real.nn.RNN(3, 4), mini.nn.RNN(3, 4)
    copy_rnn(rr, nr)
    x = np.random.default_rng(1).standard_normal((5, 2, 3)).astype(np.float32)
    rx, nx = real.tensor(x, requires_grad=True), mini.tensor(x, requires_grad=True)
    rr(rx)[0].sum().backward()
    nr(nx)[0].sum().backward()
    same(rx.grad, nx.grad, tol=1e-4, what="RNN 입력 기울기")
    same(rr.weight_hh_l0.grad, nr.weight_hh_l0.grad, tol=1e-4, what="RNN weight_hh 기울기")


def test_rnn_initial_hidden():
    """h_0 를 직접 주면 그것부터 시작해야 한다."""
    rr, nr = real.nn.RNN(3, 4), mini.nn.RNN(3, 4)
    copy_rnn(rr, nr)
    x = np.zeros((2, 1, 3), dtype=np.float32)
    h0 = np.ones((1, 1, 4), dtype=np.float32) * 0.5
    same(rr(real.tensor(x), real.tensor(h0))[0], nr(mini.tensor(x), mini.tensor(h0))[0],
         what="RNN 초기 은닉 상태")


def test_stack_and_cat_backward():
    a = mini.tensor([1.0, 2.0], requires_grad=True)
    b = mini.tensor([3.0, 4.0], requires_grad=True)
    ra = real.tensor([1.0, 2.0], requires_grad=True)
    rb = real.tensor([3.0, 4.0], requires_grad=True)
    (mini.stack([a, b]) * mini.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    (real.stack([ra, rb]) * real.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    same(ra.grad, a.grad, what="stack 기울기")

    c = mini.tensor([1.0], requires_grad=True)
    rc = real.tensor([1.0], requires_grad=True)
    mini.cat([c, c * 2]).sum().backward()
    real.cat([rc, rc * 2]).sum().backward()
    same(rc.grad, c.grad, what="cat 기울기")


def copy_state(src, dst):
    """torch 쪽 상태를 그대로 심는다. 초기값이 같아야 값을 비교할 수 있다.

    파라미터만이 아니라 **버퍼도** 옮긴다 — BatchNorm 의 running_mean 이 그것이고,
    빠뜨리면 평가 모드에서만 값이 갈린다.
    """
    assert list(src.state_dict().keys()) == list(dst.state_dict().keys()), (
        f"state_dict 키가 다르다\n  torch {list(src.state_dict())}\n  nano  {list(dst.state_dict())}")
    own = dict(dst.named_parameters())
    buffers = dict(dst.named_buffers())
    for key, value in src.state_dict().items():
        array = value.detach().numpy().copy()
        if key in own:
            own[key].data = mini.tensor(array)
        elif key in buffers:
            dst.load_state_dict({key: mini.tensor(array)}, strict=False)


@pytest.mark.parametrize("cls", ["RNN", "LSTM", "GRU"])
@pytest.mark.parametrize("kwargs", [
    {}, {"num_layers": 2}, {"batch_first": True}, {"bias": False},
])
def test_recurrent_forward(cls, kwargs):
    rr = getattr(real.nn, cls)(4, 6, **kwargs)
    nr = getattr(mini.nn, cls)(4, 6, **kwargs)
    copy_state(rr, nr)
    shape = (5, 3, 4) if kwargs.get("batch_first") else (3, 5, 4)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    ro, rs = rr(real.tensor(x))
    no, ns = nr(mini.tensor(x))
    same(ro, no, what=f"{cls} 출력 {kwargs}")
    if cls == "LSTM":
        same(rs[0], ns[0], what="LSTM h_n")
        same(rs[1], ns[1], what="LSTM c_n")
    else:
        same(rs, ns, what=f"{cls} 마지막 상태")


@pytest.mark.parametrize("cls", ["RNN", "LSTM", "GRU"])
def test_recurrent_backward(cls):
    rr, nr = getattr(real.nn, cls)(3, 4), getattr(mini.nn, cls)(3, 4)
    copy_state(rr, nr)
    x = np.random.default_rng(1).standard_normal((5, 2, 3)).astype(np.float32)
    rx, nx = real.tensor(x, requires_grad=True), mini.tensor(x, requires_grad=True)
    rr(rx)[0].sum().backward()
    nr(nx)[0].sum().backward()
    same(rx.grad, nx.grad, tol=1e-4, what=f"{cls} 입력 기울기")
    same(rr.weight_hh_l0.grad, nr.weight_hh_l0.grad, tol=1e-4, what=f"{cls} weight_hh 기울기")


def test_lstm_gate_order():
    """게이트 순서(i, f, g, o)가 틀리면 값은 그럴듯한데 학습이 안 된다."""
    rr, nr = real.nn.LSTM(2, 3), mini.nn.LSTM(2, 3)
    copy_state(rr, nr)
    x = np.ones((1, 1, 2), dtype=np.float32)
    same(rr(real.tensor(x))[1][1], nr(mini.tensor(x))[1][1], what="LSTM c_n (게이트 순서)")


def test_gru_bias_inside_reset_gate():
    """n 게이트에서 r 은 편향까지 포함한 은닉 항에 곱해진다. 밖에 두면 미세하게 어긋난다."""
    rr, nr = real.nn.GRU(2, 3), mini.nn.GRU(2, 3)
    copy_state(rr, nr)
    x = np.random.default_rng(2).standard_normal((4, 1, 2)).astype(np.float32)
    same(rr(real.tensor(x))[0], nr(mini.tensor(x))[0], what="GRU 출력")


@pytest.mark.parametrize("batch_first", [True, False])
def test_multihead_attention(batch_first):
    rm = real.nn.MultiheadAttention(8, 2, batch_first=batch_first)
    nm = mini.nn.MultiheadAttention(8, 2, batch_first=batch_first)
    copy_state(rm, nm)
    x = np.random.default_rng(0).standard_normal(
        (2, 5, 8) if batch_first else (5, 2, 8)).astype(np.float32)
    ro, rw = rm(real.tensor(x), real.tensor(x), real.tensor(x))
    no, nw = nm(mini.tensor(x), mini.tensor(x), mini.tensor(x))
    same(ro, no, what="MHA 출력")
    same(rw, nw, what="MHA 가중치(헤드 평균)")


def test_multihead_attention_mask():
    rm = real.nn.MultiheadAttention(8, 2, batch_first=True)
    nm = mini.nn.MultiheadAttention(8, 2, batch_first=True)
    copy_state(rm, nm)
    x = np.random.default_rng(0).standard_normal((2, 5, 8)).astype(np.float32)
    mask = np.triu(np.ones((5, 5), dtype=bool), 1)
    _, rw = rm(real.tensor(x), real.tensor(x), real.tensor(x), attn_mask=real.tensor(mask))
    _, nw = nm(mini.tensor(x), mini.tensor(x), mini.tensor(x), attn_mask=mini.tensor(mask))
    same(rw, nw, what="MHA 인과 마스크")
    upper = np.triu(np.ones((5, 5)), 1).astype(bool)
    assert np.abs(nw.data[0][upper]).max() < 1e-6, "가려진 자리는 0이어야 한다"


@pytest.mark.parametrize("kwargs", [
    {}, {"norm_first": True}, {"activation": "gelu"},
])
def test_encoder_layer(kwargs):
    rl = real.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0,
                                         batch_first=True, **kwargs)
    nl = mini.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0,
                                         batch_first=True, **kwargs)
    copy_state(rl, nl)
    rl.eval()
    nl.eval()
    x = np.random.default_rng(0).standard_normal((2, 5, 8)).astype(np.float32)
    same(rl(real.tensor(x)), nl(mini.tensor(x)), tol=1e-4, what=f"EncoderLayer {kwargs}")


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_encoder_layer_mask(mode):
    rl = real.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    nl = mini.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    copy_state(rl, nl)
    getattr(rl, mode)()
    getattr(nl, mode)()
    x = np.random.default_rng(0).standard_normal((2, 5, 8)).astype(np.float32)
    mask = np.triu(np.ones((5, 5), dtype=bool), 1)
    same(rl(real.tensor(x), src_mask=real.tensor(mask)),
         nl(mini.tensor(x), src_mask=mini.tensor(mask)), tol=1e-4,
         what=f"EncoderLayer 마스크 ({mode})")


def test_transformer_encoder_stack():
    rl = real.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    nl = mini.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    re_, ne = real.nn.TransformerEncoder(rl, 3), mini.nn.TransformerEncoder(nl, 3)
    copy_state(re_, ne)
    re_.eval()
    ne.eval()
    x = np.random.default_rng(0).standard_normal((2, 5, 8)).astype(np.float32)
    same(re_(real.tensor(x)), ne(mini.tensor(x)), tol=1e-4, what="TransformerEncoder 3층")


def test_encoder_backward():
    rl = real.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    nl = mini.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    copy_state(rl, nl)
    rl.eval()
    nl.eval()
    x = np.random.default_rng(3).standard_normal((2, 5, 8)).astype(np.float32)
    rx, nx = real.tensor(x, requires_grad=True), mini.tensor(x, requires_grad=True)
    rl(rx).sum().backward()
    nl(nx).sum().backward()
    same(rx.grad, nx.grad, tol=1e-4, what="EncoderLayer 입력 기울기")
    same(rl.self_attn.in_proj_weight.grad, nl.self_attn.in_proj_weight.grad, tol=1e-4,
         what="in_proj_weight 기울기")


@pytest.mark.parametrize("kwargs", [{}, {"norm_first": True}])
def test_decoder_layer(kwargs):
    rl = real.nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0,
                                         batch_first=True, **kwargs)
    nl = mini.nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0,
                                         batch_first=True, **kwargs)
    copy_state(rl, nl)
    rl.eval()
    nl.eval()
    rng = np.random.default_rng(0)
    tgt = rng.standard_normal((2, 4, 8)).astype(np.float32)
    mem = rng.standard_normal((2, 6, 8)).astype(np.float32)
    same(rl(real.tensor(tgt), real.tensor(mem)),
         nl(mini.tensor(tgt), mini.tensor(mem)), tol=1e-4, what=f"DecoderLayer {kwargs}")


def test_square_subsequent_mask():
    r = real.nn.Transformer.generate_square_subsequent_mask(4).numpy()
    n = mini.nn.Transformer.generate_square_subsequent_mask(4).data
    assert np.array_equal(np.isneginf(r), np.isneginf(n)), "가려지는 자리가 같아야 한다"
    assert np.array_equal(np.nan_to_num(r, neginf=0.0), np.nan_to_num(n, neginf=0.0))


def test_float_mask_is_added_not_thresholded():
    """실수 마스크는 점수에 **더한다.** 0 이 아니면 가린다고 뭉뚱그리면 여기서 갈린다."""
    rm = real.nn.MultiheadAttention(8, 2, batch_first=True)
    nm = mini.nn.MultiheadAttention(8, 2, batch_first=True)
    copy_state(rm, nm)
    x = np.random.default_rng(0).standard_normal((1, 4, 8)).astype(np.float32)
    bias = np.zeros((4, 4), dtype=np.float32)
    bias[0, 1] = -2.0                    # 가리는 게 아니라 **낮추는** 마스크
    _, rw = rm(real.tensor(x), real.tensor(x), real.tensor(x), attn_mask=real.tensor(bias))
    _, nw = nm(mini.tensor(x), mini.tensor(x), mini.tensor(x), attn_mask=mini.tensor(bias))
    same(rw, nw, what="실수 마스크(가중치 조절)")


def test_decoder_layer_causal_mask():
    rl = real.nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    nl = mini.nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    copy_state(rl, nl)
    rl.eval()
    nl.eval()
    rng = np.random.default_rng(0)
    tgt = rng.standard_normal((2, 4, 8)).astype(np.float32)
    mem = rng.standard_normal((2, 6, 8)).astype(np.float32)
    same(rl(real.tensor(tgt), real.tensor(mem),
            tgt_mask=real.nn.Transformer.generate_square_subsequent_mask(4)),
         nl(mini.tensor(tgt), mini.tensor(mem),
            tgt_mask=mini.nn.Transformer.generate_square_subsequent_mask(4)),
         tol=1e-4, what="DecoderLayer 인과 마스크")


def test_full_transformer():
    kw = dict(d_model=8, nhead=2, num_encoder_layers=2, num_decoder_layers=2,
              dim_feedforward=16, dropout=0.0, batch_first=True)
    rt, nt = real.nn.Transformer(**kw), mini.nn.Transformer(**kw)
    copy_state(rt, nt)
    rt.eval()
    nt.eval()
    rng = np.random.default_rng(0)
    src = rng.standard_normal((2, 6, 8)).astype(np.float32)
    tgt = rng.standard_normal((2, 4, 8)).astype(np.float32)
    same(rt(real.tensor(src), real.tensor(tgt)),
         nt(mini.tensor(src), mini.tensor(tgt)), tol=1e-4, what="nn.Transformer")


def test_transformer_backward():
    kw = dict(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
              dim_feedforward=16, dropout=0.0, batch_first=True)
    rt, nt = real.nn.Transformer(**kw), mini.nn.Transformer(**kw)
    copy_state(rt, nt)
    rt.eval()
    nt.eval()
    rng = np.random.default_rng(4)
    src = rng.standard_normal((1, 3, 8)).astype(np.float32)
    tgt = rng.standard_normal((1, 2, 8)).astype(np.float32)
    rs, ns = real.tensor(src, requires_grad=True), mini.tensor(src, requires_grad=True)
    rt(rs, real.tensor(tgt)).sum().backward()
    nt(ns, mini.tensor(tgt)).sum().backward()
    same(rs.grad, ns.grad, tol=1e-4, what="Transformer 입력 기울기")


# 커버리지를 재보니 아래 층들은 **한 번도 안 돌아본 채** 있었다. 전부 통과했지만,
# 검사가 없었을 뿐이지 맞다는 근거는 없었다 — BatchNorm2d 가 그렇게 오래 틀려 있었다.

@pytest.mark.parametrize("mode", ["train", "eval"])
def test_batchnorm1d(mode):
    rl, ml = real.nn.BatchNorm1d(4), mini.nn.BatchNorm1d(4)
    copy_state(rl, ml)
    getattr(rl, mode)()
    getattr(ml, mode)()
    x = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    same(rl(real.tensor(x)), ml(mini.tensor(x)), tol=1e-4, what=f"BatchNorm1d {mode}")


def test_batchnorm1d_backward_and_running():
    rl, ml = real.nn.BatchNorm1d(4), mini.nn.BatchNorm1d(4)
    copy_state(rl, ml)
    x = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    for _ in range(3):
        rl(real.tensor(x))
        ml(mini.tensor(x))
    assert np.allclose(rl.running_mean.numpy(), ml.running_mean, atol=1e-5)
    rx, mx = real.tensor(x, requires_grad=True), mini.tensor(x, requires_grad=True)
    rl(rx).sum().backward()
    ml(mx).sum().backward()
    same(rx.grad, mx.grad, tol=1e-4, what="BatchNorm1d 기울기")


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
    same(build(real)(real.tensor(x)), build(mini)(mini.tensor(x)), tol=1e-4,
         what=f"{type(build(mini)).__name__}")


@pytest.mark.parametrize("name", ["L1Loss", "SmoothL1Loss"])
def test_extra_losses(name):
    x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    same(getattr(real.nn, name)()(real.tensor(x), real.tensor(-x)),
         getattr(mini.nn, name)()(mini.tensor(x), mini.tensor(-x)), what=name)


def test_nll_loss_layer():
    x = np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32)
    target = np.array([0, 1, 2, 3])
    same(real.nn.NLLLoss()(real.nn.LogSoftmax(dim=-1)(real.tensor(x)), real.tensor(target)),
         mini.nn.NLLLoss()(mini.nn.LogSoftmax(dim=-1)(mini.tensor(x)), mini.tensor(target)),
         what="NLLLoss")


@pytest.mark.parametrize("name,build,data", [
    ("topk", lambda L, t: L.topk(t, 3).values, np.array([3., 1., 4., 1., 5., 9.], dtype=np.float32)),
    ("sort", lambda L, t: L.sort(t).values, np.array([3., 1., 4., 1., 5., 9.], dtype=np.float32)),
])
def test_selection_keeps_gradient(name, build, data):
    """뽑기만 하고 그래프를 끊으면 학습이 조용히 멈춘다 — top-k 샘플링이 그 자리다."""
    weights = np.arange(1.0, 4.0, dtype=np.float32)
    rt, mt = real.tensor(data, requires_grad=True), mini.tensor(data, requires_grad=True)
    rv, mv = build(real, rt), build(mini, mt)
    w = weights if rv.shape[0] == 3 else np.arange(1.0, rv.shape[0] + 1.0, dtype=np.float32)
    (rv * real.tensor(w)).sum().backward()
    (mv * mini.tensor(w)).sum().backward()
    same(rt.grad, mt.grad, what=f"{name} 기울기")


def test_no_grad_does_not_disable_leaves():
    """no_grad 는 **연산 결과**가 그래프를 안 갖게 할 뿐이다.
    직접 만든 잎까지 끄면 그 안에서 만든 파라미터가 학습에서 조용히 빠진다."""
    for lib in (real, mini):
        with lib.no_grad():
            leaf = lib.tensor([1.0], requires_grad=True)
            derived = lib.tensor([2.0], requires_grad=True) * 2
        assert leaf.requires_grad is True, f"{lib.__name__}: 잎은 켜져 있어야 한다"
        assert derived.requires_grad is False, f"{lib.__name__}: 연산 결과는 꺼져야 한다"


def test_weighted_sampler_actually_weights():
    """가중치가 큰 쪽이 실제로 더 자주 뽑혀야 한다. 도는 것만으로는 근거가 없다."""
    weights = [1.0, 1.0, 8.0]
    for lib in (real, mini):
        picks = list(lib.utils.data.WeightedRandomSampler(weights, 2000))
        share = picks.count(2) / len(picks)
        assert 0.6 < share < 0.9, f"{lib.__name__}: 세 번째가 {share:.2f} 비율로 뽑혔다"


def test_concat_dataset():
    for lib in (real, mini):
        a = lib.utils.data.TensorDataset(lib.zeros(2, 3))
        b = lib.utils.data.TensorDataset(lib.ones(3, 3))
        joined = lib.utils.data.ConcatDataset([a, b])
        assert len(joined) == 5
        assert float(joined[0][0].sum()) == 0.0
        assert float(joined[4][0].sum()) == 3.0


def test_minmax_result_unpacks_both_ways():
    for lib in (real, mini):
        result = lib.tensor([[1.0, 3.0], [4.0, 2.0]]).max(dim=1)
        values, indices = result
        assert values.tolist() == result[0].tolist() == result.values.tolist()
        assert indices.tolist() == result[1].tolist() == result.indices.tolist()


# ---------------------------------------------------------------- 손실

def test_mse_loss():
    p, t = [[1.0], [2.0]], [[1.5], [1.0]]
    same(real.nn.MSELoss()(real.tensor(p), real.tensor(t)),
         mini.nn.MSELoss()(mini.tensor(p), mini.tensor(t)), what="MSELoss")


def test_bce_with_logits():
    p, t = [[0.5], [-2.0], [8.0]], [[1.0], [0.0], [1.0]]
    same(real.nn.BCEWithLogitsLoss()(real.tensor(p), real.tensor(t)),
         mini.nn.BCEWithLogitsLoss()(mini.tensor(p), mini.tensor(t)), what="BCEWithLogitsLoss")


def test_bce_with_logits_backward():
    t_real, t_mini = real.tensor([[1.0], [0.0]]), mini.tensor([[1.0], [0.0]])
    grads_match(lambda x: real.nn.BCEWithLogitsLoss()(x, t_real),
                lambda x: mini.nn.BCEWithLogitsLoss()(x, t_mini),
                [[0.5], [-1.0]], "BCEWithLogitsLoss")


def test_cross_entropy():
    logits = [[1.0, 2.0, 0.5], [0.1, 0.2, 3.0]]
    target = [1, 2]
    same(real.nn.CrossEntropyLoss()(real.tensor(logits), real.tensor(target)),
         mini.nn.CrossEntropyLoss()(mini.tensor(logits), mini.tensor(target)),
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
    rl, ml = real.nn.Linear(2, 1), mini.nn.Linear(2, 1)
    copy_linear(rl, ml)
    _train(rl, real.tensor, real.optim.SGD(rl.parameters(), lr=0.05, momentum=momentum))
    _train(ml, mini.tensor, mini.optim.SGD(ml.parameters(), lr=0.05, momentum=momentum))
    same(rl.weight, ml.weight, what=f"SGD(momentum={momentum}) 다섯 스텝 뒤 가중치")


def test_adam_multi_step():
    rl, ml = real.nn.Linear(2, 1), mini.nn.Linear(2, 1)
    copy_linear(rl, ml)
    _train(rl, real.tensor, real.optim.Adam(rl.parameters(), lr=0.01), steps=10)
    _train(ml, mini.tensor, mini.optim.Adam(ml.parameters(), lr=0.01), steps=10)
    same(rl.weight, ml.weight, what="Adam 열 스텝 뒤 가중치")


def _lr(opt):
    """학습률을 읽는 표준 경로. 양쪽에서 같은 식으로 읽어야 차이가 드러난다 —
    전에는 torch 만 param_groups 로 읽고 nano 는 `.lr` 로 읽어, **테스트가 차이를 덮고 있었다.**"""
    return opt.param_groups[0]["lr"]


@pytest.mark.parametrize("make,steps", [
    (lambda L, o: L.optim.lr_scheduler.StepLR(o, step_size=2, gamma=0.5), 6),
    (lambda L, o: L.optim.lr_scheduler.MultiStepLR(o, milestones=[2, 4], gamma=0.5), 6),
    (lambda L, o: L.optim.lr_scheduler.ExponentialLR(o, gamma=0.9), 5),
    (lambda L, o: L.optim.lr_scheduler.CosineAnnealingLR(o, T_max=5), 5),
    (lambda L, o: L.optim.lr_scheduler.LambdaLR(o, lambda e: 1 / (1 + e)), 5),
])
def test_scheduler_trajectory(make, steps):
    """한 값이 아니라 **궤적 전체**를 본다. 마지막만 맞고 중간이 다른 경우가 있다."""
    ro = real.optim.SGD(real.nn.Linear(2, 1).parameters(), lr=1.0)
    mo = mini.optim.SGD(mini.nn.Linear(2, 1).parameters(), lr=1.0)
    rs, ms = make(real, ro), make(mini, mo)
    for epoch in range(steps):
        assert abs(_lr(ro) - _lr(mo)) < 1e-9, f"{epoch}에폭에서 갈렸다: {_lr(ro)} vs {_lr(mo)}"
        ro.step()          # torch 는 optimizer 를 먼저 부르라고 경고한다
        mo.step()
        rs.step()
        ms.step()


def test_reduce_on_plateau():
    ro = real.optim.SGD(real.nn.Linear(2, 1).parameters(), lr=1.0)
    mo = mini.optim.SGD(mini.nn.Linear(2, 1).parameters(), lr=1.0)
    rs = real.optim.lr_scheduler.ReduceLROnPlateau(ro, patience=1, factor=0.5)
    ms = mini.optim.lr_scheduler.ReduceLROnPlateau(mo, patience=1, factor=0.5)
    for metric in [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]:
        rs.step(metric)
        ms.step(metric)
        assert abs(_lr(ro) - _lr(mo)) < 1e-9, f"metric={metric} 에서 갈렸다"


@pytest.mark.parametrize("name,kwargs", [
    ("SGD", {"lr": 0.05}), ("SGD", {"lr": 0.05, "momentum": 0.9}),
    ("SGD", {"lr": 0.05, "weight_decay": 0.01}),
    ("Adam", {"lr": 0.01}), ("Adam", {"lr": 0.01, "weight_decay": 0.01}),
    ("AdamW", {"lr": 0.01, "weight_decay": 0.05}),
    ("RMSprop", {"lr": 0.01}),
])
def test_optimizer_trajectory(name, kwargs):
    rl, ml = real.nn.Linear(2, 1), mini.nn.Linear(2, 1)
    copy_linear(rl, ml)
    ro = getattr(real.optim, name)(rl.parameters(), **kwargs)
    mo = getattr(mini.optim, name)(ml.parameters(), **kwargs)
    _train(rl, real.tensor, ro, steps=8)
    _train(ml, mini.tensor, mo, steps=8)
    same(rl.weight, ml.weight, tol=1e-5, what=f"{name}{kwargs} 여덟 스텝")


def test_optimizer_state_dict_roundtrip():
    """Adam 은 파라미터마다 보폭을 기억한다. 그 기억을 버리고 이어 학습하면
    손실이 한 번 튄다 — 오류는 안 나고 곡선만 이상해진다(6장)."""
    ml = mini.nn.Linear(2, 1)
    mo = mini.optim.Adam(ml.parameters(), lr=0.01)
    _train(ml, mini.tensor, mo, steps=5)
    saved = mo.state_dict()
    before = [p.data.copy() for p in ml.parameters()]

    fresh_opt = mini.optim.Adam(ml.parameters(), lr=0.01)
    fresh_opt.load_state_dict(saved)
    _train(ml, mini.tensor, fresh_opt, steps=1)
    after_restored = [p.data.copy() for p in ml.parameters()]

    for p, b in zip(ml.parameters(), before):
        p.data = mini.tensor(b)
    _train(ml, mini.tensor, mo, steps=1)
    for restored, continued in zip(after_restored, [p.data for p in ml.parameters()]):
        assert np.allclose(restored, continued, atol=1e-6), (
            "불러온 optimizer 가 이어서 같은 걸음을 걷지 않는다")


# ---------------------------------------------------------------- 저장·불러오기

def test_state_dict_roundtrip(tmp_path):
    m = mini.nn.Sequential(mini.nn.Linear(3, 2), mini.nn.ReLU(), mini.nn.Linear(2, 1))
    x = mini.tensor([[1.0, 2.0, 3.0]])
    before = m(x)

    path = tmp_path / "m.pt"
    mini.save(m.state_dict(), str(path))

    fresh = mini.nn.Sequential(mini.nn.Linear(3, 2), mini.nn.ReLU(), mini.nn.Linear(2, 1))
    fresh.load_state_dict(mini.load(str(path)))
    same(before, fresh(x), what="저장하고 불러온 모델의 출력")


def test_state_dict_keys_match_real():
    rm = real.nn.Sequential(real.nn.Linear(3, 2), real.nn.ReLU(), real.nn.Linear(2, 1))
    mm = mini.nn.Sequential(mini.nn.Linear(3, 2), mini.nn.ReLU(), mini.nn.Linear(2, 1))
    assert list(rm.state_dict().keys()) == list(mm.state_dict().keys())


def test_load_state_dict_rejects_wrong_shape():
    m = mini.nn.Linear(3, 2)
    bad = {"weight": mini.zeros(5, 5), "bias": mini.zeros(2)}
    with pytest.raises(RuntimeError):
        m.load_state_dict(bad)


# ---------------------------------------------------------------- 데이터

def test_dataloader_batches_match():
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.float32)

    rd = real.utils.data.DataLoader(
        real.utils.data.TensorDataset(real.tensor(x), real.tensor(y)), batch_size=3, shuffle=False)
    md = mini.utils.data.DataLoader(
        mini.utils.data.TensorDataset(mini.tensor(x), mini.tensor(y)), batch_size=3, shuffle=False)

    rb = list(rd)
    mb = list(md)
    assert len(rb) == len(mb) == 4, "10개를 3씩 나누면 마지막 자투리까지 네 묶음"
    for (rx, ry), (mx, my) in zip(rb, mb):
        same(rx, mx, what="배치 x")
        same(ry, my, what="배치 y")


def test_dataloader_len():
    ds = mini.utils.data.TensorDataset(mini.zeros(10, 2), mini.zeros(10))
    assert len(mini.utils.data.DataLoader(ds, batch_size=3)) == 4
    assert len(mini.utils.data.DataLoader(ds, batch_size=3, drop_last=True)) == 3


def test_sampler_and_shuffle_conflict():
    ds = mini.utils.data.TensorDataset(mini.zeros(4, 2), mini.zeros(4))
    sampler = mini.utils.data.SequentialSampler(ds)
    with pytest.raises(ValueError):
        mini.utils.data.DataLoader(ds, sampler=sampler, shuffle=True)


# ---------------------------------------------------------------- 거절

@pytest.mark.parametrize("call", [
    lambda: mini.zeros(2).to("cuda"),
    lambda: mini.zeros(2).to(device="cuda"),
    lambda: mini.cuda.synchronize(),
    lambda: mini.nn.Module().to("mps"),
])
def test_unsupported_raises_loudly(call):
    """없는 것은 근사하지 않고 멈춘다. 조용히 다른 값을 내느니 여기서 끝낸다.

    이 목록은 지원 범위가 늘 때마다 줄어들었고, 그때마다 낡은 채로 남아 두 번 깨졌다.
    그래서 지금은 **브라우저에 존재할 수 없는 것만** 남긴다 — 늘어날 일이 없는 항목들이다.
    """
    with pytest.raises(mini.NanoTorchError):
        call()


def test_cuda_reports_unavailable():
    assert mini.cuda.is_available() is False


def test_no_grad_blocks_graph():
    r, m = pair([1.0, 2.0], requires_grad=True)
    with real.no_grad():
        assert (r * 2).requires_grad is False
    with mini.no_grad():
        assert (m * 2).requires_grad is False


def test_inplace_on_grad_tensor_raises():
    r, m = pair([1.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        r += 1
    with pytest.raises(RuntimeError):
        m += 1
