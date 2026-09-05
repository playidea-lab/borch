"""borch_webgpu — **a thin binding on top of borch.ts.**

Used as `import borch_webgpu as torch`, with hand-written WGSL kernels running
underneath.

## This name changed hands once

It used to belong to the TF.js implementation. That one was **5,307 lines**,
because TF.js supplies 104 primitive operations and nothing else, so the
autograd tape, `nn.Module` and the optimisers all had to be rebuilt in Python.
This one is **11,580 lines** — borch.ts already has all of it, so Python's job
is swapping names across.

`_data.py` is nearly identical in both. It came over unchanged, sits on numpy
and OPFS, and does not care what is underneath. Everything that shrank shrank
elsewhere.

The name describes what a user sees: in a browser, on the GPU. That meaning did
not change, so the name did not either; what was swapped is the floor.
Measured on the same benchmark at batch 64 it went 154.7ms to 123.4ms, and 859
golden cases asked both sides the same questions.

Even the name-swapping is not written by hand. `__getattr__` on the module and
on the tensor turns `masked_select` into `maskedSelect` and stops when there is
no such name. **It also stops when the other side does not accept an argument
being passed** — JavaScript discards surplus arguments silently, and while that
check was missing `sum(dim=1)` produced a full sum that ignored the dimension.

## Why it is synchronous — this was measured

WebGPU has no synchronous read. borch.ts's `toArray()` returns a promise. Laid
down as-is that becomes `await loss.item()`, and then **"tutorial code runs by
changing only the import"** — the one claim this project makes — is broken.

Pyodide's `run_sync` fills that gap, standing on JSPI. Measured:
`tests/browser/sync_probe.py` produces `[2,4,6]`. **One condition:** Python has
to be entered asynchronously (`runPythonAsync`). Enter synchronously and it
stops with `RuntimeError: No suspender`, which is not a limit of the library
but the absence of anywhere on that stack to suspend. The runners already enter
asynchronously.

## It only runs inside a browser

It reaches for `js.borch`. Importing it under native CPython stops immediately —
falling back to something else quietly would let somebody believe they ran on a
GPU, which is the kind of thing this project dislikes most. The core (`borch`,
on numpy) is the native answer.

## Where it stands

It passes **all 1830 golden cases**. The core (numpy) sees 1777 of them; the
other 53 are things the core refuses on purpose (1-D and 3-D convolution, rank
7 and 8), so it is not asked.

The golden holds the boundary. Nothing missing gets approximated into a green
tick.
"""

try:                                                    # pragma: no cover
    import js as _js
except ImportError as exc:                              # pragma: no cover
    raise ImportError(
        "borch_webgpu only runs inside a browser — outside Pyodide there is no `js`.\n"
        "  Natively, use `borch` (numpy).") from exc

# `js.borch` is the page's borch.ts when the page loaded one. **When nothing did — a
# notebook, a worker, a bare Pyodide — `_base` boots the borch.ts this wheel carries**
# (`_boot`), so the import is the whole setup. What cannot be helped is a machine with
# no WebGPU: that stops here, by name, rather than quietly running something else.
try:
    from ._base import Tensor, tensor                    # noqa: E402,F401
except Exception as exc:                                 # pragma: no cover
    if getattr(_js, "borch", None) is None:
        raise ImportError(
            "borch_webgpu could not bring up borch.ts: no WebGPU adapter, or the browser "
            "has no JSPI (`run_sync`).\n  Chrome or Edge 137+ on a machine with a GPU; on "
            f"Linux with NVIDIA, chrome://flags #enable-unsafe-webgpu and #enable-vulkan.\n  ({exc})") from exc
    raise
# **Named things are imported first.** The module's `__getattr__` only receives
# names that are not here, so things whose first argument is not a tensor —
# `linalg`, `einsum` — must not leak through to it. `linalg` did get caught as a
# function once, and `linalg.cholesky` failed outright.
from ._ops import (                                      # noqa: E402,F401
    aminmax, arange, argmax, argmin, as_tensor, cat, chunk, clamp, clip, einsum, eye,
    from_numpy,
    flatten, flip,
    full, linalg, linspace, matrix_power, memory, no_grad, norm, numel,
    ones, pow,
    quantile, rand, randn, repeat_interleave, scope, split, squeeze, stack,
    # Where cost gets measured. One group with `memory`, used by
    # `tests/browser/cost.py`. `pooled` and `empty_cache` are **names torch does
    # not have** — the same grain as `backend` and `fetch_cached`, which exist
    # only in a browser. Imitating `torch.cuda` was the other option and was
    # rejected: `is_available()` is false there, so the textbook idiom around it
    # becomes a dead line.
    dispatches, empty_cache, keep_alive, last_scope, pooled, submits,
    sum, swapdims, transpose, where, zeros,
    # Names torch offers as a second spelling — a name over a combination.
    add, adjoint, block_diag, broadcast_shapes, broadcast_tensors, broadcast_to,
    column_stack, concat, concatenate, div, divide, dstack, floor_divide, fmod,
    hstack, moveaxis, mul, multiply, remainder, row_stack, rsub, sub, subtract, t,
    true_divide, vstack,
    # Things where the computation itself was missing.
    cross, empty, empty_like, float_power, fmax, fmin, full_like, inner,
    isclose, isin, isneginf, ones_like, zeros_like,
    isposinf, isreal, kron, lerp, logical_xor, logspace, meshgrid, nan_to_num,
    rand_like, randint_like, randn_like, scalar_tensor, std_mean, var_mean, vdot,
    # The indexing side.
    bucketize, index_add, index_copy, index_fill, scatter, scatter_add,
    searchsorted, take, take_along_dim,
    # The numeric family. `lgamma`, `digamma` and `erfinv` live on the WGSL side,
    # so `__getattr__` passes those across.
    cdist, corrcoef, cov, cumulative_trapezoid, tensordot, trapezoid,
    # QR in reflector form. The counterpart to `linalg.householder_product`, so
    # it sits at the top level too.
    geqrf,
    # Window functions. They take **a count**, not a tensor, so `__getattr__`
    # cannot pass them across.
    bartlett_window, blackman_window, hamming_window, hann_window, kaiser_window,
    # Things that do different work on bool and on int, plus the out-of-place `fill`.
    bitwise_not, fill,
    # Shape and indexing. **`as_strided` is a view in torch and a copy here.**
    cartesian_prod, chain_matmul, combinations, index_put, index_put_,
    split_with_sizes, tensor_split, tril_indices, triu_indices,
    unique_consecutive, unravel_index, vander,
    # **Sparse-only, so absent.** The name stays and stops where it is called.
    sspaddmm,
    # Four whose core rows named what they are for rather than what they need.
    # borch.ts keeps all four as module functions, so `__getattr__` — which asks
    # `Tensor.prototype` — cannot reach them and each needs a line.
    cudnn_is_acceptable, is_vulkan_available, narrow_copy, segment_reduce,
    get_autocast_cpu_dtype, get_autocast_dtype, get_autocast_gpu_dtype,
    get_autocast_ipu_dtype, get_autocast_xla_dtype, is_autocast_cache_enabled,
    is_autocast_cpu_enabled, is_autocast_enabled, is_autocast_ipu_enabled,
    is_autocast_xla_enabled,
    are_deterministic_algorithms_enabled, get_deterministic_debug_mode,
    get_float32_matmul_precision, is_anomaly_check_nan_enabled, is_anomaly_enabled,
    is_deterministic_algorithms_warn_only_enabled, is_warn_always_enabled,
    get_num_interop_threads, get_num_threads,
    # Top-level linear algebra. **Only the three whose names collide with the
    # `linalg` namespace are written out** — `__getattr__` passes the rest to the
    # first argument's method.
    lu, lu_solve, lu_unpack,
    # Three names torch removed in 1.9 — written out so they refuse instead of
    # reaching the first argument's method, which computes. And three whose list the
    # forwarder got wrong: `round` (keyword-only `decimals`), `softmax`, `log_softmax`
    # (`dim` required).
    solve, lstsq, matrix_rank, round, softmax, log_softmax,
    # Statistics. Four of the random ones are frozen by **their bounds** rather
    # than their values. The last two keep a name and refuse.
    bernoulli, binomial, hash_tensor, histogramdd, normal, poisson, trapz,
    # Fourier. `fft` is a namespace; `stft` and `istft` are top level — the same
    # places torch puts them.
    fft, istft, stft,
    # **`special` is twenty names this binding already answers to, under torch's
    # second spelling for them.** A namespace rather than a passthrough, because
    # thirty-six of that namespace's names are arithmetic we do not have and a
    # passthrough would claim every one of them — see `_Special` in `_ops.py`.
    special,
    # The eight top-level recurrent ones. The same computation as the layers
    # (`nn.LSTM`) but taking the weights as a list.
    gru, gru_cell, lstm, lstm_cell, rnn_relu, rnn_relu_cell, rnn_tanh,
    rnn_tanh_cell,
    # What was left at the top level. `device` is the biggest of them — it is the
    # first line of half the tutorials.
    constant_pad_nd, dequantize, device, fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine, igamma, igammac, polygamma, resize_as_,
    # Complex. `imag` refuses on a real tensor, and **torch itself does that**
    # (measured).
    #
    # **`complex` shadows a Python builtin.** It is exported under that name
    # anyway because `torch.complex(re, im)` is torch's name, and this is torch's
    # place rather than Python's. Inside `_ops.py` the shadowing is a real
    # problem, so the complex test lives separately as `_is_cplx`.
    angle, asarray, complex, conj, conj_physical, conj_physical_,
    empty_permuted, empty_strided, frombuffer, imag, is_complex, is_conj,
    is_neg, polar, real, resolve_conj, resolve_neg, view_as_complex,
    view_as_real, range_top as range,
    # **Names that exist only at the top level.** Some have a different signature
    # from their `F` counterpart, so they are moved rather than aliased.
    alpha_dropout_, batch_norm, ctc_loss, dropout_, feature_alpha_dropout_,
    feature_dropout, feature_dropout_, grid_sampler, max_pool1d_with_indices,
    nan_to_num_,
    # Gradient mode.
    enable_grad, inference_mode, is_grad_enabled, is_inference,
    is_inference_mode_enabled, set_grad_enabled,
    # Random state.
    get_rng_state, initial_seed, seed, set_rng_state,
    # Introspection.
    can_cast, finfo, get_default_dtype, iinfo, is_distributed, is_floating_point,
    is_nonzero, is_same_size, is_signed, is_storage, is_tensor, promote_types,
    result_type,
    set_default_dtype, typename,
)
from ._ops import (                                      # noqa: E402,F401
    Generator, manual_seed, multinomial, randint, randperm,
)
from ._data import (                                     # noqa: E402,F401
    ConcatDataset, DataLoader, Dataset, ImageFiles, RandomSampler, SequentialSampler, Subset,
    TensorDataset, WeightedRandomSampler, backend, cache_get, cache_put, cuda,
    decode_cifar10, decode_images, fetch_cached, get_default_device, label_from_name,
    random_split, suspects,
    set_default_device, utils,
)
from ._serialize import load, save                       # noqa: E402,F401
from . import _onnx as onnx                              # noqa: E402,F401
from . import _hub as hub                                # noqa: E402,F401
from ._ops import __getattr__                            # noqa: E402,F401
from . import _nn as nn, _optim as optim                 # noqa: E402,F401
# **Named here or `borch_webgpu.autograd` is an `AttributeError`.** A submodule is
# not pulled in by importing the package, and `_ops.__getattr__` above answers
# unknown names — so the miss would come out as this library's *not in the browser
# subset* wording for a module that is right there on disk.
from . import autograd                                   # noqa: E402,F401

# In borch.ts a dtype is a label over float32 storage. It stays a label here
# too, but the name shown is torch's — the golden froze `torch.float32` as the
# answer.
from ._base import _DType                                # noqa: E402

# ── `out=` — the core's table, the core's rules ─────────────────────────────
#
# The table is not written twice. The core's is used as-is and only **the names
# we actually have** are wrapped here; a name that is missing has nothing to
# wrap. Let the tables diverge and three implementations give three answers,
# which is invisible until the golden catches it.
def _wrap_out_names():
    from borch import _TAKES_OUT, _TAKES_OUT_TUPLE
    from ._ops import _out

    def wrap(fn, name):
        def call(*args, **kwargs):
            out = kwargs.pop("out", None)
            return _out(fn(*args, **kwargs), out, name)
        call.__name__ = getattr(fn, "__name__", name)
        call.__doc__ = getattr(fn, "__doc__", None)
        return call

    for name in _TAKES_OUT | _TAKES_OUT_TUPLE:
        fn = globals().get(name)
        if fn is not None:
            globals()[name] = wrap(fn, name)


_wrap_out_names()


def install(name="borch_webgpu", modules=None):
    """Register submodule paths so `from <name>.nn import Linear` works.

    **Not registering is the default.** With `import borch_webgpu as torch`,
    `torch.nn.Linear` already resolves — it is attribute access, and an alias is
    enough. Most textbook code is shaped that way.

    What does not resolve is `from … import`. That consults the **path**
    registered in `sys.modules`, and an alias is one name inside one file, which
    does not reach that far. The boundary was measured and `tests/test_alias.py`
    holds it by value.

    **The name defaults to this module's own.** Registering as `torch` means
    another library's `import torch` receives the subset too, and in a mixed
    environment that becomes an error nobody can trace. Registering under its
    own name opens the submodule paths without touching anyone else's code.

    The same shape as the core's `install()`, for the same reason: submodule
    paths written by hand drift apart. The core's did drift, and
    `from torch.optim.lr_scheduler import StepLR` stopped inside a textbook.
    """
    import sys

    modules = sys.modules if modules is None else modules
    modules[name] = sys.modules[__name__]
    registered = [name]
    for path, mod in (("nn", nn), ("optim", optim), ("linalg", linalg),
                      ("nn.functional", nn.functional),
                      ("nn.utils", nn.utils), ("nn.utils.rnn", nn.utils.rnn),
                      ("nn.utils.fusion", nn.utils.fusion),
                      ("onnx", onnx),
                      ("optim.lr_scheduler", optim.lr_scheduler)):
        full = f"{name}.{path}"
        modules[full] = mod
        registered.append(full)
    return registered


float32 = _DType("float32")
int64 = _DType("int64")
# **`bool` does not go in the module globals.** It shadows the Python builtin,
# `isinstance(x, bool)` breaks, and that leaked out through `_DType.__repr__`.
# The golden calls it as `L.bool`, so the module's `__getattr__` picks that one
# name out — `_ops.__getattr__` does the work.
bool_ = _DType("bool")
complex64 = _DType("complex64")
cfloat = complex64
# **`complex128` does not even keep a name.** The core keeps one and stops
# there, because numpy can **actually produce** it by promotion and a door was
# needed. Here borch.ts does not carry `float64` at all, so that promotion has
# no path — and a door standing in front of nothing reads, to the next person,
# as a door holding something back.

# ── torch's five top-level numeric constants ───────────────────────────────
#
# **The same values** as the core's (`borch._base`). This side does not go
# through borch.ts: they are Python values, so there is nothing to ask the GPU,
# and asking raises `AttributeError` because the name is not over there — which
# is what happened.
#
# The coverage table could not see these five. `tests/torch_gap.py` counts names
# that are `callable`, and values landed in neither the numerator nor the
# denominator.
from math import e, inf, nan, pi                        # noqa: E402,F401

# torch has it as plain `None` too — a marker that it means the same as
# `x[:, None]`.
newaxis = None

# ── the words layouts and memory formats are named with ─────────────────────
#
# **The core's objects, not copies.** `x.layout` on this side hands back the
# core's `strided`, so `x.layout is borch_webgpu.strided` is true the way torch's
# is; a second set of instances here would read identically and compare false
# under `is`. They are plain Python values with nothing to ask the GPU, which is
# why they come from the core rather than through borch.ts — asking borch.ts
# produced *borch.ts does not have `contiguousFormat`*, a sentence about the far
# side for something neither side needed to hold.
from borch._tensor import (                             # noqa: E402,F401
    channels_last, channels_last_3d, contiguous_format, preserve_format,
    sparse_bsc, sparse_bsr, sparse_coo, sparse_csc, sparse_csr, strided,
)
# The narrow dtypes that have no storage here. The core keeps the **names** so
# `dtype=torch.uint8` says what is missing rather than reading as a typo, and
# this side keeps them for the same reason and in the same wording.
from borch._base import (                               # noqa: E402,F401
    bfloat16, chalf, complex32, float16, half, int8, int16, short, uint8,
)


# ── the seven losses torch keeps at top level as well as under `F` ──────────
#
# **`torch.kl_div` and `F.kl_div` are two different functions.** The reduction is an
# integer here — `0` none, `1` mean, `2` sum — where `F` takes the word, and it defaults
# to none where every `F` loss defaults to mean.
#
# They are written out rather than left to the module's `__getattr__`, which forwards an
# unknown name to **the first argument's method** — `kl_div(a, b)` would become
# `a.kl_div(b)`, and a tensor has no such method. The core has the same seven for the
# same reason; this side had none of them, and eighteen golden cases said so.
def _aten_reduction(value):
    """The integer the ATen ops take, as the word `F` takes.

    **`int` is a dtype in this module's namespace**, so the builtin is reached through
    `builtins` — asking `isinstance(value, int)` here asks whether a number is a dtype
    and raises. Measured on the core first, and it is the same shape of trap.
    """
    import builtins as _builtins
    if (not isinstance(value, _builtins.int) or isinstance(value, _builtins.bool)
            or not 0 <= value <= 2):
        raise ValueError(f"reduction has to be 0, 1 or 2, but got {value!r}.")
    return ("none", "mean", "sum")[value]


# ── the eight `sym_*` helpers, taken from the core ────────────────────────────
#
# **Arithmetic on ordinary numbers, so there is nothing here to do differently.** They
# make no tensor and run no kernel; a second copy on this side would be a second place
# for `sym_max`'s float promotion to drift, and that promotion is the only thing
# separating these from the builtins they look like.
#
# The general forwarding rule sends an unknown name to borch.ts, which has none of these
# — measured, fifteen golden cases said `borch.ts does not have symMax` after they went
# into the core. A name that is Python arithmetic has no business crossing to the GPU
# side to be refused there.
from borch import (                                          # noqa: E402
    sym_float, sym_int, sym_ite, sym_max, sym_min, sym_not, sym_sqrt, sym_sum,
)


def binary_cross_entropy_with_logits(self, target, weight=None, pos_weight=None,
                                     reduction=0):
    return nn.functional.binary_cross_entropy_with_logits(
        self, target, weight=weight, pos_weight=pos_weight,
        reduction=_aten_reduction(reduction))


def cosine_embedding_loss(input1, input2, target, margin=0.0, reduction=0):
    return nn.functional.cosine_embedding_loss(
        input1, input2, target, margin=margin, reduction=_aten_reduction(reduction))


def hinge_embedding_loss(self, target, margin=1.0, reduction=0):
    return nn.functional.hinge_embedding_loss(
        self, target, margin=margin, reduction=_aten_reduction(reduction))


def kl_div(self, target, reduction=0, *, log_target=False):
    return nn.functional.kl_div(self, target, reduction=_aten_reduction(reduction),
                                log_target=log_target)


def margin_ranking_loss(input1, input2, target, margin=0.0, reduction=0):
    return nn.functional.margin_ranking_loss(
        input1, input2, target, margin=margin, reduction=_aten_reduction(reduction))


def poisson_nll_loss(input, target, log_input, full, eps, reduction):
    """**Six required arguments and no defaults**, which is the schema."""
    return nn.functional.poisson_nll_loss(
        input, target, log_input=log_input, full=full, eps=eps,
        reduction=_aten_reduction(reduction))


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6,
                        swap=False, reduction=0):
    return nn.functional.triplet_margin_loss(
        anchor, positive, negative, margin=margin, p=p, eps=eps, swap=swap,
        reduction=_aten_reduction(reduction))

