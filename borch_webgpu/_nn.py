"""Where `torch.nn` sits. **Mostly transcribing names here too.**

Two branches.

`nn.functional.relu(x)` is `x.relu()` in borch.ts — torch itself treats the two
as the same thing, so taking the first argument and forwarding to its method is
the whole job. One `__getattr__` does all of it.

`nn.Linear(6, 8)` is a class in borch.ts. Some names diverge there
(`BatchNorm2d` is `BatchNormND` over the wall), so this side keeps a table. And
the ones torch has that borch.ts does not have as layers — `Softmax`, `ELU`,
`L1Loss` and the rest — are built as **a layer wrapping one tensor method**:
not approximating what is missing, but naming what is there.
"""

import collections as _collections
import inspect as _inspect
import warnings as _warnings

import numpy as _np

import js as _js
from pyodide.ffi import to_js as _to_js

from ._base import (
    Tensor, _js_floats, _js_list, guarded, handle, settle, tensor, wrap,
)
from ._ops import _arg, camel, positional
# **The module is held, not the name.** `manual_seed` **swaps `_ops._rng` for a
# new generator**, so importing it by name means that swap never reaches here —
# seeding changes everything except this layer. This place was caught by that
# once already.
from . import _ops as _ops_mod

_ts = _js.borch


class _Functional:
    """`nn.functional`. Forwarded to the first argument's method — torch's own
    rule."""

    def __getattr__(self, name):
        # What was hand-written on the module is used here as well — `F.pad` is
        # one.
        from . import _ops
        if name == "embedding":
            return embedding
        # **Ones whose name or arguments differ from borch.ts's.** The rule
        # cannot forward them: `rms_norm` takes a shape in torch and a count of
        # dimensions over there, and `conv_transpose2d` has no per-dimension
        # name over there — there is only `convTransposeND`.
        if name in _HAND_WRITTEN:
            return _HAND_WRITTEN[name]
        if name in ("pad", "clamp", "flip", "pow", "split", "chunk",
                    "layer_norm", "where", "squeeze", "repeat_interleave"):
            fn = getattr(_ops, name)
            return lambda *a, **k: fn(*a, **k)

        js_name = camel(name)

        def call(x, *args, **kw):
            h = handle(x)
            fn = getattr(h, js_name, None)
            if fn is None:
                raise AttributeError(
                    f"borch.ts tensors do not have `{js_name}` (F.{name})")
            return guarded(fn, *positional(name, args, kw))

        call.__name__ = name
        return call


def _rms_norm(x, normalized_shape, weight=None, eps=None):
    """torch takes **a shape** and borch.ts takes **a count of dimensions**."""
    dims = len(normalized_shape) if isinstance(normalized_shape, (list, tuple)) else 1
    out = wrap(handle(x).rmsNorm(dims) if eps is None
               else handle(x).rmsNorm(dims, eps))
    return out if weight is None else out * weight


def _conv_transpose(x, weight, bias=None, stride=1, padding=0):
    """There are no per-dimension names over there — `convTransposeND` does all
    of them."""
    return wrap(handle(x).convTransposeND(
        handle(weight), handle(bias) if bias is not None else None, stride, padding))


def _pool_fn(kind, adaptive):
    """One pooling. **No per-dimension names over there** — `poolND` does all of
    them.

    `padding` and `ceil_mode` reach `poolND`, which has taken them for the average
    since it was written and takes them for the maximum now. Absent here, they were
    not refused but **received and dropped**: `F.max_pool2d(x, 3, padding=1)` answered
    with the unpadded pooling, a smaller table of plausible numbers.
    """
    def call(x, size, stride=None, padding=0, dilation=1, return_indices=False,
             ceil_mode=False, count_include_pad=True, divisor_override=None):
        h = handle(x)
        if return_indices:
            # **`adaptive` by keyword.** Written as the fourth positional — which is
            # where it used to be — it lands in `padding` the moment that seat exists,
            # and a truthy padding switched the whole call to adaptive pooling: a
            # different algorithm, a plausible answer, no exception. Eight golden rows
            # caught it, which is the only reason this line reads as it does.
            return _pool_with_indices(x, size, stride, padding, dilation,
                                      ceil_mode, adaptive=adaptive)
        if adaptive:
            return wrap(h.adaptivePool(kind, size))
        # **`dilation` was refused here and the kernel takes it now.** The refusal's
        # reason was that borch.ts's pooling shader had no step between the cells one
        # window reads; it has one, and at 1 the generated source is the line it always
        # was. **The average has no such argument in torch at all**, so passing one is
        # a `TypeError` over there and borch.ts refuses it by name here.
        return wrap(h.poolND(kind, size, stride if stride is not None else size,
                             padding, ceil_mode, count_include_pad,
                             divisor_override, dilation))
    return call


def _pool_with_indices(x, size, stride=None, padding=0, dilation=1,
                       ceil_mode=False, return_indices=True, *, adaptive=False):
    """Hands back the values together with **where each one won.** The index is
    a flat position inside the plane.

    **torch's argument order, and `adaptive` is keyword-only behind it.** Written
    with `adaptive` fourth — which is where it was — `F.max_pool2d_with_indices(x, 2,
    None, 1)` handed the padding to it, and a truthy padding switched the whole call
    to adaptive pooling: a different algorithm, a plausible answer, no exception.

    Over there it comes as `{values, indices}`, so it is read out by name and
    turned into a tuple — torch gives a tuple and textbook code unpacks it as
    `out, idx = ...`.

    **The three window arguments were refused here and not on the plain path**, and
    the difference was the shape of the thing that carries the positions: that one
    goes through the pooling shader, which takes a padding, a dilation and a ceiling,
    while this one went through a window list whose entry was `[start, end)` — an
    interval, with no step between the cells it reads and no room to hang off an edge.
    A list of positions is both, so the refusal is gone with the interval.
    """
    h = handle(x)
    got = (h.adaptiveMaxPoolWithIndices(size) if adaptive
           else h.maxPoolWithIndices(size, stride if stride is not None else size,
                                     padding, dilation, bool(ceil_mode)))
    return wrap(got.values), wrap(got.indices)


def _unpool(x, indices, kernel_size, stride=None, padding=0, output_size=None):
    """Put the values back into the slots the index map points at. The rest are
    zero."""
    return wrap(handle(x).maxUnpool(
        handle(indices), kernel_size, stride, padding,
        _js_list(list(output_size)) if output_size is not None else None))


def _fractional(spatial):
    """Max pooling whose window positions are jittered by samples.

    **The samples are drawn in Python.** Window positions have to be settled on
    the CPU to be baked into a shader, and reading GPU random numbers back from
    borch.ts would mean waiting. numpy is already here, so they are drawn on
    this side and passed across — and `manual_seed` reaches that generator.
    """
    def call(x, kernel_size, output_size=None, output_ratio=None,
             return_indices=False, _random_samples=None):
        h = handle(x)
        shape = [int(n) for n in h.shape]
        if (output_size is None) == (output_ratio is None):
            raise ValueError(
                "fractional_max_pool takes either output_size or output_ratio, not both.")
        if output_size is not None:
            sizes = ([output_size] * spatial if isinstance(output_size, int)
                     else list(output_size))
        else:
            ratios = ([output_ratio] * spatial if isinstance(output_ratio, float)
                      else list(output_ratio))
            sizes = [int(shape[2 + k] * ratios[k]) for k in range(spatial)]

        n, c = shape[0], shape[1]
        if _random_samples is None:
            samples = _ops_mod._rng.random((n, c, spatial))
        else:
            samples = _np.asarray(_random_samples.numpy()).reshape(n, c, spatial)
        flat = [[float(v) for v in samples[i, j]] for i in range(n) for j in range(c)]
        got = h.fractionalMaxPool(
            kernel_size, _js_list(sizes),
            _to_js([_js_floats(row) for row in flat]))
        out, idx = wrap(got.values), wrap(got.indices)
        return (out, idx) if return_indices else out
    return call


def _fractional_indices(spatial):
    fn = _fractional(spatial)

    def call(x, kernel_size, output_size=None, output_ratio=None,
             _random_samples=None):
        return fn(x, kernel_size, output_size, output_ratio, True, _random_samples)
    return call


def _ints(v):
    """It may arrive as a tensor or as a list. Unrolled into a list of Python
    integers."""
    if isinstance(v, Tensor):
        return [int(round(x)) for x in _np.asarray(v.numpy()).reshape(-1)]
    return [int(x) for x in _np.asarray(v).reshape(-1)]


def _ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
              reduction="mean", zero_infinity=False):
    """The target arrives as `(N, S)` or as a concatenated 1-D tensor — torch
    takes both."""
    lens = _ints(target_lengths)
    flat = _np.asarray(targets.numpy() if isinstance(targets, Tensor) else targets)
    if flat.ndim == 1:
        rows, at = [], 0
        for n in lens:
            rows.append([int(round(v)) for v in flat[at:at + n]])
            at += n
    else:
        rows = [[int(round(v)) for v in flat[i]] for i in range(len(lens))]
    return wrap(_ts.nn.ctcLoss(
        handle(log_probs), _to_js([_js_list(r) for r in rows]),
        _js_list(_ints(input_lengths)), _js_list(lens),
        int(blank), reduction, bool(zero_infinity)))


def _lp_pool(x, norm_type, kernel_size, stride=None, ceil_mode=False):
    """**`ceil_mode` was refused here and the seat had been there for a while.**

    It began inside a `**kw` and went nowhere; the refusal that replaced it said
    borch.ts's `lpPool` had no seat for it, which was true when written. That function
    grew a `ceilMode` the day the pooling shader did — the golden's two `LPPool`
    ceiling cases are what drove it — and this door was not opened with it. A refusal
    on a call path nothing exercised is exactly the kind that outlives its gap.
    """
    return wrap(handle(x).lpPool(norm_type, kernel_size, stride, bool(ceil_mode)))


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
          scale=None):
    """A free function over there, so it does not go through a tensor method.

    **`dropout_p` and `scale` used to be taken and dropped.** borch.ts had neither,
    so the call passed `is_causal` into their place and the two arguments never
    crossed — a `scale` silently replaced by `1/√dim` is a model whose attention is
    weighted wrong and still trains."""
    return wrap(_js.borch.nn.scaledDotProductAttention(
        handle(query), handle(key), handle(value),
        handle(attn_mask) if attn_mask is not None else None,
        float(dropout_p), bool(is_causal),
        None if scale is None else float(scale)))


# ── losses and distances ────────────────────────────────────────────────
#
# **The positional order is written out by hand.** The general path
# (`positional`) fills empty slots with `None`, and torch's losses put
# `reduction` in the middle with `delta` and `margin` behind it — so naming just
# one of them opens a hole in front of it. That hole erases the other side's
# default.

def _lay_out(order, args, kw):
    """Unroll what arrived by name **into positions.** Empty slots cross as
    empty.

    **A hole in the middle must not stop it.** Written that way at first,
    `huber_loss(x, y, reduction="none")` cut off at the `delta` slot and lost
    `reduction` entirely — not an exception but **a quietly different answer
    from the default**, and thirteen cases diverged that way at once.

    Empty slots stay `None`. Pyodide passes that as `undefined`, so the other
    side's default survives. Anything trailing is trimmed.
    """
    got = dict(zip(order, args))
    got.update(kw)
    laid = [_arg(got[key]) if key in got else None for key in order]
    while laid and laid[-1] is None:
        laid.pop()
    return laid


def _loss(js_name, order, torch_order=None):
    """Read torch's call, then write **borch.ts's positional order.**

    **One table was doing both jobs and could not.** `order` named the incoming
    arguments *and* fixed the outgoing positions, so anything torch has and
    borch.ts does not — the deprecated `size_average`/`reduce` pair, a `weight`
    the subset refuses — had nowhere to be named. Left out of the table, those
    seats close up and **every later argument moves forward**:

        F.huber_loss(x, t, "sum")   torch: reduction="sum"
                                    here:  delta="sum"      — before this

    which is not an error anywhere. `delta` is only ever compared against an
    absolute difference, and a string compares false, so the answer came back as
    the mean of the L2 branch with no warning. Twelve of the fourteen entries
    below had drifted this way; `gaussian_nll_loss`, the one with no deprecated
    pair, is the control that agreed all along — which is the same control the
    core names in its own note, written before this file was measured.

    So `torch_order` names what arrives and `order` names where it goes. When
    they are the same, one is enough.

    **A third parameter, `refuse`, stood here and is gone.** It named the layer whose
    class-weight refusal to raise, for the two entries that had one; both answer now,
    and `weight` moved from *the seat that raises* to a seat in `order`.
    """
    names = order if torch_order is None else torch_order

    def call(x, *args, **kw):
        got = dict(zip(names, args))
        got.update(kw)
        if "reduction" in names:
            folded = _legacy_reduction(got.pop("size_average", None),
                                       got.pop("reduce", None),
                                       got.get("reduction"))
            if folded is not None:
                got["reduction"] = folded
        return guarded(getattr(handle(x), js_name), *_lay_out(order, (), got))

    call.__name__ = js_name
    return call


# The deprecated pair, at the seat torch keeps it in. Named once; the entries
# below splice it where it belongs.
_DEPRECATED = ("size_average", "reduce")


_LOSSES = {
    # **`binary_cross_entropy` was the third implementation left behind.** The core
    # and borch.ts both grew it in the same hour; this file did not, and only
    # `tests/browser/run.py --lib borch_webgpu` says so — the core's suite and the
    # borch.ts golden were both green while the binding could not run the case.
    #
    # Its logits form was then left behind **from that same note**: the sentence
    # above was written, the one name it named was added, and the pair beside it
    # was not looked at. Four golden cases said so for as long as they existed.
    # **`weight` and `pos_weight` were the `refuse=` seat and are outgoing seats now.**
    # borch.ts puts them after `reduction` and torch puts `weight` before the
    # deprecated pair, which is what the two tables are for.
    "binary_cross_entropy": _loss(
        "bce", ("target", "reduction", "weight"),
        ("target", "weight", *_DEPRECATED, "reduction")),
    "binary_cross_entropy_with_logits": _loss(
        "bceWithLogits", ("target", "reduction", "weight", "pos_weight"),
        # **`pos_weight` is last, after `reduction`** — the only loss torch puts
        # it there, and the core's layer carries the same note.
        ("target", "weight", *_DEPRECATED, "reduction", "pos_weight")),
    # **torch's third seat is `reduction`, not `delta`.** It is the newest of
    # these and never carried the deprecated pair, so nothing shifted and the
    # order is simply torch's own — which the table had backwards.
    # **`weight` stopped being a refusal and became a fourth outgoing seat.** borch.ts
    # puts it after `reduction`, torch after `delta`, and the two tables here are
    # exactly for that: `torch_order` says what arrives, `order` where it goes.
    "huber_loss": _loss(
        "huberLoss", ("target", "delta", "reduction", "weight"),
        ("target", "reduction", "delta", "weight")),
    "kl_div": _loss(
        "klDiv", ("target", "reduction", "log_target"),
        ("target", *_DEPRECATED, "reduction", "log_target")),
    # **The pair is split here** — `size_average` before `eps` and `reduce`
    # after it. Assuming the two are always adjacent is what a single table
    # cannot say.
    "poisson_nll_loss": _loss(
        "poissonNllLoss", ("target", "log_input", "full", "eps", "reduction"),
        ("target", "log_input", "full", "size_average", "eps", "reduce",
         "reduction")),
    # No deprecated pair, so what arrives is what goes out. **This is the
    # control**: it agreed before this change and after it.
    "gaussian_nll_loss": _loss(
        "gaussianNllLoss", ("target", "var", "full", "eps", "reduction")),
    "margin_ranking_loss": _loss(
        "marginRankingLoss", ("input2", "target", "margin", "reduction"),
        ("input2", "target", "margin", *_DEPRECATED, "reduction")),
    "cosine_embedding_loss": _loss(
        "cosineEmbeddingLoss", ("input2", "target", "margin", "reduction"),
        ("input2", "target", "margin", *_DEPRECATED, "reduction")),
    "hinge_embedding_loss": _loss(
        "hingeEmbeddingLoss", ("target", "margin", "reduction"),
        ("target", "margin", *_DEPRECATED, "reduction")),
    "soft_margin_loss": _loss(
        "softMarginLoss", ("target", "reduction"),
        ("target", *_DEPRECATED, "reduction")),
    "triplet_margin_loss": _loss(
        "tripletMarginLoss",
        ("positive", "negative", "margin", "p", "eps", "swap", "reduction"),
        ("positive", "negative", "margin", "p", "eps", "swap", *_DEPRECATED,
         "reduction")),
    # **borch.ts takes a `weight` here and the table had no seat for it**, so a
    # `reduction` given by position landed in it — a string where a tensor
    # belongs. The core answers with the weight applied (0.7314 against 0.3962
    # on the same input), so this was the one refusal-shaped hole that was not a
    # refusal: the subset has the feature and only this layer could not reach it.
    "multilabel_soft_margin_loss": _loss(
        "multilabelSoftMarginLoss", ("target", "weight", "reduction"),
        ("target", "weight", *_DEPRECATED, "reduction")),
    "multi_margin_loss": _loss(
        "multiMarginLoss", ("target", "p", "margin", "weight", "reduction"),
        ("target", "p", "margin", "weight", *_DEPRECATED, "reduction")),
    "multilabel_margin_loss": _loss(
        "multilabelMarginLoss", ("target", "reduction"),
        ("target", *_DEPRECATED, "reduction")),
    "pairwise_distance": _loss("pairwiseDistance", ("x2", "p", "eps", "keepdim")),
    "pdist": _loss("pdist", ("p",)),
    # Repositioning. Only the names differ.
    # Window unrolling and the rest. **`F.unfold` is im2col** — a different thing
    # from `Tensor.unfold`.
    "unfold": _loss("unfoldIm2col",
                    ("kernel_size", "dilation", "padding", "stride")),
    "fold": _loss("fold", ("output_size", "kernel_size", "dilation", "padding",
                           "stride")),
    "local_response_norm": _loss("localResponseNorm",
                                 ("size", "alpha", "beta", "k")),
    "rrelu": _loss("rrelu", ("lower", "upper", "training")),
    "pixel_shuffle": _loss("pixelShuffle", ("upscale_factor",)),
    "pixel_unshuffle": _loss("pixelUnshuffle", ("downscale_factor",)),
    "channel_shuffle": _loss("channelShuffle", ("groups",)),
    # **The ones that drop whole channels.** Over there it is one name,
    # `featureDropout`, and it does not check the rank — only `dropout1d` checks
    # it, so that check is attached separately below.
    "dropout2d": _loss("featureDropout", ("p", "training")),
    "dropout3d": _loss("featureDropout", ("p", "training")),
}


def _dropout1d(x, p=0.5, training=True):
    """**Refuses 4-D.** torch does, and the name carries the number of spatial
    dimensions."""
    rank = len(handle(x).shape)
    if rank not in (2, 3):
        raise RuntimeError(
            f"dropout1d: Expected 2D or 3D input, but received a {rank}D input. "
            "Note that dropout1d exists to provide channel-wise dropout on inputs "
            "with 1 spatial dimension, a channel dimension, and an optional batch "
            "dimension (i.e. 2D or 3D inputs).")
    return wrap(guarded(handle(x).featureDropout, float(p), bool(training)))


def _alpha_dropout(per_channel):
    def call(x, p=0.5, training=False):
        return wrap(guarded(handle(x).alphaDropout, float(p), bool(training),
                            per_channel))
    return call


def _triplet_with_distance(anchor, positive, negative, distance_function=None,
                           margin=1.0, swap=False, reduction="mean"):
    """The triplet that takes a distance function. What lives on the layer side
    is offered under a function name as well.

    **The callable needs two things done to it and neither is optional.**

    It is handed to JavaScript, so a bare Python function crosses as a *borrowed*
    proxy — alive for the argument's own lifetime and destroyed before the call
    that stores it uses it. The failure arrives later, as `borrowed proxy was
    automatically destroyed`, from a line that no longer names the callback.
    `create_proxy` is what keeps it, and `destroy()` is what stops it leaking.

    And what arrives at the callback is **borch.ts handles, not binding tensors**,
    so a Python body written in this library's own vocabulary — `(u - v).abs()` —
    would find none of it. The bridge wraps on the way in and unwraps on the way
    out, which is the same crossing `wrap`/`handle` make everywhere else here.

    Both were found by `tests/browser/run.py --lib borch_webgpu`, which is the only
    check that runs this file at all: the case existed and passed on the core and on
    borch.ts while this raised.
    """
    if distance_function is not None:
        from pyodide.ffi import create_proxy

        given = distance_function

        def bridge(u, v):
            return handle(given(wrap(u), wrap(v)))

        # **Rebound under its own name, not held in a `keep`.** The proxy *is* the
        # distance function, and `test_binding_arguments.py` reads this call site as
        # source: with the argument named anything else it reports position 0 as
        # carrying something that is not `distance_function`, which is exactly the
        # defect it exists to find. The name being accurate is what makes it quiet.
        distance_function = create_proxy(bridge)
    try:
        layer = _ts.nn.TripletMarginWithDistanceLoss.new(
            distance_function, float(margin), bool(swap), reduction)
        return wrap(guarded(layer.call, handle(anchor), handle(positive),
                            handle(negative)))
    finally:
        if hasattr(distance_function, "destroy"):
            distance_function.destroy()


def _interpolate(x, size=None, scale_factor=2, mode="nearest",
                 align_corners=None, recompute_scale_factor=None,
                 antialias=False):
    """**A place that accepted `mode` and never used it**, and then a third copy of
    the rules that did.

    The general path forwarded to `upsample`, which does nearest only, so asking
    for bilinear produced nearest — not an exception but a quietly different
    value. `F.pad` was caught in the same shape once, and like that time it only
    surfaced once the golden gained a case asking for that branch.

    The repair was to dispatch here, and that put a **third** implementation of the
    same arithmetic in the repository — with its own limits nobody else had, such as
    refusing a nearest `size=` that is not a whole multiple, which both the core and
    borch.ts answer. It forwards to borch.ts's own `interpolate` now: one dispatch,
    one set of rules, and the five modes arrive without being spelled out again.

    `antialias` was refused here while borch.ts had no such argument; it takes one
    now, and this hands the word over rather than stopping on it.

    **`align_corners` is the one rule that has to be checked here.** borch.ts takes
    it as a boolean, so the difference between *not given* and *given as false* is
    lost at the boundary — and torch refuses the flag outright on the modes that have
    no corners to align rather than picking one for you.
    """
    if align_corners is not None and mode not in ("linear", "bilinear", "bicubic",
                                                  "trilinear"):
        raise ValueError("align_corners option can only be set with the "
                         "interpolating modes: linear | bilinear | bicubic | "
                         "trilinear")
    return wrap(guarded(handle(x).interpolate, size, scale_factor, mode,
                        bool(align_corners), recompute_scale_factor,
                        bool(antialias)))


def _upsample(x, size=None, scale_factor=None, mode="nearest", align_corners=None,
              ):
    """`interpolate`'s old name. torch warns that it is deprecated and keeps
    taking it."""
    return _interpolate(x, size, scale_factor, mode, align_corners)


def _upsample_nearest(x, size=None, scale_factor=None):
    return _interpolate(x, size, scale_factor, "nearest")


def _upsample_bilinear(x, size=None, scale_factor=None):
    """**This is `align_corners=True`.** The default for
    `interpolate(mode='bilinear')` is false, so aliasing on the name alone
    misaligns the edges — the interior stays close enough that the eye does not
    separate them."""
    return _interpolate(x, size, scale_factor, "bilinear", True)


def _additive_mask(mask, like_shape=None):
    """Turn a boolean table into **a float that gets added.**

    torch takes two kinds — a boolean that masks the true positions, and a float
    added straight onto the scores. borch.ts takes only the additive kind, so it
    is converted here. **Not by multiplying with zero** — softmax has already
    normalised, and multiplying leaves the remaining positions not summing back
    to one.
    """
    if mask is None:
        return None
    raw = _np.asarray(mask.numpy() if isinstance(mask, Tensor) else mask)
    if raw.dtype == bool or raw.dtype.kind == "b":
        raw = _np.where(raw, -_np.inf, 0.0)
    got = raw.astype(_np.float32)
    if like_shape is not None:
        got = got.reshape(like_shape)
    return handle(tensor(got))


def _mha_forward(query, key, value, embed_dim_to_check, num_heads,
                 in_proj_weight, in_proj_bias, bias_k=None, bias_v=None,
                 add_zero_attn=False, dropout_p=0.0, out_proj_weight=None,
                 out_proj_bias=None, training=True, key_padding_mask=None,
                 need_weights=True, attn_mask=None, use_separate_proj_weight=False,
                 q_proj_weight=None, k_proj_weight=None, v_proj_weight=None,
                 static_k=None, static_v=None, average_attn_weights=True,
                 is_causal=False):
    """The computation `MultiheadAttention` performs inside. **Branches it does
    not do are refused loudly.**

    Three of these arguments used to be accepted and then dropped without a word.

    `embed_dim_to_check` is torch's own guard: the caller states the width it
    believes the projection has, and torch asserts it. Taking the number and not
    checking it turns a guard into decoration — the shape error still comes, but
    later and from somewhere else.

    **The seven refusals here are gone with borch.ts's.** They said *not here yet*
    for `bias_k`, `static_k`, the three separate projections and the two flags; each
    of those is a concatenation or a substitution over there now, and this file was
    only ever the door. The comment that stood here — *there is no arrangement of
    arguments in which this function can honour them* — was about the door and not
    about the room behind it.
    """
    width = int(handle(query).shape[-1])
    if embed_dim_to_check is not None and int(embed_dim_to_check) != width:
        raise RuntimeError(
            f"multi_head_attention_forward: query is {width} wide, "
            f"embed_dim_to_check={int(embed_dim_to_check)}.")
    n, s = int(handle(query).shape[1]), int(handle(key).shape[0])
    length = int(handle(query).shape[0])
    if is_causal and attn_mask is None:
        attn_mask = _np.triu(_np.ones((length, s), dtype=bool), k=1)
    def _h(t):
        """**Forwarded rather than checked here.** Each of these is refused by name on
        the borch.ts side now, and passing a hard `None` in its place is the same as
        dropping it — the shape this file's own check is for."""
        return None if t is None else handle(t)

    # **torch's twenty-five, in torch's order**, now that borch.ts takes them. This
    # laid thirteen out in an order of its own, and the two lists happened to agree
    # only because both were wrong in the same way — the moment borch.ts followed
    # torch, `int(num_heads)` was landing on `embed_dim_to_check` and the masks on
    # `bias_k`/`bias_v`. Nothing is re-checked here; the arguments are passed as they
    # arrived.
    #
    # **`in_proj_weight` goes through `_h` and not `handle`**, because it is `None`
    # under separate projections — which is what torch's own layer holds there. A hard
    # `handle(None)` would stop at the boundary rather than reach the branch that
    # expects it.
    #
    # (That note is here rather than beside the argument: `test_binding_arguments.py`
    # reads this call by splitting on commas, so a comment inside the parentheses
    # becomes two arguments and shifts every position after it.)
    got = _ts.nn.multiHeadAttentionForward(
        handle(query), handle(key), handle(value),
        None if embed_dim_to_check is None else int(embed_dim_to_check),
        int(num_heads),
        _h(in_proj_weight),
        handle(in_proj_bias) if in_proj_bias is not None else None,
        _h(bias_k), _h(bias_v), bool(add_zero_attn), float(dropout_p),
        handle(out_proj_weight),
        handle(out_proj_bias) if out_proj_bias is not None else None,
        bool(training),
        _additive_mask(key_padding_mask, (n, s)),
        bool(need_weights),
        _additive_mask(attn_mask),
        bool(use_separate_proj_weight),
        _h(q_proj_weight), _h(k_proj_weight), _h(v_proj_weight),
        _h(static_k), _h(static_v),
        bool(average_attn_weights), bool(is_causal))
    out = wrap(got.output)
    return (out, None) if got.weights is None else (out, wrap(got.weights))


def _affine_grid(theta, size, align_corners=False):
    return wrap(_ts.nn.affineGrid(handle(theta), _js_list(list(size)),
                                  bool(align_corners)))


def _grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                 align_corners=False):
    return wrap(_ts.nn.gridSample(handle(x), handle(grid), mode, padding_mode,
                                  bool(align_corners)))


def _batch_norm(x, running_mean=None, running_var=None, weight=None, bias=None,
                training=False, momentum=0.1, eps=1e-5):
    """The function form of the layer. **In training it edits the running
    statistics in place** — torch does that."""
    return wrap(_ts.nn.batchNorm(
        handle(x),
        handle(running_mean) if running_mean is not None else None,
        handle(running_var) if running_var is not None else None,
        handle(weight) if weight is not None else None,
        handle(bias) if bias is not None else None,
        bool(training), float(momentum), float(eps)))


def _embedding_bag(idx, weight, offsets=None, max_norm=None, norm_type=2.0,
                   scale_grad_by_freq=False, mode="mean", sparse=False,
                   per_sample_weights=None, include_last_offset=False,
                   padding_idx=None):
    """**torch's order, and the call across the boundary is positional.**

    `mode` used to be fourth here and fourth over there, and both moved to sixth —
    but they moved in two different commits, and in between this function was
    handing a mode string to `maxNorm`, which rewrites the embedding table. Neither
    compiler can see across this call: `tsc` reads the TypeScript side and stops at
    the bridge, and Python has nothing to read at all.

    That is the fourth kind of positional failure, and the only one no type system
    reaches. `test_binding_arguments.py` is what covers it.
    """
    return wrap(_ts.nn.embeddingBag(
        handle(idx), handle(weight),
        _js_list([int(v) for v in _np.asarray(
            offsets.numpy() if isinstance(offsets, Tensor) else offsets
        ).reshape(-1)]) if offsets is not None else None,
        None if max_norm is None else float(max_norm),
        float(norm_type), bool(scale_grad_by_freq),
        mode, bool(sparse),
        handle(per_sample_weights) if per_sample_weights is not None else None,
        bool(include_last_offset),
        None if padding_idx is None else int(padding_idx)))


def _gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    """**Random, and gradients still flow.** Even with `hard`, the gradient
    taken is the soft one's.

    `eps` is passed and then not used, which is what torch does with it — it has
    been deprecated there since the noise moved to an exponential draw that needs
    no floor. What torch also does, and this did not, is **say so**. A caller
    writing `eps=1e-5` was getting silence from a library that had quietly decided
    to ignore them; now they get the same warning torch gives.

    **`eps` now travels to borch.ts rather than being dropped here.** It had been
    left out of that signature entirely, which put `dim` in its seat: this call read
    `gumbelSoftmax(logits, tau, hard, int(dim), None)`, and the fourth argument
    landed on a parameter named `dim` by luck rather than by agreement. Once borch.ts
    took `eps` back, that luck ran out and `test_binding_arguments.py` said so in the
    same run — `dim` belongs at position 4 and the call has None.
    """
    if float(eps) != 1e-10:
        _warnings.warn("`eps` parameter is deprecated and has no effect.",
                       stacklevel=2)
    # **The trailing `None` was borch.ts's own sixth seat and it has gone.** That seat
    # took the Gumbel draw so a caller could reproduce the answer; torch has no such
    # argument, and `manual_seed` gives reproducibility on all three sides. Five
    # arguments now, which is torch's list.
    return wrap(_ts.nn.gumbelSoftmax(handle(logits), float(tau), bool(hard),
                                     float(eps), int(dim)))


def _functional_inplace(name):
    """`F.relu_(x)` — **the version without the underscore does the computation**
    and the result is copied into this buffer.

    The same place and the same reason as the core. Two copies of the expression
    diverge eventually and the values stay plausible enough that nobody sees it.
    A leaf with gradients on is refused, as torch does.

    **And the argument list is the base's minus `inplace`**, taken from the function
    rather than retyped — the same derivation the core makes. Written `(x, *args,
    **kw)` on both sides, `F.relu_(x, 0.5)` ran with the 0.5 landing in `inplace`
    while torch answered `TypeError`, and `F.elu_(x, inplace=True)` ran where torch
    refuses the word. `bind` is what refuses; a signature alone documents a door that
    still lets everything through.

    Six of the eight already stopped here for reasons of their own, and the two that
    did not are why this is derived rather than trusted.
    """
    # **Read off the core's own underscore form**, which derives it from its base.
    # Two things follow: there is one table rather than two, and this side cannot
    # drift from the list the argument axis measures.
    #
    # Late, because `functional` below does not exist while this dictionary is being
    # built — and from `borch` rather than from here, because `getattr(functional,
    # name)` lands on a forwarder whose own signature is the bag this is replacing.
    cache = {}

    def prepared():
        if not cache:
            from borch import nn as _core_nn
            shown = _inspect.signature(getattr(_core_nn.functional, name + "_"))
            cache["sig"] = shown
            cache["first"] = next(iter(shown.parameters))
        return cache

    def call(*args, **kw):
        got = prepared()
        bound = got["sig"].bind(*args, **kw)
        bound.apply_defaults()
        # **Handed on by position.** `bind` keeps the declared order, and several of
        # these forwarders reach borch.ts by a table that takes no keywords at all —
        # `threshold` answers *does not take keyword arguments* to its own parameter
        # names. Position is what both sides agree on.
        rest = list(bound.arguments.values())
        x = rest.pop(0)
        x._refuse_inplace_on_leaf(name + "_")
        return x._write_back(getattr(functional, name)(x, *rest))

    call.__name__ = name + "_"
    return call


def _gelu(x, approximate="none"):
    """**Two expressions.** The table's `gelu` is the exact form (erf) and
    `"tanh"` is a different kernel."""
    h = handle(x)
    if approximate == "tanh":
        return wrap(guarded(h.geluTanh))
    if approximate != "none":
        raise ValueError(
            f"gelu(): approximate is 'none' or 'tanh' (got {approximate!r})")
    return wrap(guarded(h.unary, "gelu"))


def _elu(x, alpha=1.0, inplace=False):
    """**Takes an alpha.** On the general path it reaches the table's
    no-argument version and the alpha disappears."""
    return wrap(guarded(handle(x).elu, float(alpha)))


def _bilinear(x1, x2, weight, bias=None):
    """The form that takes its weights from outside. It goes to the same tensor
    method the layer does."""
    return wrap(guarded(handle(x1).bilinear, handle(x2), handle(weight),
                        handle(bias) if bias is not None else None))


def _linear_cross_entropy(input, linear_weight, target, *,   # noqa: A002
                          linear_bias=None, weight=None, reduction="mean",
                          ignore_index=None, label_smoothing=0.0, options=None):
    """**The generic door cannot carry this one**, and the reason is the same shape as
    `special.polygamma`'s: it forwards to the first argument's method, and borch.ts has
    this as a module function taking three tensors rather than a method on one.

    `ignore_index` defaults to `None` here and to `-100` on `crossEntropy`, and `None`
    is measured to *mean* -100 — so it is mapped rather than passed through. Handed
    straight over it would mean *ignore nothing*, which agrees on every target that
    holds no -100 and differs on the ones this argument exists for.
    """
    _ = options                     # no field of it moves a value (measured)
    return wrap(guarded(
        _ts.nn.functional.linearCrossEntropy,
        handle(input), handle(linear_weight), handle(target),
        handle(linear_bias) if linear_bias is not None else None,
        handle(weight) if weight is not None else None,
        reduction, -100 if ignore_index is None else int(ignore_index),
        float(label_smoothing), None))


_HAND_WRITTEN = {
    "linear_cross_entropy": _linear_cross_entropy,
    "interpolate": _interpolate,
    "multi_head_attention_forward": _mha_forward,
    "affine_grid": _affine_grid,
    "grid_sample": _grid_sample,
    "batch_norm": _batch_norm,
    "embedding_bag": _embedding_bag,
    "gumbel_softmax": _gumbel_softmax,
    "upsample": _upsample,
    "upsample_nearest": _upsample_nearest,
    "upsample_bilinear": _upsample_bilinear,
    **{n + "_": _functional_inplace(n) for n in
       ("relu", "celu", "elu", "selu", "hardtanh", "leaky_relu", "threshold",
        "rrelu")},
    "bilinear": _bilinear,
    "gelu": _gelu,
    "elu": _elu,
    "dropout1d": _dropout1d,
    "alpha_dropout": _alpha_dropout(False),
    "feature_alpha_dropout": _alpha_dropout(True),
    "triplet_margin_with_distance_loss": _triplet_with_distance,
    "scaled_dot_product_attention": _sdpa,
    "avg_pool1d": _pool_fn("avg", False),
    # **`avg_pool2d` used to take the generic route** while its 1-D and 3-D siblings
    # took this one — and that route's row is `("kernel_size", "stride")`, so torch's
    # `padding`, `ceil_mode`, `count_include_pad` and `divisor_override` had no seat.
    # Three names, two routes, and the odd one out was the one everybody calls.
    "avg_pool2d": _pool_fn("avg", False),
    "avg_pool3d": _pool_fn("avg", False),
    "adaptive_avg_pool1d": _pool_fn("avg", True),
    # **The two-dimensional one was the gap in the middle.** 1-D and 3-D
    # were both written down and 2-D — the one the tutorials use — reached
    # the general rule instead, which asks the tensor for
    # `adaptiveAvgPool2d`. borch.ts has `adaptiveAvgPool` (1-D) and
    # `adaptivePool(kind, size)`, so the call raised `AttributeError` and no
    # golden case asked it.
    "adaptive_avg_pool2d": _pool_fn("avg", True),
    "adaptive_avg_pool3d": _pool_fn("avg", True),
    "adaptive_max_pool1d": _pool_fn("max", True),
    "adaptive_max_pool2d": _pool_fn("max", True),
    "adaptive_max_pool3d": _pool_fn("max", True),
    "lp_pool1d": _lp_pool,
    "lp_pool2d": _lp_pool,
    "lp_pool3d": _lp_pool,
    # The versions that also hand back where each one won. torch gives the same
    # computation two names — `return_indices=True` and `*_with_indices`.
    "max_pool1d_with_indices": _pool_with_indices,
    "max_pool2d_with_indices": _pool_with_indices,
    "max_pool3d_with_indices": _pool_with_indices,
    "adaptive_max_pool1d_with_indices": lambda x, s, **k: _pool_with_indices(
        x, s, adaptive=True),
    "adaptive_max_pool2d_with_indices": lambda x, s, **k: _pool_with_indices(
        x, s, adaptive=True),
    "adaptive_max_pool3d_with_indices": lambda x, s, **k: _pool_with_indices(
        x, s, adaptive=True),
    "max_unpool1d": _unpool,
    "max_unpool2d": _unpool,
    "max_unpool3d": _unpool,
    # `max_pool*d` has to accept `return_indices`, so it cannot take the general
    # path — that path forwards arguments to the method over there verbatim, and
    # `maxPool2d` does not know that name.
    "max_pool1d": _pool_fn("max", False),
    "max_pool2d": _pool_fn("max", False),
    "max_pool3d": _pool_fn("max", False),
    "ctc_loss": _ctc_loss,
    "fractional_max_pool2d": _fractional(2),
    "fractional_max_pool3d": _fractional(3),
    "fractional_max_pool2d_with_indices": _fractional_indices(2),
    "fractional_max_pool3d_with_indices": _fractional_indices(3),
    "rms_norm": _rms_norm,
    "conv_transpose1d": _conv_transpose,
    "conv_transpose2d": _conv_transpose,
    "conv_transpose3d": _conv_transpose,
    **_LOSSES,
}


functional = _Functional()


def embedding(idx, table, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    """`F.embedding(indices, table)` — pick rows out of the table by index.

    **The definition itself.** The same work `index_select` does, and the
    gradient is already known over there: an index appearing several times
    accumulates into that row several times. Naming what exists rather than
    imitating what does not, so there is nowhere for the values to diverge.

    **The naming moved over there.** While these three lines lived here, borch.ts
    had no such name, and the golden goes through this function, so the table was
    green.

    **torch's five arrived with borch.ts's.** They were absent from both, so
    `F.embedding(i, w, 0)` handed a padding index to nothing — JavaScript discards
    a surplus argument — and the un-padded answer came back under the name of a
    padded one. Two of the five are refused on the far side; that refusal is
    reachable only because the seats exist here to carry the word to it.
    """
    return wrap(_ts.nn.functional.embedding(
        handle(idx), handle(table), padding_idx, max_norm, norm_type,
        scale_grad_by_freq, sparse))


class Transformer:
    """torch's `nn.Transformer` is not here. **Only the mask-making place is.**

    `generate_square_subsequent_mask` is a function with a settled definition
    rather than a layer — the upper triangle `-inf` and the rest zero. The point
    is that the values are floats; rounding them into booleans diverges inside
    the attention, which is what the golden case names record.

    The rest — encoder, decoder — is absent. Nothing missing is imitated.
    """

    @staticmethod
    def generate_square_subsequent_mask(n):
        import numpy as _np
        from ._base import tensor as _t

        m = _np.zeros((n, n), dtype=_np.float32)
        m[_np.triu_indices(n, 1)] = -_np.inf
        return _t(m)


class _Rnn:
    """`nn.utils.rnn`. What is here right now is `pad_sequence`, and nothing
    else."""

    @staticmethod
    def pad_sequence(parts, batch_first=False, padding_value=0.0):
        return wrap(_ts.Tensor.padSequence(
            _js.Array.new(*[handle(p) for p in parts]), batch_first, padding_value))


class _Fusion:
    """`nn.utils.fusion`. The folding is done over the wall — `fuseConvBnWeights` and
    `fuseConvBnEval` in borch.ts — and this side only hands the layers across and
    wraps what comes back as the same class the convolution went in as."""

    @staticmethod
    def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w=None, bn_b=None,
                             transpose=False):
        pair = _ts.nn.fuseConvBnWeights(
            handle(conv_w), None if conv_b is None else handle(conv_b),
            handle(bn_rm), handle(bn_rv), float(bn_eps),
            None if bn_w is None else handle(bn_w), None if bn_b is None else handle(bn_b),
            bool(transpose))
        return wrap(pair[0]), wrap(pair[1])

    @staticmethod
    def fuse_conv_bn_eval(conv, bn, transpose=False):
        if transpose:
            raise NotImplementedError("fuse_conv_bn_eval: transpose=True is not here yet")
        fused = _ts.nn.fuseConvBnEval(conv._m, bn._m)
        out = type(conv).__new__(type(conv))
        Module.__init__(out, fused)
        object.__setattr__(out, "training", False)
        return out


class _Utils:
    rnn = _Rnn()
    fusion = _Fusion()


utils = _Utils()


class Module:
    """One layer. **It works both wrapped and subclassed.**

    Wrapping takes one borch.ts layer: `Module(js_layer)`.

    **Subclassing has to work too.** It is the most common thing torch code
    does — write `class Net(nn.Module)`, attach layers as attributes in
    `__init__`, and write `forward`. Every golden case built its model out of
    `nn.Sequential` alone, so this was never asked, and the benchmark building a
    real ResNet caught it with
    `Module.__init__() missing 1 required positional argument`.

    A subclassed one has no `_m`. Parameters and `state_dict` come from **walking
    the layers attached as attributes**
    — torch does the same.
    """

    def __init__(self, module=None):
        object.__setattr__(self, "_m", module)
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        """**Writing a weight has to reach the layer underneath.**

        A wrapped layer keeps its tensors in the borch.ts module, and there was no
        `__setattr__` at all — so `layer.weight = nn.Parameter(w)` put the tensor on the
        Python object while `forward` went on through `self._m` with the weights it was
        built with. No exception, the right shapes, and a plausible wrong number: on a
        3→4 convolution it answered 9.94 where 68.44 was right.

        The sister defect one level down (`p.data = t` landing on a field nothing reads)
        is fixed in `_base.py`, and both were found the same way — by training a network
        in a browser and getting a different loss on identical weights.

        **The values move; the slot does not.** borch.ts declares a layer's `weight` as
        `readonly`, so there is no slot to swap even if we wanted one, and `copyFrom`'s
        comment over there gives the reason to prefer this anyway: the optimizer holds
        the parameter's handle, so replacing it leaves training running against something
        the optimizer no longer points at. torch does replace, and after it you are
        expected to rebuild the optimizer; here you are not.

        **A different shape stops here, and torch does not** — measured, both ways. torch
        takes the assignment and the layer breaks later, in `forward`, on a shape nobody
        can trace back to the line that set it. This cannot take it at all, because the
        slot is `readonly` over there, so it says so at the assignment with both shapes in
        the message. A stop at the cause beats a stop three frames away, and it is never a
        wrong value.

        Anything that is not a slot of the wrapped layer falls through to an ordinary
        attribute, as before.
        """
        target = None
        module = self.__dict__.get("_m")
        if module is not None and isinstance(value, Tensor) and not name.startswith("_"):
            for slot in (module.namedParameters(), module.namedBuffers()):
                if name in [str(k) for k in _js.Object.keys(slot)]:
                    target = wrap(getattr(slot, name))
                    break
        if target is None:
            object.__setattr__(self, name, value)
            return
        here = tuple(int(v) for v in target.shape)
        given = tuple(int(v) for v in value.shape)
        if here != given:
            # Raised before `copyFrom`, whose own message is a JavaScript `Error` reading
            # `size mismatch` with no mention of the attribute, the layer or torch.
            # **The core's type, not a new one** — `_base.py` says why where it refuses:
            # code that catches `BorchError` has to catch this too.
            from borch._base import BorchError                       # noqa: PLC0415
            raise BorchError(
                f"{type(self).__name__}.{name} is {here} and cannot become {given}.\n"
                "  torch would put the new tensor in the slot; borch.ts declares a layer's "
                "parameters\n"
                "  readonly, so this copies values into the slot instead and a different "
                "shape has\n"
                "  nowhere to go. Build the layer with the shape you want, or use "
                "`load_state_dict`.")
        from ._ops import no_grad                                    # noqa: PLC0415
        with no_grad():
            target._h.copyFrom(handle(value))

    # ── the layers a subclass attached as attributes ──────────────────────

    def _children(self):
        """The layers and tensors attached as attributes, **in the order they
        were attached.**

        The naming rule has to match torch's — a `state_dict` key is built from
        the attribute name, as in `conv1.weight`, and the golden loads weights
        by those names.
        """
        got = []
        for key, value in vars(self).items():
            if key.startswith("_"):
                continue
            # **Leave `_Holder` out and a container stops acting like one.**
            # Those classes exist because "a bare list is invisible", and
            # omitting them here makes the container itself invisible, which
            # lands back in the same place.
            if isinstance(value, (Module, _Wrap, Sequential, _Holder, Tensor)):
                got.append((key, value))
        return got

    # ── buffers ───────────────────────────────────────────────────────────
    #
    # **Buffers existed only as a special case of `BatchNorm`.** The only path
    # was `BatchNormND` over there carrying its running statistics in its own
    # `stateDict`; `register_buffer` did not exist, so **a user could not have a
    # model with buffers** — masks, positional tables, normalisation constants
    # all use that call, and all of them stopped at an `AttributeError`.

    def register_buffer(self, name, value, persistent=True):
        """A value that is not trained but is saved and restored. It is attached
        as an attribute too.

        Attached that way, `_children()` recognises it as a tensor and it rides
        in `state_dict` — so there is nothing to write on the saving side.
        `requiresGrad` is false, so `parameters()` and `named_parameters()` do
        not pick it up.
        """
        self.__dict__.setdefault("_buffers", {})[name] = value
        if not persistent:
            self.__dict__.setdefault("_nonpersistent", set()).add(name)
        else:
            self.__dict__.get("_nonpersistent", set()).discard(name)
        object.__setattr__(self, name, value)

    def named_buffers(self, persistent_only=False):
        """`(name, buffer)` pairs.

        For a wrapped layer they come **by subtraction** — `state_dict` minus
        `named_parameters` is the buffers. Keeping another list in borch.ts would
        make it a third list, and "two lists of the same thing" is a failure this
        repository has met repeatedly.
        """
        if self._m is None:
            skip = self.__dict__.get("_nonpersistent", set()) if persistent_only else ()
            out = [(n, v) for n, v in self.__dict__.get("_buffers", {}).items()
                   if n not in skip]
            for name, m in self._children():
                if isinstance(m, Tensor):
                    continue
                out.extend((f"{name}.{k}", v)
                           for k, v in _buffers_of(m, persistent_only))
            return out
        params = {n for n, _ in self.named_parameters()}
        return [(n, v) for n, v in self.state_dict().items() if n not in params]

    def buffers(self):
        return [b for _, b in self.named_buffers()]

    def __call__(self, *args):
        # A subclass has its own `forward`. Only a wrapper forwards to JS.
        if self._m is None:
            return self.forward(*args)
        return guarded(self._m.call, *[_arg(a) for a in args])

    def forward(self, *args):
        if self._m is None:
            raise NotImplementedError(f"{type(self).__name__} has no forward")
        return self(*args)

    def __repr__(self):
        """**How it prints is asked of borch.ts too.**

        How a layer prints itself is an answer with the same standing as a value
        — textbooks call `print(model)` and the golden froze those strings.
        Assembling it again in Python makes two copies, and two copies drift.
        Where there is no `describe` over there, only the class name comes back.
        """
        if self._m is None:
            # **Written here, because there is no layer over there to ask.** These
            # are the ones this file computes in Python — they hold their own
            # parameters — so `describe` has nothing to answer with, and the rules
            # come from the core's single table rather than a second wording.
            extra = getattr(self, "extra_repr", None)
            return f"{type(self).__name__}({extra() if extra else ''})"
        fn = getattr(self._m, "describe", None)
        return str(fn()) if fn is not None else f"{type(self).__name__}()"

    def parameters(self):
        if self._m is None:
            out = []
            for _, m in self._children():
                # **A parameter attached directly as an attribute is a
                # parameter too.**
                #
                # `state_dict` had this branch from the beginning and only this
                # place did not. So things holding **a tensor rather than a
                # child layer** as an attribute — `PReLU`, `GroupNorm` — offered
                # no parameters at all. `_params_of` asks
                # `hasattr(t, "parameters")`, a tensor has no such thing, and the
                # list comes back quietly empty: no exception, no warning, and
                # **it simply does not learn.** Exactly the shape the "containers"
                # comment further down warns about — while standing in it.
                if isinstance(m, Tensor):
                    if bool(m._h.requiresGrad):
                        out.append(m)
                    continue
                out.extend(_params_of(m))
            return out
        # A JS array cannot be iterated directly from Python — `to_py` has to
        # hand back a list.
        return [wrap(p) for p in self._m.parameters().to_py()]

    def state_dict(self):
        if self._m is None:
            out = {}
            # **Not every tensor attribute rides along.**
            #
            # Every `Tensor` attached as an attribute used to. torch carries only
            # what went through registration — `self.t = torch.ones(3)` is
            # neither a parameter nor a buffer and does not appear in
            # `state_dict`. That is why the `register_buffer` API exists. A model
            # parking an intermediate value on an attribute was shipping it in
            # its checkpoint, and a receiver running torch refuses it as an
            # unexpected key.
            #
            # Leaves that take gradients still ride along — "a tensor with only
            # the flag raised counts as a parameter" is **a divergence already
            # written down** and parity holds it by value. Tightening it here too
            # would quietly reverse that rule.
            registered = self.__dict__.get("_buffers", {})
            skip = self.__dict__.get("_nonpersistent", set())
            for name, m in self._children():
                if isinstance(m, Tensor):
                    if bool(m._h.requiresGrad):
                        out[name] = m
                    elif name in registered and name not in skip:
                        out[name] = m
                    continue
                for k, v in _state_of(m).items():
                    out[f"{name}.{k}"] = v
            return out
        got = self._m.stateDict()
        return {str(k): wrap(getattr(got, k)) for k in _js.Object.keys(got)}

    def named_parameters(self):
        """`(name, tensor)` pairs. torch code takes them through `dict(...)` and
        looks names up.

        **The same naming rule** as `state_dict` — a positional number in front,
        as in `0.weight`. Cases do look weights up by those names, so the rule
        has to match.

        **It is not the same list, though.** This was one line:
        `return list(self.state_dict().items())`. On a layer holding only
        parameters the two lists are identical, which hid it for a long time, but
        `state_dict` carries buffers too — asked of `BatchNorm` it produced
        `running_mean`, `running_var` and `num_batches_tracked` **posing as
        parameters.** Code handing that to an optimiser sets out to train the
        running statistics.

        It stayed hidden because the golden only ever asked `Linear`. Asking a
        layer with buffers produced it immediately —
        `container::BatchNorm/named_parameters keys`.
        """
        if self._m is None:
            out = []
            for name, m in self._children():
                # A tensor attached directly as an attribute is judged by the
                # same measure `parameters()` uses.
                if isinstance(m, Tensor):
                    if bool(m._h.requiresGrad):
                        out.append((name, m))
                    continue
                out.extend((f"{name}.{k}", v) for k, v in _named_of(m))
            return out
        got = self._m.namedParameters()
        return [(str(k), wrap(getattr(got, k))) for k in _js.Object.keys(got)]

    def load_state_dict(self, values, strict=True):
        if self._m is None:
            # Split on the leading name and hand it to the child —
            # `conv1.weight` becomes `weight` on `conv1`.
            own = dict(self._children())
            groups = {}
            for key, v in values.items():
                head, _, rest = key.partition(".")
                if not rest and isinstance(own.get(head), Tensor):
                    # **Copied inside `no_grad`.** A parameter is a leaf that
                    # takes gradients, and editing a leaf in place is refused, as
                    # it is in torch. `loadStateDict` over there already wraps
                    # it; only this branch did not.
                    from ._ops import no_grad
                    with no_grad():
                        own[head]._h.copyFrom(handle(v))
                    continue
                groups.setdefault(head, {})[rest] = v
            for head, sub in groups.items():
                if head in own:
                    own[head].load_state_dict(sub, strict)
                elif strict:
                    raise RuntimeError(f"load_state_dict: unexpected key '{head}'")
            return
        obj = _js.Object.new()
        for k, v in values.items():
            setattr(obj, k, handle(v))
        self._m.loadStateDict(obj, strict)

    def train(self, mode=True):
        # **torch keeps the flag on the module and code reads it.** It was only
        # forwarded, so `self.training` inside a subclass's `forward` — which is how
        # every dropout in torchvision decides whether to drop — stopped with
        # `AttributeError: training`, and `__getattr__` had already forwarded the
        # name to a borch.ts layer that has no such field.
        object.__setattr__(self, "training", bool(mode))
        if self._m is None:
            for _, m in self._children():
                if hasattr(m, "train"):
                    m.train(mode)
            return self
        self._m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def modules(self):
        """**Self first, then everything below**, as torch's does. Depth first and in
        the order the children were attached, because callers walk it to initialise
        weights and the order decides which draw each layer gets."""
        yield self
        for _, m in self._children():
            inner = getattr(m, "modules", None)
            if inner is None:
                yield m
                continue
            yield from inner()

    # ── walking the tree ──────────────────────────────────────────────────
    #
    # **Ten methods no check could see were missing.** Nothing read a class's
    # *methods* — the name axis counts a namespace's top-level names, the signature
    # axis compares constructors — so `nn.Module` held 14 of torch's 50 while
    # borch.ts had `children` and `namedChildren` all along.
    # `tests/test_class_methods.py` is the axis that asks now.

    def named_children(self):
        """The immediate children — **one level.** A `Tensor` attribute is not a
        child; only what walks like a module is."""
        for name, value in self._children():
            if not isinstance(value, Tensor):
                yield (name, value)

    def children(self):
        for _, m in self.named_children():
            yield m

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        """The whole tree with dotted names, **this one first and named `""`.**"""
        memo = set() if memo is None else memo
        if id(self) in memo:
            return
        if remove_duplicate:
            memo.add(id(self))
        yield (prefix, self)
        for name, child in self.named_children():
            walk = getattr(child, "named_modules", None)
            below = f"{prefix}.{name}" if prefix else name
            if walk is None:
                yield (below, child)
                continue
            yield from walk(memo, below, remove_duplicate)

    def apply(self, fn):
        """`fn` on every module, **children first**, then this one. Returns `self`."""
        for _, child in self.named_children():
            inner = getattr(child, "apply", None)
            if inner is None:
                fn(child)
            else:
                inner(fn)
        fn(self)
        return self

    def add_module(self, name, module):
        setattr(self, name, module)

    register_module = add_module

    def register_parameter(self, name, param):
        setattr(self, name, param)

    def get_submodule(self, target):
        """One module by its dotted name — the shape of a checkpoint key."""
        if target == "":
            return self
        at = self
        for part in target.split("."):
            found = dict(at.named_children()).get(part)
            if found is None:
                raise AttributeError(
                    f"{type(at).__name__} has no attribute `{part}`")
            at = found
        return at

    def _dotted(self, target, bank, what):
        head, _, leaf = target.rpartition(".")
        owner = self.get_submodule(head)
        held = dict(bank(owner))
        if leaf not in held:
            raise AttributeError(
                f"{type(owner).__name__} has no {what} named `{leaf}`")
        return held[leaf]

    def get_parameter(self, target):
        return self._dotted(target, lambda m: m.named_parameters(), "parameter")

    def get_buffer(self, target):
        return self._dotted(target, lambda m: m.named_buffers(), "buffer")

    def requires_grad_(self, requires_grad=True):
        """Freeze or unfreeze the whole tree, handing back the model."""
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self

    def __getattr__(self, name):
        """Forward what the layer holds, such as `bn.weight`.

        **`_m` must not be asked for again here.** A subclass without one — a
        `_Wrap` — makes `__getattr__` call itself and recurse forever. The
        failure came out of a CNN training case as `RecursionError`, a long way
        from its cause.
        """
        if name.startswith("_") or self._m is None:
            raise AttributeError(name)
        # **A field that is there and null is not a field that is missing.** This read
        # `getattr(…, default=None)` and raised on both, so `bn.running_mean` under
        # `track_running_stats=False` — where torch answers `None` — came back as
        # *the borch.ts layer does not have `running_mean`*. The property is declared
        # over there and holds `null`; asking for it and catching the absence tells
        # the two apart, which the default never could.
        try:
            got = getattr(self._m, camel(name))
        except AttributeError:
            raise AttributeError(
                f"the borch.ts layer does not have `{name}`") from None
        if got is None:
            return None
        if _ts.isTensor(got):
            return wrap(got)
        # **A submodule came back as the raw borch.ts object**, so `model.fc.weight` was
        # reachable and unusable: the tensor behind it never went through `wrap`, and
        # `copy_` on it stopped at `size mismatch: [4,5] <- [torch.Size([4, 5])]` —
        # a borch.ts message holding a stringified Python argument, which is what a raw
        # handle receiving a wrapped value looks like.
        #
        # Found through `LinearCrossEntropyLoss`, which is the first layer here to own
        # a named child; every layer before it held tensors alone, so the branch had
        # nothing to be wrong about. **`forward` is the mark**, and duck-typing rather
        # than a class check is deliberate: borch.ts's layers do not share a base this
        # side can see, and asking for the one method they all have is the question
        # that survives that.
        if callable(getattr(got, "forward", None)):
            return Module(got)
        # **A method the layer has and this class does not** — bimm's `forwardFeatures`
        # and `forwardHead` are the first. Called through `guarded` so tensors cross as
        # handles and come back wrapped, the way `__call__` does it; keyword arguments
        # ride along in the order written (`forward_head(h, pre_logits=True)`).
        if callable(got):
            def call(*args, **kw):
                return guarded(got, *[_arg(a) for a in args], *[_arg(v) for v in kw.values()])
            call.__name__ = name
            return call
        return got


def _layer(js_name, *args):
    """Wrap a layer from over there.

    **Everything produced here is the one class `Module`.** So
    `type(model.fc).__name__` is always `Module` where torch says `Linear`.
    Minting a class per name would fix it, except borch.ts has no per-dimension
    names — `BatchNorm2d` is just `BatchNormND` over there — so the names would
    come out half right, and a half-right name confuses more than `Module` does.
    Splitting the names over there has to come first.
    """
    return Module(getattr(_ts.nn, js_name).new(*args))


class _Wrap:
    """The ones borch.ts does not have as layers and **does have as tensor
    methods.**

    `nn.Softmax(dim)` is `x.softmax(dim)`. Naming what exists with torch's name
    rather than approximating what does not, so the values come from the same
    place.

    **It does not subclass `Module`.** Subclassing it meant `Module`'s methods
    looked for an `_m` that was not there, `__getattr__` called itself, and it
    recursed forever — surfacing as `RecursionError` in a CNN training case, a
    long way from the cause. These layers have no parameters, so there is
    nothing to inherit from `Module` anyway.
    """

    __slots__ = ("_fn", "_shown")

    def __init__(self, fn, shown=None):
        """`shown` is **how it prints**, and there is nowhere else it can come from.

        These are the layers borch.ts has no class for, so `describe` has nobody to
        ask — and without a name the golden read
        `<borch_webgpu._nn._Wrap object at 0x…>` where torch prints
        `Flatten(start_dim=1, end_dim=-1)`. Given as the finished string rather than
        as fields, because there is no rule shared with anything: each of these is one
        line of torch's `extra_repr` and copying the fields would only put the same
        line further away.
        """
        self._fn = fn
        self._shown = shown

    def __repr__(self):
        return self._shown or f"{type(self).__name__}()"

    def forward(self, *args):
        return self(*args)

    def __call__(self, *args):
        return self._fn(*args)

    def parameters(self):
        return []

    def state_dict(self):
        return {}

    # No parameters, so an empty list. **Absent, it raises `AttributeError`** —
    # torch answers `nn.ReLU().named_parameters()` with an empty one.
    def named_parameters(self):
        return []

    def named_buffers(self, persistent_only=False):
        return []

    def load_state_dict(self, values, strict=True):
        pass

    def train(self, mode=True):
        return self

    def eval(self):
        return self


class Sequential:
    """**Chained on the Python side.**

    Handing it to borch.ts's `Sequential` would require a JavaScript object per
    layer, and things like `Softmax` and `Flatten` are Python layers wrapping a
    tensor method, so they have none. A `Lambda` slot could be added over there,
    but then the kernel-side surface grows because of layers with no parameters.
    Chaining in Python produces the same values.

    The naming rule matches borch.ts's — a positional number in front, as in
    `0.weight`, and the golden loads and reads weights by those names.

    **It is a class, and it was a function.** `def Sequential(*layers)` returning one
    of these answers every call made of an *instance* and none of the ones made of the
    *name*: `class ConvNormActivation(nn.Sequential)` — which is how torchvision builds
    every block it ships — calls the function where a metaclass belongs and stops with
    *function() argument 'code' must be code, not str*. **Twenty-seven golden cases**,
    every one of them a block whose layers all worked.

    No `__slots__`, because a subclass sets its own attributes: `ConvNormActivation`
    keeps `out_channels` on itself, as torchvision's does.
    """

    def __init__(self, *layers):
        flat = []
        for one in layers:
            flat.extend(one if isinstance(one, (list, tuple)) else [one])
        self.layers = flat

    def __call__(self, x):
        for m in self.layers:
            x = m(x)
        return x

    def forward(self, x):
        return self(x)

    def __repr__(self):
        """torch's block, and **this class had none.**

        `print(model)` — the first line anybody writes after building one — showed
        `<borch_webgpu._nn.Sequential object at 0x…>`, an address where the other two
        implementations print the layers. Found by a case reading a child's name out
        of its repr, which is a question about the walk and answered this one instead.

        Children indent by two, and a child that is itself a block indents its own
        lines again — that recursion is the whole of torch's rule.
        """
        if not self.layers:
            return f"{type(self).__name__}()"
        rows = []
        for i, layer in enumerate(self.layers):
            body = repr(layer).split("\n")
            head = f"  ({i}): {body[0]}"
            rows.append("\n".join([head] + [f"  {line}" for line in body[1:]]))
        return f"{type(self).__name__}(\n" + "\n".join(rows) + "\n)"

    def parameters(self):
        return [p for m in self.layers for p in _params_of(m)]

    def state_dict(self):
        out = {}
        for i, m in enumerate(self.layers):
            for k, v in _state_of(m).items():
                out[f"{i}.{k}"] = v
        return out

    def named_parameters(self):
        return [(f"{i}.{k}", v)
                for i, m in enumerate(self.layers) for k, v in _named_of(m)]

    def named_buffers(self, persistent_only=False):
        return [(f"{i}.{k}", v) for i, m in enumerate(self.layers)
                for k, v in _buffers_of(m, persistent_only)]

    def load_state_dict(self, values, strict=True):
        groups = {}
        for key, v in values.items():
            head, _, rest = key.partition(".")
            groups.setdefault(int(head), {})[rest] = v
        for i, sub in groups.items():
            self.layers[i].load_state_dict(sub, strict)

    def train(self, mode=True):
        self.training = bool(mode)
        for m in self.layers:
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def modules(self):
        """As `Module.modules` — **a container that does not walk stops the walk**,
        and every block in `ops` is one of these."""
        yield self
        for m in self.layers:
            inner = getattr(m, "modules", None)
            if inner is None:
                yield m
                continue
            yield from inner()

    # ── walking the tree ──────────────────────────────────────────────────
    #
    # **This class does not inherit `Module`**, so the ten written there do not
    # reach it — and `Sequential` is what every golden case walking a tree is built
    # from. The names are positional, as `state_dict`'s already are.
    #
    # `modules` above is what these were missing beside: it existed and the nine
    # torch spells out did not, in the same class, on the same walk.

    def named_children(self):
        return [(str(i), m) for i, m in enumerate(self.layers)]

    def children(self):
        return list(self.layers)

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        memo = set() if memo is None else memo
        if id(self) in memo:
            return
        if remove_duplicate:
            memo.add(id(self))
        yield (prefix, self)
        for name, child in self.named_children():
            walk = getattr(child, "named_modules", None)
            below = f"{prefix}.{name}" if prefix else name
            if walk is None:
                yield (below, child)
                continue
            yield from walk(memo, below, remove_duplicate)

    def apply(self, fn):
        for m in self.layers:
            inner = getattr(m, "apply", None)
            if inner is None:
                fn(m)
            else:
                inner(fn)
        fn(self)
        return self

    def add_module(self, name, module):
        """**Appended, not attached by name.** `Sequential`'s names are positions,
        so a name that is not the next index would make `state_dict`'s keys and the
        list disagree — torch's `Sequential.add_module` has the same rule."""
        if name != str(len(self.layers)):
            raise KeyError(
                f"Sequential names its children by position — `{name}` is not "
                f"`{len(self.layers)}`")
        self.layers.append(module)

    register_module = add_module

    def get_submodule(self, target):
        if target == "":
            return self
        at = self
        for part in target.split("."):
            found = dict(at.named_children()).get(part)
            if found is None:
                raise AttributeError(
                    f"{type(at).__name__} has no attribute `{part}`")
            at = found
        return at

    def _dotted(self, target, bank, what):
        head, _, leaf = target.rpartition(".")
        owner = self.get_submodule(head)
        held = dict(bank(owner))
        if leaf not in held:
            raise AttributeError(
                f"{type(owner).__name__} has no {what} named `{leaf}`")
        return held[leaf]

    def get_parameter(self, target):
        return self._dotted(target, lambda m: m.named_parameters(), "parameter")

    def get_buffer(self, target):
        return self._dotted(target, lambda m: m.named_buffers(), "buffer")

    def requires_grad_(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self


def _params_of(m):
    return m.parameters() if hasattr(m, "parameters") else []


def _state_of(m):
    return m.state_dict() if hasattr(m, "state_dict") else {}


def _named_of(m):
    """`(name, parameter)` pairs. **It differs from `_state_of` by exactly the
    buffers.**

    Written as one function, the buffers become parameters. That is why there
    are two.
    """
    return m.named_parameters() if hasattr(m, "named_parameters") else []


def _buffers_of(m, persistent_only=False):
    """`(name, buffer)` pairs. The third list, and exactly the difference
    between the other two."""
    if not hasattr(m, "named_buffers"):
        return []
    return m.named_buffers(persistent_only)


def _zero_grad(owner, set_to_none=True):
    """`model.zero_grad()` — **torch's, and no class on this side had it.**

    It is the line a training loop opens with as often as the optimizer's, and it
    stopped with `'Sequential' object has no attribute 'zero_grad'`. `set_to_none`
    means here what it means there: `False` leaves a zeroed tensor where `True`
    drops it, which is the difference anything reading `grad` between two calls
    depends on.

    **Written once and attached to the five classes that hold parameters.** They
    share no base — `Module`, `_Wrap`, `Sequential` and the two `_Holder` families
    each grew their own `parameters()` — so a method per class would be five copies
    of one rule, and the list below is the one place naming who has it.
    """
    from ._ops import zeros_like

    for p in owner.parameters():
        if set_to_none or p.grad is None:
            p.grad = None
        else:
            p.grad = zeros_like(p.grad)


# ── containers ──────────────────────────────────────────────────────────────
#
# **This is where things go quietly wrong.** Put layers or parameters in a bare
# list, attach it as an attribute, and `Module._children()` does not recognise
# it — `parameters()` does not offer them, the optimiser never sees them, and
# **the loss still falls** because the remaining parameters compensate. No
# exception, no warning.
#
# torch fails to recognise it in exactly the same way, which is why torch has
# these four classes.

def Parameter(value, requires_grad=True):
    """A tensor that is trained. In torch it is a subclass of `Tensor`; here it
    is **a leaf tensor.**

    borch.ts has no separate `Parameter` — a leaf that takes gradients is a
    parameter. The value makes one trip to the CPU and comes back as a new leaf,
    which runs once per layer at construction and is not a cost inside the
    training loop.
    """
    from ._base import tensor as _t

    arr = value.numpy() if isinstance(value, Tensor) else value
    return _t(arr, requires_grad=requires_grad)


class _Holder:
    """The housekeeping the four containers share. **The naming rule has to
    live in one place.**

    A `state_dict` key is built by putting the slot name in front, as in
    `layers.0.weight`, and writing that rule out per container means that when
    one of them drifts there is no finding where.
    """

    def parameters(self):
        return [p for _, m in self._entries() for p in _params_of(m)]

    def state_dict(self):
        out = {}
        for name, m in self._entries():
            for k, v in _state_of(m).items():
                out[f"{name}.{k}"] = v
        return out

    def named_parameters(self):
        return [(f"{name}.{k}", v)
                for name, m in self._entries() for k, v in _named_of(m)]

    def named_buffers(self, persistent_only=False):
        return [(f"{name}.{k}", v) for name, m in self._entries()
                for k, v in _buffers_of(m, persistent_only)]

    def load_state_dict(self, values, strict=True):
        own = dict(self._entries())
        groups = {}
        for key, v in values.items():
            head, _, rest = key.partition(".")
            groups.setdefault(head, {})[rest] = v
        for head, sub in groups.items():
            if head in own:
                own[head].load_state_dict(sub, strict)
            elif strict:
                raise RuntimeError(f"load_state_dict: unexpected key '{head}'")
        return self

    def train(self, mode=True):
        for _, m in self._entries():
            if hasattr(m, "train"):
                m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def modules(self):
        """As `Module.modules` — `ModuleList` is what `FeaturePyramidNetwork` keeps
        its blocks in, so a walk that stops here reaches none of them."""
        yield self
        for _, m in self._entries():
            inner = getattr(m, "modules", None)
            if inner is None:
                yield m
                continue
            yield from inner()


def _ordered(mapping):
    """torch's ordering rule. **A plain dict has its keys sorted on the way in.**

    An `OrderedDict` keeps insertion order and a plain `dict` is sorted. Not
    matching that diverges the order of `named_parameters`, which is the order of
    `state_dict` — and the golden did catch this: given `{"w":…, "b":…}`, torch
    produced `ws.b ws.w`.
    """
    import collections as _c

    items = dict(mapping or {})
    if isinstance(mapping, (_ModuleDict, _ParameterDict, _c.OrderedDict)):
        return list(items.items())
    return sorted(items.items(), key=lambda kv: str(kv[0]))


class _ModuleList(_Holder):
    """A list of layers. The index is the name — `layers.0.weight`.

    Without `append` there is no way to write a model whose layer count is not
    fixed in advance.
    """

    def __init__(self, mods=()):
        self.layers = list(mods)

    def _entries(self):
        return [(str(i), m) for i, m in enumerate(self.layers)]

    def append(self, module):
        self.layers.append(module)
        return self

    def extend(self, mods):
        self.layers.extend(mods)
        return self

    def insert(self, index, module):
        self.layers.insert(index, module)
        return self

    def __getitem__(self, i):
        return self.layers[i]

    def __setitem__(self, i, module):
        self.layers[i] = module

    def __iadd__(self, mods):
        return self.extend(mods)

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)


class _ModuleDict(_Holder):
    """Named layers. The name given becomes the `state_dict` key."""

    def __init__(self, mods=None):
        self.mods = dict(_ordered(mods))

    def _entries(self):
        return list(self.mods.items())

    def __getitem__(self, key):
        return self.mods[key]

    def __setitem__(self, key, module):
        self.mods[str(key)] = module

    def __contains__(self, key):
        return key in self.mods

    def __iter__(self):
        return iter(self.mods)

    def __len__(self):
        return len(self.mods)

    def keys(self):
        return self.mods.keys()

    def values(self):
        return self.mods.values()

    def items(self):
        return self.mods.items()

    def update(self, mods):
        self.mods.update(dict(_ordered(mods)))
        return self


class _ParamHolder(_Holder):
    """The one holding parameters directly. Its leaves are tensors, so
    `_Holder`'s walk does not apply."""

    def parameters(self):
        return [p for _, p in self._entries()]

    def state_dict(self):
        return dict(self._entries())

    # **Its leaves are tensors, so `_Holder`'s version does not work.** That one
    # asks each child for `named_parameters` again, a tensor has no such thing,
    # and the list comes back empty — every parameter disappears without an
    # exception. `parameters()` just above is overridden for the same reason.
    def named_parameters(self):
        return list(self._entries())

    # Everything it holds is a parameter — there are no buffers at all.
    def named_buffers(self, persistent_only=False):
        return []

    def load_state_dict(self, values, strict=True):
        own = dict(self._entries())
        for key, v in values.items():
            if key in own:
                own[key]._h.copyFrom(handle(v))
            elif strict:
                raise RuntimeError(f"load_state_dict: unexpected key '{key}'")
        return self

    def train(self, mode=True):
        return self


class _ParameterList(_ParamHolder):
    """A list of `Parameter`s. **Without it there is no substitute.**"""

    def __init__(self, params=()):
        self.params = list(params)

    def _entries(self):
        return [(str(i), p) for i, p in enumerate(self.params)]

    def append(self, param):
        self.params.append(param)
        return self

    def extend(self, params):
        self.params.extend(params)
        return self

    def __getitem__(self, i):
        return self.params[i]

    def __setitem__(self, i, param):
        self.params[i] = param

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)


class _ParameterDict(_ParamHolder):
    """Named `Parameter`s."""

    def __init__(self, params=None):
        self.params = dict(_ordered(params))

    def _entries(self):
        return list(self.params.items())

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, param):
        self.params[str(key)] = param

    def __contains__(self, key):
        return key in self.params

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def items(self):
        return self.params.items()

    def update(self, params):
        self.params.update(dict(_ordered(params)))
        return self


# **The five that hold parameters get `zero_grad`.** `_Holder` covers
# `_ModuleList` and `_ModuleDict`, `_ParamHolder` the two parameter containers,
# so naming the four bases plus `Sequential` reaches all of them. Attached rather
# than written into each, for the reason in `_zero_grad`'s own docstring.
for _owner in (Module, _Wrap, Sequential, _Holder, _ParamHolder):
    _owner.zero_grad = _zero_grad
del _owner


def ModuleList(mods=()):
    return _ModuleList(mods)


def ModuleDict(mods=None):
    return _ModuleDict(mods)


def ParameterList(params=()):
    return _ParameterList(params)


def ParameterDict(params=None):
    return _ParameterDict(params)




def Linear(inf, outf, bias=True):
    return _layer("Linear", inf, outf, bias)


ASMoutput = _collections.namedtuple("ASMoutput", ["output", "loss"])


class _AdaptiveLogSoftmax(Module):
    """`forward` takes the target as well and **produces two answers** — the log
    probability at the target and the loss.

    Over there `forward` is blocked and `run(x, target)` stands in its place, so
    that a layer shaped unlike the others says so where it is called rather than
    passing through.
    """

    def __call__(self, x, target):
        got = self._m.run(handle(x), handle(target))
        return ASMoutput(wrap(got.output), wrap(got.loss))

    def forward(self, x, target):
        return self(x, target)


def AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0,
                               head_bias=False):
    return _AdaptiveLogSoftmax(_ts.nn.AdaptiveLogSoftmaxWithLoss.new(
        in_features, n_classes, _js_list(list(cutoffs)),
        float(div_value), bool(head_bias)))


# **torch's list, and the crossing is positional.** These three read
# `(cin, cout, k, stride, padding, bias)` while borch.ts had already moved to torch's
# `(inC, outC, kernel, stride, padding, dilation, groups, bias, paddingMode)`. So
# `bias` was landing in `dilation`, and a boolean where a stride-like number belongs
# came out the far side as `TypeError: v is not iterable` — from the kernel, several
# frames below anything that mentions a convolution.
#
# The commit that moved borch.ts said it had carried the change "across core, borch.ts
# and the binding". It had carried `conv{1,2,3}d`, the functions; these three are the
# layers, in the same file, forty lines away. **Two call sites for one thing and only
# one of them in mind** — the golden's `ndim::nn.Conv1d` and both `train::CNN` rows
# have been red since, under an error naming neither convolutions nor arguments.
class _ConvND(Module):
    """**A class, because `isinstance` is a question torchvision asks.**

    These were three functions returning a plain `Module`, which answers every call
    made of an instance and none made of the name. `FeaturePyramidNetwork` initialises
    its convolutions with `isinstance(module, nn.Conv2d)` — torchvision's own line —
    and a function in the second seat is `TypeError: isinstance() arg 2 must be a
    type`. The same shape as `nn.Sequential` being a function, one file over.

    borch.ts has `Conv1d`, `Conv2d` and `Conv3d` as classes over there, so the name of
    this one picks which to stand up.
    """

    def __init__(self, cin, cout, k, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode="zeros"):
        super().__init__(getattr(_ts.nn, type(self).__name__).new(
            cin, cout, k, stride, padding, int(dilation), int(groups), bool(bias),
            padding_mode))


class Conv1d(_ConvND):
    pass


class Conv2d(_ConvND):
    pass


class Conv3d(_ConvND):
    pass


# ── one step of a recurrence ────────────────────────────────────────────
#
# **It goes through `step`, not `call`.** `Module.call(x)` over there takes one
# argument and cannot receive state. `LSTMCell` has a pair of states and answers
# with a pair, so that is unpacked here as well.

class _Cell(Module):
    _pairs = False

    def __call__(self, x, hx=None):
        if not self._pairs:
            args = (handle(x),) if hx is None else (handle(x), handle(hx))
            return wrap(guarded(self._m.step, *args))
        if hx is None:
            got = settle(self._m.step(handle(x)))
        else:
            # **`_js_list` is for lists of integers only** — a tensor stops it
            # at `int()`. The pair is built with `Array.of`.
            got = settle(self._m.step(
                handle(x), _js.Array.of(handle(hx[0]), handle(hx[1]))))
        return (got[0], got[1])


def _cell(name, pairs=False):
    def make(input_size, hidden_size, bias=True, **kw):
        args = [input_size, hidden_size, bias]
        if name == "RNNCell":
            args.append(kw.get("nonlinearity", "tanh"))
        made = _Cell(getattr(_ts.nn, name).new(*args))
        object.__setattr__(made, "_pairs", pairs)
        return made

    make.__name__ = name
    return make


RNNCell = _cell("RNNCell")
GRUCell = _cell("GRUCell")
LSTMCell = _cell("LSTMCell", pairs=True)


# ── the remaining layers ────────────────────────────────────────────────
#
# **A layer taking more than one argument cannot go through `_layer`.** The
# wrapper's `__call__` forwards to `call(x)` over there, while `Bilinear` takes
# two and `EmbeddingBag(offsets)` takes a list. Those two are wired by hand.

class _Bilinear(Module):
    def __call__(self, x1, x2):
        return wrap(guarded(self._m.call2, handle(x1), handle(x2)))


def Bilinear(in1, in2, out, bias=True):
    return _Bilinear(_ts.nn.Bilinear.new(in1, in2, out, bool(bias)))


class _EmbeddingBag(Module):
    def __call__(self, idx, offsets=None):
        if offsets is None:
            return wrap(guarded(self._m.call, handle(idx)))
        starts = [int(v) for v in _to_list(offsets)]
        return wrap(guarded(self._m.callOffsets, handle(idx),
                            _js_list(starts)))


def _to_list(t):
    got = handle(t).toArray() if hasattr(handle(t), "toArray") else t
    return settle(got) if hasattr(got, "then") else list(got)


def EmbeddingBag(num, dim, max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 mode="mean", sparse=False, _weight=None, include_last_offset=False,
                 padding_idx=None):
    """torch's list, and **the call across the boundary is positional.**

    `mode` was third here and third over there until both moved to sixth — in two
    different commits, and in between this handed a mode string to `maxNorm`, which
    rewrites the embedding table in place. It did not return a wrong number: it
    raised `a leaf Variable that requires grad is being used in an in-place
    operation`, from a layer nobody had asked to renormalise anything.

    Neither compiler reaches this call. `tsc` reads the TypeScript and stops at the
    bridge; Python has nothing to read. **The binding golden is what found it**, and
    it only runs where real torch can dump the answers.
    """
    return _EmbeddingBag(_ts.nn.EmbeddingBag.new(
        num, dim, max_norm, float(norm_type), bool(scale_grad_by_freq),
        mode, bool(sparse), handle(_weight) if _weight is not None else None,
        bool(include_last_offset), padding_idx))


def Embedding(num, dim, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False,
              ):
    """torch's list, in torch's order, across a positional bridge.

    **`padding_idx` is second and `max_norm` third**, which is the opposite of
    `EmbeddingBag` next door — there `max_norm` is third and `padding_idx` is last.
    Two neighbouring layers, two different orders, and torch is the reason for both.
    Swapping them here would not raise: `padding_idx=1` reaching `maxNorm` renormalises
    the whole table to length 1 on every forward pass, and the values would simply be
    wrong.

    Wired late. The comment on borch.ts's `embedding()` said no layer existed on any of
    the three and that the golden did not ask — true when written, false once the core
    took torch's nine, and the reason seven failing rows sat unexamined.
    """
    return Module(_ts.nn.Embedding.new(
        num, dim, padding_idx, max_norm, float(norm_type),
        bool(scale_grad_by_freq), bool(sparse),
        handle(_weight) if _weight is not None else None, bool(_freeze)))


def _undefined():
    """JavaScript's `undefined`, for the gaps `_MISC_ARGS` leaves in the middle —
    `LocalResponseNorm(2, k=2.0)` fills the first and last of four slots.

    **This is a guard, not a fix, and the docstring here said otherwise.** It claimed
    a `None` would arrive as `null`, the TypeScript default would not fire, and the
    layer would compute with it — citing an `OneCycleLR` defect in `_optim.py`.
    Measured in the browser afterwards: **Pyodide hands a Python `None` across as
    `undefined`, and the default does apply.** So this changes nothing, and the
    `OneCycleLR` `NaN` came from a stale row in that file's table rather than from
    the conversion.

    Written that way because the sentence was copied from `_optim.py` rather than
    measured — **which is how a wrong reason spreads**: it was plausible, it sat
    beside working code, and repeating it made it look confirmed. Kept as a guard
    because it costs nothing and Pyodide's conversion is not ours to promise across
    versions; if it ever starts mattering, that is a change underneath and this is
    the place that says so.
    """
    import js
    return js.undefined


# borch.ts's constructor parameters, in borch.ts's order, under torch's spelling —
# the same kind of table as `_SCHED_ARGS` in `_optim.py`, and it exists for the same
# reason: the far side takes positions and torch code writes keywords.
#
# **Before this, `_misc_layer` forwarded `*args` and threw every keyword away except
# `scale_factor`.** So `LocalResponseNorm(2, alpha=1.0, beta=2.0, k=2.0)` built the
# layer on its defaults, and `Unfold(2, dilation=2)`, `Fold(4, 2, padding=1)` and
# `RReLU(lower=0.2)` did the same. Nothing raised. The layer answered, with numbers
# from a configuration nobody asked for.
#
# It was found by a `repr` case added at a **non-default** value to catch a missing
# decimal point on the other side, and the first thing that case said was that the
# arguments never arrived at all. A case whose default answer would be right cannot
# tell those two apart, because it cannot tell either of them from working.
#
# `tests/test_scheduler_table.py` compiles this against `borch-ts/src/nn.ts`.
_MISC_ARGS = {
    "Unfold": ("kernel_size", "dilation", "padding", "stride"),
    "Fold": ("output_size", "kernel_size", "dilation", "padding", "stride"),
    "LocalResponseNorm": ("size", "alpha", "beta", "k"),
    "Softmax2d": (),
    "RReLU": ("lower", "upper", "inplace"),
    # **borch.ts takes `scale` and nothing else.** torch's `size=` is a different way
    # of asking and that side has no seat for it, so it is refused by name below rather
    # than dropped — an argument that raises with its own name beats one that silently
    # takes the default.
    # **`size` leads, because torch leads with it.** These read `("scale_factor",)`
    # while borch.ts took a scale alone; it takes both now and refuses unless exactly
    # one is given, so a row naming only the second would lay the scale into `size`.
    # Both are numbers, so nothing in three languages would have said a word.
    "UpsamplingNearest2d": ("size", "scale_factor"),
    "UpsamplingBilinear2d": ("size", "scale_factor"),
}

# Accepted by the core and with nowhere to go on the other side. Refused rather than
# ignored, and listed rather than counted.
#
# **This table is empty, and that is the interesting state.** Its one row said
# `("RReLU", "inplace"): "there is no in-place activation on that side"` — true when it
# was written and false the moment borch.ts grew `writeBack`. A refusal carries its
# reason as prose, and prose does not notice when the world moves under it; what caught
# this was `test_the_binding_sends_each_argument_to_the_seat_borch_ts_keeps_for_it`
# compiling the row against the far side and finding one more seat than the table knew.
#
# Keep the name and the empty dict rather than deleting them: the mechanism is what
# makes a refusal a listed decision instead of a silent drop, and the next argument that
# genuinely has nowhere to go belongs here.
_MISC_REFUSED = {}


def _misc_layer(name):
    def make(*args, **kw):
        # **A Python tuple must not cross as-is.** Over there it becomes a
        # briefly borrowed proxy that is discarded soon after, and the failure
        # arrives later as `borrowed proxy was automatically destroyed` —
        # `Fold((4,4), 2)` was that place.
        laid = [_js_list(list(a)) if isinstance(a, (list, tuple)) else a
                for a in args]
        order = _MISC_ARGS.get(name, ())
        for key, value in kw.items():
            if (name, key) in _MISC_REFUSED:
                raise NotImplementedError(
                    f"{name}({key}=…) is not carried into the browser — "
                    f"{_MISC_REFUSED[(name, key)]}.")
            if key not in order:
                raise TypeError(f"{name}() got an unexpected keyword argument {key!r}")
            at = order.index(key)
            # **A slot given twice is a refusal**, as it is in torch and as Python
            # enforces for the core's real signatures. `LocalResponseNorm(2, 1.0,
            # alpha=2.0)` used to take the keyword and answer.
            if at < len(args):
                raise TypeError(
                    f"{name}() got multiple values for argument {key!r}")
            while len(laid) <= at:
                laid.append(_undefined())
            laid[at] = (_js_list(list(value))
                        if isinstance(value, (list, tuple)) else value)
        return _layer(name, *laid)

    make.__name__ = name
    return make


for _misc in _MISC_ARGS:
    globals()[_misc] = _misc_layer(_misc)


# ── repositioning, and channel-wise dropout ─────────────────────────────

for _shuffle in ("PixelShuffle", "PixelUnshuffle", "ChannelShuffle",
                 "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout",
                 "FeatureAlphaDropout"):
    globals()[_shuffle] = (lambda name: lambda *a: _layer(name, *a))(_shuffle)


# ── the thirteen lazy layers ────────────────────────────────────────────
#
# All of them live in borch.ts. **They materialise over there too** — the
# prototype is swapped, so the Python wrapper stays and only its insides change.
# `__repr__` asks `describe()`, so the printed form before and after follows on
# its own.

for _lazy in ("LazyLinear",
              *(f"Lazy{k}{d}d" for k in ("Conv", "ConvTranspose", "BatchNorm",
                                         "InstanceNorm") for d in (1, 2, 3))):
    globals()[_lazy] = (lambda name: lambda *a: _layer(name, *a))(_lazy)


# ── loss layers ─────────────────────────────────────────────────────────
#
# All of them live in borch.ts. What happens here is **moving the argument
# order** — torch puts `reduction` in the middle and the other side puts it at
# the end, so it arrives by name and is unrolled into position.

_LOSS_LAYERS = {
    # **borch.ts moved this to torch's `(reduction, delta)` and the table stayed.**
    # `HuberLoss(delta=0.5)` put 0.5 into `reduction`. It raised only because that
    # constructor validates its own string — the same luck that made the scheduler
    # table's break findable, and the opposite of what happened to `nll_loss` next
    # door, where the wrong seat was a number and swallowed a string as NaN.
    # **`size_average` and `reduce` are in these rows because borch.ts now has the
    # seats.** They are torch's deprecated pair, and this table is read as *borch.ts's
    # parameter order* — a row that omits them puts the reduction two seats early, so
    # `MSELoss(reduction="sum")` would arrive as `sizeAverage="sum"` and fold to the
    # mean. The three rows where the pair is not adjacent are torch's doing, not a
    # transcription slip: `ignore_index` and `eps` sit between them.
    #
    # `HuberLoss`, `GaussianNLLLoss` and `TripletMarginWithDistanceLoss` have no pair
    # in torch either — newer names, and they are the control that says the rows above
    # are copied rather than pattern-filled.
    "HuberLoss": ("reduction", "delta"),
    "KLDivLoss": ("size_average", "reduce", "reduction", "log_target"),
    "PoissonNLLLoss": ("log_input", "full", "size_average", "eps", "reduce",
                       "reduction"),
    "GaussianNLLLoss": ("full", "eps", "reduction"),
    "MarginRankingLoss": ("margin", "size_average", "reduce", "reduction"),
    "CosineEmbeddingLoss": ("margin", "size_average", "reduce", "reduction"),
    "HingeEmbeddingLoss": ("margin", "size_average", "reduce", "reduction"),
    "SoftMarginLoss": ("size_average", "reduce", "reduction"),
    "TripletMarginLoss": ("margin", "p", "eps", "swap", "size_average", "reduce",
                          "reduction"),
    "TripletMarginWithDistanceLoss": ("distance_function", "margin", "swap",
                                      "reduction"),
    # **`weight` was missing from this row and borch.ts has taken it first for a
    # while.** `MultiLabelSoftMarginLoss(reduction="sum")` laid the string out into
    # the first seat, which is the class weights — the exact defect the comment on
    # `HuberLoss` above describes, sitting two rows below it.
    "MultiLabelSoftMarginLoss": ("weight", "size_average", "reduce", "reduction"),
    "MultiMarginLoss": ("p", "margin", "weight", "size_average", "reduce",
                        "reduction"),
    "MultiLabelMarginLoss": ("size_average", "reduce", "reduction"),
    "PairwiseDistance": ("p", "eps", "keepdim"),
    "CosineSimilarity": ("dim", "eps"),
}


def _loss_layer(name, order):
    def make(*args, **kw):
        return _layer(name, *_lay_out(order, args, kw))

    make.__name__ = name
    return make


for _loss_name, _loss_order in _LOSS_LAYERS.items():
    globals()[_loss_name] = _loss_layer(_loss_name, _loss_order)


# ── the fifteen padding layers ──────────────────────────────────────────
#
# All of them live in borch.ts. What happens here is joining the names and
# **turning Python's `int` or tuple into something JavaScript can read** — a
# Python tuple crossing as-is becomes a proxy over there that is neither
# `typeof padding === "number"` nor an array.

def _pad_arg(padding):
    return padding if isinstance(padding, int) else _js_list(list(padding))


def _pad_layer(name):
    """**`value` belongs to `ConstantPad*` and this gave it to all fifteen.**

    The twelve took it and dropped it — borch.ts's constructors have no seat for
    it, so it never crossed. Nothing diverged, which is why it lasted: an argument
    accepted and discarded looks exactly like an argument honoured until somebody
    checks the answer. torch refuses `ZeroPad2d(1, 9.0)` outright.

    Two shapes rather than one with a branch, so the signature itself says which
    layers have a value.
    """
    if name.startswith("ConstantPad"):
        def make(padding, value=0.0):
            return _layer(name, _pad_arg(padding), float(value))
    else:
        def make(padding):
            return _layer(name, _pad_arg(padding))
    make.__name__ = name
    return make


for _dims in (1, 2, 3):
    for _kind in ("Reflection", "Replication", "Circular", "Zero", "Constant"):
        _pad_name = f"{_kind}Pad{_dims}d"
        globals()[_pad_name] = _pad_layer(_pad_name)


def _batchnorm(name):
    """torch's list. `affine` and `bias` are the two halves of the same idea —
    `affine=False` is no learnable scale or shift at all, `bias=False` keeps the
    scale and drops the shift. Neither crossed before, so a checkpoint from a torch
    layer built either way could not be read here in strict mode.

    **The per-dimension name is borch.ts's now.** All three stood up `BatchNormND`,
    which is what `describe` printed — five golden cases said `BatchNormND(4, …)`
    where torch says `BatchNorm2d(4, …)`. The reason written on `_layer` for keeping
    one name, *borch.ts has no per-dimension names*, was true and stopped being true:
    `BatchNorm1d`, `2d` and `3d` are empty subclasses over there and `describe` reads
    `this.constructor.name`, so standing the right one up is the whole fix.
    """
    def make(n, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
             device=None, dtype=None, *, bias=True):
        from borch._base import _only_cpu, _unsupported

        # **The pair is carried in order to answer it**, and the note on
        # `TransformerEncoderLayer` — *no layer in this file takes them* — is one
        # exception shorter for it. Not taking them is not the same as answering
        # them: `BatchNorm2d(3, device="cpu")` came back *unexpected keyword
        # argument*, which is the screen a typo gives.
        #
        # **And then it refused `"cpu"`**, which is this library's own device — the
        # core did the same until `_only_cpu` gave the argument a rule instead of a
        # habit. `dtype` still stops at anything: a parameter here is float32.
        _only_cpu(f"nn.{name}", device)
        if dtype is not None:
            _unsupported(f"nn.{name}(dtype=…)")
        # **`track_running_stats=False` was refused here and `True` was hard-wired
        # into the call below**, which is the shape this file keeps finding: the seat
        # existed, the value never crossed. borch.ts answers the flag now — no buffers
        # and the batch's own statistics in both modes — so the argument goes over.
        #
        # **`bias` sits eighth over there**, behind the `device` and `dtype` seats
        # borch.ts carries in order to refuse. It used to be handed over sixth, which
        # is `device` — and it only became wrong the day that class grew the pair, so
        # the two `None`s are what keep this call at the argument it means.
        return _layer(name, n, eps, momentum, bool(affine),
                      bool(track_running_stats), None, None, bool(bias))
    make.__name__ = name
    return make


BatchNorm1d = _batchnorm("BatchNorm1d")
BatchNorm2d = _batchnorm("BatchNorm2d")
BatchNorm3d = _batchnorm("BatchNorm3d")


def _maybe_in_place(layer, inplace):
    """The layer as built, or the same layer with its answer written back.

    `_in_place` is defined further down and is looked up when the layer runs, not
    when this returns — so the ordering here is the file's, not a dependency.
    """
    if not inplace:
        return layer
    return _Wrap(lambda x: _in_place(x, layer(x)))


def ReLU(inplace=False):
    return _maybe_in_place(_layer("ReLU"), inplace)


def _max_pool_layer(name, wide=False):
    """**It stands borch.ts's layer up rather than wrapping the tensor method.**

    The same move `AvgPool1d` below records, and the same reason: wrapping the method
    reaches a function that takes a kernel and a stride and nothing else, so torch's
    other four arguments have no seat and a `_Wrap` has no name to print. Four golden
    cases came back `<borch_webgpu._nn._Wrap object at …>` where torch prints
    `MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)`.

    **All six are real now.** `dilation` was the last refused — the shader over there
    had no step between the cells one window reads — and the kernel takes one; at 1 the
    generated source is the line it always was. `padding` and `ceil_mode` went the same
    way earlier, when the maximum's kernel grew the guard the average had.

    `return_indices` still keeps the Python path: it hands back a pair, and the layer
    that consumes those positions reads them from here. That path used to refuse the
    same three, because its window was an interval rather than a list of positions;
    `_pool_with_indices` says what changed.
    """
    def make(kernel_size=2, stride=None, padding=0, dilation=1,
             return_indices=False, ceil_mode=False):
        if return_indices:
            return _Wrap(lambda x: _pool_with_indices(
                x, kernel_size, stride, padding, dilation, ceil_mode))
        # **All six, in torch's order**, because that is the order over there now.
        # Written as four — kernel, stride, padding, ceil — `ceil_mode` landed in
        # `dilation`'s seat and every 1-D and 3-D max pool stopped with a refusal
        # naming an argument nobody had passed. The same seat, on the same layers,
        # that the signature axis had just caught on the borch.ts side.
        return _layer(name, kernel_size, stride, padding, dilation, False,
                      ceil_mode)
    make.__name__ = name
    return make


MaxPool1d = _max_pool_layer("MaxPool1d")
MaxPool2d = _max_pool_layer("MaxPool2d", wide=True)
MaxPool3d = _max_pool_layer("MaxPool3d")


def _spread(v, n):
    """A single number spreads across the dimensions; a list is taken as it is.
    The same rule as the core's name of the same spelling."""
    return (v,) * n if isinstance(v, int) else tuple(v)


class _MaxUnpool(_Wrap):
    """Put values back into the positions `MaxPool` chose.

    **`forward` takes two arguments** — the values and the index map. Shaped
    unlike the other layers, it does not drop into a `Sequential`, and torch is
    the same. The index map has to travel with the values; hidden inside the
    layer, using the same layer twice means using somebody else's map.
    """

    def __init__(self, dim, kernel_size, stride=None, padding=0):
        super().__init__(lambda x, indices, output_size=None: _unpool(
            x, indices, kernel_size, stride, padding, output_size))
        self.dim = dim
        # **Held unrolled per dimension** — torch holds it that way and `repr`
        # prints that tuple verbatim.
        self.kernel_size = _spread(kernel_size, dim)
        self.stride = _spread(kernel_size if stride is None else stride, dim)
        self.padding = _spread(padding, dim)

    def __repr__(self):
        return (f"MaxUnpool{self.dim}d(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")


def _unpool_layer(dim):
    def make(kernel_size, stride=None, padding=0):
        return _MaxUnpool(dim, kernel_size, stride, padding)
    return make


MaxUnpool1d = _unpool_layer(1)
MaxUnpool2d = _unpool_layer(2)
MaxUnpool3d = _unpool_layer(3)


class _FractionalMaxPool(_Wrap):
    """**The `repr` is empty** — torch's `extra_repr` produces nothing
    (measured)."""

    def __init__(self, dim, kernel_size, output_size=None, output_ratio=None,
                 return_indices=False, _random_samples=None):
        # **Only one of the two** — torch stops in the constructor. The inner
        # `_fractional` checks the same thing but stops **when called**, which
        # separates the line that built the layer from the line that stops.
        if (output_size is None) == (output_ratio is None):
            raise ValueError(
                "FractionalMaxPool takes either output_size or output_ratio, not both.")
        fn = _fractional(dim)
        super().__init__(lambda x: fn(x, kernel_size, output_size, output_ratio,
                                      return_indices, _random_samples))
        self.dim = dim

    def __repr__(self):
        return f"FractionalMaxPool{self.dim}d()"


class _CTCLoss(_Wrap):
    """`forward` takes four arguments — log probabilities, targets, and two
    lengths."""

    def __init__(self, blank=0, reduction="mean", zero_infinity=False):
        super().__init__(lambda lp, t, il, tl: _ctc_loss(
            lp, t, il, tl, blank, reduction, zero_infinity))

    def __repr__(self):
        return "CTCLoss()"


def CTCLoss(*args, **kw):
    return _CTCLoss(*args, **kw)


def FractionalMaxPool2d(*args, **kw):
    return _FractionalMaxPool(2, *args, **kw)


def FractionalMaxPool3d(*args, **kw):
    return _FractionalMaxPool(3, *args, **kw)


def Flatten(start_dim=1, end_dim=-1):
    from ._ops import flatten
    return _Wrap(lambda x: flatten(x, start_dim, end_dim),
                 f"Flatten(start_dim={start_dim}, end_dim={end_dim})")


def Identity(*args, **kw):
    """torch accepts any arguments at all (measured) — `Identity(64,
    unused=True)` runs.

    **The `**kw` is the point here, not an oversight.** It was swept out with
    ninety-four others that took it and never looked at it, and this is the one where
    swallowing *is* the contract: torch declares `Identity(*args, **kwargs)` and the
    golden case is named after it. Restoring it is what the case said to do, and the
    sweep is why anybody read the case again."""
    return _layer("Identity")


def _pool_layer(kind, adaptive, shown=None):
    def make(size, stride=None, return_indices=False):
        n = size[0] if isinstance(size, (list, tuple)) else size
        fn = _pool_fn(kind, adaptive)
        return _Wrap(lambda x: fn(x, n, stride, return_indices=return_indices),
                     None if shown is None
                     else f"{shown}(output_size={_size_repr(size)})")
    return make


def _size_repr(size):
    """torch's `_AdaptiveAvgPoolNd.extra_repr` prints **the argument as given** — a
    number stays a number and a pair stays a pair, because the two are different
    layers and printing `(1, 1)` for `1` hides which one was built."""
    return repr(tuple(size)) if isinstance(size, (list, tuple)) else repr(size)


def _adaptive_layer(name, indices):
    """The adaptive poolings borch.ts has as layers. As `_max_pool_layer` above.

    All four average and maximum names are over there now — `AdaptiveAvgPool2d` was the
    one missing, on the same tensor method as its siblings, and stayed on the wrapped
    method here until it was written.
    """
    def make(output_size, return_indices=False):
        # **The pair keeps the Python path**, as `_max_pool_layer` above. borch.ts's
        # layer hands back `{values, indices}` and `Module.__call__` wraps that as one
        # tensor, so `pool(x)[1]` indexed the values — a case that had been passing
        # said `index 1 is out of bounds` the moment this was repointed.
        if indices and return_indices:
            n = (output_size[0] if isinstance(output_size, (list, tuple))
                 else output_size)
            return _Wrap(lambda x: _pool_with_indices(x, n, None, adaptive=True))
        # **A pair has to cross as a real array.** Passed as a Python tuple it arrives as a
        # borrowed proxy that Pyodide destroys after the call, and the layer keeps it for
        # `describe()` — "This borrowed proxy was automatically destroyed" at the repr.
        # Measured on `AdaptiveAvgPool2d((2, 3))`; the 1-D and 3-D goldens use integers.
        if isinstance(output_size, (list, tuple)):
            output_size = _js_list(output_size)
        return _layer(name, output_size)
    make.__name__ = name
    return make


AdaptiveAvgPool2d = _adaptive_layer("AdaptiveAvgPool2d", False)
AdaptiveAvgPool1d = _adaptive_layer("AdaptiveAvgPool1d", False)
AdaptiveAvgPool3d = _adaptive_layer("AdaptiveAvgPool3d", False)
AdaptiveMaxPool1d = _adaptive_layer("AdaptiveMaxPool1d", True)
AdaptiveMaxPool2d = _adaptive_layer("AdaptiveMaxPool2d", True)
AdaptiveMaxPool3d = _adaptive_layer("AdaptiveMaxPool3d", True)
def AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False,
              count_include_pad=True):
    """**It stands borch.ts's layer up rather than calling the functional path.**

    It used to be `_pool_layer("avg", False)`, which reaches `avg_pool1d` and takes a
    kernel and a stride and nothing else. The four arguments below live on the layer,
    so the functional route could not carry them — and forwarding a padding into a
    function with no seat for it drops it silently.

    `divisor_override` is absent because torch gives it to the 2-D and 3-D forms only.
    """
    return _layer("AvgPool1d", kernel_size, stride, padding, ceil_mode,
                  count_include_pad)


def AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False,
              count_include_pad=True, divisor_override=None):
    return _layer("AvgPool3d", kernel_size, stride, padding, ceil_mode,
                  count_include_pad, divisor_override)


def _lp_pool_layer(name):
    """As `_batchnorm` — `LPPool2d` and `LPPool3d` are subclasses over there and
    `describe` reads the constructor's name, so all three stood up `LPPool1d` and
    printed it."""
    def make(norm_type, kernel_size, stride=None, ceil_mode=False):
        return _layer(name, norm_type, kernel_size, stride, ceil_mode)
    make.__name__ = name
    return make


LPPool1d = _lp_pool_layer("LPPool1d")
LPPool2d = _lp_pool_layer("LPPool2d")
LPPool3d = _lp_pool_layer("LPPool3d")


def AvgPool2d(kernel_size=2, stride=None, padding=0, ceil_mode=False,
              count_include_pad=True, divisor_override=None):
    """torch's list, as `AvgPool1d` and `AvgPool3d` above. It took two of six here
    because the layer over there took two."""
    return _layer("AvgPool2d", kernel_size, stride, padding, ceil_mode,
                  count_include_pad, divisor_override)


# ── eight now call borch.ts's layers directly ───────────────────────────────
#
# These used to be built here by wrapping a tensor method. That makes **two
# copies of the rule**, and the two did diverge: `Softmax()`'s default dimension
# is not `-1` but varies with rank (measured: 1→0, 2→1, 3→**0**, 4→1), and both
# sides had it as `-1`. Asked only at rank 2, `dim=1` and `dim=-1` are the same
# dimension and the divergence stays invisible.
#
# The layers exist over there now, so this only moves the names. One copy of the
# rule leaves nowhere to diverge.

def Softmax(dim=None):
    return _layer("Softmax", dim)


def LogSoftmax(dim=None):
    return _layer("LogSoftmax", dim)


def LeakyReLU(negative_slope=0.01, inplace=False):
    return _maybe_in_place(_layer("LeakyReLU", negative_slope), inplace)


def ELU(alpha=1.0, inplace=False):
    """**Takes an alpha.** It did not, and `nn.ELU(0.5)` stopped on that line."""
    return _maybe_in_place(_layer("ELU", alpha), inplace)


# **`SiLU` sat here alone and took no `inplace`**, while its four siblings below —
# `Hardsigmoid`, `Hardswish`, `Mish`, `ReLU6` — share `_unary_layer` and have one.
# torch gives it one and so does the core, so `nn.SiLU(inplace=True)` stopped on an
# argument count in the binding alone. Moved down to join them rather than given a
# sixth spelling of the same three lines.


def GELU(approximate="none"):
    """**`approximate='tanh'` is a different expression** — not merely accepted,
    the values differ."""
    return _layer("GELU", approximate)


def Sigmoid():
    return _layer("Sigmoid")


def Tanh():
    return _layer("Tanh")


# ── activation layers. Each wraps one borch.ts method. ──────────────────────
#
# The ones without arguments call the `unary` table directly; the ones with
# arguments go to a kernel with that argument baked in as a constant — either
# way Python only moves the name.

def _in_place(x, out):
    """`out`'s values written into `x`'s buffer, handing back **`x` itself.**

    What `inplace=True` buys is not the value — `ReLU(inplace=True)(x)` returns
    exactly what `ReLU()(x)` returns — but that the caller's tensor moved and the
    thing returned is the caller's tensor. So the flag is honoured through the same
    `_write_back` the underscore methods use rather than by returning a new tensor,
    which would pass a value comparison and fail the only thing the flag is for.

    The leaf refusal comes with it. torch stops there, the core stops there, and
    letting it through means a backward pass reads a value that has already moved.
    """
    t = wrap(handle(x)) if not hasattr(x, "_write_back") else x
    t._refuse_inplace_on_leaf("inplace")
    return t._write_back(out)


def _unary_layer(name):
    def build(inplace=False, n=name):
        def run(x):
            out = wrap(handle(x).unary(n))
            return _in_place(x, out) if inplace else out
        return _Wrap(run)
    return build


Hardsigmoid = _unary_layer("hardsigmoid")
Hardswish = _unary_layer("hardswish")
Mish = _unary_layer("mish")
ReLU6 = _unary_layer("relu6")
SELU = _unary_layer("selu")
SiLU = _unary_layer("silu")


# **These three take no `inplace` and torch gives them none either.** Sharing
# `_unary_layer` would have handed them one for free, which is the mirror of the
# fault this repository keeps finding: an argument accepted where the authority
# declines misleads as much as one accepted and inert.
def _plain_unary_layer(name):
    return lambda: _Wrap(lambda x, n=name: wrap(handle(x).unary(n)))


LogSigmoid = _plain_unary_layer("logsigmoid")
Softsign = _plain_unary_layer("softsign")
Tanhshrink = _plain_unary_layer("tanhshrink")


def CELU(alpha=1.0, inplace=False):
    def run(x):
        out = wrap(handle(x).celu(alpha))
        return _in_place(x, out) if inplace else out
    return _Wrap(run)


def Hardshrink(lambd=0.5):
    return _Wrap(lambda x: wrap(handle(x).hardshrink(lambd)))


def Softshrink(lambd=0.5):
    return _Wrap(lambda x: wrap(handle(x).softshrink(lambd)))


def Hardtanh(min_val=-1.0, max_val=1.0, inplace=False):
    def run(x):
        out = wrap(handle(x).hardtanh(min_val, max_val))
        return _in_place(x, out) if inplace else out
    return _Wrap(run)


def Softplus(beta=1.0, threshold=20.0):
    return _Wrap(lambda x: wrap(handle(x).softplus(beta, threshold)))


def Threshold(threshold, value):
    return _Wrap(lambda x: wrap(handle(x).threshold(threshold, value)))


def Softmin(dim=-1):
    return _Wrap(lambda x: wrap(handle(x).softmin(dim)))


def GLU(dim=-1):
    return _Wrap(lambda x: wrap(handle(x).glu(dim)))


class PReLU(Module):
    """The negative slope is **learned.** The only one in this family with a
    parameter.

    That is why it is a `Module` rather than a `_Wrap` — `weight` has to appear
    in `named_parameters`, and that name becomes the `state_dict` key.
    """

    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        import numpy as _np

        self.weight = Parameter(_np.full(num_parameters, init, dtype=_np.float32))

    def forward(self, x):
        return wrap(handle(x).prelu(handle(self.weight)))

    def extra_repr(self):
        return f"num_parameters={int(handle(self.weight).size)}"


class GroupNorm(Module):
    """Normalise with the channels gathered into groups. It carries weights, so
    a `Module` rather than a `_Wrap`."""

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, *,
                 bias=True):
        """`bias=False` keeps the scale and drops the shift — torch's, and the
        half of `affine` that was missing on every normalisation layer here."""
        super().__init__()
        import numpy as _np

        self.num_groups, self.eps = num_groups, eps
        # **Stored because the repr prints them.** `num_channels` was readable off
        # the weight and unreadable with `affine=False`, and `affine` was not kept
        # at all — two golden cases printed `GroupNorm()`.
        self.num_channels, self.affine = num_channels, affine
        self.weight = self.bias = None
        if affine:
            self.weight = Parameter(_np.ones(num_channels, dtype=_np.float32))
            if bias:
                self.bias = Parameter(_np.zeros(num_channels, dtype=_np.float32))

    def extra_repr(self):
        from borch._nn import _group_norm_extra_repr
        return _group_norm_extra_repr(self)

    def forward(self, x):
        h = handle(x)
        out = wrap(h.groupNorm(self.num_groups, self.eps))
        weight = getattr(self, "weight", None)
        shift = getattr(self, "bias", None)
        width = int(handle(weight if weight is not None else shift).size) \
            if (weight is not None or shift is not None) else 0
        shape = [1, width] + [1] * (len(h.shape) - 2)
        if weight is not None:
            out = out * weight.reshape(*shape)
        return out if shift is None else out + shift.reshape(*shape)


def _instancenorm(name):
    """Per sample and per channel. **The default is `affine=False`**, as it is in
    torch.

    **It used to be a Python `Module` holding its own parameters**, and the reason
    was sound at the time: an earlier `**kw` had swallowed `affine` whole, so a layer
    built with `affine=True` ran with no parameters at all. Standing borch.ts's class
    up instead fixes that as thoroughly and keeps one implementation — which is what
    `track_running_stats` made necessary, because the buffers live over there and a
    Python copy of them would be a third place the same arithmetic is written.

    The refusal that stood here said the `state_dict` keys would be right and the
    values wrong. That was true of registering buffers nothing reads; borch.ts's
    `instanceNorm` reads them now.
    """
    def make(n=0, eps=1e-5, momentum=0.1, affine=False,
             track_running_stats=False, device=None, dtype=None, *, bias=True):
        from borch._base import _unsupported

        for what, given in (("device", device), ("dtype", dtype)):
            if given is not None:
                _unsupported(f"nn.{name}({what}=…)")
        # `bias` sits eighth over there, behind the two seats borch.ts carries in
        # order to refuse — the same shift `_batchnorm` writes out above.
        return _layer(name, n, eps, momentum, bool(affine),
                      bool(track_running_stats), None, None, bool(bias))
    make.__name__ = name
    return make


InstanceNorm1d = _instancenorm("InstanceNorm1d")
InstanceNorm2d = _instancenorm("InstanceNorm2d")
InstanceNorm3d = _instancenorm("InstanceNorm3d")


class RMSNorm(Module):
    """**The mean is not subtracted.** That is the only difference from
    `LayerNorm`."""

    def __init__(self, normalized_shape, eps=None, elementwise_affine=True):
        super().__init__()
        import numpy as _np

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.dims = len(normalized_shape)
        # **`eps` was taken here and never handed on.** `Tensor.rmsNorm(dims, eps)`
        # over there has carried it from the start — borch.ts's own `RMSNorm` note
        # records fixing exactly this defect in exactly this place — and this class
        # is a second copy of the layer that repeated it. `RMSNorm(4, eps=10.0)` set
        # nothing and answered as if it had.
        self.eps = eps
        if elementwise_affine:
            self.weight = Parameter(_np.ones(normalized_shape, dtype=_np.float32))

    def forward(self, x):
        # torch's `eps=None` means *use the dtype's epsilon*, and there is one dtype
        # here — so `None` and the kernel's own default are the same answer, which is
        # why it is passed straight through rather than filled in.
        out = wrap(handle(x).rmsNorm(self.dims, self.eps))
        return out if getattr(self, "weight", None) is None else out * self.weight


class _ConvTransposeND(Module):
    """Transposed convolution. **The weights are `(in, out, …)`** — reversed
    from `Conv2d`'s."""

    nd = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode="zeros"):
        """**torch's whole list, and it took six of ten.** `output_padding`,
        `groups` and `dilation` were absent while the tensor method over there has
        taken all three from the start — so `ConvTranspose2d(2, 3, 3, stride=2,
        output_padding=1)` stopped with a `TypeError` naming a private class, and
        the fix was to pass on what was already being received one layer down.

        They are stored as well as passed, because the repr prints them: the layer
        printed `ConvTranspose1d()` where torch prints its channels and kernel.
        """
        super().__init__()
        import numpy as _np

        nd = type(self).nd
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.output_padding, self.groups, self.dilation = (output_padding, groups,
                                                           dilation)
        self.padding_mode = padding_mode
        self.bias = None
        shape = (in_channels, out_channels // groups) + (kernel_size,) * nd
        bound = 1.0 / ((out_channels // groups) * kernel_size ** nd) ** 0.5
        rng = _np.random.default_rng(0)
        self.weight = Parameter(rng.uniform(-bound, bound, shape).astype(_np.float32))
        if bias:
            self.bias = Parameter(
                rng.uniform(-bound, bound, out_channels).astype(_np.float32))

    def forward(self, x):
        b = self.bias
        return wrap(handle(x).convTransposeND(
            handle(self.weight), handle(b) if b is not None else None,
            self.stride, self.padding, self.output_padding, self.groups,
            self.dilation))

    def extra_repr(self):
        from borch._nn import _conv_extra_repr
        return _conv_extra_repr(type(self).nd, transposed=True)(self)


class ConvTranspose1d(_ConvTransposeND):
    nd = 1


class ConvTranspose2d(_ConvTransposeND):
    nd = 2


class ConvTranspose3d(_ConvTransposeND):
    nd = 3


class Dropout(Module):
    """Drops only while training. **It is a `Module` rather than a `_Wrap`
    because of the mode** — a `_Wrap` does not carry `training`, so it keeps
    dropping after `eval()`."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True

    def train(self, mode=True):
        self.training = mode
        return self

    def forward(self, x):
        return wrap(handle(x).dropout(self.p, self.training))


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
    """**`normalized_shape` decides how many dimensions are folded** — it used
    to be discarded entirely.

    Measured only with `LayerNorm(4)` the divergence is invisible: the answer
    matches folding the last dimension alone. On top of that it was a `_Wrap`,
    so **it had no parameters** — torch's default learns `weight` and `bias`, and
    without them the layer quietly does not learn.
    """
    shape = ([normalized_shape] if isinstance(normalized_shape, int)
             else list(normalized_shape))
    return _layer("LayerNorm", _arg(shape), eps, elementwise_affine, bias)


def Unflatten(dim, sizes):
    return _layer("Unflatten", dim, _arg(list(sizes)))


def Upsample(size=None, scale_factor=None, mode="nearest", align_corners=None,
             recompute_scale_factor=None):
    """**The first position is `size`**, as it is in torch. It used to be the
    scale factor, and `mode` was accepted and unused — asking for bilinear
    quietly produced nearest.

    `recompute_scale_factor` is torch's fifth seat and it was missing here while
    borch.ts grew it: with a fractional factor torch either samples at the number it
    was given or derives the scale back from the floored output size, and those are
    different values at the same shape."""
    return _layer("Upsample", _arg(size) if size is not None else None,
                  scale_factor, mode, align_corners, recompute_scale_factor)


# **The reduction is part of the loss — it has to exist on the layer side too.**
# Fixing only the function side leaves `nn.MSELoss(reduction="sum")` stopping
# with a `TypeError`. It was in that state, and textbook code uses the layers
# more than the functions.
#
# **A reason that was true when it was written, and is why nobody looked.** It read:
#
#     `NLLLoss` and `CrossEntropyLoss` are not here yet — `nllLoss` and
#     `crossEntropy` in borch.ts produce a scalar only, so `none` cannot be
#     built here. Rather than invent what is missing, it stops.
#
# borch.ts has `ignoreIndex`, `reduction` and `labelSmoothing`, all implemented, and
# `reduceIgnoring` was written specifically so that `"none"` is buildable. Nothing was
# missing. The arguments were being handed across in the wrong seats — `reduction`
# into `ignoreIndex` — and the paragraph above explained the resulting silence as a
# decision.
#
# The failure mode is worth more than the fix: a stale reason is worse than no reason,
# because no reason invites a check. This one survived every sweep of the file.
# **`_no_class_weights` stood here and is gone**, and it is the same shape of thing
# as the stale paragraph above it. Its reason: `weight` and `pos_weight` accepted and
# unused make the loss quietly a different one, because `mean` divides by the sum of
# the weights rather than the sample count. That was true and it named the way out —
# divide by the sum of the weights. borch.ts does exactly that now, so the four losses
# below hand the tensors across instead of stopping, and nothing here implements the
# arithmetic a second time.


def _legacy_reduction(size_average, reduce, reduction):
    """torch's deprecated `size_average`/`reduce`, folded into a `reduction`.

    **The same rule as the core's helper of the same name, and it has to be here
    too.** The seven layers below call a `Tensor` method rather than standing
    borch.ts's layer up, so borch.ts's own fold never runs for them — the arguments
    would be taken and dropped, which is the shape of defect this whole file keeps
    finding.

    `reduce=False` gives `none`, else `size_average=False` gives `sum`, else `mean`.
    **The pair beats `reduction`**, which is torch's behaviour and the opposite of
    what the word *deprecated* suggests.
    """
    if size_average is None and reduce is None:
        return reduction
    got = ("none" if reduce is False
           else "sum" if size_average is False
           else "mean")
    # torch's own wording, so a caller who meets it can search for the same
    # sentence in torch's issues. It was absent here while the core and torch
    # both warned — the fold was right and the *telling* was missing, which is
    # how a deprecated argument stays in a codebase forever.
    _warnings.warn(
        f"size_average and reduce args will be deprecated, "
        f"please use reduction='{got}' instead.",
        UserWarning, stacklevel=3)
    return got


def L1Loss(size_average=None, reduce=None, reduction="mean"):
    reduction = _legacy_reduction(size_average, reduce, reduction)
    return _Wrap(lambda a, b: wrap(handle(a).l1Loss(handle(b), reduction)))


def MSELoss(size_average=None, reduce=None, reduction="mean"):
    reduction = _legacy_reduction(size_average, reduce, reduction)
    return _Wrap(lambda a, b: wrap(handle(a).mseLoss(handle(b), reduction)))


def SmoothL1Loss(size_average=None, reduce=None, reduction="mean", beta=1.0):
    # torch's four in torch's order, and `beta` last. See `borch/_nn.py`.
    reduction = _legacy_reduction(size_average, reduce, reduction)
    return _Wrap(lambda a, b: wrap(
        handle(a).smoothL1Loss(handle(b), beta, reduction)))


def NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None,
            reduction="mean"):
    """torch's order — and **the deprecated pair is not adjacent here**, with
    `ignore_index` between the two."""
    reduction = _legacy_reduction(size_average, reduce, reduction)
    w = handle(weight) if weight is not None else None
    return _Wrap(lambda a, b: wrap(
        handle(a).nllLoss(handle(b), int(ignore_index), reduction, w)))


def BCELoss(weight=None, size_average=None, reduce=None, reduction="mean"):
    """torch's order. **No `pos_weight`** — that belongs to the logits form alone,
    and offering it here would be an argument torch does not have. The core says the
    same at the same place."""
    reduction = _legacy_reduction(size_average, reduce, reduction)
    w = handle(weight) if weight is not None else None
    return _Wrap(lambda a, b: wrap(handle(a).bce(handle(b), reduction, w)))


def BCEWithLogitsLoss(weight=None, size_average=None, reduce=None,
                      reduction="mean", pos_weight=None):
    reduction = _legacy_reduction(size_average, reduce, reduction)
    w = handle(weight) if weight is not None else None
    pw = handle(pos_weight) if pos_weight is not None else None
    return _Wrap(lambda a, b: wrap(
        handle(a).bceWithLogits(handle(b), reduction, w, pw)))


def CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                     reduce=None, reduction="mean", label_smoothing=0.0):
    """torch's order — `ignore_index` sits between the deprecated pair."""
    reduction = _legacy_reduction(size_average, reduce, reduction)
    w = handle(weight) if weight is not None else None
    return _Wrap(lambda a, b: wrap(handle(a).crossEntropy(
        handle(b), int(ignore_index), reduction, float(label_smoothing), w)))


class LinearCrossEntropyOptions:
    """torch's chunking configuration, held and read by nothing.

    Every field is a memory strategy. Measured in real torch, four of them answer
    within 4.8e-7 of the default — float32 noise — so accepting it and ignoring it give
    the same number, which is what makes carrying it honest rather than the
    *accepted and never read* defect. The core's class says the same at more length.
    """

    def __init__(self, allow_retain_graph=False, batch_chunk_size=None,
                 chunking_method="auto", acc_policy="auto", acc_dtype=None):
        self.allow_retain_graph = allow_retain_graph
        self.batch_chunk_size = batch_chunk_size
        self.chunking_method = chunking_method
        self.acc_policy = acc_policy
        self.acc_dtype = acc_dtype

    def __repr__(self):
        return (f"{type(self).__name__}("
                f"allow_retain_graph={self.allow_retain_graph}, "
                f"batch_chunk_size={self.batch_chunk_size}, "
                f"chunking_method={self.chunking_method!r}, "
                f"acc_policy={self.acc_policy!r}, acc_dtype={self.acc_dtype})")


def LinearCrossEntropyLoss(in_features, num_classes, *, out_features=(),
                           bias=False, device=None, dtype=None, reduction="mean",
                           weight=None, ignore_index=None, label_smoothing=0.0,
                           options=None):
    """A `Linear` and a cross entropy, fused — **and the fusion is about memory.**

    borch.ts carries the class, so this forwards to it rather than assembling the pair
    in Python. Assembled here the golden would go green and the name would not exist
    for anyone holding borch.ts, which is the seam this repository has fallen into
    seven times.

    `out_features` is refused over there for the reason the core gives: torch reads the
    logits as `(N, C, *out_features)` and neither implementation's cross entropy takes
    a target of that rank.
    """
    from borch._base import _only_cpu, _unsupported

    # **The pair is carried in order to answer it.** Not taking them is not the same as
    # answering them — `LinearCrossEntropyLoss(5, 4, device="cpu")` would come back
    # *unexpected keyword argument*, which is the screen a typo gives.
    _only_cpu("nn.LinearCrossEntropyLoss", device)
    if dtype is not None:
        _unsupported("nn.LinearCrossEntropyLoss(dtype=…)")
    # **`out_features` and `options` go across rather than being written as constants.**
    # Hard-wired here they were two seats that took a value and sent something else,
    # which `test_binding_arguments.py` caught by reading the call — the shape that file
    # exists for. `out_features` is refused over there and `options` is inert on both
    # sides, and neither of those is a reason for the value not to travel.
    return Module(_ts.nn.LinearCrossEntropyLoss.new(
        int(in_features), int(num_classes), _js_list([int(v) for v in out_features]),
        bool(bias), None, None,
        reduction, handle(weight) if weight is not None else None,
        None if ignore_index is None else int(ignore_index),
        float(label_smoothing), options))


class _Recurrent(Module):
    """**torch's recurrent networks hand back a tuple** — `(output, final
    state)`.

    borch.ts's `forward` gives the output only and `run()` gives all three. LSTM
    has two states (`h`, `c`), so `(output, (h, c))`; the rest are
    `(output, h)`. The shapes are matched too — torch's final state is
    `(layers, batch, hidden)`, one dimension more.
    """

    def __init__(self, module, batch_first=False):
        super().__init__(module)
        object.__setattr__(self, "_batch_first", batch_first)

    def __call__(self, x, *rest):
        # **`batch_first` decides which of the first two axes is which.** borch.ts's
        # `run` takes the length first, as torch's does, so the flag is honoured by
        # turning the input on the way in and the output on the way back — the same
        # shape `_Attention` next door uses, and the same place it was being dropped.
        from ._ops import transpose as _t

        if self._batch_first:
            x = _t(x, 0, 1)
        got = self._m.run(handle(x))
        out, h = wrap(got.output), wrap(got.hidden)
        if self._batch_first:
            out = _t(out, 0, 1)
        # **Brought to three dimensions.** torch's final state is
        # `(layers, batch, hidden)`. Adding one unconditionally at first made it
        # four — already three, it is left alone.
        if h.ndim == 2:
            h = wrap(h._h.unsqueeze(0))
        # **`mode`, because torch calls it that** — this read `kind` until borch.ts
        # took torch's spelling, and a property that is not there reads as
        # `undefined`, which is not `"LSTM"`. Every recurrent case failed at once,
        # 6/6 under one prefix, and no type said a word on either side of the bridge.
        if self._m.mode == "LSTM":
            c = wrap(got.cell)
            if c.ndim == 2:
                c = wrap(c._h.unsqueeze(0))
            return out, (h, c)
        return out, h


def _recurrent(kind):
    def make(input_size, hidden_size, num_layers=1, *rest, **kw):
        """torch's ten. **This took two and swallowed the rest.**

        `GRU(3, 4, batch_first=True)` set nothing at all here and said nothing —
        `**kw` took the flag and dropped it, so the layer read `(length, batch, …)`
        while the caller had written `(batch, length, …)`. The golden agreed only
        because every case passes the default.

        **Five were refused and are forwarded now.** The reason was that borch.ts's
        `RNNBase` builds one unidirectional biased layer, so anything else had nowhere
        to go — true of that class, and that class is what changed.

        **The fourth positional is not the same argument on all three.** torch's `RNN`
        puts `nonlinearity` there and the other two put `bias`, which is why the tail
        is read per kind rather than by one signature: written as one, `RNN(3, 4, 2,
        "relu")` lands the string in `bias`, where it is true, and the layer computes
        `tanh` with biases and says nothing.
        """
        from borch._base import _unsupported

        names = (("nonlinearity", "bias", "batch_first", "dropout", "bidirectional",
                  "device", "dtype") if kind == "RNN" else
                 ("bias", "batch_first", "dropout", "bidirectional", "proj_size",
                  "device", "dtype"))
        got = dict(zip(names, rest))
        clash = [n for n in got if n in kw]
        if clash:
            raise TypeError(f"{kind}() got multiple values for argument '{clash[0]}'")
        got.update(kw)
        # **The three validations are torch's and they are checked here rather than
        # left to borch.ts.** Crossing the bridge turns a thrown `RangeError` into
        # something with a different type and a decorated message, and the wording is
        # what the cases freeze.
        if kind != "LSTM" and got.get("proj_size"):
            raise ValueError("proj_size argument is only supported for LSTM, "
                             "not RNN or GRU")
        if got.get("proj_size", 0) < 0:
            raise ValueError("proj_size should be a positive integer or zero "
                             "to disable projections")
        if got.get("proj_size", 0) >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")
        for what in ("device", "dtype"):
            if got.get(what) is not None:
                _unsupported(f"{kind}({what}=…) — one device, one precision")
        bias = got.get("bias", True)
        batch_first = got.get("batch_first", False)
        dropout = got.get("dropout", 0.0)
        bidirectional = got.get("bidirectional", False)
        proj_size = got.get("proj_size", 0)
        nonlinearity = got.get("nonlinearity", "tanh")
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be 'tanh' or 'relu'.")
        inp, hid = input_size, hidden_size
        # **`RNNBase` is the class's name now** — torch's, where borch.ts had
        # `Recurrent` alone. The alias still resolves at runtime, and
        # `test_binding_arguments.py` reads *source*: it looks for a `class` with
        # this name and found a `const`, so the call site had nothing to compare
        # against. Its own message says that is a call site with no rule.
        #
        # **`kind` leads, because torch puts `mode` first.** This read
        # `(inp, hid, kind)` until borch.ts took torch's order, and then every
        # recurrent case failed at once — six of six under one prefix, which is the
        # shape the runner calls *one cause rather than many*. The signature axis
        # could not have said so: it reads declarations, and this is a call.
        # **`nonlinearity` is set after the fact and not passed**, because torch's
        # `RNNBase` has no such seat — it carries the activation in the mode, and
        # borch.ts follows. Given as a twelfth argument the class was one longer than
        # torch's and the signature axis said so.
        built = _ts.nn.RNNBase.new(kind, inp, hid, int(num_layers), bool(bias),
                                   False, float(dropout), bool(bidirectional),
                                   int(proj_size))
        built.nonlinearity = nonlinearity
        return _Recurrent(built, batch_first)
    return make


RNN, LSTM, GRU = _recurrent("RNN"), _recurrent("LSTM"), _recurrent("GRU")


class _Attention(Module):
    """torch's attention takes three — `(query, key, value)` — and gives back
    `(output, weights)`.

    **The computation is `multi_head_attention_forward`'s.** It used to call
    borch.ts's `attend`, which is for self-attention only and puts the batch
    first, so the answers diverged wherever the three were given separately —
    the golden's "same answer as the layer" case caught it at a maximum
    difference of 1.26e-01.
    """

    def __init__(self, module, heads, batch_first=False):
        super().__init__(module)
        object.__setattr__(self, "_heads", heads)
        object.__setattr__(self, "_batch_first", batch_first)

    def __call__(self, q, k=None, v=None, attn_mask=None, need_weights=True,
                 key_padding_mask=None, average_attn_weights=True):
        from ._ops import transpose as _t
        k = q if k is None else k
        v = q if v is None else v
        # **The function form puts length first.** With `batch_first` it is
        # transposed here before crossing.
        if self._batch_first:
            q, k, v = (_t(t, 0, 1) for t in (q, k, v))
        # **Everything the layer holds crosses, including the empties.** Under
        # `kdim`/`vdim` the bundle is `null` and the three separate weights are what
        # exist; under `add_bias_kv` there are two more. Reading only `inWeight` and
        # `outWeight` was right while the rest could not exist.
        def held(name):
            got = getattr(self._m, name, None)
            return None if got is None else wrap(got)

        out, weights = _mha_forward(
            q, k, v, None, self._heads,
            held("inWeight"), held("inBias"),
            bias_k=held("biasK"), bias_v=held("biasV"),
            add_zero_attn=bool(self._m.addZeroAttn),
            out_proj_weight=wrap(self._m.outWeight),
            out_proj_bias=held("outBias"),
            attn_mask=attn_mask, need_weights=need_weights,
            key_padding_mask=key_padding_mask,
            use_separate_proj_weight=not self._m.qkvSameEmbedDim,
            q_proj_weight=held("qWeight"), k_proj_weight=held("kWeight"),
            v_proj_weight=held("vWeight"),
            average_attn_weights=average_attn_weights)
        return (_t(out, 0, 1) if self._batch_first else out), weights


def MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True,
                       add_bias_kv=False, add_zero_attn=False, kdim=None,
                       vdim=None, batch_first=False):
    """torch's nine, forwarded whole.

    **This took three and borch.ts now takes nine**, so six were accepted here and
    reached nothing — the shape JavaScript makes silent, since a surplus argument is
    discarded without a word. Forwarding them puts the refusals in one place:
    `add_bias_kv`, `add_zero_attn` and a `kdim`/`vdim` unlike the embedding stop in
    borch.ts's constructor with their own names, rather than being dropped here and
    stopping somewhere else or not at all.

    `batch_first` is passed twice on purpose — borch.ts holds it for its own
    `attend`, and `_Attention` below reads it because the computation here is
    `multi_head_attention_forward`'s rather than that method's.
    """
    return _Attention(
        _ts.nn.MultiheadAttention.new(embed_dim, num_heads, dropout, bias,
                                      add_bias_kv, add_zero_attn, kdim, vdim,
                                      batch_first),
        num_heads, batch_first)


# ── The transformer ───────────────────────────────────────────────────────
#
# **The third implementation, and it was missing again.** The core and borch.ts both
# grew these in the same hour and this file did not, so the four cases raised
# `AttributeError` while the other two sides were green. It is the second time in one
# session, and the reason is the same both times: nothing automated runs this file.
#
# torch's argument order, which is `(…, activation, layer_norm_eps, batch_first,
# norm_first, bias)` — the two middle seats were the other way round on all three
# sides until an hour ago.

def TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1,
                            activation="relu", layer_norm_eps=1e-5,
                            batch_first=False, norm_first=False, bias=True):
    # No `device`/`dtype` — no layer in this file takes them, because there is one
    # of each in a browser. The core carries the pair only to refuse it.
    return _layer("TransformerEncoderLayer", d_model, nhead, dim_feedforward,
                  float(dropout), activation, float(layer_norm_eps),
                  bool(batch_first), bool(norm_first), bool(bias))


def TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1,
                            activation="relu", layer_norm_eps=1e-5,
                            batch_first=False, norm_first=False, bias=True):
    return _layer("TransformerDecoderLayer", d_model, nhead, dim_feedforward,
                  float(dropout), activation, float(layer_norm_eps),
                  bool(batch_first), bool(norm_first), bool(bias))


def TransformerEncoder(encoder_layer, num_layers, norm=None,
                       enable_nested_tensor=True, mask_check=True):
    """**Both of torch's last two are accepted and change nothing**, as in the core:
    the first asks for a packed representation that does not exist here and the
    second guards a fast path that is not taken."""
    return _layer("TransformerEncoder", handle(encoder_layer), int(num_layers),
                  None if norm is None else handle(norm))


def TransformerDecoder(decoder_layer, num_layers, norm=None):
    return _layer("TransformerDecoder", handle(decoder_layer), int(num_layers),
                  None if norm is None else handle(norm))


def Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                dim_feedforward=2048, dropout=0.1, activation="relu",
                custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-5,
                batch_first=False, norm_first=False, bias=True):
    return _layer("Transformer", d_model, nhead, num_encoder_layers,
                  num_decoder_layers, dim_feedforward, float(dropout), activation,
                  None if custom_encoder is None else handle(custom_encoder),
                  None if custom_decoder is None else handle(custom_decoder),
                  float(layer_norm_eps), bool(batch_first), bool(norm_first),
                  bool(bias))


def _square_subsequent_mask(size):
    """`torch.nn.Transformer.generate_square_subsequent_mask` — a **float** mask
    whose upper triangle is −∞. It is *added* to the scores, which is why the
    masked positions are −∞ and not 0.

    **Hung on the function**, because every layer in this file is a factory
    function rather than a class, and a `staticmethod` has nowhere else to live.
    Two golden cases reach for it through `nn.Transformer.…` and both said
    `'function' object has no attribute` until it was attached.
    """
    return wrap(_ts.nn.Transformer.generateSquareSubsequentMask(int(size)))


Transformer.generate_square_subsequent_mask = _square_subsequent_mask
