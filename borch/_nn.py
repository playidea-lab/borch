"""A piece of borch, split out. __init__ gathers the public names."""

import collections as _collections
import inspect as _inspect
import math as _math

import numpy as _np

from ._tensor import (
    Tensor,
)
from ._base import (
    _DEFAULT_DTYPE, _like_torch, _math, _np, _unsupported,
)
from ._ops import (
    _Namespace, _gelu, _pool_all, _reduce, _renorm_rows, _rng, _spread, _wrap,
    adaptive_avg_pool1d,
    adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool1d,
    adaptive_max_pool2d, adaptive_max_pool3d, avg_pool1d, avg_pool2d, avg_pool3d,
    celu, lp_pool1d, lp_pool2d, lp_pool3d,
    conv1d,
    conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d,
    cosine_similarity, dropout, elu, embedding, gelu, glu, group_norm, hardshrink,
    hardsigmoid, hardswish, hardtanh, instance_norm, interpolate, l1_loss, layer_norm,
    leaky_relu, log_softmax, logsigmoid, max_pool1d, max_pool2d, max_pool3d, mish,
    nll_loss, no_grad, norm, normalize, pad, prelu, relu, relu6, rms_norm, selu,
    sigmoid, silu, smooth_l1_loss, softmax, softmin, softplus, softshrink, softsign,
    cat, stack, tanh, tanhshrink, zeros,
    # Losses and distances.
    cosine_embedding_loss, gaussian_nll_loss, hinge_embedding_loss, huber_loss,
    kl_div, margin_ranking_loss, multi_margin_loss, multilabel_margin_loss,
    multilabel_soft_margin_loss, pairwise_distance, pdist, poisson_nll_loss,
    soft_margin_loss, triplet_margin_loss, triplet_margin_with_distance_loss,
    # Rearrangement and channel-wise dropout.
    alpha_dropout, channel_shuffle, dropout1d, dropout2d, dropout3d,
    feature_alpha_dropout, pixel_shuffle, pixel_unshuffle,
    # Unfolding windows and the rest. **`unfold_im2col` carries a split name** —
    # it does a different job from `Tensor.unfold`, so they cannot share the
    # module slot (see the comment over there).
    amax, bilinear, fold, local_response_norm, rrelu, unfold_im2col,
    # The poolings that also give the winning positions, and the partner that
    # puts values back at them.
    adaptive_max_pool1d_with_indices, adaptive_max_pool2d_with_indices,
    adaptive_max_pool3d_with_indices, max_pool1d_with_indices,
    max_pool2d_with_indices, max_pool3d_with_indices,
    max_unpool1d, max_unpool2d, max_unpool3d,
    fractional_max_pool2d, fractional_max_pool2d_with_indices,
    fractional_max_pool3d, fractional_max_pool3d_with_indices,
    ctc_loss,
    # In-place activations and `interpolate`'s old names.
    celu_, elu_, hardtanh_, leaky_relu_, relu_, rrelu_, selu_, threshold_,
    upsample, upsample_bilinear, upsample_nearest,
    # The functions layers sit on top of. One copy of each formula.
    batch_norm, embedding_bag, gumbel_softmax,
    # The spatial transformer's pair.
    affine_grid, grid_sample,
)
# **`_wrap` must not be imported inside a function.** Left that way once,
# `tests/test_alias.py` cleared `borch.*` out of `sys.modules` and that import ran
# again, building **a second copy** of `_ops`; there were then two `Tensor`
# classes and `isinstance` went wrong. The values came out as an object array and
# running that case on its own passed, so the cause looked a long way off — a
# late import costs that much in this repository.
#
# **`threshold` alone is imported under a different name.** The `Threshold` layer
# has an attribute called `self.threshold` (torch uses that name), and then the
# same name inside `forward` points at both the function and the attribute.
# Splitting the names means that place never arises.
from ._ops import threshold as threshold_fn

# ================================================================ nn

class _NN(_Namespace):
    pass


nn = _NN()


class Parameter(Tensor):
    """What training targets. The weights `nn.Linear` makes carry requires_grad
    from the start.

    **It takes `requires_grad`.** The argument exists in torch and was missing
    here, so `nn.Parameter(t, requires_grad=False)` stopped with a `TypeError` —
    the form textbooks use when making a frozen weight.
    """

    def __init__(self, data, requires_grad=True):
        super().__init__(
            data.data if isinstance(data, Tensor) else _np.asarray(data),
            requires_grad)


class Module:
    def __init__(self):
        self._modules = {}
        self._params = {}
        self._buffers = {}          # values that are not trained and are saved and restored (running_mean and the like)
        self.training = True

    def register_buffer(self, name, value, persistent=True):
        """torch's `register_buffer`. It goes into `state_dict` and is not
        trained.

        BatchNorm's running_mean goes in here — left out, saving and loading
        sends **evaluation mode back to the initial values**, and training looks
        fine while inference alone is wrong.

        With `persistent=False` it stays out of `state_dict`. This is where a
        value that can be rebuilt, like a cache, is kept out of the checkpoint,
        and **ignoring the argument makes the keys disagree with somebody else's
        checkpoint** — read strictly on the receiving side, that is a refusal.
        """
        self.__dict__.setdefault("_buffers", {})[name] = value
        if not persistent:
            self.__dict__.setdefault("_nonpersistent", set()).add(name)
        else:
            self.__dict__.get("_nonpersistent", set()).discard(name)
        object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_params", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        elif name in self.__dict__.get("_buffers", {}):
            self._buffers[name] = value
        object.__setattr__(self, name, value)

    def named_buffers(self, prefix="", persistent_only=False):
        """The buffers with their names. **`persistent_only` is turned on by the
        saving side alone.**

        torch's `named_buffers()` produces the `persistent=False` ones too — it is
        a list of buffers rather than a list of what gets saved. Keeping them out
        of the save is `state_dict`'s job, and filtering here by default mashes
        the two lists into one.
        """
        skip = self.__dict__.get("_nonpersistent", set()) if persistent_only else ()
        for n, b in self.__dict__.get("_buffers", {}).items():
            if n not in skip:
                yield (f"{prefix}{n}", b)
        for n, m in self._modules.items():
            yield from m.named_buffers(f"{prefix}{n}.", persistent_only)

    def buffers(self):
        for _, b in self.named_buffers():
            yield b

    def parameters(self):
        for p in self._params.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix=""):
        for n, p in self._params.items():
            yield (f"{prefix}{n}", p)
        for n, m in self._modules.items():
            yield from m.named_parameters(f"{prefix}{n}.")

    def modules(self):
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *a, **k):
        for x in list(a) + list(k.values()):
            if isinstance(x, str) and x != "cpu":
                _unsupported(f"device '{x}'")
        return self

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self):
        out = {name: Tensor(p.data.copy()) for name, p in self.named_parameters()}
        # **The save leaves out `persistent=False`.** That is what the argument
        # means.
        for name, buf in self.named_buffers(persistent_only=True):
            # A buffer arrives in two shapes — the numpy value a layer
            # registered and **a tensor the user put in through
            # `register_buffer`.** The save format is a tensor throughout.
            out[name] = Tensor(buf.data.copy() if isinstance(buf, Tensor)
                               else _np.array(buf, copy=True))
        return out

    def load_state_dict(self, state, strict=True):
        own = dict(self.named_parameters())
        # Every buffer can be received, and **only the saved ones may be
        # complained about as missing.**
        buffers = dict(self.named_buffers())
        saved = dict(self.named_buffers(persistent_only=True))
        missing = [k for k in list(own) + list(saved) if k not in state]
        unexpected = [k for k in state if k not in own and k not in buffers]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"state_dict does not match. missing: {missing}, unexpected: {unexpected}"
            )
        for name, value in state.items():
            if name in buffers:
                data = value.data if isinstance(value, Tensor) else _np.asarray(value)
                holder = self
                *path, leaf = name.split(".")
                for part in path:
                    holder = holder._modules[part]
                # **Restored in the shape it went in as.** Swapping a buffer
                # registered as a tensor for numpy makes a line like
                # `self.mask.unsqueeze(0)` blow up only after loading — code that
                # ran before the save stops running after it.
                keep = isinstance(buffers[name], Tensor)
                holder.register_buffer(
                    leaf,
                    Tensor(data.copy()) if keep
                    else (data.copy() if data.ndim else data.item()),
                    persistent=leaf not in holder.__dict__.get("_nonpersistent", set()))
                continue
            if name in own:
                target = own[name]
                incoming = value.data if isinstance(value, Tensor) else _np.asarray(value)
                if incoming.shape != target.data.shape:
                    raise RuntimeError(
                        f"{name} has a different shape: {incoming.shape} vs {tuple(target.data.shape)}"
                    )
                target._array = incoming.astype(target._array.dtype).copy()
        return self

    def forward(self, *a, **k):
        raise NotImplementedError("Implement forward.")

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

    def __repr__(self):
        # **A child spanning several lines has all of them indented.** Indenting
        # the first line only put the inner one hard against the left margin when
        # a container held a container (a `Sequential` inside a `ModuleList`) and
        # the characters diverged from torch — the kind where the values are fine
        # and the picture is wrong.
        parts = []
        for name, mod in self._modules.items():
            head, *rest = repr(mod).splitlines()
            parts.append(f"  ({name}): {head}")
            parts.extend(f"  {line}" for line in rest)
        inner = "\n".join(parts)
        if inner:
            return f"{type(self).__name__}(\n{inner}\n)"
        # **`extra_repr` was defined and never called.** torch's `Module.__repr__`
        # asks each layer for the arguments worth printing; here every class that
        # wanted any wrote its own `__repr__` instead, so a base class could not add
        # one for its subclasses — `nn.ReLU(inplace=True)` printed `ReLU()` while
        # torch printed `ReLU(inplace=True)`, which the golden freezes character for
        # character.
        #
        # Only reached when there are no children, and the default returns `""`, so
        # no layer that does not define one moves.
        extra = self.extra_repr()
        return f"{type(self).__name__}({extra})"

    def extra_repr(self):
        """The arguments this layer prints inside its parentheses. Empty by default,
        which is what makes adding it here safe for every layer that has none."""
        return ""


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        _no_device_dtype("Linear", device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        # Real torch's initialisation (the Kaiming uniform family):
        # U(-1/√fan_in, 1/√fan_in)
        # **The input can be 0.** `AdaptiveLogSoftmaxWithLoss`'s tail dimension is
        # `in_features // div_value**(i+1)`, which falls to 0 — torch builds an
        # empty tensor and moves on with nothing to initialise. This divided by
        # √0 and stopped.
        bound = 1.0 / _math.sqrt(in_features) if in_features else 0.0
        self.weight = Parameter(_rng.uniform(-bound, bound, (out_features, in_features)).astype(_DEFAULT_DTYPE))
        self.bias = Parameter(_rng.uniform(-bound, bound, (out_features,)).astype(_DEFAULT_DTYPE)) if bias else None

    def forward(self, x):
        out = x @ self.weight.transpose(-2, -1)
        return out + self.bias if self.bias is not None else out

    def __repr__(self):
        # `bias=` is printed too — torch does that, and it surfaced once the
        # golden began asking about the characters of a lazy layer after it
        # materialises. Whether there is a bias is information that changes the
        # `state_dict` keys.
        return (f"Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={getattr(self, 'bias', None) is not None})")


class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        return tanh(x)


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        """`end_dim` folds a run of axes and leaves the rest — `Flatten(1, 2)` on
        `(N, C, H, W)` gives `(N, C·H, W)`. `Tensor.flatten` took it already; this
        layer did not, so the two forms of one operation disagreed about which
        arguments exist."""
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)


class Identity(Module):
    """**It takes any arguments.** torch does that (measured) —
    `Identity(64, unused=True)`.

    It is a layer used as a placeholder, so people swap the name and leave the
    replaced layer's arguments where they are. Refusing the arguments stops at
    that line.
    """

    def __init__(self, *args, **kw):
        super().__init__()

    def forward(self, x):
        return x


class Dropout(Module):
    """**The formula lived here in a second copy.** The one in `_ops.dropout`
    branches on `p == 1`, where `1/(1-p)` is a division by zero; this one did not,
    so `nn.Dropout(1.0)` produced NaN where `F.dropout(x, 1.0)` produced zeros —
    the same library disagreeing with itself. Calling the function is also what
    lets `inplace` arrive, which this layer never took at all.

    **It was briefly refused here instead**, on the reasoning that dropout's mask is
    a fresh tensor either way so honouring the flag would be a promise about memory
    that nothing catches. The reasoning was right about the danger and wrong about
    the premise: `_ops.dropout` now writes the product back through the same
    `Tensor._inplace` the underscore names use, so the caller's buffer really does
    move — measured against torch, input changed and the same object returned.
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p, self.inplace = p, inplace

    def forward(self, x):
        return dropout(x, self.p, self.training, self.inplace)

    def extra_repr(self):
        return f"p={self.p}, inplace={self.inplace}"


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self._layers = list(layers)
        for i, m in enumerate(layers):
            self._modules[str(i)] = m

    def forward(self, x):
        for m in self._layers:
            x = m(x)
        return x

    def __getitem__(self, i):
        return self._layers[i]

    def __len__(self):
        return len(self._layers)


class ModuleList(Module):
    """A list of layers. **The index is the name** — as in `layers.0.weight`.

    Without `append` there is no way to write a model whose layer count is not
    fixed. That is the commonest shape in torch code:

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(Block())
    """

    def __init__(self, mods=()):
        super().__init__()
        self._layers = list(mods)
        for i, m in enumerate(self._layers):
            self._modules[str(i)] = m

    def _renumber(self):
        """Renumber `_modules` in list order.

        With `insert` the indices shift, so they are rewritten wholesale. Fixing
        only the shifted positions leaves an old name in the middle, and that
        leaks out as a `state_dict` key.
        """
        self._modules.clear()
        for i, m in enumerate(self._layers):
            self._modules[str(i)] = m

    def append(self, module):
        self._layers.append(module)
        self._modules[str(len(self._layers) - 1)] = module
        return self

    def extend(self, mods):
        for m in mods:
            self.append(m)
        return self

    def insert(self, index, module):
        self._layers.insert(index, module)
        self._renumber()
        return self

    def __iter__(self):
        return iter(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    def __setitem__(self, i, module):
        self._layers[i] = module
        self._renumber()

    def __iadd__(self, mods):
        return self.extend(mods)

    def __len__(self):
        return len(self._layers)


def _ordered(mapping):
    """torch's ordering rule. **A plain dict has its keys sorted on the way in.**

    Given an `OrderedDict` or a container of that kind it keeps the insertion
    order, and given a plain `dict` it sorts. torch does that (dicts in old
    Python did not keep their order),
    Unmatched, `named_parameters`'s order diverges, and that order is
    `state_dict`'s.

    The golden caught this place — given `{"w": …, "b": …}`, torch produced
    `ws.b ws.w` and this produced `ws.w ws.b`. Had the two keys been chosen in
    alphabetical order it would have passed by accident.
    """
    items = dict(mapping or {})
    if isinstance(mapping, (ModuleDict, ParameterDict, _collections.OrderedDict)):
        return list(items.items())
    return sorted(items.items(), key=lambda kv: str(kv[0]))


class ModuleDict(Module):
    """A named group of layers. Used by models that select a branch by name.

    The name given instead of an index becomes the `state_dict` key —
    `blocks.down.weight`.
    """

    def __init__(self, mods=None):
        super().__init__()
        for name, m in _ordered(mods):
            self._modules[str(name)] = m

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self._modules[str(key)] = module

    def __contains__(self, key):
        return key in self._modules

    def __iter__(self):
        return iter(self._modules)

    def __len__(self):
        return len(self._modules)

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, mods):
        for name, m in _ordered(mods):
            self._modules[str(name)] = m
        return self


class ParameterList(Module):
    """A list of `Parameter`s. **Without it there is no way around.**

    Putting `Parameter`s in a bare list and attaching it as an attribute leaves
    `Module.__setattr__` unable to recognise it as either a `Parameter` or a
    `Module`. It enters no list, `parameters()` does not produce it and the
    optimiser cannot see it — and **the loss goes down anyway**, because the
    remaining parameters compensate. No exception and no warning.

    torch fails to recognise it in exactly the same way, which is why torch has
    this class.
    """

    def __init__(self, params=()):
        super().__init__()
        for i, p in enumerate(params):
            self._params[str(i)] = p

    def _at(self, i):
        keys = list(self._params)
        return keys[i]

    def append(self, param):
        self._params[str(len(self._params))] = param
        return self

    def extend(self, params):
        for p in params:
            self.append(p)
        return self

    def __getitem__(self, i):
        return self._params[self._at(i)]

    def __setitem__(self, i, param):
        self._params[self._at(i)] = param

    def __iter__(self):
        return iter(self._params.values())

    def __len__(self):
        return len(self._params)


class ParameterDict(Module):
    """A named group of `Parameter`s. It exists for `ParameterList`'s reason."""

    def __init__(self, params=None):
        super().__init__()
        for name, p in _ordered(params):
            self._params[str(name)] = p

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, param):
        self._params[str(key)] = param

    def __contains__(self, key):
        return key in self._params

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def keys(self):
        return self._params.keys()

    def values(self):
        return self._params.values()

    def items(self):
        return self._params.items()

    def update(self, params):
        for name, p in _ordered(params):
            self._params[str(name)] = p
        return self


# **How it folds is part of the loss.** `_ops._reduce` keeps that rule in one
# place and thirteen use it, and **the five commonest were not** — the four here
# and `NLLLoss`. A `.mean()` was baked in, so passing `reduction=` stopped with a
# `TypeError`.
#
# Being upside down was the clue. **Every rare loss** — `cosine_embedding`,
# `multi_margin`, `triplet` — took `reduction`, and `MSELoss` and
# `CrossEntropyLoss` did not. What was written later followed torch's signature
# and what was written first went unfixed. The golden did not see it because
# tutorials use the default `mean` only.
def _reduce_ignoring(each, idx, ignore_index, reduction):
    """Fold `each`, leaving out the rows whose target is `ignore_index`.

    **torch treats the three reductions differently and the difference is easy to
    get wrong in both directions:**

    - `mean` drops them from the **denominator** as well as the sum. A batch with
      half its targets ignored divides by the half that remain; zeroing the terms
      and averaging as usual gives a number too small by exactly that ratio, and
      nothing about it looks wrong.
    - `sum` is the same either way.
    - `none` **keeps them, as zeros.** The shape is part of the answer here, so
      dropping the rows returns a shorter vector — measured against torch, which
      returns `[0.357, 0.0, 0.511]` where dropping gives `[0.357, 0.511]`.

    The first version dropped for all three and was right about `mean` and `sum`.
    A case at `none` is what showed it, which is why the case exists.
    """
    if ignore_index is None or not (idx == ignore_index).any():
        return _reduce(each, reduction)
    mask = _wrap((idx != ignore_index).astype(each.data.dtype))
    zeroed = each * mask
    if reduction == "none":
        return zeroed
    total = zeroed.sum()
    return total if reduction == "sum" else total / int((idx != ignore_index).sum())


class _WrittenLoss(Module):
    """The base of the losses whose classes are **written out below.**

    **This and `_GeneratedLoss` were both called `_Loss`**, and both were live. The
    eight losses written by hand inherit from this one because they are defined
    before line 3151; the thirteen built from `_LOSSES` inherit from the other
    because they are built after it. Which class the name meant depended on where in
    the file a reader was standing — and searching for `_Loss` from `NLLLoss` found
    the later one, which does something else and raises nothing when mistaken for
    this.

    `reduction` is accepted in one place. Written per loss, the places
    diverge.

    **`weight` and `pos_weight` are refused here.** torch registers the two as
    buffers and ships them in `state_dict` (a loss has a checkpoint too), and
    above all the division in `mean` changes — it divides by **the sum of the
    weights** rather than by the sample count. Accepted and unused, the loss
    value quietly changes, and that leads to choosing the wrong learning rate.

    A refusal message is used because a `TypeError` does not say "absent from
    this library" — it is the same screen as a typo.
    """

    def __init__(self, reduction="mean", *, weight=None, pos_weight=None,
                 ignore_index=-100, label_smoothing=0.0):
        """**One signature for six losses is what torch does not do**, and the
        subclasses below override this with torch's own list each.

        It stays as the shared body: whatever the order the arguments arrive in,
        they end up here, and the refusals are written once.

        `weight` and `pos_weight` are refused rather than dropped — see above — but
        they now arrive **at the position torch puts them**, which is first for
        every loss that has one. `CrossEntropyLoss(class_weights)` used to put the
        tensor into `reduction` and fail later with a numpy message about comparing
        a float32 array to a string; it says what is actually wrong now.
        """
        super().__init__()
        if weight is not None:
            _unsupported(f"{type(self).__name__}(weight=…) — class weights")
        if pos_weight is not None:
            _unsupported(f"{type(self).__name__}(pos_weight=…)")
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing


class MSELoss(_WrittenLoss):
    def __init__(self, reduction="mean"):
        """**torch's `MSELoss` has no `weight` at all**, and this offered one —
        keyword-only, and refused when given, but offered. An argument that only
        exists to say no is a wrong entry in the reference and in every editor's
        completion list, and this axis found it looking the other way for once."""
        super().__init__(reduction)

    def forward(self, pred, target):
        return _reduce((pred - target) ** 2, self.reduction)


class BCEWithLogitsLoss(_WrittenLoss):
    def __init__(self, weight=None, reduction="mean", pos_weight=None):
        """torch's order — and `pos_weight` is **last**, after `reduction`, which is
        the only loss that puts it there."""
        super().__init__(reduction, weight=weight, pos_weight=pos_weight)

    def forward(self, logits, target):
        # log(1+e^-|x|) + max(x,0) - x*t  — the form that stays safe at large values
        x, t = logits, target
        return _reduce(relu(x) - x * t + (1 + (-(x.abs())).exp()).log(),
                       self.reduction)


class BCELoss(_WrittenLoss):
    def __init__(self, weight=None, reduction="mean"):
        """torch's order. **No `pos_weight`** — that belongs to the logits form
        alone, and offering it here would be an argument torch does not have."""
        super().__init__(reduction, weight=weight)

    def forward(self, p, t):
        """**torch clamps the log, and this was adding an epsilon to the
        probability.** They are not the same guard and the numbers say so.

            p = 0,     target 1     ours 27.631   torch 100.0
            p = 1e-20, target 1     ours 27.631   torch  46.052
            p = 1e-8,  target 1     ours 18.42058 torch  18.42068

        `-(p + 1e-12).log()` cannot exceed `-log(1e-12) = 27.63` whatever `p` is, so
        every confident-and-wrong prediction reported the same loss as every other
        one. torch's documented rule is that **the log's output** is clamped at
        −100, which caps the loss at 100 and leaves 1e-20 telling the truth.

        The epsilon is also wrong where nothing is clamped: at p = 1e-8 it moves the
        answer in the fourth decimal, because it is added to a number smaller
        than itself is large.

        **This is `CrossEntropyLoss`'s defect a second time** — that one was
        `-(picked + 1e-12).log()` and capped at 27.63 too. The guard and the defect
        were the same line there as well. Two losses, one habit; the search that
        found this one was for the habit rather than for the loss.
        """
        pd = _np.asarray(p.data, dtype=_DEFAULT_DTYPE)
        td = _np.asarray(t.data, dtype=_DEFAULT_DTYPE)
        with _np.errstate(divide="ignore"):
            lo = _np.maximum(_np.log(pd), -100.0)
            hi = _np.maximum(_np.log(1.0 - pd), -100.0)
        out = -(td * lo + (1 - td) * hi)

        def back(g):
            # **torch floors the denominator at 1e-12 rather than differentiating
            # through the clamp**, so the gradient saturates at 1e12 where the value
            # saturates at 100. Differentiated through, `p = 1e-20` gives −1e20 —
            # eight orders past torch, and the first optimiser step takes the weight
            # somewhere no finite learning rate was meant to reach.
            gg = _np.asarray(g, dtype=_DEFAULT_DTYPE)
            denom = _np.maximum(pd * (1.0 - pd), 1e-12)
            return (gg * (pd - td) / denom, gg * (hi - lo))

        return _reduce(p._make(out, (p, t), back, "BinaryCrossEntropyBackward0"),
                       self.reduction)


class CrossEntropyLoss(_WrittenLoss):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean",
                 label_smoothing=0.0):
        """torch's order: `weight` first, `reduction` **third.**

        This was `(reduction, *, weight, pos_weight)`, so the two arguments people
        actually pass to this constructor — class weights, and an index to skip —
        were either unreachable by position or landed on `reduction`.
        """
        super().__init__(reduction, weight=weight, ignore_index=ignore_index,
                         label_smoothing=label_smoothing)

    def forward(self, logits, target):
        """**Through `log_softmax`, not through `log(softmax(x) + 1e-12)`.**

        The old line was `-(picked + 1e-12).log()`, and the epsilon there was put in
        to keep the logarithm away from zero. It does — and in doing so it **caps the
        loss at `-log(1e-12)`, which is 27.63.** Once a true class's probability
        underflows float32 (logits separated by about 28 apart, which happens in any
        confidently-wrong prediction) the loss stops rising and the gradient stops
        with it. Measured against torch on logits of scale 20: torch 25.34, this
        14.00.

        `log_softmax` computes `x - logsumexp(x)` and never forms the probability at
        all, so there is nothing to underflow and no epsilon to become a ceiling. It
        was already in this repository, already exact against torch at every scale
        tried, and sitting one call away.

        **The guard and the defect were the same line.** Nothing was missing: the
        epsilon does prevent `log(0)`, and preventing `log(0)` this way is what put a
        ceiling on the answer. A test asking whether the loss is finite would have
        passed on both.

        Found from `maximize`, which drives cross-entropy into exactly this regime —
        the only reason anybody looked.
        """
        n = logits.data.shape[0]
        logp = log_softmax(logits, dim=-1)
        idx = target.data.astype(int)
        # **The ignored rows are gathered before they are dropped**, so their index
        # has to be a real one first — torch's own sentinel is -100 and picking with
        # it raises. Row 0 is a placeholder whose value never reaches the answer.
        safe = _np.where(idx == self.ignore_index, 0, idx)
        each = -logp[_np.arange(n), safe]
        if self.label_smoothing:
            # torch spreads ε over every class: the target term keeps 1-ε and the
            # rest share ε/C, which is the mean of every class's log-probability.
            e = self.label_smoothing
            each = each * (1 - e) + (-logp).mean(dim=-1) * e
        return _reduce_ignoring(each, idx, self.ignore_index, self.reduction)


def _nn_unsupported(name):
    def factory(*a, **k):
        _unsupported(f"nn.{name}")
    return factory


def _no_device_dtype(name, device, dtype):
    """torch's `device=` and `dtype=` occupy the last two positions of nearly every
    layer. **They are carried and refused rather than left out**: left out, a
    positional call that reaches them lands on nothing and the two signatures part
    at every layer that has them; carried, the position is torch's and the refusal
    says which name it was."""
    if device is not None:
        _unsupported(f"nn.{name}(device=…)")
    if dtype is not None:
        _unsupported(f"nn.{name}(dtype=…)")


def _conv_bound(in_channels, groups, kernel_numel):
    """torch's initialisation bound. **The fan-in is divided by `groups`** —
    each filter sees only `in_channels // groups` channels, so a grouped
    convolution initialised as if it saw all of them starts too small."""
    return 1.0 / _math.sqrt(in_channels // groups * kernel_numel)


_PADDING_MODES = ("zeros", "reflect", "replicate", "circular")


def _conv_pad(x, padding, mode, spatial):
    """The non-zero padding modes, applied before the convolution.

    torch's layer does the same: anything but `zeros` is padded here and the
    convolution is then called with padding 0. Keeping it in the layer rather
    than in `conv2d` is why `F.conv2d` has no `padding_mode` — the functional
    takes an already-padded input, and putting the mode there too would be a
    second place for the same decision.
    """
    if mode == "zeros":
        return x, padding
    if mode not in _PADDING_MODES:
        raise ValueError(f"padding_mode has to be one of {_PADDING_MODES}, got {mode!r}")
    widths = _spread(padding, spatial)
    if not any(widths):
        return x, 0
    pairs = []
    for width in reversed(widths):
        pairs += [width, width]
    return pad(_wrap(x), pairs, mode=mode), 0


class Conv2d(Module):
    """**`bias` moved from the sixth position to the eighth**, where torch has it.

    `Conv2d(3, 16, 3, 1, 1, False)` used to turn the bias off and now sets
    `dilation=False`. Every call site in this repository already passed `bias` by
    keyword, which is the only reason the move was quiet — a positional call is a
    silent bet that the callee's parameter order never moves.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros",
                 device=None, dtype=None):
        super().__init__()
        _no_device_dtype("Conv2d", device, dtype)
        if padding_mode not in _PADDING_MODES:
            raise ValueError(
                f"padding_mode has to be one of {_PADDING_MODES}, got {padding_mode!r}")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.dilation, self.groups, self.padding_mode = dilation, groups, padding_mode
        ks = _spread(kernel_size, 2)
        bound = _conv_bound(in_channels, groups, ks[0] * ks[1])
        self.weight = Parameter(_rng.uniform(
            -bound, bound,
            (out_channels, in_channels // groups, ks[0], ks[1])).astype(_DEFAULT_DTYPE))
        self.bias = Parameter(_rng.uniform(-bound, bound, (out_channels,)).astype(_DEFAULT_DTYPE)) if bias else None

    def forward(self, x):
        x, padding = _conv_pad(x, self.padding, self.padding_mode, 2)
        return conv2d(x, self.weight, self.bias, self.stride, padding,
                      self.dilation, self.groups)

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")


class MaxPool2d(Module):
    """**With `return_indices` on there are two answers** — the values and the
    winning positions.

    Those positions are handed to `MaxUnpool2d` to undo it. A common pair in an
    autoencoder, and without the positions it cannot be undone — which cell won
    is not in the values.
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation
        self.return_indices, self.ceil_mode = return_indices, ceil_mode

    def forward(self, x):
        return max_pool2d(x, self.kernel_size, self.stride, self.padding,
                          self.dilation, self.return_indices, self.ceil_mode)

    def __repr__(self):
        return _pool_repr("MaxPool2d", self)


class Embedding(Module):
    """A trainable table turning an index into a vector. The alternative chapter
    8 offers to "do not just number them".

    **It took two arguments where torch takes nine**, and a name count cannot see
    that: `torch_gap.py` counts names, `Embedding` was present, and the namespace
    read 100%. `EmbeddingBag` next door has carried torch's full list all along, so
    the two neighbours disagreed about the same five arguments.

    Three of the nine are carried and refused rather than left out, for the reason
    `_no_device_dtype` gives: left out, a positional call that reaches them lands on
    nothing, and the two signatures part at every position after the gap.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        super().__init__()
        _no_device_dtype("Embedding", device, dtype)
        if scale_grad_by_freq:
            _unsupported("Embedding(scale_grad_by_freq=True)")
        if sparse:
            _unsupported("Embedding(sparse=True) — there is no sparse gradient here")
        if padding_idx is not None:
            if padding_idx >= num_embeddings or padding_idx < -num_embeddings:
                raise ValueError("padding_idx must be within num_embeddings")
            if padding_idx < 0:
                padding_idx = num_embeddings + padding_idx
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.padding_idx = padding_idx
        self.max_norm, self.norm_type = max_norm, norm_type
        self.scale_grad_by_freq, self.sparse = scale_grad_by_freq, sparse
        if _weight is None:
            table = _rng.standard_normal(
                (num_embeddings, embedding_dim)).astype(_DEFAULT_DTYPE)
            # **A fresh table zeroes the padding row and a given one does not.**
            # torch draws that line at the same place: `from_pretrained` leaves the
            # row as it arrived, because the caller who supplied the weights meant
            # them. Measured both ways.
            if padding_idx is not None:
                table[padding_idx] = 0.0
        else:
            table = _np.asarray(_weight.data if hasattr(_weight, "data") else _weight,
                                dtype=_DEFAULT_DTYPE)
            if table.shape != (num_embeddings, embedding_dim):
                raise ValueError(
                    "Shape of weight does not match num_embeddings and embedding_dim")
        self.weight = Parameter(table)
        self.weight.requires_grad = not _freeze

    def forward(self, idx):
        ids = idx.data.astype(int)
        if self.max_norm is not None:
            # **In the table itself**, as `embedding_bag` does it and as torch does.
            # The same function, not a second copy of the rule.
            _renorm_rows(self.weight, ids, self.max_norm, self.norm_type)
        out = self.weight.data[ids]

        def back(g):
            gw = _np.zeros_like(self.weight.data)
            _np.add.at(gw, ids.reshape(-1), _np.asarray(g).reshape(-1, self.embedding_dim))
            # **The padding row learns nothing.** Left in, a pad token drifts toward
            # whatever the loss wants and the mask stops meaning "ignore this".
            if self.padding_idx is not None:
                gw[self.padding_idx] = 0.0
            return (gw,)

        return self.weight._make(out, (self.weight,), back)

    def __repr__(self):
        parts = [f"{self.num_embeddings}, {self.embedding_dim}"]
        if self.padding_idx is not None:
            parts.append(f"padding_idx={self.padding_idx}")
        if self.max_norm is not None:
            parts.append(f"max_norm={self.max_norm}")
        if self.norm_type != 2.0:
            parts.append(f"norm_type={self.norm_type}")
        if self.scale_grad_by_freq:
            parts.append("scale_grad_by_freq=True")
        if self.sparse:
            parts.append("sparse=True")
        return f"Embedding({', '.join(parts)})"


class LayerNorm(Module):
    """**`normalized_shape` decides how many axes are folded** — not the last
    axis alone.

    Measured with `LayerNorm(4)` only, this divergence is invisible. With one
    axis it gives the same answer as "fold the last axis", and that is what was
    written. `LayerNorm((3, 4))` folds the last two axes **as one block** — the
    mean and the variance come from 12 cells.

    `elementwise_affine` is taken too. Turned off the parameters disappear, and
    then the `state_dict` keys vanish wholesale — a story about the wiring rather
    than about a value.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 bias=True, device=None, dtype=None):
        super().__init__()
        _no_device_dtype("LayerNorm", device, dtype)
        shape = ((normalized_shape,) if isinstance(normalized_shape, int)
                 else tuple(normalized_shape))
        self.normalized_shape = shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(_np.ones(shape, dtype=_DEFAULT_DTYPE))
            if bias:
                self.bias = Parameter(_np.zeros(shape, dtype=_DEFAULT_DTYPE))

    def forward(self, x):
        dims = len(self.normalized_shape)
        # The last `dims` axes are folded into one and the mean and variance
        # come out together. Folded axis by axis the numbers differ — the mean of
        # means is not the mean.
        shape = tuple(int(n) for n in x.shape)
        # **A mismatched shape stops.** Being lenient quietly folds the wrong
        # axis.
        if shape[len(shape) - dims:] != self.normalized_shape:
            raise RuntimeError(_like_torch(
                f"normalized_shape={list(self.normalized_shape)} but the input is "
                f"{list(shape)}.",
                f"Given normalized_shape={list(self.normalized_shape)}, expected "
                f"input with shape [*, "
                f"{', '.join(str(n) for n in self.normalized_shape)}]"))
        lead = shape[:len(shape) - dims]
        flat = x.reshape(*lead, -1) if dims > 1 else x
        mean = flat.mean(dim=-1, keepdim=True)
        centered = flat - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        normed = centered / (var + self.eps) ** 0.5
        if dims > 1:
            normed = normed.reshape(*shape)
        if not self.elementwise_affine:
            return normed
        out = normed * self.weight
        return out + self.bias if hasattr(self, "bias") else out


class BatchNorm2d(Module):
    """This batch during training and the accumulated values during evaluation —
    the layer eval() changes in chapter 6."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None, *, bias=True):
        """torch's `affine` and `bias`, which this took neither of.

        `affine=False` is the layer with no learnable scale or shift at all —
        common enough that torch makes it a positional argument — and `bias=True`
        is the *other* half of it, keeping the scale and dropping the shift.
        Neither was here, so `BatchNorm2d(3, affine=False)` was a `TypeError` and
        the `state_dict` had two keys torch's did not.

        `bias` is keyword-only, as it is in torch, and it only means anything under
        `affine`: torch builds no bias when either is off. The signature axis
        counted this absence across thirteen layers — three `BatchNorm`s, three
        `InstanceNorm`s, `GroupNorm` and the six lazy variants — and the count is
        what made it work rather than a note somebody would meet later.
        """
        super().__init__()
        _no_device_dtype("BatchNorm2d", device, dtype)
        if not track_running_stats:
            # The forward pass reads the buffers in evaluation mode, so ignoring
            # this would leave training right and evaluation quietly wrong — the
            # shape `_InstanceNorm` next door already refuses for the same reason.
            _unsupported(f"{type(self).__name__} with track_running_stats=False")
        self.eps, self.momentum, self.affine = eps, momentum, affine
        self.weight = (Parameter(_np.ones(num_features, dtype=_DEFAULT_DTYPE))
                       if affine else None)
        self.bias = (Parameter(_np.zeros(num_features, dtype=_DEFAULT_DTYPE))
                     if affine and bias else None)
        self.register_buffer("running_mean", _np.zeros(num_features, dtype=_DEFAULT_DTYPE))
        self.register_buffer("running_var", _np.ones(num_features, dtype=_DEFAULT_DTYPE))
        self.register_buffer("num_batches_tracked", 0)

    def forward(self, x):
        # **`F.batch_norm` does the computation.** With the layer and the
        # function each writing their own they eventually diverge, and the place
        # they diverge is the running statistics, so training is fine and
        # evaluation alone is wrong.
        #
        # The rank is not examined — everything but the channel axis is reduced,
        # so (N,C,H,W) and (N,C,D,H,W) run through the same code. `BatchNorm3d`
        # inherits it unchanged.
        out = batch_norm(x, self.running_mean, self.running_var,
                         self.weight, self.bias, self.training,
                         self.momentum, self.eps)
        if self.training:
            with no_grad():
                self.num_batches_tracked = self.num_batches_tracked + 1
        return out



class _RNNBase(Module):
    """What RNN, LSTM and GRU share — building the parameters and the layer and
    time loops.

    The parameter names match torch's (`weight_ih_l0` and the rest). Matching
    names make the `state_dict` keys match, and a checkpoint crosses between the
    two.

    Time is a Python loop. Recurrence cannot be parallelised because the earlier
    step has to finish before the later one is visible (chapter 30), and that
    slowness is why the transformer exists. Since **the input-side multiplication
    does not involve h**, though, the whole of time is computed at once — only
    the hidden-side multiplication is left inside the loop.
    """

    gates = 1
    mode = None

    # torch's own set. `RNN_TANH` and `RNN_RELU` are one class with two activations,
    # which is why `RNN` takes `nonlinearity` and the mode carries it here.
    MODES = ("RNN_TANH", "RNN_RELU", "LSTM", "GRU")

    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False, proj_size=0,
                 device=None, dtype=None):
        """torch's parameter list, `mode` first.

        **This began at `input_size`**, so `RNNBase` disagreed with torch by one
        place at every position and the four arguments after `batch_first` were not
        there at all. `RNN`, `LSTM` and `GRU` pass their own mode now, which is what
        torch does too — the string is not decoration, it is what tells the base
        class which recurrence it is building.

        `dropout`, `bidirectional` and `proj_size` are **refused when asked for**
        rather than accepted and ignored. A bidirectional layer that silently runs
        one direction returns a plausible number of the wrong shape's meaning.
        """
        super().__init__()
        _no_device_dtype(type(self).__name__, device, dtype)
        if mode not in self.MODES:
            raise ValueError(f"Unrecognized RNN mode: {mode}")
        self.mode = mode
        if dropout:
            _unsupported(f"{type(self).__name__}(dropout={dropout})")
        if bidirectional:
            _unsupported(f"{type(self).__name__}(bidirectional=True)")
        if proj_size:
            _unsupported(f"{type(self).__name__}(proj_size={proj_size})")
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.has_bias = bias

        bound = 1.0 / _math.sqrt(hidden_size)
        g = self.gates
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            setattr(self, f"weight_ih_l{layer}", Parameter(
                _rng.uniform(-bound, bound, (g * hidden_size, in_size)).astype(_DEFAULT_DTYPE)))
            setattr(self, f"weight_hh_l{layer}", Parameter(
                _rng.uniform(-bound, bound, (g * hidden_size, hidden_size)).astype(_DEFAULT_DTYPE)))
            if bias:
                setattr(self, f"bias_ih_l{layer}", Parameter(
                    _rng.uniform(-bound, bound, g * hidden_size).astype(_DEFAULT_DTYPE)))
                setattr(self, f"bias_hh_l{layer}", Parameter(
                    _rng.uniform(-bound, bound, g * hidden_size).astype(_DEFAULT_DTYPE)))

    def _weights(self, layer):
        return (getattr(self, f"weight_ih_l{layer}"), getattr(self, f"weight_hh_l{layer}"),
                getattr(self, f"bias_ih_l{layer}", None), getattr(self, f"bias_hh_l{layer}", None))

    def _run(self, x, init):
        """(output, a list of per-layer final states). init is the function
        supplying the per-layer initial state."""
        if self.batch_first:
            x = x.transpose(0, 1)                       # (N,T,I) → (T,N,I)
        T, N = x.data.shape[0], x.data.shape[1]

        layer_input = x
        finals = []
        for layer in range(self.num_layers):
            w_ih, w_hh, b_ih, b_hh = self._weights(layer)
            pre = layer_input @ w_ih.transpose(0, 1)     # (T, N, gates*H) — independent of h
            if self.has_bias:
                pre = pre + b_ih
            state = init(layer, N)
            steps = []
            for t in range(T):
                state, out = self._step(pre[t], state, w_hh, b_hh)
                steps.append(out)
            layer_input = stack(steps)
            finals.append(state)

        out = layer_input
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, finals

    def _step(self, pre_t, state, w_hh, b_hh):
        raise NotImplementedError

    def __repr__(self):
        return (f"{type(self).__name__}({self.input_size}, {self.hidden_size}"
                + (f", num_layers={self.num_layers}" if self.num_layers > 1 else "")
                + (", batch_first=True" if self.batch_first else "") + ")")


class RNN(_RNNBase):
    """h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh) — what chapter 29
    teaches."""

    def __init__(self, *a, nonlinearity="tanh", **k):
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be 'tanh' or 'relu'.")
        self.nonlinearity = nonlinearity
        # The mode carries the activation, which is why `RNN_TANH` and `RNN_RELU`
        # are two of torch's four modes and one class here.
        super().__init__(f"RNN_{nonlinearity.upper()}", *a, **k)

    def _step(self, pre_t, h, w_hh, b_hh):
        act = tanh if self.nonlinearity == "tanh" else relu
        z = pre_t + h @ w_hh.transpose(0, 1)
        if self.has_bias:
            z = z + b_hh
        h = act(z)
        return h, h

    def forward(self, x, hx=None):
        if hx is None:
            hx = zeros(self.num_layers, x.data.shape[1 if not self.batch_first else 0],
                       self.hidden_size)
        out, finals = self._run(x, lambda layer, n: hx[layer])
        return out, stack(finals)


class LSTM(_RNNBase):
    """Four gates learn **what to forget and what to keep.**

        i = σ(...)  what to take in       f = σ(...)  what to drop
        g = tanh(...) the value to take in  o = σ(...)  what to send out
        c' = f·c + i·g                   h' = o·tanh(c')

    `weight_ih_l0` is (4H, I) and the row order is **i, f, g, o.** It has to
    match torch for a checkpoint to cross — reordered, the values are plausible
    and training does not happen.
    """

    gates = 4

    def __init__(self, *a, **k):
        super().__init__("LSTM", *a, **k)

    def _step(self, pre_t, state, w_hh, b_hh):
        h, c = state
        z = pre_t + h @ w_hh.transpose(0, 1)
        if self.has_bias:
            z = z + b_hh
        H = self.hidden_size
        i = sigmoid(z[:, 0 * H:1 * H])
        f = sigmoid(z[:, 1 * H:2 * H])
        g = tanh(z[:, 2 * H:3 * H])
        o = sigmoid(z[:, 3 * H:4 * H])
        c = f * c + i * g
        h = o * tanh(c)
        return (h, c), h

    def forward(self, x, hx=None):
        batch = x.data.shape[0 if self.batch_first else 1]
        if hx is None:
            zero = zeros(self.num_layers, batch, self.hidden_size)
            hx = (zero, zeros(self.num_layers, batch, self.hidden_size))
        h0, c0 = hx
        out, finals = self._run(x, lambda layer, n: (h0[layer], c0[layer]))
        return out, (stack([h for h, _ in finals]), stack([c for _, c in finals]))


class GRU(_RNNBase):
    """Three gates. Simpler than LSTM and usually behaves much the same.

        r = σ(...)  how much of the past to look at   z = σ(...)  how much to switch over
        n = tanh(W_in·x + b_in + r·(W_hn·h + b_hn))
        h' = (1-z)·n + z·h

    In the n gate **r multiplies the hidden term including its bias** — keeping
    the bias outside makes the values slightly off, and that goes unnoticed.
    """

    gates = 3

    def __init__(self, *a, **k):
        super().__init__("GRU", *a, **k)

    def _step(self, pre_t, h, w_hh, b_hh):
        H = self.hidden_size
        hh = h @ w_hh.transpose(0, 1)
        if self.has_bias:
            hh = hh + b_hh
        r = sigmoid(pre_t[:, 0 * H:1 * H] + hh[:, 0 * H:1 * H])
        z = sigmoid(pre_t[:, 1 * H:2 * H] + hh[:, 1 * H:2 * H])
        n = tanh(pre_t[:, 2 * H:3 * H] + r * hh[:, 2 * H:3 * H])
        h = (1 - z) * n + z * h
        return h, h

    def forward(self, x, hx=None):
        if hx is None:
            hx = zeros(self.num_layers, x.data.shape[0 if self.batch_first else 1],
                       self.hidden_size)
        out, finals = self._run(x, lambda layer, n: hx[layer])
        return out, stack(finals)




class RNNCellBase(Module):
    """**One step** of the recurrence. Code that wants to write the time loop by
    hand calls this.

    **The names differ from the layer's.** A layer attaches a layer index, as in
    `weight_ih_l0`, and a cell is `weight_ih` — because a cell has no layers. The
    `state_dict` keys are those names, so getting them wrong means a checkpoint
    does not match.

    Only the gate count differs and the rest is the same. The one-step formula is
    `_RNNBase`'s, used as it is — written as two copies, the day comes when the
    gate order diverges, and then the shapes match and only the values are
    wrong.
    """

    gates = 1

    def __init__(self, input_size, hidden_size, bias=True, num_chunks=None,
                 device=None, dtype=None):
        """**`num_chunks` sits fourth, where torch has it**, and it decides how many
        gate blocks the weights hold — 1 for `RNNCell`, 3 for `GRUCell`, 4 for
        `LSTMCell`. The subclasses set it as a class attribute, so it had never been
        an argument here; adding `device` and `dtype` without it put `device` in its
        seat, and `RNNCellBase(4, 8, True, 3)` would have set a device to 3.

        Caught by the signature axis on the same run as the edit that caused it —
        the `shifted` bucket, the one that means *a positional call lands on the
        wrong parameter*. It is the first row that bucket has held since the
        optimizers, and it was mine.
        """
        super().__init__()
        _no_device_dtype(type(self).__name__, device, dtype)
        self.input_size, self.hidden_size = input_size, hidden_size
        self.has_bias = bias
        bound = 1.0 / _math.sqrt(hidden_size)
        g = self.gates if num_chunks is None else int(num_chunks)
        self.weight_ih = Parameter(_rng.uniform(
            -bound, bound, (g * hidden_size, input_size)).astype(_DEFAULT_DTYPE))
        self.weight_hh = Parameter(_rng.uniform(
            -bound, bound, (g * hidden_size, hidden_size)).astype(_DEFAULT_DTYPE))
        if bias:
            self.bias_ih = Parameter(
                _rng.uniform(-bound, bound, g * hidden_size).astype(_DEFAULT_DTYPE))
            self.bias_hh = Parameter(
                _rng.uniform(-bound, bound, g * hidden_size).astype(_DEFAULT_DTYPE))

    def _pre(self, x):
        out = x @ self.weight_ih.transpose(0, 1)
        return out + self.bias_ih if self.has_bias else out

    def _zeros(self, x):
        return zeros(x.data.shape[0], self.hidden_size)

    def __repr__(self):
        tail = "" if self.has_bias else ", bias=False"
        return f"{type(self).__name__}({self.input_size}, {self.hidden_size}{tail})"


class RNNCell(RNNCellBase):
    gates = 1

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh",
                 device=None, dtype=None):
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be 'tanh' or 'relu'.")
        self.nonlinearity = nonlinearity
        super().__init__(input_size, hidden_size, bias, device=device, dtype=dtype)

    def forward(self, x, hx=None):
        h = self._zeros(x) if hx is None else hx
        z = self._pre(x) + h @ self.weight_hh.transpose(0, 1)
        if self.has_bias:
            z = z + self.bias_hh
        return (tanh if self.nonlinearity == "tanh" else relu)(z)

    def __repr__(self):
        parts = f"{self.input_size}, {self.hidden_size}"
        if not self.has_bias:
            parts += ", bias=False"
        if self.nonlinearity != "tanh":
            parts += f", nonlinearity={self.nonlinearity}"
        return f"RNNCell({parts})"


class GRUCell(RNNCellBase):
    """Three gates. **The order inside `weight_ih` is `r, z, n`** — reordered,
    only the values diverge."""

    gates = 3

    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None):
        """**torch's list, without `num_chunks`.** The base takes it because torch's
        base does; a cell with a fixed number of gates does not, and inheriting the
        base's list put `num_chunks` in `device`'s seat here while torch has `device`
        there. `GRUCell(4, 8, True, "cpu")` would have set a chunk count to a string.
        """
        super().__init__(input_size, hidden_size, bias, device=device, dtype=dtype)


    def forward(self, x, hx=None):
        h = self._zeros(x) if hx is None else hx
        H = self.hidden_size
        pre = self._pre(x)
        hh = h @ self.weight_hh.transpose(0, 1)
        if self.has_bias:
            hh = hh + self.bias_hh
        r = sigmoid(pre[:, 0:H] + hh[:, 0:H])
        z = sigmoid(pre[:, H:2 * H] + hh[:, H:2 * H])
        n = tanh(pre[:, 2 * H:3 * H] + r * hh[:, 2 * H:3 * H])
        return (1 - z) * n + z * h


class LSTMCell(RNNCellBase):
    """Four gates. **The only one that returns two things** — `(h, c)`.

    Forcing the three into one shape loses the memory cell, and then values come
    out and training does not happen. The order inside `weight_ih` is
    `i, f, g, o`.
    """

    gates = 4

    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None):
        """**torch's list, without `num_chunks`.** The base takes it because torch's
        base does; a cell with a fixed number of gates does not, and inheriting the
        base's list put `num_chunks` in `device`'s seat here while torch has `device`
        there. `LSTMCell(4, 8, True, "cpu")` would have set a chunk count to a string.
        """
        super().__init__(input_size, hidden_size, bias, device=device, dtype=dtype)


    def forward(self, x, hx=None):
        h, c = (self._zeros(x), self._zeros(x)) if hx is None else hx
        H = self.hidden_size
        z = self._pre(x) + h @ self.weight_hh.transpose(0, 1)
        if self.has_bias:
            z = z + self.bias_hh
        i = sigmoid(z[:, 0:H])
        f = sigmoid(z[:, H:2 * H])
        g = tanh(z[:, 2 * H:3 * H])
        o = sigmoid(z[:, 3 * H:4 * H])
        c = f * c + i * g
        return o * tanh(c), c


nn.RNNCellBase = RNNCellBase
nn.RNNCell = RNNCell
nn.GRUCell = GRUCell
nn.LSTMCell = LSTMCell
# `RNNBase` is the parent of `RNN`, `LSTM` and `GRU`. It cannot be constructed
# directly in torch either (ValueError).
nn.RNNBase = _RNNBase


# ------------------------------------------------------- top-level recurrence
#
# **The layer's computation, taking the weights as a list.**
# `torch.lstm(x, (h,c), params, …)` is the form, and this is what the layer calls
# inside.
#
# **A layer is built and its weights swapped out.** Writing the recurrence
# formula again here means the day comes when the gate order diverges, and then
# the shapes match and only the values are wrong — the cell's docstring already
# borrows the layer's formula for the same reason. Building it wastes one set of
# parameters, and that is time rather than a value.

def _install_weights(mod, params, num_layers, has_biases):
    """Plug a flat list of weights into the layer's named slots.

    The order is **`[w_ih, w_hh, b_ih, b_hh]` per layer** (measured). With no bias
    it is two per layer. They are not wrapped in `Parameter` — the gradient has
    to reach the tensors the caller handed over, unchanged.
    """
    per = 4 if has_biases else 2
    want = per * num_layers
    if len(params) != want:
        raise RuntimeError(_like_torch(
            f"expected {want} weights but got {len(params)} "
            f"({num_layers} layers x {per}).",
            "expected a tuple of tensors of the right length"))
    for layer in range(num_layers):
        chunk = params[layer * per:(layer + 1) * per]
        setattr(mod, f"weight_ih_l{layer}", chunk[0])
        setattr(mod, f"weight_hh_l{layer}", chunk[1])
        if has_biases:
            setattr(mod, f"bias_ih_l{layer}", chunk[2])
            setattr(mod, f"bias_hh_l{layer}", chunk[3])


def _rnn_top(cls, x, hx, params, has_biases, num_layers, dropout, train,
             bidirectional, batch_first, **kw):
    """The body the four top-level recurrent ones share.

    **Bidirectionality and inter-layer dropout are refused.** Our layers have
    neither — handing back one direction here would be caught loudly because the
    shape is half, and the dropout side would diverge with plausible values
    (training with no regularisation applied). Both stop here.
    """
    if bidirectional:
        _unsupported("bidirectional recurrence (bidirectional=True)")
    if train and dropout:
        _unsupported(f"dropout between layers (dropout={dropout})")
    first = params[0]
    hidden = first.data.shape[0] // cls.gates
    mod = cls(first.data.shape[1], hidden, num_layers, bias=bool(has_biases),
              batch_first=bool(batch_first), **kw)
    _install_weights(mod, params, num_layers, bool(has_biases))
    return mod(x, hx)


def lstm(input, hx, params, has_biases, num_layers, dropout, train,      # noqa: A002
         bidirectional, batch_first=False):
    """`(output, h_n, c_n)` — **all three spread.** The layer groups them as
    `(output, (h, c))` and the top level does not (measured). Handed over
    grouped, the caller's unpacking is off by one."""
    out, (h, c) = _rnn_top(LSTM, input, hx, params, has_biases, num_layers,
                           dropout, train, bidirectional, batch_first)
    return out, h, c


def gru(input, hx, params, has_biases, num_layers, dropout, train,      # noqa: A002
        bidirectional, batch_first=False):
    return _rnn_top(GRU, input, hx, params, has_biases, num_layers, dropout,
                    train, bidirectional, batch_first)


def rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train,  # noqa: A002
             bidirectional, batch_first=False):
    return _rnn_top(RNN, input, hx, params, has_biases, num_layers, dropout,
                    train, bidirectional, batch_first, nonlinearity="tanh")


def rnn_relu(input, hx, params, has_biases, num_layers, dropout, train,  # noqa: A002
             bidirectional, batch_first=False):
    return _rnn_top(RNN, input, hx, params, has_biases, num_layers, dropout,
                    train, bidirectional, batch_first, nonlinearity="relu")


def _cell_top(cls, x, hx, w_ih, w_hh, b_ih, b_hh, **kw):
    """The body the four cells share. Unlike the layers, the names carry no
    layer index."""
    hidden = w_ih.data.shape[0] // cls.gates
    cell = cls(w_ih.data.shape[1], hidden, bias=b_ih is not None, **kw)
    cell.weight_ih, cell.weight_hh = w_ih, w_hh
    if b_ih is not None:
        cell.bias_ih, cell.bias_hh = b_ih, b_hh
    return cell(x, hx)


def lstm_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):              # noqa: A002
    """One step. **The same value** as `nn.LSTMCell` (measured)."""
    return _cell_top(LSTMCell, input, tuple(hx), w_ih, w_hh, b_ih, b_hh)


def gru_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):              # noqa: A002
    return _cell_top(GRUCell, input, hx, w_ih, w_hh, b_ih, b_hh)


def rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):         # noqa: A002
    return _cell_top(RNNCell, input, hx, w_ih, w_hh, b_ih, b_hh,
                     nonlinearity="tanh")


def rnn_relu_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):         # noqa: A002
    return _cell_top(RNNCell, input, hx, w_ih, w_hh, b_ih, b_hh,
                     nonlinearity="relu")


class MultiheadAttention(Module):
    """The attention written on day 45, split across several perspectives.

    torch keeps the Q, K and V weights **bundled into one**, in
    `in_proj_weight` (3E, E) — so that the matmul happens once rather than three
    times, and so a checkpoint takes that shape too. Carried separately the
    values match and the `state_dict` does not.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False, device=None, dtype=None):
        """torch's parameter list, in torch's order.

        **This took `(embed_dim, num_heads, bias, batch_first)`**, so
        `MultiheadAttention(64, 8, 0.1)` — torch's own way of writing a dropout of
        0.1 — set `bias=0.1` here and nothing raised. Five arguments were missing
        from the middle and every one of them shifted what followed.

        Four of the five are **refused rather than implemented**, and the refusal
        already existed one layer down: `multi_head_attention_forward` names
        `bias_k`, `add_zero_attn` and `use_separate_proj_weight` and stops. Carrying
        the argument here means the refusal arrives with the right name attached
        instead of the value landing on a different parameter — *an absent feature
        beats a wrong answer* is this library's rule, and a wrong **position** is a
        wrong answer wearing the shape of a feature.

        `dropout` is the one that works: the function applies it while training.
        """
        super().__init__()
        _no_device_dtype("MultiheadAttention", device, dtype)
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim({embed_dim}) is not divisible by num_heads({num_heads}).")
        if add_bias_kv:
            _unsupported("MultiheadAttention(add_bias_kv=True)")
        if add_zero_attn:
            _unsupported("MultiheadAttention(add_zero_attn=True)")
        for name, given in (("kdim", kdim), ("vdim", vdim)):
            # torch only takes the separate-projection path when these differ from
            # `embed_dim`; passing the same number is the ordinary layer and is not
            # a request for anything this cannot do.
            if given is not None and given != embed_dim:
                _unsupported(f"MultiheadAttention({name}={given}) — a key or value "
                             "width unlike the embedding's")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.batch_first = batch_first

        bound = _math.sqrt(1.0 / embed_dim)
        self.in_proj_weight = Parameter(
            _rng.uniform(-bound, bound, (3 * embed_dim, embed_dim)).astype(_DEFAULT_DTYPE))
        self.in_proj_bias = Parameter(_np.zeros(3 * embed_dim, dtype=_DEFAULT_DTYPE)) if bias else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=True,
                key_padding_mask=None, average_attn_weights=True):
        key = query if key is None else key
        value = query if value is None else value
        # **The function form puts the length first.** With `batch_first` it is
        # flipped here before being handed over — the computation is kept as one
        # copy in `multi_head_attention_forward`.
        if self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))
        out, weights = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias, None, None, False, self.dropout,
            self.out_proj.weight, self.out_proj.bias, self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, weights

    def __repr__(self):
        return f"MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None):
    """The core of attention. **The computation `MultiheadAttention` was doing
    inside, exposed under a name.**

    The layer existed and this function did not. Code that writes attention by
    hand without the layer calls this name, and that is the default shape of
    modern transformer code.

    **A mask is added, not multiplied.** `-inf` is added so that softmax produces
    0; a 0 is not multiplied in — multiplying comes after softmax has already
    normalised, so the remaining positions do not sum back to 1.
    """
    query, key, value = _wrap(query), _wrap(key), _wrap(value)
    dim = query.data.shape[-1]
    factor = (1.0 / _math.sqrt(dim)) if scale is None else scale
    scores = (query @ key.transpose(-2, -1)) * factor
    if is_causal:
        # Block the upper triangle. Given a mask as well, torch applies both.
        length = query.data.shape[-2]
        upper = _np.triu(_np.ones((length, key.data.shape[-2]), dtype=bool), k=1)
        scores = scores.masked_fill(Tensor(upper), float("-inf"))
    if attn_mask is not None:
        scores = _apply_mask(scores, attn_mask)
    weights = softmax(scores, dim=-1)
    if dropout_p:
        weights = dropout(weights, dropout_p, training=True)
    return weights @ value


def multi_head_attention_forward(
        query, key, value, embed_dim_to_check, num_heads,
        in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn,
        dropout_p, out_proj_weight, out_proj_bias, training=True,
        key_padding_mask=None, need_weights=True, attn_mask=None,
        use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
        v_proj_weight=None, static_k=None, static_v=None,
        average_attn_weights=True, is_causal=False):
    """The computation `MultiheadAttention` does inside, exposed under a name.
    **The layer calls this.**

    **The input is `(L, N, E)`** — length first. The layer takes `batch_first`
    and this function is always length-first in torch too, so calling it with the
    batch first quietly mixes the wrong axis.

    It takes the weights from outside, so code assembling attention by hand
    without the layer calls this name — torch's own `MultiheadAttention` calls it
    as well.

    **What it does not do is refused loudly.** Quietly ignoring a rarely used
    branch such as `bias_k`, `add_zero_attn` or `static_k` makes the values
    plausibly different.
    """
    for name, given in (("bias_k", bias_k), ("bias_v", bias_v),
                        ("static_k", static_k), ("static_v", static_v)):
        if given is not None:
            _unsupported(f"multi_head_attention_forward({name}=…)")
    if add_zero_attn:
        _unsupported("multi_head_attention_forward(add_zero_attn=True)")
    if use_separate_proj_weight:
        _unsupported("multi_head_attention_forward(use_separate_proj_weight=True)")

    query, key, value = _wrap(query), _wrap(key), _wrap(value)
    # Inside, everything is computed batch-first. It is flipped only on the way
    # in and on the way out.
    query, key, value = (t.transpose(0, 1) for t in (query, key, value))
    B, T, E = query.data.shape
    S = key.data.shape[1]
    if embed_dim_to_check is not None and int(embed_dim_to_check) != E:
        raise AssertionError(
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {E}")
    head_dim = E // num_heads

    w, b = _wrap(in_proj_weight), None if in_proj_bias is None else _wrap(in_proj_bias)

    def project(t, index):
        piece = w[index * E:(index + 1) * E]
        out = t @ piece.transpose(0, 1)
        return out + b[index * E:(index + 1) * E] if b is not None else out

    q = _split_heads(project(query, 0), B, T, num_heads, head_dim)
    k = _split_heads(project(key, 1), B, S, num_heads, head_dim)
    v = _split_heads(project(value, 2), B, S, num_heads, head_dim)

    scores = (q @ k.transpose(-2, -1)) / _math.sqrt(head_dim)
    if is_causal and attn_mask is None:
        attn_mask = _np.triu(_np.ones((T, S), dtype=bool), k=1)
    if attn_mask is not None:
        scores = _apply_mask(scores, attn_mask)
    if key_padding_mask is not None:
        # `(N, S)` is spread to `(N, 1, 1, S)` so every head masks the same
        # positions.
        pad = _wrap(key_padding_mask)
        scores = _apply_mask(scores, pad.reshape(B, 1, 1, S))
    weights = softmax(scores, dim=-1)
    if training and dropout_p:
        weights = dropout(weights, dropout_p, True)

    merged = (weights @ v).transpose(1, 2).reshape(B, T, E)
    out = merged @ _wrap(out_proj_weight).transpose(0, 1)
    if out_proj_bias is not None:
        out = out + _wrap(out_proj_bias)
    out = out.transpose(0, 1)                       # back to length-first
    if not need_weights:
        return out, None
    return out, (weights.mean(dim=1) if average_attn_weights else weights)


def _split_heads(t, B, T, heads, head_dim):
    return t.reshape(B, T, heads, head_dim).transpose(1, 2)      # (B, heads, T, head_dim)


def _apply_mask(scores, mask):
    """torch takes a mask in two forms.

      boolean — the True positions are masked (filled with -inf)
      float   — **added** to the scores. The 0/-inf `generate_square_subsequent_mask` gives is this

    Lumping a float mask in as "mask where it is not 0" happens to get the causal
    mask right and goes wrong on a mask that adjusts the weights.
    """
    m = mask if isinstance(mask, Tensor) else Tensor(_np.asarray(mask))
    if m.data.dtype.kind == "b":
        return scores.masked_fill(m, float("-inf"))
    return scores + m


class TransformerEncoderLayer(Module):
    """Attention plus feed-forward, each with a residual and a normalisation.
    Chapter 10's Block exactly."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        """**torch's order, which is not the order this had.**

        torch is `(…, activation, layer_norm_eps, batch_first, norm_first, bias)`
        and the sixth seat here was `batch_first` — so
        `TransformerEncoderLayer(4, 2, 8, 0.1, "relu", True)` put `True` into
        torch's epsilon and the layer normalised with eps = 1. Nothing raises: the
        shapes are right, the loss goes down, and the answer is wrong by however
        much an epsilon of 1 moves a normalisation.

        It sat in the `unaligned` bucket, which says *these lists cannot be lined
        up* and then says nothing else — the same place `F.normalize` was hiding a
        missing `out=`. **Clearing the vague reason is what shows the specific one.**

        `bias`, `device` and `dtype` are torch's last three. They are carried and
        refused rather than left out, so a positional call that reaches them lands
        on the argument it was aimed at.
        """
        super().__init__()
        _no_device_dtype("TransformerEncoderLayer", device, dtype)
        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        self.norm_first = norm_first
        self.activation = relu if activation == "relu" else (
            _gelu if activation == "gelu" else activation)

    def _sa(self, x, mask):
        return self.dropout(self.self_attn(x, attn_mask=mask, need_weights=False)[0])

    def _ff(self, x):
        return self.dropout(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src, src_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa(self.norm1(x), src_mask)
            x = x + self._ff(self.norm2(x))
        else:
            x = self.norm1(x + self._sa(x, src_mask))
            x = self.norm2(x + self._ff(x))
        return x


class TransformerEncoder(Module):
    """The same layer stacked. Named `layers.N.…` as in torch."""

    def __init__(self, encoder_layer, num_layers, norm=None,
                 enable_nested_tensor=True, mask_check=True):
        """**Both of torch's last two are accepted and unused, and neither can
        change an answer.**

        `enable_nested_tensor` asks for the fast path that packs a padded batch into
        a nested tensor — a representation, not a computation, and torch falls back
        to the ordinary path whenever it cannot use it. There are no nested tensors
        here, so this is that fallback permanently.

        `mask_check` asks torch to *validate* the mask before using it. Ours
        validates unconditionally: turning a check off to go faster is a trade there
        is nothing to trade here.

        Same standing as `foreach` on the optimizers — an argument that cannot change
        the answer is not a capability being faked. The seats are torch's, so a
        positional call reaching them lands where torch lands.
        """
        super().__init__()
        del enable_nested_tensor, mask_check
        import copy as _copy
        self.layers = ModuleList([_copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self._modules["layers"] = self.layers
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.norm(x) if self.norm is not None else x



class TransformerDecoderLayer(Module):
    """Self-attention → **attention over the encoder** → feed-forward.

    The one difference from the encoder layer is in the middle —
    `multihead_attn` looks at the encoder's output (the memory) rather than at
    itself. In translation this is where "the sentence written so far" and "the
    source" are looked at together.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        """torch's order — see `TransformerEncoderLayer`, which had the same
        two seats the other way round."""
        super().__init__()
        _no_device_dtype("TransformerDecoderLayer", device, dtype)
        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        self.norm_first = norm_first
        self.activation = relu if activation == "relu" else (
            _gelu if activation == "gelu" else activation)

    def _sa(self, x, mask):
        return self.dropout(self.self_attn(x, attn_mask=mask, need_weights=False)[0])

    def _mha(self, x, memory, mask):
        return self.dropout(
            self.multihead_attn(x, memory, memory, attn_mask=mask, need_weights=False)[0])

    def _ff(self, x):
        return self.dropout(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = tgt
        if self.norm_first:
            x = x + self._sa(self.norm1(x), tgt_mask)
            x = x + self._mha(self.norm2(x), memory, memory_mask)
            x = x + self._ff(self.norm3(x))
        else:
            x = self.norm1(x + self._sa(x, tgt_mask))
            x = self.norm2(x + self._mha(x, memory, memory_mask))
            x = self.norm3(x + self._ff(x))
        return x


class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        import copy as _copy
        self.layers = ModuleList([_copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self._modules["layers"] = self.layers
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(x) if self.norm is not None else x


class Transformer(Module):
    """The encoder and the decoder together. The whole diagram from "Attention
    Is All You Need"."""

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, bias=True, device=None, dtype=None):
        """torch's order, with `custom_encoder` and `custom_decoder` in their seats.

        Those two let a caller hand in an assembled stack instead of the one built
        here. Left out, torch's eighth and ninth positions landed on
        `layer_norm_eps` and `batch_first` — the same shift as the layers above,
        two arguments wide.
        """
        super().__init__()
        _no_device_dtype("Transformer", device, dtype)
        common = dict(dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                      batch_first=batch_first, norm_first=norm_first,
                      layer_norm_eps=layer_norm_eps, bias=bias)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, **common), num_encoder_layers,
            LayerNorm(d_model, eps=layer_norm_eps))
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, **common), num_decoder_layers,
            LayerNorm(d_model, eps=layer_norm_eps))
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, mask=src_mask)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        """A **float** mask with the upper triangle filled with -inf. It is
        added.

        Chapter 32's "do not let it see the future" is this one line.
        """
        m = _np.zeros((sz, sz), dtype=_DEFAULT_DTYPE)
        m[_np.triu_indices(sz, 1)] = -_np.inf
        return Tensor(m)


nn.Module = Module
nn.Parameter = Parameter
nn.Linear = Linear
# `nn.ReLU` is assigned after the class, which now lives below `_Activation` —
# see the note there.
nn.Sigmoid = Sigmoid
nn.Tanh = Tanh
nn.Flatten = Flatten
nn.Identity = Identity
nn.Dropout = Dropout
nn.Conv2d = Conv2d
nn.Embedding = Embedding
nn.LayerNorm = LayerNorm
nn.BatchNorm2d = BatchNorm2d
class _Activation(Module):
    """An activation layer — it has no state, so it wraps one function.

    **`inplace` is torch's and thirteen layers here did not take it.**
    `nn.ReLU(inplace=True)` is a line every torch model writes, and it stopped with
    a `TypeError` about the argument count. It is not decoration: in place the
    activation writes into the tensor it was handed and hands back **the same
    object**, which is how a deep network holds its memory down.

    **It is taken only where torch takes it**, which is not everywhere on this base.
    `ReLU`, `ReLU6`, `SELU`, `SiLU`, `Mish`, `Hardsigmoid` and `Hardswish` have one;
    `LogSigmoid`, `Softsign` and `Tanhshrink` do not. Offering it on all of them
    would be an argument torch refuses — the mirror of the fault this library keeps
    finding, since *accepted where the authority declines* misleads exactly as much
    as *accepted and inert*.

    Which is which is read from the function rather than written down: a name whose
    function takes `inplace` gets one. A `fn_inplace` table beside `fn` stood here
    for a few hours and said the same thing twice — the second copy also said six of
    these have no in-place form, which was true of the functions as they were that
    morning and not of the operation. `_ops` gives every one of them a write-back
    now, so the table would have gone on refusing what the library had learned to do.
    """

    fn = staticmethod(relu)

    def __init__(self, inplace=False):
        super().__init__()
        if inplace and not self._takes_inplace():
            raise TypeError(
                f"{type(self).__name__}() got an unexpected keyword argument "
                "'inplace' — torch does not give this one an in-place form either")
        self.inplace = inplace

    @classmethod
    def _takes_inplace(cls):
        try:
            return "inplace" in _inspect.signature(cls.fn).parameters
        except (TypeError, ValueError):
            return False

    def forward(self, x):
        if self.inplace:
            return type(self).fn(x, inplace=True)
        return type(self).fn(x)

    def extra_repr(self):
        return "inplace=True" if self.inplace else ""


class ReLU(_Activation):
    """**Moved down to here from the top of the file** so that it can share
    `_Activation`, which is defined between the two. Nothing referenced it in
    between — checked before moving, since a name defined twice in one module is
    the shape `tests/test_one_definition.py` exists to catch."""

    fn = staticmethod(relu)


nn.ReLU = ReLU


class GELU(Module):
    """**It takes an argument** — `approximate='tanh'` is a different formula
    and a different value.

    Left as an `_Activation` shell, `nn.GELU('tanh')` stops with
    `Module.__init__() takes 1 positional argument`. Better than quietly
    discarding an argument that does not exist, and saying something torch has is
    absent is still a divergence.
    """

    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return gelu(x, self.approximate)


class SiLU(_Activation):
    fn = staticmethod(silu)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        """`inplace` — see `_Activation`."""
        super().__init__()
        self.negative_slope, self.inplace = negative_slope, inplace

    def forward(self, x):
        return leaky_relu(x, self.negative_slope, inplace=self.inplace)

    def extra_repr(self):
        tail = ", inplace=True" if self.inplace else ""
        return f"negative_slope={self.negative_slope}{tail}"


class ELU(Module):
    def __init__(self, alpha=1.0, inplace=False):
        """`inplace` — see `_Activation`."""
        super().__init__()
        self.alpha, self.inplace = alpha, inplace

    def forward(self, x):
        return elu(x, self.alpha, inplace=self.inplace)

    def extra_repr(self):
        tail = ", inplace=True" if self.inplace else ""
        return f"alpha={self.alpha}{tail}"


# ── the stateless activation layers. Each wraps one function. ───────────────
#
# A wrapper **calling the wrong function** is this family's only failure mode,
# and it is invisible to the eye and diverges only in the values — which is why
# the golden asks about the function form and the layer form separately.

class Hardsigmoid(_Activation):
    fn = staticmethod(hardsigmoid)


class Hardswish(_Activation):
    fn = staticmethod(hardswish)


class LogSigmoid(_Activation):
    fn = staticmethod(logsigmoid)


class Mish(_Activation):
    fn = staticmethod(mish)


class ReLU6(_Activation):
    fn = staticmethod(relu6)


class SELU(_Activation):
    fn = staticmethod(selu)


class Softsign(_Activation):
    fn = staticmethod(softsign)


class Tanhshrink(_Activation):
    fn = staticmethod(tanhshrink)


class CELU(Module):
    def __init__(self, alpha=1.0, inplace=False):
        """`inplace` — see `_Activation`."""
        super().__init__()
        self.alpha, self.inplace = alpha, inplace

    def forward(self, x):
        return celu(x, self.alpha, inplace=self.inplace)

    def extra_repr(self):
        tail = ", inplace=True" if self.inplace else ""
        return f"alpha={self.alpha}{tail}"


class Hardshrink(Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return hardshrink(x, self.lambd)


class Softshrink(Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return softshrink(x, self.lambd)


class Hardtanh(Module):
    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False,
                 min_value=None, max_value=None):
        """`min_value` and `max_value` are torch's **deprecated** spellings and it
        still takes them, so a positional call reaching that far lands on them.
        Carried and forwarded rather than left out, which is what keeps the seats
        lined up; `inplace` — see `_Activation`.
        """
        super().__init__()
        self.min_val = min_val if min_value is None else min_value
        self.max_val = max_val if max_value is None else max_value
        self.inplace = inplace

    def forward(self, x):
        return hardtanh(x, self.min_val, self.max_val, inplace=self.inplace)

    def extra_repr(self):
        tail = ", inplace=True" if self.inplace else ""
        return f"min_val={self.min_val}, max_val={self.max_val}{tail}"


class Softplus(Module):
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.beta, self.threshold = beta, threshold

    def forward(self, x):
        return softplus(x, self.beta, self.threshold)


class Threshold(Module):
    def __init__(self, threshold, value, inplace=False):
        """`inplace` — see `_Activation`."""
        super().__init__()
        self.threshold, self.value, self.inplace = threshold, value, inplace

    def forward(self, x):
        return threshold_fn(x, self.threshold, self.value, inplace=self.inplace)

    def extra_repr(self):
        tail = ", inplace=True" if self.inplace else ""
        return f"threshold={self.threshold}, value={self.value}{tail}"


class Softmin(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return softmin(x, dim=self.dim)


class GLU(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return glu(x, dim=self.dim)


class PReLU(Module):
    """It **learns** the slope on the negative side. The only one in this family
    with a parameter.

    The name `weight` becomes the `state_dict` key, so it has to match torch — a
    diverging name means somebody else's checkpoint is unreadable.
    """

    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
        super().__init__()
        _no_device_dtype("PReLU", device, dtype)
        self.num_parameters = num_parameters
        self.weight = Parameter(_np.full(num_parameters, init, dtype=_DEFAULT_DTYPE))

    def forward(self, x):
        return prelu(x, self.weight)


class GroupNorm(Module):
    """Normalise with the channels bundled into groups. **Used instead of
    BatchNorm when the batch is small.**

    BatchNorm uses batch statistics, so at a batch of 1 or 2 those statistics are
    not trustworthy. This one bundles within a single sample, so it is
    independent of the batch size.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None, *,
                 bias=True):
        """See `BatchNorm2d` on `bias` — it drops the shift and keeps the scale."""
        super().__init__()
        _no_device_dtype("GroupNorm", device, dtype)
        self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
        self.weight = (Parameter(_np.ones(num_channels, dtype=_DEFAULT_DTYPE))
                       if affine else None)
        self.bias = (Parameter(_np.zeros(num_channels, dtype=_DEFAULT_DTYPE))
                     if affine and bias else None)

    def forward(self, x):
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class _InstanceNorm(Module):
    """Per sample and per channel. **The default is `affine=False`** — that is
    what torch does.

    The opposite of `BatchNorm`, which makes it a confusing place, and flipping
    the default swaps which layers have parameters and which do not, so the
    `state_dict` keys diverge wholesale.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False, device=None, dtype=None, *, bias=True):
        """See `BatchNorm2d` on `bias`."""
        super().__init__()
        _no_device_dtype(type(self).__name__, device, dtype)
        self.num_features, self.eps = num_features, eps
        self.weight = (Parameter(_np.ones(num_features, dtype=_DEFAULT_DTYPE))
                       if affine else None)
        self.bias = (Parameter(_np.zeros(num_features, dtype=_DEFAULT_DTYPE))
                     if affine and bias else None)
        # **An argument accepted and unused becomes a lie.**
        #
        # Given `track_running_stats=True`, torch registers three running
        # statistics and **actually uses them** in evaluation mode. This was
        # quietly ignoring it — three `state_dict` keys vanish wholesale, and
        # training is fine while evaluation alone produces different values.
        # Registering the buffers without the forward pass using them only moves
        # it to a later-discovered place where the keys match and the values are
        # wrong. So **what does not work is said not to work.**
        if track_running_stats:
            _unsupported("InstanceNorm with track_running_stats=True")

    def forward(self, x):
        return instance_norm(x, self.weight, self.bias, self.eps)


class InstanceNorm1d(_InstanceNorm):
    pass


class InstanceNorm2d(_InstanceNorm):
    pass


class InstanceNorm3d(_InstanceNorm):
    pass


class RMSNorm(Module):
    """**It does not subtract the mean.** That is the only difference from
    `LayerNorm`."""

    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        _no_device_dtype("RMSNorm", device, dtype)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = (Parameter(_np.ones(self.normalized_shape, dtype=_DEFAULT_DTYPE))
                       if elementwise_affine else None)

    def forward(self, x):
        return rms_norm(x, self.normalized_shape, self.weight, self.eps)


class _ConvTransposeND(Module):
    """A transposed convolution. **The weights are `(in, out, …)`** — reversed
    from `Conv2d`.

    With a square kernel the shape fits even reversed, so it diverges only in the
    values. The `state_dict` keys are `weight` and `bias`, the same as `Conv2d`'s,
    so loading by shape alone is quietly wrong.
    """

    nd = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode="zeros", device=None, dtype=None):
        """**torch puts `dilation` after `bias` here and before it in `Conv2d`.**

        The two are not the same list in a different spelling; the eighth position
        is `bias` in one and `dilation` in the other. Following torch means
        following that too — a tidier order of our own would read as agreement and
        land a positional call somewhere else.
        """
        super().__init__()
        _no_device_dtype(type(self).__name__, device, dtype)
        if padding_mode != "zeros":
            # torch refuses this itself: only `zeros` is implemented for a
            # transposed convolution, on either side.
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose"
                f"{type(self).nd}d")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.output_padding, self.groups = output_padding, groups
        self.dilation, self.padding_mode = dilation, padding_mode
        ks = _spread(kernel_size, type(self).nd)
        shape = (in_channels, out_channels // groups) + tuple(ks)
        numel = 1
        for size in ks:
            numel *= size
        bound = _conv_bound(out_channels, groups, numel)
        self.weight = Parameter(_rng.uniform(-bound, bound, shape).astype(_DEFAULT_DTYPE))
        self.bias = (Parameter(_rng.uniform(-bound, bound, out_channels)
                               .astype(_DEFAULT_DTYPE)) if bias else None)

    def forward(self, x):
        fn = {1: conv_transpose1d, 2: conv_transpose2d, 3: conv_transpose3d}[type(self).nd]
        return fn(x, self.weight, self.bias, self.stride, self.padding,
                  self.output_padding, self.groups, self.dilation)


class ConvTranspose1d(_ConvTransposeND):
    nd = 1


class ConvTranspose2d(_ConvTransposeND):
    nd = 2


class ConvTranspose3d(_ConvTransposeND):
    nd = 3


def _default_softmax_dim(ndim):
    """The axis torch chooses when `dim` is not given.

    **It is not `-1`.** It is 0 or 1 depending on the rank, and torch even warns
    at that point ("Implicit dimension choice for softmax has been deprecated").
    The rule was measured — rank 1 → 0, 2 → 1, 3 → **0**, 4 → 1.

    **Asked at rank 2 only, this defect is invisible.** There `dim=1` and
    `dim=-1` are the same axis, so a default of `-1` gives the same answer. It
    really was left that way, quietly folding the wrong axis at rank 3.
    """
    return 0 if ndim in (0, 1, 3) else 1


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dim = _default_softmax_dim(x.dim()) if self.dim is None else self.dim
        return softmax(x, dim=dim)


class LogSoftmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dim = _default_softmax_dim(x.dim()) if self.dim is None else self.dim
        return log_softmax(x, dim=dim)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        """torch's list. Four of the six were missing, and the pair that decides the
        *divisor* is the reason it is more than plumbing — see `F.avg_pool2d`."""
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.ceil_mode, self.count_include_pad = ceil_mode, count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x):
        return avg_pool2d(x, self.kernel_size, self.stride, self.padding,
                          self.ceil_mode, self.count_include_pad,
                          self.divisor_override)


class _PoolND(Module):
    """The body of the layers that fold over a fixed window. **Only which
    function they call differs.**

    Writing a `forward` per layer means the day comes when one of them calls a
    different function, and that diverges only in the values — the activations
    had the function form and the layer form asked separately for the same
    reason.
    """

    fn = staticmethod(avg_pool2d)
    adaptive = False

    def __init__(self, size, stride=None):
        super().__init__()
        self.size, self.stride = size, stride

    def forward(self, x):
        fn = type(self).fn
        return fn(x, self.size) if type(self).adaptive else fn(x, self.size, self.stride)


class AvgPool1d(_PoolND):
    fn = staticmethod(avg_pool1d)


class AvgPool3d(_PoolND):
    fn = staticmethod(avg_pool3d)


class AdaptiveAvgPool1d(_PoolND):
    fn = staticmethod(adaptive_avg_pool1d)
    adaptive = True


class AdaptiveAvgPool3d(_PoolND):
    fn = staticmethod(adaptive_avg_pool3d)
    adaptive = True


class _AdaptiveMaxPoolND(_PoolND):
    """Adaptive max pooling. **It takes `return_indices`** — an argument the
    average side does not have."""

    adaptive = True

    def __init__(self, size, return_indices=False):
        super().__init__(size)
        self.return_indices = return_indices

    def forward(self, x):
        return type(self).fn(x, self.size, return_indices=self.return_indices)


class AdaptiveMaxPool1d(_AdaptiveMaxPoolND):
    fn = staticmethod(adaptive_max_pool1d)


class AdaptiveMaxPool2d(_AdaptiveMaxPoolND):
    fn = staticmethod(adaptive_max_pool2d)


class AdaptiveMaxPool3d(_AdaptiveMaxPoolND):
    fn = staticmethod(adaptive_max_pool3d)


class LPPool1d(Module):
    def __init__(self, norm_type, kernel_size, stride=None, ceil_mode=False):
        """`ceil_mode` adds the trailing window that rounding up allows, **clipped
        to the input**: with no padding there is nothing to pad with, so torch folds
        over the cells that are really there. It reached `avg_pool1d` and stopped
        there, which is why all three of these were short by exactly one argument."""
        super().__init__()
        self.norm_type, self.kernel_size, self.stride = norm_type, kernel_size, stride
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return lp_pool1d(x, self.norm_type, self.kernel_size, self.stride,
                         self.ceil_mode)


class LPPool2d(LPPool1d):
    def forward(self, x):
        return lp_pool2d(x, self.norm_type, self.kernel_size, self.stride,
                         self.ceil_mode)


class LPPool3d(LPPool1d):
    def forward(self, x):
        return lp_pool3d(x, self.norm_type, self.kernel_size, self.stride,
                         self.ceil_mode)


ASMoutput = _collections.namedtuple("ASMoutput", ["output", "loss"])


class AdaptiveLogSoftmaxWithLoss(Module):
    """A softmax for when there are very many tokens. **It makes the frequent
    ones cheap.**

    With a vocabulary of hundreds of thousands, the final linear layer alone is
    larger than the model. Here the tokens are bundled by frequency; the leading
    bundle (the `shortlist`) comes straight out of the head, and the later
    bundles come out as **the probability the head assigned to that bundle × the
    probability within the bundle.** The further back the bundle, the narrower
    its intermediate dimension, divided down by `div_value` — less room spent on
    rare tokens.

    ## What was measured and written down

    - **The defaults are `div_value=4.0` and `head_bias=False`.** Asked assuming
      2.0, the tail layers come out a wholly different shape
      (`tests/probe_asm.py`).
    - The intermediate dimension is `in_features // div_value**(i+1)` and **can
      reach 0.** torch builds an empty layer there too — it is not blocked.
    - `forward` produces a named tuple `(output, loss)`. `output` is the log
      probability at the target position and `loss` is its mean negated.
    """

    def __init__(self, in_features, n_classes, cutoffs, div_value=4.0,
                 head_bias=False, device=None, dtype=None):
        super().__init__()
        _no_device_dtype("AdaptiveLogSoftmaxWithLoss", device, dtype)
        cutoffs = list(cutoffs)
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(in_features, self.head_size, bias=head_bias)
        tail = []
        for i in range(self.n_clusters):
            hidden = int(in_features // (div_value ** (i + 1)))
            out = self.cutoffs[i + 1] - self.cutoffs[i]
            tail.append(Sequential(Linear(in_features, hidden, bias=False),
                                   Linear(hidden, out, bias=False)))
        self.tail = ModuleList(tail)

    def log_prob(self, x):
        """The log probability of every token, `(N, n_classes)`.

        The probability of a later bundle **has the head's log probability for
        that bundle added to it** — a multiplication is an addition in the log
        domain, which is what keeps each row summing to 1.
        """
        head = log_softmax(self.head(x), dim=1)
        parts = [head[:, :self.shortlist_size]]
        for i in range(self.n_clusters):
            inside = log_softmax(self.tail[i](x), dim=1)
            picked = head[:, self.shortlist_size + i:self.shortlist_size + i + 1]
            parts.append(inside + picked)
        return cat(parts, 1)

    def forward(self, x, target):
        lp = self.log_prob(x)
        idx = _np.asarray(target.data if isinstance(target, Tensor) else target)
        picked = lp.gather(1, Tensor(idx.reshape(-1, 1).astype(_np.int64)))
        out = picked.reshape(-1)
        return ASMoutput(out, -out.mean())

    def predict(self, x):
        return self.log_prob(x).argmax(dim=1)


class CTCLoss(Module):
    """A loss that connects audio to characters **without aligning them.**

    `forward` takes four arguments — the log probabilities, the target, the input
    lengths and the target lengths. The lengths differ per sample and that is the
    point of this loss, so it cannot take two like the others.

    **The `repr` is empty.** torch's `extra_repr` produces nothing (measured).
    """

    def __init__(self, blank=0, reduction="mean", zero_infinity=False):
        super().__init__()
        self.blank, self.reduction = blank, reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return ctc_loss(log_probs, targets, input_lengths, target_lengths,
                        self.blank, self.reduction, self.zero_infinity)

    def __repr__(self):
        return "CTCLoss()"


class _FractionalMaxPoolND(Module):
    """Max pooling that jitters the window positions at random.

    A fixed window puts the grid in the same place every time, so it sees only
    the patterns that fit that grid. Here the window start jitters per sample, so
    passing the same layer several times sees different grids — used as
    regularisation during training.

    **The `repr` is empty.** torch's `extra_repr` produces nothing, so it prints
    as `()` alone (measured). Not an imitation — that side is like that.
    """

    fn = None
    dim = 0

    def __init__(self, kernel_size, output_size=None, output_ratio=None,
                 return_indices=False, _random_samples=None):
        super().__init__()
        # **Exactly one of the two is accepted** — torch stops in the
        # constructor. Being lenient means that when both are given, which one
        # won shows only in the values, and when neither is given nobody can read
        # where the size came from.
        if (output_size is None) == (output_ratio is None):
            raise ValueError(
                "FractionalMaxPool takes either output_size or output_ratio, not both."
                "\n(torch: FractionalMaxPool2d requires specifying either "
                "an output size, or a pooling ratio)")
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices
        self._random_samples = _random_samples

    def forward(self, x):
        return type(self).fn(x, self.kernel_size, self.output_size,
                             self.output_ratio, self.return_indices,
                             self._random_samples)

    def __repr__(self):
        return f"FractionalMaxPool{type(self).dim}d()"


class FractionalMaxPool2d(_FractionalMaxPoolND):
    fn = staticmethod(fractional_max_pool2d)
    dim = 2


class FractionalMaxPool3d(_FractionalMaxPoolND):
    fn = staticmethod(fractional_max_pool3d)
    dim = 3


class AdaptiveAvgPool2d(_PoolND):
    """**The output size does not have to be 1.**

    It used to take 1 and refuse everything else. That was because averaging the
    whole thing through `_pool_all` was all there was, and now that there is
    machinery for windows placed differently per position there is no reason to
    refuse — the refusal was a different rule rather than a form of imitation.
    """

    fn = staticmethod(adaptive_avg_pool2d)
    adaptive = True


class Unflatten(Module):
    """Spread one axis into several. **The axes after it stay where they are.**

    It was written as `shape[:dim] + sizes` alone — the right answer when the
    axis being spread is the last one, so it went unseen for a long time, and
    spreading a middle axis loses everything behind it wholesale. The element
    count does not match so `reshape` stops, which means it is not quietly wrong,
    and the place it stops is far from the cause.
    """

    def __init__(self, dim, unflattened_size):
        super().__init__()
        self.dim, self.unflattened_size = dim, tuple(unflattened_size)

    def forward(self, x):
        shape = tuple(x.data.shape)
        dim = self.dim if self.dim >= 0 else self.dim + len(shape)
        return x.reshape(shape[:dim] + self.unflattened_size + shape[dim + 1:])


class L1Loss(_WrittenLoss):
    def __init__(self, reduction="mean"):
        """No `weight` in torch either — the same correction as `MSELoss`."""
        super().__init__(reduction)

    def forward(self, pred, target):
        return l1_loss(pred, target, self.reduction)


class SmoothL1Loss(_WrittenLoss):
    """**`reduction` comes first, as in torch.**

    torch's live arguments are `(reduction, beta)` — the deprecated `size_average`
    and `reduce` sit in front of them and nobody passes those. This class took
    `(beta, reduction)`, so `SmoothL1Loss("sum")` set `beta="sum"` here and
    `reduction="sum"` there. Nothing raised at construction; the failure arrived
    later, wherever `beta` was used as a number.

    borch.ts had it right, and the divergence read as the two sisters disagreeing
    rather than as one of them being wrong — which is what a comparison between two
    of our own libraries can never tell apart. It took asking real torch.
    """

    def __init__(self, reduction="mean", beta=1.0):
        super().__init__(reduction)
        self.beta = beta

    def forward(self, pred, target):
        return smooth_l1_loss(pred, target, self.beta, self.reduction)


class NLLLoss(_WrittenLoss):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean"):
        """torch's order — `weight`, then `ignore_index`, then `reduction`."""
        super().__init__(reduction, weight=weight, ignore_index=ignore_index)

    def forward(self, log_probs, target):
        idx = target.data.astype(int)
        # The gather happens inside `nll_loss`, so the sentinel has to be replaced
        # before it gets there — the same ordering `CrossEntropyLoss` needs, and for
        # the same reason: -100 is not an index.
        safe = _np.where(idx == self.ignore_index, 0, idx)
        each = nll_loss(log_probs, _wrap(safe), reduction="none")
        return _reduce_ignoring(each, idx, self.ignore_index, self.reduction)


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None, *, bias=True):
        """See `BatchNorm2d` on `affine` and `bias`."""
        super().__init__()
        _no_device_dtype("BatchNorm1d", device, dtype)
        if not track_running_stats:
            _unsupported("BatchNorm1d with track_running_stats=False")
        self.eps, self.momentum, self.affine = eps, momentum, affine
        self.weight = (Parameter(_np.ones(num_features, dtype=_DEFAULT_DTYPE))
                       if affine else None)
        self.bias = (Parameter(_np.zeros(num_features, dtype=_DEFAULT_DTYPE))
                     if affine and bias else None)
        self.register_buffer("running_mean", _np.zeros(num_features, dtype=_DEFAULT_DTYPE))
        self.register_buffer("running_var", _np.ones(num_features, dtype=_DEFAULT_DTYPE))
        self.register_buffer("num_batches_tracked", 0)

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            centered = x - mean
            var = (centered * centered).mean(dim=0)
            with no_grad():
                self.running_mean = ((1 - self.momentum) * self.running_mean
                                     + self.momentum * mean.data)
                self.running_var = ((1 - self.momentum) * self.running_var
                                    + self.momentum * x.data.var(axis=0, ddof=1))
                self.num_batches_tracked = self.num_batches_tracked + 1
            normed = centered / (var + self.eps) ** 0.5
        else:
            normed = ((x - Tensor(self.running_mean))
                      / Tensor(_np.sqrt(self.running_var + self.eps)))
        if self.weight is not None:
            normed = normed * self.weight
        return normed if self.bias is None else normed + self.bias


# ---- the 1-D and 3-D families
#
# **They were refusing stubs.** The sister library (webgpu) had the real thing
# and the core did not, which left open a direction where changing one `import`
# breaks the code — this project's promise is exactly the opposite.
#
# `conv2d` and `max_pool2d` do the arithmetic. No new im2col is written: two
# copies of the same computation means the day comes when one of them is fixed
# and they diverge.

class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros",
                 device=None, dtype=None):
        super().__init__()
        _no_device_dtype("Conv1d", device, dtype)
        if padding_mode not in _PADDING_MODES:
            raise ValueError(
                f"padding_mode has to be one of {_PADDING_MODES}, got {padding_mode!r}")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.dilation, self.groups, self.padding_mode = dilation, groups, padding_mode
        k = _spread(kernel_size, 1)[0]
        bound = _conv_bound(in_channels, groups, k)
        self.weight = Parameter(_rng.uniform(
            -bound, bound,
            (out_channels, in_channels // groups, k)).astype(_DEFAULT_DTYPE))
        self.bias = Parameter(
            _rng.uniform(-bound, bound, (out_channels,)).astype(_DEFAULT_DTYPE)) if bias else None

    def forward(self, x):
        x, padding = _conv_pad(x, self.padding, self.padding_mode, 1)
        return conv1d(x, self.weight, self.bias, self.stride, padding,
                      self.dilation, self.groups)

    def __repr__(self):
        return (f"Conv1d({self.in_channels}, {self.out_channels}, "
                f"kernel_size=({self.kernel_size},), stride=({self.stride},))")


class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros",
                 device=None, dtype=None):
        super().__init__()
        _no_device_dtype("Conv3d", device, dtype)
        if padding_mode not in _PADDING_MODES:
            raise ValueError(
                f"padding_mode has to be one of {_PADDING_MODES}, got {padding_mode!r}")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.dilation, self.groups, self.padding_mode = dilation, groups, padding_mode
        ks = _spread(kernel_size, 3)
        bound = _conv_bound(in_channels, groups, ks[0] * ks[1] * ks[2])
        shape = (out_channels, in_channels // groups) + tuple(ks)
        self.weight = Parameter(_rng.uniform(-bound, bound, shape).astype(_DEFAULT_DTYPE))
        self.bias = Parameter(
            _rng.uniform(-bound, bound, (out_channels,)).astype(_DEFAULT_DTYPE)) if bias else None

    def forward(self, x):
        x, padding = _conv_pad(x, self.padding, self.padding_mode, 3)
        return conv3d(x, self.weight, self.bias, self.stride, padding,
                      self.dilation, self.groups)

    def __repr__(self):
        return (f"Conv3d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride})")


def _pool_repr(name, layer):
    """torch's own repr, with the three new settings shown only when they are set.

    Printing them always would be torch's exact wording and would also change the
    output of every existing default construction — nothing in the golden holds a
    `MaxPool` repr today, and adding an argument is a poor moment to move a string
    that nothing is watching.
    """
    parts = [f"kernel_size={layer.kernel_size}", f"stride={layer.stride}"]
    for field, default in (("padding", 0), ("dilation", 1), ("ceil_mode", False)):
        value = getattr(layer, field)
        if value != default:
            parts.append(f"{field}={value}")
    return f"{name}({', '.join(parts)})"


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation
        self.return_indices, self.ceil_mode = return_indices, ceil_mode

    def forward(self, x):
        return max_pool1d(x, self.kernel_size, self.stride, self.padding,
                          self.dilation, self.return_indices, self.ceil_mode)

    def __repr__(self):
        return _pool_repr("MaxPool1d", self)


class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation
        self.return_indices, self.ceil_mode = return_indices, ceil_mode

    def forward(self, x):
        return max_pool3d(x, self.kernel_size, self.stride, self.padding,
                          self.dilation, self.return_indices, self.ceil_mode)

    def __repr__(self):
        return _pool_repr("MaxPool3d", self)


class _MaxUnpoolND(Module):
    """Put values back at the positions `MaxPool` chose. Only the function
    differs per dimensionality.

    **`forward` takes two arguments.** A different shape from the other layers,
    so it cannot simply go into a `Sequential`, and torch is the same — the
    positions have to flow alongside the values, and hiding them inside the layer
    means using somebody else's positions when the same layer is used twice.
    """

    fn = None
    dim = 0

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        # **Carried spread out per axis.** torch carries it that way and the
        # tuple appears in the `repr` as-is — carried as a single number it
        # prints as `kernel_size=2` and diverges.
        n = type(self).dim
        self.kernel_size = tuple(_spread(kernel_size, n))
        self.stride = tuple(_spread(kernel_size if stride is None else stride, n))
        self.padding = tuple(_spread(padding, n))

    def forward(self, x, indices, output_size=None):
        return type(self).fn(x, indices, self.kernel_size, self.stride,
                             self.padding, output_size)

    def __repr__(self):
        return (f"MaxUnpool{type(self).dim}d(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")


class MaxUnpool1d(_MaxUnpoolND):
    fn = staticmethod(max_unpool1d)
    dim = 1


class MaxUnpool2d(_MaxUnpoolND):
    fn = staticmethod(max_unpool2d)
    dim = 2


class MaxUnpool3d(_MaxUnpoolND):
    fn = staticmethod(max_unpool3d)
    dim = 3


class BatchNorm3d(BatchNorm2d):
    """**The same code** as `BatchNorm2d` — it was fixed above to stop examining
    the rank, so (N,C,D,H,W) passes through unchanged. The sister library has the
    same structure."""


# --------------------------------------------------------- the rest of the layers

class Unfold(Module):
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.kernel_size, self.dilation = kernel_size, dilation
        self.padding, self.stride = padding, stride

    def forward(self, x):
        return unfold_im2col(x, self.kernel_size, self.dilation, self.padding,
                             self.stride)

    def __repr__(self):
        return (f"Unfold(kernel_size={self.kernel_size}, "
                f"dilation={self.dilation}, padding={self.padding}, "
                f"stride={self.stride})")


class Fold(Module):
    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.output_size, self.kernel_size = output_size, kernel_size
        self.dilation, self.padding, self.stride = dilation, padding, stride

    def forward(self, x):
        return fold(x, self.output_size, self.kernel_size, self.dilation,
                    self.padding, self.stride)

    def __repr__(self):
        return (f"Fold(output_size={self.output_size}, "
                f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
                f"padding={self.padding}, stride={self.stride})")


class Bilinear(Module):
    """Mixes two inputs **at once.** The weights have three axes, so
    `(out, in1, in2)`."""

    def __init__(self, in1_features, in2_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        _no_device_dtype("Bilinear", device, dtype)
        self.in1_features, self.in2_features = in1_features, in2_features
        self.out_features = out_features
        bound = 1.0 / _math.sqrt(in1_features)
        self.weight = Parameter(_rng.uniform(
            -bound, bound, (out_features, in1_features, in2_features)
        ).astype(_DEFAULT_DTYPE))
        if bias:
            self.bias = Parameter(
                _rng.uniform(-bound, bound, out_features).astype(_DEFAULT_DTYPE))

    def forward(self, x1, x2):
        return bilinear(x1, x2, self.weight, getattr(self, "bias", None))

    def __repr__(self):
        return (f"Bilinear(in1_features={self.in1_features}, "
                f"in2_features={self.in2_features}, "
                f"out_features={self.out_features}, "
                f"bias={getattr(self, 'bias', None) is not None})")


class LocalResponseNorm(Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1.0):
        super().__init__()
        self.size, self.alpha, self.beta, self.k = size, alpha, beta, k

    def forward(self, x):
        return local_response_norm(x, self.size, self.alpha, self.beta, self.k)

    def __repr__(self):
        return (f"LocalResponseNorm({self.size}, alpha={self.alpha}, "
                f"beta={self.beta}, k={self.k})")


class Softmax2d(Module):
    """A softmax **along the channels** of `(N, C, H, W)`. The same as
    `softmax(dim=1)`."""

    def forward(self, x):
        return softmax(x, dim=1)

    def __repr__(self):
        return "Softmax2d()"


class RReLU(Module):
    def __init__(self, lower=1.0 / 8, upper=1.0 / 3, inplace=False):
        super().__init__()
        self.lower, self.upper, self.inplace = lower, upper, inplace

    def forward(self, x):
        # **`inplace` was accepted here and dropped.** It was stored, printed by the
        # repr above, and never passed on — so a caller relying on the input changing
        # got an unchanged input and no complaint. Found while giving the other
        # eleven activations the argument they had never taken at all.
        return rrelu(x, self.lower, self.upper, self.training,
                     inplace=self.inplace)

    def __repr__(self):
        return f"RReLU(lower={self.lower}, upper={self.upper})"


class _Upsampling(Module):
    """Two old names. **`UpsamplingBilinear2d` is `align_corners=True`** —
    different from `Upsample(mode='bilinear')`'s default."""

    _mode = "nearest"
    _corners = None

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return interpolate(x, size=self.size, scale_factor=self.scale_factor,
                           mode=self._mode, align_corners=self._corners)

    def __repr__(self):
        return (f"{type(self).__name__}(scale_factor="
                f"{float(self.scale_factor)}, mode={self._mode!r})")


class UpsamplingNearest2d(_Upsampling):
    _mode = "nearest"


class UpsamplingBilinear2d(_Upsampling):
    _mode = "bilinear"
    _corners = True


class EmbeddingBag(Module):
    """One row per bag. Selecting from the table and **combining** is all one
    layer.

    Given `offsets`, a 1-D row of indices is cut into bags — the shape for when
    the bags have differing lengths.
    """

    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, mode="mean", sparse=False, _weight=None,
                 include_last_offset=False, padding_idx=None, device=None,
                 dtype=None):
        """**`mode` sits sixth, where torch has it.**

        It used to be third, so `EmbeddingBag(10, 3, "sum")` set `max_norm="sum"`
        in torch and the mode here. Both sides then build a layer and return bags
        of the right shape, and only the numbers differ.
        """
        super().__init__()
        _no_device_dtype("EmbeddingBag", device, dtype)
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.max_norm, self.norm_type = max_norm, norm_type
        self.scale_grad_by_freq, self.sparse = scale_grad_by_freq, sparse
        self.mode = mode
        self.include_last_offset, self.padding_idx = include_last_offset, padding_idx
        if _weight is None:
            table = _rng.standard_normal(
                (num_embeddings, embedding_dim)).astype(_DEFAULT_DTYPE)
        else:
            table = _np.asarray(_weight.data if hasattr(_weight, "data") else _weight,
                                dtype=_DEFAULT_DTYPE)
            if table.shape != (num_embeddings, embedding_dim):
                raise ValueError(
                    "Shape of weight does not match num_embeddings and embedding_dim")
        self.weight = Parameter(table)

    def forward(self, idx, offsets=None, per_sample_weights=None):
        # `F.embedding_bag` does the computation — the layer and the function
        # are not kept as two copies. **By keyword**: this call passed `mode`
        # fourth, and the day the function took torch's order it would have been
        # handing it to `max_norm`.
        return embedding_bag(
            idx, self.weight, offsets, max_norm=self.max_norm,
            norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq,
            mode=self.mode, sparse=self.sparse,
            per_sample_weights=per_sample_weights,
            include_last_offset=self.include_last_offset,
            padding_idx=self.padding_idx)

    def __repr__(self):
        return (f"EmbeddingBag({self.num_embeddings}, {self.embedding_dim}, "
                f"mode={self.mode!r})")


for _cls in (Unfold, Fold, Bilinear, LocalResponseNorm, Softmax2d, RReLU,
             UpsamplingNearest2d, UpsamplingBilinear2d, EmbeddingBag):
    setattr(nn, _cls.__name__, _cls)


# ------------------------ rearrangement and channel-wise dropout
#
# All eight layers call one function. Only the arguments passed and the
# characters printed differ.

class _Rearrange(Module):
    _fn = None
    _arg = "factor"

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return type(self)._fn(x, self.value)

    def __repr__(self):
        return f"{type(self).__name__}({self._arg}={self.value})"


class PixelShuffle(_Rearrange):
    _fn = staticmethod(pixel_shuffle)
    _arg = "upscale_factor"


class PixelUnshuffle(_Rearrange):
    _fn = staticmethod(pixel_unshuffle)
    _arg = "downscale_factor"


class ChannelShuffle(_Rearrange):
    _fn = staticmethod(channel_shuffle)
    _arg = "groups"


class _FeatureDropout(Module):
    """The ones that drop whole channels. **`inplace` is printed too** — torch
    does that."""

    _fn = None

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p, self.inplace = p, inplace

    def forward(self, x):
        # `self.inplace` was stored and **printed in the `repr` below** while
        # `forward` never passed it on — `AlphaDropout(inplace=True)` reported a
        # flag it did not act on, across all five of these classes. The `repr` was
        # the only honest-looking part of it.
        return type(self)._fn(x, self.p, self.training, self.inplace)

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p}, inplace={self.inplace})"


def _make_feature_dropouts():
    table = (("Dropout1d", dropout1d), ("Dropout2d", dropout2d),
             ("Dropout3d", dropout3d), ("AlphaDropout", alpha_dropout),
             ("FeatureAlphaDropout", feature_alpha_dropout))
    return {name: type(name, (_FeatureDropout,), {"_fn": staticmethod(fn)})
            for name, fn in table}


for _name, _cls in {"PixelShuffle": PixelShuffle,
                    "PixelUnshuffle": PixelUnshuffle,
                    "ChannelShuffle": ChannelShuffle,
                    **_make_feature_dropouts()}.items():
    globals()[_name] = _cls
    setattr(nn, _name, _cls)


# ------------------------------------------------------------- the lazy layers
#
# **The shape is worked out at the first forward.** `nn.LazyLinear(3)` does not
# take `in_features` and decides from the first value that passes through — it
# removes counting by hand how many channels come out after a convolution, which
# is why real code uses it often.
#
# **Materialising changes the class.** That is the heart of the contract and it
# does not come out of guessing. After the first forward the object is no longer
# a `LazyLinear` but a `Linear` — the name and `isinstance` both change and the
# method `has_uninitialized_params` disappears entirely (confirmed by asking real
# torch). Handled with a flag the name does not change, and then the `repr`
# diverges.

class UninitializedParameter(Parameter):
    """A parameter whose shape is not known yet.

    **It does exist.** `parameters()` produces it and it has a `state_dict` key —
    so it can go into an optimiser before materialising (torch allows that and
    there is code written in that order). Asking for its shape or computing with
    it throws, though.
    """

    def __init__(self):
        super().__init__(_np.zeros((0,), dtype=_DEFAULT_DTYPE))

    @property
    def shape(self):
        raise RuntimeError(
            "Can't access the shape of an uninitialized parameter or buffer. "
            "This error usually happens in `load_state_dict` when the parameter "
            "has not been materialized — run one pass through it first.")

    def __repr__(self):
        return "<UninitializedParameter>"

    def _refuse(self, *_a, **_k):
        raise ValueError(
            "Attempted to use an uninitialized parameter — run one pass through it first.")

    __add__ = __radd__ = __mul__ = __rmul__ = _refuse
    __sub__ = __rsub__ = __truediv__ = __rtruediv__ = _refuse
    __matmul__ = __rmatmul__ = __pow__ = _refuse


class UninitializedBuffer(UninitializedParameter):
    def __repr__(self):
        return "<UninitializedBuffer>"


def Buffer(data):
    """`nn.Buffer(t)` — a mark saying this value is not trained and is saved.

    In torch this is the tensor itself (`isinstance(nn.Buffer(t), Tensor)` is
    true). What actually reads the mark is `register_buffer`, so it is handed
    back unchanged here.
    """
    return data


class _Lazy(Module):
    """The root of the lazy layers. `_becomes` is the class it turns into and
    `_infer` reads the shape.

    Materialising happens in `forward` — so that it passes the same place whether
    it is inside a `Sequential` or called by hand. Once materialised it **swaps
    itself for the real layer** and calls again.
    """

    _becomes = None
    _names = ()

    def __init__(self, *args, **kw):
        super().__init__()
        bound = type(self)._bind(*args, **kw)
        self._lazy_args, self._lazy_kw = list(args), dict(kw)
        self.weight = UninitializedParameter()
        # **`bias` can arrive positionally.** This read `kw.get("bias", True)`, so
        # `LazyConv2d(4, 3, 1, 0, 1, 1, False)` — every argument in torch's order,
        # the bias turned off — built an uninitialized bias anyway and then threw it
        # away at materialisation. Nothing failed; the layer simply carried a
        # parameter for one forward pass. Binding against the real signature is what
        # makes the question answerable at all.
        if bound.arguments.get("bias", True):
            self.bias = UninitializedParameter()

    @classmethod
    def _bind(cls, *args, **kw):
        """Check the call against the signature this class actually declares.

        A lazy layer forwards everything to the class it becomes, so a wrong name
        used to travel all the way there and be refused in its name instead of this
        one. Bound here, the refusal says `LazyConv2d`.
        """
        signature = _inspect.signature(cls.__init__)
        return signature.bind(None, *args, **kw)

    def has_uninitialized_params(self):
        return True

    def _infer(self, x):
        """The first argument read off the shape. It differs per layer, so this
        is where they split."""
        raise NotImplementedError

    def forward(self, x):
        cls = type(self)._becomes
        real = cls(*self._infer(x), *self._lazy_args, **self._lazy_kw)
        # The insides are swapped out wholesale. Handing back a new object makes
        # it a different thing from the parameters already in the optimiser, and
        # then training runs and the weights do not move.
        self.__dict__.clear()
        self.__dict__.update(real.__dict__)
        self.__class__ = cls
        return self(x)

    def __repr__(self):
        inner = ", ".join(f"{n}={v}" for n, v in
                          zip(self._names, [0] + self._lazy_args))
        tail = ", bias=True" if self._lazy_kw.get("bias", True) else ", bias=False"
        return f"{type(self).__name__}({inner}{tail})"


class LazyLinear(_Lazy):
    _becomes = Linear
    _names = ("in_features", "out_features")

    def _infer(self, x):
        return (x.shape[-1],)


def _lazy_channels(x):
    """The channel count from (N, C, …). Shared by the convolutions and the
    normalisations."""
    return (x.shape[1],)


def _lazy_declaration(becomes, infers):
    """The `__init__` a lazy layer declares: **its target's, minus what it infers.**

    torch's rule exactly — `LazyConv2d` is `Conv2d` without `in_channels`,
    `LazyLinear` is `Linear` without `in_features`, and so on for all twelve.
    Checked against real torch rather than assumed.

    **Derived and not written down.** A hand-copied list here would be one more
    fact that is correct on the day it is pasted: every argument the six
    convolutions gained today would have had to be typed a second time, and the
    day one is missed the lazy layer takes torch's arguments in the wrong seats
    while the eager one takes them in the right ones. Nothing would compare the
    two — a variadic signature cannot be judged, which is why these twelve sat in
    the axis's uncomparable bucket while every other layer was measured.
    """
    parameters = list(_inspect.signature(becomes.__init__).parameters.values())
    kept = parameters[1 + infers:]
    declared = _inspect.Signature(
        [_inspect.Parameter("self", _inspect.Parameter.POSITIONAL_OR_KEYWORD)] + kept)
    names = tuple(p.name for p in parameters[1:])

    def __init__(self, *args, **kw):
        _Lazy.__init__(self, *args, **kw)

    __init__.__signature__ = declared
    return __init__, names


def _make_lazies():
    """Twelve are stamped out here. Only what they become and what they read
    differ."""
    table = (
        ("LazyConv1d", Conv1d),
        ("LazyConv2d", Conv2d),
        ("LazyConv3d", Conv3d),
        ("LazyConvTranspose1d", ConvTranspose1d),
        ("LazyConvTranspose2d", ConvTranspose2d),
        ("LazyConvTranspose3d", ConvTranspose3d),
        ("LazyBatchNorm1d", BatchNorm1d),
        ("LazyBatchNorm2d", BatchNorm2d),
        ("LazyBatchNorm3d", BatchNorm3d),
        ("LazyInstanceNorm1d", InstanceNorm1d),
        ("LazyInstanceNorm2d", InstanceNorm2d),
        ("LazyInstanceNorm3d", InstanceNorm3d),
    )
    made = {}
    for name, becomes in table:
        init, names = _lazy_declaration(becomes, 1)
        made[name] = type(name, (_Lazy,), {
            "_becomes": becomes, "_names": names, "__init__": init,
            "_infer": staticmethod(_lazy_channels),
        })
    return made


# **`LazyLinear` gets the same declaration as the other eleven.** It is written by
# hand above rather than generated, so it kept `_Lazy.__init__`'s `(*args, **kw)` and
# was the one lazy layer the signature axis could not compare — `variadic` there means
# *nothing was checked*, and the paragraph in `_lazy_declaration` about exactly that
# hazard was three lines away from the class it did not cover.
LazyLinear.__init__, LazyLinear._names = _lazy_declaration(Linear, 1)


for _name, _lazy_cls in {"LazyLinear": LazyLinear, **_make_lazies()}.items():
    globals()[_name] = _lazy_cls
    setattr(nn, _name, _lazy_cls)

nn.UninitializedParameter = UninitializedParameter
nn.UninitializedBuffer = UninitializedBuffer
nn.Buffer = Buffer


# --------------------------------------------------------------- the loss layers
#
# **They all have the same shape** — take the arguments at construction and hand
# them to the function at the call. That is all a torch loss layer does, so
# writing a `forward` per layer amounts to writing the same two lines thirteen
# times. Only the argument names are kept in a table and the rest is stamped out
# here.

class _GeneratedLoss(Module):
    """The root of the losses **built from the `_LOSSES` table**, not of all of
    them — the eight written out above descend from `_WrittenLoss` instead.

    Both classes were called `_Loss`. The name said nothing about which, and the
    answer was the reader's line number.

    `_fn` names the function and `_keys` names the arguments to pass.

    **The constructor is generated per subclass with torch's own parameter list**,
    rather than shared as `(*args, reduction="mean", **kw)`. That shared version
    paired positionals with `zip(self._keys, args)`, and `zip` stops at the shorter
    side — so everything past the last key **went nowhere and raised nothing**:

        HuberLoss(0.5, "sum")        → 0.25, computed with reduction="mean"
        HuberLoss(0.5, "sum", 99)    → accepted, three positionals against one key
        SoftMarginLoss(0.5)          → accepted, `_keys` is empty

    torch answers the first of those with `ValueError: 0.5 is not a valid value for
    reduction`. Ours returned a number, and the wrong one — a loss reduced by the
    mean when the caller asked for the sum. **A silently discarded argument is worse
    than a refused one**, which is this library's own rule pointing at itself.

    The generated signature also puts `reduction` where torch puts it, which is not
    always first: `HuberLoss(reduction, delta)` but `MarginRankingLoss(margin,
    reduction)`. Getting that from a table means the order is written down once.
    """

    _fn = None
    _keys = ()

    def __init__(self, **kw):
        super().__init__()
        self.reduction = kw.pop("reduction", "mean")
        self._opts = kw

    def forward(self, *inputs):
        return type(self)._fn(*inputs, reduction=self.reduction, **self._opts)

    def __repr__(self):
        # torch prints a loss layer **with no arguments** — even
        # `HuberLoss(delta=0.5)` comes out as `HuberLoss()` (measured). The
        # characters are part of the answer, so it is followed exactly.
        return f"{type(self).__name__}()"


# Each row is **torch's own parameter list, in torch's order**, with the deprecated
# `size_average` and `reduce` left out — torch documents both as dead and ignores them
# whenever `reduction` is given, and keeping them would put two arguments nobody
# passes in front of the ones everybody does.
#
# `"*"` marks the point after which torch takes keyword arguments only, and two of
# these have one at the very front: `GaussianNLLLoss` and
# `TripletMarginWithDistanceLoss` refuse positional arguments entirely in torch.
# Following that is the difference between a subset you can practise on and one that
# teaches a call real torch will reject.
_LOSSES = (
    ("HuberLoss", "huber_loss", ("reduction='mean'", "delta=1.0")),
    ("KLDivLoss", "kl_div", ("reduction='mean'", "log_target=False")),
    ("PoissonNLLLoss", "poisson_nll_loss",
     ("log_input=True", "full=False", "eps=1e-8", "reduction='mean'")),
    ("GaussianNLLLoss", "gaussian_nll_loss",
     ("*", "full=False", "eps=1e-6", "reduction='mean'")),
    ("MarginRankingLoss", "margin_ranking_loss", ("margin=0.0", "reduction='mean'")),
    ("CosineEmbeddingLoss", "cosine_embedding_loss",
     ("margin=0.0", "reduction='mean'")),
    ("HingeEmbeddingLoss", "hinge_embedding_loss", ("margin=1.0", "reduction='mean'")),
    ("SoftMarginLoss", "soft_margin_loss", ("reduction='mean'",)),
    ("TripletMarginLoss", "triplet_margin_loss",
     ("margin=1.0", "p=2.0", "eps=1e-6", "swap=False", "reduction='mean'")),
    ("TripletMarginWithDistanceLoss", "triplet_margin_with_distance_loss",
     ("*", "distance_function=None", "margin=1.0", "swap=False", "reduction='mean'")),
    ("MultiLabelSoftMarginLoss", "multilabel_soft_margin_loss",
     ("weight=None", "reduction='mean'")),
    ("MultiMarginLoss", "multi_margin_loss",
     ("p=1", "margin=1.0", "weight=None", "reduction='mean'")),
    ("MultiLabelMarginLoss", "multilabel_margin_loss", ("reduction='mean'",)),
)


def _loss_init(params):
    """A real `__init__` with `params` as its signature, built by `exec`.

    **Generated rather than hand-written, and generated rather than shared.** Thirteen
    hand-written constructors is the same two lines thirteen times, which is what the
    shared `*args` version was avoiding and it was right to. What it got wrong was
    reaching for `*args`: a signature that accepts everything cannot refuse anything,
    and `zip` then dropped whatever it had no key for.

    `exec` is what gives a **real** signature — one `inspect` reads, one that puts
    `reduction` where torch puts it, and one that raises `TypeError` on an argument
    too many instead of swallowing it. A generated function has that; a wrapper
    carrying `functools.wraps` or a hand-set `__signature__` only looks like it does,
    and this file has been bitten twice this week by things that only looked right.
    """
    names = [p.split("=")[0] for p in params if p != "*"]
    src = (f"def __init__(self, {', '.join(params)}):\n"
           f"    _GeneratedLoss.__init__(self, "
           + ", ".join(f"{n}={n}" for n in names) + ")\n")
    scope = {"_GeneratedLoss": _GeneratedLoss}
    exec(src, scope)                                     # noqa: S102 — see docstring
    return scope["__init__"]


def _make_losses():
    return {name: type(name, (_GeneratedLoss,), {
        "_fn": staticmethod(globals()[fn]),
        "_keys": tuple(p.split("=")[0] for p in params if p != "*"),
        "__init__": _loss_init(params),
    }) for name, fn, params in _LOSSES}


for _name, _loss_cls in _make_losses().items():
    globals()[_name] = _loss_cls
    setattr(nn, _name, _loss_cls)


class PairwiseDistance(Module):
    """The distance between two paired rows. **`eps` is added to the
    difference** — written down on the function side."""

    def __init__(self, p=2.0, eps=1e-6, keepdim=False):
        super().__init__()
        self.p, self.eps, self.keepdim = p, eps, keepdim

    def forward(self, x1, x2):
        return pairwise_distance(x1, x2, p=self.p, eps=self.eps,
                                 keepdim=self.keepdim)

    def __repr__(self):
        return "PairwiseDistance()"


class CosineSimilarity(Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim, self.eps = dim, eps

    def forward(self, x1, x2):
        return cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)

    def __repr__(self):
        return "CosineSimilarity()"


nn.PairwiseDistance = PairwiseDistance
nn.CosineSimilarity = CosineSimilarity


# ------------------------------------------------------------ the padding layers
#
# **Fifteen come out of one machine.** Three (1-D, 2-D, 3-D) × five (reflect,
# replicate, zero, constant, circular), and only the mode name and the number of
# pairs differ. Written out fifteen times by hand there are fifteen places that
# can drift, and only two things actually differ.
#
# **`ConstantPad` alone prints differently** — the rest print the pairs and that
# one attaches names (`ConstantPad1d(padding=(2, 2), value=7.0)`). Real torch does
# that, and the golden pinned the characters, so the difference is part of the
# answer.

class _PadNd(Module):
    _mode = "constant"
    _dims = 1

    def __init__(self, padding, value=0.0):
        super().__init__()
        pairs = 2 * self._dims
        self.padding = ((padding,) * pairs if isinstance(padding, int)
                        else tuple(padding))
        self.value = value

    def forward(self, x):
        return pad(x, self.padding, mode=self._mode, value=self.value)

    def __repr__(self):
        return f"{type(self).__name__}({self.padding})"


class _ConstantPadNd(_PadNd):
    def __repr__(self):
        return (f"{type(self).__name__}(padding={self.padding}, "
                f"value={self.value})")


def _make_pads():
    """Fifteen are stamped out here. Only the name, the mode and the
    dimensionality differ."""
    made = {}
    for kind, mode, base in (("Reflection", "reflect", _PadNd),
                             ("Replication", "replicate", _PadNd),
                             ("Circular", "circular", _PadNd),
                             ("Zero", "constant", _PadNd),
                             ("Constant", "constant", _ConstantPadNd)):
        for dims in (1, 2, 3):
            name = f"{kind}Pad{dims}d"
            made[name] = type(name, (base,), {"_mode": mode, "_dims": dims})
    return made


for _name, _pad_cls in _make_pads().items():
    globals()[_name] = _pad_cls
    setattr(nn, _name, _pad_cls)


class Upsample(Module):
    """Enlargement. One cell is replicated into s×s, so **the backward is
    summing that block.**

    **The first position is `size`.** That is what torch does — `Upsample(2)` is
    not a factor of 2 but "make the output 2×2". The factor was in the first
    position, and then `Upsample(2)` splits into enlarging and shrinking within
    the same code. The shape is plausible, so only the values catch it.

    **`mode='bilinear'` was being refused.** The computation already existed in
    `interpolate` and ran through `F.upsample_bilinear` — one computation under
    two names with only one of them working. The form textbooks use is the
    layer.
    """

    def __init__(self, size=None, scale_factor=None, mode="nearest",
                 align_corners=None, recompute_scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor
        self.mode, self.align_corners = mode, align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, x):
        if self.size is None and self.scale_factor is None:
            raise RuntimeError(_like_torch(
                "Give either size or scale_factor.",
                "either size or scale_factor should be defined"))
        return interpolate(x, size=self.size, scale_factor=self.scale_factor,
                           mode=self.mode, align_corners=self.align_corners,
                           recompute_scale_factor=self.recompute_scale_factor)

    def __repr__(self):
        return f"Upsample(scale_factor={self.scale_factor}, mode={self.mode!r})"


for _cls in (GELU, SiLU, LeakyReLU, ELU, Softmax, LogSoftmax, AvgPool2d,
             AdaptiveAvgPool2d, Unflatten, L1Loss, SmoothL1Loss, NLLLoss, BatchNorm1d,
             Conv1d, Conv3d, MaxPool1d, MaxPool3d, BatchNorm3d, Upsample):
    setattr(nn, _cls.__name__, _cls)
nn.RNN = RNN
nn.LSTM = LSTM
nn.GRU = GRU
nn.MultiheadAttention = MultiheadAttention
nn.TransformerEncoderLayer = TransformerEncoderLayer
nn.TransformerEncoder = TransformerEncoder
nn.TransformerDecoderLayer = TransformerDecoderLayer
nn.TransformerDecoder = TransformerDecoder
nn.Transformer = Transformer
nn.MaxPool2d = MaxPool2d
nn.Sequential = Sequential
nn.ModuleList = ModuleList
nn.ModuleDict = ModuleDict
nn.ParameterList = ParameterList
nn.ParameterDict = ParameterDict
for _cls in (CELU, GLU, Hardshrink, Hardsigmoid, Hardswish, Hardtanh, LogSigmoid,
             Mish, PReLU, ReLU6, SELU, Softmin, Softplus, Softshrink, Softsign,
             Tanhshrink, Threshold,
             ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, GroupNorm,
             InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, RMSNorm,
             AvgPool1d, AvgPool3d, AdaptiveAvgPool1d, AdaptiveAvgPool3d,
             AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d,
             LPPool1d, LPPool2d, LPPool3d,
             MaxUnpool1d, MaxUnpool2d, MaxUnpool3d,
             FractionalMaxPool2d, FractionalMaxPool3d, CTCLoss,
             AdaptiveLogSoftmaxWithLoss):
    setattr(nn, _cls.__name__, _cls)
nn.MSELoss = MSELoss
nn.BCELoss = BCELoss
nn.BCEWithLogitsLoss = BCEWithLogitsLoss
nn.CrossEntropyLoss = CrossEntropyLoss


def one_hot(tensor, num_classes=-1):
    idx = tensor.data.astype(int)
    n = int(idx.max()) + 1 if num_classes == -1 else num_classes
    return Tensor(_np.eye(n, dtype=_np.int64)[idx])


class _Functional(_Namespace):
    # Losses and distances. **They point at the same functions the layers do** —
    # two copies would drift.
    huber_loss = staticmethod(huber_loss)
    kl_div = staticmethod(kl_div)
    poisson_nll_loss = staticmethod(poisson_nll_loss)
    gaussian_nll_loss = staticmethod(gaussian_nll_loss)
    margin_ranking_loss = staticmethod(margin_ranking_loss)
    cosine_embedding_loss = staticmethod(cosine_embedding_loss)
    hinge_embedding_loss = staticmethod(hinge_embedding_loss)
    soft_margin_loss = staticmethod(soft_margin_loss)
    triplet_margin_loss = staticmethod(triplet_margin_loss)
    triplet_margin_with_distance_loss = staticmethod(
        triplet_margin_with_distance_loss)
    multilabel_soft_margin_loss = staticmethod(multilabel_soft_margin_loss)
    multi_margin_loss = staticmethod(multi_margin_loss)
    multilabel_margin_loss = staticmethod(multilabel_margin_loss)
    pairwise_distance = staticmethod(pairwise_distance)
    pdist = staticmethod(pdist)
    # Rearrangement and channel-wise dropout.
    pixel_shuffle = staticmethod(pixel_shuffle)
    pixel_unshuffle = staticmethod(pixel_unshuffle)
    channel_shuffle = staticmethod(channel_shuffle)
    # `native_channel_shuffle` is not exposed. It is ATen's low-level entry
    # point and there is a separate name above that calls it — writing that down
    # in the gap table and then building it here makes the table lie. It really
    # was built once and `test_gap.py` caught the contradiction.
    dropout1d = staticmethod(dropout1d)
    dropout2d = staticmethod(dropout2d)
    dropout3d = staticmethod(dropout3d)
    alpha_dropout = staticmethod(alpha_dropout)
    feature_alpha_dropout = staticmethod(feature_alpha_dropout)
    # Unfolding windows and the rest. **`F.unfold` is im2col** — different from
    # `Tensor.unfold`.
    unfold = staticmethod(unfold_im2col)
    fold = staticmethod(fold)
    bilinear = staticmethod(bilinear)
    local_response_norm = staticmethod(local_response_norm)
    rrelu = staticmethod(rrelu)
    softmax = staticmethod(softmax)
    log_softmax = staticmethod(log_softmax)
    relu = staticmethod(relu)
    leaky_relu = staticmethod(leaky_relu)
    elu = staticmethod(elu)
    silu = staticmethod(silu)
    gelu = staticmethod(gelu)
    sigmoid = staticmethod(sigmoid)
    tanh = staticmethod(tanh)
    celu = staticmethod(celu)
    hardshrink = staticmethod(hardshrink)
    hardsigmoid = staticmethod(hardsigmoid)
    hardswish = staticmethod(hardswish)
    hardtanh = staticmethod(hardtanh)
    logsigmoid = staticmethod(logsigmoid)
    mish = staticmethod(mish)
    relu6 = staticmethod(relu6)
    selu = staticmethod(selu)
    softplus = staticmethod(softplus)
    softshrink = staticmethod(softshrink)
    softsign = staticmethod(softsign)
    tanhshrink = staticmethod(tanhshrink)
    threshold = staticmethod(threshold_fn)
    softmin = staticmethod(softmin)
    glu = staticmethod(glu)
    prelu = staticmethod(prelu)
    group_norm = staticmethod(group_norm)
    instance_norm = staticmethod(instance_norm)
    rms_norm = staticmethod(rms_norm)
    scaled_dot_product_attention = staticmethod(scaled_dot_product_attention)
    avg_pool1d = staticmethod(avg_pool1d)
    avg_pool3d = staticmethod(avg_pool3d)
    adaptive_avg_pool1d = staticmethod(adaptive_avg_pool1d)
    adaptive_avg_pool3d = staticmethod(adaptive_avg_pool3d)
    adaptive_max_pool1d = staticmethod(adaptive_max_pool1d)
    adaptive_max_pool2d = staticmethod(adaptive_max_pool2d)
    adaptive_max_pool3d = staticmethod(adaptive_max_pool3d)
    lp_pool1d = staticmethod(lp_pool1d)
    lp_pool2d = staticmethod(lp_pool2d)
    lp_pool3d = staticmethod(lp_pool3d)
    # The versions that also give the winning positions, and the partner that
    # puts values back at them.
    adaptive_max_pool1d_with_indices = staticmethod(adaptive_max_pool1d_with_indices)
    adaptive_max_pool2d_with_indices = staticmethod(adaptive_max_pool2d_with_indices)
    adaptive_max_pool3d_with_indices = staticmethod(adaptive_max_pool3d_with_indices)
    max_pool1d_with_indices = staticmethod(max_pool1d_with_indices)
    max_pool2d_with_indices = staticmethod(max_pool2d_with_indices)
    max_pool3d_with_indices = staticmethod(max_pool3d_with_indices)
    max_unpool1d = staticmethod(max_unpool1d)
    max_unpool2d = staticmethod(max_unpool2d)
    max_unpool3d = staticmethod(max_unpool3d)
    ctc_loss = staticmethod(ctc_loss)
    # In-place activations. The version without the underscore does the
    # computation and this writes back into its own buffer.
    celu_ = staticmethod(celu_)
    elu_ = staticmethod(elu_)
    hardtanh_ = staticmethod(hardtanh_)
    leaky_relu_ = staticmethod(leaky_relu_)
    relu_ = staticmethod(relu_)
    rrelu_ = staticmethod(rrelu_)
    selu_ = staticmethod(selu_)
    threshold_ = staticmethod(threshold_)
    multi_head_attention_forward = staticmethod(multi_head_attention_forward)
    affine_grid = staticmethod(affine_grid)
    grid_sample = staticmethod(grid_sample)
    batch_norm = staticmethod(batch_norm)
    embedding_bag = staticmethod(embedding_bag)
    gumbel_softmax = staticmethod(gumbel_softmax)
    upsample = staticmethod(upsample)
    upsample_bilinear = staticmethod(upsample_bilinear)
    upsample_nearest = staticmethod(upsample_nearest)
    fractional_max_pool2d = staticmethod(fractional_max_pool2d)
    fractional_max_pool3d = staticmethod(fractional_max_pool3d)
    fractional_max_pool2d_with_indices = staticmethod(fractional_max_pool2d_with_indices)
    fractional_max_pool3d_with_indices = staticmethod(fractional_max_pool3d_with_indices)
    conv_transpose1d = staticmethod(conv_transpose1d)
    conv_transpose2d = staticmethod(conv_transpose2d)
    conv_transpose3d = staticmethod(conv_transpose3d)
    one_hot = staticmethod(one_hot)
    dropout = staticmethod(dropout)
    avg_pool2d = staticmethod(avg_pool2d)
    layer_norm = staticmethod(layer_norm)
    embedding = staticmethod(embedding)
    nll_loss = staticmethod(nll_loss)
    l1_loss = staticmethod(l1_loss)
    smooth_l1_loss = staticmethod(smooth_l1_loss)
    pad = staticmethod(pad)
    normalize = staticmethod(normalize)
    cosine_similarity = staticmethod(cosine_similarity)
    # The 1-D and 3-D families. **The ones the sister library had and this did
    # not.**
    conv1d = staticmethod(conv1d)
    conv2d = staticmethod(conv2d)
    conv3d = staticmethod(conv3d)
    max_pool1d = staticmethod(max_pool1d)
    max_pool2d = staticmethod(max_pool2d)
    max_pool3d = staticmethod(max_pool3d)
    adaptive_avg_pool2d = staticmethod(adaptive_avg_pool2d)
    interpolate = staticmethod(interpolate)

    # **Every one of these passes by keyword now.** They read `MSELoss(reduction)`
    # and the like, and the day the constructors took torch's order that put a
    # string where `weight` goes — six tests went red at once and the message was
    # the class-weight refusal, which is not what any of them were doing. A
    # positional call is a silent bet that the callee's parameter order never moves,
    # and this file has just moved six of them.
    @staticmethod
    def mse_loss(pred, target, reduction="mean"):
        return MSELoss(reduction=reduction)(pred, target)

    @staticmethod
    def binary_cross_entropy_with_logits(logits, target, weight=None,
                                         reduction="mean", pos_weight=None):
        return BCEWithLogitsLoss(weight=weight, reduction=reduction,
                                 pos_weight=pos_weight)(logits, target)

    @staticmethod
    def binary_cross_entropy(p, target, weight=None, reduction="mean"):
        return BCELoss(weight=weight, reduction=reduction)(p, target)

    @staticmethod
    def linear(input, weight, bias=None):
        out = input @ weight.transpose(-2, -1)
        return out + bias if bias is not None else out

    conv2d = staticmethod(conv2d)
    max_pool2d = staticmethod(max_pool2d)

    @staticmethod
    def cross_entropy(logits, target, weight=None, ignore_index=-100,
                      reduction="mean", label_smoothing=0.0):
        """torch's argument list for the function form as well.

        **This read `CrossEntropyLoss(reduction)` — positionally.** The day the
        constructor took torch's order, `weight` moved into first place and that
        call started handing it a string; the refusal fired and six tests went red
        at once. Written with the keyword it would have survived the change, which
        is the argument for keywords at every internal call site: a positional call
        is a silent bet that the callee's order never moves.
        """
        return CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                reduction=reduction,
                                label_smoothing=label_smoothing)(logits, target)


nn.functional = _Functional()


