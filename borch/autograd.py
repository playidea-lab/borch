"""`torch.autograd` — taking a gradient **without putting it anywhere.**

`.backward()` accumulates into every leaf's `.grad`. `autograd.grad(y, x)` hands
the gradient back and leaves `.grad` exactly as it found it, which is the whole
reason the second function exists: a gradient penalty, a meta-learning inner step,
Grad-CAM and every physics-informed loss are written with it, because they need the
gradient as a *value* to keep computing with and must not disturb the optimiser's
accumulation while they do.

**This module was absent, with no reason ever written down.** `borch.autograd`
raised `AttributeError: module 'borch' has no attribute 'autograd'` — a sentence
that names neither gradients nor what is missing, so a reader could not tell an
absent feature from a typo. Every other gap in this library states itself.

`create_graph=True` is refused here in the same words `backward` refuses it: the
backward functions are numpy closures rather than graph nodes, so nothing records
the backward pass and a second derivative has nowhere to come from.
"""

import numpy as _np

from ._base import _unsupported, _like_torch
from ._tensor import Tensor, _flow


def _listed(what):
    """One tensor or a sequence of them, as a list. torch takes either."""
    if isinstance(what, Tensor):
        return [what]
    return list(what)


def _seeds(outputs, grad_outputs):
    """One seed per output, in torch's order of complaint.

    Left out, an output has to be a scalar; given, the shape has to match. Both
    checks are `Tensor.backward`'s, in torch's own words — the two functions reach
    the same tape and a caller who moves between them should not meet two wordings
    for one rule.
    """
    if grad_outputs is None:
        given = [None] * len(outputs)
    elif isinstance(grad_outputs, Tensor):
        given = [grad_outputs]
    else:
        given = list(grad_outputs)
    seeds = []
    for out, seed in zip(outputs, given):
        if seed is None:
            if out.data.size != 1:
                raise RuntimeError(_like_torch(
                    "A tensor with more than one value needs a gradient. "
                    "Usually the loss is reduced to a scalar first.",
                    "grad can be implicitly created only for scalar outputs"))
            seeds.append(_np.ones_like(out.data))
            continue
        arr = _np.asarray(seed.data if isinstance(seed, Tensor) else seed,
                          dtype=out.data.dtype)
        if arr.shape != out.data.shape:
            raise RuntimeError(_like_torch(
                f"The gradient shape {tuple(arr.shape)} differs from the value shape "
                f"{tuple(out.data.shape)}.",
                f"Mismatch in shape: grad_output[0] has a shape of "
                f"torch.Size({list(arr.shape)}) and output[0] has a shape of "
                f"torch.Size({list(out.data.shape)})."))
        seeds.append(arr)
    return seeds


def grad(outputs, inputs, grad_outputs=None, retain_graph=None,
         create_graph=False, only_inputs=True, allow_unused=None,
         is_grads_batched=False, materialize_grads=False):
    """The gradient of `outputs` with respect to `inputs`, as a tuple.

    **`.grad` is not touched** — measured against torch, which leaves it `None` on a
    leaf that has never been through `.backward()`. That is the difference from
    `backward()` and it is the point; a version that accumulated would silently add
    a penalty term's gradient to the one the optimiser is about to step on.

    The refusals are torch's, measured one at a time:

    - an input the graph never reached is a `RuntimeError` naming **its index**,
      unless `allow_unused=True`, which puts `None` in that slot instead;
    - `materialize_grads=True` puts zeros there rather than `None`, and does **not**
      need `allow_unused` (measured — it implies it);
    - an input that does not require grad, an empty `inputs`, a non-scalar output
      with no seed, and a second walk over a released graph each stop with torch's
      own sentence.

    `only_inputs` is torch's own dead argument — it has defaulted to `True` for
    years and the `False` branch was removed from torch itself, so passing `False`
    is refused rather than quietly ignored.

    `is_grads_batched` wants `vmap`, which is not here.
    """
    if create_graph:
        _unsupported("autograd.grad(create_graph=True) — double backward")
    if is_grads_batched:
        _unsupported("autograd.grad(is_grads_batched=True) — it needs `vmap`")
    if not only_inputs:
        _unsupported("autograd.grad(only_inputs=False) — torch removed that branch too")

    outs = _listed(outputs)
    ins = _listed(inputs)
    # **First of all**, ahead of every other refusal — the same order `backward`
    # follows for its own `inputs=[]`.
    if not ins:
        raise RuntimeError("`inputs` argument to `grad()` cannot be empty.")
    for out in outs:
        if not out.requires_grad:
            raise RuntimeError(_like_torch(
                "grad() cannot be taken of a tensor that does not require grad.",
                "element 0 of tensors does not require grad and does not have a grad_fn"))
        if out._freed:
            raise RuntimeError(_like_torch(
                "This graph has already been walked. Going back once releases it — "
                "recompute, or pass `retain_graph=True`.",
                "Trying to backward through the graph a second time"))
    seeds = _seeds(outs, grad_outputs)
    # **After the seed, not before it** — torch settles the gradient's shape first
    # and only then complains about a name that cannot hold one.
    for one in ins:
        if not one.requires_grad:
            raise RuntimeError("One of the differentiated Tensors does not require grad")

    order, grads = _flow(outs, seeds)

    out_grads = []
    for at, one in enumerate(ins):
        got = grads.get(id(one))
        if got is not None:
            out_grads.append(Tensor(got))
        elif materialize_grads:
            out_grads.append(Tensor(_np.zeros_like(one.data)))
        elif allow_unused:
            out_grads.append(None)
        else:
            raise RuntimeError(
                f"The differentiated Tensor at index {at} appears to not have been "
                "used in the graph. Set allow_unused=True if this is the desired "
                "behavior.")

    # `retain_graph` defaults to `create_graph`, which is refused above and so is
    # always False here — releasing is torch's default and the memory is the reason.
    if not retain_graph:
        for t in order:
            if t._backward is not None:
                t._freed = True
    return tuple(out_grads)


def backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False,
             grad_variables=None, inputs=None):
    """The free-function spelling of `y.backward()`, accumulating into `.grad`.

    **It forwards rather than walking.** Written as its own loop it would be a
    second copy of the accumulation rules — leaves, `retain_grad()` nodes, and the
    `inputs` list that both restricts and retains — and the copy is what drifts.

    `grad_variables` is torch's own deprecated spelling of `grad_tensors`; torch
    still accepts it, so it is accepted and merged here.
    """
    if grad_variables is not None:
        if grad_tensors is not None:
            raise RuntimeError(
                "'grad_tensors' and 'grad_variables' (deprecated) arguments "
                "both passed to backward(). Please only use 'grad_tensors'.")
        grad_tensors = grad_variables
    outs = _listed(tensors)
    if grad_tensors is None:
        given = [None] * len(outs)
    elif isinstance(grad_tensors, Tensor):
        given = [grad_tensors]
    else:
        given = list(grad_tensors)
    for out, seed in zip(outs, given):
        out.backward(seed, retain_graph=bool(retain_graph),
                     create_graph=create_graph, inputs=inputs)
