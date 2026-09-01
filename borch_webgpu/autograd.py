"""`torch.autograd` — taking a gradient **without putting it anywhere.**

The core's module of the same name says why it exists. Here the whole of it is
carried across to borch.ts's `Tensor.grad`, which is the same walk `backward` takes
with the accumulation left off — a second walk on this side would be a fourth copy
of arithmetic three implementations already have to agree on.

What is Python's alone stays here: torch takes one tensor *or* a sequence in both
positions, and it takes `grad_variables`, its own deprecated spelling.
"""

from borch._base import _unsupported

from ._base import Tensor, guarded, handle, _ts, wrap


def _js_tensors(what):
    """One tensor or a sequence of them, as a list of borch.ts handles.

    **A plain Python list crosses, and it was written as `js.Array.from_` first.**
    The note copied from `Tensor.backward` next door said a Python list arrives a
    proxy that `instanceof Tensor` rejects and that spreads empty. Half of that is
    true and it does not apply here: `backward` tests **the argument itself** with
    `instanceof`, so a proxy in that seat really is received and does nothing, while
    `Tensor.grad` spreads both of its seats — and a proxy of a Python list is
    iterable in JavaScript, so the spread yields the handles.

    Measured, not reasoned: the conversion was removed and all 4,270 browser cases
    still agreed, which is what a precaution nothing defends looks like.
    """
    listed = [what] if isinstance(what, Tensor) else list(what)
    return [handle(t) if isinstance(t, Tensor) else t for t in listed]


def grad(outputs, inputs, grad_outputs=None, retain_graph=None,
         create_graph=False, only_inputs=True, allow_unused=None,
         is_grads_batched=False, materialize_grads=False):
    """The gradient of `outputs` with respect to `inputs`, as a tuple.

    The three refusals below are this side's because the seat is Python's:
    `create_graph` is the tape's one genuine limit (`backward` refuses it in the
    same words), `is_grads_batched` wants `vmap`, and `only_inputs=False` is a
    branch torch removed from itself — swallowed, each would answer a question
    nobody asked.
    """
    if create_graph:
        _unsupported("autograd.grad(create_graph=True) — double backward")
    if is_grads_batched:
        _unsupported("autograd.grad(is_grads_batched=True) — it needs `vmap`")
    if not only_inputs:
        _unsupported("autograd.grad(only_inputs=False) — torch removed that branch too")
    seeds = None if grad_outputs is None else _js_tensors(grad_outputs)
    got = guarded(_ts.Tensor.grad, _js_tensors(outputs), _js_tensors(inputs),
                  seeds, bool(retain_graph), bool(allow_unused),
                  bool(materialize_grads))
    # **A tuple, and `None` where a gradient is absent.** torch returns a tuple even
    # for a single input, and `allow_unused=True` puts `None` in the slot — a list
    # would unpack the same way and compare unequal to torch's answer.
    return tuple(None if one is None else wrap(one) for one in got)


def backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False,
             grad_variables=None, inputs=None):
    """The free-function spelling of `y.backward()`, accumulating into `.grad`.

    **It forwards rather than walking**, for the reason at the top of this file:
    the accumulation rules — leaves, retained nodes, and the `inputs` list that both
    restricts and retains — live in one place.
    """
    if grad_variables is not None:
        if grad_tensors is not None:
            raise RuntimeError(
                "'grad_tensors' and 'grad_variables' (deprecated) arguments "
                "both passed to backward(). Please only use 'grad_tensors'.")
        grad_tensors = grad_variables
    outs = [tensors] if isinstance(tensors, Tensor) else list(tensors)
    if grad_tensors is None:
        given = [None] * len(outs)
    elif isinstance(grad_tensors, Tensor):
        given = [grad_tensors]
    else:
        given = list(grad_tensors)
    for out, seed in zip(outs, given):
        out.backward(seed, retain_graph=bool(retain_graph),
                     create_graph=create_graph, inputs=inputs)
