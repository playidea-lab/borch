"""**The fourth axis: the methods of the classes the other three do not read.**

Three axes were already here and none of them looks at a method of a class:

    ts_axis.py           a namespace's top-level **names**, core ↔ borch.ts
    ts_signatures.py     those names' **constructors**, core ↔ borch.ts
    test_torch_names.py  torch's runtime **argument names**, torch ↔ core

`torch_gap` counts `Tensor`'s methods because `Tensor` is one of its spaces. Every
other class is counted **as a name** — `nn.Module` present, tick — and what is
inside it is read by nobody.

## What that hid

`Optimizer.add_param_group` was missing. It is the line fine-tuning is written with,
torch has it, borch.ts has it as `addParamGroup`, and this side did not — invisible
to every instrument here. Looking for its neighbours found `nn.Module` holding **14
of torch's 50 public methods**, with `children` and `named_children` among the
absent while borch.ts has had both all along. A three-way divergence, not a shared
limit, and green everywhere.

Eleven closed in the same commit as this file: the ten that walk the module tree,
and `add_param_group`.

## What this asks

Only *does the name exist*. Not the arguments — `test_torch_names.py` asks that of
what it can reach — and not the values, which the golden asks. One question:
**can a caller write the line torch's documentation writes?**

Every gap needs a reason, and a reason that outlives its gap is a failure too. That
second half is what stops this table from becoming the place gaps go to be forgotten:
the `AvgPool2d` row in `borch-ts/test/run.py` sat true for an hour after its cause
was gone, and only because the person who ended it happened to read the line.
"""

import pytest

torch = pytest.importorskip("torch")
import borch                                                       # noqa: E402


def _public(cls):
    """The public callables — what a caller can reach by name."""
    return {n for n in dir(cls)
            if not n.startswith("_") and callable(getattr(cls, n, None))}


def _classes():
    sched = torch.optim.lr_scheduler
    ours = borch.optim.lr_scheduler
    return [
        ("nn.Module", torch.nn.Module, borch.nn.Module),
        ("optim.Optimizer", torch.optim.Optimizer, borch.optim.Optimizer),
        ("optim.lr_scheduler.LRScheduler", sched.LRScheduler, ours.LRScheduler),
        ("optim.lr_scheduler.ReduceLROnPlateau",
         sched.ReduceLROnPlateau, ours.ReduceLROnPlateau),
        ("utils.data.DataLoader",
         torch.utils.data.DataLoader, borch.utils.data.DataLoader),
    ]


# **One device, one precision.** Written once at the top of `borch/_base.py` and
# inherited by every name below it: there is one backend and it is float32, so a
# method whose whole purpose is to move a model to another device or narrow it to
# another dtype has nothing to move it to. `Module.to` is here and refuses a device
# it does not have, which is the door these would come through.
_ONE_DEVICE = ("there is one device and one precision — `Module.to` is the door, "
               "and it refuses anything else by name")

# **Hooks are torch's extension mechanism and not an operation.** A hook changes
# nothing about what a model computes; it lets somebody else's code run in the
# middle. Porting them means porting the ordering, the removable handles and the
# `always_call` semantics, and nothing in this library or its golden asks for one.
# Named rather than left out so that the day something does ask, this row is where
# the question already is.
_HOOKS = ("a hook runs somebody else's code mid-pass and changes no value here; "
          "the ordering, the removable handle and `always_call` are the port, and "
          "nothing asks for one yet")

DECLINED = {
    # ── nn.Module ──
    **{("nn.Module", n): _ONE_DEVICE for n in
       ("bfloat16", "cpu", "cuda", "double", "float", "half", "ipu", "mtia",
        "share_memory", "to_empty", "type", "xpu")},
    **{("nn.Module", n): _HOOKS for n in
       ("register_backward_hook", "register_forward_hook",
        "register_forward_pre_hook", "register_full_backward_hook",
        "register_full_backward_pre_hook", "register_load_state_dict_post_hook",
        "register_load_state_dict_pre_hook", "register_state_dict_post_hook",
        "register_state_dict_pre_hook")},
    ("nn.Module", "compile"):
        "`torch.compile` is declined at the top level too — there is no graph "
        "capture and no second backend to compile for, and a `compile` that hands "
        "back the same module unchanged is the accepted-and-inert shape this "
        "repository spends its checks on",
    ("nn.Module", "get_extra_state"):
        "the pair exists so a subclass can put something of its own into "
        "`state_dict`. torch's own is a stub that raises unless overridden, so "
        "there is nothing here to port — a subclass wanting it writes both",
    ("nn.Module", "set_extra_state"): "as `get_extra_state`, its other half",
    ("nn.Module", "set_submodule"):
        "`get_submodule` is here and this is its write side. Replacing a layer by "
        "its dotted name is a real thing to want; it is absent rather than "
        "declined, and this row is the record of that",

    # ── optim.Optimizer ──
    **{("optim.Optimizer", n): _HOOKS for n in
       ("register_load_state_dict_post_hook", "register_load_state_dict_pre_hook",
        "register_state_dict_post_hook", "register_state_dict_pre_hook",
        "register_step_post_hook", "register_step_pre_hook")},
    ("optim.Optimizer", "OptimizerPreHook"):
        "a typing alias torch re-exports through the class, not a method — "
        "`callable()` says yes because a `typing` construct is callable",
    ("optim.Optimizer", "OptimizerPostHook"): "as `OptimizerPreHook`",
    ("optim.Optimizer", "profile_hook_step"):
        "the decorator torch wraps `step` in so the profiler can see it. There is "
        "no profiler here for it to feed",

    # ── optim.lr_scheduler ──
    ("optim.lr_scheduler.ReduceLROnPlateau", "get_lr"):
        "**torch's raises.** It is `LRScheduler.get_lr`'s abstract, inherited and "
        "not overridden — measured: `ReduceLROnPlateau(opt).get_lr()` gives "
        "`NotImplementedError`. A name that cannot be called is not a name to port",

    # ── utils.data ──
    ("utils.data.DataLoader", "check_worker_number_rationality"):
        "it warns when `num_workers` looks wrong for the machine's core count. "
        "**One host stream** — the decision written at the top of "
        "`borch-ts/src/data.ts` — so there is no worker count to be wrong",
}


def _gaps():
    """`{(class, method)}` — every public method of torch's that ours has not."""
    out = set()
    for name, theirs, ours in _classes():
        for method in sorted(_public(theirs) - _public(ours)):
            out.add((name, method))
    return out


def test_every_missing_method_carries_a_reason():
    """A method torch has and this side does not **has to say why.**

    The number is not held here — only the reasons are. A gap that closes needs no
    edit, and the test below is what makes sure a closed one does not go on being
    excused.
    """
    unexplained = sorted(_gaps() - set(DECLINED))
    assert not unexplained, (
        "these methods are torch's and are not here, with no reason written:\n  "
        + "\n  ".join(f"{cls}.{name}" for cls, name in unexplained)
        + "\n\nWrite one into `DECLINED`, or implement the method. A reason is a "
          "sentence about why it cannot or should not be carried — not a note that "
          "it is missing, which the table already says.")


def test_no_reason_outlives_the_gap_it_explains():
    """**A reason for a gap that has closed is worse than no reason.** It reads as
    a live limit and it is a record of the past.

    This is `borch-ts/test/run.py`'s `AvgPool2d` row, generalised: that one said
    *AvgPool2d takes no padding*, which stopped being true the moment the layer
    moved to `poolND`, and it survived only because the person who ended it read
    the line on their way past.
    """
    stale = sorted(set(DECLINED) - _gaps())
    assert not stale, (
        "these are excused in `DECLINED` and are no longer missing:\n  "
        + "\n  ".join(f"{cls}.{name}" for cls, name in stale)
        + "\n\nTake the row out. The reason describes a state that has ended.")


def test_the_classes_are_actually_being_read():
    """**A denominator that goes to nothing reads as agreement.**

    If `_public` stopped answering — a rename, an import that fails softly, a class
    that stops being a class — every set difference would be empty and both tests
    above would pass while measuring nothing. `Tensor` is not in this axis and its
    surface is the one this file is *not* about, so the floor is set from what these
    five carry, which is a number that only moves when torch moves.
    """
    counted = {name: len(_public(theirs)) for name, theirs, _ in _classes()}
    thin = {n: c for n, c in counted.items() if c < 1}
    assert not thin, (
        f"a class came back with no public methods at all: {thin}\n"
        "  That is the shape of a reader that stopped, not of a class that is empty.")
    assert sum(counted.values()) >= 60, (
        f"the five classes hold {sum(counted.values())} public methods between "
        f"them: {counted}. It was 75 when this was written — a fall that large is "
        "the reader, not torch.")
