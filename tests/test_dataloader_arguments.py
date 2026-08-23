"""Every `DataLoader` argument does something, is refused, or is ignored on purpose.

**A positional case proves the value arrived, not that it works.** `tests/cases.py`
gives the loader its arguments by position, because that is the only way to see a
wrong seat — `collate_fn` is torch's seventh and `drop_last` its ninth, and here they
were sixth and seventh, so a boolean was landing in a callable's slot. Those cases
are right and they cannot tell *in the right seat and working* from *in the right
seat and dropped*.

Two arguments here **are** dropped on purpose: `pin_memory` and `pin_memory_device`
ask for page-locked host memory so a copy to a GPU can be asynchronous, and there is
no device to copy to. That is a correct decision and an invisible one — nothing in
the positional cases distinguishes it from an oversight, and nothing would notice if
a third argument joined them by accident.

So every argument is named with what it is expected to do, and the three outcomes are
exhaustive:

    CHANGES    a value that must change what comes out of the loader
    REFUSED    a value that must raise — torch's own refusals where torch has them
    IGNORED    accepted, changes nothing, and the reason is written here

`test_every_argument_is_accounted_for` fails when a name appears in none of the
three, so a new argument cannot arrive unexamined.
"""

import inspect

import pytest

import borch

ITEMS = list(range(10))


def _out(**kw):
    """The loader's whole output, as something comparable."""
    return [[int(v) for v in batch] for batch in
            borch.utils.data.DataLoader(ITEMS, **kw)]


def _sampler():
    return borch.utils.data.SequentialSampler(ITEMS[:4])


def _batch_sampler():
    return borch.utils.data.BatchSampler(
        borch.utils.data.SequentialSampler(ITEMS), 4, True)


def _seeded(seed):
    g = borch.Generator()
    g.manual_seed(seed)
    return g


# **Each value has to be one the default answer cannot match.** `drop_last=True` on
# ten items in batches of one drops nothing, and `generator` changes nothing without
# `shuffle`, so both are given alongside what makes them mean something — the same
# rule as "a case for an argument has to be one whose default answer would be wrong".
CHANGES = {
    "batch_size": {"batch_size": 3},
    "shuffle": {"shuffle": True, "generator": _seeded(1)},
    "sampler": {"sampler": _sampler()},
    "batch_sampler": {"batch_sampler": _batch_sampler()},
    "collate_fn": {"batch_size": 2, "collate_fn": lambda b: [-1]},
    "drop_last": {"batch_size": 3, "drop_last": True},
}

# torch's own refusals where torch has them: it rejects `prefetch_factor` and
# `persistent_workers` when `num_workers` is 0, and that is always true here.
REFUSED = {
    "timeout": {"timeout": 5},
    "worker_init_fn": {"worker_init_fn": lambda i: None},
    "multiprocessing_context": {"multiprocessing_context": "fork"},
    "prefetch_factor": {"prefetch_factor": 2},
    "persistent_workers": {"persistent_workers": True},
    "in_order": {"in_order": False},
}

IGNORED = {
    "num_workers": ({"num_workers": 2},
                    "one process — a browser has no fork, and the values are the "
                    "same either way"),
    "pin_memory": ({"pin_memory": True},
                   "page-locked host memory for an async copy to a device that is "
                   "not here; no value changes"),
    "pin_memory_device": ({"pin_memory_device": "cuda"}, "as `pin_memory`"),
}

# `generator` is checked apart from the three: it changes the answer, but only
# against *another seed*, not against the default — with no `shuffle` there is
# nothing to draw and with `shuffle` the default already draws.
SEEDED = "generator"


def _names():
    params = inspect.signature(borch.utils.data.DataLoader.__init__).parameters
    return [p for p in params][2:]          # past `self` and `dataset`


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_an_argument_that_should_change_the_answer_does(name):
    assert _out(**CHANGES[name]) != _out(), (
        f"`{name}` was given a value the default cannot produce and the loader "
        "returned the default's answer. In the right seat and dropped looks exactly "
        "like in the right seat and working.")


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_an_argument_that_should_be_refused_is(name):
    with pytest.raises(ValueError):
        _out(**REFUSED[name])


@pytest.mark.parametrize("name", sorted(IGNORED))
def test_an_argument_ignored_on_purpose_changes_nothing(name):
    kw, why = IGNORED[name]
    assert _out(**kw) == _out(), (
        f"`{name}` is recorded as ignored on purpose ({why}) and it changed the "
        "answer. Either it does something now — move it to CHANGES — or something "
        "else moved underneath it.")


def test_the_generator_decides_the_order():
    same = _out(batch_size=3, shuffle=True, generator=_seeded(7))
    again = _out(batch_size=3, shuffle=True, generator=_seeded(7))
    other = _out(batch_size=3, shuffle=True, generator=_seeded(11))
    assert same == again, "the same seed gave two different orders"
    assert same != other, (
        "two seeds gave the same order — the generator is being accepted and not "
        "used, which is the shape this file exists to catch")


def test_every_argument_is_accounted_for():
    """**The list is exhaustive or it says nothing.** A name in none of the three
    tables is an argument nobody asked about, which reads from outside exactly like
    an argument that works."""
    named = set(CHANGES) | set(REFUSED) | set(IGNORED) | {SEEDED}
    missing = [n for n in _names() if n not in named]
    assert not missing, (
        f"these `DataLoader` arguments are in none of the tables: {missing}\n"
        "  Give each a value and say which of the three it is.")
    stale = sorted(named - set(_names()))
    assert not stale, f"these are named and no longer arguments: {stale}"
