"""Checks that the gap tables **do not lie.**

The tables in `tests/torch_gap.py` are ones where every name written down either raises our
percentage (`NOT_API`) or shortens the to-do list (`SKIPPED`). Tables like that widen
quietly over time — what stops it is not human will but the checks written here.

Three things are looked at.

- **No dead rows.** A row that matches no torch name means torch changed, and its reason is
  then a reason about nothing.
- **A reason exists.** An empty reason is "declined, no more", and then the table does no
  work.
- **No contradictions.** If `NOT_API` says "not API" about a name we have already
  implemented, one of the two is false.
"""

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

torch = pytest.importorskip("torch")
# **The vision half is not optional to this file.** The rows written for
# `transforms` only match while those namespaces are on the list, so with
# torchvision absent every one of them reads as a dead row and this file goes red
# for a reason that is about the environment rather than the tables. Skipping is
# the honest answer; `pyproject.toml`'s dev extra installs it.
pytest.importorskip("torchvision")

import torchvision  # noqa: E402

sys.path.insert(0, str(ROOT))
import borchvision  # noqa: E402

from torch_gap import (  # noqa: E402
    DELIBERATE, NOT_API, SKIPPED, _look, _public, _spaces,
)


def _every_torch_name():
    """Every name on torch's side, gathered per namespace and **also given with the namespace attached.**"""
    got = set()
    for space, theirs, _ours in _spaces():
        for name in _public(theirs):
            got.add(name)
            got.add(f"{space}.{name}")
    return got


def _api(namespace, space):
    """`_public` **minus the names the tables say are not API.**

    The report subtracts them — `transforms.functional.Tensor` is an imported label and
    a row says so — and a README count taken from `_public` raw would be one larger than
    the number the tool prints. Two denominators for one question is how a figure ends up
    right in one place and wrong in another with nobody able to say which.
    """
    return {n for n in _public(namespace) if not _look(NOT_API, n, f"{space}.{n}")}


def test_every_namespace_meant_to_be_counted_is_counted():
    """**A namespace off the list has no rule.**

    `transforms` and `transforms.functional` were absent from `_spaces()` for the whole
    life of `borchvision`, so no name inside them was counted and none was asked for a
    reason. The count in circulation was made by hand.

    Nothing above would have noticed. Every check in this file reads `_spaces()`, so a
    namespace missing from it is missing from the checks too — the tables stay consistent
    and the measure is silent. **This is the only check that can speak about what is not
    being measured**, which is why it names the namespaces literally rather than deriving
    them.
    """
    listed = {space for space, _theirs, _ours in _spaces()}
    want = {"torch", "Tensor", "nn", "nn.functional", "optim", "optim.lr_scheduler",
            "linalg", "utils.data", "transforms", "transforms.functional"}
    assert want <= listed, (
        f"namespaces that should be counted are off the list: {sorted(want - listed)}\n"
        "  A namespace not in `_spaces()` is not counted and not asked for reasons. If one\n"
        "  was removed on purpose, remove it from this check too — deliberately, with the\n"
        "  reason written down.")


def test_the_readme_transform_count_is_the_measured_one():
    """The README says **"21 of the 41"**, and here is where those two numbers are.

    It is not in `test_docs.py` with the other README numbers because that file runs
    without torch, and this pair cannot be measured without real torchvision to count
    against. A number nobody can measure is the kind that goes stale.
    """
    theirs, ours = _api(torchvision.transforms, "transforms"), _public(borchvision.transforms)
    row = (ROOT / "README.md").read_text(encoding="utf-8")
    said = re.search(r"\*\*(\d+) of the (\d+) names `torchvision.transforms` carries", row)
    assert said is not None, (
        "the README's torchvision row no longer states its two numbers in the form this\n"
        "  check reads. Reword the check with it rather than dropping it.")
    assert (int(said.group(1)), int(said.group(2))) == (len(theirs & ours), len(theirs)), (
        f"the README says {said.group(1)} of {said.group(2)}; measured is "
        f"{len(theirs & ours)} of {len(theirs)}.")


def test_the_readme_functional_count_is_the_measured_one():
    """The same for `transforms.functional`, and it is a **second** claim rather than a
    second number in the first one.

    Written as one sentence covering both namespaces, one of the two could go stale
    while the sentence stayed true of the other — and a reader has no way to tell which
    half they are reading. Two claims, two checks.
    """
    theirs = _api(torchvision.transforms.functional, "transforms.functional")
    ours = _public(borchvision.transforms.functional)
    row = (ROOT / "README.md").read_text(encoding="utf-8")
    said = re.search(
        r"holds (\d+) of the (\d+) names `torchvision.transforms.functional` carries", row)
    assert said is not None, (
        "the README's `transforms.functional` claim no longer states its two numbers in "
        "the form this check reads. Reword the check with it rather than dropping it.")
    assert (int(said.group(1)), int(said.group(2))) == (len(theirs & ours), len(theirs)), (
        f"the README says {said.group(1)} of {said.group(2)}; measured is "
        f"{len(theirs & ours)} of {len(theirs)}.")


def test_no_table_entry_matches_nothing():
    """**There must be no dead rows.**

    When torch deletes or renames something, the row stays and catches nothing. The reason
    left behind reads to the next person as "this was declined", and the name no longer
    exists.
    """
    names = _every_torch_name()
    dead = []
    for table, label in ((NOT_API, "NOT_API"), (SKIPPED, "SKIPPED")):
        for key in table:
            if not any(_look({key: "x"}, n, n) for n in names):
                dead.append(f"{label}['{key}']")
    assert not dead, (
        "the table has rows that match no name:\n  " + "\n  ".join(dead) +
        "\n\ntorch changed, or it was a typo from the start. Delete it or fix it — the reason "
        "on a row that matches nothing is a reason about nothing.")


def test_no_deliberate_prefix_matches_nothing():
    """The same for the namespace table.

    **`_spaces()` alone is not enough to look at.** Most of this table's prefixes point at
    submodules we never count (`torch.jit`, `torch.distributed`), so sweeping the eight
    namespaces makes all of them look like dead rows — written that way at first, thirteen
    were flagged at once. What is checked is whether torch actually has that name.
    """
    dead = [key for key in DELIBERATE
            if not hasattr(torch, key.split(".")[0])
            and not any(n.startswith(key) for n in _every_torch_name())]
    assert not dead, (
        f"`DELIBERATE` has prefixes that match nothing: {dead}\n"
        "  torch removed that namespace, or it was a typo from the start.")


def test_every_reason_says_something():
    """A reason that is empty or a single character is the same as no reason."""
    thin = []
    for table, label in ((DELIBERATE, "DELIBERATE"), (NOT_API, "NOT_API"),
                         (SKIPPED, "SKIPPED")):
        for key, reason in table.items():
            if not reason or len(reason.strip()) < 4:
                thin.append(f"{label}['{key}'] = {reason!r}")
    assert not thin, (
        "rows with no reason:\n  " + "\n  ".join(thin) +
        "\n\nA row whose reason cannot be written is a gap — take it out of the table.")


def test_not_api_does_not_claim_what_we_implement():
    """**Catches contradictions.**

    If we have already built that name, it is API. With `NOT_API` saying "not API" at the
    same time, one of the two is false, and either way the table stops being believable.
    """
    clashes = []
    for space, _theirs, ours in _spaces():
        for name in _public(ours):
            full = f"{space}.{name}"
            reason = _look(NOT_API, name, full)
            if reason:
                clashes.append(f"{full} — '{reason}'")
    assert not clashes, (
        "`NOT_API` says names we built are not API:\n  " +
        "\n  ".join(clashes) +
        "\n\nBuilt means API. Take it out of the table, or delete what was built.")


def test_skipped_does_not_claim_what_we_actually_do():
    """**A name written into `SKIPPED` that actually runs makes that row false.**

    `NOT_API` has its contradiction caught above and `SKIPPED` did not. There is a reason it
    looked unnecessary — the classifier asks for a reason only about **names we do not
    have.** So nobody reads the reason attached to a name we built, and **the number stays
    right while the documentation goes false.** A stale number surfaces on being measured
    again; a stale reason does not.

    Twelve of them were exactly that. The reasons on the `complex`, `real`, `imag` and `conj`
    family still read "there is no complex dtype here" long after `complex64` went in. Anyone
    reading that row and taking away "complex is not supported" **learns something false.**

    ## A name existing does not mean it runs

    Some places, such as `q_scale` and `int_repr`, **carry a name in order to refuse.** Their
    reasons are still true. Separating with `hasattr` catches those too, so it **calls
    them** — our refusals stop with `BorchError` regardless of the arguments, and that is
    what separates them.

    A name that could not be called is **not judged.** Pretending to know what is unknown is
    where this table starts lying again.
    """
    import numpy as np

    import borch
    from borch import BorchError

    probe = borch.tensor(np.array([1.0, 2.0], dtype=np.float32))

    def verdict(fn):
        """`True` = it runs · `False` = it refuses · `None` = it could not be called."""
        seen_type_error = False
        for args in ((), (probe,), (probe, probe)):
            try:
                fn(*args)
            except BorchError:
                return False
            except TypeError:
                seen_type_error = True
            except Exception:                                   # noqa: BLE001
                continue
            else:
                return True
        return None if seen_type_error else None

    alive = []
    for space, _theirs, ours in _spaces():
        for name in _public(ours):
            full = f"{space}.{name}"
            reason = _look(SKIPPED, name, full)
            if not reason:
                continue
            got = getattr(ours, name, None)
            if not callable(got):
                continue
            if verdict(got) is True:
                alive.append(f"{full} — '{reason}'")
    assert not alive, (
        "`SKIPPED` says we decline **things we actually do**:\n  " +
        "\n  ".join(sorted(alive)) +
        "\n\nDelete those rows. Unlike a number, nobody measures a reason again, so once it\n"
        "goes stale it keeps handing a falsehood to whoever reads it.")


def test_the_tool_still_runs():
    """Whether the tool itself runs. Breaking the syntax while editing the tables stops here."""
    import torch_gap

    assert torch_gap.main([]) == 0
