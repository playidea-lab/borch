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
    DELIBERATE, NOT_API, SKIPPED, _look, _public, _spaces, _why,
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
            "linalg", "utils.data", "transforms", "transforms.functional",
            "transforms.v2", "ops", "datasets"}
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


def test_the_v2_namespace_is_counted_and_its_overlap_is_not_hidden():
    """**v2 is a superset of v1, and the count has to keep saying so.**

    This was written when `borchvision.transforms.v2` did not exist and the table read
    0 of 72 — a true sentence about the namespace and a false one about the library,
    since 38 of those names were already here one namespace over. The namespace exists
    now and reads 52 of 72, so the zero it explained is gone.

    What it measures did not change and is worth keeping: **torchvision's own v2 still
    contains v1's names.** That containment is the reason the transforms here can
    subclass their v1 versions and override the repr alone. If v2 ever stops being a
    superset, that design stops being sound, and this is where it would show.
    """
    v2 = _api(torchvision.transforms.v2, "transforms.v2")
    v1 = _api(torchvision.transforms, "transforms")
    ours = _public(borchvision.transforms)
    assert not (v1 - v2), (
        f"v1 has names v2 does not: {sorted(v1 - v2)}\n"
        "  The docstring says v2 is a superset. If that stopped being true, the "
        "explanation of the zero is wrong too.")
    assert len(v2 & ours) >= 38, (
        f"only {len(v2 & ours)} of v2's names exist here; the docstring says 38")


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


def test_a_namespaced_wildcard_stays_inside_its_namespace():
    """**The reason this table could not count `transforms.v2.functional` at all.**

    114 of that namespace's 165 names are `<operation>_<type>` dispatch kernels and one
    reason covers all of them, but the only wildcard this matcher understood was flat.
    `"*_image"` written flat matches `to_pil_image` in v1 as well, and attaching a
    sentence about v2's type dispatch to *that* name is the failure this whole file
    exists to catch — a name not counted for a false reason, which nobody re-reads.

    So the namespace is part of the key now, and this pins the containment. Flatten the
    matcher again and `to_pil_image` starts answering with the dispatch sentence, which
    is the shape that has to stay impossible.
    """
    kernel = _why("transforms.v2.functional", "affine_image")
    assert kernel and "dispatch kernel" in kernel[1], (
        "the namespaced wildcard stopped matching inside its own namespace")

    for space, name in (("transforms.functional", "to_pil_image"),
                        ("transforms", "ToPILImage")):
        found = _why(space, name)
        assert found and "dispatch" not in found[1], (
            f"{space}.{name} is being explained by v2's dispatch reason. The wildcard "
            "went flat, and a name explained by the wrong reason reads as explained.")


def test_a_namespaced_wildcard_needs_the_whole_leaf():
    """`*_image` is a suffix on the **leaf**, not on the full path, and not a substring.

    Both directions are asked. A name that merely contains the word does not match, and
    a name that ends with something longer does not either — either miss would let this
    row claim names it was never written about, and the count would go on looking right.
    """
    for name in ("image", "my_image_thing", "affine_image_extra"):
        assert _why("transforms.v2.functional", name) is None, (
            f"{name} matched a kernel row it has nothing to do with")


def test_the_kernel_rows_cover_every_kernel_and_nothing_else():
    """**A count, because a wildcard cannot be read.**

    Five rows stand for 114 names and no reader can check that by eye. Measured against
    torchvision directly: every public name ending in one of the dispatch suffixes has
    a reason, and the number of names those rows account for is the number that exists.
    A sixth suffix appearing in a future torchvision shows up here as an uncovered name
    rather than as a quietly smaller denominator.

    There were six rows when this was written. `*_batch` was one of them and it matched
    nothing — a reason about nothing, put there because six suffixes looked like a
    natural set. `test_no_table_entry_matches_nothing` said so before this check ran,
    which is the second time today that guessing at a family has cost more than
    measuring it.
    """
    suffixes = ("_image", "_video", "_mask", "_bounding_boxes", "_keypoints")
    names = _api(torchvision.transforms.v2.functional, "transforms.v2.functional")
    kernels = [n for n in names if n.endswith(suffixes)]
    assert len(kernels) == 114, (
        f"torchvision now has {len(kernels)} dispatch kernels rather than 114 — the "
        "rows below may no longer say what they say. Re-read them, then move the "
        "number.")
    unexplained = [n for n in kernels if not _why("transforms.v2.functional", n)]
    assert not unexplained, f"kernels with no reason: {unexplained[:8]}"


def test_the_not_api_bin_has_not_grown():
    """**The one bin that comes out of the denominator, and the one nothing watched.**

    Every other absence stays in: what was declined is a choice we made, so we carry
    the cost of it in the percentage. `NOT_API` is subtracted before the fraction is
    taken, which is right when the call is right — and it means a wrong call there is a
    wrong call **that also makes the number look better.** The comment beside the
    denominator has always said this about `SKIPPED`, and then granted `NOT_API` the
    exemption without anything watching it.

    So the bin has a written size. Growing it is an edit to that number, which makes it
    a decision somebody made rather than one that drifted. The contents were read once,
    on the day this went in: of 203 names four carried an `Example::` in torch's own
    docstring, three of those were fairly called internals, and the fourth —
    `narrow_copy` — moved to `SKIPPED`, where it costs a percentage point rather than
    being free.
    """
    from torch_gap import NOT_API_SIZE, _why                     # noqa: PLC0415

    measured, wrong = {}, []
    for space, theirs, ours in _spaces():
        have = _public(ours)
        found = sum(1 for name in _public(theirs)
                    if name not in have
                    and (_why(space, name) or ("",))[0] == "not API")
        if found:
            measured[space] = found
    for space, found in sorted(measured.items()):
        allowed = NOT_API_SIZE.get(space)
        if allowed is None:
            wrong.append(f"{space}: {found} not-API names and no size written down")
        elif found > allowed:
            wrong.append(f"{space}: {found} not-API names, {allowed} written down")
    # Shrinking is not a failure — but a size that is now too big describes a bin that
    # no longer exists, and the next person reads it as room.
    for space, allowed in sorted(NOT_API_SIZE.items()):
        found = measured.get(space, 0)
        if found < allowed:
            wrong.append(f"{space}: {found} not-API names, {allowed} written down — "
                         "lower it")
    assert not wrong, (
        "the not-API bin no longer matches what is written down:\n  "
        + "\n  ".join(wrong)
        + "\n\nThis bin is subtracted from the denominator. A name put here rather than "
          "in SKIPPED raises the percentage, so growing it is a decision, not a tidy-up "
          "— move the number deliberately, with the reason in the commit message.")
