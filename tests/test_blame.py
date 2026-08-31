"""**Pins the failure grouping against the run it was built from.**

`tests/browser/blame.py` turns a flat list of golden failures into *how many causes*.
The check it has to pass is not "does it produce output" — it is whether it produces
**the thing a person read by hand**, on the run where a person read it.

That run is recorded below. The binding golden reported 94 failures; about 64 came from
one wrong argument seat, because every `opt::` case trains through a single
`CrossEntropyLoss()`. What settled it was not the count of failures but the *helper with
none*: `lr_trace` steps the same twenty-four optimizers with no loss anywhere and every
one of its cases passed.

So the two assertions are the two halves of that reading, and they are separate tests on
purpose: an implementation that names the failing helper and cannot find a clean sibling
has produced a list of symptoms, which is what the failure list already was.

## Why the input is frozen, whole, rather than generated or abridged

The grouping's whole claim is about a specific historical run. A synthesised failure
list would test the mechanism against a situation chosen to suit it — and the case that
made this worth building is one nobody would have invented, because the informative part
was a helper that *did not appear in the failure list at all.*

**The list below is all 94 rows, and the first version of this file was not.** It had 33
— "abridged to the two families that carry the finding" — and the abridgement destroyed
the finding: the grouping's rule is that *every* case through a helper failed, so
leaving out two-thirds of the `train::` rows made `CrossEntropyLoss` a helper with some
failures rather than all of them, and it vanished from the report. Both assertions
failed against a correct implementation.

That is worth more than the twenty minutes it cost. **A sample of a failure list is not
a failure list**, because what the grouping reads is the shape of the whole set — which
rows are absent matters exactly as much as which are present. The same sentence covers
`lr_trace`: it is decisive because none of its cases appear, and no abridgement chosen
for relevance would have kept it.
"""

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "browser"))

pytest.importorskip("torch")
pytest.importorskip("numpy")

import blame                                                     # noqa: E402


NINETY_FOUR = [
    'train::SGD/손실: max diff nan',
    'train::SGD/0.weight: max diff nan',
    'train::SGD(모멘텀)/손실: max diff nan',
    'train::SGD(모멘텀)/0.weight: max diff nan',
    'train::Adam/손실: max diff nan',
    'train::Adam/0.weight: max diff nan',
    'train::CNN/손실: RuntimeError — TypeError: v is not iterable',
    'train::CNN/conv.weight: RuntimeError — TypeError: v is not iterable',
    'train::RMSprop/0.weight: max diff nan',
    'vision::LinearTransformation: ValueError — The image flattens to 0 and the matrix is 60 wide — they do ',
    'vision::RandomErasing(p=0): shape (3, 5, 4) vs (3, 5, 4, 0)',
    'vision::RandomErasing(ten draws all miss): shape (3, 5, 4) vs (3, 5, 4, 0)',
    "vision::F.erase: TypeError — 'Tensor' object does not support item assignment",
    'v2f::elastic: ValueError — cannot reshape array of size 0 into shape (5,4,2)',
    'ndim::nn.Conv1d: RuntimeError — TypeError: v is not iterable',
    "loss::층::HuberLoss: RuntimeError — 0.5 is not a valid value for reduction ('none' | 'mean' | 's",
    "loss::층::CrossEntropyLoss(ignore_index, mean): TypeError — CrossEntropyLoss() got an unexpected keyword argument 'ignor",
    "loss::층::CrossEntropyLoss(ignore_index, sum): TypeError — CrossEntropyLoss() got an unexpected keyword argument 'ignor",
    "loss::층::CrossEntropyLoss(ignore_index, none): TypeError — CrossEntropyLoss() got an unexpected keyword argument 'ignor",
    "loss::층::CrossEntropyLoss(label_smoothing): TypeError — CrossEntropyLoss() got an unexpected keyword argument 'label",
    "loss::층::NLLLoss(ignore_index): TypeError — NLLLoss() got an unexpected keyword argument 'ignore_index'",
    'loss::reduction::cross_entropy(none): shape (2,) vs ()',
    'loss::reduction::nll_loss(none): shape (2,) vs ()',
    'loss::reduction::nn.CrossEntropyLoss(none): shape (2,) vs ()',
    'loss::reduction::nn.NLLLoss(none): shape (2,) vs ()',
    'loss::reduction::cross_entropy(mean): max diff nan',
    'loss::reduction::nll_loss(mean): max diff nan',
    'loss::reduction::nn.CrossEntropyLoss(mean): max diff nan',
    'loss::reduction::nn.NLLLoss(mean): max diff nan',
    'loss::reduction::cross_entropy(sum): max diff nan',
    'loss::reduction::nll_loss(sum): max diff nan',
    'loss::reduction::nn.CrossEntropyLoss(sum): max diff nan',
    'loss::reduction::nn.NLLLoss(sum): max diff nan',
    'misc::층::EmbeddingBag(max_norm): RuntimeError — a leaf Variable that requires grad is being used in an in-pl',
    'misc::층::EmbeddingBag(max_norm)/표가 줄었다: RuntimeError — a leaf Variable that requires grad is being used in an in-pl',
    'misc::층::EmbeddingBag(max_norm)/색인 안 된 행: RuntimeError — a leaf Variable that requires grad is being used in an in-pl',
    'misc::층::EmbeddingBag(max_norm, norm_type=1): RuntimeError — a leaf Variable that requires grad is being used in an in-pl',
    "misc::층::Embedding(padding_idx): AttributeError — module 'borch_webgpu._nn' has no attribute 'Embedding'",
    "misc::층::Embedding(max_norm): AttributeError — module 'borch_webgpu._nn' has no attribute 'Embedding'",
    "misc::층::Embedding(max_norm)/표가 줄었다: AttributeError — module 'borch_webgpu._nn' has no attribute 'Embedding'",
    "misc::층::Embedding(max_norm, norm_type=1)/표: AttributeError — module 'borch_webgpu._nn' has no attribute 'Embedding'",
    "misc::repr::Embedding(전부): AttributeError — module 'borch_webgpu._nn' has no attribute 'Embedding'",
    "misc::grad::Embedding(padding_idx): AttributeError — module 'borch_webgpu._nn' has no attribute 'Embedding'",
    "misc::층::Embedding(padding_idx)/새 표와 준 표: AttributeError — module 'borch_webgpu._nn' has no attribute 'Embedding'",
    'opt::Adagrad/0.weight: max diff nan',
    'opt::Adagrad/손실: max diff nan',
    'opt::Adadelta/0.weight: max diff nan',
    'opt::Adadelta/손실: max diff nan',
    'opt::Adamax/0.weight: max diff nan',
    'opt::Adamax/손실: max diff nan',
    'opt::NAdam/0.weight: max diff nan',
    'opt::NAdam/손실: max diff nan',
    'opt::RAdam/0.weight: max diff nan',
    'opt::RAdam/손실: max diff nan',
    'opt::ASGD/0.weight: max diff nan',
    'opt::ASGD/손실: max diff nan',
    'opt::Rprop/0.weight: max diff 3.72e-01',
    'opt::Rprop/손실: max diff nan',
    'opt::Adafactor/0.weight: max diff nan',
    'opt::Adafactor/손실: max diff nan',
    'opt::Adam(weight_decay)/0.weight: max diff nan',
    'opt::Adam(weight_decay)/손실: max diff nan',
    "opt::AdamW/0.weight: AttributeError — module 'borch_webgpu._optim' has no attribute 'AdamW'",
    "opt::AdamW/손실: AttributeError — module 'borch_webgpu._optim' has no attribute 'AdamW'",
    'opt::Adagrad(weight_decay)/0.weight: max diff nan',
    'opt::Adagrad(weight_decay)/손실: max diff nan',
    'opt::Adadelta(weight_decay)/0.weight: max diff nan',
    'opt::Adadelta(weight_decay)/손실: max diff nan',
    'opt::Adamax(weight_decay)/0.weight: max diff nan',
    'opt::Adamax(weight_decay)/손실: max diff nan',
    'opt::NAdam(weight_decay)/0.weight: max diff nan',
    'opt::NAdam(weight_decay)/손실: max diff nan',
    'opt::RAdam(weight_decay)/0.weight: max diff nan',
    'opt::RAdam(weight_decay)/손실: max diff nan',
    'opt::RMSprop(weight_decay)/0.weight: max diff nan',
    'opt::RMSprop(weight_decay)/손실: max diff nan',
    'opt::ASGD(weight_decay)/0.weight: max diff nan',
    'opt::ASGD(weight_decay)/손실: max diff nan',
    'opt::SGD(weight_decay)/0.weight: max diff nan',
    'opt::SGD(weight_decay)/손실: max diff nan',
    'opt::SGD(dampening)/0.weight: max diff nan',
    'opt::SGD(dampening)/손실: max diff nan',
    'opt::SGD(nesterov)/0.weight: max diff nan',
    'opt::SGD(nesterov)/손실: max diff nan',
    'opt::SGD(maximize)/0.weight: max diff nan',
    'opt::SGD(maximize)/손실: max diff nan',
    'opt::Adagrad(initial_accumulator_value)/0.weight: max diff nan',
    'opt::Adagrad(initial_accumulator_value)/손실: max diff nan',
    'opt::SGD/이어서 학습하기: max diff nan',
    'opt::Adam/이어서 학습하기: max diff nan',
    'opt::RMSprop/이어서 학습하기: max diff nan',
    'opt::상태를 안 옮기면 갈린다: max diff nan',
    'spot::unique(dim=0): shape (2, 2) vs (4,)',
    'spot::unique(dim=0, inverse): shape (3,) vs (6,)',
]

# The run that is finished: the two rows left are `unique(dim=)`, which is a decision
# rather than a defect. **Nothing here shares a helper**, so the grouping must be silent.
THE_TWO = [
    "spot::unique(dim=0): shape (2, 2) vs (4,)",
    "spot::unique(dim=0, inverse): shape (3,) vs (6,)",
]


# **Cases added after the 94 were measured.** The failure list above is frozen and the
# case table below is read live, so the grouping's rule — *every case through this
# helper failed* — is asked with a denominator that keeps growing while its numerator
# does not. Three `opt::` cases added later go through `trained` and therefore through
# `CrossEntropyLoss`, and they did not fail because by then it was fixed: the helper
# stopped being named and both assertions went red against a correct implementation.
#
# That is this file's own sentence one level over. It says **a sample of a failure list
# is not a failure list**, and the reason given is that which rows are *absent* carries
# the finding. The same is true of the case table, and only one of the two was pinned —
# so the fixture stayed a snapshot at one end and a live reading at the other.
#
# Named rather than filtered by a rule (a date, a prefix) so that the next addition
# fails here and is looked at, instead of being absorbed by a pattern.
#
# **The cost of naming them is that this list needs adding to**, and it will fire on
# any future `opt::` or `train::` case that trains — which has nothing to do with the
# grouping and everything to do with the era. So `_era_gap` below turns that failure
# from a puzzle into an instruction: it works out which cases are making the helper
# stop being named and prints them ready to paste. A test that must be updated is
# fine; a test that must be *understood again* every time is not.
ADDED_AFTER = frozenset({
    "opt::Adam(maximize)/0.weight", "opt::Adam(maximize)/손실",
    "opt::RMSprop(maximize)/0.weight", "opt::RMSprop(maximize)/손실",
    "opt::SGD(the default rate)/0.weight", "opt::SGD(the default rate)/손실",
    # The four algorithm variants that were absent from the optimizers — `amsgrad`,
    # `centered`, `momentum`, `decoupled_weight_decay`. Same era, same reason: they
    # train through `CrossEntropyLoss` and they pass, so the helper stops being one
    # *every* failing case went through.
    "opt::Adam(amsgrad)/0.weight", "opt::Adam(amsgrad)/손실",
    "opt::RMSprop(centered)/0.weight", "opt::RMSprop(centered)/손실",
    "opt::RMSprop(momentum)/0.weight", "opt::RMSprop(momentum)/손실",
    "opt::NAdam(decoupled_weight_decay)/0.weight",
    "opt::NAdam(decoupled_weight_decay)/손실",
    # The same flag on the other two torch gives it to, carried across to borch.ts
    # in the same edit. Same era, same reason as the four above.
    "opt::Adam(decoupled_weight_decay)/0.weight",
    "opt::Adam(decoupled_weight_decay)/손실",
    "opt::RAdam(decoupled_weight_decay)/0.weight",
    "opt::RAdam(decoupled_weight_decay)/손실",
    # The class weight, which was a refusal when the 94 were measured — the five
    # cases below could not have been among them, because the call they make stopped
    # before it reached the loss. Same era rule as the rows above.
    "loss::층::CrossEntropyLoss(weight, mean)",
    "loss::층::CrossEntropyLoss(weight, sum)",
    "loss::층::CrossEntropyLoss(weight, none)",
    "loss::층::CrossEntropyLoss(weight, ignore_index)",
    "loss::층::CrossEntropyLoss(weight, label_smoothing)",
})


def _era_gap(cases, helper):
    """The cases through `helper` that are **not** in the frozen failure list.

    Those are exactly the rows that stop `blame.group` naming it, and exactly what
    belongs in `ADDED_AFTER` if they were added after the 94 were measured. Returned
    sorted so the failure message can be pasted straight in.

    **Only the first assertion needs this.** Measured by emptying `ADDED_AFTER`: the
    sibling test still passes, because `controls` asks for a helper with *no*
    failures and a newly-added passing case cannot create one. So era drift shows up
    at one end and not the other — which is worth knowing, since a reader who sees
    one red test may reasonably conclude the other half is unaffected, and here that
    happens to be true.
    """
    failed = set()
    ordered = sorted((n for n, _ in cases), key=len, reverse=True)
    for line in NINETY_FOUR:
        for name in ordered:
            if line == name or line.startswith(name + ":"):
                failed.add(name)
                break
    return sorted(n for n, fn in cases
                  if helper in blame._helpers(fn) and n not in failed)


@pytest.fixture(scope="module")
def cases():
    spec = importlib.util.spec_from_file_location("bt_cases", ROOT / "tests" / "cases.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    built = mod.golden_cases(mod.golden_inputs())
    kept = {n: f for n, f in dict(built).items() if n not in ADDED_AFTER}
    missing = ADDED_AFTER - set(dict(built))
    assert not missing, (
        f"ADDED_AFTER names cases that no longer exist: {sorted(missing)}\n"
        "  A name here that matches nothing removes nothing, and this fixture would "
        "go on looking correct.")
    return list(kept.items())


def test_it_names_the_helper_every_failing_case_went_through(cases):
    """`CrossEntropyLoss`, which is the actual cause and not merely a shared ancestor.

    `trained` is the helper a person found by eye and it is also correct; the grouping
    reaches one level further, to the thing inside `trained` that was broken.
    """
    rows = blame.group(NINETY_FOUR, cases)
    named = {helper for helper, *_ in rows}
    assert "CrossEntropyLoss" in named, (
        f"the grouping did not name the loss every failing case trains through.\n"
        f"  it named: {sorted(named) or '(nothing)'}\n\n"
        "  These cases go through `CrossEntropyLoss` and are not in the frozen\n"
        "  failure list, so the helper no longer fails whole:\n"
        + "".join(f'        "{n}",\n' for n in _era_gap(cases, "CrossEntropyLoss"))
        + "  If they were added after the 94 were measured, paste them into\n"
          "  `ADDED_AFTER`. If one of them is old and newly passing, that is a\n"
          "  different thing and worth reading before it is silenced.")


def test_it_finds_the_sibling_that_passed(cases):
    """**The half that does the work.**

    A helper with many failures is a list of symptoms. A sibling under the same prefix,
    stepping the same optimizers, whose cases all passed is what makes it one cause.
    Separate from the test above because an implementation can do the first and not the
    second, and the first alone adds nothing to the failure list.
    """
    rows = blame.group(NINETY_FOUR, cases)
    clean = {helper for helper, *_ in blame.controls(rows, NINETY_FOUR, cases)}
    assert "lr_trace" in clean, (
        "the grouping did not surface `lr_trace`, which steps the same optimizers with "
        "no loss\n  and passed entirely. That contrast is the finding; without it this "
        "file reports\n  what the failure list already said.\n"
        f"  it offered: {sorted(clean) or '(nothing)'}")


def test_it_says_nothing_when_the_failures_share_nothing(cases):
    """**Silence is a result.**

    A grouping that always produces a ranking hands a reader a most-blamed helper on a
    run whose failures are unrelated, and a plausible wrong lead costs more than no
    lead — which is a lesson this repository learned from a runner that answered
    `3255/3255` about a library nobody had asked about.
    """
    assert blame.report(THE_TWO, cases) == [], (
        "the grouping spoke about two unrelated failures:\n  "
        + "\n  ".join(blame.report(THE_TWO, cases)))


def test_building_the_case_list_moves_no_state_another_test_reads():
    """**This file executes every section builder in `cases.py` to get its fixture.**

    A sweep in a sister file reached `set_printoptions`, handed it a tensor, and left a
    global precision behind; six tests in a different file then failed with
    `Format specifier missing precision`, a message naming neither printing nor the
    sweep. An instrument that changes what it measures is worse than one that measures
    wrongly, because the damage lands somewhere it cannot be traced from.

    Building a case list is not calling a case, so this should be safe — and *should be*
    is what the check is for.

    **The first version of this reseeded before taking the second reading**, which makes
    an RNG shift impossible to observe: it reset the very thing it was watching. That is
    the shape this repository has spent the day removing, arriving inside a check written
    to prevent it.
    """
    import importlib.util                                        # noqa: PLC0415

    import numpy as np                                           # noqa: PLC0415

    import borch                                                 # noqa: PLC0415

    def draw():
        return tuple(np.asarray(borch.rand(3).data).round(9).tolist())

    borch.manual_seed(0)
    untouched = draw()
    printing = dict(np.get_printoptions())

    borch.manual_seed(0)
    spec = importlib.util.spec_from_file_location("bt_cases", ROOT / "tests" / "cases.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.golden_cases(mod.golden_inputs())
    after = draw()

    assert after == untouched, (
        "building the case list consumed random numbers.\n"
        "  Every test that seeds and draws after this one gets a different stream, and\n"
        "  the failure appears wherever that stream is read — not here.")
    assert dict(np.get_printoptions()) == printing, (
        "building the case list changed numpy's print options.\n"
        "  Anything comparing a formatted number afterwards fails with a message about\n"
        "  formatting.")
    assert borch.is_grad_enabled(), "building the case list left autograd disabled"


def test_a_case_is_not_grouped_by_the_data_it_closes_over(cases):
    """Inputs group as strongly as helpers and mean nothing.

    Every `opt::` case closes over one input tensor, so *all 23 cases through `yin`
    failed* is true and says nothing about a cause. The cell contents settle it — a
    helper is callable and an input is not — and without that filter the first version
    put `yin`, `img` and `chans` beside `CrossEntropyLoss` with equal weight.
    """
    rows = blame.group(NINETY_FOUR, cases)
    named = {helper for helper, *_ in rows}
    data = named & {"yin", "xin", "img", "chans", "weights", "table"}
    assert not data, f"grouped by closed-over data rather than by machinery: {sorted(data)}"
