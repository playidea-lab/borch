"""The seven that draw from a distribution and fill in place — **the half the golden cases cannot ask about.**

The golden cases' contract is that real torch produces the answer, and their power over
three implementations comes from that. A distribution's properties are outside that
contract — the answer comes from **a predicate we chose** rather than from torch, so a row
in the table would measure two implementations and not torch. And it means choosing a sample
count and a tolerance, which needs somewhere to choose them.

**Shapes, types and refusals are in the golden cases** (`inplace::분포::*`). What is here is
the value side.

## Asking only about the range is shallow

`uniform_(a, b)` passes "inside [a, b]" **even filling everything with `a`.** The same story
as a predicate that is always true — it cannot be told from an implementation returning a
constant. So along with the range it looks at **whether there are many distinct values**, and
whether the mean and standard deviation sit near the theoretical ones.

## Seeds are not asked about here

The three generators differ (a place already accepted at `randn`). The same seed giving the
same numbers means something **only within each of them**, so the core asks here and borch.ts
is asked by parity. Putting something the three cannot be asked about into the table leaves
the next person unable to read what that row measures.
"""

import numpy as np
import pytest

import borch

N = 20000          # the sample count. The tolerances below are tied to it.


def drawn(name, *args, dtype=None, size=N):
    x = (borch.zeros(size) if dtype is None
         else borch.tensor(np.zeros(size, dtype=dtype)))
    getattr(x, name)(*args)
    return np.asarray(x.tolist(), dtype=np.float64)


def test_uniform_is_not_a_constant():
    """**What a range-only check cannot catch.** Filling everything with `a` passes the range."""
    got = drawn("uniform_", -1.0, 3.0)
    assert got.min() >= -1.0 and got.max() < 3.0, "outside the range"
    assert len(np.unique(got)) > N // 2, "too few distinct values — is this a constant"
    assert abs(got.mean() - 1.0) < 0.05, got.mean()
    # A uniform distribution's standard deviation is (b-a)/√12. A constant gives 0, and a range check misses it.
    assert abs(got.std() - 4.0 / np.sqrt(12)) < 0.05, got.std()


def test_normal_matches_its_mean_and_spread():
    got = drawn("normal_", 5.0, 2.0)
    assert abs(got.mean() - 5.0) < 0.05, got.mean()
    assert abs(got.std() - 2.0) < 0.05, got.std()


def test_exponential_is_positive_with_mean_one_over_lambda():
    got = drawn("exponential_", 2.0)
    assert (got > 0).all(), "the exponential is positive"
    assert abs(got.mean() - 0.5) < 0.02, got.mean()


def test_log_normal_is_positive_and_its_log_is_normal():
    """**It is normal after taking logs.** The mean of the values themselves fits badly, the tail being heavy."""
    got = drawn("log_normal_", 0.0, 1.0)
    assert (got > 0).all(), "the log-normal is positive"
    logged = np.log(got)
    assert abs(logged.mean()) < 0.05, logged.mean()
    assert abs(logged.std() - 1.0) < 0.05, logged.std()


def test_cauchy_has_heavy_tails_and_no_useful_mean():
    """**The mean cannot ask it** — Cauchy has no mean. It is looked at through the median and the quartiles."""
    got = drawn("cauchy_", 1.0, 0.5)
    assert abs(np.median(got) - 1.0) < 0.05, np.median(got)
    # The interquartile width is 2·sigma.
    spread = np.percentile(got, 75) - np.percentile(got, 25)
    assert abs(spread - 1.0) < 0.1, spread
    # Pins the heaviness of the tail itself — a normal would not reach this far.
    assert np.abs(got - 1.0).max() > 20, "the tail is too well behaved"


def test_geometric_is_a_count_starting_at_one():
    """torch's `geometric_` is **the number of trials up to the first success**, so it starts at 1, not 0."""
    got = drawn("geometric_", 0.3)
    assert got.min() >= 1, got.min()
    assert (got == np.floor(got)).all(), "these have to be integers"
    assert abs(got.mean() - 1 / 0.3) < 0.1, got.mean()


def test_geometric_fills_an_integer_tensor_too():
    """**Being discrete, it has an answer inside an integer slot.** The one that parts from the five continuous ones."""
    got = drawn("geometric_", 0.5, dtype=np.int64)
    assert got.min() >= 1


def test_random_respects_its_range_and_uses_all_of_it():
    got = drawn("random_", 0, 5, dtype=np.int64, size=2000)
    assert got.min() >= 0 and got.max() < 5
    # **It looks at whether the endpoints appear.** An implementation using a narrower range passes a range check.
    assert set(np.unique(got)) == {0, 1, 2, 3, 4}, np.unique(got)


def test_random_on_bool_is_zero_or_one():
    x = borch.tensor(np.zeros(200, dtype=bool))
    x.random_()
    got = np.asarray(x.tolist())
    assert set(np.unique(got)) <= {False, True}
    assert len(np.unique(got)) == 2, "only one value came out"


@pytest.mark.parametrize("name,args", [
    ("normal_", ()), ("uniform_", ()), ("exponential_", ()),
    ("cauchy_", ()), ("log_normal_", ()), ("geometric_", (0.5,)),
    ("random_", ()),
    # **`bernoulli_` was missing.** It is the one implemented outside the table, so this list
    # did not count it, and meanwhile it was using numpy's **global** generator, which the
    # seed never reached. A list counting the seven added later cannot see the one that was
    # there first.
    ("bernoulli_", (0.5,)),
])
def test_the_same_seed_gives_the_same_draw(name, args):
    """**A question meaningful only within each implementation.** The three generators differ, so
    the table cannot ask it.

    An unhonoured seed makes an experiment irreproducible, and the symptom is "the values
    differ slightly", which a value comparison does not catch.
    """
    def once():
        borch.manual_seed(7)
        x = borch.zeros(50)
        getattr(x, name)(*args)
        return x.tolist()

    assert once() == once()
