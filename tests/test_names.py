"""Checks about the case **names.**

It looks at the table itself rather than the values. The golden data lands in a dictionary
keyed by name while the table is a list, and anything that disappears quietly between the
two is caught here.
"""

import collections

import cases as cases_mod


def test_case_names_are_unique():
    """A duplicate name makes **the earlier case disappear.**

    `golden.dump` stores with `data[name] = ...` and `export_json` stores into a dictionary
    too. With two cases of the same name, the later overwrites the earlier and the earlier's
    expectation survives nowhere. The case is in the table and nobody asks about it, and the
    counts do not give it away — dump counts the list's length (duplicates included) and the
    comparison counts the dictionary's (duplicates excluded).
    """
    names = [name for name, _ in cases_mod.golden_cases(cases_mod.golden_inputs())]
    dup = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dup, "duplicate case names — the earlier one is overwritten and lost:\n  " + "\n  ".join(dup)
