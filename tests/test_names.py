"""A check on the case **names.**

It looks at the table itself rather than at values. The golden is stored in a
dictionary keyed by name and the table is a list. Anything that vanishes quietly
between the two is caught here.
"""

import collections

import cases as cases_mod


def test_case_names_are_unique():
    """A duplicate name makes **the earlier case disappear.**

    `golden.dump` stores with `data[name] = ...` and `export_json` stores into a
    dictionary too. With two cases under one name the later overwrites the
    earlier, and the earlier's expected value survives nowhere. The table holds a
    case that nobody asks about, and counting alone cannot notice — dump counts
    the list's length (duplicates included) and the comparison counts the
    dictionary's (duplicates excluded).
    """
    names = [name for name, _ in cases_mod.golden_cases(cases_mod.golden_inputs())]
    dup = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dup, ("case names collide — the earlier one is overwritten and "
                     "lost:\n  ") + "\n  ".join(dup)
