"""**The Python library carries no Korean.** Not in a message, not in a comment.

## Why the rule is this wide

It started narrower — "the error messages a user sees are in English" — and the check
looked for Korean inside `raise`. That check was evaded four times, each time by a shape
it had no reason to expect:

- a helper that wraps the wording (`_unsupported("a tensor exponent")`), so the Korean
  sits at a call site the regex does not open
- a helper's own default (`_no_complex128(what="이 연산")`), which is not a call site at
  all and interpolated straight into an English sentence
- a table of wording (`_ABSENT`'s values, `_bind_absent(_n, "희소 텐서")`), where the
  string never appears next to the helper's name
- `raise error(...)`, where the class arrived in a lowercase variable and
  `raise \\w*Error\\(` did not match

Each fix widened the list, and each time the list was the rule the next shape walked
around. A fifth gap needed no cleverness at all: `borchvision.py` was not in the list of
files.

So the rule stopped being about messages. Every comment and docstring in these files is
English now, which makes "no Korean anywhere" both true and checkable, and no new helper,
table, alias or default can step around it.

> **This rule was lost once and restored.** A merge (`3253fc9`) resolved seventy conflict
> hunks with "take the shared branch throughout" — sound for seven files that were two
> English translations of the same Korean, and wrong here, because this file was not a
> translation but a **strictly stronger rule**. The narrower construct-keyed version came
> back and nothing failed, since everything was already English by then. Measured on
> restoration: Korean planted in a `raise ValueError(...)` in `borchvision.py` passed the
> narrow rule. A blanket resolution is itself a claim keyed above its evidence.

## What is allowed

Golden case names. `tests/cases.py` names its cases in Korean and those names are keys in
the committed `tests/golden.json`, so a docstring that cites one
(`opt::StepLR/이어서 학습하기`) is quoting an identifier rather than writing Korean prose.
Those names stay Korean by decision — the reasoning is at the top of `tests/cases.py` — so
this allowance is permanent, and the two checks below keep it small and honest.

## What this file does **not** cover

`borch-ts/src` is checked by construct, not by directory: `throw new …Error(`. That is the
shape of rule this file exists to warn against, and it is knowingly weak — sixteen Korean
strings reach a user through a named constant, a `console.error`, a `||` fallback label
and a table of wording, and not one of them is a throw site. It stays narrow because that
source is mid-translation under another hand; it becomes a whole-directory rule in the
same commit that finishes the pass, not before, since a check that is red on arrival gets
skipped rather than obeyed.
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANGUL = re.compile(r"[가-힣]")

# The Python library, all of it.
SOURCES = (
    ("borch", "*.py"),
    ("borch_webgpu", "*.py"),
    (".", "borchvision.py"),
)

# Golden case names cited from a docstring. Each is a key in `tests/golden.json`, and
# adding one here is a claim that it is. `test_the_quoted_case_names_exist` checks that
# claim against the file rather than against `tests/cases.py` — the names there are built
# with f-strings (`OPT_PREFIX + f"{name}/자취"`) and never appear as literals, so the
# source cannot answer the question and the golden can.
QUOTED_CASE_NAMES = (
    "repr::스칼라",
    "opt::StepLR/이어서 학습하기",
)

# **A throw site is not only a `raise`.** Some places hand the wording to a helper that
# throws inside it. Kept for the TypeScript side, where the whole-directory rule cannot
# land yet.
TS_THROW = re.compile(r"throw new \w*Error\(")


def _without_quoted_names(text):
    for name in QUOTED_CASE_NAMES:
        text = text.replace(name, "")
    return text


def _files():
    for folder, glob in SOURCES:
        yield from sorted((ROOT / folder).glob(glob))


def _sites(text, opener):
    """From the opener until the brackets balance — a message spanning lines is one site."""
    found = []
    for m in opener.finditer(text):
        start = m.end() - 1
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    found.append((text[:start].count("\n") + 1, text[start:i + 1]))
                    break
    return found


def test_the_python_library_carries_no_korean():
    """Every line of every file, rather than the lines a pattern found interesting."""
    bad = []
    for path in _files():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if HANGUL.search(_without_quoted_names(line)):
                bad.append(f"{path.relative_to(ROOT)}:{number}  {line.strip()[:90]}")
    assert not bad, (
        f"{len(bad)} lines carry Korean. The Python library is English throughout — "
        "messages, comments and docstrings alike.\n  "
        + "\n  ".join(bad[:40])
        + (f"\n  … and {len(bad) - 40} more" if len(bad) > 40 else ""))


def test_the_files_the_rule_covers_are_all_there():
    """A directory rule is only as wide as the files it actually opens.

    `borchvision.py` sits at the repository root and matched no folder glob, which is how
    five of its messages stayed Korean while a check said the library was clean. So the
    file list is asserted rather than assumed: if the library grows a module this rule
    does not reach, that is a hole and it should be loud.
    """
    covered = {p.resolve() for p in _files()}
    real = {p.resolve() for p in ROOT.glob("borch*/*.py")} | {(ROOT / "borchvision.py").resolve()}
    real = {p for p in real if "borch-ts" not in p.parts and "dist" not in p.parts}
    missed = sorted(str(p.relative_to(ROOT)) for p in real - covered)
    assert not missed, (
        "these Python files belong to the library and this rule does not open them:\n  "
        + "\n  ".join(missed) + "\n\nadd them to SOURCES.")


def test_the_quoted_case_names_exist():
    """An allowance that names something absent stops being an allowance.

    If a case is renamed and this list is not, the entry silently permits Korean that no
    longer quotes anything.

    A docstring may cite a case by its full name or by the tail alone, so a listed entry
    counts as present when it is a key or the end of one.
    """
    names = list(json.loads((ROOT / "tests" / "golden.json").read_text())["cases"])
    missing = [q for q in QUOTED_CASE_NAMES
               if not any(name == q or name.endswith(q) for name in names)]
    assert not missing, (
        "these are listed as quoted golden case names and are not keys in "
        f"tests/golden.json: {missing}. Either the case was renamed — in which case fix "
        "the docstring citing it — or the allowance is stale.")


def test_the_allowance_is_used():
    """Every entry earns its place, or it is dead permission."""
    text = "\n".join(p.read_text() for p in _files())
    unused = [name for name in QUOTED_CASE_NAMES if name not in text]
    assert not unused, (
        f"listed as allowed and cited nowhere: {unused}. Remove them — an allowance "
        "nothing uses is a hole waiting for something else.")


def test_the_typescript_throw_sites_are_english():
    """The narrow half, over `borch-ts/src`, kept until the whole-directory rule can land.

    An error message is what an English reader meets **the first time anything breaks**.
    Even after the documentation and the site were entirely English, 81% of the messages
    were Korean — the largest Korean surface left, at 303 throw sites across the three
    libraries.
    """
    bad = []
    for path in sorted((ROOT / "borch-ts" / "src").glob("*.ts")):
        for line, site in _sites(path.read_text(), TS_THROW):
            if HANGUL.search(site):
                first = HANGUL.search(site)
                snippet = site[max(0, first.start() - 30):first.start() + 40]
                bad.append(f"{path.relative_to(ROOT)}:{line}  …{snippet.strip()}…")
    assert not bad, (
        f"{len(bad)} thrown messages are Korean. They are the surface a user meets first "
        "and have to be English.\n  " + "\n  ".join(bad[:40])
        + (f"\n  … and {len(bad) - 40} more" if len(bad) > 40 else ""))


# ── the one sentence three implementations have to share ──────────────
#
# Twenty-one golden cases ask whether a name absent from the browser subset is refused
# **with the same wording** everywhere: `'half' is not defined` is indistinguishable from
# a typo, and three implementations refusing in three sentences read to a learner as three
# different products. Each side finds that sentence by looking for a fragment of it, and
# the case returns a verdict word.
#
# **The verdict word is what the golden freezes, so the sentences could drift apart under
# a green suite — and they did.** Translating the Python message left `tests/cases.py`
# looking for `is not in the browser subset` while `borch-ts/test/cases.ts` still looked
# for `브라우저 축소판에 없습니다`. Both sides self-consistent, both returning the same
# word, twenty-one cases green, and the property the group exists to check was false for a
# week. A check that lets each side supply its own input measures self-consistency and
# reports it in the vocabulary of agreement.
FRAGMENT_SITES = (
    ("tests/cases.py", re.compile(r'mark = "([^"]+)"')),
    ("borch-ts/test/cases.ts", re.compile(r'const MARK = "([^"]+)"')),
    ("borch-ts/src/errors.ts", re.compile(r'absent: "([^"]+)"')),
)


def test_the_three_sides_look_for_one_sentence():
    """Each end of the contract, and the library that has to satisfy it."""
    found = {}
    for rel, pattern in FRAGMENT_SITES:
        hits = pattern.findall((ROOT / rel).read_text(encoding="utf-8"))
        assert hits, (
            f"{rel} no longer states the fragment this reads. It is one half of a wording "
            "contract, so a rename has to bring this pattern with it — matching nothing "
            "here would pass in silence, which is how the contract broke the first time.")
        assert len(set(hits)) == 1, f"{rel} states {len(set(hits))} different fragments: {sorted(set(hits))}"
        found[rel] = hits[0]

    # **The two searching sides are compared to each other, not each to the message.**
    # Containment alone is too weak in a way worth naming: shortening one side's fragment
    # keeps it a substring, so it passes while checking less — and the limit of that is a
    # side looking for `browser`, agreeing with everything and asserting nothing. Measured:
    # cutting `is not in the browser subset` down to `not in the browser subset` on the TS
    # side passed the containment form of this test.
    python_looks_for = found["tests/cases.py"]
    ts_looks_for = found["borch-ts/test/cases.ts"]
    assert python_looks_for == ts_looks_for, (
        f"tests/cases.py looks for {python_looks_for!r} and borch-ts/test/cases.ts looks "
        f"for {ts_looks_for!r} — both suites pass while they describe different refusals, "
        "because what the golden freezes is the verdict word and not the sentence.")

    emitted = found["borch-ts/src/errors.ts"]
    assert ts_looks_for in emitted, (
        f"borch-ts/test/cases.ts looks for {ts_looks_for!r} and borch.ts says {emitted!r}")

    # **And the sentence has to be one the Python library actually produces.** Two files
    # agreeing about a string neither library emits would pass everything above.
    import borch

    try:
        borch.tensor([1.0]).half()
    except Exception as exc:                                        # noqa: BLE001
        said = str(exc)
    else:
        raise AssertionError("`.half()` no longer refuses — the twenty-one cases are stale")
    assert found["tests/cases.py"] in said, (
        f"the fragment the cases look for is {found['tests/cases.py']!r} and borch says "
        f"{said!r}")
