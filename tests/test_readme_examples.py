"""The README's Python examples are **run**, not read.

`borch-ts/test/readme.ts` already does this for the TypeScript side, and its first
paragraph gives the reason: an example in a document is a claim about what the code does,
and a claim nobody executes is a claim nobody checked. That file records two occasions
where this repository shipped install instructions that did not work, and one more where
an example added the same day was wrong the moment it was written — the author edited the
document and not the file that runs it.

There was no equivalent on the Python side. The README's central example block — the one
whose paragraph promises the library **stops loudly rather than quietly producing a
different value** — was four statements and four outputs that nothing had ever compared
against the library.

**The expectations are read out of the README rather than written here.** Copied into this
file they would be a second original, and then a wrong document and a right test agree
with each other while the reader is misled. Editing the block edits the test.
"""

import io
import pathlib
import re
from contextlib import redirect_stdout

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# A trailing note on an output line — `tensor([2.])        # what it means`. Two spaces or
# more, so a `#` inside an actual output is left alone.
NOTE = re.compile(r"\s{2,}#.*$")


def _python_blocks():
    text = README.read_text(encoding="utf-8")
    for m in re.finditer(r"```python\n(.*?)```", text, re.S):
        yield text[:m.start()].count("\n") + 1, m.group(1)


def _doctest_block():
    """The one block written as a session. Its absence is a failure, not a skip."""
    found = [(line, body) for line, body in _python_blocks() if ">>>" in body]
    assert len(found) == 1, (
        f"expected one `>>>` block in the README, found {len(found)} — "
        "if the examples were reorganised, this file has to follow them.")
    return found[0]


def _pairs(body):
    """`>>> statement` and the lines under it, up to the next statement or a blank line."""
    out, stmt, said = [], None, []
    for raw in body.splitlines():
        if raw.startswith(">>> "):
            if stmt is not None:
                out.append((stmt, "\n".join(said)))
            stmt, said = raw[4:], []
        elif not raw.strip():
            if stmt is not None:
                out.append((stmt, "\n".join(said)))
            stmt, said = None, []
        elif stmt is not None:
            said.append(NOTE.sub("", raw).rstrip())
    if stmt is not None:
        out.append((stmt, "\n".join(said)))
    return out


def _answer(statement, env):
    """What a session would print — the value's repr, or the exception as it reads."""
    try:
        value = eval(statement, env)                                # noqa: S307
    except Exception as exc:                                        # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return repr(value)


def test_the_readme_session_block_prints_what_it_says():
    """Every statement in the block, against the library.

    Both halves matter and they fail differently. A changed value is caught by the
    comparison; a changed *sentence* — the wording of a refusal — is caught only because
    the exception is compared as the reader sees it, name and message together. This
    repository has translated those messages once already, and two documents quoting them
    went stale in the same week.
    """
    line, body = _doctest_block()
    pairs = _pairs(body)
    assert len(pairs) >= 4, f"README.md:{line} has {len(pairs)} statements — it had four"

    import borch

    env = {"torch": borch}
    wrong = []
    for statement, said in pairs:
        got = _answer(statement, env)
        if got != said:
            wrong.append(f"  >>> {statement}\n  says: {said}\n  does: {got}")
    assert not wrong, (
        f"README.md:{line} — the session block says something the library does not:\n"
        + "\n\n".join(wrong))


def test_the_readme_autograd_example_prints_what_its_comment_claims():
    """The first example in the file, whose expected value is a comment on the print.

    It is the first code a reader meets and it states one number. Nothing was comparing
    that number either.
    """
    blocks = [(line, body) for line, body in _python_blocks()
              if "backward()" in body and "print(" in body]
    assert len(blocks) == 1, f"expected one autograd example, found {len(blocks)}"
    line, body = blocks[0]

    claimed = [NOTE.search(raw) for raw in body.splitlines() if raw.startswith("print(")]
    assert claimed and claimed[0], (
        f"README.md:{line} — the print no longer carries the value it claims, so nothing "
        "here can be checked. Put the expected value back as a trailing comment.")
    said = claimed[0].group(0).split("#", 1)[1].strip()

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(body, str(README), "exec"), {})                # noqa: S102
    got = buffer.getvalue().strip()
    assert got == said, (
        f"README.md:{line} — the example claims it prints {said!r} and it prints {got!r}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
