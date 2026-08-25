"""The documentation's Python examples are **run**, not read.

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


def test_the_readme_vision_example_prints_the_shape_it_claims():
    """The `borchvision` entry block — **it was two imports and nothing ran them.**

    The paragraph above it in the README says the first ten lines of an introductory
    PyTorch tutorial are torchvision, and that is what this library's vision half
    exists for. What stood under that sentence was `import borchvision as torchvision`
    and one more import, which prove the module exists and nothing else: an import
    resolves whatever `ToTensor` then does with the axes.

    So the block is a pipeline now, and the value it claims is the shape. `(H, W, C)`
    in and `(C, H, W)` out is the single fact a reader copying these lines is relying
    on, and it is torchvision's convention rather than an obvious one — an
    implementation that skipped the transposition would still hand back a tensor of
    the right size in the wrong order.
    """
    blocks = [(line, body) for line, body in _python_blocks()
              if "borchvision" in body and "print(" in body]
    assert len(blocks) == 1, (
        f"expected one runnable borchvision example, found {len(blocks)} — the entry "
        "block under \"torchvision — `transforms` only\" is the one meant here")
    line, body = blocks[0]

    claimed = [NOTE.search(raw) for raw in body.splitlines() if raw.startswith("print(")]
    assert claimed and claimed[0], (
        f"README.md:{line} — the print no longer carries the value it claims. Put the "
        "expected shape back as a trailing comment.")
    said = claimed[0].group(0).split("#", 1)[1].strip()

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(body, str(README), "exec"), {})                # noqa: S102
    got = buffer.getvalue().strip()
    assert got == said, (
        f"README.md:{line} — the example claims it prints {said!r} and it prints {got!r}")


# ── the same message, quoted in a second document ─────────────────────
#
# ROADMAP.md records what the matmul message looked like when it was Korean, and then
# says "the same call today gives:" and quotes the English one. The first block is
# history and must not move — this file's neighbour, `test_docs.py`, has a rule saying
# changing a past number to the current one is forging the record. The second is
# **present tense**, and present tense rots.
#
# It also makes that sentence live in two documents at once, which is the shape that
# already failed here: `test_site.py` records a bundle size copied from the README into
# two site pages, where a stale source produced stale copies. The README's copy is
# checked by the session block above; this checks the other one against the same source
# of truth rather than against the README, so neither document is the original.
TODAY = "the same call today gives:"


def test_the_roadmap_quotes_the_message_the_library_emits_today():
    """The present-tense half of a document whose other half is deliberately frozen."""
    text = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    after = text.split(TODAY)
    assert len(after) == 2, (
        f"ROADMAP.md no longer says {TODAY!r} — the quote it introduces is the thing this "
        "checks, so a rewrite has to bring this pattern with it.")
    quoted = re.search(r"```\n(.*?)```", after[1].replace("> ", ""), re.S)
    assert quoted, "ROADMAP.md stopped quoting a message under that sentence"

    import borch

    try:
        borch.randn(3, 4) @ borch.randn(3, 2)
    except Exception as exc:                                        # noqa: BLE001
        got = str(exc)
    else:
        raise AssertionError("the matmul no longer refuses — the whole passage is stale")

    assert quoted.group(1).strip() == got.strip(), (
        "ROADMAP.md quotes a message the library does not emit:\n"
        f"  says: {quoted.group(1).strip()}\n  does: {got.strip()}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
