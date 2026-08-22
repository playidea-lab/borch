"""The verdict is read **from state**, not from the report's prose.

Six runners judged in the same shape:

    return 0 if "전부 통과" in result["text"] else 1

A judgement that scans a sentence goes quietly wrong three ways.

1. **Reword it and the answer changes.** Where the page writes "all passed" differently,
   the runner reads every run as a failure; where the word happens to appear in a failing
   line, it reads a pass.
2. **A partial pass reads as a pass.** `readme.py` did. Its verdict word was
   `"그대로 돌고"`, and that phrase sits in the success sentence of **both** README
   examples — so with the first example failing and only LBFGS passing, the runner exited
   0. Leaving an example whose loss does not go down in the documentation.
3. **One document is read two different ways.** `cost.html` judged with
   `!text.includes("갈렸다")` and `cost.py` with `"전부 통과" in text` — two judgements of
   one report, and nothing to say which wins when they disagree.

So the page hands over `checks` (`{name, ok, note}`) as they are and the counting happens
here. The prose stays as the shadow a person reads.

**Absent `checks` is an error rather than a pass.** Reading their absence as "0 failures"
brings back, under another name, exactly the silence this file exists to remove.
"""

import sys


def failures(result, what):
    """The failing checks. It stops where it stands if `checks` cannot be found.

    Args:
        result: the object the page handed over. It has to carry `checks`.
        what: which runner is measuring — used in the error sentence.

    Returns:
        The list of failing checks. Empty when everything passed.

    Raises:
        SystemExit: when `checks` is absent or empty. It means there is no state to judge,
            and that is not a pass.
    """
    checks = result.get("checks")
    if not isinstance(checks, list) or not checks:
        raise SystemExit(
            f"{what}: the page handed over no `checks` — there is no state to judge.\n"
            f"  keys received: {sorted(result)}\n"
            "  `report()` has to return {text, checks} and the html has to carry it.")
    return [c for c in checks if not c.get("ok")]


def verdict(result, what, stream=sys.stderr):
    """The runner's exit code. Where something failed, it names **what** failed."""
    bad = failures(result, what)
    if not bad:
        return 0
    print(f"\n**{what} — {len(bad)} failed**", file=stream)
    for c in bad:
        note = f" — {c['note']}" if c.get("note") else ""
        print(f"  ✗ {c.get('name', '(unnamed)')}{note}", file=stream)
    return 1
