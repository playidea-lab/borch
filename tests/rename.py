"""Renaming done by a machine rather than by hand.

    uv run python tests/rename.py --check     # counts what is left
    uv run python tests/rename.py --apply     # changes them

**Sweeping forty-one files by hand leaves one behind, quietly.** And the one left behind
does not blow up at once as an import error; it surfaces at runtime somewhere the name is
only ever a string, such as the browser runner's `?lib=` query. Exactly the shape of defect
this repository keeps catching.

There are three rules and **the order matters** — the longer names go first. Changing
`browsertorch` first does still take `browsertorch_webgpu` somewhere, but it collides with
`browsertorch_vision` becoming `borch_vision` and leaves no way to confirm what went where.
Descending by length prevents that.

`--check` stays because this file is not single-use. The next time a name changes, the same
work is not done by hand again.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Longest first. Changing a short one first catches the front of a longer name.
#
# What is in the table now is **the binding inheriting the sister's name.** The TF.js sister
# was deleted and its name handed to the binding on borch.ts — what the name means to a user
# (browser, GPU) is unchanged and only the floor beneath it moved.
#
# **An underscore, a hyphen and a dot are different things.** Only `borch_ts` (the Python
# package) changes; `borch-ts` (the TypeScript directory) and `borch.ts` (the name used in
# prose) stay. That these three look alike to the eye is what called for this rename in the
# first place.
#
# **So a retired token must not appear in a new name.** Python cannot use a dot, so writing
# `borch.ts` as an identifier makes it `borch_ts`, which is exactly the retired package's
# name. `_touches_borch_ts` (does it call borch.ts) went through this tool and came out as
# `_touches_borch_webgpu` (does it call the binding) — **the code changes consistently so
# nothing breaks, what is wrong is only the meaning, and no check cries.** New names use a
# word that is not in the table, such as `ts` or `the_ts_side`.
RULES = [
    ("borch_ts", "borch_webgpu"),
    # **`torchvision` has no underscore.** This project's point is leaving torch's structure
    # as it is, and `borch_vision` was adding an underscore the corresponding name does not
    # have. `import borchvision as torchvision` is what keeps that point.
    #
    # The distribution name (`[project] name`) is left alone — that is a separate problem,
    # and `borch` on PyPI is currently someone else's (Desupervised, probabilistic
    # programming). Moving the module name and the distribution name together hides which
    # changed because of what.
    ("borch_vision", "borchvision"),
    # **A different spelling is a different rule.** Running `browsertorch` → `borch` did not
    # catch this name — the rule was lowercase and the class was `BrowserTorchError`. There
    # are no word boundaries inside a lowercase run (`browsertorch`), so the tool has no way
    # to infer `BrowserTorch`. So it is **written here rather than generated**, and in place
    # of generating it, `RETIRED` counts what is left case-insensitively.
    ("BrowserTorch", "Borch"),
]

# **Names that must never appear again, in any spelling.** The branches `RULES` cannot catch
# are caught here — counted case-insensitively and **not fixed.** What each becomes differs
# per spelling (`BrowserTorchError` is `BorchError`, not `Borchtorcherror`), and a machine
# making that judgement quietly produces strange names.
#
# Why this exists: after a rename ran, `BrowserTorchError` remained in 37 places and the
# tool answered "nothing to change". It happened **outside the tool's own rules** so the tool
# could not see it, and that is precisely the thing this tool exists to prevent.
RETIRED = ["browsertorch"]

# **Places that write the old name on purpose.** A sentence telling history has to call the
# old name by its name — the same place `test_docs.py` records as "changing a past number to
# the current one is forging history". Catching those too makes the check cry wolf, and a
# check that cries wolf gets switched off.
#
# **A reason per file.** With no reason to write, it is not history but a leftover.
HISTORY = {
    "tests/test_site.py": "the sentence recording why the site links went stale without breaking during the rename",
}

# `.claude` and `.mcp.json` are this machine's configuration, not the project's — they match
# because a repository path is written in them, and changing them touches someone else's tool
# settings.
SKIP_DIRS = {".git", "node_modules", "vendor", "dist", "__pycache__", ".venv", ".claude"}
# Lock files are left alone — the tools regenerate them, and fixing one here breaks its
# hashes. This file itself is excluded too: a rules table that becomes
# `("borch_webgpu", "borch_webgpu")` makes the next run find nothing and not say so.
SKIP_FILES = {"uv.lock", "package-lock.json", ".mcp.json", pathlib.Path(__file__).name}
SUFFIXES = {".py", ".ts", ".js", ".html", ".json", ".md", ".yml", ".toml", ".cfg", ".txt"}


def targets():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in SUFFIXES:
            continue
        if p.name in SKIP_FILES or SKIP_DIRS & set(p.relative_to(ROOT).parts):
            continue
        yield p


def survivors():
    """Where a retired name survives **in any spelling.** Counted only, never fixed.

    What remains after every rule in `RULES` has run is caught here. Case is ignored because
    the branch that gets missed has always been case — running a rename with one lowercase
    rule and leaving `BrowserTorchError` in 37 places is the example.
    """
    found = []
    for path in targets():
        if path.relative_to(ROOT).as_posix() in HISTORY:
            continue
        text = path.read_text(encoding="utf-8")
        for old, fresh in RULES:
            text = text.replace(old, fresh)
        low = text.lower()
        spellings = set()
        for gone in RETIRED:
            at = low.find(gone)
            while at != -1:
                spellings.add(text[at:at + len(gone)])
                at = low.find(gone, at + 1)
        if spellings:
            found.append((path.relative_to(ROOT), spellings))
    return found


def main(argv):
    apply = "--apply" in argv
    touched, remaining = [], 0
    for path in targets():
        text = path.read_text(encoding="utf-8")
        new = text
        for old, fresh in RULES:
            new = new.replace(old, fresh)
        if new == text:
            continue
        hits = sum(text.count(old) for old, _ in RULES)
        remaining += hits
        touched.append((path.relative_to(ROOT), hits))
        if apply:
            path.write_text(new, encoding="utf-8")

    # The label has to read at zero. Carrying the Korean across as it stood gave
    # "there is something to change — 0 files", which contradicts itself.
    verb = "changed" if apply else "to change"
    print(f"{verb} — {len(touched)} files, {remaining} places")
    for rel, hits in touched:
        print(f"  {hits:4d}  {rel}")

    left = survivors()
    if left:
        print("\n✘ a retired name survives — a spelling the rules did not catch:")
        for rel, spellings in left:
            print(f"  {rel}: {' · '.join(sorted(spellings))}")
        print("  decide the new name and write the spelling into `RULES` as it stands — what\n"
              "  each becomes differs per spelling, so the tool does not decide it.")

    # In check mode, anything left is reported through the exit code so CI can see it.
    # **A retired name is a failure even after `--apply`** — it means something could not be
    # changed, and finishing successfully there is exactly the moment that was missed last
    # time.
    return 1 if left or (not apply and touched) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
