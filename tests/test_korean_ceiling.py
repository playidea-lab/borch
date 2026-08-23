"""**The Korean left in `borch-ts` may shrink and may not grow.**

The rest of the repository is English: the Python library, every root document, the
configuration, the workflows, the site's own pages. What remains is `borch-ts`, and it is
under another session's hand while features land in it daily.

## Why a ceiling rather than a rule

The obvious check — "no Korean in `borch-ts`" — is the right rule and it cannot land yet:
red on arrival, it would be skipped rather than obeyed, which is worse than absent because
it teaches people to switch checks off. So the rule that *can* land today is the one that
stops the loss getting larger.

**The measurement that argued for it.** Over thirty hours `borch-ts/src` went from 40,698
Korean characters to 45,480 and `borch-ts/test` from 44,491 to 52,721, across fifteen
commits of which one was a translation. Nothing was going wrong — features were landing,
and each arrived with Korean comments, because that is what the surrounding file looks
like. Waiting was not holding position; it was losing ground at about 11% a day.

## What a green run here does **not** mean

It does not mean a directory is English. It does not mean a file is. **The ceiling is a
derivative** — it answers "did this grow" and is silent about every absolute fact, and a
green run is compatible with 40,000 Korean characters sitting exactly where they were.

This is written down because it already misled somebody. A session translated the
characters its own commit had added, ran this, saw green, and reported "vision.ts is
translated" — the check answered *did this grow* and the sentence claimed *is this
English*. The file still held 2,883 Korean characters, and the report was believed
downstream until somebody grepped.

Having a green test in front of you is what makes it easy to stop looking. To claim a
file is English, count it:

    grep -c "[가-힣]" path/to/file

## What it costs

Nothing to read and nothing to run. It asks only that **new comments in these directories
be written in English**, which is the direction the repository has already taken
everywhere else. Translating an existing block lowers the number, and lowering it is
always allowed.

## When a number moves

Lower it. The ceilings below are a record of a debt, not a budget to spend: after a
translation pass, set them to what was measured and the ratchet holds the new floor. The
failure message prints the number to write.

If a genuinely new Korean string has to go in — a case name, a fixture, something quoted
from a Korean page — raise the ceiling **in the same commit**, with the reason in the
commit message. That makes it a decision somebody made rather than a number that drifted.

## Why the failure message asks git where the number came from

A ceiling says *how much*. It never said *measured when, in whose tree, over which files*,
and both of the ways this file has misled somebody are that missing half.

**Sideways.** A translation pass lowered `borch-ts/test` to 27,201 in a tree that did not
contain another branch's new rows. Those rows carried 204 Korean characters. Both branches
were right about their own tree and the merge was over by exactly that much, so main went
red on arrival and the first guess was that the merge had reverted the translation. It had
not. A ceiling is measured at a moment, and with two branches there are two moments.

**Downward.** The tightening test below demands the ceiling follow the count down. Move a
big file out of the directory and the count falls with no translation behind it, and the
test will *insist* the new floor be locked in — printing the line to paste. A drop nobody
earned, arriving with instructions. A ratchet cannot be surprised: it compares to its own
last value, so any movement in the good direction agrees with it.

Neither is detectable from in here. Both are the same shape as the other checks that have
gone quiet in this repository — the check is right and its input is wider than the check
claims — and a check cannot measure its own boundary. So the ceiling does not try. It
makes the **red run explain itself**, which is the hour that was actually lost.

The provenance is asked of git and not written down beside the number. Written down it
would be one more fact copied by hand, going stale the first time somebody tightens the
count without touching the note — which is the other failure this repository keeps having,
and it is the one that *is* preventable: where a fact already has a home, read it there.
"""

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANGUL = re.compile(r"[가-힣]")
SUFFIXES = (".ts", ".py", ".html")

# Measured 2026-08-22. Lower these after a translation pass; see the module docstring
# before raising one.
CEILINGS = {
    "borch-ts/src": 18,          # 3132 → 3133. **One character, and it is the same key as the last two rises.**
    # Two case names moved from `브라우저는거절` to `우리는거절` when the core stopped
    # granting `.double()` in name and answering in `float32` — so a place where the
    # implementations parted became a place where they agree, and the name had to say
    # so. Shorter by two characters each, and a gap ledger row's marker made up the
    # difference, which is why this is +1 rather than −4.
    #
    # The markers (`아직`, `별칭`, `파이썬`, `없음`) and these verdict words are keys
    # that `test_alias_rows.py`, `test_site.py` and the golden case names match on.
    # They move when those move and not before.
    # 3133 → 3135. The `act::` row returning to the ledger, whose marker word `아직`
    # is what `test_site.py` matches on to tell owed work from declined work.
    #
    # 3135 → 3139. **+4, and all four are markers being named rather than used.**
    # `opt::` moved from `아직` to `없음` when borch.ts gained `maximize`, `amsgrad`,
    # `centered`, `momentum` and `decoupled_weight_decay`, and the note beside the row
    # says which marker it left and which it took — a change of ledger verdict that
    # does not say what it changed from is the kind this repository keeps finding.
    # The twelve case names that came with it are ASCII and cost nothing here.
    #
    # 3139 → 3141. **+2, the same kind again**: the `act::` row's cause was rewritten
    # (no second shader is needed — `copyFrom` is the write-back and borch.ts has it),
    # and the note says the marker `아직` was right while the reason under it was not.
    # Naming the marker costs two characters and is the whole point of the note.
    # 3141 → 3143. One golden case name, `배율` — `scaled_dot_product_attention`'s
    # `scale`, which was accepted and dropped. Names are keys: the Python and
    # TypeScript tables have to agree on the string or the row reconciles against
    # nothing.
    # 3143 → 3145. One case name, `표` — `nonzero`'s table form beside its new
    # tuple form. Both had to be asked: `torch.nonzero` is unreadable to `inspect`,
    # so the argument had never been compared on either axis.
    "borch-ts/test": 3145,
}


def _countable(folder):
    """The files this rule counts. `_provenance` measures the same set in an old tree."""
    return [path for path in sorted((ROOT / folder).rglob("*"))
            if path.is_file() and path.suffix in SUFFIXES
            and "dist" not in path.parts and "node_modules" not in path.parts]


def _count(folder):
    total, per_file = 0, {}
    for path in _countable(folder):
        found = len(HANGUL.findall(path.read_text(errors="ignore")))
        if found:
            per_file[str(path.relative_to(ROOT))] = found
            total += found
    return total, per_file


def _git(*args):
    """git, or None when it cannot answer. A message is never worth failing over."""
    try:
        done = subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _ceiling_commit(folder):
    """The commit that last wrote this folder's number, and the date, from `git blame`.

    Asked rather than recorded: a hash written into the table above would be correct on
    the day it was pasted and silently wrong from the first tightening that forgot it.
    """
    here = pathlib.Path(__file__)
    for number, line in enumerate(here.read_text().splitlines(), 1):
        if line.strip().startswith(f'"{folder}":'):
            blamed = _git("blame", "--porcelain", f"-L{number},{number}", "--",
                          str(here.relative_to(ROOT)))
            if not blamed:
                return None
            sha = blamed.split()[0][:12]
            return sha, (_git("log", "-1", "--format=%ad", "--date=short", sha) or "?")
    return None


def _files_at(sha, folder):
    """How many countable files that folder held in that commit's tree."""
    listing = _git("ls-tree", "-r", "--name-only", sha, "--", folder)
    if listing is None:
        return None
    return sum(1 for name in listing.splitlines() if name.endswith(SUFFIXES))


def _provenance(folder):
    """Where this number came from, and what has moved under it since.

    Two sentences, and both of them are about the same missing half: a ceiling is a
    measurement of one tree at one moment, and it is being compared against another.
    """
    found = _ceiling_commit(folder)
    if not found:
        return ""
    sha, date = found
    now_files = len(_countable(folder))
    said = [f"this ceiling was last written in {sha} ({date})"]

    merges = _git("rev-list", "--count", "--merges", f"{sha}..HEAD")
    if merges and merges != "0":
        said.append(
            f"{merges} merge(s) have landed since — a merge can put two numbers together "
            "that were each correct in their own branch, which reads exactly like growth")

    then = _files_at(sha, folder)
    if then is not None and then != now_files:
        way = "left" if then > now_files else "arrived"
        said.append(
            f"the directory held {then} countable files then and {now_files} now, so "
            f"files have {way} — a count that moved with them did not move by translation")
    return "\n  ".join(said)


def test_the_korean_left_in_borch_ts_does_not_grow():
    """The ceiling, per directory, with the worst files named when it is breached."""
    over = []
    for folder, ceiling in CEILINGS.items():
        total, per_file = _count(folder)
        if total > ceiling:
            worst = sorted(per_file.items(), key=lambda kv: -kv[1])[:5]
            where = _provenance(folder)
            over.append(
                f"{folder}: {total} Korean characters against a ceiling of {ceiling} "
                f"(+{total - ceiling})\n    "
                + "\n    ".join(f"{n}  {c}" for n, c in worst)
                + (f"\n  {where}" if where else ""))
    assert not over, (
        "Korean grew in a directory that is being translated:\n  " + "\n  ".join(over)
        + "\n\n  New comments in borch-ts go in English — everything else in this "
          "repository already does.\n  Raising a ceiling is allowed when a Korean string "
          "genuinely has to go in (a case\n  name, a quoted fixture); do it in the same "
          "commit and say why.")


def test_the_ceilings_name_directories_that_exist():
    """A ceiling over a directory that moved is a budget nobody is spending.

    It would sit at zero, pass forever, and read as a directory under control.
    """
    missing = [folder for folder in CEILINGS if not (ROOT / folder).is_dir()]
    assert not missing, (
        f"these ceilings name directories that are not there: {missing}. The code moved — "
        "point the ceiling at where it went, or drop the row if the Korean is gone.")


def test_a_ceiling_that_is_far_too_high_is_tightened():
    """**A ratchet nobody tightens is a ratchet that stopped working.**

    After a translation pass the count drops, the ceiling stays where it was, and the
    headroom left behind quietly permits new Korean back up to the old number — the pass
    is undone over the following weeks and every commit doing it is green.

    So it fails, in the same commit that earned the drop, and prints the line to paste.
    Tightening is copying a number, not taking a measurement.
    """
    slack = {}
    for folder, ceiling in CEILINGS.items():
        total, _ = _count(folder)
        if total and ceiling - total > ceiling * 0.1:
            slack[folder] = (total, ceiling, _provenance(folder))
    assert not slack, (
        "these ceilings are more than 10% above what is actually there — tighten them:\n  "
        + "\n  ".join(f'"{f}": {t},   # was {c}' + (f"\n  {w}" if w else "")
                      for f, (t, c, w) in slack.items())
        + "\n\n  Paste the number only where the drop was earned. This test cannot tell "
          "translation\n  from a file that left the directory, and it asks for the same "
          "line either way.")


def test_the_failure_message_can_say_where_the_number_came_from():
    """**The provenance runs only when something is red, so it is exercised here.**

    A path that runs only on failure rots without anybody noticing, and it is needed on
    exactly the day nobody wants a second problem. This calls it on a green tree.

    It is allowed to say nothing — outside a git checkout there is nothing to ask — but
    where git answers at all, the sentence has to name the commit that wrote the number.
    """
    if _git("rev-parse", "--git-dir") is None:
        return
    folder = next(iter(CEILINGS))
    said = _provenance(folder)
    assert said, (
        f"git is here and the provenance for {folder} came out empty — the blame lookup "
        "is keyed on the line that starts with the folder name, so it breaks when the "
        "table is reformatted.")
    assert "last written in" in said, said


# Measured floors, well under today's counts. They exist to catch a sweep that stopped
# finding files, not to track the directory's size.
FLOORS = {"borch-ts/src": 15, "borch-ts/test": 20}


def test_the_sweep_still_finds_files_to_count():
    """**A ceiling over an empty sweep is green forever, and reads as a clean directory.**

    `_count` walks `rglob("*")` and filters on suffix and on path parts. Narrow that
    filter by accident — a suffix dropped, a `dist` guard that starts matching real
    paths, a directory renamed — and it counts nothing. Nothing is under every ceiling,
    so the ratchet passes, the tightening test skips on `if total`, and both report
    exactly what they report when a translation pass has finished.

    There is no residue to find afterwards. A file that is never visited cannot be
    counted as unvisited, and the summary is small and healthy-looking either way.

    So the floor is asserted separately, on the **file count** rather than the character
    count, because the character count is supposed to fall to zero and the file count is
    not. Another session hit this exact shape today: a filter keyed on a field that does
    not exist reported `agree 0 / differ 0 / unreadable 0` for a namespace of 144 layers,
    and it was found by the row being too clean rather than by anything failing.
    """
    thin = {}
    for folder, floor in FLOORS.items():
        found = len(_countable(folder))
        if found < floor:
            thin[folder] = (found, floor)
    assert not thin, (
        "the sweep found fewer files than it should — it is measuring less than it "
        "claims:\n  "
        + "\n  ".join(f"{f}: {n} files against a floor of {fl}" for f, (n, fl) in thin.items())
        + "\n\n  A ceiling over a sweep that found nothing passes forever. Check SUFFIXES, "
          "the\n  `dist`/`node_modules` guards, and whether the directory moved.")


def test_every_ceiling_has_a_floor():
    """A ceiling added without a floor is the case above, waiting.

    The floor is cheap and its absence is invisible, which is the combination that means
    it will be forgotten unless something asks.
    """
    missing = [folder for folder in CEILINGS if folder not in FLOORS]
    assert not missing, (
        f"these ceilings have no floor beneath them: {missing}. Measure the file count "
        "and\n  write it into FLOORS well under what is there — it catches a sweep that "
        "stops\n  finding files, which no ceiling can.")
