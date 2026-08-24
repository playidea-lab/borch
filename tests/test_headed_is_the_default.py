"""**A window is the default, and going without one has to be asked for.**

`tests/browser/launch.py` opens every browser this repository uses. Its own opening
paragraph says headless quietly hands back a software rasteriser — and for as long as
that paragraph existed, `headless=not headed` with `headed=False` made headless the
default. The trap was described where the browser opens and set two lines below.

What it cost, measured on one machine over the same 3433 binding cases:

    headless    [borch.ts — google / swiftshader]    3433/3433
    --headed    [borch.ts — apple / metal-3]         3433/3433

Nothing was missing from the machine. Two days of browser goldens across three
implementations and two sessions — including newly written WGSL whose whole risk is
integer division and boundary handling, the two things vendor compilers part on — were
run on a CPU, with the warning on screen every time and the number at the bottom.

## Why the flip could not be a default

All fifteen runners spell it `headed="--headed" in argv`, which passes **False
explicitly** when the flag is absent. A new default would have been overridden fifteen
times over. So the decision is made inside the door, and `False` now means *no
preference* rather than *no*.

## What this file holds

- **A window unless refused**, and refusing takes a word.
- **The refusal actually reachable.** The first version of the flip left no
  `--headless` on the two runners that use `argparse`, so the failure message told the
  reader to pass a flag argparse then rejected. A door with no handle on the inside.
- **Not that the adapter is real.** A machine may have no GPU at all; that is what the
  adapter line on the score row is for.
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAUNCH = ROOT / "tests" / "browser" / "launch.py"


def _launch():
    if str(LAUNCH.parent) not in sys.path:
        sys.path.insert(0, str(LAUNCH.parent))
    spec = importlib.util.spec_from_file_location("_borch_launch", LAUNCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_borch_launch", mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not LAUNCH.exists(), reason="no tests/browser/launch.py")
def test_a_window_is_what_you_get_without_saying_anything(monkeypatch):
    mod = _launch()
    monkeypatch.setattr(sys, "argv", ["run.py", "--lib", "borch_webgpu"])
    monkeypatch.delenv("BORCH_HEADLESS", raising=False)
    assert mod._headed(False) is True, (
        "a bare run went headless again — that is a software adapter, and the run\n"
        "  still prints `agreeing` for it.")
    assert mod._headed(True) is True


@pytest.mark.skipif(not LAUNCH.exists(), reason="no tests/browser/launch.py")
def test_the_cpu_can_still_be_asked_for_on_purpose(monkeypatch):
    """**Both spellings, because a machine with no display needs one that is not a flag.**

    `--headless` is for a person at a terminal; `BORCH_HEADLESS` is for a job that
    cannot edit the command line it was handed.
    """
    mod = _launch()
    monkeypatch.delenv("BORCH_HEADLESS", raising=False)
    monkeypatch.setattr(sys, "argv", ["run.py", "--lib", "borch", "--headless"])
    assert mod._headed(False) is False

    monkeypatch.setattr(sys, "argv", ["run.py", "--lib", "borch"])
    monkeypatch.setenv("BORCH_HEADLESS", "1")
    assert mod._headed(False) is False


@pytest.mark.skipif(not LAUNCH.exists(), reason="no tests/browser/launch.py")
def test_an_explicit_headed_beats_the_refusal(monkeypatch):
    """`--headed --headless` is a contradiction and the window wins.

    Asking for the thing is louder than not asking against it, and the direction
    matters: the wrong resolution here downgrades silently, which is the whole defect.
    """
    mod = _launch()
    monkeypatch.delenv("BORCH_HEADLESS", raising=False)
    monkeypatch.setattr(sys, "argv", ["run.py", "--headed", "--headless"])
    assert mod._headed(True) is True


def _parsers():
    """The runners that use `argparse` — the ones that can reject a flag before
    `launch` ever reads `sys.argv`."""
    return [ROOT / "tests" / "browser" / "run.py",
            ROOT / "tests" / "browser" / "why_failing.py"]


@pytest.mark.parametrize("path", _parsers(), ids=lambda p: p.name)
def test_every_argparse_runner_accepts_the_word_the_error_tells_you_to_use(path):
    """**The escape has to be reachable from the command line that failed.**

    `launch._headed` reads `--headless` out of `sys.argv`, which works for the twelve
    runners that read `sys.argv` themselves. The two that build an `ArgumentParser`
    stop first with `unrecognized arguments: --headless` — and that is what happened on
    the first run after the flip: the new failure message named a flag the program
    refused. Measured by parsing, not by grepping for the string.
    """
    if not path.exists():
        pytest.skip(f"no {path.name}")
    # **Asked of the runner's own parser, by running it.** The first version of this
    # searched the source for the string and then exercised a *fresh* parser it built
    # itself — which proves that `argparse` works, not that this file uses it. That is
    # the shape named twice already this week: a check asserting less than its name.
    #
    # **`--headless` on its own does not ask the question**, and the second version of
    # this check thought it did. `run.py` requires `--lib`, and argparse reports the
    # missing requirement *before* it looks for unknown flags — so the same sentence
    # comes back whether the flag is declared or not, and deleting it from `run.py`
    # left this passing. Measured both ways, which is what showed it.
    #
    # A junk flag alongside is what separates them: argparse lists **every** name it
    # did not recognise, so the marker is there either way and `--headless` joins it
    # only when it is missing.
    #
    #     declared    unrecognized arguments: --zzz-marker
    #     missing     unrecognized arguments: --headless --zzz-marker
    #
    # `--lib borch` gets past the required check; the junk flag stops the run before
    # a browser opens.
    got = subprocess.run(
        [sys.executable, str(path), "--lib", "borch", "--headless", "--zzz-marker"],
        capture_output=True, text=True, cwd=ROOT, timeout=60)
    # **The error sentence, not the whole of stderr.** argparse prints the `usage:`
    # line first and that line lists every declared flag — including `--headless` when
    # it is declared. Searching all of stderr therefore finds the word in exactly the
    # case that should pass, and the check failed while the code was right.
    complaint = [ln for ln in got.stderr.splitlines()
                 if "unrecognized arguments" in ln]
    assert complaint, (
        f"{path.name} did not stop on a junk flag, so this check never reached the\n"
        f"  question it asks:\n\n  {got.stderr.strip()[-300:]}")
    assert "--headless" not in complaint[0], (
        f"{path.name} builds an ArgumentParser and does not declare `--headless`.\n"
        "  `launch._headed` never sees it — argparse rejects the run first, and the\n"
        "  failure message this repository prints tells the reader to pass exactly\n"
        f"  that flag.\n\n  {got.stderr.strip()[-300:]}")
