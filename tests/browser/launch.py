"""One place that opens a browser.

Five runners were each calling `p.chromium.launch(...)` with diverging arguments —
`tests/browser/run.py` had no flags at all. On macOS WebGPU came out anyway so it went
unseen, but the moment two numbers measured under diverged conditions are put side by side
and called "the same yardstick", that is a lie. So they are gathered here.

**Give `BORCH_CHROME_CHANNEL` to use the system browser.** Playwright uses the Chromium it
downloaded itself by default, and a GPU server may not have that while having a distribution
Chrome installed. Using what is there beats downloading one more browser, and above all
**the browser the user actually uses** is that one.

    BORCH_CHROME_CHANNEL=chrome DISPLAY=:1 uv run ... --headed

`DISPLAY` is not handled here — Playwright passes the process environment through, so putting
it in front of the command is enough. That headless does not work is this repository's
repeated lesson (a software rasteriser comes out quietly, without exception), and it is why
the runners print the adapter first.

**And that lesson was written here while the default did the opposite.** Headless was the
default for every runner, so the paragraph above described the trap and the code set it.
Measured on one machine, the same 3433 binding cases:

    headless    [borch.ts — google / swiftshader]    3433/3433
    --headed    [borch.ts — apple / metal-3]         3433/3433

Nothing was missing from the machine. The window is the default now — see `_headed` — and
`--headless` is how a CPU run is asked for on purpose.
"""

import os
import re
import sys

# Turns WebGPU on and makes Linux use Vulkan. macOS is on Metal, so the second is ignored.
FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan"]

# Implementations that run on the CPU. Chrome has SwiftShader; Linux Mesa has lavapipe (llvmpipe).
_SOFTWARE = re.compile(r"swiftshader|llvmpipe|lavapipe|software", re.I)


def is_software(adapter):
    """Is this adapter a CPU? Told apart by name — WebGPU does not report it."""
    return bool(adapter and _SOFTWARE.search(str(adapter)))


def refuse_if_software(adapter, what):
    """**Measuring time and resources is void on a CPU.** Returns True when void.

    A device does not change values, so the golden answers are valid evidence on a software
    adapter too — evidence that the logic is right, just not that the GPU path runs. So the
    golden run is not blocked and only the benchmarks and accuracy are.

    That distinction is why this function exists. Running the golden cases headless on a Linux
    GPU server gave 845/845 while the adapter was `google / swiftshader` — the pass was real
    and the claim "confirmed on another vendor" was false. Unless a person reads the first line
    of the log, it goes straight through.
    """
    if not is_software(adapter):
        return False
    print(f"**Software adapter ({adapter}) — {what} is void on this device.**\n"
          "  It ran on the CPU, so this number is not a GPU's number. Measure again with\n"
          "  `--headed`, on a screen with a real GPU attached.", file=sys.stderr)
    return True


def warn_if_software(adapter, what):
    """For things that **ask about values only**, as the golden run does. Does not block; only
    narrows what was proved.

    **It went unread again, a whole session of it.** Every browser golden run in one day —
    twenty and more, across three implementations, including four freshly written WGSL kernels
    — came out of `google / swiftshader`, and the warning was on screen every time. It was
    read past because the reader was tailing the last three lines for the pass count, and this
    line is at the top.

    The docstring above already predicted exactly that: *unless a person reads the first line
    of the log, it goes straight through.* Predicting it did not stop it. What would is putting
    the sentence where the number is rather than where the run starts — that is a change to
    every runner's output and is written down here rather than made, because the split between
    this and `refuse_if_software` is deliberate and a person should choose whether to widen it.
    """
    if is_software(adapter):
        print(f"  (Software adapter — that {what} is right is proved, but this run does not\n"
              "   prove that the GPU path runs.)")


import contextlib


@contextlib.contextmanager
def browser(playwright, headed=False, flags=FLAGS):
    """**The only door that opens a browser.** Closing happens here too.

    All twelve runners had `browser.close()` as the last line of their `with`, and an exception
    before it means **the browser does not close.**

    It does not leak quietly — the leftover Chromium keeps using CPU. One runner really was
    still alive at two minutes forty-two, and in that time another session was measuring
    benchmarks. Their number came out at sixteen times the documented one, and **both sides
    spent time** chasing the cause (it turned out that sixteen times had a different cause, but
    ruling this one out cost time again).

    A value check will never see it. This is the kind where a test apparatus does not clean up
    after itself and **damages another measurement.**

    **So there is no non-closing edition outside.** The first fix moved only six and left
    `launch` public, and the other six went on using it — three of them being `bench`, `cost`
    and `accuracy`, exactly where a leak does the most harm. Two lists diverge, and with two
    doors only one gets fixed.
    """
    got = _open(playwright, headed=headed, flags=flags)
    try:
        yield got
    finally:
        got.close()


def _headed(asked):
    """**A real adapter is the default now, and the caller cannot fall back by accident.**

    This used to be `headless=not headed` with `headed=False` as the default, and the
    docstring said what that cost in one line: *without `headed` a software adapter
    comes out.* True the whole time, and it was written where the browser is opened
    rather than where the result is read.

    What it cost: **every browser golden run in two days, across three
    implementations and two sessions, came off SwiftShader.** Twenty-odd runs,
    including newly written WGSL kernels whose whole risk is integer division and
    boundary handling — the two things vendor compilers part on. The warning printed
    every time, at the top, and the number is at the bottom.

    Flipping the default alone would not have done it. All fifteen runners spell it
    `headed="--headed" in argv`, which passes **False explicitly** when the flag is
    absent, so a new default would have been overridden fifteen times. The decision
    has to be made here, at the one door — which is what this module is for; it exists
    because five runners were each launching with different arguments.

    So `headed=True` still means yes, and `False` now means *no preference*. Saying no
    takes `--headless` or `BORCH_HEADLESS=1`, and it is worth having: a machine with no
    display cannot open a window, and there the choice is a real one rather than a
    silent downgrade.
    """
    if asked:
        return True
    if os.environ.get("BORCH_HEADLESS"):
        return False
    return "--headless" not in sys.argv


def _open(playwright, headed=False, flags=FLAGS):
    """Opens headed unless told otherwise — see `_headed`."""
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    want = _headed(headed)
    try:
        return playwright.chromium.launch(
            headless=not want,
            channel=channel,
            args=list(flags),
        )
    except Exception as exc:                                    # noqa: BLE001
        if not want:
            raise
        # **Failing loudly here is the point.** A machine with no display cannot open
        # a window, and the old behaviour was to quietly hand back a CPU rasteriser
        # and let the run report a pass. Saying which flag turns that back on is not
        # the same as turning it on.
        raise RuntimeError(
            f"could not open a browser window ({type(exc).__name__}: {exc}).\n"
            "  A window is now the default, because headless quietly gives a software\n"
            "  adapter and the golden then proves the values and not the GPU path.\n"
            "  On a machine with a display, give `DISPLAY=:1` (Linux) or run locally.\n"
            "  To measure on the CPU on purpose, say so: `--headless`, or "
            "`BORCH_HEADLESS=1`.") from exc
