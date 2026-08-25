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
#
# **The last two are what finally got a second vendor measured**, and they were needed
# because Chrome refuses before the driver is ever asked. On an RTX 5080 (Ubuntu 24.04,
# driver 580.159.04) with a window on Xvfb and every permission in place,
# `requestAdapter()` returned null and the runner said *"No WebGPU adapter could be
# obtained"* — while `vulkaninfo --summary` on that same machine listed the card as GPU0
# with Vulkan 1.4.312.
#
# **Two tools, one machine, two answers**, and the difference is where each one looks:
# `vulkaninfo` asks the driver, Chrome asks its own blocklist first. Linux plus the
# proprietary NVIDIA driver is on that list. So the flag is not a workaround for this
# repository's code — it is how you ask Chrome to go and look.
#
# They went in as a pair and **which of the two is decisive was not separated**. The run
# that produced `passed 2901 / failed 0  [nvidia / blackwell]` had both.
FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan",
         "--ignore-gpu-blocklist", "--disable-gpu-driver-bug-workarounds"]

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
    # The two callers pass a noun phrase — `the values`, `lifetime rules` — so the
    # sentence is built around one. It read *that {what} is right is proved* and came
    # out as "that the values is right", which is the ordinary cost of interpolating a
    # phrase into a slot shaped for a clause.
    if is_software(adapter):
        print(f"  (Software adapter — {what} being right is proved, but this run does\n"
              "   not prove that the GPU path runs.)")


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

    ## Why headless gives a CPU — **two causes, and this paragraph used to say one**

    The file's opening called headless a repeated lesson without saying what makes it
    one. There are two mechanisms, they are independent, and either alone is enough:

    **1. Playwright headless is a different executable.** It downloads
    `chromium-<rev>` and `chromium_headless_shell-<rev>` as separate binaries, and the
    headless one has **no GPU process at all** — so `--enable-unsafe-webgpu
    --enable-features=Vulkan` are not ignored, there is nothing running that could
    receive them. Met from both sides on the same day: the session holding the site
    found the shell binary eating 100% CPU under `ps`, and a run of mine that dropped
    `BORCH_CHROME_CHANNEL` failed with `Executable doesn't exist at
    .../chromium_headless_shell-1234/chrome-headless-shell` — the binary had never
    been fetched.

    **2. A real browser with no window still falls back.** `BORCH_CHROME_CHANNEL=chrome`
    uses the machine's own Chrome, which is **one binary** running `--headless=new`, so
    the GPU process does exist — and it starts in under a second where the shell binary
    is absent entirely. It still reports `google / swiftshader`. Measured on two Linux
    machines with `nvidia_icd.json` installed.

    An earlier draft of this paragraph added *forcing `--use-angle=vulkan` there gives
    no adapter at all rather than a real one, which is the same fact from the other
    side.* **The observation is real; the conclusion drawn from it was not.** ANGLE
    translates GL and WebGL, and WebGPU does not go through it — Dawn talks to Vulkan
    directly. Losing the adapter under that flag therefore says something about GPU
    process startup and nothing about whether Vulkan can reach the card. A peer session
    caught it while the commit was still unpushed. What is removed is the conclusion,
    not the measurement, and re-measuring without that flag is still owed.

    Keeping them apart matters because they have different fixes. The first is a flag
    away. The second is not: it wants a display.

    ## And a display is not the same as a display server that answers

    The 4090 has a logged-in session on `:1`, and Chrome opens no window there —
    measured four ways: `DISPLAY` alone, `DISPLAY` with the session's `XAUTHORITY`,
    and again with five minutes of patience. All three fail identically, so it is
    neither the cookie nor the wait. Chrome prints nothing while failing.

    **A second client settles what one could not.** `Xephyr`, a nested X server, was
    pointed at the same `:1`: it does not exit, does not complain, and never opens its
    socket. Two unrelated programs, the same silence — so the fault is the display
    server, not the browser. A browser that hangs and an X server that hangs look the
    same from the outside, and only asking twice tells them apart.

    That also rules out the nested route on principle rather than by trying harder:
    `Xephyr` draws into its parent, so anything built on it inherits the wall. A
    standalone framebuffer (`Xvfb`) is the one shape that does not, and it is not
    installed there.

    **And then the process table said it outright.** `Xorg` on that machine has been
    burning **81% of a core for twelve days**, while `gnome-shell` is fifty-four seconds
    old — restarted over and over, with a dozen crash-reporter and update-manager
    windows piled up behind it from previous rounds. The session is not idle and
    unresponsive; it is **crash-looping**. That is why every client meets silence rather
    than an error: nothing there fails, and nothing gets serviced either.

    Worth recording because of how long it took to reach. Four launch variations, a
    second X client, a device-node ACL check — all of them pointing at *something about
    the display* — and then one `ps` sorted by CPU named the process and the duration.
    **The cheapest question was asked last.**

    ## And the display was not the wall either

    All of the above shares one assumption: that a working screen is what is missing. It
    is not. Late in the same session, headless Chrome — which needs no display at all —
    **stopped starting on that machine too.** `--headless=new --dump-dom about:blank`, a
    one-second job, produced zero bytes in forty-seven seconds, spawned no child
    processes (no zygote, no GPU process), and ignored `SIGTERM`. No process sat in `D`
    state, so it was not stuck inside the driver; it was waiting on something, very
    early. Two sessions hit it independently, through Playwright and through the binary
    directly.

    **The same invocation returned `google / swiftshader` in under a second that
    morning.** So this is not a property of the machine that was always there and finally
    got noticed — the machine degraded over the twelve-day `Xorg` crash-loop, during the
    hours it was being measured.

    That retires the plan this section was building toward. `Xvfb` would supply a screen,
    and a screen is not what is missing; nothing installable fixes a browser that cannot
    reach its own first child process. What that machine needs is physical: one of its
    two cards reads `rev ff` on the PCI bus — every configuration read returning `0xff`,
    the signature of a device that has fallen off — and that does not come back from a
    reboot.

    **Kept here rather than dropped**, because "we could not measure NVIDIA" and "we
    could not measure NVIDIA *because the only machine available was failing*" are
    different sentences, and the second one is the one that stops somebody spending
    another day on it.

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
