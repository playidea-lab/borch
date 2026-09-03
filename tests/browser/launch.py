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
# They went in as a pair, were separated (`--ignore-gpu-blocklist` alone finds the
# adapter, so `--disable-gpu-driver-bug-workarounds` was not doing the work), and **both
# have now left.** The reason they stayed as long as they did is worth keeping: the
# number the documents quoted came from a run that had them, and **the number the
# documents quote and the flags this file ships have to be the same configuration.**
# Dropping a flag without re-measuring makes those two things different, and then the
# quoted score belongs to a browser nobody launches.
#
# So the list shrank only once every adapter had been asked with **the shipped list
# itself**, and the quoted numbers were taken from those runs. See the table below.
#
# **These four are not the WebGPU working group's four.** Its implementation status
# gives Linux as `--enable-unsafe-webgpu --ozone-platform=x11 --use-angle=vulkan
# --enable-features=Vulkan,VulkanFromANGLE`, which shares one entry with this list.
#
# **That measurement has now been made, on a second card, and this list holds.**
#
#     flags                                    5080/580/151  4090/550/143  M4 Max
#     none                                     none          none          metal-3
#     --enable-unsafe-webgpu                   swiftshader   swiftshader   metal-3
#       + --ignore-gpu-blocklist               swiftshader¹  swiftshader   metal-3
#       + --enable-features=Vulkan             blackwell     lovelace      metal-3
#     --enable-unsafe-webgpu
#       + --enable-features=Vulkan  (= FLAGS)  blackwell     lovelace      metal-3
#
# **The third column is the browser**, and it is there because leaving it out made this
# table lie by omission. The two Linux columns differ in card, driver **and Chrome major
# version at once** (151 against 143), so a difference between them cannot be attributed
# to the hardware — which the sentence that used to stand here did, twice.
#
# ¹ **This row read `blackwell` until Chrome 151 was asked.** Reproduced three times on
#   the newer browser. The old reading came from Playwright's bundled Chromium and the
#   new one from the system Chrome, so what moved is not established and is not claimed —
#   only that the row is no longer what it says it was.
#
# **`--enable-features=Vulkan` is the flag that carries, on both cards.** The blocklist
# override opens no adapter that it does not, on any of the three; macOS is on Metal and
# gives an adapter with no flags at all, so the pair there only has to break nothing, and
# does not. That is why the list is two and not four.
#
# ## This comment said the opposite for one commit
#
# It read *this list does not generalise*, drawn from a ladder whose third rung was
# `--enable-unsafe-webgpu --ignore-gpu-blocklist` — **not `FLAGS`.** `FLAGS` carries
# `--enable-features=Vulkan`, that flag is the carrier on Lovelace, and it was absent
# from the rung the conclusion was drawn about. Re-run with the shipped list verbatim,
# the 4090 reports `nvidia / lovelace`.
#
# **The shipped list was four lines below the table being compared to it**, and the two
# were never put side by side. Fifth time in this repository that two lists nobody
# compared produced a confident wrong sentence, and the first time it happened inside
# the file that carries one of them.
#
# **The lesson is cheaper than the one first written here.** That draft reached for *one
# observation does not establish a mechanism*, which is the right sentence about the
# ANGLE argument below and the wrong one about this: the observation was sound and
# sufficient, and **what was written down was the wrong name for what had been run.** No
# amount of extra measurement fixes a mislabelled one.
#
# So the fix is a label that is checked rather than a habit of measuring more. The rung
# in the README marked `(= FLAGS)` is compared to this list by
# `test_the_ladder_row_that_says_it_is_FLAGS_is_FLAGS`, and it earned its place
# immediately: the label was added as the structural fix for the mistake above and **was
# itself one flag short**, which the check found on its first run. Naming a thing is not
# comparing to it.
#
# ## ANGLE does nothing here, in either direction
#
# The documented incantation reaches the 4090, and so does `--enable-features=Vulkan`
# alone: it worked **because Vulkan is inside it**, not because of ANGLE.
# `--use-angle=vulkan` on its own returns no adapter — ANGLE without Vulkan breaks what
# Vulkan alone fixes. `--ozone-platform=x11` is not needed either.
#
# So dropping the ANGLE flags from this list was right, and both later retractions of
# that were premature — including the one that briefly put a `LOVELACE` constant here
# for a card that never needed it.
#
# Still unmeasured: the 5080 on `--enable-features=Vulkan` alone, which is what would
# let this list shrink.
FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan"]

# Implementations that run on the CPU. Chrome has SwiftShader; Linux Mesa has lavapipe (llvmpipe).
_SOFTWARE = re.compile(r"swiftshader|llvmpipe|lavapipe|software", re.I)


def is_software(adapter):
    """Is this adapter a CPU? Told apart by name — WebGPU does not report it."""
    return bool(adapter and _SOFTWARE.search(str(adapter)))


def screen_is_off():
    """On Linux with an X display: is the monitor blanked (DPMS)? `None` when unknown.

    **A blanked monitor turns every GPU wait into a one-second wait.** Measured on the
    4090 (Xorg + GNOME, monitor connected, DPMS "Monitor is Off" after GNOME's five idle
    minutes): the adapter request took 3.0 s, every readback after a DOM write 1.0 s, a
    training loop that prints its loss seven seconds; with the monitor woken, 52 ms,
    46 ms and 0.56 s. A day of "NVIDIA one-second quanta" was this. The number is not
    the GPU's and is not measured.
    """
    import os
    import shutil
    import subprocess
    if not os.environ.get("DISPLAY") or not shutil.which("xset"):
        return None
    try:
        out = subprocess.run(["xset", "q"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if "Monitor is" not in out:
        return None
    return "Monitor is Off" in out or "Monitor is in Standby" in out or "Monitor is in Suspend" in out


def refuse_if_screen_off(what):
    """**A time taken in front of a blanked monitor is void.** Returns True when void."""
    if screen_is_off():
        print(f"{what}: the monitor is off (DPMS) — every GPU wait on this machine then takes a "
              f"second, and the number would be the monitor's, not the GPU's. Wake it "
              f"(`xset dpms force on`) and measure again.", file=sys.stderr)
        return True
    return False


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
    side.* **The conclusion does not follow from that one observation** — losing an
    adapter under a flag does not establish that Vulkan cannot reach the card — so it
    was removed, and removing it was right.

    **The reason given for removing it was wrong, and that is worth more than the
    sentence was.** It read: *ANGLE translates GL and WebGL, and WebGPU does not go
    through it — Dawn talks to Vulkan directly.* The WebGPU working group's own
    implementation status lists Linux as behind a flag and gives the incantation:

        --enable-unsafe-webgpu --ozone-platform=x11 --use-angle=vulkan
        --enable-features=Vulkan,VulkanFromANGLE

    `--use-angle=vulkan` is in it, and there is a feature named `VulkanFromANGLE`. On
    Linux, ANGLE is **on** WebGPU's documented path rather than beside it. So the flag
    that got waved away was a documented one.

    **And what was actually run was not that path.** The probe passed
    `--use-gl=angle --use-angle=vulkan --enable-features=Vulkan` — ANGLE's Vulkan
    backend without `VulkanFromANGLE` and without the ozone platform. Half an
    incantation is its own configuration, and "no adapter" under it is a fact about that
    half rather than about ANGLE.

    Two people reached the same wrong reason from opposite directions and neither
    checked it: one wrote it into a commit, the other had asserted it first. The page
    that settles it is public and took one fetch. **A correction can be right about what
    to delete and wrong about why**, and only the second half was load-bearing for
    anything else.

    **That has since been run, and it is Vulkan doing the work, not ANGLE.** The
    documented four reach a 4090 — and so does `--enable-unsafe-webgpu
    --enable-features=Vulkan` with no ANGLE in it at all. The incantation works because
    Vulkan is inside it. `--use-angle=vulkan` alone still returns no adapter, which now
    reads as ANGLE without Vulkan breaking what Vulkan alone fixes, rather than as
    anything about half an incantation.

    So the original deletion was right **and so was its stated reason's conclusion**,
    arrived at from a premise that was wrong: ANGLE really is on the documented path and
    really does nothing here. Two retractions of that deletion have now been made and
    both were premature.

    What survives is the narrower sentence this paragraph opened with — **one observation
    does not establish a mechanism** — and it belongs here rather than to the flag-ladder
    mistake above, which was a mislabelled measurement and a cheaper failure entirely.
    Here it has been the correct sentence three times running, and every conclusion drawn
    past it has been wrong.

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
