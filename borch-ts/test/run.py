"""Run the borch.ts golden runner in a browser.

    npm run build:ts
    uv run --with playwright python borch-ts/test/run.py

The same shape as `tests/browser/run.py` — put the repository root on a temporary port,
open the page with Playwright, read the result off it. It is written separately because
that one is a dedicated runner that loads the Python library into Pyodide through
`runner.html?lib=`, and this one needs no Pyodide at all. borch.ts is JS the browser
simply reads.

**What did not run is not mixed in with what passed.** If the page throws, the exit code
is not 0, and a single "registered but not in the golden" name is a failure too — running
0 cases through a typo and seeing green is the worst outcome available in this project.
"""

import functools
import http.server
import pathlib
import socketserver
import sys
import threading

from launch import browser as browser_of, warn_if_software

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGE = "/borch-ts/test/index.html"
# **Give it room.** This was 120 seconds, and with the table grown and headless dropping
# to a software adapter it cannot finish inside that. The screen it produces then cannot be
# told apart from "it hung", and a defect that did not exist really was chased once — even
# with the last case name printed. Running out of time and never finishing are different
# events, so the budget is set past any doubt.
TIMEOUT_MS = 600_000


def require_fresh_dist(root=ROOT):
    """**Stop here when the source is newer than `dist`.**

    What the runner loads is `borch-ts/dist`, and that is in `.gitignore`, so it is in no
    commit. Rebase or move between branches and the source changes while the emit stays
    old.

    **The runner cannot report that state.** A stale `dist` and a genuine hole come out
    with **the same wording**: `borch.ts does not have X`. Two people walked into it
    separately — one added 119 new cases, saw the runner's count not move by a single
    case, and went looking for a typo in a name; the other reported 31 red cases in the
    binding as a regression. Those 31 names were **there** in `tensor.ts`. They were
    reading the source while the runner read `dist`.

    `check:ts` is `--noEmit`, so it does not mend this place. Forgetting the build is what
    the structure invites, so it is better to stop when it is forgotten — **before
    anything else is suspected.**

    ## An emit whose source is gone reads as staleness the build cannot fix

    The age test takes the *oldest* `.js`, so one leftover drags the whole verdict down.
    Delete a `.ts` — a scratch probe, a renamed module — and `tsc` leaves its `.js`
    behind: it emits, it does not sweep. From then on this function fires on every run
    and prescribes `npm run build:ts`, which rebuilds everything and **changes nothing
    about the file that is failing.** A check that is always red is ignored exactly as
    fast as one that is always green, and this one arrives with a remedy that cannot
    work, so the reader concludes the check is broken rather than the tree.

    Found by walking into it: a probe under `test/` was written, compiled, and removed,
    and the next four runs all stopped here with a correct verdict and the wrong cause.
    So the orphans are named separately — the sweep is one line, but only for someone
    who knows that is what they are looking at.
    """
    dist = root / "borch-ts" / "dist"
    if not dist.exists():
        raise SystemExit(f"no emit: {dist}\n  first: npm run build:ts")
    src_root = root / "borch-ts"
    newest_src = max(
        (p.stat().st_mtime for p in src_root.rglob("*.ts")
         if "dist" not in p.parts and "node_modules" not in p.parts),
        default=0)

    outs = list(dist.rglob("*.js"))
    orphans = [p for p in outs
               if not (src_root / p.relative_to(dist).with_suffix(".ts")).exists()]
    if orphans:
        listed = "\n".join(f"    {p.relative_to(root)}" for p in sorted(orphans)[:10])
        raise SystemExit(
            "emitted files whose source is gone — `tsc` emits but does not sweep.\n"
            f"{listed}\n"
            "  delete them; rebuilding will not, and they hold the age test down\n"
            "  because it reads the *oldest* emit.")

    oldest_out = min((p.stat().st_mtime for p in outs), default=0)
    if newest_src > oldest_out:
        raise SystemExit(
            "the emit is older than the source — the runner loads `borch-ts/dist`.\n"
            "  first: npm run build:ts\n"
            "  (run it as it is and a new name comes out as `not in borch.ts`, which is\n"
            "   **the same wording** as a genuine hole, so the cause is invisible.)")


def require_fresh_golden(root=ROOT):
    """**Stop here when `cases.py` is newer than `golden.json`.**

    The golden has the same trap as `dist`. What the runner reads is `tests/golden.json`,
    and that comes out of `tests/cases.py` → `golden.npz` → `golden.json`, two steps, and
    the `npz` in the middle is in `.gitignore` and so in no commit.

    So write a new case, forget to dump, and **that case comes out as "the name is not in
    the golden"** — **the same wording** as a typo in the name. Nine cases really were
    added, that screen came back, and a typo that did not exist was searched for first.

    Two steps are easy to forget, and the first of them wants real torch, so it does not
    run everywhere. Better to stop when it is forgotten, **before anything else is
    suspected.**

    ## mtime is a proxy, not the fact

    At first only the time was looked at. But changing **a comment alone** in `cases.py`
    moves the time — another session translated that file into English and the runner
    stopped, with the golden perfectly fine. The case names and the values were untouched
    and one timestamp produced "dump it again".

    Leave a false alarm standing and people learn to walk past that warning, and then they
    **walk past the real one too.** So when the times disagree it does not stop there: it
    **compares the name table itself** (`manifest`). Equal means only the time moved, and
    it goes through.

    **It does not go as far as the values.** That is `tests/test_committed_golden.py`'s
    job, which runs without real torch, and a second copy of it here would part from the
    first eventually.
    """
    cases = root / "tests" / "cases.py"
    exported = root / "tests" / "golden.json"
    if not exported.exists():
        raise SystemExit(f"no golden: {exported}")
    if cases.stat().st_mtime <= exported.stat().st_mtime:
        return
    if _names_still_match(root, exported):
        return
    raise SystemExit(
        "the golden is older than the case table — the runner reads "
        "`tests/golden.json`.\n"
        "  first: uv run --with numpy --with torch --with torchvision "
        "python tests/golden.py dump\n"
        "  then: uv run --with numpy python tests/export_json.py\n"
        "  (run it as it is and a new case comes out as `the name is not in the\n"
        "   golden`, which is **the same wording** as getting the name wrong, so the\n"
        "   cause is invisible.)")


def _names_still_match(root, exported):
    """Is the frozen name table the one `cases.py` produces today?

    **Where it cannot measure, it answers that it cannot.** Importing `cases.py` wants
    numpy, and there are places this runner runs without it. Answering "equal" there would
    turn this branch into a switch that turns the check off — better noisy than
    manufacturing a confidence nobody has.
    """
    import json
    import sys

    try:
        doc = json.loads(exported.read_text(encoding="utf-8"))
        stamped = doc.get("manifest")
        if not stamped:
            return False
        sys.path.insert(0, str(root))
        try:
            from tests import cases as cases_mod
        except ImportError:
            import cases as cases_mod
        return stamped == cases_mod.manifest_hash(cases_mod.golden_cases())
    except Exception:
        return False


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(root):
    """Put the repository root on a temporary port; return (port, shutdown)."""
    handler = functools.partial(_ReportMissing, directory=str(root))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1], httpd.shutdown


class _ReportMissing(_Quiet):
    """**Print what was not found.** An unexplained 404 must not be papered over — a
    runner in this repository once took 404 HTML as a Python file and blew up somewhere
    else entirely. What the browser fetches on its own (favicon) lands here too, so it
    shows itself for what it is."""

    def send_error(self, code, message=None, explain=None):
        if code == 404:
            print(f"  [404] {self.path}")
        super().send_error(code, message, explain)


_STARTED = "[golden] "


def run(headed=False, verbose=False):
    """With `verbose`, print the console in full.

    **It catches a silent hang.** The runner try/catches every case, so an exception rides
    out in the report. What does not ride out is a case that **never finishes**, and then
    no report is built at all and the time simply passes with nothing on screen.

    So the line the runner prints as it starts each case is caught here, and when the time
    runs out it says **the last name started** — that is the culprit. Leaning on terminal
    scrollback buries it under 1,199 lines, and it really was buried once.

    The prefix it watches for is written on the other side, in `golden.ts`. Translate one
    of the two alone and this quietly catches nothing: the trace goes empty, the timeout
    screen loses its only useful line, and nothing fails.
    """
    from playwright.sync_api import sync_playwright

    require_fresh_dist()
    require_fresh_golden()
    port, stop = serve(ROOT)
    url = f"http://127.0.0.1:{port}{PAGE}"
    last = []

    def on_console(m):
        if m.text.startswith(_STARTED):
            last.append(m.text[len(_STARTED):])
        if verbose or m.type == "error":
            print(f"  [browser] {m.text}")

    try:
        # **`with` closes the browser too** — put on the last line instead, an exception
        # before it leaves it open, and the leftover Chromium ruins another measurement.
        with sync_playwright() as p, browser_of(p, headed=headed) as browser:
            # Headless Chromium gives no WebGPU adapter by default — ask and null comes
            # back rather than an exception. The old TF.js build never showed this
            # problem, because failing to get one it dropped quietly to WebGL — which is
            # not better for being invisible: the numbers measured then were not the
            # GPU's.
            page = browser.new_page()
            page.set_default_timeout(0)
            # Shader compile errors come out on the console alone. Swallow them and the
            # cause cannot be found.
            page.on("console", on_console)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}"))
            # An unexplained 404 must not be papered over — a runner in this repository
            # once took 404 HTML as a Python file and blew up somewhere else entirely.
            page.on("response", lambda r: print(f"  [404] {r.url}")
                    if r.status == 404 else None)
            page.goto(url)
            try:
                page.wait_for_function("window.__borchReport !== undefined",
                                       timeout=TIMEOUT_MS)
            except Exception:
                where = last[-1] if last else "(not one of them started)"
                print(f"no report came out. last case started: {where}\n"
                      f"  {len(last)} started — that case is the one that never "
                      "finished.", file=sys.stderr)
                raise
            report = page.evaluate("window.__borchReport")
    finally:
        stop()
    return report


# Why they were not ported. **Every prefix owes a reason and a count.**
#
# For a long time this was one line: "N cases were deliberately not ported". Three
# different things were mixed inside that sentence — what has no value to port, what has
# not been ported yet, and **what is not in borch.ts at all.** Printing the count alone
# made the three look identical on screen, and `rnntop::`, 35 cases, really was sitting
# in there as something forgotten. Only once they were written out here did it come out
# that `opt::LBFGS` and `index::searchsorted` were **absent names**.
#
# The counts have to be **exact**. Leave slack and a newly unported case hides in the gap
# — which is the very thing this check exists to stop. Grow the cases and either port
# them or raise the number with the reason it cannot be ported.
#
# Markers:  별칭   = porting it asks the same question twice (alias)
#           파이썬 = it belongs to the Python surface, with no TS counterpart (python)
#           아직   = there is value in porting it and it has not been done (**a backlog**)
#           없음   = the name is not in borch.ts (**a hole**)
#
# The four markers are keys rather than prose: `test_alias_rows.py` reads `별칭` off the
# head of a reason string and `test_site.py` reads `아직` to split the remainder into
# backlog and never. They move when those checks move, and not before.
NOT_PORTED = {
    # **v2's type system.** `tv_tensors` are `torch.Tensor` subclasses that carry a
    # label — a box's format and canvas, a mask's being a mask — and the thirteen names
    # here are the dispatch that reads those labels off a flattened sample.
    #
    # borch.ts has no subclassable tensor to hang a label on: `Tensor` there is a handle
    # to a GPU buffer, and the thing that makes a tv_tensor work on the Python side is
    # that `isinstance` answers. Carrying this over is designing a second mechanism, not
    # writing a second body — a tagged wrapper, or a parallel dict of metadata, and
    # which one is a decision nobody has taken.
    "v2::": (13, "없음 — the tv_tensor dispatch. borch.ts's `Tensor` is a handle "
                        "to a buffer and cannot be subclassed to carry a label, so this "
                        "needs a mechanism that side does not have rather than a body "
                        "it has not been given"),
    # **The seven losses torch keeps at top level as well as under `F`.** They were
    # declined as *the raw ATen op — its signature differs from F's*, which was about
    # why the name looks redundant rather than about what was missing; the arithmetic
    # was `F`'s and already there.
    #
    # borch.ts has the `F` side of all seven. What it does not have is a **top level**
    # to put a second name on: over there `F` is the namespace and there is no `torch.`
    # beside it, so `borch.klDiv` would be a name this side invented rather than one
    # torch has. The integer reduction is the same — `F`'s takes the word, and a second
    # spelling that takes a number is a Python-side fact about two bindings of one
    # library.
    "loss::": (18, "파이썬 — torch keeps these seven at top level as well as "
                         "under `F`, with an integer reduction and a different default. "
                         "borch.ts has one namespace, so there is no second place to "
                         "put them"),
    # **`unpool::` came back at 46 and is down to 6.** The Python side printed
    # twenty-nine of sixty-four layers differently from torch and nothing was watching;
    # this side had the same hole for the same reason, and forty of the strings now
    # cross. The row was owed for one run, which was true: it was a hole nobody had
    # looked in rather than one looked in and turned down.
    #
    # What is left is a missing name and not the writing: four of the six ask about an
    # argument borch.ts does not take and two about a name it does not have.
    # `AdaptiveAvgPool2d` is absent there; `MaxPool2d`'s `padding`, `dilation` and
    # `ceilMode` are held and refused until the pooling kernel implements them;
    # `Flatten` takes no arguments at all. Each is a question this side cannot be asked
    # yet, which the name and signature axes already count — writing the case would
    # only move the refusal.
    #
    # **6 → 5: `AvgPool2d`'s padding case was in this list and is now written.** The
    # row said "AvgPool2d takes no padding", which was true while the layer called the
    # two-dimensional kernel and stopped being true the moment it moved to `poolND`,
    # the path its 1-D and 3-D siblings already used. A reason recorded here goes on
    # being read after its cause is gone; this one lasted about an hour, and only
    # because the same change that ended it was made by someone who then read this
    # line.
    "unpool::": (4, "없음 — two ask about arguments borch.ts does not take "
                        "(MaxPool2d's `dilation`, which no pooling kernel here "
                        "dilates, and Flatten's two) and two about "
                        "`AdaptiveAvgPool2d`, which is not a name over there. "
                        "**The maximum's padding and ceilMode left this row** with "
                        "AvgPool2d's: they were refused on the ground that the "
                        "maximum's backward reads the input at each window position "
                        "and a padded one has none, and the average had answered that "
                        "one function away — take the padding off the coordinate and "
                        "skip what falls outside"),
    # **`ops::` was a row here once, at 11, and was emptied by porting all of them.**
    # It is back at 8 for a different reason: the eleven were box geometry on plain
    # arrays and these eight are `nn.Module` layers whose weights have to be written
    # from the same numbers on both sides. borch.ts has the layers; what is missing is
    # the harness that fills a module's parameters and buffers by name, which the
    # Python cases grew for this and the TS runner has never needed.
    "ops::": (14, "아직 — eight need a way to **write parameters and buffers by "
                        "name**, which is what the Python side's `_fill` is and this "
                        "side has no counterpart to: the two blocks, the two "
                        "multi-layer perceptrons, the excitation, the pyramid, the "
                        "deformable module and the frozen norm. The other six are the "
                        "level-picker and three reprs, and they want only the class. "
                        "**The eight RoI names crossed**, then the deformable "
                        "convolution and the dropouts' deterministic halves, then the "
                        "eleven layers that hold settings and no weights. What is left "
                        "of the dropouts is the coin itself, which has no shared answer "
                        "rather than no implementation"),
    # 104 → 103. What `dtype::없는이름::` was asking about moved to `narrow_copy` and
    # `unsafe_chunk` — torch actually answers the earlier one, so it was no absent name.
    # 103 → 111 → 147 → 156 → 157. Whether the dtype aliases and the factories really
    # listen to `dtype=` and `requires_grad=`. **A Python-side matter** — borch.ts takes a
    # dtype as a string, so the name `torch.float` does not exist at all, and its factories
    # are `Tensor.zeros(shape)` and take no dtype argument. Gradients too: over there
    # `requiresGrad()` is called separately.
    #
    # 147 → 156 is **the nine that were left outside the fourteen**. Five window functions
    # were swallowing `requires_grad=` (quietly handing back a leaf with no gradient) and
    # four of the numbering factories did not take `dtype=` at all. The earlier batch had
    # been mended as a list, so what lay outside that list stayed as it was.
    # 156 → 157 is `norm(dtype=)` — alone among the reductions in not listening.
    # 157 → 160 is the last two of `normal` and `frombuffer`, and the place `frombuffer`
    # was **quietly dropping an unknown dtype to float32**. Four candidates came out of a
    # `**kw` grep and reading torch's signatures one by one left two — build the list
    # mechanically and things that do not belong come with it.
    # 160 → 164 is the four asking whether `out=` is refused rather than swallowed.
    # borch.ts has no such argument at all.
    # 164 → 172. `out=` was **actually implemented** (refusal used to be the answer):
    # writing into a tensor made in advance and returning that same object, re-taking it
    # when the shape differs, refusing on dtype and on gradient. **A Python-side matter** —
    # there is no `out=` argument in borch.ts.
    # 172 → 158. **Fourteen named dtype conversions were ported.** Refusal is the answer
    # for eight of them, and what is pinned there is not a value but the **wording** — a
    # missing name produces `'half' is not there`, which cannot be told from a typo. The
    # remaining 158 cannot be ported: `자리만::`, `공장::`, `별칭::` and `out::` are all
    # about Python signatures, and borch.ts takes a dtype as a string, so the name
    # `torch.float` does not exist.
    # Four gradient cases are left. **A Python-side helper is what makes the leaf** —
    # `_grad_of` is where a leaf is checked for an arrived gradient and it is taken out, so
    # carrying that shape into TS would make a second copy of it. The seventeen value cases
    # were ported, and it was those that caught a defect in the kernel.
    # **`inplace` on the thirteen activations is a kernel question, not an argument
    # one.** Over there an activation is a WGSL shader that writes to a fresh output
    # buffer; writing back into the input means a second shader per activation, not a
    # flag threaded through. Twelve rows — six values and six identities.
    #
    # **That reason was wrong within the hour and the row was kept to say so.** No
    # second shader is needed: `Tensor.copyFrom` already writes one buffer into
    # another, and the binding honours the flag with exactly that — run the ordinary
    # shader, copy the answer back over the input, hand the input back. It is the same
    # call `Optimizer.update` has been making all along. So the row stayed `아직`, which
    # was the right marker, on a cause that turned out to be "the TS cases were not
    # written" rather than "the mechanism is missing". A reason that names a difficulty
    # the library does not actually have is the shape this repository keeps finding —
    # it reads like a decision and outlives the fact it was built on.
    #
    # **The row is gone and the corrected reason held exactly.** Six classes gained an
    # `inplace` seat and one line each: run the ordinary path, then `copyFrom`. No new
    # shader, no new kernel, and the leaf refusal came free because `copyFrom` goes
    # through `mutate`, which already stops on a leaf that requires grad.
    #
    # Six and not thirteen. torch gives `inplace` to more of them (`ReLU6`,
    # `Hardsigmoid`, `Hardswish`, `Mish`), and those seats are **absent rather than
    # declared-and-unmeasured** — the `pool::` row three screens down is what that
    # distinction costs when it goes the other way.
    "fft::": (4, "파이썬 — the gradient helper is what makes the leaf"),
    # **`opt::` has a row again, and its own comment above says the last three reasons
    # written here were each wrong in turn.** So this one names exactly what cannot
    # cross and nothing more.
    #
    # `maximize` went onto ten core optimizers and borch.ts has it on `SGD` alone —
    # carrying it means a WGSL kernel per optimizer, which is work rather than a
    # decision, hence `아직`. The third is `SGD(the default rate)`, which cannot exist
    # over there at all: borch.ts's `SGD` requires `lr`, so there is no default to
    # disagree about. That one is `없음` in a row of `아직`, and it is counted here
    # rather than split into a row of its own because a one-case prefix would read as
    # a category.
    # 6 → 14. The core took torch's whole optimizer surface — `amsgrad`, `centered`,
    # `momentum`, `decoupled_weight_decay` and the execution switches — and borch.ts
    # has none of the four. **They are algorithm, not performance**: each changes the
    # values, so each is a kernel over there rather than an argument threaded through.
    #
    # **14 → 2. The work was done, so the reason expired.** borch.ts has all five now:
    # `maximize` is one negation on the `Optimizer` base (torch applies it before
    # weight decay in every one of its optimizers, so it is not eleven decisions), and
    # `amsgrad`, `centered` and `momentum` are one extra buffer each in the Adam and
    # RMSprop shaders, with `decoupled_weight_decay` moving NAdam's decay onto the
    # weight.
    #
    # **2 → 0, and the last row marked `없음` is gone.** It said borch.ts's `SGD`
    # requires `lr` and so has no default to disagree about — true, and the wrong way
    # round: torch's `SGD` has a default, so `new SGD(params)` is a line torch takes
    # and this stopped on an argument count. The one optimizer a first tutorial builds
    # with nothing but its parameters.
    #
    # The number is torch's `1e-3` and not a rounder one. The Python side carried
    # `0.01` for a while — ten times too large — and **a default is the one value a
    # case cannot check by using it**, because every other row names its own rate.
    # That is why this case names none.
    # 158 → 86. **One reason was covering eight groups.** `아직` means a backlog, and
    # what was actually backed up was `자리만::` 63 and `묻는것::` 9 alone — both pure
    # properties of the other side, asking `t.dtype` and nothing else, and both should have
    # been asked long before. Ported.
    #
    # The rest is not backlog but **nowhere to ask it**:
    #
    #   `공장::` 40   — the factories over there take no `dtype=` or `requires_grad=`.
    #                   torch writes `zeros(3, dtype=int64)`; here it is two steps, make it
    #                   and then convert. An API difference written down in no table, and
    #                   this is the record of it.
    #   `별칭::` 16   — the name `torch.float` does not exist (a dtype is a string).
    #   `out::` 12    — there is no `out=` argument. Imitated, it saves nothing.
    #   `없는이름::` 11 · `조밀에도답::` 6 · `없는형::` 1 — the Python surface.
    # 86 → 88. Two spellings of one refusal — `.to(float64)` and `.type(float64)` —
    # went in beside `.double()`, which the core had been granting in name and
    # answering in `float32`. All three arrive at one gate now, and the rows are
    # here rather than portable for the reason the rest of this group is: they ask
    # about a Python signature, and borch.ts takes its dtypes as strings.
    "dtype::": (88, "파이썬 — the factories' `dtype=`, the dtype aliases, `out=`"),
    # This number jumping from 82 to 88 was caught **the day the check went in** — the
    # batch that turned `x.real` and `x.device` into properties grew the cases by six.
    # 88 → 116. Twenty predicates and eight unpaired in-place variants went in. Both are
    # **matters of the Python surface** — `is_cuda` has nowhere to be asked in borch.ts,
    # and `apply_` and `map_` hang a Python function on every slot, which a GPU cannot do.
    # 116 → 144. Looking into the storage (`stride`, `nbytes`), the three names for
    # transpose, `new_*` and `retain_grad` went in. **They are here for different reasons
    # by group** — `stride` parts because the other side makes no views (that parting is
    # itself a case), `new_*` is Python's dtype inheritance, and `H` and `mT` are names
    # that do not exist over there.
    # 144 → 158. Seven sparse accessors (`values`, `indices`, `crow_indices` …), the
    # missing storage and quantisation features, and `is_set_to`. **Refusal is the answer
    # in every one** — borch.ts handles dense tensors only, so there is nowhere over there
    # to ask it.
    # 158 → 137. **One reason was covering fourteen.** "alias or python" was true of the
    # big groups (`술어::` 23 is the `is_cuda`/`is_mps` kind, `저장::` 10 is `stride` and
    # `layout`, and the ungrouped 47 is copy semantics), but `분포::` 22 was hidden
    # underneath it — and **fifteen of those were things this code was not doing.**
    #
    # Five distributions went in without their refusals. They did not stop on an integer
    # slot (the five continuous ones have to), and the argument domains were not looked at
    # (`p` is an open interval, `lambda` is positive). Porting twenty-one filled those
    # three in. The one left (`random_(int64)`'s upper bound) **cannot be done** — torch's
    # is 2⁶², and int64 here sits in an f32 slot and cannot count above 2²⁴.
    #
    # The other 116 really are alias or python. Of the 40 in `짝에서::` only `i0_` has a
    # name over there, and they are mostly torch's **second spelling** (`divide_` = `div_`)
    # or the in-place bitwise and logical variants, so porting them asks the same question
    # twice.
    # 137 → 97. The 40 in `짝에서::` were ported. They were written down as "alias", and
    # **only ten of them were aliases** (second spellings like `divide_` = `div_`).
    # Seventeen had the arithmetic and lacked only the underscore name, and eleven were in
    # the kernel table alone, reachable only as `binary("gcd", …)` — while the line code
    # transcribed from torch types is `x.gcd(y)`.
    #
    # The remaining 97 really are python: `술어::` 23 (`is_cuda`, `is_mps`), `저장::` 10
    # (`stride`, `layout`), the ungrouped 47 (copy semantics, `from_numpy`), `희소::` 5.
    "inplace::": (97, "파이썬 — views, sharing, properties, predicates, storage"),
    # `method2::` used to be here — 60 cases, "alias — Python's second name, as
    # `multiply` = `mul`". Some of them were aliases, and **nine had no name over there at
    # all** (`fmax`, `vdot`, `moveaxis`, `t`, `broadcast_to`, four comparisons). They went
    # in and all of them were ported.
    #
    # Porting turned up two more. `fmax` and `fmin` were assemblies that lived only in the
    # binding, and `remainder` took a number alone, so `x.remainder(y)` did not run — **a
    # name that is there but narrow** is harder to find than one that is absent.
    # 48 → 9. Thirty-nine neighbours of the complex numbers were ported — `real`, `conj`,
    # `conjPhysical`, `resolveConj`, `resolveNeg` and `angle`, asked by value and by dtype
    # across three dtypes, and three judgements besides. The four that did not exist
    # (`resolveConj`, `resolveNeg`, `isConj`, `isNeg`) were built. **Having no lazy
    # conjugate bit is a circumstance of the implementation, not a reason for the question
    # to lose its meaning.**
    #
    # The nine left are **factories that do not exist over there** — `range` (it includes
    # the end), `frombuffer` and `asarray`. The first two have no name at all, and
    # `Tensor.from` is what stands where `asarray` does in TS. And `arange` took **one**
    # argument (torch takes three).
    # 9 → 2. `range` and `frombuffer` went in over there and `arange` takes three arguments
    # now. The two left are `asarray`, which takes a numpy array or a Python list, and TS
    # has neither.
    "make::": (2, "파이썬 — `asarray` takes an ndarray or a list; TS has neither"),
    # **A question TypeScript cannot be asked.** Python refuses `ZeroPad2d(1, 9.0)`
    # with a `TypeError` because the constructor takes one argument; JavaScript keeps
    # no arity and simply does not receive the second. So the same case name would
    # freeze `거절` on one side and `안 던졌다` on the other, and reconciling that
    # would mean writing an arity check into fifteen constructors that no other layer
    # in `nn.ts` has.
    #
    # **borch.ts was right about this from the start** — each pad class writes its
    # own constructor and only `ConstantPad*` has a value. The core handed it to all
    # fifteen, and `ZeroPad2d(1, 9.0)` filled with 9 while `ReflectionPad2d(1, 9.0)`
    # took it and dropped it. The four cases are the core's own.
    "pad::": (4, "파이썬 — JS keeps no arity, so a surplus argument is not received "
                 "rather than refused"),
    # `pool::` **was** a row here, reading *아직 — the arguments work in borch.ts; only
    # the TS case bodies are unwritten*, and that was the whole story: the eleven bodies
    # were written against `nn.AvgPool1d/3d` and `nn.LPPool1d/2d` exactly as they
    # already stood, and all eleven passed first time.
    #
    # It is worth keeping why they were owed rather than declined. The arguments —
    # `padding`, `countIncludePad`, `divisorOverride`, `ceilMode` — were **reachable
    # here before any case asked**, and a seat that is declared and a seat that works
    # look identical to the name axis, which counts declared names. So "borch.ts has
    # the argument" was never evidence that passing it did anything, and this row was
    # the ordinary kind of owed: a case list not carried across, not an argument that
    # does nothing.
    # `seq::` **was** a row here: *borch.ts's RNNBase has no batchFirst seat*. True when
    # written, and the sentence beside it explained the absence as a rule — accepting an
    # argument and not using it is a lie, so it was not accepted at all.
    #
    # **The rule was right and it was applied one step too far.** `batchFirst` is not
    # something this side cannot honour; it is a turn on the way in and a turn on the
    # way out. Leaving it out was declining to do something possible, and it was not
    # neutral either: `numLayers` and `bias` sit between it and `hidden` in torch, so a
    # line copied positionally landed `numLayers` in `batchFirst`. That is the defect
    # `EmbeddingBag`'s `mode` comment records having actually shipped.
    #
    # So the three seats went in at torch's indices, with the two that cannot be
    # honoured **refused rather than ignored** — which is the rule the row was appealing
    # to, applied to the two arguments it actually covers.
    # 47 → 50. The **kinds** of `finfo` and `iinfo`, and the no-argument default dtype. A
    # Python-side matter — neither name is in borch.ts.
    # 50 → 39. **The reason was explaining only eleven of them.** "top-level in-place
    # functions" was about the ten in `제자리::` (the four dropouts and `nan_to_num_`), and
    # those four could not be ported because they had no name over there — they went in and
    # were ported.
    #
    # The 39 left are **the Python surface**, and they part into three kinds:
    #
    #   `살펴보기::` 22 — `finfo`, `iinfo`, `can_cast`, `promote_types`, `result_type`,
    #                     `typename`. Looking into a dtype as a value, which has nowhere
    #                     to sit over there, where a dtype is a string.
    #   `device::` 9    — `torch.device` is an **object** with `.type` and `.index`.
    #                     `t.device` over there is a string, and `device()`, which hands
    #                     back the adapter, is an entirely different function.
    #   the other 14    — `resize_as_` (it swaps the handle out) · `inference_mode` (a with
    #                     statement) · round-tripping the random state · top-level
    #                     signatures taking an integer enum.
    # 39 → 45. Six `result_type` cases, added by the session holding the core because
    # **that name had no case at all** and what was public under it was the internal
    # numpy-dtype helper — torch takes tensors there and refuses dtypes, and this took
    # dtypes and refused tensors. Inverted end to end with nothing asking.
    #
    # They land here for the reason already written above rather than a new one: five of
    # the six hand back a dtype as a value, and over there a dtype is a string. The sixth
    # is a refusal whose whole content is which of two functions you called.
    "top::": (45, "파이썬 — dtype introspection, the `device` object, with, integer enums"),
    # `spot::` was gone — 47 cases, all ported — then back with two, and gone again.
    #
    # The two were `unique(dim=)`, and the reason written here said it is a different
    # operation from the value-wise form: with no `dim` it folds *values*, with `dim`
    # it folds **whole slices**, sorting them lexicographically and reporting an
    # inverse over slices rather than elements. **That was true, and it was doing an
    # excuse's work** — both forms are torch's, and a difference in kind is a reason to
    # write a second implementation, not a reason to have none. Two rows sat red under
    # it while the sentence read as a decision.
    #
    # `uniqueAlong` is that second implementation. The index arithmetic was checked
    # against numpy at fourteen shape-and-axis pairs before the browser saw it, because
    # gathering a slice out of a flat row-major buffer is where this goes wrong and a
    # golden run costs ten minutes to say so.
    "toplin::": (42, "별칭 — a top-level second name, as `lu` = `linalg.lu_factor`"),
    # `stat::` used to be here — 42 cases. 31 of them had simply never been asked, and the
    # other 11 could not be ported because **the name was not over there**, so those five
    # went into borch.ts.
    # `keep::` used to be here — 35 cases, `아직`. All ported, so the row is gone.
    #
    # Porting them showed **thirty-four values right on the first try.** The reductions'
    # `dtype=` rule (convert before accumulating) was already held, and so was the parting
    # where `sum(→bool)` is allowed and `cumsum(→bool)` is not. One thing was not held —
    # **`sum` alone had nowhere to take `dtype=`.** Its neighbours (`mean`, `prod`,
    # `nansum`, `cumsum`, `sumDim`) all took it, and the one name called most often of all
    # the reductions was missing it.
    # `blend::` used to be here — 34 cases. All ported, so the row is gone.
    #
    # **Thirty-four right on the first try.** Where `beta=0` drops out of the value and
    # stays in the graph, where `input` broadcasts as `(4,)` or as a scalar, where an
    # in-place variant returns itself — all already correct. No hole came out here — **a
    # place that was merely never asked is not the same as a place that is wrong, and
    # asking is what separates the two.**
    "fname::": (28, "별칭 — `F`'s in-place variants. The method side asks them already"),
    # `bit::` used to be here — 24 cases, "alias — the method names of the bit
    # operations". **The point was that those names were not over there**, and the reason
    # called them aliases. They went in and all were ported, so the row is gone.
    #
    # Porting them parted `bitwise_not(bool)`: a kernel comment had written down that "on
    # bool this is logical negation and the binding does the parting" — which leaves the
    # TypeScript side receiving `-2`. **Not a missing answer but a wrong one.** The parting
    # moved over there.
    # **It was not "mostly `repr`".** Laid out with `--show unpool`, six of the twenty-two
    # are `repr` and the other fourteen ask values, gradients and shapes. And those names
    # are **already in** borch.ts — `CTCLoss`, `FractionalMaxPool` and
    # `AdaptiveLogSoftmaxWithLoss` all stand as classes and only the cases never came.
    #
    # A reason frozen into one line hides the kinds parting inside it. This row showed that
    # for the sixth time, which is why the counts are written out by kind.
    #
    # **The row is gone and the correction it carried was the useful half.** "Mostly
    # `repr`" was wrong: six of the twenty were, and the other fourteen asked values,
    # gradients and shapes — which is what made this a case list rather than a hole, and
    # the classes really were all standing already.
    #
    # Fifteen of the twenty were right on the first run. The five that were not are worth
    # keeping because none is derivable: `MaxUnpool` prints its three arguments **spread
    # across the rank** and a `stride` left unset prints the kernel rather than `None`;
    # Python's one-element tuple carries a trailing comma, which showed up on one row of
    # one state-dict case (`head.bias(5,)`); and `LPPool3d`'s fixture is a 4³ volume, where
    # a 2³ gives a single window and an answer that agrees with nothing.
    #
    # One thing did move in the library: `ctcLoss` now takes the targets **flat as well as
    # padded**, cut by `targetLengths`. torch takes both, the flat form is what a real
    # loader produces, and a caller handing it over got the padded reading — comparing the
    # wrong letters and still getting a number.
    # `linalg::` used to be here — 17 cases. Sixteen had simply never been asked, and one
    # (`ldl_factor_ex`) could not be asked because the binding was standing three of its
    # slots up by hand.
    "grad::": (12, "별칭 — a vjp is `backward(seed)`, and parity asks it already"),
    "cplx::": (10, "파이썬 — a complex `repr` belongs to Python's formatter"),
    # Five of the ten buffer cases were ported (registration, keeping one out of the
    # state, listing, a value round trip) and five are left. **The reason parts in two for
    # what remains.** The three `InstanceNorm` cases are here because borch.ts has no such
    # **layer** — the tensor method `instanceNorm` is there and the layer is stood up on
    # the Python side. The two loss cases ask about a refusal, and in TypeScript passing an
    # argument that does not exist is a **compile error** rather than a refusal at run
    # time, so there is nowhere to ask it.
    "container::": (5, "파이썬 — the binding stands the InstanceNorm layer up, and the "
                       "refusal is a Python argument"),
    # `torch.pi`, `inf`, `nan` and `newaxis` are **values at Python's top level**. borch.ts
    # is a bundle of classes rather than a module, and JS already has `Math.PI`, `Infinity`
    # and `null`, so there is nowhere to offer the same names again — `x[:, None]` is
    # `unsqueeze(1)` over there too, so the indexing syntax itself is a Python matter.
    "const::": (6, "파이썬 — top-level values and indexing syntax. JS has no seat for it"),
    # 5 → 2. `searchSorted` and `bucketize` arrived (one binary-search kernel). The two
    # left are **refusals** and a Python matter — torch takes the same thing under two
    # names, `right` (a boolean) and `side` (a word), and stops when the two disagree.
    # borch.ts knows `right` alone, so there is no partner to disagree with.
    "index::": (2, "파이썬 — reconciling `side` with `right`. TS knows only one"),
    # 4 → 10. Six "resume training" cases arrived. **Those already tread on borch.ts** —
    # the binding's optimizers and schedulers call the ones over there as they are, so
    # those six running under `--lib borch_webgpu` are measuring borch.ts's `StepLR` and
    # its state-bank round trip, and `serialize` on the TS side pins the same thing again
    # in bytes. Four were still `LBFGS`.
    # 10 → 14. Four `save`/`load` round trips arrived. **On the TS side `serialize`
    # already asks it in bytes** — it round-trips the same codec and goes as far as
    # checking whether others (numpy, Python `borch`) can read it. Asking it once more
    # through the golden is Python's `torch.save(path)` surface, and borch.ts handles bytes
    # alone (files are the page's business), so there is nowhere to ask it.
    # 14 → 17. Three LBFGS cases arrived — the existing three **never trod on the
    # quasi-Newton part at all.** The closure fed the gradient in as a constant, so `y = 0`
    # and `ys = 0` and nothing accumulated in the history. The name was LBFGS and what was
    # measured was the first iteration's gradient descent, and nothing else.
    #
    # **The reason it is not in borch.ts is not "nobody built it".** This algorithm's
    # control flow depends on values (`ys > 1e-10`, the convergence test) and every read
    # over there is asynchronous — a number on the GPU cannot be looked at from inside a
    # synchronous `step()`. The binding manages because it has `run_sync`. Putting it in
    # would mean `async step(closure)`, and then it would be **the one asynchronous
    # optimizer** over there.
    # `opt::` has **no line left at all.** The three reasons written down were each wrong
    # in turn — "LBFGS cannot be used from a synchronous step" (true, and the conclusion
    # was wrong: make `step` asynchronous), "resuming training treads on the binding"
    # (simply untrue), "save/load takes a flat tensor table only" (true then; it carries a
    # tree now).
    # `vision::` was not in the list, so its cases were left **with no reason** and the
    # runner refused them (after `fda5540`). Written as one line saying `아직`, the kinds
    # parting inside it are invisible, so the counts are written out — the same reason this
    # file splits by prefix, one level further in.
    #
    # **42 → 38 is what that split did.** Written out, the 42 was "38 backlog + 4 holes".
    # The four holes were conversions the other side **already had**, with arguments too
    # narrow to match (`Resize`'s `max_size`, two of `describe()`'s four slots,
    # `RandomCrop`'s `padding` default), and `ae60832` mended them so all four answered.
    # Written as a single number, those four would have counted as backlog and nobody would
    # have touched them.
    #
    # **The `vision::` row is gone: it reached 0 and this table's own rule is that a
    # row with nothing left must be deleted.**
    #
    # It went 57 → 19 → 50 → 40 → 9 → 3 → 0, which is not a number failing to fall.
    # 94 cases were carried across while the Python side kept freezing more, and one
    # figure cannot show a debt being paid and taken on at the same time — so while
    # the row existed it carried both. That is the same reason this table splits by
    # prefix at all, one level further in.
    #
    # **The policy layer was narrow on the Python side too**, and that was said
    # rather than discovered. AutoAugment, RandAugment, TrivialAugmentWide and AugMix
    # all draw on every call, so what could be frozen was the three learned tables
    # as text plus RandAugment(num_ops=0), the one configuration of any of them that
    # does not draw. Without being told that boundary, an hour goes into hunting for
    # an AugMix value case that does not exist and concluding something was missed.
    # `ops::` **was** a row here, reading *아직 — the eleven box-geometry functions.
    # Pure arithmetic, so carrying them across needs no model.* It is gone because
    # `borch-ts/src/ops.ts` exists now and all sixteen cases passed on their first run.
    #
    # The prediction in that row is worth keeping even though the row is not. It said
    # the marker was `아직` and not `없음` — that borch.ts having no `nms` was true but
    # not the point, because these eleven touch no weights and no feature map and so
    # needed no model to carry across. That turned out to be the whole story: the port
    # is arithmetic on rows of four numbers, read back once per call. **A debt written
    # down with the reason it was cheap gets paid; one written as a number does not.**
    #
    # The other twenty-eight of torchvision's `ops` (RoI, FPN, detection losses) do want
    # a detector, and they are absent from the Python side too — so they were never part
    # of what this row recorded, and their absence is not a debt this file can see.
    # 52 of the 71 are **repr strings**. v2 computes what v1 computes and differs only
    # in what it prints, and that difference is the whole reason these names are not a
    # re-export — so the strings are frozen as strings. The other 19 are values at the
    # settings where the draw has nothing to draw. Portable, both halves: the arithmetic
    # is v1's, already here, and a repr is a string comparison on either side.
    # Six, and all six are **decoders on bytes built in the case table** — an IDX
    # header and a CIFAR batch, written out rather than downloaded. Nothing here
    # touches a network, so they port like any other value case; what does not port
    # is the rest of `datasets`, which on that side is a `fetch` and an OPFS cache
    # and has no case in this table at all.
    # `v2f::` **was** a row here — *the nine v2 adds, and eight of v1's re-exports*.
    # `borch-ts/src/vision_v2.ts` exists now and all seventeen passed first time. The
    # eight were bound rather than wrapped, because a wrapper is a body and a body is
    # the thing those eight cases exist to prove is absent.
    #
    # One of the seventeen had to be paid for differently. `elastic`'s displacement is
    # built on the Python side with numpy's `default_rng(7)`, and PCG64 has no
    # counterpart in TS — **a field from a different generator is a different input**,
    # and two answers to two different questions do not compare. So the forty numbers
    # are transcribed into the case table. That is the second fixture in this repository
    # written out rather than drawn, and both times for the same reason: the alternative
    # was a green run that had compared nothing.
    # **The first `misc::` row.** Every case with that prefix had been carried across
    # until now, which is why this table has never had one — and it is worth a line
    # because a prefix appearing here for the first time reads like an oversight.
    #
    # Eleven, and they were one kind: `Embedding` and `EmbeddingBag` with `max_norm`,
    # plus `padding_idx`. **The row is gone** — borch.ts already did all of it
    # (`renormRows`, and the padding row's gradient cut by splitting the table into a
    # part the gradient flows through and a detached part), so this was the same shape
    # as `pool::`: a case list not carried across, not a mechanism missing.
    #
    # Its prediction is the part worth keeping. Four of the eleven read the **weight
    # table after the call**, because that is the only place the difference lives —
    # `max_norm` shortens rows in place, and an implementation that shortened a copy
    # would return the same numbers forever.
    #
    # **Measured, and the first measurement was of the wrong thing.** A mutation that
    # scaled a copy and then looked up from the *original* reddened seven cases, value
    # cases included — but that is not the defect these were written for, it is a
    # cruder one. Scaling a copy and looking up from **that copy** is the real shape:
    # `Embedding`'s value case stays green and only the table cases move. Two more,
    # each landing where its name says: leaving `paddingIdx`'s gradient uncut reddens
    # `grad::Embedding(padding_idx)` and nothing else, and ignoring `normType` reddens
    # the two `norm_type=1` rows and nothing else.
    #
    # Worth writing down because the claim was in this comment before the run was: it
    # said the copy mutation "reddens exactly those four", and the number was seven.
    # A prediction in the same sentence position as a measurement reads as one.
    # 19 → 3. Sixteen went across into `borch-ts/src/datasets.ts` — IDX (including the
    # int32 two-axis table QMNIST's labels have), STL10, a `.npy` reader for
    # MovingMNIST, and FER2013's CSV in both of its layouts.
    #
    # **The three left are CIFAR, and the reason is Python's, not a difficulty.** Its
    # batches ship as **pickles**, and a pickle is Python's own serialization rather
    # than a data format. Checked rather than assumed: `site/assets/datasets.js` reaches
    # CIFAR through a **JPEG sprite**, so nothing on the browser side ever meets one.
    # Carrying a pickle parser across would be building a reader for a format that side
    # does not encounter, and its own risk — a silent mis-parse of a nested numpy
    # `__reduce__` — is the shape this repository spends its time removing.
    # 3 → 5. **`ImageFolder` arrived on the Python side and cannot arrive here**, and
    # it is the clearest case in this whole table: the class walks a directory and
    # calls a loader on each path, and a page has no directories to walk. Everything
    # else under `dataset::` is a format that could in principle cross; this is a
    # filesystem, which is not a format at all.
    "dataset::": (54, "파이썬 — CIFAR's batches are pickles and the browser side "
                      "reaches CIFAR through a JPEG sprite instead; `ImageFolder`, "
                      "CLEVR, RenderedSST2, the twelve stereo and flow sets, Kitti, PhotoTour and the eleven that walk a listing and call a loader "
                      "walk directories and read archives off disk, and a page has "
                      "neither"),
    # `v2::` **was** a row here — 71, then 52, then gone. The nineteen that needed no v1
    # field access went first; the fifty-two twins followed once it was clear the fields
    # did not have to be read at all. Each twin extends the v1 class and keeps its own
    # copy of the arguments, so `vision.ts` did not have to open thirty-six
    # constructors for this file's benefit.
    #
    # **Thirty-one of the fifty-two reprs were right on the first run and twenty-one
    # were not**, and what the wrong ones were wrong about is the useful record: v2
    # prints `interpolation` as the plain value where v1 prints the enum; `Resize` shows
    # its size as a list while every crop shows the expanded pair as a tuple;
    # `ColorJitter` clamps three of its four spans at zero and not the fourth;
    # `RandomAdjustSharpness` prints its factor as given while the `p` beside it carries
    # a decimal point; and `ElasticTransform` recovers two fields from the v1 object
    # while keeping the third as the argument. Not one of those is derivable — every one
    # came back from the golden.
    "cache::": (4, "별칭 — parity asks the same thing about soiling a global constant"),
    # 3 → 7 → 6. Four `DataLoader` rows went in with this reason:
    #
    #     **borch.ts has no `DataLoader` at all** — over there a batch comes from
    #     `Dataset.batches()`, so there is nothing to give a generator to.
    #
    # **That sentence was false the day it was written.** `borch-ts/src/data.ts` has
    # exported `class DataLoader` since 2026-08-17 and the reason above is dated
    # 2026-08-23 — six days later, in bold, about a file that already had the class.
    # Not a reason that went stale; one that was never true.
    #
    # The cost is not that the number was wrong — it was right, the four rows really
    # could not be asked. The cost is that **a false reason points at the wrong
    # repair.** "no DataLoader" says *build one*, and nobody was going to; the truth
    # was "the DataLoader takes three options and torch's list is seventeen", which
    # says *give the one you have a sampler*, and that took an afternoon.
    #
    # `batch_sampler` is asked now. What is left:
    #   · two `generator` rows — one host stream, the decision at the top of data.ts
    #   · `drop_last` given by **position** — borch.ts takes an options object, so
    #     there is no ninth seat to put it in. That one is the language, not a gap.
    "dataconv::": (6, "파이썬 — `default_convert`, `get_worker_info`, the two "
                      "`generator` rows (one host stream), and `drop_last` given by "
                      "position (borch.ts takes an options object)"),
}


def unasked_report(report, show=None):
    """Show what was never asked **grouped by prefix**, and sift out what has no reason.

    Print the count alone and nobody knows what the number is made of. `679 cases` looks
    exactly the same whether or not "deliberately not ported" and "forgotten" are mixed
    inside it.

    Give `show` a prefix and it **lays out every name in that seat.** From a count and a
    one-line reason there was no way to ask "what is missing" — porting a group means
    seeing its list first, and with nowhere handing that list over the case table had to be
    searched by hand. That a reason frozen into one line hides the kinds parting inside it
    is why this repository started writing the reasons out per prefix, and the same reason
    was waiting one level further in.
    """
    import json

    doc = json.loads((ROOT / "tests" / "golden.json").read_text(encoding="utf-8"))
    asked = set(report.get("asked", ()))
    rest = [n for n in doc["cases"] if n not in asked]
    groups = {}
    for name in rest:
        head = name.split("::", 1)[0] + "::" if "::" in name else "(no prefix)"
        groups.setdefault(head, []).append(name)

    if show is not None:
        want = show if show.endswith("::") else show + "::"
        names = groups.get(want, [])
        out = [f"  never asked in {want} — {len(names)}:"]
        out.extend(f"    · {n}" for n in sorted(names))
        return out

    lines = []
    surprise = []
    for head, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        entry = NOT_PORTED.get(head)
        if entry is None:
            lines.append(f"    ✘ {head:18s} {len(names):>4}  "
                         f"**no reason is written down**  [{names[0]}]")
            surprise.append(f"{head} ({len(names)}, no reason)")
            continue
        frozen, why = entry
        mark = " " if len(names) == frozen else "✘"
        lines.append(f"    {mark} {head:18s} {len(names):>4}  {why}")
        if len(names) != frozen:
            surprise.append(f"{head} (written {frozen}, actual {len(names)})")
    if lines:
        lines.insert(0, "  never asked — by prefix:")
    # A row in the list with nothing left under it is **fully ported**. That row has to be
    # deleted — left standing, the next person reads it as work not yet done.
    for head, (frozen, _) in NOT_PORTED.items():
        if head not in groups:
            surprise.append(f"{head} (all ported — delete the row)")
    if surprise:
        lines.append("  ✘ places that do not reconcile: " + " · ".join(surprise))
        lines.append("     Ported, lower the number; not ported, raise it with the "
                     "reason.")
    return lines


def main(argv):
    dist = ROOT / "borch-ts" / "dist" / "test" / "golden.js"
    if not dist.exists():
        # Better not to run at all than to run on a stale dist.
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2

    report = run(headed="--headed" in argv, verbose="--verbose" in argv)
    if "error" in report:
        print(f"it could not run: {report['error']}", file=sys.stderr)
        return 1

    # **Say which device it ran on, and say it beside the score.** It used to be printed
    # here, nineteen lines above `passed N / failed M`, with the failing cases in between
    # — and a whole session of runs went by on `google / swiftshader` with nobody
    # reading it, because whoever wants the number tails the last three lines.
    #
    # The same repair was made one file over for the *library* name, and its note says
    # why in a sentence that fits this exactly: **the name goes on the line that carries
    # the number, not in the header, because the header is not where somebody looking
    # for a score reads.** The adapter was in the identical position and was not moved,
    # because that fix was about `lib` rather than about the line a person reads. **A fix
    # aimed at one symptom leaves the same class one variable over.**
    adapter = report.get("adapter", "(unknown)")
    gap = report["total"] - report["registered"]
    print(f"{report['registered']} of the golden's {report['total']} are written in TS "
          f"— {gap} have not been asked yet.")
    show = argv[argv.index("--show") + 1] if "--show" in argv else None
    gap_lines = unasked_report(report, show)
    for line in gap_lines:
        print(line)
    gap_ok = not any("✘" in line for line in gap_lines)
    for name in report["unknown"]:
        print(f"  ? the name is not in the golden: {name}")
    for f in report["failed"]:
        print(f"  ✘ {f['name']} — {f['why']}")
    print(f"passed {report['passed']} / failed {len(report['failed'])}  [{adapter}]")
    # **It does not block.** The golden asks about values and the device does not change
    # them, so a pass on a CPU is a real pass — refusing to run at all on the one machine
    # there is trades a real check for no check. What must not stand is the word `passed`
    # with nothing beside it, and now it cannot.
    warn_if_software(adapter, "the values")

    # **What was never asked is a failure too** — where no reason is written down or the
    # numbers do not reconcile. One line of count cannot show the golden growing while the
    # TS side fails to follow.
    ok = not report["failed"] and not report["unknown"] and gap_ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
