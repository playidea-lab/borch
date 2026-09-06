# borch — rules for an agent working here

Four rules, each written after it cost something measurable. The reasons are the rules.

## 1. One session, one worktree — never share the checkout

Two sessions worked in this directory on 2026-09-02 and collided four times in one
afternoon: a commit landed on top of somebody's uncommitted file; the branch was switched
underneath a running session; a commit vanished from a branch that was rewritten; a
cleanup raced a checkout. Nothing was lost — each time by luck and a hash check.

So: **before touching files, make your own worktree and work there.**

    git worktree add ../borch-<what-you-are-doing> origin/main
    cd ../borch-<…> && npm ci --ignore-scripts && npm run build:ts

**`npm ci`, not a symlink to the main checkout's `node_modules`.** The symlink was the
rule until 2026-09-06, and the nightly's golden rows were red for three nights because the
link's target held what somebody had installed in August; esbuild, bimm-ts and borch-hub
had joined the lock and never arrived there. Thirty seconds of `npm ci` reads the lock the
worktree was checked out with. `uv run --project /Users/changmin/git/borch …` still reuses
the main environment (torch is large). Push from the worktree — a PR, or
`git push origin HEAD:main` — and remove the worktree after.

The main checkout `/Users/changmin/git/borch` is a **mirror of `origin/main` and nothing
else**: no branch of its own, no edits, no commits. On 2026-09-07 it was found on a
version branch 132 commits behind, holding 576 lines of a session's leftover work and an
untracked copy of `nightly.py` that launchd had been running for four days. Its
`node_modules` is kept installed for whoever still links to it.

## 2. Nothing runs the browser checks unless something is scheduled

Thirteen entry points need a browser and CI runs none of them (`.github/workflows/gpu.yml`
says why and counts them; `tests/test_browser_entry_points.py` holds the count). When
they were run by hand after a gap, two were red for nobody knew how long and a real
defect in the core was visible only through Pyodide.

`tests/browser/nightly.py` runs the correctness checks (twenty-eight rows; the first-run
clock twice: this checkout, and the deployed site) in the worktree `../borch-nightly`,
moved to `origin/main` each night; `~/Library/LaunchAgents/co.pilab.borch-nightly.plist`
fires it at 04:30 **from that worktree's copy of the script**, so the run always uses
main's `nightly.py` as of the night before. Logs are under `~/Library/Logs/borch-nightly/`.
**Read the last log before assuming green.** If the agent is not loaded, load it:

    launchctl load ~/Library/LaunchAgents/co.pilab.borch-nightly.plist

**One machine is one adapter.** The second is an RTX 5080 on Ubuntu (`/home/pi/borch-nv`,
Chrome 151, driver 580, `DISPLAY=:1`); `tests/browser/borch-nightly.{service,timer}` are
its systemd user units and the install command is in the service file. Installing them is
a person's decision — that machine runs other people's work. Until both logs exist, a
sentence that says "verified on the GPU" names one adapter.

## 3. A name bound to another name's argument list is a defect, and it recurs

`linalg.norm` → `torch.norm`, `linalg.svd` → `torch.svd`, `linalg.pinv` → `pinverse`,
`special.softmax` → `F.softmax`: four times one function stood under two torch names
whose documented lists differ, and each time the shared binding accepted what one refuses.
`tests/test_one_name_one_list.py` now holds this for every namespace. When it fires,
**split the binding** — do not attest the row unless torch's two lists genuinely agree.

## 4. Many sessions, one machine — the machine is the shared thing now

Rule 1 kept sessions out of each other's files. On 2026-09-06 and 07 they met on the
machine instead: a headed Chromium never launched while another session's headless one
was up; two runs fought over the uv cache lock; a cleanup by process name took another
session's browser with it; a nightly run died the same way; and one session's push turned
main red for an hour while another's PR waited behind it.

- **One browser probe at a time.** Every probe takes `tests/browser/launch.py:probe_lock`
  through `serve()` and waits, saying whose turn it is. Do not work around it; do not run
  browser probes between 04:30 and 06:00, which is the nightly's.
- **Never kill a browser by name.** `pkill chromium` takes every session's. Kill your own
  launch's process tree by PID, or nothing.
- **Whoever turns main red fixes main first.** Before your next push, `git fetch` and
  rebase onto `origin/main`, `npx tsc -p borch-ts/tsconfig.json --noEmit`, and `npm run
  sync` (the API index, the counts on the pages, and a list of names still without their
  Korean). A stale count or index is the commonest way main goes red, and CI catches it
  only after it has.
- **Say what you are doing on the machine** when it takes the GPU for minutes — a
  cross-session message costs a line; a killed run costs the night.

## Everything else

`README.md` is a front door and `docs/BOOK.md` is the long document it opens onto — the
book was the README until 2026-09-03, and the checks in `tests/` police both. Put a
measured claim in the book; put a link to it on the door.


The rest is in the code, which says why beside what. Reasons for declined names live in
`tests/torch_gap.py`; the browser-side ledger in `borch-ts/test/run.py`; the argument
axes in `tests/test_torch_signatures_core.py` and `tests/test_ts_signatures.py`. Every
count in those files carries the reason it last moved. Move a count only with one.

Measure against real torch before writing a case; a probe that prints seventy
characters will get transcribed as the whole message.
