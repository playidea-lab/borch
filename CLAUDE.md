# borch — rules for an agent working here

Three rules, each written after it cost something measurable. The reasons are the rules.

## 1. One session, one worktree — never share the checkout

Two sessions worked in this directory on 2026-09-02 and collided four times in one
afternoon: a commit landed on top of somebody's uncommitted file; the branch was switched
underneath a running session; a commit vanished from a branch that was rewritten; a
cleanup raced a checkout. Nothing was lost — each time by luck and a hash check.

So: **before touching files, make your own worktree and work there.**

    git worktree add ../borch-<what-you-are-doing> origin/main
    ln -s /Users/changmin/git/borch/node_modules ../borch-<…>/node_modules
    cd ../borch-<…> && npm run build:ts

`uv run --project /Users/changmin/git/borch …` reuses the main environment (torch is
large). Push with `git push origin HEAD:main` from the worktree and remove it after.
Nine worktrees already exist beside this checkout; the practice is established, the
main checkout is the exception — and it is the one that gets shared.

## 2. Nothing runs the browser checks unless something is scheduled

Thirteen entry points need a browser and CI runs none of them (`.github/workflows/gpu.yml`
says why and counts them; `tests/test_browser_entry_points.py` holds the count). When
they were run by hand after a gap, two were red for nobody knew how long and a real
defect in the core was visible only through Pyodide.

`tests/browser/nightly.py` runs the twelve correctness checks in a worktree of
`origin/main`; `~/Library/LaunchAgents/co.pilab.borch-nightly.plist` fires it at 04:30.
Logs are under `~/Library/Logs/borch-nightly/`. **Read the last log before assuming
green.** If the agent is not loaded, load it:

    launchctl load ~/Library/LaunchAgents/co.pilab.borch-nightly.plist

## 3. A name bound to another name's argument list is a defect, and it recurs

`linalg.norm` → `torch.norm`, `linalg.svd` → `torch.svd`, `linalg.pinv` → `pinverse`,
`special.softmax` → `F.softmax`: four times one function stood under two torch names
whose documented lists differ, and each time the shared binding accepted what one refuses.
`tests/test_one_name_one_list.py` now holds this for every namespace. When it fires,
**split the binding** — do not attest the row unless torch's two lists genuinely agree.

## Everything else

The rest is in the code, which says why beside what. Reasons for declined names live in
`tests/torch_gap.py`; the browser-side ledger in `borch-ts/test/run.py`; the argument
axes in `tests/test_torch_signatures_core.py` and `tests/test_ts_signatures.py`. Every
count in those files carries the reason it last moved. Move a count only with one.

Measure against real torch before writing a case; a probe that prints seventy
characters will get transcribed as the whole message.
