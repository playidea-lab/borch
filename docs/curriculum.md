# The borch learning tree

**How the lessons are organised, and how to add one.** The site's Learn section is a tree
of lessons rendered from a single manifest; this page is for whoever grows it — a person or
an agent. The lessons themselves live at
https://playidea-lab.github.io/borch/site/learn/ ; this is the map behind them.

The guiding idea is the same one the rest of borch runs on: **code beside an explanation
that does not run is a picture.** Every lesson's examples run in the reader's tab, on their
GPU, and a lesson with an exercise says whether the reader got it. So a new lesson is not
prose — it is runnable code with a verdict, held to the same library the tests run against.

---

## One source of truth

Everything about the tree — the course home, each reader's progress, the "new since your
last visit" badges — is rendered from **`site/assets/curriculum.json`**. Adding a lesson is,
at its core, adding one entry there. Do not hand-maintain a second list of lessons anywhere;
a number written into prose is stale the day the next lesson lands.

Three files turn that manifest into the course:

| File | Role |
|---|---|
| `site/assets/curriculum.json` | The tree as data — parts, lessons, prerequisites, status. The SSOT. |
| `site/assets/course.js` | Renders the course home from the manifest: parts → lessons, a progress bar, per-part fractions, a resume link, NEW badges. |
| `site/assets/progress.js` | Per-browser completion in `localStorage`, keyed by a lesson's site-relative path. No login, nothing leaves the tab. |

Completion is set two ways, and the reader controls the first: the **checkmark in front of a
lesson toggles it done**, and a lesson whose exercise **verdict** passes also marks itself
(via `runnable.js`). The verdict is a bonus on top, never the gate — a visitor with no WebGPU
adapter, or on a browser without JSPI for the Python twin, can never make a verdict pass, and
must still be able to record progress.

---

## The shape of the manifest

`curriculum.json` has two arrays. `parts` orders the sections; `lessons` places each lesson
in a part with an order and its prerequisites. The live content is in the file — read it
rather than trusting a copy here — but the schema is:

```json
{
  "parts":   [ { "id": "foundations", "title": {"en": "...", "ko": "..."},
                 "blurb": {"en": "...", "ko": "..."} } ],
  "lessons": [ { "id": "f-vectors", "part": "foundations", "order": 1,
                 "prereq": ["..."],
                 "title": {"en": "...", "ko": "..."},
                 "url": "foundations/01-vectors.html",
                 "ko_url": "ko/foundations/01-vectors.html",
                 "added": "2026-09-14", "verdict": true, "status": "exists" } ]
}
```

- **`status`** is `exists` (live, has a page) or `new` (planned; `url`/`ko_url`/`added` are
  `null`). A `new` leaf shows on the course home as *soon*, so the arc is visible before the
  page is written.
- **`added`** dates a leaf so the course home can badge it NEW for returning readers. Lessons
  that predate the manifest share one baseline date; new leaves carry the real date.
- **`prereq`** is what the lesson leans on. It is the DAG of the tree — keep it honest, it is
  what lets the arc be read as a dependency order rather than a numbered list.
- **`verdict`** records whether the lesson has a fix-it exercise. It is metadata; the actual
  verdict lives in the page's `data-verdict` attribute.

The arc, in dependency order: **onboarding → foundations (the math, run) → basics (tensor to
training loop) → depth (real networks, data, debugging) → transformers → papers (one idea
from a paper, wrapped in code you run) → extras.**

---

## Where a lesson's files live — and why it matters

Lessons are plain static HTML under `site/`, one file per language:

| Section | Directory | English + Korean |
|---|---|---|
| Basics / depth / transformers | `site/learn/` | `learn/NN-slug.html`, `ko/learn/NN-slug.html` |
| Project walk-throughs | `site/tutorials/` | `tutorials/NN-slug.html`, `ko/tutorials/…` |
| Foundations (math) | `site/foundations/` | `foundations/NN-slug.html`, `ko/foundations/…` |
| Paper case studies | `site/papers/` | `papers/NN-slug.html`, `ko/papers/…` |

**Foundations and papers are deliberately their own directories, not part of `learn/`.** A
site guard (`tests/test_site.py::test_the_site_counts_the_pages_it_links_to`) checks every
"N lessons" claim on the site against the number of files in `site/learn/`, and several pages
say *eleven lessons*. Dropping new files into `learn/` would silently break those claims. The
course home unifies all the sections from the manifest anyway, so a new cluster gets its own
directory and leaves the `learn/` count alone.

---

## Adding a lesson — the checklist

1. **Write the page** at `site/<section>/NN-slug.html`, and its Korean twin at
   `site/ko/<section>/NN-slug.html`. Copy an existing lesson in the same section as the
   scaffold — the head (title, description, the `%OG_BASE%` share tags, the CF beacon), the
   global nav, and the `<aside class="sidebar">` must match its neighbours exactly.
2. **Give it runnable blocks.** Each `<div class="runnable">` holds a JavaScript source and,
   where the binding supports it, a Python twin — the same work on each, so a reader can press
   and compare. See the conventions below.
3. **Add the manifest entry** to `curriculum.json` — one object in `lessons`, with the part,
   order, prereqs, both titles, both urls, today's date, and `status: "exists"`.
4. **Update the section's sidebars.** Every page in a section lists every numbered page of
   that section, in order (`test_a_section_sidebar_lists_every_page_of_its_section`). Adding
   a leaf means adding its line to the sidebar of every existing page in the section, both
   languages.
5. **Add both pages to the nightly runner** — `borch-ts/test/lessons.py`'s `PAGES` list. This
   is what presses the Run buttons on a real GPU each night; a page absent from it is reported
   as *wants reviewing*.
6. **Run the guards:** `uv run --with pytest --with playwright pytest tests/test_site.py
   tests/test_lessons_coverage.py -q`. They are fast and they catch broken links, dangling
   classes, a nav that drifted, a dual-language block that lost a half, and a stale sidebar.

---

## Runnable-block conventions

- **Two languages, same work.** Write a `data-lang="js"` source and a `data-lang="py"` twin.
  The reader switches with one tab and compares the numbers — better than the sentence "they
  agree." Keep the code identical between the English and Korean pages; translate only the
  prose. Code comments stay in **English** (this is public OSS); the surrounding prose is
  Korean on the `ko/` pages.
- **JS-only is allowed, when declared.** If a feature is in `borch-ts` but not yet in the
  Python binding (LoRA / `borch.peft` is the current example), ship a JS-only lesson and say
  so in a `note-box`. State the split; do not leave it silent. `--py` in the runner simply
  skips a block with no Python twin.
- **Do not redeclare an injected name.** The runner spreads these into every JS block, so a
  `const` of the same name makes the joined module a syntax error
  (`test_no_block_declares_a_name_the_runner_injects` catches it): `init, Tensor, nn, optim,
  data, vision, ops, fft, linalg, scope, keepAlive, noGrad, manualSeed, einsum, slice, save,
  load, isAvailable, probe, currentDevice, device, Device`, plus `log, plot, show, stopped,
  datasets`. Namespaces not on that list — `peft`, for one — are reached through the `borch`
  object (`const { peft } = borch;`).
- **JS reads values with `await`.** `item()` and `toArray()` cross back from the GPU and
  return promises; forward and backward are synchronous. Python does not await.

---

## Verdicts — and the traps that are worth avoiding

A lesson with a fix-it exercise carries `data-verdict="loss<X>"` on the block. The runner
reads the last `loss N` the block printed and judges it; the reader is told *learned* or
*not yet*. A block ships in its broken state (the reader fixes one marked line), and a
failing verdict is `verdict bad`, **not** an error — the nightly presses the shipped state and
passes it, because the intended failure is not a broken page.

Three real traps, each caught the hard way on a real adapter:

1. **`data-verdict` must be a plain decimal.** The rule is matched by `^loss<([0-9.]+)$` —
   `loss<1e-6` will not fire. Write `loss<0.0001`.
2. **Do not guess a convergence hint you cannot run.** The LoRA leaf shipped
   `try 0.1`, which *diverged* (the adapter is a bilinear product); the Adam leaf shipped
   `try 0.1`, which merely fell short of the bar in the step budget. Both misled the reader,
   and both were only found by pressing the fixed path in a browser. The nightly presses the
   *broken* state, so it does not check your hint. Prefer a **deterministic verdict** (see
   below); if the exercise really is about convergence, have someone run the fixed value
   before shipping the hint.
3. **A training target must not carry the graph.** Building a target from `model.call(x)`
   when the model's parameters require grad, then reusing that target across steps, throws
   *"backward through the graph a second time"* on the second step. Wrap the target in
   `noGrad(() => …)` so it is a constant; the loss rebuilds a fresh graph each step.

**Prefer deterministic verdicts.** The strongest exercises do not depend on an optimiser
converging at all — Dropout and the two norm lessons check a property that is exactly true
when fixed and exactly false when broken (a train/eval mode switch; a row's mean going to
zero). There is no step budget to tune and nothing to run to be sure of the threshold.

The nightly can check the *fixed* path automatically when the fix is a value: it reads the
value from the `// <-- fix me (try X)` comment and presses the block with it applied. So the
hint comment is the one source of that value — do not also declare it in an attribute, or the
two will drift.

---

## Verification is the nightly's job

These pages cannot be trusted from a headless Mac — it falls back to SwiftShader, a software
adapter, which is not what a reader has. The arbiter is `tests/browser/lessons.py`, run each
night on a real adapter (windowed, so the probes get the GPU), which presses every block on
every page in `PAGES`. A green local `pytest` run means the structure is sound — links, nav,
sidebars, dual-language halves — not that the code runs. Ship on the structural guards, and
let the nightly confirm the run.
