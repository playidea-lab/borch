# site — the explainer pages and the playground

Static pages that show what `borch` is in thirty seconds and let you run it once on the
spot. No build tool and no framework — HTML, CSS and ES modules the browser simply reads,
so this page keeps the repository's zero-runtime-dependency rule too.

```bash
npm run build:ts          # a browser cannot read TypeScript as it is
npm run site              # extracts the API list and serves it · http://127.0.0.1:8123/site/
```

Korean is below that, at `/ko/`.

`python3 -m http.server` works too, as long as it serves **the repository root.** Serving
`site/` alone does not — the pages call `../borch-ts/dist` (and, in Python mode,
`../vendor/pyodide` and `../borch*/`).

## What is where

| Place | What it does |
|---|---|
| `index.html` | The landing page. The hero's demo really runs |
| `learn/` | Ten lessons (concepts). **Every code block runs where it stands** |
| `tutorials/` | Ten projects. Things that finish if you follow them |
| `api/` | The API reference. The list comes from `assets/api.json` |
| `playground.html` | A large editor · a loss curve · GPU instrumentation |
| `ko/` | The Korean edition of the four above. Uses the same assets through `../assets/` |
| `build_api.py` | `borch-ts/dist/src/*.d.ts` → `assets/api.json` |
| `fetch_data.py` | A CIFAR subset → `assets/data/` (gitignored). Tutorials 4 and 5 use it |
| `assets/runner.js` | Where borch is loaded and user code is run. **Every page uses the same one** |
| `assets/runnable.js` | The runnable code blocks in the lessons and tutorials |
| `assets/render.js` | Tensors as pictures and curves (`show`, the loss curve) |
| `assets/datasets.js` | Reads one sprite sheet back into a tensor |
| `assets/docs.js` | The API reference screen (search, sidebar) |
| `assets/examples.js` | The playground examples. Six JavaScript, five Python — each `{en, ko}` |
| `assets/i18n.js` | The two languages for wording the code produces |
| `assets/playground.js` · `assets/home.js` | The wiring for each screen |
| `assets/style.css` | Both the light and the dark side. No fonts fetched from outside either |
| `serve.py` | Serves the root and opens the page |

## Information architecture

**The global menu is one set on every page** — `Learn · API · Playground · GitHub · language`.
Where you are is marked with a single underline (`.top nav a.on`).

For a while the landing page alone carried extra `Why · How` anchors, and that made **the menu
look like it changed** on the way from Learn to Playground. Global navigation and in-page
navigation are different things, so the layers were separated:

| Layer | What | Where |
|---|---|---|
| Global | Learn · API · Playground · GitHub · language | The top of every page, in the same order |
| In-page | The landing page's section list | A thin strip just under the top bar (`.subnav`) |
| In-page | The lesson list · the module list | The left sidebar |

API and Playground links went into the sidebar too and then came out — a second road to the
same place as the global menu is not help, it is one more choice.

## How the two languages are kept

**The prose is two sets of HTML, and only the wording the code produces lives in one place,
`i18n.js`.**

Injecting the prose with JavaScript too would give a blank page to a browser with JavaScript
off, and to a link preview. An explainer page in that state has no reason to exist. The cost
taken instead is that **the two sets can drift** — changing wording means changing
`index.html` and `ko/index.html` together.

Which language it is, is decided by `<html lang>`. That travels with the path (`/ko/`) but does
not read the path — reading it would silently revert the wording to English the moment a file
moves.

**The example code has to be identical in both languages.** What differs is the comments and
the printed wording, nothing else. If the code differs, this page's claim — that JavaScript and
Python give the same values — becomes false.

## The API reference is not written by hand

`build_api.py` extracts it from the declaration files (`.d.ts`). `tsc` keeps the TSDoc comments
as they are, so **the original of every description is always the source**, and a wrong
description is fixed in the source.

A list of hundreds written by hand is right for the first week only. This repository has caught
four stale numbers and all four were found by eye — something far larger cannot be managed the
same way.

`tests/test_site.py` checks that `assets/api.json` equals what is extracted from the source
right now. Edit the source without re-extracting and it stops there. On a checkout with no
declaration files it skips.

> What actually happened while using the generator: counting bracket depth separately per
> branch caught **18 of the 422 tensor methods.** The screen looked fine — what is absent and
> what was not found look the same. Leaving out a modifier (`protected`) and an optional
> property (`initialLr?`) had the same symptom, and 80 names were quietly missing.

## The data is kept outside git

Only tutorials 4 and 5 use a real dataset. The original is 29MB per batch and is not committed
— the same rule as `vendor/pyodide` and `borch-ts/dist`. `fetch_data.py` turns 2,000+500 images
into one sprite sheet and leaves about 1MB in `assets/data/`, and that folder is gitignored too.
The deploy workflow builds the same thing in CI with `--download`.

**Being JPEG, the pixels are not identical to the original.** That was chosen by measuring —
the same 2,000 images are 5.9MB as originals, 4.1MB as PNG and 0.84MB as JPEG. The question a
tutorial answers is "does it learn", not the absolute value of an accuracy, and that fact is
written on every page where a number appears. Where a value has to be exact (the golden
answers), this file is not used.

## The lessons and tutorials are run, not read

The code blocks in `learn/` sit in `<script type="text/plain">` and `assets/runnable.js` turns
them into an editor with a Run button. Whoever is reading can edit and run.

**The code has to be identical in both languages.** What differs is the comments and the printed
wording — if the code differs, the claim that JavaScript and Python, and the two language
editions, give the same values becomes false.

## How it runs

**JavaScript mode** — user code is made into an ES module (a Blob) and hung on `import()`.
`new Function` accepts neither top-level `await` nor `import`, and borch's first line is
`await init()`, so both are needed.

**Python mode** — the same thing `tests/browser/runner.html` used to do. borch.ts loads first,
gets the adapter, and goes into `globalThis.borch`, from where `borch_webgpu` inside Pyodide
picks it up as `js.borch`. Pyodide, numpy and every Python package come from this repository.

**Syntax highlighting** — a coloured `<pre>` lies underneath and the `textarea` above it keeps
its letters transparent, showing only the caret. Typing, selection, undo and Hangul composition
all belong to the browser, so there is nothing to imitate. The two layers' font, line height,
padding and tab width are set **once** in CSS and inherited by both — one of them out of step
and the colour slides off the letters, widening further down.

Neither mode **goes out over the network.** The computation happens on the GPU in the tab and
the values stay there.

## Where to make changes

- **To change an example**, change `assets/examples.js` and nothing else. The landing page's
  links point at the same list through `#example=<id>`.
- **To change an API description**, change the comment in `borch-ts/src/*.ts` and run
  `npm run docs:api`. Editing `assets/api.json` directly loses it at the next extraction, and a
  check stops before that.
- **After adding an example, press the button and run it.** Writing six lessons surfaced four
  places — `max()`, `t()`, `at(null, 2)`, `load()` — and writing six tutorials surfaced
  `sub_(tensor)` turning quietly into NaN, and `probe` being an injected name. All of it came
  out only from pressing the button.
- **Re-emitting a page with a generation script overwrites what was fixed by hand.** The `probe`
  fix really did disappear that way once — after re-emitting, press and check.
- **When changing a number**, do not change the "measured" section alone; write down where the
  number was measured. Every number quoted on these pages has its grounds in `README.md` or
  `BORCH-TS.md`.
- **Do not write an absent feature as though it were there.** The "what exists now · what does
  not yet" table is that promise, and Models, Hub and the WASM fallback are written as absent.

## Who edits this folder

`site/` belongs to whichever session is holding this site. Another session sometimes needs to
change a **number** written here (a golden count, an artifact size), and the order matters then.

**Commit the origin first.** If cases were added, land the cases; if source was added, land the
source. The site is then briefly **stale**, and `tests/test_docs.py`'s failure message describes
exactly that situation. The site's line follows afterwards.

**Not the other way round.** Writing a number into the site before it exists makes the site
**lie**, and the check is equally red — this time not because it is stale but because it is
wrong. Those are different: a stale number gets corrected eventually, while a wrong number gets
quoted by whoever read it.

**Pressing the button on someone's behalf is a one-time thing.** A session that cannot make a
change because it is outside its write scope can ask the session holding this folder. But if
that repeats, it is not a favour — it is **a scope decision for a person to make.** As the
number of people pressing on others' behalf grows, nobody knows any more who approved what.
(This paragraph is the result of receiving that request twice and checking with the user.)

## Publishing it

`.github/workflows/pages.yml` puts it on GitHub Pages. **It does not run automatically** — it is
`workflow_dispatch` only. This repository is private and going public is a decision a person
makes, so hanging it on push would make the next commit a publication (the same judgement that
kept `gpu.yml` off a self-hosted runner).

The workflow uploads **the repository shape as it is** and puts a one-line page at the root that
sends you to `/site/`. The pages call `../borch-ts/dist` and `../../vendor/pyodide` by relative
path, so putting `site/` alone at the root sends those paths outside the root. The collected
result is 31MB, and a rehearsal confirmed Python mode runs inside it.

They are static files, so hosting them yourself works. Two things to keep:

- `borch-ts/dist` has to go up with it — it is gitignored and so is not in the commit.
- `site/assets/api.json` is committed, because a static deploy cannot run the generator.
  `tests/test_site.py` checks it has not gone stale instead.
- To keep Python mode alive, `vendor/pyodide/` (26MB) and `borch/` and `borch_webgpu/` go too.
  Pyodide is MPL-2.0, so the way to obtain its source has to be on the page, and that wording is
  under the playground.
