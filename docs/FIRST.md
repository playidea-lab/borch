# The first time a user meets it

> 2026-09-21. The performance programme closed with the table ahead on every card
> (`docs/BOOK.md`); what is left to make better is the moment a person first runs this
> library — on the machine they have, with the code a tutorial gives them. Three things
> are ugly there today, and this plan takes them in order. Predictions are written before
> the numbers; the ledger (§4) says which were wrong.

## 0. What a first visit pays today

Measured 2026-09-21 (`capture:ts`, the ResNet-18 training step and the fused inference
forward, `Compiled.firstCall`):

| first call of a shape | apple / metal-3 | RTX 5080 (Vulkan) | RTX 5050 Laptop (D3D12) |
|---|---|---|---|
| recording | 3 ms | 3 | 19–27 |
| the step's own first run | 49 | 139 | 389–663 |
| tuning, of which the compile wave | 56 / 30 | 188 / 118 | 2,961 / **2,590** |
| inference forward, tuning / compile | 46 / 23 | 213 / 110 | 3,175 / **2,773** |
| a replay afterwards | 16.8 | 8.3 | 30–35 |

(The laptop's rows are from an afternoon its owner was using it; quiet, its compile
wave is 536 ms and the inference forward's first call 1.4 s — 1a.) On D3D12 a first
visit compiled for seconds before the first answer, once per shape per adapter, cached
after (`localStorage`). And a page that calls the model eagerly
— every tutorial's first page — does not see the table's numbers: on the 5080 the eager
fused forward at batch 1 is 3.28 ms against the captured 0.56.

## 1. The first load · size M

- **1a — measure the compile wave by pipeline.** `Device.tuneCompiles`: each pipeline the
  warm wave made, with its wall time and WGSL size; `capture:ts` prints the dearest.
  *Predict*: the staged convolution's three tile variants are more than half of the wave
  on every card; DXC's time scales with the WGSL's size.
- **1b — take the tuning off the first call.** The first call records with the rule's
  picks, runs, and answers; the candidates compile and time afterwards, in idle time
  (`requestIdleCallback`, or the next call's tail), and a decision that changed re-records
  a pure step before its next call. *Gate*: the D3D12 first call within the rule's own
  first run + 100 ms; the answers bit for bit what they are today; `capture:ts` 23 / 23,
  with the tuner's checks now waiting for the deferred pass.
- **1c — the second session's wave.** Whether Chrome keeps the compiled pipelines across
  sessions on D3D12 (Dawn has a pipeline cache; whether it hits through a fresh page is
  unmeasured). The runners open a fresh profile every time, so this needs a
  `--profile=<dir>` option and two runs. *Predict*: the second session's wave is under a
  quarter of the first's on D3D12 and unchanged on Metal. If so, 1b is enough and the
  candidate count stays; if not, the variants are trimmed to what the tuner has ever
  taken.

## 2. The eager path · size M

- **2a — measure the eager / captured gap.** The 5080's 3.28 against 0.56 at batch 1:
  where the 2.7 ms goes — JavaScript per dispatch, bind-group creation, the allocator,
  submit count. *Predict*: bind groups and the allocator, not the kernels (the GPU time
  is the same 0.9 ms either way).
- **2b — the lever, chosen after 2a.** Candidates: a bind-group cache keyed by pipeline
  and buffers; the intrinsic fusions applied on the eager path as they are on the
  captured one; and, in the book, `compiled` on the first page of every tutorial. *Gate*:
  eager within 1.5× captured on the 5080 at batch 1.

## 3. The traps, named by the library · size S

Five things a learner's code does that make it silently slow or wrong, and one sentence
the library says first: an inference loop without a `scope`; an inference forward under
gradient mode; a shape off the multiple of eight the subgroup kernels want; a laptop
on battery (the afternoon runs of the ledger); the first call's wait. *Gate*: each of the
five scenarios prints its sentence once, and none of them prints on correct code.

## 4. Ledger

(entries follow, newest last)
- **2026-09-21, 1c — the second session (0c13d4f).** `capture:ts --profile=<dir>`
  (`BORCH_CHROME_PROFILE`, a persistent context) twice on the laptop: **the compile wave
  517 → 51 ms, the inference forward's first call 158 → 15 ms** — Chrome keeps D3D12's
  compiled pipelines in the profile, and the prediction ("under a quarter") held at a
  tenth. The tuning ran again (850 → 326 ms, 888 → 547) because the runner serves each
  run on a fresh port and `localStorage` is per origin; a page at one address keeps its
  decisions, and its second visit pays the 51 and nothing else. So the candidate count
  stays: what a first visit on D3D12 pays after its first answer is 0.85 s once per
  origin, of which a third is the timing itself on a card whose replay is 27 ms.

