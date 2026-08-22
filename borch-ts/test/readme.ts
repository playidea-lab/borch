/**
 * The examples written in the README and in `index.ts`, **run exactly as written.**
 *
 * Code in a document rots unless it runs — a renamed name, a changed argument order, one
 * missing `await`, and nobody says a word until the first user is stuck on it. This
 * repository has twice caught the README's install instructions not actually working
 * (`1b5a1e9`, `e41c043`).
 *
 * So what is here has to be **the original of the example, not a copy of it.** Change the
 * document and this file changes with it. Whether the values are right is not asked — the
 * golden does that. One thing is asked here: **does it run when typed exactly as
 * written.**
 */

import { init, keepAlive, nn, optim, scope, Tensor } from "../src/index.js";

interface Check { name: string; ok: boolean; note: string }

/**
 * `checks` is the authority in this report. `text` is the shadow a person reads.
 *
 * **This file is where that difference showed itself most expensively.** `readme.py`
 * judged a pass by `"그대로 돌고" in text`, and that phrase sat in the success sentence of
 * **both** examples. So with the first example failing and only LBFGS passing, the phrase
 * was still there and the runner returned 0 — leaving an example whose loss does not go
 * down in the documentation.
 */
export interface Report { text: string; checks: Check[] }

export async function report(): Promise<Report> {
  await init();

  const model = new nn.Sequential(
    new nn.Linear(784, 128), new nn.ReLU(), new nn.Linear(128, 10));
  const opt = new optim.SGD(model.parameters(), 0.05, 0.9);
  const crit = new nn.CrossEntropyLoss();

  // Where the example's `pixels` and `labels` sit. The values do not matter; the shapes
  // do.
  const pixels = new Float32Array(32 * 784);
  for (let i = 0; i < pixels.length; i++) pixels[i] = (i % 17) / 17 - 0.5;
  const labels = new Float32Array(32);
  for (let i = 0; i < labels.length; i++) labels[i] = i % 10;

  const x = keepAlive(Tensor.from(pixels, [32, 784]));
  const y = keepAlive(Tensor.from(labels, [32], { dtype: "int64" }));

  const seen: number[] = [];
  for (let i = 0; i < 3; i++) {
    await scope(async () => {
      opt.zeroGrad();
      const loss = crit.call(model.call(x), y);
      loss.backward();
      opt.step();
      seen.push(await loss.item());
    });
  }

  // **The loss has to go down.** Running and learning are different, and what the example
  // shows is the second — three steps giving the same number make the example a lie.
  const [first, last] = [seen[0] ?? NaN, seen[seen.length - 1] ?? NaN];
  const ok = Number.isFinite(first) && Number.isFinite(last) && last < first;

  // ── The README's `LBFGS` example ────────────────────────────────────
  //
  // **The rule this file's opening paragraph sets is one I broke.** It says *change the
  // document and this file changes with it*, and the LBFGS example went into the README
  // without this file being touched. So that example went into the documentation having
  // never once been run, and **it was wrong** — `new LBFGS([p], …)` optimised one
  // parameter while the loss was computed from `model`. Where `p` is not that model's, the
  // step moves nothing.
  //
  // It is the kind that raises nothing, so reading does not show it. Running does.
  const lb = new nn.Sequential(
    new nn.Linear(784, 128), new nn.ReLU(), new nn.Linear(128, 10));
  const lbOpt = new optim.LBFGS(lb.parameters(), 0.1);
  const before = await scope(async () => await crit.call(lb.call(x), y).item());
  await scope(async () => {
    await lbOpt.step(() => {
      lbOpt.zeroGrad();
      const loss = crit.call(lb.call(x), y);
      loss.backward();
      return loss;
    });
  });
  const after = await scope(async () => await crit.call(lb.call(x), y).item());
  const lbOk = Number.isFinite(before) && Number.isFinite(after) && after < before;

  const checks: Check[] = [
    {
      name: "the README example runs as written and the loss goes down",
      ok,
      note: `loss ${seen.map((v) => v.toFixed(4)).join(" → ")}`,
    },
    {
      name: "the README LBFGS example lowers the loss in one step",
      ok: lbOk,
      note: `loss ${before.toFixed(4)} → ${after.toFixed(4)}`,
    },
  ];
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  const failed = checks.filter((c) => !c.ok);
  lines.push(failed.length === 0
    ? `all ${checks.length} README examples passed`
    : `**${failed.length} failed** / ${checks.length}`);
  return { text: lines.join("\n"), checks };
}
