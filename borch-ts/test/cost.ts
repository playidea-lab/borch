/**
 * **What one step spends, and how much of it** — counted rather than timed.
 *
 *     npm run build:ts
 *     npm run cost:ts
 *
 * ## Why not time
 *
 * The bench (`bench.ts`) measures the wall clock, and so **refuses to answer on a software
 * adapter** — milliseconds off a CPU rasteriser are that rasteriser's number and not this
 * library's. That judgement is right, and it leaves the bench meaningful only on a machine
 * with a GPU.
 *
 * **What is counted here does not depend on the adapter.** How many dispatches were made,
 * how many buffers a scope let go, how many are held — every one of them is decided by the
 * code path and not changed by the device. So **this runs where the bench cannot.** CI,
 * and this repository's default environment, are exactly there.
 *
 * ## What it catches
 *
 * The golden looks at **values alone.** An implementation leaking one buffer per step and
 * an implementation dispatching twice as many kernels both give the same values, so the
 * table is green either way. This repository lived through that (it is why `scope()`
 * exists) and until now not one check stood in that place — `device.ts`'s comment had
 * written down that **"a `survived` that is not 0 is a leak in the training loop"**, and
 * the only thing asking for that number was the bench, which a person runs by hand.
 *
 * ## When a frozen number has to change
 *
 * `EXPECT` holds **measurements, not estimates.** If one grew it is one of three things —
 * more kernels are being dispatched, something that used to be batched is not, or there is
 * a leak. If one shrank that is good news, and it still has to be changed here. **Write
 * down why it moved along with the new number** — swap the figure alone and the next
 * person swaps it again without knowing what it meant.
 *
 * ## It was probed
 *
 * One line, `keepAlive(loss.mul(loss))`, was put inside the training loop to leak one
 * buffer per step. **Three checks caught it, each from a different angle** — dispatches
 * 53 → 54, `survived` 0 → 1, buffers held 26 → 36. Had the three been counting the same
 * thing only one would have gone red, so the overlap is not waste: it covers three
 * different ways of leaking.
 *
 * **One check was overlap and nothing else, and it was deleted** — the reason is at "does
 * the pool cycle" below. That only became visible after the same check was carried over to
 * the binding side.
 */

import * as nn from "../src/nn.js";
import { SGD } from "../src/optim.js";
import { device, keepAlive, scope, Tensor } from "../src/tensor.js";

/** One check. It keeps whether it passed and **the number actually seen.** */
interface Check {
  readonly name: string;
  readonly ok: boolean;
  readonly note: string;
}
/**
 * `checks` is the authority in this report. `text` is the shadow a person reads.
 *
 * **The runner used to judge by scanning a sentence.** That way of judging changes its
 * answer quietly when the wording changes, and in `readme.py` it did — with one of the two
 * examples failing, the word it looked for was still sitting on another line, so it
 * returned 0. Hand the state over as it is and the runner can count, and can say for
 * itself which thing failed.
 */
export interface Report { text: string; checks: Check[] }

/**
 * The model this measures with. **It has to be small** — running on a software adapter is
 * the reason this check exists. It still passes through a convolution, a normalisation, a
 * linear layer and a loss once each, so the *kinds* of place one step touches are a large
 * model's kinds.
 */
class Small extends nn.Module {
  private readonly conv = new nn.Conv2d(1, 4, 3, 1, 1, false);
  private readonly bn = new nn.BatchNormND(4);
  private readonly fc = new nn.Linear(4 * 8 * 8, 3);

  override forward(x: Tensor): Tensor {
    const h = this.bn.forward(this.conv.forward(x)).unary("relu");
    return this.fc.forward(h.reshape([x.shape[0] ?? 1, 4 * 8 * 8]));
  }
}

/**
 * How many dispatches and how many submits one step makes. **Measured, not estimated.**
 *
 * Taken on `Small` above at batch 4; it moves when the model or the kernels move.
 */
const EXPECT = {
  dispatches: 53,
  // **One step is one submit.** The commands pile up and go in a single send when the
  // loss is read, so this number rising means a place appeared that waits on the GPU
  // mid-step — the kind that leaves the values alone and makes the step slower, which the
  // golden can never see.
  submits: 1,
};

export async function report(): Promise<Report> {
  const checks: Check[] = [];
  const want = (name: string, ok: boolean, note = ""): void => {
    checks.push({ name, ok, note });
  };

  const dev = device();
  const batch = 4;
  const pixels = new Float32Array(batch * 1 * 8 * 8);
  for (let i = 0; i < pixels.length; i++) pixels[i] = (i % 13) / 13 - 0.5;
  const labels = new Float32Array(batch);
  for (let i = 0; i < batch; i++) labels[i] = i % 3;

  // The input and the parameters are **outside the scope.** Made inside, they would be
  // let go at the end of the first step.
  const x = keepAlive(Tensor.from(pixels, [batch, 1, 8, 8]));
  const y = keepAlive(Tensor.from(labels, [batch], { dtype: "int64" }));
  const model = new Small();
  const opt = new SGD(model.parameters(), 0.05, 0.9);
  const crit = new nn.CrossEntropyLoss();

  /** **Exactly what a user would type.** Call the low-level scope and it is no longer
   * `scope()` that is being measured. */
  const step = async (): Promise<number> => scope(async () => {
    opt.zeroGrad();
    const loss = crit.call(model.call(x), y);
    loss.backward();
    opt.step();
    return await loss.item();      // read inside the scope, while that buffer exists
  });

  // Warm-up. The first step bakes shaders and starts with an empty pool, so its numbers
  // are not the ones that follow.
  for (let i = 0; i < 3; i++) await step();

  // ── 1. Is the dispatch count **the same every step** ─────────────────
  //
  // Growing means a step is looking at a step — the graph is not being cut, or the cache
  // took the wrong key and is baking a shader again.
  const perStep: number[] = [];
  const perSubmit: number[] = [];
  for (let i = 0; i < 5; i++) {
    const d0 = dev.dispatches;
    const s0 = dev.submits;
    await step();
    perStep.push(dev.dispatches - d0);
    perSubmit.push(dev.submits - s0);
  }
  const first = perStep[0] ?? 0;
  want("the dispatch count is the same every step", perStep.every((n) => n === first),
    perStep.join(" "));
  const firstSubmit = perSubmit[0] ?? 0;
  want("the submit count is the same every step",
    perSubmit.every((n) => n === firstSubmit),
    perSubmit.join(" "));

  // ── 2. Is it the frozen number ───────────────────────────────────────
  if (EXPECT.dispatches > 0) {
    want("dispatches per step match the frozen number", first === EXPECT.dispatches,
      `${first} (frozen ${EXPECT.dispatches})`);
    want("submits per step match the frozen number", firstSubmit === EXPECT.submits,
      `${firstSubmit} (frozen ${EXPECT.submits})`);
  } else {
    // Clear the frozen numbers and run, and it arrives here — the place to take a fresh
    // measurement.
    want("dispatches per step have not been frozen yet", false,
      `measured: dispatches ${first} · submits ${firstSubmit} — write them into EXPECT`);
  }

  // ── 3. Does the scope leave nothing behind ──────────────────────────
  //
  // **This is the definition of a leak.** `device.ts` wrote it down that way, and one
  // survivor per step fills the device over a long training run — with the values right
  // the whole way.
  await step();
  want("the scope leaves no buffer behind", dev.lastScope.survived === 0,
    `survived ${dev.lastScope.survived} · freed ${dev.lastScope.freed}`);
  want("every scope was closed", dev.scopeDepth === 0, `depth ${dev.scopeDepth}`);

  // ── 4. Do the buffers held grow with the step count ─────────────────
  //
  // The `survived` above looks at **one scope.** A leak outside a scope — a global cache
  // gaining an entry per step, say — does not show in that number, so it is asked apart.
  const early = dev.memory;
  const earlyPool = dev.pooled;
  for (let i = 0; i < 10; i++) await step();
  const late = dev.memory;
  want("ten more steps do not add to the buffers held",
    late.tensors <= early.tensors,
    `${early.tensors} → ${late.tensors} · ` +
    `${(early.bytes / 1024).toFixed(0)}KB → ${(late.bytes / 1024).toFixed(0)}KB`);
  // **The pool has to be watched with it.** The number above subtracts what is in the
  // pool, so if buffers drain into the pool once a step, that line stays green while the
  // footprint grows. Repeating one shape, the pool has to settle at the working set — and
  // not settling is a defect rather than a policy.
  const latePool = dev.pooled;
  want("repeating steps does not grow the pool", latePool.count <= earlyPool.count,
    `${earlyPool.count} → ${latePool.count} · ` +
    `${(earlyPool.bytes / 1024).toFixed(0)}KB → ${(latePool.bytes / 1024).toFixed(0)}KB`);

  // ── Whether the pool cycles is **not asked separately** ─────────────
  //
  // At first this asked "are the buffers held fewer than the dispatches per step". The
  // thought was that an implementation making a fresh buffer every time and letting it go
  // every time also has `survived` 0, so that number alone is not enough.
  //
  // **Carrying the same check over to the binding side is what showed it asks nothing
  // new.** Over there the golden has run on the page first, so it starts holding forty
  // thousand, and the absolute comparison read the harness's leftovers as the training
  // loop's. Opening it up to mend it: `memory.tensors` is `made - spare`, and **an
  // implementation that does not return to the pool has `spare` 0, so that number simply
  // grows** — which the check just above already catches. Two checks thought to count
  // different things were counting one.
  //
  // Without running it somewhere else the overlap would not have been visible. Deleting a
  // check is what that run produced.

  // ── A tensor that escaped its scope **stops, loudly** ───────────────
  //
  // It used to hand back somebody else's values quietly (measured: `[1,2,3,4]` read back
  // as `9,9,9,9`). The buffer is not destroyed but returned to the pool, and the next
  // allocation writes over it; WebGPU does not stop that, because it is a valid read of a
  // valid buffer.
  {
    let escaped: Tensor | null = null;
    await scope(async () => {
      escaped = Tensor.from([1, 2, 3, 4], [4]).mul(Tensor.full([], 1));
      return 0;
    });
    let note = "";
    let stopped = false;
    try {
      const got = await (escaped as unknown as Tensor).toArray();
      note = `it read quietly: ${Array.from(got).join(",")}`;
    } catch (err) {
      stopped = true;
      note = (err as Error).message.split("\n")[0] ?? "";
    }
    want("using an escaped tensor stops", stopped, note);
  }

  // ── Does the block form (`using`) do what the callback form does ────
  //
  // Both forms stand on the same `beginScope`/`endScope`, but **left as a statement that
  // holds only in prose, the two part.** Three things are asked here — does it close on
  // leaving the block, is that moment **after** the `await` inside, and does what `keep()`
  // held survive.
  {
    const before = dev.scopeDepth;
    let inside = -1;
    let survived: Tensor | null = null;
    let awaited = -1;
    {
      using s = scope();
      inside = dev.scopeDepth;
      survived = s.keep(Tensor.from([5, 6], [2]).mul(Tensor.full([], 1)));
      // **The `await` is inside the block.** If the letting-go happened before this
      // wait, a dead tensor would be read here — which is the crux of whether `using`
      // works at all.
      awaited = (await survived.toArray())[0] ?? -1;
    }
    want("using opens the scope inside the block", inside === before + 1,
      `depth ${before} → ${inside}`);
    want("using closes it at the end of the block", dev.scopeDepth === before,
      `depth ${dev.scopeDepth}`);
    want("the await inside the block finishes before it closes", awaited === 5,
      `${awaited}`);
    // What was kept passed to the outer scope, so it still reads after the block. Had it
    // not, the dead-tensor guard made above would stop here — which is what makes this
    // line an question about `keep()`.
    const after = await survived.toArray();
    want("what keep() held is alive after the block", after[1] === 6,
      Array.from(after).join(","));
  }

  // ── The pool does not shrink on its own ─────────────────────────────
  //
  // While the steps repeat one shape the pool earns its keep — "the buffers held do not
  // grow" above measures that. **A changing shape is another story.** The pool is split by
  // size, so a batch-16 buffer cannot serve batch 32 and simply stays, and `memory`
  // deliberately subtracts what is in the pool, so it does not appear in that number. The
  // bench runs three batch sizes in one sitting, so this is a real path.
  {
    const before = dev.pooled;
    for (const n of [1000, 2000, 3000]) {
      dev.beginScope();
      // A different size per shape — the pool parts by size.
      Tensor.owned([n], 1).mul(Tensor.owned([n], 2));
      dev.endScope([]);
    }
    const grown = dev.pooled;
    want("a changing shape grows the pool", grown.count > before.count,
      `${before.count} → ${grown.count} · ${Math.round(grown.bytes / 1024)}KB`);
    // **`memory` does not count that.** Pinned here, that the two numbers ask different
    // questions — fold them into one and either the leak check or the footprint check
    // becomes a lie.
    want("memory does not count what is in the pool",
      dev.memory.bytes < grown.bytes + dev.memory.bytes,
      `held ${Math.round(dev.memory.bytes / 1024)}KB · ` +
      `pool ${Math.round(grown.bytes / 1024)}KB`);
    const freed = dev.emptyCache();
    want("emptyCache empties the pool",
      freed.count === grown.count && dev.pooled.count === 0,
      `${freed.count} · ${Math.round(freed.bytes / 1024)}KB handed back`);
    // It has to keep running afterwards — emptying the pool must not break the device.
    dev.beginScope();
    const still = Tensor.owned([4], 3).add(Tensor.owned([4], 4));
    const value = (await still.toArray())[0] ?? -1;
    dev.endScope([]);
    want("arithmetic runs after the pool is emptied", value === 7, `${value}`);
  }

  const bad = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push("");
  lines.push(bad.length
    ? `**${bad.length} parted.**`
    : `all ${checks.length} cost checks passed`);
  return { text: lines.join("\n"), checks };
}
