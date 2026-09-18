/**
 * **Streaming a sequential stack through a frozen-weight window** — `docs/SCALE.md` Step 3
 * ④. Each block's frozen weights are placed in the window just before the block runs and
 * evicted just after, so only a few blocks are resident at once and a model larger than the
 * window fits. The block's own intermediates are freed in a scope, keeping only the output
 * that flows to the next block.
 *
 * This is the scheduler abstraction: a caller (the bimm adapter, the workbench feature pass)
 * turns a real model into a list of `StreamBlock`s — each carrying its weight bytes and a
 * forward closure — and this runs them with bounded memory. The weights are frozen
 * (`requiresGrad` false), so this is a no-grad feature/inference pass; a trainable adapter
 * on top (LoRA, Step 7) stays resident and is passed into the closure separately.
 *
 * Prefetch-one-ahead (placing block k+1 while k runs) is a later optimisation: the window's
 * single staging buffer serialises fills today, so blocks stream one at a time. What this
 * establishes is correctness and bounded residency; overlap is a speed lever for when the
 * staging ring exists.
 */

import { Window } from "./device.js";
import { scope, Tensor } from "./tensor.js";

/** One block of a streamed stack: its frozen weights (in the order `run` consumes them),
 *  their shapes, and the forward that takes the previous output and the windowed weights. */
export interface StreamBlock {
  readonly weights: readonly Float32Array[];
  readonly shapes: readonly (readonly number[])[];
  run(h: Tensor, weights: readonly Tensor[]): Tensor;
}

/**
 * Runs `blocks` in order over `input`, streaming each block's weights through `win`. Returns
 * the final output, kept alive past the streaming scopes. `f16` stores the weights at half
 * precision (unpacked on use) for half the resident bytes.
 */
export async function streamSequential(
  win: Window,
  input: Tensor,
  blocks: readonly StreamBlock[],
  opts: { f16?: boolean } = {},
): Promise<Tensor> {
  let h = input;
  for (const blk of blocks) {
    // Place this block's weights into the window.
    const placed: { weight: Tensor; evict: () => void }[] = [];
    for (let i = 0; i < blk.weights.length; i++) {
      const data = blk.weights[i];
      const shape = blk.shapes[i];
      if (data === undefined || shape === undefined) continue;
      placed.push(opts.f16
        ? await Tensor.inWindowF16(win, data, shape)
        : await Tensor.inWindow(win, data, shape));
    }
    // Run the block in its own scope, keeping only the output — the block's intermediates
    // go back to the pool at once, so residency stays at one block's activations.
    let out: Tensor | undefined;
    const weights = placed.map((p) => p.weight);
    // eslint-disable-next-line no-await-in-loop
    await scope(async () => { out = blk.run(h, weights); }, () => (out ? [out] : []));
    // The block is done — free its window slots for the next block.
    for (const p of placed) p.evict();
    h = out as Tensor;
  }
  return h;
}
