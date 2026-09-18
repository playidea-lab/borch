/**
 * **Training an adapter on a streamed frozen backbone** — `docs/SCALE.md` Step 7, the residency
 * rule. A block's frozen weights are resident only while that block runs, in both passes:
 *
 *   - **forward**: place block k's frozen weights in the window, run it, keep only the output
 *     (the boundary activation), evict — so at most one block's weights and the boundary
 *     activations are resident, never the whole backbone. The block's own intermediates are
 *     dropped, exactly as gradient checkpointing drops them.
 *   - **backward**: walk the blocks in reverse; refill block k's weights, recompute its forward
 *     *with the tape on* from the saved boundary input, flow the incoming gradient through that
 *     fresh graph into the resident adapters and into the block input, evict. This is QLoRA's
 *     per-layer dequant in both passes, with "dequant" replaced by "refill".
 *
 * ## Why this is not `checkpoint()`
 *
 * `checkpoint` (`checkpoint.ts`) recomputes inside the autograd backward, which is
 * **synchronous** — but refilling the window is `async` (a staging-buffer map). So the reverse
 * pass is written out here, awaiting each refill, rather than hung off `Tensor.backward()`. The
 * gradient arithmetic is the same `flow` the tape uses; only the driving loop differs.
 *
 * ## What is resident
 *
 * The frozen base weights stream (one block at a time in the window); the adapters stay resident
 * and trainable; the boundary activations — one tensor per block edge — stay resident so the
 * recompute has its input. That is the checkpointing memory profile, with the weights bounded by
 * the window on top. The base weights never form a gradient (`requiresGrad` is false), so nothing
 * is spent taping them.
 */

import { enableGrad, flow } from "./autograd.js";
import { noGrad, scope, Tensor } from "./tensor.js";
import type { Window } from "./device.js";

/** One block of a streamed training stack: its frozen weight bytes (streamed through the
 *  window), their shapes, the resident trainable parameters whose gradients this fills (the
 *  block's adapters), and a forward that consumes the previous output and the windowed frozen
 *  weights — reading its adapters from its own closure. */
export interface TrainBlock {
  readonly weights: readonly Float32Array[];
  readonly shapes: readonly (readonly number[])[];
  readonly params: readonly Tensor[];
  run(h: Tensor, windowed: readonly Tensor[]): Tensor;
}

/** Place a block's frozen weights into the window, returning the windowed tensors and one
 *  eviction that frees every slot the block took. */
async function placeBlock(win: Window, blk: TrainBlock): Promise<{ weights: Tensor[]; evict: () => void }> {
  const placed: { weight: Tensor; evict: () => void }[] = [];
  for (let i = 0; i < blk.weights.length; i++) {
    const data = blk.weights[i];
    const shape = blk.shapes[i];
    if (data === undefined || shape === undefined) continue;
    // eslint-disable-next-line no-await-in-loop
    placed.push(await Tensor.inWindow(win, data, shape));
  }
  return {
    weights: placed.map((p) => p.weight),
    evict: () => { for (const p of placed) p.evict(); },
  };
}

/**
 * One training step over a streamed frozen backbone. Runs `blocks` in order over `input` with
 * the residency rule above, evaluates `loss` on the final output, and fills each block's
 * `params[*].grad` — accumulating, as `Tensor.backward` does, so a caller zeroes grads between
 * steps. Returns the scalar loss (kept alive past the internal scopes); the caller reads it and
 * then steps its optimiser over the adapters.
 *
 * `loss` must return a scalar (a `[]`-shaped tensor). The frozen weights are streamed, so the
 * window holds at most one block's worth at a time in either pass.
 */
export async function streamTrainStep(
  win: Window,
  input: Tensor,
  blocks: readonly TrainBlock[],
  loss: (output: Tensor) => Tensor,
): Promise<Tensor> {
  // ── Forward: stream, keeping only the boundary activation between blocks. ──
  // boundaries[k] is the input to block k; boundaries[n] is the final output.
  const boundaries: Tensor[] = [input];
  let h = input;
  for (const blk of blocks) {
    // eslint-disable-next-line no-await-in-loop
    const placed = await placeBlock(win, blk);
    let out: Tensor | undefined;
    // Keep only the block's output; its intermediates go back to the pool at the scope's close.
    {
      using s = scope();
      out = s.keep(noGrad(() => blk.run(h, placed.weights)));
    }
    placed.evict();
    boundaries.push(out);
    h = out;
  }

  // ── The loss, on a detached leaf, taped so its gradient to the output is exact. ──
  const finalOut = boundaries[boundaries.length - 1] as Tensor;
  const hLeaf = finalOut.detach();
  hLeaf.requiresGrad = true;
  let lossVal: Tensor | undefined;
  let g: Tensor;
  {
    using s = scope();
    const l = enableGrad(() => loss(hLeaf));
    const seed = Tensor.full(l.shape, 1);
    const grads = flow([l], [seed], (a, b) => a.add(b));
    g = s.keep(grads.get(hLeaf) ?? Tensor.zeros(hLeaf.shape));
    lossVal = s.keep(l);
  }

  // ── Backward: refill each block in reverse, recompute with the tape on, flow the gradient
  //    into the adapters (resident) and into the block input (passed to the previous block). ──
  for (let k = blocks.length - 1; k >= 0; k--) {
    const blk = blocks[k] as TrainBlock;
    const xk = (boundaries[k] as Tensor).detach();
    xk.requiresGrad = true;
    // eslint-disable-next-line no-await-in-loop
    const placed = await placeBlock(win, blk);
    let gradX: Tensor;
    {
      using s = scope();
      const y = enableGrad(() => blk.run(xk, placed.weights));
      const grads = flow([y], [g], (a, b) => a.add(b));
      // Each adapter is used only in its own block, so its whole gradient is here. Accumulate
      // into `.grad` the way `backward` does, keeping the result past the scope.
      for (const p of blk.params) {
        const gp = grads.get(p);
        if (gp === undefined) continue;
        p.grad = p.grad === null ? s.keep(gp) : s.keep(p.grad.add(gp));
      }
      gradX = s.keep(grads.get(xk) ?? Tensor.zeros(xk.shape));
    }
    placed.evict();
    g = gradX;
  }

  return lossVal as Tensor;
}
