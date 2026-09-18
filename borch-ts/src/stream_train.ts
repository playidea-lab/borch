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
import { Module } from "./nn.js";
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
 * window holds at most one block's worth at a time in either pass. `opts.lossParams` are
 * resident trainable tensors used **inside `loss`** rather than in a streamed block — a new
 * classification head, say — whose gradients are harvested from the loss's own backward.
 */
export async function streamTrainStep(
  win: Window,
  input: Tensor,
  blocks: readonly TrainBlock[],
  loss: (output: Tensor) => Tensor,
  opts: { lossParams?: readonly Tensor[] } = {},
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
    // Resident trainable params used inside the loss (a new head) get their gradient here — the
    // loss's backward already computed it; extract and accumulate it before the scope closes.
    for (const p of opts.lossParams ?? []) {
      const gp = grads.get(p);
      if (gp !== undefined) p.grad = p.grad === null ? s.keep(gp) : s.keep(p.grad.add(gp));
    }
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

/** Set a field (a parameter or a buffer) by its dotted name — the owner's field assignment,
 *  which is what `namedBuffers`/`namedParameters` read back. */
function setField(root: Module, dotted: string, value: Tensor): void {
  const cut = dotted.lastIndexOf(".");
  const owner = root.getSubmodule(cut < 0 ? "" : dotted.slice(0, cut));
  const leaf = cut < 0 ? dotted : dotted.slice(cut + 1);
  (owner as unknown as Record<string, Tensor>)[leaf] = value;
}

/** Which of a block's frozen buffers stream — a window slot is a weight operand only, so the
 *  default is buffers with rank ≥ 2 (conv kernels, linear weights). Batch-norm running stats
 *  (rank 1) stay resident, as a frozen eval reads them generically. */
export type BufferSelect = (shape: readonly number[]) => boolean;

const isWeightOperandBuffer: BufferSelect = (shape) => shape.length >= 2;

/**
 * Turns a real (LoRA-adapted, frozen-base) `Module` into a {@link TrainBlock}: its frozen weight
 * buffers stream, its trainable parameters (the adapters) are harvested for gradients, and the
 * forward swaps the windowed buffers into the module's fields, runs `module.forward`, and
 * restores. The bridge from `applyLora`'d modules to {@link streamTrainStep}.
 */
export async function trainBlock(module: Module, opts: { select?: BufferSelect } = {}): Promise<TrainBlock> {
  const select = opts.select ?? isWeightOperandBuffer;
  const buffers = module.namedBuffers();
  const names = Object.keys(buffers).filter((n) => {
    const b = buffers[n];
    return b !== undefined && select(b.shape);
  });
  const weights: Float32Array[] = [];
  const shapes: number[][] = [];
  for (const name of names) {
    const b = buffers[name] as Tensor;
    // eslint-disable-next-line no-await-in-loop
    weights.push(await b.toArray());
    shapes.push([...b.shape]);
  }
  const params = module.parameters();
  return {
    weights,
    shapes,
    params,
    run(h: Tensor, windowed: readonly Tensor[]): Tensor {
      const saved = names.map((n) => module.getBuffer(n));
      names.forEach((n, i) => setField(module, n, windowed[i] as Tensor));
      const y = module.forward(h);
      names.forEach((n, i) => setField(module, n, saved[i] as Tensor));
      return y;
    },
  };
}

/**
 * One training step over a sequence of real modules, streaming each module's frozen weight
 * buffers through `win` — the model-facing form of {@link streamTrainStep}. The caller passes
 * `applyLora`'d modules (frozen base + resident adapter); this fills each adapter's `.grad`.
 */
export async function streamTrainSequence(
  win: Window,
  input: Tensor,
  modules: readonly Module[],
  loss: (output: Tensor) => Tensor,
  opts: { select?: BufferSelect; lossParams?: readonly Tensor[] } = {},
): Promise<Tensor> {
  const selectOpt = opts.select ? { select: opts.select } : {};
  const blocks = await Promise.all(modules.map((m) => trainBlock(m, selectOpt)));
  return streamTrainStep(win, input, blocks, loss, opts.lossParams ? { lossParams: opts.lossParams } : {});
}
