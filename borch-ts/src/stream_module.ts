/**
 * **A real module turned into a `StreamBlock`** — `docs/SCALE.md` Step 3 ④, the adapter.
 *
 * `streamSequential` (`src/stream.ts`) streams a list of `StreamBlock`s — each a plain bag
 * of weight bytes plus a forward closure — through a window with bounded residency. That
 * interface is deliberately free of `nn`: the scheduler does not know what a module is. This
 * file is the bridge. Given a real `Module` (a bimm `BasicBlock`, a `Sequential` stage), it
 * snapshots the module's frozen weight kernels as host bytes and builds the closure that, on
 * each call, swaps the **windowed** tensors into the module's own parameter fields, runs the
 * genuine `forward`, and restores the resident tensors so the model stays runnable.
 *
 * **Only weight operands may be windowed.** A window slot lives at an offset inside one
 * shared buffer and is bound as a slice — which is exactly how matmul's `mat2` and conv's
 * kernel are read, and *not* how a generic op reads a tensor (batch-norm's scale, a bias
 * add, an embedding gather all read from offset 0). So the default selects a module's 4-D
 * kernels (conv weights); everything else — biases, batch-norm affine, running statistics —
 * stays resident, which is what a frozen eval pass wants anyway. A caller streaming Linear
 * `weight`s (2-D, read as `mat2`) passes a `select` that admits them.
 *
 * The swap-and-restore keeps this a **non-consuming** pass: the module is unchanged after it
 * runs, so the same model can be streamed again and a streamed result compared against the
 * resident one.
 */

import { noGrad, Tensor } from "./tensor.js";
import { Module } from "./nn.js";
import { streamSequential, type StreamBlock } from "./stream.js";
import type { Window } from "./device.js";

/** Which of a module's parameters go through the window. A window slot is a weight operand
 *  only, so the default admits 4-D kernels (conv weights) and leaves the rest resident. */
export type ParamSelect = (name: string, param: Tensor) => boolean;

const isConvKernel: ParamSelect = (_name, p) => p.shape.length === 4;

export interface StreamBlockOptions {
  /** The parameters to stream; the rest stay resident. Defaults to 4-D conv kernels. */
  readonly select?: ParamSelect;
}

/** Set a parameter by its dotted name — `getSubmodule` reaches the owner, and assigning the
 *  field *is* the registration (`namedParameters` reads the owner's own fields). */
function setParameter(root: Module, dotted: string, value: Tensor): void {
  const cut = dotted.lastIndexOf(".");
  const owner = root.getSubmodule(cut < 0 ? "" : dotted.slice(0, cut));
  const leaf = cut < 0 ? dotted : dotted.slice(cut + 1);
  (owner as unknown as Record<string, Tensor>)[leaf] = value;
}

/**
 * Turns one `Module` into a `StreamBlock`: its selected frozen weights become the streamed
 * bytes (in `namedParameters` order), and the forward swaps the windowed tensors into the
 * module's fields, runs `module.forward`, and restores. `async` because reading the weight
 * bytes off the device is async.
 */
export async function streamBlock(module: Module, opts: StreamBlockOptions = {}): Promise<StreamBlock> {
  const select = opts.select ?? isConvKernel;
  const named = module.namedParameters();
  const names = Object.keys(named).filter((n) => {
    const p = named[n];
    return p !== undefined && select(n, p);
  });
  const weights: Float32Array[] = [];
  const shapes: number[][] = [];
  for (const name of names) {
    const p = named[name] as Tensor;
    // eslint-disable-next-line no-await-in-loop
    weights.push(await p.toArray());
    shapes.push([...p.shape]);
  }
  return {
    weights,
    shapes,
    run(h: Tensor, windowed: readonly Tensor[]): Tensor {
      const saved = names.map((n) => module.getParameter(n));
      names.forEach((n, i) => setParameter(module, n, windowed[i] as Tensor));
      const y = noGrad(() => module.forward(h));
      names.forEach((n, i) => setParameter(module, n, saved[i] as Tensor));
      return y;
    },
  };
}

/**
 * Streams a sequence of modules through `win`, in order, over `input`. Each module's selected
 * frozen weights are placed just before it runs and evicted just after, so residency stays at
 * a few modules' worth of weights regardless of the sequence length. `f16` stores the streamed
 * weights half-precision (unpacked on use); `select` chooses which parameters stream.
 */
export async function streamSequence(
  win: Window,
  input: Tensor,
  modules: readonly Module[],
  opts: { f16?: boolean; select?: ParamSelect } = {},
): Promise<Tensor> {
  const blockOpts: StreamBlockOptions = opts.select ? { select: opts.select } : {};
  const blocks = await Promise.all(modules.map((m) => streamBlock(m, blockOpts)));
  return streamSequential(win, input, blocks, opts.f16 !== undefined ? { f16: opts.f16 } : {});
}
