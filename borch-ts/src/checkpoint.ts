/**
 * **Gradient checkpointing** — `torch.utils.checkpoint.checkpoint`.
 *
 * A segment run under `checkpoint` keeps **only its output** through the forward, not the
 * intermediates a backward would need; at the backward it runs the segment again, with the
 * tape on, and flows the incoming gradient through that fresh graph. The trade is the
 * plan's Step 6 (`docs/SCALE.md`): O(√n) activation memory for one extra forward, so a
 * deeper model — or a streamed backbone whose block must be resident only while it runs —
 * fits where the full tape would not.
 *
 * ## Why it lives here and not in a caller
 *
 * The pieces it needs are not exported: `flow` (the backward walk), `enableGrad`, and the
 * node funnel. So the primitive is written once, beside them, and callers get a function.
 *
 * ## What makes it correct rather than a silently-wrong gradient
 *
 * - **The forward holds nothing.** It runs under `noGrad` inside a `scope()` that keeps
 *   only the output; every intermediate is returned to the pool at the scope's close, so
 *   the tape carries none of them. That is the whole memory win.
 * - **The recompute is a fresh graph on detached leaves.** The inputs are `detach()`ed
 *   (sharing their buffers, no copy) and marked `requiresGrad` exactly where the originals
 *   were, so `flow` stops at them and records their gradient. Fresh tensors each time means
 *   the version guard (`savedVersions`) has nothing stale to fire on.
 * - **Gradient goes only to the arguments, and a captured one is refused, not dropped.**
 *   `checkpoint` routes gradient to the tensors passed as `inputs` — those are the node's
 *   parents. A tensor that requires grad and is used inside `fn` but was *closed over*
 *   rather than passed would get no gradient at all, silently; so the recompute's graph is
 *   walked and any grad-requiring leaf that is not one of the inputs makes the backward
 *   throw, naming the count. Pass every trainable tensor the segment differentiates (a
 *   LoRA adapter, say) as an argument; a frozen backbone does not require grad and is not
 *   flagged. This is the difference between the plan's Step 6 and a silently wrong gradient.
 * - **The recompute frees itself.** It runs inside its own `scope()` that keeps only the
 *   gradients it returns; the recomputed intermediates and the accumulation graph go back
 *   to the pool at once. So only one segment's worth is resident at a time in the backward,
 *   which is the other half of the win.
 * - **The node carries no saved tensors.** `checkpoint` saves nothing itself (that is the
 *   point), so it never refuses a later in-place edit for a value it is not holding.
 *
 * `checkpoint` is verified against the plain (fully-taped) backward in
 * `tests/browser/checkpoint_probe.py` — the gradients must agree within the golden
 * tolerance, and `cost.ts` holds that the buffers at the backward peak actually fall.
 */

import { enableGrad, flow } from "./autograd.js";
import { makeNode, noGrad, scope, Tensor } from "./tensor.js";

/**
 * How many grad-requiring **leaves** the recompute graph rooted at `root` has that are not
 * in `known` (the detached inputs). A leaf is a tensor with no backward — a parameter or an
 * input. One that is not `known` was closed over by `fn` rather than passed, so its
 * gradient would go nowhere; the caller turns a non-zero count into a throw. Iterative, so a
 * deep block does not overflow the stack.
 */
function strayGradLeaves(root: Tensor, known: ReadonlySet<Tensor>): number {
  const seen = new Set<Tensor>();
  const stack: Tensor[] = [root];
  let stray = 0;
  while (stack.length > 0) {
    const n = stack.pop() as Tensor;
    if (seen.has(n)) continue;
    seen.add(n);
    if (!n.backwardFn || n.parents.length === 0) {
      if (n.requiresGrad && !known.has(n)) stray += 1;
      continue;
    }
    for (const p of n.parents) stack.push(p);
  }
  return stray;
}

/**
 * Runs `fn(...inputs)` so that its intermediates are recomputed in the backward rather
 * than held. The returned tensor behaves as if `fn` had been called normally — the same
 * value, and a backward that reaches every input that required grad — but the forward
 * keeps only the output.
 *
 * `fn` must be a pure function of its tensor arguments: it is called a second time in the
 * backward, so anything it reads that is not an argument (a captured tensor, a module
 * parameter) must not have changed in between. A frozen backbone block is exactly this.
 */
export function checkpoint(
  fn: (...xs: Tensor[]) => Tensor,
  ...inputs: Tensor[]
): Tensor {
  // The forward, tape off, nothing but the output surviving the scope.
  let out: Tensor;
  {
    using s = scope();
    out = s.keep(noGrad(() => fn(...inputs)));
  }

  const backwardFn = (grad: Tensor): readonly (Tensor | null)[] => {
    using s = scope();
    // Detached leaves that require grad exactly where the originals did, so the recomputed
    // graph is rooted at them and `flow` records each one's gradient.
    const leaves = inputs.map((x) => {
      const leaf = x.detach();
      leaf.requiresGrad = x.requiresGrad;
      return leaf;
    });
    const y = enableGrad(() => fn(...leaves));
    // A grad-requiring tensor used inside `fn` but not passed in would get no gradient.
    // Refuse loudly rather than return a silent zero for it.
    const stray = strayGradLeaves(y, new Set(leaves));
    if (stray > 0) {
      throw new Error(
        `checkpoint: the function differentiates ${stray} tensor(s) that were not passed ` +
          "to it. A tensor that requires grad and is closed over — a parameter, an adapter " +
          "— gets no gradient through a checkpoint; pass every such tensor as an argument.",
      );
    }
    const grads = flow([y], [grad], (a, b) => a.add(b));
    const parts = inputs.map((x, i) =>
      x.requiresGrad ? (grads.get(leaves[i] as Tensor) ?? null) : null);
    // Keep only the gradients that leave here; the recompute's own buffers go back now.
    for (const p of parts) if (p) s.keep(p);
    return parts;
  };

  // `makeNode` returns a bare tensor when the tape is off or no input needs grad, so the
  // guard is the funnel's and not restated here. `out` and this node share the one buffer.
  return makeNode(out.raw, out.shape, inputs, backwardFn, "CheckpointBackward0", out.dtype);
}
