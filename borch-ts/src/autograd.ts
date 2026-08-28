/**
 * The automatic differentiation tape.
 *
 * **The same shape** as the core (`borch`) and the sister library (`borch_webgpu`). With
 * all three alike, a fix made in one place moves easily to the others, and this project
 * has seen that pay off several times — the sister library's `roll` and `masked_select`
 * having right values and a severed graph was found by looking at the same place in the
 * core.
 *
 * The graph data structure does not know about tensors. Left as `T`, it imports no
 * `tensor.ts`, and so the tape can be tested on its own.
 */

/**
 * A node in the graph. Tensor implements it.
 */
export interface Node<T> {
  /**
   * The inputs this value was made from. Empty at a leaf.
   */
  readonly parents: readonly Node<T>[];
  /**
   * Whether it receives gradient.
   */
  requiresGrad: boolean;
  /**
   * Accumulates on leaves, and on a derived node that asked to keep its gradient
   * — `retainGrad()`, or being named in `backward(…, inputs)`.
   */
  grad: T | null;
  /**
   * Set by `retainGrad()`. A derived node does not keep its gradient otherwise:
   * the intermediate values are the bulk of a training loop's memory, and torch
   * makes keeping one an explicit request for the same reason.
   */
  retainsGrad?: boolean;
  /**
   * Takes one output gradient and returns **as many as there are parents.**
   *
   * A parent that gets nothing is `null`. `null` and "filled with zeros"
   * are different — step functions (`sign`, `floor`) **do pass zero
   * through.** The sister project wrote this once as a refusal, and a loss
   * with a step in it ran under torch while stopping only for us.
   */
  readonly backwardFn: ((grad: T) => readonly (T | null)[]) | null;
  /**
   * torch's `grad_fn` name. Error messages and the golden `repr::` cases
   * use it.
   */
  readonly gradName: string;
  /**
   * Whether backward has already passed through this node.
   *
   * torch releases the graph once gradient has flowed, because the
   * intermediate values are holding memory. A second call is a refusal, not
   * a recomputation.
   */
  freed: boolean;
}

/**
 * Keeps the switch that turns gradient on and off **inside an object.**
 *
 * It must not be a module-level variable. On the Python side, splitting a
 * single file into eight broke `no_grad` quietly for exactly this reason —
 * each module got its own copy, and all 374 golden cases passed straight
 * through it. Keeping the value in one place stops that from happening.
 */
export const gradMode = { enabled: true };

/**
 * `torch.no_grad()`. Restores even if an exception is thrown.
 */
export function noGrad<R>(body: () => R): R {
  const before = gradMode.enabled;
  gradMode.enabled = false;
  try {
    return body();
  } finally {
    gradMode.enabled = before;
  }
}

/** A topological sort. It lays them out so a parent always comes after its child. */
function topoOrder<T>(root: Node<T>): Node<T>[] {
  const order: Node<T>[] = [];
  const seen = new Set<Node<T>>();
  // Written recursively, a deep graph (ResNet-18 has hundreds of nodes) overflows the
  // stack. This is an explicit stack with two states, so depth does not bind.
  const stack: { node: Node<T>; expanded: boolean }[] = [{ node: root, expanded: false }];
  while (stack.length > 0) {
    const frame = stack.pop();
    if (!frame) break;
    if (frame.expanded) {
      order.push(frame.node);
      continue;
    }
    if (seen.has(frame.node)) continue;
    seen.add(frame.node);
    stack.push({ node: frame.node, expanded: true });
    for (const parent of frame.node.parents) {
      if (!seen.has(parent)) stack.push({ node: parent, expanded: false });
    }
  }
  order.reverse();
  return order;
}

/**
 * Backpropagation. Seeds at `root` and accumulates into the leaves' `grad`.
 *
 * @param add how two gradients are added. A value used in several places is
 *   added that many times.
 * @param options.only torch's `backward(…, inputs)`: when given, **only** these
 *   nodes are written to. Every other leaf keeps whatever `grad` it had, which
 *   is how one branch is differentiated without disturbing the rest.
 *
 *   The two rules do not cancel. A node in `only` is written to whether it is a
 *   leaf or not, and a node that called `retainGrad()` is written to even when
 *   it is outside `only` — measured against torch, which fills a retained
 *   intermediate's `grad` while `inputs` names something else entirely.
 */
export function backward<T>(
  root: Node<T>,
  seed: T,
  add: (a: T, b: T) => T,
  options: {
    retainGraph?: boolean;
    onSecondPass?: () => never;
    only?: ReadonlySet<Node<T>>;
  } = {},
): void {
  if (!root.requiresGrad) {
    throw new Error(
      "backward() was called on a tensor that does not require grad. " +
        "(It was made under no_grad, or it passed through an operation that breaks the graph.)",
    );
  }
  const { only } = options;
  const keep = (node: Node<T>): boolean => only === undefined || only.has(node);
  const grads = new Map<Node<T>, T>();
  grads.set(root, seed);
  for (const node of topoOrder(root)) {
    const g = grads.get(node);
    if (g === undefined) continue;
    if (!node.backwardFn || node.parents.length === 0) {
      // A leaf. As in torch, it accumulates here — unless `inputs` named others.
      if (keep(node)) node.grad = node.grad === null ? g : add(node.grad, g);
      continue;
    }
    // A derived node keeps its gradient only when asked: `retainGrad()`, or
    // being named in `inputs`.
    if (node.retainsGrad || (only !== undefined && only.has(node))) {
      node.grad = node.grad === null ? g : add(node.grad, g);
    }
    if (node.freed && options.onSecondPass) options.onSecondPass();
    if (!options.retainGraph) node.freed = true;
    const parts = node.backwardFn(g);
    if (parts.length !== node.parents.length) {
      // It stops rather than being quietly wrong. A derivative that omits one parent has
      // plausible values, passes the golden, and shows up only as a failure to train.
      throw new Error(
        `${node.gradName}: backward returned ${parts.length} gradients, but the node ` +
          `has ${node.parents.length} parents.`,
      );
    }
    for (const [i, part] of parts.entries()) {
      const parent = node.parents[i];
      if (!parent || part === null || !parent.requiresGrad) continue;
      const had = grads.get(parent);
      grads.set(parent, had === undefined ? part : add(had, part));
    }
  }
}
