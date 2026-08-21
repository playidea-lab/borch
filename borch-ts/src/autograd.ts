/**
 * 자동 미분 테이프.
 *
 * 코어(`borch`)·자매(`borch_webgpu`)와 **같은 모양**이다. 셋이 같으면
 * 한 곳에서 고친 것을 다른 곳으로 옮기기 쉽고, 이번 프로젝트에서 그 값어치를
 * 여러 번 봤다 — 자매의 `roll`·`masked_select` 가 값만 맞고 그래프가 끊긴 것을
 * 코어의 같은 자리를 보고 찾았다.
 *
 * 그래프 자료구조는 텐서를 모른다. `T` 로 두었기에 `tensor.ts` 를 import 하지 않고,
 * 그래서 테이프만 따로 시험할 수 있다.
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
   * Accumulates on leaves only, as in torch.
   */
  grad: T | null;
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

/** 위상 정렬. 부모가 항상 자식보다 뒤에 오도록 늘어놓는다. */
function topoOrder<T>(root: Node<T>): Node<T>[] {
  const order: Node<T>[] = [];
  const seen = new Set<Node<T>>();
  // 재귀로 쓰면 깊은 그래프(ResNet-18 은 마디 수백 개)에서 스택이 넘는다.
  // 두 상태를 쓰는 명시적 스택이라 깊이에 안 걸린다.
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
 */
export function backward<T>(
  root: Node<T>,
  seed: T,
  add: (a: T, b: T) => T,
  options: { retainGraph?: boolean; onSecondPass?: () => never } = {},
): void {
  if (!root.requiresGrad) {
    throw new Error(
      "backward() was called on a tensor that does not require grad. " +
        "(It was made under no_grad, or it passed through an operation that breaks the graph.)",
    );
  }
  const grads = new Map<Node<T>, T>();
  grads.set(root, seed);
  for (const node of topoOrder(root)) {
    const g = grads.get(node);
    if (g === undefined) continue;
    if (!node.backwardFn || node.parents.length === 0) {
      // 잎이다. torch 와 같이 여기에만 쌓는다.
      node.grad = node.grad === null ? g : add(node.grad, g);
      continue;
    }
    if (node.freed && options.onSecondPass) options.onSecondPass();
    if (!options.retainGraph) node.freed = true;
    const parts = node.backwardFn(g);
    if (parts.length !== node.parents.length) {
      // 조용히 틀리는 대신 멈춘다. 부모 하나를 빠뜨린 미분식은 값이 그럴듯해서
      // 골든을 통과하고, 학습이 안 되는 것으로만 드러난다.
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
