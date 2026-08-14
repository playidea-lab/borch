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

/** 그래프의 마디. 텐서가 이것을 구현한다. */
export interface Node<T> {
  /** 이 값을 만든 입력들. 잎이면 비어 있다. */
  readonly parents: readonly Node<T>[];
  /** 기울기를 받을 것인가. */
  requiresGrad: boolean;
  /** 잎에만 쌓인다. torch 와 같다. */
  grad: T | null;
  /**
   * 출력 기울기 하나를 받아 **부모 수만큼** 돌려준다.
   *
   * 흘리지 않는 부모 자리는 `null` 이다. `null` 과 "0 으로 채운 것" 은 다르다 —
   * 계단 함수(`sign`·`floor`)는 **0 을 흘린다**. 자매에서 이것을 한 번 거절로
   * 잘못 적었고, 계단을 낀 손실이 torch 에서는 도는데 우리에게서만 멈췄다.
   */
  readonly backwardFn: ((grad: T) => readonly (T | null)[]) | null;
  /** torch 의 `grad_fn` 이름. 오류 메시지와 골든의 `repr::` 케이스가 쓴다. */
  readonly gradName: string;
  /**
   * 역전파가 이미 지나간 마디인가.
   *
   * torch 는 한 번 흘리고 나면 그래프를 놓아준다 — 중간 값들이 메모리를 붙들고
   * 있기 때문이다. 두 번째 호출은 거절이지 다시 계산이 아니다.
   */
  freed: boolean;
}

/**
 * 기울기를 켜고 끄는 스위치를 **객체 안에** 둔다.
 *
 * 모듈 수준 변수로 두면 안 된다. 파이썬 쪽에서 단일 파일을 8개로 쪼갤 때 정확히
 * 이것 때문에 `no_grad` 가 조용히 안 먹었다 — 모듈마다 자기 사본이 생겼고, 골든
 * 374건이 전부 통과하는 채로 지나갔다. 값을 한 군데 두면 그 일이 안 생긴다.
 */
export const gradMode = { enabled: true };

/** `torch.no_grad()`. 예외가 나도 되돌린다. */
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
 * 역전파. `root` 에서 씨앗을 흘려 잎의 `grad` 에 쌓는다.
 *
 * @param add 기울기 둘을 더하는 법. 한 값이 여러 곳에 쓰였으면 그만큼 더해진다.
 */
export function backward<T>(
  root: Node<T>,
  seed: T,
  add: (a: T, b: T) => T,
  options: { retainGraph?: boolean; onSecondPass?: () => never } = {},
): void {
  if (!root.requiresGrad) {
    throw new Error(
      "requiresGrad 가 아닌 텐서에서 backward 를 불렀다. " +
        "(연산 중 no_grad 안이었거나, 흐름을 끊는 연산을 지났다)",
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
        `${node.gradName}: 역방향이 ${parts.length}개를 냈는데 ` +
          `부모는 ${node.parents.length}개다.`,
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
