/**
 * 대괄호 자리 — `x[...]` 를 옮겨 적는 문법.
 *
 * ## 왜 필요한가
 *
 * torch 코드에서 대괄호는 어디에나 나오는데 **자바스크립트는 `[]` 를 오버로드할 수
 * 없다.** 그래서 여기서는 줄마다 다른 메서드가 된다 — `select`·`narrow`·
 * `indexSelect`·`gather`·`maskedSelect`… 열다섯 개다. 값은 전부 맞지만, 옮겨 적는
 * 사람이 **줄마다 "이건 어느 메서드지" 를 골라야 한다.** `x[1:3]` 이 `narrow` 인지
 * `select` 인지, 인자가 `(dim, start, length)` 인지 `(dim, start, end)` 인지.
 *
 * `at()` 은 그 열다섯 갈래를 문 하나로 좁힌다. 없애는 것이 아니라 **문을 하나 더
 * 내는 것**이고, 기존 메서드는 그대로 있다.
 *
 * ## 슬라이스가 왜 함수인가
 *
 * `x.at([1, 3])` 이 "축 0 은 1, 축 1 은 3" 인지 "1:3 을 자른다" 인지 **배열만으로는
 * 구분이 안 된다.** 파이썬은 문법이 갈라 주지만(`x[1, 3]` 대 `x[1:3]`) 여기서는
 * 둘 다 배열이다.
 *
 * 그래서 슬라이스를 이름 있는 것으로 만든다. 파이썬의 `x[1:3]` 도 실은
 * `x[slice(1, 3)]` 로 풀리므로 **같은 이름을 쓴다** — 새로 배울 것이 아니라 원래
 * 그 자리에 있던 이름이다.
 *
 * ```ts
 * x.at(0)                     // x[0]
 * x.at([null, 1])             // x[:, 1]        null 이 파이썬의 `:` 다
 * x.at(slice(1, 3))           // x[1:3]
 * x.at([0, slice(1, 3)])      // x[0, 1:3]
 * x.at(slice(null, null, 2))  // x[::2]
 * x.at([[0, 2]])              // x[[0, 2]]      대괄호 둘 — numpy 와 같은 모양이다
 * ```
 *
 * **맨 바깥 배열은 언제나 축 목록이다.** 번호표로 고르려면 한 겹 더 싼다 —
 * numpy 의 `x[0, 1]` 과 `x[[0, 1]]` 이 갈리는 자리와 같은 모양이다.
 */

import { RuntimeError } from "./errors.js";

/** `x[start:end:step]`. 직접 만들지 말고 `slice()` 를 쓴다. */
export interface Slice {
  readonly kind: "slice";
  readonly start: number | null;
  readonly end: number | null;
  readonly step: number;
}

/**
 * 자를 구간. 비운 자리는 파이썬처럼 끝까지다.
 *
 * **걸음이 음수면 거절한다.** 파이썬의 `x[::-1]` 은 뒤집기인데, 여기에는 그 일을
 * 하는 `flip` 이 따로 있다. 두 문으로 같은 일을 하게 두면 어느 쪽이 정본인지가
 * 흐려지고, 그것은 이 저장소가 반복해서 피해 온 자리다.
 */
export function slice(
  start: number | null = null,
  end: number | null = null,
  step = 1,
): Slice {
  if (!Number.isInteger(step) || step < 1) {
    throw new RuntimeError(
      `slice step must be a positive integer: ${step}` +
        (step < 0 ? " — use flip() to reverse." : ""),
    );
  }
  return { kind: "slice", start, end, step };
}

export function isSlice(v: unknown): v is Slice {
  return typeof v === "object" && v !== null
    && (v as Slice).kind === "slice";
}

/**
 * 한 축을 어떻게 볼 것인가 — **값이 아니라 계획이다.**
 *
 * 파싱을 실행에서 떼어 두면 이 파일이 `Tensor` 를 안 부른다(순환이 없다). 그리고
 * 범위·차례를 여기서 다 따지므로, 실행하는 쪽은 이미 맞는 수만 받는다.
 */
export type AxisPlan =
  | { kind: "int"; at: number }
  | { kind: "range"; start: number; length: number }
  | { kind: "picks"; indices: number[] }
  | { kind: "whole" };

/**
 * 파이썬의 슬라이스 셈. 음수는 뒤에서 세고, 넘치면 잘라 맞춘다.
 *
 * **여기서 던지지 않는다** — 파이썬도 `x[5:99]` 를 빈 것으로 준다. 범위를 넘긴 것이
 * 오류인 곳은 정수 인덱스 쪽이고 그것은 따로 본다.
 */
function resolveSlice(s: Slice, size: number): AxisPlan {
  const clamp = (v: number): number => Math.min(Math.max(v, 0), size);
  const from = clamp(s.start === null ? 0 : s.start < 0 ? s.start + size : s.start);
  const to = clamp(s.end === null ? size : s.end < 0 ? s.end + size : s.end);
  if (s.step === 1) {
    return { kind: "range", start: from, length: Math.max(0, to - from) };
  }
  const indices: number[] = [];
  for (let i = from; i < to; i += s.step) indices.push(i);
  return { kind: "picks", indices };
}

/**
 * 축 하나짜리 인덱스를 계획으로.
 *
 * @param axis 몇 번째 축인지. 오류 문구에만 쓴다 — 어느 자리가 틀렸는지 안 적으면
 *   랭크가 넷쯤 될 때 사람이 세어 봐야 한다.
 */
export function planAxis(
  index: number | null | Slice | readonly number[],
  size: number,
  axis: number,
): AxisPlan {
  if (index === null) return { kind: "whole" };
  if (typeof index === "number") {
    if (!Number.isInteger(index)) {
      throw new RuntimeError(`index for dimension ${axis} is not an integer: ${index}`);
    }
    const at = index < 0 ? index + size : index;
    if (at < 0 || at >= size) {
      throw new RuntimeError(
        `index ${index} is out of bounds for dimension ${axis} with size ${size}`,
      );
    }
    return { kind: "int", at };
  }
  if (isSlice(index)) return resolveSlice(index, size);
  if (Array.isArray(index)) {
    const picks = index.map((v) => {
      const at = v < 0 ? v + size : v;
      if (!Number.isInteger(v) || at < 0 || at >= size) {
        throw new RuntimeError(
          `index ${v} is out of bounds for dimension ${axis} with size ${size}`,
        );
      }
      return at;
    });
    return { kind: "picks", indices: picks };
  }
  throw new RuntimeError(`cannot read the index for dimension ${axis}: ${String(index)}`);
}
