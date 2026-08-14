/**
 * dtype 과 승격 규칙.
 *
 * ## 저장은 float32 하나다
 *
 * `int64` 는 **이름표**다. WGSL 에 64비트 정수가 없으므로 진짜로 담을 방법이 없고,
 * 32비트 정수로 담으면 그건 `int64` 가 아니라 `int32` 다. 그래서 값은 float32 에
 * 두고 dtype 만 따로 든다 — 자매가 같은 선택을 했고, 자매 쪽 이유(TF.js 의 cast 가
 * 깨져서)와 달리 여기 이유는 표현할 수단 자체가 없다는 것이다.
 *
 * **대가를 적어둔다**: 2²⁴ 를 넘는 정수는 float32 에 정확히 안 담긴다. 색인이나
 * 개수로 쓰는 범위에서는 한참 아래지만, 큰 정수를 담으면 조용히 어긋난다.
 *
 * ## 왜 범주로 가르는가
 *
 * torch 는 `bool < 정수 < 실수` 로 범주를 두고 **그 안에서만** 올린다. numpy 규칙을
 * 물려받으면 `float32 + int64` 가 float64 가 되는데, 그러면 배우는 사람이 틀린 규칙을
 * 배운다. 여기 폭이 범주마다 하나뿐이라 규칙이 더 짧아진다.
 */

import { RuntimeError } from "./errors.js";

export type DType = "float32" | "int64" | "bool";

/** 낮을수록 아래. 승격은 높은 쪽으로만 간다. */
const RANK: Readonly<Record<DType, number>> = { bool: 0, int64: 1, float32: 2 };

const BY_RANK: readonly DType[] = ["bool", "int64", "float32"];

export function dtypeName(d: DType): string {
  return `torch.${d}`;
}

/** 파이썬 쪽 스칼라에 해당하는 형. JS 에는 정수·실수 구분이 없어 값으로 가른다. */
export function scalarDType(v: number | boolean): DType {
  if (typeof v === "boolean") return "bool";
  return Number.isInteger(v) ? "int64" : "float32";
}

/**
 * 이항 연산의 결과 형.
 *
 * 세 규칙뿐이다.
 * 1. **나눗셈은 언제나 실수다.** `int64 / int64` 도 float32 다 — torch 의 `/` 는
 *    참나눗셈이고, 정수 나눗셈을 원하면 `//` 를 따로 부른다.
 * 2. **뺄셈에 bool 이 끼면 거절한다.** 참에서 거짓을 빼는 것이 무슨 뜻인지 정해진
 *    답이 없어서, torch 는 답을 지어내는 대신 멈춘다.
 * 3. 나머지는 높은 범주로 올린다.
 */
export function promote(a: DType, b: DType, op: "+" | "-" | "*" | "/"): DType {
  if (op === "-" && (a === "bool" || b === "bool")) throw BoolSubtraction();
  if (op === "/") return "float32";
  return (RANK[a] >= RANK[b] ? a : b);
}

/** 순위로 형을 되찾는다. 축약처럼 범주를 한 칸 올릴 때 쓴다. */
export function byRank(rank: number): DType {
  return BY_RANK[Math.min(Math.max(rank, 0), BY_RANK.length - 1)] ?? "float32";
}

export function rankOf(d: DType): number {
  return RANK[d];
}

/**
 * 뺄셈에 bool 이 낀 경우.
 *
 * torch 가 내는 문구를 그대로 담는다 — 막힌 사람이 검색할 것이 그 문구다.
 * **`RuntimeError` 를 그대로 쓴다**: 골든이 예외의 종류 이름을 굳혔고, 여기서 전용
 * 클래스를 세우면 그 이름이 갈린다.
 */
function BoolSubtraction(): RuntimeError {
  return new RuntimeError(
    "Subtraction, the `-` operator, with a bool tensor is not supported. " +
      "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.",
  );
}
