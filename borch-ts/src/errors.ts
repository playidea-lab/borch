/**
 * torch 모양의 예외.
 *
 * ## 왜 이름을 흉내 내는가
 *
 * 이 라이브러리를 쓰는 사람은 torch 를 쓰던 사람이다. 막혔을 때 하는 일은 오류
 * 문구를 그대로 검색하는 것이고, 그때 나오는 것은 torch 의 문서와 질문들이다.
 * 우리 문구만 있으면 그 검색이 통하지 않는다 — 답이 세상에 이미 있는데 못 닿는다.
 *
 * 그래서 **torch 가 같은 조건에서 내는 문구를 그대로 담고**, 우리 설명을 뒤에 붙인다.
 * 앞쪽이 검색용이고 뒤쪽이 이 자리의 사정이다.
 *
 * ## 어디까지 흉내 내는가
 *
 * torch 에 대응하는 실패가 있는 자리만이다. 우리에게만 있는 실패(WGSL 컴파일,
 * dispatch 한계, 아직 없는 기능)는 우리 말로 적는다 — 없는 torch 문구를 지어내면
 * 검색이 통하는 것이 아니라 엉뚱한 곳으로 간다.
 */

/**
 * Present in Python, absent in JavaScript. The golden cases froze the
 * **kind name** of the exception as well (`RuntimeError|message=True`), so
 * the class name is part of the answer.
 */
export class RuntimeError extends Error {
  override readonly name = "RuntimeError";
}

/**
 * An index past the end.
 *
 * **Not a `RuntimeError`.** torch uses Python's `IndexError` here, and the
 * golden cases froze the kind of exception too, so it has to be that name.
 * Blurring the kinds means code catching `except IndexError` stops catching
 * — the kind of an exception is part of the API.
 */
export class IndexError extends Error {
  override readonly name = "IndexError";
}

/**
 * A combination that does not exist yet.
 *
 * **Refusals have kinds too.** torch raises "this padding parity at this
 * rank is not supported" as `NotImplementedError`, and that says something
 * different from "the value is wrong" (`RuntimeError`) — the first is
 * something that may exist one day, the second is the caller being wrong.
 */
export class NotImplementedError extends Error {
  override readonly name = "NotImplementedError";
}

/**
 * An argument that makes no sense on its own — two mutually exclusive
 * options given together, or a value outside the domain.
 *
 * **It says something different from `RuntimeError`.** Python uses
 * `ValueError` in this position and so does torch (`FractionalMaxPool2d`
 * stops with that kind when given both a size and a ratio). The golden
 * cases freeze the **kind name** of the exception as the answer, so
 * throwing a different name here leaves the binding nothing to carry
 * across.
 */
export class ValueError extends Error {
  override readonly name = "ValueError";
}

/**
 * Linear algebra with no answer to give — a singular matrix, a Cholesky
 * that is not positive definite.
 *
 * **The name does work.** Code that can meet a singular matrix usually
 * wraps the call in `except LinAlgError`, and throwing a plain
 * `RuntimeError` walks straight past that wrapper and kills the program.
 * Same reason `IndexError` is kept separate — **the kind of an exception is
 * part of the API.**
 */
export class LinAlgError extends Error {
  override readonly name = "LinAlgError";
}

/**
 * The wording torch itself produces. Kept verbatim so that searching for it
 * works.
 */
export const TORCH = {
  matmulShape: "shapes cannot be multiplied",
  // **이 축소판에 칸이 없는 것.** torch 의 문구가 아니라 우리 문구인데, 여기 두는
  // 까닭은 같다 — 세 구현이 같은 문장으로 멈추고, 골든이 그 **조각**을 찾는다.
  // 자리마다 손으로 적으면 구현마다 다른 문장이 생기고, 배우는 사람은 그것을 다른
  // 규칙으로 읽는다.
  //
  // **이 두 줄이 한국어였고, 그래서 스물한 케이스가 거짓으로 초록이었다.** 파이썬
  // 쪽(`borch/_base.py` 의 `_unsupported`)이 영어로 옮겨졌는데 이쪽은 안 옮겨졌다.
  // 골든이 굳혀 둔 답은 `기대대로` 라는 **판정 낱말**이고, 케이스는 각자 자기 쪽
  // 문장 조각이 들어 있는지만 본다 — 그래서 두 문장이 아무리 벌어져도 양쪽이 각자
  // 자기와 일치하며 스물한 건이 전부 통과한다. 상호 대조가 아니라 자기 대조였다.
  //
  // 문장은 `_base.py` 에서 **그대로 옮겨 적었다.** 유도하면 또 갈린다.
  // 앞에 공백을 넣지 않는다 — 부르는 자리(`tensor.ts`)가 이미 한 칸 띄우고 있어서
  // 넣으면 두 칸이 된다. 파이썬은 `f"{what} is not in the browser subset."` 한 칸이다.
  absent: "is not in the browser subset.",
  absentAdvice: "Use real PyTorch on your own machine (`uv add torch`) — this subset "
    + "is for practising the syntax, and imitating what is missing teaches the wrong thing.",
  broadcast: "must match the size of tensor",
  reshapeSize: "is invalid for input of size",
  nonScalarBackward: "grad can be implicitly created only for scalar outputs",
  noGrad: "does not require grad",
  // torch 는 "a Tensor with 3 elements cannot be converted to Scalar" 라고 낸다.
  // 골든이 찾는 조각은 뒤쪽이므로 개수는 자리마다 채워 넣는다.
  itemScalar: "cannot be converted to Scalar",
  // torch 는 "... but found at least two devices, cuda:0 and cpu!" 로 이어 붙인다.
  // 장치 이름은 자리마다 다르므로 앞머리만 여기 둔다.
  crossDevice: "Expected all tensors to be on the same device",
  // 비스칼라 역방향에 어긋난 씨앗을 넘겼을 때. torch 는 뒤에 두 모양을 이어 붙인다.
  gradShape: "Mismatch in shape",
  secondBackward:
    "Trying to backward through the graph a second time (or directly access " +
    "saved tensors after they have already been freed)",
} as const;
