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
 * 파이썬에 있고 JS 에 없는 것. 골든이 예외의 **종류 이름**까지 굳혔으므로
 * (`RuntimeError|문구=True`) 클래스 이름이 그대로 답의 일부가 된다.
 */
export class RuntimeError extends Error {
  override readonly name = "RuntimeError";
}

/** torch 가 내는 문구. 검색이 통하라고 원문 그대로 둔다. */
export const TORCH = {
  matmulShape: "shapes cannot be multiplied",
  broadcast: "must match the size of tensor",
  reshapeSize: "is invalid for input of size",
  nonScalarBackward: "grad can be implicitly created only for scalar outputs",
  noGrad: "does not require grad",
  // torch 는 "a Tensor with 3 elements cannot be converted to Scalar" 라고 낸다.
  // 골든이 찾는 조각은 뒤쪽이므로 개수는 자리마다 채워 넣는다.
  itemScalar: "cannot be converted to Scalar",
  secondBackward:
    "Trying to backward through the graph a second time (or directly access " +
    "saved tensors after they have already been freed)",
} as const;
