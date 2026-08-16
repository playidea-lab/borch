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

/**
 * 색인이 범위를 넘은 것.
 *
 * **`RuntimeError` 가 아니다.** torch 는 여기에 파이썬의 `IndexError` 를 쓰고, 골든이
 * 예외의 종류까지 굳혔으므로 그 이름이어야 한다. 종류를 뭉뚱그리면 `except IndexError`
 * 로 잡던 코드가 안 잡힌다 — 예외의 종류도 API 다.
 */
export class IndexError extends Error {
  override readonly name = "IndexError";
}

/**
 * 조합이 아직 없는 것.
 *
 * **거절도 종류가 있다.** torch 는 "이 랭크에 이 패딩 짝수는 안 된다" 를
 * `NotImplementedError` 로 내고, 그것은 "값이 잘못됐다"(`RuntimeError`)와 다른
 * 말이다 — 앞은 언젠가 생길 수 있는 것이고 뒤는 부른 쪽이 틀린 것이다.
 */
export class NotImplementedError extends Error {
  override readonly name = "NotImplementedError";
}

/**
 * 선형대수가 답을 못 내는 것 — 특이행렬, 양정부호가 아닌 콜레스키.
 *
 * **이름이 하는 일이 있다.** 특이행렬을 만날 수 있는 코드는 `except LinAlgError` 로
 * 감싸는 것이 보통인데, 우리가 그냥 `RuntimeError` 를 던지면 그 감싸기를 지나쳐
 * 프로그램이 죽는다. `IndexError` 를 따로 둔 것과 같은 이유다 — **예외의 종류도 API 다.**
 */
export class LinAlgError extends Error {
  override readonly name = "LinAlgError";
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
  // torch 는 "... but found at least two devices, cuda:0 and cpu!" 로 이어 붙인다.
  // 장치 이름은 자리마다 다르므로 앞머리만 여기 둔다.
  crossDevice: "Expected all tensors to be on the same device",
  secondBackward:
    "Trying to backward through the graph a second time (or directly access " +
    "saved tensors after they have already been freed)",
} as const;
