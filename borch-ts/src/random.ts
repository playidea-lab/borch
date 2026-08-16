/**
 * 난수기 하나. **층 초기화와 텐서 팩토리가 같은 줄기를 쓴다.**
 *
 * 여기 있던 것은 `nn.ts` 안에 갇혀 있었다. 그래서 `Tensor.randn` 을 만들 자리가
 * 없었고 — `tensor.ts` 가 `nn.ts` 를 부르면 순환이 된다 — torch 예제의 첫 줄인
 * `torch.randn(...)` 을 옮길 방법이 없었다. 이 파일은 아무것도 안 부르므로 양쪽이
 * 부를 수 있다.
 *
 * **씨앗 하나가 전부를 되돌려야 한다.** torch 의 `manual_seed` 가 그렇고, 그것을
 * 기대하는 사람이 가장 먼저 확인하는 자리가 층 초기화와 dropout 이다. 그래서 이
 * 파일은 되돌릴 것을 가진 쪽이 등록해 두는 자리(`onSeed`)를 같이 든다.
 *
 * **torch 와 같은 수를 내지 않는다.** 같을 수가 없고 같은 척해서도 안 된다. 골든은
 * 초기값을 안 묻고 가중치를 늘 밖에서 넣으므로 여기서 갈릴 것이 없다.
 */

/** xorshift32. 씨앗을 안 심었을 때의 출발점은 황금비 상수다. */
const rng = { state: 0x9e3779b9 };

/** 씨앗을 심을 때 같이 되돌려야 하는 것들. dropout 계수기가 첫 등록자다. */
const resets: ((seed: number) => void)[] = [];

/**
 * 씨앗을 심고 등록된 것들을 되돌린다.
 *
 * **0 을 1 로 바꾼다.** xorshift 는 상태가 0 이면 영원히 0 을 낸다 — `manualSeed(0)`
 * 이 난수를 죽이는 것은 아무도 예상하지 않는다.
 */
export function manualSeed(value: number): void {
  const seed = (value >>> 0) || 1;
  rng.state = seed;
  for (const reset of resets) reset(seed);
}

/**
 * 씨앗을 심을 때 같이 되돌릴 것을 등록한다. **씨앗 값을 받는다.**
 *
 * 처음에는 인자 없이 불렀고 dropout 계수기는 늘 1 로 되돌아갔다. 같은 씨앗에 같은
 * 결과라는 약속은 지켜지지만 **다른 씨앗에 다른 결과가 안 나온다** — 씨앗을 다섯 개
 * 돌려 분산을 재는 사람은 dropout 마스크가 다섯 번 다 같은 줄 모르고, 가중치
 * 초기화만 흔들린 수를 실험 분산으로 읽는다.
 */
export function onSeed(reset: (seed: number) => void): void {
  resets.push(reset);
}

/** `[0, 1)` 균등분포 하나. */
export function uniform(): number {
  let x = rng.state;
  x ^= x << 13; x >>>= 0;
  x ^= x >> 17;
  x ^= x << 5; x >>>= 0;
  rng.state = x;
  return x / 0x100000000;
}

/**
 * 표준정규분포 하나. Box–Muller 다.
 *
 * **두 값 중 하나만 쓰고 버린다.** 짝을 캐시해 두면 씨앗을 심었을 때 캐시도 같이
 * 비워야 하고, 안 비우면 심기 직전의 값이 심은 뒤에 나온다 — 되돌린 것처럼 보이는데
 * 첫 수만 다른 상태가 되고 그것이 가장 찾기 어려운 종류다.
 */
export function gauss(): number {
  // `1 - uniform()` 으로 `(0, 1]` 을 만든다 — 0 이 들어오면 log 가 -∞ 다.
  const u = 1 - uniform();
  const v = uniform();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** `[-bound, bound]` 균등분포로 채운 배열. 층 초기화가 쓴다. */
export function uniformArray(n: number, bound: number): Float32Array {
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = (uniform() * 2 - 1) * bound;
  return data;
}
