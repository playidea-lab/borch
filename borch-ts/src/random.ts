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
 * Plants the seed and rewinds everything registered with it.
 *
 * **It turns 0 into 1.** xorshift emits zero forever once its state is zero
 * — nobody expects `manualSeed(0)` to be the call that kills randomness.
 */
export function manualSeed(value: number): void {
  const seed = (value >>> 0) || 1;
  rng.state = seed;
  for (const reset of resets) reset(seed);
}

/**
 * Registers something to be rewound when a seed is planted. **It receives
 * the seed value.**
 *
 * It was first called with no argument, and the dropout counter always
 * rewound to 1. The promise that the same seed gives the same result still
 * held, but **different seeds stopped giving different results** — someone
 * running five seeds to measure variance would not know the dropout masks
 * were identical all five times, and would read a number shaken only by
 * weight initialisation as experimental variance.
 */
export function onSeed(reset: (seed: number) => void): void {
  resets.push(reset);
}

/**
 * One sample from the uniform distribution on `[0, 1)`.
 */
export function uniform(): number {
  let x = rng.state;
  x ^= x << 13; x >>>= 0;
  x ^= x >> 17;
  x ^= x << 5; x >>>= 0;
  rng.state = x;
  return x / 0x100000000;
}

/**
 * One sample from the standard normal. Box–Muller.
 *
 * **Of the two values it produces, one is used and one is thrown away.**
 * Caching the pair would mean the cache has to be cleared whenever a seed
 * is planted, and if it is not, a value from just before the planting
 * appears just after it — a state that looks rewound but whose first number
 * is wrong, which is the hardest kind to find.
 */
export function gauss(): number {
  // `1 - uniform()` 으로 `(0, 1]` 을 만든다 — 0 이 들어오면 log 가 -∞ 다.
  const u = 1 - uniform();
  const v = uniform();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * An array filled from the uniform distribution on `[-bound, bound]`. Layer
 * initialisation uses it.
 */
export function uniformArray(n: number, bound: number): Float32Array {
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = (uniform() * 2 - 1) * bound;
  return data;
}
