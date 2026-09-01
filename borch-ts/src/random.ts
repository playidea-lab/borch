/**
 * One generator. **Layer initialisation and the tensor factories draw from one stream.**
 *
 * What is here used to be trapped inside `nn.ts`. That left nowhere to build
 * `Tensor.randn` — `tensor.ts` calling `nn.ts` would be a cycle — and so no way to port
 * the first line of a torch example, `torch.randn(...)`. This file calls nothing, so both
 * sides can call it.
 *
 * **One seed has to reset everything.** torch's `manual_seed` does, and the first two
 * places somebody expecting that looks are layer initialisation and dropout. So this file
 * also holds the slot (`onSeed`) where anything with state to reset registers itself.
 *
 * **It does not produce torch's numbers.** It cannot, and it must not pretend to. The
 * golden does not ask about initial values and always plants the weights, so there is
 * nothing here to diverge.
 */

/** xorshift32. Unseeded, it starts from the golden-ratio constant. */
const rng = { state: 0x9e3779b9 };

/** What has to be reset along with the seed. Dropout's counter is the first
 *  registrant. */
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
 * `generator=` — **carried in order to refuse it**, wherever torch takes one.
 *
 * There is one stream in this file and a `Generator` is a second one. torch's
 * argument names a stream to draw from; here there is only the stream above, and
 * the honest answer is to say so rather than to accept the object and draw from
 * the global one anyway.
 *
 * **Accepting it is the failure this exists to stop, and it is silent twice
 * over.** JavaScript discards a surplus argument without a word, so
 * `x.bernoulli_(0.5, g)` ran and ignored `g`; and what it produces is a *random
 * number*, so nothing downstream looks wrong. Somebody running five seeds to
 * measure variance would get five streams that were never separate.
 *
 * `null` and `undefined` both pass: torch's default is `None`, and the Python
 * binding cannot send `undefined`.
 */
export function refuseGenerator(who: string, generator: unknown): void {
  if (generator !== undefined && generator !== null) {
    throw new Error(
      `${who}(generator=…) is not in the browser subset — there is one stream here, `
      + "and `manualSeed` rewinds it. A Generator is a second stream, and drawing "
      + "from the global one while holding your object would be worse than stopping.");
  }
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
  // `1 - uniform()` gives `(0, 1]` — a 0 arriving makes log -∞.
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
