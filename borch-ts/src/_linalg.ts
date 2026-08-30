/**
 * Linear algebra on small matrices — **it runs on the CPU.**
 *
 * ## Why the CPU
 *
 * LU, QR and Jacobi rotations are all **sequential.** Each step's result is the next
 * step's input, so there is almost nothing to spread in parallel, and at 2×2 or 3×3 the
 * round trip of launching a kernel and reading the result back costs more than the
 * computation. The sister library reaches the same conclusion and uses numpy.
 *
 * The price is that these operations become **asynchronous**, since a value has to be read
 * back from the GPU. What the backward needs (the inverse, the determinant and so on) is
 * computed in the forward and captured, so `backward()` itself stays synchronous.
 *
 * ## It computes in double precision
 *
 * The input and the output are float32 and the arithmetic is float64. Elimination loses
 * digits easily, and that loss is visible on an ill-conditioned matrix — the core uses
 * float64 in `erf` for the same reason.
 */

/**
 * A square matrix laid out row-major.
 */
export type Mat = Float64Array;

export function fromF32(a: Float32Array): Mat {
  return Float64Array.from(a);
}

export function toF32(a: Mat): Float32Array {
  return Float32Array.from(a);
}

export function matmul(a: Mat, b: Mat, n: number, k: number, m: number): Mat {
  const out = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      const av = a[i * k + t] ?? 0;
      if (av === 0) continue;
      for (let j = 0; j < m; j++) {
        out[i * m + j] = (out[i * m + j] ?? 0) + av * (b[t * m + j] ?? 0);
      }
    }
  }
  return out;
}

export function transpose(a: Mat, n: number, m: number): Mat {
  const out = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) out[j * n + i] = a[i * m + j] ?? 0;
  }
  return out;
}

export interface LU {
  readonly lu: Mat;
  readonly piv: Int32Array;
  /**
   * The sign of the number of row swaps. The determinant's sign comes from
   * here.
   */
  readonly sign: number;
  readonly n: number;
  readonly singular: boolean;
}

/**
 * LU factorisation with partial pivoting.
 *
 * **Pivoting is not optional.** Without it, a matrix whose first element is
 * zero or very small either blows up the division or loses the digits
 * entirely.
 */
export function lu(a: Mat, n: number): LU {
  const m = Float64Array.from(a);
  const piv = new Int32Array(n);
  let sign = 1;
  let singular = false;
  for (let i = 0; i < n; i++) piv[i] = i;
  for (let k = 0; k < n; k++) {
    let best = k;
    let bestAbs = Math.abs(m[k * n + k] ?? 0);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(m[i * n + k] ?? 0);
      if (v > bestAbs) { bestAbs = v; best = i; }
    }
    if (bestAbs === 0) { singular = true; continue; }
    if (best !== k) {
      for (let j = 0; j < n; j++) {
        const t = m[k * n + j] ?? 0;
        m[k * n + j] = m[best * n + j] ?? 0;
        m[best * n + j] = t;
      }
      const tp = piv[k] ?? 0;
      piv[k] = piv[best] ?? 0;
      piv[best] = tp;
      sign = -sign;
    }
    const pivot = m[k * n + k] ?? 1;
    for (let i = k + 1; i < n; i++) {
      const f = (m[i * n + k] ?? 0) / pivot;
      m[i * n + k] = f;
      for (let j = k + 1; j < n; j++) {
        m[i * n + j] = (m[i * n + j] ?? 0) - f * (m[k * n + j] ?? 0);
      }
    }
  }
  return { lu: m, piv, sign, n, singular };
}

export function det(f: LU): number {
  if (f.singular) return 0;
  let d = f.sign;
  for (let i = 0; i < f.n; i++) d *= f.lu[i * f.n + i] ?? 0;
  return d;
}

/**
 * The sign and the log absolute value, separately. Safer than `log(det)`
 * when the determinant is very small.
 */
export function slogdet(f: LU): { sign: number; logabs: number } {
  if (f.singular) return { sign: 0, logabs: -Infinity };
  let sign = f.sign;
  let logabs = 0;
  for (let i = 0; i < f.n; i++) {
    const d = f.lu[i * f.n + i] ?? 0;
    if (d < 0) sign = -sign;
    logabs += Math.log(Math.abs(d));
  }
  return { sign, logabs };
}

/**
 * Solves `A X = B`. `B` has `m` columns.
 */
export function solve(f: LU, b: Mat, m: number): Mat {
  const n = f.n;
  const x = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    const src = f.piv[i] ?? i;
    for (let j = 0; j < m; j++) x[i * m + j] = b[src * m + j] ?? 0;
  }
  // The lower triangle — the diagonal is 1, so there is no division.
  for (let i = 1; i < n; i++) {
    for (let k = 0; k < i; k++) {
      const f2 = f.lu[i * n + k] ?? 0;
      if (f2 === 0) continue;
      for (let j = 0; j < m; j++) {
        x[i * m + j] = (x[i * m + j] ?? 0) - f2 * (x[k * m + j] ?? 0);
      }
    }
  }
  // The upper triangle.
  for (let i = n - 1; i >= 0; i--) {
    for (let k = i + 1; k < n; k++) {
      const f2 = f.lu[i * n + k] ?? 0;
      if (f2 === 0) continue;
      for (let j = 0; j < m; j++) {
        x[i * m + j] = (x[i * m + j] ?? 0) - f2 * (x[k * m + j] ?? 0);
      }
    }
    const d = f.lu[i * n + i] ?? 1;
    for (let j = 0; j < m; j++) x[i * m + j] = (x[i * m + j] ?? 0) / d;
  }
  return x;
}

export function inverse(a: Mat, n: number): Mat {
  const eye = new Float64Array(n * n);
  for (let i = 0; i < n; i++) eye[i * n + i] = 1;
  return solve(lu(a, n), eye, n);
}

/**
 * The lower-triangular Cholesky. Throws if the matrix is not symmetric
 * positive definite — it does not quietly return NaN.
 */
export function cholesky(a: Mat, n: number): Mat {
  const l = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j] ?? 0;
      for (let k = 0; k < j; k++) s -= (l[i * n + k] ?? 0) * (l[j * n + k] ?? 0);
      if (i === j) {
        if (s <= 0) {
          throw new Error(
            "cholesky: the input is not symmetric positive definite (a leading principal minor is not positive)",
          );
        }
        l[i * n + j] = Math.sqrt(s);
      } else {
        l[i * n + j] = s / (l[j * n + j] ?? 1);
      }
    }
  }
  return l;
}

/**
 * Householder QR.
 *
 * **Householder, not Gram–Schmidt.** The two give the same factorisation
 * with different signs. LAPACK uses Householder and torch uses LAPACK, so
 * matching the golden `R` down to its signs requires the same method — the
 * golden cases ask about `Q` in absolute value, but they ask about `R` as
 * it is.
 */
export function qr(a: Mat, n: number, m: number): { q: Mat; r: Mat } {
  const r = Float64Array.from(a);
  const q = new Float64Array(n * n);
  for (let i = 0; i < n; i++) q[i * n + i] = 1;
  const steps = Math.min(n - 1, m);
  for (let k = 0; k < steps; k++) {
    let norm = 0;
    for (let i = k; i < n; i++) norm += (r[i * m + k] ?? 0) ** 2;
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const head = r[k * m + k] ?? 0;
    // Taking the sign to match the head is what keeps the subtraction from losing
    // digits. LAPACK does the same.
    const alpha = head >= 0 ? -norm : norm;
    const v = new Float64Array(n);
    v[k] = head - alpha;
    for (let i = k + 1; i < n; i++) v[i] = r[i * m + k] ?? 0;
    let vv = 0;
    for (let i = k; i < n; i++) vv += (v[i] ?? 0) ** 2;
    if (vv === 0) continue;
    for (let j = 0; j < m; j++) {
      let dot = 0;
      for (let i = k; i < n; i++) dot += (v[i] ?? 0) * (r[i * m + j] ?? 0);
      const f = (2 * dot) / vv;
      for (let i = k; i < n; i++) {
        r[i * m + j] = (r[i * m + j] ?? 0) - f * (v[i] ?? 0);
      }
    }
    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let i = k; i < n; i++) dot += (v[i] ?? 0) * (q[j * n + i] ?? 0);
      const f = (2 * dot) / vv;
      for (let i = k; i < n; i++) {
        q[j * n + i] = (q[j * n + i] ?? 0) - f * (v[i] ?? 0);
      }
    }
  }
  return { q, r };
}

/**
 * Keeps one triangle and mirrors it into the other.
 *
 * **`eigh` accepts matrices that are not symmetric.** By default torch
 * reads only the lower triangle and ignores the upper — `[[4,99],[1,3]]`
 * and `[[4,1],[1,3]]` give the same answer (asked of real torch). Jacobi
 * looks at the whole matrix, so without mirroring first, a non-symmetric
 * input diverges quietly here. **It is a difference that never shows as
 * long as you pass something symmetric.**
 */
export function mirror(a: Mat, n: number, upper: boolean): Mat {
  const out = Float64Array.from(a);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      const keep = upper ? (a[j * n + i] ?? 0) : (a[i * n + j] ?? 0);
      out[i * n + j] = keep;
      out[j * n + i] = keep;
    }
  }
  return out;
}

/**
 * Eigenvalues and eigenvectors of a symmetric matrix — Jacobi rotations.
 *
 * Eigenvalues come back in **ascending** order, as torch's `linalg.eigh`
 * gives them.
 */
export function eigh(a: Mat, n: number): { values: Float64Array; vectors: Mat } {
  const m = Float64Array.from(a);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < JACOBI_SWEEPS; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += (m[p * n + q] ?? 0) ** 2;
    }
    if (off < JACOBI_TOL) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = m[p * n + q] ?? 0;
        if (Math.abs(apq) < JACOBI_TOL) continue;
        const app = m[p * n + p] ?? 0;
        const aqq = m[q * n + q] ?? 0;
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) /
          (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let i = 0; i < n; i++) {
          const mip = m[i * n + p] ?? 0;
          const miq = m[i * n + q] ?? 0;
          m[i * n + p] = c * mip - s * miq;
          m[i * n + q] = s * mip + c * miq;
        }
        for (let j = 0; j < n; j++) {
          const mpj = m[p * n + j] ?? 0;
          const mqj = m[q * n + j] ?? 0;
          m[p * n + j] = c * mpj - s * mqj;
          m[q * n + j] = s * mpj + c * mqj;
        }
        for (let i = 0; i < n; i++) {
          const vip = v[i * n + p] ?? 0;
          const viq = v[i * n + q] ?? 0;
          v[i * n + p] = c * vip - s * viq;
          v[i * n + q] = s * vip + c * viq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = m[i * n + i] ?? 0;
  const order = [...Array(n).keys()].sort(
    (x, y) => (values[x] ?? 0) - (values[y] ?? 0));
  const sortedValues = new Float64Array(n);
  const sortedVectors = new Float64Array(n * n);
  for (const [to, from] of order.entries()) {
    sortedValues[to] = values[from] ?? 0;
    for (let i = 0; i < n; i++) sortedVectors[i * n + to] = v[i * n + from] ?? 0;
  }
  return { values: sortedValues, vectors: sortedVectors };
}

const JACOBI_SWEEPS = 60;
const JACOBI_TOL = 1e-30;

/* ── The non-symmetric eigendecomposition ──────────────────────────────
 *
 * `eigh` takes symmetric matrices only and its answers are real. For a general matrix
 * **the answer is always complex** — a rotation matrix has no real eigenvalue at all
 * (±i). So it is a different function and a different route.
 *
 * The route has three parts.
 *
 * 1. **Hessenberg reduction** — Householder reflections zero everything below the first
 *    subdiagonal. It is an orthogonal transformation, so the eigenvalues do not change,
 *    and it takes the next stage's per-iteration cost from `O(n³)` to `O(n²)`.
 * 2. **Shifted QR iteration** — a Francis double shift pushes it to the real Schur form.
 *    The result is quasi-triangular: the diagonal mixes 1×1 blocks (real eigenvalues) and
 *    2×2 blocks (conjugate complex pairs).
 * 3. **Eigenvectors** — per eigenvalue, the null space of `(A − λI)v = 0` is solved by
 *    complex elimination.
 *
 * **Why go to Schur in real arithmetic.** A complex QR could go there directly, and a
 * real input's complex eigenvalues always come in conjugate pairs, so the real form keeps
 * that pairing **exactly.** Through the complex route the pair separates in the last
 * digits and an identity such as `sum = trace` is off by that rounding — which is exactly
 * what the golden asks about.
 */

/** An iteration that does not finish is a defect rather than convergence. LAPACK caps it
 *  too. */
const QR_ITERATIONS = 100;

/** The threshold below which a subdiagonal counts as 0. Relative to the size of the
 *  neighbouring diagonal entries. */
const DEFLATE_EPS = 1e-14;

/**
 * Householder reflections push it to **upper Hessenberg.** `A = Q·H·Qᵀ`.
 *
 * The transform is orthogonal, so the eigenvalues do not change — that is the
 * property which makes this step free.
 */
export function hessenberg(a: Mat, n: number): { h: Mat; q: Mat } {
  const h = Float64Array.from(a);
  const q = new Float64Array(n * n);
  for (let i = 0; i < n; i++) q[i * n + i] = 1;

  for (let k = 0; k < n - 2; k++) {
    let norm = 0;
    for (let i = k + 1; i < n; i++) norm += (h[i * n + k] ?? 0) ** 2;
    norm = Math.sqrt(norm);
    if (norm === 0) continue;

    const first = h[(k + 1) * n + k] ?? 0;
    const alpha = first >= 0 ? -norm : norm;
    const v = new Float64Array(n);
    v[k + 1] = first - alpha;
    for (let i = k + 2; i < n; i++) v[i] = h[i * n + k] ?? 0;
    let vv = 0;
    for (let i = k + 1; i < n; i++) vv += (v[i] ?? 0) ** 2;
    if (vv === 0) continue;

    // `H ← (I − 2vvᵀ/vᵀv)·H·(I − 2vvᵀ/vᵀv)`. Reflecting from both sides is what makes
    // it a similarity transformation — one side alone gives the Hessenberg shape with
    // different eigenvalues.
    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let i = k + 1; i < n; i++) dot += (v[i] ?? 0) * (h[i * n + j] ?? 0);
      const f = (2 * dot) / vv;
      for (let i = k + 1; i < n; i++) {
        h[i * n + j] = (h[i * n + j] ?? 0) - f * (v[i] ?? 0);
      }
    }
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = k + 1; j < n; j++) dot += (h[i * n + j] ?? 0) * (v[j] ?? 0);
      const f = (2 * dot) / vv;
      for (let j = k + 1; j < n; j++) {
        h[i * n + j] = (h[i * n + j] ?? 0) - f * (v[j] ?? 0);
      }
    }
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = k + 1; j < n; j++) dot += (q[i * n + j] ?? 0) * (v[j] ?? 0);
      const f = (2 * dot) / vv;
      for (let j = k + 1; j < n; j++) {
        q[i * n + j] = (q[i * n + j] ?? 0) - f * (v[j] ?? 0);
      }
    }
  }
  return { h, q };
}

/**
 * The eigenvalues — **a conjugate pair comes out of one 2×2 block together.**
 *
 * Shifted QR pushes it to quasi-triangular, then the diagonal blocks are read.
 * A 1×1 block is one real eigenvalue; a 2×2 block gives one pair from its
 * quadratic. A non-negative discriminant means two reals — a matrix can be
 * asymmetric and still have only real eigenvalues, and that branch runs here.
 */
export function eigvals(a: Mat, n: number): { re: Float64Array; im: Float64Array } {
  const { h } = hessenberg(a, n);
  const re = new Float64Array(n);
  const im = new Float64Array(n);

  let high = n - 1;
  let spent = 0;
  while (high >= 0) {
    if (high === 0) {
      re[0] = h[0] ?? 0;
      break;
    }
    // Where the subdiagonal is negligible against its neighbouring diagonals, it splits
    // there.
    let low = high;
    while (low > 0) {
      const sub = Math.abs(h[low * n + (low - 1)] ?? 0);
      const near = Math.abs(h[(low - 1) * n + (low - 1)] ?? 0)
        + Math.abs(h[low * n + low] ?? 0);
      if (sub <= DEFLATE_EPS * (near || 1)) {
        h[low * n + (low - 1)] = 0;
        break;
      }
      low -= 1;
    }

    if (low === high) {
      re[high] = h[high * n + high] ?? 0;
      high -= 1;
      spent = 0;
      continue;
    }
    if (low === high - 1) {
      const [p, q] = quadratic(
        h[low * n + low] ?? 0, h[low * n + high] ?? 0,
        h[high * n + low] ?? 0, h[high * n + high] ?? 0);
      re[low] = p.re; im[low] = p.im;
      re[high] = q.re; im[high] = q.im;
      high -= 2;
      spent = 0;
      continue;
    }

    if (spent >= QR_ITERATIONS) {
      throw new Error(
        `eig did not converge in ${QR_ITERATIONS} iterations on a ${n}x${n} matrix`);
    }
    spent += 1;
    qrSweep(h, n, low, high);
  }
  return { re, im };
}

/** A 2×2 block's eigenvalues — the two roots of the quadratic. A negative discriminant
 *  means a conjugate pair. */
function quadratic(a: number, b: number, c: number, d: number): [
  { re: number; im: number }, { re: number; im: number },
] {
  const tr = a + d;
  const det = a * d - b * c;
  const disc = (tr / 2) ** 2 - det;
  if (disc >= 0) {
    const root = Math.sqrt(disc);
    return [{ re: tr / 2 + root, im: 0 }, { re: tr / 2 - root, im: 0 }];
  }
  const root = Math.sqrt(-disc);
  return [{ re: tr / 2, im: root }, { re: tr / 2, im: -root }];
}

/**
 * One shifted QR sweep. It uses the **Wilkinson shift**, taking the real part alone when
 * the shift is complex.
 *
 * There is only real arithmetic here, so while converging towards a conjugate pair the
 * shift stops at the real part. A 2×2 block is left, and `quadratic` above produces that
 * pair exactly — **not narrowing the pair by iteration is the point.** Narrowing separates
 * the two roots in the last digits.
 */
function qrSweep(h: Mat, n: number, low: number, high: number): void {
  const a = h[(high - 1) * n + (high - 1)] ?? 0;
  const b = h[(high - 1) * n + high] ?? 0;
  const c = h[high * n + (high - 1)] ?? 0;
  const d = h[high * n + high] ?? 0;
  const [p, q] = quadratic(a, b, c, d);
  // Whichever of the two roots is closer to `d`. When they are complex both share a real
  // part, so either gives the same.
  const shift = Math.abs(p.re - d) <= Math.abs(q.re - d) ? p.re : q.re;

  const size = high - low + 1;
  const sub = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) sub[i * size + j] = h[(low + i) * n + (low + j)] ?? 0;
    sub[i * size + i] = (sub[i * size + i] ?? 0) - shift;
  }
  const { q: qq, r } = qr(sub, size, size);
  // `RQ + σI` — this is the similarity transformation. `Q` is orthogonal, so
  // `RQ = Qᵀ(A−σI)Q`.
  const next = matmul(r, qq, size, size, size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      h[(low + i) * n + (low + j)] = (next[i * size + j] ?? 0) + (i === j ? shift : 0);
    }
  }
}

/**
 * The eigenvector belonging to one eigenvalue — the null space of `(A − λI)v = 0`.
 *
 * `λ` is an eigenvalue only within rounding, so `A − λI` is **nearly** singular. Partial
 * pivoting elimination is pushed to the end and a position whose pivot is effectively 0 is
 * taken as a free variable and given 1. That is the null space direction, and it is the
 * classical prescription.
 *
 * **The magnitude is normalised to 1 and the sign is not fixed.** torch itself gives
 * opposite signs at float32 and float64 (measured). The golden does not lean on the sign
 * either; it asks about the definition (`A·V = V·diag(λ)`).
 */
function nullVector(
  a: Mat, n: number, lRe: number, lIm: number,
): { re: Float64Array; im: Float64Array } {
  // `A − λI` laid out as complex. The real and imaginary parts are held side by side.
  const mr = new Float64Array(n * n);
  const mi = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) mr[i * n + j] = a[i * n + j] ?? 0;
    mr[i * n + i] = (mr[i * n + i] ?? 0) - lRe;
    mi[i * n + i] = -lIm;
  }

  const where = new Int32Array(n).fill(-1);
  let row = 0;
  for (let col = 0; col < n && row < n; col++) {
    let best = row;
    let mag = 0;
    for (let i = row; i < n; i++) {
      const m = Math.hypot(mr[i * n + col] ?? 0, mi[i * n + col] ?? 0);
      if (m > mag) { mag = m; best = i; }
    }
    if (mag < 1e-12) continue;          // this column is a free variable
    if (best !== row) {
      for (let j = 0; j < n; j++) {
        let t = mr[row * n + j] ?? 0;
        mr[row * n + j] = mr[best * n + j] ?? 0;
        mr[best * n + j] = t;
        t = mi[row * n + j] ?? 0;
        mi[row * n + j] = mi[best * n + j] ?? 0;
        mi[best * n + j] = t;
      }
    }
    const pr = mr[row * n + col] ?? 0;
    const pi = mi[row * n + col] ?? 0;
    const den = pr * pr + pi * pi;
    for (let i = row + 1; i < n; i++) {
      const ar = mr[i * n + col] ?? 0;
      const ai = mi[i * n + col] ?? 0;
      // `f = (a/p)`, in complex.
      const fr = (ar * pr + ai * pi) / den;
      const fi = (ai * pr - ar * pi) / den;
      if (fr === 0 && fi === 0) continue;
      for (let j = col; j < n; j++) {
        const br = mr[row * n + j] ?? 0;
        const bi = mi[row * n + j] ?? 0;
        mr[i * n + j] = (mr[i * n + j] ?? 0) - (fr * br - fi * bi);
        mi[i * n + j] = (mi[i * n + j] ?? 0) - (fr * bi + fi * br);
      }
    }
    where[col] = row;
    row += 1;
  }

  const vr = new Float64Array(n);
  const vi = new Float64Array(n);
  // **One free variable is taken and given 1.** With none (when it looks numerically
  // non-singular) the last column is treated as free — as long as `λ` is an eigenvalue the
  // null space is certainly there, and not finding one means the threshold was tight
  // rather than that it does not exist.
  let free = n - 1;
  for (let col = n - 1; col >= 0; col--) {
    if (where[col] === -1) { free = col; break; }
  }
  vr[free] = 1;
  for (let col = free - 1; col >= 0; col--) {
    const r = where[col];
    if (r === undefined || r === -1) continue;
    let sr = 0;
    let si = 0;
    for (let j = col + 1; j < n; j++) {
      const br = mr[r * n + j] ?? 0;
      const bi = mi[r * n + j] ?? 0;
      sr += br * (vr[j] ?? 0) - bi * (vi[j] ?? 0);
      si += br * (vi[j] ?? 0) + bi * (vr[j] ?? 0);
    }
    const pr = mr[r * n + col] ?? 0;
    const pi = mi[r * n + col] ?? 0;
    const den = pr * pr + pi * pi;
    // `v = −s/p`.
    vr[col] = -(sr * pr + si * pi) / den;
    vi[col] = -(si * pr - sr * pi) / den;
  }

  let norm = 0;
  for (let i = 0; i < n; i++) norm += (vr[i] ?? 0) ** 2 + (vi[i] ?? 0) ** 2;
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < n; i++) {
    vr[i] = (vr[i] ?? 0) / norm;
    vi[i] = (vi[i] ?? 0) / norm;
  }
  return { re: vr, im: vi };
}

/**
 * Eigenvalues and eigenvectors. Where `torch.linalg.eig` sits.
 *
 * The vectors stand in **columns** — `V[:, k]` belongs to `values[k]`, which is
 * what makes `A·V = V·diag(λ)` hold.
 */
export function eig(a: Mat, n: number): {
  re: Float64Array; im: Float64Array; vecRe: Mat; vecIm: Mat;
} {
  const { re, im } = eigvals(a, n);
  const vecRe = new Float64Array(n * n);
  const vecIm = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    const v = nullVector(a, n, re[k] ?? 0, im[k] ?? 0);
    for (let i = 0; i < n; i++) {
      vecRe[i * n + k] = v.re[i] ?? 0;
      vecIm[i * n + k] = v.im[i] ?? 0;
    }
  }
  return { re, im, vecRe, vecIm };
}

/**
 * Singular value decomposition. **Rectangular input is accepted.**
 *
 * It is obtained from the eigendecomposition of `AᵀA` — the shortest route
 * that holds for small matrices. Singular values come back in
 * **descending** order (as in torch). On large or ill-conditioned matrices
 * this method loses digits, but that is not the size being pushed here.
 *
 * `u` is **reduced** — `rows × k` (`k = min(rows, cols)`). torch's default,
 * `full_matrices=True`, wants `rows × rows`, so `completeBasis` fills the
 * remaining columns. The two are kept apart because the reduced form is
 * always unique while the filled one is **not** once more than one
 * dimension is left over.
 */
export function svd(
  a: Mat, rows: number, cols = rows,
): { u: Mat; s: Float64Array; vt: Mat } {
  const k = Math.min(rows, cols);
  const at = transpose(a, rows, cols);
  const ata = matmul(at, a, cols, rows, cols);
  const { values, vectors } = eigh(ata, cols);
  // eigh is ascending, so it is reversed.
  const s = new Float64Array(k);
  const v = new Float64Array(cols * k);
  for (let j = 0; j < k; j++) {
    const from = cols - 1 - j;
    s[j] = Math.sqrt(Math.max(0, values[from] ?? 0));
    for (let i = 0; i < cols; i++) v[i * k + j] = vectors[i * cols + from] ?? 0;
  }
  const av = matmul(a, v, rows, cols, k);
  const u = new Float64Array(rows * k);
  for (let j = 0; j < k; j++) {
    const sj = s[j] ?? 0;
    for (let i = 0; i < rows; i++) {
      u[i * k + j] = sj > SVD_ZERO ? (av[i * k + j] ?? 0) / sj : 0;
    }
  }
  return { u, s, vt: transpose(v, cols, k) };
}

const SVD_ZERO = 1e-12;

/**
 * Fills `rows × k` orthonormal columns out to `rows × rows`.
 *
 * It finds directions orthogonal to the existing columns by Gram–Schmidt.
 * **With more than one dimension left over the answer is not unique** — any
 * rotation inside that subspace is the same factorisation. That is why the
 * golden cases ask in absolute value, and only measure where exactly one
 * dimension is left.
 */
export function completeBasis(u: Mat, rows: number, k: number): Mat {
  const out = new Float64Array(rows * rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < k; j++) out[i * rows + j] = u[i * k + j] ?? 0;
  }
  let filled = k;
  for (let seed = 0; seed < rows && filled < rows; seed++) {
    const v = new Float64Array(rows);
    v[seed] = 1;
    for (let j = 0; j < filled; j++) {
      let dot = 0;
      for (let i = 0; i < rows; i++) dot += (v[i] ?? 0) * (out[i * rows + j] ?? 0);
      for (let i = 0; i < rows; i++) {
        v[i] = (v[i] ?? 0) - dot * (out[i * rows + j] ?? 0);
      }
    }
    let norm = 0;
    for (let i = 0; i < rows; i++) norm += (v[i] ?? 0) ** 2;
    norm = Math.sqrt(norm);
    if (norm < SVD_ZERO) continue;
    for (let i = 0; i < rows; i++) out[i * rows + filled] = (v[i] ?? 0) / norm;
    filled += 1;
  }
  return out;
}

/**
 * The Moore–Penrose pseudoinverse. Directions whose singular value is near
 * zero are dropped.
 */
export function pinverse(a: Mat, rows: number, cols = rows, rcond?: number): Mat {
  const k = Math.min(rows, cols);
  const { u, s, vt } = svd(a, rows, cols);
  // **`rcond` is relative to the largest singular value and the default is not.**
  // numpy's `lstsq` cutoff is `rcond * s[0]`; the machine-precision default carries
  // a `max(rows, cols)` factor that `rcond` does not. Multiplying the given number
  // by that factor too would make `rcond=0.9` cut a different set on a 4×2 than on
  // a 2×4, which is not what either torch or numpy does.
  const tol = rcond === undefined
    ? (s[0] ?? 0) * Math.max(rows, cols) * Number.EPSILON
    : (s[0] ?? 0) * rcond;
  const inv = new Float64Array(k * k);
  for (let j = 0; j < k; j++) {
    const sj = s[j] ?? 0;
    if (sj > tol) inv[j * k + j] = 1 / sj;
  }
  const v = transpose(vt, k, cols);
  return matmul(matmul(v, inv, cols, k, k), transpose(u, rows, k), cols, k, rows);
}

/**
 * How many singular values survive `pinverse`'s `rcond`.
 *
 * The same cutoff `pinverse` applies, counted rather than inverted, so the caller
 * can ask **whether the cutoff bites** without a second convention. `lstsq` needs
 * exactly that: under the default driver the answer is only ours while the matrix
 * stays full rank.
 */
export function keptUnder(a: Mat, rows: number, cols: number,
                          rcond: number): number {
  const { s } = svd(a, rows, cols);
  const tol = (s[0] ?? 0) * rcond;
  let kept = 0;
  for (let j = 0; j < Math.min(rows, cols); j++) if ((s[j] ?? 0) > tol) kept += 1;
  return kept;
}

/**
 * The count of non-zero singular values. What counts as zero is scaled to
 * the largest singular value.
 */
export function matrixRank(a: Mat, rows: number, cols = rows,
                           given?: number): number {
  const k = Math.min(rows, cols);
  const { s } = svd(a, rows, cols);
  // **A given tolerance is absolute.** numpy's `matrix_rank(tol=)` compares the
  // singular values to it directly, and only the default is scaled to the largest
  // one — which is why this is not the `rcond` of `pinverse` above under another
  // name.
  const tol = given ?? (s[0] ?? 0) * Math.max(rows, cols) * Number.EPSILON;
  let rank = 0;
  for (let j = 0; j < k; j++) if ((s[j] ?? 0) > tol) rank += 1;
  return rank;
}

/**
 * What `lu_factor` produces — `L` and `U` packed into one plate, plus the
 * pivot table.
 *
 * **The pivot table counts from 1.** That is the LAPACK convention and
 * torch inherited it as-is. A 2×2 with no swap gives `[1, 2]`, not `[0, 1]`
 * — count from zero and `luSolveFactored` gives a different answer without
 * a sound. Matched by asking real torch.
 */
export interface LuPacked {
  readonly lu: Mat;
  readonly piv: Int32Array;
  readonly rows: number;
  readonly cols: number;
}

export function luFactor(a: Mat, rows: number, cols: number): LuPacked {
  const m = Float64Array.from(a);
  const k = Math.min(rows, cols);
  const piv = new Int32Array(k);
  for (let col = 0; col < k; col++) {
    let best = col;
    let bestAbs = Math.abs(m[col * cols + col] ?? 0);
    for (let i = col + 1; i < rows; i++) {
      const v = Math.abs(m[i * cols + col] ?? 0);
      if (v > bestAbs) { bestAbs = v; best = i; }
    }
    piv[col] = best + 1;
    if (best !== col) {
      for (let j = 0; j < cols; j++) {
        const t = m[col * cols + j] ?? 0;
        m[col * cols + j] = m[best * cols + j] ?? 0;
        m[best * cols + j] = t;
      }
    }
    const pivot = m[col * cols + col] ?? 0;
    if (pivot === 0) continue;
    for (let i = col + 1; i < rows; i++) {
      const f = (m[i * cols + col] ?? 0) / pivot;
      m[i * cols + col] = f;
      for (let j = col + 1; j < cols; j++) {
        m[i * cols + j] = (m[i * cols + j] ?? 0) - f * (m[col * cols + j] ?? 0);
      }
    }
  }
  return { lu: m, piv, rows, cols };
}

/**
 * Unpacks the packed plate into `P`, `L` and `U`.
 */
export function luExpand(f: LuPacked): { p: Mat; l: Mat; u: Mat } {
  const { rows, cols, lu: packed, piv } = f;
  const k = Math.min(rows, cols);
  const l = new Float64Array(rows * k);
  const u = new Float64Array(k * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < k; j++) {
      l[i * k + j] = i === j ? 1 : (i > j ? (packed[i * cols + j] ?? 0) : 0);
    }
  }
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < cols; j++) u[i * cols + j] = i <= j ? (packed[i * cols + j] ?? 0) : 0;
  }
  // The swaps are retraced to stand up the permutation matrix. `A = P L U`, so `P` is
  // **the inverse** of the swaps.
  const order = new Int32Array(rows);
  for (let i = 0; i < rows; i++) order[i] = i;
  for (let col = 0; col < k; col++) {
    const src = (piv[col] ?? col + 1) - 1;
    if (src !== col) {
      const t = order[col] ?? 0;
      order[col] = order[src] ?? 0;
      order[src] = t;
    }
  }
  const p = new Float64Array(rows * rows);
  for (let i = 0; i < rows; i++) p[(order[i] ?? i) * rows + i] = 1;
  return { p, l, u };
}

/**
 * Solves **knowing** the matrix is triangular. One pass, forward or
 * backward, and it is done.
 */
export function solveTriangular(
  a: Mat, b: Mat, n: number, m: number,
  upper: boolean, unit: boolean,
): Mat {
  const x = new Float64Array(n * m);
  for (let i = 0; i < n * m; i++) x[i] = b[i] ?? 0;
  const order = upper
    ? Array.from({ length: n }, (_, i) => n - 1 - i)
    : Array.from({ length: n }, (_, i) => i);
  for (const i of order) {
    for (let k = 0; k < n; k++) {
      if (upper ? k <= i : k >= i) continue;
      const c = a[i * n + k] ?? 0;
      if (c === 0) continue;
      for (let j = 0; j < m; j++) {
        x[i * m + j] = (x[i * m + j] ?? 0) - c * (x[k * m + j] ?? 0);
      }
    }
    // **With `unit` the diagonal is not read** — it is taken as 1. Not honouring that
    // changes the values quietly.
    const d = unit ? 1 : (a[i * n + i] ?? 1);
    for (let j = 0; j < m; j++) x[i * m + j] = (x[i * m + j] ?? 0) / d;
  }
  return x;
}

/** What counts as "small" for scaling and squaring. Below this 1-norm the Taylor series
 *  converges quickly. */
const EXP_SMALL = 0.5;
/** How many terms that condition needs. `0.5^18/18!` is far below double precision's
 *  floor. */
const EXP_TERMS = 18;

/**
 * The matrix exponential `e^A` — **scaling and squaring.**
 *
 * Taylor alone does not converge on large matrices. The answer for `A*5` is
 * 4.8e+10, and at that point the growing terms overflow first. Bringing the
 * 1-norm of `A/2^s` below 0.5, running the series, and then squaring `s`
 * times gives the same answer safely — `e^A = (e^{A/2^s})^{2^s}`.
 */
export function matrixExp(a: Mat, n: number): Mat {
  let norm = 0;
  for (let j = 0; j < n; j++) {
    let col = 0;
    for (let i = 0; i < n; i++) col += Math.abs(a[i * n + j] ?? 0);
    norm = Math.max(norm, col);
  }
  const squarings = norm > EXP_SMALL
    ? Math.max(0, Math.ceil(Math.log2(norm / EXP_SMALL)))
    : 0;
  const scale = 2 ** squarings;
  const scaled = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) scaled[i] = (a[i] ?? 0) / scale;
  let term: Mat = new Float64Array(n * n);
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    term[i * n + i] = 1;
    out[i * n + i] = 1;
  }
  for (let k = 1; k <= EXP_TERMS; k++) {
    const next = matmul(term, scaled, n, n, n);
    for (let i = 0; i < n * n; i++) next[i] = (next[i] ?? 0) / k;
    term = next;
    for (let i = 0; i < n * n; i++) out[i] = (out[i] ?? 0) + (term[i] ?? 0);
  }
  let result: Mat = out;
  for (let s = 0; s < squarings; s++) result = matmul(result, result, n, n, n);
  return result;
}

/**
 * Freezes the backward of `e^A` **into a single matrix.**
 *
 * The Fréchet derivative has this identity — the upper-right block of
 * `expm([[Aᵀ, E],[0, Aᵀ]])` is the derivative in the direction `E`. It is
 * an identity, not an approximation.
 *
 * **That route is not available here, though.** The `Ḡ` arriving in the
 * backward is on the GPU and this file is CPU — you cannot wait on the GPU
 * from inside a backward. So it uses the fact that `Ḡ` enters **linearly**:
 * solve once for each `E_k` with a single one in one slot, build an `n²×n²`
 * table, and the backward is one matrix multiply.
 *
 * The values are exact (each column is a real derivative, not a finite
 * difference). The price is `n²` calls to a `2n×2n` `expm` in the forward,
 * which is cheap at the sizes being pushed here.
 */
export function matrixExpAdjointMap(a: Mat, n: number): Mat {
  const at = transpose(a, n, n);
  const size = n * n;
  const wide = 2 * n;
  const map = new Float64Array(size * size);
  const block = new Float64Array(wide * wide);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      block[i * wide + j] = at[i * n + j] ?? 0;
      block[(n + i) * wide + (n + j)] = at[i * n + j] ?? 0;
    }
  }
  for (let k = 0; k < size; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) block[i * wide + n + j] = 0;
    }
    block[Math.floor(k / n) * wide + n + (k % n)] = 1;
    const e = matrixExp(block, wide);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        map[(i * n + j) * size + k] = e[i * wide + n + j] ?? 0;
      }
    }
  }
  return map;
}

/**
 * `F_ij = 1/(λⱼ − λᵢ)`, which enters `eigh`'s eigenvector backward.
 *
 * **Repeated eigenvalues blow it up.** torch blows up with it, so this is
 * the same limit rather than an imitation. The diagonal is a difference
 * with itself, so it is zero by definition rather than a division.
 */
export function eighGap(values: Float64Array, n: number): Mat {
  const f = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const gap = (values[j] ?? 0) - (values[i] ?? 0);
      f[i * n + j] = gap === 0 ? 0 : 1 / gap;
    }
  }
  return f;
}

/**
 * Solves `A x = b` using what `lu_factor` produced, or `Aᵀ x = b` with `adjoint`.
 *
 * `A = P L U`, so the plain solve moves `b` onto `L U x = Pᵀ b` and runs forward
 * then back. **The adjoint runs the whole thing in reverse**: `Aᵀ = Uᵀ Lᵀ Pᵀ`, so
 * `Uᵀ` goes first — lower triangular, and its diagonal is `U`'s rather than ones —
 * `Lᵀ` second with a unit diagonal, and the permutation lands on the *answer*
 * instead of on the right-hand side. Applying the same swaps in reverse order is
 * `P` where applying them forward is `Pᵀ`.
 *
 * Storage is real, so torch's `Aᴴ` is a transpose here.
 */
export function luSolveFactored(f: LuPacked, b: Mat, m: number,
                                adjoint = false): Mat {
  const n = f.rows;
  const x = new Float64Array(n * m);
  for (let i = 0; i < n * m; i++) x[i] = b[i] ?? 0;
  if (adjoint) {
    // Uᵀ w = b, forward.
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < i; k++) {
        const u = f.lu[k * f.cols + i] ?? 0;
        if (u === 0) continue;
        for (let j = 0; j < m; j++) {
          x[i * m + j] = (x[i * m + j] ?? 0) - u * (x[k * m + j] ?? 0);
        }
      }
      const d = f.lu[i * f.cols + i] ?? 1;
      for (let j = 0; j < m; j++) x[i * m + j] = (x[i * m + j] ?? 0) / d;
    }
    // Lᵀ z = w, back. The diagonal is one, so there is nothing to divide by.
    for (let i = n - 1; i >= 0; i--) {
      for (let k = i + 1; k < n; k++) {
        const l = f.lu[k * f.cols + i] ?? 0;
        if (l === 0) continue;
        for (let j = 0; j < m; j++) {
          x[i * m + j] = (x[i * m + j] ?? 0) - l * (x[k * m + j] ?? 0);
        }
      }
    }
    // x = P z — the forward's swaps, undone.
    for (let col = f.piv.length - 1; col >= 0; col--) {
      const src = (f.piv[col] ?? col + 1) - 1;
      if (src === col) continue;
      for (let j = 0; j < m; j++) {
        const t = x[col * m + j] ?? 0;
        x[col * m + j] = x[src * m + j] ?? 0;
        x[src * m + j] = t;
      }
    }
    return x;
  }
  // The swaps made in the forward are applied to the right-hand side in the same
  // order.
  for (let col = 0; col < f.piv.length; col++) {
    const src = (f.piv[col] ?? col + 1) - 1;
    if (src === col) continue;
    for (let j = 0; j < m; j++) {
      const t = x[col * m + j] ?? 0;
      x[col * m + j] = x[src * m + j] ?? 0;
      x[src * m + j] = t;
    }
  }
  for (let i = 1; i < n; i++) {
    for (let k = 0; k < i; k++) {
      const f2 = f.lu[i * f.cols + k] ?? 0;
      if (f2 === 0) continue;
      for (let j = 0; j < m; j++) {
        x[i * m + j] = (x[i * m + j] ?? 0) - f2 * (x[k * m + j] ?? 0);
      }
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let k = i + 1; k < n; k++) {
      const f2 = f.lu[i * f.cols + k] ?? 0;
      if (f2 === 0) continue;
      for (let j = 0; j < m; j++) {
        x[i * m + j] = (x[i * m + j] ?? 0) - f2 * (x[k * m + j] ?? 0);
      }
    }
    const d = f.lu[i * f.cols + i] ?? 1;
    for (let j = 0; j < m; j++) x[i * m + j] = (x[i * m + j] ?? 0) / d;
  }
  return x;
}

/**
 * Cholesky's backward.
 *
 * The gradient towards `A` when `L = chol(A)`. The derivation is not short,
 * so the formula is written out — with `P = Φ(Lᵀ·L̄)`, where `Φ` takes the
 * lower triangle and halves the diagonal, symmetrise `Ā = L⁻ᵀ·P·L⁻¹`. The
 * symmetrisation is needed because `A` is symmetric, so its upper and lower
 * triangles share the same degrees of freedom.
 */
export function choleskyBackward(l: Mat, lbar: Mat, n: number): Mat {
  const lt = transpose(l, n, n);
  const m = matmul(lt, lbar, n, n, n);
  const p = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      p[i * n + j] = (m[i * n + j] ?? 0) * (i === j ? 0.5 : 1);
    }
  }
  const linv = inverse(l, n);
  const abar = matmul(matmul(transpose(linv, n, n), p, n, n, n), linv, n, n, n);
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out[i * n + j] = 0.5 * ((abar[i * n + j] ?? 0) + (abar[j * n + i] ?? 0));
    }
  }
  return out;
}
