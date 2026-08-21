/**
 * 작은 행렬의 선형대수 — **CPU 에서 돈다.**
 *
 * ## 왜 CPU 인가
 *
 * LU·QR·야코비 회전은 전부 **순차적**이다. 앞 단계의 결과가 다음 단계의 입력이라
 * 병렬로 펼 자리가 거의 없고, 2×2 나 3×3 에서는 커널을 띄우고 결과를 읽어 오는
 * 왕복이 계산보다 비싸다. 자매도 같은 결론으로 numpy 를 쓴다.
 *
 * 대가는 이 연산들이 **비동기**가 된다는 것이다. GPU 에서 값을 읽어 와야 하니까.
 * 역방향에 필요한 것(역행렬, 행렬식 같은)은 순방향에서 미리 구해 붙잡아 두므로,
 * `backward()` 자체는 그대로 동기다.
 *
 * ## 배정도로 센다
 *
 * 입력도 출력도 float32 지만 계산은 float64 다. 소거법은 자릿수를 잘 잃고, 그 손실이
 * 조건수가 나쁜 행렬에서 눈에 보인다 — 코어가 `erf` 에서 같은 이유로 float64 를 쓴다.
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
  // 아래 삼각 — 대각이 1 이라 나눗셈이 없다.
  for (let i = 1; i < n; i++) {
    for (let k = 0; k < i; k++) {
      const f2 = f.lu[i * n + k] ?? 0;
      if (f2 === 0) continue;
      for (let j = 0; j < m; j++) {
        x[i * m + j] = (x[i * m + j] ?? 0) - f2 * (x[k * m + j] ?? 0);
      }
    }
  }
  // 위 삼각.
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
    // 부호를 머리와 같게 잡아야 뺄셈에서 자릿수를 안 잃는다. LAPACK 도 그렇게 한다.
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
  // eigh 는 오름차순이라 뒤집는다.
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
export function pinverse(a: Mat, rows: number, cols = rows): Mat {
  const k = Math.min(rows, cols);
  const { u, s, vt } = svd(a, rows, cols);
  const tol = (s[0] ?? 0) * Math.max(rows, cols) * Number.EPSILON;
  const inv = new Float64Array(k * k);
  for (let j = 0; j < k; j++) {
    const sj = s[j] ?? 0;
    if (sj > tol) inv[j * k + j] = 1 / sj;
  }
  const v = transpose(vt, k, cols);
  return matmul(matmul(v, inv, cols, k, k), transpose(u, rows, k), cols, k, rows);
}

/**
 * The count of non-zero singular values. What counts as zero is scaled to
 * the largest singular value.
 */
export function matrixRank(a: Mat, rows: number, cols = rows): number {
  const k = Math.min(rows, cols);
  const { s } = svd(a, rows, cols);
  const tol = (s[0] ?? 0) * Math.max(rows, cols) * Number.EPSILON;
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
  // 교환을 되짚어 순열 행렬을 세운다. `A = P L U` 이므로 `P` 는 교환의 **역**이다.
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
    // **`unit` 이면 대각을 안 본다** — 1 로 친다. 안 지키면 값이 조용히 달라진다.
    const d = unit ? 1 : (a[i * n + i] ?? 1);
    for (let j = 0; j < m; j++) x[i * m + j] = (x[i * m + j] ?? 0) / d;
  }
  return x;
}

/** 스케일링·제곱에서 무엇을 "작다" 로 볼지. 1-노름이 이 아래면 테일러가 빨리 모인다. */
const EXP_SMALL = 0.5;
/** 그 조건에서 필요한 항의 개수. `0.5^18/18!` 은 배정도의 바닥보다 한참 아래다. */
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
 * Solves `A x = b` using what `lu_factor` produced.
 */
export function luSolveFactored(f: LuPacked, b: Mat, m: number): Mat {
  const n = f.rows;
  const x = new Float64Array(n * m);
  for (let i = 0; i < n * m; i++) x[i] = b[i] ?? 0;
  // 순방향에서 한 교환을 오른쪽에도 같은 순서로 적용한다.
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
