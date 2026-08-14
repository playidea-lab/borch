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

/** 행 우선으로 늘어놓은 정사각 행렬. */
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
  /** 행을 맞바꾼 횟수의 부호. 행렬식의 부호가 여기서 나온다. */
  readonly sign: number;
  readonly n: number;
  readonly singular: boolean;
}

/**
 * 부분 피벗을 쓰는 LU 분해.
 *
 * **피벗은 선택이 아니다.** 안 하면 첫 원소가 0 이거나 아주 작은 행렬에서 나눗셈이
 * 터지거나 자릿수가 통째로 날아간다.
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

/** 부호와 로그 절댓값을 따로. 행렬식이 아주 작을 때 `log(det)` 보다 안전하다. */
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

/** `A X = B` 를 푼다. `B` 는 열이 `m` 개다. */
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

/** 아래 삼각 콜레스키. 대칭 양정부호가 아니면 던진다 — 조용히 NaN 을 내지 않는다. */
export function cholesky(a: Mat, n: number): Mat {
  const l = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j] ?? 0;
      for (let k = 0; k < j; k++) s -= (l[i * n + k] ?? 0) * (l[j * n + k] ?? 0);
      if (i === j) {
        if (s <= 0) {
          throw new Error(
            "cholesky: 대칭 양정부호가 아니다 (주소행렬식이 0 이하다)",
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
 * 하우스홀더 QR.
 *
 * **그람-슈미트가 아니라 하우스홀더다.** 둘은 같은 분해를 다른 부호로 낸다. LAPACK 이
 * 하우스홀더를 쓰고 torch 가 LAPACK 을 쓰므로, 골든의 `R` 과 부호까지 맞추려면
 * 같은 방식이어야 한다 — `Q` 는 골든이 절댓값으로 묻지만 `R` 은 그대로 묻는다.
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
 * 대칭 행렬의 고윳값·고유벡터 — 야코비 회전.
 *
 * 고윳값을 **오름차순**으로 준다. torch 의 `linalg.eigh` 가 그렇다.
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
 * 정사각 행렬의 특이값 분해.
 *
 * `AᵀA` 의 고유분해에서 얻는다 — 작은 행렬에서 통하는 가장 짧은 길이다. 특이값을
 * **내림차순**으로 준다(torch 와 같다). 큰 행렬이나 조건수가 나쁜 행렬에서는
 * 이 방법이 자릿수를 잃는데, 여기서 미는 크기가 아니다.
 */
export function svd(a: Mat, n: number): { u: Mat; s: Float64Array; vt: Mat } {
  const at = transpose(a, n, n);
  const ata = matmul(at, a, n, n, n);
  const { values, vectors } = eigh(ata, n);
  // eigh 는 오름차순이라 뒤집는다.
  const s = new Float64Array(n);
  const v = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    const from = n - 1 - k;
    s[k] = Math.sqrt(Math.max(0, values[from] ?? 0));
    for (let i = 0; i < n; i++) v[i * n + k] = vectors[i * n + from] ?? 0;
  }
  const av = matmul(a, v, n, n, n);
  const u = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    const sk = s[k] ?? 0;
    for (let i = 0; i < n; i++) {
      u[i * n + k] = sk > SVD_ZERO ? (av[i * n + k] ?? 0) / sk : 0;
    }
  }
  return { u, s, vt: transpose(v, n, n) };
}

const SVD_ZERO = 1e-12;

/** 무어-펜로즈 유사역행렬. 특이값이 0 에 가까운 방향은 버린다. */
export function pinverse(a: Mat, n: number): Mat {
  const { u, s, vt } = svd(a, n);
  const smax = s[0] ?? 0;
  const tol = smax * n * Number.EPSILON;
  const inv = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    const sk = s[k] ?? 0;
    if (sk > tol) inv[k * n + k] = 1 / sk;
  }
  const v = transpose(vt, n, n);
  return matmul(matmul(v, inv, n, n, n), transpose(u, n, n), n, n, n);
}

/** 0 이 아닌 특이값의 개수. 무엇을 0 으로 볼지는 가장 큰 특이값에 맞춘다. */
export function matrixRank(a: Mat, n: number): number {
  const { s } = svd(a, n);
  const tol = (s[0] ?? 0) * n * Number.EPSILON;
  let rank = 0;
  for (let k = 0; k < n; k++) if ((s[k] ?? 0) > tol) rank += 1;
  return rank;
}

/**
 * 콜레스키의 역방향.
 *
 * `L = chol(A)` 일 때 `A` 로 가는 기울기다. 유도가 짧지 않아서 식을 그대로 적는다 —
 * `P = Φ(Lᵀ·L̄)` 에서 `Φ` 는 아래 삼각을 취하되 대각을 반으로 줄이는 것이고,
 * `Ā = L⁻ᵀ·P·L⁻¹` 을 대칭화한다. 대칭화가 필요한 이유는 `A` 가 대칭이라 위·아래
 * 삼각이 같은 자유도를 공유하기 때문이다.
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
