/**
 * `torch.linalg` — the namespace, over the methods that already carry the arithmetic.
 *
 * Every name here forwards to a `Tensor` method. The values are the methods' values, and
 * the golden already holds those against real torch, so **nothing here is a second
 * implementation.** What this file decides is the *shape of the call*: which argument is
 * the receiver, and what the defaults are.
 *
 * ## Why it is written out by hand rather than generated
 *
 * A loop over the method names would be shorter and wrong in three places, and wrong in
 * the way that does not raise:
 *
 * - **`lu_solve` is received by the factors.** torch is `linalg.lu_solve(LU, pivots, B)`
 *   and the method here is `B.luSolve(LU, pivots)` — the receiver is the right-hand side.
 *   Forward mechanically and `linalg.luSolve(B, LU, pivots)` still has the right name and
 *   the right number of arguments, so nothing catches it and only the value is wrong. With
 *   square matrices even the shape agrees.
 * - **`diagonal` reads different axes.** `torch.diagonal` takes the first two (`0, 1`) and
 *   `torch.linalg.diagonal` the last two (`-2, -1`). Given `(2,3,4)` that is `(4,2)` against
 *   `(2,3)` — the same name, and even the shape differs.
 * - **Five names are spelled differently.** torch's `linalg.inv`, `linalg.pinv` and
 *   `linalg.matmul` are `inverse`, `pinverse` and `mm` as methods, and `tensorsolve` and
 *   `tensorinv` are `tensorSolve` and `tensorInv`.
 *
 * The last two nearly went in here as fresh implementations, because the search that found
 * them missing used torch's spelling against camelCase method names. What caught it was
 * `test_binding_fills_in.py` noticing the two declarations landing on one lookup key —
 * **a normaliser is a claim about which differences do not matter**, and case was carrying
 * the whole difference between a name that exists and a name that does not.
 *
 * `parity.ts` writes the general form of this down: *who the receiver is belongs to the
 * name too.* It is the reason `Tensor.lu_solve` is checked there by value against the
 * other spelling rather than by its signature.
 *
 * ## What is not here
 *
 * `cholesky_ex` and `inv_ex` return a status code instead of raising. Nothing here
 * produces that status, and a wrapper that always reports success would be a check that
 * cannot fail — the refusal these methods raise is the answer, so the `_ex` pair stays
 * absent rather than becoming a lie with a torch-shaped name.
 */

import { RuntimeError } from "./errors.js";
import type { DType } from "./dtype.js";
import { Tensor } from "./tensor.js";

// ── Forwarded, receiver unchanged ───────────────────────────────────────

/** `det(A)` — the determinant. */
export function det(a: Tensor): Promise<Tensor> {
  return a.det();
}

/** `slogdet(A)` — the sign and the log of the absolute determinant. */
export function slogdet(a: Tensor): Promise<{ sign: Tensor; logabs: Tensor }> {
  return a.slogdet();
}

/** `matrix_power(A, n)` — `A` multiplied by itself `n` times. */
export function matrixPower(input: Tensor, n: number): Tensor {
  return input.matrixPower(n);
}

/** `matrix_rank(A, tol=None)` — how many singular values are above the tolerance. */
export function matrixRank(input: Tensor, tol?: number): Promise<Tensor> {
  return input.matrixRank(tol);
}

/** `matrix_exp(A)` — the matrix exponential. */
export function matrixExp(input: Tensor): Promise<Tensor> {
  return input.matrixExp();
}

/** `cholesky(A, upper=False)` — the Cholesky factor. */
export function cholesky(input: Tensor, upper = false): Promise<Tensor> {
  return input.cholesky(upper);
}

/** `qr(A, mode="reduced")` — the QR decomposition. */
export function qr(
  a: Tensor, mode: "reduced" | "complete" = "reduced",
): Promise<{ q: Tensor; r: Tensor }> {
  return a.qr(mode !== "complete");
}

/** `svd(A, full_matrices=True)` — the singular value decomposition. */
export function svd(
  a: Tensor, fullMatrices = true,
): Promise<{ u: Tensor; s: Tensor; vt: Tensor }> {
  return a.linalgSvd(fullMatrices);
}

/** `svdvals(A)` — the singular values alone. */
export function svdvals(a: Tensor): Promise<Tensor> {
  return a.svdvals();
}

/** `eig(A)` — eigenvalues and eigenvectors of a general square matrix. */
export function eig(a: Tensor): Promise<{ values: Tensor; vectors: Tensor }> {
  return a.eig();
}

/** `eigvals(A)` — the eigenvalues alone. */
export function eigvals(input: Tensor): Promise<Tensor> {
  return input.eigvals();
}

/** `eigh(A, UPLO="L")` — the symmetric (Hermitian) eigendecomposition. */
export function eigh(
  input: Tensor, UPLO: "L" | "U" = "L",
): Promise<{ values: Tensor; vectors: Tensor }> {
  return input.eigh(UPLO);
}

/** `eigvalsh(A, UPLO="L")` — the symmetric eigenvalues alone. */
export function eigvalsh(input: Tensor, UPLO: "L" | "U" = "L"): Promise<Tensor> {
  return input.eigvalsh(UPLO);
}

/** `solve(A, B)` — solve `A x = B`. */
export function solve(a: Tensor, b: Tensor): Promise<Tensor> {
  return a.solve(b);
}

/** `solve_triangular(A, B, upper, left=True, unitriangular=False)`. */
export function solveTriangular(
  input: Tensor, b: Tensor, upper: boolean, left = true, unitriangular = false,
): Promise<Tensor> {
  return input.solveTriangular(b, upper, left, unitriangular);
}

/**
 * `lstsq(A, B, rcond=None, *, driver=None)` — the least-squares solution.
 *
 * The last two used to be absent, so `rcond` was **received by JavaScript and
 * discarded** — a cutoff that reads as a tuning knob and moved nothing.
 *
 * **The four drivers are four algorithms and this is one of them.** The
 * pseudoinverse is the SVD, which is `gelsd`. At full rank all four of torch's
 * agree to float noise; once the cutoff bites they separate, measured on
 * `[[1,1],[1,2],[1,3],[1,4]]` against `[6,5,7,10]` with `rcond=0.9`: `gels`
 * 3.500/1.400, `gelsy` 0.770/2.310, `gelsd` and `gelss` 0.790/2.322.
 *
 * So `gels` takes no cutoff — torch's assumes full rank and never reads `rcond` —
 * and `gelsy`, *the default*, is refused where the cutoff would change the answer,
 * because it solves with a pivoted QR there and handing back the SVD's numbers
 * under its name is a wrong answer with nothing attached to it. The core refuses
 * the same call for the same reason.
 *
 * The arithmetic is here rather than on `Tensor.lstsq`, which torch removed and
 * the core keeps as a one-argument tombstone.
 *
 * **A batch is answered now, and the refusal that stood here was one line from
 * the answer.** It read `input.shape.length !== 2` and threw — while `pinverse`
 * one line below has batched all along, and `matmul` batches too. What was
 * genuinely missing was smaller than the refusal: the cut-off test compared one
 * count against `Math.min(...input.shape)`, which on a batch is the batch size.
 *
 * **The two readings of `B` do not broadcast the same way** — measured on torch,
 * and the core carries the same rule with the same measurement written out. A
 * right-hand side one dimension shorter is a vector per matrix and its leading
 * dimensions must *equal* the matrix's; one of equal rank is `(*, m, k)` and its
 * leading dimensions broadcast.
 */
export async function lstsq(input: Tensor, b: Tensor, rcond?: number,
                            driver = "gelsy"): Promise<Tensor> {
  if (!["gels", "gelsy", "gelsd", "gelss"].includes(driver)) {
    throw new RuntimeError(
      "lstsq: parameter `driver` should be one of (gels, gelsy, gelsd, gelss)");
  }
  const rank = input.shape.length;
  const lead = input.shape.slice(0, -2);
  const vector = b.shape.length === rank - 1
    && b.shape.slice(0, -1).join() === lead.join();
  const matrix = b.shape.length === rank && b.shape[rank - 2] === input.shape[rank - 2];
  if (!vector && !matrix) {
    throw new RuntimeError(b.shape.length < rank - 1
      ? "lstsq: input.dim() must be greater or equal to other.dim() and "
        + "(input.dim() - other.dim()) <= 1"
      : "lstsq: input.size(-2) should match other.size(-2)");
  }
  const cut = driver === "gels" ? undefined : rcond;
  if (driver === "gelsy" && cut !== undefined && await cutBites(input, cut)) {
    throw new RuntimeError(
      `lstsq(rcond=${rcond}, driver="gelsy") on a cut that leaves the matrix `
      + "rank-deficient is not in the browser subset — a pivoted QR and the SVD "
      + 'part there and only the SVD is here. Ask for driver="gelsd" on purpose.');
  }
  // The least-squares solution **is** the pseudoinverse applied to `B`, and that
  // one takes the cut-off — so this composes two public pieces rather than opening
  // the matrix a second time. `Tensor.lstsq` next door is the same thing without a
  // cut-off, which is what torch's removed method computed.
  const p = await input.pinverse(cut);
  return vector ? p.matmul(b.unsqueeze(-1)).squeeze(-1) : p.matmul(b);
}

/**
 * Does this cut-off drop a singular value? `lstsq`'s default driver hinges on it.
 *
 * **Any matrix in the batch is enough**, which is what the core does by refusing
 * inside its per-matrix loop. The comparison is per matrix: the largest singular
 * value is taken along the last axis and kept there, so each row scales by its own.
 */
async function cutBites(input: Tensor, rcond: number): Promise<boolean> {
  const s = await input.svdvals();
  const kept = s.gt(s.amax(-1, true).mul(Tensor.full([], rcond))).sum(-1, false);
  // `Math.min(...input.shape)` here is a defect a batch hides: over `[2, 4, 2]` it
  // gives 2, which is also `min(m, n)`, and over `[1, 4, 2]` it gives 1 and the
  // refusal disappears. The two trailing dimensions are the matrix.
  const k = Math.min(input.shape[input.shape.length - 2] as number,
                     input.shape[input.shape.length - 1] as number);
  return Array.from(await kept.toArray()).some((c) => c < k);
}

/** `matrix_norm(A, ord="fro", dim=(-2,-1), keepdim=False)`. */
export function matrixNorm(input: Tensor, ord: number | string = "fro",
                           dim: readonly [number, number] = [-2, -1],
                           keepdim = false): Promise<Tensor> {
  return input.matrixNorm(ord, dim, keepdim);
}

/** `vector_norm(x, ord=2, dim=None, keepdim=False)`. */
export function vectorNorm(input: Tensor, ord = 2, dim?: number,
                           keepdim = false): Tensor {
  return input.vectorNorm(ord, dim, keepdim);
}

/**
 * `norm(A, ord=None, dim=None, keepdim=False, dtype=None)`.
 *
 * **All four were missing and the method had all four.** This read `norm(input)` and
 * called `input.norm()`, so `linalg.norm(x, 2, 1)` — the line a torch tutorial writes —
 * threw away the order and the axis and returned the Frobenius norm of everything.
 * JavaScript discards surplus arguments without a word, so it was not an error but a
 * different number, of a different rank, that flows on and breaks somewhere unrelated.
 *
 * torch spells the first one `ord`; the method spells it `p`. Nothing else differs.
 */
export async function norm(input: Tensor, ord?: number, dim?: number,
                           keepdim = false, dtype?: DType): Promise<Tensor> {
  const x = dtype === undefined ? input : input.to(dtype);
  // **It dispatches, and forwarding to `Tensor.norm` was wrong.** The seats line up and
  // the meanings do not: with an `ord` and no `dim` on a matrix torch takes the
  // *largest singular value*, where the method takes the elementwise p-norm. Measured on
  // `[[1..9]]`: 16.848 against 16.882 — close enough to read as rounding and produced by
  // a different formula. The golden case added with this caught it on its first run.
  if (dim === undefined && ord !== undefined && x.shape.length === 2) {
    const m = await x.matrixNorm(ord);
    return keepdim ? m.reshape([1, 1]) : m;
  }
  // Everything else is a vector norm: no `ord` means the whole thing flattened, and an
  // `ord` with a `dim` means along that axis. torch's two-axis `dim` reaches
  // `matrix_norm` as well; borch.ts takes a single axis, so that spelling does not arise.
  return x.vectorNorm(ord ?? 2, dim, keepdim);
}

/** `cond(A, p=None)` — the condition number. */
export function cond(input: Tensor, p: number | string | null = null): Promise<Tensor> {
  return input.cond(p);
}

/** `cross(a, b, dim=-1)` — the cross product. */
export function cross(input: Tensor, other: Tensor, dim = -1): Tensor {
  return input.cross(other, dim);
}

/** `householder_product(A, tau)` — the product of the Householder reflectors. */
export function householderProduct(input: Tensor, tau: Tensor): Promise<Tensor> {
  return input.householderProduct(tau);
}

/** `lu_factor(A)` — the packed LU factors and the pivots. */
export function luFactor(a: Tensor): Promise<{ LU: Tensor; pivots: Tensor }> {
  return a.luFactor();
}

/**
 * `lu(A, *, pivot=True)` — the LU decomposition, expanded into `P`, `L` and `U`.
 *
 * **`pivot` is carried in order to refuse it**, which is what the core does with the
 * same argument. Without pivoting the factorisation is a different one and exists only
 * where no pivot is ever needed; there is no such path here. Left out of the signature
 * the word would be discarded by JavaScript and the pivoted answer returned under its
 * name — a wrong factorisation that looks like a right one.
 */
export function lu(A: Tensor,
                   pivot = true): Promise<{ P: Tensor; L: Tensor; U: Tensor }> {
  if (!pivot) {
    throw new RuntimeError(
      "lu(pivot=false) is not in the browser subset — the factorisation without "
      + "pivoting is a different one and only the pivoted answer is computed here.");
  }
  return A.lu();
}

/** `vander(x, N=None)` — the Vandermonde matrix. */
export function vander(x: Tensor, n?: number): Tensor {
  return x.vander(n);
}

// ── Spelled differently, same operation ─────────────────────────────────

/** `inv(A)` — the inverse. The method is `inverse`. */
export function inv(a: Tensor): Promise<Tensor> {
  return a.inverse();
}

/** `pinv(A)` — the Moore-Penrose pseudo-inverse. The method is `pinverse`. */
export function pinv(input: Tensor, rcond?: number): Promise<Tensor> {
  // **Carried in order to refuse, and now carried in order to compute.** The seat had
  // to exist here for `pinverse`'s stop to be reachable at all — left out, the door was
  // narrower than the room behind it and `linalg.pinv(a, 1e-6)` discarded the number in
  // silence. `pinverse` takes the cut-off now, so the same seat delivers it, and the
  // core has been giving torch's answer for this all along.
  return input.pinverse(rcond);
}

/** `matmul(A, B)` — matrix multiplication. The method is `mm`. */
export function matmul(input: Tensor, other: Tensor): Tensor {
  return input.mm(other);
}

// ── The two whose receiver is not the first argument ────────────────────

/**
 * `lu_solve(LU, pivots, B)` — solve using factors from `lu_factor`.
 *
 * **The factors come first here and the method is received by `B`.** Forwarding this one
 * positionally would swap `LU` and `B`, and with square matrices nothing about the call
 * would look wrong.
 *
 * **`left` and `adjoint` were carried in order to refuse, and now they answer.** The
 * refusal said each solves a different system than the one these factors were made
 * for, which was true and was also the reason it could be closed: `A = P L U` gives
 * `Aᵀ = Uᵀ Lᵀ Pᵀ`, so the adjoint is the same three pieces in the other order, and
 * `X A = B` is `Aᵀ Xᵀ = Bᵀ` — the right-hand solve is the left one with the sides
 * transposed and the flag flipped. All four combinations are measured against torch.
 */
export function luSolve(LU: Tensor, pivots: Tensor, b: Tensor,
                        left = true, adjoint = false): Promise<Tensor> {
  return LU.luSolveFactored(pivots, b, left, adjoint);
}

/**
 * `diagonal(A, offset=0, dim1=-2, dim2=-1)`.
 *
 * **The defaults are not the method's.** `torch.diagonal` reads axes `0, 1` and
 * `torch.linalg.diagonal` reads `-2, -1`; given `(2,3,4)` that is `(4,2)` against `(2,3)`.
 * The method takes the axes as arguments precisely so both spellings can be right.
 */
export function diagonal(a: Tensor, offset = 0, dim1 = -2, dim2 = -1): Tensor {
  return a.diagonal(offset, dim1, dim2);
}

// ── Built here, out of the ones above ───────────────────────────────────
//
// One name only. Everything else in this file forwards, and a second implementation of
// arithmetic the methods already carry is a day when only one of the two gets fixed.

/**
 * `multi_dot([A, B, C, …])` — a chain of matrix products.
 *
 * Matrix multiplication is associative, so **every order gives the same matrix**; what
 * differs is the work. `(A·B)·C` and `A·(B·C)` cost `n·m·p + n·p·q` and `m·p·q + n·m·q`,
 * which for a thin middle factor differ by orders of magnitude. So the cheapest
 * parenthesisation is found first, by the usual dynamic program, which is the whole reason
 * torch offers this name at all.
 *
 * **The values are not bit-identical across orders.** Floating-point addition does not
 * associate, so the chosen order is part of the answer to a last-digit comparison.
 */
export function multiDot(tensors: readonly Tensor[]): Tensor {
  if (tensors.length === 0) throw new RuntimeError("linalg.multiDot: the tensors is empty");
  if (tensors.length === 1) return tensors[0]!;
  if (tensors.length === 2) return tensors[0]!.mm(tensors[1]!);

  // `dims[i]` is the row count of factor `i`, and the last entry its column count.
  const dims: number[] = [];
  for (const t of tensors) {
    if (t.shape.length !== 2) {
      throw new RuntimeError(
        `linalg.multiDot: every factor has to be a matrix — got [${t.shape}]`);
    }
    dims.push(t.shape[0]!);
  }
  dims.push(tensors[tensors.length - 1]!.shape[1]!);

  const n = tensors.length;
  const cost = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const split = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let len = 2; len <= n; len++) {
    for (let i = 0; i + len - 1 < n; i++) {
      const j = i + len - 1;
      cost[i]![j] = Infinity;
      for (let k = i; k < j; k++) {
        const here = cost[i]![k]! + cost[k + 1]![j]!
          + dims[i]! * dims[k + 1]! * dims[j + 1]!;
        if (here < cost[i]![j]!) {
          cost[i]![j] = here;
          split[i]![j] = k;
        }
      }
    }
  }
  const walk = (i: number, j: number): Tensor => {
    if (i === j) return tensors[i]!;
    const k = split[i]![j]!;
    return walk(i, k).mm(walk(k + 1, j));
  };
  return walk(0, n - 1);
}

/**
 * `tensorsolve(A, B)` — solve `A x = B` where `A` and `B` carry more than two axes.
 *
 * The method is `tensorSolve`. **It was already there and this file nearly reimplemented
 * it**, because the search for it used torch's spelling (`tensorsolve`) against the
 * camelCase method names and matched nothing. `test_binding_fills_in.py` caught the two
 * declarations colliding on one lookup key — the collision check another session added
 * this morning, after a normaliser folded `eq_` onto `eq`.
 */
export function tensorsolve(input: Tensor, b: Tensor,
                            dims?: readonly number[]): Promise<Tensor> {
  // **`dims` was carried in order to refuse and now it answers.** It moves the named
  // axes of `A` to the end before the fold, so the matrix that gets solved is a
  // different one and so is the answer's shape — which is a permute away, and the
  // permute was already here.
  return input.tensorSolve(b, dims);
}

/**
 * `tensorinv(A, ind=2)` — the inverse of `A` folded into a matrix at axis `ind`.
 *
 * The method is `tensorInv`, and it was already there too. See `tensorsolve` above.
 */
export function tensorinv(input: Tensor, ind = 2): Promise<Tensor> {
  return input.tensorInv(ind);
}
