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
export function matrixPower(a: Tensor, n: number): Tensor {
  return a.matrixPower(n);
}

/** `matrix_rank(A)` — how many singular values are above the tolerance. */
export function matrixRank(a: Tensor): Promise<Tensor> {
  return a.matrixRank();
}

/** `matrix_exp(A)` — the matrix exponential. */
export function matrixExp(a: Tensor): Promise<Tensor> {
  return a.matrixExp();
}

/** `cholesky(A, upper=False)` — the Cholesky factor. */
export function cholesky(a: Tensor, upper = false): Promise<Tensor> {
  return a.cholesky(upper);
}

/** `qr(A, mode="reduced")` — the QR decomposition. */
export function qr(
  a: Tensor, mode: "reduced" | "complete" = "reduced",
): Promise<{ q: Tensor; r: Tensor }> {
  return a.qr(mode);
}

/** `svd(A, full_matrices=True)` — the singular value decomposition. */
export function svd(
  a: Tensor, fullMatrices = true,
): Promise<{ u: Tensor; s: Tensor; vt: Tensor }> {
  return a.svd(fullMatrices);
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
export function eigvals(a: Tensor): Promise<Tensor> {
  return a.eigvals();
}

/** `eigh(A, UPLO="L")` — the symmetric (Hermitian) eigendecomposition. */
export function eigh(
  a: Tensor, uplo: "L" | "U" = "L",
): Promise<{ values: Tensor; vectors: Tensor }> {
  return a.eigh(uplo);
}

/** `eigvalsh(A, UPLO="L")` — the symmetric eigenvalues alone. */
export function eigvalsh(a: Tensor, uplo: "L" | "U" = "L"): Promise<Tensor> {
  return a.eigvalsh(uplo);
}

/** `solve(A, B)` — solve `A x = B`. */
export function solve(a: Tensor, b: Tensor): Promise<Tensor> {
  return a.solve(b);
}

/** `solve_triangular(A, B, upper, left=True, unitriangular=False)`. */
export function solveTriangular(
  a: Tensor, b: Tensor, upper: boolean, left = true, unitriangular = false,
): Promise<Tensor> {
  return a.solveTriangular(b, upper, left, unitriangular);
}

/** `lstsq(A, B)` — the least-squares solution. */
export function lstsq(a: Tensor, b: Tensor): Promise<Tensor> {
  return a.lstsq(b);
}

/** `matrix_norm(A, ord="fro")`. */
export function matrixNorm(a: Tensor, ord: number | string = "fro"): Promise<Tensor> {
  return a.matrixNorm(ord);
}

/** `vector_norm(x, ord=2, dim=None)`. */
export function vectorNorm(x: Tensor, ord = 2, dim?: number): Tensor {
  return x.vectorNorm(ord, dim);
}

/** `norm(A)` — the Frobenius norm of the whole tensor. */
export function norm(a: Tensor): Tensor {
  return a.norm();
}

/** `cond(A, p=None)` — the condition number. */
export function cond(a: Tensor, p: number | string | null = null): Promise<Tensor> {
  return a.cond(p);
}

/** `cross(a, b, dim=-1)` — the cross product. */
export function cross(a: Tensor, b: Tensor, dim = -1): Tensor {
  return a.cross(b, dim);
}

/** `householder_product(A, tau)` — the product of the Householder reflectors. */
export function householderProduct(a: Tensor, tau: Tensor): Promise<Tensor> {
  return a.householderProduct(tau);
}

/** `lu_factor(A)` — the packed LU factors and the pivots. */
export function luFactor(a: Tensor): Promise<{ LU: Tensor; pivots: Tensor }> {
  return a.luFactor();
}

/** `lu(A)` — the LU decomposition, expanded into `P`, `L` and `U`. */
export function lu(a: Tensor): Promise<{ P: Tensor; L: Tensor; U: Tensor }> {
  return a.lu();
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
export function pinv(a: Tensor): Promise<Tensor> {
  return a.pinverse();
}

/** `matmul(A, B)` — matrix multiplication. The method is `mm`. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  return a.mm(b);
}

// ── The two whose receiver is not the first argument ────────────────────

/**
 * `lu_solve(LU, pivots, B)` — solve using factors from `lu_factor`.
 *
 * **The factors come first here and the method is received by `B`.** Forwarding this one
 * positionally would swap `LU` and `B`, and with square matrices nothing about the call
 * would look wrong.
 */
export function luSolve(luData: Tensor, pivots: Tensor, b: Tensor): Promise<Tensor> {
  return luData.luSolveFactored(pivots, b);
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
export function multiDot(chain: readonly Tensor[]): Tensor {
  if (chain.length === 0) throw new RuntimeError("linalg.multiDot: the chain is empty");
  if (chain.length === 1) return chain[0]!;
  if (chain.length === 2) return chain[0]!.mm(chain[1]!);

  // `dims[i]` is the row count of factor `i`, and the last entry its column count.
  const dims: number[] = [];
  for (const t of chain) {
    if (t.shape.length !== 2) {
      throw new RuntimeError(
        `linalg.multiDot: every factor has to be a matrix — got [${t.shape}]`);
    }
    dims.push(t.shape[0]!);
  }
  dims.push(chain[chain.length - 1]!.shape[1]!);

  const n = chain.length;
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
    if (i === j) return chain[i]!;
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
export function tensorsolve(a: Tensor, b: Tensor): Promise<Tensor> {
  return a.tensorSolve(b);
}

/**
 * `tensorinv(A, ind=2)` — the inverse of `A` folded into a matrix at axis `ind`.
 *
 * The method is `tensorInv`, and it was already there too. See `tensorsolve` above.
 */
export function tensorinv(a: Tensor, ind = 2): Promise<Tensor> {
  return a.tensorInv(ind);
}
