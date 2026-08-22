/**
 * Exceptions shaped like torch's.
 *
 * ## Why the names are imitated
 *
 * Somebody using this library used torch before. What they do when stuck is search for the
 * error text verbatim, and what comes back is torch's documentation and its questions.
 * With only our wording that search does not work — the answer already exists in the world
 * and cannot be reached.
 *
 * So **the text torch produces under the same condition goes in verbatim** and our
 * explanation follows it. The front half is for searching and the back half is this
 * place's own circumstance.
 *
 * ## How far the imitation goes
 *
 * Only where there is a corresponding failure in torch. Failures that are ours alone (WGSL
 * compilation, dispatch limits, a feature that does not exist yet) are written in our own
 * words — inventing a torch message that does not exist does not make the search work, it
 * sends it somewhere else.
 */

/**
 * Present in Python, absent in JavaScript. The golden cases froze the
 * **kind name** of the exception as well (`RuntimeError|message=True`), so
 * the class name is part of the answer.
 */
export class RuntimeError extends Error {
  override readonly name = "RuntimeError";
}

/**
 * An index past the end.
 *
 * **Not a `RuntimeError`.** torch uses Python's `IndexError` here, and the
 * golden cases froze the kind of exception too, so it has to be that name.
 * Blurring the kinds means code catching `except IndexError` stops catching
 * — the kind of an exception is part of the API.
 */
export class IndexError extends Error {
  override readonly name = "IndexError";
}

/**
 * A combination that does not exist yet.
 *
 * **Refusals have kinds too.** torch raises "this padding parity at this
 * rank is not supported" as `NotImplementedError`, and that says something
 * different from "the value is wrong" (`RuntimeError`) — the first is
 * something that may exist one day, the second is the caller being wrong.
 */
export class NotImplementedError extends Error {
  override readonly name = "NotImplementedError";
}

/**
 * An argument that makes no sense on its own — two mutually exclusive
 * options given together, or a value outside the domain.
 *
 * **It says something different from `RuntimeError`.** Python uses
 * `ValueError` in this position and so does torch (`FractionalMaxPool2d`
 * stops with that kind when given both a size and a ratio). The golden
 * cases freeze the **kind name** of the exception as the answer, so
 * throwing a different name here leaves the binding nothing to carry
 * across.
 */
export class ValueError extends Error {
  override readonly name = "ValueError";
}

/**
 * Linear algebra with no answer to give — a singular matrix, a Cholesky
 * that is not positive definite.
 *
 * **The name does work.** Code that can meet a singular matrix usually
 * wraps the call in `except LinAlgError`, and throwing a plain
 * `RuntimeError` walks straight past that wrapper and kills the program.
 * Same reason `IndexError` is kept separate — **the kind of an exception is
 * part of the API.**
 */
export class LinAlgError extends Error {
  override readonly name = "LinAlgError";
}

/**
 * The wording torch itself produces. Kept verbatim so that searching for it
 * works.
 */
export const TORCH = {
  matmulShape: "shapes cannot be multiplied",
  // **Something this subset has no cell for.** It is our wording rather than torch's,
  // and the reason it lives here is the same — three implementations stop with the same
  // sentence, and the golden looks for **a fragment** of it. Written out by hand at each
  // site, each implementation grows a different sentence, and somebody learning reads that
  // as a different rule.
  //
  // **These two lines were Korean, and that is why twenty-one cases were falsely green.**
  // The Python side (`_unsupported` in `borch/_base.py`) was translated to English and
  // this side was not. What the golden froze is the **verdict word** `기대대로`, and each
  // case only looks for its own side's fragment — so however far the two sentences drift,
  // each agrees with itself and all twenty-one pass. It was self-comparison rather than
  // mutual comparison.
  //
  // The sentence is **copied verbatim** from `_base.py`. Derived, it diverges again.
  // No leading space — the call site (`tensor.ts`) already puts one in, so adding one here
  // gives two. Python is `f"{what} is not in the browser subset."` with a single space.
  absent: "is not in the browser subset.",
  absentAdvice: "Use real PyTorch on your own machine (`uv add torch`) — this subset "
    + "is for practising the syntax, and imitating what is missing teaches the wrong thing.",
  broadcast: "must match the size of tensor",
  reshapeSize: "is invalid for input of size",
  nonScalarBackward: "grad can be implicitly created only for scalar outputs",
  noGrad: "does not require grad",
  // torch produces "a Tensor with 3 elements cannot be converted to Scalar". The
  // fragment the golden looks for is the back half, so the count is filled in per site.
  itemScalar: "cannot be converted to Scalar",
  // torch continues with "... but found at least two devices, cuda:0 and cpu!". The
  // device names differ per site, so only the head lives here.
  crossDevice: "Expected all tensors to be on the same device",
  // When a mismatched seed is passed to a non-scalar backward. torch appends the two
  // shapes.
  gradShape: "Mismatch in shape",
  secondBackward:
    "Trying to backward through the graph a second time (or directly access " +
    "saved tensors after they have already been freed)",
} as const;
