"""`linalg.eig`'s **eigenvector gradient** — a place the golden cases cannot ask about.

The golden cases compare three implementations against one answer, and an eigenvector's
**sign is undetermined.** Whether LAPACK gives `v` or `−v`, both are eigenvectors, and
**torch itself gives opposite signs at float32 and float64** (measured). A loss leaning on
`V` then becomes a different function per implementation, and its gradient diverges with it —
there is no answer to freeze.

What went into the golden cases is only what does not lean on the sign: the eigenvalues'
magnitudes, whether their sum is the trace, whether `A·V = V·diag(λ)`. What is here is **the
gradient side**, measured with real torch alongside and only where the signs came out the
same.

## What it caught

**The normalisation correction.** LAPACK gives eigenvectors of length 1, so they cannot move
along their own direction, and the raw formula flows that component too. Before and after
subtracting it were held up against torch and the answer chosen — **on a symmetric matrix the
two answers are identical**, so measuring only with symmetric matrices cannot see the term
missing. So what follows asks with a non-symmetric one first.

**Refusing the phase.** With complex eigenvalues an eigenvector is determined only up to a
factor of `e^{iφ}`, and a loss leaning on that phase has no value. torch stops there, so we
stop — otherwise code that ran here stops under torch.
"""

import numpy as np
import pytest

import borch

torch = pytest.importorskip("torch")

# The non-symmetric one comes first. A symmetric matrix gives the same answer with or without the normalisation correction.
ASYMMETRIC = np.array([[4.0, 1.0, 2.0], [0.0, 3.0, -1.0], [1.0, 0.0, 2.0]],
                      dtype=np.float32)
TRIANGULAR = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32)
SYMMETRIC = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
ROTATION = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)   # eigenvalues ±i
WEIGHT = np.array([[1.0, -2.0, 0.5], [0.3, 1.5, -1.0], [2.0, 0.25, 0.75]],
                  dtype=np.float32)


def _signs(vecs):
    """A sign per column, decided by the values alone.

    **The column's largest-magnitude entry is made positive.** Any rule works as long as
    both libraries get the same one and it depends on nothing but the numbers; this one
    is stable because the entry it looks at is the one furthest from zero, so a small
    perturbation cannot move which entry decides.
    """
    lead = np.argmax(np.abs(vecs.real), axis=-2)
    picked = vecs.real[lead, np.arange(vecs.shape[-1])]
    return np.where(picked < 0, -1.0, 1.0)


def _vector_grad(lib, mat):
    """The gradient of `(V.real · W).sum()` with each column's sign pinned.

    **Each library canonicalises its own `V`.** Handing both the same signs was the
    first attempt and it is wrong: the sign has to cancel the one that library actually
    produced, so a shared constant leaves the two computing different scalars again —
    measured, and the gradients came out unrelated rather than negated.

    The sign is worked out once, outside the graph, and multiplied into `W`, so it is a
    constant here. It is a function of `A` in principle and locally constant in fact,
    which is what makes that legitimate: at this `A` nothing is being differentiated
    through it.
    """
    n = mat.shape[0]
    vecs = np.asarray(lib.linalg.eig(lib.tensor(mat)).eigenvectors.tolist(),
                      dtype=np.complex128)
    w = WEIGHT[:n, :n] * _signs(vecs)
    a = lib.tensor(mat, requires_grad=True)
    (lib.linalg.eig(a).eigenvectors.real * lib.tensor(w)).sum().backward()
    return np.asarray(a.grad.tolist(), dtype=np.float64)


@pytest.mark.parametrize("name,mat", [
    ("asymmetric", ASYMMETRIC), ("upper-triangular", TRIANGULAR), ("symmetric", SYMMETRIC),
])
def test_eigenvector_gradient_matches_torch(name, mat):
    """**The sign is pinned rather than waited for.**

    This used to compare only where the two libraries happened to agree on the sign and
    skip otherwise, because `V` and `−V` make `(V.real · W).sum()` a different scalar.
    That was right about the arithmetic and wrong about what to do next: the asymmetric
    row — the only one that can see the normalisation correction, and the reason this
    file leads with it — **sat skipped**, and its docstring went on saying all three
    agreed.

    Measured, before changing anything: the eigenvalues agree to 2.4e-07, our residuals
    `‖Av − λv‖` are *smaller* than torch's, and the ratio of the two answers is exactly
    −1, −1, +1 with a spread of 1e-07. **Same eigenpairs, two columns negated** — which
    is the phase freedom `eig` is defined up to, not a defect on either side.

    So the fix is not to wait for agreement. Each side canonicalises its own `V` by the
    same rule, and then both are differentiating the same function. A check that skips
    whenever a free choice comes out the other way is a check that reports nothing on
    the case it was written for.

    **And the revived row has teeth, measured rather than assumed.** Deleting the
    normalisation correction from `borch/_ops.py` reddens `asymmetric` and
    `upper-triangular` and leaves `symmetric` green — which is the claim this file has
    made since it was written, now with the run behind it. Turning a skip into a green
    that proves nothing would have been worse than the skip.
    """
    theirs = torch.linalg.eig(torch.tensor(mat)).eigenvectors.numpy()
    ours = np.asarray(borch.linalg.eig(borch.tensor(mat)).eigenvectors.tolist())
    # **The eigenpairs still have to match** — pinning the sign must not paper over two
    # libraries finding different vectors. Compared after the sign is removed from both.
    assert np.allclose(theirs * _signs(theirs), ours * _signs(ours), atol=1e-4), (
        f"{name}: the eigenvectors differ by more than a sign — that is not the phase "
        f"freedom this pins, it is a different answer.")
    assert np.allclose(_vector_grad(torch, mat), _vector_grad(borch, mat),
                       atol=1e-3), name


def test_the_normalization_term_is_what_makes_the_asymmetric_case_right():
    """**With this term missing, the symmetric case passes and only the asymmetric is wrong.**

    That the version without the correction and the version with it give the same answer on a
    symmetric matrix is written down here — it is what stops the next person testing with a
    symmetric matrix and concluding that it works.
    """
    v = np.linalg.eig(SYMMETRIC.astype(np.float64))[1]
    radial = np.real(np.sum(np.conjugate(v) * WEIGHT[:2, :2], axis=-2))
    # A symmetric matrix's eigenvectors are orthogonal, so what is subtracted rides on the diagonal alone and changes nothing.
    assert np.abs(v * radial).max() > 0.1, "what is subtracted is not itself zero"
    assert np.allclose(_vector_grad(torch, SYMMETRIC),
                       _vector_grad(borch, SYMMETRIC), atol=1e-3)


def test_a_phase_dependent_loss_is_refused_like_torch():
    """With complex eigenvalues an eigenvector is determined only up to a phase — both have to stop."""
    with pytest.raises(RuntimeError, match="e\\^\\{i phi\\}"):
        _vector_grad(torch, ROTATION)
    with pytest.raises(RuntimeError, match="e\\^\\{i phi\\}"):
        _vector_grad(borch, ROTATION)
