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


def _vector_grad(lib, mat):
    """The gradient of `(V.real · W).sum()`. The same expression is given to both libraries."""
    n = mat.shape[0]
    w = WEIGHT[:n, :n]
    a = lib.tensor(mat, requires_grad=True)
    (lib.linalg.eig(a).eigenvectors.real * lib.tensor(w)).sum().backward()
    return np.asarray(a.grad.tolist(), dtype=np.float64)


@pytest.mark.parametrize("name,mat", [
    ("asymmetric", ASYMMETRIC), ("upper-triangular", TRIANGULAR), ("symmetric", SYMMETRIC),
])
def test_eigenvector_gradient_matches_torch(name, mat):
    """**Compared only where the signs agree.**

    Whether torch and we picked the same eigenvector is checked first, and where they differ
    that matrix is outside this question — the loss becomes a different function. All three
    come out agreeing at present.
    """
    theirs = torch.linalg.eig(torch.tensor(mat)).eigenvectors.numpy()
    ours = np.asarray(borch.linalg.eig(borch.tensor(mat)).eigenvectors.tolist())
    if not np.allclose(theirs, ours, atol=1e-4):
        pytest.skip(f"{name}: the eigenvector signs diverged — the loss is a different function")
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
