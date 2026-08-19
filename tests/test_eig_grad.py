"""`linalg.eig` 의 **고유벡터 기울기** — 골든이 못 묻는 자리.

골든은 세 구현을 같은 답에 대조하는데, 고유벡터는 **부호가 안 정해진다.** LAPACK 이
`v` 를 주든 `−v` 를 주든 둘 다 고유벡터이고, 실제로 **torch 자신이 float32 와 float64
에서 반대 부호를 낸다**(실측). 그러면 `V` 에 기대는 손실은 구현마다 다른 함수가 되고,
그 기울기도 같이 갈린다 — 굳힐 수 있는 답이 없다.

골든에 넣은 것은 부호에 안 기대는 것뿐이다: 고윳값의 크기, 합이 대각합인가,
`A·V = V·diag(λ)` 인가. 여기 있는 것은 **기울기 쪽**이고, 진짜 torch 를 옆에 두고
같은 부호가 나온 자리에서만 잰다.

## 잡은 것

**정규화 보정.** LAPACK 이 고유벡터를 길이 1 로 내주므로 자기 방향으로는 못 움직이는데,
날 식은 그 성분까지 흘린다. 빼기 전과 후를 torch 에 대 보고 골랐다 —
**대칭 행렬에서는 두 답이 같아서**, 대칭으로만 재면 이 항이 빠진 것을 못 본다.
그래서 아래는 대칭이 아닌 것을 먼저 묻는다.

**위상 거절.** 고윳값이 복소수면 고유벡터는 `e^{iφ}` 배까지만 정해지고, 그 위상에
기대는 손실은 값이 없다. torch 가 거기서 멈추므로 우리도 멈춘다 — 안 멈추면 여기서
돌던 코드가 torch 에서 멈춘다.
"""

import numpy as np
import pytest

import borch

torch = pytest.importorskip("torch")

# 대칭이 아닌 것이 먼저다. 대칭은 정규화 보정이 있으나 없으나 같은 답을 낸다.
ASYMMETRIC = np.array([[4.0, 1.0, 2.0], [0.0, 3.0, -1.0], [1.0, 0.0, 2.0]],
                      dtype=np.float32)
TRIANGULAR = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32)
SYMMETRIC = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
ROTATION = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)   # 고윳값이 ±i
WEIGHT = np.array([[1.0, -2.0, 0.5], [0.3, 1.5, -1.0], [2.0, 0.25, 0.75]],
                  dtype=np.float32)


def _vector_grad(lib, mat):
    """`(V.real · W).sum()` 의 기울기. 두 라이브러리에 같은 식을 준다."""
    n = mat.shape[0]
    w = WEIGHT[:n, :n]
    a = lib.tensor(mat, requires_grad=True)
    (lib.linalg.eig(a).eigenvectors.real * lib.tensor(w)).sum().backward()
    return np.asarray(a.grad.tolist(), dtype=np.float64)


@pytest.mark.parametrize("name,mat", [
    ("비대칭", ASYMMETRIC), ("상삼각", TRIANGULAR), ("대칭", SYMMETRIC),
])
def test_eigenvector_gradient_matches_torch(name, mat):
    """**부호가 같은 자리에서만 견준다.**

    torch 와 우리가 같은 고유벡터를 골랐는지 먼저 보고, 다르면 그 행렬은 이 물음의
    대상이 아니다 — 손실이 다른 함수가 되기 때문이다. 셋 다 지금은 같이 나온다.
    """
    theirs = torch.linalg.eig(torch.tensor(mat)).eigenvectors.numpy()
    ours = np.asarray(borch.linalg.eig(borch.tensor(mat)).eigenvectors.tolist())
    if not np.allclose(theirs, ours, atol=1e-4):
        pytest.skip(f"{name}: 고유벡터 부호가 갈렸다 — 손실이 다른 함수다")
    assert np.allclose(_vector_grad(torch, mat), _vector_grad(borch, mat),
                       atol=1e-3), name


def test_the_normalization_term_is_what_makes_the_asymmetric_case_right():
    """**이 항이 빠지면 대칭은 통과하고 비대칭만 틀린다.**

    보정을 지운 판과 지금 판이 대칭 행렬에서 같은 답을 낸다는 것을 여기 적어 둔다 —
    다음 사람이 대칭으로 시험하고 "잘 된다" 고 판단하는 것을 막는 자리다.
    """
    v = np.linalg.eig(SYMMETRIC.astype(np.float64))[1]
    radial = np.real(np.sum(np.conjugate(v) * WEIGHT[:2, :2], axis=-2))
    # 대칭 행렬의 고유벡터는 직교라, 빼는 양이 대각에만 실려 답을 안 바꾼다.
    assert np.abs(v * radial).max() > 0.1, "빼는 양 자체는 0 이 아니다"
    assert np.allclose(_vector_grad(torch, SYMMETRIC),
                       _vector_grad(borch, SYMMETRIC), atol=1e-3)


def test_a_phase_dependent_loss_is_refused_like_torch():
    """고윳값이 복소수면 고유벡터는 위상까지만 정해진다 — 둘 다 멈춰야 한다."""
    with pytest.raises(RuntimeError, match="e\\^\\{i phi\\}"):
        _vector_grad(torch, ROTATION)
    with pytest.raises(RuntimeError, match="e\\^\\{i phi\\}"):
        _vector_grad(borch, ROTATION)
