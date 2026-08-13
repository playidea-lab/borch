"""`browsertorch_vision` 에서 **골든이 못 보는 부분**을 본다.

골든은 진짜 torchvision 과 값을 맞춰보는데, 무작위 변환은 그럴 수가 없다 — torch 의
난수기를 우리가 못 쓰기 때문이다. 그래서 골든은 확률을 0·1 로 못 박거나 자를 자리가
하나뿐인 경우만 묻는다.

그러면 **뽑기가 실제로 도는지는 아무도 안 본 채** 남는다. 뽑기가 고장나도 (예: 항상
같은 자리를 자르거나, 배치 전체에 같은 뽑기를 쓰거나) 골든은 초록이다. 여기가 그 자리다.
"""

import pathlib
import sys

import numpy as np
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent


if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import browsertorch as BT                                            # noqa: E402
import browsertorch_vision as V                                      # noqa: E402

V.use(BT)

# 한쪽 끝에만 표시를 둔 그림. 뒤집히면 표시가 반대쪽으로 간다 — 뒤집혔는지를
# 값 하나로 판정할 수 있어서 이 모양을 쓴다.
_MARKED = np.zeros((4, 4, 3), dtype=np.uint8)
_MARKED[:, 0, :] = 255


def _flipped(img):
    return bool(img[0, -1, 0] == 255)


def test_flip_with_half_probability_produces_both_outcomes():
    """p=0.5 인데 한쪽만 나오면 **뽑기가 아니라 상수다.** 골든은 그것을 못 본다."""
    V.manual_seed(0)
    flip = V.RandomHorizontalFlip(p=0.5)
    seen = {_flipped(flip(_MARKED)) for _ in range(60)}
    assert seen == {True, False}, f"60번 뽑았는데 한쪽만 나왔다: {seen}"


def test_crop_with_padding_visits_more_than_one_offset():
    """자를 자리가 여럿인데 늘 같은 자리를 자르면 augmentation 이 아니다."""
    V.manual_seed(0)
    crop = V.RandomCrop(4, padding=2)
    seen = {crop(_MARKED).tobytes() for _ in range(60)}
    assert len(seen) > 1, "60번 잘랐는데 결과가 한 가지뿐이다 — 뽑기가 죽었다"


def test_manual_seed_makes_the_same_draws_again():
    """torch 와 같은 장면은 못 주지만, **우리 안에서는** 재현되어야 한다."""
    def draw():
        V.manual_seed(7)
        crop = V.RandomCrop(4, padding=2)
        return [crop(_MARKED).tobytes() for _ in range(10)]

    assert draw() == draw()


def test_augment_batch_draws_per_image_not_once_per_batch():
    """`augment_batch` 의 docstring 이 주장하는 바로 그것.

    배치 전체에 같은 뽑기를 쓰면 배치 안에서는 늘어난 것이 없다. torchvision 의
    클래스들은 한 번 부를 때 한 번 뽑으므로 여기서 갈리고, 그래서 이름을 따로 뒀다.
    주장을 적어만 두고 안 재면 다음 사람이 그 주장을 믿는다.
    """
    V.manual_seed(0)
    x = np.zeros((64, 1, 4, 4), dtype=np.float32)
    x[:, :, :, 0] = 1.0                                  # 왼쪽 끝에만 표시
    out = V.augment_batch(x, crop=4, padding=0, hflip_p=0.5)
    flipped = out[:, 0, 0, -1] == 1.0
    assert flipped.any() and not flipped.all(), (
        f"64장 중 뒤집힌 것이 {int(flipped.sum())}장 — 배치 전체가 같은 뽑기를 받았다")


def test_augment_batch_keeps_shape_and_dtype():
    x = np.zeros((5, 3, 8, 8), dtype=np.float32)
    out = V.augment_batch(x, crop=8, padding=4, hflip_p=0.5)
    assert out.shape == (5, 3, 8, 8)
    assert out.dtype == np.float32


def test_augment_batch_rejects_wrong_rank():
    with pytest.raises(ValueError, match="N,C,H,W"):
        V.augment_batch(np.zeros((3, 8, 8), dtype=np.float32))


def test_crop_given_a_tensor_says_where_to_put_totensor():
    """텐서를 넣으면 거절한다 — 장당 텐서를 만들면 자매 쪽에서 GPU 버퍼가 장당 생긴다.

    거절 자체보다 **무엇을 해야 하는지 말해주는가**를 본다. 이 프로젝트의 오류 메시지
    규격이 그렇다.
    """
    with pytest.raises(TypeError, match="ToTensor"):
        V.RandomCrop(4)(BT.tensor(np.zeros((3, 4, 4), dtype=np.float32)))


def test_totensor_does_not_divide_a_float_image():
    """uint8 만 255 로 나눈다. 실수를 한 번 더 나누면 **예외 없이** 255배 어두워지고
    학습만 조용히 안 된다 — 값으로 붙잡는다."""
    img = np.full((2, 2, 3), 0.5, dtype=np.float32)
    assert np.allclose(V.ToTensor()(img).numpy(), 0.5)


def test_normalize_accepts_numpy_and_tensor_alike():
    """배치를 numpy 로 정규화하는 길과 텐서로 하는 길이 **같은 답**을 내야 한다.
    두 길이 갈리면 학습 파이프라인과 튜토리얼이 다른 것을 배운다."""
    arr = np.random.default_rng(0).random((3, 4, 4)).astype(np.float32)
    norm = V.Normalize((0.5, 0.4, 0.3), (0.2, 0.3, 0.4))
    assert np.allclose(norm(arr), norm(BT.tensor(arr)).numpy(), atol=1e-6)
