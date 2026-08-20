"""borch-vision — torchvision 모양의 얇은 층. **`transforms` 만이다.**

## 왜 별도 파일인가

`torchvision.transforms` 이지 `torch.transforms` 가 아니다. 코어 안에 넣으면
`borch.transforms` 가 되는데 그건 진짜 torch 에 **없는 자리**다. 구조를 그대로
흉내 내는 것이 이 프로젝트의 요점인데, 없는 자리를 만들면 그 요점이 먼저 깨진다.

    import borchvision as torchvision
    from borchvision import transforms

## 어느 라이브러리의 텐서를 만드는가

텐서를 만드는 것은 `ToTensor` 하나다. 기본은 코어 `borch` 이고, 자매(WebGPU)
쪽에 붙이려면 `use(L)` 를 부른다.

## 없는 것과 그 이유

- **`datasets`** — 받아오는 쪽이 막혀 있다. 실측이다: `cs.toronto.edu` 가
  `Access-Control-Allow-Origin` 을 안 줘서 브라우저가 CIFAR 원본을 못 받는다. 부지런하면
  되는 종류가 아니다. 게다가 torch 의 `download=True` 는 `root` 에 받아두고 다음 실행에
  재사용하는데 Pyodide 의 파일시스템은 새로고침에 날아간다 — 같은 코드가 예외 없이
  다르게 구는 자리라 흉내 내면 안 된다. **바이트를 손에 넣은 뒤는 이미 된다**
  (`fetch_cached`·`cache_put`·`TensorDataset`)
- **`ops`** — `nms` 는 numpy 로 짧아서 "크다"는 이유는 거짓이다. 진짜 이유는 아무도 그
  앞에 안 선다는 것이다. 검출은 사전학습 백본과 COCO 급 데이터가 있어야 끝까지 가고,
  그때까지 `nms` 는 갈 곳 없는 정확한 함수로 남는다. 사용자 없는 표면은 케이스가 안
  생기고, **케이스 없는 표면이 이 저장소가 조용히 틀렸던 자리 전부였다**
- **사전학습 가중치** — `.pth` 는 zip 안의 pickle 이고 `torch.load` 는 모듈 경로로
  클래스를 되살린다. 읽으려면 torch 내부 구조를 흉내 내야 하고 미묘하게 틀리면 **모양
  맞는 가중치에 틀린 수**가 들어온다. 거기다 ResNet-18 만 45MB 이고, 무엇보다
  `pretrained=True` 가 돌면 사람들은 발표된 top-1 과 비교한다 — 비트 동등은 이 프로젝트의
  명시적 비목표라 지킬 수 없는 약속이다

## 난수는 torch 와 다르다 — 흉내 낼 수 없어서 적어 둔다

`RandomCrop`·`RandomHorizontalFlip` 은 torch 의 난수기를 못 쓴다. 그래서 **같은 씨앗을
줘도 torchvision 과 같은 장면이 나오지 않는다.** 값이 갈리는 것이 아니라 뽑기가 갈리는
것이다. 확률이 0 이나 1 로 고정된 자리는 결정적이므로 골든은 그쪽으로 대조한다.
"""

import numpy as _np

_rng = _np.random.default_rng()
_lib = None


def use(L):
    """`ToTensor` 가 만들 텐서의 라이브러리를 정한다. 자매 쪽에 붙일 때 부른다."""
    global _lib
    _lib = L
    return L


def _backend():
    global _lib
    if _lib is None:
        import borch as _core
        _lib = _core
    return _lib


def manual_seed(seed):
    """**torch 와 같은 장면을 주지 않는다.** 이 모듈 안에서만 재현된다."""
    global _rng
    _rng = _np.random.default_rng(seed)


# --- 뼈대: 축을 인자로 받는다 ------------------------------------------------
# 장당 변환은 (H,W,C) 를, 배치 변환은 (N,C,H,W) 를 다룬다. 자리가 다르다고 같은 일을
# 두 벌 쓰면 언젠가 갈리므로, 높이·너비가 **몇 번째 축인지**만 받아서 한 벌로 쓴다.

def _hflip(arr, w_axis):
    return _np.flip(arr, axis=w_axis)


def _crop(arr, h_axis, top, height, w_axis, left, width):
    idx = [slice(None)] * arr.ndim
    idx[h_axis] = slice(top, top + height)
    idx[w_axis] = slice(left, left + width)
    return arr[tuple(idx)]


def _pad_hw(arr, h_axis, w_axis, padding, fill):
    if not padding:
        return arr
    pads = [(0, 0)] * arr.ndim
    pads[h_axis] = pads[w_axis] = (padding, padding)
    return _np.pad(arr, pads, constant_values=fill)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        inner = "".join(f"\n    {t}" for t in self.transforms)
        return f"{type(self).__name__}({inner}\n)"


class ToTensor:
    """(H,W,C) 또는 (H,W) → (C,H,W) 텐서.

    **uint8 일 때만 255 로 나눈다** — torchvision 의 규칙이 그렇다. 실수 배열을 넣으면
    나누지 않고 그대로 옮긴다. 이 구분을 놓치면 이미 [0,1] 인 데이터가 한 번 더 나뉘어
    255 배 어두워지는데, 예외는 안 나고 학습만 안 된다.
    """

    def __call__(self, pic):
        arr = _np.asarray(pic)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(
                f"ToTensor 는 (H,W) 나 (H,W,C) 를 받습니다 — {arr.shape} 를 받았습니다.\n"
                "(torch: pic should be 2/3 dimensional)")
        out = arr.transpose(2, 0, 1)
        out = (out.astype(_np.float32) / 255.0 if arr.dtype == _np.uint8
               else out.astype(_np.float32))
        return _backend().tensor(_np.ascontiguousarray(out))

    def __repr__(self):
        return f"{type(self).__name__}()"


class Normalize:
    """채널마다 `(x - mean) / std`. 텐서와 numpy 배열을 둘 다 받는다.

    numpy 도 받는 이유는 배치를 한 번에 정규화하는 자리가 있어서다(`augment_batch` 참고).
    같은 산수를 두 벌로 쓰지 않으려고 축만 맞춰 한 식으로 둔다.
    """

    def __init__(self, mean, std, inplace=False):
        # 받은 것을 **그대로** 들고 있는다. torchvision 도 그러며, repr 이 튜플로 준 것을
        # 튜플로 리스트로 준 것을 리스트로 찍는 것이 거기서 나온다.
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, x):
        m = _np.atleast_1d(_np.asarray(self.mean, dtype=_np.float32))
        s = _np.atleast_1d(_np.asarray(self.std, dtype=_np.float32))
        shape = (m.size, 1, 1)
        m, s = m.reshape(shape), s.reshape(shape)
        if isinstance(x, _np.ndarray):
            return (x - m) / s
        L = _backend()
        return (x - L.tensor(m)) / L.tensor(s)

    def __repr__(self):
        return f"{type(self).__name__}(mean={self.mean}, std={self.std})"


class RandomHorizontalFlip:
    """좌우로 뒤집는다. **(H,W,C) numpy 배열을 받는다** — torchvision 에서 이 자리에
    오는 것이 PIL 이미지이고, 우리에게 PIL 이 없어서 그 자리를 배열이 대신한다.

    텐서를 넣으면 거절한다. 장당 텐서를 만들면 자매 쪽에서 GPU 버퍼가 장당 하나씩
    생기는데, 그건 되는 것처럼 보이다가 메모리로 무너지는 종류다. 순서를 바꾸라고
    말해주는 편이 낫다.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        if _rng.random() >= self.p:
            return img
        return _np.ascontiguousarray(_hflip(img, 1))

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p})"


class RandomCrop:
    """가장자리를 채운 뒤 무작위로 잘라낸다. `RandomHorizontalFlip` 과 같은 자리다."""

    def __init__(self, size, padding=0, fill=0):
        self.size = (int(size), int(size)) if isinstance(size, int) else tuple(size)
        self.padding = int(padding)
        self.fill = fill

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        img = _pad_hw(img, 0, 1, self.padding, self.fill)
        th, tw = self.size
        h, w = img.shape[0], img.shape[1]
        if h < th or w < tw:
            raise ValueError(
                f"자를 크기 {self.size} 가 이미지 {(h, w)} 보다 큽니다.\n"
                "(torch: Required crop size is larger than input image size)")
        top = int(_rng.integers(0, h - th + 1))
        left = int(_rng.integers(0, w - tw + 1))
        return _np.ascontiguousarray(_crop(img, 0, top, th, 1, left, tw))

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size}, padding={self.padding})"


def _require_hwc(img, who):
    if isinstance(img, _np.ndarray):
        return img
    raise TypeError(
        f"{who} 는 (H,W,C) numpy 배열을 받습니다 — 텐서가 왔습니다.\n"
        "  torchvision 에서 이 자리에 오는 것은 PIL 이미지입니다. `ToTensor()` 를\n"
        "  뒤에 두세요: Compose([RandomCrop(...), RandomHorizontalFlip(), ToTensor()])")


def augment_batch(x, crop=None, padding=0, hflip_p=0.0, fill=0.0):
    """**torchvision 에 없는 우리 것이다.** (N,C,H,W) 배치를 한 번에 늘린다.

    왜 필요한가. 장당 `ToTensor` 를 하면 자매 쪽에서 **GPU 버퍼가 장당 하나씩** 생긴다.
    한 에폭이 만 장이면 만 개다. 배치를 numpy 에서 다 늘린 뒤 텐서를 **하나** 만드는
    것이 감당되는 유일한 순서라, 그 순서를 이름 붙여 내놓는다.

    뽑기는 **장마다 따로** 한다. 배치 전체에 같은 자르기·뒤집기를 쓰면 배치 안에서는
    늘어난 것이 없어 augmentation 의 효과가 사라진다. torchvision 의 클래스들은 한 번
    부를 때 한 번 뽑으므로(배치를 주면 배치 전체가 같은 뽑기를 받는다) 여기서 갈린다 —
    그래서 torchvision 의 이름을 쓰지 않고 우리 이름을 쓴다.
    """
    if x.ndim != 4:
        raise ValueError(f"augment_batch 는 (N,C,H,W) 를 받습니다 — {x.shape} 를 받았습니다")
    out = _pad_hw(x, 2, 3, padding, fill)
    n, _, h, w = out.shape
    th, tw = (h, w) if crop is None else (
        (int(crop), int(crop)) if isinstance(crop, int) else tuple(crop))
    if h < th or w < tw:
        raise ValueError(f"자를 크기 {(th, tw)} 가 이미지 {(h, w)} 보다 큽니다")

    tops = _rng.integers(0, h - th + 1, size=n)
    lefts = _rng.integers(0, w - tw + 1, size=n)
    flips = _rng.random(n) < hflip_p
    pieces = _np.empty((n, x.shape[1], th, tw), dtype=x.dtype)
    for i in range(n):
        one = _crop(out[i], 1, int(tops[i]), th, 2, int(lefts[i]), tw)
        pieces[i] = _hflip(one, 2) if flips[i] else one
    return pieces


class _Transforms:
    """`from borchvision import transforms` 를 위한 자리. torchvision 의
    모듈 구조를 그대로 두려고 이름만 빌린다."""


transforms = _Transforms()
for _name in ("Compose", "ToTensor", "Normalize", "RandomHorizontalFlip", "RandomCrop"):
    setattr(transforms, _name, globals()[_name])
