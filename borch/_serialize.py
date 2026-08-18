"""`torch.save` · `torch.load` — **safetensors** 형식.

## 왜 이것이 필요한가

`state_dict()` 는 셋 다 있는데 **그것을 파일로 쓸 방법이 파이썬 쪽에 없었다.**
튜토리얼은 거의 다 저장으로 끝나고, 브라우저는 탭을 새로고침하면 학습 결과가
사라지는 자리라 데스크톱보다 오히려 더 아쉬운 구멍이었다.

## 왜 pickle 이 아닌가

torch 의 `save`/`load` 는 pickle 이다. 파이썬 객체를 **실행하며** 푸는 형식이라
브라우저로 옮길 수도 없고 옮겨서도 안 된다. 그래서 남의 `.pt` 는 못 읽는다 —
이 라이브러리로 만든 것을 이 라이브러리로 되읽는 일만 한다.

형식은 `borch-ts/src/serialize.ts` 가 이미 정해 두었고 여기서 **같은 것**을 쓴다.
셋이 같은 파일을 주고받는 것이 이 형식을 고른 이유이므로, 한쪽이 자기 형식을
만들면 그 값이 통째로 없어진다.

    [8바이트 LE u64: 머리 길이 N][N바이트 JSON 머리][몸: 텐서 바이트가 이어 붙는다]

## 몸은 언제나 float32 다

borch.ts 의 버퍼가 float32 하나뿐이라 저쪽은 선택지가 없다. 여기서 진짜 `I64` 로
적으면 **저쪽이 우리 파일을 못 읽는다** — 그러면 형식을 맞춘 뜻이 없어진다.
그래서 여기도 F32 로 적고 borch 의 이름표는 `__metadata__` 에 따로 싣는다.

대신 **정수가 f32 로 정확히 안 담기면 거절한다.** 조용히 반올림된 체크포인트는
몇 시간 뒤에야 드러나고, 그때는 무엇이 틀렸는지 알 길이 없다.

## 중첩은 나무로 적는다

`torch.save({"model": sd, "opt": sd, "epoch": 3}, path)` 가 교재의 관용구다.
safetensors 는 **평평한** 텐서 사전이므로 중첩을 펴야 하는데, 편 이름을 점으로
다시 쪼개 되돌리면 안 된다 — `state_dict` 의 열쇠에 이미 점이 들어 있어서
(`fc.weight`) `{"model": {"fc.weight": t}}` 가 `{"model": {"fc": {"weight": t}}}`
로 돌아온다. 그래서 구조를 `borch.tree` 에 따로 적고, 파일 안의 이름은 남이
읽었을 때 뜻이 통하도록 편 이름을 그대로 쓴다.
"""

import json as _json
import pathlib as _pathlib

import numpy as _np

from ._base import BrowserTorchError
from ._tensor import Tensor

# safetensors 가 정한 자리. 머리 길이를 적는다.
_LENGTH_FIELD = 8
# 참조 구현이 머리를 이 배수로 맞춘다. 어긋나면 numpy 쪽이 몸을 그대로 못 본다.
_ALIGN = 8
_BYTES_PER_F32 = 4
# borch 이름표를 싣는 열쇠의 앞머리. float32 인 것은 안 적는다 — 기본값이다.
_DTYPE_KEY = "borch.dtype:"
# 중첩 구조를 싣는 열쇠. 이것이 없으면 평평한 텐서 사전으로 읽는다.
_TREE_KEY = "borch.tree"
# f32 가 정수를 빠짐없이 담는 범위. 2^24 를 넘으면 이웃한 정수가 같은 값이 된다.
_EXACT_INT = 2 ** 24


def _labelled_dtype(array):
    """이 배열을 borch 의 어느 이름표로 적을 것인가. float32 면 안 적는다."""
    if array.dtype == _np.bool_:
        return "bool"
    if _np.issubdtype(array.dtype, _np.integer):
        return "int64"
    return None


def _as_f32(name, array):
    """몸에 실을 float32 배열. **정수가 안 담기면 거절한다.**"""
    if _np.issubdtype(array.dtype, _np.complexfloating):
        raise BrowserTorchError(
            f"'{name}' 이 복소수다 — 아직 저장 못 한다.\n"
            "`view_as_real()` 로 실수 짝을 저장하고 읽을 때 `view_as_complex()` 로 되돌리세요.")
    if _np.issubdtype(array.dtype, _np.integer) and array.size:
        if int(_np.abs(array).max()) > _EXACT_INT:
            raise BrowserTorchError(
                f"'{name}' 의 정수가 너무 큽니다 — 이 형식의 몸은 float32 입니다.\n"
                f"{_EXACT_INT} 를 넘는 정수는 저장하면 이웃한 값으로 바뀝니다. "
                "조용히 반올림된 체크포인트는 나중에 원인을 못 찾으므로 여기서 멈춥니다.")
    return _np.ascontiguousarray(array, dtype=_np.float32)


def encode(tensors, metadata=None):
    """`{이름: 배열}` 을 바이트로. **이름 순서를 고정한다** — 두 번 저장하면 같은 바이트다."""
    meta = dict(metadata or {})
    header = {}
    bodies = []
    offset = 0
    for name in sorted(tensors):
        array = _np.asarray(tensors[name])
        values = _as_f32(name, array)
        nbytes = values.size * _BYTES_PER_F32
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        label = _labelled_dtype(array)
        if label is not None:
            meta[_DTYPE_KEY + name] = label
        bodies.append(values)
        offset += nbytes

    if meta:
        header["__metadata__"] = meta
    text = _json.dumps(header).encode("utf-8")
    padding = (_ALIGN - ((_LENGTH_FIELD + len(text)) % _ALIGN)) % _ALIGN
    # 남는 자리는 공백이다 — JSON 파서가 뒤에 붙은 공백을 그냥 지난다.
    head = text + b" " * padding
    out = bytearray()
    out += len(head).to_bytes(_LENGTH_FIELD, "little")
    out += head
    for values in bodies:
        out += values.tobytes()
    return bytes(out)


def decode(blob):
    """바이트를 `({이름: 배열}, 메타데이터)` 로. **깨진 파일은 조용히 안 지난다.**"""
    if len(blob) < _LENGTH_FIELD:
        raise BrowserTorchError(f"체크포인트가 너무 짧습니다: {len(blob)} 바이트")
    head_len = int.from_bytes(blob[:_LENGTH_FIELD], "little")
    body_at = _LENGTH_FIELD + head_len
    if body_at > len(blob):
        raise BrowserTorchError(
            f"머리 길이가 파일을 넘습니다: {head_len} (파일 {len(blob)})")
    try:
        header = _json.loads(blob[_LENGTH_FIELD:body_at].decode("utf-8"))
    except Exception as exc:                                    # noqa: BLE001
        raise BrowserTorchError("체크포인트 머리가 JSON 이 아닙니다") from exc

    metadata = header.get("__metadata__") or {}
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        if begin > end or body_at + end > len(blob):
            raise BrowserTorchError(
                f"'{name}' 의 자리가 파일을 넘습니다: [{begin}, {end}]")
        shape = tuple(entry["shape"])
        flat = _np.frombuffer(blob, dtype=_np.float32,
                              count=(end - begin) // _BYTES_PER_F32,
                              offset=body_at + begin)
        want = 1
        for dim in shape:
            want *= dim
        if flat.size != want:
            raise BrowserTorchError(
                f"'{name}' 의 몸이 모양과 안 맞습니다: {flat.size} 대 {want}")
        array = flat.reshape(shape).copy()
        label = metadata.get(_DTYPE_KEY + name)
        if label == "int64":
            array = array.astype(_np.int64)
        elif label == "bool":
            array = array.astype(_np.bool_)
        tensors[name] = array
    return tensors, metadata


# ── 중첩 ────────────────────────────────────────────────────────────────────

def _flatten(obj, path, tensors, seen, to_array):
    """구조는 나무로, 텐서는 평평한 사전으로 나눈다.

    **텐서를 알아보는 일과 배열로 바꾸는 일만 밖에서 받는다.** 결속의 `Tensor` 는
    이것과 다른 클래스이고 값이 GPU 에 있으므로 그 두 가지가 다르다. 나머지 —
    중첩을 펴는 규칙, 이름 겹침 거절, 담을 수 있는 종류 — 는 **같아야 한다.**
    두 벌로 두면 한쪽만 고쳐지고, 그때 한쪽이 쓴 파일을 다른 쪽이 못 읽는다.
    """
    if to_array(obj) is not None:
        name = ".".join(path) or "tensor"
        if name in seen:
            # 서로 다른 자리가 같은 이름으로 펴졌다. 하나가 다른 하나를 덮으면
            # 되돌릴 때 두 자리가 같은 값을 갖는데, 그것은 예외보다 나쁘다.
            raise BrowserTorchError(
                f"'{name}' 이 두 번 나옵니다 — 편 이름이 겹쳐서 저장할 수 없습니다.")
        seen.add(name)
        tensors[name] = to_array(obj)
        return {"t": "T", "v": name}
    if isinstance(obj, dict):
        return {"t": "d",
                "v": {str(k): _flatten(v, [*path, str(k)], tensors, seen, to_array)
                      for k, v in obj.items()}}
    if isinstance(obj, (list, tuple)):
        kind = "u" if isinstance(obj, tuple) else "l"
        return {"t": kind,
                "v": [_flatten(v, [*path, str(i)], tensors, seen, to_array)
                      for i, v in enumerate(obj)]}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"t": "j", "v": obj}
    raise BrowserTorchError(
        f"{type(obj).__name__} 은(는) 저장할 수 없습니다 — 텐서·사전·목록·수·글자만 됩니다.\n"
        "이 형식은 pickle 이 아니라서 임의의 파이썬 객체를 못 담습니다.")


def _unflatten(node, tensors, make):
    kind = node["t"]
    if kind == "T":
        return make(tensors[node["v"]])
    if kind == "d":
        return {k: _unflatten(v, tensors, make) for k, v in node["v"].items()}
    if kind == "l":
        return [_unflatten(v, tensors, make) for v in node["v"]]
    if kind == "u":
        return tuple(_unflatten(v, tensors, make) for v in node["v"])
    return node["v"]


def _open_bytes(where):
    """경로든 파일 객체든 받는다 — torch 가 둘 다 받는다."""
    if hasattr(where, "write") or hasattr(where, "read"):
        return None, where
    return _pathlib.Path(where), None


def dump(obj, where, to_array):
    """구조를 펴서 바이트로 쓴다. **결속도 이것을 부른다** — 코덱은 하나다."""
    tensors = {}
    tree = _flatten(obj, [], tensors, set(), to_array)
    blob = encode(tensors, {_TREE_KEY: _json.dumps(tree)})
    path, handle = _open_bytes(where)
    if handle is not None:
        handle.write(blob)
        return
    path.write_bytes(blob)


def parse(where, make, **kw):
    """바이트를 읽어 구조를 되세운다. **결속도 이것을 부른다.**

    `weights_only` 는 받되 무시한다 — 이 형식은 코드를 안 실행하므로 **언제나**
    그쪽이다. torch 코드가 그 인자를 붙여 부르는 일이 많아서 받아만 둔다.
    """
    kw.pop("weights_only", None)
    kw.pop("map_location", None)
    if kw:
        raise BrowserTorchError(f"load 가 모르는 인자입니다: {', '.join(sorted(kw))}")
    path, handle = _open_bytes(where)
    blob = handle.read() if handle is not None else path.read_bytes()
    tensors, metadata = decode(blob)
    tree = metadata.get(_TREE_KEY)
    if tree is None:
        # 남이 만든 safetensors 다. 평평한 텐서 사전으로 준다.
        return {name: make(a) for name, a in tensors.items()}
    return _unflatten(_json.loads(tree), tensors, make)


def _array_of(obj):
    """이 코어의 텐서면 그 배열을, 아니면 `None`. `_flatten` 이 텐서를 가리는 잣대다."""
    return obj.data if isinstance(obj, Tensor) else None


def save(obj, where):
    """체크포인트를 쓴다. `obj` 는 텐서, 또는 텐서·수·글자를 담은 사전/목록이다."""
    dump(obj, where, _array_of)


def load(where, **kw):
    """`save` 가 쓴 것을 되읽는다."""
    return parse(where, Tensor, **kw)
