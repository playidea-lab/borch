"""`save`/`load` 의 **골든이 안 묻는 자리**.

골든은 값을 묻는다 — 쓴 것을 되읽으면 같은가. 그것은 세 구현에서 다 돌고 실제로
돈다. 여기서 묻는 것은 그 바깥이다.

- **거절.** 이 형식이 못 담는 것을 조용히 담지 않는가. 거절 코드는 짜기는 쉬운데
  한 번도 안 돌려보면 문구만 있고 동작이 없는 채로 남는다.
- **바이트 동등.** 같은 것을 두 번 저장하면 같은 파일인가. 이름 순서를 고정한
  이유가 그것이고, 안 물으면 사전 순서가 바뀌는 날 조용히 깨진다.
- **남의 파일.** `borch.tree` 가 없는 safetensors 를 평평한 사전으로 읽는가.
  브라우저가 쓴 파일이 그 모양이라 이 갈래가 실제 경로다.
- **결속의 갈래.** 코덱은 하나지만 텐서를 꺼내고 되돌리는 두 함수는 다르다.
  브라우저 없이 그 자리에 numpy 를 끼워 같은 바이트가 나오는지 본다.
"""

import pathlib
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import borch
from borch._base import BrowserTorchError
from borch._serialize import dump, encode, parse


def _tmp(name):
    return pathlib.Path(tempfile.mkdtemp()) / name


def test_save_refuses_integers_float32_cannot_hold_exactly():
    # 2^24 를 넘는 정수는 f32 에서 이웃한 값과 구별되지 않는다. 몸이 f32 인 것은
    # borch.ts 가 읽을 수 있게 하려는 선택이므로, 그 대가를 조용히 치르면 안 된다.
    big = borch.tensor(np.array([2 ** 24 + 1], dtype=np.int64))
    with pytest.raises(BrowserTorchError, match="정수가 너무 큽니다"):
        borch.save({"labels": big}, _tmp("big.bin"))


def test_save_accepts_integers_at_the_exact_boundary():
    ok = borch.tensor(np.array([2 ** 24], dtype=np.int64))
    path = _tmp("edge.bin")
    borch.save({"labels": ok}, path)
    assert int(borch.load(path)["labels"].data[0]) == 2 ** 24


def test_save_refuses_an_object_the_format_cannot_hold():
    # pickle 은 아무거나 담았다. 이 형식은 못 담고, 못 담는 것은 말해야 한다.
    with pytest.raises(BrowserTorchError, match="저장할 수 없습니다"):
        borch.save({"fn": len}, _tmp("obj.bin"))


def test_save_refuses_complex():
    z = borch.tensor(np.array([1 + 2j], dtype=np.complex64))
    with pytest.raises(BrowserTorchError, match="복소수"):
        borch.save({"z": z}, _tmp("z.bin"))


def test_save_refuses_two_paths_that_flatten_to_one_name():
    # `{"a": {"b": t}}` 와 `{"a.b": t}` 는 같은 이름으로 펴진다. 하나가 다른 하나를
    # 덮으면 되돌릴 때 두 자리가 같은 값을 갖는데, 그것은 예외보다 나쁘다.
    t = borch.tensor(np.zeros(2, dtype=np.float32))
    with pytest.raises(BrowserTorchError, match="두 번 나옵니다"):
        borch.save({"a": {"b": t}, "a.b": t}, _tmp("dup.bin"))


def test_saving_twice_gives_the_same_bytes():
    obj = {"w": borch.tensor(np.arange(6, dtype=np.float32).reshape(2, 3)),
           "b": borch.tensor(np.zeros(3, dtype=np.float32)),
           "epoch": 7}
    first, second = _tmp("a.bin"), _tmp("b.bin")
    borch.save(obj, first)
    borch.save(obj, second)
    assert first.read_bytes() == second.read_bytes()


def test_load_reads_a_file_object_and_save_writes_one():
    obj = {"w": borch.tensor(np.array([1.5, -2.5], dtype=np.float32))}
    path = _tmp("handle.bin")
    with open(path, "wb") as f:
        borch.save(obj, f)
    with open(path, "rb") as f:
        got = borch.load(f)
    assert np.array_equal(got["w"].data, obj["w"].data)


def test_load_ignores_weights_only_and_map_location():
    # torch 코드가 이 둘을 붙여 부른다. 이 형식은 코드를 안 실행하므로 언제나
    # `weights_only` 쪽이고, 받아만 두는 것이 임포트만 바꾸는 길을 지킨다.
    path = _tmp("kw.bin")
    borch.save({"w": borch.tensor(np.ones(2, dtype=np.float32))}, path)
    got = borch.load(path, weights_only=True, map_location="cpu")
    assert np.array_equal(got["w"].data, np.ones(2, dtype=np.float32))


def test_load_refuses_an_argument_it_does_not_know():
    path = _tmp("kw2.bin")
    borch.save({"w": borch.tensor(np.ones(2, dtype=np.float32))}, path)
    with pytest.raises(BrowserTorchError, match="모르는 인자"):
        borch.load(path, pickle_module=None)


def test_a_foreign_safetensors_reads_as_a_flat_dict():
    # `borch.tree` 가 없는 파일이다 — borch.ts 가 쓴 것이 이 모양이고, 남의 도구가
    # 쓴 것도 그렇다. 나무가 없으면 중첩을 지어내지 말고 평평하게 준다.
    blob = encode({"fc.weight": np.array([[1.0, 2.0]], dtype=np.float32)})
    path = _tmp("foreign.bin")
    path.write_bytes(blob)
    got = borch.load(path)
    assert sorted(got) == ["fc.weight"]
    assert np.array_equal(got["fc.weight"].data, np.array([[1.0, 2.0]], dtype=np.float32))


def test_a_truncated_file_is_refused_not_guessed():
    path = _tmp("short.bin")
    path.write_bytes(b"\x00\x03")
    with pytest.raises(BrowserTorchError, match="너무 짧습니다"):
        borch.load(path)


def test_a_header_longer_than_the_file_is_refused():
    path = _tmp("liar.bin")
    path.write_bytes((10 ** 6).to_bytes(8, "little") + b"{}")
    with pytest.raises(BrowserTorchError, match="파일을 넘습니다"):
        borch.load(path)


def test_the_binding_shaped_hooks_write_the_same_bytes_as_the_core():
    """**결속이 쓴 파일을 코어가 읽는가** — 브라우저 없이 그 자리를 재는 법.

    결속은 코덱을 그대로 쓰고 두 갈래만 바꾼다(텐서를 알아보는 잣대, 값을 꺼내고
    되돌리는 길). 그 두 갈래를 numpy 로 흉내 내면 저쪽이 쓸 바이트가 나온다.

    이것이 없으면 "코덱이 하나니까 같다" 는 주장이지 측정이 아니다. 실제로 이
    프로젝트는 그 주장 하나를 믿고 파이썬 쪽이 pickle 을 쓰는 것을 오래 못 봤다.
    """
    class Fake:                       # 결속의 `Tensor` 자리
        def __init__(self, array):
            self.array = array

    def array_of(obj):
        return obj.array if isinstance(obj, Fake) else None

    payload = {"model": {"fc.weight": Fake(np.array([[1.5, -2.25]], dtype=np.float32))},
               "epoch": 3}
    theirs = _tmp("binding.bin")
    dump(payload, theirs, array_of)

    # 코어가 그 파일을 연다 — 구조와 값이 그대로여야 한다.
    got = borch.load(theirs)
    assert sorted(got) == ["epoch", "model"]
    assert got["epoch"] == 3
    assert np.array_equal(got["model"]["fc.weight"].data,
                          np.array([[1.5, -2.25]], dtype=np.float32))

    # 그리고 코어가 같은 것을 쓰면 **같은 바이트**여야 한다.
    mine = _tmp("core.bin")
    borch.save({"model": {"fc.weight": borch.tensor(
        np.array([[1.5, -2.25]], dtype=np.float32))}, "epoch": 3}, mine)
    assert theirs.read_bytes() == mine.read_bytes()


def test_parse_gives_back_what_dump_took_for_lists_and_tuples():
    # 목록과 튜플이 섞인 것도 모양을 지켜야 한다 — 옵티마이저 상태가 그 모양이다.
    payload = {"xs": [borch.tensor(np.ones(2, dtype=np.float32)), 1, "two"],
               "pair": (3, 4.5)}
    path = _tmp("mixed.bin")
    borch.save(payload, path)
    got = borch.load(path)
    assert isinstance(got["xs"], list) and isinstance(got["pair"], tuple)
    assert got["xs"][1] == 1 and got["xs"][2] == "two"
    assert got["pair"] == (3, 4.5)


def test_dtype_labels_survive_the_round_trip():
    payload = {"labels": borch.tensor(np.array([3, 1, 4], dtype=np.int64)),
               "flags": borch.tensor(np.array([True, False], dtype=np.bool_))}
    path = _tmp("labels.bin")
    borch.save(payload, path)
    got = borch.load(path)
    assert got["labels"].data.dtype == np.int64
    assert got["flags"].data.dtype == np.bool_
    assert list(got["labels"].data) == [3, 1, 4]


def test_parse_is_the_same_function_the_binding_calls():
    # `parse` 를 직접 불러 본다 — 결속이 부르는 것이 이것이고, 여기서 안 물으면
    # 그 경로는 브라우저에서만 돌아 실패가 늦게 보인다.
    blob = encode({"w": np.array([2.0], dtype=np.float32)})
    path = _tmp("parse.bin")
    path.write_bytes(blob)
    got = parse(path, lambda a: np.asarray(a) * 10)
    assert got["w"][0] == 20.0
