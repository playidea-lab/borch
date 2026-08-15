"""뽑은 JSON 이 **실제로 쓸 수 있는가.**

파일이 생겼다는 것과 그것으로 대조가 된다는 것은 다르다. 여기서는 npz 를 **안 보고**
JSON 만으로 코어를 대조해본다 — 다른 언어 쪽이 하게 될 일과 같은 절차다.

그리고 이 파일이 지키는 약속이 하나 더 있다: **JSON 이 낡으면 조용히 통과하면
안 된다.** 케이스 표가 바뀌었는데 옛 JSON 으로 초록이 나오면 그건 대조가 아니다.
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

_here = pathlib.Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exporter = _load("bt_export", _here / "export_json.py")
golden = _load("bt_golden", _here / "golden.py")
cases_mod = exporter.cases_mod


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    """굳히고 뽑는다 — 저장소에 놓인 파일이 아니라 **방금 만든 것**을 본다."""
    npz = tmp_path_factory.mktemp("golden") / "golden.npz"
    out = tmp_path_factory.mktemp("golden") / "golden.json"
    golden.dump(npz)
    exporter.export(npz, out)
    return json.loads(out.read_text(encoding="utf-8"))


def _restore(entry):
    """JSON 한 칸을 numpy 로 되돌린다 — 읽는 쪽이 할 일을 흉내 낸다."""
    if entry["kind"] == "string":
        return entry["value"]
    values = list(entry["values"])
    if entry["kind"] == "float":
        # **종류까지 되살린다.** 전부 `nan` 으로 되돌리던 때에는 `inf` 가 `nan` 이
        # 되어, 답에 무한대가 있는 케이스가 "최대차 0 인데 실패" 로 나왔다.
        for i, kind in entry["nonfinite"]:
            values[i] = float(kind)
        arr = np.asarray(values, dtype=np.float64)
    elif entry["kind"] == "int":
        arr = np.asarray(values, dtype=np.int64)
    else:
        arr = np.asarray(values, dtype=bool)
    return arr.reshape(entry["shape"])


def test_json_alone_can_check_the_core(doc):
    """**npz 를 안 보고** JSON 만으로 코어를 대조한다.

    이것이 되어야 다른 언어 쪽이 이 파일을 쓸 수 있다. 안 되면 뽑은 것은 그냥
    큰 파일이지 자산이 아니다.
    """
    core = golden.load_borch()
    inp = cases_mod.golden_inputs()
    bad, checked = [], 0
    for name, fn in cases_mod.golden_cases(inp):
        if name.startswith(cases_mod.WEBGPU_PREFIX):
            continue                            # 코어가 일부러 거절하는 것
        want = _restore(doc["cases"][name])
        got = fn(core)
        got = np.asarray(got) if isinstance(got, str) else cases_mod.to_numpy(got)
        checked += 1
        if isinstance(want, str):
            if want != str(got):
                bad.append(f"{name}: {want} 여야 하는데 {got}")
        elif want.shape != got.shape:
            bad.append(f"{name}: 모양 {want.shape} vs {got.shape}")
        elif not np.allclose(want, got, atol=1e-4, rtol=1e-4, equal_nan=True):
            bad.append(f"{name}: 최대차 {np.nanmax(np.abs(want - got)):.2e}")
    assert checked > 700, f"대조한 것이 {checked}건뿐이다 — 표가 안 실렸다"
    assert not bad, "JSON 과 갈렸다:\n  " + "\n  ".join(bad[:10])


def test_json_carries_every_case_name(doc):
    """이름이 하나라도 빠지면 다른 언어 쪽은 **그 케이스가 있는 줄도 모른다.**"""
    names = {name for name, _ in cases_mod.golden_cases()}
    assert set(doc["cases"]) == names


def test_json_carries_the_shared_inputs(doc):
    """케이스가 쓰는 입력도 같이 나가야 한다 — 답만 있고 문제가 없으면 못 푼다."""
    inp = cases_mod.golden_inputs()
    assert set(doc["inputs"]) == set(inp)
    for key, arr in inp.items():
        assert np.allclose(_restore(doc["inputs"][key]), arr, equal_nan=True), key


def test_stale_json_is_detectable(doc):
    """**표가 바뀌면 알아챌 수 있어야 한다.**

    JSON 은 npz 와 달리 `check` 를 안 거치므로, 낡았을 때 막아주는 것이 매니페스트
    해시뿐이다. 그것이 실제로 갈리는지 본다 — 안 갈리면 다른 언어 쪽은 옛 답을
    새 표에 맞춰보고 통과 도장을 찍는다.
    """
    assert doc["manifest"] == cases_mod.manifest_hash(cases_mod.golden_cases())
    assert doc["manifest"] != cases_mod.manifest_hash([("표가바뀌었다", None)])


def test_nonfinite_survives_the_round_trip():
    """`nan` 과 무한대는 JSON 에 그대로 못 적는다. 자리와 **종류**를 따로 적는다.

    오래 자리 번호만 적었고, 읽는 쪽이 전부 `nan` 으로 되살렸다. 그때는 답에
    무한대가 없어서 안 걸렸는데(`nanmean` 류는 `nan` 이 입력에만 있고 답은 유한하다),
    `fmax` 케이스가 처음으로 무한대를 답에 담으면서 드러났다 — **최대차 0 인데
    실패**로 나왔다. `inf` 와 `nan` 을 같은 것으로 되살리고 있었기 때문이다.

    묻는 것은 **왕복이 종류까지 도는가**다.
    """
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
    entry = exporter._value(arr)
    assert entry["nonfinite"] == [[1, "nan"], [2, "inf"], [3, "-inf"]]
    back = _restore(entry)
    assert back[0] == 1.0 and back[4] == 2.0
    assert np.isnan(back[1])
    assert back[2] == np.inf and back[3] == -np.inf


def test_inputs_carry_their_nan():
    """입력 쪽에는 `nan` 이 실제로 있다 — `nanmean`·`nansum` 이 그것을 쓴다.
    그 배열이 왕복에서 유한한 수로 바뀌면 읽는 쪽은 **다른 문제를 푼다.**"""
    inp = cases_mod.golden_inputs()
    withnan = [k for k, v in inp.items()
               if v.dtype.kind == "f" and not np.all(np.isfinite(v))]
    for key in withnan:
        restored = _restore(exporter._value(inp[key]))
        assert np.array_equal(np.isnan(restored), np.isnan(inp[key])), key
