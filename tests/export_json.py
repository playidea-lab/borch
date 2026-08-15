"""골든을 **언어 중립 형식**으로 뽑는다.

    uv run --with numpy python tests/export_json.py

## 왜

지금 골든은 `tests/golden.npz`(numpy)와 `tests/cases.py`(파이썬 람다)로 되어 있다.
파이썬 라이브러리 둘을 대조하는 데는 그것으로 충분한데, 파이썬이 아닌 구현이
생기면 **기대값에 닿을 방법이 없다.** 진짜 torch 로 굳힌 798건은 이 저장소에서
가장 비싼 자산이고, 그것을 파이썬 안에만 두면 다음 구현은 검증 없이 자란다.

## 옮겨지는 것과 안 옮겨지는 것 — 섞지 않는다

**옮겨진다**: 케이스 이름, 기대값(모양·dtype 포함), 케이스가 쓰는 입력 배열,
케이스 목록의 해시. 즉 **"무엇이 답인가"** 전부.

**안 옮겨진다**: 케이스 본문. `lambda L: L.amax(L.tensor(tie))` 는 파이썬 코드이고
기계적으로 다른 언어가 되지 않는다. 다른 언어 쪽은 **같은 이름의 케이스를 자기
언어로 써야 하고**, 이 파일이 주는 것은 그때 맞춰볼 답이다.

그 나눔이 이 파일의 요점이다. 비싼 절반(진짜 torch 를 돌려 얻은 숫자)은 옮겨지고,
싼 절반(호출 한 줄)은 다시 쓰면 된다. 반대로 적으면 안 된다 — "골든을 통째로
옮겼다"고 말하는 순간 다음 사람이 안 쓴 케이스를 썼다고 믿는다.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

_here = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = _here / "golden.json"

_spec = importlib.util.spec_from_file_location("bt_cases", _here / "cases.py")
cases_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cases_mod)


def _value(arr):
    """numpy 배열 하나를 JSON 이 담을 수 있는 모양으로.

    값은 **평평하게** 담고 모양을 따로 준다 — 중첩 리스트로 두면 랭크 8 짜리가
    괄호로 뒤덮이고, 읽는 쪽도 평평한 것을 더 쉽게 받는다.
    """
    if arr.dtype.kind == "U":
        return {"kind": "string", "value": str(arr)}
    flat = np.asarray(arr).reshape(-1)
    if arr.dtype.kind == "b":
        return {"kind": "bool", "shape": list(arr.shape),
                "values": [bool(v) for v in flat]}
    if arr.dtype.kind in "iu":
        return {"kind": "int", "shape": list(arr.shape),
                "values": [int(v) for v in flat]}
    # 실수는 **float64 로 적는다.** float32 로 반올림해 적으면 읽는 쪽이 우리보다
    # 정확할 때 그 차이가 우리 반올림 탓인지 구현 탓인지 못 가른다.
    # **유한하지 않은 값은 종류까지 적는다.** JSON 에 `nan`·`Infinity` 를 그대로
    # 적으면 엄밀한 파서가 거절하므로 자리를 따로 남기는데, 오래 자리 번호만 적고
    # 종류를 안 적었다 — 읽는 쪽이 전부 `nan` 으로 되살리므로 `inf` 와 `nan` 이
    # 같아졌다. 답에 무한대가 없는 동안은 안 걸렸고, `fmax` 케이스가 처음으로
    # 무한대를 답에 담으면서 드러났다(최대차 0 인데 실패로 나왔다).
    def kind_of(v):
        if np.isnan(v):
            return "nan"
        return "inf" if v > 0 else "-inf"

    return {"kind": "float", "shape": list(arr.shape),
            "values": [None if not np.isfinite(v) else float(v) for v in flat],
            "nonfinite": [[i, kind_of(v)] for i, v in enumerate(flat)
                          if not np.isfinite(v)]}


def export(npz_path=None, out_path=DEFAULT_OUT):
    npz_path = npz_path or (_here / "golden.npz")
    if not npz_path.exists():
        raise SystemExit(
            f"골든이 없다: {npz_path}\n"
            "  먼저: uv run --with numpy --with torch python tests/golden.py dump")
    z = np.load(npz_path, allow_pickle=False)

    inp = cases_mod.golden_inputs()
    names = [name for name, _ in cases_mod.golden_cases(inp)]
    if str(z["__manifest__"]) != cases_mod.manifest_hash(cases_mod.golden_cases(inp)):
        raise SystemExit("골든이 낡았다 — 케이스 표가 바뀌었다. dump 를 다시 돌려라.")

    doc = {
        "note": ("진짜 PyTorch 로 굳힌 기대값이다. 케이스 **본문**은 여기 없다 — "
                 "다른 언어 쪽은 같은 이름의 케이스를 자기 언어로 쓰고, 그 답을 "
                 "여기서 맞춰본다."),
        "tolerance": {"atol": 1e-4, "rtol": 1e-4,
                      "note": "비트 동등은 이 프로젝트의 명시적 비목표다"},
        "manifest": str(z["__manifest__"]),
        "inputs_fingerprint": str(z["__inputs__"]),
        # 케이스가 공유하는 입력. 이름이 `golden_inputs()` 의 키와 같다.
        "inputs": {k: _value(v) for k, v in inp.items()},
        "cases": {},
    }
    missing = []
    for name in names:
        key = "case::" + name
        if key not in z.files:
            missing.append(name)
            continue
        doc["cases"][name] = _value(z[key])
    if missing:
        raise SystemExit("골든에 없는 케이스가 있다:\n  " + "\n  ".join(missing))

    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    out_path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return len(doc["cases"]), out_path, len(text), digest


def main(argv):
    count, path, size, digest = export()
    numeric = sum(1 for c in json.loads(path.read_text(encoding="utf-8"))["cases"].values()
                  if c["kind"] != "string")
    print(f"뽑았다 — 케이스 {count}개 (수치 {numeric}, 문자열 {count - numeric})")
    print(f"  {path}  {size / 1024:.0f}KB  sha256:{digest}")
    print("  케이스 **본문**은 안 들어 있다 — 답만 들어 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
