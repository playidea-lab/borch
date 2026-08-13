"""골든 파일 하네스 — 진짜 torch 가 없는 곳에서 대조한다.

GPU 경로는 브라우저에서만 돌고, 진짜 torch 는 브라우저에 없다. 셋을 한 프로세스에서
나란히 부를 수 없으므로 둘로 나눈다.

    1단계(네이티브)  uv run --with numpy --with torch python tests/golden.py dump
    2단계(어디서든)  uv run --with numpy python tests/golden.py check

2단계는 대조할 라이브러리를 골라 받는다. 지금은 browsertorch 뿐이고, GPU 백엔드가
생기면 같은 자리에 그것을 넣는다 — 그때 하네스는 안 고친다.

골든에 값만 담지 않는다. **케이스 목록의 해시**와 **입력의 지문**을 같이 담아, 표가
바뀌었거나 입력이 달라졌으면 통과가 아니라 실패가 나오게 한다. 대조하지 않은 것을
대조했다고 말하는 것이 안 하는 것보다 나쁘다.
"""

import importlib.util
import pathlib
import sys

import numpy as np

_here = pathlib.Path(__file__).resolve().parent
DEFAULT_PATH = _here / "golden.npz"

_cases_spec = importlib.util.spec_from_file_location("bt_cases", _here / "cases.py")
cases_mod = importlib.util.module_from_spec(_cases_spec)
_cases_spec.loader.exec_module(cases_mod)

# 넓은 표면 하네스와 같은 허용치. 비트 동등(T4)은 이 프로젝트의 명시적 비목표다.
ATOL = RTOL = 1e-4
_PREFIX = "case::"
_INPUT_PREFIX = "input::"


def load_browsertorch():
    spec = importlib.util.spec_from_file_location(
        "browsertorch", _here.parent / "browsertorch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dump(path=DEFAULT_PATH):
    """1단계 — 진짜 torch 의 기대값을 굳힌다. 여기서만 torch 가 필요하다."""
    import torch as real

    inp = cases_mod.golden_inputs()
    cases = cases_mod.golden_cases(inp)
    data, broken = {}, []
    for name, fn in cases:
        try:
            data[_PREFIX + name] = cases_mod.to_numpy(fn(real))
        except Exception as exc:                                    # noqa: BLE001
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
    if broken:
        # 진짜 torch 가 못 하는 것은 기대값이 아니라 **틀린 케이스**다. 굳히면 안 된다.
        raise SystemExit("진짜 torch 에서 실패한 케이스가 있다:\n  " + "\n  ".join(broken))
    # 키별 지문도 같이 굳힌다 — 갈렸을 때 **어느 입력이** 갈렸는지 말해주기 위해서다.
    for key, digest in cases_mod.input_fingerprints(inp).items():
        data[_INPUT_PREFIX + key] = np.array(digest)
    np.savez(path,
             __manifest__=np.array(cases_mod.manifest_hash(cases)),
             __inputs__=np.array(cases_mod.input_fingerprint(inp)),
             **data)
    return len(cases), path


def check(lib, path=DEFAULT_PATH):
    """2단계 — 골든과 대조한다. (갈린 곳 목록, 케이스 수)."""
    z = np.load(path, allow_pickle=False)
    inp = cases_mod.golden_inputs()
    cases = cases_mod.golden_cases(inp)

    if str(z["__manifest__"]) != cases_mod.manifest_hash(cases):
        raise SystemExit(
            "골든이 낡았다 — 케이스 표가 바뀌었다. dump 를 다시 돌려라.")
    if str(z["__inputs__"]) != cases_mod.input_fingerprint(inp):
        mine = cases_mod.input_fingerprints(inp)
        drifted = [k for k, d in mine.items()
                   if _INPUT_PREFIX + k not in z or str(z[_INPUT_PREFIX + k]) != d]
        detail = ", ".join(
            f"{k}(여기서는 {inp[k].dtype} {inp[k].shape})" for k in drifted) or "(어느 것인지 못 짚었다)"
        raise SystemExit(
            "입력이 골든과 다르다 — 이 상태의 비교는 대조가 아니다.\n"
            f"  갈린 입력: {detail}")

    bad = []
    for name, fn in cases:
        want = z[_PREFIX + name]
        try:
            got = cases_mod.to_numpy(fn(lib))
        except Exception as exc:                                    # noqa: BLE001
            bad.append(f"{name}: {type(exc).__name__} — {str(exc).splitlines()[0][:60]}")
            continue
        if want.shape != got.shape:
            bad.append(f"{name}: 모양 {want.shape} vs {got.shape}")
        elif not np.allclose(want, got, atol=ATOL, rtol=RTOL):
            bad.append(f"{name}: 최대차 {np.abs(want - got).max():.2e}")
    return bad, len(cases)


def main(argv):
    what = argv[1] if len(argv) > 1 else "check"
    if what == "dump":
        count, path = dump()
        print(f"골든을 굳혔다 — 케이스 {count}개 → {path}")
        return 0
    if what == "check":
        if not DEFAULT_PATH.exists():
            print(f"골든이 없다: {DEFAULT_PATH}\n"
                  "  먼저: uv run --with numpy --with torch python tests/golden.py dump")
            return 1
        bad, total = check(load_browsertorch())
        print(f"골든 대조 — 케이스 {total}개")
        print(f"  일치 {total - len(bad)}/{total}")
        if bad:
            print("\n갈린 곳:")
            for why in bad:
                print(f"  ✗ {why}")
        return 1 if bad else 0
    print("쓰는 법: golden.py [dump|check]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
