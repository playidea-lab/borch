"""골든 파일 하네스 — 진짜 torch 가 없는 곳에서 대조한다.

GPU 경로는 브라우저에서만 돌고, 진짜 torch 는 브라우저에 없다. 셋을 한 프로세스에서
나란히 부를 수 없으므로 둘로 나눈다.

    1단계(네이티브)  uv run --with numpy --with torch python tests/golden.py dump
    2단계(어디서든)  uv run --with numpy python tests/golden.py check

2단계는 대조할 라이브러리를 골라 받는다. 지금은 borch 뿐이고, GPU 백엔드가
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


def load_borch():
    """저장소 루트의 `borch` 를 들여온다.

    예전에는 파일 하나를 경로로 집어 들였다. 패키지가 되면서 그 방법이 안 통한다 —
    `__init__.py` 만 실행해도 상대 임포트가 패키지 문맥을 요구하기 때문이다.
    """
    root = str(_here.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import borch
    return borch


def dump(path=DEFAULT_PATH):
    """1단계 — 진짜 torch 의 기대값을 굳힌다. 여기서만 torch 가 필요하다."""
    import torch as real

    inp = cases_mod.golden_inputs()
    cases = cases_mod.golden_cases(inp)
    data, broken = {}, []
    for name, fn in cases:
        try:
            got = fn(real)
            # dtype 케이스는 값이 아니라 **형 이름**을 묻는다. 문자열 그대로 굳힌다.
            data[_PREFIX + name] = (np.array(got) if isinstance(got, str)
                                    else cases_mod.to_numpy(got))
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


def check(lib, path=DEFAULT_PATH, faults=None):
    """2단계 — 골든과 대조한다. (갈린 곳 목록, 케이스 수).

    `faults` 는 **지금까지 난 GPU 검증 오류 수**를 돌려주는 함수다(안 주면 안 본다).

    WebGPU 의 검증 오류는 예외가 아니다. 무효한 명령 버퍼는 조용히 아무것도 안 하고,
    그래서 **범인은 통과하고 뒤에 줄 선 케이스가 대신 빨개진다.** 세 번 겪었다 —
    `as_strided_` 의 초과 복사, 옵티마이저 상태의 버퍼 공유, 그리고 아무것도 안 고르는
    `index_select` 가 0 으로 나누는 셰이더를 굽던 자리.

    케이스마다 수를 재면 **그 자리를 이름으로 짚을 수 있다.** 셋 다 원인에서 한두 칸
    떨어진 자리를 보며 시작했고, 그 거리가 이 검사의 값어치다.
    """
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

    # 두 라이브러리의 범위가 갈린다. 자매 쪽에만 있는 것은 자매만 대조한다 —
    # 코어가 일부러 거절하는 것을 코어에게 물으면 그건 검사가 아니라 오답이다.
    is_webgpu = hasattr(lib, "backend")
    bad, skipped = [], 0
    seen_faults = faults() if faults else 0
    for name, fn in cases:
        if name.startswith(cases_mod.WEBGPU_PREFIX) and not is_webgpu:
            skipped += 1
            continue
        want = z[_PREFIX + name]
        try:
            raw = fn(lib)
            got = np.array(raw) if isinstance(raw, str) else cases_mod.to_numpy(raw)
        except Exception as exc:                                    # noqa: BLE001
            bad.append(f"{name}: {type(exc).__name__} — {str(exc).splitlines()[0][:60]}")
            continue
        finally:
            # **값이 맞아도 검증 오류를 냈으면 빨갛다.** 이 케이스는 통과할 수 있어도
            # 명령 버퍼가 무효가 된 채 다음으로 넘어간다.
            if faults:
                now = faults()
                if now > seen_faults:
                    bad.append(f"{name}: GPU 검증 오류 {now - seen_faults}건 "
                               f"(값과 무관하게 여기서 났다)")
                    seen_faults = now
        if want.dtype.kind == "U" or got.dtype.kind == "U":
            if str(want) != str(got):
                bad.append(f"{name}: {want} 여야 하는데 {got}")
            continue
        if want.shape != got.shape:
            bad.append(f"{name}: 모양 {want.shape} vs {got.shape}")
        elif not np.allclose(want, got, atol=ATOL, rtol=RTOL):
            bad.append(f"{name}: 최대차 {np.abs(want - got).max():.2e}")
    return bad, len(cases) - skipped


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
        bad, total = check(load_borch())
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
