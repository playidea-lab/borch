"""체크포인트가 실제 브라우저에서 왕복하고, 재개가 이어지는지 본다.

    npm run build:ts
    uv run --with playwright python borch-ts/test/serialize.py [--headed]

**왕복만 보면 부족하다.** 저장했다 읽어서 값이 같은지는 코덱만 묻는 것이다. 진짜
물음은 그 뒤다 — *끊었다 이은 학습이 안 끊고 돌린 학습과 같은가.* 모멘텀 하나, 스텝
계수기 하나, 스케줄러의 에폭 하나만 빠져도 왕복은 초록인 채로 재개만 갈린다.

전부 결정론적이라 **비트 단위로 같아야 한다.** 그래서 이 러너에는 허용 오차가 없다.

가중치만 되돌리고 나머지를 버린 경로도 같이 돌린다 — 그쪽이 **갈려야** 위의 동등성
검사가 무언가를 재고 있다는 뜻이다.
"""

import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/serialize.html"
TIMEOUT_MS = 300_000


def main(argv):
    # **A stale emit is as bad as none** — edit the source, forget the build, and you
    # measure the old code. `require_fresh_dist` watches that place (`run.py`).
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "serialize.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        # **`with` closes it too** — put on the last line instead, an exception before
        # it leaves it open, and the leftover Chromium ruins another measurement.
        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  [browser] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}"))
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__borchSerialize !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchSerialize")
    finally:
        stop()

    if "error" in result:
        print(f"**체크포인트 점검이 터졌다**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"adapter: {result.get('adapter', '(unknown)')}")
    print(result["text"])
    if verdict(result, "checkpoints") or not cross_language(result.get("sample")):
        return 1
    return 0 if cross_tree(result.get("nested")) else 1


def cross_language(sample):
    """브라우저가 쓴 파일을 **numpy 로만** 뜯는다. borch 코드는 한 줄도 안 쓴다.

    우리 코덱이 우리 코덱과 왕복하는 것은 자체 형식으로도 된다. safetensors 를 든
    값어치는 **남이 읽는다**는 데 있고, 그 주장이 참인지는 여기서만 확인된다.
    """
    import json
    import struct

    import numpy as np

    if not sample:
        print("표본이 없다 — 페이지가 sample() 을 안 내보냈다", file=sys.stderr)
        return False

    blob = bytes(bytearray(sample))
    (head_len,) = struct.unpack_from("<Q", blob, 0)
    header = json.loads(blob[8:8 + head_len])
    body = blob[8 + head_len:]

    got = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        # dtype 은 언제나 F32 다. borch 의 int64·bool 은 이름표라 머리에만 적힌다 —
        # 몸에 I64 라고 써 놓으면 4 바이트짜리와 어긋나 이 줄이 깨진다.
        assert entry["dtype"] == "F32", entry["dtype"]
        got[name] = np.frombuffer(body[begin:end], dtype="<f4").reshape(entry["shape"])

    want = {
        "fc.weight": np.array([[1.5, -2.25, 0.5], [7.0, -0.125, 3.0]], dtype="<f4"),
        "fc.labels": np.array([3.0, 1.0, 4.0], dtype="<f4"),
    }
    for name, expected in want.items():
        if name not in got or not np.array_equal(got[name], expected):
            print(f"**numpy 가 읽은 값이 다르다** — {name}: {got.get(name)}",
                  file=sys.stderr)
            return False

    labels = header["__metadata__"].get("borch.dtype:fc.labels")
    if labels != "int64":
        print(f"**형 이름표가 안 실렸다** — {labels}", file=sys.stderr)
        return False

    print(f"  ✓ numpy 가 같은 파일을 읽는다 — 텐서 {len(got)}개, "
          f"이름표 fc.labels={labels}")
    return cross_library(blob)


def cross_library(blob):
    """브라우저가 쓴 파일을 **파이썬 `borch` 가** 읽는다.

    `serialize.ts` 의 첫 문단이 safetensors 를 고른 이유로 "파이썬 `borch`·numpy·HF
    도구가 같은 파일을 읽는다" 를 든다. **그 문장이 오래 거짓이었다** — 파이썬 쪽
    `save`/`load` 는 pickle 이었고, 그래서 브라우저에서 학습해 자기 컴퓨터로
    가져가는 길이 막혀 있었다. 그 길이 이 프로젝트가 그 형식을 고른 유일한 이유다.

    위의 numpy 검사로는 이것이 안 보인다. 저쪽은 **형식**이 열려 있는지를 묻고
    이쪽은 **우리 파이썬 코드가 실제로 그 문을 여는지**를 묻는다 — 형식이 맞아도
    읽는 함수가 없으면 사용자에게는 없는 것과 같다.
    """
    import pathlib
    import sys as _sys
    import tempfile

    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    import borch

    path = pathlib.Path(tempfile.mkdtemp()) / "from_browser.bin"
    path.write_bytes(blob)
    # 최상위가 텐서 사전이라 나무가 있으나 없으나 같은 것이 나온다.
    got = borch.load(path)
    if "fc.weight" not in got:
        print(f"**borch 가 못 읽었다** — 열쇠 {sorted(got)}", file=_sys.stderr)
        return False
    first = float(got["fc.weight"].data.reshape(-1)[0])
    if abs(first - 1.5) > 1e-6:
        print(f"**borch 가 읽은 값이 다르다** — {first}", file=_sys.stderr)
        return False
    print(f"  ✓ 파이썬 borch 가 브라우저의 파일을 읽는다 — fc.weight[0]={first}")
    return True


def cross_tree(nested):
    """브라우저가 쓴 **중첩** 파일을 파이썬 `borch` 가 구조 그대로 읽는가.

    나무 스킴(`borch.tree`, 마디 `T`/`d`/`l`/`j`)이 이제 두 벌 있다 — `serialize.ts`
    와 `_serialize.py`. 같은 글자를 쓰기로 되어 있는데 **그 약속을 아무도 안 쟀다.**
    한쪽만 고쳐지면 한쪽이 쓴 체크포인트를 다른 쪽이 못 읽고, 그때 나오는 것은 예외가
    아니라 **구조가 다른 사전**이라 훨씬 늦게 들킨다.

    위의 `cross_library` 로는 안 보인다. 그 표본은 최상위가 텐서 사전이라 나무가
    있으나 없으나 같은 것이 나온다 — 평평한 것만 물으면 나무는 한 번도 안 밟힌다.
    """
    import pathlib
    import sys as _sys
    import tempfile

    if not nested:
        print("중첩 표본이 없다 — 페이지가 sampleNested() 를 안 내보냈다",
              file=_sys.stderr)
        return False

    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    import borch

    path = pathlib.Path(tempfile.mkdtemp()) / "nested_from_browser.bin"
    path.write_bytes(bytes(bytearray(nested)))
    got = borch.load(path)

    def fail(said):
        print(f"**{said}**", file=_sys.stderr)
        return False

    if not isinstance(got, dict) or sorted(got) != [
            "done", "epoch", "model", "note", "nothing", "steps"]:
        return fail(f"열쇠가 다르다 — {sorted(got) if isinstance(got, dict) else type(got)}")
    if not isinstance(got["model"], dict) or "fc.weight" not in got["model"]:
        # 점을 다시 쪼갰으면 여기서 `{"fc": {"weight": …}}` 가 나온다.
        return fail(f"중첩이 안 왔다 — model={got['model']}")
    if not isinstance(got["steps"], list) or len(got["steps"]) != 2:
        return fail(f"배열이 안 왔다 — steps={got['steps']}")
    if float(got["steps"][0].data.reshape(-1)[0]) != 7.0 or got["steps"][1] != 3:
        return fail(f"배열 안이 다르다 — {got['steps']}")
    if (got["epoch"], got["note"], got["done"], got["nothing"]) != (
            5, "nested", False, None):
        return fail("텐서가 아닌 값들이 다르다 — "
                    f"{got['epoch']} {got['note']} {got['done']} {got['nothing']}")
    if float(got["model"]["fc.weight"].data.reshape(-1)[0]) != 1.5:
        return fail(f"값이 다르다 — {got['model']['fc.weight'].data}")

    print("  ✓ 파이썬 borch 가 브라우저의 **중첩** 파일을 구조 그대로 읽는다")
    return True


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
