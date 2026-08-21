"""배움터와 길잡이의 **실행 가능한 블록이 실제로 도는가.**

    npm run build:ts
    uv run --with playwright python borch-ts/test/lessons.py

## 왜 있는가

이 저장소는 사이트 페이지의 JS 를 한 번도 안 돌려 봤다. `test_site.py` 는 글자와
링크와 스프라이트를 보고, 골든 러너는 `borch-ts/test/` 안의 제 페이지만 본다.
그 사이에 **사용자가 실제로 누르는 코드**가 있고 아무도 안 봤다.

그 자리가 값을 물은 것은 `save`/`load` 를 중첩 쌍으로 바꿀 때다. 열 곳이 `load(x)
.tensors` 로 쓰고 있었고 새 `load` 는 표를 바로 준다 — 고치는 것은 한 줄씩이었지만,
**고쳤는지 아는 방법이 사람의 눈뿐이었다.** 다음에 공개 이름이 바뀌면 같은 자리가
같은 방식으로 조용히 깨진다.

## 무엇을 재는가

각 페이지의 `data-lang="js"` 블록을 **누른다.** 그리고 출력이 나왔는지, 그 안에
예외 문구가 없는지를 본다. 값이 맞는지는 안 본다 — 그것은 골든의 일이다. 여기가
잡는 것은 **터지는가**이고, 이름이 바뀌면 터지는 것이 정확히 그 방식이다.

파이썬 블록(`data-lang="py"`)은 안 누른다. Pyodide 를 받아야 하고 그것은 이 검사가
재려는 것과 다른 종류의 느림이다.
"""

import pathlib
import sys

import run as runner
from launch import browser as browser_of

ROOT = runner.ROOT

# **누를 페이지.** 실행 가능한 JS 블록이 있고 borch.ts 를 쓰는 것들이다.
PAGES = [
    "/site/learn/06-save-load.html",
    "/site/ko/learn/06-save-load.html",
    "/site/tutorials/01-quickstart.html",
    "/site/ko/tutorials/01-quickstart.html",
    "/site/tutorials/03-curve-fitting.html",
    "/site/ko/tutorials/03-curve-fitting.html",
    "/site/tutorials/05-adversarial.html",
    "/site/ko/tutorials/05-adversarial.html",
]

# **`04-image-classifier` 는 뺐다.** CIFAR 를 받아 conv 를 여러 에폭 돌리는 페이지라
# 소프트웨어 어댑터에서 몇 분이 든다 — 이 검사가 재려는 것(이름이 바뀌어 터지는가)에
# 비해 값이 안 맞는다. 그 페이지의 `load` 자리는 같은 한 줄이고 위 넷이 그것을 본다.

# **먼저 구조로 본다.** 페이지가 예외를 잡아 화면에 적으므로(`runnable.js` 의
# `write(describeError(err), "err")`) 던진 줄은 `class="err"` 를 달고 온다. 글자가
# 아니라 그 표시를 세면 문구가 바뀌어도, 페이지 언어가 달라도 안 죽는다.
#
# 처음에는 글자로만 봤고 목록에 `"실패"` 가 들어 있었다. **그 패턴은 죽어 있었다** —
# `describeError` 가 내는 것은 `err.name: err.message` 이고 거기 그 낱말이 올 자리가
# 없다. 죽은 패턴은 화면에서 드문 패턴과 구별이 안 된다.
ERROR_CLASS = "div.err"

# 글자 쪽은 **그물이지 문이 아니다.** 안 던지고 잘못 도는 자리를 잡으려는 것이고,
# 여기 없는 문구가 나온다고 통과가 되는 것은 아니다 — 그 문은 위의 `div.err` 가
# 지킨다.
BAD = ("is not a function", "undefined is not", "Cannot read", "TypeError",
       "ReferenceError", "Error:", "throw")

TIMEOUT_MS = 300_000


def run_page(page, path):
    """한 페이지의 JS 블록을 전부 누르고 (통과, 적을 말) 을 돌려준다."""
    page.goto(path)
    page.wait_for_selector("div.runnable button.go", timeout=TIMEOUT_MS)

    said = []
    blocks = page.query_selector_all("div.runnable")
    pressed = 0
    for i, block in enumerate(blocks):
        # 파이썬 블록은 건너뛴다 — Pyodide 를 받는 일이라 여기서 잴 것이 아니다.
        if (block.get_attribute("data-lang") or "js") != "js":
            continue
        go = block.query_selector("button.go")
        if go is None:
            said.append(f"블록 {i} 에 실행 단추가 없다")
            continue
        go.click()
        pressed += 1
        # **단추가 다시 눌리게 되면 끝난 것이다**(`runnable.js` 의 `runBtn.disabled`).
        #
        # 처음에는 단추 **글자**가 "도는 중" 에서 돌아오는 것을 봤다. 그 낱말은 저쪽
        # 파일의 것이고 이 파일이 그 철자를 알고 있어야 하는데, **아무도 그 둘을 붙잡고
        # 있지 않다.** 저쪽이 문구를 고치면 여기는 틀렸다고 말하지 못하고 **영원히
        # 기다린다** — 값이 틀리는 것보다 나쁘다, 무엇을 기다리는지가 화면에 안 나오니까.
        # `disabled` 는 같은 사실을 문구 없이 말한다.
        page.wait_for_function("el => !el.disabled", arg=go, timeout=TIMEOUT_MS)
        out = block.query_selector("pre.out, .out")
        text = (out.inner_text() if out else "").strip()
        if not text:
            said.append(f"블록 {i} 가 아무것도 안 냈다")
            continue
        # **표시가 먼저다.** 페이지가 예외를 잡아 적은 줄이 여기 걸린다.
        for line in (out.query_selector_all(ERROR_CLASS) if out else []):
            said.append(f"블록 {i} — {line.inner_text().strip().splitlines()[0][:120]}")
        for bad in BAD:
            if bad in text:
                said.append(f"블록 {i} — {text.splitlines()[0][:120]}")
                break

    if pressed == 0:
        # **0 건을 돌리고 초록을 보는 것이 제일 나쁜 결과다.**
        said.append("누를 JS 블록이 하나도 없었다 — 선택자가 낡았을 수 있다")
    return not said, pressed, said


def main(argv):
    runner.require_fresh_dist()
    port, stop = runner.serve(ROOT)
    rows = []
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("pageerror", lambda e: rows.append((False, 0, [f"페이지 예외: {e}"])))
            for rel in PAGES:
                ok, pressed, said = run_page(page, f"http://127.0.0.1:{port}{rel}")
                rows.append((rel, ok, pressed, said))
    finally:
        stop()

    bad = 0
    for rel, ok, pressed, said in rows:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {rel} — JS 블록 {pressed}개")
        for line in said:
            print(f"      {line}", file=sys.stderr)
        if not ok:
            bad += 1
    print(f"페이지 {len(rows)}개 중 {len(rows) - bad}개 통과")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    raise SystemExit(main(sys.argv[1:]))
