"""`borch-ts/dist/src/*.d.ts` 에서 API 레퍼런스를 뽑는다.

    npm run build:ts        # 선언 파일부터 만든다
    python3 site/build_api.py

`site/assets/api.json` 이 나오고 `site/api/` 가 그것을 읽는다.

## 왜 뽑는가 — 손으로 적으면 낡는다

이 저장소는 문서의 수가 낡는 것을 이미 네 번 잡았고(`b00e693`·`b3d7453`·`e41c043`,
그리고 `2709`), 네 번 다 **사람이 눈으로** 찾았다. API 목록은 그보다 훨씬 크다 —
텐서 메서드만 수백 개다. 손으로 적은 목록은 첫 주에만 맞는다.

선언 파일은 `tsc` 가 소스에서 만들고 **TSDoc 주석을 그대로 남긴다.** 그래서 설명문의
원본은 언제나 소스이고, 여기서는 옮기지 않는다. 설명이 틀렸으면 소스를 고쳐야 한다 —
그것이 이 방향의 요점이다.

## 설명문은 한국어다

소스 주석이 한국어라 뽑은 설명도 한국어다. **번역본을 여기서 지어내지 않는다** —
지으면 소스와 갈리기 시작하고, 갈린 뒤에는 어느 쪽이 사실인지 아무도 모른다.
영어 페이지는 시그니처·분류·torch 이름 대응을 보여주고(그쪽은 언어 중립이다)
설명문이 소스에서 온 한국어라고 화면에 적는다.

## 무엇을 안 하는가

타입 검사기가 아니다. 선언 파일은 이미 `tsc` 가 만든 것이라 문법이 맞다고 보고,
여기서는 **괄호 깊이만 세어** 선언 하나의 끝을 찾는다. 파서를 제대로 쓰려면
TypeScript 컴파일러 API 를 불러야 하는데, 그러면 이 저장소의 런타임 의존성 0 이
문서 생성에서 깨진다 — 받아야 하는 것은 `typescript` 뿐이지만 그것이 첫 예외가 된다.
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECL = ROOT / "borch-ts" / "dist" / "src"
OUT = ROOT / "site" / "assets" / "api.json"

# 공개 표면. `index.ts` 가 내보내는 것들이고, 순서가 곧 사이드바 순서다.
# 안쪽 사정(`kernels`·`repr`·`functional`)은 여기 없다 — 그것은 쓰는 사람의 것이 아니다.
MODULES = [
    ("tensor", "Tensor", "텐서와 그 위의 연산. `scope`·`keepAlive` 도 여기 있다."),
    ("nn", "nn", "층·손실·활성. `torch.nn` 자리."),
    ("optim", "optim", "옵티마이저와 학습률 스케줄러. `torch.optim` 자리."),
    ("data", "data", "데이터셋과 배치. `torch.utils.data` 자리."),
    ("vision", "vision", "이미지 변환. `torchvision.transforms` 자리."),
    ("fft", "fft", "푸리에 변환. `torch.fft` 자리."),
    ("linalg", "linalg", "선형대수 — 분해·풀이·노름."),
    ("serialize", "serialize", "체크포인트. 형식은 safetensors 다."),
    ("indexing", "indexing", "대괄호 자리 — `x[1:3]` 에 해당하는 것."),
    ("einsum", "einsum", "아인슈타인 합 표기."),
    ("device", "device", "어댑터를 잡고 상태를 묻는 자리."),
    ("random", "random", "난수의 씨앗."),
    ("autograd", "autograd", "기울기 스위치."),
    ("special", "special", "특수함수. `torch.special` 자리."),
    ("rnn", "rnn", "순환 신경망 유틸."),
    ("errors", "errors", "예외 종류. torch 와 같은 이름을 쓴다."),
    ("dtype", "dtype", "자료형."),
]

# 선언의 시작. `export declare class Tensor ... {` 같은 것들.
TOP = re.compile(
    r"^export\s+(?:declare\s+)?(class|abstract class|function|interface|type|const|enum)\s+"
    r"([A-Za-z_$][\w$]*)")
# **재수출은 선언이 아니다.** `export { x } from "./y.js"` 와 `export * as nn from …` 가
# 그것이고, 걸러내지 않으면 `export` 라는 이름의 심볼이 목록에 앉는다(실제로 앉았다).
REEXPORT = re.compile(r"^export\s*[{*]")
# 클래스·인터페이스 안의 한 칸. `add(other: Tensor, alpha?: number): Tensor;`
# 꾸밈말이 여럿 붙는다(`static readonly`), 그리고 물음표가 붙는 칸도 있다
# (`initialLr?: number`). **둘 다 처음에 빠뜨렸고 증상은 조용한 누락이었다** —
# `Module.claim` 과 선택 속성 전부가 목록에서 사라져 있었다.
MEMBER = re.compile(
    r"^\s+(?:(?:static|readonly|get|set|protected|abstract|override|async)\s+)*"
    r"([A-Za-z_$][\w$]*)\??\s*(\(|:|<)")
# 안쪽 것. 선언 파일에도 남지만 쓰는 사람의 것이 아니다.
#
# **`protected` 는 뺐다가 되돌렸다.** 층을 직접 만드는 사람이 부르는 자리가 거기
# 있다 — `Module.claim()` 이 그것이고, 그것 없이 만든 파라미터는 `parameters()` 에
# 안 나오고 **예외 없이** 학습만 안 된다. 확장 표면도 표면이라 목록에 둔다.
PRIVATE = re.compile(r"^\s+private\s")
PROTECTED = re.compile(r"^\s+protected\s")


def _depth(line, depth):
    """괄호 깊이. 문자열 안의 괄호는 안 센다 — 선언 파일에도 `\"constant\"` 가 있다."""
    out = depth
    quote = None
    prev = ""
    for ch in line:
        if quote:
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch in "{[(":
            out += 1
        elif ch in "}])":
            out -= 1
        prev = ch
    return out


TAG = re.compile(r"^@(param|returns?|throws|see|example)\s*(.*)$")


def _split_tags(text):
    """본문과 `@param`·`@returns` 를 가른다.

    떼지 않으면 설명 첫 줄이 `@param alpha …` 로 시작하는 항목이 생긴다 — 화면에서
    그것은 설명이 아니라 표에 들어가야 하는 것이다.
    """
    body, tags = [], []
    cur = None
    for line in text.splitlines():
        hit = TAG.match(line.strip())
        if hit:
            cur = {"tag": hit.group(1), "text": hit.group(2).strip()}
            tags.append(cur)
        elif cur is not None and line.strip():
            cur["text"] += " " + line.strip()
        elif cur is not None:
            cur = None
        else:
            body.append(line)
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(body).strip(), tags


def _doc(block):
    """`/** ... */` 를 사람이 읽는 줄들로. 마크다운은 그대로 둔다 — 화면에서 얇게 그린다."""
    lines = []
    for raw in block:
        line = raw.strip()
        if line.startswith("/**"):
            line = line[3:]
        if line.endswith("*/"):
            line = line[:-2]
        line = re.sub(r"^\s*\*\s?", "", line)
        lines.append(line.rstrip())
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _torch_name(name):
    """camelCase → snake_case. **대응의 힌트이지 동등하다는 말이 아니다.**

    이 저장소가 torch 이름을 그대로 쓰되 자바스크립트 관습으로 적었으므로 기계적으로
    되돌릴 수 있다. 아닌 것들(`call`·`scope`·`keepAlive`)은 `OURS` 에서 뺀다.

    **대문자로 시작하는 것에는 안 단다.** 클래스와 타입은 torch 에서도 대문자
    그대로다 — `Linear` 에 `linear` 를, `PadMode` 에 `pad_mode` 를 달면 없는 이름을
    있다고 적는 것이 된다(실제로 그렇게 나왔다).
    """
    if not name[:1].islower():
        return ""
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return snake if snake != name else ""


# torch 에 없는 이름들. 힌트를 달면 거짓말이 된다.
OURS = {"scope", "keepAlive", "call", "describe", "init", "probe", "isAvailable",
        "currentDevice", "webgpu", "emptyCache", "pooled", "dispatches", "faults",
        "lastScope", "pipelineCount", "synchronize", "adapterInfo"}


def parse(path):
    """선언 파일 하나 → (모듈 설명, 심볼들).

    **깊이는 줄마다 꼭 한 번만 갱신한다.** 처음에는 가지마다 따로 갱신했는데, 어느
    가지가 `continue` 로 빠져나가면 그 줄의 괄호가 안 세어져서 그 뒤로 깊이가 통째로
    어긋났다 — 증상은 예외가 아니라 **조용히 적게 나오는 목록**이었다(텐서 메서드
    수백 개가 18 개로 나왔다). 없는 것과 못 찾은 것이 같은 모양이라 눈에 안 띈다.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    module_doc = ""
    symbols = []
    pending = []          # 바로 위에 붙은 주석
    depth = 0             # 줄 **시작** 시점의 괄호 깊이
    current = None        # 지금 들어가 있는 클래스·인터페이스
    buffer = []           # 여러 줄에 걸친 선언
    skip_to = -1

    for i, line in enumerate(lines):
        if i <= skip_to:
            continue
        stripped = line.strip()
        after = _depth(line, depth)          # 이 줄을 지난 뒤의 깊이

        if stripped.startswith("/**"):
            block, j = [line], i
            while not lines[j].rstrip().endswith("*/"):
                j += 1
                block.append(lines[j])
            skip_to = j
            nxt = lines[j + 1].strip() if j + 1 < len(lines) else ""
            # 파일 맨 앞의 주석이 import 나 재수출 앞에 있으면 모듈 설명이다.
            if (not symbols and not module_doc and depth == 0
                    and (nxt.startswith("import") or nxt.startswith("export {"))):
                module_doc = _doc(block)
                pending = []
            else:
                pending = block
            continue
        if not stripped or stripped.startswith(("*", "//")):
            depth = after
            continue

        if buffer:
            buffer.append(stripped)
            base = 1 if current is not None else 0
            if after <= base and stripped.endswith((";", "}", "{")):
                _emit(" ".join(buffer), pending, current, symbols)
                buffer, pending = [], []
            depth = after
            continue

        if REEXPORT.match(stripped):
            pending = []
            depth = after
            continue

        top = TOP.match(stripped) if depth == 0 else None
        if top:
            kind, name = top.group(1), top.group(2)
            entry = _top_entry(kind, name, stripped, pending, symbols)
            pending = []
            if after > 0 and ("class" in kind or kind == "interface"):
                current = entry              # 몸통으로 들어간다
            elif not stripped.endswith(";"):
                buffer = [stripped]          # 시그니처가 다음 줄로 이어진다
            depth = after
            continue

        if current is not None and depth == 1:
            if stripped.startswith("}"):
                current = None
                pending = []
                depth = after
                continue
            if PRIVATE.match(line) or stripped.startswith("#"):
                pending = []
                depth = after
                continue
            if MEMBER.match(line):
                if after <= 1 and stripped.endswith(";"):
                    _emit(stripped, pending, current, symbols)
                    pending = []
                else:
                    buffer = [stripped]
                depth = after
                continue

        pending = []
        depth = after

    return module_doc, symbols


def _top_entry(kind, name, stripped, pending, symbols):
    """최상위 심볼 하나. 같은 이름이 두 번 나오면(`class` 와 `interface`) 합친다 —
    torch 도 그 둘을 나눠 보여주지 않는다."""
    sig = stripped.rstrip("{").strip().rstrip(";")
    same = next((s for s in symbols if s["name"] == name), None)
    if same:
        if not same["doc"] and pending:
            same["doc"] = _doc(pending)
        return same
    body, tags = _split_tags(_doc(pending) if pending else "")
    entry = {
        "kind": "class" if "class" in kind else kind,
        "name": name,
        "signature": sig,
        "doc": body,
        "members": [],
    }
    if tags:
        entry["tags"] = tags
    if name not in OURS:
        hint = _torch_name(name)
        if hint:
            entry["torch"] = hint
    symbols.append(entry)
    return entry


def _emit(text, pending, current, symbols):
    """모은 선언 한 줄을 심볼 또는 멤버로 넣는다."""
    text = re.sub(r"\s+", " ", text).strip().rstrip(";").rstrip("{").strip()
    if not text or text.startswith(("}", "//")):
        return
    # **여러 줄에 걸친 최상위 선언이 여기로 온다.** 그때 첫 낱말은 `export` 이므로
    # 그것을 이름으로 삼으면 모듈마다 `export` 라는 심볼이 하나씩 앉는다(실제로 앉았다).
    top = TOP.match(text)
    if top and current is None:
        entry = _top_entry(top.group(1), top.group(2), text, pending, symbols)
        entry["signature"] = text
        return
    name = re.sub(r"^(?:(?:static|readonly|get|set|abstract|declare|export|protected|override|async) )+", "", text)
    name = re.split(r"[(:<;=\s]", name, maxsplit=1)[0].strip()
    # 선택 속성의 물음표는 시그니처에 남기고 이름에서는 뗀다 — 이름은 검색어다.
    optional = name.endswith("?")
    name = name.rstrip("?")
    if not name or name.startswith(("#", "[", '"')):
        return
    body, tags = _split_tags(_doc(pending) if pending else "")
    item = {"name": name, "signature": text, "doc": body}
    if text.startswith("protected "):
        item["protected"] = True
    if optional:
        item["optional"] = True
    if tags:
        item["tags"] = tags
    if name not in OURS:
        hint = _torch_name(name)
        if hint:
            item["torch"] = hint

    bag = current["members"] if current is not None else symbols
    old = next((m for m in bag if m["name"] == name), None)
    if old:
        # **오버로드는 버리지 않는다.** `scope()` 는 꼴이 둘이고 둘 다 쓴다.
        sigs = old.setdefault("overloads", [])
        if item["signature"] != old["signature"] and item["signature"] not in sigs:
            sigs.append(item["signature"])
        if not old["doc"] and item["doc"]:
            old["doc"] = item["doc"]
        return
    if current is None:
        item = {"kind": "function", **item, "members": []}
    bag.append(item)


def main():
    if not DECL.exists():
        raise SystemExit(f"선언 파일이 없다: {DECL}\n  먼저: npm run build:ts")

    modules = []
    for name, title, blurb in MODULES:
        path = DECL / f"{name}.d.ts"
        if not path.exists():
            continue
        doc, symbols = parse(path)
        # 이름만 있고 설명도 시그니처도 없는 것은 버린다 — 목록만 부풀린다.
        symbols = [s for s in symbols if s["signature"]]
        modules.append({
            "name": name, "title": title, "blurb": blurb,
            "doc": doc, "symbols": symbols,
            "count": sum(1 + len(s["members"]) for s in symbols),
        })

    payload = {
        "source": "borch-ts/dist/src/*.d.ts",
        "note": "설명문은 소스의 TSDoc 을 그대로 옮긴 것이다. 고칠 곳은 소스다.",
        "modules": modules,
        "total": sum(m["count"] for m in modules),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)} — 모듈 {len(modules)}개 · 항목 {payload['total']}개")
    for m in modules:
        print(f"  {m['name']:<12} {m['count']:>5}")


if __name__ == "__main__":
    main()
