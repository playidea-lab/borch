"""Pulls the API reference out of `borch-ts/dist/src/*.d.ts`.

    npm run build:ts        # the declaration files come first
    python3 site/build_api.py

It produces `site/assets/api.json`, and `site/api/` reads it.

## Why it is generated — written by hand, it goes stale

This repository has already caught four stale numbers in its own documentation
(`b00e693`, `b3d7453`, `e41c043` and `2709`), and all four were found **by eye**. The API
index is far larger than that — the tensor methods alone run to hundreds. An index
written by hand is right for the first week.

The declaration files are made from the source by `tsc` and **keep the TSDoc comments as
they are.** So the original of every description is always the source, and nothing is
transcribed here. A wrong description has to be fixed in the source — that direction is
the point.

## The descriptions are English, and the Korean sits beside them

The source comments are English, so the descriptions pulled out are English. The Korean
lives in `site/api_ko.json`, not here and not in the source, and every entry carries a
fingerprint of the English it was made from. The fear this file used to record — that a
translation drifts from the source the day it is written — is right, and the fingerprint
is the answer to it: drift is not prevented, it is prevented from being quiet.

The roles used to be the other way round, and the table turned over rather than being
rewritten (`feb5e5d`). Which direction is the original matters here: the original is
whatever `tsc` copies out of the source, and the translation is whatever a fingerprint
has to guard.

Signatures, kinds and the torch name mapping are language-neutral and identical on both
pages.

## What it does not do

It is not a type checker. The declaration files were made by `tsc` and so are assumed to
be syntactically sound; here **only bracket depth is counted** to find where a
declaration ends. A proper parser would mean calling the TypeScript compiler API, and
that breaks this repository's zero runtime dependencies at documentation time — one
package, `typescript`, but it would be the first exception.
"""

import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECL = ROOT / "borch-ts" / "dist" / "src"
OUT = ROOT / "site" / "assets" / "api.json"
# A small table finding a place from one name. `code` in the lesson prose reads it and
# becomes a link — a 350KB body cannot be loaded on every lesson page.
INDEX = ROOT / "site" / "assets" / "api-index.json"

# The public surface — what `index.ts` exports, in the order the sidebar shows.
# Internal business (`kernels`, `repr`, `functional`) is not here; it is not the user's.
MODULES = [
    # **Start at the package root.** `index.ts` is mostly re-exports, but **some things are
    # declared only there** — `isTensor` is one, and the binding uses it. While this file
    # went unswept, that name was in neither the reference nor the name index. Another
    # session saw the hole first, in a check of its own.
    ("index", "borch",
     {"ko": "`import … from \"borch\"` 가 바로 주는 것. 나머지는 아래 이름 공간에 있다.",
      "en": "What `import … from \"borch\"` hands you directly. The rest is in the namespaces below."}),
    ("version", "VERSION",
     {"ko": "이 배포판의 판 번호. 매니페스트의 `runtime.ts` 범위를 대조할 때 쓴다.",
      "en": "Which build this is. Used when comparing a manifest's `runtime.ts` range."}),
    ("tensor", "Tensor",
     {"ko": "텐서와 그 위의 연산. `scope`·`keepAlive` 도 여기 있다.",
      "en": "Tensors and the operations on them. `scope` and `keepAlive` live here too."}),
    ("nn", "nn",
     {"ko": "층·손실·활성. `torch.nn` 자리.",
      "en": "Layers, losses, activations. Where `torch.nn` would be."}),
    # **The same hole as `index` above, one module over.** `functional.ts` exists so
    # that a line written `F.conv2d(x, w, b)` can be copied across, and it was in
    # neither the reference nor the name index — so `tests/ts_axis.py` could not see
    # a single name that lives only there. Eighteen `F.` names were added and the
    # gap count moved by two: the two that also became `Tensor` methods.
    #
    # The note above this list describes that failure and names `isTensor` as the
    # one it caught. **It was written, and the next module along was not checked
    # against it** — a reason that is true, in place, about a case adjacent to the
    # one in front of it.
    ("functional", "functional",
     {"ko": "`torch.nn.functional` 자리. 값은 전부 `Tensor` 메서드에서 온다 — 여기는 이름과 인자 순서뿐이다.",
      "en": "Where `torch.nn.functional` would be. Every value comes from a `Tensor` "
            "method; this module is one name and one argument order."}),
    ("optim", "optim",
     {"ko": "옵티마이저와 학습률 스케줄러. `torch.optim` 자리.",
      "en": "Optimizers and learning-rate schedulers. Where `torch.optim` would be."}),
    ("data", "data",
     {"ko": "데이터셋과 배치. `torch.utils.data` 자리.",
      "en": "Datasets and batching. Where `torch.utils.data` would be."}),
    ("vision", "vision",
     {"ko": "이미지 변환. `torchvision.transforms` 자리.",
      "en": "Image transforms. Where `torchvision.transforms` would be."}),
    ("vision_v2", "vision_v2",
     {"ko": "v2 의 함수형 변환. `torchvision.transforms.v2.functional` 자리 — "
            "새 이름 아홉과, 나머지는 v1 을 그대로 내보낸다.",
      "en": "v2's functional transforms. Where "
            "`torchvision.transforms.v2.functional` would be — nine new names, and "
            "the rest bound straight through to v1."}),
    # **A module that is not listed here is not stale, it is unseen.** `vision_v2`
    # re-exports these, and a re-export does not appear in its own declaration file —
    # so thirty-six classes were invisible to this reference while every check on it
    # stayed green.
    #
    # `ops` was the same failure a second time and it lasted longer, because nothing
    # about it looked wrong: `ops.ts` exists, its eleven functions have golden cases,
    # and those cases pass. What they measure is the **value**. The name and the
    # argument list are measured by `ts_axis.py` and `ts_signatures.py`, and both of
    # those read this generator's output — so a module off this list is off both axes,
    # and asking them about `ops` answered "borch.ts does not have `nms`" about a file
    # that has it. Measured: eleven names, all eleven reported absent.
    ("ops", "ops",
     {"ko": "박스 기하 — IoU·NMS. `torchvision.ops` 자리.",
      "en": "Box geometry — IoU and NMS. Where `torchvision.ops` would be."}),
    ("datasets", "datasets",
     {"ko": "데이터셋 **디코더**. `torchvision.datasets` 의 형식 절반 — 주소 절반"
            "(내려받기·캐시·체크섬)은 여기 없다. 조용히 틀리는 쪽은 형식이다.",
      "en": "The dataset **decoders**. The format half of `torchvision.datasets` — "
            "the address half (fetching, caching, checksums) is not here. The format "
            "is the half that goes wrong quietly."}),
    ("vision_v2_twins", "vision_v2_twins",
     {"ko": "v2 의 변환 클래스. `torchvision.transforms.v2` 자리 — v1 의 계산에 "
            "v2 의 출력 표면을 씌운 것이고, 다른 것은 print 되는 내용뿐이다.",
      "en": "v2's transform classes. Where `torchvision.transforms.v2` would be — "
            "v1's arithmetic under v2's printed surface, and what differs between "
            "the two namespaces is only what they print."}),
    ("fft", "fft",
     {"ko": "푸리에 변환. `torch.fft` 자리.",
      "en": "Fourier transforms. Where `torch.fft` would be."}),
    ("linalg", "linalg",
     {"ko": "선형대수 — 분해·풀이·노름.",
      "en": "Linear algebra — decompositions, solves, norms."}),
    ("serialize", "serialize",
     {"ko": "체크포인트. 형식은 safetensors 다.",
      "en": "Checkpoints. The format is safetensors."}),
    ("indexing", "indexing",
     {"ko": "대괄호 자리 — `x[1:3]` 에 해당하는 것.",
      "en": "The bracket position — what `x[1:3]` would be."}),
    ("einsum", "einsum",
     {"ko": "아인슈타인 합 표기.", "en": "Einstein summation."}),
    ("device", "device",
     {"ko": "어댑터를 잡고 상태를 묻는 자리.",
      "en": "Acquiring the adapter, and asking it what it is doing."}),
    ("random", "random", {"ko": "난수의 씨앗.", "en": "Seeding the random draws."}),
    ("autograd", "autograd", {"ko": "기울기 스위치.", "en": "The gradient switch."}),
    ("special", "special",
     {"ko": "특수함수. `torch.special` 자리.",
      "en": "Special functions. Where `torch.special` would be."}),
    ("rnn", "rnn", {"ko": "순환 신경망 유틸.", "en": "Recurrent-network utilities."}),
    ("errors", "errors",
     {"ko": "예외 종류. torch 와 같은 이름을 쓴다.",
      "en": "Exception types, under the names torch uses."}),
    ("dtype", "dtype", {"ko": "자료형.", "en": "Data types."}),
]

# English names for the sections the source divided itself into. **Only the ones with a
# translation are carried across** — otherwise the source's own name is shown. Inventing
# one here is where drift from the source begins.
SECTION_EN = {
    "픽셀 연산 여섯, 그리고 그것을 확률로 감싸는 여섯":
        "Six pixel operations, and the six that apply them with a probability",
    "만들기": "Creating", "원소별": "Elementwise", "행렬곱": "Matrix products",
    "축약": "Reductions", "모양": "Shape", "창 펴기": "Windows",
    "나머지 층이 쓰는 것들": "Used by the layers", "자리 옮기기": "Moving elements",
    "손실과 거리": "Losses and distances", "addmm 계열": "The addmm family",
    "결과 크기가 값에 달린 것들": "Output size depends on the values",
    "선형대수": "Linear algebra", "최상위 선형대수": "Linear algebra (top level)",
    "제자리 연산": "In place", "정렬 계열": "Sorting",
    "번호표로 읽고 쓰기": "Indexed read and write", "합성곱·풀링": "Convolution and pooling",
    "이긴 자리를 함께 내는 풀링": "Pooling that also returns indices",
    "CTC": "CTC", "복소수": "Complex", "역전파": "Backward",
    "장치 옮기기": "Moving between devices", "읽기": "Reading values",
    "복소수 커널": "Complex kernels",
    "torch 의 둘째 철자들": "torch's second spellings",
    "결속에만 있던 이름들": "Names that existed only in the binding",
    "제자리 판 서른여덟": "The thirty-eight in-place forms",
    "커널 표에는 있는데 이름이 없던 것들": "In the kernel tables, with no name to type",
    "표가 다는 제자리 판": "In-place forms the tables attach",
}

# The start of a declaration — things like `export declare class Tensor ... {`.
TOP = re.compile(
    r"^export\s+(?:declare\s+)?(class|abstract class|function|interface|type|const|enum)\s+"
    r"([A-Za-z_$][\w$]*)")
# **A re-export is not a declaration.** `export { x } from "./y.js"` and
# `export * as nn from …` are those, and unfiltered a symbol named `export` sits in the
# index (which it did).
REEXPORT = re.compile(r"^export\s*[{*]")
# One slot inside a class or interface: `add(other: Tensor, alpha?: number): Tensor;`
# Several modifiers attach (`static readonly`), and some slots carry a question mark
# (`initialLr?: number`). **Both were missed at first and the symptom was a quiet
# absence** — `Module.claim` and every optional property had vanished from the index.
MEMBER = re.compile(
    r"^\s+(?:(?:static|readonly|get|set|protected|abstract|override|async)\s+)*"
    r"([A-Za-z_$][\w$]*)\??\s*(\(|:|<)")
# Internal. It survives into the declaration file but it is not the user's.
#
# **`protected` was taken out and put back.** A place someone building their own layer
# calls lives there — `Module.claim()` — and a parameter made without it is absent from
# `parameters()`, with **no exception** and only the training failing. An extension
# surface is a surface, so it stays in the index.
PRIVATE = re.compile(r"^\s+private\s")
PROTECTED = re.compile(r"^\s+protected\s")


def _depth(line, depth):
    """Bracket depth. Brackets inside strings are not counted — declaration files contain `\"constant\"` too."""
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
    """Separates the body from `@param` and `@returns`.

    Left attached, entries appear whose description begins `@param alpha …` — and on
    screen that is not a description but something that belongs in a table.
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
    """`/** ... */` into lines a person reads. The markdown is left alone — the screen draws it thinly."""
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
    """camelCase → snake_case. **A hint at the correspondence, not a claim of equality.**

    This repository uses torch's names written in JavaScript convention, so the mapping
    reverses mechanically. The ones that are not (`call`, `scope`, `keepAlive`) are
    excluded through `OURS`.

    **Nothing is attached to a name starting with a capital.** Classes and types keep
    their capitals in torch too — attaching `linear` to `Linear`, or `pad_mode` to
    `PadMode`, writes down a name that does not exist (which is what came out).
    """
    if not name[:1].islower():
        return ""
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return snake if snake != name else ""


# Names torch does not have. A hint attached to these would be a lie.
OURS = {"scope", "keepAlive", "call", "describe", "init", "probe", "isAvailable",
        "currentDevice", "webgpu", "emptyCache", "pooled", "dispatches", "faults",
        "lastScope", "pipelineCount", "synchronize", "adapterInfo"}


SECTION = re.compile(r"^\s*// ── (.+?) ─")
TS_CLASS = re.compile(r"^export\s+(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")
# **An interface opens a body too.** The `export interface Tensor` at the end of
# `tensor.ts` holds the mirrored names, and unread, those fifty-five **inherit the name
# of the section just above** (measured: every one came out as "complex kernels"). The
# section ends here — the mirrored names are not a place the author divided but names
# that attach automatically.
TS_IFACE = re.compile(r"^export\s+interface\s+([A-Za-z_$][\w$]*)")
TS_MEMBER = re.compile(r"^  (?:(?:static|readonly|get|set|protected|override|async|abstract)\s+)*"
                       r"([A-Za-z_$][\w$]*)\??\s*[(<:]")


def sections_of(name):
    """`(class, member) → section name`. **What the source already divided is used as it is.**

    `tensor.ts` carries twenty-four sections such as `// ── 축약 ──`. That is the
    classification of the person who wrote this surface, and dividing it again covers
    their judgement with our guess.

    **It does not survive into the declaration files** — `tsc` carries the TSDoc and drops
    line comments. So this is the one place that reads the source. The same reason the
    original of every description is the source.
    """
    path = ROOT / "borch-ts" / "src" / f"{name}.ts"
    if not path.exists():
        return {}
    out = {}
    current_class = ""
    current_section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        head = SECTION.match(line)
        if head:
            current_section = head.group(1).strip()
            continue
        klass = TS_CLASS.match(line) or TS_IFACE.match(line)
        if klass:
            current_class = klass.group(1)
            # A new class opening means the previous section is not that class's.
            current_section = ""
            continue
        member = TS_MEMBER.match(line)
        if member and current_class and current_section:
            out.setdefault((current_class, member.group(1)), current_section)
    return out


_TUPLE_ALIAS = re.compile(
    r"^type\s+(\w+)\s*=\s*\[(.*?)\];", re.M | re.S)
_REST_ARG = re.compile(r"\.\.\.\w+\s*:\s*(\w+)")


def _tuple_aliases(text):
    """`{name: "a: number, b?: number"}` for every **labelled tuple** alias.

    TypeScript lets a tuple's elements carry names — `type ConvArgs = [outChannels:
    number, kernelSize: number, …]` — and a class then spreads it as
    `constructor(...a: ConvArgs)`. The editor shows the labels; a reader of the
    declaration file sees one parameter called `a`.

    Only labelled tuples are collected. An unlabelled `[number, number]` has no names
    to give and expanding it would invent `arg0`, `arg1` — **a fold that manufactures
    the thing it is folding**, which is worse than the rest parameter it replaces.
    """
    out = {}
    for name, body in _TUPLE_ALIAS.findall(text):
        parts, depth, current = [], 0, ""
        for ch in body:
            if ch in "<([{":
                depth += 1
            elif ch in ">)]}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())
        if parts and all(re.match(r"^\w+\??\s*:", p) for p in parts):
            out[name] = ", ".join(parts)
    return out


def _expand_rest(sig, aliases):
    """`(...a: ConvArgs)` → the tuple's own labelled parameters.

    **The six `LazyConv` classes were reported `unaligned` because of this**, which is
    a *wrong* verdict rather than an absent one: the axis read one parameter called
    `a` against the core's ten, could line up neither, and said so. Expanded they are
    `shorter` — torch's tail is missing and that is a thing to act on.

    A signature the reader cannot resolve is not neutral. It lands in the bucket that
    means *somebody has to look at this*, and then nobody does, because the row says
    nothing about what is wrong.
    """
    def swap(m):
        return aliases.get(m.group(1), m.group(0))

    return _REST_ARG.sub(swap, sig) if aliases else sig


def parse(path):
    """One declaration file → (module description, symbols).

    **The depth is updated exactly once per line.** It was first updated inside each
    branch, and a branch leaving through `continue` left that line's brackets uncounted,
    which threw the depth out for everything after — the symptom was not an exception but
    **an index quietly coming up short** (hundreds of tensor methods arrived as eighteen).
    Absent and not-found have the same shape, so nothing draws the eye.
    """
    text = path.read_text(encoding="utf-8")
    aliases = _tuple_aliases(text)
    lines = text.splitlines()
    module_doc = ""
    symbols = []
    pending = []          # the comment attached just above
    depth = 0             # bracket depth at the **start** of the line
    current = None        # the class or interface currently open
    buffer = []           # a declaration spanning several lines
    skip_to = -1

    for i, line in enumerate(lines):
        if i <= skip_to:
            continue
        stripped = line.strip()
        after = _depth(line, depth)          # the depth after this line

        if stripped.startswith("/**"):
            block, j = [line], i
            while not lines[j].rstrip().endswith("*/"):
                j += 1
                block.append(lines[j])
            skip_to = j
            nxt = lines[j + 1].strip() if j + 1 < len(lines) else ""
            # A comment at the head of the file, ahead of imports or re-exports, is the
            # module description.
            #
            # **The next line was allowed to be an import and nothing else, and that
            # made the module description depend on an accident.** A declaration file
            # keeps an import only when a type it names survives into a signature, so a
            # module whose imports are all values — thrown errors, called helpers —
            # opens with the head comment and then a declaration. The description was
            # then filed as the first symbol's `pending` and thrown away when that
            # symbol turned out to have its own.
            #
            # Six modules were in that state at once: `indexing`, `device`, `random`,
            # `autograd`, `errors` and `dtype`. Each has a written description in its
            # source and an empty one in the reference, and nothing cried — measured,
            # not guessed.
            #
            # The added clause is **a doc comment starting on the next line**, which is
            # what tells the two apart. A head comment followed by another `/**` cannot
            # be that symbol's description, because the symbol already has one.
            #
            # `dtype` stays empty after this and that is the honest end of the rule: its
            # head comment is followed by an undocumented `export type DType`, so a
            # machine cannot tell a module description from that symbol's. Widening the
            # clause to any declaration would steal a symbol's own doc elsewhere.
            if (not symbols and not module_doc and depth == 0
                    and (nxt.startswith("import") or nxt.startswith("export {")
                         or nxt.startswith("/**"))):
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
                current = entry              # step into the body
            elif not stripped.endswith(";"):
                buffer = [stripped]          # the signature continues onto the next line
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

    # **Expanded once, at the end, over everything the file produced.** Doing it at
    # each of the three places a signature is set would be three places to forget,
    # and the alias can be declared after the class that spreads it — `ConvArgs` is.
    def expand(item):
        item["signature"] = _expand_rest(item["signature"], aliases)
        # **The overloads too.** `scope()` has two shapes and both are kept; a fold
        # applied to the first alone would leave the second reading `...a: ConvArgs`
        # beside a sibling that reads the real names, which looks like two different
        # functions.
        for i, sig in enumerate(item.get("overloads", ())):
            item["overloads"][i] = _expand_rest(sig, aliases)

    for symbol in symbols:
        expand(symbol)
        for member in symbol.get("members", ()):
            expand(member)

    return module_doc, symbols


def _top_entry(kind, name, stripped, pending, symbols):
    """One top-level symbol. A name appearing twice (as `class` and as `interface`) is merged —
    torch does not show those two apart either."""
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
    """Files one gathered declaration line as a symbol or a member."""
    text = re.sub(r"\s+", " ", text).strip().rstrip(";").rstrip("{").strip()
    if not text or text.startswith(("}", "//")):
        return
    # **A top-level declaration spanning several lines arrives here.** Its first word is
    # `export`, so taking that as the name seats one symbol called `export` per module
    # (which it did).
    top = TOP.match(text)
    if top and current is None:
        entry = _top_entry(top.group(1), top.group(2), text, pending, symbols)
        entry["signature"] = text
        return
    name = re.sub(r"^(?:(?:static|readonly|get|set|abstract|declare|export|protected|override|async) )+", "", text)
    name = re.split(r"[(:<;=\s]", name, maxsplit=1)[0].strip()
    # An optional property's question mark stays in the signature and comes off the name — the name is a search term.
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
        # **Overloads are not discarded.** `scope()` has two shapes and both are used.
        sigs = old.setdefault("overloads", [])
        if item["signature"] != old["signature"] and item["signature"] not in sigs:
            sigs.append(item["signature"])
        if not old["doc"] and item["doc"]:
            old["doc"] = item["doc"]
        return
    if current is None:
        item = {"kind": "function", **item, "members": []}
    bag.append(item)


# Where the Korean descriptions live. **Here, not in the source** — the TSDoc is English.
#
# It used to be the other way round. The source was Korean, this file held the English,
# and the fingerprint caught an English translation whose Korean source had moved. Once
# the comments became English the direction had to turn over, and turning it over is the
# more useful direction: the source will change in English from now on, so the side at
# risk of going stale is the Korean.
#
# The mechanism is unchanged. Every entry carries **a hash of the English it was made
# from**, and when the source moves this says so by name. Drift cannot be prevented; being
# quiet about it can.
KO = ROOT / "site" / "api_ko.json"


def _key(mod, sym=None, member=None):
    """The key naming one description — module, symbol and member on one line."""
    if sym is None:
        return f"{mod}/"
    return f"{mod}/{sym}" + (f".{member}" if member else "")


def _fingerprint(text):
    """The source the translation saw. Only the outer whitespace is stripped — changed wrapping does not make it stale."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12]


def attach_korean(modules):
    """Attaches `doc_ko`, and returns counts of the stale and the missing."""
    table = json.loads(KO.read_text(encoding="utf-8")) if KO.exists() else {}
    stale, missing, done = [], [], 0
    for mod in modules:
        rows = [(_key(mod["name"]), mod)]
        for sym in mod["symbols"]:
            rows.append((_key(mod["name"], sym["name"]), sym))
            for mem in sym["members"]:
                rows.append((_key(mod["name"], sym["name"], mem["name"]), mem))
        for key, node in rows:
            src = (node.get("doc") or "").strip()
            if not src:
                continue
            got = table.get(key)
            if not got:
                missing.append(key)
            elif got.get("src") != _fingerprint(src):
                stale.append(key)
            else:
                node["doc_ko"] = got["ko"]
                done += 1
    return done, stale, missing


def main():
    if not DECL.exists():
        raise SystemExit(f"no declaration files: {DECL}\n  first: npm run build:ts")

    modules = []
    for name, title, blurb in MODULES:
        path = DECL / f"{name}.d.ts"
        if not path.exists():
            continue
        doc, symbols = parse(path)
        marks = sections_of(name)
        for sym in symbols:
            for member in sym["members"]:
                mark = marks.get((sym["name"], member["name"]))
                if mark:
                    member["section"] = {"ko": mark, "en": SECTION_EN.get(mark, mark)}
        # A name with neither description nor signature is dropped — it only pads the index.
        symbols = [s for s in symbols if s["signature"]]
        modules.append({
            "name": name, "title": title, "blurb": blurb,
            "doc": doc, "symbols": symbols,
            "count": sum(1 + len(s["members"]) for s in symbols),
        })

    done, stale, missing = attach_korean(modules)

    payload = {
        "source": "borch-ts/dist/src/*.d.ts",
        "note": "The descriptions are the source's own TSDoc; fix them in the source. "
                "The Korean lives in site/api_ko.json and says so when its source has moved.",
        "korean": {"done": done, "stale": len(stale), "missing": len(missing)},
        "modules": modules,
        "total": sum(m["count"] for m in modules),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")

    if stale or missing:
        print(f"  Korean descriptions — attached {done} · stale {len(stale)} · missing {len(missing)}")
        for key in stale[:5]:
            print(f"    stale: {key}")
    else:
        print(f"  Korean descriptions — attached {done} (none stale, none missing)")

    # **First seated wins.** A name living in several classes, such as `forward`, goes to
    # the first place — sending it somewhere beats sending it nowhere, and the module order
    # is the order of importance (Tensor first).
    index = {}
    for mod in modules:
        for sym in mod["symbols"]:
            index.setdefault(sym["name"], f"{mod['name']}.{sym['name']}")
            for member in sym["members"]:
                index.setdefault(member["name"],
                                 f"{mod['name']}.{sym['name']}.{member['name']}")
    INDEX.write_text(json.dumps(index, ensure_ascii=False, sort_keys=True) + "\n",
                     encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)} — {len(modules)} modules · {payload['total']} entries")
    print(f"{INDEX.relative_to(ROOT)} — {len(index)} names")
    for m in modules:
        print(f"  {m['name']:<12} {m['count']:>5}")


if __name__ == "__main__":
    main()
