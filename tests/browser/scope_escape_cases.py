"""구역 밖으로 텐서를 들고 나오는 두 길이 **다르게** 도는지 잰다.

거절하는 쪽도 같이 본다 — `keep` 없이 만든 것이 살아 있으면 구역이 일을 안 하는
것이고, 그러면 학습 루프가 메모리로 무너진다.
"""

import borch_webgpu as torch

_lines = []


def _say(ok, name, note=""):
    _lines.append(f"  {'○' if ok else '×'} {name}{' — ' + note if note else ''}")
    return ok


def _usable(t):
    """이 텐서를 아직 쓸 수 있는가. 죽었으면 저쪽이 멈춘다."""
    try:
        t.sum().item()
        return True
    except Exception:
        return False


def run():
    ok = []

    # 1) 아무것도 안 하면 죽어야 한다. 이것이 참이어야 나머지가 뜻을 갖는다.
    with torch.scope():
        loose = torch.randn(4)
    ok.append(_say(not _usable(loose), "keep 없이 만든 것은 구역 뒤에 죽는다"))

    # 2) scope.keep — 이번 구역을 벗어난다.
    with torch.scope() as s:
        carried = s.keep(torch.randn(4))
    ok.append(_say(_usable(carried), "scope.keep 은 구역 밖에서 산다"))

    # 3) keep_alive — 영구.
    with torch.scope():
        forever = torch.keep_alive(torch.randn(4))
    ok.append(_say(_usable(forever), "keep_alive 는 구역 밖에서 산다"))

    # 4) 준 것을 그대로 돌려주는가. 자매가 그렇고, 아니면 `x = keep_alive(x)` 가
    #    다른 텐서를 가리키게 된다.
    t = torch.randn(4)
    ok.append(_say(torch.keep_alive(t) is t, "keep_alive 는 준 것을 그대로 돌려준다"))
    with torch.scope() as s:
        u = torch.randn(4)
        ok.append(_say(s.keep(u) is u, "scope.keep 도 준 것을 그대로 돌려준다"))
        s.keep(u)

    # 5) **둘의 차이.** 안쪽에서 살린 것을 바깥이 닫을 때 어떻게 되는가.
    with torch.scope():
        with torch.scope() as inner:
            promoted = inner.keep(torch.randn(4))
            permanent = torch.keep_alive(torch.randn(4))
        both = _usable(promoted) and _usable(permanent)
        ok.append(_say(both, "중첩: 안쪽을 나온 직후에는 둘 다 산다"))
    ok.append(_say(not _usable(promoted),
                   "바깥이 닫히면 scope.keep 한 것은 놓인다"))
    ok.append(_say(_usable(permanent),
                   "바깥이 닫혀도 keep_alive 한 것은 산다"))

    # 6) 호스트에 있는 텐서를 거절하지 않는다 — 살릴 것이 없을 뿐이다.
    try:
        with torch.scope() as s:
            s.keep(torch.randn(4).cpu())
        ok.append(_say(True, "CPU 텐서를 줘도 멈추지 않는다"))
    except Exception as e:
        ok.append(_say(False, "CPU 텐서를 줘도 멈추지 않는다", str(e)))

    # 7) 텐서가 아닌 것은 거절한다.
    try:
        torch.keep_alive(3)
        ok.append(_say(False, "텐서가 아니면 거절한다", "안 거절했다"))
    except TypeError:
        ok.append(_say(True, "텐서가 아니면 거절한다"))

    # 8) **문구가 참인가.** 없는 이름과, borch.ts 에 모듈 함수로 있는 이름을 가른다.
    try:
        torch.definitely_not_a_kernel
        ok.append(_say(False, "없는 이름은 없다고 말한다", "안 멈췄다"))
    except AttributeError as e:
        ok.append(_say("없다" in str(e) and "definitelyNotAKernel" in str(e),
                       "없는 이름은 없다고 말한다", str(e)))
    # `gradMode` 는 `index.ts` 가 내보내고, `Tensor.prototype` 에는 없고, 이 결속이
    # 아직 안 이었다 — 셋을 다 만족하는 이름이라 이 문구가 갈리는 자리를 정확히 짚는다.
    #
    # 처음에 `make_node` 로 잡았다가 틀렸다. 그것은 `tensor.ts` 에 있지만 `index.ts`
    # 가 안 내보내므로 `js.borch` 에서 안 보이고, 그러면 "없다" 가 **맞는 말**이다.
    # 결속이 보는 borch.ts 는 공개된 표면이지 소스 전체가 아니다.
    try:
        torch.grad_mode
        ok.append(_say(False, "모듈 함수는 모듈 함수라고 말한다", "안 멈췄다"))
    except AttributeError as e:
        ok.append(_say("모듈 함수" in str(e), "모듈 함수는 모듈 함수라고 말한다", str(e)))

    head = "구역 탈출이 돈다" if all(ok) else "**어딘가 안 된다**"
    return "\n".join([head, *_lines])
