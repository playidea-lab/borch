"""여러 값이 **한 칸으로 접힐 때** 기울기가 어디로 돌아가는가 — 남은 절반.

**표는 골든으로 갔다**(`grad::접힘::*`, 스물넷). 셋이 다 답하게 됐으므로 셋을 함께
묻는 자리로 옮긴 것이고, 여기 남은 것은 **골든이 물을 수 없는 것**뿐이다.

## 규칙 (골든의 스물넷이 지키는 것)

    번호를 **건네는** 연산은 고른 자리 하나로   (max(dim)·mode·nanmedian(dim)·kthvalue)
    번호를 **안 건네는** 연산은 값이 같은 칸에 고르게 (max()·min()·median()·nanmedian())
    **정렬 자리**로 접는 연산은 그 자리들로     (quantile — 보간이면 둘)

동점이 있어야 열리는 물음이다. 값이 전부 다르면 세 규칙이 같은 답을 내므로 절반만
맞는 구현이 통과한다 — 골든의 기울기 케이스가 오래 그 상태였다.

## 잡은 것 (이력)

일곱 자리가 갈렸고 갈래는 둘이었다. 다섯은 `Tensor(...)` 를 맨손으로 만들어 **그래프가
아예 없었고**(값 검사는 값이 맞아서 통과한다), 둘은 이어져 있는데 규칙이 틀렸다 —
`median` 에는 "나머지 원소를 흔들어도 답이 안 움직인다" 는 근거까지 적혀 있었는데,
동점이 아닐 때만 맞는 말이었다.

borch.ts 쪽에서는 `i0` 이 **0 을 흘리고** 있었고 그 주석이 코어의 구멍을 근거로
대고 있었다. 값이 0 인 기울기와 기울기가 없는 것은 다른 말인데, 베낄 때 뒤가 앞으로
바뀌었다.
"""

import numpy as np

import borch

EVEN = [1.0, 5.0, 5.0, 5.0]


def test_median_and_quantile_disagree_on_ties():
    """**같은 값을 내면서 다른 것을 떠받치는 두 연산.** 이 한 줄이 표를 설명한다.

    [1,5,5,5] 의 중앙값도 0.5 분위수도 5 다. 그런데 median 은 5 인 칸 **셋 전부**에
    ⅓ 씩 주고, quantile 은 정렬해서 쓴 **두 자리**에 ½ 씩 준다. 값만 재면 둘이 같아
    보이므로 이 갈림은 되짚어야만 드러난다.

    골든에도 두 케이스가 다 있지만 **나란히 놓여 있지는 않다.** 여기서 한 함수 안에
    붙여 두는 이유는, 표에서는 둘이 각자 맞는지만 보이고 **서로 다르다는 것**은
    안 보이기 때문이다.
    """
    def grad(fn):
        t = borch.tensor(EVEN, requires_grad=True)
        fn(t).backward()
        return [round(v, 4) for v in t.grad.tolist()]

    assert grad(lambda t: t.median()) == [0.0, 0.3333, 0.3333, 0.3333]
    assert grad(lambda t: t.quantile(0.5)) == [0.0, 0.5, 0.5, 0.0]


def test_a_folding_op_that_does_not_carry_grad_is_a_defect_not_a_choice():
    """접히는 연산은 **하나도 빠짐없이** 기울기를 나른다.

    끊긴 그래프는 값 검사로 안 잡힌다 — 값은 맞기 때문이다. 골든도 못 잡는다.
    골든이 묻는 것은 "이 기울기가 맞는가" 이고, 여기서 묻는 것은 **"기울기가 있는가"**
    다. 새 축약을 넣는 사람이 `Tensor(...)` 로 결과를 만들면 이 검사가 그 이름을
    대며 빨개진다 — 케이스를 같이 안 넣어도 걸린다는 것이 요점이다.
    """
    tie = [3.0, 5.0, 5.0, 1.0, 5.0]
    nan = [3.0, float("nan"), 5.0, 1.0, 5.0]
    folding = [
        ("max()", tie, lambda t: t.max()),
        ("min()", tie, lambda t: t.min()),
        ("amax()", tie, lambda t: t.amax()),
        ("amin()", tie, lambda t: t.amin()),
        ("median()", tie, lambda t: t.median()),
        ("median(dim=0)", tie, lambda t: t.median(dim=0).values),
        ("nanmedian()", nan, lambda t: t.nanmedian()),
        ("nanmedian(dim=0)", nan, lambda t: t.nanmedian(0).values),
        ("mode()", [1.0, 1.0, 2.0, 2.0], lambda t: t.mode().values),
        ("kthvalue(2)", tie, lambda t: t.kthvalue(2).values),
        ("quantile(0.5)", tie, lambda t: t.quantile(0.5)),
        ("norm(inf)", tie, lambda t: t.norm(float("inf"))),
        ("norm(3)", tie, lambda t: t.norm(3)),
        ("angle()", [0.5, -1.0, 2.0], lambda t: t.angle()),
        ("i0()", [0.5, -1.0, 2.0], lambda t: t.i0()),
        ("topk(3)", tie, lambda t: t.topk(3).values),
        ("sort()", tie, lambda t: t.sort().values),
        ("cummax(0)", tie, lambda t: t.cummax(0).values),
    ]
    stuck = []
    for name, values, fn in folding:
        out = fn(borch.tensor(values, requires_grad=True))
        if not out.requires_grad:
            stuck.append(name)
    assert not stuck, (
        f"기울기를 안 나르는 연산: {stuck}\n"
        "`Tensor(...)` 를 맨손으로 만들면 부모가 안 달려 그래프가 조용히 끊긴다.\n"
        "`t._make(값, (t,), 역방향)` 을 쓴다."
    )


def test_the_i1_series_is_convergence_not_approximation():
    """`i0` 의 도함수를 급수로 짰다. **근사면 이 저장소에 못 들어간다.**

    항이 전부 양수라 서로 지우지 않고, 앞 항에 곱해 이어 가므로 계승이 안 넘친다.
    골든은 세 점만 묻는다 — 넓은 구간을 재는 것은 여기 몫이다.
    """
    torch = __import__("torch")
    xs = np.linspace(-30, 30, 601)
    want = torch.special.i1(torch.tensor(xs, dtype=torch.float64)).numpy()
    got = borch._ops._i1(xs)
    rel = np.abs(want - got) / np.maximum(np.abs(want), 1e-300)
    assert rel.max() < 1e-12, f"최대 상대오차 {rel.max():.3e}"


def test_retain_grad_actually_retains():
    """**거절만 맞추면 반쪽이다.** 이 이름이 하려는 일은 파생 텐서의 `.grad` 를
    남기는 것이고, 그것을 안 하면 잎에서 멈추는 것까지만 맞는 껍데기가 된다.

    골든에 못 두는 이유는 borch.ts 가 파생 텐서의 `.grad` 를 안 내주기 때문이다 —
    셋을 함께 묻는 자리가 아니다.
    """
    x = borch.tensor([1.0, 2.0], requires_grad=True)
    plain = x * 2
    plain.sum().backward()
    assert plain.grad is None, "안 청했는데 남았다"

    y = borch.tensor([1.0, 2.0], requires_grad=True)
    kept = y * 3
    kept.retain_grad()
    kept.sum().backward()
    assert kept.grad is not None and kept.grad.tolist() == [1.0, 1.0]
    assert kept.retains_grad is True
    # **잎은 청해도 거짓으로 남는다** — 남기고 있는 것이 아니라 원래 쌓이는 것이다.
    assert y.retains_grad is False
