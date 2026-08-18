"""결속의 한 스텝이 **무엇을 얼마나 쓰는가** — 시간이 아니라 세는 것으로.

    npm run build:ts
    uv run --with playwright python tests/browser/run.py --lib borch_webgpu --cost

`borch-ts/test/cost.ts` 와 같은 질문을 **결속 쪽 길**에 던진다. 사용자가 실제로
지나는 길이 이쪽이고, 여기에는 TS 쪽에 없는 새는 자리가 하나 더 있다 —
**파이썬 객체가 JS 손잡이를 쥔다.** 스텝마다 딕셔너리에 하나씩 쌓는 결속이 있으면
GPU 버퍼도 같이 안 놓이는데, borch.ts 쪽 검사는 그것을 못 본다.

## 벤치와 다른 점

`bench.py` 는 벽시계를 재고 그래서 장치에 달렸다. 여기서 세는 것은 코드 경로가
정하므로 소프트웨어 어댑터에서도 같은 수가 나온다 — 막지 않는다.

## 파이썬 쓰레기를 먼저 치우는 이유

`bench.py` 가 적어 둔 그대로다. 안 치우면 워밍업이 남긴 것이 측정 창 안에서 뒤늦게
풀리고, 그것이 스텝당 비율로 둔갑한다. 실제로 "누수 -24.8" 이 나온 적이 있고 그건
스텝당이 아니라 한 번짜리였다.
"""

import gc as _gc

import numpy as np

# 스텝 하나가 거는 dispatch 수와 제출 수. **재서 넣은 값이다.**
# borch.ts 쪽 `cost.ts` 와 같은 모델·같은 배치라 **같은 수여야 한다** — 갈리면
# 결속이 커널을 더 걸고 있다는 뜻이고, 그 자체가 답이다.
EXPECT = {"dispatches": 53, "submits": 1}


def _model(L):
    """`cost.ts` 의 `Small` 과 같은 모델. 잣대가 다르면 비교가 아니다."""

    class Small(L.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = L.nn.Conv2d(1, 4, 3, stride=1, padding=1, bias=False)
            self.bn = L.nn.BatchNorm2d(4)
            self.fc = L.nn.Linear(4 * 8 * 8, 3)

        def forward(self, x):
            h = L.relu(self.bn(self.conv(x)))
            return self.fc(h.reshape(x.shape[0], 4 * 8 * 8))

    return Small()


def report(L):
    checks = []

    def want(name, ok, note=""):
        checks.append((name, bool(ok), note))

    batch = 4
    pixels = np.array([(i % 13) / 13 - 0.5 for i in range(batch * 8 * 8)],
                      dtype=np.float32).reshape(batch, 1, 8, 8)
    labels = np.array([i % 3 for i in range(batch)], dtype=np.int64)

    # 입력과 파라미터는 **구역 밖**이다. 안에서 만들면 첫 스텝 끝에 놓인다.
    x = L.tensor(pixels)
    y = L.tensor(labels)
    model = _model(L)
    opt = L.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = L.nn.CrossEntropyLoss()

    def one():
        with L.scope():
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            return loss.item()      # 구역 안에서 읽어야 그 버퍼가 있다

    for _ in range(3):
        one()
    _gc.collect()

    # ── 1. 스텝마다 같은 수인가 ────────────────────────────────────────────
    per_step, per_submit = [], []
    for _ in range(5):
        d0, s0 = L.dispatches(), L.submits()
        one()
        per_step.append(L.dispatches() - d0)
        per_submit.append(L.submits() - s0)
    first = per_step[0]
    want("스텝마다 dispatch 수가 같다", all(n == first for n in per_step),
         " ".join(str(n) for n in per_step))
    first_submit = per_submit[0]
    want("스텝마다 제출 수가 같다", all(n == first_submit for n in per_submit),
         " ".join(str(n) for n in per_submit))

    # ── 2. 굳힌 수와 같은가 — **borch.ts 쪽과도 같아야 한다** ──────────────
    want("스텝당 dispatch 가 굳힌 수와 같다", first == EXPECT["dispatches"],
         f"{first} (굳힌 것 {EXPECT['dispatches']})")
    want("스텝당 제출이 굳힌 수와 같다", first_submit == EXPECT["submits"],
         f"{first_submit} (굳힌 것 {EXPECT['submits']})")

    # ── 3. 구역이 아무것도 안 남기는가 ────────────────────────────────────
    one()
    last = L.last_scope()
    want("구역이 버퍼를 안 남긴다", last["survived"] == 0,
         f"살아남은 것 {last['survived']} · 놓은 것 {last['freed']}")

    # ── 4. 파이썬 쓰레기를 치운 뒤에도 안 자라는가 ────────────────────────
    #
    # **여기가 결속에만 있는 자리다.** 파이썬 객체가 손잡이를 쥐고 있으면 구역이
    # 놓으려 해도 못 놓는데, `survived` 만 보면 그 차이가 안 드러날 수 있다.
    # `gc.collect()` 를 앞뒤로 부르고 비교해야 "파이썬이 쥐고 있는가" 를 묻는 것이다.
    #
    # **절대값이 아니라 늘어난 양을 본다.** 이 프로브는 골든 2765 건이 먼저 돈 뒤에
    # 같은 페이지에서 도므로 시작부터 버퍼를 4 만 개쯤 잡고 있다 — 그것은 하네스가
    # 남긴 것이지 학습 루프의 것이 아니다. 절대값으로 물으면 그 잔여물을 이 루프의
    # 몫으로 읽는다.
    _gc.collect()
    early = L.memory()["tensors"]
    for _ in range(10):
        one()
    _gc.collect()
    late = L.memory()["tensors"]
    want("스텝을 열 번 더 돌려도 잡은 버퍼가 안 는다", late <= early,
         f"{early} → {late} 개 (하네스가 남긴 것이 대부분이다)")

    # ── 통을 물을 수 있는가, 비울 수 있는가 ──────────────────────────────────
    #
    # `memory()` 는 통에 든 것을 **일부러 뺀다** — 저것은 "새는가" 를 묻는 수라서
    # 그게 맞다. 그래서 "얼마나 쥐고 있는가" 를 물을 자리가 따로 있어야 하고,
    # 그것이 없는 동안 아무도 진짜 발자국을 못 물었다(벤치가 269.7MB 라고 적는
    # 동안 통에 1,699.6MB 가 있었다).
    #
    # **borch.ts 쪽에만 있으면 소용이 없다.** 브라우저에서 학습하는 쪽이 여기다.
    held = L.pooled()
    want("통을 물을 수 있다", held["count"] > 0,
         f"{held['count']} 개 · {held['bytes'] // 1024}KB")
    freed = L.empty_cache()
    want("empty_cache 가 통을 비운다",
         freed["count"] == held["count"] and L.pooled()["count"] == 0,
         f"{freed['count']} 개 · {freed['bytes'] // 1024}KB 돌려줬다")
    # 비운 뒤에도 학습이 돌아야 한다 — 통을 비우는 것이 장치를 망가뜨리면 안 된다.
    one()
    want("통을 비운 뒤에도 스텝이 돈다", L.dispatches() > 0, f"{L.dispatches()}")

    bad = [c for c in checks if not c[1]]
    lines = [f"  {'✓' if ok else '✗'} {name}{f' — {note}' if note else ''}"
             for name, ok, note in checks]
    lines.append("")
    lines.append(f"**{len(bad)}건이 갈렸다.**" if bad
                 else f"결속 비용 검사 {len(checks)}건 전부 통과")
    return "\n".join(lines)
