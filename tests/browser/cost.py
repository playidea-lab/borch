"""**What one step of the binding spends, and how much** — counted rather than timed.

    npm run build:ts
    uv run --with playwright python tests/browser/run.py --lib borch_webgpu --cost

Throws the same question as `borch-ts/test/cost.ts` at **the binding's path.** That is the
path a user actually walks, and it has one more leaky place than the TS side does —
**a Python object holds a JS handle.** If a binding piles one per step into a dictionary, the
GPU buffer is not released either, and the borch.ts-side check cannot see that.

## How it differs from the benchmark

`bench.py` measures wall-clock and so depends on the device. What is counted here is decided
by the code path, so the same numbers come out on a software adapter — it is not blocked.

## Why Python garbage is collected first

Exactly as `bench.py` writes down. Left uncollected, what the warm-up left behind is released
late, inside the measurement window, and that turns into a per-step ratio. A "leak of -24.8"
really did come out once, and it was a one-off rather than per-step.
"""

import gc as _gc

import numpy as np

# The dispatch and submit counts one step makes. **Measured and then written in.**
# Same model and same batch as borch.ts's `cost.ts`, so **the numbers have to match** — a
# divergence means the binding is making more kernel calls, and that is itself the answer.
EXPECT = {"dispatches": 53, "submits": 1}


def _model(L):
    """The same model as `cost.ts`'s `Small`. A different yardstick is not a comparison."""

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

    # Inputs and parameters are **outside the scope.** Built inside, they are released at the first step's end.
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
            return loss.item()      # read inside the scope, while that buffer still exists

    for _ in range(3):
        one()
    _gc.collect()

    # ── 1. is the count the same every step ───────────────────────────────
    per_step, per_submit = [], []
    for _ in range(5):
        d0, s0 = L.dispatches(), L.submits()
        one()
        per_step.append(L.dispatches() - d0)
        per_submit.append(L.submits() - s0)
    first = per_step[0]
    want("the dispatch count is the same every step", all(n == first for n in per_step),
         " ".join(str(n) for n in per_step))
    first_submit = per_submit[0]
    want("the submit count is the same every step", all(n == first_submit for n in per_submit),
         " ".join(str(n) for n in per_submit))

    # ── 2. does it match the frozen count — **and borch.ts's too** ────────
    want("dispatches per step match the frozen count", first == EXPECT["dispatches"],
         f"{first} (frozen: {EXPECT['dispatches']})")
    want("submits per step match the frozen count", first_submit == EXPECT["submits"],
         f"{first_submit} (frozen: {EXPECT['submits']})")

    # ── 3. does the scope leave nothing behind ────────────────────────────
    one()
    last = L.last_scope()
    want("the scope leaves no buffer behind", last["survived"] == 0,
         f"survived {last['survived']} · freed {last['freed']}")

    # ── 4. does it stay flat once Python garbage is collected ─────────────
    #
    # **This is the place that exists in the binding alone.** While a Python object holds the
    # handle the scope cannot release it even trying to, and looking at `survived` alone may
    # not show that difference. Calling `gc.collect()` on both sides and comparing is what asks
    # "is Python holding it".
    #
    # **The growth is looked at, not the absolute.** This probe runs on the same page after
    # 2,765 golden cases have run, so it starts out holding some forty thousand buffers — those
    # are the harness's leftovers, not the training loop's. Asked in absolute terms, that
    # residue reads as this loop's share.
    _gc.collect()
    early = L.memory()["tensors"]
    for _ in range(10):
        one()
    _gc.collect()
    late = L.memory()["tensors"]
    want("ten more steps do not grow the held buffers", late <= early,
         f"{early} → {late} (mostly the harness's leftovers)")

    # ── can the pool be asked about, and can it be emptied ────────────────
    #
    # `memory()` **deliberately excludes** what is in the pool — that number asks "is it
    # leaking", so that is right. Which means there has to be a separate place to ask "how much
    # is being held", and while there was none nobody could ask about the real footprint (the
    # benchmark was writing 269.7MB while 1,699.6MB sat in the pool).
    #
    # **Having it on the borch.ts side alone is useless.** Training in a browser happens here.
    held = L.pooled()
    want("the pool can be asked about", held["count"] > 0,
         f"{held['count']} buffers · {held['bytes'] // 1024}KB")
    freed = L.empty_cache()
    want("empty_cache empties the pool",
         freed["count"] == held["count"] and L.pooled()["count"] == 0,
         f"gave back {freed['count']} buffers · {freed['bytes'] // 1024}KB")
    # Training has to run after emptying too — emptying the pool must not break the device.
    one()
    want("a step runs after the pool is emptied", L.dispatches() > 0, f"{L.dispatches()}")

    bad = [c for c in checks if not c[1]]
    lines = [f"  {'✓' if ok else '✗'} {name}{f' — {note}' if note else ''}"
             for name, ok, note in checks]
    lines.append("")
    lines.append(f"**{len(bad)} diverged.**" if bad
                 else f"all {len(checks)} binding cost checks passed")
    return "\n".join(lines)
