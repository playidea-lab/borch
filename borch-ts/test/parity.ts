/**
 * Does torch, transcribed, run — parameter registration, parameter groups, random
 * factories.
 *
 * **The golden cannot catch these three.** It plants the weights from outside for every
 * case, so it never asks how parameters are gathered and never looks at an initial value.
 * What is caught here is not a value but the **wiring** — and the first of the three
 * raises nothing when it is wrong, it only stops the learning, quietly, so without a
 * runner it is never seen at all.
 */

import {
  device, type DType, init, keepAlive, linalg, manualSeed, nn, noGrad, optim, scope,
  slice, Tensor, vision,
} from "../src/index.js";

interface Check { name: string; ok: boolean; note: string }

/**
 * `checks` is the authority in this report. `text` is the shadow a person reads.
 *
 * **The runner used to judge by scanning a sentence.** That way of judging changes its
 * answer quietly when the wording changes, and in `readme.py` it did — with one of the two
 * examples failing, the word it looked for was still sitting on another line, so it
 * returned 0. Hand the state over as it is and the runner can count, and can say for
 * itself which thing failed.
 */
export interface Report { text: string; checks: Check[] }
const checks: Check[] = [];

function want(name: string, ok: boolean, note = ""): void {
  checks.push({ name, ok, note });
}

function near(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol;
}

/** Where it has to throw. **Not throwing is the failure** — pass quietly and the value
 * is wrong. */
function wantThrow(name: string, fragment: string, body: () => unknown): void {
  try {
    body();
    want(name, false, "it did not throw");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    want(name, message.includes(fragment),
      message.includes(fragment) ? "" : `different wording: ${message}`);
  }
}

function same(a: Float32Array, b: Float32Array): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

/** The shape somebody building a layer from outside would write. Registration is never
 * spelled out by hand. */
class Net extends nn.Module {
  fc1 = new nn.Linear(4, 8);
  fc2 = new nn.Linear(8, 2);

  override forward(x: Tensor): Tensor {
    return this.fc2.call(this.fc1.call(x).relu());
  }
}

/**
 * **A child put in a plain object is not registered.** torch is the same (which is why
 * `nn.ModuleDict` exists) and by itself that is right. What has to be caught is what comes
 * next — overriding `children()` alone, so that **the two disagree.**
 *
 * Parameters are gathered by `namedChildren()`, so a child written into `children()` only
 * **stops learning, with no exception at all.** The bench's ResNet-18 was in exactly that
 * state and six shortcut layers never learned once — and the loss went down. The rest
 * compensate.
 */
class SplitBrain extends nn.Module {
  readonly a = new nn.Linear(2, 2);
  // A plain object — it does not catch on `namedChildren()`'s `instanceof Module`.
  readonly side = { fc: new nn.Linear(2, 2) };

  override children(): nn.Module[] {
    return [this.a, this.side.fc];      // ← written here and nowhere else
  }

  override forward(x: Tensor): Tensor {
    return this.a.call(x).add(this.side.fc.call(x));
  }
}

/** Not every tensor field is a parameter. */
class WithConstant extends nn.Module {
  weight = Tensor.from([2, 3], [2], { requiresGrad: true });
  mask = Tensor.from([1, 0], [2]);          // constant — no optimizer may step on it

  override forward(x: Tensor): Tensor {
    return x.mul(this.weight).mul(this.mask);
  }
}

export async function report(): Promise<Report> {
  await init();

  // ── 1. Parameters registering themselves ─────────────────────────────
  const net = new Net();
  want("child layers are picked up from the fields", net.children().length === 2);
  want("every parameter is gathered", net.parameters().length === 4,
    `${net.parameters().length} of them`);

  const names = Object.keys(net.namedParameters()).sort();
  want("the names are the field names",
    names.join(",") === "fc1.bias,fc1.weight,fc2.bias,fc2.weight", names.join(","));

  const wc = new WithConstant();
  want("requiresGrad is what marks a parameter",
    Object.keys(wc.ownParameters()).join(",") === "weight",
    Object.keys(wc.ownParameters()).join(","));

  // **When `children()` and `namedChildren()` disagree, that child does not learn.**
  //
  // Only `namedChildren()` gathers parameters. Override `children()` alone and the layer
  // is visible while its parameters are not picked up — **the learning, and nothing else,
  // stops right there, with no exception and no warning.** The loss goes down, because the
  // rest compensate.
  //
  // The bench's ResNet-18 was in exactly that state (six shortcut layers). What caught it
  // was not a value check but the **dead-tensor guard**: a leaf the optimizer cannot see
  // does not get `zeroGrad()` either, so last step's gradient stays, and that was a buffer
  // already handed back to the pool.
  const split = new SplitBrain();
  let splitNote = "it did not stop";
  let splitStopped = false;
  try {
    split.parameters();
  } catch (err) {
    splitStopped = true;
    splitNote = (err as Error).message.split("\n")[0] ?? "";
  }
  want("children() and namedChildren() disagreeing stops it", splitStopped, splitNote);

  // ── `nn.Parameter` — **the two places we part from torch, pinned by value** ──
  //
  // Write a divergence in a comment only and the next person changes it without reading
  // the comment. Written here, changing it goes red on the spot, and that is the moment
  // somebody decides again whether the divergence was **meant.**
  const src = Tensor.from([1, 2, 3], [3]);
  const par = new nn.Parameter(src);
  want("Parameter raises the learn-me flag", par.requiresGrad);
  // torch leaves the original alone — so do we.
  want("Parameter leaves the original's flag alone", !src.requiresGrad);
  // **The storage is not shared.** torch shares it; we have no views.
  // The Python binding made the same choice, so the two GPU implementations do not part.
  // The second argument turns the flag off — without that, a leaf cannot be written in
  // place and the question itself will not stand. It asks, as a bonus, whether that
  // argument arrives at all.
  await (async () => {
    const p2 = new nn.Parameter(src, false);
    want("Parameter(t, false) raises no flag", !p2.requiresGrad);
    p2.copyFrom(Tensor.from([9, 9, 9], [3]));
    const after = await src.toArray();
    want("Parameter does not share storage — here we part from torch",
      after[0] === 1, `first slot of the original ${after[0]}`);
  })();
  // **A plain `requiresGrad` tensor counts as a parameter too.** torch counts only what
  // is wrapped in `Parameter` and does not count this (measured) — move the rule that way
  // and today's code, which raises the flag with `claim()`, quietly loses parameters.
  class Bare extends nn.Module {
    marked = new nn.Parameter(Tensor.from([1, 1], [2]));
    flagged = Tensor.from([1, 1], [2], { requiresGrad: true });

    override forward(x: Tensor): Tensor {
      return x;
    }
  }
  want("a tensor with only the flag counts too — here we part from torch",
    Object.keys(new Bare().ownParameters()).sort().join(",") === "flagged,marked",
    Object.keys(new Bare().ownParameters()).sort().join(","));

  // A container has to keep its slot numbers — the golden plants weights by those names.
  const seq = new nn.Sequential(new nn.Linear(2, 2), new nn.ReLU());
  want("Sequential keeps its slot numbers",
    Object.keys(seq.namedParameters()).sort().join(",") === "0.bias,0.weight",
    Object.keys(seq.namedParameters()).join(","));
  const list = new nn.ModuleList([new nn.Linear(2, 2)]);
  want("ModuleList keeps them too",
    Object.keys(list.namedParameters()).sort().join(",") === "0.bias,0.weight",
    Object.keys(list.namedParameters()).join(","));

  // state_dict round trip. A name that parts is caught here.
  const target = new Net();
  target.loadStateDict(net.stateDict());
  const a0 = await net.parameters()[0]!.toArray();
  const b0 = await target.parameters()[0]!.toArray();
  want("a state_dict round trip carries the values", same(a0, b0));

  // **This is the real question.** Where registration leaks the loss does not go down —
  // and nothing is raised.
  const trained = new Net();
  for (const p of trained.parameters()) keepAlive(p);
  const opt = new optim.SGD(trained.parameters(), 0.1);
  const crit = new nn.CrossEntropyLoss();
  const x = keepAlive(Tensor.from(
    Array.from({ length: 8 * 4 }, (_, i) => (i % 7) / 7 - 0.5), [8, 4]));
  const y = keepAlive(Tensor.from([0, 1, 0, 1, 0, 1, 0, 1], [8], { dtype: "int64" }));
  const seen: number[] = [];
  for (let i = 0; i < 5; i++) {
    await scope(async () => {
      opt.zeroGrad();
      const loss = crit.call(trained.call(x), y);
      loss.backward();
      opt.step();
      seen.push(await loss.item());
    });
  }
  want("registration alone makes it learn",
    (seen[4] ?? NaN) < (seen[0] ?? NaN),
    seen.map((v) => v.toFixed(4)).join(" → "));

  // ── 2. Parameter groups ──────────────────────────────────────────────
  const mk = () => keepAlive(Tensor.from([1], [1], { requiresGrad: true }));
  const slow = mk();
  const fast = mk();
  slow.grad = Tensor.from([1], [1]);
  fast.grad = Tensor.from([1], [1]);
  const two = new optim.SGD(
    [{ params: [slow], lr: 0.1 }, { params: [fast], lr: 0.5 }], 0.01);
  want("the groups come out as two", two.paramGroups.length === 2);
  two.step();
  want("each group moves at its own learning rate",
    near(await slow.item(), 0.9, 1e-6) && near(await fast.item(), 0.5, 1e-6),
    `${(await slow.item()).toFixed(4)} / ${(await fast.item()).toFixed(4)}`);

  // A value given to the group and the same value given to the constructor have to come
  // out the same. The formula is not asked here; only the wiring is.
  const viaCtor = mk();
  const viaGroup = mk();
  viaCtor.grad = Tensor.from([1], [1]);
  viaGroup.grad = Tensor.from([1], [1]);
  // `weightDecay` is the fifth argument now, behind torch's `dampening`.
  new optim.SGD([viaCtor], 0.1, 0, 0, 0.5).step();
  new optim.SGD([{ params: [viaGroup], weightDecay: 0.5 }], 0.1, 0, 0, 0).step();
  want("per-group weightDecay bites the same as the constructor's",
    near(await viaCtor.item(), await viaGroup.item(), 1e-6),
    `${(await viaCtor.item()).toFixed(6)} / ${(await viaGroup.item()).toFixed(6)}`);

  // A group added later has to grow the state bank with it — otherwise the next step
  // blows up.
  const first = mk();
  const later = mk();
  const adam = new optim.Adam([first], 0.1);
  adam.addParamGroup({ params: [later], lr: 0.1 });
  first.grad = Tensor.from([1], [1]);
  later.grad = Tensor.from([1], [1]);
  let added = true;
  try {
    adam.step();
  } catch (err) {
    added = false;
    want("step after addParamGroup", false, String(err));
  }
  if (added) {
    want("addParamGroup grows the state bank too",
      (await later.item()) !== 1 && (await first.item()) !== 1,
      `${(await first.item()).toFixed(4)} / ${(await later.item()).toFixed(4)}`);
  }

  // A scheduler has to drive every group, and to keep the ratio between them.
  const s1 = mk();
  const s2 = mk();
  const sched = new optim.SGD(
    [{ params: [s1], lr: 0.1 }, { params: [s2], lr: 1.0 }], 0.1);
  const step = new optim.StepLR(sched, 1, 0.1).start();
  step.step();
  want("the scheduler drives every group and keeps the ratio",
    near(sched.paramGroups[0]!.lr, 0.01, 1e-9)
      && near(sched.paramGroups[1]!.lr, 0.1, 1e-9),
    `${sched.paramGroups[0]!.lr} / ${sched.paramGroups[1]!.lr}`);

  // ── What is mended in place has to own its buffer ─────────────────────
  // The `addParamGroup` check above caught this. `Tensor.zeros([1])` returns a **global
  // constant** cached by value, and optimizers and running statistics write into it.
  // **Every optimizer is put on the hook.** The defect is not one class's slip but the
  // whole habit of making state with `Tensor.zeros` and `Tensor.full`, so it walks back in
  // with each new optimizer — `Rprop` arrived that way.
  const makers: [string, (p: Tensor) => optim.Optimizer][] = [
    ["SGD(momentum)", (p) => new optim.SGD([p], 0.1, 0.9)],
    ["Adam", (p) => new optim.Adam([p], 0.1)],
    ["RMSprop", (p) => new optim.RMSprop([p], 0.1)],
    ["Adagrad", (p) => new optim.Adagrad([p], 0.1)],
    ["Adadelta", (p) => new optim.Adadelta([p], 0.1)],
    ["Adamax", (p) => new optim.Adamax([p], 0.1)],
    ["NAdam", (p) => new optim.NAdam([p], 0.1)],
    ["RAdam", (p) => new optim.RAdam([p], 0.1)],
    ["ASGD", (p) => new optim.ASGD([p], 0.1)],
    ["Rprop", (p) => new optim.Rprop([p], 0.1)],
    ["Adafactor", (p) => new optim.Adafactor([p], 0.1)],
  ];
  const before = device().faults.count;
  const moved: string[] = [];
  const stuck: string[] = [];
  for (const [label, make] of makers) {
    const p = mk();
    const o = make(p);
    // **Step twice.** The state has to survive, and a first step that ruins it shows up
    // on the second.
    for (let i = 0; i < 2; i++) {
      p.grad = Tensor.from([1], [1]);
      o.step();
    }
    ((await p.item()) !== 1 ? moved : stuck).push(label);
  }
  want("every optimizer runs on a one-element parameter",
    stuck.length === 0,
    stuck.length ? `did not move: ${stuck.join(", ")}` : `${moved.length} of them`);
  want("no validation fault along the way", device().faults.count === before,
    device().faults.first);

  // Where state rode the cache, this changes. `0.1` is the learning rate used above.
  //
  // **A tolerance is needed here.** 0.1 does not land exactly in float32, so reading it
  // back gives 0.10000000149… Asked at 1e-9 at first, this caught on that rounding — not
  // contamination but representation, and a place where a check raised a false alarm
  // against itself.
  const canary = [
    await Tensor.full([1], 0).item(),
    await Tensor.full([1], 1).item(),
    await Tensor.full([1], 0.1).item(),
  ];
  want("the global 0, 1 and 0.1 constants are unsoiled",
    canary[0] === 0 && canary[1] === 1 && near(canary[2] ?? NaN, 0.1, 1e-6),
    canary.join(" / "));

  // An optimizer carrying state that is not parameter-shaped has to grow its groups too.
  const af1 = mk();
  const af2 = Tensor.from([1, 2, 3, 4], [2, 2], { requiresGrad: true });
  keepAlive(af2);
  const af = new optim.Adafactor([af1], 0.1);
  af.addParamGroup({ params: [af2], lr: 0.1 });
  af1.grad = Tensor.from([1], [1]);
  af2.grad = Tensor.from([1, 1, 1, 1], [2, 2]);
  af.step();
  want("Adafactor still runs after addParamGroup",
    (await af1.item()) !== 1 && (await af2.toArray())[0] !== 1);

  const p1 = new nn.PReLU();
  const p2 = new nn.PReLU();
  noGrad(() => { p1.weight.fill_(9); });
  want("two PReLU default weights are independent of each other",
    (await p2.weight.item()) === 0.25, `${await p2.weight.item()}`);
  want("the global 0.25 constant is not soiled",
    (await Tensor.full([1], 0.25).item()) === 0.25);

  const bn = new nn.BatchNormND(1);
  noGrad(() => { bn.runningVar.fill_(5); });
  want("BatchNorm(1)'s running statistics do not overlap the global 1",
    (await Tensor.ones([1]).item()) === 1, `${await Tensor.ones([1]).item()}`);

  // ── Backward given its seed by hand ──────────────────────────────────
  // Whether the values are right is the golden's business, against real torch
  // (`grad::vjp::*`, where the binding's runner goes through borch.ts). What is asked here
  // is the **TS surface** — argument order and refusal wording.
  const leaf = keepAlive(Tensor.from([1, 2, 3], [3], { requiresGrad: true }));
  const out = leaf.mul(leaf);
  out.backward(Tensor.from([1, 10, 100], [3]));
  // d(x²)/dx · v = 2x·v = [2, 40, 600]
  want("backward with a seed gives the Jacobian-vector product",
    same(await leaf.grad!.toArray(), Float32Array.from([2, 40, 600])),
    `${Array.from(await leaf.grad!.toArray()).join(",")}`);

  wantThrow("a non-scalar with no seed is refused",
    "grad can be implicitly created only for scalar outputs",
    () => Tensor.from([1, 2], [2], { requiresGrad: true }).backward());
  wantThrow("a seed of the wrong shape is refused", "Mismatch in shape",
    () => Tensor.from([1, 2], [2], { requiresGrad: true })
      .backward(Tensor.from([1, 2, 3], [3])));

  // The second slot is `retainGraph` — torch's argument order.
  const twice = keepAlive(Tensor.from([2], [1], { requiresGrad: true }));
  const held = twice.mul(twice);
  held.backward(Tensor.from([1], [1]), true);
  held.backward(Tensor.from([1], [1]), true);
  want("the second slot is retainGraph — flow twice and it doubles",
    (await twice.grad!.item()) === 8, `${await twice.grad!.item()}`);

  // ── The square-bracket seat ──────────────────────────────────────────
  //
  // **`at()` makes no values.** It passes everything to `select`, `narrow` and
  // `indexSelect`, and the golden already holds those three against real torch. So there
  // is one thing to ask here — **does `at()` answer the same as what it delegated to** —
  // and then the values ride on the golden.
  //
  // Write the values out by hand and compare, and when that hand is wrong the check is
  // wrong with it.
  const cube = keepAlive(Tensor.from(
    Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]));

  const agrees = async (
    name: string, got: Tensor, expected: Tensor,
  ): Promise<void> => {
    const shapeOk = got.shape.join(",") === expected.shape.join(",");
    want(name, shapeOk && same(await got.toArray(), await expected.toArray()),
      shapeOk ? "" : `shape [${got.shape}] vs [${expected.shape}]`);
  };

  await agrees("at(0) is select", cube.at(0), cube.select(0, 0));
  await agrees("at(-1) counts from the back", cube.at(-1), cube.select(0, 1));
  await agrees("at([null, 1]) is a select on the second axis",
    cube.at([null, 1]), cube.select(1, 1));
  await agrees("at(slice(1, 3)) is narrow",
    cube.at(slice(1, 3)), cube.narrow(0, 1, 1));
  await agrees("an open slice runs to the end",
    cube.at([null, slice(1)]), cube.narrow(1, 1, 2));
  await agrees("a slice with a stride goes through indexSelect",
    cube.at([null, null, slice(null, null, 2)]),
    cube.indexSelect(2, Tensor.from([0, 2], [2], { dtype: "int64" })));
  await agrees("a list of indices takes two brackets",
    cube.at([[1, 0]]),
    cube.indexSelect(0, Tensor.from([1, 0], [2], { dtype: "int64" })));
  await agrees("a tensor of indices is taken too",
    cube.at(Tensor.from([1], [1], { dtype: "int64" })),
    cube.indexSelect(0, Tensor.from([1], [1], { dtype: "int64" })));

  // **The axis numbers shift.** An integer removes an axis, so the second index names
  // the original axis 1 — which, in what is left, is axis 0. Miss this place and it
  // quietly cuts a different axis.
  await agrees("an index after an integer names the original axis",
    cube.at([0, slice(1, 3)]), cube.select(0, 0).narrow(0, 1, 2));
  await agrees("two integers in a row shift correctly too",
    cube.at([1, 2]), cube.select(0, 1).select(0, 2));
  want("give fewer and the remaining axes come whole",
    cube.at(0).shape.join(",") === "3,4", cube.at(0).shape.join(","));

  // Empty is an answer too — Python gives `x[5:99]` as empty.
  want("a slice past the end becomes empty",
    cube.at(slice(5, 99)).shape.join(",") === "0,3,4",
    cube.at(slice(5, 99)).shape.join(","));

  wantThrow("an integer past the end is refused", "out of bounds", () => cube.at(9));
  wantThrow("more indices than axes is refused", "too many indices",
    () => cube.at([0, 0, 0, 0]));
  wantThrow("a negative stride points at flip", "flip()",
    () => slice(0, 3, -1));

  // ── The dtype a reduction returns ────────────────────────────────────
  //
  // The table was frozen by asking torch, and `tests/test_reduce_dtype.py` holds the core
  // side. This is **the same table on the borch.ts side** — the three have to answer alike.
  //
  // **Both int64 and bool are asked.** Ask int64 alone and "it keeps the dtype" and "it
  // promotes bool" look the same, and an implementation that is half right passes.
  const ints = Tensor.from([3, 1, 4], [3], { dtype: "int64" });
  const flags = Tensor.from([1, 0, 1], [3], { dtype: "bool" });
  const table: [string, DType, DType][] = [
    // Accumulating — it makes values. 3 does not fit in a true/false slot, so bool is
    // promoted.
    ["sum", ints.sum().dtype, flags.sum().dtype],
    ["prod", ints.prod().dtype, flags.prod().dtype],
    ["cumsum", ints.cumsum(0).dtype, flags.cumsum(0).dtype],
    ["cumprod", ints.cumprod(0).dtype, flags.cumprod(0).dtype],
    // Choosing — it hands over a value that was already there. The dtype goes through.
    ["amax", ints.amax().dtype, flags.amax().dtype],
    ["amin", ints.amin().dtype, flags.amin().dtype],
    // Fixed
    ["any", ints.any().dtype, flags.any().dtype],
    ["all", ints.all().dtype, flags.all().dtype],
    ["countNonzero", ints.countNonzero().dtype, flags.countNonzero().dtype],
    ["argmax", ints.argmax().dtype, flags.argmax().dtype],
    ["logsumexp", ints.logsumexp(0).dtype, flags.logsumexp(0).dtype],
  ];
  const expected: Record<string, [DType, DType]> = {
    sum: ["int64", "int64"], prod: ["int64", "int64"],
    cumsum: ["int64", "int64"], cumprod: ["int64", "int64"],
    amax: ["int64", "bool"], amin: ["int64", "bool"],
    any: ["bool", "bool"], all: ["bool", "bool"],
    countNonzero: ["int64", "int64"], argmax: ["int64", "int64"],
    logsumexp: ["float32", "float32"],
  };
  const wrong = table.filter(([name, i, b]) => {
    const [wi, wb] = expected[name] as [DType, DType];
    return i !== wi || b !== wb;
  });
  want("reduction dtypes match torch's table", wrong.length === 0,
    wrong.map(([n, i, b]) => `${n}: ${i}/${b}`).join(", ") || "11 of them");

  // Whether accumulating and choosing **part** is asked separately. Should the table
  // above be wrong as a whole in one direction, this line survives it and keeps the rule
  // that the two are different things.
  want("on bool, accumulating and choosing part",
    flags.sum().dtype === "int64" && flags.amax().dtype === "bool",
    `${flags.sum().dtype} / ${flags.amax().dtype}`);

  // The four that take floats only. They have to stop where torch stops.
  for (const [name, call] of [
    ["mean", () => ints.mean()], ["variance", () => ints.variance()],
    ["std", () => ints.std()], ["norm", () => ints.norm()],
  ] as [string, () => Tensor][]) {
    wantThrow(`${name} refuses an integer`, "torch:", call);
  }

  // ── nn.functional ─────────────────────────────────────────────────────
  //
  // **It makes no values.** Everything is passed to a `Tensor` method, so the golden
  // already holds those values. Two things are asked here — **does it answer the same as
  // what it delegated to**, and **have the places that must not be joined by name been
  // left unjoined.**
  const F = nn.functional;
  const fx = keepAlive(Tensor.from([1, -2, 3, -4], [2, 2]));

  want("nn.functional opens", typeof F === "object" && F !== null);
  same(await F.relu(fx).toArray(), await fx.relu().toArray())
    ? want("F.relu is the method", true)
    : want("F.relu is the method", false);
  want("F.leakyRelu is the method",
    same(await F.leakyRelu(fx, 0.2).toArray(), await fx.leakyRelu(0.2).toArray()));
  want("F.softmax is the method",
    same(await F.softmax(fx, 1).toArray(), await fx.softmax(1).toArray()));

  // **Where the name is the same and the operation is not.** Joined automatically, a
  // different thing is picked up, quietly.
  want("F.batchNorm is the layer's free function — not Tensor.batchNorm",
    F.batchNorm.length >= 5, `${F.batchNorm.length} arguments`);
  want("F.unfold is im2col — not Tensor.unfold",
    same(await F.unfold(fx.reshape([1, 1, 2, 2]), 2).toArray(),
      await fx.reshape([1, 1, 2, 2]).unfoldIm2col(2).toArray()));
  // torch orders huberLoss (reduction, delta), so the positional arguments swap.
  want("F.huberLoss uses torch's argument order",
    same(await F.huberLoss(fx, fx.zerosLike(), "mean", 2).toArray(),
      await fx.huberLoss(fx.zerosLike(), 2, "mean").toArray()));

  // **Who the receiver is belongs to the name too.** In torch it is the right-hand side
  // that receives `Tensor.lu_solve` — let the factors receive it instead and the name and
  // the argument count both still fit, so nothing catches there and only the value is
  // wrong. The one the factors receive is separate, as `luSolveFactored`.
  const lu = await Tensor.from([4, 3, 6, 3], [2, 2]).luFactor();
  const rhs = Tensor.from([1, 2], [2, 1]);
  const viaMethod = await rhs.luSolve(lu.LU, lu.pivots);
  const viaFactored = await lu.LU.luSolveFactored(lu.pivots, rhs);
  want("lu_solve is received by b",
    same(await viaMethod.toArray(), await viaFactored.toArray()),
    `${Array.from(await viaMethod.toArray()).join(",")}`);

  // The ones that must not be joined have to be **absent**. Present, they are quietly a
  // different operation.
  for (const missing of ["layerNorm", "rmsNorm", "pad", "upsample"]) {
    want(`F.${missing} is not offered — it is a different operation`,
      (F as Record<string, unknown>)[missing] === undefined);
  }

  // ── 3. Random factories ──────────────────────────────────────────────
  const N = 4096;
  const g = await Tensor.randn([N]).toArray();
  const mean = g.reduce((a, b) => a + b, 0) / N;
  const sd = Math.sqrt(g.reduce((a, b) => a + (b - mean) ** 2, 0) / N);
  want("randn is close to the standard normal", near(mean, 0, 0.08) && near(sd, 1, 0.08),
    `mean ${mean.toFixed(4)}, sd ${sd.toFixed(4)}`);

  const u = await Tensor.rand([N]).toArray();
  want("rand lands inside [0, 1)", u.every((v) => v >= 0 && v < 1));

  const ri = Tensor.randint(3, 7, [N]);
  const riv = await ri.toArray();
  want("randint is an integer inside [low, high)",
    ri.dtype === "int64"
      && riv.every((v) => Number.isInteger(v) && v >= 3 && v < 7)
      && riv.some((v) => v === 6) && !riv.some((v) => v === 7));

  const perm = Array.from(await Tensor.randperm(64).toArray()).sort((a, b) => a - b);
  want("randperm is a permutation", perm.every((v, i) => v === i));

  // **The golden asks these four at their endpoints only.** `p=0`, `p=1`, `std=0` and
  // `λ=0` are deterministic and can be frozen; the middle cannot, so on the golden alone a
  // `normal` that never multiplies by `std` and returns the mean is green — at `std=0` the
  // answers agree. It is the place the seed cases taught above: **holding the endpoints is
  // holding half.**
  const bern = await Tensor.zeros([N]).add(Tensor.full([], 0.25)).bernoulli()
    .toArray();
  const hits = bern.reduce((a, b) => a + b, 0) / N;
  want("bernoulli actually looks at the probability",
    bern.every((v) => v === 0 || v === 1) && near(hits, 0.25, 0.03),
    `share that came out 1: ${hits.toFixed(4)}`);

  const nm = await Tensor.normal(
    Tensor.zeros([N]).add(Tensor.full([], 5)), 2).toArray();
  const nmMean = nm.reduce((a, b) => a + b, 0) / N;
  const nmSd = Math.sqrt(nm.reduce((a, b) => a + (b - nmMean) ** 2, 0) / N);
  want("normal uses both the mean and the standard deviation",
    near(nmMean, 5, 0.15) && near(nmSd, 2, 0.15),
    `mean ${nmMean.toFixed(4)}, sd ${nmSd.toFixed(4)}`);

  // Poisson has **the same mean and variance** — get only one of them right and it is a
  // different distribution.
  const po = await Tensor.zeros([N]).add(Tensor.full([], 4)).poisson()
    .then((t) => t.toArray());
  const poMean = po.reduce((a, b) => a + b, 0) / N;
  const poVar = po.reduce((a, b) => a + (b - poMean) ** 2, 0) / N;
  want("poisson gives mean λ and variance λ",
    po.every((v) => Number.isInteger(v) && v >= 0)
      && near(poMean, 4, 0.25) && near(poVar, 4, 0.5),
    `mean ${poMean.toFixed(3)}, variance ${poVar.toFixed(3)}`);

  const bi = await Tensor.zeros([N]).add(Tensor.full([], 10))
    .binomial(Tensor.zeros([N]).add(Tensor.full([], 0.5)))
    .then((t) => t.toArray());
  const biMean = bi.reduce((a, b) => a + b, 0) / N;
  want("binomial gives n·p",
    bi.every((v) => Number.isInteger(v) && v >= 0 && v <= 10)
      && near(biMean, 5, 0.2),
    `mean ${biMean.toFixed(3)}`);

  // **The golden cannot see these five in-place fills.** The values are random and cannot
  // be frozen, and the binding makes these names on its own numpy stem, so the cases never
  // even reach the borch.ts side. Measuring one landmark of each distribution here is all
  // that can be done for the five.
  const drawn = async (fill: (t: Tensor) => Tensor): Promise<Float32Array> =>
    fill(Tensor.zeros([N])).toArray();
  const avg = (v: Float32Array): number => v.reduce((a, b) => a + b, 0) / N;

  const ex = await drawn((t) => t.exponential_(2));
  want("exponential_ has mean 1/lambd",
    ex.every((v) => v >= 0) && near(avg(ex), 0.5, 0.03),
    `mean ${avg(ex).toFixed(4)}`);

  // Cauchy **has no mean** — measured by the sample mean it jumps from run to run. It is
  // measured by the median.
  const ca = Array.from(await drawn((t) => t.cauchy_(3, 1))).sort((a, b) => a - b);
  const mid = ca[N >> 1] ?? 0;
  want("cauchy_'s middle value is the median", near(mid, 3, 0.1),
    `median ${mid.toFixed(4)}`);

  // A log-normal is normal **once the log is taken** — that is what the name defines.
  const ln = await drawn((t) => t.logNormal_(0, 1));
  const logged = Array.from(ln).map((v) => Math.log(v));
  const lnMean = logged.reduce((a, b) => a + b, 0) / N;
  want("log_normal_ is normal once the log is taken",
    ln.every((v) => v > 0) && near(lnMean, 0, 0.05),
    `mean of the log ${lnMean.toFixed(4)}`);

  const ge = await drawn((t) => t.geometric_(0.25));
  want("geometric_ has mean 1/p",
    ge.every((v) => Number.isInteger(v) && v >= 1) && near(avg(ge), 4, 0.25),
    `mean ${avg(ge).toFixed(3)}`);

  const ra = await drawn((t) => t.random_(5, 9));
  want("random_ is an integer in [from, to)",
    ra.every((v) => Number.isInteger(v) && v >= 5 && v < 9)
      && new Set(ra).size === 4);

  want("randnLike borrows the shape",
    Tensor.zeros([2, 3]).randnLike().shape.join(",") === "2,3");

  // One seed has to reset the tensors and the layers together.
  manualSeed(7);
  const r1 = await Tensor.randn([8]).toArray();
  manualSeed(7);
  const r2 = await Tensor.randn([8]).toArray();
  want("the same seed gives the same randn", same(r1, r2));

  manualSeed(11);
  const w1 = await new nn.Linear(3, 2).parameters()[0]!.toArray();
  manualSeed(11);
  const w2 = await new nn.Linear(3, 2).parameters()[0]!.toArray();
  want("the same seed initialises a layer the same", same(w1, w2));

  // xorshift with a zero state gives zero forever. Seed 0 must not kill the randomness.
  manualSeed(0);
  const z = await Tensor.rand([4]).toArray();
  want("manualSeed(0) does not kill the randomness", new Set(z).size > 1);

  // **Different seeds have to give different results.** Holding only "same seed, same
  // result" is holding half — while the dropout counter was being reset to 1 every time,
  // five seeds gave the same mask five times over, and then the variance of an experiment
  // comes from the weight initialisation alone.
  const gpuDraw = async (seed: number): Promise<Float32Array> => {
    manualSeed(seed);
    return Tensor.uniform([16]).toArray();      // GPU stem — it uses the dropout counter
  };
  want("different seeds move the GPU stem too",
    !same(await gpuDraw(1), await gpuDraw(2)));
  want("the same seed gives the same GPU stem",
    same(await gpuDraw(7), await gpuDraw(7)));

  // ── The names the binding had been filling in ────────────────────────
  //
  // **The golden cannot see these six, structurally.** Every case goes through
  // `borch_webgpu`, and that side **builds the layers itself** on top of the tensor
  // methods, so the Python side was fine with no class in borch.ts at all. The name was
  // missing only for somebody writing `new nn.MSELoss()` in TypeScript.
  //
  // What is asked here is not a value but **whether the name is there and whether the
  // argument arrives.**
  const lx = () => Tensor.from([0.5, -1, 2, 1.5], [2, 2]);
  const ly = () => Tensor.from([1, 0, -1, 0.5], [2, 2]);
  const label = () => Tensor.from([1, 0], [2], { dtype: "int64" as DType });
  const lossLayers: [string, (r: "none" | "sum") => Tensor][] = [
    ["MSELoss", (r) => new nn.MSELoss(r).call(lx(), ly())],
    ["L1Loss", (r) => new nn.L1Loss(r).call(lx(), ly())],
    ["SmoothL1Loss", (r) => new nn.SmoothL1Loss(r).call(lx(), ly())],
    // These three take torch's argument list now — `weight` first, and for the two
    // that have it `ignoreIndex` before the reduction. Written positionally they
    // would set a class weight to `"sum"`, which is why `tsc` names each one.
    ["BCEWithLogitsLoss", (r) => new nn.BCEWithLogitsLoss(undefined, r).call(
      lx(), Tensor.from([1, 0, 1, 0], [2, 2]))],
    ["NLLLoss", (r) => new nn.NLLLoss(undefined, -100, r).call(
      Tensor.from([-1.6, -0.7, -0.5, -1.2], [2, 2]), label())],
    ["CrossEntropyLoss",
      (r) => new nn.CrossEntropyLoss(undefined, -100, r).call(lx(), label())],
  ];
  for (const [name, call] of lossLayers) {
    // `none` is before the fold, so it has many elements, and `sum` is a scalar. **The
    // two shapes differing** is what says the argument actually arrived — that alone
    // separates them, without looking at a value.
    want(`nn.${name} is there and reduction arrives`,
      call("none").size > 1 && call("sum").size === 1,
      `none=${call("none").size} sum=${call("sum").size}`);
  }

  // **The forty-seven layer factories the binding writes by hand** — the ones torch has.
  //
  // The binding (`borch_webgpu/_nn.py`) builds them as factories on top of the tensor
  // methods, so the Python side is fine. Every golden case goes through the binding, so
  // **the table cannot see these, structurally** — a name is missing only for somebody
  // writing `new nn.GELU()` in TypeScript.
  //
  // **It is not left red.** A runner that is always red is a runner the next person stops
  // reading, so what is asked is not "are they all there" but **"is anything newly gone,
  // beyond what we know about"**. Fill one in and it comes off the list below; leave it on
  // and the run stays green — only the growing side goes red.
  //
  // Filtering the list and then comparing the result back against the same list is always
  // green and asks nothing (it was written that way first, then fixed).
  const FILLED_IN = [
    "AdaptiveLogSoftmaxWithLoss", "AvgPool2d", "BCEWithLogitsLoss", "Bilinear",
    "CELU", "CTCLoss", "Conv1d", "Conv2d", "Conv3d", "CrossEntropyLoss", "ELU",
    "EmbeddingBag", "Flatten", "FractionalMaxPool2d", "FractionalMaxPool3d",
    "GELU", "GLU", "Hardshrink", "Hardtanh", "Identity", "L1Loss", "LPPool1d",
    "LayerNorm", "LeakyReLU", "Linear", "LogSoftmax", "MSELoss", "ModuleDict",
    "ModuleList", "MultiheadAttention", "NLLLoss", "Parameter", "ParameterDict",
    "ParameterList", "ReLU", "Sequential", "SiLU", "Sigmoid", "SmoothL1Loss",
    "Softmax", "Softmin", "Softplus", "Softshrink", "Tanh", "Threshold",
    "Unflatten", "Upsample",
  ];
  // **What is missing right now. All seventeen have been filled in.** An empty list still
  // means something: let one disappear and it goes red as `newly gone`.
  const KNOWN_ABSENT = new Set<string>();
  const bag = nn as unknown as Record<string, unknown>;
  const absent = FILLED_IN.filter((n) => !(n in bag));
  const surprise = absent.filter((n) => !KNOWN_ABSENT.has(n));
  want("no new hole among the layer names the binding fills in", surprise.length === 0,
    `known ${absent.length}/${KNOWN_ABSENT.size}` +
    (surprise.length ? ` · **newly gone**: ${surprise.join(", ")}` : ""));

  // **`SmoothL1Loss`'s first argument differs from the core's.** The core is
  // `(beta, reduction)` and this is `(reduction, beta)`. torch itself takes both by
  // keyword only, so this cannot be called parting from torch — but **the sisters have
  // parted**, and somebody transcribing the same code catches on the first argument.
  // Written here so it is not forgotten the next time this is tidied.
  want("SmoothL1Loss takes reduction first",
    new nn.SmoothL1Loss("none").call(lx(), ly()).size > 1);

  // ── linalg: a namespace over methods that already carry the arithmetic ──
  //
  // **The golden cannot ask this either.** Every value here belongs to a `Tensor` method
  // the golden already holds against real torch, so asking the values again would freeze a
  // second copy of the same answers. What is unasked anywhere else is **the shape of the
  // call** — which argument receives, and what the defaults are — and that is what a
  // namespace is.
  //
  // The three below are the reason it is written by hand rather than generated. A loop
  // over the method names produces all three defects, and none of them raises.
  const sq = keepAlive(Tensor.from([4, 3, 6, 3], [2, 2]));
  const rhs2 = keepAlive(Tensor.from([1, 2], [2, 1]));

  const agreesWith = async (
    name: string, got: Promise<Tensor> | Tensor, expected: Promise<Tensor> | Tensor,
  ): Promise<void> => {
    const [a, b] = [await got, await expected];
    const shapeOk = a.shape.join(",") === b.shape.join(",");
    want(name, shapeOk && same(await a.toArray(), await b.toArray()),
      shapeOk ? "" : `shape [${a.shape}] vs [${b.shape}]`);
  };

  await agreesWith("linalg.det is the method", linalg.det(sq), sq.det());
  await agreesWith("linalg.inv is `inverse`", linalg.inv(sq), sq.inverse());
  await agreesWith("linalg.pinv is `pinverse`", linalg.pinv(sq), sq.pinverse());
  await agreesWith("linalg.matmul is `mm`", linalg.matmul(sq, sq), sq.mm(sq));
  await agreesWith("linalg.solve is the method", linalg.solve(sq, rhs2), sq.solve(rhs2));

  // **`lu_solve` is received by the right-hand side.** torch orders it (LU, pivots, B) and
  // the method is `B.luSolve(LU, pivots)`. Forwarded positionally the first argument would
  // be `B`, and with a square matrix the name, the argument count and the shapes all still
  // agree — only the value is wrong.
  const fac = await sq.luFactor();
  await agreesWith("linalg.luSolve takes the factors first",
    linalg.luSolve(fac.LU, fac.pivots, rhs2), rhs2.luSolve(fac.LU, fac.pivots));
  //
  // Asked with a **square** right-hand side on purpose. With `[2,1]` the swap is refused
  // on shape and the check would be measuring the shape rule instead — the danger is the
  // case where nothing objects, so that is the case asked.
  const rhsSq = keepAlive(Tensor.from([1, 0, 0, 1], [2, 2]));
  const straight = Array.from(
    await (await linalg.luSolve(fac.LU, fac.pivots, rhsSq)).toArray()).join(",");
  const swapped = await linalg.luSolve(rhsSq, fac.pivots, fac.LU).then(
    async (t) => Array.from(await t.toArray()).join(","), () => "(it refused)");
  want("swapping lu_solve's arguments answers, and answers differently",
    swapped !== straight && swapped !== "(it refused)",
    `${straight}  vs  ${swapped}`);

  // **`diagonal` reads different axes under the two names.** `torch.diagonal` takes the
  // first two and `torch.linalg.diagonal` the last two, so on rank 3 even the shape
  // differs — `(2,3,4)` gives `(2,3)` here and `(4,2)` through the method's defaults.
  want("linalg.diagonal reads the last two axes",
    linalg.diagonal(cube).shape.join(",") === "2,3",
    linalg.diagonal(cube).shape.join(","));
  want("the method's defaults still read the first two",
    cube.diagonal().shape.join(",") === "4,2", cube.diagonal().shape.join(","));

  // `multiDot` chooses the cheapest parenthesisation, and every order gives the same
  // matrix, so it is asked against the plain chain.
  const thin = keepAlive(Tensor.from([1, 2, 3, 4, 5, 6], [3, 2]));
  const wide = keepAlive(Tensor.from([1, 0, 2, 0, 1, 3], [2, 3]));
  await agreesWith("multiDot is the chain",
    linalg.multiDot([thin, wide, thin]), thin.mm(wide).mm(thin));

  const folded = keepAlive(Tensor.from([4, 3, 6, 3], [2, 1, 2]));
  const back = await linalg.tensorinv(folded, 2);
  want("tensorinv folds, inverts and unfolds",
    back.shape.join(",") === "2,2,1", back.shape.join(","));

  // **The internal kernel must not be reachable from the namespace.** `_linalg.ts` takes
  // flat `Float64Array`s and dimension counts — `matmul(a, b, n, k, m)` — and forty of
  // those names were being swept into the published reference as though they were the API.
  // Re-export it and the reference fills with signatures nobody can call; that is what
  // this row is here to catch.
  const kernelOnly = [
    "fromF32", "toF32", "mirror", "completeBasis", "eighGap", "hessenberg",
    "luExpand", "luSolveFactored", "choleskyBackward", "matrixExpAdjointMap",
  ];
  const leaked = kernelOnly.filter(
    (n) => (linalg as unknown as Record<string, unknown>)[n] !== undefined);
  want("the internal kernel is not re-exported", leaked.length === 0,
    leaked.length ? `leaked: ${leaked.join(", ")}` : `${kernelOnly.length} checked`);

  // `cholesky_ex` and `inv_ex` return a status instead of raising. Nothing here produces
  // that status, and a wrapper always reporting success would be a check that cannot fail.
  for (const missing of ["choleskyEx", "invEx"]) {
    want(`linalg.${missing} is not offered — the refusal is the answer`,
      (linalg as unknown as Record<string, unknown>)[missing] === undefined);
  }

  // ── std and variance: the axis argument that was not there ──────────
  //
  // **The golden asks `std()` three times and never with an argument**, so it could not
  // see this. `std(correction = 1)` took the correction first, alone among the reductions
  // — `mean`, `sumDim`, `amax` all take `dim` — and `x.std(0)`, which is what anybody
  // transcribing torch writes, compiled, ran, and returned a **scalar at correction 0**
  // where torch returns one value per column. Not a crash: an answer, of a different
  // rank, breaking somewhere else entirely.
  //
  // Found by the signature axis (`tests/ts_signatures.py`), not by any value check. The
  // values below are real torch's, taken on the same input.
  const grid = keepAlive(Tensor.from([1, 2, 3, 4, 6, 8], [2, 3]));

  const nearAll = async (
    name: string, got: Tensor, expected: number[],
  ): Promise<void> => {
    const seen = Array.from(await got.toArray());
    want(name, seen.length === expected.length
      && seen.every((v, i) => near(v, expected[i] ?? NaN, 1e-5)),
      `${seen.map((v) => v.toFixed(4)).join(", ")}`);
  };

  await nearAll("std(0) folds axis 0 — torch's answer",
    grid.std(0), [2.1213202, 2.8284271, 3.5355339]);
  await nearAll("std(1) folds axis 1", grid.std(1), [1, 2]);
  await nearAll("variance(0, 0) takes the correction second",
    grid.variance(0, 0), [2.25, 4, 6.25]);
  want("std() with no axis is still the whole tensor",
    near(await grid.std().item(), 2.6076810, 1e-5), `${await grid.std().item()}`);
  want("keepdim keeps the folded axis",
    grid.std(1, 1, true).shape.join(",") === "2,1", grid.std(1, 1, true).shape.join(","));

  // **`stdMean` has to fold the same axis in both halves.** It returned `this.mean()`
  // regardless of the axis, so asking for a per-column standard deviation gave it beside
  // the mean of everything — two answers of different rank in one object.
  const pair = grid.stdMean(0);
  want("stdMean folds the same axis in both halves",
    pair.std.shape.join(",") === "3" && pair.mean.shape.join(",") === "3",
    `std [${pair.std.shape}] · mean [${pair.mean.shape}]`);
  await nearAll("stdMean's mean is the mean over that axis", pair.mean, [2.5, 4, 5.5]);

  // ── conv: the arguments that arrived after the core moved ────────────
  //
  // **The golden cannot ask these.** Its cases go through the binding, which calls
  // `convND` with the four arguments it always had; `dilation` and `groups` are
  // reachable only from TypeScript until the binding grows them. And no value is
  // written down here — each check is an **equivalence**, so what is asked is that
  // the new argument means what it says rather than that some array is right.
  const cx = keepAlive(Tensor.from(
    Array.from({ length: 2 * 4 * 6 * 6 }, (_, i) => ((i * 7) % 13) / 13 - 0.5),
    [2, 4, 6, 6]));

  // `groups=2` is two convolutions on the channel halves, joined. Written out by
  // hand here, so the check knows nothing about how the argument is implemented.
  const gw = keepAlive(Tensor.from(
    Array.from({ length: 6 * 2 * 3 * 3 }, (_, i) => ((i * 5) % 11) / 11 - 0.5),
    [6, 2, 3, 3]));
  const grouped = cx.convND(gw, null, 1, 0, 1, 2);
  const byHand = Tensor.cat([
    cx.narrow(1, 0, 2).convND(gw.narrow(0, 0, 3), null, 1, 0),
    cx.narrow(1, 2, 2).convND(gw.narrow(0, 3, 3), null, 1, 0),
  ], 1);
  await agrees("groups=2 is two convolutions on the halves", grouped, byHand);

  // `dilation=2` is the same convolution with the filter's cells spread out. A
  // 3×3 dilated by 2 covers 5×5 with zeros between, so the equivalence is against
  // a kernel written that way — which asks the shader's index arithmetic and
  // nothing else.
  const dw = keepAlive(Tensor.from(
    Array.from({ length: 2 * 4 * 3 * 3 }, (_, i) => ((i * 3) % 7) / 7 - 0.5),
    [2, 4, 3, 3]));
  const spread = new Float32Array(2 * 4 * 5 * 5);
  const flat = await dw.toArray();
  for (let o = 0; o < 2; o++) {
    for (let c = 0; c < 4; c++) {
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          spread[((o * 4 + c) * 5 + i * 2) * 5 + j * 2] =
            flat[((o * 4 + c) * 3 + i) * 3 + j] ?? 0;
        }
      }
    }
  }
  await agrees("dilation=2 is the filter with its cells spread apart",
    cx.convND(dw, null, 1, 0, 2), cx.convND(keepAlive(Tensor.from(spread, [2, 4, 5, 5]))));

  // The layer's non-zero padding mode pads and then convolves with padding 0 —
  // the same split torch's layer makes, and the reason `F.conv2d` has no such
  // argument.
  {
    const layer = new nn.Conv2d(4, 2, 3, 1, 1, 1, 1, false, "reflect");
    const plain = cx.padND([1, 1, 1, 1], "reflect")
      .convND(layer.weight, null, 1, 0);
    await agrees("padding_mode pads first and convolves with 0",
      layer.call(cx), plain);
  }

  want("Conv2d takes bias eighth, as torch does",
    new nn.Conv2d(4, 2, 3, 1, 0, 1, 1, false).bias === null,
    "a positional `false` in the sixth seat would be a dilation now");

  // ── convTranspose: the three that followed the core ──────────────────
  //
  // Equivalences again, so nothing here writes a value down.
  const tx = keepAlive(Tensor.from(
    Array.from({ length: 2 * 4 * 5 * 5 }, (_, i) => ((i * 11) % 17) / 17 - 0.5),
    [2, 4, 5, 5]));
  const tw = keepAlive(Tensor.from(
    Array.from({ length: 4 * 3 * 3 * 3 }, (_, i) => ((i * 5) % 13) / 13 - 0.5),
    [4, 3, 3, 3]));

  await agrees("transpose groups=2 is two transposes on the halves",
    tx.convTransposeND(tw, null, 1, 0, 0, 2),
    Tensor.cat([
      tx.narrow(1, 0, 2).convTransposeND(tw.narrow(0, 0, 2), null, 1, 0),
      tx.narrow(1, 2, 2).convTransposeND(tw.narrow(0, 2, 2), null, 1, 0),
    ], 1));

  // **`outputPadding` reaches back into what the padding trim threw away.** With
  // padding 1 and outputPadding 1 the answer has to be the untrimmed transpose cut
  // at `[p, len - p + op)` — real values, not zeros. Writing zeros there agrees on
  // the shape and differs in the values, which is how the core found it.
  {
    const full = tx.convTransposeND(tw, null, 2, 0);
    const len = full.shape[2] ?? 0;
    await agrees("outputPadding takes computed values, not zeros",
      tx.convTransposeND(tw, null, 2, 1, 1),
      full.narrow(2, 1, len - 1).narrow(3, 1, len - 1));
  }

  // `dilation=2` against the same filter spread apart, as for the convolution.
  {
    const spread = new Float32Array(4 * 3 * 5 * 5);
    const flat = await tw.toArray();
    for (let i = 0; i < 4; i++) {
      for (let o = 0; o < 3; o++) {
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) {
            spread[((i * 3 + o) * 5 + a * 2) * 5 + b * 2] =
              flat[((i * 3 + o) * 3 + a) * 3 + b] ?? 0;
          }
        }
      }
    }
    await agrees("transpose dilation=2 is the filter spread apart",
      tx.convTransposeND(tw, null, 1, 0, 0, 1, 2),
      tx.convTransposeND(keepAlive(Tensor.from(spread, [4, 3, 5, 5]))));
  }

  want("ConvTranspose2d takes bias eighth and dilation ninth",
    new nn.ConvTranspose2d(4, 2, 3, 1, 0, 0, 1, false).bias === null,
    "torch orders this one differently from Conv2d, and so does this");

  // ── EmbeddingBag: the arguments, and the one that only shows twice ───
  //
  // `mode` moved from third to sixth. `tsc` named all eight call sites the moment
  // it did — five golden cases and the layer's own two — because a mode string
  // does not fit a `number | null`. The same move on the Python side was silent.
  const ebW = () => keepAlive(Tensor.from(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [5, 3]));
  const ebI = keepAlive(Tensor.from([0, 1, 2, 3], [2, 2], { dtype: "int64" }));

  // **`paddingIdx` leaves the bag rather than contributing zero to it.** Under
  // `sum` those are the same thing and under `mean` they are not, so the check is
  // at `mean`, as an equivalence: bag 0 is rows 0 and 1 with row 1 padded, so it
  // has to be **row 0 itself** — not their sum halved.
  //
  // Written by hand first, and the hand was wrong: the expected array had bag 1 as
  // row 3 rather than the mean of rows 2 and 3, so the check failed against a
  // correct implementation. `parity.ts` says two hundred lines up that writing
  // values out by hand makes the check wrong wherever the hand is, and it was
  // right within the hour.
  {
    const padded = nn.embeddingBag(
      ebI, ebW(), null, null, 2, false, "mean", false, null, false, 1);
    await agrees("paddingIdx at mean gives the unpadded row itself",
      padded.narrow(0, 0, 1), ebW().narrow(0, 0, 1));
    // And it is not the zero-contributing reading, which would halve it.
    const zeroed = ebW().narrow(0, 0, 1).mul(Tensor.full([], 0.5));
    const seen = Array.from(await padded.narrow(0, 0, 1).toArray());
    const halved = Array.from(await zeroed.toArray());
    want("paddingIdx is not the same as contributing zero",
      seen.some((v, i) => Math.abs(v - (halved[i] ?? 0)) > 1e-6),
      `${seen.join(",")} against ${halved.join(",")}`);
  }

  // `includeLastOffset` means the last entry closes the final bag rather than
  // opening a new one, so the same bags come out of one more offset.
  const flatIdx = keepAlive(Tensor.from([0, 1, 2, 3], [4], { dtype: "int64" }));
  await agrees("includeLastOffset closes the last bag",
    nn.embeddingBag(flatIdx, ebW(), [0, 2, 4], null, 2, false, "sum", false, null, true),
    nn.embeddingBag(flatIdx, ebW(), [0, 2], null, 2, false, "sum"));

  // **`maxNorm` rewrites the table, and the check has to look at the table.**
  //
  // A version that renormalised a copy returns **the same numbers forever** — not
  // just on the first call. Renormalising an already-short row is a no-op, so both
  // implementations agree on the output at any number of repetitions (measured
  // against real torch, three deep, identical to seven figures). They part on the
  // *state*: `[0, 0.4472, 0.8944]` against `[0, 1, 2]`, at once.
  //
  // Every other instrument here compares a returned value — the golden runs a case,
  // the axes read a signature, every check above weighs an output — and **no number
  // of repetitions turns a value comparison into a state comparison.** So the check
  // reads `weight` after the call, which is the only thing that separates the two.
  //
  // This comment said "agrees on the first call and parts on the second" for an
  // hour. That reads correctly and points at the wrong check: run it twice, see
  // agreement, call it confirmed. A peer re-measured and it was wrong.
  {
    const table = ebW();
    const before = Array.from(await table.toArray());
    nn.embeddingBag(ebI, table, null, 1.0, 2, false, "sum");
    const after = Array.from(await table.toArray());
    want("maxNorm shortens the rows in the table itself",
      after.some((v, i) => v !== before[i]),
      "a copy would leave the parameter untouched and agree on the output");

    // The rows `idx` never named keep their length. A whole-table renormalisation
    // is the obvious shortcut and torch does not do it.
    want("maxNorm leaves the rows nobody looked up",
      after[12] === before[12] && after[13] === before[13],
      `row 4: ${after.slice(12).join(",")}`);

    // Renormalised once, a row is exactly at the limit, so asking again changes
    // nothing. **This one separates nothing** — a copy-based version passes it, and
    // so does a whole-table one. It is here because it pins torch's idempotence,
    // which the two above do not, and it is labelled rather than left to look like
    // part of the discrimination.
    nn.embeddingBag(ebI, table, null, 1.0, 2, false, "sum");
    const twice = Array.from(await table.toArray());
    want("maxNorm a second time is a no-op",
      twice.every((v, i) => Math.abs(v - (after[i] ?? 0)) < 1e-6),
      `${twice.slice(0, 3).map((v) => v.toFixed(4)).join(",")}`);
  }

  // ── vision: the place a widened type opened ──────────────────────────
  //
  // **The golden cannot ask this.** `Transform` widened to take an array as well, for
  // `FiveCrop` and `TenCrop`, and with that `Compose([new FiveCrop(3), new ToTensor()])`
  // passes the type check. There is no counterpart on the Python side — its `ToTensor`
  // takes a numpy array and a tuple dies there some other way — so there is no case to put
  // to the golden at all.
  //
  // Before it was stopped, this blew up as `shape [,,] does not match 0 elements`
  // (measured). It does blow up, so "it refuses" was true, but nothing in the accident
  // said what had been done wrong. This is where the **TS surface** is asked, so it goes
  // here.
  const pic = vision.image(new Float64Array(12), 2, 2, 3, false);
  wantThrow("ToTensor refuses several pictures", "it received 5 of them",
    () => new vision.ToTensor().apply(new vision.FiveCrop([1, 1]).apply(pic)));
  wantThrow("inside Compose it stops in the same place", "Lambda",
    () => new vision.Compose([new vision.FiveCrop([1, 1]), new vision.ToTensor()])
      .apply(pic));

  // **One validation fault and the green above cannot be believed.** WebGPU drops an
  // invalid command buffer quietly, so a check can come to read an unchanged value as a
  // pass.
  want("no WebGPU validation fault", device().faults.count === 0,
    device().faults.first);

  const failed = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push(failed.length === 0
    ? `all ${checks.length} torch wiring checks passed`
    : `**${failed.length} failed** / ${checks.length}`);
  return { text: lines.join("\n"), checks };
}
