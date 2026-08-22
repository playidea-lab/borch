/**
 * Checkpoints — the round trip and **resume equivalence.**
 *
 * **The round trip alone is not enough.** Saving and reading back and finding the values
 * equal asks about the codec and nothing else; the real question comes after it: *is
 * training that was stopped and resumed the same as training that never stopped.* Leave
 * out one momentum buffer, one step counter, one scheduler epoch, and the round trip
 * stays green while the resume alone parts.
 *
 * All of it is deterministic, so **it has to be equal bit for bit.** A tolerance here
 * would read a piece of state left behind as rounding, so only exact equality is asked.
 */

import {
  decode, encode, init, keepAlive, load, manualSeed, metaToNumbers, nn,
  numbersToMeta, optim, prefixed, save, scope, Tensor, unprefixed,
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

function same(a: Float32Array, b: Float32Array): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

class Net extends nn.Module {
  fc1 = new nn.Linear(6, 5);
  fc2 = new nn.Linear(5, 3);

  override forward(x: Tensor): Tensor {
    return this.fc2.call(this.fc1.call(x).relu());
  }
}

/** The same model, optimizer and scheduler, stood up from the same seed. */
function build(): {
  model: Net; opt: optim.Adam; sched: optim.StepLR; crit: nn.CrossEntropyLoss;
} {
  manualSeed(20260817);
  const model = new Net();
  for (const p of model.parameters()) keepAlive(p);
  const opt = new optim.Adam(model.parameters(), 0.05);
  return {
    model, opt,
    sched: new optim.StepLR(opt, 2, 0.5).start(),
    crit: new nn.CrossEntropyLoss(),
  };
}

const BATCH = 8;
const FEATURES = 6;

function inputs(): { x: Tensor; y: Tensor } {
  const pixels = new Float32Array(BATCH * FEATURES);
  for (let i = 0; i < pixels.length; i++) pixels[i] = (i % 11) / 11 - 0.5;
  const labels = new Float32Array(BATCH);
  for (let i = 0; i < BATCH; i++) labels[i] = i % 3;
  return {
    x: keepAlive(Tensor.from(pixels, [BATCH, FEATURES])),
    y: keepAlive(Tensor.from(labels, [BATCH], { dtype: "int64" })),
  };
}

async function train(
  kit: ReturnType<typeof build>, data: ReturnType<typeof inputs>, steps: number,
): Promise<number[]> {
  const seen: number[] = [];
  for (let i = 0; i < steps; i++) {
    await scope(async () => {
      kit.opt.zeroGrad();
      const loss = kit.crit.call(kit.model.call(data.x), data.y);
      loss.backward();
      kit.opt.step();
      seen.push(await loss.item());
    });
    kit.sched.step();
  }
  return seen;
}

/**
 * One sample for the Python side to take apart.
 *
 * **The reason for choosing this format is proved here and nowhere else.** Our codec
 * round-tripping with our codec would work in a format of our own; what carrying
 * safetensors buys is that **somebody else can read it.** The runner takes these bytes and
 * parses them with numpy directly — without a line of borch code.
 */
export async function sample(): Promise<number[]> {
  await init();
  const bytes = await save({
    "fc.weight": Tensor.from([1.5, -2.25, 0.5, 7, -0.125, 3], [2, 3]),
    "fc.labels": Tensor.from([3, 1, 4], [3], { dtype: "int64" }),
  }, { note: "cross-language" });
  return Array.from(bytes);
}

/**
 * One nested sample. **There are two copies of the tree scheme, and this is the only
 * place the two are put against each other.**
 *
 * `serialize.ts` and Python's `_serialize.py` are supposed to write the same node kinds
 * (`T`/`d`/`l`/`j`) under the same letter (`borch.tree`), and until now nobody measured
 * that promise. Mend one side alone and **a checkpoint written by one cannot be read by
 * the other** — and what comes out then is not an exception but a dictionary of a
 * different shape, which is found later.
 *
 * The `sample` above cannot see it. Its top level is a dictionary of tensors, so the same
 * thing comes out with or without the tree — **ask only about the flat case and the tree
 * is never trodden on.**
 */
export async function sampleNested(): Promise<number[]> {
  await init();
  const bytes = await save({
    model: { "fc.weight": Tensor.from([1.5, -2.25], [2]) },
    steps: [Tensor.from([7], [1]), 3],
    epoch: 5,
    note: "nested",
    done: false,
    nothing: null,
  });
  return Array.from(bytes);
}

export async function report(): Promise<Report> {
  await init();
  const data = inputs();

  // ── The codec ───────────────────────────────────────────────────────
  const original = {
    weight: Tensor.from([1.5, -2.25, 0, 7], [2, 2]),
    labels: Tensor.from([3, 1, 4], [3], { dtype: "int64" }),
    flags: Tensor.from([1, 0], [2], { dtype: "bool" }),
    empty: Tensor.from(new Float32Array(0), [0]),
  };
  // **The codec is called directly.** `save`/`load` are where the tree sits on top of
  // it, and what this passage asks about is the bytes underneath — with the tree in the
  // way, what broke is blurred.
  const bytes = await encode(original, { note: "a borch checkpoint" });
  const back = decode(bytes);

  want("the names come through unchanged",
    Object.keys(back.tensors).sort().join(",") === "empty,flags,labels,weight",
    Object.keys(back.tensors).join(","));
  want("the values are exactly equal",
    same(await original.weight.toArray(), await back.tensors.weight!.toArray()));
  want("the shape comes through unchanged",
    back.tensors.weight!.shape.join(",") === "2,2");
  // **The label does not ride in the body.** The values are always float32 and int64 and
  // bool are written in the header — otherwise it would be a four-byte body labelled I64,
  // and somebody else's reader breaks on it.
  want("the dtype label survives",
    back.tensors.labels!.dtype === "int64" && back.tensors.flags!.dtype === "bool",
    `${back.tensors.labels!.dtype} / ${back.tensors.flags!.dtype}`);
  want("an empty tensor round-trips too",
    back.tensors.empty!.size === 0 && back.tensors.empty!.shape.join(",") === "0");
  want("the metadata rides along", back.metadata.note === "a borch checkpoint");

  // safetensors writes the header length in front as 8 bytes LE and starts the body at
  // byte 8.
  const headerLength = Number(new DataView(bytes.buffer, bytes.byteOffset)
    .getBigUint64(0, true));
  want("the header is aligned to 8 bytes", (8 + headerLength) % 8 === 0,
    `header ${headerLength}`);
  want("saving twice gives the same bytes",
    same(new Float32Array((await encode(original)).buffer.slice(0)),
      new Float32Array((await encode(original)).buffer.slice(0))));

  // A broken file must not become a strange tensor, quietly.
  for (const [name, broken] of [
    ["a truncated file", bytes.subarray(0, 4)],
    ["a file whose body is short", bytes.subarray(0, bytes.length - 4)],
  ] as [string, Uint8Array][]) {
    let threw = false;
    try { decode(broken); } catch { threw = true; }
    want(`${name} is refused`, threw);
  }

  // ── The model round trip ────────────────────────────────────────────
  const trained = build();
  await train(trained, data, 3);
  const restored = build();
  restored.model.loadStateDict(
    decode(await encode(trained.model.stateDict())).tensors);
  want("a model state_dict crosses the bytes",
    same(await trained.model.parameters()[0]!.toArray(),
      await restored.model.parameters()[0]!.toArray()));

  // ── Resume equivalence ──────────────────────────────────────────────
  // **This is the reason the runner exists.** Ten steps run straight through and five
  // stopped and resumed have to give the same loss trajectory.
  const straightKit = build();
  const straight = await train(straightKit, data, 10);

  const first = build();
  const early = await train(first, data, 5);
  const optState = first.opt.stateDict();
  const checkpoint = await encode(
    {
      ...prefixed("model", first.model.stateDict()),
      ...prefixed("opt", optState.tensors),
    },
    {
      ...numbersToMeta("opt", optState.numbers),
      ...numbersToMeta("sched", first.sched.stateDict()),
    },
  );

  const second = build();
  const read = decode(checkpoint);
  second.model.loadStateDict(unprefixed("model", read.tensors));
  second.opt.loadStateDict({
    tensors: unprefixed("opt", read.tensors),
    numbers: metaToNumbers("opt", read.metadata),
  });
  second.sched.loadStateDict(metaToNumbers("sched", read.metadata));
  const resumed = await train(second, data, 5);

  const joined = [...early, ...resumed];
  const exact = joined.length === straight.length
    && joined.every((v, i) => v === straight[i]);
  want("stopped-and-resumed matches straight-through bit for bit", exact,
    exact ? `${straight.length} steps`
      : `straight ${straight.slice(4, 7).map((v) => v.toFixed(6)).join(" ")}\n` +
        `      resumed ${joined.slice(4, 7).map((v) => v.toFixed(6)).join(" ")}`);

  // Leaving out one piece of state has to break the check above — this is what asks
  // whether the check measures anything at all.
  const naive = build();
  naive.model.loadStateDict(unprefixed("model", read.tensors));   // the weights alone
  const careless = await train(naive, data, 5);
  want("restoring the weights alone makes the trajectory part",
    !careless.every((v, i) => v === resumed[i]),
    "if this one is green, the equivalence above is measuring nothing");

  // ── Scheduler state ─────────────────────────────────────────────────
  // **The comparison has to be against the straight-through run.** It first weighed the
  // stopped side (`first`, 5 steps) against the resumed one (`second`, 10 in total) and
  // caught — of course those two differ. The check was wrong about itself.
  want("a resumed scheduler is at the same learning rate as a straight run",
    second.opt.paramGroups[0]!.lr === straightKit.opt.paramGroups[0]!.lr,
    `resumed ${second.opt.paramGroups[0]!.lr} / `
    + `straight ${straightKit.opt.paramGroups[0]!.lr}`);

  // ── The same thing again, nested ────────────────────────────────────
  //
  // The resume above laid everything flat under prefixed names and pulled the numbers out
  // into the metadata. That was the only road before this layer existed, and **it is not
  // what the textbooks write** — a textbook saves `{model: …, opt: …, epoch: 3}` whole.
  //
  // The same trajectory has to come out. If it does not, the tree dropped something, and
  // **unasked here that dropping is invisible to the golden** — the golden looks at key
  // names and the values beside them and never resumes a training run.
  const third = build();
  const early3 = await train(third, data, 5);
  const state3 = third.opt.stateDict();
  const nested = await save({
    model: third.model.stateDict(),
    opt: { tensors: state3.tensors, numbers: state3.numbers },
    sched: third.sched.stateDict(),
    epoch: 5,
    note: "half way",
  });

  const fourth = build();
  const back3 = load(nested) as {
    model: Record<string, Tensor>;
    opt: { tensors: Record<string, Tensor>; numbers: Record<string, number> };
    sched: Record<string, number>;
    epoch: number;
    note: string;
  };
  fourth.model.loadStateDict(back3.model);
  fourth.opt.loadStateDict(back3.opt);
  fourth.sched.loadStateDict(back3.sched);
  const resumed3 = await train(fourth, data, 5);

  const joined3 = [...early3, ...resumed3];
  want("a resume saved nested also matches straight-through bit for bit",
    joined3.length === straight.length && joined3.every((v, i) => v === straight[i]),
    `${joined3.length} steps`);
  // What is not a tensor rides along too — that is where it parts from a flat table.
  want("the tree carries numbers and words alongside",
    back3.epoch === 5 && back3.note === "half way",
    `epoch=${String(back3.epoch)} note=${String(back3.note)}`);

  // **A dotted key must not be split again.** `stateDict`'s names already carry dots
  // (`fc1.weight`). Rebuild by splitting the flattened names on the dot and every value is
  // there while the structure is different.
  const dotted = load(await save({ model: third.model.stateDict() })) as
    { model: Record<string, Tensor> };
  want("a dotted key inside the tree is not split",
    Object.keys(dotted.model).includes("fc1.weight"),
    Object.keys(dotted.model).sort().join(" "));

  // A file with no tree — somebody else's safetensors. It has to arrive as a flat table.
  const foreign = load(await encode({ w: Tensor.from([1, 2], [2]) })) as
    Record<string, Tensor>;
  want("a file with no tree arrives as a flat table",
    foreign.w !== undefined && foreign.w.shape.join(",") === "2",
    Object.keys(foreign).join(","));

  // JSON cannot write an infinity. `ReduceLROnPlateau`'s `best` starts at one.
  const plateau = new optim.ReduceLROnPlateau(build().opt);
  const meta = numbersToMeta("p", plateau.stateDict());
  want("an infinity crosses the header",
    metaToNumbers("p", meta).best === Infinity, meta["p.best"] ?? "(absent)");

  const failed = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push(failed.length === 0
    ? `all ${checks.length} checkpoint checks passed`
    : `**${failed.length} failed** / ${checks.length}`);
  return { text: lines.join("\n"), checks };
}
