/**
 * Does `nn.LoRALinear` behave — the invariants that stand in for a golden, since
 * torch.nn has no LoRA to match against.
 *
 *     uv run --with playwright python borch-ts/test/lora.py [--headed]
 *
 * Checks, in order: the zero-init layer is exactly its base; the low-rank forward
 * equals the folded full weight `W + (alpha/r)·B·A` (which also proves `merge()`);
 * `merge()` reproduces the forward; `parameters()` is the adapter alone while the base
 * stays in `stateDict()`; and a backward reaches A and B but not the frozen base.
 */
import { Conv2d, Linear } from "../src/nn.js";
import { LoRAConv2d, LoRALinear } from "../src/peft.js";
import { SGD } from "../src/optim.js";
import { Tensor } from "../src/tensor.js";

export interface Check { name: string; ok: boolean; note: string }

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  return worst;
}

function pixels(n: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; out[i] = (s / 4294967296) * 2 - 1; }
  return out;
}

const GATE = 1e-5;

export async function report(): Promise<{ text: string; checks: Check[] }> {
  const lines: string[] = [];
  const checks: Check[] = [];
  const IN = 12, OUT = 6, R = 4, B = 3;

  const lora = new LoRALinear(IN, OUT, { r: R, alpha: 8 });
  const x = Tensor.from(pixels(B * IN, 7), [B, IN]);

  // 1) zero-init: B is zero, so the layer is exactly base = x·Wᵀ + b.
  {
    const ours = await lora.forward(x).toArray();
    const base = await x.linear(lora.weight).add(lora.bias!).toArray();
    const gap = maxAbsDiff(ours, base);
    checks.push({ name: "zero-init layer equals its base Linear", ok: gap <= GATE, note: `max |Δ| ${gap.toExponential(2)}` });
    lines.push(`zero-init: max |Δ| ${gap.toExponential(2)}`);
  }

  // Make the adapter non-zero with one optimiser step, so B ≠ 0 for the rest.
  const opt = new SGD(lora.parameters(), 0.5);
  { opt.zeroGrad(); const l = lora.forward(x).sum(); l.backward(); opt.step(); }

  // 2) the low-rank forward equals the folded full weight W' = W + (alpha/r)·(B·A).
  {
    const Wp = lora.weight.add(lora.loraB.matmul(lora.loraA).mul(Tensor.full([], lora.scaling))); // (out,in)
    const ref = await x.linear(Wp).add(lora.bias!).toArray();
    const ours = await lora.forward(x).toArray();
    const gap = maxAbsDiff(ours, ref);
    checks.push({ name: "low-rank forward equals the folded full weight", ok: gap <= GATE, note: `max |Δ| ${gap.toExponential(2)}` });
    lines.push(`folded-weight: max |Δ| ${gap.toExponential(2)}`);
  }

  // 3) merge() gives a plain Linear that reproduces the forward.
  {
    const merged = lora.merge();
    const ok0 = merged instanceof Linear;
    const ours = await lora.forward(x).toArray();
    const m = await merged.forward(x).toArray();
    const gap = maxAbsDiff(ours, m);
    checks.push({ name: "merge() is a Linear that reproduces the forward", ok: ok0 && gap <= GATE, note: `Linear=${ok0} · max |Δ| ${gap.toExponential(2)}` });
    lines.push(`merge: max |Δ| ${gap.toExponential(2)}`);
  }

  // 4) parameters() is the adapter alone; the base is in stateDict() as a buffer.
  {
    const params = Object.keys(lora.namedParameters()).sort();
    const state = Object.keys(lora.stateDict()).sort();
    const adapter = Object.keys(lora.adapterState()).sort();
    const okParams = params.length === 2 && params.join(",") === "lora_A,lora_B";
    const okState = state.includes("weight") && state.includes("bias")
      && state.includes("lora_A") && state.includes("lora_B");
    const okAdapter = adapter.join(",") === "lora_A,lora_B";
    checks.push({ name: "parameters() is the adapter; base stays in stateDict()", ok: okParams && okState && okAdapter,
      note: `params=[${params}] state=[${state}] adapter=[${adapter}]` });
    lines.push(`surface: params=[${params}] · state=[${state}]`);
  }

  // 5) a backward reaches A and B, and not the frozen base.
  const amax = async (t: Tensor): Promise<number> =>
    Math.max(0, ...(await t.abs().toArray() as unknown as number[]));
  {
    const fresh = new LoRALinear(IN, OUT, { r: R, alpha: 8 });
    const o2 = new SGD(fresh.parameters(), 0.5);
    o2.zeroGrad(); fresh.forward(x).sum().backward(); o2.step();       // B ≠ 0 now
    fresh.forward(x).sum().backward();                                  // grads to measure
    const gA = fresh.loraA.grad, gB = fresh.loraB.grad;
    const aOk = gA != null && await amax(gA) > 0;
    const bOk = gB != null && await amax(gB) > 0;
    const baseFrozen = fresh.weight.grad == null && (fresh.bias == null || fresh.bias.grad == null)
      && fresh.weight.requiresGrad === false;
    checks.push({ name: "backward reaches A and B, not the frozen base", ok: aOk && bOk && baseFrozen,
      note: `A.grad≠0=${aOk} · B.grad≠0=${bOk} · base frozen=${baseFrozen}` });
    lines.push(`grad: A≠0=${aOk} B≠0=${bOk} base-frozen=${baseFrozen}`);
  }

  // ── LoRAConv2d — the same invariants on a convolution ──
  const CIN = 4, COUT = 6, K = 3, HW = 8;
  const img = Tensor.from(pixels(B * CIN * HW * HW, 21), [B, CIN, HW, HW]);
  const conv = new LoRAConv2d(CIN, COUT, K, { r: R, alpha: 8, padding: 1 });

  // 6) zero-init conv equals its base convolution.
  {
    const ours = await conv.forward(img).toArray();
    const base = await img.conv2d(conv.weight, conv.bias, 1, 1, 1, 1).toArray();
    const gap = maxAbsDiff(ours, base);
    checks.push({ name: "zero-init LoRAConv2d equals its base conv", ok: gap <= GATE, note: `max |Δ| ${gap.toExponential(2)}` });
    lines.push(`conv zero-init: max |Δ| ${gap.toExponential(2)}`);
  }

  // make the conv adapter non-zero.
  const co = new SGD(conv.parameters(), 0.2);
  { co.zeroGrad(); conv.forward(img).sum().backward(); co.step(); }

  // 7) merge() folds the two convs into one that reproduces the forward.
  {
    const merged = conv.merge();
    const ok0 = merged instanceof Conv2d;
    const ours = await conv.forward(img).toArray();
    const m = await merged.forward(img).toArray();
    const gap = maxAbsDiff(ours, m);
    checks.push({ name: "LoRAConv2d.merge() is a Conv2d that reproduces the forward", ok: ok0 && gap <= GATE, note: `Conv2d=${ok0} · max |Δ| ${gap.toExponential(2)}` });
    lines.push(`conv merge: max |Δ| ${gap.toExponential(2)}`);
  }

  // 8) conv params are the adapter alone; base stays in stateDict(); backward reaches down/up not base.
  {
    const params = Object.keys(conv.namedParameters()).sort();
    const state = Object.keys(conv.stateDict()).sort();
    const okSurface = params.join(",") === "lora_down,lora_up"
      && state.includes("weight") && state.includes("lora_down") && state.includes("lora_up");
    conv.forward(img).sum().backward();
    const dOk = conv.down.grad != null && await amax(conv.down.grad) > 0;
    const uOk = conv.up.grad != null && await amax(conv.up.grad) > 0;
    const frozen = conv.weight.grad == null && conv.weight.requiresGrad === false;
    checks.push({ name: "LoRAConv2d: adapter-only params, base frozen, grads reach down/up", ok: okSurface && dOk && uOk && frozen,
      note: `params=[${params}] · down≠0=${dOk} up≠0=${uOk} base-frozen=${frozen}` });
    lines.push(`conv surface+grad: params=[${params}] down≠0=${dOk} up≠0=${uOk}`);
  }

  const failed = checks.filter((c) => !c.ok);
  lines.push(failed.length ? `${failed.length} check(s) failed` : `all ${checks.length} LoRA checks passed`);
  return { text: lines.join("\n"), checks };
}
