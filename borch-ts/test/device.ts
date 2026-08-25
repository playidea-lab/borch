/**
 * Device handling — negotiation, availability and placement, in a browser.
 *
 * **The golden does not catch this.** The golden is an instrument for asking whether a
 * value equals torch's, and what is asked here is not a value but *where it is* and *what
 * is said when it is not there.* Those are different questions, so the runner is separate.
 *
 * It does not refuse on a software adapter — placement follows the same rules on any.
 */

import {
  currentDevice,
  device,
  init,
  isAvailable,
  keepAlive,
  probe,
  scope,
  Tensor,
} from "../src/index.js";

const CROSS_DEVICE = "Expected all tensors to be on the same device";

interface Check {
  name: string;
  ok: boolean;
  note: string;
}

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

/** Where it has to throw. **Not throwing is the failure** — pass quietly and the value
 * is wrong. */
function wantThrow(name: string, fragment: string, body: () => unknown): void {
  try {
    body();
    want(name, false, "it did not throw — it passed quietly");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    want(name, message.includes(fragment), message.includes(fragment)
      ? "" : `different wording: ${message}`);
  }
}

function same(a: Float32Array, b: Float32Array): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export async function report(): Promise<Report> {
  // ── Before it attaches ──────────────────────────────────────────────
  // **The order matters.** Asked after `init()`, the unattached state is never seen.
  want("currentDevice() is null before init", currentDevice() === null);

  const first = await probe();
  want("probe() finds an adapter", first.ok,
    first.ok ? "" : `${first.why}: ${first.message}`);
  want("probe() gives the adapter's name",
    first.ok && first.adapter.length > 0, first.ok ? first.adapter : "");
  want("isAvailable() is true", await isAvailable());

  // Does a reason come out when something absent is asked for. An environment where it
  // genuinely is absent cannot be built here, so a software adapter is forced and all that
  // is looked at is **whether the same path returns an adapter.**
  const fallback = await probe({ forceFallbackAdapter: true });
  want("asking for a fallback adapter gives an answer with a reason",
    fallback.ok || fallback.why === "no-adapter",
    fallback.ok ? fallback.adapter : fallback.why);

  // **`software` is the field a caller acts on, and the name is trivia.**
  // Knowing that `swiftshader`, `llvmpipe` and `lavapipe` mean the CPU is knowledge
  // this library has and its callers should not need — three files were each keeping
  // their own copy of that list before it moved here.
  want("probe() says whether the adapter is software — a GPU is not",
    first.ok && first.software === false, first.ok ? String(first.software) : "");
  want("probe() says so when the software adapter was asked for",
    !fallback.ok || fallback.software === true,
    fallback.ok ? `${fallback.adapter} → software=${fallback.software}` : fallback.why);
  want("and the two answers differ, so the field is reading the adapter",
    !fallback.ok || first.ok && first.software !== fallback.software);

  // Called in the form the README writes down — code in a document rots unless it runs.
  await init({ powerPreference: "high-performance" });
  want("currentDevice() is webgpu after init", currentDevice() === "webgpu");
  want("the device is alive", device().alive && device().lost === null);

  // ── Placement ───────────────────────────────────────────────────────
  const values = [1, 2, 3, 4];
  const g = Tensor.from(values, [2, 2]);
  want("the default placement is webgpu", g.device === "webgpu");

  const c = await g.cpu();
  want("after cpu() it is on the cpu", c.device === "cpu");
  want("cpu() carries the values unchanged",
    same(await c.toArray(), Float32Array.from(values)));
  want("a cpu tensor keeps its shape and dtype too",
    c.shape.length === 2 && c.shape[0] === 2 && c.dtype === g.dtype);

  const one = await Tensor.from([7], [1]).cpu();
  want("item() runs on a cpu tensor", (await one.item()) === 7);
  want("repr() runs on a cpu tensor", (await one.repr()).includes("7"));

  // ── Devices that have parted throw ──────────────────────────────────
  wantThrow("an operation on a cpu tensor stops with torch's wording", CROSS_DEVICE,
    () => c.sum());
  wantThrow("mixing it with a gpu tensor stops too", CROSS_DEVICE, () => g.add(c));
  wantThrow("the cpu tensor on the left stops too", CROSS_DEVICE, () => c.add(g));

  // ── Bringing it back up ─────────────────────────────────────────────
  const back = c.webgpu();
  want("after webgpu() it is on the gpu", back.device === "webgpu");
  want("arithmetic runs again on what came back", (await back.sum().item()) === 10);
  want("the values survive the round trip",
    same(await back.toArray(), Float32Array.from(values)));

  // Already in place, it does nothing. Another round trip would be waste.
  want("cpu() is harmless on a cpu tensor", (await c.cpu()) === c);
  want("webgpu() is harmless on a gpu tensor", g.webgpu() === g);

  // ── Put on the host from the start ──────────────────────────────────
  const source = Float32Array.from([9, 8]);
  const host = Tensor.from(source, [2], { device: "cpu" });
  want("made with device: 'cpu' it is on the cpu", host.device === "cpu");
  source[0] = 0;
  want("changing the array afterwards does not change the tensor",
    (await host.toArray())[0] === 9);
  const copy = await host.toArray();
  copy[1] = 0;
  want("toArray() hands back a copy", (await host.toArray())[1] === 8);

  // A scope handles GPU buffers only. `keepAlive(await t.cpu())` must not catch on the
  // guard.
  want("keepAlive() does not refuse a cpu tensor", keepAlive(c) === c);
  let scoped: Tensor | null = null;
  await scope(async () => { scoped = await g.cpu(); });
  want("a cpu tensor still reads after leaving the scope",
    scoped !== null && same(await (scoped as Tensor).toArray(),
      Float32Array.from(values)));

  // ── Gradients ───────────────────────────────────────────────────────
  const leaf = Tensor.from([1, 2], [2], { requiresGrad: true });
  const dropped = await leaf.cpu();
  want("cpu() cuts the graph", !dropped.requiresGrad);

  // ── Synchronising ───────────────────────────────────────────────────
  // It has to be possible to wait for completion without reading a value. Without this
  // the bench mixes the readback into its measurement.
  const before = device().submits;
  Tensor.from([1, 2, 3], [3]).sum();
  await device().synchronize();
  want("synchronize() sends what piled up and waits", device().submits > before);

  const failed = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push(
    failed.length === 0
      ? `all ${checks.length} device-handling checks passed`
      : `**${failed.length} failed** / ${checks.length}`,
  );
  return { text: lines.join("\n"), checks };
}
