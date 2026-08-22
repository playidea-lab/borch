/**
 * The golden runner — borch.ts against the answers in `tests/golden.json`.
 *
 * The expectations were frozen with real PyTorch, and two Python implementations are
 * already held against them. A third implementation sees the same table **from its first
 * commit.**
 *
 * ## What is counted before the passes
 *
 * This runner puts **"how many were never asked at all"** ahead of "how many passed".
 * Write one case name wrong and that case quietly does not run, and if the rest pass the
 * screen goes green. Most of the eight defects this repository caught in a week had
 * exactly that shape — the values were plausible and nobody was looking.
 */

import { Device } from "../src/device.js";
import { init } from "../src/tensor.js";
import { cases as registered, Inputs } from "./cases.js";

interface GoldenValue {
  kind: "float" | "int" | "bool" | "string";
  shape?: number[];
  values?: (number | boolean | null)[];
  /** `[index, kind]` pairs. Older files carry the index alone, so both are taken. */
  nonfinite?: (number | [number, string])[];
  value?: string;
}

interface GoldenDoc {
  tolerance: { atol: number; rtol: number };
  manifest: string;
  /** The inputs the cases share. The names are `golden_inputs()`'s keys. */
  inputs: Record<string, GoldenValue>;
  cases: Record<string, GoldenValue>;
}

export interface Failure {
  name: string;
  why: string;
}

export interface Report {
  /** How many cases the golden holds in all. */
  total: number;
  /**
   * How many cases are written in TS **and are in the golden too.**
   *
   * This used to be the size of the table. With that, holding seven names the golden does
   * not have while leaving seven of the golden's own unwritten comes out as **"859 of 859,
   * 0 left"** — the counts match, so it reconciles. That state really was read as a pass
   * once. Where what is counted differs from what was asked, the instrument is worse than
   * no instrument.
   */
  registered: number;
  passed: number;
  failed: Failure[];
  /** **Names registered that the golden does not have.** A typo, or a stale golden. */
  unknown: string[];
  /** The names actually asked. Counting what is left wants names, not a number. */
  asked: string[];
  /**
   * Which adapter it ran on.
   *
   * The device does not change the values, so it has no bearing on passing. It is written
   * down because a headless browser will sometimes hand over a software adapter, and not
   * knowing that is how **whoever measures performance falls into the same trap** — which
   * is what happened in this repository.
   */
  adapter: string;
  manifest: string;
}

function describe(shape: readonly number[]): string {
  return shape.length === 0 ? "scalar" : `[${shape}]`;
}

/** Weigh one value. The tolerance is the one the golden brought — we do not set it. */
function close(got: number, want: number, atol: number, rtol: number): boolean {
  if (Number.isNaN(want)) return Number.isNaN(got);
  if (!Number.isFinite(want)) return got === want;
  return Math.abs(got - want) <= atol + rtol * Math.abs(want);
}

function compare(
  got: Float32Array,
  gotShape: readonly number[],
  want: GoldenValue,
  tol: { atol: number; rtol: number },
): string | null {
  if (want.kind === "string") {
    return `a string case answered with a tensor (expected: ${want.value})`;
  }
  const shape = want.shape ?? [];
  const values = want.values ?? [];
  if (got.length !== values.length) {
    return `different number of elements: ${got.length} vs ${values.length} ` +
      `(ours ${describe(gotShape)}, golden ${describe(shape)})`;
  }
  if (gotShape.length !== shape.length ||
      gotShape.some((d, i) => d !== shape[i])) {
    // A different shape with the same element count is **the quietly wrong kind.**
    // Caught before the values are looked at.
    return `different shape: ours ${describe(gotShape)}, golden ${describe(shape)}`;
  }
  // **The kind comes too.** For a long time only the index rode along, so this weighed
  // no further than "it is not finite" — and then `NaN`, `+inf` and `-inf` all pass as
  // each other. The hole came out when a case arrived with an infinity in its answer.
  const nonfinite = new Map<number, string>(
    (want.nonfinite ?? []).map((e) =>
      Array.isArray(e) ? [e[0] as number, e[1] as string] : [e as number, "?"]),
  );
  for (let i = 0; i < values.length; i++) {
    const raw = values[i];
    if (raw === undefined) return `[${i}] the golden's value is empty — the file is broken`;
    if (nonfinite.has(i) || raw === null) {
      const g = got[i] ?? Number.NaN;
      if (Number.isFinite(g)) return `[${i}] should not be finite and is ${g}`;
      const kind = nonfinite.get(i) ?? "?";
      const mine = Number.isNaN(g) ? "nan" : (g > 0 ? "inf" : "-inf");
      if (kind !== "?" && kind !== mine) return `[${i}] is ${mine} and should be ${kind}`;
      continue;
    }
    const want1 = typeof raw === "boolean" ? (raw ? 1 : 0) : raw;
    const g = got[i] ?? Number.NaN;
    if (!close(g, want1, tol.atol, tol.rtol)) {
      return `[${i}] ${g} ≠ ${want1}`;
    }
  }
  return null;
}

/**
 * @param url where `golden.json` is. The page passes a relative path — no host is nailed in.
 */
export async function run(url: string): Promise<Report> {
  const res = await fetch(url);
  // fetch does not throw on a 404. Unchecked, HTML gets read as JSON and it blows up
  // somewhere else entirely — a browser runner in this repository walked into that once.
  if (!res.ok) throw new Error(`could not fetch the golden: ${res.status} ${url}`);
  const doc = (await res.json()) as GoldenDoc;

  await init();
  // **The inputs come from the golden.** Write the arrays out again here and that becomes
  // a place that can be wrong, and being wrong it only ever looks like "our value
  // differs". The very numbers used at freezing time are the ones used.
  const table = registered(new Inputs(doc.inputs));
  const report: Report = {
    total: Object.keys(doc.cases).length,
    registered: 0,                  // only what the loop below actually asks is counted
    passed: 0,
    failed: [],
    unknown: [],
    asked: [],
    adapter: Device.adapterInfo,
    manifest: doc.manifest,
  };

  for (const [name, body] of table) {
    const want = doc.cases[name];
    if (!want) {
      report.unknown.push(name);
      continue;
    }
    report.asked.push(name);
    report.registered += 1;
    // **Write the start down first.** Every case is try/caught, so an exception rides out
    // in the report. What does not ride out is a case that never finishes, and then
    // nothing is left on screen — with this line, the last name printed is the culprit.
    // Seen with `run.py --verbose`.
    //
    // **The prefix is a contract with `run.py`**, which watches for it to name the case
    // that hung. Change one side alone and the trace goes quietly empty and nothing fails;
    // `test_messages.py` holds the two spellings together.
    console.debug(`[golden] ${name}`);
    let why: string | null;
    try {
      const result = await body();
      if (typeof result === "string") {
        // A case that froze a **judgement** rather than a value — whether `equal` is
        // true, which exception comes out. These have to be exactly equal, not near.
        why = want.kind === "string"
          ? (result === want.value ? null : `"${result}" ≠ "${want.value}"`)
          : `answered with a string where the golden is ${want.kind}`;
      } else {
        const got = await result.toArray();
        why = compare(got, result.shape, want, doc.tolerance);
      }
    } catch (err) {
      why = `it threw: ${err instanceof Error ? err.message : String(err)}`;
    }
    if (why === null) report.passed += 1;
    else report.failed.push({ name, why });
  }
  return report;
}

/** As lines for a person. **What was never asked** is written before the passes. */
export function format(report: Report): string {
  const lines: string[] = [];
  const gap = report.total - report.registered;
  lines.push(`adapter: ${report.adapter}`);
  lines.push(`${report.registered} of the golden's ${report.total} are written in TS — ` +
    `${gap} have not been asked yet.`);
  if (report.unknown.length > 0) {
    lines.push(`**${report.unknown.length} names do not match** ` +
      "(registered and not in the golden — a typo, or a stale golden):");
    for (const n of report.unknown) lines.push(`  ? ${n}`);
  }
  lines.push(`passed ${report.passed} / failed ${report.failed.length}`);
  for (const f of report.failed) lines.push(`  ✘ ${f.name} — ${f.why}`);
  return lines.join("\n");
}
