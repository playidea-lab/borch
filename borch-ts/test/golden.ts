/**
 * 골든 러너 — `tests/golden.json` 의 답과 borch.ts 를 대조한다.
 *
 * 진짜 PyTorch 로 굳힌 기대값이고, 파이썬 두 구현이 이미 이것으로 대조된다.
 * 셋째 구현이 **첫 커밋부터** 같은 표를 본다.
 *
 * ## 통과보다 먼저 세는 것
 *
 * 이 러너는 "몇 건 통과"보다 앞에 **"몇 건을 아예 안 물었는가"** 를 놓는다.
 * 케이스 이름을 하나 잘못 적으면 그 케이스는 조용히 안 돌고, 남은 것만 통과하면
 * 화면에는 초록색이 뜬다. 이 저장소가 이번 주에 잡은 결함 여덟 개 중 다수가
 * 정확히 그 모양이었다 — 값은 그럴듯하고 아무도 안 보고 있었다.
 */

import { init } from "../src/tensor.js";
import { cases as registered, Inputs } from "./cases.js";

interface GoldenValue {
  kind: "float" | "int" | "bool" | "string";
  shape?: number[];
  values?: (number | boolean | null)[];
  nonfinite?: number[];
  value?: string;
}

interface GoldenDoc {
  tolerance: { atol: number; rtol: number };
  manifest: string;
  /** 케이스가 함께 쓰는 입력. 이름이 `golden_inputs()` 의 키와 같다. */
  inputs: Record<string, GoldenValue>;
  cases: Record<string, GoldenValue>;
}

export interface Failure {
  name: string;
  why: string;
}

export interface Report {
  /** 골든이 들고 있는 전체 케이스 수. */
  total: number;
  /** 우리가 TS 로 쓴 케이스 수. */
  registered: number;
  passed: number;
  failed: Failure[];
  /** **등록했는데 골든에 없는 이름.** 오타이거나 골든이 낡았다. */
  unknown: string[];
  manifest: string;
}

function describe(shape: readonly number[]): string {
  return shape.length === 0 ? "스칼라" : `[${shape}]`;
}

/** 값 하나를 견준다. 허용 오차는 골든이 들고 온 것을 쓴다 — 우리가 정하지 않는다. */
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
    return `문자열 케이스인데 텐서로 답했다 (기대: ${want.value})`;
  }
  const shape = want.shape ?? [];
  const values = want.values ?? [];
  if (got.length !== values.length) {
    return `원소 수가 다르다: ${got.length} vs ${values.length} ` +
      `(우리 ${describe(gotShape)}, 골든 ${describe(shape)})`;
  }
  if (gotShape.length !== shape.length ||
      gotShape.some((d, i) => d !== shape[i])) {
    // 모양이 다른데 원소 수만 같은 것은 **조용히 틀리는 쪽**이다. 값을 보기 전에 잡는다.
    return `모양이 다르다: 우리 ${describe(gotShape)}, 골든 ${describe(shape)}`;
  }
  const nonfinite = new Set(want.nonfinite ?? []);
  for (let i = 0; i < values.length; i++) {
    const raw = values[i];
    if (raw === undefined) return `[${i}] 골든에 값이 비어 있다 — 파일이 깨졌다`;
    // 비유한값은 JSON 에 null 로 적히고 자리만 따로 온다. NaN 인지 ±inf 인지는
    // JSON 이 구분해 담지 못하므로, 여기서는 "유한하지 않다"까지만 견준다.
    if (nonfinite.has(i) || raw === null) {
      const g = got[i] ?? Number.NaN;
      if (Number.isFinite(g)) return `[${i}] 유한하지 않아야 하는데 ${g} 다`;
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
 * @param url `golden.json` 의 자리. 페이지에서 상대 경로로 준다 — 호스트를 안 박는다.
 */
export async function run(url: string): Promise<Report> {
  const res = await fetch(url);
  // fetch 는 404 에 던지지 않는다. 확인 안 하면 HTML 을 JSON 으로 읽다가
  // 엉뚱한 자리에서 터진다 — 이 저장소의 브라우저 러너에서 이미 한 번 밟았다.
  if (!res.ok) throw new Error(`골든을 못 받았다: ${res.status} ${url}`);
  const doc = (await res.json()) as GoldenDoc;

  await init();
  // **입력은 골든에서 받는다.** 여기서 배열을 다시 적으면 그 자리가 틀릴 자리가 되고,
  // 틀려도 "우리 값이 다르다" 로만 보인다. 굳힐 때 쓴 바로 그 숫자를 쓴다.
  const table = registered(new Inputs(doc.inputs));
  const report: Report = {
    total: Object.keys(doc.cases).length,
    registered: table.size,
    passed: 0,
    failed: [],
    unknown: [],
    manifest: doc.manifest,
  };

  for (const [name, body] of table) {
    const want = doc.cases[name];
    if (!want) {
      report.unknown.push(name);
      continue;
    }
    let why: string | null;
    try {
      const result = await body();
      if (typeof result === "string") {
        // 값이 아니라 **판정**을 굳힌 케이스다 — `equal` 이 참인가, 어떤 예외가
        // 나는가 같은 것. 이런 것은 근사가 아니라 정확히 같아야 한다.
        why = want.kind === "string"
          ? (result === want.value ? null : `"${result}" ≠ "${want.value}"`)
          : `문자열로 답했는데 골든은 ${want.kind} 다`;
      } else {
        const got = await result.toArray();
        why = compare(got, result.shape, want, doc.tolerance);
      }
    } catch (err) {
      why = `던졌다: ${err instanceof Error ? err.message : String(err)}`;
    }
    if (why === null) report.passed += 1;
    else report.failed.push({ name, why });
  }
  return report;
}

/** 사람이 읽을 줄로. 통과 수보다 **안 물은 수**를 먼저 적는다. */
export function format(report: Report): string {
  const lines: string[] = [];
  const gap = report.total - report.registered;
  lines.push(`골든 ${report.total}건 중 ${report.registered}건을 TS 로 썼다 — ` +
    `${gap}건은 아직 안 물었다.`);
  if (report.unknown.length > 0) {
    lines.push(`**이름이 안 맞는 것 ${report.unknown.length}건** ` +
      "(등록했는데 골든에 없다 — 오타이거나 골든이 낡았다):");
    for (const n of report.unknown) lines.push(`  ? ${n}`);
  }
  lines.push(`통과 ${report.passed} / 실패 ${report.failed.length}`);
  for (const f of report.failed) lines.push(`  ✘ ${f.name} — ${f.why}`);
  return lines.join("\n");
}
