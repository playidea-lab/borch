/**
 * 장치 관리 — 협상·가용성·배치를 브라우저에서 확인한다.
 *
 * **골든이 이것을 안 잡는다.** 골든은 값이 torch 와 같은지를 묻는 장치이고, 여기서
 * 묻는 것은 값이 아니라 *어디에 있는가* 와 *없을 때 뭐라고 하는가* 다. 둘은 다른
 * 물음이라 러너를 따로 둔다.
 *
 * 소프트웨어 어댑터에서도 막지 않는다 — 배치는 어느 어댑터에서나 같은 규칙이다.
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
 * 이 보고의 정본은 `checks` 다. `text` 는 사람이 읽는 그림자다.
 *
 * **러너가 문장을 훑어 통과를 판정하고 있었다.** 그 방식은 문구가 바뀌면 조용히
 * 답을 바꾸고, `readme.py` 에서는 실제로 그랬다 — 두 예시 중 하나만 통과해도
 * 찾던 낱말이 다른 줄에 남아 있어 0 을 냈다. 상태를 그대로 넘기면 러너가 셀 수
 * 있고, 무엇이 실패했는지도 제 입으로 말한다.
 */
export interface Report { text: string; checks: Check[] }

const checks: Check[] = [];

function want(name: string, ok: boolean, note = ""): void {
  checks.push({ name, ok, note });
}

/** 던져야 하는 자리. **안 던지는 것이 실패다** — 조용히 지나가면 값이 틀린다. */
function wantThrow(name: string, fragment: string, body: () => unknown): void {
  try {
    body();
    want(name, false, "안 던졌다 — 조용히 지나갔다");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    want(name, message.includes(fragment), message.includes(fragment)
      ? "" : `문구가 다르다: ${message}`);
  }
}

function same(a: Float32Array, b: Float32Array): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export async function report(): Promise<Report> {
  // ── 붙기 전 ────────────────────────────────────────────────────────────
  // **순서가 중요하다.** `init()` 뒤에 물으면 안 붙은 상태를 영영 못 본다.
  want("init 전 currentDevice() 는 null", currentDevice() === null);

  const first = await probe();
  want("probe() 가 어댑터를 찾는다", first.ok,
    first.ok ? "" : `${first.why}: ${first.message}`);
  want("probe() 가 어댑터 이름을 준다",
    first.ok && first.adapter.length > 0, first.ok ? first.adapter : "");
  want("isAvailable() 이 참", await isAvailable());

  // 없는 것을 물었을 때 이유가 나오는가. 진짜로 없는 환경을 만들 수 없으므로
  // 소프트웨어 어댑터를 강제해 **같은 경로가 어댑터를 돌려주는지**만 본다.
  const fallback = await probe({ forceFallbackAdapter: true });
  want("폴백 어댑터 요청이 이유 있는 답을 준다",
    fallback.ok || fallback.why === "no-adapter",
    fallback.ok ? fallback.adapter : fallback.why);

  // README 가 적어 놓은 형태 그대로 부른다 — 문서의 코드는 안 돌리면 썩는다.
  await init({ powerPreference: "high-performance" });
  want("init 후 currentDevice() 는 webgpu", currentDevice() === "webgpu");
  want("장치가 살아 있다", device().alive && device().lost === null);

  // ── 배치 ──────────────────────────────────────────────────────────────
  const values = [1, 2, 3, 4];
  const g = Tensor.from(values, [2, 2]);
  want("기본 배치는 webgpu", g.device === "webgpu");

  const c = await g.cpu();
  want("cpu() 뒤에는 cpu", c.device === "cpu");
  want("cpu() 가 값을 그대로 옮긴다",
    same(await c.toArray(), Float32Array.from(values)));
  want("cpu 텐서도 모양과 형을 지킨다",
    c.shape.length === 2 && c.shape[0] === 2 && c.dtype === g.dtype);

  const one = await Tensor.from([7], [1]).cpu();
  want("cpu 텐서에서 item() 이 돈다", (await one.item()) === 7);
  want("cpu 텐서에서 repr() 이 돈다", (await one.repr()).includes("7"));

  // ── 갈린 장치는 던진다 ────────────────────────────────────────────────
  wantThrow("cpu 텐서에 연산을 걸면 torch 문구로 멈춘다", CROSS_DEVICE,
    () => c.sum());
  wantThrow("gpu 텐서와 섞어도 멈춘다", CROSS_DEVICE, () => g.add(c));
  wantThrow("cpu 텐서가 왼쪽이어도 멈춘다", CROSS_DEVICE, () => c.add(g));

  // ── 되올리기 ──────────────────────────────────────────────────────────
  const back = c.webgpu();
  want("webgpu() 뒤에는 webgpu", back.device === "webgpu");
  want("되올린 값으로 연산이 다시 돈다", (await back.sum().item()) === 10);
  want("왕복해도 값이 같다",
    same(await back.toArray(), Float32Array.from(values)));

  // 이미 그 자리에 있으면 아무 일도 안 한다. 왕복을 한 번 더 도는 것은 낭비다.
  want("cpu() 는 cpu 텐서에 무해", (await c.cpu()) === c);
  want("webgpu() 는 gpu 텐서에 무해", g.webgpu() === g);

  // ── 처음부터 호스트에 두기 ────────────────────────────────────────────
  const source = Float32Array.from([9, 8]);
  const host = Tensor.from(source, [2], { device: "cpu" });
  want("device: 'cpu' 로 만들면 cpu", host.device === "cpu");
  source[0] = 0;
  want("넘긴 배열을 나중에 고쳐도 텐서는 안 바뀐다",
    (await host.toArray())[0] === 9);
  const copy = await host.toArray();
  copy[1] = 0;
  want("toArray() 가 사본을 준다", (await host.toArray())[1] === 8);

  // 구역은 GPU 버퍼만 다룬다. `keepAlive(await t.cpu())` 가 가드에 걸리면 안 된다.
  want("keepAlive() 가 cpu 텐서를 거절하지 않는다", keepAlive(c) === c);
  let scoped: Tensor | null = null;
  await scope(async () => { scoped = await g.cpu(); });
  want("구역을 나가도 cpu 텐서는 읽힌다",
    scoped !== null && same(await (scoped as Tensor).toArray(),
      Float32Array.from(values)));

  // ── 기울기 ────────────────────────────────────────────────────────────
  const leaf = Tensor.from([1, 2], [2], { requiresGrad: true });
  const dropped = await leaf.cpu();
  want("cpu() 는 그래프를 끊는다", !dropped.requiresGrad);

  // ── 동기화 ────────────────────────────────────────────────────────────
  // 값을 안 읽고 완료를 기다릴 수 있어야 한다. 이것이 없으면 벤치가 readback 을
  // 측정에 섞는다.
  const before = device().submits;
  Tensor.from([1, 2, 3], [3]).sum();
  await device().synchronize();
  want("synchronize() 가 쌓인 것을 보내고 기다린다", device().submits > before);

  const failed = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push(
    failed.length === 0
      ? `장치 관리 ${checks.length}건 전부 통과`
      : `**${failed.length}건 실패** / ${checks.length}건`,
  );
  return { text: lines.join("\n"), checks };
}
