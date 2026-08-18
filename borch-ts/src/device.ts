/**
 * WebGPU 장치·버퍼·파이프라인 캐시.
 *
 * ## 파이프라인 캐시가 왜 여기 있나
 *
 * 모양을 셰이더에 굽는 것이 이 라이브러리의 전제다(`kernels.ts` 참고). 그러면 같은
 * 연산도 모양이 다르면 다른 셰이더이고, 층을 한 번 지나갈 때마다 컴파일하면 커널이
 * 빠른 것이 의미가 없다. 그래서 **모양 서명 → 파이프라인** 이 자료구조로 들어간다.
 * 최적화가 아니라 굽기로 얻은 속도를 지키는 값이다.
 *
 * ## 한계는 조용히 온다
 *
 * WebGPU 는 버퍼 크기나 dispatch 한계를 넘으면 **던지지 않고 안 한다.** 벤치에서
 * 두 번 밟았다 — 128MB 초과에서 24만 GFLOPS, dispatch 65,535 초과에서 "144%".
 * 둘 다 값을 안 봤으면 믿었을 수치라, 여기서는 한계를 **미리 재고 넘으면 던진다.**
 */

import { grid1d, reduceParts, reduceSum, WORKGROUP } from "./kernels.js";

const BYTES_PER_F32 = 4;

/** 검증 오류를 몇 건까지 찍을 것인가. 첫 건이 원인이고 나머지는 그 여파다. */
const MAX_REPORTED_ERRORS = 3;

/**
 * 텐서가 어디에 있는가.
 *
 * torch 의 `'cuda'`·`'cpu'` 자리다. **인덱스가 없다** — WebGPU 는 어댑터를 열거하는
 * 방법을 주지 않으므로 `'webgpu:1'` 이 가리킬 대상이 아예 없다. 문자열로 충분하다.
 */
export type DeviceKind = "webgpu" | "cpu";

/** 어댑터를 어떻게 고를 것인가. torch 의 `CUDA_VISIBLE_DEVICES` 자리다. */
export interface InitOptions {
  /**
   * 기본값이 `"high-performance"` 인 것은 이것이 **재는 라이브러리**이기 때문이다.
   * 브라우저 기본값은 노트북에서 통합 GPU 를 고를 수 있고, 그러면 같은 코드가 같은
   * 기계에서 다른 수를 낸다 — 무엇을 쟀는지 모르는 수가 나온다.
   */
  powerPreference?: GPUPowerPreference;
  /** 소프트웨어 어댑터를 강제한다. 폴백 경로 자체를 시험할 때 쓴다. */
  forceFallbackAdapter?: boolean;
}

/**
 * WebGPU 를 쓸 수 있는가. **예외가 아니라 값으로 답한다.**
 *
 * `why` 가 값어치다 — `no-api`(브라우저가 낡았거나 안전한 문맥이 아니다)와
 * `no-adapter`(드라이버 차단 목록·가상 머신·GPU 없는 헤드리스)는 쓰는 사람이 할 수
 * 있는 일이 전혀 다른데, 예외 하나로 뭉치면 그 갈림이 사라진다.
 */
export type Availability =
  | { ok: true; adapter: string }
  | { ok: false; why: "no-api" | "no-adapter"; message: string };

const NO_API =
  "WebGPU 가 없다. Chrome/Edge 113+ 또는 Safari 18+ 가 필요하고, " +
  "리눅스에서는 플래그가 필요할 수 있다. https 또는 localhost 여야 한다.";

const NO_ADAPTER =
  "WebGPU 어댑터를 못 얻었다 — 드라이버 차단 목록, 가상 머신, 또는 GPU 가 없는 " +
  "헤드리스 환경일 수 있다.";

/** 어느 어댑터인지 한 줄로. 빈 칸은 뺀다 — 브라우저가 대부분을 가린다. */
function describe(adapter: GPUAdapter): string {
  const info: Partial<GPUAdapterInfo> = adapter.info ?? {};
  return [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean).join(" / ") || "(알 수 없음)";
}

function askAdapter(options: InitOptions): Promise<GPUAdapter | null> {
  return navigator.gpu.requestAdapter({
    powerPreference: options.powerPreference ?? "high-performance",
    forceFallbackAdapter: options.forceFallbackAdapter ?? false,
  });
}

/**
 * 붙어 보지 않고 붙을 수 있는지만 묻는다. 장치는 안 만든다.
 *
 * **`init()` 을 대신하지 않는다** — 여기를 지나도 `requestDevice` 가 거절할 수 있고
 * 그것은 여전히 `init()` 에서 예외로 온다. 이 함수가 답하는 것은 "어댑터가 있는가"
 * 까지이고, 실제로 막히는 자리가 대부분 그 앞이라 그것만으로 값이 있다.
 */
export async function probe(options: InitOptions = {}): Promise<Availability> {
  if (!("gpu" in navigator)) return { ok: false, why: "no-api", message: NO_API };
  const adapter = await askAdapter(options);
  if (!adapter) return { ok: false, why: "no-adapter", message: NO_ADAPTER };
  return { ok: true, adapter: describe(adapter) };
}

/**
 * WebGPU 를 쓸 수 있는가만 묻는다. `torch.cuda.is_available()` 자리다.
 *
 * **torch 와 달리 비동기다** — 어댑터를 얻는 것이 비동기라 피할 길이 없다. 왜 안
 * 되는지가 필요하면 `probe()` 를 써라.
 */
export async function isAvailable(options: InitOptions = {}): Promise<boolean> {
  return (await probe(options)).ok;
}

/** 오류 메시지가 가리키는 줄을 찾을 수 있게 셰이더에 번호를 붙인다. */
function numbered(code: string): string {
  return code
    .split("\n")
    .map((line, i) => `${String(i + 1).padStart(3)} | ${line}`)
    .join("\n");
}

export class Device {
  private readonly device: GPUDevice;
  private readonly limits: GPUSupportedLimits;
  /** 모양까지 포함한 서명 → 파이프라인. */
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  /**
   * 파이프라인 → 바인드 그룹 배치.
   *
   * `getBindGroupLayout` 은 부를 때마다 **새 객체를 만든다** — 사양이 캐시를 약속하지
   * 않는다. dispatch 마다 부르면 그만큼 만들고 버리는 것이고, 스텝당 칠백 번이다.
   */
  private readonly layouts = new WeakMap<GPUComputePipeline, GPUBindGroupLayout>();
  /**
   * 읽어올 때 쓰는 staging 버퍼의 **놀고 있는 것들**. 크기별로 여러 개다.
   *
   * 처음에는 크기마다 하나만 두고 돌려썼는데, 읽기 둘이 겹치면 같은 버퍼를 두 번
   * 매핑하게 되어 "Buffer already has an outstanding map pending" 으로 터졌다.
   * `equal` 이 두 텐서를 `Promise.all` 로 읽자마자 나왔다 — 겹쳐 읽는 것은 흔한 일이라
   * 하나로는 안 된다.
   */
  private readonly stagingFree = new Map<number, GPUBuffer[]>();

  private constructor(device: GPUDevice) {
    this.device = device;
    this.limits = device.limits;
  }

  static async create(options: InitOptions = {}): Promise<Device> {
    if (!("gpu" in navigator)) throw new Error(NO_API);
    const adapter = await askAdapter(options);
    if (!adapter) throw new Error(NO_ADAPTER);
    // **어느 장치인지 알아야 잰 수가 뜻을 갖는다.** 헤드리스 브라우저는 진짜 GPU 대신
    // 소프트웨어 어댑터를 주는 일이 있고, 그것도 어댑터라 예외가 안 난다 — 그러면
    // 벽시계는 멀쩡히 돌고 "느리다" 는 결론만 남는다. 재는 쪽이 이것을 봐야 한다.
    Device.adapterInfo = describe(adapter);
    Device.adapterFeatures = [...adapter.features].sort().join(" ");
    // 기본 한계를 그대로 쓰지 않고 어댑터가 주는 최대치를 요청한다. 기본
    // maxStorageBufferBindingSize 는 128MB 이고, 그 위에서 조용히 틀린 답이 나온다.
    const want: Record<string, number> = {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    };
    // **`timestamp-query` 는 있으면 받아 둔다.** 요청해 두어도 안 켜면 비용이 없고,
    // 나중에 켜려면 장치를 다시 만들어야 한다 — 재는 사람이 그 시점에 그것을 알 수
    // 없다. 없는 어댑터에서 요청하면 `requestDevice` 가 거절하므로 있을 때만 넣는다.
    const canTime = adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice({
      requiredLimits: want,
      requiredFeatures: canTime ? ["timestamp-query"] : [],
    });
    // 검증 오류도 예외로 안 온다. 붙잡지 않으면 잘못 만든 파이프라인이 조용히
    // 아무것도 안 하고, 그 결과를 우리는 "값이 틀렸다" 로만 보게 된다.
    //
    // 처음 것들만 낸다 — 셰이더 하나가 깨지면 그 뒤 dispatch 마다 같은 오류가 다시
    // 나서 진짜 원인(첫 줄)이 스크롤 밖으로 밀린다. 실측으로 그렇게 됐다.
    // **줄이는 것이지 삼키는 것이 아니다.** 몇 건을 접었는지는 마지막에 적는다.
    //
    // **세어서 밖으로 내보낸다.** 찍기만 하면 재는 쪽이 그것을 못 본다 — ResNet 벤치가
    // 무효한 명령 버퍼를 안고도 ms/step 을 냈고, 그 수는 측정이 아니라 학습이 안 되는
    // 상태의 벽시계였다. 재는 쪽이 이 수를 보고 결과를 거절할 수 있어야 한다.
    const made = new Device(device);
    const seen = made.faults;
    device.addEventListener("uncapturederror", (event) => {
      seen.count += 1;
      const err = (event as GPUUncapturedErrorEvent).error;
      if (seen.first === "") seen.first = err.message;
      if (seen.count <= MAX_REPORTED_ERRORS) {
        console.error(`[borch.ts] WebGPU 검증 오류 ${seen.count}: ${err.message}`);
      } else if (seen.count === MAX_REPORTED_ERRORS + 1) {
        console.error(
          `[borch.ts] 검증 오류가 ${MAX_REPORTED_ERRORS} 건을 넘었다 — ` +
            "이후는 안 찍는다. 원인은 위의 첫 건이다.",
        );
      }
    });
    device.lost
      .then((info) => {
        // **찍기만 하면 안 된다.** 장치를 잃으면 그 뒤의 모든 텐서와 모든 수가 뜻을
        // 잃는데, 로그는 재는 쪽이 읽지 않는다 — `faults` 를 밖으로 내보낸 것과 같은
        // 이유로 이것도 물어볼 수 있는 상태여야 한다. 그래야 벤치가 결과를 거절한다.
        made.lost = { reason: String(info.reason), message: info.message };
        console.error(`[borch.ts] WebGPU 장치를 잃었다: ${info.reason} — ${info.message}`);
      })
      .catch(() => {
        /* lost 는 거절되지 않지만, 거절되더라도 여기서 더 할 일이 없다 */
      });
    return made;
  }

  /**
   * 장치를 잃었으면 그 사연, 아니면 `null`.
   *
   * torch 에 대응물이 없다 — CUDA 문맥은 프로세스와 함께 산다. 브라우저에서는 다른
   * 탭이나 드라이버 사정으로 남이 우리 장치를 회수할 수 있고, 그때 예외는 안 난다.
   */
  lost: { reason: string; message: string } | null = null;

  /** 아직 쓸 수 있는가. 긴 학습 루프가 스텝마다 볼 자리다. */
  get alive(): boolean {
    return this.lost === null;
  }

  /**
   * 지금까지 난 검증 오류.
   *
   * **재는 쪽이 이것을 봐야 한다.** 무효한 명령 버퍼는 예외를 안 던지고 그냥 아무것도
   * 안 하므로, 그 상태에서도 벽시계는 돌고 수는 나온다 — 측정처럼 보이는 것이 나온다.
   */
  faults: { count: number; first: string } = { count: 0, first: "" };

  /**
   * 지금까지 보낸 dispatch 수.
   *
   * "느리다" 에서 멈추면 다음에 무엇을 할지 모른다. 스텝당 dispatch 수를 알면 느린
   * 것이 **커널 자체인지 부르는 횟수인지**가 갈린다 — 지금 이 설계는 연산 하나마다
   * 명령 인코더를 새로 만들어 제출하므로, 그 수가 크면 원인이 거기다.
   */
  dispatches = 0;

  /**
   * 커널 종류별 dispatch 수.
   *
   * 총수만으로는 다음에 무엇을 고칠지 모른다. 1,636 개가 conv 스무 번인지 BatchNorm
   * 조립 오백 번인지에 따라 할 일이 다르다 — 그 갈림을 재려고 둔다.
   */
  readonly byKind = new Map<string, number>();

  /** 지금 어느 커널을 부르는지. `pipeline` 이 서명의 앞머리를 여기 남긴다. */
  private current = "?";
  /** 지금 굽는 파이프라인의 **서명 전체**. 프로파일이 이것으로 쌓는다. */
  private currentSig = "?";

  /**
   * 아직 안 보낸 명령들.
   *
   * **연산마다 제출하면 안 된다.** 처음에는 dispatch 마다 명령 인코더를 새로 만들어
   * 제출했는데, 배치를 4 배로 늘렸을 때 시간이 2.1 배밖에 안 늘었다 — 직선으로 맞추면
   * 배치와 무관한 고정비가 스텝당 5.2 초, dispatch 당 7.4ms 였다. 제출 하나에 그런
   * 값이 들 리 없으니 그것이 곧 제출 횟수의 값이었다.
   *
   * 이제 한 인코더에 쌓아 두고 **읽을 때** 한 번 보낸다. WebGPU 는 한 패스 안의
   * dispatch 사이에 장벽을 알아서 넣으므로 순서는 그대로 지켜진다.
   */
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  /** 지금까지 실제로 보낸 제출 수. 묶기가 도는지 재는 쪽이 본다. */
  submits = 0;

  /**
   * 셰이더 컴파일 오류를 삼키지 않는다.
   *
   * **WGSL 컴파일 실패는 예외로 오지 않는다.** `createShaderModule` 은 그냥 돌아오고,
   * 그 파이프라인으로 dispatch 하면 아무 일도 안 일어난다 — 결과 버퍼가 0 인 채로
   * 남고 화면에는 "값이 다르다"만 뜬다. 실제로 이 러너에서 축약 커널이 그렇게
   * 0 을 냈고, 그때 아무 오류도 안 보였다. 그래서 진단 정보를 직접 꺼내 본다.
   */
  pipeline(signature: string, source: () => string): GPUComputePipeline {
    // 서명의 첫 토막이 커널 종류다(`cnt:...`, `u:relu:...`). 모양까지 세면 종류가
    // 수백 개가 되어 어디가 무거운지가 안 보인다.
    this.current = signature.split(":")[0] ?? "?";
    // **프로파일 중에는 서명 전체를 쓴다.** 종류만으로는 "gb 가 94%" 까지밖에 못 가고,
    // 그 다음 물음(어느 규칙·어느 모양인가)에서 막힌다 — 실제로 거기서 막혔다.
    // 켰을 때만 쌓이므로 평소에는 값이 없다.
    this.currentSig = signature;
    const hit = this.pipelines.get(signature);
    if (hit) return hit;
    const code = source();
    const module = this.device.createShaderModule({ code });
    void module.getCompilationInfo().then((info) => {
      for (const m of info.messages) {
        if (m.type !== "error" && m.type !== "warning") continue;
        console.error(
          `[borch.ts] ${signature} 셰이더 ${m.type} ${m.lineNum}:${m.linePos} — ` +
            `${m.message}\n${numbered(code)}`,
        );
      }
    });
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.pipelines.set(signature, pipeline);
    return pipeline;
  }

  /** 지금까지 구운 셰이더 수. 캐시가 도는지 테스트가 본다. */
  get pipelineCount(): number {
    return this.pipelines.size;
  }

  /**
   * 지금 열려 있는 구역들. `alloc` 이 만든 것을 여기 적어 두고 구역이 닫힐 때 놓는다.
   *
   * **없으면 학습이 안 돈다.** ResNet 한 스텝이 중간 버퍼를 수천 개 만드는데, GPU
   * 버퍼는 자바스크립트의 쓰레기 수집이 제때 안 놓아준다 — 손잡이가 사라져도 메모리가
   * 남는다. 자매도 같은 이유로 `scope()` 를 든다.
   */
  private readonly scopes: Set<GPUBuffer>[] = [];
  /** 구역이 닫혀도 살아남는 것 — 파라미터와 옵티마이저 상태다. */
  private readonly kept = new WeakSet<GPUBuffer>();
  /**
   * 놓인 버퍼를 크기별로 되쓴다.
   *
   * `createBuffer` 는 드라이버를 거쳐 GPU 메모리를 잡는 일이고, 학습 한 스텝이 그것을
   * 수백 번 한다. 그런데 **스텝마다 같은 크기가 되풀이된다** — 모양이 매번 같기
   * 때문이다. 파괴하고 다시 만드는 대신 돌려쓰면 그 일이 한 번으로 준다.
   */
  private readonly spare = new Map<number, GPUBuffer[]>();
  /** 버퍼가 실제로 몇 바이트인지. 반납할 때 어느 통에 넣을지가 여기서 나온다. */
  private readonly sizes = new WeakMap<GPUBuffer, number>();

  /**
   * 버퍼가 **몇 번째 삶인가.** 통에 돌아갈 때마다 하나씩 오른다.
   *
   * ## 왜 필요한가 — 실측
   *
   * 구역이 닫힐 때 버퍼는 파괴되지 않고 통에 돌아간다(그게 통이 있는 이유다).
   * 그런데 그 버퍼를 가리키던 텐서가 구역 밖으로 샜으면, 그 텐서는 여전히 같은
   * `GPUBuffer` 를 들고 있고 **다음 할당이 그것을 꺼내 덮어쓴다.**
   *
   * 재봤다. `[1,2,3,4]` 를 담은 텐서를 구역 밖으로 흘리고 같은 크기를 네 번 더
   * 잡은 뒤 읽으니 **`9,9,9,9`** 가 나왔다 — 남의 값이, 예외 없이.
   * 이 저장소의 첫 문장이 "조용히 다른 값을 내느니 시끄럽게 멈춘다" 인데 그 반대가
   * 핵심 학습 루프에서 일어나고 있었다.
   *
   * 텐서는 태어날 때 이 수를 적어 두고, 값에 닿을 때 견준다. 어긋나면 그 텐서는
   * **이미 죽은 것**이고 거기서 멈춘다. 골든은 이것을 못 본다 — 케이스마다 페이지가
   * 깨끗해서 통이 휘저어질 일이 없다.
   */
  private readonly ages = new WeakMap<GPUBuffer, number>();

  /** 이 버퍼의 지금 삶. 텐서가 태어날 때와 쓸 때 이 수를 견준다. */
  age(buffer: GPUBuffer): number {
    return this.ages.get(buffer) ?? 0;
  }

  /**
   * 삶을 하나 올린다 — **이 순간 그 버퍼를 가리키던 텐서는 전부 죽는다.**
   *
   * 통에 실제로 다시 꺼내 쓸 때가 아니라 **돌려놓을 때** 올린다. 꺼낼 때 올리면
   * "아직 아무도 안 가져갔으니 읽히긴 한다" 는 구간이 생기고, 그 구간에서만
   * 통과하는 코드가 나온다 — 재현이 할당 순서에 달린 결함이 그렇게 만들어진다.
   */
  private retire(buffer: GPUBuffer): void {
    this.ages.set(buffer, this.age(buffer) + 1);
  }

  beginScope(): void {
    this.scopes.push(new Set());
  }

  /**
   * 구역을 닫고 그 안에서 만든 것을 놓는다.
   *
   * @param keep 살려 둘 것. 바깥 구역이 있으면 그쪽으로 넘긴다 — 안 넘기면 바깥이
   *   닫힐 때 아무도 안 놓아준다.
   * @returns 놓은 수와 **살아남은 수**. 둘 다 준다 — 살아남은 수가 곧 이 구역이
   *   바깥으로 흘린 것이고, 학습 루프에서 그것이 0 이 아니면 스텝마다 쌓인다.
   */
  endScope(keep: readonly GPUBuffer[] = []): { freed: number; survived: number } {
    const frame = this.scopes.pop();
    if (!frame) return { freed: 0, survived: 0 };
    const spare = new Set(keep);
    const outer = this.scopes[this.scopes.length - 1];
    let freed = 0;
    let survived = 0;
    for (const buf of frame) {
      if (spare.has(buf) || this.kept.has(buf)) {
        outer?.add(buf);
        survived += 1;
        continue;
      }
      // **여기서 죽는다.** 이 버퍼를 들고 밖으로 샌 텐서가 있으면 이제부터 그
      // 텐서는 쓰면 멈춘다 — 안 그러면 다음 할당이 덮어쓴 값을 조용히 읽는다.
      this.retire(buf);
      // 파괴하지 않고 통에 돌려놓는다. 다음 스텝이 같은 크기를 다시 부른다.
      const size = this.sizes.get(buf);
      if (size === undefined) {
        // 여기 오는 것은 `alloc` 이 안 만든 버퍼다. 아직 안 보낸 명령이 이것을
        // 가리킬 수 있으므로 보내고 나서 놓는다.
        this.flush();
        buf.destroy();
      } else {
        let pool = this.spare.get(size);
        if (!pool) {
          pool = [];
          this.spare.set(size, pool);
        }
        pool.push(buf);
      }
      freed += 1;
    }
    // **마지막 셈을 남긴다.** 이 값이 필요해서 `scope()` 를 못 쓰고 `beginScope`/
    // `endScope` 를 직접 부르던 자리가 있었다 — 벤치가 누수를 재느라 그랬다.
    // 권하는 길을 쓰면 못 보는 것이 있으면 그 권함은 안 지켜진다.
    this.lastScope = { freed, survived };
    return this.lastScope;
  }

  /**
   * 가장 최근에 닫힌 구역의 셈. `scope()` 로 닫아도 여기 남는다.
   *
   * **`survived` 가 0 이 아니면 스텝마다 쌓인다** — 학습 루프에서 그것이 누수다.
   */
  lastScope: { freed: number; survived: number } = { freed: 0, survived: 0 };

  /**
   * 지금 잡고 있는 버퍼의 수와 바이트.
   *
   * 벤치가 누수를 재려면 밖에서 이것을 물을 수 있어야 한다. 자매 쪽 벤치는
   * `js.tf.memory()` 를 직접 불렀는데, 그러면 계측이 TF.js 에 묶여서 같은 벤치를
   * 다른 구현으로 못 돌린다 — 실제로 그것 때문에 못 돌렸다.
   *
   * `spare` 에 든 것은 뺀다. 통에 돌아와 다음 스텝을 기다리는 버퍼는 잡고 있는
   * 것이 맞지만 **새는 것은 아니다** — 그것을 세면 누수가 아닌 것을 누수로 읽는다.
   */
  get memory(): { tensors: number; bytes: number } {
    let spare = 0;
    let spareBytes = 0;
    for (const [size, pool] of this.spare) {
      spare += pool.length;
      spareBytes += size * pool.length;
    }
    return { tensors: this.made - spare, bytes: this.madeBytes - spareBytes };
  }

  // `sizes` 는 WeakMap 이라 셀 수 없다 — 셀 수 있게 두면 버퍼가 안 죽는다.
  // 그래서 만들 때 센다. **빼는 자리는 없다** — `alloc` 이 만든 버퍼는 파괴하지
  // 않고 통에 돌려놓기 때문이다(`endScope`). 파괴하는 두 자리(`alloc` 밖에서 온
  // 버퍼, 읽기용 staging)는 애초에 여기 안 세어졌다.
  private made = 0;
  private madeBytes = 0;

  /** 구역과 무관하게 살려 둔다. 파라미터와 옵티마이저 상태가 이것을 쓴다. */
  keep(buffer: GPUBuffer): void {
    this.kept.add(buffer);
  }

  /** 열려 있는 구역의 깊이. 테스트가 균형을 본다. */
  get scopeDepth(): number {
    return this.scopes.length;
  }

  /**
   * @param recycle 통에 있는 것을 꺼내 써도 되는가.
   *
   * **올리기는 꺼내 쓰면 안 된다.** `writeBuffer` 는 큐의 지금 자리에서 도는데, 우리는
   * 명령을 쌓아 두었다가 나중에 보낸다 — 통에서 꺼낸 버퍼를 아직 안 보낸 dispatch 가
   * 읽을 참이면 그것을 덮어쓰게 된다. 새로 만들면 그 자리가 아예 없다.
   */
  alloc(count: number, recycle = true): GPUBuffer {
    const bytes = count * BYTES_PER_F32;
    const max = this.limits.maxStorageBufferBindingSize;
    if (bytes > max) {
      // 넘긴 채로 돌리면 WebGPU 는 조용히 일부만 쓴다. 여기서 멈추는 편이 낫다.
      throw new Error(
        `버퍼가 한계를 넘는다: ${(bytes / 1048576).toFixed(1)}MB > ` +
          `${(max / 1048576).toFixed(0)}MB (maxStorageBufferBindingSize)`,
      );
    }
    const size = Math.max(bytes, BYTES_PER_F32);
    const reused = recycle ? this.spare.get(size)?.pop() : undefined;
    const buf = reused ?? this.device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    if (!reused) {
      this.made += 1;
      this.madeBytes += size;
    }
    this.sizes.set(buf, size);
    this.scopes[this.scopes.length - 1]?.add(buf);
    return buf;
  }

  upload(data: Float32Array): GPUBuffer {
    const buf = this.alloc(data.length, false);
    this.device.queue.writeBuffer(buf, 0, data as unknown as BufferSource);
    return buf;
  }

  /**
   * 커널 한 번. `groups` 는 **워크그룹 수**이고, 축당 한계를 여기서 다시 확인한다 —
   * `kernels.ts` 가 격자를 접어 주지만 손으로 부르는 경로가 생길 수 있다.
   */
  run(
    pipeline: GPUComputePipeline,
    buffers: readonly GPUBuffer[],
    groups: readonly [number, number, number],
  ): void {
    const cap = this.limits.maxComputeWorkgroupsPerDimension;
    for (const [axis, count] of groups.entries()) {
      if (count > cap) {
        throw new Error(
          `dispatch ${axis} 축이 한계를 넘는다: ${count} > ${cap}. ` +
            "WebGPU 는 이것을 던지지 않고 조용히 안 한다.",
        );
      }
    }
    let layout = this.layouts.get(pipeline);
    if (!layout) {
      layout = pipeline.getBindGroupLayout(0);
      this.layouts.set(pipeline, layout);
    }
    const bindGroup = this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const pass = this.openPass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
    this.dispatches += 1;
    this.byKind.set(this.current, (this.byKind.get(this.current) ?? 0) + 1);
  }

  /** 1차원 작업을 격자로 펴서 돌린다. `kernels.ts` 의 인덱싱과 짝이다. */
  run1d(pipeline: GPUComputePipeline, buffers: readonly GPUBuffer[], n: number): void {
    const g = grid1d(n);
    this.run(pipeline, buffers, [g.x, g.y, 1]);
  }

  /**
   * 전체 합. 부분합이 하나가 될 때까지 같은 커널을 다시 부른다.
   *
   * **원자 연산을 안 쓴다** — 부동소수 덧셈은 순서가 바뀌면 값이 달라지고, 그러면
   * 같은 씨앗으로 두 번 돌린 학습이 갈린다. 느린 쪽이 재현된다.
   */
  sumAll(input: GPUBuffer, n: number): GPUBuffer {
    let src = input;
    let count = n;
    let owned: GPUBuffer | null = null;
    while (count > 1) {
      const parts = reduceParts(count);
      const dst = this.alloc(parts);
      const size = count;
      this.run1d(
        this.pipeline(`reduceSum:${size}`, () => reduceSum(size)),
        [src, dst],
        size,
      );
      // **여기서 놓으면 안 된다.** 명령을 쌓아 두었다가 나중에 보내므로, 방금 건
      // dispatch 가 아직 이 버퍼를 읽을 참이다. 구역이 닫힐 때 통으로 돌아간다.
      owned = dst;
      src = dst;
      count = parts;
    }
    if (owned) return owned;
    // 원소가 하나면 접을 것이 없다. 입력을 그대로 돌려주면 호출자가 남의 버퍼를
    // 파괴하게 되므로 복사해서 준다.
    //
    // **쌓아 둔 줄에 얹어야 한다.** 여기서 인코더를 따로 만들어 바로 제출했더니, 아직
    // 안 보낸 명령이 만들 값을 **먼저** 복사해서 0 이 나왔다 — 뒤늦게 그 값이 계산돼도
    // 복사본은 이미 떠난 뒤다. 예외도 NaN 도 아니고 **그냥 0** 이라, `x.mean()` 의
    // `x` 가 원소 하나일 때 손실이 조용히 0 이 되는 자리였다. 원소가 하나인 텐서를
    // 접는 일이 드물어서 골든 1,399 건이 초록인 채로 지나갔다.
    const copy = this.alloc(1);
    this.copyInto(copy, input, 1);
    return copy;
  }

  /**
   * 버퍼 하나를 다른 버퍼에 덮어쓴다. 제자리 연산이 쓴다.
   *
   * 커널이 아니라 복사다 — 결과를 새 버퍼에 만든 뒤 원래 자리로 옮긴다. 원래
   * 버퍼를 읽으면서 같이 쓰면 스레드끼리 순서가 없어 값이 섞인다.
   */
  copyInto(dst: GPUBuffer, src: GPUBuffer, count: number): void {
    const bytes = Math.max(count * BYTES_PER_F32, BYTES_PER_F32);
    // 복사는 계산 패스 안에 못 들어간다. 패스를 닫고 같은 인코더에 얹으면 순서는
    // 그대로이고 제출은 여전히 한 번이다.
    this.openEncoder().copyBufferToBuffer(src, 0, dst, 0, bytes);
  }

  /**
   * 커널마다 GPU 시간을 잰다. **기본은 꺼져 있다.**
   *
   * ## 왜 있는가
   *
   * 벽시계로는 스텝 전체밖에 못 잰다. 429 개 dispatch 중 어느 것이 비싼지 물으려
   * 했더니 물을 방법이 없었다 — 종류별 **횟수**는 있는데 종류별 **시간**이 없었고,
   * 횟수는 배치가 커져도 그대로라 아무것도 안 가리켰다.
   *
   * ## 켜면 무엇이 달라지는가
   *
   * 평소에는 모든 dispatch 가 계산 패스 **하나**를 함께 쓴다(제출도 스텝당 한 번).
   * 타임스탬프는 패스 단위라, 그 상태로는 패스 전체의 시작·끝밖에 못 찍는다.
   * 그래서 켜면 **dispatch 마다 패스를 연다.**
   *
   * **그러면 절대값은 평소보다 커진다.** 패스를 여는 값이 붙기 때문이다. 여기서
   * 얻으려는 것은 절대 시간이 아니라 **어느 커널이 몫이 큰가** 이고, 그 비율은
   * 남는다. 절대값을 재려면 끄고 벤치를 쓴다.
   */
  private profiling = false;
  /** 켰을 때 커널 종류별로 쌓인 GPU 시간(나노초). */
  readonly nsByKind = new Map<string, number>();
  private querySet: GPUQuerySet | null = null;
  private queryUsed = 0;
  private queryKinds: string[] = [];
  /**
   * 자리가 없어 **안 잰** dispatch 수. 부르는 쪽이 이것을 같이 적어야 한다.
   *
   * 0 이 아니면 `nsByKind` 는 스텝의 일부만 담은 것이고, 그런데도 "합" 처럼 읽힌다.
   * 잘린 것을 안 적으면 "전부 쟀다" 로 읽히는데, 그게 이 저장소가 세어 온 거짓말의
   * 한 종류다.
   */
  profileDropped = 0;
  /** 질의 집합의 크기. 한 제출에 이보다 많이 재면 나머지는 안 잰다. */
  private static readonly MAX_QUERIES = 4096;

  /**
   * `body` 를 재면서 돌린다. **끝나면 반드시 끈다 — 예외로 나가도 끈다.**
   *
   * 켜고 끄기를 부르는 쪽에 맡기면 안 된다. 프로파일 중에는 dispatch 마다 패스를
   * 여느라 시간이 부풀고, 켜진 채로 새어 나가면 **그 뒤의 모든 측정이 조용히
   * 부풀어서** 나온다. 벤치는 배치를 여러 번 재므로, 한 배치에서 예외가 나면
   * 다음 배치의 ms/step 이 측정이 아니라 프로파일된 수가 된다 — 그런데 화면에는
   * 똑같이 생긴 수로 찍힌다. 문은 하나여야 하고 뒷정리는 문이 해야 한다.
   */
  async profile<T>(body: () => Promise<T>): Promise<T> {
    this.profiling = true;
    this.nsByKind.clear();
    this.queryUsed = 0;
    this.queryKinds = [];
    this.profileDropped = 0;
    try {
      return await body();
    } finally {
      this.profiling = false;
      await this.collectProfile();
    }
  }

  /** 계산 패스를 연다. 평소에는 하나를 함께 쓰고, 프로파일 중에는 하나씩 연다. */
  private openPass(): GPUComputePassEncoder {
    if (!this.profiling) {
      if (!this.pass) this.pass = this.openEncoder().beginComputePass();
      return this.pass;
    }
    // 프로파일 중 — 앞 패스를 닫고 타임스탬프를 낀 새 패스를 연다.
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    const encoder = this.encoder ?? (this.encoder = this.device.createCommandEncoder());
    this.querySet ??= this.device.createQuerySet({
      type: "timestamp", count: Device.MAX_QUERIES,
    });
    if (this.queryUsed + 2 > Device.MAX_QUERIES) {
      // 자리가 없으면 그냥 평소처럼 연다 — **안 잰 것을 0 으로 세면 안 된다.**
      // 세어는 둔다. 안 세면 잘린 표가 온전한 표와 똑같이 생긴다.
      this.profileDropped += 1;
      this.pass = encoder.beginComputePass();
      return this.pass;
    }
    const at = this.queryUsed;
    this.queryUsed += 2;
    this.queryKinds.push(this.currentSig);
    this.pass = encoder.beginComputePass({
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassWriteIndex: at,
        endOfPassWriteIndex: at + 1,
      },
    });
    return this.pass;
  }

  /**
   * 찍어 둔 타임스탬프를 읽어 종류별로 더한다. **제출 뒤에 불러야 한다.**
   *
   * 해석 버퍼와 읽기 버퍼를 그때그때 만들고 버린다 — 프로파일은 드물게 도는 길이라
   * 통을 쓸 값어치가 없고, 통을 쓰면 재는 장치가 재는 대상을 건드린다.
   */
  private async collectProfile(): Promise<void> {
    if (!this.querySet || this.queryUsed === 0) return;
    const count = this.queryUsed;
    const kinds = this.queryKinds;
    this.queryUsed = 0;
    this.queryKinds = [];
    const bytes = count * 8;
    const resolved = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const stage = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.resolveQuerySet(this.querySet, 0, count, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, stage, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const times = new BigUint64Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    resolved.destroy();
    for (const [i, kind] of kinds.entries()) {
      const start = times[i * 2];
      const end = times[i * 2 + 1];
      if (start === undefined || end === undefined || end <= start) continue;
      this.nsByKind.set(kind, (this.nsByKind.get(kind) ?? 0) + Number(end - start));
    }
  }

  /** 인코더를 연다. 계산 패스가 열려 있으면 닫는다 — 복사가 그 밖에 있어야 한다. */
  private openEncoder(): GPUCommandEncoder {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    this.encoder ??= this.device.createCommandEncoder();
    return this.encoder;
  }

  /**
   * 쌓아 둔 명령을 보낸다.
   *
   * 값을 읽기 전에 반드시 지나야 한다 — 안 보낸 명령의 결과를 읽으면 옛 값이 나온다.
   */
  flush(): void {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    if (!this.encoder) return;
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    this.submits += 1;
  }

  /**
   * 보낸 일이 **실제로 끝날 때까지** 기다린다. `torch.cuda.synchronize()` 자리다.
   *
   * `flush()` 는 큐에 넣기만 하고 돌아온다 — 그것으로 시간을 재면 GPU 가 일하는 동안
   * 벽시계는 이미 멈춰 있다. 지금까지 이 저장소가 완료를 강제한 방법은 값을 하나
   * 읽는 것(`item()`)이었는데, 그러면 **readback 왕복이 측정에 섞인다.** 재는 것이
   * 커널인지 버스인지가 흐려지는 자리이고, 그것을 가르는 것이 이 함수다.
   */
  async synchronize(): Promise<void> {
    this.flush();
    await this.device.queue.onSubmittedWorkDone();
  }

  async read(buffer: GPUBuffer, count: number): Promise<Float32Array> {
    // 빈 텐서를 읽으면 빈 것이 나와야 한다. 버퍼는 최소 한 칸을 잡으므로, 그것을
    // 그대로 읽으면 있지도 않은 원소 하나가 딸려 나온다.
    if (count === 0) return new Float32Array(0);
    const bytes = Math.max(count * BYTES_PER_F32, BYTES_PER_F32);
    let free = this.stagingFree.get(bytes);
    if (!free) {
      free = [];
      this.stagingFree.set(bytes, free);
    }
    const stage = free.pop() ?? this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      // 쌓아 둔 명령까지 한 인코더에 얹고 **여기서 한 번** 보낸다.
      this.openEncoder().copyBufferToBuffer(buffer, 0, stage, 0, bytes);
      this.flush();
      await stage.mapAsync(GPUMapMode.READ);
      // 매핑된 메모리는 unmap 하면 사라진다. 반드시 복사해서 내보낸다.
      const out = new Float32Array(stage.getMappedRange().slice(0));
      stage.unmap();
      // **성공했을 때만 돌려놓는다.** 실패한 버퍼는 매핑 상태를 모르고, 그것을
      // 풀에 넣으면 다음 사람이 깨진 상태를 물려받아 원인이 한 단계 멀어진다.
      free.push(stage);
      return out;
    } catch (err) {
      stage.destroy();
      throw err;
    }
  }

  /** 워크그룹 크기. 커널과 장치가 같은 값을 봐야 한다. */
  static readonly workgroup = WORKGROUP;

  /** 어느 어댑터에 붙었는가. 성능을 재는 쪽이 반드시 같이 적어야 하는 값이다. */
  static adapterInfo = "(아직 안 붙음)";

  /**
   * 어댑터가 주는 선택 기능들. **`timestamp-query` 가 여기 있어야 커널별 시간을 잰다.**
   *
   * 벽시계로는 스텝 전체밖에 못 재고, 그러면 429 개 dispatch 중 어느 것이 비싼지
   * 물을 방법이 없다 — 실제로 그 자리에서 막혔다.
   */
  static adapterFeatures = "";
}
