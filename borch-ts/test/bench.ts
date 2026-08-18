/**
 * ResNet-18(CIFAR 판)로 **진짜 학습 스텝**을 잰다.
 *
 * `tests/browser/bench.py` 와 **같은 잣대**다 — 같은 모델, 같은 옵티마이저, 같은
 * 배치, 같은 계산식. 잣대가 다르면 비교가 아니라 두 개의 다른 수다.
 *
 * 층 하나로 잰 값은 BN·ReLU·잔차 덧셈처럼 대역폭에 묶인 연산을 안 센다. "에폭 몇 분"
 * 은 그것들까지 포함한 실제 스텝에서만 나온다 — FLOPs 로 나눈 추정이 아니라.
 */

import * as nn from "../src/nn.js";
import { SGD } from "../src/optim.js";
import { Device } from "../src/device.js";
import { device, keepAlive, noGrad, scope, Tensor } from "../src/tensor.js";

/** 에폭 시간을 내는 데 쓰는 장수. 파이썬 벤치와 같은 수여야 비교가 된다. */
const CIFAR_TRAIN_IMAGES = 50000;

/** ResNet 의 기본 블록. 지름길이 모양을 바꿔야 할 때만 1×1 을 둔다. */
export class Block extends nn.Module {
  private readonly conv1: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly conv2: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  /**
   * **필드로 꺼내 둔다.** 전에는 `{ conv, bn }` 이라는 평범한 객체였고
   * `children()` 에만 적혀 있었다 — `namedChildren()` 은 `instanceof Module` 인
   * 필드만 훑으므로 그 둘을 못 봤고, **지름길 층 여섯이 한 번도 안 배웠다.**
   * 손실은 내려갔다. 나머지가 대신 맞추기 때문이다.
   *
   * torch 도 파이썬 dict 에 담은 층을 등록 안 한다(그래서 `nn.ModuleDict` 가
   * 있다). 라이브러리가 옳았고 이 파일이 틀렸다.
   */
  private readonly downConv: nn.Conv2d | null;
  private readonly downBn: nn.BatchNormND | null;

  constructor(cin: number, cout: number, stride: number) {
    super();
    this.conv1 = new nn.Conv2d(cin, cout, 3, stride, 1, false);
    this.bn1 = new nn.BatchNormND(cout);
    this.conv2 = new nn.Conv2d(cout, cout, 3, 1, 1, false);
    this.bn2 = new nn.BatchNormND(cout);
    const shrinks = stride !== 1 || cin !== cout;
    this.downConv = shrinks ? new nn.Conv2d(cin, cout, 1, stride, 0, false) : null;
    this.downBn = shrinks ? new nn.BatchNormND(cout) : null;
  }

  override forward(x: Tensor): Tensor {
    let out = this.bn1.forward(this.conv1.forward(x)).unary("relu");
    out = this.bn2.forward(this.conv2.forward(out));
    const side = this.downConv && this.downBn
      ? this.downBn.forward(this.downConv.forward(x))
      : x;
    return out.add(side).unary("relu");
  }
}

/**
 * CIFAR 판 — 3×3 스템에 맥스풀이 없다. 32×32 를 7×7 스템으로 받으면 너무 줄어든다.
 */
export class ResNet18 extends nn.Module {
  private readonly stem: nn.Conv2d;
  private readonly bn: nn.BatchNormND;
  private readonly body: nn.Sequential;
  private readonly fc: nn.Linear;

  constructor(classes = 10) {
    super();
    this.stem = new nn.Conv2d(3, 64, 3, 1, 1, false);
    this.bn = new nn.BatchNormND(64);
    this.body = new nn.Sequential([
      new Block(64, 64, 1), new Block(64, 64, 1),
      new Block(64, 128, 2), new Block(128, 128, 1),
      new Block(128, 256, 2), new Block(256, 256, 1),
      new Block(256, 512, 2), new Block(512, 512, 1),
    ]);
    this.fc = new nn.Linear(512, classes);
  }

  // `children()` 을 안 덮어쓴다 — 넷 다 필드라 기본 훑기가 찾는다. 덮어쓰면
  // `namedChildren()` 과 어긋날 자리가 생기고, 그 어긋남이 바로 위 `Block` 을
  // 여섯 층 동안 안 배우게 만든 것이다.

  override forward(x: Tensor): Tensor {
    let h = this.bn.forward(this.stem.forward(x)).unary("relu");
    h = this.body.forward(h).adaptiveAvgPool(1);
    return this.fc.forward(h.reshape([h.shape[0] ?? 1, 512]));
  }
}

export interface StepResult {
  batch: number;
  params: number;
  msPerStep: number;
  epochMin: number;
  /** 스텝당 안 놓인 버퍼 수. 0 이 아니면 새는 것이다. */
  leakPerStep: number;
  /** 스텝당 dispatch 수. 느린 것이 커널인지 부르는 횟수인지 가른다. */
  dispatches: number;
  /** dispatch 하나당 벽시계. 커널 시간이 아니라 **부르는 값**의 상한이다. */
  usPerDispatch: number;
  /** 커널 종류별 dispatch 수, 많은 것부터. 다음에 무엇을 고칠지가 여기서 나온다. */
  kinds: [string, number][];
  /** 스텝당 실제 제출 수. dispatch 수와 갈리는 만큼이 묶인 것이다. */
  submits: number;
  /**
   * 스텝이 끝난 뒤 잡고 있는 GPU 메모리(MB).
   *
   * **파이썬 벤치는 이것을 재는데 이쪽은 안 쟀다.** 같은 잣대라고 부르면서 한쪽만
   * 보는 칸이 있으면 그 칸에서 갈린 것은 비교에 안 나온다.
   */
  gpuMb: number;
  /** 순방향만 돌렸을 때의 벽시계. 나머지가 역방향과 옵티마이저의 몫이다. */
  msForward: number;
  /** 가중치 기울기를 끈 스텝. 전체에서 빼면 `gradWeight` 의 몫이다. */
  msNoWeightGrad: number;
  /** 그때의 dispatch 수. 안 줄었으면 **깃발이 안 먹은 것**이다. */
  noWeightDispatches: number;
  loss: number;
}

/**
 * 한 스텝의 벽시계 시간.
 *
 * **워밍업 뒤에 기준선을 잡는다.** 안 그러면 첫 스텝의 셰이더 컴파일이 측정 창에
 * 들어와 스텝당 비용으로 둔갑한다 — 모양마다 셰이더를 굽는 설계라 그 값이 크다.
 */
export async function runStep(
  batch = 32, steps = 5, warmup = 2,
): Promise<StepResult> {
  // 입력은 **구역 밖**에서 만든다. 안에서 만들면 첫 스텝이 끝날 때 놓인다.
  const rng = { s: 12345 };
  const next = (): number => {
    let x = rng.s; x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
    rng.s = x;
    return x / 0x100000000;
  };
  const pixels = new Float32Array(batch * 3 * 32 * 32);
  for (let i = 0; i < pixels.length; i++) pixels[i] = next() * 2 - 1;
  const labels = new Float32Array(batch);
  for (let i = 0; i < batch; i++) labels[i] = Math.floor(next() * 10);
  const x = keepAlive(Tensor.from(pixels, [batch, 3, 32, 32]));
  const y = keepAlive(Tensor.from(labels, [batch], { dtype: "int64" }));

  const model = new ResNet18();
  const params = model.parameters();
  const count = params.reduce((n, p) => n + p.size, 0);
  const opt = new SGD(params, 0.05, 0.9);
  const crit = new nn.CrossEntropyLoss();

  // **쓰는 사람이 칠 그대로 친다.** 벤치는 남이 보는 유일한 학습 루프이므로, 여기가
  // 저수준 `beginScope`/`endScope` 를 부르면 `scope()` 를 권한 것이 무의미해진다.
  // 누수는 `device().lastScope` 로 본다 — 그것 때문에 저수준으로 내려가던 자리였다.
  const one = async (): Promise<number> => scope(async () => {
    opt.zeroGrad();
    const out = crit.call(model.call(x), y);
    out.backward();
    opt.step();
    // **구역 안에서 읽어야 한다** — 나가면 그 버퍼가 없다.
    return await out.item();
  });

  for (let i = 0; i < warmup; i++) await one();

  const t0 = performance.now();
  const d0 = device().dispatches;
  const s0 = device().submits;
  const k0 = new Map(device().byKind);
  let last = 0;
  for (let i = 0; i < steps; i++) last = await one();
  const perStep = (performance.now() - t0) / steps;
  const perStepDispatches = (device().dispatches - d0) / steps;
  const perStepSubmits = (device().submits - s0) / steps;
  // 종류별로 갈라 둔다 — 총수만으로는 다음에 무엇을 고칠지 안 나온다.
  const kinds: [string, number][] = [];
  for (const [kind, n] of device().byKind) {
    const grew = (n - (k0.get(kind) ?? 0)) / steps;
    if (grew > 0) kinds.push([kind, Math.round(grew)]);
  }
  kinds.sort((a, b) => b[1] - a[1]);

  // 누수는 **구역이 안 놓고 내보낸 버퍼 수**다.
  //
  // 처음에는 열린 구역의 깊이를 뺐는데, 그건 구역 밖에서 늘 0 이라 언제나 0 을 낸다 —
  // 이름은 누수인데 재는 것이 아무것도 없는 계측이다. 파이썬 벤치가 같은 함정을
  // 적어두었고("이름이 말하는 것과 다른 수를 내는 계측은 진짜 누수를 만났을 때도
  // 못 믿는다") 나는 그것을 읽고도 같은 것을 만들었다.
  await one();
  const leak = device().lastScope.survived;

  /**
   * 순방향만.
   *
   * **역방향의 몫을 추측하지 않으려고 잰다.** 전체에서 이것을 빼면 남는 것이
   * 역방향과 옵티마이저이고, 그 둘 중 어느 쪽을 먼저 고칠지가 거기서 정해진다.
   * 손실을 읽는 것까지 같이 해야 GPU 를 기다리는 지점이 같다.
   */
  const forwardOnly = async (): Promise<void> => {
    const d = device();
    d.beginScope();
    try {
      // `noGrad` 는 **동기** 본문을 받는다 — 비동기를 넣으면 기울기 스위치가 읽기보다
      // 먼저 되돌아간다. 그래프 세우기만 감싸고 읽기는 밖에서 기다린다.
      const out = noGrad(() => crit.forward(model.forward(x), y));
      await out.item();
    } finally {
      d.endScope();
    }
  };
  await forwardOnly();
  const f0 = performance.now();
  for (let i = 0; i < steps; i++) await forwardOnly();
  const perForward = (performance.now() - f0) / steps;

  /**
   * 가중치 기울기를 끄고 한 스텝.
   *
   * **어느 커널이 무거운지를 추측 대신 뺄셈으로 낸다.** `gradWeight` 는 출력이 작고
   * 축약이 큰 GEMM 이라 층에 따라 워크그룹이 한 개까지 떨어진다 — 그것이 실제로
   * 비싼지는 켜고 끈 차이로만 알 수 있다. 학습은 이 동안 틀리지만 재는 것은 시간이다.
   */
  const convWeights = params.filter((p) => p.shape.length >= 3);
  for (const p of convWeights) p.requiresGrad = false;
  await one();
  const w0 = performance.now();
  const wd0 = device().dispatches;
  for (let i = 0; i < steps; i++) await one();
  const perNoWeightGrad = (performance.now() - w0) / steps;
  // **깃발이 진짜로 일을 줄이는가.** 시간만 보면 "gradWeight 가 싸다" 와 "깃발이
  // 안 먹는다" 가 같은 화면이다. dispatch 수가 안 줄면 뒤쪽이다.
  const noWeightDispatches = (device().dispatches - wd0) / steps;
  for (const p of convWeights) p.requiresGrad = true;

  // **검증 오류가 하나라도 났으면 수를 안 낸다.**
  //
  // 무효한 명령 버퍼는 예외를 안 던지고 그냥 아무것도 안 한다. 그 상태에서도
  // 벽시계는 돌아서 ms/step 이 나오는데, 그건 측정이 아니라 학습이 안 되는 상태의
  // 벽시계다. 실제로 그렇게 됐다 — 평균 풀링의 역방향이 통째로 무효였고 손실이
  // 2.27 에서 안 움직이는 채로 "13070 ms/step" 이 나왔다.
  const faults = device().faults;
  if (faults.count > 0) {
    throw new Error(
      `WebGPU 검증 오류 ${faults.count}건이 났다 — 이 수는 측정이 아니다.\n` +
        `첫 건: ${faults.first}`,
    );
  }

  const stepsPerEpoch = Math.ceil(CIFAR_TRAIN_IMAGES / batch);
  return {
    batch,
    params: count,
    gpuMb: Math.round(device().memory.bytes / 1e5) / 10,
    msPerStep: Math.round(perStep * 10) / 10,
    epochMin: Math.round((perStep * stepsPerEpoch) / 600) / 100,
    leakPerStep: leak,
    dispatches: Math.round(perStepDispatches),
    usPerDispatch: Math.round((perStep * 1000) / Math.max(1, perStepDispatches)),
    kinds,
    submits: Math.round(perStepSubmits),
    msForward: Math.round(perForward * 10) / 10,
    msNoWeightGrad: Math.round(perNoWeightGrad * 10) / 10,
    noWeightDispatches: Math.round(noWeightDispatches),
    loss: Math.round(last * 10000) / 10000,
  };
}

/** 파이썬 벤치와 같은 배치들을 같은 차례로. */
export async function report(batches: readonly number[] = [16, 32, 64]): Promise<string> {
  const lines: string[] = [];
  for (const b of batches) {
    try {
      const r = await runStep(b);
      lines.push(
        `batch ${String(r.batch).padStart(3)}  ` +
        `${r.msPerStep.toFixed(1).padStart(8)} ms/step  ` +
        `(순방향 ${r.msForward.toFixed(1).padStart(7)} · dW뺀것 ` +
        `${r.msNoWeightGrad.toFixed(1).padStart(7)}/${r.noWeightDispatches}d)  ` +
        `에폭 ${r.epochMin.toFixed(2).padStart(6)}분  ` +
        `dispatch ${String(r.dispatches).padStart(5)}  ` +
        `제출 ${String(r.submits).padStart(3)}  ` +
        `${String(r.usPerDispatch).padStart(5)}µs/dispatch  ` +
        `누수 ${r.leakPerStep.toFixed(1).padStart(5)}  ` +
        `${r.gpuMb.toFixed(1).padStart(6)}MB  ` +
        `손실 ${r.loss.toFixed(4)}`,
      );
      // 종류별 내역은 배치마다 같으므로 한 번만 찍는다.
      if (b === batches[0]) {
        const top = r.kinds.slice(0, 8)
          .map(([kind, n]) => `${kind} ${n}`).join(" · ");
        lines.push(`         dispatch 내역: ${top}`);
      }
    } catch (err) {
      lines.push(`batch ${String(b).padStart(3)}  실패: ` +
        `${err instanceof Error ? `${err.constructor.name}: ${err.message.slice(0, 120)}` : String(err)}`);
      // **어디서 났는지가 무엇이 났는지만큼 중요하다.** 문구만 남기면 같은 문구를
      // 내는 자리가 여럿일 때 어느 쪽인지 못 가른다.
      if (err instanceof Error && err.stack) {
        lines.push(`         ${err.stack.split("\n").slice(3, 11).join("\n         ")}`);
      }
    }
  }
  // **어느 장치에서 잰 것인지가 수보다 먼저 온다.** 헤드리스 브라우저가 소프트웨어
  // 어댑터를 주면 예외 없이 느린 수가 나오고, 그것은 라이브러리의 성적이 아니다.
  return `ResNet-18(CIFAR) · 배치별 실제 학습 스텝\n어댑터: ${Device.adapterInfo}\n`
    + lines.join("\n");
}

/** 구역을 밖에서도 쓸 수 있게 — 러너가 전체를 한 겹 더 감싼다. */
export { scope };

/**
 * 조각을 하나씩 떼어 **자리마다 다른 가중치로** 역전파해 본다.
 *
 * 전체 모델만 견주면 갈렸을 때 어디인지 모른다. 그리고 그냥 `sum()` 으로 접으면
 * 상류 기울기가 전부 1 이라 **BatchNorm 역방향의 보정항 둘이 정확히 상쇄된다** —
 * 골든의 `grad::BatchNorm2d/x` 가 기대값 4.7e-10 인 이유가 그것이고, 그래서 그
 * 케이스는 보정항을 하나도 안 물어본다.
 */
export async function dumpPieces(): Promise<Record<string, {
  params: number[][];
  shapes: number[][];
  input: number[];
  inputShape: number[];
  output: number[];
  inputGrad: number[];
  paramGrads: (number[] | null)[];
}>> {
  const pieces: Record<string, () => { mod: nn.Module; shape: number[] }> = {
    bn: () => ({ mod: new nn.BatchNormND(3), shape: [2, 3, 4, 4] }),
    avgpool: () => ({ mod: new AvgPoolTo1(), shape: [2, 3, 4, 4] }),
    relu: () => ({ mod: new nn.ReLU(), shape: [2, 3, 4, 4] }),
    conv: () => ({ mod: new nn.Conv2d(3, 4, 3, 1, 1, false), shape: [2, 3, 4, 4] }),
    block: () => ({ mod: new Block(3, 3, 1), shape: [2, 3, 4, 4] }),
    blockDown: () => ({ mod: new Block(3, 6, 2), shape: [2, 3, 8, 8] }),
  };
  const out: Record<string, {
    params: number[][]; shapes: number[][]; input: number[]; inputShape: number[];
    output: number[]; inputGrad: number[]; paramGrads: (number[] | null)[];
  }> = {};
  for (const [name, build] of Object.entries(pieces)) {
    const { mod, shape } = build();
    const n = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = Math.sin(i * 0.31) * 0.7;
    const x = keepAlive(Tensor.from(data, shape, { requiresGrad: true }));
    const y = mod.forward(x);
    // **자리마다 다른 가중치.** 전부 1 이면 보정항이 상쇄되어 아무것도 안 묻는다.
    const w = new Float32Array(y.size);
    for (let i = 0; i < w.length; i++) w[i] = ((i % 7) + 1) * 0.3;
    y.mul(Tensor.from(w, y.shape)).sum().backward();
    const grad = x.grad;
    if (!grad) throw new Error(`${name}: 입력 기울기가 안 왔다`);
    const params = mod.parameters();
    out[name] = {
      params: await Promise.all(params.map(async (p) => [...await p.toArray()])),
      shapes: params.map((p) => [...p.shape]),
      input: [...data],
      inputShape: shape,
      output: [...await y.toArray()],
      inputGrad: [...await grad.toArray()],
      paramGrads: await Promise.all(params.map(async (p) =>
        p.grad ? [...await p.grad.toArray()] : null)),
    };
  }
  return out;
}

/** 비교용 껍데기 — 파이썬 쪽 `AdaptiveAvgPool2d(1)` 과 짝이다. */
class AvgPoolTo1 extends nn.Module {
  override forward(x: Tensor): Tensor {
    return x.adaptiveAvgPool(1);
  }
}

/**
 * 이 ResNet-18 이 **파이썬 쪽 것과 같은 모델인지** 확인할 재료를 낸다.
 *
 * 벤치의 모델은 `tests/browser/bench.py` 를 눈으로 읽어 옮긴 것이라 골든 밖에 있다.
 * 블록 구성이나 BN 자리가 미묘하게 다르면 속도도 정확도도 **다른 모델끼리 비교한
 * 것**이 되는데, 그 갈림은 값으로만 보인다.
 *
 * 파라미터를 **순서대로** 내보낸다. 구조가 같으면 모양 목록이 정확히 같고, 다르면
 * 거기서 먼저 갈린다 — 이름 짓는 규칙이 두 언어에서 다르므로 자리로 맞춘다.
 */
export async function dumpForComparison(batch = 2): Promise<{
  shapes: number[][];
  params: number[][];
  input: number[];
  output: number[];
  loss: number;
  inputGrad: number[];
}> {
  const model = new ResNet18();
  const params = model.parameters();
  // 입력도 라벨도 못 박는다 — 두 쪽이 같은 수를 봐야 비교다.
  const pixels = new Float32Array(batch * 3 * 32 * 32);
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = Math.sin(i * 0.017) * 0.5;
  }
  const labels = new Float32Array(batch);
  for (let i = 0; i < batch; i++) labels[i] = i % 10;

  const x = keepAlive(Tensor.from(pixels, [batch, 3, 32, 32], { requiresGrad: true }));
  const y = keepAlive(Tensor.from(labels, [batch], { dtype: "int64" }));
  const out = model.forward(x);
  const loss = new nn.CrossEntropyLoss().forward(out, y);
  loss.backward();

  const grad = x.grad;
  if (!grad) throw new Error("입력 기울기가 안 왔다 — 그래프가 끊겼다");
  return {
    shapes: params.map((p) => [...p.shape]),
    params: await Promise.all(params.map(async (p) => [...await p.toArray()])),
    input: [...pixels],
    output: [...await out.toArray()],
    loss: await loss.item(),
    inputGrad: [...await grad.toArray()],
  };
}
