// conv 도 WGSL 로 TF.js 만큼 나오는가.
//
// **지금 이 파일은 안 돈다** — `wgsl_bench.js` 와 같은 이유다. TF.js 기준선을
// 부르는데 그 벤더 파일을 더는 안 받는다. 결론을 낸 방법으로 남긴다.
//
// 행렬곱은 잰 결과 우리 커널이 TF.js 의 115~217% 였다(wgsl_bench.js). conv 는 그
// 결과가 그대로 오는지가 관건이다 — im2col 로 펴서 행렬곱에 태우면 오고, 안 오면
// 어디서 새는지 봐야 한다. ResNet-18(CIFAR) 이 **실제로 쓰는 모양**으로만 잰다.
//
// 두 조각을 따로 잰다: im2col 자체와 그 뒤의 행렬곱. 합쳐서만 재면 느릴 때 어느
// 쪽이 범인인지 못 짚는다.

const IM2COL_WGSL = `
struct P {
  N: u32, C: u32, H: u32, W: u32,
  KH: u32, KW: u32, OH: u32, OW: u32,
  SH: u32, SW: u32, PH: u32, PW: u32,
};
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Cols: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

// 출력 한 칸 = (n, oh, ow) 한 자리의 (C·KH·KW) 벡터 중 하나.
//
// **격자 보폭 반복문을 쓴다.** 칸마다 스레드 하나씩 띄우면 워크그룹이
// 차원당 워크그룹 한계 65,535 를 넘는다 — 64채널 3×3 에 배치 64 면
// 589,824개가 필요하다. 넘으면 **조용히 안 돈다.** 처음 재봤을 때 여섯 중 다섯이
// 값이 틀렸고, 유일하게 맞은 스템은 채널이 3이라 27,648개로 한계 아래였다.
// 속도만 보고 있었으면 "TF.js 대비 144%" 를 그대로 믿었을 자리다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let inner = p.C * p.KH * p.KW;
  let rows = p.N * p.OH * p.OW;
  let total = rows * inner;
  let stride = nwg.x * 64u;

  for (var idx = gid.x; idx < total; idx = idx + stride) {
    let r = idx / inner;                        // 어느 (n, oh, ow)
    let c = idx % inner;                        // 그 안의 몇 번째

    let ow = r % p.OW;
    let oh = (r / p.OW) % p.OH;
    let n  = r / (p.OW * p.OH);

    let kw = c % p.KW;
    let kh = (c / p.KW) % p.KH;
    let ch = c / (p.KW * p.KH);

    let ih = i32(oh * p.SH + kh) - i32(p.PH);
    let iw = i32(ow * p.SW + kw) - i32(p.PW);

    var v = 0.0;
    if (ih >= 0 && iw >= 0 && u32(ih) < p.H && u32(iw) < p.W) {
      v = X[((n * p.C + ch) * p.H + u32(ih)) * p.W + u32(iw)];
    }
    Cols[idx] = v;
  }
}`;

// 행렬곱은 wgsl_bench.js 에서 이긴 그 커널과 같은 설계다 — 누산기를 이름 붙인
// 스칼라 16개로 **펼쳐서** 둔다. array 에 넣고 변수로 인덱싱하면 레지스터에서
// 떨어져 24배 느려진다(실측).
const unrolled = () => {
  const decl = [], zero = [], fma = [], store = [];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      decl.push(`  var c${i}${j}: f32;`);
      zero.push(`  c${i}${j} = 0.0;`);
      fma.push(`      c${i}${j} = fma(a${i}, b${j}, c${i}${j});`);
      store.push(`  { let r = row0 + ${i}u; let c = col0 + ${j}u;` +
                 ` if (r < d.M && c < d.N) { C[r * d.N + c] = c${i}${j}; } }`);
    }
  }
  return { decl: decl.join("\n"), zero: zero.join("\n"),
           fma: fma.join("\n"), store: store.join("\n") };
};
const U4 = unrolled();

const MATMUL_WGSL = `
struct Dims { M: u32, K: u32, N: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.y * 16u + lid.x;
  let row0 = wid.y * 64u + lid.y * 4u;
  let col0 = wid.x * 64u + lid.x * 4u;
${U4.decl}
${U4.zero}
  let tiles = (d.K + 15u) / 16u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    for (var s = 0u; s < 4u; s = s + 1u) {
      let idx = s * 256u + tid;
      let ar = idx / 16u;
      let ak = idx % 16u;
      let arow = wid.y * 64u + ar;
      let acol = t * 16u + ak;
      As[idx] = select(0.0, A[arow * d.K + acol], arow < d.M && acol < d.K);
      let bk = idx / 64u;
      let bc = idx % 64u;
      let brow = t * 16u + bk;
      let bcol = wid.x * 64u + bc;
      Bs[idx] = select(0.0, B[brow * d.N + bcol], brow < d.K && bcol < d.N);
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k = k + 1u) {
      let a0 = As[(lid.y * 4u + 0u) * 16u + k];
      let a1 = As[(lid.y * 4u + 1u) * 16u + k];
      let a2 = As[(lid.y * 4u + 2u) * 16u + k];
      let a3 = As[(lid.y * 4u + 3u) * 16u + k];
      let b0 = Bs[k * 64u + lid.x * 4u + 0u];
      let b1 = Bs[k * 64u + lid.x * 4u + 1u];
      let b2 = Bs[k * 64u + lid.x * 4u + 2u];
      let b3 = Bs[k * 64u + lid.x * 4u + 3u];
${U4.fma}
    }
    workgroupBarrier();
  }
${U4.store}
}`;

// 융합 conv — **im2col 을 따로 돌리지 않는다.**
//
// im2col 이 진 이유는 계산이 아니라 메모리다. 3×3 이면 입력을 9배로 부풀려 쓰고
// 다시 읽는다(64→64 층에서 151MB). TF.js 가 그 왕복이 없는 이유는 conv 커널이
// 입력을 직접 읽기 때문이고, 여기서 같은 것을 한다 — 위 행렬곱 커널에서 **A 조각을
// 싣는 줄만** 바꿔 그 자리에서 (n, oh, ow, c, kh, kw) 를 풀어 입력에서 바로 읽는다.
//
// 나머지(타일 크기, 누산기 16개를 펼쳐 두는 것, 공유메모리 재사용)는 그대로다.
const FUSED_WGSL = `
struct P {
  N: u32, C: u32, H: u32, W: u32,
  KH: u32, KW: u32, OH: u32, OW: u32,
  SH: u32, SW: u32, PH: u32, PW: u32,
  M: u32, K: u32, F: u32, _pad: u32,
};
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;   // (K, F)
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;

var<workgroup> As: array<f32, 1024>;            // 64 행 × 16
var<workgroup> Bs: array<f32, 1024>;            // 16 × 64 열

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.y * 16u + lid.x;
  let row0 = wid.y * 64u + lid.y * 4u;
  let col0 = wid.x * 64u + lid.x * 4u;
${U4.decl}
${U4.zero}

  let tiles = (p.K + 15u) / 16u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    for (var s = 0u; s < 4u; s = s + 1u) {
      let idx = s * 256u + tid;

      // --- A 조각: im2col 색인을 **여기서** 푼다. 중간 버퍼가 없다.
      let ar = idx / 16u;                       // 0..63, 출력 행 안의 자리
      let ak = idx % 16u;                       // 0..15, K 타일 안의 자리
      let arow = wid.y * 64u + ar;              // 전체 행 = (n, oh, ow)
      let acol = t * 16u + ak;                  // 전체 열 = (c, kh, kw)
      var av = 0.0;
      if (arow < p.M && acol < p.K) {
        let ow = arow % p.OW;
        let oh = (arow / p.OW) % p.OH;
        let n  = arow / (p.OW * p.OH);
        let kw = acol % p.KW;
        let kh = (acol / p.KW) % p.KH;
        let ch = acol / (p.KW * p.KH);
        let ih = i32(oh * p.SH + kh) - i32(p.PH);
        let iw = i32(ow * p.SW + kw) - i32(p.PW);
        if (ih >= 0 && iw >= 0 && u32(ih) < p.H && u32(iw) < p.W) {
          av = X[((n * p.C + ch) * p.H + u32(ih)) * p.W + u32(iw)];
        }
      }
      As[idx] = av;

      // --- B 조각: 가중치는 이미 (K, F) 라 그대로 읽는다.
      let bk = idx / 64u;
      let bc = idx % 64u;
      let brow = t * 16u + bk;
      let bcol = wid.x * 64u + bc;
      Bs[idx] = select(0.0, Wt[brow * p.F + bcol], brow < p.K && bcol < p.F);
    }
    workgroupBarrier();

    for (var k = 0u; k < 16u; k = k + 1u) {
      let a0 = As[(lid.y * 4u + 0u) * 16u + k];
      let a1 = As[(lid.y * 4u + 1u) * 16u + k];
      let a2 = As[(lid.y * 4u + 2u) * 16u + k];
      let a3 = As[(lid.y * 4u + 3u) * 16u + k];
      let b0 = Bs[k * 64u + lid.x * 4u + 0u];
      let b1 = Bs[k * 64u + lid.x * 4u + 1u];
      let b2 = Bs[k * 64u + lid.x * 4u + 2u];
      let b3 = Bs[k * 64u + lid.x * 4u + 3u];
${U4.fma}
    }
    workgroupBarrier();
  }

${U4.store.replace(/d\.M/g, "p.M").replace(/d\.N/g, "p.F").replace(/C\[/g, "Out[")}
}`;

// 융합 conv, **모양을 상수로 구워 넣은 판.**
//
// 앞의 융합 커널은 채널이 늘수록 im2col 판보다도 느려졌다(512→512 에서 절반).
// 원인은 읽기 패턴이 아니라 **나눗셈**이다. `p.OW`·`p.KW`·`p.C` 가 유니폼 값이라
// 컴파일러가 나눗셈을 곱셈·시프트로 못 바꾸고, GPU 에는 정수 나눗셈 하드웨어가 없다.
// 타일을 실을 때마다 원소당 나눗셈·나머지 7번을 다시 하고, 512채널 3×3 이면 타일이
// 288번이다. im2col 은 같은 나눗셈을 **원소당 한 번만** 하고 펴놓은 것을 모두가 읽었다.
//
// 그래서 모양을 셰이더 문자열에 박는다. 모양마다 컴파일이 한 번 더 들지만 — 진짜
// 라이브러리도 모양 서명으로 셰이더를 캐시한다 — 나눗셈이 전부 상수 접기로 사라진다.
const fusedSpecialised = (s) => `
struct P { N: u32, M: u32, K: u32, F: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;

const C: u32 = ${s.C}u;   const H: u32 = ${s.H}u;   const W: u32 = ${s.W}u;
const KH: u32 = ${s.KH}u; const KW: u32 = ${s.KW}u;
const OH: u32 = ${s.OH}u; const OW: u32 = ${s.OW}u;
const SH: u32 = ${s.S}u;  const SW: u32 = ${s.S}u;
const PH: i32 = ${s.P};   const PW: i32 = ${s.P};
const KHW: u32 = ${s.KH * s.KW}u;
const OHW: u32 = ${s.OH * s.OW}u;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.y * 16u + lid.x;
  let row0 = wid.y * 64u + lid.y * 4u;
  let col0 = wid.x * 64u + lid.x * 4u;
${U4.decl}
${U4.zero}

  let tiles = (p.K + 15u) / 16u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    for (var s = 0u; s < 4u; s = s + 1u) {
      let idx = s * 256u + tid;
      let ar = idx / 16u;
      let ak = idx % 16u;
      let arow = wid.y * 64u + ar;
      let acol = t * 16u + ak;
      var av = 0.0;
      if (arow < p.M && acol < p.K) {
        // 제수가 전부 상수라 여기 나눗셈은 곱셈·시프트로 접힌다.
        let ow = arow % OW;
        let oh = (arow / OW) % OH;
        let n  = arow / OHW;
        let kw = acol % KW;
        let kh = (acol / KW) % KH;
        let ch = acol / KHW;
        let ih = i32(oh * SH + kh) - PH;
        let iw = i32(ow * SW + kw) - PW;
        if (ih >= 0 && iw >= 0 && u32(ih) < H && u32(iw) < W) {
          av = X[((n * C + ch) * H + u32(ih)) * W + u32(iw)];
        }
      }
      As[idx] = av;

      let bk = idx / 64u;
      let bc = idx % 64u;
      let brow = t * 16u + bk;
      let bcol = wid.x * 64u + bc;
      Bs[idx] = select(0.0, Wt[brow * p.F + bcol], brow < p.K && bcol < p.F);
    }
    workgroupBarrier();

    for (var k = 0u; k < 16u; k = k + 1u) {
      let a0 = As[(lid.y * 4u + 0u) * 16u + k];
      let a1 = As[(lid.y * 4u + 1u) * 16u + k];
      let a2 = As[(lid.y * 4u + 2u) * 16u + k];
      let a3 = As[(lid.y * 4u + 3u) * 16u + k];
      let b0 = Bs[k * 64u + lid.x * 4u + 0u];
      let b1 = Bs[k * 64u + lid.x * 4u + 1u];
      let b2 = Bs[k * 64u + lid.x * 4u + 2u];
      let b3 = Bs[k * 64u + lid.x * 4u + 3u];
${U4.fma}
    }
    workgroupBarrier();
  }

${U4.store.replace(/d\.M/g, "p.M").replace(/d\.N/g, "p.F").replace(/C\[/g, "Out[")}
}`;

const U = () => GPUBufferUsage;

function pipe(device, code) {
  return device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
  });
}

function bindOf(device, pipeline, buffers) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

async function readBack(device, buf, floats) {
  const out = device.createBuffer({ size: floats * 4, usage: U().COPY_DST | U().MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, out, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await out.mapAsync(GPUMapMode.READ);
  const arr = new Float32Array(out.getMappedRange()).slice();
  out.unmap();
  out.destroy();
  return arr;
}

window.wgslConv = async function (cases, iters) {
  const adapter = await navigator.gpu.requestAdapter();
  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
    },
  });
  const imPipe = pipe(device, IM2COL_WGSL);
  const mmPipe = pipe(device, MATMUL_WGSL);
  const fuPipe = pipe(device, FUSED_WGSL);
  const maxWg = lim.maxComputeWorkgroupsPerDimension;
  const lines = [`한계: 차원당 워크그룹 ${maxWg.toLocaleString()}개`, ""];

  for (const cs of cases) {
    const { N, C, H, W, F, KH, KW, S, P } = cs;
    const OH = Math.floor((H + 2 * P - KH) / S) + 1;
    const OW = Math.floor((W + 2 * P - KW) / S) + 1;
    const M = N * OH * OW;           // im2col 의 행
    const K = C * KH * KW;           // 안쪽 차원
    const flops = 2 * M * K * F;

    const x = new Float32Array(N * C * H * W);
    const w = new Float32Array(F * C * KH * KW);
    for (let i = 0; i < x.length; i++) x[i] = ((i * 37) % 19) / 19 - 0.5;
    for (let i = 0; i < w.length; i++) w[i] = ((i * 53) % 23) / 23 - 0.5;

    // --- TF.js 기준선. NHWC 로 넣어야 빠른 길을 탄다(실측: NCHW 는 1/7).
    const xNHWC = tf.tensor(x, [N, C, H, W]).transpose([0, 2, 3, 1]);
    const wHWIO = tf.tensor(w, [F, C, KH, KW]).transpose([2, 3, 1, 0]);
    let ref;
    {
      const o = tf.conv2d(xNHWC, wHWIO, [S, S], P === 0 ? "valid" : "same");
      ref = await o.data();
      o.dispose();
      const t0 = performance.now();
      for (let i = 0; i < iters; i++) {
        const r = tf.conv2d(xNHWC, wHWIO, [S, S], P === 0 ? "valid" : "same");
        if (i === iters - 1) await r.data();
        r.dispose();
      }
      cs.tfMs = (performance.now() - t0) / iters;
    }
    xNHWC.dispose();
    wHWIO.dispose();

    // --- 우리 커널. im2col 과 행렬곱을 따로 잰다.
    const mk = (n, usage) => device.createBuffer({ size: n * 4, usage });
    const bufX = mk(x.length, U().STORAGE | U().COPY_DST);
    const bufCols = mk(M * K, U().STORAGE | U().COPY_SRC);
    const bufP = device.createBuffer({ size: 48, usage: U().UNIFORM | U().COPY_DST });
    device.queue.writeBuffer(bufX, 0, x);
    device.queue.writeBuffer(bufP, 0, new Uint32Array(
      [N, C, H, W, KH, KW, OH, OW, S, S, P, P]));
    const imBind = bindOf(device, imPipe, [bufX, bufCols, bufP]);

    // 가중치는 (F, C·KH·KW) 로 두고 전치해서 (K, F) 로 쓴다 — 행렬곱이 B 를
    // 행 우선으로 읽으므로 여기서 한 번만 맞춰두면 된다.
    const wT = new Float32Array(K * F);
    for (let f = 0; f < F; f++) for (let k = 0; k < K; k++) wT[k * F + f] = w[f * K + k];
    const bufW = mk(K * F, U().STORAGE | U().COPY_DST);
    const bufOut = mk(M * F, U().STORAGE | U().COPY_SRC);
    const bufD = device.createBuffer({ size: 16, usage: U().UNIFORM | U().COPY_DST });
    device.queue.writeBuffer(bufW, 0, wT);
    device.queue.writeBuffer(bufD, 0, new Uint32Array([M, K, F, 0]));
    const mmBind = bindOf(device, mmPipe, [bufCols, bufW, bufOut, bufD]);

    const runIm = () => {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(imPipe);
      pass.setBindGroup(0, imBind);
      // 한계 안으로 자른다 — 남는 것은 셰이더의 격자 보폭 반복문이 가져간다.
      pass.dispatchWorkgroups(Math.min(Math.ceil(M * K / 64), maxWg));
      pass.end();
      device.queue.submit([enc.finish()]);
    };
    const runMm = () => {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(mmPipe);
      pass.setBindGroup(0, mmBind);
      const gx = Math.ceil(F / 64);
      const gy = Math.ceil(M / 64);
      // 행렬곱은 워크그룹 하나가 64×64 를 덮어 한계에 여유가 크다. 그래도 넘으면
      // **조용히 안 도므로** 짐작하지 않고 확인한다.
      if (gx > maxWg || gy > maxWg) throw new Error(`행렬곱 dispatch 가 한계를 넘는다: ${gx}×${gy}`);
      pass.dispatchWorkgroups(gx, gy);
      pass.end();
      device.queue.submit([enc.finish()]);
    };

    runIm(); runMm();
    await device.queue.onSubmittedWorkDone();

    let t0 = performance.now();
    for (let i = 0; i < iters; i++) runIm();
    await device.queue.onSubmittedWorkDone();
    const imMs = (performance.now() - t0) / iters;

    t0 = performance.now();
    for (let i = 0; i < iters; i++) runMm();
    await device.queue.onSubmittedWorkDone();
    const mmMs = (performance.now() - t0) / iters;

    // 값 대조. 우리 출력은 (M, F) = (N·OH·OW, F) 이고 TF.js 는 NHWC 라 같은 순서다.
    const got = await readBack(device, bufOut, M * F);
    let diff = 0;
    for (let i = 0; i < got.length; i++) diff = Math.max(diff, Math.abs(got[i] - ref[i]));

    // --- 융합 커널. 중간 버퍼 없이 입력에서 바로 읽는다.
    const bufFP = device.createBuffer({ size: 64, usage: U().UNIFORM | U().COPY_DST });
    device.queue.writeBuffer(bufFP, 0, new Uint32Array(
      [N, C, H, W, KH, KW, OH, OW, S, S, P, P, M, K, F, 0]));
    const bufFOut = mk(M * F, U().STORAGE | U().COPY_SRC);
    const fuBind = bindOf(device, fuPipe, [bufX, bufW, bufFOut, bufFP]);
    const runFused = () => {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(fuPipe);
      pass.setBindGroup(0, fuBind);
      pass.dispatchWorkgroups(Math.ceil(F / 64), Math.ceil(M / 64));
      pass.end();
      device.queue.submit([enc.finish()]);
    };
    runFused();
    await device.queue.onSubmittedWorkDone();
    t0 = performance.now();
    for (let i = 0; i < iters; i++) runFused();
    await device.queue.onSubmittedWorkDone();
    const fuMs = (performance.now() - t0) / iters;

    const fuGot = await readBack(device, bufFOut, M * F);
    let fuDiff = 0;
    for (let i = 0; i < fuGot.length; i++) {
      fuDiff = Math.max(fuDiff, Math.abs(fuGot[i] - ref[i]));
    }

    // --- 모양을 구워 넣은 융합. 셰이더 컴파일 시간은 재는 구간 밖에 둔다.
    const t1 = performance.now();
    const spPipe = pipe(device, fusedSpecialised({ ...cs, OH, OW }));
    const compileMs = performance.now() - t1;
    const bufSOut = mk(M * F, U().STORAGE | U().COPY_SRC);
    const bufSP = device.createBuffer({ size: 16, usage: U().UNIFORM | U().COPY_DST });
    device.queue.writeBuffer(bufSP, 0, new Uint32Array([N, M, K, F]));
    const spBind = bindOf(device, spPipe, [bufX, bufW, bufSOut, bufSP]);
    const runSp = () => {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(spPipe);
      pass.setBindGroup(0, spBind);
      pass.dispatchWorkgroups(Math.ceil(F / 64), Math.ceil(M / 64));
      pass.end();
      device.queue.submit([enc.finish()]);
    };
    runSp();
    await device.queue.onSubmittedWorkDone();
    t0 = performance.now();
    for (let i = 0; i < iters; i++) runSp();
    await device.queue.onSubmittedWorkDone();
    const spMs = (performance.now() - t0) / iters;

    const spGot = await readBack(device, bufSOut, M * F);
    let spDiff = 0;
    for (let i = 0; i < spGot.length; i++) {
      spDiff = Math.max(spDiff, Math.abs(spGot[i] - ref[i]));
    }

    const ours = imMs + mmMs;
    const g = (ms) => flops / (ms / 1000) / 1e9;
    lines.push(
      `${cs.name}  (N=${N} C=${C} ${H}×${W} → F=${F} ${KH}×${KW} s${S} p${P})` +
      `\n  TF.js conv   ${cs.tfMs.toFixed(2).padStart(7)} ms  ${g(cs.tfMs).toFixed(0).padStart(5)} GFLOPS` +
      `\n  우리 합계    ${ours.toFixed(2).padStart(7)} ms  ${g(ours).toFixed(0).padStart(5)} GFLOPS` +
      `  (TF.js 대비 ${(cs.tfMs / ours * 100).toFixed(0)}%)` +
      `\n    im2col     ${imMs.toFixed(2).padStart(7)} ms  (합계의 ${(imMs / ours * 100).toFixed(0)}%)` +
      `\n    행렬곱     ${mmMs.toFixed(2).padStart(7)} ms  ${g(mmMs).toFixed(0).padStart(5)} GFLOPS` +
      `\n    ${diff < 1e-3 ? "최대차 " + diff.toExponential(1) : "값 틀림! 최대차 " + diff.toExponential(1)}` +
      `\n  우리 융합    ${fuMs.toFixed(2).padStart(7)} ms  ${g(fuMs).toFixed(0).padStart(5)} GFLOPS` +
      `  (TF.js 대비 ${(cs.tfMs / fuMs * 100).toFixed(0)}%)` +
      `\n    ${fuDiff < 1e-3 ? "최대차 " + fuDiff.toExponential(1) : "값 틀림! 최대차 " + fuDiff.toExponential(1)}` +
      `\n  융합+상수    ${spMs.toFixed(2).padStart(7)} ms  ${g(spMs).toFixed(0).padStart(5)} GFLOPS` +
      `  (TF.js 대비 ${(cs.tfMs / spMs * 100).toFixed(0)}%)  컴파일 ${compileMs.toFixed(0)}ms` +
      `\n    ${spDiff < 1e-3 ? "최대차 " + spDiff.toExponential(1) : "값 틀림! 최대차 " + spDiff.toExponential(1)}`
    );

    [bufX, bufCols, bufP, bufW, bufOut, bufD, bufFP, bufFOut, bufSOut, bufSP]
      .forEach((b) => b.destroy());
  }
  return lines.join("\n");
};
