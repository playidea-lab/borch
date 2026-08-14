// conv 도 WGSL 로 TF.js 만큼 나오는가.
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

    const ours = imMs + mmMs;
    const g = (ms) => flops / (ms / 1000) / 1e9;
    lines.push(
      `${cs.name}  (N=${N} C=${C} ${H}×${W} → F=${F} ${KH}×${KW} s${S} p${P})` +
      `\n  TF.js conv   ${cs.tfMs.toFixed(2).padStart(7)} ms  ${g(cs.tfMs).toFixed(0).padStart(5)} GFLOPS` +
      `\n  우리 합계    ${ours.toFixed(2).padStart(7)} ms  ${g(ours).toFixed(0).padStart(5)} GFLOPS` +
      `  (TF.js 대비 ${(cs.tfMs / ours * 100).toFixed(0)}%)` +
      `\n    im2col     ${imMs.toFixed(2).padStart(7)} ms  (합계의 ${(imMs / ours * 100).toFixed(0)}%)` +
      `\n    행렬곱     ${mmMs.toFixed(2).padStart(7)} ms  ${g(mmMs).toFixed(0).padStart(5)} GFLOPS` +
      `\n  ${diff < 1e-3 ? "최대차 " + diff.toExponential(1) : "값 틀림! 최대차 " + diff.toExponential(1)}`
    );

    [bufX, bufCols, bufP, bufW, bufOut, bufD].forEach((b) => b.destroy());
  }
  return lines.join("\n");
};
