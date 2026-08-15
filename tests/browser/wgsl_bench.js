// WGSL 을 직접 쓰면 TF.js 만큼 나오는가 — 그것만 재는 파일이다.
//
// **지금 이 파일은 안 돈다.** 답이 "그렇다" 로 나와서 borch.ts 를 썼고, 그 뒤
// TF.js 판을 지우면서 벤더 파일(`vendor/tf.min.js`)도 안 받는다 — 기준선이 없다.
// 남겨 둔 것은 결론이 아니라 **결론을 낸 방법** 때문이다. 다시 돌리려면
// `tests/browser/vendor.py` 에 TF.js 를 되돌리면 된다(이력에 있다).
//
// 왜 이것을 재는가. 지금 자매 라이브러리는 TF.js 커널 위에 서 있고, 헤드라인
// ("ResNet-18 에폭 2분")도 거기서 나온다. WGSL 을 직접 쓰면 NCHW 강제도 CPU 왕복도
// 사라지지만, **행렬곱이 느려지면 그 헤드라인이 통째로 없어진다.** 그 하나가 방향을
// 정하므로 논쟁 대신 잰다.
//
// 재는 것 셋: TF.js `matMul`, 순진한 WGSL(스레드 하나가 출력 하나), 타일링 WGSL
// (16×16 워크그룹 공유메모리). **값이 맞는지 먼저 보고** 시간을 잰다 — 빠르고 틀린
// 커널은 아무 값어치가 없다.

const TILE = 16;

const NAIVE_WGSL = `
struct Dims { M: u32, K: u32, N: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= d.M || col >= d.N) { return; }
  var acc = 0.0;
  for (var k = 0u; k < d.K; k = k + 1u) {
    acc = acc + A[row * d.K + k] * B[k * d.N + col];
  }
  C[row * d.N + col] = acc;
}`;

// 타일링. 각 워크그룹이 A·B 의 16×16 조각을 공유메모리로 끌어와 재사용한다 —
// 순진한 판은 같은 값을 전역 메모리에서 몇 번이고 다시 읽는다.
const TILED_WGSL = `
struct Dims { M: u32, K: u32, N: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

var<workgroup> As: array<f32, ${TILE * TILE}>;
var<workgroup> Bs: array<f32, ${TILE * TILE}>;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let T = ${TILE}u;
  let row = wid.y * T + lid.y;
  let col = wid.x * T + lid.x;
  var acc = 0.0;
  let tiles = (d.K + T - 1u) / T;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let aCol = t * T + lid.x;
    let bRow = t * T + lid.y;
    As[lid.y * T + lid.x] = select(0.0, A[row * d.K + aCol], row < d.M && aCol < d.K);
    Bs[lid.y * T + lid.x] = select(0.0, B[bRow * d.N + col], bRow < d.K && col < d.N);
    workgroupBarrier();
    for (var k = 0u; k < T; k = k + 1u) {
      acc = acc + As[lid.y * T + k] * Bs[k * T + lid.x];
    }
    workgroupBarrier();
  }
  if (row < d.M && col < d.N) { C[row * d.N + col] = acc; }
}`;

// 레지스터 블로킹. 스레드 하나가 출력 **4×4** 를 맡는다.
//
// 단순 타일링만으로는 TF.js 의 20~26% 밖에 안 나왔다(실측). 차이는 스레드당 출력
// 개수에서 온다 — 하나만 맡으면 공유메모리에서 읽은 값을 한 번 쓰고 버리는데,
// 4×4 를 맡으면 A 조각 4개와 B 조각 4개를 읽어 곱셈 16번에 쓴다. 산술 강도가 4배다.
// **이 판이 천장인지 바닥인지를 갈라야** "WGSL 로 가도 되는가"에 답할 수 있다.
const RT = 4;                                   // 스레드당 출력 한 변
const BLOCKED_WGSL = `
struct Dims { M: u32, K: u32, N: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

const TS = ${TILE * RT}u;                       // 워크그룹이 맡는 출력 한 변 (64)
var<workgroup> As: array<f32, ${TILE * RT * TILE}>;   // 64 × 16
var<workgroup> Bs: array<f32, ${TILE * TILE * RT}>;   // 16 × 64

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let T = ${TILE}u;
  let R = ${RT}u;
  let row0 = wid.y * TS + lid.y * R;
  let col0 = wid.x * TS + lid.x * R;
  var acc: array<f32, ${RT * RT}>;
  for (var i = 0u; i < R * R; i = i + 1u) { acc[i] = 0.0; }

  let tiles = (d.K + T - 1u) / T;
  for (var t = 0u; t < tiles; t = t + 1u) {
    // A 조각 (64×16) 과 B 조각 (16×64) 을 워크그룹 256 스레드가 나눠 싣는다.
    for (var r = 0u; r < R; r = r + 1u) {
      let ar = lid.y * R + r;
      let arow = wid.y * TS + ar;
      let acol = t * T + lid.x;
      As[ar * T + lid.x] = select(0.0, A[arow * d.K + acol], arow < d.M && acol < d.K);

      let bc = lid.x * R + r;
      let brow = t * T + lid.y;
      let bcol = wid.x * TS + bc;
      Bs[lid.y * TS + bc] = select(0.0, B[brow * d.N + bcol], brow < d.K && bcol < d.N);
    }
    workgroupBarrier();

    for (var k = 0u; k < T; k = k + 1u) {
      var av: array<f32, ${RT}>;
      var bv: array<f32, ${RT}>;
      for (var i = 0u; i < R; i = i + 1u) { av[i] = As[(lid.y * R + i) * T + k]; }
      for (var j = 0u; j < R; j = j + 1u) { bv[j] = Bs[k * TS + lid.x * R + j]; }
      for (var i = 0u; i < R; i = i + 1u) {
        for (var j = 0u; j < R; j = j + 1u) {
          acc[i * R + j] = acc[i * R + j] + av[i] * bv[j];
        }
      }
    }
    workgroupBarrier();
  }

  for (var i = 0u; i < R; i = i + 1u) {
    for (var j = 0u; j < R; j = j + 1u) {
      let r = row0 + i;
      let c = col0 + j;
      if (r < d.M && c < d.N) { C[r * d.N + c] = acc[i * R + j]; }
    }
  }
}`;

// 레지스터 블로킹, **완전히 펼친 판.**
//
// 앞의 4×4 판은 오히려 4배 느렸다(값은 맞았다). 원인 가설: 누산기를 `array<f32,16>`
// 으로 두고 `acc[i * R + j]` 처럼 **변수로 인덱싱**했는데, WGSL 에서 그러면 레지스터에
// 못 두고 메모리로 떨어진다. 그러면 블로킹의 목적인 재사용이 통째로 사라진다.
//
// 그래서 여기서는 누산기 16개를 **이름 붙은 스칼라**로 두고 곱셈 16개를 손으로 펼친다.
// 읽기는 나쁘지만, 이 파일의 목적은 읽히는 것이 아니라 **850 이 내 한계인지
// WebGPU 의 한계인지 가르는 것**이다.
const unrolled = () => {
  const decl = [];
  const zero = [];
  const fma = [];
  const store = [];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      decl.push(`  var c${i}${j}: f32;`);
      zero.push(`  c${i}${j} = 0.0;`);
      fma.push(`      c${i}${j} = fma(a${i}, b${j}, c${i}${j});`);
      store.push(
        `  { let r = row0 + ${i}u; let c = col0 + ${j}u;` +
        ` if (r < d.M && c < d.N) { C[r * d.N + c] = c${i}${j}; } }`);
    }
  }
  return { decl: decl.join("\n"), zero: zero.join("\n"), fma: fma.join("\n"),
           store: store.join("\n") };
};
const U4 = unrolled();

const BLOCKED2_WGSL = `
struct Dims { M: u32, K: u32, N: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

// 워크그룹 16×16 스레드가 출력 64×64 를 맡는다. K 는 16 씩 민다.
var<workgroup> As: array<f32, 1024>;            // 64 행 × 16
var<workgroup> Bs: array<f32, 1024>;            // 16 × 64 열

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.y * 16u + lid.x;                // 0..255
  let row0 = wid.y * 64u + lid.y * 4u;
  let col0 = wid.x * 64u + lid.x * 4u;

${U4.decl}
${U4.zero}

  let tiles = (d.K + 15u) / 16u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    // 256 스레드가 1024 칸을 넷씩 나눠 싣는다. 한 번에 한 줄씩 훑어 인접 스레드가
    // 인접 주소를 읽게 둔다 — 합쳐진 읽기가 흩어진 읽기보다 훨씬 싸다.
    for (var s = 0u; s < 4u; s = s + 1u) {
      let idx = s * 256u + tid;                 // 0..1023
      let ar = idx / 16u;                       // A 조각의 행 (0..63)
      let ak = idx % 16u;
      let arow = wid.y * 64u + ar;
      let acol = t * 16u + ak;
      As[idx] = select(0.0, A[arow * d.K + acol], arow < d.M && acol < d.K);

      let bk = idx / 64u;                       // B 조각의 행 (0..15)
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

function makePipeline(device, code) {
  const module = device.createShaderModule({ code });
  return device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
}

async function runWgsl(device, pipeline, A, B, M, K, N, iters, perThread = 1) {
  const bytes = (n) => n * 4;
  const mk = (size, usage) => device.createBuffer({ size, usage });
  const U = GPUBufferUsage;

  const bufA = mk(bytes(M * K), U.STORAGE | U.COPY_DST);
  const bufB = mk(bytes(K * N), U.STORAGE | U.COPY_DST);
  const bufC = mk(bytes(M * N), U.STORAGE | U.COPY_SRC);
  const bufD = mk(16, U.UNIFORM | U.COPY_DST);
  device.queue.writeBuffer(bufA, 0, A);
  device.queue.writeBuffer(bufB, 0, B);
  device.queue.writeBuffer(bufD, 0, new Uint32Array([M, K, N, 0]));

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufC } },
      { binding: 3, resource: { buffer: bufD } },
    ],
  });

  const dispatch = () => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    const span = TILE * perThread;              // 워크그룹 하나가 덮는 출력 한 변
    pass.dispatchWorkgroups(Math.ceil(N / span), Math.ceil(M / span));
    pass.end();
    device.queue.submit([enc.finish()]);
  };

  dispatch();                                   // 워밍업 — 셰이더 컴파일을 시간에서 뺀다
  await device.queue.onSubmittedWorkDone();

  const t0 = performance.now();
  for (let i = 0; i < iters; i++) dispatch();
  await device.queue.onSubmittedWorkDone();
  const ms = (performance.now() - t0) / iters;

  // 값을 읽어온다 — 맞는지 봐야 시간이 뜻을 가진다.
  const read = device.createBuffer({ size: bytes(M * N), usage: U.COPY_DST | U.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(bufC, 0, read, 0, bytes(M * N));
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(read.getMappedRange()).slice();
  read.unmap();

  [bufA, bufB, bufC, bufD, read].forEach((b) => b.destroy());
  return { ms, out };
}

async function timeTfMatmul(A, B, M, K, N, iters) {
  const a = tf.tensor(A, [M, K]);
  const b = tf.tensor(B, [K, N]);
  let c = tf.matMul(a, b);
  await c.data();                               // 워밍업
  const ref = await c.data();
  c.dispose();

  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    const r = tf.matMul(a, b);
    if (i === iters - 1) await r.data();        // 마지막에만 동기화한다
    r.dispose();
  }
  const ms = (performance.now() - t0) / iters;
  a.dispose();
  b.dispose();
  return { ms, ref };
}

function maxDiff(x, y) {
  let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i] - y[i]));
  return m;
}

window.wgslBench = async function (shapes, iters) {
  const adapter = await navigator.gpu.requestAdapter();
  // **한계를 최대로 올려서 받는다.** 기본 maxStorageBufferBindingSize 는 128MB 이고,
  // 그것을 넘는 버퍼는 조용히 안 돈다 — 처음 재봤을 때 65536×576 행렬(151MB)에서
  // 24만 GFLOPS 라는 물리적으로 불가능한 수가 나왔고, 값도 틀려 있었다(최대차 1.9).
  // 값을 같이 안 봤으면 그 수를 믿을 뻔했다.
  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
    },
  });
  const naive = makePipeline(device, NAIVE_WGSL);
  const tiled = makePipeline(device, TILED_WGSL);
  const blocked = makePipeline(device, BLOCKED_WGSL);
  const blocked2 = makePipeline(device, BLOCKED2_WGSL);

  const lines = [
    `한계: 저장버퍼 ${(lim.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)}MB` +
    `  워크그룹메모리 ${(lim.maxComputeWorkgroupStorageSize / 1024).toFixed(0)}KB`,
    "",
  ];
  for (const [M, K, N] of shapes) {
    const need = Math.max(M * K, K * N, M * N) * 4;
    if (need > lim.maxStorageBufferBindingSize) {
      lines.push(`${M}×${K}×${N}  건너뜀 — 버퍼 ${(need / 2 ** 20).toFixed(0)}MB 가 한계를 넘는다`);
      continue;
    }
    const A = new Float32Array(M * K);
    const B = new Float32Array(K * N);
    // 값은 작게 둔다 — float32 누적 오차로 대조가 흐려지면 안 된다.
    for (let i = 0; i < A.length; i++) A[i] = ((i * 37) % 19) / 19 - 0.5;
    for (let i = 0; i < B.length; i++) B[i] = ((i * 53) % 23) / 23 - 0.5;

    const flops = 2 * M * K * N;
    const g = (ms) => flops / (ms / 1000) / 1e9;

    const t = await timeTfMatmul(A, B, M, K, N, iters);
    const runs = [
      ["WGSL 순진   ", await runWgsl(device, naive, A, B, M, K, N, iters)],
      ["WGSL 타일링 ", await runWgsl(device, tiled, A, B, M, K, N, iters)],
      ["WGSL 4×4블록", await runWgsl(device, blocked, A, B, M, K, N, iters, RT)],
      ["WGSL 펼침   ", await runWgsl(device, blocked2, A, B, M, K, N, iters, RT)],
    ];
    const rows = runs.map(([label, r]) => {
      const diff = maxDiff(r.out, t.ref);
      // **값이 틀리면 시간은 뜻이 없다.** 눈에 띄게 적는다.
      const verdict = diff < 1e-3 ? `최대차 ${diff.toExponential(1)}`
                                  : `값 틀림! 최대차 ${diff.toExponential(1)}`;
      return `\n  ${label} ${r.ms.toFixed(2).padStart(8)} ms  ${g(r.ms).toFixed(0).padStart(6)} GFLOPS` +
             `  (TF.js 대비 ${(t.ms / r.ms * 100).toFixed(0)}%)  ${verdict}`;
    });
    lines.push(
      `${M}×${K}×${N}` +
      `\n  TF.js        ${t.ms.toFixed(2).padStart(8)} ms  ${g(t.ms).toFixed(0).padStart(6)} GFLOPS` +
      rows.join("")
    );
  }
  return lines.join("\n");
};
