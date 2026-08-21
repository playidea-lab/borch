// Does writing WGSL directly match TF.js — this file measures that and nothing else.
//
// **This file does not run at present.** The answer came out "yes", borch.ts was written, and
// deleting the TF.js edition after that stopped the vendor file (`vendor/tf.min.js`) being
// fetched — there is no baseline. What is kept is not the conclusion but **the method that
// reached it.** To run it again, put TF.js back into `tests/browser/vendor.py` (it is in the
// history).
//
// Why measure this. The sister library stands on TF.js kernels today, and the headline ("a
// ResNet-18 epoch in 2 minutes") comes from there. Writing WGSL directly removes both the
// forced NCHW and the CPU round trip, but **if the matrix multiply gets slower that headline
// disappears whole.** That one thing decides the direction, so it is measured rather than
// argued.
//
// Three things are measured: TF.js `matMul`, naive WGSL (one thread per output), and tiled
// WGSL (a 16×16 workgroup with shared memory). **The values are checked first** and then the
// time — a fast, wrong kernel is worth nothing.

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

// Tiled. Each workgroup pulls a 16×16 piece of A and B into shared memory and reuses it —
// the naive edition rereads the same value from global memory again and again.
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

// Register blocking. One thread takes **4×4** outputs.
//
// Plain tiling alone reached only 20-26% of TF.js (measured). The difference comes from the
// outputs per thread — taking one means a value read from shared memory is used once and
// thrown away, while taking 4×4 means reading four pieces of A and four of B and using them
// across sixteen multiplies. Four times the arithmetic intensity.
// **Telling whether this edition is the ceiling or the floor** is what answers "can we go WGSL".
const RT = 4;                                   // outputs per thread, per side
const BLOCKED_WGSL = `
struct Dims { M: u32, K: u32, N: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

const TS = ${TILE * RT}u;                       // the output side a workgroup takes (64)
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
    // The workgroup's 256 threads share the load of A's piece (64×16) and B's (16×64).
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

// Register blocking, **fully unrolled.**
//
// The 4×4 edition above was four times slower instead (the values were right). Hypothesis: the
// accumulator was an `array<f32,16>` **indexed by a variable**, as `acc[i * R + j]`, and in
// WGSL that cannot stay in registers and falls into memory. Which loses the reuse that is
// blocking's whole purpose.
//
// So here the sixteen accumulators are **named scalars** and the sixteen multiplies are
// unrolled by hand. It reads badly, but this file's purpose is not to be read — it is to
// **tell whether 850 is my limit or WebGPU's.**
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

// A workgroup of 16×16 threads takes a 64×64 output. K advances by 16.
var<workgroup> As: array<f32, 1024>;            // 64 rows × 16
var<workgroup> Bs: array<f32, 1024>;            // 16 × 64 columns

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
    // 256 threads load 1024 slots, four each. Sweeping one row at a time keeps adjacent
    // threads on adjacent addresses — a coalesced read is far cheaper than a scattered one.
    for (var s = 0u; s < 4u; s = s + 1u) {
      let idx = s * 256u + tid;                 // 0..1023
      let ar = idx / 16u;                       // the row in A's piece (0..63)
      let ak = idx % 16u;
      let arow = wid.y * 64u + ar;
      let acol = t * 16u + ak;
      As[idx] = select(0.0, A[arow * d.K + acol], arow < d.M && acol < d.K);

      let bk = idx / 64u;                       // the row in B's piece (0..15)
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
    const span = TILE * perThread;              // the output side one workgroup covers
    pass.dispatchWorkgroups(Math.ceil(N / span), Math.ceil(M / span));
    pass.end();
    device.queue.submit([enc.finish()]);
  };

  dispatch();                                   // warm-up — keeps shader compilation out of the time
  await device.queue.onSubmittedWorkDone();

  const t0 = performance.now();
  for (let i = 0; i < iters; i++) dispatch();
  await device.queue.onSubmittedWorkDone();
  const ms = (performance.now() - t0) / iters;

  // Reads the values back — the time only means something once they are checked.
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
  await c.data();                               // warm-up
  const ref = await c.data();
  c.dispose();

  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    const r = tf.matMul(a, b);
    if (i === iters - 1) await r.data();        // synchronises on the last one only
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
  // **The limits are requested at their maximum.** The default maxStorageBufferBindingSize is
  // 128MB, and a buffer past it quietly does not run — the first measurement gave a physically
  // impossible 240,000 GFLOPS on a 65536×576 matrix (151MB), and the values were wrong too (a
  // maximum difference of 1.9). Without looking at the values alongside, that number was very
  // nearly believed.
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
    `limits: storage buffer ${(lim.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)}MB` +
    `  workgroup memory ${(lim.maxComputeWorkgroupStorageSize / 1024).toFixed(0)}KB`,
    "",
  ];
  for (const [M, K, N] of shapes) {
    const need = Math.max(M * K, K * N, M * N) * 4;
    if (need > lim.maxStorageBufferBindingSize) {
      lines.push(`${M}×${K}×${N}  skipped — a ${(need / 2 ** 20).toFixed(0)}MB buffer is past the limit`);
      continue;
    }
    const A = new Float32Array(M * K);
    const B = new Float32Array(K * N);
    // The values are kept small — float32 accumulation error must not blur the comparison.
    for (let i = 0; i < A.length; i++) A[i] = ((i * 37) % 19) / 19 - 0.5;
    for (let i = 0; i < B.length; i++) B[i] = ((i * 53) % 23) / 23 - 0.5;

    const flops = 2 * M * K * N;
    const g = (ms) => flops / (ms / 1000) / 1e9;

    const t = await timeTfMatmul(A, B, M, K, N, iters);
    const runs = [
      ["WGSL naive   ", await runWgsl(device, naive, A, B, M, K, N, iters)],
      ["WGSL tiled   ", await runWgsl(device, tiled, A, B, M, K, N, iters)],
      ["WGSL 4×4block", await runWgsl(device, blocked, A, B, M, K, N, iters, RT)],
      ["WGSL unrolled", await runWgsl(device, blocked2, A, B, M, K, N, iters, RT)],
    ];
    const rows = runs.map(([label, r]) => {
      const diff = maxDiff(r.out, t.ref);
      // **A wrong value makes the time meaningless.** Written where it will be noticed.
      const verdict = diff < 1e-3 ? `max diff ${diff.toExponential(1)}`
                                  : `WRONG VALUES! max diff ${diff.toExponential(1)}`;
      return `\n  ${label} ${r.ms.toFixed(2).padStart(8)} ms  ${g(r.ms).toFixed(0).padStart(6)} GFLOPS` +
             `  (${(t.ms / r.ms * 100).toFixed(0)}% of TF.js)  ${verdict}`;
    });
    lines.push(
      `${M}×${K}×${N}` +
      `\n  TF.js        ${t.ms.toFixed(2).padStart(8)} ms  ${g(t.ms).toFixed(0).padStart(6)} GFLOPS` +
      rows.join("")
    );
  }
  return lines.join("\n");
};
