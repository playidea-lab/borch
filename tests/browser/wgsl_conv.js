// Does conv reach TF.js in WGSL too?
//
// **This file does not run at present** — the same reason as `wgsl_bench.js`. It calls a TF.js
// baseline and that vendor file is no longer fetched. Kept as the method that reached the
// conclusion.
//
// For the matrix multiply, measurement put our kernel at 115-217% of TF.js (wgsl_bench.js).
// For conv the question is whether that result carries over — flattening with im2col and
// riding the matrix multiply carries it, and if it does not, where it leaks has to be found.
// Measured only in **the shapes ResNet-18 (CIFAR) actually uses.**
//
// Two pieces are measured separately: im2col itself and the matrix multiply after it. Measured
// only together, a slow result cannot say which of the two is the culprit.

const IM2COL_WGSL = `
struct P {
  N: u32, C: u32, H: u32, W: u32,
  KH: u32, KW: u32, OH: u32, OW: u32,
  SH: u32, SW: u32, PH: u32, PW: u32,
};
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Cols: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

// One output slot = one entry of the (C·KH·KW) vector at one (n, oh, ow).
//
// **A grid-stride loop is used.** One thread per slot puts the workgroup count past the
// per-dimension limit of 65,535 — 64 channels, 3×3, batch 64 needs 589,824. Past it, **it
// quietly does not run.** In the first measurement five of six had wrong values, and the one
// that was right was the stem, whose 3 channels made 27,648, under the limit. Looking at speed
// alone, "144% of TF.js" would have been believed as it stood.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let inner = p.C * p.KH * p.KW;
  let rows = p.N * p.OH * p.OW;
  let total = rows * inner;
  let stride = nwg.x * 64u;

  for (var idx = gid.x; idx < total; idx = idx + stride) {
    let r = idx / inner;                        // which (n, oh, ow)
    let c = idx % inner;                        // which entry within it

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

// The matrix multiply is the same design as the kernel that won in wgsl_bench.js — the
// accumulators are **unrolled** into sixteen named scalars. Put in an array and indexed by a
// variable they fall out of the registers and get 24 times slower (measured).
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

// Fused conv — **im2col is not run separately.**
//
// im2col lost on memory, not on arithmetic. At 3×3 it writes the input out ninefold and reads
// it back (151MB at a 64→64 layer). TF.js has no such round trip because its conv kernel reads
// the input directly, and the same is done here — **only the lines that load A's piece** in
// the matrix-multiply kernel above change, unpacking (n, oh, ow, c, kh, kw) on the spot and
// reading straight from the input.
//
// The rest (the tile size, the sixteen unrolled accumulators, the shared-memory reuse) is
// unchanged.
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

var<workgroup> As: array<f32, 1024>;            // 64 rows × 16
var<workgroup> Bs: array<f32, 1024>;            // 16 × 64 columns

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

      // --- A's piece: the im2col index is unpacked **here.** No intermediate buffer.
      let ar = idx / 16u;                       // 0..63, position within the output row
      let ak = idx % 16u;                       // 0..15, position within the K tile
      let arow = wid.y * 64u + ar;              // the full row = (n, oh, ow)
      let acol = t * 16u + ak;                  // the full column = (c, kh, kw)
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

      // --- B's piece: the weights are already (K, F), so they are read as they are.
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

// Fused conv, **with the shapes baked in as constants.**
//
// The fused kernel above got slower than the im2col edition as channels grew (half the speed
// at 512→512). The cause is not the read pattern but **division.** `p.OW`, `p.KW` and `p.C`
// are uniform values, so the compiler cannot turn the divisions into multiplies and shifts,
// and a GPU has no integer division hardware. Every tile load redoes seven divisions and
// remainders per element, and 512 channels at 3×3 means 288 tiles. im2col did the same
// divisions **once per element** and everyone read the flattened result.
//
// So the shapes go into the shader string. Each shape costs one more compilation — a real
// library caches shaders by a shape signature too — and every division disappears into
// constant folding.
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
        // Every divisor is a constant, so these divisions fold into multiplies and shifts.
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
  const lines = [`limit: ${maxWg.toLocaleString()} workgroups per dimension`, ""];

  for (const cs of cases) {
    const { N, C, H, W, F, KH, KW, S, P } = cs;
    const OH = Math.floor((H + 2 * P - KH) / S) + 1;
    const OW = Math.floor((W + 2 * P - KW) / S) + 1;
    const M = N * OH * OW;           // im2col's rows
    const K = C * KH * KW;           // the inner dimension
    const flops = 2 * M * K * F;

    const x = new Float32Array(N * C * H * W);
    const w = new Float32Array(F * C * KH * KW);
    for (let i = 0; i < x.length; i++) x[i] = ((i * 37) % 19) / 19 - 0.5;
    for (let i = 0; i < w.length; i++) w[i] = ((i * 53) % 23) / 23 - 0.5;

    // --- The TF.js baseline. It takes the fast path only in NHWC (measured: NCHW is 1/7).
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

    // --- Our kernels. im2col and the matrix multiply are measured separately.
    const mk = (n, usage) => device.createBuffer({ size: n * 4, usage });
    const bufX = mk(x.length, U().STORAGE | U().COPY_DST);
    const bufCols = mk(M * K, U().STORAGE | U().COPY_SRC);
    const bufP = device.createBuffer({ size: 48, usage: U().UNIFORM | U().COPY_DST });
    device.queue.writeBuffer(bufX, 0, x);
    device.queue.writeBuffer(bufP, 0, new Uint32Array(
      [N, C, H, W, KH, KW, OH, OW, S, S, P, P]));
    const imBind = bindOf(device, imPipe, [bufX, bufCols, bufP]);

    // The weights are held as (F, C·KH·KW) and transposed to (K, F) — the matrix multiply
    // reads B row-major, so lining it up once here is enough.
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
      // Clamped under the limit — the shader's grid-stride loop takes the remainder.
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
      // For the matrix multiply one workgroup covers 64×64, so there is plenty of headroom.
      // Past the limit it still **quietly does not run**, so it is checked rather than assumed.
      if (gx > maxWg || gy > maxWg) throw new Error(`the matmul dispatch is past the limit: ${gx}×${gy}`);
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

    // Value comparison. Our output is (M, F) = (N·OH·OW, F) and TF.js is NHWC, the same order.
    const got = await readBack(device, bufOut, M * F);
    let diff = 0;
    for (let i = 0; i < got.length; i++) diff = Math.max(diff, Math.abs(got[i] - ref[i]));

    // --- The fused kernel. Reads straight from the input with no intermediate buffer.
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

    // --- The shape-baked fusion. Shader compilation is kept outside the measured window.
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
      `\n  ours, total  ${ours.toFixed(2).padStart(7)} ms  ${g(ours).toFixed(0).padStart(5)} GFLOPS` +
      `  (${(cs.tfMs / ours * 100).toFixed(0)}% of TF.js)` +
      `\n    im2col     ${imMs.toFixed(2).padStart(7)} ms  (${(imMs / ours * 100).toFixed(0)}% of the total)` +
      `\n    matmul     ${mmMs.toFixed(2).padStart(7)} ms  ${g(mmMs).toFixed(0).padStart(5)} GFLOPS` +
      `\n    ${diff < 1e-3 ? "max diff " + diff.toExponential(1) : "WRONG VALUES! max diff " + diff.toExponential(1)}` +
      `\n  ours, fused  ${fuMs.toFixed(2).padStart(7)} ms  ${g(fuMs).toFixed(0).padStart(5)} GFLOPS` +
      `  (${(cs.tfMs / fuMs * 100).toFixed(0)}% of TF.js)` +
      `\n    ${fuDiff < 1e-3 ? "max diff " + fuDiff.toExponential(1) : "WRONG VALUES! max diff " + fuDiff.toExponential(1)}` +
      `\n  fused+const  ${spMs.toFixed(2).padStart(7)} ms  ${g(spMs).toFixed(0).padStart(5)} GFLOPS` +
      `  (${(cs.tfMs / spMs * 100).toFixed(0)}% of TF.js)  compile ${compileMs.toFixed(0)}ms` +
      `\n    ${spDiff < 1e-3 ? "max diff " + spDiff.toExponential(1) : "WRONG VALUES! max diff " + spDiff.toExponential(1)}`
    );

    [bufX, bufCols, bufP, bufW, bufOut, bufD, bufFP, bufFOut, bufSOut, bufSP]
      .forEach((b) => b.destroy());
  }
  return lines.join("\n");
};
