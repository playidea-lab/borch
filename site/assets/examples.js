/**
 * The playground's examples.
 *
 * **Every piece of code here actually runs.** An example that is never run rots, and
 * that is precisely where a first user steps — this repository has twice caught
 * installation instructions in its own documentation that did not work. The order
 * follows the PRD's learning path (Tensor → Autograd → regression → MLP → CNN).
 *
 * Titles, descriptions and comments are written in both, as `{en, ko}`. **The code
 * itself has to be identical** — two languages running different code would make this
 * page's claim that the values agree a false one. Only the comments and the printed
 * wording differ.
 */

import { pick } from "./i18n.js";

const JS_EXAMPLES = [
  {
    id: "tensor",
    title: {
      en: "1 · Tensors — make, add, multiply",
      ko: "1 · 텐서 — 만들고, 더하고, 곱한다" },
    blurb: {
      en: "Same names as torch, same values. The one difference: reading a value is async.",
      ko: "torch 와 같은 이름, 같은 값. 다른 것은 값을 읽는 것이 비동기라는 것뿐이다." },
    code: {
      en: `// Acquiring the adapter is async, so this line comes first.
await init();

const a = Tensor.from([1, 2, 3, 4], [2, 2]);
const b = Tensor.from([10, 20, 30, 40], [2, 2]);

log("a =\\n" + await a.repr());
log("a + b =\\n" + await a.add(b).repr());
log("a @ b =\\n" + await a.mm(b).repr());

// Reading a value is awaited — it copies back from GPU memory.
log("a.mean() =", (await a.mean().item()).toFixed(4));
log("shape", JSON.stringify(a.shape), "· dtype", a.dtype, "· device", a.device);`,
      ko: `// borch 는 어댑터를 잡는 것이 비동기라 이 한 줄이 먼저다.
await init();

const a = Tensor.from([1, 2, 3, 4], [2, 2]);
const b = Tensor.from([10, 20, 30, 40], [2, 2]);

log("a =\\n" + await a.repr());
log("a + b =\\n" + await a.add(b).repr());
log("a @ b =\\n" + await a.mm(b).repr());

// 값을 읽는 것은 await 다 — GPU 메모리를 도로 가져오는 일이다.
log("a.mean() =", (await a.mean().item()).toFixed(4));
log("shape", JSON.stringify(a.shape), "· dtype", a.dtype, "· device", a.device);` },
  },
  {
    id: "autograd",
    title: {
      en: "2 · Autograd — gradients flow",
      ko: "2 · Autograd — 기울기가 흐른다" },
    blurb: {
      en: "Backward is synchronous. The smallest problem that shows it: pull w to 5.",
      ko: "역방향은 동기다. w 를 5 에 붙이는 가장 작은 문제로 그것을 본다." },
    code: {
      en: `await init();

// Same as the Python example in the README: loss = (w - 5)^2
const w = Tensor.from([3], [], { requiresGrad: true });
const five = Tensor.full([], 5);

const loss = w.sub(five).powScalar(2);
loss.backward();

log("loss   =", (await loss.item()).toFixed(4));
log("dL/dw  =", (await w.grad.item()).toFixed(4));
log("by hand = 2(w - 5) = 2(3 - 5) = -4");`,
      ko: `await init();

// README 의 파이썬 예시와 같은 것: loss = (w - 5)^2
const w = Tensor.from([3], [], { requiresGrad: true });
const five = Tensor.full([], 5);

const loss = w.sub(five).powScalar(2);
loss.backward();

log("loss   =", (await loss.item()).toFixed(4));
log("dL/dw  =", (await w.grad.item()).toFixed(4));
log("해석해  = 2(w - 5) = 2(3 - 5) = -4");` },
  },
  {
    id: "linreg",
    title: {
      en: "3 · Linear regression — recover y = 3x + 2",
      ko: "3 · 선형회귀 — y = 3x + 2 를 찾아낸다" },
    blurb: {
      en: "One layer, SGD, MSE. 200 steps and the coefficients show up. Loss curve is bottom right.",
      ko: "층 하나, SGD, MSE. 200 스텝이면 계수가 드러난다. 손실 곡선은 오른쪽 아래." },
    code: {
      en: `await init();
manualSeed(0);

// Data with a known answer — we want to see training actually find it.
const N = 128;
const xs = new Float32Array(N), ys = new Float32Array(N);
for (let i = 0; i < N; i++) {
  const x = (i / N) * 4 - 2;
  xs[i] = x;
  ys[i] = 3 * x + 2;
}

// Anything that must outlive a scope is marked with keepAlive.
const x = keepAlive(Tensor.from(xs, [N, 1]));
const y = keepAlive(Tensor.from(ys, [N, 1]));

const model = new nn.Linear(1, 1);
const opt = new optim.SGD(model.parameters(), 0.1);
const crit = new nn.MSELoss();

for (let step = 0; step <= 200; step++) {
  if (stopped()) break;
  // Release the scratch buffers one step makes. Without this the device fills up.
  await scope(async () => {
    opt.zeroGrad();
    const loss = crit.call(model.call(x), y);
    loss.backward();
    opt.step();

    const v = await loss.item();
    plot("loss", v);
    if (step % 40 === 0) log(\`step \${String(step).padStart(3)}   loss \${v.toFixed(6)}\`);
  });
}

log("");
log("w =", (await model.weight.item()).toFixed(4), "(should be 3)");
log("b =", (await model.bias.item()).toFixed(4), "(should be 2)");`,
      ko: `await init();
manualSeed(0);

// 정답이 있는 데이터를 만든다 — 학습이 실제로 그것을 찾는지 보려는 것이다.
const N = 128;
const xs = new Float32Array(N), ys = new Float32Array(N);
for (let i = 0; i < N; i++) {
  const x = (i / N) * 4 - 2;
  xs[i] = x;
  ys[i] = 3 * x + 2;
}

// 구역 밖에서 살아남아야 하는 것은 keepAlive 로 표시한다.
const x = keepAlive(Tensor.from(xs, [N, 1]));
const y = keepAlive(Tensor.from(ys, [N, 1]));

const model = new nn.Linear(1, 1);
const opt = new optim.SGD(model.parameters(), 0.1);
const crit = new nn.MSELoss();

for (let step = 0; step <= 200; step++) {
  if (stopped()) break;
  // 한 스텝이 만드는 중간 버퍼를 나갈 때 놓는다. 없으면 몇 스텝 만에 장치가 찬다.
  await scope(async () => {
    opt.zeroGrad();
    const loss = crit.call(model.call(x), y);
    loss.backward();
    opt.step();

    const v = await loss.item();
    plot("loss", v);
    if (step % 40 === 0) log(\`step \${String(step).padStart(3)}   loss \${v.toFixed(6)}\`);
  });
}

log("");
log("w =", (await model.weight.item()).toFixed(4), "(정답 3)");
log("b =", (await model.bias.item()).toFixed(4), "(정답 2)");` },
  },
  {
    id: "mlp",
    title: {
      en: "4 · MLP — the code written in the README",
      ko: "4 · MLP — README 에 적힌 그 코드" },
    blurb: {
      en: "784 → 128 → 10. The repository's `npm run example:ts` runs exactly this.",
      ko: "784 → 128 → 10. 저장소의 `npm run example:ts` 가 이것을 그대로 돌린다." },
    code: {
      en: `await init();

const model = new nn.Sequential(
  new nn.Linear(784, 128), new nn.ReLU(), new nn.Linear(128, 10));
const opt = new optim.SGD(model.parameters(), 0.05, 0.9);
const crit = new nn.CrossEntropyLoss();

// Where MNIST would go. These values only have the right shape —
// what we are watching is whether the loss goes down, not accuracy.
const pixels = new Float32Array(32 * 784);
for (let i = 0; i < pixels.length; i++) pixels[i] = (i % 17) / 17 - 0.5;
const labels = new Float32Array(32);
for (let i = 0; i < labels.length; i++) labels[i] = i % 10;

const x = keepAlive(Tensor.from(pixels, [32, 784]));
const y = keepAlive(Tensor.from(labels, [32], { dtype: "int64" }));

for (let i = 0; i < 100; i++) {
  if (stopped()) break;
  await scope(async () => {
    opt.zeroGrad();
    const loss = crit.call(model.call(x), y);
    loss.backward();
    opt.step();

    const v = await loss.item();
    plot("loss", v);
    if (i % 20 === 0) log(\`step \${String(i).padStart(3)}   loss \${v.toFixed(4)}\`);
  });
}
log("Done. Bottom right: the dispatches this loop issued and the memory it holds.");`,
      ko: `await init();

const model = new nn.Sequential(
  new nn.Linear(784, 128), new nn.ReLU(), new nn.Linear(128, 10));
const opt = new optim.SGD(model.parameters(), 0.05, 0.9);
const crit = new nn.CrossEntropyLoss();

// MNIST 가 들어갈 자리. 여기서는 모양만 맞춘 값을 쓴다 —
// 보려는 것은 정확도가 아니라 "손실이 내려가는가" 다.
const pixels = new Float32Array(32 * 784);
for (let i = 0; i < pixels.length; i++) pixels[i] = (i % 17) / 17 - 0.5;
const labels = new Float32Array(32);
for (let i = 0; i < labels.length; i++) labels[i] = i % 10;

const x = keepAlive(Tensor.from(pixels, [32, 784]));
const y = keepAlive(Tensor.from(labels, [32], { dtype: "int64" }));

for (let i = 0; i < 100; i++) {
  if (stopped()) break;
  await scope(async () => {
    opt.zeroGrad();
    const loss = crit.call(model.call(x), y);
    loss.backward();
    opt.step();

    const v = await loss.item();
    plot("loss", v);
    if (i % 20 === 0) log(\`step \${String(i).padStart(3)}   loss \${v.toFixed(4)}\`);
  });
}
log("끝. 오른쪽 아래 수가 이 루프가 건 dispatch 와 쥐고 있는 메모리다.");` },
  },
  {
    id: "cnn",
    title: {
      en: "5 · CNN — convolution, pooling, flatten",
      ko: "5 · CNN — 합성곱·풀링·평탄화" },
    blurb: {
      en: "One batch of 28×28 through the net. Convolution running on the browser's GPU.",
      ko: "28×28 한 배치를 통과시킨다. 브라우저 GPU 에서 도는 합성곱이다." },
    code: {
      en: `await init();
manualSeed(0);

const model = new nn.Sequential(
  new nn.Conv2d(1, 8, 3, 1, 1), new nn.ReLU(), new nn.MaxPool2d(2),
  new nn.Conv2d(8, 16, 3, 1, 1), new nn.ReLU(), new nn.MaxPool2d(2),
  new nn.Flatten(), new nn.Linear(16 * 7 * 7, 10));

const opt = new optim.Adam(model.parameters(), 1e-3);
const crit = new nn.CrossEntropyLoss();

const B = 16;
const x = keepAlive(Tensor.randn([B, 1, 28, 28]));
const labels = new Float32Array(B);
for (let i = 0; i < B; i++) labels[i] = i % 10;
const y = keepAlive(Tensor.from(labels, [B], { dtype: "int64" }));

for (let i = 0; i < 30; i++) {
  if (stopped()) break;
  await scope(async () => {
    opt.zeroGrad();
    const out = model.call(x);
    if (i === 0) log("logits shape:", JSON.stringify(out.shape));
    const loss = crit.call(out, y);
    loss.backward();
    opt.step();

    const v = await loss.item();
    plot("loss", v);
    if (i % 10 === 0) log(\`step \${String(i).padStart(3)}   loss \${v.toFixed(4)}\`);
  });
}`,
      ko: `await init();
manualSeed(0);

const model = new nn.Sequential(
  new nn.Conv2d(1, 8, 3, 1, 1), new nn.ReLU(), new nn.MaxPool2d(2),
  new nn.Conv2d(8, 16, 3, 1, 1), new nn.ReLU(), new nn.MaxPool2d(2),
  new nn.Flatten(), new nn.Linear(16 * 7 * 7, 10));

const opt = new optim.Adam(model.parameters(), 1e-3);
const crit = new nn.CrossEntropyLoss();

const B = 16;
const x = keepAlive(Tensor.randn([B, 1, 28, 28]));
const labels = new Float32Array(B);
for (let i = 0; i < B; i++) labels[i] = i % 10;
const y = keepAlive(Tensor.from(labels, [B], { dtype: "int64" }));

for (let i = 0; i < 30; i++) {
  if (stopped()) break;
  await scope(async () => {
    opt.zeroGrad();
    const out = model.call(x);
    if (i === 0) log("logits shape:", JSON.stringify(out.shape));
    const loss = crit.call(out, y);
    loss.backward();
    opt.step();

    const v = await loss.item();
    plot("loss", v);
    if (i % 10 === 0) log(\`step \${String(i).padStart(3)}   loss \${v.toFixed(4)}\`);
  });
}` },
  },
  {
    id: "bench",
    title: {
      en: "6 · Measure — what does this machine's GPU do",
      ko: "6 · 재보기 — 이 기계의 GPU 는 얼마나 나오나" },
    blurb: {
      en: "512×512 matmul. Wall clock, so do not trust this number on a software adapter.",
      ko: "512×512 행렬곱. 벽시계라 소프트웨어 어댑터에서는 이 수를 믿으면 안 된다." },
    code: {
      en: `await init();
log("adapter:", Device.adapterInfo);

const n = 512, R = 20;
const a = keepAlive(Tensor.randn([n, n]));
const b = keepAlive(Tensor.randn([n, n]));

// Warm up — the first call includes shader compilation.
await scope(async () => { a.mm(b); await device().synchronize(); });

const t0 = performance.now();
await scope(async () => {
  for (let i = 0; i < R; i++) a.mm(b);
  await device().synchronize();   // wait until the queue drains
});
const ms = (performance.now() - t0) / R;

log(\`\${n}×\${n} matmul — \${ms.toFixed(2)} ms each\`);
log(\`             \${(2 * n ** 3 / (ms * 1e6)).toFixed(1)} GFLOP/s\`);
log("");
log("If this looks oddly low, read the adapter name — on a software rasterizer");
log("that number belongs to the rasterizer, not to this library.");`,
      ko: `await init();
log("어댑터:", Device.adapterInfo);

const n = 512, R = 20;
const a = keepAlive(Tensor.randn([n, n]));
const b = keepAlive(Tensor.randn([n, n]));

// 예열 — 첫 회는 셰이더를 컴파일하는 시간이 섞인다.
await scope(async () => { a.mm(b); await device().synchronize(); });

const t0 = performance.now();
await scope(async () => {
  for (let i = 0; i < R; i++) a.mm(b);
  await device().synchronize();   // 명령이 다 끝날 때까지 기다린다
});
const ms = (performance.now() - t0) / R;

log(\`\${n}×\${n} 행렬곱 — \${ms.toFixed(2)} ms/회\`);
log(\`             \${(2 * n ** 3 / (ms * 1e6)).toFixed(1)} GFLOP/s\`);
log("");
log("이 수가 이상하게 낮으면 어댑터 이름을 보라 — 소프트웨어 래스터라이저면");
log("그것은 이 라이브러리의 수가 아니라 그 래스터라이저의 수다.");` },
  },
];

/**
 * The Python side — the same kernels, called through `borch_webgpu`.
 *
 * **No `await` appears.** Pyodide's `run_sync` (on JSPI) fills that place and the
 * binding hides the rest. So the code here, but for one import line and one `scope()`
 * line, has the shape of the textbook's PyTorch — which is this binding's only claim.
 */
const PY_EXAMPLES = [
  {
    id: "py-tensor",
    title: {
      en: "1 · Tensors — the same code, one import changed",
      ko: "1 · 텐서 — 임포트만 바꾼 그 코드" },
    blurb: {
      en: "torch becomes borch_webgpu. Underneath, the same WGSL kernels run.",
      ko: "torch 를 borch_webgpu 로 바꿔 부른다. 밑에서는 같은 WGSL 커널이 돈다." },
    code: {
      en: `import borch_webgpu as torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[10., 20.], [30., 40.]])

print("a =", a)
print("a + b =", a + b)
print("a @ b =", a @ b)
print("a.mean() =", a.mean().item())
print("shape", a.shape, "· dtype", a.dtype, "· device", a.device)`,
      ko: `import borch_webgpu as torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[10., 20.], [30., 40.]])

print("a =", a)
print("a + b =", a + b)
print("a @ b =", a @ b)
print("a.mean() =", a.mean().item())
print("shape", a.shape, "· dtype", a.dtype, "· device", a.device)` },
  },
  {
    id: "py-autograd",
    title: {
      en: "2 · Autograd — no await anywhere",
      ko: "2 · Autograd — await 이 없다" },
    blurb: {
      en: "WebGPU has no synchronous read, yet .item() just returns a value. run_sync fills that gap.",
      ko: "WebGPU 에 동기 읽기가 없는데도 .item() 이 그냥 값을 준다. run_sync 가 그 자리다." },
    code: {
      en: `import borch_webgpu as torch

w = torch.tensor(3.0, requires_grad=True)
loss = (w - 5.0) ** 2
loss.backward()

print("loss  =", loss.item())
print("dL/dw =", w.grad.item())
print("by hand = 2(w - 5) = -4")`,
      ko: `import borch_webgpu as torch

w = torch.tensor(3.0, requires_grad=True)
loss = (w - 5.0) ** 2
loss.backward()

print("loss  =", loss.item())
print("dL/dw =", w.grad.item())
print("해석해 = 2(w - 5) = -4")` },
  },
  {
    id: "py-linreg",
    title: {
      en: "3 · Linear regression — y = 3x + 2",
      ko: "3 · 선형회귀 — y = 3x + 2" },
    blurb: {
      en: "A PyTorch training loop as written — exactly one line added, with torch.scope().",
      ko: "PyTorch 학습 루프 그대로 — 딱 한 줄, with torch.scope() 만 늘었다." },
    code: {
      en: `import borch_webgpu as torch

nn, optim = torch.nn, torch.optim
torch.manual_seed(0)

N = 128
xs = [[(i / N) * 4 - 2] for i in range(N)]
ys = [[3 * row[0] + 2] for row in xs]
x = torch.tensor(xs)
y = torch.tensor(ys)

model = nn.Linear(1, 1)
opt = optim.SGD(model.parameters(), lr=0.1)
crit = nn.MSELoss()

for step in range(201):
    if stopped():
        break
    # Release the scratch buffers one step makes, or GPU memory keeps growing
    # (watch the number bottom right). Not a torch concept — the browser needs it.
    with torch.scope():
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

        plot("loss", loss.item())
        if step % 40 == 0:
            print(f"step {step:3d}   loss {loss.item():.6f}")

print("")
print("w =", round(model.weight.item(), 4), "(should be 3)")
print("b =", round(model.bias.item(), 4), "(should be 2)")`,
      ko: `import borch_webgpu as torch

nn, optim = torch.nn, torch.optim
torch.manual_seed(0)

N = 128
xs = [[(i / N) * 4 - 2] for i in range(N)]
ys = [[3 * row[0] + 2] for row in xs]
x = torch.tensor(xs)
y = torch.tensor(ys)

model = nn.Linear(1, 1)
opt = optim.SGD(model.parameters(), lr=0.1)
crit = nn.MSELoss()

for step in range(201):
    if stopped():
        break
    # 한 스텝이 만드는 중간 버퍼를 나갈 때 놓는다. 없으면 GPU 메모리가 계속 자란다
    # (오른쪽 아래 수를 보라). torch 에 없는 개념이고 브라우저라서 필요한 것이다.
    with torch.scope():
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

        plot("loss", loss.item())
        if step % 40 == 0:
            print(f"step {step:3d}   loss {loss.item():.6f}")

print("")
print("w =", round(model.weight.item(), 4), "(정답 3)")
print("b =", round(model.bias.item(), 4), "(정답 2)")` },
  },
  {
    id: "py-mlp",
    title: {
      en: "4 · MLP — 784 → 128 → 10",
      ko: "4 · MLP — 784 → 128 → 10" },
    blurb: {
      en: "Same model and same kernels as JavaScript #4. Only this surface differs.",
      ko: "자바스크립트 쪽 4번과 같은 모델, 같은 커널. 다른 것은 이 표면뿐이다." },
    code: {
      en: `import borch_webgpu as torch

nn, optim = torch.nn, torch.optim

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
crit = nn.CrossEntropyLoss()

# Where MNIST would go. These values only have the right shape.
pixels = [(i % 17) / 17 - 0.5 for i in range(32 * 784)]
x = torch.tensor(pixels).reshape(32, 784)
y = torch.tensor([i % 10 for i in range(32)]).long()

for i in range(100):
    if stopped():
        break
    with torch.scope():
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

        plot("loss", loss.item())
        if i % 20 == 0:
            print(f"step {i:3d}   loss {loss.item():.4f}")`,
      ko: `import borch_webgpu as torch

nn, optim = torch.nn, torch.optim

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
crit = nn.CrossEntropyLoss()

# MNIST 가 들어갈 자리. 여기서는 모양만 맞춘 값을 쓴다.
pixels = [(i % 17) / 17 - 0.5 for i in range(32 * 784)]
x = torch.tensor(pixels).reshape(32, 784)
y = torch.tensor([i % 10 for i in range(32)]).long()

for i in range(100):
    if stopped():
        break
    with torch.scope():
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

        plot("loss", loss.item())
        if i % 20 == 0:
            print(f"step {i:3d}   loss {loss.item():.4f}")` },
  },
  {
    id: "py-cnn",
    title: {
      en: "5 · CNN — convolution too",
      ko: "5 · CNN — 합성곱까지" },
    blurb: {
      en: "Calling convolution that runs on the browser's GPU, from Python.",
      ko: "브라우저 GPU 에서 도는 합성곱을 파이썬으로 부른다." },
    code: {
      en: `import borch_webgpu as torch

nn, optim = torch.nn, torch.optim
torch.manual_seed(0)

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

B = 16
x = torch.randn(B, 1, 28, 28)
y = torch.tensor([i % 10 for i in range(B)]).long()

for i in range(30):
    if stopped():
        break
    with torch.scope():
        opt.zero_grad()
        out = model(x)
        if i == 0:
            print("logits shape:", out.shape)
        loss = crit(out, y)
        loss.backward()
        opt.step()

        plot("loss", loss.item())
        if i % 10 == 0:
            print(f"step {i:3d}   loss {loss.item():.4f}")`,
      ko: `import borch_webgpu as torch

nn, optim = torch.nn, torch.optim
torch.manual_seed(0)

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

B = 16
x = torch.randn(B, 1, 28, 28)
y = torch.tensor([i % 10 for i in range(B)]).long()

for i in range(30):
    if stopped():
        break
    with torch.scope():
        opt.zero_grad()
        out = model(x)
        if i == 0:
            print("logits shape:", out.shape)
        loss = crit(out, y)
        loss.backward()
        opt.step()

        plot("loss", loss.item())
        if i % 10 == 0:
            print(f"step {i:3d}   loss {loss.item():.4f}")` },
  },
];

/** Folds `{en, ko}` down to the current language, so the rest of the code need not know about languages. */
function localize(list) {
  return list.map((ex) => ({
    id: ex.id,
    title: pick(ex.title),
    blurb: pick(ex.blurb),
    code: pick(ex.code),
  }));
}

/** The examples per language. The playground's picker reads this. */
export const EXAMPLES = { js: localize(JS_EXAMPLES), py: localize(PY_EXAMPLES) };

/** What is pinned in the hero — short, with few places to fail. */
export const HERO_CODE = pick({
  en: `await init();                       // acquire the WebGPU adapter

const w = Tensor.from([3], [], { requiresGrad: true });
const loss = w.sub(Tensor.full([], 5)).powScalar(2);
loss.backward();                    // backward is synchronous

log("loss  =", (await loss.item()).toFixed(2));
log("dL/dw =", (await w.grad.item()).toFixed(2));`,
  ko: `await init();                       // WebGPU 어댑터를 잡는다

const w = Tensor.from([3], [], { requiresGrad: true });
const loss = w.sub(Tensor.full([], 5)).powScalar(2);
loss.backward();                    // 역방향은 동기다

log("loss  =", (await loss.item()).toFixed(2));
log("dL/dw =", (await w.grad.item()).toFixed(2));`,
});
