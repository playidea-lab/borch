/**
 * 데이터셋과 적재기 — 배치가 맞게 나오는가, 섞기가 씨앗을 따르는가.
 *
 * **빠른 길과 느린 길이 같은 답을 내야 한다.** `TensorDataset` 은 배치를 한 번에
 * 뽑고(`narrow`·`indexSelect`), `gather` 가 없는 데이터셋은 하나씩 꺼내 쌓는다.
 * 둘이 갈리면 **빠른 쪽만 틀리고 아무도 못 본다** — 값이 그럴듯하게 나오기 때문이다.
 * 그래서 여기서는 두 길을 나란히 돌려 견준다.
 */

import {
  data, device, init, keepAlive, manualSeed, nn, optim, scope, Tensor,
} from "../src/index.js";

interface Check { name: string; ok: boolean; note: string }
const checks: Check[] = [];

function want(name: string, ok: boolean, note = ""): void {
  checks.push({ name, ok, note });
}

function wantThrow(name: string, body: () => unknown): void {
  try {
    body();
    want(name, false, "안 던졌다");
  } catch {
    want(name, true);
  }
}

/** `gather` 가 없는 데이터셋. 적재기가 느린 길로 가는 것을 강제한다. */
class Plain implements data.Dataset {
  constructor(private readonly inner: data.Dataset) {}
  get length(): number { return this.inner.length; }
  get(index: number): readonly Tensor[] { return this.inner.get(index); }
}

const ROWS = 10;
const WIDTH = 3;

/** 값이 자리를 말해 주는 데이터. `x[i][j] === i * 10 + j` 라 섞여도 추적된다. */
function makeSet(): data.TensorDataset {
  const xs = new Float32Array(ROWS * WIDTH);
  for (let i = 0; i < ROWS; i++) {
    for (let j = 0; j < WIDTH; j++) xs[i * WIDTH + j] = i * 10 + j;
  }
  const ys = Float32Array.from({ length: ROWS }, (_, i) => i);
  return new data.TensorDataset(
    keepAlive(Tensor.from(xs, [ROWS, WIDTH])),
    keepAlive(Tensor.from(ys, [ROWS], { dtype: "int64" })),
  );
}

/** 배치를 전부 훑어 첫 열의 값을 모은다 — 어떤 표본이 어느 차례로 왔는지. */
async function rowsOf(loader: data.DataLoader): Promise<number[]> {
  const seen: number[] = [];
  for (const [x] of loader) {
    const flat = await (x as Tensor).toArray();
    for (let i = 0; i < flat.length; i += WIDTH) seen.push((flat[i] as number) / 10);
  }
  return seen;
}

export async function report(): Promise<string> {
  await init();
  const set = makeSet();

  // ── 데이터셋 ──────────────────────────────────────────────────────────
  want("길이가 표본 수다", set.length === ROWS, `${set.length}`);
  const item = set.get(3);
  want("한 표본은 첫 축이 빠진다",
    item.length === 2 && item[0]!.shape.join(",") === `${WIDTH}`
      && item[1]!.shape.length === 0,
    `${item[0]!.shape} / ${item[1]!.shape}`);
  want("표본의 값이 맞다", (await item[0]!.toArray())[0] === 30);
  // **이 검사가 결함 하나를 꺼냈다.** 모양·색인 연산이 형 이름표를 잃고 있었다 —
  // `select`·`narrow`·`indexSelect` 가 `Tensor.make` 를 dtype 없이 불렀고 그 기본값이
  // float32 였다. int64 라벨에서 표본을 꺼내면 float32 로 붙어 나왔고, 값은 맞고
  // 이름만 갈리는 종류라 눈으로만 잡힌다.
  //
  // 밑동에서 고쳐졌으므로 여기 있던 되붙이기(`keepLabel`)는 지웠다. **검사는 남긴다** —
  // 계약을 쥐는 것이 방어 코드가 아니라 이 세 줄이어야 한다. 밑이 되돌아가면 여기가
  // 빨개진다.
  want("표본이 dtype 이름표를 지킨다", item[1]!.dtype === "int64", item[1]!.dtype);
  want("이어진 배치도 이름표를 지킨다",
    set.gather([0, 1, 2])[1]!.dtype === "int64", set.gather([0, 1, 2])[1]!.dtype);
  want("흩어진 배치도 이름표를 지킨다",
    set.gather([3, 1, 2])[1]!.dtype === "int64", set.gather([3, 1, 2])[1]!.dtype);

  // 표본 수가 다른 텐서를 묶으면 라벨이 조용히 밀린다. 거기서 멈춰야 한다.
  wantThrow("표본 수가 다르면 거절한다", () => new data.TensorDataset(
    Tensor.from([1, 2, 3], [3]), Tensor.from([1, 2], [2])));
  wantThrow("텐서가 없으면 거절한다", () => new data.TensorDataset());

  // ── 배치 세기 ─────────────────────────────────────────────────────────
  want("배치 수가 올림이다",
    new data.DataLoader(set, { batchSize: 3 }).length === 4);
  want("dropLast 는 내림이다",
    new data.DataLoader(set, { batchSize: 3, dropLast: true }).length === 3);
  want("length 는 표본이 아니라 배치를 센다",
    new data.DataLoader(set, { batchSize: 10 }).length === 1);
  wantThrow("batchSize 0 을 거절한다",
    () => new data.DataLoader(set, { batchSize: 0 }));

  // ── 순서와 내용 ───────────────────────────────────────────────────────
  const straight = await rowsOf(new data.DataLoader(set, { batchSize: 3 }));
  want("안 섞으면 차례대로 전부 나온다",
    straight.join(",") === "0,1,2,3,4,5,6,7,8,9", straight.join(","));

  const dropped = await rowsOf(
    new data.DataLoader(set, { batchSize: 3, dropLast: true }));
  want("dropLast 가 남는 것을 버린다",
    dropped.join(",") === "0,1,2,3,4,5,6,7,8", dropped.join(","));

  // ── 빠른 길과 느린 길 ─────────────────────────────────────────────────
  // **이것이 이 러너의 이유다.** `gather` 가 있는 쪽과 없는 쪽이 갈리면 빠른 쪽만
  // 틀리고, 값이 그럴듯해서 안 보인다.
  manualSeed(5);
  const fast = await rowsOf(new data.DataLoader(set, { batchSize: 4, shuffle: true }));
  manualSeed(5);
  const slow = await rowsOf(
    new data.DataLoader(new Plain(set), { batchSize: 4, shuffle: true }));
  want("빠른 길과 느린 길이 같은 답을 낸다",
    fast.join(",") === slow.join(","), `${fast.join(",")} / ${slow.join(",")}`);

  // 섞어도 표본은 전부 한 번씩이어야 한다.
  want("섞어도 빠짐도 겹침도 없다",
    [...fast].sort((a, b) => a - b).join(",") === "0,1,2,3,4,5,6,7,8,9",
    fast.join(","));

  // ── 섞기가 씨앗을 따르는가 ────────────────────────────────────────────
  const draw = async (seed: number): Promise<string> => {
    manualSeed(seed);
    return (await rowsOf(new data.DataLoader(set, { batchSize: 4, shuffle: true })))
      .join(",");
  };
  want("같은 씨앗이면 같은 차례", (await draw(9)) === (await draw(9)));
  want("다른 씨앗이면 다른 차례", (await draw(1)) !== (await draw(2)));

  // **에폭마다 다시 섞어야 한다.** 생성자에서 한 번만 정하면 두 번째 에폭이 첫
  // 번째와 같은 차례로 돌고, 섞는 이유가 사라진다.
  manualSeed(3);
  const loader = new data.DataLoader(set, { batchSize: 4, shuffle: true });
  const epoch1 = (await rowsOf(loader)).join(",");
  const epoch2 = (await rowsOf(loader)).join(",");
  want("에폭마다 다시 섞는다", epoch1 !== epoch2, `${epoch1} / ${epoch2}`);

  // ── 나누기 ────────────────────────────────────────────────────────────
  manualSeed(11);
  const [train, valid] = data.randomSplit(set, [7, 3]);
  want("나눈 길이가 맞다", train!.length === 7 && valid!.length === 3);
  const trainRows = await rowsOf(new data.DataLoader(train!, { batchSize: 7 }));
  const validRows = await rowsOf(new data.DataLoader(valid!, { batchSize: 3 }));
  want("나눈 것이 안 겹치고 전부를 덮는다",
    [...trainRows, ...validRows].sort((a, b) => a - b).join(",")
      === "0,1,2,3,4,5,6,7,8,9",
    `${trainRows.join(",")} | ${validRows.join(",")}`);
  wantThrow("합이 안 맞으면 거절한다", () => data.randomSplit(set, [5, 3]));

  // Subset 도 빠른 길을 타야 한다 — 학습·검증을 나누는 것이 예사로운 일이라
  // 여기서 느린 길로 떨어지면 대부분의 학습이 느린 길로 간다.
  manualSeed(4);
  const subFast = await rowsOf(new data.DataLoader(train!, { batchSize: 3, shuffle: true }));
  manualSeed(4);
  const subSlow = await rowsOf(
    new data.DataLoader(new Plain(train!), { batchSize: 3, shuffle: true }));
  want("Subset 도 두 길이 같다", subFast.join(",") === subSlow.join(","),
    `${subFast.join(",")} / ${subSlow.join(",")}`);

  // ── 이어 붙이기 ───────────────────────────────────────────────────────
  const joined = new data.ConcatDataset([set, set]);
  want("이어 붙인 길이가 합이다", joined.length === ROWS * 2);
  want("이어 붙인 뒤쪽이 두 번째 것을 가리킨다",
    (await joined.get(ROWS + 2)[0]!.toArray())[0] === 20);

  // ── 실제로 학습이 도는가 ──────────────────────────────────────────────
  //
  // **"내려갔다" 만 물으면 공짜로 통과한다.** 처음에 위의 `set` 을 그대로 썼는데
  // 특징이 0~92 라 첫 에폭만 크고 곧장 우연 수준(클래스 10 개면 ln 10 = 2.303)에
  // 붙어 버렸다. 그 상태에서도 `마지막 < 처음` 은 참이라 검사가 초록이었다.
  //
  // 그래서 배울 수 있는 문제를 따로 만들고, **우연보다 뚜렷이 낮은 곳까지 가는지**를
  // 묻는다. 적재기가 라벨을 밀어 놓으면 여기가 우연에서 안 내려간다.
  const CLASSES = 3;
  const SAMPLES = 60;
  const feats = new Float32Array(SAMPLES * WIDTH);
  const labels = new Float32Array(SAMPLES);
  for (let i = 0; i < SAMPLES; i++) {
    const cls = i % CLASSES;
    // 클래스마다 한 축만 세워 둔다 — 선형으로 갈릴 수 있는 문제여야 적재기가
    // 라벨을 제대로 붙이는지가 손실에 드러난다.
    for (let j = 0; j < WIDTH; j++) feats[i * WIDTH + j] = j === cls ? 1 : -1;
    labels[i] = cls;
  }
  const learnable = new data.TensorDataset(
    keepAlive(Tensor.from(feats, [SAMPLES, WIDTH])),
    keepAlive(Tensor.from(labels, [SAMPLES], { dtype: "int64" })),
  );

  manualSeed(7);
  const model = new nn.Sequential(new nn.Linear(WIDTH, 8), new nn.ReLU(),
    new nn.Linear(8, CLASSES));
  for (const p of model.parameters()) keepAlive(p);
  const opt = new optim.Adam(model.parameters(), 0.05);
  const crit = new nn.CrossEntropyLoss();
  const training = new data.DataLoader(learnable, { batchSize: 10, shuffle: true });
  const losses: number[] = [];
  for (let epoch = 0; epoch < 12; epoch++) {
    let total = 0;
    for (const [x, y] of training) {
      await scope(async () => {
        opt.zeroGrad();
        const loss = crit.call(model.call(x as Tensor), y as Tensor);
        loss.backward();
        opt.step();
        total += await loss.item();
      });
    }
    losses.push(total / training.length);
  }
  const chance = Math.log(CLASSES);
  const last = losses[losses.length - 1] ?? NaN;
  want("적재기로 돌린 학습이 우연 밑으로 내려간다", last < chance * 0.3,
    `${losses.map((v) => v.toFixed(3)).join(" → ")} (우연 ${chance.toFixed(3)})`);

  // 값이 안 바뀌는데 초록으로 읽히는 자리를 막는다.
  want("WebGPU 검증 오류가 없다", device().faults.count === 0,
    device().faults.first);

  const failed = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push(failed.length === 0
    ? `데이터 적재 ${checks.length}건 전부 통과`
    : `**${failed.length}건 실패** / ${checks.length}건`);
  return lines.join("\n");
}
