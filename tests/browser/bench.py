"""ResNet-18(CIFAR 판)로 **진짜 학습 스텝**을 잰다.

층 하나로 잰 값은 BN·ReLU·잔차 덧셈처럼 대역폭에 묶인 연산을 안 센다. 목표
("에폭 몇 분")에 닿았는지는 그것들까지 포함한 실제 스텝에서만 나온다 — 그래서
FLOPs 로 나눈 추정 대신 이 파일이 있다.

브라우저 안에서 돈다:

    uv run --with playwright python tests/browser/run.py --lib borch_webgpu --headed --bench
"""

import asyncio
import gc as _gc

import js
import numpy as np

import borchvision as vision

CIFAR_TRAIN_IMAGES = 50000
# CIFAR-10 의 통상값. 정규화를 빼면 첫 에폭이 눈에 띄게 느리게 붙는다.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
_shuffle = np.random.default_rng(0)


def _block(L, cin, cout, stride):
    """ResNet 의 기본 블록. 지름길(shortcut)이 모양을 바꿔야 할 때만 1×1 을 둔다."""

    class Block(L.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = L.nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.bn1 = L.nn.BatchNorm2d(cout)
            self.conv2 = L.nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
            self.bn2 = L.nn.BatchNorm2d(cout)
            self.shrinks = stride != 1 or cin != cout
            if self.shrinks:
                self.dconv = L.nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
                self.dbn = L.nn.BatchNorm2d(cout)

        def forward(self, x):
            out = L.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            side = self.dbn(self.dconv(x)) if self.shrinks else x
            return L.relu(out + side)

    return Block()


def resnet18(L, num_classes=10):
    """CIFAR 판 — 3×3 스템에 맥스풀이 없다. 32×32 를 7×7 스템으로 받으면 너무 줄어든다."""

    class Net(L.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = L.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            self.bn = L.nn.BatchNorm2d(64)
            self.body = L.nn.Sequential(
                _block(L, 64, 64, 1), _block(L, 64, 64, 1),
                _block(L, 64, 128, 2), _block(L, 128, 128, 1),
                _block(L, 128, 256, 2), _block(L, 256, 256, 1),
                _block(L, 256, 512, 2), _block(L, 512, 512, 1))
            self.pool = L.nn.AdaptiveAvgPool2d(1)
            self.fc = L.nn.Linear(512, num_classes)

        def forward(self, x):
            x = L.relu(self.bn(self.stem(x)))
            x = self.pool(self.body(x))
            return self.fc(x.flatten(1))

    return Net()


def run(L, batch=32, steps=5, warmup=2):
    """한 스텝의 벽시계 시간을 잰다. (결과 dict)"""
    rng = np.random.default_rng(0)
    x = L.tensor(rng.standard_normal((batch, 3, 32, 32)).astype(np.float32))
    y = L.tensor(rng.integers(0, 10, batch).astype(np.int64))

    model = resnet18(L)
    n_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    opt = L.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = L.nn.CrossEntropyLoss()

    def one():
        with L.scope():
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            return loss.item()          # 스코프 안에서 읽어야 한다

    for _ in range(warmup):
        one()

    # **기준선을 잡기 전에 파이썬 쓰레기를 먼저 치운다.**
    #
    # 안 그러면 워밍업이 남긴 것이 측정 창 안에서 뒤늦게 풀리고, 아래 나눗셈이 그
    # 한 번의 정리를 **스텝당 비율로 둔갑시킨다.** 실제로 배치 32 에서 "누수 -24.8"
    # 이 나왔는데, 스텝 수를 5·10·20 으로 바꿔 보니 총량이 늘 0 아니면 정확히 -124 였다 —
    # 스텝당이 아니라 한 번짜리라는 뜻이다. 음수라 새는 쪽은 아니었지만, 이름이
    # 말하는 것과 다른 수를 내는 계측은 다음에 진짜 누수를 만났을 때도 못 믿는다.
    _gc.collect()
    # **계측을 라이브러리에 묻는다.** 여기서 `js.tf.memory()` 를 직접 불렀었고, 그래서
    # 이 벤치가 TF.js 에 묶여 다른 구현으로는 못 돌았다 — 두 밑바닥을 같은 잣대로
    # 재려다 이 세 줄에서 막혔다. 같은 잣대라는 말이 성립하려면 잣대가 한쪽 밑바닥을
    # 알면 안 된다.
    before = L.memory()["tensors"]
    t0 = js.performance.now()
    last = None
    for _ in range(steps):
        last = one()
    per_step = (js.performance.now() - t0) / steps
    leak = (L.memory()["tensors"] - before) / steps

    steps_per_epoch = -(-CIFAR_TRAIN_IMAGES // batch)
    return {
        "batch": batch,
        "params": n_params,
        "ms_per_step": round(per_step, 1),
        "epoch_min": round(per_step * steps_per_epoch / 60000, 2),
        "leak_per_step": round(leak, 1),
        "loss": round(float(last), 4),
        "gpu_mb": round(L.memory()["bytes"] / 1e6, 1),
    }


def run_with_loader(L, batch=128, images=5120, steps=5, warmup=2):
    """DataLoader 를 거쳐 잰다 — **배치마다 CPU→GPU 업로드가 붙는다.**

    위의 `run` 은 같은 텐서를 계속 쓴다. 실제 학습은 배치마다 새 데이터를 올리므로,
    그 값이 스텝 시간에 얼마나 붙는지는 따로 재야 안다.

    이미지 수를 줄여 둔 것은 의도다 — 5만 장을 float32 로 들면 614MB 이고,
    그건 이 측정이 묻는 것(업로드 비용)이 아니라 파이썬 힙을 묻게 된다.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((images, 3, 32, 32)).astype(np.float32)
    y = rng.integers(0, 10, images).astype(np.int64)
    loader = L.utils.data.DataLoader(
        L.utils.data.TensorDataset(x, y), batch_size=batch, shuffle=True, drop_last=True)

    model = resnet18(L)
    opt = L.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = L.nn.CrossEntropyLoss()

    def go(limit):
        done, last = 0, None
        for bx, by in loader:
            with L.scope():
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
                last = loss.item()
            done += 1
            if done >= limit:
                break
        return last

    go(warmup)
    t0 = js.performance.now()
    last = go(steps)
    per_step = (js.performance.now() - t0) / steps
    steps_per_epoch = -(-CIFAR_TRAIN_IMAGES // batch)
    return {"batch": batch, "ms_per_step": round(per_step, 1),
            "epoch_min": round(per_step * steps_per_epoch / 60000, 2),
            "loss": round(float(last), 4)}


async def cifar_from(L, url, name="cifar-batch1.bin"):
    """CIFAR-10 한 덩이를 받아 캐시하고 푼다.

    주소를 인자로 받는 이유: **원본은 CORS 헤더를 안 준다**(실측). 브라우저에서 직접
    못 받으므로, 직접 호스팅하거나 CORS 를 주는 미러를 쓰거나 사용자가 파일을 골라
    `cache_put` 으로 넣는다. 여기서는 첫 번째 — 로컬 서버가 서빙한다.
    """
    raw = await L.fetch_cached(url, name)
    return L.decode_cifar10(raw)


def run_real(L, x, y, batch=128, steps=20, lr=0.05):
    """진짜 이미지로 학습한다. **손실이 내려가는지**를 본다 — 합성 데이터로는
    라벨이 무작위라 내려갈 수가 없어서 그 부분이 확인이 안 됐다."""
    loader = L.utils.data.DataLoader(
        L.utils.data.TensorDataset(x, y), batch_size=batch, shuffle=True, drop_last=True)
    model = resnet18(L)
    opt = L.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = L.nn.CrossEntropyLoss()

    losses, done = [], 0
    t0 = js.performance.now()
    while done < steps:
        for bx, by in loader:
            with L.scope():
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
                losses.append(round(float(loss.item()), 4))
            done += 1
            if done >= steps:
                break
    per_step = (js.performance.now() - t0) / steps
    return {"steps": steps, "batch": batch,
            "first": losses[0], "last": losses[-1],
            "curve": losses[::max(1, steps // 8)],
            "ms_per_step": round(per_step, 1),
            "epoch_min": round(per_step * -(-CIFAR_TRAIN_IMAGES // batch) / 60000, 2)}


async def _breathe():
    """이벤트 루프에 숨 쉴 틈을 준다.

    **없으면 측정이 끝까지 못 간다.** 학습을 동기 루프로 돌리면 50~70 초쯤에서 페이지가
    통째로 죽는다(실측: 세 번 다 그 언저리). 누수가 아니라는 것은 따로 확인했다 —
    에폭 사이 텐서 수 5148 로, GPU 바이트 136MB 로 완전히 평평했다. 오래 도는 것이
    목적인 측정이라 이 한 줄이 측정 자체의 조건이다.
    """
    await asyncio.sleep(0)


async def accuracy(L, model, x, y, batch=250):
    """맞힌 비율. **학습에 안 쓴 데이터로** 재는 것이 이 함수의 전부다.

    `model.eval()` 을 부르는 쪽은 부르는 자리다 — BatchNorm 이 학습 통계를 쓰면
    배치 구성에 따라 값이 흔들려서, 재는 것이 정확도가 아니라 배치 운이 된다.
    """
    right = 0
    with L.no_grad():
        for i in range(0, len(y), batch):
            with L.scope():
                out = model(L.tensor(_prep(x[i:i + batch])))
                pred = out.numpy().argmax(axis=1)
            right += int((pred == y[i:i + batch]).sum())
            await _breathe()
    return right / len(y)


def _prep(x, augment=False):
    """[0,1] NCHW 를 모델에 넣을 모양으로. 늘리기는 **정규화보다 먼저** 한다 —
    가장자리를 0 으로 채운 뒤 정규화해야 그 0 이 다른 화소와 같은 자로 재진다."""
    if augment:
        x = vision.augment_batch(x, crop=32, padding=4, hflip_p=0.5)
    return vision.Normalize(CIFAR_MEAN, CIFAR_STD)(x).astype(np.float32)


async def train_eval(L, train, test, epochs=6, batch=128, lr=0.05, augment=False):
    """에폭마다 **학습 정확도와 시험 정확도를 둘 다** 잰다.

    둘 다 재는 이유가 이 함수의 요점이다. 학습 정확도만 보면 늘 오르므로 잘 되는 것처럼
    보인다. 둘의 **차이**가 과적합이고, augmentation 이 줄이라고 있는 것이 그 차이다.
    시험 정확도만 봐도 안 된다 — 안 오를 때 아직 덜 배운 것인지 이미 외운 것인지가
    안 갈린다.
    """
    xtr, ytr = train
    xte, yte = test
    model = resnet18(L)
    opt = L.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    crit = L.nn.CrossEntropyLoss()
    n = len(ytr) - len(ytr) % batch
    rows = []
    for _ in range(epochs):
        order = _shuffle.permutation(len(ytr))[:n]
        model.train()
        t0 = js.performance.now()
        last = 0.0
        for i in range(0, n, batch):
            sel = order[i:i + batch]
            bx = _prep(xtr[sel], augment=augment)
            with L.scope():
                opt.zero_grad()
                loss = crit(model(L.tensor(bx)), L.tensor(ytr[sel]))
                loss.backward()
                opt.step()
                last = float(loss.item())
            await _breathe()
        secs = (js.performance.now() - t0) / 1000
        model.eval()
        row = {"train": await accuracy(L, model, xtr, ytr),
               "test": await accuracy(L, model, xte, yte),
               "loss": round(last, 3), "sec": round(secs, 1)}
        rows.append(row)
        # 에폭마다 내보낸다. 긴 측정이 도중에 죽으면 반환값은 통째로 사라지지만
        # 콘솔에 나간 것은 남는다 — 실제로 한 번 죽어서 아무것도 못 건졌다.
        # 텐서 수와 바이트도 같이 낸다. 긴 측정이 죽었을 때 **누수인지 브라우저가
        # 죽인 것인지**를 가르는 것이 이 두 수다 — 없으면 둘 다 그럴듯해서 못 고른다.
        mem = L.memory()
        js.console.log(f"[bench] 에폭 {len(rows)}/{epochs} 늘리기={augment} "
                       f"학습 {row['train']:.3f} 시험 {row['test']:.3f} {row['sec']}초 "
                       f"텐서 {mem['tensors']} GPU {mem['bytes'] / 1e6:.0f}MB")
    return rows


async def report_accuracy(L, train, test, epochs=6, batch=128, only=None):
    """늘리기를 켠 쪽과 끈 쪽을 **같은 조건에서** 돌린다.

    한쪽만 재면 "정확도 X%" 라는 숫자 하나가 남는데, 그 숫자만으로는 augmentation 이
    들어갔는지 알 수 없다. 이 파일이 지금까지 ms/step 만 재온 것과 같은 종류의 구멍이다.

    `only` 로 한쪽만 돌릴 수 있고, **그 편이 실험으로 낫다.** 둘을 한 세션에서 이어
    돌리면 둘째 모델은 난수기가 이미 진행된 뒤라 **초기 가중치가 다르다** — 재려는 것이
    늘리기의 효과인데 초기값 차이가 섞인다. 페이지를 새로 띄우면 `_rng` 가 씨앗 0 에서
    다시 시작하므로 양쪽이 같은 자리에서 출발한다.
    """
    picked = (("늘리기 없음", False), ("늘리기 있음", True))
    if only is not None:
        picked = tuple(p for p in picked if p[1] == only)
    out = []
    for tag, aug in picked:
        rows = await train_eval(L, train, test, epochs=epochs, batch=batch, augment=aug)
        out.append(f"{tag}")
        for i, r in enumerate(rows, 1):
            out.append(f"  에폭 {i}  학습 {r['train']:.1%}  시험 {r['test']:.1%}  "
                       f"차이 {r['train'] - r['test']:+.1%}  손실 {r['loss']}  {r['sec']}초")
    return ("ResNet-18(CIFAR) · 정확도\n"
            f"학습 {len(train[1])}장 / 시험 {len(test[1])}장, 배치 {batch}\n"
            + "\n".join(out))


def report(L, batches=(16, 32, 64)):
    lines = []
    for b in batches:
        try:
            r = run(L, batch=b)
            lines.append(
                f"batch {r['batch']:>3}  {r['ms_per_step']:>8.1f} ms/step  "
                f"에폭 {r['epoch_min']:>5.2f}분  누수 {r['leak_per_step']:>4.1f}  "
                f"GPU {r['gpu_mb']:>6.1f}MB  손실 {r['loss']}")
        except Exception as exc:                                    # noqa: BLE001
            lines.append(f"batch {b:>3}  실패: {type(exc).__name__}: {str(exc)[:120]}")
    return "ResNet-18(CIFAR) · 배치별 실제 학습 스텝\n" + "\n".join(lines)
