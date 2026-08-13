"""ResNet-18(CIFAR 판)로 **진짜 학습 스텝**을 잰다.

층 하나로 잰 값은 BN·ReLU·잔차 덧셈처럼 대역폭에 묶인 연산을 안 센다. 목표
("에폭 몇 분")에 닿았는지는 그것들까지 포함한 실제 스텝에서만 나온다 — 그래서
FLOPs 로 나눈 추정 대신 이 파일이 있다.

브라우저 안에서 돈다:

    uv run --with playwright python tests/browser/run.py --lib browsertorch_webgpu --headed --bench
"""

import js
import numpy as np

CIFAR_TRAIN_IMAGES = 50000


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

    before = int(js.tf.memory().numTensors)
    t0 = js.performance.now()
    last = None
    for _ in range(steps):
        last = one()
    per_step = (js.performance.now() - t0) / steps
    leak = (int(js.tf.memory().numTensors) - before) / steps

    steps_per_epoch = -(-CIFAR_TRAIN_IMAGES // batch)
    return {
        "batch": batch,
        "params": n_params,
        "ms_per_step": round(per_step, 1),
        "epoch_min": round(per_step * steps_per_epoch / 60000, 2),
        "leak_per_step": round(leak, 1),
        "loss": round(float(last), 4),
        "gpu_mb": round(float(js.tf.memory().numBytes) / 1e6, 1),
    }


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
