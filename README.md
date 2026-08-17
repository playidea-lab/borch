# borch

**브라우저에서 도는 PyTorch.** numpy 위에 얹은 얇은 층이고, 설치 없이 PyTorch 문법을 연습한다.

> ## 먼저, 이것이 **아닌** 것
>
> PyTorch 가 아니다. 표면의 **11%**, 코드로는 **0.1%** 다.
> `CUDA` · 분산 · 혼합정밀도 · `torch.compile` · 사전학습 가중치는 **영원히 없다** —
> 브라우저에 존재할 수 없거나, 그것을 배우려면 브라우저를 벗어나야 하는 것들이다.
>
> 할 수 있는 말은 하나다. **입문 튜토리얼 코드가 임포트만 바꿔 같은 값을 낸다.**
> 그건 실측했다(아래 적합성).
>
> 이름이 비슷한 다른 프로젝트들과 무관하다 —
> [minitorch](https://minitorch.github.io) · [nanotorch](https://pypi.org/project/nanotorch/) ·
> [edutorch](https://pypi.org/project/edutorch/).

## 세 가지가 이 이름을 쓴다

같은 골든(진짜 PyTorch 로 굳힌 기대값)을 셋이 함께 본다. **표가 하나여야 갈리는
것이 보인다** — 서로를 대조해서 잡은 결함이 이 저장소 이력의 큰 몫이다.

| | 무엇 위에 | 어디서 | 천장 |
|---|---|---|---|
| **`borch`** (PyPI) | numpy | 어디서나 · Pyodide | MNIST 급 |
| **`borch`** (npm) — borch.ts | **WGSL 직접, 의존성 0** | 브라우저 안에서만 | CIFAR ResNet-18 **에폭 1.5분** |
| **`borch-webgpu`** (파이썬) | 위의 borch.ts | 브라우저 안에서만 | 같은 것이 **에폭 1.6분** |

아래 둘은 **같은 커널 위에 있다.** `borch-webgpu` 는 borch.ts 를 파이썬에서 부르는
5,961 줄짜리 결속이고, 그 차이(1.5분 대 1.6분)가 Pyodide 를 한 번 지나는 값이다.

> 그 자리에 한동안 **TF.js 판**이 있었다. 같은 이름, 다른 밑바닥, **5,307 줄**. 직접 쓴
> WGSL 이 같은 벤치에서 20% 빨랐고(배치 64 에서 154.7ms → 123.4ms) 랭크 한계도
> 없어서 걷어냈다. 그 결정과 실측은 [BORCH-TS.md](BORCH-TS.md) 에 있다.

아래는 파이썬 쪽 이야기다. TypeScript 쪽은 [borch.ts](#borchts--typescript-와-wgsl) 절에 있다.

## 설치

순수 파이썬 휠 하나다. 의존성은 numpy 뿐이고, Pyodide 에는 numpy 가 이미 있다.
`borch` 패키지와 `borch_vision` 모듈이 들어 있다.

```bash
uv add ./borch-1.4.0-py3-none-any.whl        # 릴리스에서 받은 파일
```

브라우저(Pyodide)에서는 휠 바이트를 가상 파일시스템에 써넣고 `micropip` 으로 건다.

```js
// 파일 이름을 그대로 써야 한다 — micropip 이 이름에서 패키지명과 버전을 읽는다.
py.FS.writeFile("/borch-1.4.0-py3-none-any.whl", new Uint8Array(wheelBytes));
await py.runPythonAsync(`
import micropip
await micropip.install("emfs:/borch-1.4.0-py3-none-any.whl")
`);
```

> **저장소가 private 이라 릴리스 URL 을 그대로 `micropip.install()` 에 넣을 수 없다.**
> 익명 요청은 404 를 받는다(실제로 그렇게 해보고 알았다). 공개로 돌리면 URL 한 줄로 끝난다.

```python
import borch as torch

w = torch.tensor(3.0, requires_grad=True)
loss = (w - 5.0) ** 2
loss.backward()
print(w.grad.item())               # -4.0
```

### 이름을 어떻게 붙일 것인가 — 셋이고, 경계는 재봤다

`import borch as torch` 는 **그 파일 안의 이름** 하나를 만든다. `from X.Y import Z` 는
`sys.modules` 에 등록된 **경로**를 보므로 별칭이 안 닿는다. 그 차이가 어디서 갈리는지를
`tests/test_alias.py` 가 못 박는다.

| | `torch.nn.Linear` | `from borch.nn import Linear` | `from torch.… import` | 남의 `import torch` |
|---|---|---|---|---|
| `import borch as torch` | ✅ | ❌ | ❌ | 안 건드림 |
| `borch.install("borch")` | ✅ | ✅ | ❌ | 안 건드림 |
| `sys.modules["torch"] = borch` | ✅ | ✅ | ✅ | **가로챔** |

**기본은 첫 줄이다.** 대부분의 교재 코드가 `torch.nn.Linear` 처럼 속성으로 닿고, 그건
별칭만으로 된다.

`from … import` 가 필요하면 **자기 이름으로 심는다** — `borch.install("borch")`. 하위
경로가 열리면서 남의 코드는 안 건드린다.

`torch` 로 심는 것은 마지막이다. 그 뒤로는 **남의 라이브러리가 하는 `import torch` 도**
축소판을 받는다. 학습자 한 명의 연습 환경에서는 편의지만, 다른 코드가 섞인 곳에서는
원인을 못 찾는 오류가 된다.

```python
import sys, borch
sys.modules["torch"] = borch          # 정말 필요할 때만
```

---

## 왜 만들었나

PyTorch 는 WebAssembly 로 포팅되지 않는다. 수백 MB의 네이티브 코드에, 손튜닝된 AVX·NEON
커널은 wasm SIMD 로 옮겨지지 않고, OpenMP 스레드는 Pyodide 가 싣지 않는 헤더를 요구한다.

**그런데 문법을 익히는 데는 그중 아무것도 필요하지 않다.** numpy 는 Pyodide 안에 이미 있다.

비행 시뮬레이터에 가깝다 — **조종간은 진짜 조종간이고, 밑의 물리가 흉내다.**
학습자가 타이핑하는 코드는 진짜 PyTorch 코드이고, 자기 컴퓨터에서 그대로 돈다.

## 설계 원칙 — 틀린 답보다 없는 기능이 낫다

흉내가 진짜와 *거의* 같으면 그것을 배우는 사람은 거짓을 배운다.
그래서 **범위 밖은 근사하지 않고 예외를 던진다.**

```python
>>> torch.tensor([-1.5, 2.0, -0.25])
tensor([-1.5000,  2.0000, -0.2500])          # 진짜와 같은 자리·같은 정렬

>>> torch.tensor([1.0], requires_grad=True) * 2
tensor([2.], grad_fn=<MulBackward0>)

>>> torch.randn(3, 4) @ torch.randn(3, 2)
RuntimeError: 행렬곱의 모양이 안 맞습니다 (3x4 @ 3x2) — 앞의 열(4)과 뒤의 행(3)이 같아야 합니다.
(torch: mat1 and mat2 shapes cannot be multiplied (3x4 and 3x2))

>>> torch.nn.LSTM(2, 2)
BrowserTorchError: nn.LSTM 은(는) 브라우저 축소판에 없습니다.
자기 컴퓨터에서 `uv add torch` 로 진짜 PyTorch 를 쓰세요 — 축소판은 문법 연습용이고,
없는 것을 흉내 내면 틀린 것을 배우게 됩니다.
```

조용히 다른 값을 내느니 시끄럽게 멈춘다.

**한 자리만 예외다.** `svd_lowrank`·`pca_lowrank` 의 `niter` 는 받되 안 쓴다. torch 는
무작위 부분공간을 `niter` 번 다듬는데, 그 반복이 있는 이유는 사영이 근사이기
때문이다. 우리는 전체 SVD 를 구해 앞을 자르므로 다듬을 것이 없다 — 그래서 **작은
`niter` 에서 torch 와 값이 갈린다.** 우리 쪽이 torch 의 `niter` 를 크게 키운 극한에
있는, 더 정확한 갈림이다. 멈추지 않는 이유는 여기서 멈추면 정확한 답을 거절하게
되기 때문이고, 적어 두는 이유는 **갈림은 더 정확해도 갈림**이기 때문이다.

## 파일 구조 — 한 파일이 아니다

`borch` 는 **패키지다**. 처음에는 파일 하나였고 그것이 "얹으면 끝"이라는 배포
이야기의 일부였는데, 3,300 줄이 되면서 그 명분이 먼저 사라졌다.

| | 줄 | 무엇 |
|---|---|---|
| `_base.py` | 275 | dtype · 오류 규격 · `repr` |
| `_tensor.py` | 1,025 | `Tensor` 와 autograd |
| `_ops.py` | 7,507 | 수학 · 모양 · `nn.functional` |
| `_nn.py` | 2,771 | 층 · 순환 · 트랜스포머 |
| `_optim.py` | 1,000 | 옵티마이저 · 스케줄러 |
| `_fft.py` | 364 | `torch.fft` · `stft`/`istft` |
| `_data.py` | 355 | `Dataset` · `DataLoader` |
| `_rnn.py` | 65 | `nn.utils.rnn` |
| `__init__.py` | 338 | 전부 모으고 `torch` 로 심는 자리 |

**공개 이름은 안 바뀌었다** — 쪼개기 전후로 197개 그대로다. `import borch` 는
같은 것을 준다. `borch_webgpu` 도 같은 모양이고 5,961 줄에 여섯 파일이다.

손으로 옮기지 않았다. 자르는 자리만 정하고 나머지는 스크립트가 했다 — 3천 줄을 사람이
오려 붙이면 조용히 한 줄이 사라지고, 그건 골든이 잡기 전까지 아무도 모른다.

## 어떻게 보증하는가

`tests/test_diff.py` 가 **같은 연산을 진짜 torch 와 borch 양쪽에 넣고 숫자를 비교한다.**
pytest 180개, **코드 커버리지 93%**.

```bash
uv run --with pytest --with numpy --with torch pytest tests/
```

> **GPU 쪽은 코드 커버리지를 못 잰다.** 브라우저 안에서만 돌아서 `pytest --cov` 가
> 닿지 않는다. 그쪽에 대해 말할 수 있는 것은 **골든 2641건이 지난다**는 것뿐이고,
> 그것은 표면 검사이지 줄 검사가 아니다. 두 수를 같은 것처럼 적지 않는다.

이 검사가 첫 실행에서 잡은 것: PyTorch 의 `BatchNorm2d` 는 **같은 forward 안에서 분산을
두 가지로 쓴다** — 정규화는 편향(ddof=0), `running_var` 갱신은 비편향(ddof=1).
둘 다 편향으로 두면 출력이 2.6% 어긋난다. 코드를 읽어서는 나오지 않는 종류다.

### 초록색이 거짓일 수 있는 다섯 자리

표가 커질수록 "몇 건이 지난다" 는 말의 값어치가 떨어진다. **묻지 않은 것은 안 맞는다**
가 이 저장소에서 반복된 유일한 규칙이고, 아래 다섯은 각각 **두 번 이상 실제로 물린**
자리다. 새 케이스를 쓰거나 결함을 쫓을 때 먼저 의심할 곳이다.

**같은 계산에 이름이 둘이면 한쪽만 맞을 수 있다** (3회). `norm(p)` 은 틀렸는데
`linalg.vector_norm(p)` 은 맞았고, `searchsorted(side=)` 는 버려졌는데
`bucketize(right=)` 는 맞았고, `max_pool2d` 와 `max_pool2d_with_indices` 도 그랬다.
맞는 쪽만 물으면 틀린 쪽은 표에서 없는 것과 같다.

**대칭이 있는 입력은 대칭을 깨는 결함을 가린다** (여러 번). 값이 전부 다르면 접힘의
세 규칙이 같은 답을 내고, 창과 보폭이 같으면 보폭을 흘려도 답이 같고, 상삼각과
하삼각이 같은 대각행렬은 `upper` 를 안 묻는다. **입력을 고르는 것이 케이스의 절반이다.**

**정의역이 있는 인자는 전부 넣어 봐야 한다** (2회). `if p == 1: … else: L2` 처럼
`else` 가 한 값의 이름을 달고 나머지를 삼키면, 인자를 흔들었을 때 답은 변하므로
"인자가 쓰인다" 는 검사를 통과한다. 변한 답이 틀렸을 뿐이다.

**NaN 이 있을 수 있는 자리에서 마스크는 곱하는 것이 아니라 고르는 것이다** (3회).
`0 × NaN = NaN` 이라 마스크 곱셈은 걸러 낸 자리를 통째로 오염시킨다. `where` 를 쓴다.
코어의 `median`, borch.ts 의 `median`, 그리고 `nanmedian` 에서 한 번 더 물렸다.

**셋이 서로를 대조하므로, 한쪽이 다른 쪽의 구멍을 베끼면 표는 초록이다** (1회지만
가장 위험하다). borch.ts 의 `i0` 은 기울기를 0 으로 두고 **"코어도 그래프를 끊고
있어서 맞춘 것"** 이라고 주석에 적혀 있었다. 둘이 일치했고 torch 와는 갈렸다.
게다가 베낄 때 나빠졌다 — 끊긴 그래프는 멈추지만 `0` 은 안 멈춘다. **기울기가 0 인
것과 기울기가 없는 것은 다른 말이다.**

곁들여, 검사 자체가 한쪽 편을 드는 자리도 세 번 나왔다: **평범한 텐서도
`.values`·`.indices` 를 갖는다**(희소용). 튜플인지 먼저 보지 않으면 텐서의 첫
원소를 집고, 그러면 없는 결함이 무더기로 보인다.

그리고 위 다섯과 **증상이 반대인** 자리가 하나 있다. 다섯은 결함이 안 보이는
쪽인데, 이쪽은 **빨강이 엉뚱한 이름을 달고 나온다** — 케이스가 공유 상태를 건드리면
그 뒤에 오는 케이스들이 대신 빨개진다. `tensor()` 가 사본을 안 뜨던 동안 케이스
하나가 입력 배열을 1 만큼 올렸고, torch 는 사본을 떠서 안 샜기 때문에 **코어에서만**
열여섯 자리가 틀렸다. 하나씩 돌리면 전부 통과해서 원인이 자기 자리에 없었다.
처방도 다르다 — 케이스를 더 넣는 것이 아니라 **격리하는 것**이다.

---

## 얼마나 빠른가

브라우저(Pyodide) 기준 실측이다.

| | 시간 |
|---|---|
| MLP 학습 1스텝 (256×64) | 3.3ms |
| conv2d 순방향 (32×1×28²) | 1.9ms |
| **MNIST CNN 학습 1에폭** | **약 2분** |

네이티브에서는 torch 와 비슷하거나 빠르다 — 둘 다 BLAS 를 부르고, 작은 텐서에서는
torch 의 디스패처 오버헤드가 더 크다. 느려지는 것은 wasm 탓이고, 그중에서도
큰 행렬곱만 유독 나쁘다(Pyodide 의 BLAS 가 SIMD·멀티스레드를 못 쓴다).

**MNIST 급까지는 브라우저에서 실제로 학습된다.** 그 위는 자기 컴퓨터나 원격 장비다 —
또는 아래의 GPU 배포판이다.

## 그 위가 필요하면 — `borch-webgpu`

이것(코어)은 **numpy 위에서 MNIST 급까지**다. 그 경계를 넘고 싶으면 별도 배포판이 있다.

| | 코어 `borch` | `borch-webgpu` |
|---|---|---|
| 무엇 위에 | numpy | borch.ts (직접 쓴 WGSL) |
| 휠 | 순수 파이썬 42KB | 휠에 없다(브라우저 전용, borch.ts 는 페이지가 싣는다) |
| 어디서 | 어디서나 | **브라우저 안에서만** |
| 천장 | MNIST 급 | **CIFAR ResNet-18 이 에폭 약 1.6분** (실측) |
| 읽히는가 | 그것이 전부다 | 아니다. 성능이 목적이다 |
| 제자리 연산 | `x.add_(1)` 도, **뷰를 통한 전파도** 된다 | `x.add_(1)` 은 되고, **뷰 전파는 거절한다** |

### 그래서 몇 %인가

천장을 **속도로만** 말해왔다. "에폭 2분"은 맞지만 "그래서 몇 %"는 오래 아무도 안 물었다.
잰 값은 아래와 같다 — CIFAR-10 학습 1만 장, **학습에 안 쓴** 시험 1만 장, ResNet-18,
10 에폭, 배치 128.

> **이 표는 TF.js 바닥에서 잰 것이다.** 그 뒤 같은 이름의 패키지가 borch.ts 위로
> 옮겨갔고, 아직 다시 안 쟀다. 그때 골든이 두 바닥에서 같은 값을 냈으므로 크게
> 달라질 이유는 없지만, **재지 않은 것을 잰 것처럼 적지 않는다.**

| 10 에폭 뒤 | 늘리기 없음 | 늘리기 있음 |
|---|---|---|
| 학습 정확도 | 80.9% | 64.8% |
| **시험 정확도** | 59.9% | **60.4%** |
| **둘의 차이(과적합)** | **+21.0%** | **+4.3%** |
| 가장 좋았던 시험 | 61.4% (8에폭) | 62.2% (9에폭) |

**학습 정확도만 봤으면 늘리기가 해롭다고 결론냈을 것이다**(80.9% → 64.8%). 시험 정확도만
봤어도 거의 같아서(59.9% vs 60.4%) 아무 일도 없는 줄 알았을 것이다. 둘을 같이 봐야
보이는 것이 **과적합이 21.0%에서 4.3%로 줄었다**는 것이고, 늘리기가 하라고 있는 일이
정확히 그것이다. 늘리기 없는 쪽은 8 에폭에서 시험 정확도가 꺾여 내려간다.

두 조건은 **각자 새 페이지에서** 돌린다. 한 세션에 이어 돌리면 둘째 모델은 난수기가
진행된 뒤라 초기 가중치가 달라지고, 재려는 것이 늘리기의 효과인데 초기값 차이가 섞인다.

```bash
# cifar-batch1.bin(학습)과 cifar-batch-test.bin(시험)이 저장소 루트에 있어야 한다.
# 원본은 CORS 로 못 받으므로 직접 가져다 둔다 — 아래 transforms 절 참고.
uv run --with playwright python tests/browser/run.py \
    --lib borch_webgpu --headed --accuracy --epochs 10 --augment off
uv run --with playwright python tests/browser/run.py \
    --lib borch_webgpu --headed --accuracy --epochs 10 --augment on
```

> 이 수는 **1만 장으로 10 에폭**을 돌린 값이다. CIFAR 전체(5만 장)나 더 긴 학습의 값이
> 아니고, 발표된 ResNet-18 수치와 비교할 것도 아니다. 여기서 말하려는 것은 절대 수치가
> 아니라 **재는 자리가 생겼다**는 것과 늘리기가 실제로 듣는다는 것이다.

**코어를 대체하지 않는다.** 왜 하나로 합치지 않았는지는 [ROADMAP 의 ADR-001](ROADMAP.md)
에 적었다 — 요약하면 휠의 성질이 전염되고, 브라우저·드라이버 실패가 `import` 로 올라오고,
"임포트만 바꾸면 같은 코드"라는 약속이 `device`·비동기와 양립하지 않기 때문이다.

설계와 실측은 [WEBGPU-DESIGN.md](WEBGPU-DESIGN.md) 에 있다.

## 지원 범위

| | |
|---|---|
| **텐서** | 모양·브로드캐스팅·dtype 승격 · 인덱싱 · reshape/view/permute/squeeze · split·chunk·flip·roll·gather·narrow·index_select·masked_select |
| **autograd** | `requires_grad` · `backward()` · `.grad` · `no_grad()` · `detach()` · 누적 |
| **축약** | `sum`·`mean`·`max`·`min`·`prod`·`median`·`norm`·`cumsum`·`topk`·`sort`·`unique`·`std` — 역전파 포함 |
| **nn** | `Module` · `Linear` · `Conv1d/2d/3d` · `MaxPool1d/2d/3d` · `Upsample` · `Embedding` · `LayerNorm` · `BatchNorm1d/2d/3d` · `Dropout` · `Sequential` · `ModuleList` |
| **순환** | `RNN` · `LSTM` · `GRU` — 다층 · `batch_first` · 초기 상태. **최상위 함수 꼴도** (`torch.lstm`·`lstm_cell` 등 여덟) — 가중치를 목록으로 받는다. 양방향과 층간 드롭아웃은 거절한다 |
| **트랜스포머** | `MultiheadAttention` · 인코더·디코더 층 · `nn.Transformer` — 불리언·실수 마스크 · `norm_first` · gelu |
| **손실** | `MSELoss`·`L1Loss`·`SmoothL1Loss`·`BCELoss`·`BCEWithLogitsLoss`·`CrossEntropyLoss`·`NLLLoss` |
| **optim** | `SGD`(momentum·weight_decay) · `Adam` · `AdamW` · `RMSprop` — `param_groups` · `state_dict` |
| **스케줄러** | `StepLR` · `MultiStepLR` · `ExponentialLR` · `CosineAnnealingLR` · `LambdaLR` · `ReduceLROnPlateau` |
| **데이터** | `Dataset` · `TensorDataset` · `Subset` · `ConcatDataset` · `DataLoader` · `WeightedRandomSampler` · `random_split(generator=)` · `collate_fn` |
| **저장** | `state_dict` · `load_state_dict` · `save`/`load` · 버퍼(`running_mean` 등) 포함 |
| **nn.functional** | 25종 — 활성·손실·`pad`·`normalize`·`cosine_similarity`·`one_hot`·`layer_norm`·`embedding` |
| **복소수** | `complex64` 만 — `complex`·`polar`·`view_as_real`/`view_as_complex`·`real`/`imag`/`conj`/`angle`/`abs` · 산술 · autograd. **셋 다에 있다** (아래) |
| **푸리에** | `fft.fft`/`ifft`/`rfft`/`irfft` · `fftfreq`/`rfftfreq`/`fftshift`/`ifftshift` · `stft`/`istft` — `n`·`dim`·`norm` 과 역전파. **셋 다에 있다** |

### 복소수 — `complex64` 만

`complex128` 은 **영원히 없다.** WGSL 에 `f64` 가 없어서 브라우저에서 도는 절반이
그것을 못 든다. 그래서 이름은 두되, `complex64 + float64` 처럼 승격이 그것을 만들려는
자리에서 **조용히 내려앉지 않고 멈춘다** — torch 는 여기서 `complex128` 을 준다.
값이 반쯤 맞는 것보다 여기서 서는 편이 낫다는 이 저장소의 같은 선택이다.

기울기 규약은 재서 못 박았다: **torch 는 복소 손실에 `backward()` 를 거절한다.**
손실이 늘 실수라면 규약은 하나로 정해진다 —

    z.grad = ∂L/∂re + i·∂L/∂im

그 위에서 **정칙 함수의 역방향에는 켤레가 붙고**(곱셈·나눗셈: `conj(f'(z))·g`),
실수를 내는 `abs` 에는 안 붙는다(`z/|z|`). 실수 입력만으로는 이 차이가 안 보인다 —
켤레가 실수에서 항등이라서다. 그래서 골든이 셋을 한 표에서 묻는다.

**`conj` 가 torch 와 갈린다.** torch 의 `conj` 는 **게으르다** — 켤레 비트만 세우고
값을 안 뒤집는다. 그래서 `torch.is_conj(torch.conj(z))` 가 `True` 이고,
`view_as_real` 은 "풀지 않은 켤레" 라며 거절한다. 우리 것은 즉시 뒤집으므로
그 상태가 아예 없고, **`is_conj` 는 언제나 `False`** 다. 값은 같다 —
`conj_physical` 로 물으면 양쪽이 같은 답을 낸다.

**borch.ts 의 저장은 인터리브다** — `[re, im, re, im, …]` 한 버퍼. 그래서
`view_as_real`·`view_as_complex` 가 **진짜 뷰**이고(torch 도 그렇다), 대신 오래된
불변식 `size = 버퍼 길이` 가 `버퍼 길이 = size × 2` 로 갈린다. 그것을 모르는 커널이
복소수 버퍼를 읽으면 앞쪽 절반만 실수로 보고 **예외 없이** 틀린 답을 내므로,
**기본값이 거절**이다 — 복소수를 아는 연산만 따로 문을 지난다. `Tensor.from` 으로
형만 `complex64` 라고 붙이는 것, 이름표 갈이(`to`), 체크포인트 저장도 같은 이유로
막혀 있다.

### 푸리에 — `torch.fft` 와 `stft`

**복소수 위에 선다.** `stft` 는 오래 거절이었고 거절문에 "복소수 규약을 안 정했다"
고 적혀 있었다. 그 이유가 **정확했기 때문에** 규약이 정해진 날 문이 열렸다 — "저장이
없다" 로 적어 두었으면 저장이 생긴 뒤에도 아무도 다시 안 물었을 것이다.

`stft` 는 새 커널이 아니라 **조립이다** — 자르고 · 창을 곱하고 · `rfft`. 셋 다 이미
미분되는 이름이라 **기울기가 저절로 맞는다.** 손으로 커널을 쓰면 순방향은 금방 맞고
역방향이 창과 겹침을 다 지나야 해서, 틀리면 값은 그럴듯하고 학습만 안 된다.

브라우저 쪽은 **DFT 를 그대로 돈다 — O(n²) 다.** 쿨리-튜키는 2 의 거듭제곱에서만
빠르고 아닌 길이는 블루스타인이 따로 필요한데, **값은 어느 쪽이든 같고** 이 프로젝트의
천장에서는 차이가 안 보인다. 빨라야 하는 날이 오면 그때 바꾸되, 지금 없는 속도를
있는 것처럼 적지 않는다. 회전인자는 **호스트에서 배정도로 만들어 올린다** —
셰이더의 `cos`/`sin` 은 정확도가 구현에 맡겨져 있고, 실제로 사각창 `stft` 한 자리가
상대오차 2.7e-4 로 골든을 벗어났다(f32 반올림으로는 설명이 안 되는 크기다).

기울기에서 어려운 자리는 값이 아니라 **어느 쪽 반쪽을 세는가** 다. `rfft` 의 역방향은
저장된 반쪽에만 기울기가 오므로 켤레 짝을 **안** 더하고(더하면 두 배), `irfft` 는
되살린 켤레 짝이 같은 칸에서 왔으므로 **가장자리만 한 번, 가운데는 두 번** 센다.
둘 다 순방향 값은 멀쩡한 채로 틀릴 수 있는 자리다.

> **`abs` 의 칼날.** 골든의 `stft` 기울기 케이스가 고르지 않은 수를 쓰는 데는 이유가
> 있다. 경사 신호는 나이퀴스트 칸이 **정확히 0** 이 되는데 거기서 `abs` 는 미분
> 불가능하고 부호가 반올림에 달린다 — 우리는 float64 로 누산해 +1 을, torch 는
> float32 FFT 라 0 을 골랐다. 규칙이 갈린 것이 아니라 **케이스가 칼날 위에 선 것**
> 이고, 그런 것을 굳히면 골든이 부동소수 우연을 명세로 박제한다.

**`print` 도 명세다.** torch 는 복소수를 찍을 때 **실수부와 허수부를 따로 잰다** —
`[1+2j, -0.5-1j]` 에서 실수부는 소수 네 자리를 요구하고 허수부는 정수라
`1.0000+2.j` 가 된다. 한 형식으로 재면 `1.0000+2.0000j` 가 나오는데, 값이 전부
맞는 채로 글자만 갈린다. 자리맞춤도 실수부에만 걸려서 `1.-0.j` 처럼 **음의 0** 이
산다 — 그 부호 하나가 이 결속의 읽기 경로에서 `-0.0` 을 잃고 있던 것을 잡았다.
`-0.0 == 0.0` 이라 값 대조로는 영영 안 걸렸을 자리다.

## torchvision — `transforms` 만 (`borch_vision`)

파이토치 입문 튜토리얼의 **첫 열 줄이 torchvision** 이다.

```python
datasets.MNIST(root, transform=transforms.ToTensor())
```

"임포트만 바꿔 같은 값을 낸다"는 약속이 여기서 먼저 걸리므로, `transforms` 는 있다.
별도 파일인 이유는 `torchvision.transforms` 이지 `torch.transforms` 가 아니어서다 —
코어 안에 넣으면 진짜 torch 에 **없는 자리**를 만들게 된다.

```python
import borch_vision as torchvision
from borch_vision import transforms
```

| 있는 것 | `Compose` · `ToTensor` · `Normalize` · `RandomHorizontalFlip` · `RandomCrop` |
|---|---|
| **`datasets`** | 없다. 받아오는 쪽이 막혀 있다 — `cs.toronto.edu` 가 CORS 헤더를 안 준다(실측). 게다가 torch 의 `download=True` 는 받아두고 재사용하는데 Pyodide 파일시스템은 새로고침에 날아간다. **바이트를 손에 넣은 뒤는 이미 된다** (`fetch_cached`·`cache_put`·`TensorDataset`) |
| **`ops`** | 없다. `nms` 는 numpy 로 짧아서 "크다"는 이유는 거짓이고, 진짜 이유는 아무도 그 앞에 안 선다는 것이다 — 검출은 사전학습 백본과 COCO 급 데이터가 있어야 끝까지 간다 |
| **사전학습 가중치** | 없다. `.pth` 는 pickle 이라 torch 내부 클래스를 흉내 내야 읽히고, 미묘하게 틀리면 모양 맞는 가중치에 틀린 수가 들어온다. ResNet-18 만 45MB 이며, 무엇보다 `pretrained=True` 가 돌면 사람들은 발표된 top-1 과 비교한다 — 비트 동등은 명시적 비목표라 지킬 수 없는 약속이다 |

**난수는 torch 와 다르다.** 같은 씨앗을 줘도 torchvision 과 같은 장면이 나오지 않는다 —
torch 의 난수기를 쓸 수 없어서다. 그래서 골든은 확률을 0·1 로 못 박은 자리만 대조하고,
뽑기가 실제로 도는지는 `tests/test_vision.py` 가 분포로 본다. 둘 중 하나만 하면
"무작위니까 못 잰다"로 안 잰 것을 잰 것처럼 적게 된다.

## borch.ts — TypeScript 와 WGSL

파이썬을 안 거친다. **TF.js 도 안 거친다** — 커널을 WGSL 로 직접 썼다.
런타임 의존성이 **0개**이고, 브라우저가 그냥 읽는 ES 모듈이다(232KB, 압축 전).

```bash
npm install borch
```

```ts
import { init, Tensor, nn, optim, scope, keepAlive } from "borch";

await init();                                   // WebGPU 어댑터를 잡는다

const model = new nn.Sequential(
  new nn.Linear(784, 128), new nn.ReLU(), new nn.Linear(128, 10));
const opt = new optim.SGD(model.parameters(), 0.05, 0.9);
const crit = new nn.CrossEntropyLoss();

const x = keepAlive(Tensor.from(pixels, [32, 784]));
const y = keepAlive(Tensor.from(labels, [32], { dtype: "int64" }));

for (let i = 0; i < steps; i++) {
  await scope(async () => {                     // 한 스텝의 중간 버퍼를 놓는다
    opt.zeroGrad();
    const loss = crit.call(model.call(x), y);
    loss.backward();
    opt.step();
    console.log(await loss.item());
  });
}
```

**이 예시는 실제로 돈다** — `npm run example:ts` 가 그대로 실행하고 손실이 내려가는
것까지 본다. 문서의 코드는 안 돌리면 썩고, 이 저장소는 설치 안내가 실제로는 안 듣던
것을 이미 두 번 잡았다.

### Pyodide 에서 파이썬으로 — `borch_webgpu`

파이썬 코드를 **borch.ts 위에서** 돌린다. 결속이 하는 일은 위의 네 가지
차이(`await init()`·`await item()`·`scope()`·`.call()`)를 감추는 것이다.

```python
import borch_webgpu as torch          # 별칭이면 대부분 된다
```

WebGPU 에 동기 읽기가 없는데도 **`await` 이 안 나온다.** Pyodide 의 `run_sync`(JSPI)가
그 자리를 메운다 — 재봤고(`tests/browser/sync_probe.py`), 조건이 하나 있다: 페이지가
비동기로 파이썬에 들어와야 한다. 러너는 이미 그렇게 들어간다.

`from borch_webgpu.nn import Linear` 처럼 하위 경로가 필요하면 `borch_webgpu.install()` 을
부른다. 기본값이 자기 이름이라 남의 `import torch` 는 안 건드린다 — 위의 표와 같은
선택이다.

**골든 2641 건 전부**를 지난다 — 표에서 이쪽만 건너뛰는 것이 하나도 없다. 코어는
그중 2588 건을 보는데, 나머지 53 건은 코어가 일부러 거절하는 것들(1·3 차원 합성곱,
랭크 7·8)이라 안 묻는다.

borch.ts 자신은 2015 건에 TS 본문을 써 두었다. 나머지 583 건은 **일부러 안 옮겼다** —
결속(`borch-webgpu`)이 이미 그 케이스들에서 borch.ts 커널을 지나므로 **값은 검증되고
있고**, TS 본문이 추가로 증명하는 것은 값이 아니라 이쪽 표면(이름과 인자 순서)이다.
그중 상당수는 파이썬 이름 별칭을 묻는 것이라 옮기면 같은 질문이 두 번이 된다.
줄지 않는 그 수를 러너가 계속 찍는다 — 은근히 사라지는 것보다 낫다.

### torch 와 갈리는 다섯 자리

첫 열 줄에서 전부 만나므로 미리 적는다.

| | 왜 |
|---|---|
| `await init()` 을 먼저 | WebGPU 어댑터를 잡는 것이 비동기다 |
| `await loss.item()` | GPU 메모리를 도로 가져온다. 순방향·역방향은 동기다 |
| `scope()` 로 감싼다 | JS 의 쓰레기 수집이 GPU 메모리를 제때 안 놓는다. 한 스텝이 중간 버퍼를 수천 개 만든다 |
| `model.call(x)` | JS 는 객체를 그냥 못 부른다 |
| `'cpu'` 로는 연산이 안 된다 | 값을 내려두는 자리이지 커널이 있는 장치가 아니다 (아래 절) |

`scope()` 는 torch 에 없다 — TF.js 의 `tidy` 와 같은 자리이고 이유도 같다. 파라미터처럼
살아남아야 하는 것은 `keepAlive` 로 표시한다 — **안 감싸면 몇 스텝 만에 장치가 찬다.**

### 장치를 다루는 자리

`torch.cuda.is_available()` 자리는 이렇다. **비동기다** — 어댑터를 얻는 것이 비동기라
피할 길이 없다.

```ts
import { init, isAvailable, probe, currentDevice, Tensor } from "borch";

if (!(await isAvailable())) { /* 이 브라우저에서는 못 쓴다 */ }

const p = await probe();          // 왜 안 되는지까지 필요하면
if (!p.ok) console.log(p.why);    // 'no-api' | 'no-adapter'

await init({ powerPreference: "high-performance" });   // 기본값이 이것이다
currentDevice();                  // 'webgpu' — 안 붙었으면 null
```

`why` 를 가르는 것이 요점이다. `no-api` 는 브라우저가 낡았거나 https 가 아닌 것이고
`no-adapter` 는 드라이버 차단 목록·가상 머신·GPU 없는 헤드리스다 — 쓰는 사람이 할 수
있는 일이 서로 다른데 예외 하나로 뭉치면 그 갈림이 사라진다.

텐서가 어디 있는지는 `t.device` 가 답하고, `await t.cpu()` 로 내리고 `t.webgpu()` 로
올린다. **`'cpu'` 는 값이 담긴 그릇이지 연산되는 장치가 아니다** — borch 에 CPU 커널은
없다. 내려온 텐서는 읽을 수는 있어도(`toArray`·`item`·`repr`) 연산에 넣으면 torch 와
같은 문구로 멈춘다.

```ts
const g = Tensor.from([1, 2, 3, 4], [2, 2]);   // 'webgpu'
const c = await g.cpu();                        // 'cpu'
await c.item();                                 // 된다 — 읽기다
c.sum();                                        // RuntimeError:
                                                // Expected all tensors to be on
                                                // the same device, ...
c.webgpu().sum();                               // 다시 된다
```

내리는 것만 비동기인 것은 그쪽만 왕복이기 때문이다. 올리는 것은 큐에 쓰기 하나다.

`torch.cuda.synchronize()` 자리는 `await device().synchronize()` 다. 지금까지 완료를
강제하는 방법은 값을 하나 읽는 것이었는데 그러면 **readback 왕복이 측정에 섞인다.**

### 층을 직접 만들 때

**필드에 두면 등록된다.** torch 가 `__setattr__` 로 하는 일과 같은 자리다.

```ts
class Net extends nn.Module {
  fc1 = new nn.Linear(4, 8);
  fc2 = new nn.Linear(8, 2);
  override forward(x: Tensor): Tensor {
    return this.fc2.call(this.fc1.call(x).relu());
  }
}
new Net().parameters();       // 넷 다 나온다
new Net().namedParameters();  // fc1.weight, fc1.bias, fc2.weight, fc2.bias
```

텐서를 직접 파라미터로 둘 때는 `claim()` 으로 세운다 — torch 의 `nn.Parameter` 자리다.
안 세운 텐서 필드는 상수로 보고 옵티마이저가 안 밟는다. **배열은 안 훑는다**(torch 도
파이썬 list 를 등록하지 않는다) — `nn.ModuleList` 를 쓴다.

### 파라미터 그룹

```ts
const opt = new optim.SGD([
  { params: backbone.parameters(), lr: 1e-3 },
  { params: head.parameters(),     lr: 1e-2, weightDecay: 0 },
], 1e-3);
opt.addParamGroup({ params: extra.parameters(), lr: 5e-4 });
```

스케줄러는 그룹 전부를 몰고 그룹 사이 비율을 지킨다 — torch 가 `base_lrs` 를 그룹마다
드는 것과 같은 결과다.

### 난수

`manualSeed` 하나가 텐서 팩토리·층 초기화·dropout 을 같이 되돌린다.

```ts
manualSeed(42);
Tensor.randn([2, 3]);           // 표준정규
Tensor.rand([4]);               // [0, 1)
Tensor.randint(0, 10, [8]);     // [low, high) 정수, int64
Tensor.randperm(64);
t.randnLike();
```

### `nn.functional` — `F.` 로 적힌 줄

torch 는 같은 연산을 두 이름으로 갖는다. `x.relu()` 도 되고 `F.relu(x)` 도 되며,
교재 코드는 층을 쓸 때 앞쪽을, 손실·합성곱을 직접 부를 때 뒤쪽을 쓴다. borch 에는
앞쪽만 있어서 `F.` 로 적힌 줄을 통째로 다시 써야 했다.

```ts
import { nn } from "borch";
const F = nn.functional;                 // torch.nn.functional 과 같은 경로다

F.relu(x);
F.conv2d(x, weight, bias);
F.crossEntropy(logits, target);
```

**메서드를 안 없앤다.** torch 가 둘 다 갖고 있으므로 우리도 둘 다 갖는다 —
`x.relu()` 로 적힌 코드가 이 변경으로 멈출 이유가 없다. `Tensor` 가 작아지지도
않는다. 없던 문을 내는 것이지 있던 것을 치우는 것이 아니다.

**이름이 같은데 연산이 다른 다섯은 안 낸다.** `F.layer_norm`·`F.rms_norm`·`F.pad`·
`F.upsample` 은 torch 와 인자 규약이 다르고, `F.batch_norm` 은 `Tensor.batchNorm`
(축만 바꾼 `layerNorm`)이 아니라 층 쪽 자유 함수로 나간다. 이름으로 이으면 조용히
다른 연산이 걸리므로, 없는 것은 없다고 둔다.

### 대괄호 자리 — `x[...]`

**자바스크립트는 `[]` 를 오버로드할 수 없다.** 그래서 torch 의 대괄호 한 줄이 여기서는
`select`·`narrow`·`indexSelect`… 열다섯 갈래로 흩어지고, 옮겨 적는 사람이 줄마다
어느 것인지 골라야 했다. `at()` 이 그 갈래를 문 하나로 좁힌다.

```ts
x.at(0)                     // x[0]           축이 사라진다
x.at([null, 1])             // x[:, 1]        null 이 파이썬의 `:` 다
x.at(slice(1, 3))           // x[1:3]         축이 남는다
x.at([0, slice(1, 3)])      // x[0, 1:3]
x.at(slice(null, null, 2))  // x[::2]
x.at([[0, 2]])              // x[[0, 2]]      대괄호 둘 — numpy 와 같은 모양
x.at(idx)                   // x[idx]         int64 텐서
```

**슬라이스가 함수인 이유**: `x.at([1, 3])` 이 "축 0 은 1, 축 1 은 3" 인지 "1:3 을
자른다" 인지 배열만으로는 안 갈린다. 파이썬의 `x[1:3]` 도 실은 `x[slice(1, 3)]` 로
풀리므로 같은 이름을 쓴다 — 새로 배울 것이 아니라 원래 그 자리에 있던 이름이다.

**맨 바깥 배열은 언제나 축 목록이다.** 적게 주면 남은 축은 통째로 온다. 번호표로
고르려면 한 겹 더 싼다 — numpy 의 `x[0, 1]` 과 `x[[0, 1]]` 이 갈리는 것과 같다.

`at()` 은 값을 안 만든다. 전부 기존 메서드로 넘기므로 **골든이 이미 그 값들을
지킨다** — 이 메서드가 지는 책임은 어느 문으로 보내는가뿐이다. 기존 메서드는 그대로
있고, 없애는 것이 아니라 문을 하나 더 낸 것이다.

**참·거짓 마스크는 안 받는다.** `x[mask]` 는 `await x.maskedSelect(mask)` 로 남는다 —
결과의 길이가 값에 달려 있어 GPU 에서 한 번 읽어야 알 수 있고, 그것 하나 때문에
`at()` 을 비동기로 만들면 나머지 모든 쓰임이 이유 없이 `await` 를 달게 된다.

### 데이터 먹이기

`torch.utils.data` 자리다. **배치는 GPU 텐서라 `scope()` 안에서 받아야 한다** —
적재기가 대신 감쌀 수 없다. 텐서가 구역 밖으로 나가는 것이 목적이기 때문이다.

```ts
const set = new data.TensorDataset(images, labels);
const [train, valid] = data.randomSplit(set, [800, 200]);
const loader = new data.DataLoader(train, { batchSize: 32, shuffle: true });

for (let epoch = 0; epoch < 10; epoch++) {
  for (const [x, y] of loader) {           // 동기 반복자다
    await scope(async () => {
      opt.zeroGrad();
      const loss = crit.call(model.call(x), y);
      loss.backward();
      opt.step();
    });
  }
}
loader.length;    // 표본 수가 아니라 **배치 수**. torch 와 같다
```

섞기는 `manualSeed` 를 따른다 — torch 는 적재기에 별도 generator 를 두는데 여기서는
호스트 줄기 하나를 쓴다. 씨앗 하나가 층 초기화·dropout·텐서 팩토리에 이어 배치
순서까지 되돌린다. 에폭마다 다시 섞는 것은 torch 와 같다.

**`sampler` 와 `num_workers` 는 없다.** 앞은 지금 받쳐 줄 것이 없어서이고(이름만
놓으면 넣은 것이 조용히 무시된다), 뒤는 워커로 GPU 손잡이가 안 건너가서다. 있는
것은 `shuffle`·`dropLast`·`Subset`·`randomSplit`·`ConcatDataset` 이다.

### 저장하고 이어서 하기

**형식은 safetensors 다.** torch 의 `save`/`load` 는 pickle 이라 브라우저로 옮길 수도
옮겨서도 안 된다. 대신 이쪽을 들면 **파이썬 `borch`·numpy·HF 도구가 같은 파일을
읽는다** — 브라우저에서 학습해 자기 컴퓨터로 가져가는 길이 그것으로 열린다.

```ts
import { save, load, prefixed, unprefixed, numbersToMeta, metaToNumbers } from "borch";

const state = opt.stateDict();
const bytes = await save(
  { ...prefixed("model", model.stateDict()), ...prefixed("opt", state.tensors) },
  { ...numbersToMeta("opt", state.numbers),
    ...numbersToMeta("sched", sched.stateDict()) },
);
// bytes 는 Uint8Array — IndexedDB 에 넣든 파일로 내리든 쓰는 쪽 몫이다
```

되돌릴 때는 **모델·옵티마이저·스케줄러를 같은 인자로 다시 세운 뒤** 얹는다.

```ts
const read = load(bytes);
model.loadStateDict(unprefixed("model", read.tensors));
opt.loadStateDict({ tensors: unprefixed("opt", read.tensors),
                    numbers: metaToNumbers("opt", read.metadata) });
sched.loadStateDict(metaToNumbers("sched", read.metadata));
```

**가중치만 되돌리면 안 된다.** 모멘텀·스텝 계수기·스케줄러의 에폭이 같이 가야 재개한
다음 스텝이 안 끊고 돌린 것과 같은 수를 낸다. `npm run serialize:ts` 가 그것을 비트
단위로 확인한다 — 열 스텝을 통으로 돌린 궤적과 다섯에서 끊었다 이은 궤적이 정확히
같아야 통과하고, 같은 러너가 그 파일을 **numpy 로만** 다시 뜯어 본다.

값은 언제나 float32 로 나간다. borch 의 `int64`·`bool` 은 이름표라 머리의
`__metadata__` 에 실린다 — 4 바이트짜리 몸에 `I64` 라고 적으면 남의 리더가 깨진다.

### 어디서 도는가

WebGPU 가 필요하다. **없으면 폴백하지 않고 거절한다.** 여기 있던 TF.js 판은 WebGPU 를
못 얻으면 WebGL 로 조용히 내려갔고, 그 때문에 한동안 **CPU 소프트웨어 경로에서 잰
성능 수치**를 GPU 의 것으로 읽었다. 조용히 느려지느니 안 도는 편이 낫다.

`--headed` 로만 잰다. 헤드리스 브라우저는 소프트웨어 래스터라이저(SwiftShader)를
주는데 **예외를 안 던지고 수만 이상해진다.** 그래서 러너가 어댑터를 먼저 찍고,
벤치와 정확도는 소프트웨어 어댑터에서 아예 거부한다.

### 얼마나 되나

Apple Metal 과 NVIDIA(RTX 4090) 두 벤더에서 **골든이 전건 같다** — 손으로 쓴
WGSL 이 Metal 전용이 아니라는 뜻이다. (4090 쪽은 그때의 표로 쟀다. 표가 자란 뒤로는
그 기계를 못 써서 다시 안 쟀고, 안 잰 것을 잰 것처럼 적지 않는다.)
벤치의 ResNet-18 자체도 진짜 torch 와 순방향·손실·역방향이 맞는 것을 확인했다.

**TF.js 판을 지운 근거가 이 표다.** 같은 기계·같은 벤치에서 나란히 잰 기록이고,
지금은 왼쪽 열이 없다.

| CIFAR ResNet-18, 배치 64 | TF.js 판 (지금은 없다) | **borch.ts** | **borch_webgpu** |
|---|---|---|---|
| ms/step | 154.9 | **118.5** | 123.4 |
| 에폭 | 2.02분 | **1.55분** | 1.61분 |
| 시험 정확도 (10 에폭) | 60.4% | **65.5%** | 안 쟀다 |

오른쪽 두 열은 **같은 커널**이다. 차이 4.9ms 가 파이썬을 한 번 지나는 값이고,
그것이 이 결속의 값을 재는 유일한 수다.

설계와 실측 근거는 [BORCH-TS.md](BORCH-TS.md) 에 있다.

## 일부러 지원하지 않는 것

`CUDA` · 사전학습 가중치 · 혼합정밀도 · 분산 · `torch.compile`

**거절 목록이 긴 것이 의도다.** GPU·저장된 모델·사전학습은 브라우저를 벗어나야 배우는 것들이고,
여기서 흉내 내면 그 교훈이 사라진다.

## 적합성

목표는 "PyTorch 재현"이 아니라 **커리큘럼이 쓰는 범위 안에서의 동등성**이다.
왜 그렇게 잡았는지와 앞으로의 순서는 [ROADMAP.md](ROADMAP.md) 에 있다.

| 등급 | 지금 |
|---|---|
| **T1 값·기울기** (`allclose 1e-5`) | **100%** — 생성 케이스 132개 |
| **T2 오류 동등** | **12/12** — 예외 종류 · 검색 가능한 메시지 9/9 |
| **T3 표현(`repr`) 동등** | **15/15** |
| **dtype 승격** | **112/112** — 4 dtype × 4 연산 × 텐서·스칼라 |
| **저장소 공유** (view·slice) | **13/13** |
| **통합 시나리오** | **6/6** — 같은 코드를 임포트만 바꿔 돌린 결과 |
| **넓은 표면** (수학·모양·functional) | **67/67** |
| **흔한 API 이름** | **144/144** |
| T4 비트 동등 | **명시적 비목표** |

그중 **53건은 값이 아니라 "기울기가 흐르는가"만 묻는다.** 값만 대조하는 검사는
그래프가 끊긴 것을 못 본다 — 값은 맞기 때문이다. 실제로 GPU 쪽의 `roll` 과
`masked_select` 가 그렇게 조용히 끊겨 있었고 그때 골든은 전부 초록이었다.

그리고 **골든 2641건**이 세 구현을 **같은 기대값**에 대조한다. 코어는 브라우저
전용(1·3 차원 합성곱처럼 코어가 일부러 거절하는 것) 53 건을 빼고 2545 건을 본다 —
없는 것을 물으면 그건 검사가 아니라 오답이다. 진짜 torch 를 브라우저에 넣을 수
없어서, 네이티브에서 기대값을 굳혀 브라우저로 들고 간다.

```bash
uv run --with numpy --with torch python tests/golden.py dump   # 1단계: 굳힌다
uv run --with numpy python tests/golden.py check               # 2단계: 대조한다
uv run --with numpy python tests/export_json.py                # 3단계: 밖으로 뽑는다
```

3단계가 `tests/golden.json`(722KB)을 만든다. **파이썬이 아닌 구현도 이 기대값을 쓸 수
있게** 하려는 것이다 — 진짜 torch 를 돌려 얻은 숫자가 이 저장소에서 가장 비싼 자산인데,
파이썬 안에만 두면 다음 구현은 검증 없이 자란다.

**케이스 본문은 안 들어 있다.** `lambda L: L.amax(...)` 는 기계적으로 다른 언어가 되지
않는다. 받는 쪽은 같은 이름의 케이스를 자기 언어로 쓰고, 그 답을 여기서 맞춘다 —
비싼 절반(숫자)은 건너가고 싼 절반(호출 한 줄)은 다시 쓴다. borch.ts 가 실제로 그렇게
쓰고 1779 건을 지난다.

**이름이 안 맞으면 러너가 그것을 센다.** 한동안 안 셌는데, 그때 골든에 없는 이름
일곱을 들고 있으면서 골든의 다른 일곱을 안 쓴 상태가 "859 중 859, 0 건 남음" 으로
보였다 — 개수가 같아서 맞물렸다.

```bash
uv run --with numpy --with torch python tests/conformance.py
```

## 라이선스

Apache-2.0 · PI Lab

의존은 numpy(BSD-3-Clause) 하나다. 순수 파이썬 휠이라 다른 것을 묶어 팔지 않는다.

> **브라우저에 띄우는 쪽은 Pyodide 를 함께 서빙하게 된다.** Pyodide 는 MPL-2.0 이고,
> 실행 형태로 배포하면 **소스를 구할 길을 알려야 한다**(MPL §3.2). 우리 코드로 번지지는
> 않는다 — 파일 단위 약한 카피레프트라 borch 는 Apache-2.0 그대로다.
> 페이지 어딘가에 이 한 줄을 두면 된다:
>
> > 이 페이지는 [Pyodide](https://github.com/pyodide/pyodide) 를 포함하며 Mozilla Public
> > License 2.0 을 따릅니다. 소스는 해당 저장소에서 받을 수 있습니다.
>
> 무엇에 기대고 무엇을 지켜야 하는지는 [THIRD-PARTY.md](THIRD-PARTY.md) 에 정리했다.
