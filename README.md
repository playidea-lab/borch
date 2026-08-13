# browsertorch

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

## 설치

순수 파이썬 휠 하나(34KB)다. 의존성은 numpy 뿐이고, Pyodide 에는 numpy 가 이미 있다.

```bash
uv add ./browsertorch-1.3.0-py3-none-any.whl        # 릴리스에서 받은 파일
```

브라우저(Pyodide)에서는 휠 바이트를 가상 파일시스템에 써넣고 `micropip` 으로 건다.

```js
// 파일 이름을 그대로 써야 한다 — micropip 이 이름에서 패키지명과 버전을 읽는다.
py.FS.writeFile("/browsertorch-1.3.0-py3-none-any.whl", new Uint8Array(wheelBytes));
await py.runPythonAsync(`
import micropip
await micropip.install("emfs:/browsertorch-1.3.0-py3-none-any.whl")
`);
```

```python
import sys, browsertorch
sys.modules["torch"] = browsertorch          # 이 뒤로는 `import torch` 가 그대로 통한다
```

> **`torch` 로 심는 것은 강력하고 위험하다.** 그 뒤로는 **남의 라이브러리가 하는 `import torch` 도**
> 축소판을 받는다. 학습자 한 명의 연습 환경에서는 그게 편의지만, 다른 코드가 섞인 곳에서는
> 원인을 못 찾는 오류가 된다. 섞이는 자리에서는 `import browsertorch as torch` 를 쓴다.

> **저장소가 private 이라 릴리스 URL 을 그대로 `micropip.install()` 에 넣을 수 없다.**
> 익명 요청은 404 를 받는다(실제로 그렇게 해보고 알았다). 공개로 돌리면 URL 한 줄로 끝난다.

```python
import browsertorch as torch

w = torch.tensor(3.0, requires_grad=True)
loss = (w - 5.0) ** 2
loss.backward()
print(w.grad.item())               # -4.0
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

## 어떻게 보증하는가

`tests/test_diff.py` 가 **같은 연산을 진짜 torch 와 browsertorch 양쪽에 넣고 숫자를 비교한다.**
76개, 커버리지 86%.

```bash
uv run --with pytest --with numpy --with torch pytest tests/
```

이 검사가 첫 실행에서 잡은 것: PyTorch 의 `BatchNorm2d` 는 **같은 forward 안에서 분산을
두 가지로 쓴다** — 정규화는 편향(ddof=0), `running_var` 갱신은 비편향(ddof=1).
둘 다 편향으로 두면 출력이 2.6% 어긋난다. 코드를 읽어서는 나오지 않는 종류다.

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
또는 아래의 자매 라이브러리다.

## 그 위가 필요하면 — `browsertorch-webgpu`

이것(코어)은 **numpy 위에서 MNIST 급까지**다. 그 경계를 넘고 싶으면 별도 배포판이 있다.

| | 코어 `browsertorch` | 자매 `browsertorch-webgpu` |
|---|---|---|
| 무엇 위에 | numpy | TF.js WebGPU |
| 휠 | 순수 파이썬 34KB | 순수 파이썬(TF.js 는 페이지가 싣는다) |
| 어디서 | 어디서나 | **브라우저 안에서만** |
| 천장 | MNIST 급 | **CIFAR ResNet-18 이 에폭 약 2분** (실측) |
| 읽히는가 | 그것이 전부다 | 아니다. 성능이 목적이다 |

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
| **nn** | `Module` · `Linear` · `Conv2d` · `MaxPool2d` · `Embedding` · `LayerNorm` · `BatchNorm2d` · `Dropout` · `Sequential` · `ModuleList` |
| **순환** | `RNN` · `LSTM` · `GRU` — 다층 · `batch_first` · 초기 상태 |
| **트랜스포머** | `MultiheadAttention` · 인코더·디코더 층 · `nn.Transformer` — 불리언·실수 마스크 · `norm_first` · gelu |
| **손실** | `MSELoss`·`L1Loss`·`SmoothL1Loss`·`BCELoss`·`BCEWithLogitsLoss`·`CrossEntropyLoss`·`NLLLoss` |
| **optim** | `SGD`(momentum·weight_decay) · `Adam` · `AdamW` · `RMSprop` — `param_groups` · `state_dict` |
| **스케줄러** | `StepLR` · `MultiStepLR` · `ExponentialLR` · `CosineAnnealingLR` · `LambdaLR` · `ReduceLROnPlateau` |
| **데이터** | `Dataset` · `TensorDataset` · `Subset` · `ConcatDataset` · `DataLoader` · `WeightedRandomSampler` · `random_split(generator=)` · `collate_fn` |
| **저장** | `state_dict` · `load_state_dict` · `save`/`load` · 버퍼(`running_mean` 등) 포함 |
| **nn.functional** | 25종 — 활성·손실·`pad`·`normalize`·`cosine_similarity`·`one_hot`·`layer_norm`·`embedding` |

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

그리고 **골든 385건**이 코어와 자매 라이브러리를 **같은 기대값**에 대조한다.
(코어는 그중 자매 전용 52건을 건너뛰어 333건을 본다 — 코어가 일부러 거절하는 것을
코어에게 물으면 그건 검사가 아니라 오답이다.) 진짜 torch 를
브라우저에 넣을 수 없어서, 네이티브에서 기대값을 굳혀 브라우저로 들고 간다.

```bash
uv run --with numpy --with torch python tests/golden.py dump   # 1단계: 굳힌다
uv run --with numpy python tests/golden.py check               # 2단계: 대조한다
```

```bash
uv run --with numpy --with torch python tests/conformance.py
```

## 라이선스

Apache-2.0 · PI Lab

의존은 numpy(BSD-3-Clause) 하나다. 순수 파이썬 휠이라 다른 것을 묶어 팔지 않는다.

> **브라우저에 띄우는 쪽은 Pyodide 를 함께 서빙하게 된다.** Pyodide 는 MPL-2.0 이고,
> 실행 형태로 배포하면 **소스를 구할 길을 알려야 한다**(MPL §3.2). 우리 코드로 번지지는
> 않는다 — 파일 단위 약한 카피레프트라 browsertorch 는 Apache-2.0 그대로다.
> 페이지 어딘가에 이 한 줄을 두면 된다:
>
> > 이 페이지는 [Pyodide](https://github.com/pyodide/pyodide) 를 포함하며 Mozilla Public
> > License 2.0 을 따릅니다. 소스는 해당 저장소에서 받을 수 있습니다.
>
> 무엇에 기대고 무엇을 지켜야 하는지는 [THIRD-PARTY.md](THIRD-PARTY.md) 에 정리했다.
