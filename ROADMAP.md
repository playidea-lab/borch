# borch — how far this goes

> The name was `minitorch` first, then `nanotorch`, before arriving here. The
> first two collided with existing projects (`edutorch` did too), because all
> three aimed at the same spot — **"small + educational + torch".** What sets this
> apart from the others is not its size or its purpose but that **it runs in a
> browser**, and the name should say so — limits included.

"Reproduce PyTorch" is not a measurable goal. It is replaced with
**conformance.**

---

## First: why reproduction cannot be the goal

### The denominator is 1,800

| | count |
|---|---|
| `torch.*` | 1,025 |
| `torch.nn.*` | 182 |
| `torch.Tensor` methods | 604 |
| ATen kernels (forward and backward each) | 863 |

What is actually used is **95 of them** — 5%. The other 95% is surface nobody uses
and that can still be wrong.

### Bit equivalence is unreachable

```
summing 100,000 float32s
  numpy  -90.82513427734375
  torch  -90.82499694824219
  diff    1.373e-04          ← 1,000× float32's eps
```

The **order** of summation alone opens that gap. Matching it would mean
reproducing torch's reduction kernel order, and that order changes again with the
SIMD width and the thread count. It is not a goal that can be chased.

### Full reproduction inverts this project's principle

borch's principle is **"an absent feature beats a wrong answer".** Chasing
reproduction shrinks the refusal list and **grows the surface that can be quietly
wrong.** Unless the verification grows in proportion, more coverage means more
danger.

---

## So the goal is this

> **Within the range the curriculum uses: real torch's values, real torch's
> errors, real torch's printed form.**

### Fidelity grades

| grade | meaning | target | at the time |
|---|---|---|---|
| **T1 values** | values, shapes and gradients equal under `allclose(1e-5)` | **100%** of the supported range | **100%** (132/132) |
| **T2 errors** | the same exception type plus a searchable message | the main errors | **12/12 · 9/9** |
| **T3 printed form** | `print(t)` and `repr` identical | the common ones | **15/15** |
| **T4 bits** | identical bits | **a non-goal** | — |

Writing T4 down as a non-goal is the heart of this document. Left unwritten,
somebody eventually chases it.

### How it is measured

Two layers.

- `tests/test_diff.py` — 76 written by hand. They aim at specific traps
- `tests/conformance.py` — 132 **generated from a table.** Shapes, dtypes and
  arguments multiplied out

The second is the route to growing it. One more row in the table adds dozens of
cases.

```bash
uv run --with numpy --with torch python tests/conformance.py     # the score
uv run --with pytest --with numpy --with torch pytest tests/     # everything
```

---

## 순서

### 1. ~~T2 — 오류를 진짜와 맞춘다~~ · **완료**

메시지 규격을 정했다 — **한국어 설명 + torch 의 정규 영문 문구**.

```
행렬곱의 모양이 안 맞습니다 (3x4 @ 3x2) — 앞의 열(4)과 뒤의 행(3)이 같아야 합니다.
(torch: mat1 and mat2 shapes cannot be multiplied (3x4 and 3x2))
```

한국어만 두면 검색해서 답을 못 찾고, 영문만 베끼면 이 교재가 한국어인 이유가 사라진다.
**둘 다 넣는다** — 설명은 읽고, 영문은 검색한다.

그리고 이 작업이 **메시지가 아닌 결함 둘**을 드러냈다.

- `.item()` 이 원소 3개짜리에서 **조용히 첫 값을 돌려주고 있었다.** 이 프로젝트가
  하지 않겠다고 한 바로 그 짓이다
- `backward()` 를 두 번 불러도 그냥 통과했다. torch 는 그래프를 놓고 예외를 낸다

둘 다 고쳤고, 그래프 해제와 `retain_graph=True` 를 구현했다.

### 2. ~~T3 — `repr` 을 맞춘다~~ · **완료**

torch 의 규칙을 따랐다 — 값이 전부 정수면 `1.`, 아니면 **소수 네 자리**, 범위가 넓으면 지수.
원소는 **같은 너비로 오른쪽 정렬**(음수가 섞이면 양수 앞에 자리가 생긴다),
이어지는 줄은 `tensor(` 만큼 **8칸** 들여쓴다. 기본이 아닌 dtype 은 접미사로 붙인다.

그리고 비잎 노드는 `requires_grad=True` 가 아니라 **`grad_fn=<MulBackward0>`** 으로 찍는다.
학습자가 "이 텐서는 그래프 안에 있다"를 눈으로 아는 자리다.

```
>>> torch.tensor([-1.5, 2.0, -0.25])
tensor([-1.5000,  2.0000, -0.2500])
>>> torch.tensor([1.0], requires_grad=True) * 2
tensor([2.], grad_fn=<MulBackward0>)
```

### 3. ~~dtype 승격표를 전부 덮는다~~ · **완료**

4개 dtype × 4개 연산 × (텐서·스칼라) = **112건**을 전부 훑었고, 처음엔 54건이 갈렸다.

원인 하나가 대부분이었다 — torch 는 **범주**(bool < 정수 < 실수)로 먼저 가르고
**그 범주 안에서만** 올린다. 낮은 범주는 높은 것을 끌어올리지 않는다.

```
float32 + int64   torch: float32     numpy: float64
int64 / int64     torch: float32     numpy: float64
bool - bool       torch: RuntimeError (`~` 를 쓰라고 안내)
```

나눗셈은 정수끼리여도 기본 실수형을 내고, 뺄셈은 불리언에서 거부한다.
지금 112/112.

### 4. ~~view 의미론~~ · **완료 (그리고 이 항목은 틀린 전제였다)**

여기 "우리는 복사한다, 구현 비용이 크다"고 적어뒀었다. **재보지 않고 코드만 읽고 쓴 말이었고,
틀렸다.** numpy 의 reshape·swapaxes·슬라이스가 이미 뷰를 주고 우리는 그것을 그대로 들고
있으므로, 저장소 공유는 처음부터 되고 있었다.

```python
a = torch.zeros(4); b = a.view(2, 2); b[0, 0] = 9
a          # [9., 0., 0., 0.]  — torch 와 같다
```

실제로 갈린 것은 하나뿐이었다. torch 의 `view` 는 **메모리 순서가 어긋난 텐서를 거부**하고
`reshape` 을 쓰라고 안내한다 — 둘의 차이가 거기에 있다. 그것을 맞췄다.

저장소 공유 13가지(뷰 8종 · `clone` 은 독립 · `detach` 는 공유 · 팬시 인덱싱은 사본 ·
`view`/`reshape` 의 차이)를 검사기에 넣었다. 13/13.

**교훈**: 로드맵에 "안 된다"를 적을 때도 재보고 적어야 한다. 안 재고 적으면
있지도 않은 일을 몇 시간 계획하게 된다.

### 5. ~~`nn.RNN`~~ · **완료**

시간 방향이 파이썬 반복문이다. **그 느림이 곧 30장의 내용이기도 하다** —
순환은 앞을 끝내야 뒤를 볼 수 있어서 병렬화가 안 되고, 트랜스포머가 나온 이유가 그것이다.

`num_layers` · `batch_first` · `nonlinearity` · `bias` · `h_0` 전부 맞고,
파라미터 이름을 torch 와 같게 둬서 `state_dict` 키가 맞는다(저장·불러오기가 통한다).
`stack`·`cat` 의 역전파도 이 김에 제대로 고쳤다 — RNN 이 그 위에 서 있다.

### 6. `LSTM`·`GRU`·트랜스포머 인코더 · **완료**

커리큘럼이 요구하는 범위를 넘어선다. **그건 알고 넣는다** — 이 물건 자체로 쓸 만하게
만들려는 것이고, 그 결정은 기록해 둔다.

넣으면서 지킨 선:

- **파라미터 이름과 배치를 torch 와 같게.** LSTM 의 게이트 순서는 `i, f, g, o`,
  torch 는 Q·K·V 가중치를 `in_proj_weight` (3E, E) 하나로 묶어 든다.
  순서나 배치가 다르면 값은 그럴듯한데 체크포인트가 안 통하고, 그건 조용히 틀리는 종류다
- **GRU 의 `n` 게이트에서 `r` 은 편향까지 포함한 은닉 항에 곱한다.** 편향을 밖에 두면
  미세하게 어긋나고 눈에 안 띈다
### 7. 디코더와 `nn.Transformer` · **완료**

인코더까지가 교재 범위라 거절하려 했는데, 넣기로 했다. 디코더 층은 인코더 층과
**가운데 하나만 다르다** — `multihead_attn` 이 자기 자신이 아니라 인코더 출력을 본다.

그 김에 마스크 의미를 제대로 고쳤다. torch 의 마스크는 두 가지다.

- **불리언** — True 인 자리를 가린다(-inf)
- **실수** — 점수에 **더한다.** `generate_square_subsequent_mask` 가 주는 0/-inf 가 그것이다

전에는 실수 마스크를 "0 이 아니면 가림"으로 뭉뚱그렸다. 인과 마스크는 우연히 맞지만
**가중치를 조절하는 마스크에서 어긋난다** — 그 경우를 테스트로 박아뒀다.

### 8. 통합 리뷰에서 나온 것 · **완료**

튜토리얼처럼 쓴 코드를 양쪽에서 통째로 돌려봤다(MLP 학습·CNN·LSTM·트랜스포머·저장/불러오기).
6개 중 2개가 갈렸고, **둘 다 단위 대조가 못 잡던 종류**였다.

- **BatchNorm 역방향이 틀렸다.** 평균·분산을 numpy 로 빼서 상수처럼 썼더니 x → mean → y
  로 흐르는 길이 끊겼다. 입력 기울기가 1.17 어긋나고 **weight 기울기는 아예 안 왔다**(None).
  순방향만 대조하고 있었기 때문에 오래 남았다 — **"학습이 돌아가고 손실도 내려가는데 값이 다른"**
  가장 나쁜 종류다
- **`p.data = ndarray` 를 받아주고 있었다.** torch 는 거부한다. 관대한 것도 갈리는 것이고,
  브라우저에서 되던 코드가 자기 컴퓨터에서 깨진다

그 검사를 다섯 층에 걸어뒀다 — **가중치에 기울기가 실제로 도착하는가.**
그래프가 끊기면 `None` 으로 드러난다. 그리고 그 검사가 **또 하나를 잡았다**:
BatchNorm 의 `running_mean`·`running_var` 가 `state_dict` 에 없었다.
저장했다 불러오면 **평가 모드가 초기값으로 돌아간다** — 학습은 멀쩡해 보이고 추론만 틀린다.
버퍼(`register_buffer`) 개념을 들여와 고쳤다.

통합 시나리오는 `tests/scenario.py` 로 남겼다. 단위 대조가 연산 하나씩만 보는 반면
이쪽은 **조각이 엮였을 때**를 본다 — 세 결함 전부 그 자리에서 나왔다.

### 9. 학습 루프에 필요한 것들 · **완료**

"예제로 학습을 돌리려면 DataLoader·옵티마이저·스케줄러는 있어야 하지 않나"에서 출발했다.
**이름은 다 있었는데 쓰는 법이 달랐다** — 16가지를 재보니 13개가 갈렸다.

가장 큰 것은 `param_groups` 였다. torch 에서 학습률을 읽고 쓰는 표준 경로가
`opt.param_groups[0]["lr"]` 이고 스케줄러도 그것을 고친다. 우리는 `opt.lr` 이었고,
그러면 **남의 코드가 안 돌고 남의 스케줄러를 못 쓴다.**

그리고 **내 테스트가 그 차이를 덮고 있었다.** StepLR 대조가 torch 는 `param_groups[0]["lr"]`,
nano 는 `.lr` 로 읽고 있었다 — 양쪽을 같은 식으로 읽지 않으면 검사가 아니라 합리화다.
지금은 같은 헬퍼로 읽고, **한 값이 아니라 궤적 전체**를 본다.

더한 것: `AdamW`·`RMSprop`, 스케줄러 6종(`ReduceLROnPlateau` 포함),
옵티마이저 `state_dict`(6장의 "이어서 학습하기"가 여기 걸려 있다),
`WeightedRandomSampler`(5장이 가르치는데 없었다) · `Subset` · `ConcatDataset` ·
`Generator`(`random_split` 을 고정하는 것) · `sin`/`cos`(10장 위치 인코딩) ·
`F.mse_loss`·`F.one_hot`.

**범위는 교재가 정했다.** 교재·랩이 언급하는 이름을 전부 뽑아 없는 것만 채웠고,
남은 것은 `amp`·`backends`·`float16` — 전부 GPU 장이고 의도적 거절이다.

### 10. 넓은 표면 · **완료**

흔히 쓰는 144개를 훑으니 **56개(38%)** 뿐이었다. 88개를 채워 100% 가 됐다 —
수학 원소별 함수, 비교, `split`·`chunk`·`gather`·`flip`·`roll` 같은 모양 조작,
`topk`·`sort`·`unique`·`cumsum`, 선형대수, `nn.functional` 25종, 활성·손실 층 15종.

**이름만 맞추고 끝내지 않았다.** 있는 것을 전부 값으로 대조했고 셋이 갈렸다.

- `median` — torch 는 원소가 짝수일 때 **가운데 둘 중 작은 쪽**을 준다. numpy 는 평균을 낸다
- `cumsum`·`cumprod` — torch 는 `dim` 이 **필수**다. 기본값을 두면 남의 코드가 다르게 돈다

미분이 정의되지 않는 것들(`sign`·`floor`·`ceil`·`round`)은 기울기를 0 으로 둔다 —
torch 도 그렇게 한다. 계단 함수의 미분은 거의 모든 곳에서 0 이기 때문이다.

### 11. 리뷰 — 넓힌 뒤의 사각지대 · **완료**

88개를 더한 직후의 리뷰다. **값은 맞는데 기울기를 안 본 것**이 다시 나왔다.

- `topk`·`sort` 가 **그래프를 끊고 있었다.** 값만 떼어 돌려주니 뽑은 자리로 기울기가
  안 갔다 — top-k 샘플링이나 정렬을 끼운 손실에서 **학습이 조용히 멈춘다**
- `no_grad` 안에서 만든 **잎**의 `requires_grad` 를 꺼버렸다. torch 는 켜둔다.
  그 블록 안에서 만든 파라미터가 학습 대상에서 빠지는 종류의 차이다

커버리지를 재보니 `BatchNorm1d`·`AdaptiveAvgPool2d`·활성 층 4종·`ConcatDataset`·
`WeightedRandomSampler` 가 **한 번도 안 돌아본 채** 있었다. 전부 통과했지만
**검사가 없었을 뿐이지 맞다는 근거는 없었다** — BatchNorm2d 가 그렇게 오래 틀려 있었다.

그리고 "도는가"가 아니라 "맞는가"를 묻게 고쳤다. `WeightedRandomSampler` 는
길이만 보던 것을 **가중치 큰 쪽이 실제로 더 자주 뽑히는지**로 바꿨다.

---

## 로드맵이 비었다

여섯 항목이 전부 닫혔다. 다음을 정할 때 참고할 것:

- **커리큘럼이 요구하지 않는 것은 넣지 않는다.** 표면이 늘면 조용히 틀릴 자리가 는다
- **재보고 적는다.** 4번은 "안 된다"고 적었는데 이미 되고 있었다
- 남은 자연스러운 후보: `torch.compile` 같은 것 말고, **학습자가 실제로 만나는 것** —
  `nn.functional` 의 나머지, `Tensor.scatter_`·`gather`, 옵티마이저 몇 종

---

## 하지 않을 것

| | 왜 |
|---|---|
| **CUDA · 분산 · 혼합정밀도** | 브라우저에 없다. 흉내 내면 교훈이 사라진다 |
| **사전학습 가중치** | 받아오는 것 자체가 배울 점이다 |
| **JIT · `torch.compile`** | 범위 밖 |
| **속도** | 아래 참고 |

### 속도 — 실측

처음에 "네이티브 torch 보다 수백 배 느리다"고 적었는데 **재보니 틀렸다.**
행렬곱은 양쪽 다 BLAS 를 부르고, 작은 텐서에서는 torch 의 디스패처 오버헤드가 더 커서
오히려 borch 가 빠르다.

| | torch (CPU) | borch (네이티브) | borch (브라우저) |
|---|---|---|---|
| matmul 512² | 0.28ms | 0.13ms | 82.4ms |
| 작은 연산 50회 | 0.15ms | 0.17ms | 0.74ms |
| MLP 학습 1스텝 | 0.36ms | 0.18ms | 3.34ms |
| conv2d 순방향 | 0.22ms | 0.44ms | 1.88ms |
| conv2d 역방향 | 3.97ms | 1.11ms | 6.49ms |
| **MNIST CNN 학습 1배치** | — | — | **65.7ms** |

**MNIST CNN 1에폭이 브라우저에서 약 2분이다.** 학습이 실제로 된다.

느려지는 원인은 구현이 아니라 wasm 이다. Pyodide 의 BLAS 는 wasm 빌드라 SIMD·멀티스레드를
못 쓰고, 그래서 **큰 행렬곱만 유독 나쁘다**(294×). 실습이 실제로 하는 크기에서는 5~10× 이고,
그건 체감되지 않는다.

경계선은 **"MNIST 급까지"** 다. CIFAR·ResNet 급은 자기 컴퓨터나 원격 장비이고,
그것이 8장(GPU)이 가르치는 내용이기도 하다.

### 빠른 런타임을 목표로 삼지 않는 이유

빠르게 만들려면 Rust/C++ 로 다시 써서 wasm 으로 컴파일해야 한다.
그 순간 **읽을 수 있는 교육용 구현**이라는 성질을 잃는다 — 그게 이 프로젝트의 전부인데.

그리고 그 자리에는 이미 ONNX Runtime Web 과 WebGPU 가 있다(추론 전용).
브라우저에서 진짜로 학습을 돌려야 한다면, 답은 borch 가 아니라 원격 장비다.

**둘 다는 못 가진다.** 읽히는 구현이거나 빠른 런타임이거나.
borch 는 전자다.

---

## ADR-001: 천장은 자매 라이브러리가 올린다

- **상태**: 승인
- **맥락**

  로드맵 열한 항목이 전부 닫힌 뒤, 다음 수를 정해야 했다. 요구는 **CIFAR·ResNet 급
  학습을 브라우저에서 돌리는 것**이고, 이는 바로 위 절이 "경계선은 MNIST 급까지"라고
  적어둔 것과 정면으로 부딪힌다.

  재보면 요구 배수가 나온다. 지금 실효 처리량은 MNIST CNN 1에폭 2분에서 역산해
  **약 3 GFLOPS** 다. CIFAR-10 ResNet-18 은 학습 스텝이 이미지당 약 1.7 GFLOPs,
  5만 장이면 **에폭당 84 TFLOPs** — 지금 속도로 **7.7시간**이다.
  에폭 몇 분으로 만들려면 **300배**가 필요하다.

  | | 1에폭 |
  |---|---|
  | 지금 (3 GFLOPS) | 7.7시간 |
  | wasm SIMD (×4) | 1.9시간 |
  | WebGPU 나이브 셰이더 | ~14분 |
  | WebGPU 튜닝 (~1 TFLOPS) | ~1.4분 |

  wasm SIMD 는 3~5배다. **300배의 1%를 채우고 끝난다.** 즉 이 요구는 GPU 없이는
  어떤 방법으로도 닿지 않는다.

- **결정**

  코어 `borch` 는 **"MNIST 급까지"와 순수 파이썬 17KB 휠을 그대로 유지한다.**
  천장을 올리는 일은 **별도 배포판**(`borch-webgpu`)이 맡는다.
  설계는 [WEBGPU-DESIGN.md](WEBGPU-DESIGN.md) 에 있다.

- **근거**

  - **휠의 성질은 전염된다.** 코어에 백엔드를 넣으면 `py3-none-any` 가 아니게 되고,
    네이티브에서 진짜 torch 와 대조하는 지금의 검증 경로부터 아키텍처 종속이 된다
  - **실패 모드가 올라온다.** WebGPU 는 브라우저·드라이버별로 깨진다. 코어에 있으면
    "문법 연습하러 온 학습자가 드라이버 때문에 import 부터 실패"가 가능해진다
  - **하네스가 2자에서 3자가 된다.** 그 부담을 T1 100% 를 유지 중인 코어 위에 얹으면
    100% 가 먼저 깨진다
  - **약속이 깨진다.** 코어가 파는 것은 "임포트만 바꾸면 같은 코드"다. `device` 개념과
    비동기 읽기는 그 약속과 양립하지 않는다

- **대안**

  | 고려한 것 | 왜 안 골랐나 |
  |---|---|
  | 코어에 백엔드를 넣는다 | 위 근거 네 가지 |
  | libtorch 를 wasm 으로 포팅한다 | person-years 를 들여 표면 1,800개를 사는데, **속도는 못 산다.** 병목이 라이브러리가 아니라 wasm 이라 MKL·FBGEMM 이 없는 곳에서는 옮겨도 빨라지지 않는다 |
  | 현행 유지 | 천장이 안 올라간다. 요구를 안 푼다 |

- **결과**

  - 바로 위 "하지 않을 것 · 속도" 항목은 이제 **코어에 한정된 말**이다
  - **코어에 `device` 개념·비동기 API·GPU 코드가 들어오면 이 ADR 위반이다.**
    적어두지 않으면 언젠가 누가 그걸 한다 — 이 문서가 존재하는 이유다
  - 코어의 다음 할 일은 바뀌지 않는다: 적합성 표 확장, 커버리지 사각지대,
    그리고 배포(저장소 공개)
