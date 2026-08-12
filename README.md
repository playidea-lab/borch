# nanotorch

**numpy 위에 얹은 PyTorch 모양의 얇은 층.** 설치 없이 브라우저에서 PyTorch 문법을 연습한다.

```python
import nanotorch as torch          # 또는 sys.modules["torch"] 로 심어 쓴다

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
>>> torch.nn.LSTM(2, 2)
NanoTorchError: nn.LSTM 은(는) 브라우저 축소판에 없습니다.
자기 컴퓨터에서 `uv add torch` 로 진짜 PyTorch 를 쓰세요 — 축소판은 문법 연습용이고,
없는 것을 흉내 내면 틀린 것을 배우게 됩니다.
```

조용히 다른 값을 내느니 시끄럽게 멈춘다.

## 어떻게 보증하는가

`tests/test_diff.py` 가 **같은 연산을 진짜 torch 와 nanotorch 양쪽에 넣고 숫자를 비교한다.**
76개, 커버리지 86%.

```bash
uv run --with pytest --with numpy --with torch pytest tests/
```

이 검사가 첫 실행에서 잡은 것: PyTorch 의 `BatchNorm2d` 는 **같은 forward 안에서 분산을
두 가지로 쓴다** — 정규화는 편향(ddof=0), `running_var` 갱신은 비편향(ddof=1).
둘 다 편향으로 두면 출력이 2.6% 어긋난다. 코드를 읽어서는 나오지 않는 종류다.

---

## 지원 범위

| | |
|---|---|
| **텐서** | 모양·브로드캐스팅·dtype 승격 규칙 · 인덱싱 · reshape/view/permute/squeeze |
| **autograd** | `requires_grad` · `backward()` · `.grad` · `no_grad()` · `detach()` · 누적 |
| **축약** | `sum`·`mean`·`max`·`min`(values/indices) · `std`(편향/비편향) — 전부 역전파 포함 |
| **nn** | `Module` · `Linear` · `Conv2d` · `MaxPool2d` · `Embedding` · `LayerNorm` · `BatchNorm2d` · `Dropout` · `Sequential` · `ModuleList` |
| **손실** | `MSELoss` · `BCELoss` · `BCEWithLogitsLoss` · `CrossEntropyLoss` |
| **optim** | `SGD`(momentum) · `Adam` · `lr_scheduler.StepLR` |
| **데이터** | `Dataset` · `TensorDataset` · `DataLoader` · 샘플러 · `random_split` · `collate_fn` |
| **저장** | `state_dict` · `load_state_dict` · `save`/`load` (Pyodide 가상 FS 에서도 된다) |

## 일부러 지원하지 않는 것

`CUDA` · 사전학습 가중치 · `TransformerEncoderLayer` · `LSTM`/`GRU`/`RNN` · 혼합정밀도

**거절 목록이 긴 것이 의도다.** GPU·저장된 모델·사전학습은 브라우저를 벗어나야 배우는 것들이고,
여기서 흉내 내면 그 교훈이 사라진다.

## 적합성

목표는 "PyTorch 재현"이 아니라 **커리큘럼이 쓰는 범위 안에서의 동등성**이다.
왜 그렇게 잡았는지와 앞으로의 순서는 [ROADMAP.md](ROADMAP.md) 에 있다.

| 등급 | 지금 |
|---|---|
| **T1 값·기울기** (`allclose 1e-5`) | **100%** — 생성 케이스 132개 |
| T2 오류 동등 | 일부 |
| T3 표현(`repr`) 동등 | 없음 |
| T4 비트 동등 | **명시적 비목표** |

```bash
uv run --with numpy --with torch python tests/conformance.py
```

## 라이선스

Apache-2.0 · PI Lab
