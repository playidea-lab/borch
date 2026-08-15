# 남의 것 — 무엇에 기대고, 무엇을 지켜야 하나

borch 는 Apache-2.0 이다. 여기 적은 것은 **실행할 때 필요한 남의 코드**와
그 조건이다. 라이선스는 기억으로 적지 않고 **받은 파일에서 직접 확인했다**.

> **이 저장소는 아래 것들을 재배포하지 않는다.** `vendor/` 는 `.gitignore` 에 있고,
> 커밋되는 것은 sha256 몇 줄(`tests/browser/assets.lock`)뿐이다.
> 다만 **브라우저에 띄우는 쪽은 재배포하게 된다** — 그때 아래 조건이 적용된다.

---

## 코어 `borch`

| | 라이선스 | 확인 |
|---|---|---|
| **numpy** | BSD-3-Clause | 휠 안 `numpy-2.0.2.dist-info/LICENSE.txt` |

이게 전부다. 순수 파이썬 휠 하나에 의존은 numpy 뿐이다.

브라우저에서 쓰려면 Pyodide 가 필요하지만, 그건 **호스트 페이지가 싣는 것**이지
borch 가 묶어 파는 것이 아니다.

## 브라우저 `borch-webgpu` · `borch.ts`

| | 라이선스 | 확인 |
|---|---|---|
| **Pyodide** (파이썬 쪽만) | **MPL-2.0** | pyodide/pyodide 0.27.2 의 `LICENSE` |
| **CPython 표준 라이브러리** (파이썬 쪽만) | PSF License | `python_stdlib.zip` 에 동봉 |
| **numpy** (파이썬 쪽만) | BSD-3-Clause | 위와 같음 |

**`borch.ts` 는 이 표가 비어 있다.** TypeScript 와 WGSL 만이고 실행 시 의존이 없다.
브라우저의 WebGPU 를 직접 부른다.

여기 **TensorFlow.js 가 있었다**(Apache-2.0, Copyright 2024 Google LLC). 파이썬 쪽
GPU 구현이 그 위에 서 있었고, `tf.min.js` 와 `tf-backend-webgpu.min.js` 를 CDN 에서
받아 페이지가 실었다. 손으로 쓴 WGSL 로 갈아치우면서 **의존이 통째로 없어졌다** —
재배포하는 쪽이 지켜야 할 조건도 그만큼 줄었다.

---

## 지켜야 하는 것

### MPL-2.0 (Pyodide) — **소스를 구할 길을 알려야 한다**

이것 하나가 성질이 다르다. MPL 은 **파일 단위 약한 카피레프트**다.

- 우리 코드로 **번지지 않는다.** Apache-2.0 인 borch 와 한 페이지에 있어도
  borch 가 MPL 이 되지 않는다("Larger Work" 조항)
- 그러나 **Pyodide 를 실행 형태로 배포하면**(= 페이지에서 `pyodide.asm.wasm` 등을
  서빙하면) 그 파일들의 **소스 형태를 구할 방법을 받는 사람에게 알려야 한다**(§3.2)

실무적으로는 배포물 옆에 이 줄을 두면 된다:

> 이 페이지는 Pyodide (https://github.com/pyodide/pyodide) 를 포함하며,
> Mozilla Public License 2.0 을 따릅니다. 소스는 위 주소에서 받을 수 있습니다.

### PSF · BSD-3-Clause — 표시를 남긴다

저작권 표시와 라이선스 전문을 함께 둔다. 추가 의무는 없다.

---

## 데이터

**CIFAR-10** 은 이 저장소에 없다(`.gitignore`). 받아서 쓰는 쪽은 관례대로 인용한다.

> Krizhevsky, A. *Learning Multiple Layers of Features from Tiny Images.* 2009.

명시적 라이선스가 붙어 있지 않으므로, 연구·학습 용도를 벗어나 재배포할 계획이면
출처를 먼저 확인하는 편이 안전하다.

## PyTorch 와의 관계

borch 는 PyTorch(BSD-3-Clause)의 **코드를 가져오지 않았다.** API 모양만 맞췄고,
값 대조는 진짜 torch 를 **테스트에서만** 부른다(`dev` 추가 의존).

이름과 `sys.modules["torch"] = borch` 에 대해서는 README 가 이미 경고를 달고
있다. 상표 문제로 번질 수 있는 자리라, 공개 배포 전에 한 번 짚어보는 편이 좋다.

---

## 이 문서의 한계

여기 적은 것은 **파일에서 확인한 사실**과 각 라이선스 조문이 말하는 바다.
법률 자문이 아니다 — 특히 MPL-2.0 항목은 실제 배포 형태에 따라 달라지므로,
공개 전에 라이선스를 보는 사람에게 한 번 확인받는 것이 맞다.
