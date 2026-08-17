/**
 * 최상위 순환 여덟 — `torch.lstm` · `torch.gru` · `torch.rnn_tanh` · `torch.rnn_relu`
 * 와 그 `_cell` 짝.
 *
 * **층(`nn.Recurrent`)과 다른 것은 가중치를 어디서 얻는가뿐이다.** 층은 자기가 들고
 * 있고 이쪽은 **목록으로 받는다.** torch 도 그 둘을 다 주고, 층이 안에서 부르는 것이
 * 이 함수 쪽이다.
 *
 * 그래서 걸음 식을 여기 한 벌만 둔다. 층 쪽과 두 벌로 두면 **게이트 순서가 갈리는
 * 날**이 오고, 그때 모양은 같고 값만 틀린다 — 이 저장소가 셀과 층 사이에서 이미 그
 * 이유로 식을 공유하고 있다.
 *
 * ## 게이트 순서
 *
 * `weight_ih` 의 **행 순서**가 규약이다(실측):
 *
 *     LSTM  i, f, g, o          GRU  r, z, n          RNN  (하나)
 *
 * 바꾸면 값만 갈리고 모양은 그대로다.
 */

import { RuntimeError } from "./errors.js";
import { Tensor } from "./tensor.js";

export type RnnKind = "lstm" | "gru" | "rnn_tanh" | "rnn_relu";

function gatesOf(kind: RnnKind): number {
  return kind === "lstm" ? 4 : kind === "gru" ? 3 : 1;
}

/** `x·Wᵀ (+ b)`. 편향이 없으면 안 더한다. */
function affine(x: Tensor, w: Tensor, b: Tensor | null): Tensor {
  const out = x.linear(w);
  return b === null ? out : out.add(b);
}

/** 게이트 하나를 잘라 온다. `(배치, 게이트·H)` 에서 `k` 번째 토막이다. */
function gate(z: Tensor, k: number, H: number): Tensor {
  return z.narrow(1, k * H, H);
}

/**
 * 한 걸음. **네 종류가 전부 여기를 지난다.**
 *
 * @returns `[다음 h, 다음 c]` — c 는 LSTM 만 쓰고 나머지는 h 를 그대로 넘긴다.
 */
export function rnnStep(
  kind: RnnKind, x: Tensor, h: Tensor, c: Tensor,
  wIh: Tensor, wHh: Tensor, bIh: Tensor | null, bHh: Tensor | null,
): [Tensor, Tensor] {
  const H = h.shape[1] ?? 0;
  const gi = affine(x, wIh, bIh);
  const gh = affine(h, wHh, bHh);
  if (kind === "rnn_tanh" || kind === "rnn_relu") {
    const z = gi.add(gh);
    const next = kind === "rnn_tanh" ? z.unary("tanh") : z.unary("relu");
    return [next, next];
  }
  if (kind === "gru") {
    // **`n` 게이트만 `r` 을 은닉 쪽에 곱한다.** 입력 쪽에는 안 곱한다 — 그 자리를
    // 바꾸면 값이 그럴듯하게 틀린다.
    const r = gate(gi, 0, H).add(gate(gh, 0, H)).unary("sigmoid");
    const z = gate(gi, 1, H).add(gate(gh, 1, H)).unary("sigmoid");
    const n = gate(gi, 2, H).add(r.mul(gate(gh, 2, H))).unary("tanh");
    const one = Tensor.full([], 1);
    const next = one.sub(z).mul(n).add(z.mul(h));
    return [next, next];
  }
  const z = gi.add(gh);
  const i = gate(z, 0, H).unary("sigmoid");
  const f = gate(z, 1, H).unary("sigmoid");
  const g = gate(z, 2, H).unary("tanh");
  const o = gate(z, 3, H).unary("sigmoid");
  const nextC = f.mul(c).add(i.mul(g));
  return [o.mul(nextC.unary("tanh")), nextC];
}

/** 평평한 가중치 목록을 층별 넷으로 쪼갠다. 차례는 `[w_ih, w_hh, b_ih, b_hh]` 다. */
function layerWeights(params: readonly Tensor[], layers: number, hasBiases: boolean):
  [Tensor, Tensor, Tensor | null, Tensor | null][] {
  const per = hasBiases ? 4 : 2;
  if (params.length !== per * layers) {
    throw new RuntimeError(
      `가중치가 ${per * layers} 개여야 하는데 ${params.length} 개다 ` +
        `(층 ${layers} × ${per}).`);
  }
  const out: [Tensor, Tensor, Tensor | null, Tensor | null][] = [];
  for (let k = 0; k < layers; k++) {
    const at = k * per;
    out.push([
      params[at] as Tensor, params[at + 1] as Tensor,
      hasBiases ? (params[at + 2] as Tensor) : null,
      hasBiases ? (params[at + 3] as Tensor) : null,
    ]);
  }
  return out;
}

export interface RnnOptions {
  hasBiases?: boolean;
  numLayers?: number;
  dropout?: number;
  train?: boolean;
  bidirectional?: boolean;
  batchFirst?: boolean;
}

/**
 * 여러 걸음. `hx` 는 `(층, 배치, H)` 다.
 *
 * **양방향과 층간 드롭아웃은 거절한다.** 없는 것을 반쪽으로 흉내 내면, 앞쪽은 모양이
 * 절반이라 시끄럽게 걸리지만 **드롭아웃 쪽은 값이 그럴듯한 채로 갈린다**(정칙화가
 * 안 걸린 학습). 둘 다 여기서 멈춘다.
 */
export function rnnApply(
  kind: RnnKind, input: Tensor, h0: Tensor, c0: Tensor | null,
  params: readonly Tensor[], options: RnnOptions = {},
): { output: Tensor; hidden: Tensor; cell: Tensor } {
  const {
    hasBiases = true, numLayers = 1, dropout = 0, train = false,
    bidirectional = false, batchFirst = false,
  } = options;
  if (bidirectional) {
    throw new RuntimeError("양방향 순환(bidirectional=true)은 여기 없다.");
  }
  if (train && dropout) {
    throw new RuntimeError(`층간 드롭아웃(dropout=${dropout})은 여기 없다.`);
  }
  const gates = gatesOf(kind);
  const weights = layerWeights(params, numLayers, hasBiases);
  let x = batchFirst ? input.movedim(0, 1) : input;
  const steps = x.shape[0] ?? 0;
  const H = ((weights[0]?.[0].shape[0] ?? 0) / gates) | 0;

  const lastH: Tensor[] = [];
  const lastC: Tensor[] = [];
  for (const [layer, [wIh, wHh, bIh, bHh]] of weights.entries()) {
    let h = h0.select(0, layer);
    let c = c0 === null ? Tensor.zeros(h.shape) : c0.select(0, layer);
    const outs: Tensor[] = [];
    for (let t = 0; t < steps; t++) {
      [h, c] = rnnStep(kind, x.select(0, t), h, c, wIh, wHh, bIh, bHh);
      outs.push(h);
    }
    x = Tensor.stack(outs, 0);
    lastH.push(h);
    lastC.push(c);
  }
  void H;
  const output = batchFirst ? x.movedim(0, 1) : x;
  return {
    output,
    hidden: Tensor.stack(lastH, 0),
    cell: Tensor.stack(lastC, 0),
  };
}

/** `torch.lstm` — **셋을 편다**(`출력, h_n, c_n`). 층 쪽처럼 묶지 않는다. */
export function lstm(input: Tensor, hx: readonly [Tensor, Tensor],
                     params: readonly Tensor[], options: RnnOptions = {}):
  [Tensor, Tensor, Tensor] {
  const got = rnnApply("lstm", input, hx[0], hx[1], params, options);
  return [got.output, got.hidden, got.cell];
}

export function gru(input: Tensor, hx: Tensor, params: readonly Tensor[],
                    options: RnnOptions = {}): [Tensor, Tensor] {
  const got = rnnApply("gru", input, hx, null, params, options);
  return [got.output, got.hidden];
}

export function rnnTanh(input: Tensor, hx: Tensor, params: readonly Tensor[],
                        options: RnnOptions = {}): [Tensor, Tensor] {
  const got = rnnApply("rnn_tanh", input, hx, null, params, options);
  return [got.output, got.hidden];
}

export function rnnRelu(input: Tensor, hx: Tensor, params: readonly Tensor[],
                        options: RnnOptions = {}): [Tensor, Tensor] {
  const got = rnnApply("rnn_relu", input, hx, null, params, options);
  return [got.output, got.hidden];
}

export function lstmCell(input: Tensor, hx: readonly [Tensor, Tensor],
                         wIh: Tensor, wHh: Tensor,
                         bIh: Tensor | null = null,
                         bHh: Tensor | null = null): [Tensor, Tensor] {
  return rnnStep("lstm", input, hx[0], hx[1], wIh, wHh, bIh, bHh);
}

export function gruCell(input: Tensor, hx: Tensor, wIh: Tensor, wHh: Tensor,
                        bIh: Tensor | null = null,
                        bHh: Tensor | null = null): Tensor {
  return rnnStep("gru", input, hx, hx, wIh, wHh, bIh, bHh)[0];
}

export function rnnTanhCell(input: Tensor, hx: Tensor, wIh: Tensor, wHh: Tensor,
                            bIh: Tensor | null = null,
                            bHh: Tensor | null = null): Tensor {
  return rnnStep("rnn_tanh", input, hx, hx, wIh, wHh, bIh, bHh)[0];
}

export function rnnReluCell(input: Tensor, hx: Tensor, wIh: Tensor, wHh: Tensor,
                            bIh: Tensor | null = null,
                            bHh: Tensor | null = null): Tensor {
  return rnnStep("rnn_relu", input, hx, hx, wIh, wHh, bIh, bHh)[0];
}
