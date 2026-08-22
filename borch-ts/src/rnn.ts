/**
 * The eight top-level recurrences — `torch.lstm`, `torch.gru`,
 * `torch.rnn_tanh`, `torch.rnn_relu`, and their `_cell` counterparts.
 *
 * **The only difference from the layer (`nn.Recurrent`) is where the
 * weights come from.** The layer holds its own; this side **receives them
 * as a list.** torch offers both, and what the layer calls internally is
 * this side.
 *
 * So the step equations live here in one copy. Two copies, here and in the
 * layer, means a day when **the gate order diverges**, and then the shape
 * is right and only the values are wrong — this repository already shares
 * the equations between cell and layer for that same reason.
 *
 * ## Gate order
 *
 * The **row order** of `weight_ih` is the convention (measured):
 *
 *     LSTM  i, f, g, o          GRU  r, z, n          RNN  (just one)
 *
 * Change it and only the values diverge; the shape stays.
 */

import { RuntimeError } from "./errors.js";
import { Tensor } from "./tensor.js";

export type RnnKind = "lstm" | "gru" | "rnn_tanh" | "rnn_relu";

function gatesOf(kind: RnnKind): number {
  return kind === "lstm" ? 4 : kind === "gru" ? 3 : 1;
}

/** `x·Wᵀ (+ b)`. With no bias, nothing is added. */
function affine(x: Tensor, w: Tensor, b: Tensor | null): Tensor {
  const out = x.linear(w);
  return b === null ? out : out.add(b);
}

/** Slices out one gate. The `k`th segment of `(batch, gates·H)`. */
function gate(z: Tensor, k: number, H: number): Tensor {
  return z.narrow(1, k * H, H);
}

/**
 * One step. **All four kinds pass through here.**
 *
 * @returns `[next h, next c]` — only LSTM uses c; the rest pass h straight
 *   through.
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
    // **Only the `n` gate multiplies the hidden side by `r`.** The input side is not
    // multiplied — swapping those makes the values plausibly wrong.
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

/** Splits the flat weight list into four per layer. The order is
 *  `[w_ih, w_hh, b_ih, b_hh]`. */
function layerWeights(params: readonly Tensor[], layers: number, hasBiases: boolean):
  [Tensor, Tensor, Tensor | null, Tensor | null][] {
  const per = hasBiases ? 4 : 2;
  if (params.length !== per * layers) {
    throw new RuntimeError(
      `expected ${per * layers} weights but got ${params.length} ` +
        `(${layers} layers x ${per}).`);
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
 * Several steps. `hx` is `(layer, batch, H)`.
 *
 * **Bidirectional and inter-layer dropout are refused.** Half-imitating
 * something that is not there catches loudly for the first — the shape
 * comes out halved — but **dropout diverges with plausible values**
 * (training with no regularisation applied). Both stop here.
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
    throw new RuntimeError("bidirectional recurrence (bidirectional=true) is not here.");
  }
  if (train && dropout) {
    throw new RuntimeError(`dropout between layers (dropout=${dropout}) is not here.`);
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

/**
 * `torch.lstm` — **it spreads three** (`output, h_n, c_n`), rather than
 * bundling them the way the layer does.
 */
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
