/**
 * The eight top-level recurrences — `torch.lstm`, `torch.gru`,
 * `torch.rnn_tanh`, `torch.rnn_relu`, and their `_cell` counterparts.
 *
 * **The only difference from the layer (`nn.Recurrent`) is where the
 * weights come from.** The layer holds its own; this side **receives them
 * as a list.** torch offers both, and what the layer calls internally is
 * this side.
 *
 * **This claimed the step equations lived here in one copy, and there were two.**
 * `nn.RNNBase.run` wrote out its own layer loop, and the day came the sentence warned
 * about: this side fell behind, refusing `bidirectional` and inter-layer `dropout` as
 * *not here* while the layer one file away had both. `rnnApply` below stands up an
 * `RNNBase` and fills its slots now, so the layer loop really is in one place; what
 * stays here is `rnnStep`, which the four `_cell` names need and the layer does not
 * call.
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
import { RNNBase } from "./nn.js";
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

/**
 * The names the flat weight list fills, in order.
 *
 * **It is `namedParameters()`'s own order** — four per *direction*, the reverse one
 * second, layers outermost, and `weight_hr` last within a direction under a
 * projection. Measured against `torch._VF.lstm` with a two-layer bidirectional net;
 * with one direction, directions and layers cannot be told apart.
 */
function slotNames(layers: number, directions: number, hasBiases: boolean,
                   projSize: number): string[] {
  const out: string[] = [];
  for (let layer = 0; layer < layers; layer++) {
    for (let d = 0; d < directions; d++) {
      const tail = `_l${layer}` + (d ? "_reverse" : "");
      out.push("weight_ih" + tail, "weight_hh" + tail);
      if (hasBiases) out.push("bias_ih" + tail, "bias_hh" + tail);
      if (projSize) out.push("weight_hr" + tail);
    }
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
 * Several steps. `hx` is `(layer · directions, batch, H)`.
 *
 * **This was a second layer loop and it is a call now.** The header above says the
 * step equations live here in one copy; they did not — `nn.RNNBase.run` wrote them
 * out again, and this side then fell behind it: `bidirectional` and inter-layer
 * `dropout` were refused here on the ground that they were not implemented, and by
 * then they were, one file away. So the whole of this function is standing up an
 * `RNNBase` with the flags the caller gave, filling its named slots from the flat
 * list, and calling `run`.
 *
 * **`projSize` is read off the shapes**, because torch has no seat for it here:
 * `weight_hh` is `(gates·H, proj or H)`, and torch infers it the same way.
 *
 * **`dropout` only crosses when it can act.** At one layer it has nowhere to go and
 * the layer warns about exactly that, while `torch._VF.lstm` does not (measured).
 */
export function rnnApply(
  kind: RnnKind, input: Tensor, h0: Tensor, c0: Tensor | null,
  params: readonly Tensor[], options: RnnOptions = {},
): { output: Tensor; hidden: Tensor; cell: Tensor } {
  const {
    hasBiases = true, numLayers = 1, dropout = 0, train = false,
    bidirectional = false, batchFirst = false,
  } = options;
  const gates = gatesOf(kind);
  const first = params[0];
  const second = params[1];
  if (!first || !second) {
    throw new RuntimeError("expected at least two weights, got " + params.length);
  }
  const hidden = ((first.shape[0] ?? 0) / gates) | 0;
  const carried = second.shape[1] ?? hidden;
  const projSize = carried === hidden ? 0 : carried;
  const directions = bidirectional ? 2 : 1;
  const slots = slotNames(numLayers, directions, hasBiases, projSize);
  if (params.length !== slots.length) {
    const per = slots.length / (numLayers * directions);
    throw new RuntimeError(
      `expected ${slots.length} weights but got ${params.length} ` +
        `(${numLayers} layers x ${directions} directions x ${per}).`);
  }
  const mode = kind === "lstm" ? "LSTM" : kind === "gru" ? "GRU" : "RNN";
  const layer = new RNNBase(mode, first.shape[1] ?? 0, hidden, numLayers,
    hasBiases, false, numLayers > 1 ? dropout : 0, bidirectional, projSize);
  layer.nonlinearity = kind === "rnn_relu" ? "relu" : "tanh";
  if (train) layer.train(); else layer.eval();
  // **Not `loadStateDict`** — that copies values into the layer's own tensors, and
  // the gradient has to reach the ones the caller handed over, unchanged.
  layer.installFlat(slots, params);
  const src = batchFirst ? input.movedim(0, 1) : input;
  const got = layer.run(src, h0, c0);
  return {
    output: batchFirst ? got.output.movedim(0, 1) : got.output,
    hidden: got.hidden,
    cell: got.cell,
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
