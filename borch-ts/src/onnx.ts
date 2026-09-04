/**
 * `torch.onnx.export` — **one forward, traced, written as an ONNX file.**
 *
 * Training happens here; serving happens wherever the weights are wanted, and the
 * file every runtime reads is ONNX. torch's answer is `torch.onnx.export(model, args,
 * f)`; this one is `exportOnnx(model, sample)` and it hands the bytes back, because a
 * page has no file to write to.
 *
 * **How it traces.** The tensor methods a network is built from (`convND`, `unary`,
 * `binary`, `poolND`, `reshape`, `matmul`, `linear`, `adaptiveAvgPool`,
 * `batchNormEval`) each pass through `traced`, which does nothing until an export is
 * open. Then every outermost call is one node — an op built from other ops (a
 * `linear` is a transpose and a matmul) records itself and not its parts. The
 * forward runs under `noGrad`, in eval mode.
 *
 * **What it refuses.** An op with no ONNX spelling here (`cannot export …`, naming
 * it), a training-mode network (its batch norms would trace the batch's statistics),
 * an adaptive pool to anything but 1 × 1. The refusal names the op, so the reader
 * knows what to change rather than what to debug.
 *
 * **The bytes are the wire format itself** — a protobuf writer of a few dozen lines,
 * the same choice as `serialize.ts` writing safetensors by hand: no dependency, and
 * the checker that matters is whether somebody else's reader (ORT Web, `onnx.checker`)
 * opens the file. `test/onnx.ts` asks exactly that.
 */
import { noGrad } from "./autograd.js";
import type { Module } from "./nn.js";
import type { Tensor } from "./tensor.js";

export type Attr = number | readonly number[] | string;

export interface TraceNode {
  op: string;
  inputs: (Tensor | null)[];
  output: Tensor;
  attrs: Record<string, Attr>;
}

interface Trace {
  nodes: TraceNode[];
  /** Above 0 while inside a recorded op — nested ops then pass through unrecorded. */
  depth: number;
}

let active: Trace | null = null;

/**
 * Runs `body`, and — while an export is open — records it as one node. Costs one
 * comparison when nothing is open, so the hooks stay in the hot path.
 */
export function traced(
  op: string, inputs: readonly (Tensor | null)[], attrs: Record<string, Attr>,
  body: () => Tensor,
): Tensor {
  const trace = active;
  if (!trace || trace.depth > 0) return body();
  trace.depth += 1;
  let out: Tensor;
  try {
    out = body();
  } finally {
    trace.depth -= 1;
  }
  trace.nodes.push({ op, inputs: [...inputs], output: out, attrs });
  return out;
}

// ── the wire format ────────────────────────────────────────────────────────

/** A protobuf message under construction. Fields are appended in call order. */
class Pb {
  private readonly parts: Uint8Array[] = [];
  private size = 0;

  private push(bytes: Uint8Array): void {
    this.parts.push(bytes);
    this.size += bytes.length;
  }

  private varint(value: bigint): void {
    const out: number[] = [];
    let v = value;
    while (v >= 0x80n) {
      out.push(Number(v & 0x7fn) | 0x80);
      v >>= 7n;
    }
    out.push(Number(v));
    this.push(Uint8Array.from(out));
  }

  private key(field: number, wire: number): void {
    this.varint(BigInt((field << 3) | wire));
  }

  /** An `int64` (or enum). Negative values take the ten-byte two's complement form. */
  int(field: number, value: number): this {
    this.key(field, 0);
    this.varint(BigInt.asUintN(64, BigInt(Math.trunc(value))));
    return this;
  }

  float(field: number, value: number): this {
    this.key(field, 5);
    const b = new Uint8Array(4);
    new DataView(b.buffer).setFloat32(0, value, true);
    this.push(b);
    return this;
  }

  bytes(field: number, value: Uint8Array): this {
    this.key(field, 2);
    this.varint(BigInt(value.length));
    this.push(value);
    return this;
  }

  string(field: number, value: string): this {
    return this.bytes(field, new TextEncoder().encode(value));
  }

  message(field: number, value: Pb): this {
    return this.bytes(field, value.finish());
  }

  finish(): Uint8Array {
    const out = new Uint8Array(this.size);
    let at = 0;
    for (const p of this.parts) {
      out.set(p, at);
      at += p.length;
    }
    return out;
  }
}

// onnx.proto field numbers. Written out here so the file has one place to be wrong.
const MODEL = { irVersion: 1, producerName: 2, producerVersion: 3, graph: 7, opsetImport: 8 };
const OPSET = { domain: 1, version: 2 };
const GRAPH = { node: 1, name: 2, initializer: 5, input: 11, output: 12 };
const NODE = { input: 1, output: 2, name: 3, opType: 4, attribute: 5 };
const ATTR = { name: 1, f: 2, i: 3, s: 4, ints: 8, type: 20 };
const ATTR_TYPE = { FLOAT: 1, INT: 2, STRING: 3, INTS: 7 };
const TENSOR = { dims: 1, dataType: 2, name: 8, rawData: 9 };
const DATA_TYPE = { FLOAT: 1, INT64: 7 };
const VALUE_INFO = { name: 1, type: 2 };
const TYPE = { tensorType: 1 };
const TENSOR_TYPE = { elemType: 1, shape: 2 };
const SHAPE = { dim: 1 };
const DIM = { value: 1, param: 2 };

const IR_VERSION = 8;
const DEFAULT_OPSET = 17;
/** Attributes ONNX types as `float`; every other number here is an `int`. */
const FLOAT_ATTRS = new Set(["epsilon", "alpha", "beta", "momentum"]);

function attribute(name: string, value: Attr): Pb {
  const a = new Pb().string(ATTR.name, name);
  if (typeof value === "string") return a.string(ATTR.s, value).int(ATTR.type, ATTR_TYPE.STRING);
  if (typeof value === "number") {
    return FLOAT_ATTRS.has(name)
      ? a.float(ATTR.f, value).int(ATTR.type, ATTR_TYPE.FLOAT)
      : a.int(ATTR.i, value).int(ATTR.type, ATTR_TYPE.INT);
  }
  for (const v of value) a.int(ATTR.ints, v);
  return a.int(ATTR.type, ATTR_TYPE.INTS);
}

function floatTensor(name: string, dims: readonly number[], data: Float32Array): Pb {
  const t = new Pb();
  for (const d of dims) t.int(TENSOR.dims, d);
  t.int(TENSOR.dataType, DATA_TYPE.FLOAT).string(TENSOR.name, name);
  return t.bytes(TENSOR.rawData, new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}

function int64Tensor(name: string, values: readonly number[]): Pb {
  const raw = new Uint8Array(values.length * 8);
  const view = new DataView(raw.buffer);
  values.forEach((v, i) => view.setBigInt64(i * 8, BigInt(Math.trunc(v)), true));
  return new Pb().int(TENSOR.dims, values.length).int(TENSOR.dataType, DATA_TYPE.INT64)
    .string(TENSOR.name, name).bytes(TENSOR.rawData, raw);
}

function valueInfo(name: string, shape: readonly number[], batchParam: string | null): Pb {
  const dims = new Pb();
  shape.forEach((d, i) => {
    const dim = new Pb();
    if (i === 0 && batchParam !== null) dim.string(DIM.param, batchParam);
    else dim.int(DIM.value, d);
    dims.message(SHAPE.dim, dim);
  });
  const tensorType = new Pb().int(TENSOR_TYPE.elemType, DATA_TYPE.FLOAT).message(TENSOR_TYPE.shape, dims);
  return new Pb().string(VALUE_INFO.name, name)
    .message(VALUE_INFO.type, new Pb().message(TYPE.tensorType, tensorType));
}

// ── the ops ────────────────────────────────────────────────────────────────

const UNARY: Record<string, string> = {
  relu: "Relu", sigmoid: "Sigmoid", tanh: "Tanh", exp: "Exp", log: "Log", neg: "Neg",
  abs: "Abs", sqrt: "Sqrt", floor: "Floor", ceil: "Ceil", softplus: "Softplus",
  // MobileNetV3's pair. Both are opset 14 ops; the default here is 17.
  hardswish: "HardSwish", hardsigmoid: "HardSigmoid",
};
const BINARY: Record<string, string> = {
  add: "Add", sub: "Sub", mul: "Mul", div: "Div", pow: "Pow", maximum: "Max", minimum: "Min",
};

/** The previous node's output, in a chain — `emit` may spell one traced op as several. */
const PREV = Symbol("previous");
type EmittedInput = Tensor | null | typeof PREV;

interface Emitted {
  opType: string;
  inputs: EmittedInput[];
  attrs: Record<string, Attr>;
  /** An int64 constant this node takes as an extra input (Reshape's shape). */
  shapeInput?: readonly number[];
}

/** One traced node, spelled as ONNX (one node, or a chain) — or a refusal that names the op. */
function emit(node: TraceNode, batch: number, dynamicBatch: boolean): Emitted[] {
  const { op, inputs, attrs } = node;
  if (op === "unary:silu") {
    // ONNX has no SiLU; it is `x · sigmoid(x)`, two nodes. EfficientNet is made of it —
    // the workbench's frozen-backbone export stopped here first.
    const [x = null] = inputs;
    return [{ opType: "Sigmoid", inputs: [x], attrs: {} }, { opType: "Mul", inputs: [x, PREV], attrs: {} }];
  }
  if (op.startsWith("unary:")) {
    const opType = UNARY[op.slice(6)];
    if (!opType) throw new Error(`cannot export ${op.slice(6)}: no ONNX spelling for it here`);
    return [{ opType, inputs, attrs: {} }];
  }
  if (op.startsWith("binary:")) {
    const opType = BINARY[op.slice(7)];
    if (!opType) throw new Error(`cannot export ${op.slice(7)}: no ONNX spelling for it here`);
    return [{ opType, inputs, attrs: {} }];
  }
  if (op === "ConvFused") {
    // The file is the unfused network's: the epilogue is written as the ops it stands
    // for, so a reader that knows Conv, Add and Relu needs nothing else.
    const { relu, residual, ...conv } = attrs;
    const [x = null, w = null, b = null, r = null] = inputs;
    const chain: Emitted[] = [{ opType: "Conv", inputs: [x, w, b], attrs: conv }];
    if (residual === 1) chain.push({ opType: "Add", inputs: [PREV, r], attrs: {} });
    if (relu === 1) chain.push({ opType: "Relu", inputs: [PREV], attrs: {} });
    return chain;
  }
  return [emitOne(op, inputs, attrs, batch, dynamicBatch)];
}

function emitOne(
  op: string, inputs: (Tensor | null)[], attrs: Record<string, Attr>, batch: number, dynamicBatch: boolean,
): Emitted {
  switch (op) {
    case "Conv":
    case "MatMul":
    case "BatchNormalization":
    case "ReduceMean":
      return { opType: op, inputs, attrs };
    case "linear": {
      const x = inputs[0];
      if (x && x.shape.length === 2) return { opType: "Gemm", inputs, attrs: { transB: 1 } };
      throw new Error(`cannot export linear on a ${x?.shape.length ?? 0}-D input: only 2-D reaches Gemm here`);
    }
    case "pool": {
      const kind = attrs["kind"];
      const rest = { ...attrs };
      delete rest["kind"];
      if (rest["divisor_override"] !== undefined) throw new Error("cannot export avg_pool with divisor_override");
      delete rest["divisor_override"];
      if (kind === "max") {
        delete rest["count_include_pad"];
        return { opType: "MaxPool", inputs, attrs: rest };
      }
      delete rest["dilations"];
      return { opType: "AveragePool", inputs, attrs: rest };
    }
    case "adaptive_avg_pool":
      if (attrs["output_size"] === 1) return { opType: "GlobalAveragePool", inputs, attrs: {} };
      throw new Error(`cannot export adaptive_avg_pool(${String(attrs["output_size"])}): only 1 × 1 is GlobalAveragePool`);
    case "reshape": {
      const want = [...(attrs["shape"] as readonly number[])];
      const x = inputs[0];
      // A leading dimension equal to the batch is "the batch, whatever it is" — ONNX's
      // 0 copies the input's — so the file runs at a batch it was not traced at.
      if (dynamicBatch && x && want[0] === batch && x.shape[0] === batch) want[0] = 0;
      return { opType: "Reshape", inputs: [x ?? null], attrs: {}, shapeInput: want };
    }
    default:
      throw new Error(`cannot export ${op}: no ONNX spelling for it here`);
  }
}

// ── the export ─────────────────────────────────────────────────────────────

export interface ExportOptions {
  /** The graph input's name. `input` by default. */
  inputName?: string;
  /** The graph output's name. `output` by default. */
  outputName?: string;
  /** ONNX opset. 17 by default — what ORT Web 1.29 and `torch.onnx.export` agree on. */
  opset?: number;
  /**
   * Name the leading dimension `N` instead of pinning it to the sample's batch, so the
   * file takes any batch. On by default — torch's `dynamic_axes`, without the dict.
   */
  dynamicBatch?: boolean;
}

/** What `exportOnnx` returns beside the bytes — for the check, and for the curious. */
export interface Exported {
  bytes: Uint8Array;
  /** Node op types, in graph order. */
  ops: string[];
  /** Initializer names — every weight the file carries. */
  initializers: string[];
}

/**
 * What a trace leaves behind, before any value is read back: the graph as nodes, the
 * names, and the weights the file will carry. `encodeOnnx` turns it into bytes once
 * the values are in hand — split this way because reading a GPU tensor is asynchronous
 * in borch.ts and synchronous in the Python binding, and the encoder should not care.
 */
export interface OnnxPlan {
  emitted: Emitted[][];
  nodes: TraceNode[];
  names: Map<Tensor, string>;
  sample: Tensor;
  output: Tensor;
  /** Every tensor the file carries as an initializer, in first-use order. */
  weights: Tensor[];
  inputName: string;
  outputName: string;
  opset: number;
  dynamicBatch: boolean;
}

/**
 * Opens a trace. Every outermost traced op from here to `endTrace` becomes a node.
 * `traceOnnx` does this around `model.forward`; the Python binding does it around a
 * forward written in Python, whose ops still pass through the same tensor methods.
 */
export function beginTrace(): void {
  if (active) throw new Error("a trace is already open");
  active = { nodes: [], depth: 0 };
}

/** Closes the trace and hands back what it recorded. */
export function endTrace(): TraceNode[] {
  const trace = active;
  if (!trace) throw new Error("no trace is open");
  active = null;
  return trace.nodes;
}

/**
 * Plans the graph from recorded nodes: names, the output, the weights the file will
 * carry. `names` gives the model's own names for its parameters and buffers — a
 * tensor not in it, not the sample and not made by a node is a `const_N`.
 */
export function planOnnx(
  nodes: TraceNode[], sample: Tensor, output: Tensor, names: Map<Tensor, string>,
  options: ExportOptions = {},
): OnnxPlan {
  const inputName = options.inputName ?? "input";
  const outputName = options.outputName ?? "output";
  const opset = options.opset ?? DEFAULT_OPSET;
  const dynamicBatch = options.dynamicBatch ?? true;
  const batch = sample.shape[0] ?? 1;
  if (output === sample) throw new Error("the forward returned its input — nothing to export");

  names.set(sample, inputName);
  const produced = new Set<Tensor>();
  const counts = new Map<string, number>();
  const emitted = nodes.map((node) => emit(node, batch, dynamicBatch));
  emitted.forEach((chain, i) => {
    const node = nodes[i];
    const last = chain[chain.length - 1];
    if (!node || !last) return;
    produced.add(node.output);
    if (node.output === output) {
      names.set(node.output, outputName);
      return;
    }
    const n = (counts.get(last.opType) ?? 0) + 1;
    counts.set(last.opType, n);
    names.set(node.output, `${last.opType.toLowerCase()}_${n}`);
  });
  if (!produced.has(output)) {
    throw new Error("the forward's result was not made by any traced op — nothing reaches the output");
  }

  // Every input no node made and the caller did not hand in is a weight the file carries.
  const weights: Tensor[] = [];
  const seen = new Set<Tensor>();
  let constants = 0;
  for (const e of emitted.flat()) {
    for (const t of e.inputs) {
      if (!t || t === PREV || t === sample || produced.has(t) || seen.has(t)) continue;
      seen.add(t);
      if (!names.has(t)) names.set(t, `const_${++constants}`);
      weights.push(t);
    }
  }
  return { emitted, nodes, names, sample, output, weights, inputName, outputName, opset, dynamicBatch };
}

/** Runs the forward under the tracer and plans the graph. Synchronous. */
export function traceOnnx(model: Module, sample: Tensor, options: ExportOptions = {}): OnnxPlan {
  const wasTraining = model.training;
  model.eval();
  beginTrace();
  let output: Tensor;
  let nodes: TraceNode[];
  try {
    output = noGrad(() => model.forward(sample));
  } finally {
    nodes = endTrace();
    if (wasTraining) model.train();
  }
  const names = new Map<Tensor, string>();
  for (const [name, t] of Object.entries(model.namedParameters())) names.set(t, name);
  for (const [name, t] of Object.entries(model.namedBuffers())) names.set(t, name);
  return planOnnx(nodes, sample, output, names, options);
}

/** Writes the planned graph, taking each weight's values from `read`. Synchronous. */
export function encodeOnnx(plan: OnnxPlan, read: (t: Tensor) => Float32Array): Exported {
  const { emitted, nodes, names } = plan;
  const graph = new Pb();
  const initializers: string[] = [];
  for (const t of plan.weights) {
    const name = names.get(t) ?? "";
    graph.message(GRAPH.initializer, floatTensor(name, t.shape, read(t)));
    initializers.push(name);
  }

  const ops: string[] = [];
  emitted.forEach((chain, i) => {
    const node = nodes[i];
    if (!node) return;
    const finalName = names.get(node.output) ?? "";
    chain.forEach((e, j) => {
      const pb = new Pb();
      // A trailing absent input (a convolution's bias) is left off rather than written as
      // "" — the spec allows either, and ORT Web's WebGPU convolution reads "" as a
      // tensor with no shape (measured: `The input shape must not be empty`).
      const inputs = [...e.inputs];
      while (inputs.length && inputs[inputs.length - 1] === null) inputs.pop();
      for (const t of inputs) {
        pb.string(NODE.input, t === PREV ? `${finalName}_${j - 1}` : t ? names.get(t) ?? "" : "");
      }
      const outName = j === chain.length - 1 ? finalName : `${finalName}_${j}`;
      if (e.shapeInput) {
        const shapeName = `${outName}_shape`;
        graph.message(GRAPH.initializer, int64Tensor(shapeName, e.shapeInput));
        initializers.push(shapeName);
        pb.string(NODE.input, shapeName);
      }
      pb.string(NODE.output, outName).string(NODE.name, `${e.opType}_${i}_${j}`).string(NODE.opType, e.opType);
      for (const [k, v] of Object.entries(e.attrs)) pb.message(NODE.attribute, attribute(k, v));
      graph.message(GRAPH.node, pb);
      ops.push(e.opType);
    });
  });
  const batchParam = plan.dynamicBatch ? "N" : null;
  graph.string(GRAPH.name, "borch")
    .message(GRAPH.input, valueInfo(plan.inputName, plan.sample.shape, batchParam))
    .message(GRAPH.output, valueInfo(plan.outputName, plan.output.shape, batchParam));

  const model = new Pb().int(MODEL.irVersion, IR_VERSION)
    .string(MODEL.producerName, "borch.ts")
    .message(MODEL.opsetImport, new Pb().string(OPSET.domain, "").int(OPSET.version, plan.opset))
    .message(MODEL.graph, graph);
  return { bytes: model.finish(), ops, initializers };
}

/**
 * Traces `model.forward(sample)` and returns the ONNX file. The model is run in
 * eval mode under `noGrad`, and put back the way it was.
 */
export async function exportOnnx(
  model: Module, sample: Tensor, options: ExportOptions = {},
): Promise<Exported> {
  const plan = traceOnnx(model, sample, options);
  const values = new Map<Tensor, Float32Array>();
  for (const t of plan.weights) values.set(t, await t.toArray());
  return encodeOnnx(plan, (t) => values.get(t) ?? new Float32Array());
}
