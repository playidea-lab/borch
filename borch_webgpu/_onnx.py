"""`torch.onnx.export` — **the file is written over the wall, the values are read here.**

## What it is

`onnx.export(model, args, f=None)` traces one forward of a binding `Module` and
writes ONNX — the file every serving runtime reads. borch.ts does the tracing and
the encoding (`traceOnnx`, `encodeOnnx`); what this side adds is the two things
that differ, the same split as `_serialize.py`:

1. **Reading a weight's values is synchronous here.** borch.ts's `exportOnnx` awaits
   each readback; the Python API is synchronous throughout (`tests/browser/sync_probe.py`
   keeps that promise), and `.numpy()` already makes the round trip synchronously —
   so the encoder is handed a reader that does that, and the file comes out with no
   `await` anywhere.
2. **Where the bytes go.** A browser has no real filesystem; Pyodide supplies a virtual
   one, so `export(model, x, "model.onnx")` runs as written and the page reads the file
   back out with `FS.readFile`. With no `f` the bytes are returned.

## What it refuses

An op with no ONNX spelling (borch.ts names it: `cannot export erf`), a training-mode
network, an adaptive pool to anything but 1 × 1. `dynamic_axes` is not a dict here:
the leading dimension is dynamic unless `dynamic_axes=None` is passed explicitly,
which is what nearly every call of torch's spells out at length.
"""
import numpy as _np

from ._base import _js, _ts, _to_js, handle, wrap


def _named(model):
    """The model's names for its tensors — parameters and buffers, `state_dict` order."""
    try:
        return list(model.state_dict().items())
    except Exception:                                   # a model with no state_dict — unnamed is fine
        return []


def export(model, args, f=None, *, input_names=None, output_names=None,
           opset_version=17, dynamic_axes=True, **unused):
    """Trace `model(args)` and write ONNX. Returns the bytes when `f` is None."""
    if isinstance(args, (tuple, list)):
        if len(args) != 1:
            raise NotImplementedError("onnx.export: one input tensor is traced here, "
                                      f"not {len(args)}")
        args = args[0]
    if unused:
        raise TypeError("onnx.export: not here — " + ", ".join(sorted(unused)))
    options = {"opset": int(opset_version), "dynamicBatch": bool(dynamic_axes)}
    if input_names:
        options["inputName"] = str(input_names[0])
    if output_names:
        options["outputName"] = str(output_names[0])

    # **The forward runs here, not over the wall.** A model written in Python has its
    # `forward` in Python; borch.ts only sees the tensor ops it makes, and those are
    # what the tracer records between `beginTrace` and `endTrace`.
    was_training = bool(getattr(model, "training", False))
    model.eval()
    _ts.onnx.beginTrace()
    try:
        from ._ops import no_grad
        with no_grad():
            out = model(args)
    finally:
        nodes = _ts.onnx.endTrace()
        if was_training:
            model.train()
    names = _js.Map.new()
    for name, p in _named(model):
        names.set(handle(p), name)
    plan = _ts.onnx.planOnnx(nodes, handle(args), handle(out), names,
                             _to_js(options, dict_converter=_js.Object.fromEntries))

    # **Every value is read before the encoder runs.** The encoder calls back for each
    # weight, and a callback entered from JavaScript has no suspension point under it —
    # `run_sync` there fails with "No suspender" (measured). Read here, in Python, then
    # hand the encoder a lookup that suspends nothing.
    values = _js.Map.new()
    for t in plan.weights:
        flat = wrap(t).numpy().ravel().astype(_np.float32)
        values.set(t, _js.Float32Array.new(_to_js(flat)))

    result = _ts.onnx.encodeOnnx(plan, values.get)
    data = result.bytes.to_bytes()
    if f is None:
        return data
    if hasattr(f, "write"):
        f.write(data)
        return None
    with open(f, "wb") as out:
        out.write(data)
    return None
