"""Streaming a frozen backbone through a bounded window — borch.ts's scale primitives on the
WebGPU surface (`docs/SCALE.md` Steps 3, 5, 7). Bridges to `_ts.trainBlock` /
`streamTrainStep` / `streamSequential`, so the workbench can fine-tune a backbone larger than
the GPU by streaming its frozen weights.

These are borch's own, not torch names (there is no `torch.streaming`); they sit under
`torch.streaming` on this surface. Async on the borch.ts side — `run_sync` awaits through JSPI,
the same machinery every readback uses, so a caller writes no `await`.
"""
import js as _js
from pyodide.ffi import create_proxy as _create_proxy, run_sync as _run_sync, to_js as _to_js

from ._base import _js_options, handle, wrap

_ts = _js.borch


class Window:
    """A bounded slice of GPU memory the frozen weights stream through. `free()` releases it."""

    def __init__(self, js_window):
        self._w = js_window

    def free(self):
        self._w.free()

    @property
    def used(self):
        return int(self._w.used)


def window(byte_size):
    """A window of `byte_size` bytes (capped at the device's binding tier)."""
    return Window(_ts.device().window(int(byte_size)))


def train_block(module, offload=False):
    """Turn a frozen (`apply_lora`'d) `Module` into a streamable train block. `offload` frees the
    module's resident base weights after reading their bytes, so the backbone is streamed rather
    than resident — the block is then usable only through streaming."""
    opts = _js_options(offload=True) if offload else _js_options()
    return _run_sync(_ts.trainBlock(module._m, opts))


def stream_train_step(win, input, blocks, loss, *, loss_params=None):
    """One training step over the streamed blocks. `loss` is a Python `f(output_tensor) ->
    scalar_tensor`; `loss_params` are resident trainable tensors used inside it (a head). Fills
    each adapter's `.grad` and returns the scalar loss."""
    bridge = _create_proxy(lambda js_out: handle(loss(wrap(js_out))))
    opts = {}
    if loss_params is not None:
        opts["lossParams"] = _to_js([handle(p) for p in loss_params])
    try:
        out = _run_sync(_ts.streamTrainStep(
            win._w, handle(input), _to_js(list(blocks)), bridge, _js_options(**opts)))
    finally:
        bridge.destroy()
    return wrap(out)


def stream_sequential(win, input, blocks, *, int8=False, f16=False):
    """A no-grad forward over the streamed blocks — inference on a streamed backbone. `int8`/`f16`
    stream each weight quantised (a quarter / half the window bytes)."""
    opts = {}
    if int8:
        opts["int8"] = True
    if f16:
        opts["f16"] = True
    out = _run_sync(_ts.streamSequential(
        win._w, handle(input), _to_js(list(blocks)), _js_options(**opts)))
    return wrap(out)
