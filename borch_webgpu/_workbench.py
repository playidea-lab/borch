"""`workbench` bound to this surface — the hub and the exporter are here.

The recipe itself is the numpy core's (`borch/_workbench.py`), so it is verified without a
browser; what a surface adds is `hub` and `onnx`, which the session asks for by name and
refuses for by name. The import is inside the call because this module is imported while
`borch_webgpu` is still being built.
"""

from borch._workbench import Session                     # noqa: F401 — re-exported below


def setup(files, **config):
    """Every setting in one call, on the WebGPU surface. Returns a `Session`."""
    import borch_webgpu

    return Session(borch_webgpu, files, **config)
