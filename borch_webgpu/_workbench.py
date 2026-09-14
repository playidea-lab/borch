"""`workbench` bound to this surface — the hub and the exporter are here.

The recipe itself is the numpy core's (`borch/_workbench.py`), so it is verified without a
browser; what a surface adds is `hub` and `onnx`, which the session asks for by name and
refuses for by name. The import is inside the call because this module is imported while
`borch_webgpu` is still being built.
"""

from borch._workbench import Session                     # noqa: F401 — re-exported below
from borch._workbench import candidates as _candidates, compare as _compare, pick as _pick
from borch._workbench import interval, resolution, say   # noqa: F401 — read as they are


def setup(files, **config):
    """Every setting in one call, on the WebGPU surface. Returns a `Session`."""
    import borch_webgpu

    return Session(borch_webgpu, files, **config)


def candidates(budget_mb=60, task="image-classification"):
    """The registry's models this data could be handed to, smallest first."""
    import borch_webgpu

    return _candidates(borch_webgpu, budget_mb=budget_mb, task=task)


def compare(files, **config):
    """Every candidate under the budget, same split, best first."""
    import borch_webgpu

    return _compare(borch_webgpu, files, **config)


def pick(files, **config):
    """The smallest backbone that clears the bar — `(name, rows)`."""
    import borch_webgpu

    return _pick(borch_webgpu, files, **config)
