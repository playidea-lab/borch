"""`torch.report(**facts)` — what a support engineer asks for, as one dictionary.

    rep = torch.report(images=len(ds), classes=len(ds.classes), model="frozen backbone",
                       train_s=4.2, accuracy=0.996)
    rep["warnings"]        # [] when nothing is off; else the reasons, in words
    json.dumps(rep)        # → report.json, sent beside model.onnx

Everything here is already counted somewhere — the adapter name, the validation faults
and the first of them, whether the device is alive or lost, what is held and pooled,
how many dispatches and pipelines, the wheel's version, the bundle's VERSION file when
the page came from one. This puts them in one place with a timestamp, so the answer to
"it does not work" is a file rather than a conversation. No file names and no pixels:
the facts are about the machine and the run, not the data.
"""
import time

import js as _js

from ._data import backend
from ._ops import _ts, dispatches, memory, pooled

# The adapter names that mean "this is the CPU drawing pictures of a GPU" — the same
# words tests/browser/launch.py refuses a timing on.
_SOFTWARE = ("swiftshader", "llvmpipe", "software", "lavapipe")


def _device():
    dev = _ts.device()
    faults = dev.faults
    lost = dev.lost
    lost_str = None
    if lost is not None and str(type(lost).__name__) not in ("JsNull", "jsnull"):
        try:
            lost_str = f"{lost.reason}: {lost.message}"
        except Exception:                                                # noqa: BLE001
            lost_str = str(lost)
    return {
        "faults": int(faults.count),
        "first_fault": str(faults.first)[:300] if int(faults.count) else None,
        "alive": bool(dev.alive),
        "lost": lost_str,
        "pipelines": int(getattr(dev, "pipelineCount", 0) or 0),
    }


def _bundle_version():
    """The VERSION file beside the page, when the page is the offline bundle."""
    try:
        from ._hub import _page_base
        from pyodide.ffi import run_sync
        response = run_sync(_js.fetch(_page_base() + "VERSION"))
        if not response.ok:
            return None
        text = str(run_sync(response.text()))
        return text.strip() if text.lstrip().startswith("borch") else None
    except Exception:                                                    # noqa: BLE001
        return None


def report(**facts):
    """The machine, the run and `facts` as one dictionary, with `warnings` filled in."""
    try:
        import importlib.metadata as md
        wheel = md.version("pyborch")
    except Exception:                                                    # noqa: BLE001
        wheel = None
    adapter = backend()
    dev = _device()
    out = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "adapter": adapter,
        "adapter_features": str(getattr(_ts.Device, "adapterFeatures", "") or ""),
        "browser": str(_js.navigator.userAgent),
        "wheel": wheel,
        "bundle": _bundle_version(),
        "memory_mb": round(memory()["bytes"] / 1e6, 1),
        "pooled_mb": round(pooled()["bytes"] / 1e6, 1),
        "dispatches": dispatches(),
        **dev,
        **facts,
    }
    warnings = []
    if any(word in adapter.lower() for word in _SOFTWARE):
        warnings.append(f"the adapter is a software rasteriser ({adapter}) — every timing is the CPU's; the GPU was not handed over")
    if dev["faults"]:
        warnings.append(f"{dev['faults']} validation fault(s); the first: {dev['first_fault']}")
    if dev["lost"]:
        warnings.append(f"the device was lost: {dev['lost']}")
    if not dev["alive"]:
        warnings.append("the device is not alive")
    out["warnings"] = warnings
    return out
