"""`borch_cpu` — the catalogue's pretrained models and a head on their features, on the
machine with **no WebGPU adapter**, from Python in a browser.

    import borch_cpu
    borch_cpu.available()                                   # WebAssembly SIMD in this engine
    borch_cpu.threads()                                     # workers in the pool, 0 on one thread
    backbone = borch_cpu.load("imagenet-efficientnet-b0")   # features=True by default
    feats = backbone.features(x)                            # x: (N, 3, 224, 224) float32 NCHW → (N, 1280)
    head = borch_cpu.LinearHead(backbone.num_features, classes, lr=0.05, momentum=0.9)
    losses = head.fit(feats, labels, steps=300)
    pred = head.predict(feats).argmax(1)
    doubt = borch_cpu.suspects(feats, labels, k=5)

## Why a module of its own, and not a mode of `borch_webgpu`

`import borch_webgpu` boots borch.ts **and brings the device up**, and on the machine this
module is for that second step fails by name — there is no adapter to bring up. Making
that import succeed anyway would leave a `Tensor` with nowhere to live. So this module
loads the same bundle (`borch_webgpu/_borch.js`: borch.ts, bimm-ts, borch-hub) **without
`init()`**, and exposes the three things the labelling tool needs from borch-ts's `cpu`
namespace: a checkpoint's forward (`cpu.CpuRunner` on a graph `bimm.cpuGraphFor` built
from the same plan tables the WebGPU model uses), a linear head (`cpu.LinearHead`), and
the neighbour score — the last one from the numpy core, which never needed a device.

The workbench's first cell tries `import borch_webgpu as torch` and, when that fails by
name, comes here. Same manifest, same bytes, same tables:

    adapter → hub.load() → nn.Module → WebGPU
    none    → borch_cpu.load() → cpuGraphFor() → cpu.CpuRunner

## Threads

Where the page is cross-origin isolated (COOP `same-origin` + COEP `require-corp` — the
repository's servers send both; `file://` cannot) the first `kernels()` spawns a pool of
Web Workers over one shared wasm memory and every forward is cut across them; B0 went
from 14 to 2.6 ms an image on eight workers in the book's measurement. Elsewhere the same
forward runs on this thread. `threads()` says which, `POOL_ERROR` says why not.

## What it does not do

Train a backbone, export ONNX, run MobileNet or ViT (the device has no hardsigmoid and no
attention). Those are named as absent by `bimm.cpuGraphFor` and by this file, not
imitated. Everything measured about the device is in the book under "The `cpu` device".

## Registry resolution is written here a second time

`borch_webgpu._hub.entries` resolves a name the same way — the bundle's `models/index.json`
beside the page first, the public registry after. Importing it would boot the device, so
the twenty lines stand here as well; the rule is the same and the two say so.
"""

import importlib.util as _ilu
import pathlib as _pathlib

import numpy as _np

from borch._data import ImageFiles, decode_images, label_from_name, suspects  # noqa: F401 — the files, the decoder and the review score: the numpy core's, no device needed

try:
    import js as _js
    from pyodide.ffi import run_sync as _run_sync, to_js as _to_js
    from pyodide.code import run_js as _run_js
except ImportError as _e:  # pragma: no cover — this module only runs inside Pyodide
    raise ImportError("borch_cpu runs in a browser (Pyodide): it drives borch.ts's wasm kernels through JavaScript") from _e

REGISTRY = "https://models.pilab.kr/index.json"


def _bundle():
    """borch.ts with `cpu`, `hub` and `bimm` on it — the page's, or the wheel's own, **not initialised**.

    The page may have put its borch.ts on `globalThis.borch`; if it carries the three
    namespaces it is used as it is. Otherwise the bundle inside `borch_webgpu` is imported
    through a blob URL, exactly as `borch_webgpu._base._boot` does — minus the `init()`
    that needs an adapter. The location is found with `find_spec`, which does not import
    the package (importing it would boot).
    """
    existing = getattr(_js, "borch", None)
    if existing is not None and str(type(existing).__name__) not in ("JsNull", "jsnull"):
        if all(hasattr(existing, name) for name in ("cpu", "hub", "bimm")):
            return existing
    spec = _ilu.find_spec("borch_webgpu")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("borch_webgpu is not installed beside borch_cpu — the wheel carries both")
    source = (_pathlib.Path(next(iter(spec.submodule_search_locations))) / "_borch.js").read_text(encoding="utf-8")
    blob = _js.Blob.new(_to_js([source]), _to_js({"type": "text/javascript"}, dict_converter=_js.Object.fromEntries))
    url = _js.URL.createObjectURL(blob)
    try:
        module = _run_sync(_run_js(f"import({url!r})"))
    finally:
        _js.URL.revokeObjectURL(url)
    return module


_ts = _bundle()
_kernels = None
_pool = None
_threads = 0
#: Why no pool was spawned when one was asked for by default — `None` when there is a pool or none was wanted.
POOL_ERROR = None


def _obj(**kw):
    """A plain JS object, not a `Map` — the same reason as `borch_webgpu._base._js_options`."""
    return _to_js(kw, dict_converter=_js.Object.fromEntries)


def _f32(array):
    """A float32 numpy array as a JS `Float32Array` (a copy, contiguous)."""
    return _js.Float32Array.new(_to_js(_np.ascontiguousarray(array, dtype=_np.float32).ravel()))


def _back(typed):
    """A JS `Float32Array` as numpy — through `to_py()` so the bytes come as they are, and a copy."""
    return _np.frombuffer(typed.to_py(), dtype=_np.float32).copy()


def available():
    """Whether this engine runs the kernels — WebAssembly with SIMD. Every browser that ships WebGPU does."""
    return bool(_ts.cpu.available())


def kernels(threads=None):
    """The wasm kernel module, loaded once — one linear memory, so one handle.

    `threads` decides the first call: `None` spawns a worker pool of `cpu.defaultWorkers()`
    where the page is cross-origin isolated and runs on this thread elsewhere; a number asks
    for that many workers and raises where a pool cannot be had; `0` never spawns one. When
    the default spawn fails the module falls back to one thread and keeps the reason in
    `POOL_ERROR` — the forward is the same, only slower. Later calls return the handle.
    """
    global _kernels, _pool, _threads, POOL_ERROR
    if _kernels is None:
        want = threads
        if want is None:
            want = int(_ts.cpu.defaultWorkers()) if bool(_ts.cpu.threadsAvailable()) else 0
        if want and int(want) > 1:
            try:
                _pool = _run_sync(_ts.cpu.WorkerPool.spawn(int(want)))
                _kernels, _threads = _pool.kernels, int(want)
            except Exception as e:  # noqa: BLE001 — the reason is kept, and an explicit ask is not swallowed
                if threads is not None:
                    raise
                POOL_ERROR = f"{type(e).__name__}: {str(e)[:200]}"
        if _kernels is None:
            _kernels = _run_sync(_ts.cpu.loadKernels())
    return _kernels


def threads():
    """How many workers run the forward — 0 when it runs on this thread alone."""
    kernels()
    return _threads


def _fetch_json(url):
    response = _run_sync(_js.fetch(url))
    if not response.ok:
        raise RuntimeError(f"fetch failed {response.status}: {url}")
    return _run_sync(response.json()).to_py()


def _page_base():
    href = str(_js.location.href)
    if "/assets/" in href:
        return href.split("/assets/")[0] + "/"
    return href.rsplit("/", 1)[0] + "/"


def entries():
    """Every model the registry lists — the bundle's `models/index.json` beside the page first, the public registry after."""
    base = _page_base()
    tried = []
    for url in (base + "models/index.json", REGISTRY):
        try:
            index = _fetch_json(url)
        except Exception as e:  # noqa: BLE001 — both failing is reported with both reasons
            tried.append(f"{url} → {type(e).__name__}: {str(e)[:120]}")
            continue
        rows = (index.get("models") or index.get("entries") or []) if isinstance(index, dict) else index
        out = []
        for row in rows:
            row = dict(row)
            murl = row.get("manifestUrl", "")
            if murl and "://" not in murl:
                row["manifestUrl"] = base + murl
            out.append(row)
        return out
    raise RuntimeError("no model registry reachable — " + "; ".join(tried))


def _manifest_url(name_or_url):
    if "://" in name_or_url:
        return name_or_url
    rows = [r for r in entries() if r.get("name") == name_or_url]
    if not rows:
        names = sorted({r.get("name") for r in entries()})
        raise KeyError(f"{name_or_url!r} is not in the registry; there: {', '.join(names)}")
    return rows[-1]["manifestUrl"]


class Backbone:
    """A pretrained classifier's forward on the CPU device — features by default, logits if asked.

    Built from the manifest's `arch` (library, factory, numClasses) and the checkpoint's
    bytes, through `bimm.cpuGraphFor` and `cpu.CpuRunner`. The bytes come through
    borch-hub's `fetchWeights`, so the hash is checked and the Cache API is used, the same
    as on the WebGPU side.
    """

    def __init__(self, name_or_url, features=True, cache=True):
        url = _manifest_url(name_or_url)
        self.manifest = _fetch_json(url)
        arch = self.manifest.get("arch") or {}
        args = arch.get("args") or {}
        self.num_classes = int(args.get("numClasses") or args.get("num_classes") or 1000)
        manifest_js = _to_js(self.manifest, dict_converter=_js.Object.fromEntries)
        bytes_js = _run_sync(_ts.hub.fetchWeights(manifest_js, url, _obj(cache=bool(cache))))
        state = _ts.cpu.readSafetensors(bytes_js)
        self.name = f"{arch.get('library')}/{arch.get('factory')}"
        graph = _ts.bimm.cpuGraphFor(_obj(library=arch.get("library"), factory=arch.get("factory")), state,
                                     _obj(numClasses=self.num_classes, features=bool(features)))
        self.num_features = int(graph.outputChannels)
        self._runner = _ts.cpu.CpuRunner.new(kernels(), graph, _pool)
        self.input_size = tuple(int(v) for v in ((self.manifest.get("preprocess") or {}).get("inputSize") or (3, 224, 224)))

    def features(self, x):
        """`(N, C, H, W)` float32 NCHW → `(N, num_features)`. One forward, all `N` at once."""
        x = _np.ascontiguousarray(x, dtype=_np.float32)
        if x.ndim != 4:
            raise ValueError(f"expected (N, C, H, W), got {x.shape}")
        n, _c, h, w = x.shape
        out = self._runner.forward(_f32(x), int(n), int(h), int(w))
        return _back(out).reshape(n, self.num_features)

    __call__ = features


def load(name_or_url, features=True, cache=True):
    """A pretrained model's forward on the CPU device, by registry name or manifest URL."""
    return Backbone(name_or_url, features=features, cache=cache)


class LinearHead:
    """A linear layer trained with full-batch SGD on cached features — `cpu.LinearHead`, from Python.

    `fit` returns the mean cross-entropy before each step, the same numbers torch prints;
    `state_dict` hands the weight back in torch's `[classes × features]` order.
    """

    def __init__(self, in_features, num_classes, lr=0.05, momentum=0.0, weight_decay=0.0):
        self.in_features, self.num_classes = int(in_features), int(num_classes)
        self._head = _ts.cpu.LinearHead.new(kernels(), self.in_features, self.num_classes,
                                            _obj(lr=float(lr), momentum=float(momentum), weightDecay=float(weight_decay)))

    def fit(self, features, labels, steps):
        features = _np.ascontiguousarray(features, dtype=_np.float32)
        n = features.shape[0]
        labels = [int(v) for v in _np.asarray(labels).ravel()]
        if len(labels) != n:
            raise ValueError(f"{len(labels)} labels for {n} rows")
        return _back(self._head.fit(_f32(features), _to_js(labels), int(n), int(steps)))

    def predict(self, features):
        features = _np.ascontiguousarray(features, dtype=_np.float32)
        n = features.shape[0]
        return _back(self._head.predict(_f32(features), int(n))).reshape(n, self.num_classes)

    def state_dict(self):
        st = self._head.stateDict()
        return {"weight": _back(st.weight).reshape(self.num_classes, self.in_features), "bias": _back(st.bias)}

    __call__ = predict
