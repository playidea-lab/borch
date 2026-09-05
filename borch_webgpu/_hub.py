"""`torch.hub` — the catalogue's pretrained models, as `nn.Module`s in Python.

    model = torch.hub.load("imagenet-efficientnet-b0")      # by name, through the registry
    model = torch.hub.load("https://…/manifest/")            # or by manifest URL
    with torch.no_grad():
        logits = model(x)                                    # (N, 1000)

The loading is borch-hub's (JavaScript): fetch, hash check, a sample the model has to
reproduce on this device, the Cache API so the 21 MB is fetched once. What is here is only
the seat: the wheel's bundle carries bimm-ts and borch-hub next to borch.ts (`_entry.js`),
so `js.borch.hub` exists wherever the wheel booted borch.ts itself. A page that put its own
borch.ts on `globalThis.borch` has to put `borch.hub` there too, or this says so.

`torch.hub` in real torch takes GitHub repositories; the shape kept here is the name —
`load` returns a model — and the source is the catalogue, which is the only place a
browser can get weights it can verify.
"""
import js as _js
from pyodide.ffi import run_sync as _run_sync, to_js as _to_js

from ._nn import Module
from ._ops import _ts

# The public catalogue. A constant, not configuration: it is the one address the models
# page and the envelope bench already read, and a model loaded by name is resolved here.
REGISTRY = "https://models.pilab.kr/index.json"


def _hub():
    hub = getattr(_ts, "hub", None)
    if hub is None or str(type(hub).__name__) in ("JsNull", "jsnull"):
        raise RuntimeError(
            "this borch.ts carries no hub — the wheel's own bundle does; a page that "
            "placed its own borch.ts on `globalThis.borch` has to expose `borch.hub` too")
    return hub


def _fetch_json(url):
    response = _run_sync(_js.fetch(url))
    if not response.ok:
        raise RuntimeError(f"fetch failed {response.status}: {url}")
    return _run_sync(response.json()).to_py()


def _page_base():
    """The page's folder as an absolute URL. The kernel may run in a worker whose
    `location` sits under `<page>/assets/`; the page is one level up from there."""
    href = str(_js.location.href)
    if "/assets/" in href:
        return href.split("/assets/")[0] + "/"
    return href.rsplit("/", 1)[0] + "/"


def entries():
    """Every model the registry lists — name, version, task, bytes, manifestUrl.

    A `models/index.json` beside the page is asked first — that is the offline
    bundle's catalogue, its manifest paths relative to the page — and the public
    catalogue after it. Relative `manifestUrl`s are made absolute here so `load`
    can hand them to borch-hub as they are.
    """
    base = _page_base()
    for url in (base + "models/index.json", REGISTRY):
        try:
            index = _fetch_json(url)
        except Exception:                                                # noqa: BLE001
            continue
        # `list` is torch's name for this function below, so the builtin is not asked here.
        rows = (index.get("models") or index.get("entries") or []) if isinstance(index, dict) else index
        out = []
        for row in rows:
            row = dict(row)
            murl = row.get("manifestUrl", "")
            if murl and "://" not in murl:
                row["manifestUrl"] = base + murl
            out.append(row)
        return out
    raise RuntimeError(f"no model registry reachable — neither {base}models/index.json nor {REGISTRY}")


def load(name_or_url, cache=True, verify=True, timeout_ms=None):
    """A pretrained model as an `nn.Module`, in eval mode, with `.manifest` attached.

    `name_or_url` is a registry name (`"imagenet-efficientnet-b0"`, the newest version)
    or a manifest URL. `verify` keeps borch-hub's check that the model reproduces its
    sample on this device — off only when measuring the load itself.
    """
    if "://" in name_or_url:
        url = name_or_url
    else:
        rows = [r for r in entries() if r.get("name") == name_or_url]
        if not rows:
            names = sorted({r.get("name") for r in entries()})
            raise KeyError(f"{name_or_url!r} is not in the registry; there: {', '.join(names)}")
        url = rows[-1]["manifestUrl"]
    opts = {"cache": bool(cache), "verify": bool(verify)}
    if timeout_ms is not None:
        opts["timeoutMs"] = int(timeout_ms)
    loaded = _run_sync(_hub().load(url, _to_js(opts, dict_converter=_js.Object.fromEntries)))
    model = Module(loaded.model)
    model.eval()
    manifest = loaded.manifest
    object.__setattr__(model, "manifest", manifest.to_py() if hasattr(manifest, "to_py") else manifest)
    return model


# torch's spelling. The builtin is not used after this line.
list = entries   # noqa: A001
