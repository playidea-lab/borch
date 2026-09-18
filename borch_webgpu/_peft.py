"""`borch.peft` on the WebGPU runtime — LoRA, mirroring `borch-ts`'s `peft`.

torch.nn has no LoRA, so this is borch's own namespace (see `borch-ts/src/peft.ts`). The
individual layers bridge to `_ts.peft` — `LoRALinear`/`LoRAConv2d` are classes over there. But
`apply_lora` cannot: the WebGPU binding composes models **Python-side** (`nn.Sequential` here is
a Python class, not a `_ts` module), so a model built in Python has no `_ts` tree for the
TypeScript `applyLora` to walk. So the walk is Python — over `named_modules()` — and each matched
leaf is swapped for its LoRA wrapper, whose base is bridged from the leaf's `_ts` layer.
"""
import js as _js

from ._base import _js_options, _to_js
from ._nn import Module

_ts = _js.borch


def _opts(r, alpha, **extra):
    kw = dict(extra)
    if r is not None:
        kw["r"] = int(r)
    if alpha is not None:
        kw["alpha"] = int(alpha)
    return _js_options(**kw)


def LoRALinear(in_features, out_features, r=8, alpha=None, bias=True):
    """A `Linear` with a frozen base and a trainable low-rank adapter. `parameters()` returns
    only the adapter; the base rides along in `state_dict()` as a buffer."""
    return Module(_ts.peft.LoRALinear.new(int(in_features), int(out_features), _opts(r, alpha, bias=bool(bias))))


def LoRAConv2d(in_channels, out_channels, kernel_size, r=8, alpha=None, bias=True,
               stride=1, padding=0, dilation=1, groups=1):
    """A 2-D convolution with a frozen base kernel and a trainable low-rank adapter."""
    return Module(_ts.peft.LoRAConv2d.new(int(in_channels), int(out_channels), int(kernel_size),
        _opts(r, alpha, bias=bool(bias), stride=int(stride), padding=int(padding),
              dilation=int(dilation), groups=int(groups))))


def _kind(module):
    """`"Linear"`, `"Conv2d"`, or the leading name of `describe()` — the class name over there.
    Every wrapped leaf is the one Python class `Module`, so the type is read from `describe`
    (a frozen, tested string) rather than `type(module)`."""
    describe = getattr(module, "describe", None)
    if describe is None:
        return None
    text = describe()
    return text.split("(", 1)[0] if text else None


def _adapter_for(module, r, alpha):
    """The LoRA wrapper for a matched leaf, its base bridged from the leaf's `_ts` layer."""
    kind = _kind(module)
    base = getattr(module, "_m", None)
    if base is None:
        return None
    if kind == "Linear":
        return Module(_ts.peft.LoRALinear.fromLinear(base, _opts(r, alpha)))
    if kind == "Conv2d":
        return Module(_ts.peft.LoRAConv2d.fromConv2d(base, _opts(r, alpha)))
    return None


def _make_match(targets):
    """A predicate on the dotted name. `None` matches every adaptable leaf; a list of names
    matches when the name equals a target, ends with `.<target>`, or has it as a segment."""
    if targets is None:
        return lambda name: True
    names = list(targets)

    def match(name):
        return any(name == t or name.endswith(f".{t}") or t in name.split(".") for t in names)

    return match


def _replace_submodule(model, dotted, adapter):
    """Swap the submodule at `dotted`, then read it back to prove the swap took — a leaf held in
    a Python `Sequential`'s list (not as a named attribute) is not reached by `setattr`, and
    without the read-back the swap would silently do nothing."""
    cut = dotted.rfind(".")
    parent = model.get_submodule(dotted[:cut] if cut >= 0 else "")
    leaf = dotted[cut + 1:] if cut >= 0 else dotted
    parent.add_module(leaf, adapter)
    if model.get_submodule(dotted) is not adapter:
        raise RuntimeError(
            f"apply_lora could not replace `{dotted}`: it is held in a container's list, not as "
            f"a named attribute, so setattr does not reach it. Target a module held as a field.")


def apply_lora(model, targets=None, r=8, alpha=None):
    """Adapt a model in place: swap every matched `Linear`/`Conv2d` for its LoRA wrapper, so
    afterwards `model.parameters()` is the adapters alone. `targets` is a list of names (a
    predicate is not bridged here); the default is every adaptable leaf. Returns the dotted
    names swapped. The adapters start at zero, so the forward is unchanged until they train, and
    the call is idempotent — a `LoRALinear` is not a `Linear`."""
    # A **wrapped** borch.ts module (a hub-loaded backbone) keeps its whole tree on the TS side,
    # invisible to a Python walk (`_children` reads Python attributes only). The TS `applyLora`
    # walks that tree, so delegate to it and the adapters live where `forward_features` reads them.
    base = getattr(model, "_m", None)
    if base is not None:
        opts = {}
        if r is not None:
            opts["r"] = int(r)
        if alpha is not None:
            opts["alpha"] = int(alpha)
        if targets is not None:
            opts["targets"] = _to_js(list(targets))
        return [str(s) for s in _ts.peft.applyLora(base, _js_options(**opts))]
    match = _make_match(targets)
    hits = []
    for name, module in model.named_modules():
        if not name:
            continue
        if _kind(module) in ("Linear", "Conv2d") and match(name):
            hits.append((name, module))
    swapped = []
    for name, module in hits:
        adapter = _adapter_for(module, r, alpha)
        if adapter is None:
            continue
        _replace_submodule(model, name, adapter)
        swapped.append(name)
    return swapped
