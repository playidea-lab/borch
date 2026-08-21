"""`torch.save` and `torch.load` — the **safetensors** format.

## Why this is needed

All three have `state_dict()`, and **the Python side had no way of writing it to
a file.** Almost every tutorial ends in saving, and a browser is where a refresh
loses the training, so the gap was felt more here than on a desktop.

## Why not pickle

torch's `save`/`load` is pickle. It is a format that unpacks Python objects **by
executing them**, so it cannot be carried into a browser and should not be.
Somebody else's `.pt` is therefore unreadable here — this reads back what this
library wrote.

The format was already settled by `borch-ts/src/serialize.ts` and **the same one**
is used here. The three exchanging the same file is the reason for choosing this
format, so one of them inventing its own erases that value entirely.

    [8-byte LE u64: header length N][N-byte JSON header][body: tensor bytes, concatenated]

## The body is always float32

borch.ts's buffers are float32 and nothing else, so that side has no choice.
Writing a real `I64` here means **that side cannot read our files** — and then
matching the format was pointless. So F32 is written here too and borch's dtype
label rides separately in `__metadata__`.

In exchange, **an integer that does not fit exactly in f32 is refused.** A
quietly rounded checkpoint surfaces hours later, and by then there is no way to
tell what went wrong.

## Nesting is recorded as a tree

`torch.save({"model": sd, "opt": sd, "epoch": 3}, path)` is the textbook idiom.
safetensors is a **flat** dict of tensors, so the nesting has to be flattened,
and the flattened names must not be split back on the dots — `state_dict`'s keys
already contain dots (`fc.weight`), so `{"model": {"fc.weight": t}}` returns as
`{"model": {"fc": {"weight": t}}}`. So the structure is recorded separately in
`borch.tree`, and the names inside the file stay flattened so they still mean
something to somebody else reading it.
"""

import json as _json
import pathlib as _pathlib

import numpy as _np

from ._base import BorchError
from ._tensor import Tensor

# The place safetensors fixed. The header length goes here.
_LENGTH_FIELD = 8
# The reference implementation aligns the header to this multiple. Off by any
# amount and numpy cannot view the body directly.
_ALIGN = 8
_BYTES_PER_F32 = 4
# The prefix of the keys carrying borch's dtype labels. float32 is not written —
# it is the default.
_DTYPE_KEY = "borch.dtype:"
# The key carrying the nested structure. Without it the file reads as a flat
# dict of tensors.
_TREE_KEY = "borch.tree"
# The range over which f32 holds every integer. Past 2^24 neighbouring integers
# collapse onto the same value.
_EXACT_INT = 2 ** 24


def _labelled_dtype(array):
    """Which borch dtype label to record for this array. float32 records
    none."""
    if array.dtype == _np.bool_:
        return "bool"
    if _np.issubdtype(array.dtype, _np.integer):
        return "int64"
    return None


def _as_f32(name, array):
    """The float32 array to carry in the body. **An integer that does not fit is
    refused.**"""
    if _np.issubdtype(array.dtype, _np.complexfloating):
        raise BorchError(
            f"'{name}' is complex — saving that is not supported yet.\n"
            "Store the real pair with `view_as_real()` and restore it with "
            "`view_as_complex()` on load.")
    if _np.issubdtype(array.dtype, _np.integer) and array.size:
        if int(_np.abs(array).max()) > _EXACT_INT:
            raise BorchError(
                f"'{name}' holds an integer too large — the body of this format is float32.\n"
                f"Integers past {_EXACT_INT} become a neighbouring value once stored. "
                "A quietly rounded checkpoint leaves no way to find the cause later, "
                "so it stops here.")
    return _np.ascontiguousarray(array, dtype=_np.float32)


def encode(tensors, metadata=None):
    """`{name: array}` to bytes. **The name order is fixed** — saving twice gives
    the same bytes."""
    meta = dict(metadata or {})
    header = {}
    bodies = []
    offset = 0
    for name in sorted(tensors):
        array = _np.asarray(tensors[name])
        values = _as_f32(name, array)
        nbytes = values.size * _BYTES_PER_F32
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        label = _labelled_dtype(array)
        if label is not None:
            meta[_DTYPE_KEY + name] = label
        bodies.append(values)
        offset += nbytes

    if meta:
        header["__metadata__"] = meta
    text = _json.dumps(header).encode("utf-8")
    padding = (_ALIGN - ((_LENGTH_FIELD + len(text)) % _ALIGN)) % _ALIGN
    # The remaining room is spaces — a JSON parser walks past trailing
    # whitespace.
    head = text + b" " * padding
    out = bytearray()
    out += len(head).to_bytes(_LENGTH_FIELD, "little")
    out += head
    for values in bodies:
        out += values.tobytes()
    return bytes(out)


def decode(blob):
    """Bytes to `({name: array}, metadata)`. **A corrupt file does not pass
    quietly.**"""
    if len(blob) < _LENGTH_FIELD:
        raise BorchError(f"the checkpoint is too short: {len(blob)} bytes")
    head_len = int.from_bytes(blob[:_LENGTH_FIELD], "little")
    body_at = _LENGTH_FIELD + head_len
    if body_at > len(blob):
        raise BorchError(
            f"the header length runs past the file: {head_len} (file {len(blob)})")
    try:
        header = _json.loads(blob[_LENGTH_FIELD:body_at].decode("utf-8"))
    except Exception as exc:                                    # noqa: BLE001
        raise BorchError("the checkpoint header is not JSON") from exc

    metadata = header.get("__metadata__") or {}
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        if begin > end or body_at + end > len(blob):
            raise BorchError(
                f"'{name}' points past the end of the file: [{begin}, {end}]")
        shape = tuple(entry["shape"])
        flat = _np.frombuffer(blob, dtype=_np.float32,
                              count=(end - begin) // _BYTES_PER_F32,
                              offset=body_at + begin)
        want = 1
        for dim in shape:
            want *= dim
        if flat.size != want:
            raise BorchError(
                f"'{name}' body does not match its shape: {flat.size} vs {want}")
        array = flat.reshape(shape).copy()
        label = metadata.get(_DTYPE_KEY + name)
        if label == "int64":
            array = array.astype(_np.int64)
        elif label == "bool":
            array = array.astype(_np.bool_)
        tensors[name] = array
    return tensors, metadata


# ── nesting ─────────────────────────────────────────────────────────────────

def _flatten(obj, path, tensors, seen, to_array):
    """Split the structure into a tree and the tensors into a flat dict.

    **Only recognising a tensor and converting it to an array come from
    outside.** The binding's `Tensor` is a different class from this one and its
    values live on the GPU, so those two differ. The rest — the flattening rules,
    the refusal on name collisions, the kinds that can be carried — **has to be
    the same.** Kept as two copies, one of them gets fixed, and then one side
    cannot read the other's files.
    """
    if to_array(obj) is not None:
        name = ".".join(path) or "tensor"
        if name in seen:
            # Two different places flattened to the same name. One overwriting
            # the other means both hold the same value when restored, and that
            # is worse than an exception.
            raise BorchError(
                f"'{name}' appears twice — the flattened names collide and cannot be stored.")
        seen.add(name)
        tensors[name] = to_array(obj)
        return {"t": "T", "v": name}
    if isinstance(obj, dict):
        return {"t": "d",
                "v": {str(k): _flatten(v, [*path, str(k)], tensors, seen, to_array)
                      for k, v in obj.items()}}
    if isinstance(obj, (list, tuple)):
        kind = "u" if isinstance(obj, tuple) else "l"
        return {"t": kind,
                "v": [_flatten(v, [*path, str(i)], tensors, seen, to_array)
                      for i, v in enumerate(obj)]}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"t": "j", "v": obj}
    raise BorchError(
        f"{type(obj).__name__} cannot be stored — only tensors, dicts, lists, numbers "
        "and strings.\n"
        "This format is not pickle, so it cannot hold arbitrary Python objects.")


def _unflatten(node, tensors, make):
    kind = node["t"]
    if kind == "T":
        return make(tensors[node["v"]])
    if kind == "d":
        return {k: _unflatten(v, tensors, make) for k, v in node["v"].items()}
    if kind == "l":
        return [_unflatten(v, tensors, make) for v in node["v"]]
    if kind == "u":
        return tuple(_unflatten(v, tensors, make) for v in node["v"])
    return node["v"]


def _open_bytes(where):
    """Takes a path or a file object — torch takes both."""
    if hasattr(where, "write") or hasattr(where, "read"):
        return None, where
    return _pathlib.Path(where), None


def dump(obj, where, to_array):
    """Flatten the structure and write the bytes. **The binding calls this
    too** — there is one codec."""
    tensors = {}
    tree = _flatten(obj, [], tensors, set(), to_array)
    blob = encode(tensors, {_TREE_KEY: _json.dumps(tree)})
    path, handle = _open_bytes(where)
    if handle is not None:
        handle.write(blob)
        return
    path.write_bytes(blob)


def parse(where, make, **kw):
    """Read the bytes and rebuild the structure. **The binding calls this too.**

    `weights_only` is accepted and ignored — this format executes no code, so it
    is **always** that way. torch code passes the argument often enough that it
    is accepted.
    """
    kw.pop("weights_only", None)
    kw.pop("map_location", None)
    if kw:
        raise BorchError(f"load got an unexpected argument: {', '.join(sorted(kw))}")
    path, handle = _open_bytes(where)
    blob = handle.read() if handle is not None else path.read_bytes()
    tensors, metadata = decode(blob)
    tree = metadata.get(_TREE_KEY)
    if tree is None:
        # Somebody else's safetensors. Handed back as a flat dict of tensors.
        return {name: make(a) for name, a in tensors.items()}
    return _unflatten(_json.loads(tree), tensors, make)


def _array_of(obj):
    """This core's tensor gives its array, anything else gives `None`. The
    measure `_flatten` uses to tell a tensor apart."""
    return obj.data if isinstance(obj, Tensor) else None


def save(obj, where):
    """Write a checkpoint. `obj` is a tensor, or a dict/list holding tensors,
    numbers and strings."""
    dump(obj, where, _array_of)


def load(where, **kw):
    """Read back what `save` wrote."""
    return parse(where, Tensor, **kw)
