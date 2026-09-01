"""**A parameter the body never reads is a seat that accepts and does nothing.**

The call reads exactly like torch's, the request reaches nowhere, and no exception is
raised. It is the quietest way to be wrong that this repository has found, and it was
found late: one sweep of the three Python surfaces turned up **142** such seats, and
measuring each against torch — call it twice with the argument changed, and see whether
torch's answer moves — showed twelve of them were real.

    F.layer_norm(normalized_shape)   folded the last axis whatever it was told
    istft(return_complex)            handed back complex where torch gives a waveform
    zero_grad(set_to_none)           always set None, so `False` did the other thing
    nn.RMSNorm(eps)                  the layer took it and never handed it on
    isclose(equal_nan)               NaN against NaN came back False
    quantile(dim)                    the flattened scalar, not one number per row
    rand/randn(dtype)                float32 under whatever name was asked for
    SequentialLR(last_epoch)         resuming put the chain back at its first interval
    convert_bounding_box_format(inplace)  the caller's boxes kept their old corners
    multinomial(replacement)         `False` still drew with replacement
    layer_norm(weight)/(bias)        the affine silently missing from a block

Every one of them had a case that passed. `layer_norm` had eleven.

## Why this is a check and not a habit

The sweep was run by hand and the twelve were fixed. **A sweep run by hand is a thing
somebody has to remember**, which is the problem it exists to solve one level up — the
same argument `tests/test_binding_fills_in.py` makes about its own table. So the sweep
runs here, and a seat that is new blows up naming itself.

## What is detected and what is attested

Two shapes are recognised rather than listed, because listing them would be listing a
rule:

  - **A body that refuses outright.** `empty_strided` raises on its first real line;
    every parameter is unread because there is nowhere to read one.
  - **A Python protocol seat.** `__exit__(exc)` and `__get__(owner)` are positions the
    interpreter fills, and a context manager that restores state does not consult them.

Everything else is named below **with why it is right**, and a row that stops appearing
fails too — a reason for something that is no longer there is the defect this
repository spends its time removing.

## What this cannot do

It reads the source, so it finds a parameter the body never *mentions*. It does not
find one that is mentioned and then has no effect — `borch/_ops.py`'s `topk(sorted)`
would still be caught, but a version that wrote `if sorted: pass` would not. That half
stays a judgement, and it is the same boundary `test_inert_arguments.py` draws from the
other side: that one reads the frozen answers and finds arguments that provably changed
nothing, and between the two the gap is an argument that is read, has an effect, and
the effect is wrong.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SURFACES = ("borch", "borch_webgpu")

# Calls that refuse. A body reaching one of these unconditionally has no room to read
# anything, so its parameters are not counted.
REFUSERS = {"_unsupported", "_absent_dtype", "_unknown_tensor_type"}

# Positions the interpreter fills. `__exit__` gets the exception triple whether or not
# there was one, and a context manager that restores state does not look.
PROTOCOL_SEATS = {
    ("__exit__", "exc"), ("__exit__", "exc_type"), ("__exit__", "tb"),
    ("__get__", "owner"), ("__set_name__", "owner"),
}

# ── the reasons, written once and shared ────────────────────────────────────
_RETURN_INDICES = (
    "the function's whole job is to hand the positions back — its name says so, and "
    "torch's `return_indices=False` reaches the plain `max_pool*d` instead. Measured: "
    "torch gives the pair either way at this name"
)
_PREDICATE = (
    "a predicate with one answer here. torch reads the argument and answers about it; "
    "there is one storage, one device and no inference mode, so the answer does not "
    "depend on which tensor is asked. Measured against torch on a dense CPU tensor"
)
_HERMITIAN = (
    "`hermitian` and `check_errors` change nothing for a real symmetric input, which "
    "is the only kind reachable here — measured: torch's answer is identical with and "
    "without. `pivot=False` is refused by torch too"
)
_NO_ITERATION = (
    "an iterative solver's knobs, and this is exact. torch iterates so that a few "
    "eigenpairs come cheaply out of a large sparse matrix; there is no sparse layout "
    "here and the sizes are small, so the answer is computed outright — measured to "
    "within 7e-6 of torch's, below this repository's tolerance"
)
_NOT_ASYNC = (
    "it asks torch not to wait on an asynchronous device copy, and nothing here is "
    "asynchronous — the copy has finished by the time the call returns. Measured: "
    "torch's answer is the same with and without"
)
_INPLACE_IDENTITY = (
    "`inplace` changes the identity and not the value — the numbers are the ones it "
    "would have computed anyway, which is what the flag is for. Measured against "
    "torch, whose values also match"
)
_SHAPE_ONLY = (
    "the seat exists to hold torch's position, so a positional call lands where torch "
    "lands rather than one argument earlier. Nothing is behind it to read"
)
_NO_PROCESSES = (
    "it starts worker processes, and a browser tab has none — the whole point of this "
    "subset. torch's own answer is the same batches in the same order, which is what "
    "the single process produces"
)
_V2_DETERMINISTIC = (
    "the v2 transform protocol: `make_params` draws once per sample and `transform` "
    "receives what it drew. These transforms are deterministic, so the base's "
    "`make_params` returns `{}` and there is nothing in it — measured: torchvision's "
    "own bodies do not read it either"
)

ATTESTED = {
    # ── the pair-returning pools ──
    **{f"{name}(return_indices)": _RETURN_INDICES for name in (
        "max_pool1d_with_indices", "max_pool2d_with_indices",
        "max_pool3d_with_indices", "adaptive_max_pool1d_with_indices",
        "adaptive_max_pool2d_with_indices", "adaptive_max_pool3d_with_indices",
        "fractional_max_pool2d_with_indices", "fractional_max_pool3d_with_indices",
        "_pool_with_indices")},
    # ── predicates with one answer ──
    **{key: _PREDICATE for key in (
        "is_inference(input)", "is_inference(t)", "is_storage(x)",
        "is_distributed(x)", "is_conj(input)", "is_conj(t)", "is_neg(t)")},
    # ── decompositions whose flags do not move a real symmetric answer ──
    **{key: _HERMITIAN for key in (
        "lu_factor_ex(check_errors)", "lu_factor_ex(pivot)",
        "ldl_factor(hermitian)", "ldl_factor_ex(hermitian)",
        "ldl_factor_ex(check_errors)", "ldl_solve(hermitian)")},
    # ── the iterative solver's knobs ──
    **{f"lobpcg({name})": _NO_ITERATION for name in (
        "n", "iK", "niter", "tol", "method", "tracker",
        "ortho_iparams", "ortho_fparams", "ortho_bparams")},
    "svd_lowrank(niter)":
        "the same as `lobpcg`'s above — torch iterates towards the decomposition and "
        "this one computes it. Measured: the singular values agree at `niter=0` and "
        "`niter=8` alike",
    # ── nothing here is asynchronous ──
    **{key: _NOT_ASYNC for key in (
        "type(non_blocking)", "copy_(non_blocking)")},
    "numpy(force)":
        "it tells torch to copy off a device or out of a graph rather than refuse. "
        "There is one device and `numpy()` already detaches, so there is nothing the "
        "flag would unblock — measured: torch's answer is the same either way",
    # ── identity, not value ──
    **{key: _INPLACE_IDENTITY for key in ("rrelu(inplace)", "_elu(inplace)")},
    # ── seats held for torch's positions ──
    **{key: _SHAPE_ONLY for key in (
        "__init__(args)", "__init__(kw)", "Identity(args)", "Identity(kw)",
        "fftfreq(kw)", "rfftfreq(kw)", "__call__(rest)", "call(rest)")},
    "__init__(device)":
        "`Generator(device=…)` — torch takes it and there is one device. The stream is "
        "numpy's either way, and `_only_cpu` guards the seats where a device changes "
        "where a tensor lands",
    # ── workers ──
    **{key: _NO_PROCESSES for key in ("__init__(num_workers)",)},
    # ── torchvision's v2 protocol ──
    **{key: _V2_DETERMINISTIC for key in (
        "make_params(flat_inputs)", "transform(params)")},
    # ── the rest, one at a time ──
    "topk(sorted)":
        "**torch does not promise an order when it is false** — the documentation says "
        "the result is not necessarily sorted, and measured it sometimes is and "
        "sometimes is not. Always sorting is inside that promise; matching torch's "
        "particular unsorted order would be freezing an implementation detail",
    "unique(sorted)": "as `topk(sorted)` above — measured, torch returns the same "
                      "sorted answer either way on this input",
    "repeat_interleave(output_size)":
        "a hint that lets torch size the output without reading the repeat counts off "
        "the device. The counts are already here, so it is a cost and not an answer — "
        "measured: torch's values are identical with and without",
    "manual_seed_all(seed)":
        "it seeds every CUDA device, and there are none. torch's is a no-op on a "
        "machine without CUDA too (measured)",
    "batch_norm_aten(cudnn_enabled)": "which kernel torch picks. There is no cuDNN and "
                                      "the answer does not depend on it (measured)",
    "batch_norm(cudnn_enabled)": "as `batch_norm_aten(cudnn_enabled)` above",
    "TransformerEncoder(enable_nested_tensor)":
        "a fast path torch takes when the padding mask allows it. It changes how long "
        "the answer takes and not what it is — torch documents that, and it is why the "
        "flag can default differently between versions without moving a value",
    "TransformerEncoder(mask_check)":
        "torch validates the mask's shape when it is on. Ours validates always, so "
        "turning it off would be offering to skip a check we do not skip",
    "named_buffers(persistent_only)":
        "these two classes hold no buffers at all and return `[]`. The argument selects "
        "among buffers, and selecting among none is the same list",
    "train(mode)":
        "these two classes wrap something with no training-dependent behaviour — no "
        "dropout, no running statistics — so both modes compute the same thing. The "
        "classes that do have one read it",
    "handle_starttag(attrs)":
        "`HTMLParser`'s protocol hands over the tag's attributes; this parser only "
        "counts tags by name",
    # ── private helpers, where the seat is about the caller and not the answer ──
    "_window_of(win_length)":
        "the window is fitted to `n_fft` and not to `win_length` — measured against "
        "torch, and a comment at the call site records the day fitting to `win_length` "
        "stopped on a shape",
    "_window_of(like)": "it is there so a caller can pass the tensor whose dtype the "
                        "window should take. There is one float dtype",
    "back(m)": "the backward closure's shape is fixed by the forward it belongs to; "
               "this one's gradient does not depend on the mask it is handed",
    "_step(w_hr)":
        "the GRU and LSTM cells share this signature, and `w_hr` is the projection "
        "weight only an LSTM with `proj_size` has. The GRU's step is handed one and "
        "has nowhere to apply it",
    "_norm_flat(groups)": "the caller has already laid the groups onto one axis, so "
                          "the count is spent by the time this is reached",
    "_weighted_reduce(where_)":
        "torch's `where=` on a reduction selects which elements take part. The callers "
        "that reach this helper never pass one; the seat keeps the signature aligned "
        "with the reductions that do",
    "_feature_dropout(name)": "it names the caller for a refusal that no longer fires "
                              "on this path",
    "_max_pool_layer(wide)": "it marks the one spelling whose signature torch widened. "
                             "The layers are built from a table and this row's flag is "
                             "read where the table is written, not here",
    "__init__(custom_encoder)": "torch lets a caller substitute the whole encoder "
                                "stack. Ours builds its own — the seat is carried so "
                                "the positions after it land where torch puts them",
    "__init__(custom_decoder)": "as `__init__(custom_encoder)` above",
}


class _Reads(ast.NodeVisitor):
    """Every name the body mentions, nested functions included — a parameter closed
    over by an inner function is read."""

    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.id)


def _params(fn):
    a = fn.args
    out = []
    for group in (a.posonlyargs, a.args, a.kwonlyargs):
        out += [p.arg for p in group]
    if a.vararg:
        out.append(a.vararg.arg)
    if a.kwarg:
        out.append(a.kwarg.arg)
    return out


def _refuses_outright(fn):
    """The body raises before it branches. Guard calls and imports do not count as
    branching — `empty_strided` calls `_no_out(out)` and then raises."""
    for stmt in fn.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue                                        # the docstring
        if isinstance(stmt, ast.Raise):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            called = stmt.value.func
            if isinstance(called, ast.Name) and called.id in REFUSERS:
                return True
            continue
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        return False
    return False


def _stub(fn):
    """`...`, `pass`, or a bare raise — a declaration rather than a body."""
    body = [s for s in fn.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if not body:
        return True
    return len(body) == 1 and isinstance(body[0], (ast.Raise, ast.Pass))


def _files():
    found = [ROOT / "borchvision.py"]
    for pkg in SURFACES:
        found += sorted((ROOT / pkg).glob("*.py"))
    return [p for p in found if p.exists()]


def unread_seats():
    """`{"function(param)": [where, …]}` for every seat the body never mentions."""
    found = {}
    for path in _files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _stub(node) or _refuses_outright(node):
                continue
            reader = _Reads()
            for stmt in node.body:
                reader.visit(stmt)
            for name in _params(node):
                if name in ("self", "cls") or name.startswith("_"):
                    continue
                if name in reader.names or (node.name, name) in PROTOCOL_SEATS:
                    continue
                key = f"{node.name}({name})"
                where = f"{path.relative_to(ROOT)}:{node.lineno}"
                found.setdefault(key, []).append(where)
    return found


def test_no_seat_accepts_and_does_nothing_without_a_reason():
    """**A new unread parameter blows up naming itself.**

    Two ways out. If torch's answer moves when the argument changes, the seat is a
    defect and reading it is the fix — that is what the twelve at the top of this file
    were. If torch's answer does not move, write it into `ATTESTED` **with what was
    measured.** Raising a count is the same as switching this check off.
    """
    seats = unread_seats()
    surprise = sorted(set(seats) - set(ATTESTED))
    assert not surprise, (
        f"{len(surprise)} parameters are accepted and never read, with no reason "
        "written:\n  " + "\n  ".join(
            f"{key}  ({', '.join(seats[key])})" for key in surprise)
        + "\n\n  Call torch twice with the argument changed. If its answer moves, the "
          "seat is a defect —\n  the twelve at the top of this file all were. If it "
          "does not, put the name in `ATTESTED`\n  with what you measured.")


def test_no_attested_row_outlives_its_seat():
    """**A row whose seat is gone has to go too.**

    Left in, the next reader takes it for a live decision, and this repository has
    spent a great deal of time removing reasons for things that stopped being true.
    A seat leaves this list two ways: the parameter started being read, or it went
    away — and both mean the row is finished.
    """
    gone = sorted(set(ATTESTED) - set(unread_seats()))
    assert not gone, (
        "these seats are read now, or gone — delete their rows from `ATTESTED`:\n  "
        + "\n  ".join(gone))
