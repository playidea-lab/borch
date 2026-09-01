"""Counts the names the **core** has that **borch.ts** does not, and the reverse.

    uv run --with numpy --with torch --with torchvision python tests/ts_axis.py
    uv run --with numpy --with torch --with torchvision python tests/ts_axis.py --show nn

## The axis nothing was looking along

Three files already measure this repository's surface, and none of them measures this
pair:

    tests/torch_gap.py           core Python  ↔ real torch          names
    tests/test_torch_signatures  borchvision  ↔ real torchvision    parameters, in order
    tests/test_binding_arguments borch_webgpu ↔ borch.ts            parameters, in order

The core and borch.ts are **two independent implementations of the same surface** —
one over numpy, one over WGSL — and the golden holds their *values* against each
other. What no check holds is their *names*. A name the core has and borch.ts does
not is not a wrong answer anywhere; it is a line of tutorial code that runs in one
and raises in the other, and nothing goes red.

That state was real. Measured by hand in one session, seventeen `nn` names stood in
the core and not in borch.ts, and every golden case was green throughout — because a
case can only ask about a name somebody wrote a case for.

## What this counts, and the two things it cannot

It counts **names**, from the same enumeration `torch_gap.py` uses on the Python side
and from the generated index on the TypeScript side.

It cannot see a **signature**. `MaxPool2d` present in both, taking `(kernel)` here and
`(kernel, stride, padding, dilation, returnIndices)` there, counts as agreement — the
exact defect `test_torch_signatures.py` was written for, on the axis it does not
cover. Five of those were found in one day and none was visible to a count.

It cannot see a **value**. That is the golden's job, and the golden is why the two
implementations agree at all.

So a green run of this file says *the same names exist on both sides*. It does not say
they mean the same thing. Reading it as coverage is reading it as a sentence it does
not support.

## Why the TypeScript side is read from the index rather than the source

`site/assets/api-index.json` is generated from the `.d.ts` files, which is what a
consumer of the package actually gets. Reading `src/*.ts` instead would count names
that never reach a user, and reading the case table would count only what has a case.

**That makes this measurement a build artefact, and a stale one lies in our favour** —
names added since the last build read as absent from borch.ts, which is the same
sentence as a real gap. `test_ts_axis.py` refuses to run against a stale bundle for
that reason, the same rule `test_site.py` applies to its counts.

## Filling this in is the same dangerous work `torch_gap.py` describes

Every name written into `DELIBERATE` below raises the agreement figure, so the work
slides towards making the number look good. The rule is the one that file uses:
**every row carries a reason, and a row whose reason cannot be written is a gap.**
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

INDEX = ROOT / "site" / "assets" / "api-index.json"

# Which Python namespaces have a borch.ts side to compare against.
#
# **`torch` — the top level — is not on this list, and that is a finding rather than
# an omission.** The first version measured it and reported 202 core-only names.
# Reading them showed the question does not have an answer: `borch/__init__.py`
# mirrors *every* `Tensor` method into module scope (`for _name in dir(Tensor)`), the
# way real torch offers `torch.numel(x)` beside `x.numel()`. borch.ts does not — its
# methods live on the class and its `index` module carries two names. So comparing the
# two top levels counts every method a second time and answers a question about
# module structure while looking like one about features.
#
# The methods themselves are still compared: they are the `Tensor` row.
#
# `transforms` and `transforms.functional` are `borchvision`'s, and **they used to be
# left out.** The reason written here was that their TypeScript side is `vision.ts` and
# the golden's `vision::` cases hold it name by name, so measuring it here would ask the
# same question twice.
#
# That sentence was true and it was not true of everything it covered. `five_crop`,
# `ten_crop` and `to_grayscale` were in `borchvision` and absent from `vision.ts`, and
# **the golden had no case for any of the three** — their classes had cases, the
# functions did not. So this file did not ask because the golden was assumed to, and the
# golden did not ask at all. A name can hide in the seam between two checks that each
# name the other, and neither check is wrong when it happens.
#
# The overlap the old reason worried about is real: most of these names do have a
# `vision::` case, and this row will re-ask about them. That is the cheaper mistake.
# Asking twice costs a listing; asking neither time costs `F.five_crop(x, 32)` — the
# line a tutorial writes — stopping at a name nobody had.
SPACES = frozenset({
    "Tensor", "nn", "nn.functional",
    "optim", "optim.lr_scheduler", "linalg", "utils.data",
    "transforms", "transforms.functional",
    # **These four were off the list, and off the list has no rule.** Their values are
    # measured — the golden asks every one of them — and the value is not what this
    # axis is for: a name that torch has and borch.ts spells differently, or does not
    # have at all, is invisible to a comparison of numbers. `ops` was the sharp case,
    # because it was off the *generator's* list too, so this axis could not have seen
    # it even if it had been named here.
    "ops", "transforms.v2", "transforms.v2.functional", "datasets",
    # **`fft` and `special` were off this list for the same reason `transforms` was:
    # nobody had put them on it.** Both were declined whole in `torch_gap.py` until the
    # day that blanket was measured and found false, and adding them there did not add
    # them here — so a name the core gained and borch.ts did not had no ledger at all
    # between the two. `special` matters most: thirty-four of its names are numpy in
    # `borch/_ops.py` with no WGSL behind them, and without this row the only thing
    # recording that would be a prefix in the golden's core-only list.
    "fft", "special",
})


def refused():
    """Names the core carries **only in order to refuse them.**

    `borch/_tensor.py` binds a stub for each: `to_mkldnn` raises "MKL-DNN is not in
    the browser subset", `symeig` says torch removed it and points at `linalg.eigh`.
    They are the design principle working — *an absent feature beats a wrong answer* —
    and they are not features.

    borch.ts not carrying the stub is therefore **not a missing feature**. It is a
    worse error message: `x.to_mkldnn()` says "not a function" there and says what is
    wrong and why here. Worth fixing one day, and not the same list as `maximum`.

    **Asked of the bound method, not of a table.** The first version imported
    `_GONE`, `_NO_MACHINERY` and `_ABSENT_DTYPES` and found fourteen. Reading the
    source showed why that was too few: as many again are bound by **inline loops**
    with no table to import — sparse, quantisation, storage, the four accelerators.
    A list of table names would have gone on missing those and the miss would have
    been silent, because a refusal counted as a gap looks exactly like a gap.

    So the marker is where the function came from. Every stub is built by one of a
    few factories and keeps that factory in its `__qualname__`; nothing else in the
    core does. If that ever stops being true this raises rather than quietly
    reporting zero — a rule that misses in our favour is worse than no rule, which is
    `torch_gap.py`'s sentence and the reason this one is written to fail closed.
    """
    from borch import _tensor

    # **`_sparse_only` was missing from this tuple and three stubs were being
    # counted as gaps** — `resize_as_sparse_`, `sparse_resize_` and
    # `sparse_resize_and_clear_`, all built by it and all carrying
    # `_sparse_only.<locals>.method`.
    #
    # The docstring above names the hazard exactly ("a refusal counted as a gap
    # looks exactly like a gap") and the floor below is what was meant to stop it.
    # **A floor on the total cannot see partial drift.** Four factories of five
    # still finds forty, which is comfortably over twenty, so the rule that was
    # written to fail closed stayed open for three names.
    factories = ("_bind_gone", "_bind_absent", "_needs_sparse", "_bind_absent_dtype",
                 "_sparse_only")
    out = {name for name in dir(_tensor.Tensor)
           if not name.startswith("_")
           and getattr(getattr(_tensor.Tensor, name, None), "__qualname__", "")
           .startswith(factories)}
    # `_bind_absent_dtype` rewrites its own `__qualname__`, so its names are read
    # from the one table that does exist. Both paths are kept because either alone
    # was measured to be short.
    out |= set(_tensor._ABSENT_DTYPES)
    if len(out) < 20:
        raise SystemExit(
            f"only {len(out)} refusal stubs found in borch/_tensor.py — the factory "
            "names above have probably changed. Fix them rather than letting "
            "refusals be counted as gaps.")
    return out

# **Names the core has and borch.ts is not going to.** Each row is a judgement and
# carries its reason; a name absent from both this table and borch.ts is the to-do
# list. Keyed by `space::name` so that a reason about one name cannot excuse a whole
# namespace — the shape `test_binding_arguments.py` found by keying its own table the
# wrong way first.
#
# **Every row here raises the agreement figure**, so the work slides towards writing
# rows. The rule is `torch_gap.py`'s: a reason has to be checkable, and one that
# cannot be written is a gap. Each row below cites where its reason is already
# established in this repository rather than asserting it fresh.

_ALIAS = "torch's own alias for {}; borch.ts carries the one name"
_DEVICE = ("one device in a browser — borch/__init__.py's `_NOT_OURS` says the same "
           "for `cpu`: there is one device to choose, so we have none")
_STORAGE = ("no storage layer to look into — borch/__init__.py's `_NOT_OURS`, and "
            "the core answers `.storage()` with 'use `.numpy()`'")
_NUMPY = "the numpy bridge; borch.ts has no numpy on the other side of it"

DELIBERATE: dict[str, str] = {
    # torch keeps both spellings of six arcs and four arithmetic names. The core
    # mirrors torch, borch.ts carries the primary — a learner who types the alias
    # gets an error rather than a different answer, which is this project's rule.
    **{f"Tensor::{n}": _ALIAS.format(p)
       for n, p in (("arccos", "acos"), ("arccos_", "acos_"),
                    ("arccosh", "acosh"), ("arccosh_", "acosh_"),
                    ("arcsin", "asin"), ("arcsin_", "asin_"),
                    ("arcsinh", "asinh"), ("arcsinh_", "asinh_"),
                    ("clip", "clamp"), ("divide", "div"), ("subtract", "sub"),
                    ("ndimension", "dim"), ("nelement", "numel"),
                    ("swapdims", "transpose"), ("swapdims_", "transpose_"))},
    # There is one device and no storage objects. Both reasons are the core's own,
    # written into `_NOT_OURS` when the same question was asked of torch.
    # **Three left this list by being answered rather than by being carried.**
    # `pin_memory`, `record_stream` and `untyped_storage` are refusal stubs in the
    # core now, so they sit in the `refusal stub` column and no longer in `gaps` —
    # and a row here that no count reads is dead weight that reads as a live limit.
    # Caught by `test_no_reason_outlives_the_gap_it_explains` on its first run, which
    # is what that check is for.
    **{f"Tensor::{n}": _DEVICE
       for n in ("get_device", "is_pinned", "share_memory_",
                 "is_shared", "is_distributed")},
    **{f"Tensor::{n}": _STORAGE
       for n in ("data_ptr", "const_data_ptr", "storage_offset", "element_size",
                 "is_set_to", "dim_order")},
    # The numpy bridge, which has nothing on the far side in TypeScript.
    **{f"Tensor::{n}": _NUMPY for n in ("numpy", "tolist", "new_tensor")},
    # Worker processes do not exist in a browser, which is `run.py`'s stated reason
    # for the two `dataconv::` cases it leaves unported.
    "utils.data::get_worker_info":
        "no worker processes in a browser — borch-ts/test/run.py says the same",
    "utils.data::default_convert":
        "converts numpy and Python containers into tensors; neither is on the "
        "TypeScript side of the bridge",
    # **Three marks with nothing to read them.**
    #
    # `nn.Buffer(t)` is torch's way of saying *this value is not trained and is
    # saved*, and in torch it is the tensor itself — what acts on the mark is
    # `register_buffer`. borch.ts has `registerBuffer(name, value, persistent)`
    # taking the tensor directly, so a `Buffer` wrapper would be a name that
    # produces its own argument back and is read by nobody. **That is what this
    # project calls a hollow name**, and `data.ts`'s header argues against exactly
    # it: torch's shape with nothing inside, quietly ignoring what the caller
    # passes.
    "nn::Buffer":
        "a mark that `registerBuffer` reads; borch.ts's takes the tensor directly, "
        "so the wrapper would hand its argument back and nothing would read it",
    # `LazyModule` holds `built: Module | null` and makes the real layer when the
    # size arrives — before that there are **no parameters at all**, not parameters
    # in an uninitialised state. The core has these because Python's `parameters()`
    # has to produce *something* and its lazy layers put an object in a
    # `state_dict`; borch.ts has nothing for the name to name.
    **{f"nn::{n}": "borch.ts's LazyModule has no parameters before it is built — "
                   "not uninitialised ones, none — so there is nothing to name"
       for n in ("UninitializedParameter", "UninitializedBuffer")},
    # **The sparse queries, which the core answers only because a dense tensor is
    # the trivial case.** `dense_dim` is the rank, `sparse_dim` is 0, `to_dense` is
    # the tensor itself, and `is_coalesced` refuses because a dense tensor has no
    # coalesce state at all. Every one is a question about a layout borch.ts does
    # not have, and the three that answer do so by having nothing to answer about.
    #
    # Held apart from the `_sparse_only` stubs above: those three raise, and these
    # four return. A name that answers and a name that refuses are different rows
    # even when the reason underneath is one reason.
    **{f"Tensor::{n}": "a question about the sparse layout; borch.ts has one layout, "
                       "and the core only answers because dense is the trivial case"
       for n in ("dense_dim", "sparse_dim", "is_coalesced", "to_dense")},
    "Tensor::sspaddmm": "sparse-by-sparse addmm; there is no sparse layout here",
    # **Held absent on purpose, and checked in both directions.** A lesson page
    # teaches a reader what to do without this class, and pins the absence at both
    # ends — the sentence must stay and the name must not appear. Filling it would
    # leave a page teaching a detour around a road that exists.
    #
    # The pooling itself is here (`adaptivePool`), and the nine `nn.functional`
    # adaptive names are too — those are camelCase and `_folds_onto` refuses to fold
    # a capitalised name, so the class stays a gap however many of them go in. That
    # was measured against the index rather than assumed, including the positive
    # control the page relies on (`AdaptiveAvgPool1d` must be *findable*).
    #
    # It moves when that page does, and not before.
    "nn::AdaptiveAvgPool2d": "held — a lesson page teaches the way around this "
                             "absence and pins it at both ends; the pooling itself "
                             "is here as `adaptivePool`",
    # **A per-element callback needs the values, and getting them is asynchronous.**
    # torch's three run a Python function over every cell and are CPU-only for that
    # reason. On this side the values live in a GPU buffer and `toArray()` returns a
    # promise, so a *synchronous* `apply_` cannot exist — and an asynchronous one
    # would be a different function wearing the name.
    **{f"Tensor::{n}": "a per-element callback needs the values, and reading them "
                       "back from the GPU is asynchronous — a synchronous apply_ "
                       "cannot exist here"
       for n in ("apply_", "map_", "map2_")},
    # Storage surgery. borch.ts's tensor owns a GPU buffer of a fixed size and every
    # shape operation materialises a new one through a plan; there is no second
    # tensor pointing at the same storage for these to re-aim.
    **{f"Tensor::{n}": "these re-aim or re-size the storage; a borch.ts tensor owns "
                       "its buffer and shape operations materialise a new one"
       for n in ("resize_", "resize_as", "resize_as_", "set_", "new")},
    # `torch.inference_mode` is a second switch beside `no_grad`, and borch.ts has
    # only the one. A tensor cannot be *in* a mode that does not exist.
    "Tensor::is_inference": "there is no inference mode on this side, only no_grad",
    # The core carries this to refuse it (no uint64, no settled hash spec) and the
    # refusal is not built by a stub factory, so `refused()` cannot see it.
    "Tensor::hash_tensor": "the core carries this only to refuse it — no uint64 and "
                           "no settled hash spec — and the refusal is written by "
                           "hand rather than by a stub factory",
    # `igamma`/`igammac` are module functions in `special.ts`, which imports `Tensor`;
    # a `Tensor.igamma` calling back into it would close the import cycle. The
    # underscore forms need the plain ones to exist first, so both wait on that.
    **{f"Tensor::{n}": "the partner lives in special.ts, which imports Tensor — a "
                       "method calling back into it closes the import cycle"
       for n in ("igamma_", "igammac_")},
    # **Two that are owed rather than refused**, and saying so is the point of the
    # row: `sum_to_size` folds broadcast axes back and `multinomial` draws by weight,
    # and both are ordinary work nobody has done. They are here so that "no reason"
    # means *nobody has looked*, and these have been looked at.
    #
    # **`retain_grad` was the third and it has gone.** Its row read *borch.ts keeps
    # gradients on leaves only, and this asks for one on a non-leaf* — true when
    # written, and the mechanism arrived with `backward(…, inputs)`, which needs the
    # same thing. One row leaving with work it did not name is the shape an `owed`
    # entry is supposed to have: it stated what was missing, not who would do it.
    "Tensor::sum_to_size": "owed — folds broadcast axes back; ordinary work, not yet done",
    "Tensor::multinomial": "owed — draws by weight; `WeightedRandomSampler` in data.ts "
                           "does the same arithmetic and could be shared",
    # **Not absent — spelled as a type.** The core needs a runtime object because
    # Python has no union of literals; TypeScript has one, so `vision.ts` writes the
    # parameter as `"bilinear" | "nearest"` and the compiler refuses a third value
    # before it runs. An exported enum beside it would be a second spelling of the
    # same choice, and a reader who typed the wrong one would learn it at run time
    # instead of while typing.
    **{f"{ns}::InterpolationMode":
       'not a name here: `vision.ts` takes `"bilinear" | "nearest"` as a type, so the '
       "compiler refuses a wrong mode before the call runs"
       for ns in ("transforms", "transforms.functional",
                  "transforms.v2", "transforms.v2.functional")},
    # **`datasets` is a decoder here and a catalogue there, and the difference is the
    # half that cannot cross.** A dataset is an address and a format. The address —
    # fetching, caching, checksums — needs hosts that send a CORS header, and
    # torchvision's own do not (`cs.toronto.edu` and `ossci-datasets.s3.amazonaws.com`,
    # measured), so a `MNIST` class here would be a constructor whose `download=True`
    # raises. What crossed is the format, which is the half that goes wrong quietly:
    # `datasets.ts` reads IDX, STL10's bytes, `.npy` and FER2013's CSV, and the golden
    # holds those decoders against real torchvision.
    #
    # So these eighteen are not eighteen decisions. They are one, taken once, and the
    # row is written per name because a table that groups them would hide the day one
    # of them becomes possible — a `FakeData` needs no network at all.
    **{f"datasets::{n}":
       "the class is the address half — fetching, caching and checksums — and "
       "torchvision's hosts send no CORS header. The format half is in `datasets.ts` "
       "and the golden holds it"
       for n in ("VisionDataset", "MNIST", "FashionMNIST", "KMNIST", "QMNIST",
                 "EMNIST", "CIFAR10", "CIFAR100", "FakeData", "SEMEION", "USPS",
                 "STL10", "SVHN", "Omniglot", "GTSRB", "FER2013", "MovingMNIST",
                 "DatasetFolder", "ImageFolder", "CLEVRClassification",
                 "RenderedSST2", "Sintel")},
    # **The other thirty-three, and their reason is not the twenty-two's.** Those have
    # a format half in `datasets.ts` and are held back by the address half alone.
    # These have **no format half at all**: every one of them is a reader that walks a
    # directory tree or opens an archive — a `.zip` of per-class folders, a `.tar` of
    # `.pfm` disparity maps, a `.mat` of annotations — and a page has neither a
    # filesystem nor an archive to walk. `borchvision` has them because it runs on
    # Python beside a disk.
    #
    # Written out per name rather than as a rule about the namespace, for the reason
    # the twenty-two above give: a table that groups them hides the day one of them
    # becomes possible. Two here are closer than the rest — `Imagenette` and
    # `Country211` are folder-of-folders and would arrive the day a directory can be
    # handed in — and grouping would bury that.
    **{f"datasets::{n}":
       "a reader that walks a directory tree or opens an archive; a page has neither "
       "a filesystem nor an archive, and unlike the twenty-two above there is no "
       "format half of this one in `datasets.ts`"
       for n in ("CREStereo", "Caltech101", "Caltech256", "CarlaStereo",
                 "Country211", "DTD", "ETH3DStereo", "EuroSAT", "FGVCAircraft",
                 "FallingThingsStereo", "Flickr30k", "Flickr8k", "FlyingChairs",
                 "FlyingThings3D", "Food101", "HD1K", "INaturalist", "Imagenette",
                 "InStereo2k", "Kitti", "Kitti2012Stereo", "Kitti2015Stereo",
                 "KittiFlow", "Middlebury2014Stereo", "OxfordIIITPet", "PhotoTour",
                 "SUN397", "SceneFlowStereo", "SintelStereo", "StanfordCars",
                 "VOCDetection", "VOCSegmentation", "WIDERFace")},
    # ── the tv_tensor family ──
    #
    # **One decision, and it is about the type system rather than about any of these
    # twenty.** torchvision marks a tensor as a bounding box, a mask, a set of
    # keypoints or a video by *subclassing* `torch.Tensor`, and every name below
    # either reads that mark, writes it, or dispatches on it. borch.ts's `Tensor` is a
    # handle to a GPU buffer and cannot be subclassed to carry a label — the golden's
    # `v2::` and `v2f::` skip rows say the same thing and are where this was first
    # written down.
    #
    # The `_video` / `_bounding_boxes` / `_keypoints` / `_mask` suffixes are the
    # kernel entries that dispatch reaches; the unsuffixed `uniform_temporal_subsample`
    # is not here because borch.ts has it, which is the shape of the distinction.
    **{f"transforms.v2::{n}":
       "reads, writes or dispatches on torchvision's tv_tensor mark, which is a "
       "`Tensor` subclass; borch.ts's is a handle to a buffer and cannot carry a label"
       for n in ("ClampBoundingBoxes", "ClampKeyPoints", "ConvertBoundingBoxFormat",
                 "SanitizeBoundingBoxes", "SanitizeKeyPoints", "SetClampingMode",
                 "check_type", "get_bounding_boxes", "get_keypoints", "has_all",
                 "has_any", "query_chw", "query_size")},
    **{f"transforms.v2.functional::{n}":
       "a tv_tensor kernel entry — the suffix names the mark it dispatches on, and "
       "borch.ts's `Tensor` cannot carry one"
       for n in ("clamp_bounding_boxes", "convert_bounding_box_format",
                 "get_num_frames_video", "get_size_bounding_boxes",
                 "get_size_keypoints", "get_size_mask", "is_pure_tensor",
                 "uniform_temporal_subsample_video")},
    # **The thirty-three `*_video` kernels, and the reason is a type rather than a
    # gap.** The core's picture kernels count their axes from the end now, so
    # `(H, W, C)` and `(T, H, W, C)` name the same two and a video is an image with
    # something in front of it. borch.ts's vision side does not take a ranked array at
    # all: `Image` in `vision.ts` is `{ data: Float64Array, height, width, channels,
    # isByte }` — one picture by construction, with nowhere to put a frame axis.
    #
    # So this is the `v2::` row's shape and not the `ops::` row's: a mechanism that
    # side has not got, rather than a body it has not been given. Carrying the names
    # across would mean giving `Image` a rank, which is a change to every transform in
    # that file and not thirty-three aliases.
    # **`special` had thirty-four rows here and has none.** Worth an account, because
    # the number came down in steps and the steps were the point: the thirty-four were
    # not one kind of job, and treating them as one is what had kept all of them
    # written down rather than done.
    #
    # 34 → 16: eighteen were **arrangements of operations borch.ts already had** —
    # twelve orthogonal recurrences (`mul` and `sub` in a loop) and six compositions
    # whose *safe* form is a composition. `ndtr` is the one worth naming: it looks like
    # it belongs with the sixteen and does not, because `erfc(-x/√2)/2` needs no kernel
    # while `(1 + erf(x/√2))/2` would have needed one or a wrong answer.
    #
    # 16 → 0: fifteen became entries in the `UNARY` table in `kernels.ts`, and `zeta` a
    # loop over tensor operations, since that table has no seat for a binary op and
    # inventing one for a single name is a mechanism with one user.
    #
    # **The three that were wrong the first time are the reason the golden asks at the
    # tails.** WGSL refuses a constant NaN, and a shader that fails to compile does not
    # raise — `bessel_y0(0.001)` came back 99.97, which was `k1`'s value from the
    # dispatch before it. `k`'s series lost seven digits to cancellation in f32 and
    # became the Abramowitz & Stegun minimax table. And Airy's seam moved from 8 to 6
    # for the same reason, at a third the golden's tolerance. None of the three is
    # visible from an ordinary input.
    #
    # Nothing is written below for that namespace, which is what a closed row looks
    # like here — `test_ts_axis.py` holds its count at 0 and `torch_gap.py` holds the
    # core at 56 of 56.
    **{f"transforms.v2.functional::{n}_video":
       "borch.ts's `Image` is `{data, height, width, channels}` — one picture by "
       "construction, with no axis for frames. The core's kernels count from the end "
       "and take a video for free; giving that side a rank is a change to the type, "
       "not an alias"
       for n in ("adjust_brightness", "adjust_contrast", "adjust_gamma", "adjust_hue",
                 "adjust_saturation", "adjust_sharpness", "affine", "autocontrast",
                 "center_crop", "crop", "elastic", "equalize", "erase", "five_crop",
                 "gaussian_blur", "gaussian_noise", "get_dimensions",
                 "get_num_channels", "get_size", "horizontal_flip", "invert",
                 "normalize", "pad", "permute_channels", "perspective", "posterize",
                 "resize", "resized_crop", "rotate", "solarize", "ten_crop",
                 "to_dtype", "vertical_flip")},
}

# **Names borch.ts has and the core does not.** The reverse direction is not
# symmetric: borch.ts is a browser library and carries things a numpy core has no
# reason to (`init`, `device`, `keepAlive`, `scope`). Left empty until measured —
# writing rows here before running it would be inventing the answer.
EXTRA: dict[str, str] = {}


def ts_names():
    """Every name in the generated index, as one set.

    **The index records one home per name and the comparison must not use it.** It is
    `{name: "module.path.name"}`, and `det` is recorded as `tensor.Tensor.det` — the
    method — not as `linalg.det`. Asking "is `det` in the `linalg` module of the
    index" therefore answers no about a name that is present.

    The first version of this file did exactly that and reported **1,137 core-only
    names**, of which nearly all were a name filed under a different home. A number
    that large reads as a finding; it was a mapping error, and it was caught by the
    one namespace whose whole content came back missing at once.

    So membership is asked of the **whole surface**: does borch.ts have this name
    anywhere. What that gives up is "and in the right namespace", which the index
    cannot answer — said here rather than left for the next reader to assume.
    """
    if not INDEX.exists():
        raise SystemExit(f"no {INDEX.relative_to(ROOT)} — run npm run docs:api first")
    raw = json.loads(INDEX.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"{INDEX.relative_to(ROOT)} is not the expected name → path map")
    return set(raw)


def _camel(name):
    """`return_indices` → `returnIndices`. **Only the spelling, not a translation.**

    borch.ts is camelCase and the core is snake_case, so comparing them raw reports
    every multi-word name as missing on both sides at once. That is not a gap; it is
    two spellings of one name. A name with no underscore passes through unchanged,
    which is most class names.

    **A trailing underscore is kept, and the first version dropped it.** torch marks
    in-place with it — `eq_` writes into the receiver where `eq` returns a new
    tensor, and they are two functions rather than two spellings. Splitting on `_`
    naively gives `eq_` an empty last part, so it camelled to `eq` and matched the
    out-of-place method.

    That was caught by the number improving too much: adding six comparison methods
    dropped the count by twelve. `torch_gap.py` says a rule that misses in our favour
    is worse than no rule, and this one missed in our favour by exactly the six
    in-place forms nobody had written.

    **A leading underscore is kept too, and it was not.** The paragraph above was
    written about the other end of the string and applied to one end only:
    `_weight` split to `["", "weight"]`, the empty head vanished, and the name came
    out `Weight` → `weight`. So the core's `_random_samples` folded onto borch.ts's
    `randomSamples` and `FractionalMaxPool2d` and `3d` reported **`agree` against
    parameter lists they did not match**.

    That is the one direction an instrument must not fail in. A row wrongly saying
    *these differ* costs an hour; a row wrongly saying *these agree* costs nothing
    until it costs a user. Four parameters are in the class — `_weight`, `_freeze`,
    `_random_samples`, `_stacklevel` — and torch marks all four private on purpose.

    `test_fold_is_lossless.py` holds the rule rather than the row: the fold has to
    keep every letter, keep the underscores at both ends, and never put two names on
    one string. The last of those is the half no single comparison can show.
    """
    leading = "_" * (len(name) - len(name.lstrip("_")))
    trailing = "_" if name.endswith("_") else ""
    head, *rest = name.strip("_").split("_")
    return (leading + head
            + "".join(p[:1].upper() + p[1:] for p in rest) + trailing)


def _folds_onto(name, lowered):
    """Whether `name` matches something in borch.ts once case is set aside.

    **Needed, because `_camel` cannot bridge every spelling.** The core writes
    `tensorinv` and `tensorsolve` with no underscore at all, and borch.ts writes
    `tensorInv` and `tensorSolve`; there is nothing for a split-on-underscore rule
    to work with. Twelve names sit in that shape and dropping the fold invents
    twelve gaps that are not there — measured.

    **But the first letter is kept, and that is the whole care in this function.**
    In torch, an initial capital is the class/function boundary: `nn.Embedding` is
    a layer and `nn.functional.embedding` is a function, and they are not each
    other. Folding both would report the layer as present because the function is,
    which is a normaliser erasing a distinction the domain makes — the same fault
    as `_camel` stripping the in-place underscore, one line down and pointing the
    other way.

    So: any normaliser is a claim about which differences do not matter. This one
    claims that internal capitalisation does not and that the first letter does.
    """
    if not name:
        return False
    if name[0].isupper():                        # a class; the capital is the name
        return False
    return name.lower() in lowered


def compare():
    """`{space: (gaps, refusals)}`, with the two spellings reconciled.

    Both are core-only. They are returned apart because they are different work: a
    gap wants the feature written, a refusal wants the message carried across.
    """
    import torch_gap

    theirs = ts_names()
    lowered = {n.lower() for n in theirs}
    stubs = refused()
    out = {}
    for space, _real, ours in torch_gap._spaces():
        if space not in SPACES:                  # borchvision's spaces are held elsewhere
            continue
        missing = [n for n in torch_gap._public(ours)
                   if _camel(n) not in theirs and not _folds_onto(n, lowered)]
        out[space] = (sorted(n for n in missing if n not in stubs),
                      sorted(n for n in missing if n in stubs))
    return out


def main(argv):
    show = argv[argv.index("--show") + 1] if "--show" in argv else None
    rows = compare()
    unexplained = 0
    stubs = 0
    for space, (gaps, refusals) in rows.items():
        loose = [n for n in gaps if f"{space}::{n}" not in DELIBERATE]
        unexplained += len(loose)
        stubs += len(refusals)
        mark = " " if not loose else "✘"
        print(f"  {mark} {space:22s} gap {len(gaps):>4}  "
              f"without a reason {len(loose):>4}   refusal stub {len(refusals):>4}")
        if show is not None and space.startswith(show):
            for name in gaps:
                why = DELIBERATE.get(f"{space}::{name}", "**no reason**")
                print(f"      · {name}  {why}")
            for name in refusals:
                print(f"      · {name}  (the core only refuses it too)")
    print(f"\n이름만 센다 — 서명도 값도 안 본다.")
    print(f"까닭 없는 결손 {unexplained}건 · 거절 스텁 {stubs}건.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
