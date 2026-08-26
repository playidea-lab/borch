"""Counts the names real torch has that **we do not**.

    uv run --with numpy --with torch --with torchvision python tests/torch_gap.py
    uv run --with numpy --with torch --with torchvision python tests/torch_gap.py --extra
    uv run --with numpy --with torch --with torchvision python tests/torch_gap.py --show nn

`tests/conformance.py` asks whether **what is there is right**. This file asks **what is
not there**. They are different questions, and measuring only the first gives 100% while
saying nothing about how narrow the surface is — which is very nearly how it was read.

## What is missing is not all of one kind

It divides in **four**, and that division is this file's point.

- **Not API to begin with** — `AliasDb`, `ClassType`, `boolean_dispatch`, `ByteStorage`.
  Public only in that the name has no underscore; not a thing to call. TorchScript IR
  internals, dispatcher internals and old classes torch itself threw away live here.
- **Deliberately declined** — `cuda`, `compile`, distribution, quantisation. Things that
  cannot exist in a browser, or that have to be learned outside one. Imitating them
  loses the lesson.
- **Outside the curriculum** — `torch.fft`, `torch.sparse`, `torch.onnx`. They could go
  in, but an introductory tutorial does not call them, and a wider surface is more room
  to be quietly wrong.
- **Simply absent** — it should be there and nobody asked. **This list is the one worth
  having.** `torch.sum` was on it (it existed as a method with no module-level function).

The first three are judgements and the last is a defect. A machine cannot tell the four
apart, so the first three are written into the tables below — **whatever is not written
is what wants reviewing.**

## torchvision is counted here too, and was not for a long time

`transforms` and `transforms.functional` were **not on the namespace list**, so every name
absent from `borchvision` was absent from the measure as well: no count, and no reason
demanded of it. The number in circulation, "7 of 41", was arrived at by hand, which is the
method this whole file exists to replace.

That shape has now cost this repository four times — a helper list that lost to new shapes,
a file simply not in `test_messages.py`'s list, a README row nothing compared name by name,
and this. **What is off the list has no rule.**

`transforms.functional` was counted against an **empty** namespace at first, because we had
none. It could have been left out on the grounds that it did not exist, and that is exactly
the move that hid it: absent from the list reads as zero to review, and absent from *us*
while present in the list reads as thirty-six to review. The second was the true sentence,
and it is what put the namespace on the to-do list it has now come off.

## `datasets` reads 15 of 72, and **nothing in this file is absent without a reason**

The count of unexplained gaps was eight this morning. It is zero.

That number was never the goal — a zero bought by inventing reasons is worse than
an eight — but the way it got there is worth the paragraph. Five were built because
the list sat there being uncomfortable. Two more were downloads: `EMNIST` at 562MB
and `STL10` at 2.6GB were declined with "a cost, not an impossibility", and the way
they came off was somebody waiting.

**The last one came off because the reason was wrong.** `FER2013` was written down
as impossible to check: torchvision has no `download` for it, it wants a Kaggle
account, so — the sentence went — there is nothing here to compare an implementation
against. That is *cannot fetch the data* carried over into *cannot check the code*,
and they are different claims. torchvision's reader takes a directory. A CSV written
in the case table goes to both sides and the comparison is as real as every other one
here. **It was the same over-wide refusal this row has now produced three times**,
and each time the shape was identical: a true sentence about one thing, used as a
reason about another.

The 57 that stay declined are the codec: most of torchvision's datasets are folders
of JPEG or PNG, numpy decodes neither, and adding a decoder is the dependency this
library does without — the same answer PIL gets in `transforms`, arriving from the
other side. That reason has been checked against every name it covers, which is what
the three corrections were for.

## `transforms.v2` is on the list and `transforms.v2.functional` is not

v2 is torchvision's current recommended API and it was **invisible to this measure**
until it was put on the list — not declined, not counted, just never asked about,
because `dir(torchvision.transforms)` does not carry `v2` until something imports it.
That is the same shape as `transforms` itself being off the list, one level up.

Counted against an absent namespace it read 0 of 72, which was the true sentence about
the namespace and a false one about the library: 38 of those names already existed
here under `transforms`, computing the same values. It now reads 52 of 72, and the
gap between those two numbers is the whole reason a namespace is put on the list
before it is built rather than after.

**What v2 changes over v1 is what it prints, not what it computes.** Measured across
the comparable names: values agreed everywhere, and 21 of 33 reprs differed —
`Resize(5)` keeps its size as `[5]`, `ColorJitter` drops the arguments left at `None`
instead of printing them. So the transforms here subclass v1's and override only the
repr, and the golden file freezes 52 of those strings against real torchvision's. Four
were wrong before they were right, every one found by comparing.

The 20 still declined are the tv_tensor half — boxes, masks, keypoints and video
travelling alongside the picture — plus the base class whose body *is* that dispatch.
`MixUp` and `CutMix` were in that group and are not: they need a batch and a label,
which is unlike everything else here, but they need nothing this library lacks, and
"it is unlike the others" is not a reason.

`transforms.v2.functional` **is on the list now, and was not.** It reads 43 of 165.

It was off with a paragraph rather than off silently, and the paragraph said what was
blocking it: 114 of its 165 public names are `<operation>_<type>` dispatch kernels —
`affine_image`, `affine_mask`, `affine_bounding_boxes`, `affine_keypoints` and
`affine_video` are one operation counted five times — and one reason covers all of
them, but the only wildcard this matcher understood was flat. `"*_image"` written flat
also swallows `to_pil_image` in v1, attaching a sentence about v2's type dispatch to a
name that has nothing to do with it.

So the matcher takes namespaced wildcards now, and the paragraph became five rows and
a number. That is the shape worth noticing: **a namespace stayed uncounted for as long
as the tool could not express its reason**, which is the same failure as being off the
list, wearing an explanation. A paragraph is not a number, and nothing was watching it.

(The old paragraph said 128 kernels. Measured now: 114. Neither this file nor anything
else was checking that figure, which is the point being made one line up.)

**And the number counts names, not signatures.** `Grayscale` present with the wrong luma
weights counts the same as `Grayscale`; so does `Pad` present without `padding_mode`. This
is not hypothetical — measured on the borch.ts side, `MaxPool2d` was present taking
`(kernel)` alone against the core's `return_indices`, and `InstanceNorm` took `(eps?)`
against five arguments. **A name comparison finds absent names and cannot find a name that
lies about what it accepts.** What holds signatures here is the golden cases and the
arguments they pass, not this count — so quoting "21 of 41" as coverage is quoting the wrong
number for that question.

## Filling in this table is dangerous work

**Every name written down raises our percentage.** So the work slides, by its nature,
towards "making the number look good". There is one thing stopping it — **every line
carries a reason, and the reason has to be checkable.** A line whose reason cannot be
written is a gap.

**Trying to divide them by machine went badly wrong once.** The rule "anything whose
`__module__` starts with `torch._C` is an internal type" looked plausible and swallowed
`linalg`'s `eig`, `eigvals` and `ldl_factor` — seven of them — along with `F.elu_`,
`hardtanh_` and `leaky_relu_`. **All of those are real API**; most of torch's functions
live in C. With that rule in, `linalg` would have jumped from 83% to 100% and the seven
real gaps would have vanished from the list. **An automatic rule that misses in our
favour is worse than no rule.** So what follows was read and written by hand.
"""

import sys

import torch

try:                                            # the vision half needs one more package
    import torchvision
    # **`v2` has to be imported by name.** It is not an attribute of
    # `torchvision.transforms` until something imports it, which is precisely why this
    # measure could not see it: `dir()` on the parent does not list it, so no sweep
    # that walks attributes reaches it. A namespace can be invisible to a measure
    # without being hidden.
    import torchvision.transforms.v2                              # noqa: F401
    import torchvision.transforms.v2.functional                   # noqa: F401
    import torchvision.ops                                        # noqa: F401
    import torchvision.datasets                                   # noqa: F401
except ImportError:                             # pragma: no cover - measured by test_gap
    torchvision = None

import borch
import borchvision

# ------------------------------------------------------------- tables
#
# All three tables share the same **spelling**:
#   "name"     — that one name
#   "prefix*"  — names beginning with it
#   "*suffix"  — names ending with it
#
# Substrings are not used. An earlier version filtered with `key in full`, which let a key
# like `"special"` swallow unrelated names — nothing went wrong, but it was wide.

# **A whole namespace.** Matched on the prefix of the namespace's name.
DELIBERATE = {
    "cuda": "there is no CUDA in a browser. Imitating it loses the lesson about GPUs",
    "mps": "same reason",
    "xpu": "same reason",
    "mtia": "same reason",
    "distributed": "this is one tab. Learning distribution means leaving for several machines",
    "compile": "TorchDynamo rewrites CPython bytecode. It does not sit on wasm",
    "jit": "same reason",
    "export": "same reason",
    "fx": "same reason",
    "onnx": "exporting is deployment's job and this is grammar practice",
    "quantiz": "quantisation means something only on real hardware",
    "sparse": "outside the curriculum",
    # **The eight one-dimensional ones are done** (`fft`, `ifft`, `rfft`, `irfft`,
    # `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`). What is not done is 2-D, N-D and the
    # fifteen Hermitian variants. While this read only "outside the curriculum", those
    # eight were already running — declining a whole namespace hides **what is done there
    # too.**
    "fft": "only the eight one-dimensional ones — 2-D, N-D and the Hermitian variants are outside the curriculum",
    "special": "outside the curriculum",
    "futures": "outside the curriculum",
    "package": "outside the curriculum",
    "profiler": "outside the curriculum",
    "utils.tensorboard": "outside the curriculum",
    "backends": "there is one backend to choose",
    "multiprocessing": "a browser has no processes",
}

# The old per-type classes torch used to carry. The list is closed, so it is spelled out
# here — a suffix wildcard would catch `torch.Tensor` itself.
#
# **The two are not the same list.** The complex and quantised types have a `Storage` and
# no `Tensor` — written as one, seven dead rows appeared and `test_gap.py` caught them.
_OLD_STORAGE = ("BFloat16", "Bool", "Byte", "Char", "ComplexDouble", "ComplexFloat",
                "Double", "Float", "Half", "Int", "Long", "QInt32", "QInt8",
                "QUInt2x4", "QUInt4x2", "QUInt8", "Short")
_OLD_TENSOR = ("BFloat16", "Bool", "Byte", "Char", "Double", "Float", "Half",
               "Int", "Long", "Short")

# TorchScript's type names. This list is closed too.
_IR_TYPES = ("Any", "Await", "Bool", "Class", "Complex", "DeviceObj", "Dict", "Enum",
             "Float", "Future", "Int", "Interface", "List", "None", "Number",
             "Optional", "PyObject", "RRef", "StreamObj", "String", "SymBool",
             "SymInt", "Tensor", "Tuple", "Union")


# **Names that are not API to begin with.** Public only in having no underscore; not things to call.
NOT_API = {
    **{f"{d}Storage": "a per-type Storage class — torch deprecated it" for d in _OLD_STORAGE},
    **{f"{d}Tensor": "a per-type Tensor class — torch deprecated it (FloatTensor …)"
       for d in _OLD_TENSOR},
    **{f"{t}Type": "the TorchScript type system (AnyType, ListType, TensorType …)"
       for t in _IR_TYPES},
    "Type": "the TorchScript type system",
    # TorchScript's IR and type system. Internal things `torch.jit` uses put names at the
    # top level too. `DELIBERATE`'s "jit" matches only namespace names and does not reach here.
    "AliasDb": "TorchScript IR internals",
    "Argument": "TorchScript IR internals",
    "ArgumentSpec": "TorchScript IR internals",
    "CompleteArgumentSpec": "TorchScript IR internals",
    "Block": "TorchScript IR internals",
    "CallStack": "TorchScript IR internals",
    "Capsule": "TorchScript IR internals",
    "Code": "TorchScript IR internals",
    "CompilationUnit": "TorchScript IR internals",
    "ConcreteModuleType": "TorchScript IR internals",
    "ConcreteModuleTypeBuilder": "TorchScript IR internals",
    "DeepCopyMemoTable": "TorchScript IR internals",
    "ErrorReport": "TorchScript IR internals",
    "ExecutionPlan": "TorchScript IR internals",
    "FileCheck": "a TorchScript testing tool",
    "FunctionSchema": "TorchScript IR internals",
    "Gradient": "TorchScript IR internals",
    "Graph": "TorchScript IR internals",
    "GraphExecutorState": "TorchScript IR internals",
    "IODescriptor": "TorchScript IR internals",
    "InferredType": "TorchScript IR internals",
    "JITException": "TorchScript IR internals",
    "LiteScriptModule": "TorchScript IR internals",
    "LockingLogger": "TorchScript IR internals",
    "LoggerBase": "TorchScript IR internals",
    "NoopLogger": "TorchScript IR internals",
    "Node": "TorchScript IR internals",
    "OperatorInfo": "TorchScript IR internals",
    "PyTorchFileReader": "serialisation internals",
    "PyTorchFileWriter": "serialisation internals",
    "SerializationStorageContext": "serialisation internals",
    "DeserializationStorageContext": "serialisation internals",
    "StaticModule": "TorchScript IR internals",
    "Tag": "TorchScript IR internals",
    "TracingState": "TorchScript IR internals",
    "Use": "TorchScript IR internals",
    "Value": "TorchScript IR internals",
    "FatalError": "TorchScript IR internals",
    "AggregationType": "a profiler aggregation kind — there is no profiler to measure with",
    "Script*": "a TorchScript object (ScriptModule, ScriptFunction …)",
    "import_ir_module": "TorchScript IR internals",
    "import_ir_module_from_buffer": "TorchScript IR internals",
    "merge_type_from_type_comment": "TorchScript IR internals",
    "parse_ir": "TorchScript IR internals",
    "parse_schema": "TorchScript IR internals",
    "parse_type_comment": "TorchScript IR internals",
    "unify_type_list": "TorchScript IR internals",
    "fork": "TorchScript async execution",
    "wait": "TorchScript async execution",
    "ThroughputBenchmark": "a TorchScript benchmarking tool",
    "BenchmarkConfig": "a TorchScript benchmarking tool",
    "BenchmarkExecutionStats": "a TorchScript benchmarking tool",

    # The per-type Storage and Tensor classes. torch threw them away itself —
    # `torch.FloatTensor(...)` warns if used today.
    #
    # **It must not be written as `"*Tensor"`.** Written that way it swallowed
    # `torch.Tensor` itself — the name at the centre of this library. `test_gap.py` caught
    # it. A suffix wildcard only ever misses in the direction of swallowing real names, so
    # it is not used.
    "Storage": "a per-type Storage class — torch deprecated it",
    "StorageBase": "a per-type Storage class — torch deprecated it",
    "TypedStorage": "a per-type Storage class — torch deprecated it",
    "UntypedStorage": "a per-type Storage class — torch deprecated it",
    "set_default_tensor_type": "deprecated along with the per-type Tensor classes",

    # Dispatcher internals. The place that chooses which backend an operation goes to, and
    # we have no backend to choose.
    "DispatchKey": "dispatcher internals",
    "DispatchKeySet": "dispatcher internals",
    "ExcludeDispatchKeyGuard": "dispatcher internals",
    "DisableTorchFunction": "dispatcher internals",
    "DisableTorchFunctionSubclass": "dispatcher internals",
    "boolean_dispatch": "dispatcher internals",
    "handle_torch_function": "dispatcher internals",
    "has_torch_function": "dispatcher internals",
    "has_torch_function_unary": "dispatcher internals",
    "has_torch_function_variadic": "dispatcher internals",
    "assert_int_or_pair": "an argument-checking tool — used only inside F",
    "factory_kwargs": "an internal tool layers use to pass device and dtype along",
    "swap_in_optimizer_params_and_state": "an optimizer internal tool",
    "argument_validation": "a DataPipe internal tool",
    "functional_datapipe": "a DataPipe internal tool",
    "classproperty": "a Python helper — not an operation",

    # Labels imported from elsewhere. **Written with their namespace attached** — written as
    # bare names, `"Optimizer"` swallowed `torch.optim.Optimizer` (real API, and we built
    # it) and `"Tensor"` swallowed `torch.Tensor`. `test_gap.py` caught it.
    "nn.functional.Tensor": "an imported label — not F's API",
    "transforms.functional.Tensor": "an imported label — not that namespace's API",
    "optim.lr_scheduler.Tensor": "an imported label — not that namespace's API",
    "optim.lr_scheduler.Optimizer": "an imported label — not that namespace's API",
    "ScalingType": "an imported label — an fp8 scaling kind",
    "SwizzleType": "an imported label — an fp8 layout kind",

    # The variants the functionalisation pass uses. Not names a user calls — a real name
    # doing the same work sits beside each (`view` next to `view_copy`).
    #
    # **It must not be written as `"*_copy"`.** Written that way it swallowed `index_copy`,
    # which is not a functionalisation variant but a real operation we had already built.
    # Long as the list is, it is written by hand — a wide net here only ever misses in our
    # favour.
    "alias_copy": "a functionalisation-pass variant — the real name is `alias`",
    "as_strided_copy": "a functionalisation-pass variant — the real name is `as_strided`",
    "ccol_indices_copy": "a functionalisation-pass variant (sparse layout)",
    "col_indices_copy": "a functionalisation-pass variant (sparse layout)",
    "crow_indices_copy": "a functionalisation-pass variant (sparse layout)",
    "row_indices_copy": "a functionalisation-pass variant (sparse layout)",
    "indices_copy": "a functionalisation-pass variant (sparse layout)",
    "values_copy": "a functionalisation-pass variant (sparse layout)",
    "detach_copy": "a functionalisation-pass variant — the real name is `detach`",
    "diagonal_copy": "a functionalisation-pass variant — the real name is `diagonal`",
    "expand_copy": "a functionalisation-pass variant — the real name is `expand`",
    "permute_copy": "a functionalisation-pass variant — the real name is `permute`",
    "select_copy": "a functionalisation-pass variant — the real name is `select`",
    "slice_copy": "a functionalisation-pass variant — the real name is slicing",
    "split_copy": "a functionalisation-pass variant — the real name is `split`",
    "split_with_sizes_copy": "a functionalisation-pass variant — the real name is `split`",
    "squeeze_copy": "a functionalisation-pass variant — the real name is `squeeze`",
    "t_copy": "a functionalisation-pass variant — the real name is `t`",
    "transpose_copy": "a functionalisation-pass variant — the real name is `transpose`",
    "unbind_copy": "a functionalisation-pass variant — the real name is `unbind`",
    "unfold_copy": "a functionalisation-pass variant — the real name is `unfold`",
    "unsqueeze_copy": "a functionalisation-pass variant — the real name is `unsqueeze`",
    "view_copy": "a functionalisation-pass variant — the real name is `view`",
    "view_as_real_copy": "a functionalisation-pass variant (complex)",
    "view_as_complex_copy": "a functionalisation-pass variant (complex)",
    "unsafe_chunk": "a functionalisation-pass variant — the real name is `chunk`",
    "unsafe_split": "a functionalisation-pass variant — the real name is `split`",
    "unsafe_split_with_sizes": "a functionalisation-pass variant — the real name is `split`",
    "slice_inverse": "a functionalisation-pass variant — an internal op for undoing",

    # ATen's lower entry points. A name the user calls sits above each
    # (`layer_norm` above `native_layer_norm`).
    "native_*": "a lower ATen entry point — a name to call sits above it",
    "batch_norm_*": "a lower ATen entry point (statistics and backward pieces)",
    "affine_grid_generator": "a lower ATen entry point — `affine_grid` sits above it",
    "grid_sampler_2d": "a lower ATen entry point — `grid_sample` sits above it",
    "grid_sampler_3d": "a lower ATen entry point — `grid_sample` sits above it",
    "norm_except_dim": "a lower ATen entry point — `weight_norm` uses it",
    "embedding_renorm_": "a lower ATen entry point — `Embedding(max_norm=)` uses it",
    "convolution": "a lower ATen entry point — `conv1d/2d/3d` sit above it",
    "init_num_threads": "runtime internals",
    "thread_safe_generator": "runtime internals",
    "get_file_path": "runtime internals",
    "sym_fresh_size": "symbolic-size internals",
}

# **Real API, and declined.** What is written here is a judgement, not a defect.
SKIPPED = {
    # Devices and precision.
    #
    # **This read "the GPU has f32 and nothing else", which is not a fact about hardware.**
    # WebGPU has `shader-f16` and so does this machine (Apple metal-3, measured). What is
    # absent is not the hardware but **our decision that our shaders use f32 only.** Writing
    # cannot and will-not in the same sentence stops the next person reviewing it.
    #
    # **Why half precision is declined** (decided after measuring):
    #
    # - `shader-f16` is an **optional feature.** Machines that have it and machines that do
    #   not diverge, so opening it makes the same program answer differently per machine —
    #   and the golden cases lose an answer to freeze. `float64` is a different place: it is
    #   **always** absent, so the divergence itself was frozen as the answer.
    # - Imitating it in float32 does not work. Half precision's lesson is **overflow and
    #   rounding** (65600 → inf, 1+0.0005 → 1), and counted in f32 neither happens.
    # - `bfloat16` has **no substrate** in numpy or in WGSL. Imitated in software it becomes
    #   a thing with f32's speed and worse precision — there is nothing to learn from it.
    #
    # **Chosen in the reversible direction.** Adding it later is an addition; adding and
    # then removing it breaks someone else's code. What has to be measured again when it is
    # added is the first line above — whether `shader-f16` has become effectively
    # universal.
    "autocast": "mixed precision — our shaders use f32 only (a decision, not the hardware)",
    "autocast_*": "mixed precision — as above",
    "clear_autocast_cache": "mixed precision — as above",
    "get_autocast_*": "mixed precision — as above",
    "set_autocast_*": "mixed precision — as above",
    "is_autocast_*": "mixed precision — as above",
    "GradScaler": "mixed precision's loss scaling — as above",
    "get_num_threads": "inside one tab there is no thread count to choose",
    "set_num_threads": "inside one tab there is no thread count to choose",
    "get_num_interop_threads": "it is inside one tab",
    "set_num_interop_threads": "it is inside one tab",
    "Stream": "device streams — there is one",
    "Event": "device events — there is one stream to measure",
    "get_device_module": "there is one device",
    "get_default_device": "there is one device",
    "set_default_device": "there is one device",
    "is_vulkan_available": "not a backend a browser chooses",
    "cudnn_is_acceptable": "there is no cuDNN",
    "AcceleratorError": "there is one accelerator",
    "OutOfMemoryError": "we do not separate out device memory errors",
    "DataParallel": "for several devices — this is one tab",
    "SyncBatchNorm": "for distributed training — this is inside one tab",
    "DistributedSampler": "for distributed training — this is inside one tab",

    # Vendor kernels. They mean something only where that hardware is.
    "cudnn_*": "an NVIDIA kernel — not in a browser",
    "miopen_*": "an AMD kernel — not in a browser",
    "mkldnn_*": "an Intel kernel — not in a browser",
    "fbgemm_*": "a quantised GEMM kernel — it means something only on real hardware",
    "q_per_channel_axis": "a quantised tensor's scale — that dtype does not exist here",
    "q_per_channel_scales": "a quantised tensor's scale — that dtype does not exist here",
    "q_per_channel_zero_points": "a quantised tensor's scale — that dtype does not exist here",
    "q_scale": "a quantised tensor's scale — that dtype does not exist here",
    "q_zero_point": "a quantised tensor's scale — that dtype does not exist here",
    "qscheme": "a quantisation scheme — that dtype does not exist here",
    "int_repr": "a quantised tensor's integer representation — that dtype does not exist here",
    "choose_qparams_optimized": "choosing a quantisation scale — it means something only on real hardware",
    "fused_moving_avg_obs_fake_quant": "a quantisation observer — it means something only on real hardware",

    # Sparse tensors. `DELIBERATE`'s "sparse" matches namespace names only.
    "smm": "sparse matrix products — outside the curriculum",
    "spmm": "sparse matrix products — outside the curriculum",
    "hsmm": "sparse matrix products — outside the curriculum",
    "dsmm": "sparse matrix products — outside the curriculum",
    "hspmm": "sparse matrix products — outside the curriculum",
    "saddmm": "sparse matrix products — outside the curriculum",
    "sspaddmm": "sparse matrix products — outside the curriculum",
    "segment_reduce": "for sparse and ragged bundles — outside the curriculum",
    "resize_as_sparse_": "sparse tensors only — as above",
    # **By its name it looks like `nn.ParameterDict`'s counterpart, and it is not.**
    # Measured, its constructor takes a single `torch._C.ScriptModule` and it does not live
    # under `nn` — it is TorchScript internals. Built as a dictionary, it would hand over
    # something with the same name and a different nature.
    "BufferDict": "TorchScript internals — the constructor takes a ScriptModule",
    # **torch itself cannot make one.** Called with a float tensor it raises
    # `NotImplementedError` (measured) — it needs a quantised dtype to stand on. Its absence
    # here is not us being stingy.
    "empty_quantized": "an empty quantised tensor — torch cannot make one without that dtype either",

    # Places with one thing to choose. **Keeping the name makes it look choosable.**
    #
    # `layout` and `memory_format` are types, and each has exactly one value — `strided`
    # and `contiguous_format`. An argument choosing among one value teaches that choosing
    # changes something, and here it does not.
    "layout": "`strided` is the only layout to choose",
    "memory_format": "`contiguous_format` is the only layout to choose",
    "prepare_multiprocessing_environment": "inside one tab there are no processes",
    # **torch itself refuses a dense gradient** — the measured wording is
    # "SparseAdam does not support dense gradients, please consider Adam instead". With no
    # sparse tensors here, there is no input this optimizer could accept. Built, it would be
    # a thing that only ever refuses, which is worse than counting it as present.
    "SparseAdam": "sparse gradients only — torch refuses a dense one too",

    # Complex. Our dtypes are three: float32, int64 and bool.
    "imag": "there is no complex dtype here",
    "view_as_real": "there is no complex dtype here",

    # Symbolic sizes and graph capture. The same reason as `DELIBERATE`'s compile and
    # export, but the names do not match there.
    "Sym*": "symbolic sizes — for graph capture, and they do not sit on wasm",
    "sym_*": "symbolic sizes — as above",
    "cond": "a control-flow capture op — it is for export",
    "while_loop": "a control-flow capture op — it is for export",
    "vmap": "functorch's auto-batching — outside the curriculum",

    # Exchange with the outside. Inside a browser there is nobody to hand to.
    "from_dlpack": "DLPack exchange — there is nobody to exchange with inside a browser",
    "to_dlpack": "DLPack exchange — as above",
    "from_file": "file mapping — a browser has no such file layer",

    # Debug switches. We have neither nondeterminism to turn on nor an anomaly detector.
    "use_deterministic_algorithms": "there is no nondeterministic kernel to choose",
    "are_deterministic_algorithms_enabled": "as above",
    "is_deterministic_algorithms_warn_only_enabled": "as above",
    "get_deterministic_debug_mode": "as above",
    "set_deterministic_debug_mode": "as above",
    "is_anomaly_enabled": "there is no anomaly detector",
    "set_anomaly_enabled": "there is no anomaly detector",
    "is_anomaly_check_nan_enabled": "there is no anomaly detector",
    "set_warn_always": "a warning-policy switch — outside the curriculum",
    "is_warn_always_enabled": "a warning-policy switch — outside the curriculum",
    "set_flush_denormal": "a subnormal-handling switch — WGSL does not offer it",
    "get_float32_matmul_precision": "a TF32 switch — that hardware is not here",
    "set_float32_matmul_precision": "a TF32 switch — that hardware is not here",

    # Things torch deprecated or folded itself.
    "symeig": "torch deprecated it — `eigh` replaced it",
    "frobenius_norm": "torch deprecated it — `linalg.matrix_norm` replaced it",
    "nuclear_norm": "torch deprecated it — it is `linalg.matrix_norm(ord='nuc')`",
    "range": "torch deprecated it — use `arange`",
    "Container": "torch deprecated it — `Sequential` replaced it",
    "NLLLoss2d": "torch deprecated it — `NLLLoss` takes that shape",
    "CrossMapLRN2d": "an old layer — `LocalResponseNorm` stands there now",
    "IterDataPipe": "torch's folded DataPipes experiment",
    "MapDataPipe": "torch's folded DataPipes experiment",
    "DFIterDataPipe": "torch's folded DataPipes experiment",
    "DataChunk": "torch's folded DataPipes experiment",
    "guaranteed_datapipes_determinism": "torch's folded DataPipes experiment",
    "non_deterministic": "torch's folded DataPipes experiment",
    "runtime_validation": "torch's folded DataPipes experiment",
    "runtime_validation_disabled": "torch's folded DataPipes experiment",
    "Future": "a TorchScript async promise — this is one tab",
    "conv_tbc": "an old-layout (time-batch-channel) convolution — no code calls it",

    # **The raw ATen losses at the top level.** They share `F`'s names and **are not the
    # same functions** — their default reduction is `none` and `reduction` is an integer
    # rather than a string (0, 1, 2). `torch.kl_div(a, b)` gives `[2,2]` and
    # `F.kl_div(a, b)` gives a scalar.
    #
    # So aliasing `F`'s version at the top level **diverges at the shape.** What the
    # tutorials call is `F`'s, and this side is only ATen having put a name out. Measured:
    # the only two that are the same function are `pairwise_distance` and `pdist`, and both
    # are offered. **Written with the namespace attached** — as bare names they would
    # swallow `F`'s identically named ones, and `F.kl_div` we built while `F.ctc_loss` is a
    # real gap.
    "torch.binary_cross_entropy_with_logits":
        "the top-level one is the raw ATen op — its default reduction is none and its argument is an integer",
    "torch.cosine_embedding_loss": "the top-level one is the raw ATen op — its signature differs from F's",
    "torch.hinge_embedding_loss": "the top-level one is the raw ATen op — its signature differs from F's",
    "torch.kl_div": "the top-level one is the raw ATen op — its signature differs from F's",
    "torch.margin_ranking_loss": "the top-level one is the raw ATen op — its signature differs from F's",
    "torch.poisson_nll_loss": "the top-level one is the raw ATen op — it has no defaults at all",
    "torch.triplet_margin_loss": "the top-level one is the raw ATen op — its signature differs from F's",

    # Not settled yet.
    "LinearCrossEntropyLoss": "newly arrived in torch — looked at once it settles",
    "LinearCrossEntropyOptions": "as above",
    "linear_cross_entropy": "the functional counterpart of the layer above — looked at when they settle together",
    "Muon": "an optimizer newly arrived in torch — looked at once it settles",
    "grouped_mm": "fp8 and grouped GEMM — that hardware is not here",
    "scaled_grouped_mm": "fp8 and grouped GEMM — that hardware is not here",
    "scaled_mm": "fp8 GEMM — that hardware is not here",

    # torchvision. **Only what is declined for good is written here** — everything else
    # absent from `transforms` is the to-do list, and it should read as one.
    #
    # The two PIL names are the load-bearing pair. This library's stand-in for a PIL image
    # is an (H,W,C) array, and every transform says so at its door (`_require_hwc`).
    "transforms.ToPILImage": "there is no PIL here — an (H,W,C) array stands in for a PIL "
                             "image, and handing back a real one would mean depending on "
                             "Pillow to return the thing this library does without",
    "transforms.PILToTensor": "it takes a PIL image, and nothing here produces one — "
                              "`ToTensor` is the same journey from the array that does "
                              "stand in",
    # **Measured rather than judged**: `torch.uint8` is an `_AbsentDtype` in the core, so
    # the conversion this class exists for has no destination.
    "transforms.ConvertImageDtype": "uint8 has no storage in this subset, so float-to-uint8 "
                                    "with its x255 has nothing to convert into — and "
                                    "`ToTensor` already does the divide the other way",
    # The same three, under the names `transforms.functional` gives them. **Written out
    # rather than wildcarded**: `*_pil_image` would also catch a name torchvision has not
    # invented yet, and a row matching something nobody has read is how a reason ends up
    # attached to the wrong thing.
    # **Moved out of `NOT_API` after being read.** It was called a
    # functionalisation-pass variant of `narrow`, and it is not one: torch documents
    # it, with an example, and it does something `narrow` does not — it copies where
    # `narrow` gives a view. It exists because sparse tensors have no view-narrow, and
    # sparse is declined in the core, which is a reason and a different one.
    #
    # The move matters beyond this name. `NOT_API` comes out of the denominator and
    # `SKIPPED` does not, so a wrong reason there is a wrong reason **that also
    # improves the number** — which is why that bin is now frozen by size below.
    "narrow_copy": "`narrow` and then a copy. torch has it because sparse tensors have "
                   "no view-narrow, and sparse is declined in the core, so what is "
                   "left is `narrow(...).clone()`",

    "transforms.functional.to_pil_image": "there is no PIL here — as `ToPILImage`",
    "transforms.functional.pil_to_tensor": "it takes a PIL image and nothing here makes "
                                           "one — as `PILToTensor`",
    "transforms.functional.convert_image_dtype": "uint8 has no storage in this subset — "
                                                 "as `ConvertImageDtype`",

    # `transforms.v2`. **Only what is declined for the same reason `ops` is** — v2's
    # whole addition over v1 is that a transform carries boxes, masks, keypoints and
    # video alongside the image, and that pays off with a detector. There is no
    # detector in the catalogue, so the type system that exists to keep boxes in step
    # with the picture has nothing to keep in step.
    #
    # Everything else absent from v2 is the to-do list and reads as one: 38 of its 72
    # names are the transforms this library already has, one namespace over.
    "transforms.v2.ClampBoundingBoxes": "boxes travelling with the picture — the point "
                                        "of v2's type system, and it pays off with a "
                                        "detector, which the catalogue has none of",
    "transforms.v2.ClampKeyPoints": "as above, for keypoints",
    "transforms.v2.SanitizeBoundingBoxes": "as above",
    "transforms.v2.SanitizeKeyPoints": "as above",
    "transforms.v2.ConvertBoundingBoxFormat": "as above",
    "transforms.v2.SetClampingMode": "as above",
    "transforms.v2.RandomIoUCrop": "a detection augmentation — it crops by how much of "
                                   "a box survives, so it is boxes or it is nothing",
    "transforms.v2.get_bounding_boxes": "reads the boxes out of a v2 sample — as above",
    "transforms.v2.get_keypoints": "as above",
    "transforms.v2.UniformTemporalSubsample": "video. There is no video anywhere in "
                                              "this project and a tutorial's first ten "
                                              "lines do not open one",
    "transforms.v2.ToPILImage": "there is no PIL here — as in v1",
    "transforms.v2.PILToTensor": "it takes a PIL image and nothing here makes one — "
                                 "as in v1",
    "transforms.v2.ConvertImageDtype": "uint8 has no storage in this subset — as in v1",
    "transforms.v2.Transform": "**the base class every v2 transform inherits**, and its "
                               "body is the tv_tensor dispatch: flatten the sample, "
                               "decide per leaf whether this leaf gets transformed, "
                               "reassemble. Here the transforms inherit v1's classes "
                               "instead and take a picture, so there is no sample to "
                               "walk. Present as a name it would be an empty class that "
                               "subclassing gets nothing from",
    "transforms.v2.query_size": "reads `(H, W)` out of **a sample** — a dict or tuple "
                                "of tv_tensors where the picture has to be found first. "
                                "On a bare array that is `.shape`, so what the function "
                                "is for is the part that is missing",
    "transforms.v2.query_chw": "as above, for `(C, H, W)`",
    "transforms.v2.has_any": "asks which tv_tensor types are in a sample — as above",
    "transforms.v2.has_all": "as above",
    "transforms.v2.check_type": "as above",
    "transforms.v2.JPEG": "it encodes and decodes JPEG. numpy has no codec, and adding "
                          "one is the dependency this library does without — the same "
                          "answer PIL gets",


    # `datasets`. **The line is a codec, not a network.** Seven of these are built
    # here; of the rest, most read JPEG or PNG, and numpy has no decoder for either.
    # Adding one is the dependency this library does without — the same answer PIL
    # gets in `transforms`, arriving from the other side.
    #
    # Eight names are **absent with no reason on purpose**; see this file's docstring.

    "datasets.Caltech101": "as above — a codec",
    "datasets.Caltech256": "as above — a codec",
    "datasets.CelebA": "as above — a codec",
    "datasets.Cityscapes": "its pictures are PNG and `_png_read` handles those now — the wall left is the dataset rather than the format: thirty splits crossed with five target types, polygon annotations in JSON, and a 60GB archive behind a login. **The codec was never the expensive half of this one**, which the old row (`as above — a codec`) could not say because it pointed at a sentence about JPEG",
    "datasets.CocoCaptions": "as above — a codec",
    "datasets.CocoDetection": "as above — a codec",
    "datasets.Country211": "as above — a codec",
    "datasets.DTD": "as above — a codec",
    "datasets.EuroSAT": "as above — a codec",
    "datasets.FGVCAircraft": "as above — a codec",
    "datasets.Flickr30k": "as above — a codec",
    "datasets.Flickr8k": "as above — a codec",
    "datasets.Flowers102": "its pictures are JPEG, which is the codec wall. Its labels are a `.mat` and that half is no longer a reason — `_mat_read` in borchvision handles those now, which is what let `SVHN` in. This row used to read *as `SVHN` for the labels*, and pointed at a refusal that has since been lifted",
    "datasets.Food101": "as above — a codec",
    "datasets.HMDB51": "video. There is no video anywhere in this project and a tutorial's first ten lines do not open one",
    "datasets.INaturalist": "as above — a codec",
    "datasets.ImageNet": "as above — a codec",
    "datasets.Imagenette": "as above — a codec",
    "datasets.Kinetics": "as above",
    "datasets.Kitti": "as above — a codec",
    "datasets.LFWPairs": "as above — a codec",
    "datasets.LFWPeople": "as above — a codec",
    "datasets.LSUN": "the pictures live in an LMDB database. That is a second dependency before the codec is even reached",
    "datasets.LSUNClass": "as above",
    "datasets.OxfordIIITPet": "as above — a codec",
    "datasets.PCAM": "the whole set is one HDF5 file, so it is `h5py` rather than a codec. Same answer: the dependency",
    "datasets.PhotoTour": "as above — a codec",
    "datasets.Places365": "as above — a codec",
    "datasets.SBDataset": "its pictures are JPEG, which is the codec wall. Its `.mat` annotations are not — `_mat_read` handles those, and this row named them as a second reason when there was only ever one",
    "datasets.SBU": "as above — a codec",
    "datasets.SUN397": "as above — a codec",
    "datasets.StanfordCars": "as above — a codec",
    "datasets.UCF101": "as above",
    "datasets.VOCDetection": "as above — a codec",
    "datasets.VOCSegmentation": "as above — a codec",
    "datasets.WIDERFace": "as above — a codec",

    # The stereo and optical-flow sets are one kind and are declined as one. Each is a
    # **pair** of pictures plus a disparity or flow field in `.pfm` or `.flo`, so they
    # need the codec and then a second format on top of it, and what they exist to
    # train has no model here either.
    "datasets.CREStereo": "stereo or optical flow — paired pictures plus a "
                            "disparity field, so a codec and then another format",
    "datasets.CarlaStereo": "as above",
    "datasets.ETH3DStereo": "as above",
    "datasets.FallingThingsStereo": "as above",
    "datasets.FlyingChairs": "as above",
    "datasets.FlyingThings3D": "as above",
    "datasets.HD1K": "as above",
    "datasets.InStereo2k": "as above",
    "datasets.Kitti2012Stereo": "as above",
    "datasets.Kitti2015Stereo": "as above",
    "datasets.KittiFlow": "as above",
    "datasets.Middlebury2014Stereo": "as above",
    "datasets.SceneFlowStereo": "as above",
    "datasets.Sintel": "as above",
    "datasets.SintelStereo": "as above",

    # `transforms.v2.functional`. **114 of its 165 names are one operation counted
    # five times**, and this is the row that says so — the first namespaced wildcard
    # this table has been able to write. `affine_image`, `affine_mask`,
    # `affine_bounding_boxes`, `affine_keypoints` and `affine_video` are v2's dispatch
    # kernels: the type of what arrives decides which body runs, and that type system
    # is the half of v2 declined one namespace up.
    #
    # Written flat as `"*_image"` these rows would also swallow v1's `to_pil_image`
    # and attach a sentence about dispatch to a name that has nothing to do with it.
    # That was the reason this namespace was **off the list entirely** — described in
    # a paragraph rather than counted — and a namespace off the list is a namespace
    # nobody counts, which is how `transforms` stayed invisible for the whole life of
    # this library. The paragraph is gone; the number is here.
    "transforms.v2.functional.*_image": "a dispatch kernel — v2 routes by the type of "
                                        "what arrives, and the image body is the "
                                        "public name one namespace over",
    "transforms.v2.functional.*_video": "as above, and there is no video in this "
                                        "project",
    "transforms.v2.functional.*_mask": "as above, for segmentation masks — a tv_tensor "
                                       "type, and the type system is declined in `v2`",
    "transforms.v2.functional.*_bounding_boxes": "as above, for boxes",
    "transforms.v2.functional.*_keypoints": "as above, for keypoints",

    # The eight left over are not kernels and not v1's. Each carries its own reason.
    "transforms.v2.functional.convert_bounding_box_format": "boxes travelling with the "
                                                            "picture — as in `v2`",
    "transforms.v2.functional.convert_image_dtype": "uint8 has no storage in this "
                                                    "subset — as in v1",
    "transforms.v2.functional.pil_to_tensor": "it takes a PIL image and nothing here "
                                              "makes one — as in v1",
    "transforms.v2.functional.jpeg": "it encodes and decodes JPEG. numpy has no codec",
    "transforms.v2.functional.uniform_temporal_subsample": "video",
    "transforms.v2.functional.get_num_frames": "video — it answers how many frames a "
                                               "clip has, and nothing here is a clip",
    "transforms.v2.functional.register_kernel": "**the dispatch registry itself.** It "
                                                "attaches a body to a (functional, "
                                                "tv_tensor type) pair, and there are "
                                                "no tv_tensor types here to attach to",
    "transforms.v2.functional.is_pure_tensor": "asks whether a tensor is a plain one "
                                               "rather than a tv_tensor subclass. Here "
                                               "every tensor is plain, so it could "
                                               "only ever answer True — **a question "
                                               "with one answer is not a question**, "
                                               "and present it would read as support "
                                               "for the type system it is testing for",

    # `ops`. **The eleven that are here are box geometry and the twenty-eight that are
    # not need a model.** The old one-line reason covered all thirty-nine and
    # justified twenty-eight of them, which is why it is written out by kind now.
    #
    # The layers first — every one of these is a piece of a detector or a backbone,
    # and the catalogue (`bimm`) has one classifier in it.
    "ops.Conv2dNormActivation": "a backbone's building block — it needs a model to be "
                                "a block of",
    "ops.Conv3dNormActivation": "as above, and 3-D convolution is declined in the core",
    "ops.DeformConv2d": "deformable convolution — a detector's, and there is no "
                        "detector in the catalogue",
    "ops.deform_conv2d": "as above",
    "ops.FeaturePyramidNetwork": "the detector's neck. Nothing feeds it here",
    "ops.MultiScaleRoIAlign": "as above",
    "ops.RoIAlign": "it crops from a feature map. A feature map comes from a model",
    "ops.RoIPool": "as above",
    "ops.PSRoIAlign": "as above",
    "ops.PSRoIPool": "as above",
    "ops.roi_align": "as above",
    "ops.roi_pool": "as above",
    "ops.ps_roi_align": "as above",
    "ops.ps_roi_pool": "as above",
    "ops.FrozenBatchNorm2d": "batch norm with the statistics frozen — it exists for "
                             "fine-tuning a pre-trained detector",
    "ops.DropBlock2d": "structured dropout for convolutional backbones",
    "ops.DropBlock3d": "as above",
    "ops.drop_block2d": "as above",
    "ops.drop_block3d": "as above",
    "ops.StochasticDepth": "it drops whole residual blocks — it needs blocks",
    "ops.stochastic_depth": "as above",
    "ops.SqueezeExcitation": "a backbone's attention block — as above",
    "ops.MLP": "a stack of linear layers with dropout. `nn.Sequential` is that, and "
               "torchvision's version exists to be a piece of its own models",
    "ops.Permute": "an `nn.Module` wrapper around `permute`, so that it can sit in a "
                   "`Sequential`. The core has the function",
    # The losses need predictions, which need a model.
    "ops.sigmoid_focal_loss": "a detection loss — it takes a model's predictions",
    "ops.complete_box_iou_loss": "as above",
    "ops.distance_box_iou_loss": "as above",
    "ops.generalized_box_iou_loss": "as above. **The IoU itself is here** — what is "
                                    "absent is the loss around it, which is where the "
                                    "predictions come in",
}

# **How big the `NOT_API` bin is allowed to be, per namespace.**
#
# Every other absence stays in the denominator. This one does not — a name called "not
# API" is subtracted before the percentage is taken, which is right when the call is
# right and is **the one place in this file where a wrong call makes the number look
# better.** The comment beside the denominator already says this about `SKIPPED` and
# then grants `NOT_API` the exemption; nothing was watching the exemption.
#
# So the bin has a size and the size is written down. Growing it is an edit here, which
# makes it a decision somebody made rather than a number that drifted — the same
# arrangement `tests/test_korean_ceiling.py` uses, and for the same reason.
#
# Measured 2026-08-22. The contents were read at the same time: of 203 names, four carry
# an `Example::` in torch's own docstring, and three of those four are fairly called
# internals (deprecated, an optimizer tool, a runtime one). The fourth was `narrow_copy`
# and it has been moved to `SKIPPED`.
NOT_API_SIZE = {
    "torch": 180,
    "Tensor": 4,
    "nn": 1,
    "nn.functional": 10,
    "optim": 1,
    "optim.lr_scheduler": 2,
    "utils.data": 2,
    "transforms.functional": 1,
}

# The namespaces looked at. (display name, torch's side, ours)
class _Absent:
    """**A namespace we do not have, standing where it would be.**

    `_spaces()` drops a `None`, so declaring an absent namespace that way counts it as
    nothing to review — which is exactly the move that hid `transforms` this morning.
    An empty object counts it as everything to review, and that is the honest number.

    This class was here once for `transforms.functional`, was deleted when that
    namespace was built, came back for `transforms.v2`, and now stands empty again.
    Worth keeping rather than tidying: it has been needed twice and deleted twice, so
    what it represents recurs. Deleting it costs nothing to write back and costs the
    next person the reasoning above, which is the part that was expensive.
    """


def _spaces():
    got = [("torch", torch, borch),
           ("Tensor", torch.Tensor, borch.Tensor),
           ("nn", torch.nn, borch.nn),
           ("nn.functional", torch.nn.functional, borch.nn.functional),
           ("optim", torch.optim, borch.optim),
           ("optim.lr_scheduler", torch.optim.lr_scheduler, borch.optim.lr_scheduler),
           ("linalg", torch.linalg, borch.linalg),
           ("utils.data", torch.utils.data, borch.utils.data)]
    if torchvision is not None:
        got += [("transforms", torchvision.transforms, borchvision.transforms),
                ("transforms.functional", torchvision.transforms.functional,
                 borchvision.transforms.functional),
                ("transforms.v2", torchvision.transforms.v2, borchvision.transforms.v2),
                ("transforms.v2.functional", torchvision.transforms.v2.functional,
                 borchvision.transforms.v2.functional),
                ("ops", torchvision.ops, borchvision.ops),
                ("datasets", torchvision.datasets, borchvision.datasets)]
    return [(name, a, b) for name, a, b in got if b is not None]


def _public(obj):
    """Public names only — narrowed to **what can be called.**

    The denominator was fixed three times, and the path is left here.

    1. `dir()` as it stands → a torch surface of 1,013. `Callable` and `Optional` (typing
       imports) and `AnyType` and `ArgumentSpec` (internal types of the C extension) were
       counted as API.
    2. `__all__` → worse at the top level. torch's top-level `__all__` is not a hand-picked
       public list but something generated that carries internal C types too, giving 905,
       and the overlap fell to 3. **A false denominator makes a false percentage.**
    3. Now — only what is callable (functions and classes), and names imported from other
       modules are excluded.

    It is still not perfect. This number is not one to boast "what percentage of the
    surface" with; it is one for **pointing at which places are empty.**
    """
    out = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        thing = getattr(obj, name, None)
        if thing is None or not callable(thing):
            continue
        # A name imported from another module (typing and the like) is not that namespace's API.
        home = getattr(thing, "__module__", "") or ""
        if home and not (home.startswith("torch") or home.startswith("borch")):
            continue
        out.add(name)
    return out


def _props(obj):
    """**Public names that cannot be called** — attributes and dtype constants.

    `_public` counts only what is `callable`. So names **used without parentheses**, such as
    `x.real`, `x.mT` and `x.is_cuda`, appear in neither the numerator nor the denominator.
    The "100% of the Tensor surface" produced in that state is 100% *of the methods*, not of
    the surface.

    They are counted apart because the two questions differ. A missing method raises
    `AttributeError`, and a missing attribute **raises the same exception with a different
    place to fix** — the first wants a function written and the second a `@property`.
    Merged into one number, which of the two it is cannot be seen.
    """
    import types as _types

    out = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        thing = getattr(obj, name, None)
        if callable(thing):
            continue
        # **Modules are not what this question is about.** `torch.math` and `torch.sys` are
        # just torch not hiding its own imports, and a real sub-namespace such as
        # `torch.nn.init` is covered by the per-namespace table above. Unfiltered, the noise
        # buries the signal.
        if isinstance(thing, _types.ModuleType):
            continue
        out.add(name)
    return out


def _look(table, name, full=None):
    """Finds the reason in the tables. In the order **namespaced name → name → wildcard.**

    The order was a problem twice.

    With `"*Tensor"` written ahead of `"Tensor"`, `nn.functional.Tensor` was explained as
    "a per-type Tensor class — torch deprecated it". The category was right and **the reason
    was false** — it is simply an imported label.

    Then a bare `"Optimizer"` swallowed `torch.optim.Optimizer` as well. That one is real
    API and we had already built it — `test_gap.py` caught the contradiction. So names can
    now be written with their namespace attached.

    That shape is the worst thing in this table. Something not counted draws the eye;
    something **not counted for a false reason** does not, and the next person to read it
    believes the falsehood.
    """
    if full and full in table:
        return table[full]
    if name in table:
        return table[name]
    for key, reason in table.items():
        if "." in key:
            # **A namespaced wildcard matches inside its namespace and nowhere else.**
            # Dotted keys used to be skipped here entirely, and that is what kept
            # `transforms.v2.functional` off the list: 114 of its 165 public names are
            # `<operation>_<type>` dispatch kernels, one reason covers all of them, and
            # the only way to write that reason was `"*_image"` — which is not
            # namespaced, so it would have swallowed v1's `to_pil_image` too and
            # attached a sentence about dispatch to a name that has nothing to do with
            # it. A namespace off the list is a namespace nobody counts, and that is
            # how `transforms` stayed invisible for the whole life of this library.
            head, _, leaf = key.rpartition(".")
            if "*" not in leaf or not full or not full.startswith(head + "."):
                continue
            # **Matched against the leaf of `full`, not against `name`.** The two are
            # the same when `_why` calls this, and they are not when
            # `test_no_table_entry_matches_nothing` does: that check hands the same
            # namespaced string in both positions, deliberately, because it is asking
            # whether a row matches anything at all. Reading `name` here made every one
            # of these rows look dead — six reasons about nothing, and the check was
            # right to say so about what it was actually being shown.
            rest = full[len(head) + 1:]
            if "." in rest:                     # a name further in, not this leaf
                continue
            if _leaf_match(leaf, rest):
                return reason
            continue
        if key.endswith("*") and name.startswith(key[:-1]):
            return reason
        if key.startswith("*") and name.endswith(key[1:]):
            return reason
    return None


def _leaf_match(pattern, name):
    """`*` at one end only, as the flat wildcards have always been.

    Deliberately not `fnmatch`: a pattern language nobody asked for is a pattern
    language that will eventually match something nobody meant, and this table's whole
    job is to not do that.
    """
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    if pattern.startswith("*"):
        return name.endswith(pattern[1:])
    return False


def _why(space, name):
    """Why it is all right for this to be absent. None where there is no reason — **and that
    is exactly what wants reviewing.**

    It returns `(category, reason)`. Without the category, "not API, so not counted" and
    "API, and declined" collapse into one number, and to a reader those say different
    things.

    **The top level carries its namespace too** (`torch.kl_div`). Without it, a judgement
    meant for the top level alone swallowed `nn.functional`'s identically named one — that
    `kl_div` we built, and `ctc_loss` there is still a real gap.
    """
    full = f"{space}.{name}"
    for key, reason in DELIBERATE.items():
        if full.startswith(key) or name.startswith(key):
            return ("namespace", reason)
    found = _look(NOT_API, name, full)
    if found:
        return ("not API", found)
    found = _look(SKIPPED, name, full)
    if found:
        return ("declined", found)
    return None


def props_report():
    """Counts **the names used without parentheses**, separately.

    This file's main reckoning counts only what is `callable`. So when the coverage table said "100%
    of the Tensor surface", that was 100% **of the methods**, and names such as `x.real`,
    `x.mT` and `x.is_cuda` were in neither the numerator nor the denominator — nobody even
    knew they were missing.

    No reasons are attached to this table — attaching them would make it "the place where
    the number gets made to look good" again. Here it only **shows what is absent**, and the
    judgement is written by a person into the two tables above.
    """
    for space, theirs, ours in _spaces():
        a, b = _props(theirs), _props(ours)
        # What the two tables above already explain is not asked again here — namespaces
        # declined whole, things that are not API, things declined. What remains is the
        # question.
        gap = sorted(n for n in a - b if _why(space, n) is None)
        print(f"\n{space} — {len(a & b)} of {len(a)} non-callable public names present")
        if gap:
            print(f"  absent with no reason, {len(gap)}: " + ", ".join(gap))
    return 0


def main(argv):
    show = None
    if "--show" in argv:
        show = argv[argv.index("--show") + 1]
    if "--props" in argv:
        return props_report()
    extra = "--extra" in argv

    total_missing = total_have = 0
    for space, theirs, ours in _spaces():
        # `--show nn` looks at that namespace alone. Printing everything buries the line you came for.
        if show not in (None, "all") and space != show:
            continue
        a, b = _public(theirs), _public(ours)
        gap = sorted(a - b) if not extra else sorted(b - a)
        judged = [(n, _why(space, n)) for n in gap]
        unexplained = [n for n, why in judged if why is None]

        if extra:
            print(f"\n{space} — {len(gap)} we have that torch does not")
            for n in gap:
                print(f"  + {n}")
            continue

        # **The only thing removed from the denominator is "the ones that are not API".**
        #
        # What was declined stays in. We chose it, so we carry the cost — removed, every
        # name written into the table raises the percentage, and then the table becomes the
        # place where the number gets made to look good. Left in, at least that temptation
        # is gone.
        not_api = {n for n, why in judged if why and why[0] == "not API"}
        skipped = [(n, why[1]) for n, why in judged if why and why[0] == "declined"]
        by_space = [n for n, why in judged if why and why[0] == "namespace"]
        api = a - not_api
        covered = len(api & b)

        total_have += covered
        total_missing += len(unexplained)
        print(f"\n{space} — {covered} of {len(api)} API names present "
              f"({covered * 100 // max(1, len(api))}%)")
        # **The same fraction with the exempt bin put back.** One number cannot show
        # both what was judged and what was excused from being judged, and the gap
        # between the two is exactly what calling something "not API" buys.
        if not_api:
            print(f"  {covered} of {len(a)} counting the not-API bin "
                  f"({covered * 100 // max(1, len(a))}%)")
        parts = [f"not API, uncounted {len(not_api)}"]
        if by_space:
            parts.append(f"namespace declined {len(by_space)}")
        if skipped:
            parts.append(f"declined {len(skipped)}")
        print("  " + " · ".join(parts) + f" · **wants reviewing {len(unexplained)}**")
        if show in (space, "all"):
            for n, reason in skipped:
                print(f"    – {n}: {reason}")
            for n in unexplained:
                print(f"    ? {n}")
        elif unexplained:
            head = ", ".join(unexplained[:8])
            more = f" … and {len(unexplained) - 8} more" if len(unexplained) > 8 else ""
            print(f"    {head}{more}")

    if not extra:
        print(f"\ntotal — {total_have} names in common, {total_missing} gaps with no reason")
        print("  `--show <namespace>` spreads that namespace out. `--show all` spreads everything.")
        print("  `–` is declined (a reason is attached); `?` wants reviewing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
