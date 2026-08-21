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
    "narrow_copy": "a functionalisation-pass variant — the real name is `narrow`",
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
    "transforms.functional.to_pil_image": "there is no PIL here — as `ToPILImage`",
    "transforms.functional.pil_to_tensor": "it takes a PIL image and nothing here makes "
                                           "one — as `PILToTensor`",
    "transforms.functional.convert_image_dtype": "uint8 has no storage in this subset — "
                                                 "as `ConvertImageDtype`",
}

# The namespaces looked at. (display name, torch's side, ours)
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
                 borchvision.transforms.functional)]
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
            continue
        if key.endswith("*") and name.startswith(key[:-1]):
            return reason
        if key.startswith("*") and name.endswith(key[1:]):
            return reason
    return None


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
