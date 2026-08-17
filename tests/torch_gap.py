"""진짜 torch 가 가진 이름 중 **우리에게 없는 것**을 센다.

    uv run --with numpy --with torch python tests/torch_gap.py
    uv run --with numpy --with torch python tests/torch_gap.py --extra   # 반대 방향
    uv run --with numpy --with torch python tests/torch_gap.py --show nn # 한 자리만 전부

`tests/conformance.py` 는 **있는 것이 맞는가**를 묻는다. 이 파일은 **무엇이 없는가**를
묻는다. 둘은 다른 질문이고, 앞의 것만 재면 100% 가 나오면서 표면이 얼마나 좁은지는
안 보인다 — 실제로 그렇게 읽힐 뻔했다.

## 없는 것이 다 같은 종류가 아니다

**넷**으로 갈린다. 이 구분이 이 파일의 요점이다.

- **애초에 API 가 아니다** — `AliasDb`·`ClassType`·`boolean_dispatch`·`ByteStorage`.
  이름이 밑줄 없이 공개돼 있을 뿐 부를 것이 아니다. TorchScript IR 내부, 디스패처
  내부, torch 가 스스로 버린 옛 클래스가 여기 든다.
- **일부러 거절한다** — `cuda`·`compile`·분산·양자화. 브라우저에 존재할 수 없거나,
  그것을 배우려면 브라우저를 벗어나야 하는 것들이다. 흉내 내면 교훈이 사라진다.
- **커리큘럼 밖이다** — `torch.fft`·`torch.sparse`·`torch.onnx`. 넣을 수는 있지만
  입문 튜토리얼이 안 부른다. 표면이 늘면 조용히 틀릴 자리가 는다.
- **그냥 없다** — 있어야 하는데 아무도 안 물어서 없는 것. **이 목록이 값어치다.**
  실제로 `torch.sum` 이 여기 있었다(메서드로만 있고 모듈 함수가 없었다).

앞의 셋은 판단이고 마지막 하나는 결함이다. 기계는 넷을 못 가르므로 아래 표에
앞의 셋을 적어 둔다 — **적히지 않은 것이 곧 검토 대상이다.**

## 이 표를 채우는 것은 위험한 일이다

**이름 하나를 적을 때마다 우리 비율이 올라간다.** 그러니 이 작업은 태생적으로
"수를 예쁘게 만들기" 로 미끄러진다. 막는 것은 하나뿐이다 — **각 줄이 사유를 갖고,
그 사유가 확인 가능해야 한다.** 사유를 못 적겠으면 그것은 빈자리다.

**기계로 가르려다 한 번 크게 틀렸다.** "`__module__` 이 `torch._C` 로 시작하면
내부 타입" 이라는 규칙이 그럴듯해 보였는데, `linalg` 의 `eig`·`eigvals`·`ldl_factor`
일곱과 `F.elu_`·`hardtanh_`·`leaky_relu_` 를 통째로 삼켰다. **전부 진짜 API 다** —
torch 의 함수 대부분이 C 에 산다. 그 규칙을 넣었으면 `linalg` 가 83% 에서 100% 로
뛰고 남은 진짜 빈자리 일곱이 목록에서 사라졌을 것이다. **우리에게 유리한 쪽으로
빗나가는 자동 규칙은 규칙이 없느니만 못하다.** 그래서 아래는 손으로 읽어 적었다.
"""

import sys

import torch

import borch

# ---------------------------------------------------------------- 표
#
# 세 표 다 **맞춤법**은 같다:
#   "이름"     — 그 이름 하나
#   "앞머리*"  — 그것으로 시작하는 이름들
#   "*꼬리"    — 그것으로 끝나는 이름들
#
# 부분 문자열은 안 쓴다. 앞의 판에서 `key in full` 로 걸러서 `"special"` 같은 키가
# 엉뚱한 이름을 삼킬 수 있었다 — 사고는 안 났지만 넓었다.

# **이름 공간 통째로.** 자리 이름의 앞머리로 건다.
DELIBERATE = {
    "cuda": "브라우저에 CUDA 가 없다. 흉내 내면 GPU 를 배우는 교훈이 사라진다",
    "mps": "같은 이유",
    "xpu": "같은 이유",
    "mtia": "같은 이유",
    "distributed": "한 탭 안이다. 분산을 배우려면 여러 기계로 나가야 한다",
    "compile": "TorchDynamo 는 CPython 바이트코드를 고쳐 쓴다. wasm 에 못 얹는다",
    "jit": "같은 이유",
    "export": "같은 이유",
    "fx": "같은 이유",
    "onnx": "내보내기는 배포의 일이고 여기는 문법 연습이다",
    "quantiz": "양자화는 실제 하드웨어 위에서만 뜻이 있다",
    "sparse": "커리큘럼 밖",
    "fft": "커리큘럼 밖",
    "special": "커리큘럼 밖",
    "futures": "커리큘럼 밖",
    "package": "커리굴럼 밖",
    "profiler": "커리큘럼 밖",
    "utils.tensorboard": "커리큘럼 밖",
    "backends": "고를 백엔드가 하나다",
    "multiprocessing": "브라우저에 프로세스가 없다",
}

# torch 가 형마다 하나씩 두었던 옛 클래스들. 목록이 닫혀 있어 여기 편다 —
# 꼬리 와일드카드로 잡으면 `torch.Tensor` 자체가 걸린다.
#
# **둘이 같은 목록이 아니다.** 복소수와 양자화 형에는 `Storage` 만 있고 `Tensor` 는
# 없다 — 하나로 적었더니 죽은 줄 일곱이 생겼고 `test_gap.py` 가 잡았다.
_OLD_STORAGE = ("BFloat16", "Bool", "Byte", "Char", "ComplexDouble", "ComplexFloat",
                "Double", "Float", "Half", "Int", "Long", "QInt32", "QInt8",
                "QUInt2x4", "QUInt4x2", "QUInt8", "Short")
_OLD_TENSOR = ("BFloat16", "Bool", "Byte", "Char", "Double", "Float", "Half",
               "Int", "Long", "Short")

# TorchScript 의 타입 이름들. 이것도 닫힌 목록이다.
_IR_TYPES = ("Any", "Await", "Bool", "Class", "Complex", "DeviceObj", "Dict", "Enum",
             "Float", "Future", "Int", "Interface", "List", "None", "Number",
             "Optional", "PyObject", "RRef", "StreamObj", "String", "SymBool",
             "SymInt", "Tensor", "Tuple", "Union")


# **애초에 API 가 아닌 이름.** 밑줄 없이 공개돼 있을 뿐 부를 것이 아니다.
NOT_API = {
    **{f"{d}Storage": "형별 Storage 클래스 — torch 가 폐기했다" for d in _OLD_STORAGE},
    **{f"{d}Tensor": "형별 Tensor 클래스 — torch 가 폐기했다 (FloatTensor …)"
       for d in _OLD_TENSOR},
    **{f"{t}Type": "TorchScript 타입 체계 (AnyType·ListType·TensorType …)"
       for t in _IR_TYPES},
    "Type": "TorchScript 타입 체계",
    # TorchScript 의 IR 과 타입 체계. `torch.jit` 이 쓰는 내부 물건이 최상위에도
    # 이름을 냈다. `DELIBERATE` 의 "jit" 은 자리 이름에만 걸려서 여기까지 안 온다.
    "AliasDb": "TorchScript IR 내부",
    "Argument": "TorchScript IR 내부",
    "ArgumentSpec": "TorchScript IR 내부",
    "CompleteArgumentSpec": "TorchScript IR 내부",
    "Block": "TorchScript IR 내부",
    "CallStack": "TorchScript IR 내부",
    "Capsule": "TorchScript IR 내부",
    "Code": "TorchScript IR 내부",
    "CompilationUnit": "TorchScript IR 내부",
    "ConcreteModuleType": "TorchScript IR 내부",
    "ConcreteModuleTypeBuilder": "TorchScript IR 내부",
    "DeepCopyMemoTable": "TorchScript IR 내부",
    "ErrorReport": "TorchScript IR 내부",
    "ExecutionPlan": "TorchScript IR 내부",
    "FileCheck": "TorchScript 테스트 도구",
    "FunctionSchema": "TorchScript IR 내부",
    "Gradient": "TorchScript IR 내부",
    "Graph": "TorchScript IR 내부",
    "GraphExecutorState": "TorchScript IR 내부",
    "IODescriptor": "TorchScript IR 내부",
    "InferredType": "TorchScript IR 내부",
    "JITException": "TorchScript IR 내부",
    "LiteScriptModule": "TorchScript IR 내부",
    "LockingLogger": "TorchScript IR 내부",
    "LoggerBase": "TorchScript IR 내부",
    "NoopLogger": "TorchScript IR 내부",
    "Node": "TorchScript IR 내부",
    "OperatorInfo": "TorchScript IR 내부",
    "PyTorchFileReader": "직렬화 내부",
    "PyTorchFileWriter": "직렬화 내부",
    "SerializationStorageContext": "직렬화 내부",
    "DeserializationStorageContext": "직렬화 내부",
    "StaticModule": "TorchScript IR 내부",
    "Tag": "TorchScript IR 내부",
    "TracingState": "TorchScript IR 내부",
    "Use": "TorchScript IR 내부",
    "Value": "TorchScript IR 내부",
    "FatalError": "TorchScript IR 내부",
    "AggregationType": "프로파일러 집계 종류 — 잴 프로파일러가 없다",
    "Script*": "TorchScript 객체 (ScriptModule·ScriptFunction …)",
    "import_ir_module": "TorchScript IR 내부",
    "import_ir_module_from_buffer": "TorchScript IR 내부",
    "merge_type_from_type_comment": "TorchScript IR 내부",
    "parse_ir": "TorchScript IR 내부",
    "parse_schema": "TorchScript IR 내부",
    "parse_type_comment": "TorchScript IR 내부",
    "unify_type_list": "TorchScript IR 내부",
    "fork": "TorchScript 비동기 실행",
    "wait": "TorchScript 비동기 실행",
    "ThroughputBenchmark": "TorchScript 벤치 도구",
    "BenchmarkConfig": "TorchScript 벤치 도구",
    "BenchmarkExecutionStats": "TorchScript 벤치 도구",

    # 형별 Storage·Tensor 클래스. torch 가 스스로 버린 것들이다 —
    # `torch.FloatTensor(...)` 는 지금 쓰면 경고가 난다.
    #
    # **`"*Tensor"` 로 적으면 안 된다.** 그렇게 두었더니 `torch.Tensor` 자체를
    # 삼켰다 — 이 라이브러리의 한복판인 이름이다. `test_gap.py` 가 잡았다.
    # 꼬리 와일드카드는 진짜 이름을 삼키는 쪽으로만 빗나가므로 안 쓴다.
    "Storage": "형별 Storage 클래스 — torch 가 폐기했다",
    "StorageBase": "형별 Storage 클래스 — torch 가 폐기했다",
    "TypedStorage": "형별 Storage 클래스 — torch 가 폐기했다",
    "UntypedStorage": "형별 Storage 클래스 — torch 가 폐기했다",
    "set_default_tensor_type": "형별 Tensor 클래스와 함께 폐기됐다",

    # 디스패처 내부. 연산을 어느 백엔드로 보낼지 고르는 자리이고,
    # 우리에게는 고를 백엔드가 없다.
    "DispatchKey": "디스패처 내부",
    "DispatchKeySet": "디스패처 내부",
    "ExcludeDispatchKeyGuard": "디스패처 내부",
    "DisableTorchFunction": "디스패처 내부",
    "DisableTorchFunctionSubclass": "디스패처 내부",
    "boolean_dispatch": "디스패처 내부",
    "handle_torch_function": "디스패처 내부",
    "has_torch_function": "디스패처 내부",
    "has_torch_function_unary": "디스패처 내부",
    "has_torch_function_variadic": "디스패처 내부",
    "assert_int_or_pair": "인자 검사 도구 — F 안에서만 쓴다",
    "factory_kwargs": "층이 device·dtype 를 넘길 때 쓰는 내부 도구",
    "swap_in_optimizer_params_and_state": "옵티마이저 내부 도구",
    "argument_validation": "DataPipe 내부 도구",
    "functional_datapipe": "DataPipe 내부 도구",
    "classproperty": "파이썬 도우미 — 연산이 아니다",

    # 다른 자리에서 들어온 이름표. **자리를 붙여 적는다** — 이름만 적었더니
    # `"Optimizer"` 가 `torch.optim.Optimizer`(진짜 API 이고 우리가 만들었다)까지
    # 삼켰고 `"Tensor"` 가 `torch.Tensor` 를 삼켰다. `test_gap.py` 가 잡았다.
    "nn.functional.Tensor": "들여온 이름표 — F 의 API 가 아니다",
    "optim.lr_scheduler.Tensor": "들여온 이름표 — 그 자리의 API 가 아니다",
    "optim.lr_scheduler.Optimizer": "들여온 이름표 — 그 자리의 API 가 아니다",
    "ScalingType": "들여온 이름표 — fp8 스케일링 종류",
    "SwizzleType": "들여온 이름표 — fp8 배치 종류",

    # 함수화(functionalization) 패스가 쓰는 변종. 사용자가 부르는 이름이 아니다 —
    # 같은 일을 하는 진짜 이름이 따로 있다(`view_copy` 옆에 `view`).
    #
    # **`"*_copy"` 로 적으면 안 된다.** 그렇게 두었더니 `index_copy` 를 삼켰는데
    # 그것은 함수화 변종이 아니라 진짜 연산이고 우리가 이미 만들어 두었다. 목록이
    # 길어도 손으로 적는다 — 여기서 넓은 그물은 우리에게 유리한 쪽으로만 빗나간다.
    "alias_copy": "함수화 패스용 변종 — 진짜 이름은 `alias` 다",
    "as_strided_copy": "함수화 패스용 변종 — 진짜 이름은 `as_strided` 다",
    "ccol_indices_copy": "함수화 패스용 변종 (희소 레이아웃)",
    "col_indices_copy": "함수화 패스용 변종 (희소 레이아웃)",
    "crow_indices_copy": "함수화 패스용 변종 (희소 레이아웃)",
    "row_indices_copy": "함수화 패스용 변종 (희소 레이아웃)",
    "indices_copy": "함수화 패스용 변종 (희소 레이아웃)",
    "values_copy": "함수화 패스용 변종 (희소 레이아웃)",
    "detach_copy": "함수화 패스용 변종 — 진짜 이름은 `detach` 다",
    "diagonal_copy": "함수화 패스용 변종 — 진짜 이름은 `diagonal` 다",
    "expand_copy": "함수화 패스용 변종 — 진짜 이름은 `expand` 다",
    "narrow_copy": "함수화 패스용 변종 — 진짜 이름은 `narrow` 다",
    "permute_copy": "함수화 패스용 변종 — 진짜 이름은 `permute` 다",
    "select_copy": "함수화 패스용 변종 — 진짜 이름은 `select` 다",
    "slice_copy": "함수화 패스용 변종 — 진짜 이름은 자르기다",
    "split_copy": "함수화 패스용 변종 — 진짜 이름은 `split` 이다",
    "split_with_sizes_copy": "함수화 패스용 변종 — 진짜 이름은 `split` 이다",
    "squeeze_copy": "함수화 패스용 변종 — 진짜 이름은 `squeeze` 다",
    "t_copy": "함수화 패스용 변종 — 진짜 이름은 `t` 다",
    "transpose_copy": "함수화 패스용 변종 — 진짜 이름은 `transpose` 다",
    "unbind_copy": "함수화 패스용 변종 — 진짜 이름은 `unbind` 다",
    "unfold_copy": "함수화 패스용 변종 — 진짜 이름은 `unfold` 다",
    "unsqueeze_copy": "함수화 패스용 변종 — 진짜 이름은 `unsqueeze` 다",
    "view_copy": "함수화 패스용 변종 — 진짜 이름은 `view` 다",
    "view_as_real_copy": "함수화 패스용 변종 (복소수)",
    "view_as_complex_copy": "함수화 패스용 변종 (복소수)",
    "unsafe_chunk": "함수화 패스용 변종 — 진짜 이름은 `chunk` 다",
    "unsafe_split": "함수화 패스용 변종 — 진짜 이름은 `split` 이다",
    "unsafe_split_with_sizes": "함수화 패스용 변종 — 진짜 이름은 `split` 이다",
    "slice_inverse": "함수화 패스용 변종 — 되돌리기용 내부 연산",

    # ATen 의 밑단 진입점. 위에 사용자가 부르는 이름이 따로 있다
    # (`native_layer_norm` 위에 `layer_norm`).
    "native_*": "ATen 밑단 진입점 — 위에 부르는 이름이 따로 있다",
    "batch_norm_*": "ATen 밑단 진입점 (통계 조각·역방향 조각)",
    "affine_grid_generator": "ATen 밑단 진입점 — 위가 `affine_grid` 다",
    "grid_sampler_2d": "ATen 밑단 진입점 — 위가 `grid_sample` 이다",
    "grid_sampler_3d": "ATen 밑단 진입점 — 위가 `grid_sample` 이다",
    "norm_except_dim": "ATen 밑단 진입점 — `weight_norm` 이 쓴다",
    "embedding_renorm_": "ATen 밑단 진입점 — `Embedding(max_norm=)` 이 쓴다",
    "convolution": "ATen 밑단 진입점 — 위가 `conv1d/2d/3d` 다",
    "init_num_threads": "런타임 내부",
    "thread_safe_generator": "런타임 내부",
    "get_file_path": "런타임 내부",
    "sym_fresh_size": "심볼 크기 내부",
}

# **진짜 API 인데 안 한다.** 여기 적힌 것은 결함이 아니라 판단이다.
SKIPPED = {
    # 장치·정밀도. GPU 가 f32 하나뿐이고 탭 하나 안이다.
    "autocast": "혼합정밀 — 우리 GPU 는 f32 하나뿐이라 섞을 것이 없다",
    "autocast_*": "혼합정밀 — 위와 같다",
    "clear_autocast_cache": "혼합정밀 — 위와 같다",
    "get_autocast_*": "혼합정밀 — 위와 같다",
    "set_autocast_*": "혼합정밀 — 위와 같다",
    "is_autocast_*": "혼합정밀 — 위와 같다",
    "GradScaler": "혼합정밀의 손실 스케일 — 섞을 정밀도가 없다",
    "get_num_threads": "탭 하나 안이라 고를 스레드 수가 없다",
    "set_num_threads": "탭 하나 안이라 고를 스레드 수가 없다",
    "get_num_interop_threads": "탭 하나 안이다",
    "set_num_interop_threads": "탭 하나 안이다",
    "Stream": "장치 스트림 — 하나뿐이다",
    "Event": "장치 이벤트 — 잴 스트림이 하나다",
    "get_device": "장치가 하나다",
    "get_device_module": "장치가 하나다",
    "get_default_device": "장치가 하나다",
    "set_default_device": "장치가 하나다",
    "is_vulkan_available": "브라우저에서 고를 백엔드가 아니다",
    "cudnn_is_acceptable": "cuDNN 이 없다",
    "AcceleratorError": "가속기가 하나다",
    "OutOfMemoryError": "장치 메모리 오류를 우리가 구분해 내지 않는다",
    "DataParallel": "여러 장치용 — 탭 하나 안이다",
    "SyncBatchNorm": "분산 학습용 — 탭 하나 안이다",
    "DistributedSampler": "분산 학습용 — 탭 하나 안이다",

    # 벤더 커널. 그 하드웨어가 있어야 뜻이 있다.
    "cudnn_*": "NVIDIA 커널 — 브라우저에 없다",
    "miopen_*": "AMD 커널 — 브라우저에 없다",
    "mkldnn_*": "Intel 커널 — 브라우저에 없다",
    "fbgemm_*": "양자화 GEMM 커널 — 실제 하드웨어 위에서만 뜻이 있다",
    "q_per_channel_axis": "양자화된 텐서의 눈금 — 그 dtype 이 없다",
    "q_per_channel_scales": "양자화된 텐서의 눈금 — 그 dtype 이 없다",
    "q_per_channel_zero_points": "양자화된 텐서의 눈금 — 그 dtype 이 없다",
    "q_scale": "양자화된 텐서의 눈금 — 그 dtype 이 없다",
    "q_zero_point": "양자화된 텐서의 눈금 — 그 dtype 이 없다",
    "qscheme": "양자화 방식 — 그 dtype 이 없다",
    "int_repr": "양자화된 텐서의 정수 표현 — 그 dtype 이 없다",
    "choose_qparams_optimized": "양자화 눈금 고르기 — 실 하드웨어 위에서만 뜻이 있다",
    "fused_moving_avg_obs_fake_quant": "양자화 관찰자 — 실 하드웨어 위에서만 뜻이 있다",

    # 희소 텐서. `DELIBERATE` 의 "sparse" 는 자리 이름에만 걸린다.
    "smm": "희소 행렬곱 — 커리큘럼 밖",
    "spmm": "희소 행렬곱 — 커리큘럼 밖",
    "hsmm": "희소 행렬곱 — 커리큘럼 밖",
    "dsmm": "희소 행렬곱 — 커리큘럼 밖",
    "hspmm": "희소 행렬곱 — 커리큘럼 밖",
    "saddmm": "희소 행렬곱 — 커리큘럼 밖",
    "sspaddmm": "희소 행렬곱 — 커리큘럼 밖",
    "segment_reduce": "희소·불규칙 묶음용 — 커리큘럼 밖",
    "resize_as_sparse_": "희소 텐서 전용 — 위와 같다",
    # **이름만 보면 `nn.ParameterDict` 의 짝 같은데 아니다.** 재보니 생성자가
    # `torch._C.ScriptModule` 하나만 받고 `nn` 아래에는 없다 — TorchScript 내부다.
    # 사전처럼 쓰려고 만들면 이름은 같고 물건이 다른 것을 주게 된다.
    "BufferDict": "TorchScript 내부 — 생성자가 ScriptModule 을 받는다",
    # **torch 자신이 못 만든다.** 실수 텐서로 부르면 `NotImplementedError` 다(실측) —
    # 양자화 dtype 이 있어야 자리가 잡힌다. 우리가 인색해서 없는 것이 아니다.
    "empty_quantized": "양자화된 빈 텐서 — torch 도 그 dtype 없이는 못 만든다",

    # 고를 것이 하나뿐인 자리. **이름을 두면 고를 수 있는 것처럼 보인다.**
    #
    # `layout` 과 `memory_format` 은 형(type)이고, 그 형의 값이 각각 `strided` 와
    # `contiguous_format` 하나씩이다. 하나뿐인 값을 고르는 인자를 두면 "고르면
    # 달라지는구나" 를 가르치게 되는데, 여기서는 안 달라진다.
    "layout": "고를 배치가 `strided` 하나뿐이다",
    "memory_format": "고를 배치가 `contiguous_format` 하나뿐이다",
    "prepare_multiprocessing_environment": "탭 하나 안에 프로세스가 없다",
    # **torch 자신이 촘촘한 기울기를 거절한다** — 실측한 문구가
    # "SparseAdam does not support dense gradients, please consider Adam instead" 다.
    # 여기 희소 텐서가 없으므로 이 옵티마이저가 받을 수 있는 입력이 하나도 없다.
    # 만들어 두면 언제나 거절만 하는 물건이 되고, 그것은 있다고 세는 것보다 나쁘다.
    "SparseAdam": "희소 기울기 전용 — 촘촘한 기울기는 torch 도 거절한다",

    # 복소수. 우리 dtype 은 float32·int64·bool 셋이다.
    "complex": "복소수 dtype 이 없다",
    "polar": "복소수 dtype 이 없다",
    "real": "복소수 dtype 이 없다",
    "imag": "복소수 dtype 이 없다",
    "angle": "복소수 dtype 이 없다",
    "conj": "복소수 dtype 이 없다",
    "conj_physical": "복소수 dtype 이 없다",
    "conj_physical_": "복소수 dtype 이 없다",
    "is_complex": "복소수 dtype 이 없다",
    "is_conj": "복소수 dtype 이 없다",
    "is_neg": "켤레·부호 비트 machinery — 복소수와 함께 온다",
    "resolve_conj": "복소수 dtype 이 없다",
    "resolve_neg": "복소수 dtype 이 없다",
    "view_as_complex": "복소수 dtype 이 없다",
    "view_as_real": "복소수 dtype 이 없다",
    "eig": "**복소수 dtype 이 없다** — 비대칭 행렬의 고윳값은 복소수로 나온다",
    "eigvals": "위와 같다. 대칭 행렬 쪽(`eigh`·`eigvalsh`)은 있다",

    # 심볼 크기와 그래프 캡처. `DELIBERATE` 의 compile·export 와 같은 이유인데
    # 이름이 그 자리에 안 걸린다.
    "Sym*": "심볼 크기 — 그래프 캡처용, wasm 에 못 얹는다",
    "sym_*": "심볼 크기 — 위와 같다",
    "cond": "제어 흐름 캡처 연산 — export 용이다",
    "while_loop": "제어 흐름 캡처 연산 — export 용이다",
    "vmap": "functorch 의 자동 배치 — 커리큘럼 밖",

    # 밖과 주고받기. 브라우저 안에는 넘겨줄 상대가 없다.
    "from_dlpack": "DLPack 교환 — 브라우저 안에 상대가 없다",
    "to_dlpack": "DLPack 교환 — 위와 같다",
    "from_file": "파일 매핑 — 브라우저에 그 파일 계층이 없다",
    "from_numpy": "numpy 는 코어 쪽 이야기다",

    # 디버그 스위치. 우리에게는 켤 비결정성도 이상 검출기도 없다.
    "use_deterministic_algorithms": "고를 비결정 커널이 없다",
    "are_deterministic_algorithms_enabled": "위와 같다",
    "is_deterministic_algorithms_warn_only_enabled": "위와 같다",
    "get_deterministic_debug_mode": "위와 같다",
    "set_deterministic_debug_mode": "위와 같다",
    "is_anomaly_enabled": "이상 검출기가 없다",
    "set_anomaly_enabled": "이상 검출기가 없다",
    "is_anomaly_check_nan_enabled": "이상 검출기가 없다",
    "set_warn_always": "경고 정책 스위치 — 커리큘럼 밖",
    "is_warn_always_enabled": "경고 정책 스위치 — 커리큘럼 밖",
    "set_flush_denormal": "비정규수 처리 스위치 — WGSL 이 안 내준다",
    "get_float32_matmul_precision": "TF32 스위치 — 그 하드웨어가 없다",
    "set_float32_matmul_precision": "TF32 스위치 — 그 하드웨어가 없다",

    # torch 가 스스로 폐기했거나 접은 것.
    "symeig": "torch 가 폐기했다 — `eigh` 로 대체됐다",
    "frobenius_norm": "torch 가 폐기했다 — `linalg.matrix_norm` 으로 대체됐다",
    "nuclear_norm": "torch 가 폐기했다 — `linalg.matrix_norm(ord='nuc')` 이다",
    "range": "torch 가 폐기했다 — `arange` 를 쓴다",
    "Container": "torch 가 폐기했다 — `Sequential` 로 대체됐다",
    "NLLLoss2d": "torch 가 폐기했다 — `NLLLoss` 가 그 모양을 받는다",
    "CrossMapLRN2d": "옛 층 — `LocalResponseNorm` 이 그 자리다",
    "IterDataPipe": "torch 가 접은 DataPipes 실험",
    "MapDataPipe": "torch 가 접은 DataPipes 실험",
    "DFIterDataPipe": "torch 가 접은 DataPipes 실험",
    "DataChunk": "torch 가 접은 DataPipes 실험",
    "guaranteed_datapipes_determinism": "torch 가 접은 DataPipes 실험",
    "non_deterministic": "torch 가 접은 DataPipes 실험",
    "runtime_validation": "torch 가 접은 DataPipes 실험",
    "runtime_validation_disabled": "torch 가 접은 DataPipes 실험",
    "Future": "TorchScript 비동기 실행의 약속 — 탭 하나 안이다",
    "conv_tbc": "옛 레이아웃(시간-배치-채널) 합성곱 — 부르는 코드가 없다",

    # **최상위의 날 ATen 손실들.** 이름은 `F` 와 같은데 **같은 함수가 아니다** —
    # 기본 reduction 이 `none` 이고 `reduction` 이 문자열이 아니라 정수다(0·1·2).
    # `torch.kl_div(a, b)` 는 `[2,2]` 를 내고 `F.kl_div(a, b)` 는 스칼라를 낸다.
    #
    # 그래서 `F` 의 것을 최상위에 별명으로 걸면 **모양부터 갈린다.** 튜토리얼이
    # 부르는 것은 `F` 쪽이고, 이쪽은 ATen 이 이름을 낸 것뿐이다. 재서 확인했다 —
    # 같은 함수인 것은 `pairwise_distance` 와 `pdist` 둘뿐이고 그 둘은 냈다.
    # **자리를 붙여 적는다.** 이름만 적으면 `F` 쪽 같은 이름까지 삼킨다 — `F.kl_div`
    # 는 우리가 만들었고 `F.ctc_loss` 는 진짜 빈자리다.
    "torch.binary_cross_entropy_with_logits":
        "최상위는 날 ATen 연산 — 기본 reduction 이 none 이고 인자가 정수다",
    "torch.cosine_embedding_loss": "최상위는 날 ATen 연산 — F 쪽과 서명이 다르다",
    "torch.hinge_embedding_loss": "최상위는 날 ATen 연산 — F 쪽과 서명이 다르다",
    "torch.kl_div": "최상위는 날 ATen 연산 — F 쪽과 서명이 다르다",
    "torch.margin_ranking_loss": "최상위는 날 ATen 연산 — F 쪽과 서명이 다르다",
    "torch.poisson_nll_loss": "최상위는 날 ATen 연산 — 기본값이 아예 없다",
    "torch.triplet_margin_loss": "최상위는 날 ATen 연산 — F 쪽과 서명이 다르다",

    # 아직 안 굳은 것.
    "LinearCrossEntropyLoss": "torch 에 갓 들어온 것 — 굳으면 본다",
    "LinearCrossEntropyOptions": "위와 같다",
    "linear_cross_entropy": "위 층의 함수 짝 — 같이 굳으면 같이 본다",
    "Muon": "torch 에 갓 들어온 옵티마이저 — 굳으면 본다",
    "grouped_mm": "fp8·묶음 GEMM — 그 하드웨어가 없다",
    "scaled_grouped_mm": "fp8·묶음 GEMM — 그 하드웨어가 없다",
    "scaled_mm": "fp8 GEMM — 그 하드웨어가 없다",
}

# 볼 자리. (보이는 이름, torch 쪽, 우리 쪽)
def _spaces():
    got = [("torch", torch, borch),
           ("Tensor", torch.Tensor, borch.Tensor),
           ("nn", torch.nn, borch.nn),
           ("nn.functional", torch.nn.functional, borch.nn.functional),
           ("optim", torch.optim, borch.optim),
           ("optim.lr_scheduler", torch.optim.lr_scheduler, borch.optim.lr_scheduler),
           ("linalg", torch.linalg, borch.linalg),
           ("utils.data", torch.utils.data, borch.utils.data)]
    return [(name, a, b) for name, a, b in got if b is not None]


def _public(obj):
    """공개 이름만 — **부를 수 있는 것**으로 좁힌다.

    분모를 세 번 고쳤고 그 과정을 남긴다.

    1. `dir()` 그대로 → torch 표면 1,013 개. `Callable`·`Optional`(typing 임포트)과
       `AnyType`·`ArgumentSpec`(C 확장 내부 타입)이 API 로 세어졌다.
    2. `__all__` → 최상위에서 더 나빠졌다. torch 의 최상위 `__all__` 은 손으로 고른
       공개 목록이 아니라 C 내부 타입까지 담은 생성물이라 905 개가 나오고, 겹치는
       것이 3 개로 떨어졌다. **분모가 거짓이면 비율도 거짓이다.**
    3. 지금 — 부를 수 있는 것(함수·클래스)만, 그리고 남의 모듈에서 들어온 이름은 뺀다.

    그래도 완벽하지 않다. 이 수는 "표면의 몇 %" 라고 자랑할 값이 아니라 **어느 자리가
    비었는지 짚는 데** 쓰는 값이다.
    """
    out = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        thing = getattr(obj, name, None)
        if thing is None or not callable(thing):
            continue
        # 남의 모듈에서 들어온 이름(typing 등)은 그 자리의 API 가 아니다.
        home = getattr(thing, "__module__", "") or ""
        if home and not (home.startswith("torch") or home.startswith("borch")):
            continue
        out.add(name)
    return out


def _look(table, name, full=None):
    """표에서 사유를 찾는다. **자리 붙은 이름 → 이름 → 와일드카드** 순이다.

    순서가 두 번 문제였다.

    `"*Tensor"` 가 `"Tensor"` 보다 앞에 적혀 있었더니 `nn.functional.Tensor` 가
    "형별 Tensor 클래스 — torch 가 폐기했다" 로 설명됐다. 갈래는 맞았지만 **사유가
    거짓이었다** — 그것은 그냥 들여온 이름표다.

    그 다음엔 이름만 적은 `"Optimizer"` 가 `torch.optim.Optimizer` 까지 삼켰다.
    그쪽은 진짜 API 이고 우리가 이미 만들어 두었다 — `test_gap.py` 가 그 모순을
    잡았다. 그래서 자리를 붙여 적을 수 있게 했다.

    이 표에서 가장 나쁜 것이 그 모양이다. 안 세는 것은 눈에 띄지만 **틀린 이유로
    안 세는 것**은 안 띄고, 다음에 읽는 사람이 그 거짓을 믿는다.
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
    """왜 없어도 되는지. 없으면 None — **그것이 곧 검토 대상이다.**

    돌려주는 것은 `(갈래, 사유)` 다. 갈래를 안 나누면 "API 가 아니라 안 센다" 와
    "API 인데 안 한다" 가 한 수로 뭉개지고, 그 둘은 읽는 사람에게 다른 말이다.

    **최상위도 자리를 붙인다**(`torch.kl_div`). 안 붙였더니 최상위만 두고 싶은 판단이
    `nn.functional` 의 같은 이름까지 삼켰다 — 그쪽 `kl_div` 는 우리가 만들었고
    `ctc_loss` 는 아직 진짜 빈자리다.
    """
    full = f"{space}.{name}"
    for key, reason in DELIBERATE.items():
        if full.startswith(key) or name.startswith(key):
            return ("자리", reason)
    found = _look(NOT_API, name, full)
    if found:
        return ("API 아님", found)
    found = _look(SKIPPED, name, full)
    if found:
        return ("안 함", found)
    return None


def main(argv):
    show = None
    if "--show" in argv:
        show = argv[argv.index("--show") + 1]
    extra = "--extra" in argv

    total_missing = total_have = 0
    for space, theirs, ours in _spaces():
        # `--show nn` 은 그 자리만 본다. 전부 찍으면 찾던 줄이 묻힌다.
        if show not in (None, "all") and space != show:
            continue
        a, b = _public(theirs), _public(ours)
        gap = sorted(a - b) if not extra else sorted(b - a)
        judged = [(n, _why(space, n)) for n in gap]
        unexplained = [n for n, why in judged if why is None]

        if extra:
            print(f"\n{space} — torch 에 없는데 우리에게 있는 것 {len(gap)}개")
            for n in gap:
                print(f"  + {n}")
            continue

        # **분모에서 빼는 것은 "API 가 아닌 것" 뿐이다.**
        #
        # 안 하기로 한 것은 남긴다. 우리가 고른 것이므로 비용을 져야 한다 — 빼 주면
        # 표에 이름을 적을 때마다 비율이 올라가고, 그러면 이 표가 "수를 예쁘게
        # 만드는 자리" 가 된다. 남겨 두면 적어도 그 유혹이 없어진다.
        not_api = {n for n, why in judged if why and why[0] == "API 아님"}
        skipped = [(n, why[1]) for n, why in judged if why and why[0] == "안 함"]
        by_space = [n for n, why in judged if why and why[0] == "자리"]
        api = a - not_api
        covered = len(api & b)

        total_have += covered
        total_missing += len(unexplained)
        print(f"\n{space} — API {len(api)}개 중 {covered}개 있다 "
              f"({covered * 100 // max(1, len(api))}%)")
        parts = [f"API 아니라 안 셈 {len(not_api)}"]
        if by_space:
            parts.append(f"자리째 거절 {len(by_space)}")
        if skipped:
            parts.append(f"안 하기로 함 {len(skipped)}")
        print("  " + " · ".join(parts) + f" · **검토 대상 {len(unexplained)}**")
        if show in (space, "all"):
            for n, reason in skipped:
                print(f"    – {n}: {reason}")
            for n in unexplained:
                print(f"    ? {n}")
        elif unexplained:
            head = ", ".join(unexplained[:8])
            more = f" … 외 {len(unexplained) - 8}" if len(unexplained) > 8 else ""
            print(f"    {head}{more}")

    if not extra:
        print(f"\n합계 — 겹치는 이름 {total_have}개, 설명 안 된 빈자리 {total_missing}개")
        print("  `--show <자리>` 로 그 자리를 전부 편다. `--show all` 은 전부.")
        print("  `–` 는 안 하기로 한 것(사유가 붙는다), `?` 는 검토 대상이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
