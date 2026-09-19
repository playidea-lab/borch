//! 브라우저(WebGPU)에서 Burn wgpu+autodiff 백엔드로 ResNet-18(CIFAR 변형)을
//! 한 스텝씩 학습하는 wasm 모듈. 다른 프레임워크와 같은 페이지에서 시간을 재기 위한 벤치용.
//!
//! JS 진입점: `init()` → `train_step(x, y, batch)` 반복, `backend()`는 라벨.

use std::cell::RefCell;

use burn::backend::wgpu::{RuntimeOptions, Wgpu, WgpuDevice, graphics::WebGpu, init_setup_async};
use burn::backend::Autodiff;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{GradientsParams, Optimizer, Sgd, SgdConfig};
use burn::prelude::*;
use wasm_bindgen::prelude::*;

/// 학습용 백엔드: wgpu 위에 autodiff.
type AD = Autodiff<Wgpu>;

const IMAGE_CHANNELS: usize = 3;
const IMAGE_SIZE: usize = 32;
const NUM_CLASSES: usize = 10;
const STAGE_CHANNELS: [usize; 4] = [64, 128, 256, 512];
const LEARNING_RATE: f64 = 0.05;
const MOMENTUM: f64 = 0.9;

// ---------------------------------------------------------------------------
// 모델
// ---------------------------------------------------------------------------

/// 3×3 conv, bias 없음, 패딩 1 (다른 프레임워크의 벤치 정의와 동일).
fn conv3x3<B: Backend>(cin: usize, cout: usize, stride: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([cin, cout], [3, 3])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(false)
        .init(device)
}

/// 1×1 stride-s conv + BN 숏컷 (stage 2~4의 첫 블록).
#[derive(Module, Debug)]
struct Downsample<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> Downsample<B> {
    fn new(cin: usize, cout: usize, stride: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([cin, cout], [1, 1])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(false)
            .init(device);
        let bn = BatchNormConfig::new(cout).init(device);
        Self { conv, bn }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.bn.forward(self.conv.forward(x))
    }
}

/// BasicBlock: conv3×3(stride) → BN → ReLU → conv3×3 → BN, 숏컷 합산, ReLU.
#[derive(Module, Debug)]
struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    downsample: Option<Downsample<B>>,
    relu: Relu,
}

impl<B: Backend> BasicBlock<B> {
    fn new(cin: usize, cout: usize, stride: usize, device: &B::Device) -> Self {
        // 채널이 바뀌거나 stride가 2면 projection 숏컷, 아니면 identity
        let downsample = if stride != 1 || cin != cout {
            Some(Downsample::new(cin, cout, stride, device))
        } else {
            None
        };
        Self {
            conv1: conv3x3(cin, cout, stride, device),
            bn1: BatchNormConfig::new(cout).init(device),
            conv2: conv3x3(cout, cout, 1, device),
            bn2: BatchNormConfig::new(cout).init(device),
            downsample,
            relu: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let shortcut = match &self.downsample {
            Some(ds) => ds.forward(x.clone()),
            None => x.clone(),
        };
        let out = self.relu.forward(self.bn1.forward(self.conv1.forward(x)));
        let out = self.bn2.forward(self.conv2.forward(out));
        self.relu.forward(out + shortcut)
    }
}

/// ResNet-18 CIFAR 변형: stem 3×3 s1 → 4 stage × 2 block → GAP → Linear 512→10.
#[derive(Module, Debug)]
struct ResNet18<B: Backend> {
    stem_conv: Conv2d<B>,
    stem_bn: BatchNorm<B>,
    relu: Relu,
    blocks: Vec<BasicBlock<B>>,
    pool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> ResNet18<B> {
    fn new(device: &B::Device) -> Self {
        let mut blocks = Vec::with_capacity(STAGE_CHANNELS.len() * 2);
        let mut cin = STAGE_CHANNELS[0];
        for (stage, &cout) in STAGE_CHANNELS.iter().enumerate() {
            // 첫 stage는 stride 1, 나머지는 첫 블록만 stride 2
            let first_stride = if stage == 0 { 1 } else { 2 };
            blocks.push(BasicBlock::new(cin, cout, first_stride, device));
            blocks.push(BasicBlock::new(cout, cout, 1, device));
            cin = cout;
        }
        Self {
            stem_conv: conv3x3(IMAGE_CHANNELS, STAGE_CHANNELS[0], 1, device),
            stem_bn: BatchNormConfig::new(STAGE_CHANNELS[0]).init(device),
            relu: Relu::new(),
            blocks,
            pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc: LinearConfig::new(STAGE_CHANNELS[3], NUM_CLASSES).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut out = self.relu.forward(self.stem_bn.forward(self.stem_conv.forward(x)));
        for block in &self.blocks {
            out = block.forward(out);
        }
        // [B, 512, 1, 1] → [B, 512]
        let pooled = self.pool.forward(out).flatten::<2>(1, 3);
        self.fc.forward(pooled)
    }
}

// ---------------------------------------------------------------------------
// 전역 상태 (모델 + 옵티마이저)
// ---------------------------------------------------------------------------

type SgdOptim = OptimizerAdaptor<Sgd<Wgpu>, ResNet18<AD>, AD>;

struct State {
    device: WgpuDevice,
    model: ResNet18<AD>,
    optim: SgdOptim,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn js_err(msg: impl Into<String>) -> JsValue {
    JsValue::from_str(&msg.into())
}

// ---------------------------------------------------------------------------
// wasm-bindgen 진입점
// ---------------------------------------------------------------------------

/// WebGPU 디바이스를 비동기로 생성하고 모델·옵티마이저를 초기화한다.
#[wasm_bindgen]
pub async fn init() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let device = WgpuDevice::DefaultDevice;
    // wasm에서는 동기 init이 panic하므로 반드시 async 경로를 쓴다
    init_setup_async::<WebGpu>(&device, RuntimeOptions::default()).await;

    let model = ResNet18::<AD>::new(&device);
    let optim: SgdOptim = SgdConfig::new()
        .with_momentum(Some(MomentumConfig {
            momentum: MOMENTUM,
            dampening: 0.0,
            nesterov: false,
        }))
        .init();

    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            device,
            model,
            optim,
        })
    });
    Ok(())
}

/// forward + backward + SGD 한 스텝. GPU가 끝난 뒤 loss를 f32로 반환한다.
///
/// `x`: Float32Array [batch*3*32*32] NCHW, `y`: Int32Array/Uint32Array [batch] 정수 라벨.
#[wasm_bindgen]
pub async fn train_step(x: &[f32], y: JsValue, batch: usize) -> Result<f32, JsValue> {
    let expected = batch * IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE;
    if x.len() != expected {
        return Err(js_err(format!(
            "x length {} != batch*3*32*32 = {}",
            x.len(),
            expected
        )));
    }
    // Int32Array·Uint32Array 모두 받기 위해 JS 쪽에서 Int32Array로 복사
    let labels: Vec<i32> = js_sys::Int32Array::new(&y).to_vec();
    if labels.len() != batch {
        return Err(js_err(format!("y length {} != batch {}", labels.len(), batch)));
    }

    // await 동안 RefCell borrow를 들고 있지 않도록 상태를 꺼내서 쓴다
    let State {
        device,
        model,
        mut optim,
    } = STATE
        .with(|s| s.borrow_mut().take())
        .ok_or_else(|| js_err("call init() first"))?;

    let x_t = Tensor::<AD, 4>::from_data(
        TensorData::new(x.to_vec(), [batch, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]),
        &device,
    );
    let y_t = Tensor::<AD, 1, Int>::from_data(TensorData::new(labels, [batch]), &device);

    let logits = model.forward(x_t);
    let loss = CrossEntropyLossConfig::new()
        .init(&device)
        .forward(logits, y_t);
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let model = optim.step(LEARNING_RATE, model, grads);

    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            device,
            model,
            optim,
        })
    });

    // 옵티마이저 스텝까지 큐에 넣은 뒤 읽으므로, 이 readback 완료 = 스텝 전체 GPU 완료
    let data = loss
        .into_data_async()
        .await
        .map_err(|e| js_err(format!("loss readback failed: {e:?}")))?;
    let values = data
        .to_vec::<f32>()
        .map_err(|e| js_err(format!("loss decode failed: {e:?}")))?;
    values
        .first()
        .copied()
        .ok_or_else(|| js_err("empty loss tensor"))
}

/// 벤치 표시용 라벨.
#[wasm_bindgen]
pub fn backend() -> String {
    "burn 0.21 wgpu (wasm)".to_string()
}
