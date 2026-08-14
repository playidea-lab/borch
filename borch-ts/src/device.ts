/**
 * WebGPU 장치·버퍼·파이프라인 캐시.
 *
 * ## 파이프라인 캐시가 왜 여기 있나
 *
 * 모양을 셰이더에 굽는 것이 이 라이브러리의 전제다(`kernels.ts` 참고). 그러면 같은
 * 연산도 모양이 다르면 다른 셰이더이고, 층을 한 번 지나갈 때마다 컴파일하면 커널이
 * 빠른 것이 의미가 없다. 그래서 **모양 서명 → 파이프라인** 이 자료구조로 들어간다.
 * 최적화가 아니라 굽기로 얻은 속도를 지키는 값이다.
 *
 * ## 한계는 조용히 온다
 *
 * WebGPU 는 버퍼 크기나 dispatch 한계를 넘으면 **던지지 않고 안 한다.** 벤치에서
 * 두 번 밟았다 — 128MB 초과에서 24만 GFLOPS, dispatch 65,535 초과에서 "144%".
 * 둘 다 값을 안 봤으면 믿었을 수치라, 여기서는 한계를 **미리 재고 넘으면 던진다.**
 */

import { grid1d, reduceParts, reduceSum, WORKGROUP } from "./kernels.js";

const BYTES_PER_F32 = 4;

export class Device {
  private readonly device: GPUDevice;
  private readonly limits: GPUSupportedLimits;
  /** 모양까지 포함한 서명 → 파이프라인. */
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  /** 읽어올 때 쓰는 staging 버퍼. 크기별로 하나씩 재사용한다. */
  private readonly staging = new Map<number, GPUBuffer>();

  private constructor(device: GPUDevice) {
    this.device = device;
    this.limits = device.limits;
  }

  static async create(): Promise<Device> {
    const gpu = navigator.gpu;
    if (!gpu) {
      throw new Error(
        "WebGPU 가 없다. Chrome/Edge 113+ 또는 Safari 18+ 가 필요하고, " +
          "리눅스에서는 플래그가 필요할 수 있다.",
      );
    }
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU 어댑터를 못 얻었다.");
    // 기본 한계를 그대로 쓰지 않고 어댑터가 주는 최대치를 요청한다. 기본
    // maxStorageBufferBindingSize 는 128MB 이고, 그 위에서 조용히 틀린 답이 나온다.
    const want: Record<string, number> = {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    };
    const device = await adapter.requestDevice({ requiredLimits: want });
    device.lost
      .then((info) => {
        // 장치 소실은 조용히 오면 안 된다 — 이후 모든 결과가 의미 없어진다.
        console.error(`[borch.ts] WebGPU 장치를 잃었다: ${info.reason} — ${info.message}`);
      })
      .catch(() => {
        /* lost 는 거절되지 않지만, 거절되더라도 여기서 더 할 일이 없다 */
      });
    return new Device(device);
  }

  /** 셰이더 컴파일 오류를 삼키지 않는다. */
  pipeline(signature: string, source: () => string): GPUComputePipeline {
    const hit = this.pipelines.get(signature);
    if (hit) return hit;
    const module = this.device.createShaderModule({ code: source() });
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.pipelines.set(signature, pipeline);
    return pipeline;
  }

  /** 지금까지 구운 셰이더 수. 캐시가 도는지 테스트가 본다. */
  get pipelineCount(): number {
    return this.pipelines.size;
  }

  alloc(count: number): GPUBuffer {
    const bytes = count * BYTES_PER_F32;
    const max = this.limits.maxStorageBufferBindingSize;
    if (bytes > max) {
      // 넘긴 채로 돌리면 WebGPU 는 조용히 일부만 쓴다. 여기서 멈추는 편이 낫다.
      throw new Error(
        `버퍼가 한계를 넘는다: ${(bytes / 1048576).toFixed(1)}MB > ` +
          `${(max / 1048576).toFixed(0)}MB (maxStorageBufferBindingSize)`,
      );
    }
    return this.device.createBuffer({
      size: Math.max(bytes, BYTES_PER_F32),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  upload(data: Float32Array): GPUBuffer {
    const buf = this.alloc(data.length);
    this.device.queue.writeBuffer(buf, 0, data as unknown as BufferSource);
    return buf;
  }

  /**
   * 커널 한 번. `groups` 는 **워크그룹 수**이고, 축당 한계를 여기서 다시 확인한다 —
   * `kernels.ts` 가 격자를 접어 주지만 손으로 부르는 경로가 생길 수 있다.
   */
  run(
    pipeline: GPUComputePipeline,
    buffers: readonly GPUBuffer[],
    groups: readonly [number, number, number],
  ): void {
    const cap = this.limits.maxComputeWorkgroupsPerDimension;
    for (const [axis, count] of groups.entries()) {
      if (count > cap) {
        throw new Error(
          `dispatch ${axis} 축이 한계를 넘는다: ${count} > ${cap}. ` +
            "WebGPU 는 이것을 던지지 않고 조용히 안 한다.",
        );
      }
    }
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** 1차원 작업을 격자로 펴서 돌린다. `kernels.ts` 의 인덱싱과 짝이다. */
  run1d(pipeline: GPUComputePipeline, buffers: readonly GPUBuffer[], n: number): void {
    const g = grid1d(n);
    this.run(pipeline, buffers, [g.x, g.y, 1]);
  }

  /**
   * 전체 합. 부분합이 하나가 될 때까지 같은 커널을 다시 부른다.
   *
   * **원자 연산을 안 쓴다** — 부동소수 덧셈은 순서가 바뀌면 값이 달라지고, 그러면
   * 같은 씨앗으로 두 번 돌린 학습이 갈린다. 느린 쪽이 재현된다.
   */
  sumAll(input: GPUBuffer, n: number): GPUBuffer {
    let src = input;
    let count = n;
    let owned: GPUBuffer | null = null;
    while (count > 1) {
      const parts = reduceParts(count);
      const dst = this.alloc(parts);
      const size = count;
      this.run1d(
        this.pipeline(`reduceSum:${size}`, () => reduceSum(size)),
        [src, dst],
        size,
      );
      if (owned) owned.destroy();
      owned = dst;
      src = dst;
      count = parts;
    }
    if (owned) return owned;
    // 원소가 하나면 접을 것이 없다. 입력을 그대로 돌려주면 호출자가 남의 버퍼를
    // 파괴하게 되므로 복사해서 준다.
    const copy = this.alloc(1);
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(input, 0, copy, 0, BYTES_PER_F32);
    this.device.queue.submit([encoder.finish()]);
    return copy;
  }

  async read(buffer: GPUBuffer, count: number): Promise<Float32Array> {
    const bytes = count * BYTES_PER_F32;
    let stage = this.staging.get(bytes);
    if (!stage) {
      stage = this.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.staging.set(bytes, stage);
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, stage, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    // 매핑된 메모리는 unmap 하면 사라진다. 반드시 복사해서 내보낸다.
    const out = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    return out;
  }

  /** 워크그룹 크기. 커널과 장치가 같은 값을 봐야 한다. */
  static readonly workgroup = WORKGROUP;
}
