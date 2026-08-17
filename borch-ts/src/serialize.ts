/**
 * 체크포인트를 바이트로, 바이트를 체크포인트로 — **safetensors** 형식.
 *
 * ## 왜 이것이 필요한가
 *
 * `stateDict()` 는 있었는데 **저장할 방법이 없었다.** 탭을 새로고침하면 학습 결과가
 * 사라진다. 브라우저는 프로세스가 데스크톱보다 훨씬 자주 죽는 자리라 torch 보다
 * 오히려 더 아쉬운 구멍이었다.
 *
 * ## 왜 우리 형식을 안 만들었나
 *
 * torch 의 `save`/`load` 는 pickle 이다. 파이썬 객체를 실행하며 푸는 형식이라
 * 브라우저로 옮길 수도 없고 옮겨서도 안 된다.
 *
 * safetensors 는 머리가 JSON 이고 몸이 연속된 바이트다. **코덱이 이 파일 하나이고,
 * 그 값으로 파이썬 `borch`·numpy·HF 도구가 같은 파일을 읽는다.** 자체 형식을 만들면
 * 그것이 안 된다 — 브라우저에서 학습해 자기 컴퓨터로 가져가는 길이 이 프로젝트의
 * 이야기 절반이므로, 읽는 쪽이 우리뿐인 형식은 절반을 버리는 것이다.
 *
 * ```
 * [8바이트 LE u64: 머리 길이 N][N바이트 JSON 머리][몸: 텐서 바이트가 이어 붙는다]
 * ```
 *
 * ## dtype 을 정직하게 적는 법
 *
 * borch 의 `int64`·`bool` 은 **이름표일 뿐 값은 float32 버퍼에 있다**(`dtype.ts`).
 * 머리에 `I64` 라고 적으면 4 바이트짜리 몸과 어긋나서 **남의 리더가 깨진다.**
 *
 * 그래서 `dtype` 은 언제나 `F32` 로 적고, borch 의 이름표는 `__metadata__` 에 따로
 * 싣는다. 남이 읽으면 float32 배열이 나오고(맞다), 우리가 읽으면 이름표까지 돌아온다.
 */

import { RuntimeError } from "./errors.js";
import { DTYPES, type DType } from "./dtype.js";
import { Tensor } from "./tensor.js";

const BYTES_PER_F32 = 4;

/** 머리 길이를 적는 자리. safetensors 가 정한 크기다. */
const LENGTH_FIELD = 8;

/**
 * 머리를 이 배수로 맞춰 몸이 8 바이트에서 시작하게 한다.
 *
 * 사양이 요구하지는 않지만 참조 구현이 공백으로 채워 맞춘다. 어긋나면 numpy 쪽에서
 * 몸을 그대로 매핑하지 못하고 복사가 한 번 더 든다.
 */
const ALIGN = 8;

/** borch 이름표를 싣는 열쇠의 앞머리. float32 인 것은 안 적는다 — 기본값이다. */
const DTYPE_KEY = "borch.dtype:";

export interface Bundle {
  tensors: Record<string, Tensor>;
  metadata: Record<string, string>;
}

interface Entry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

/**
 * 체크포인트를 바이트로.
 *
 * **비동기다** — 값이 GPU 에 있으므로 되가져오는 왕복이 든다. 호스트에 내려둔
 * 텐서(`await t.cpu()`)를 주면 그 왕복이 없다.
 */
export async function save(
  tensors: Record<string, Tensor>,
  metadata: Record<string, string> = {},
): Promise<Uint8Array> {
  const names = Object.keys(tensors);
  // **이름 순서를 고정한다.** 객체의 열쇠 순서는 만든 차례를 따르므로, 같은 모델을
  // 두 번 저장하면 같은 바이트가 나와야 한다는 약속이 그것 하나에 걸린다.
  names.sort();

  const header: Record<string, Entry | Record<string, string>> = {};
  const meta: Record<string, string> = { ...metadata };
  const bodies: Float32Array[] = [];
  let offset = 0;

  for (const name of names) {
    const t = tensors[name] as Tensor;
    // **복소수는 아직 이 파일 형식에 안 들어간다.** 저장이 칸당 f32 두 개라
    // `shape` 와 몸 길이가 어긋나고, 그 상태는 저장은 되는데 **읽을 때** 터진다 —
    // 체크포인트가 몇 시간 뒤에 못 읽히는 것이 제일 나쁜 실패다.
    if (t.dtype === "complex64") {
      throw new RuntimeError(
        `'${name}' 이 complex64 다 — 아직 저장 못 한다. ` +
          "`viewAsReal()` 로 실수 짝을 저장하고 읽을 때 `viewAsComplex()` 로 되돌려라.",
      );
    }
    const values = await t.toArray();
    const bytes = values.length * BYTES_PER_F32;
    header[name] = {
      dtype: "F32",
      shape: [...t.shape],
      data_offsets: [offset, offset + bytes],
    };
    if (t.dtype !== "float32") meta[DTYPE_KEY + name] = t.dtype;
    bodies.push(values);
    offset += bytes;
  }

  if (Object.keys(meta).length > 0) header.__metadata__ = meta;

  const json = new TextEncoder().encode(JSON.stringify(header));
  const padding = (ALIGN - ((LENGTH_FIELD + json.length) % ALIGN)) % ALIGN;
  const headerLength = json.length + padding;

  const out = new Uint8Array(LENGTH_FIELD + headerLength + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerLength), true);
  out.set(json, LENGTH_FIELD);
  // 남는 자리는 공백이다 — JSON 파서가 뒤에 붙은 공백을 그냥 지난다.
  out.fill(0x20, LENGTH_FIELD + json.length, LENGTH_FIELD + headerLength);

  let at = LENGTH_FIELD + headerLength;
  for (const body of bodies) {
    out.set(new Uint8Array(body.buffer, body.byteOffset, body.byteLength), at);
    at += body.byteLength;
  }
  return out;
}

/**
 * 바이트를 체크포인트로. **동기다** — 올리는 것은 큐에 쓰기 하나다.
 *
 * 깨진 파일에서 조용히 이상한 텐서를 만들지 않는다. 길이·자리·형을 다 확인하고
 * 안 맞으면 던진다 — 체크포인트가 조용히 틀리면 그 뒤 학습 전체가 뜻을 잃는다.
 */
export function load(bytes: Uint8Array): Bundle {
  if (bytes.byteLength < LENGTH_FIELD) {
    throw new RuntimeError(`체크포인트가 너무 짧다: ${bytes.byteLength} 바이트`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  const bodyAt = LENGTH_FIELD + headerLength;
  if (!Number.isSafeInteger(headerLength) || bodyAt > bytes.byteLength) {
    throw new RuntimeError(
      `머리 길이가 파일을 넘는다: ${headerLength} (파일 ${bytes.byteLength})`,
    );
  }

  const text = new TextDecoder().decode(
    bytes.subarray(LENGTH_FIELD, LENGTH_FIELD + headerLength),
  );
  let header: Record<string, unknown>;
  try {
    header = JSON.parse(text) as Record<string, unknown>;
  } catch {
    throw new RuntimeError("체크포인트 머리가 JSON 이 아니다");
  }

  const raw = header.__metadata__;
  const metadata: Record<string, string> = isStringMap(raw) ? raw : {};
  const tensors: Record<string, Tensor> = {};

  for (const [name, value] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const entry = asEntry(name, value);
    const [begin, end] = entry.data_offsets;
    if (begin > end || bodyAt + end > bytes.byteLength) {
      throw new RuntimeError(
        `'${name}' 의 자리가 파일을 넘는다: [${begin}, ${end}]`,
      );
    }
    const count = (end - begin) / BYTES_PER_F32;
    const size = entry.shape.reduce((a, b) => a * b, 1);
    if (count !== size) {
      throw new RuntimeError(
        `'${name}' 의 모양 [${entry.shape}] 는 원소 ${count} 개와 안 맞는다`,
      );
    }
    // **사본을 뜬다.** 원본 바이트 배열이 8 바이트 정렬이 아닐 수 있고, 그때
    // `new Float32Array(buffer, offset)` 은 그냥 던진다. 여기서 한 번 복사하면
    // 정렬 걱정이 사라지고, 어차피 GPU 로 올리며 또 복사한다.
    const values = new Float32Array(
      bytes.slice(bodyAt + begin, bodyAt + end).buffer,
    );
    const label = metadata[DTYPE_KEY + name];
    tensors[name] = Tensor.from(values, entry.shape, {
      ...(label === undefined ? {} : { dtype: asDType(name, label) }),
    });
  }
  return { tensors, metadata };
}

/** 머리에 든 것이 우리가 아는 모양인가. 아니면 그 이름을 대고 던진다. */
function asEntry(name: string, value: unknown): Entry {
  const e = value as Partial<Entry> | null;
  if (
    !e || typeof e !== "object"
    || !Array.isArray(e.shape) || !e.shape.every((n) => Number.isInteger(n))
    || !Array.isArray(e.data_offsets) || e.data_offsets.length !== 2
    || typeof e.data_offsets[0] !== "number" || typeof e.data_offsets[1] !== "number"
  ) {
    throw new RuntimeError(`'${name}' 의 머리 항목이 깨졌다`);
  }
  if (e.dtype !== "F32") {
    // **근사하지 않는다.** 남이 만든 F16·I64 파일을 float32 로 읽어 주면 값은
    // 나오는데 그 값이 무엇인지는 아무도 모른다.
    throw new RuntimeError(
      `'${name}' 의 dtype 이 ${String(e.dtype)} 다 — borch 는 F32 만 읽는다`,
    );
  }
  return { dtype: "F32", shape: e.shape, data_offsets: [e.data_offsets[0], e.data_offsets[1]] };
}

function asDType(name: string, label: string): DType {
  if (!(DTYPES as readonly string[]).includes(label)) {
    throw new RuntimeError(`'${name}' 의 형 이름표가 낯설다: ${label}`);
  }
  return label as DType;
}

function isStringMap(v: unknown): v is Record<string, string> {
  return typeof v === "object" && v !== null
    && Object.values(v).every((x) => typeof x === "string");
}

/**
 * 이름 앞에 꼬리표를 붙인다. 모델과 옵티마이저를 **한 파일에** 담을 때 쓴다.
 *
 * ```ts
 * const bytes = await save({
 *   ...prefixed("model", model.stateDict()),
 *   ...prefixed("opt", opt.stateDict().tensors),
 * }, { ...numbersToMeta("opt", opt.stateDict().numbers) });
 * ```
 */
export function prefixed<T>(
  prefix: string, entries: Record<string, T>,
): Record<string, T> {
  const out: Record<string, T> = {};
  for (const [name, value] of Object.entries(entries)) out[`${prefix}.${name}`] = value;
  return out;
}

/** 꼬리표를 뗀다. 앞머리가 다른 것은 안 준다. */
export function unprefixed<T>(
  prefix: string, entries: Record<string, T>,
): Record<string, T> {
  const head = `${prefix}.`;
  const out: Record<string, T> = {};
  for (const [name, value] of Object.entries(entries)) {
    if (name.startsWith(head)) out[name.slice(head.length)] = value as T;
  }
  return out;
}

/**
 * 수를 머리에 실을 수 있게 문자열로. safetensors 의 `__metadata__` 는 문자열만 받는다.
 *
 * **`JSON.stringify` 로 적는다.** `String(0.1)` 은 `"0.1"` 이고 그것을 다시 읽으면
 * 같은 배정도 수가 나오지만, `Infinity` 는 `"Infinity"` 가 되어 `JSON.parse` 가
 * 거절한다 — `ReduceLROnPlateau` 의 `best` 가 정확히 무한대에서 시작한다.
 */
export function numbersToMeta(
  prefix: string, numbers: Record<string, number>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [name, value] of Object.entries(numbers)) {
    out[`${prefix}.${name}`] = Number.isFinite(value) ? String(value)
      : value > 0 ? "Infinity" : value < 0 ? "-Infinity" : "NaN";
  }
  return out;
}

/** 위의 되돌림. 앞머리가 맞는 것만 준다. */
export function metaToNumbers(
  prefix: string, metadata: Record<string, string>,
): Record<string, number> {
  const head = `${prefix}.`;
  const out: Record<string, number> = {};
  for (const [name, text] of Object.entries(metadata)) {
    if (!name.startsWith(head)) continue;
    out[name.slice(head.length)] = text === "Infinity" ? Infinity
      : text === "-Infinity" ? -Infinity : Number(text);
  }
  return out;
}
