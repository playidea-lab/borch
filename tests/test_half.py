"""`f32ToF16Bit` 을 Node 의 `Float16Array` 와 비트 단위로 대조한다.

`Float16Array` 는 IEEE 반정밀도, 가장 가까운 값·동률은 짝수로 반올림한다 — WGSL 의
`pack2x16float` 와 같은 규칙이고, 창(window)에 f16 으로 들어가는 가중치가 GPU 에서
풀릴 때 기대하는 규칙이다. 2026-09-24 리뷰 전 구현은 NaN 을 inf 로, 65504 를 넘는
유한값 일부를 NaN 으로 만들었고 동률을 올림했다. 기준이 되는 Node(22+) 나 빌드된
`dist` 가 없으면 건너뛴다.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HALF = ROOT / "borch-ts" / "dist" / "src" / "half.js"

SCRIPT = r"""
import { f32ToF16Bit } from %s;
const ref = (v) => new Uint16Array(new Float16Array([v]).buffer)[0];
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
const edges = [0, -0, 1, -1, 1 + 2 ** -11, 1 + 3 * 2 ** -11, 65504, 65519.99, 65520, 65536, 70000, -1e6,
  Infinity, -Infinity, NaN, 2 ** -24, 2 ** -25, 2 ** -25 * 1.0001, 2 ** -26, 2 ** -14, 2 ** -14 - 2 ** -25,
  6.1e-5, 5.96e-8, 1e-8, 3.14159, -2.71828, 0.1, 1234.5678];
const bad = [];
const check = (v) => { const a = f32ToF16Bit(v), b = ref(v);
  if (a !== b && !(Number.isNaN(v) && (a & 0x7c00) === 0x7c00 && (a & 0x3ff) !== 0)) bad.push([v, a, b]); };
edges.forEach(check);
// 모든 지수에 걸친 무작위 f32 비트 패턴 — 반올림 경계는 하위 13비트에 몰려 있다.
let s = 12345;
for (let i = 0; i < 400000; i++) { s = (s * 1103515245 + 12345) >>> 0; u32[0] = s ^ (i << 7); check(f32[0]); }
// 동률(하위 13비트가 정확히 0x1000)만 따로 — 짝수 규칙이 여기서 갈린다.
for (let e = 100; e < 145; e++) for (let m = 0; m < 64; m++) { u32[0] = (e << 23) | (m << 13) | 0x1000; check(f32[0]); }
console.log(JSON.stringify({ bad: bad.slice(0, 10), count: bad.length }));
"""


def test_f32_to_f16_matches_float16array_bit_for_bit():
    node = shutil.which("node")
    if node is None or not HALF.exists():
        pytest.skip("node 또는 borch-ts/dist 가 없다 — npm run build:ts")
    has = subprocess.run([node, "-e", "process.exit(typeof Float16Array === 'function' ? 0 : 1)"])
    if has.returncode != 0:
        pytest.skip("이 Node 에는 Float16Array 가 없다 (22+ 필요)")
    code = SCRIPT % json.dumps(HALF.as_uri())
    out = subprocess.run([node, "--input-type=module", "-e", code], capture_output=True, text=True, check=True)
    got = json.loads(out.stdout)
    assert got["count"] == 0, f"f32ToF16Bit differs from Float16Array on {got['count']} values, e.g. {got['bad']}"
