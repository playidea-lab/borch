"""**The CPU device, checked where there is no GPU.**

`borch-ts/test/cpu.py` proves the device against the WebGPU device on real checkpoints —
and needs a browser, an adapter and the hub to do it. This device exists for the machines
that have none of those, so a check that needs all three is the wrong shape to be the only
one. `borch-ts/test/cpu_node.ts` runs the same wiring against a scalar reference on a
small random network — every node kind, the head, the neighbours — under node, and this
file runs it under `pytest`. No browser, no network, one second.

It is not a browser entry point (`gpu.yml`'s census is about what needs one), and it does
not replace `cpu.py`: the reference here is our own arithmetic in float64, the reference
there is torch's numbers through another device. The two ask different questions.
"""

import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "borch-ts" / "test" / "cpu_node.ts"
EMIT = ROOT / "borch-ts" / "dist" / "test" / "cpu_node.js"
CPU_SRC = ROOT / "borch-ts" / "src" / "cpu"

RUN = """
const { check } = await import(process.env.CPU_NODE);
console.log(JSON.stringify(await check()));
"""


def _stale():
    if not EMIT.exists():
        return f"no emit: {EMIT.relative_to(ROOT)} — first: npm run build:ts"
    newest = max([SOURCE.stat().st_mtime] + [p.stat().st_mtime for p in CPU_SRC.glob("*.ts")])
    if newest > EMIT.stat().st_mtime:
        return "the emit is older than the source — npm run build:ts"
    return None


@pytest.fixture(scope="module")
def report():
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node")
    stale = _stale()
    if stale:
        pytest.fail(stale)
    out = subprocess.run([node, "--input-type=module", "-e", RUN], capture_output=True, text=True,
                         cwd=ROOT, env={"CPU_NODE": EMIT.as_uri(), "PATH": ""}, timeout=120)
    assert out.returncode == 0, f"the node check did not run:\n{out.stderr[-2000:]}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_cpu_device_matches_the_scalar_reference_without_a_gpu(report):
    """Every check the node page reports has to hold. The note carries the measured gap."""
    failed = [f"{c['name']} — {c['note']}" for c in report["checks"] if not c["ok"]]
    assert report["checks"], "the node check reported nothing — absent checks are not a pass"
    assert not failed, "cpu device checks failed:\n  " + "\n  ".join(failed) + "\n\n" + report["text"]


def test_the_node_check_covers_the_three_halves(report):
    """The forward, the head, the neighbours — a check that quietly dropped one would still
    be green above. The names are read here so that it cannot."""
    names = " ".join(c["name"] for c in report["checks"])
    for word in ("forward", "head", "neighbours"):
        assert word in names, f"the node check no longer reports on the {word}"


def test_the_pool_survives_a_long_dispatch_and_a_trapping_worker():
    """`borch-ts/test/threads_check.mjs`: 300 convolution blocks on three workers come out to
    the bit of one thread (the column buffer a worker uses is chosen by its position in the
    chunk, and three does not divide the chunk size), and a worker that traps makes the
    main side throw instead of waiting for a count that never arrives. The second half runs
    in a child with a ten-second limit — the main side spins synchronously, so a hang can
    only be seen from outside it."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node")
    stale = _stale()
    if stale:
        pytest.fail(stale)
    out = subprocess.run([node, str(ROOT / "borch-ts" / "test" / "threads_check.mjs")], capture_output=True, text=True, cwd=ROOT, timeout=180)
    assert out.stdout.strip(), f"the pool check reported nothing:\n{out.stderr[-2000:]}"
    report = json.loads(out.stdout.strip().splitlines()[-1])
    failed = [f"{c['name']} — {c['note']}" for c in report["checks"] if not c["ok"]]
    assert len(report["checks"]) == 2, report
    assert not failed, "pool checks failed:\n  " + "\n  ".join(failed)
