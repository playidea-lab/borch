"""Whether a checkpoint round-trips in a real browser, and whether a resume continues.

    npm run build:ts
    uv run --with playwright python borch-ts/test/serialize.py [--headed]

**The round trip alone is not enough.** Saving and reading back and finding the values
equal asks about the codec and nothing else. The real question comes after it — *is
training that was stopped and resumed the same as training that never stopped.* Leave out
one momentum buffer, one step counter, one scheduler epoch, and the round trip stays green
while the resume alone parts.

All of it is deterministic, so **it has to be equal bit for bit.** There is no tolerance in
this runner.

The path that restores the weights and throws the rest away is run alongside — **it has to
part** for the equivalence check above to be measuring anything.
"""

import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/serialize.html"
TIMEOUT_MS = 300_000


def main(argv):
    # **A stale emit is as bad as none** — edit the source, forget the build, and you
    # measure the old code. `require_fresh_dist` watches that place (`run.py`).
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "serialize.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        # **`with` closes it too** — put on the last line instead, an exception before
        # it leaves it open, and the leftover Chromium ruins another measurement.
        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  [browser] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}"))
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__borchSerialize !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchSerialize")
    finally:
        stop()

    if "error" in result:
        print(f"**the checkpoint check blew up**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"adapter: {result.get('adapter', '(unknown)')}")
    print(result["text"])
    if verdict(result, "checkpoints") or not cross_language(result.get("sample")):
        return 1
    return 0 if cross_tree(result.get("nested")) else 1


def cross_language(sample):
    """Take the browser's file apart **with numpy alone.** Not one line of borch code.

    Our codec round-tripping with our codec would work in a format of our own. What
    carrying safetensors buys is that **somebody else can read it**, and whether that
    claim is true is confirmed here and nowhere else.
    """
    import json
    import struct

    import numpy as np

    if not sample:
        print("no sample — the page did not hand out sample()", file=sys.stderr)
        return False

    blob = bytes(bytearray(sample))
    (head_len,) = struct.unpack_from("<Q", blob, 0)
    header = json.loads(blob[8:8 + head_len])
    body = blob[8 + head_len:]

    got = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        # The dtype is always F32. borch's int64 and bool are labels and are written in
        # the header alone — write I64 in the body and it disagrees with four bytes, and
        # this line breaks.
        assert entry["dtype"] == "F32", entry["dtype"]
        got[name] = np.frombuffer(body[begin:end], dtype="<f4").reshape(entry["shape"])

    want = {
        "fc.weight": np.array([[1.5, -2.25, 0.5], [7.0, -0.125, 3.0]], dtype="<f4"),
        "fc.labels": np.array([3.0, 1.0, 4.0], dtype="<f4"),
    }
    for name, expected in want.items():
        if name not in got or not np.array_equal(got[name], expected):
            print(f"**the value numpy read differs** — {name}: {got.get(name)}",
                  file=sys.stderr)
            return False

    labels = header["__metadata__"].get("borch.dtype:fc.labels")
    if labels != "int64":
        print(f"**the dtype label did not ride along** — {labels}", file=sys.stderr)
        return False

    print(f"  ✓ numpy reads the same file — {len(got)} tensors, "
          f"label fc.labels={labels}")
    return cross_library(blob)


def cross_library(blob):
    """The browser's file, read by **Python `borch`.**

    `serialize.ts`'s opening paragraph gives as its reason for choosing safetensors that
    "Python `borch`, numpy and the HF tools read the same file". **That sentence was false
    for a long time** — `save`/`load` on the Python side were pickle, so the road from
    training in a browser to carrying the result to your own machine was closed. That road
    is the only reason this project chose the format.

    The numpy check above cannot see this. That one asks whether the **format** is open and
    this one asks whether **our Python code actually opens that door** — with the format
    right and no function to read it, there is nothing there as far as a user is
    concerned.
    """
    import pathlib
    import sys as _sys
    import tempfile

    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    import borch

    path = pathlib.Path(tempfile.mkdtemp()) / "from_browser.bin"
    path.write_bytes(blob)
    # The top level is a dictionary of tensors, so the same thing comes out with or
    # without the tree.
    got = borch.load(path)
    if "fc.weight" not in got:
        print(f"**borch could not read it** — keys {sorted(got)}", file=_sys.stderr)
        return False
    first = float(got["fc.weight"].data.reshape(-1)[0])
    if abs(first - 1.5) > 1e-6:
        print(f"**the value borch read differs** — {first}", file=_sys.stderr)
        return False
    print(f"  ✓ Python borch reads the browser's file — fc.weight[0]={first}")
    return True


def cross_tree(nested):
    """Whether Python `borch` reads the browser's **nested** file with its shape intact.

    There are two copies of the tree scheme (`borch.tree`, nodes `T`/`d`/`l`/`j`) now —
    `serialize.ts` and `_serialize.py`. They are supposed to write the same letters, and
    **nobody measured that promise.** Mend one side alone and a checkpoint written by one
    cannot be read by the other, and what comes out then is not an exception but **a
    dictionary of a different shape**, which is found far later.

    `cross_library` above cannot see it. That sample's top level is a dictionary of
    tensors, so the same thing comes out with or without the tree — ask only about the flat
    case and the tree is never trodden on.
    """
    import pathlib
    import sys as _sys
    import tempfile

    if not nested:
        print("no nested sample — the page did not hand out sampleNested()",
              file=_sys.stderr)
        return False

    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    import borch

    path = pathlib.Path(tempfile.mkdtemp()) / "nested_from_browser.bin"
    path.write_bytes(bytes(bytearray(nested)))
    got = borch.load(path)

    def fail(said):
        print(f"**{said}**", file=_sys.stderr)
        return False

    if not isinstance(got, dict) or sorted(got) != [
            "done", "epoch", "model", "note", "nothing", "steps"]:
        return fail("the keys differ — "
                    f"{sorted(got) if isinstance(got, dict) else type(got)}")
    if not isinstance(got["model"], dict) or "fc.weight" not in got["model"]:
        # Split on the dot again and `{"fc": {"weight": …}}` comes out here.
        return fail(f"the nesting did not arrive — model={got['model']}")
    if not isinstance(got["steps"], list) or len(got["steps"]) != 2:
        return fail(f"the array did not arrive — steps={got['steps']}")
    if float(got["steps"][0].data.reshape(-1)[0]) != 7.0 or got["steps"][1] != 3:
        return fail(f"the array's contents differ — {got['steps']}")
    if (got["epoch"], got["note"], got["done"], got["nothing"]) != (
            5, "nested", False, None):
        return fail("the values that are not tensors differ — "
                    f"{got['epoch']} {got['note']} {got['done']} {got['nothing']}")
    if float(got["model"]["fc.weight"].data.reshape(-1)[0]) != 1.5:
        return fail(f"the value differs — {got['model']['fc.weight'].data}")

    print("  ✓ Python borch reads the browser's **nested** file with its shape intact")
    return True


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
