# The workbench notebook — upload images, train on this tab's GPU, review the labels
# the model doubts, and take the model away as ONNX. marimo, in the browser (Pyodide);
# built into `site/marimo/` by `site/build_marimo.py`.
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="borch workbench")


@app.cell
async def _():
    import js, micropip, marimo as mo
    # The wheel sits beside this page. The kernel is a worker whose `location` is its
    # script's — under `<page>/assets/` — so the page's directory is two steps up.
    href = str(js.location.href)
    base = href.split("/assets/")[0] if "/assets/" in href else str(js.location.origin)
    await micropip.install(f"{base}/pyborch-1.9.0-py3-none-any.whl")
    # Two doors, one notebook. `borch_webgpu` boots borch.ts and brings the WebGPU device
    # up; where there is no adapter that fails by name, and `borch_cpu` takes the same
    # bundle without a device — the frozen backbone and the head run on wasm SIMD instead.
    try:
        import borch_webgpu as torch
        cpu = None
        adapter = str(js.borch.Device.adapterInfo)
        boot = f"**borch on `{adapter}`** — `import borch_webgpu as torch` booted borch.ts in this kernel. Nothing was installed on this machine."
    except Exception as no_gpu:  # noqa: BLE001 — the reason is shown, not swallowed
        import borch_cpu as cpu
        torch = None
        adapter = "cpu (no WebGPU adapter)"
        why = str(no_gpu).removeprefix("Error: ").split(" — ")[0][:90]   # the reason, not the whole paragraph
        boot = (f"**borch on the CPU** — no WebGPU adapter here (`{why}`), so `borch_cpu` runs the frozen backbone "
                "and the head on WebAssembly SIMD, one thread. Slower, same numbers; the small-CNN path and the ONNX export need the GPU.")
    mo.md(boot)
    return adapter, cpu, mo, torch


@app.cell
def _(mo):
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".zip"], multiple=True, label="Images to review — files (label = the part of the name before the first '_') or a zipped folder (label = the folder)")
    mo.vstack([mo.md("### 1 · Images"), upload, mo.md("A zip of class folders is the way to bring thousands: nothing is decoded until a batch asks for it. Nothing uploaded? A synthetic set of three classes is used, so the notebook runs as it is.")])
    return (upload,)


@app.cell
def _(mo, torch):
    FROZEN, SCRATCH = "frozen backbone + head", "small CNN from scratch"
    path = mo.ui.radio(options=[FROZEN, SCRATCH] if torch is not None else [FROZEN], value=FROZEN, label="Model")
    mo.vstack([path, mo.md("*Frozen backbone*: ImageNet EfficientNet-B0 stays as it is, one feature vector per image is computed once, and only a linear head learns — the field trainer's path, hundreds of images are enough. *From scratch*: a small CNN, every weight learns.")])
    return FROZEN, SCRATCH, path


@app.cell
def _(FROZEN, cpu, path, torch, upload):
    import io
    import numpy as np
    from PIL import Image
    SIDE = 224 if path.value == FROZEN else 64            # the backbone was trained at 224
    if upload.value:
        # Files or a zipped folder → `torch.ImageFiles` (the numpy core's, so `borch_cpu` has it too): names, labels and classes now,
        # pixels only when a batch asks. Five thousand camera images stay as their bytes.
        ds = (torch or cpu).ImageFiles(upload.value, size=SIDE)
    else:
        # Three classes: a low-frequency template each, plus noise — the same set
        # tests/browser/envelope2.html trains on — written as ninety PNGs, so the
        # synthetic path goes through the same dataset as an upload.
        rng = np.random.default_rng(7)
        cells, S0 = 6, 64
        templates = [rng.standard_normal((cells, cells, 3)).astype(np.float32) for _ in "abc"]
        idx = np.arange(S0) * cells // S0
        files = []
        for i in range(90):
            k = i % 3
            img = 0.5 + 0.3 * templates[k][idx][:, idx] + 0.15 * rng.standard_normal((S0, S0, 3)).astype(np.float32)
            buf = io.BytesIO(); Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).save(buf, format="PNG")
            files.append((f"{'abc'[k]}_{i:03d}.png", buf.getvalue()))
        ds = (torch or cpu).ImageFiles(files, size=SIDE)
    CLASSES, names, labels, y = ds.classes, ds.names, ds.labels, ds.targets
    return CLASSES, Image, ds, io, labels, names, np, y


@app.cell
def _(CLASSES, FROZEN, cpu, ds, mo, np, path, torch, y):
    # 2 · Train. Two paths, one output: `model` (what is exported), `feats` (what the
    # review ranks on), `pred`, and a line saying how well the labels are agreed with.
    import time
    K, N = len(CLASSES), len(ds)
    t0 = time.perf_counter(); losses = []
    if torch is None:
        # No adapter: the same frozen backbone through `bimm.cpuGraphFor` + `cpu.CpuRunner`,
        # the same head through `cpu.LinearHead` — full-batch SGD with momentum (there is
        # no Adam on this side; the numbers land within 3.5e-5 of torch's step for step).
        backbone = cpu.load("imagenet-efficientnet-b0", features=True)
        chunks = [backbone.features(xb) for xb, _idx in ds.batches(16)]
        feats = np.concatenate(chunks)                     # (N, 1280)
        feat_s = time.perf_counter() - t0
        head = cpu.LinearHead(backbone.num_features, K, lr=0.05, momentum=0.9)
        all_losses = head.fit(feats, y, steps=300)
        losses = [float(all_losses[i]) for i in list(range(0, 300, 50)) + [299]]
        pred = head.predict(feats).argmax(1)
        model = None                                       # nothing to export as ONNX on this side — the head's weights are in `head`
        how = f"EfficientNet-B0 frozen, on the CPU · features for {N} images in **{feat_s:.1f} s** · head 300 steps"
    elif path.value == FROZEN:
        nn = torch.nn
        crit = torch.nn.CrossEntropyLoss()
        # The backbone runs forward only, once per image, in batches of 16; what is
        # kept is the 1280-d vector before its classifier (`pre_logits`). The head is
        # the only thing that learns — 300 full-batch steps on the cached features.
        backbone = torch.hub.load("imagenet-efficientnet-b0")
        chunks = []                                        # marimo: one name per cell, `rows` is the table's
        with torch.no_grad():
            for xb, _idx in ds.batches(16):                # decoded here, sixteen at a time
                with torch.scope():
                    maps = backbone.forward_features(torch.tensor(xb))
                    chunks.append(backbone.forward_head(maps, pre_logits=True).numpy())
        feats = np.concatenate(chunks)                     # (N, 1280)
        feat_s = time.perf_counter() - t0
        head = nn.Linear(backbone.num_features, K)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        Ft, yt = torch.tensor(feats), torch.tensor(y)
        for step in range(300):
            with torch.scope():
                opt.zero_grad(); loss = crit(head(Ft), yt); loss.backward(); opt.step()
                if step % 50 == 0 or step == 299:
                    losses.append(loss.item())
        class Frozen(nn.Module):
            # backbone → pre-logits → head, as one module, so the export is the whole thing
            def __init__(self):
                super().__init__()
                self.head = head
            def forward(self, x):
                return self.head(backbone.forward_head(backbone.forward_features(x), pre_logits=True))
        model = Frozen()
        model.eval()
        with torch.no_grad():
            pred = head(Ft).argmax(1).numpy()
        how = f"EfficientNet-B0 frozen · features for {N} images in **{feat_s:.1f} s** · head 300 steps"
    else:
        nn = torch.nn
        crit = torch.nn.CrossEntropyLoss()
        head = None                                        # the small CNN has no separate head to hand on
        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]
        model = nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, K))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        Xt, yt = torch.tensor(ds.stack()), torch.tensor(y)  # all of it, at 64 px
        BATCH, EPOCHS = 16, 12
        steps = max(1, N // BATCH)
        for epoch in range(EPOCHS):
            for s in range(steps):
                xb = Xt[s * BATCH:(s + 1) * BATCH]; yb = yt[s * BATCH:(s + 1) * BATCH]
                with torch.scope():
                    opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
                    last = loss.item()          # read inside the scope — outside it the buffer is gone
            losses.append(last)
        model.eval()
        with torch.no_grad():
            logits = model(Xt)
            pred = logits.argmax(1).numpy()
            feats = logits.numpy()
        how = f"small CNN · {EPOCHS} epochs × {steps} steps"
    train_s = time.perf_counter() - t0
    acc = float((pred == y).mean())
    mo.md(f"### 2 · Trained\n{how} in **{train_s:.1f} s** · loss {losses[0]:.2f} → {losses[-1]:.3f} · agrees with the given labels on **{acc * 100:.0f}%**")
    return acc, feats, head, how, model, pred, train_s


@app.cell
def _(CLASSES, Image, cpu, ds, feats, io, labels, mo, names, np, pred, torch, y):
    # 3 · Review — the labels the model doubts, first. `torch.suspects` scores each
    # image by how many of its five nearest neighbours (cosine, on the model's
    # features) carry a different label: a wrong label sits among images that disagree.
    # The score is the numpy core's, so it is the same function on either side.
    # Five neighbours for a few hundred images; twenty at a few thousand — measured on
    # CIFAR-100N with 10 % of labels flipped, AUROC 0.913 at k=5 and 0.962 at k=20.
    suspect = (torch or cpu).suspects(feats, y, k=5 if len(y) < 2000 else 20)
    order = np.argsort(-suspect)
    def thumb(i):
        buf = io.BytesIO(); Image.fromarray(ds.thumb(int(i), 56)).save(buf, format="PNG"); return buf.getvalue()
    rows = [{"file": names[i], "image": mo.image(thumb(i), width=56), "given": labels[i], "predicted": CLASSES[int(pred[i])],
             "suspect": round(float(suspect[i]), 2)} for i in order]
    table = mo.ui.table(rows, selection="single", page_size=10, label="review queue — most doubted first")
    mo.vstack([mo.md("### 3 · Review"), table])
    return suspect, table


@app.cell
def _(mo, table):
    picked = table.value
    mo.md(f"picked: **{picked[0]['file']}** — given `{picked[0]['given']}`, predicted `{picked[0]['predicted']}`" if picked else "*pick a row above to inspect it*")
    return


@app.cell
def _(ds, head, mo, model, torch):
    # 4 · Take it away — the trained model as ONNX, the file every serving runtime reads.
    if torch is None:
        # The CPU side exports no ONNX (the graph is bimm's, not a traced module). What
        # leaves is the head — weight `[classes × 1280]` and bias, torch's order, as JSON —
        # to be laid on the same EfficientNet-B0 wherever it is served. (A marimo cell has
        # no early return, so both sides build `_view` and the cell shows it.)
        import json as _json
        _st = head.state_dict()
        _head_json = _json.dumps({"backbone": "imagenet-efficientnet-b0", "weight": _st["weight"].round(6).tolist(), "bias": _st["bias"].round(6).tolist()})
        _view = mo.vstack([mo.md(f"### 4 · Export\n{len(_head_json) / 1e3:.0f} KB of head weights (JSON) — the ONNX export runs on the WebGPU side"), mo.download(data=lambda: _head_json, filename="head.json", label="Download head.json")])
    else:
        sample = torch.tensor(ds[0][0][None])
        data = torch.onnx.export(model, sample)
        # The frozen path's file is the whole backbone, 16 MB. Handed to `mo.download` as
        # bytes it is written into the cell's output as base64 and the cell never comes back
        # (measured); as a callable it is fetched once, on the click.
        _view = mo.vstack([mo.md(f"### 4 · Export\n{len(data) / 1e3:.0f} KB of ONNX (`torch.onnx.export`)"), mo.download(data=lambda: data, filename="model.onnx", label="Download model.onnx")])
    _view
    return


@app.cell
def _(acc, ds, how, mo, path, torch, train_s):
    # 5 · Report — the machine and the run as one file, for whoever is asked "it does not
    # work". `torch.report` gathers the adapter, the faults, memory, the wheel and bundle
    # versions; the facts of this run ride along. No file names, no pixels.
    import json
    if torch is None:
        rep = {"adapter": "cpu (no WebGPU adapter)", "faults": 0, "warnings": ["no WebGPU adapter — the frozen backbone and the head ran on wasm SIMD, one thread; the small CNN and the ONNX export were not available"],
               "images": len(ds), "classes": len(ds.classes), "model": path.value, "train_s": round(train_s, 2), "accuracy": round(acc, 4), "how": how}
    else:
        rep = torch.report(images=len(ds), classes=len(ds.classes), model=path.value, train_s=round(train_s, 2), accuracy=round(acc, 4), how=how)
    text = json.dumps(rep, indent=2)
    warn = rep["warnings"]
    rep_head = mo.md(f"### 5 · Report\n`{rep['adapter']}` · faults {rep['faults']} · warnings {len(warn)}" + (" — **read them before the numbers**" if warn else ""))
    rep_body = mo.callout(mo.md("\n".join(f"- {w}" for w in warn)), kind="warn") if warn else mo.md("Nothing is off: a real GPU, no validation fault, the device alive.")
    mo.vstack([rep_head, rep_body, mo.download(data=lambda: text, filename="report.json", label="Download report.json")])
    return


if __name__ == "__main__":
    app.run()
