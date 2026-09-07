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
        workers = cpu.threads()
        boot = (f"**borch on the CPU{f', {workers} workers' if workers else ''}** — no WebGPU adapter here (`{why}`), so `borch_cpu` runs the frozen backbone "
                f"and the head on WebAssembly SIMD, {f'{workers} threads' if workers else 'one thread'}. Slower, same numbers; the small-CNN path and the ONNX export need the GPU.")
    mo.md(boot)
    return adapter, cpu, mo, torch


@app.cell
def _(mo):
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".zip"], multiple=True, label="Images to review — files (label = the part of the name before the first '_') or a zipped folder (label = the folder)")
    mo.vstack([mo.md("### 1 · Images"), upload, mo.md("A zip of class folders is the way to bring thousands: nothing is decoded until a batch asks for it. **Segmentation**: a zip with `images/` and `masks/` — the same file names in both, a mask white where the object is — and the notebook trains a U-Net and reviews the masks instead. Nothing uploaded? A synthetic set of three classes is used, so the notebook runs as it is.")])
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
    from borch._data import image_entries                  # the core's zip reader — both sides carry the core
    masks = None                                           # a segmentation set carries one per image
    entries = image_entries(upload.value) if upload.value else []
    pairs = {name.split("/", 1)[1]: data for name, data in entries if name.startswith("images/")}
    mask_of = {name.split("/", 1)[1]: data for name, data in entries if name.startswith("masks/")}
    if pairs and mask_of:
        # `images/` + `masks/`, matched by name: a segmentation set. 96 px (measured on
        # Kvasir-SEG: 800 images × 30 epochs in 34 s on a laptop GPU), the U-Net path.
        SIDE = 96
        keep = sorted(k for k in pairs if k in mask_of)
        ds = (torch or cpu).ImageFiles([(k, pairs[k]) for k in keep], size=SIDE, label=lambda _n: "image")
        masks = (torch or cpu).ImageFiles([(k, mask_of[k]) for k in keep], size=SIDE, label=lambda _n: "mask")
    elif upload.value:
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
    return CLASSES, Image, ds, io, labels, masks, names, np, y


@app.cell
def _(CLASSES, FROZEN, cpu, ds, masks, mo, np, path, torch, y):
    # 2 · Train. Three paths, one output: `model` (what is exported), `feats` (what the
    # review ranks on), `pred`, and a line saying how well the labels are agreed with.
    import time
    K, N = len(CLASSES), len(ds)
    t0 = time.perf_counter(); losses = []
    if masks is not None and torch is None:
        raise RuntimeError("segmentation needs the WebGPU device — the CPU side runs the frozen backbone and the head only")
    if masks is not None:
        # Segmentation: a three-level U-Net from scratch on the given masks (the model
        # of tests/seg_eval.py, where it is measured against torch), one logit per pixel.
        # `pred` is the model's mask per image; `feats` carries the given masks — the
        # review ranks masks by their disagreement with the model's, not by neighbours.
        nn = torch.nn
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        class UNet(nn.Module):
            def __init__(self, w=16):
                super().__init__()
                self.e1, self.e2, self.e3 = block(3, w), block(w, 2 * w), block(2 * w, 4 * w)
                self.pool = nn.MaxPool2d(2)
                self.u2, self.u1 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2), nn.ConvTranspose2d(2 * w, w, 2, stride=2)
                self.d2, self.d1 = block(4 * w, 2 * w), block(2 * w, w)
                self.out = nn.Conv2d(w, 1, 1)
            def forward(self, x):
                a = self.e1(x); b = self.e2(self.pool(a)); c = self.e3(self.pool(b))
                y_ = self.d2(torch.cat([self.u2(c), b], 1))
                return self.out(self.d1(torch.cat([self.u1(y_), a], 1)))
        model = UNet()
        head = None
        crit = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        Xt = ds.stack()                                            # (N, 3, 96, 96)
        Mt = (masks.stack()[:, :1] > 0.5).astype(np.float32)       # (N, 1, 96, 96): white is the object
        BATCH = 16
        EPOCHS = 30 if N <= 1000 else 15
        steps = max(1, N // BATCH)
        shuffled = np.random.default_rng(0).permutation(N)
        # The first step is recorded under `torch.capture()` and every other step replays
        # it — the same dispatches, the next batch copied into the same input buffers,
        # no Python between the kernels — after `fuse()` merged the elementwise chains into
        # single kernels. The eager step to a rounding (measured), a sixth faster on a laptop GPU.
        xb = torch.tensor(Xt[shuffled[:BATCH]]); mb = torch.tensor(Mt[shuffled[:BATCH]])
        with torch.capture() as rec:
            with torch.scope():
                opt.zero_grad(); loss = crit(model(xb), mb); loss.backward(); opt.step()
        rec.fuse()
        for epoch in range(EPOCHS):
            for s in range(steps):
                if epoch == 0 and s == 0:
                    continue                                   # recorded above
                take = shuffled[s * BATCH:(s + 1) * BATCH]
                xb.copy_(torch.tensor(Xt[take])); mb.copy_(torch.tensor(Mt[take]))
                rec.replay()
            losses.append(loss.item())
        rec.dispose()
        model.eval()
        pred = np.zeros((N, 1, ds.size, ds.size), np.float32)
        with torch.no_grad():
            for s in range(0, N, 32):
                with torch.scope():
                    pred[s:s + 32] = (model(torch.tensor(Xt[s:s + 32])).numpy() > 0)
        feats = Mt
        hit = (pred * Mt).sum(axis=(1, 2, 3)); joined = ((pred + Mt) > 0).sum(axis=(1, 2, 3))
        ious = np.where(joined > 0, hit / np.maximum(joined, 1), 1.0)
        acc = float(ious.mean())
        how = f"U-Net (width 16) from scratch at {ds.size} px · {EPOCHS} epochs × {steps} steps"
        headline = f"{how} in **{time.perf_counter() - t0:.1f} s** · loss {losses[0]:.3f} → {losses[-1]:.3f} · mean IoU with the given masks **{acc:.2f}**"
    elif torch is None:
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
        # Full-batch: the input never changes, so the recorded step replays as it is.
        with torch.capture() as rec:
            with torch.scope():
                opt.zero_grad(); loss = crit(head(Ft), yt); loss.backward(); opt.step()
        rec.fuse()
        losses.append(loss.item())
        for step in range(1, 300):
            rec.replay()
            if step % 50 == 0 or step == 299:
                losses.append(loss.item())
        rec.dispose()
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
        # One step recorded, the rest replayed with each batch copied into the captured
        # inputs — see the U-Net path above for why.
        xb = torch.tensor(ds.stack()[:BATCH]); yb = torch.tensor(y[:BATCH])
        with torch.capture() as rec:
            with torch.scope():
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        rec.fuse()
        for epoch in range(EPOCHS):
            for s in range(steps):
                if epoch == 0 and s == 0:
                    continue
                xb.copy_(Xt[s * BATCH:(s + 1) * BATCH]); yb.copy_(yt[s * BATCH:(s + 1) * BATCH])
                rec.replay()
            losses.append(loss.item())
        rec.dispose()
        model.eval()
        with torch.no_grad():
            logits = model(Xt)
            pred = logits.argmax(1).numpy()
            feats = logits.numpy()
        how = f"small CNN · {EPOCHS} epochs × {steps} steps"
    train_s = time.perf_counter() - t0
    if masks is None:
        acc = float((pred == y).mean())
        headline = f"{how} in **{train_s:.1f} s** · loss {losses[0]:.2f} → {losses[-1]:.3f} · agrees with the given labels on **{acc * 100:.0f}%**"
    mo.md(f"### 2 · Trained\n{headline}")
    return acc, feats, head, how, model, pred, train_s


@app.cell
def _(CLASSES, Image, cpu, ds, feats, io, labels, masks, mo, names, np, pred, torch, y):
    # 3 · Review — the labels the model doubts, first. `torch.suspects` scores each
    # image by how many of its five nearest neighbours (cosine, on the model's
    # features) carry a different label: a wrong label sits among images that disagree.
    # The score is the numpy core's, so it is the same function on either side.
    # Five neighbours for a few hundred images; twenty at a few thousand — measured on
    # CIFAR-100N with 10 % of labels flipped, AUROC 0.913 at k=5 and 0.962 at k=20.
    # A mask has no neighbourhood to vote in: its score is one minus the IoU between
    # the given mask and the model's — the mask the model could not learn to reproduce
    # (measured on Kvasir-SEG with a fifth of the masks swapped or shifted: AUROC 0.90,
    # a third of the queue for 90 % of the wrong ones).
    def png(arr):
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); return buf.getvalue()
    def thumb(i):
        return png(ds.thumb(int(i), 56))
    if masks is not None:
        given_m = feats
        inter = (pred * given_m).sum(axis=(1, 2, 3)); union = ((pred + given_m) > 0).sum(axis=(1, 2, 3))
        suspect = 1 - np.where(union > 0, inter / np.maximum(union, 1), 1.0)
        order = np.argsort(-suspect)
        def mask_png(m):
            return png(np.asarray(Image.fromarray((m[0] * 255).astype("uint8")).resize((56, 56))))
        rows = [{"file": names[i], "image": mo.image(thumb(i), width=56), "given": mo.image(mask_png(given_m[i]), width=56),
                 "predicted": mo.image(mask_png(pred[i]), width=56), "suspect": round(float(suspect[i]), 2)} for i in order]
    else:
        suspect = (torch or cpu).suspects(feats, y, k=5 if len(y) < 2000 else 20)
        order = np.argsort(-suspect)
        rows = [{"file": names[i], "image": mo.image(thumb(i), width=56), "given": labels[i], "predicted": CLASSES[int(pred[i])],
                 "suspect": round(float(suspect[i]), 2)} for i in order]
    table = mo.ui.table(rows, selection="single", page_size=10, label="review queue — most doubted first" + (" (given mask · model's mask · 1 − IoU)" if masks is not None else ""))
    mo.vstack([mo.md("### 3 · Review"), table])
    return suspect, table


@app.cell
def _(mo, table):
    picked = table.value
    def _text(v):
        return v if isinstance(v, str) else "mask"
    mo.md(f"picked: **{picked[0]['file']}** — given `{_text(picked[0]['given'])}`, predicted `{_text(picked[0]['predicted'])}` · suspect {picked[0]['suspect']}" if picked else "*pick a row above to inspect it*")
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
def _(acc, ds, how, masks, mo, path, torch, train_s):
    # 5 · Report — the machine and the run as one file, for whoever is asked "it does not
    # work". `torch.report` gathers the adapter, the faults, memory, the wheel and bundle
    # versions; the facts of this run ride along. No file names, no pixels.
    import json
    if torch is None:
        rep = {"adapter": "cpu (no WebGPU adapter)", "faults": 0, "warnings": ["no WebGPU adapter — the frozen backbone and the head ran on wasm SIMD, one thread; the small CNN and the ONNX export were not available"],
               "images": len(ds), "classes": len(ds.classes), "model": path.value, "train_s": round(train_s, 2), "accuracy": round(acc, 4), "how": how}
    else:
        rep = (torch.report(images=len(ds), task="segmentation", model="U-Net", train_s=round(train_s, 2), mean_iou=round(acc, 4), how=how) if masks is not None
               else torch.report(images=len(ds), classes=len(ds.classes), model=path.value, train_s=round(train_s, 2), accuracy=round(acc, 4), how=how))
    text = json.dumps(rep, indent=2)
    warn = rep["warnings"]
    rep_head = mo.md(f"### 5 · Report\n`{rep['adapter']}` · faults {rep['faults']} · warnings {len(warn)}" + (" — **read them before the numbers**" if warn else ""))
    rep_body = mo.callout(mo.md("\n".join(f"- {w}" for w in warn)), kind="warn") if warn else mo.md("Nothing is off: a real GPU, no validation fault, the device alive.")
    mo.vstack([rep_head, rep_body, mo.download(data=lambda: text, filename="report.json", label="Download report.json")])
    return


if __name__ == "__main__":
    app.run()
