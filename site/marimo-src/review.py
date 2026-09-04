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
    await micropip.install(f"{base}/pyborch-1.6.0-py3-none-any.whl")
    import borch_webgpu as torch
    adapter = str(js.borch.Device.adapterInfo)
    mo.md(f"**borch on `{adapter}`** — `import borch_webgpu as torch` booted borch.ts in this kernel. Nothing was installed on this machine.")
    return adapter, mo, torch


@app.cell
def _(mo):
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg"], multiple=True, label="Images to review (label = the part of the file name before the first '_')")
    mo.vstack([mo.md("### 1 · Images"), upload, mo.md("Nothing uploaded? A synthetic set of three classes is used, so the notebook runs as it is.")])
    return (upload,)


@app.cell
def _(torch, upload):
    import numpy as np
    SIDE = 64
    if upload.value:
        # Files in hand → NCHW in [0, 1], labels from the names: `torch.decode_images`.
        X, y, names, CLASSES = torch.decode_images(upload.value, size=SIDE)
        labels = [CLASSES[i] for i in y]
    else:
        # Three classes: a low-frequency template each, plus noise — the same set
        # tests/browser/envelope2.html trains on.
        CLASSES = ["a", "b", "c"]
        rng = np.random.default_rng(7)
        cells = 6
        templates = [rng.standard_normal((cells, cells, 3)).astype(np.float32) for _ in CLASSES]
        idx = np.arange(SIDE) * cells // SIDE
        names, images, labels = [], [], []
        for i in range(90):
            k = i % 3
            img = 0.5 + 0.3 * templates[k][idx][:, idx] + 0.15 * rng.standard_normal((SIDE, SIDE, 3)).astype(np.float32)
            names.append(f"{CLASSES[k]}_{i:03d}.png"); images.append(np.clip(img, 0, 1)); labels.append(CLASSES[k])
        X = np.stack(images).transpose(0, 3, 1, 2).astype(np.float32)        # N, 3, S, S
        y = np.array([CLASSES.index(l) for l in labels], dtype=np.int64)
    return CLASSES, X, labels, names, np, y


@app.cell
def _(CLASSES, X, mo, np, torch, y):
    # 2 · Train — a small CNN written as torch writes it. Swap this cell for your own model.
    import time
    def block(cin, cout):
        return [torch.nn.Conv2d(cin, cout, 3, padding=1), torch.nn.BatchNorm2d(cout), torch.nn.ReLU(), torch.nn.MaxPool2d(2)]
    model = torch.nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(64, len(CLASSES)))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    BATCH, EPOCHS = 16, 12
    steps = max(1, len(X) // BATCH)
    t0 = time.perf_counter(); losses = []
    for epoch in range(EPOCHS):
        for s in range(steps):
            xb = Xt[s * BATCH:(s + 1) * BATCH]; yb = yt[s * BATCH:(s + 1) * BATCH]
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        losses.append(loss.item())
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        logits = model(Xt)
        pred = logits.argmax(1).numpy()
        feats = logits.numpy()
    acc = float((pred == y).mean())
    mo.md(f"### 2 · Trained\n{EPOCHS} epochs × {steps} steps in **{train_s:.1f} s** · loss {losses[0]:.2f} → {losses[-1]:.3f} · agrees with the given labels on **{acc * 100:.0f}%**")
    return acc, feats, model, pred


@app.cell
def _(CLASSES, X, feats, labels, mo, names, np, pred, torch, y):
    # 3 · Review — the labels the model doubts, first. `torch.suspects` scores each
    # image by how many of its five nearest neighbours (cosine, on the model's
    # features) carry a different label: a wrong label sits among images that disagree.
    import io
    from PIL import Image          # named here so marimo loads Pillow for the thumbnails
    suspect = torch.suspects(feats, y, k=5)
    order = np.argsort(-suspect)
    def thumb(i):
        buf = io.BytesIO(); Image.fromarray((X[i].transpose(1, 2, 0) * 255).astype("uint8")).save(buf, format="PNG"); return buf.getvalue()
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
def _(X, mo, model, torch):
    # 4 · Take it away — the trained model as ONNX, the file every serving runtime reads.
    sample = torch.tensor(X[:1])
    data = torch.onnx.export(model, sample)
    mo.vstack([mo.md(f"### 4 · Export\n{len(data) / 1e3:.0f} KB of ONNX (`torch.onnx.export`)"), mo.download(data=data, filename="model.onnx", label="Download model.onnx")])
    return


if __name__ == "__main__":
    app.run()
