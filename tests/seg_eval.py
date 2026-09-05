"""Segmentation, the same two questions asked of classification: does the browser learn
what torch learns, and does a review queue put the wrong masks first? Measured on
Kvasir-SEG, a thousand endoscopy images with a polyp mask each.

    uv run --project ~/git/borch python tests/seg_eval.py prepare          # → kvasir_96.npz
    uv run --project ~/git/borch python tests/seg_eval.py [--epochs=30] [--train=800] [--noise=0.2]

A mask has no neighbourhood to vote in, so the queue for segmentation is a different
signal from `suspects`: the model is trained on the given masks, and each training
image is scored by how far the model's own mask is from the one it was given (one minus
IoU). A wrong mask is the one the model could not learn to reproduce. The noise here is
the two ways a mask goes wrong at a bench: swapped with another image's mask, and shifted
off its object by a third of the width; `--noise` is the share of training masks hit,
half each.

The code is written against a torch-shaped library `L`, so `run(torch, ...)` here and
`run(borch_webgpu, ...)` in the browser (tests/browser/seg_eval_py.py) are the same
lines. Numbers go to tests/seg_eval.json.
"""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from review_eval import auroc, queue_metrics  # noqa: E402

DATA = pathlib.Path.home() / "data" / "kvasir"
NPZ = DATA / "kvasir_96.npz"
OUT = HERE / "seg_eval.json"
SIZE = 96


def prepare():
    """Kvasir-SEG's thousand pairs at 96×96 as one npz: images uint8 NCHW, masks 0/1."""
    from PIL import Image
    root = DATA / "Kvasir-SEG"
    names = sorted(p.stem for p in (root / "images").glob("*.jpg"))
    x = np.stack([np.asarray(Image.open(root / "images" / f"{n}.jpg").convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.uint8) for n in names])
    m = np.stack([np.asarray(Image.open(root / "masks" / f"{n}.jpg").convert("L").resize((SIZE, SIZE), Image.BILINEAR), np.uint8) for n in names])
    np.savez_compressed(NPZ, x=x.transpose(0, 3, 1, 2), m=(m > 127).astype(np.uint8))
    print(f"{len(names)} pairs → {NPZ} ({NPZ.stat().st_size / 1e6:.1f} MB)")


def load(blob=None):
    """(images float32 NCHW in [0,1], masks float32 N1HW) — from the npz, or from its bytes."""
    import io
    z = np.load(io.BytesIO(blob) if blob is not None else NPZ)
    return z["x"].astype(np.float32) / 255, z["m"][:, None].astype(np.float32)


def corrupt(masks, noise, rng):
    """`noise` of the masks go wrong, half swapped with another image's, half shifted by a
    third of the width. Returns (masks, wrong flags)."""
    given = masks.copy()
    n = len(masks)
    hit = rng.choice(n, size=int(round(noise * n)), replace=False)
    swap, shift = hit[: len(hit) // 2], hit[len(hit) // 2:]
    given[swap] = masks[(swap + rng.integers(1, n, len(swap))) % n]
    given[shift] = np.roll(masks[shift], SIZE // 3, axis=3)
    wrong = np.zeros(n, bool); wrong[hit] = True
    return given, wrong


def build_unet(L, width=16):
    """Three levels, a skip at each — the shape every segmentation model still has."""
    def block(i, o):
        return L.nn.Sequential(L.nn.Conv2d(i, o, 3, padding=1), L.nn.BatchNorm2d(o), L.nn.ReLU(),
                               L.nn.Conv2d(o, o, 3, padding=1), L.nn.BatchNorm2d(o), L.nn.ReLU())

    class UNet(L.nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = block(3, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.pool = L.nn.MaxPool2d(2)
            self.u2, self.u1 = L.nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2), L.nn.ConvTranspose2d(2 * w, w, 2, stride=2)
            self.d2, self.d1 = block(4 * w, 2 * w), block(2 * w, w)
            self.out = L.nn.Conv2d(w, 1, 1)

        def forward(self, x):
            a = self.e1(x)
            b = self.e2(self.pool(a))
            c = self.e3(self.pool(b))
            y = self.d2(L.cat([self.u2(c), b], 1))
            y = self.d1(L.cat([self.u1(y), a], 1))
            return self.out(y)
    return UNet()


def iou(pred, mask):
    """Per-image IoU of two 0/1 arrays N1HW; an empty pair scores 1."""
    inter = (pred * mask).sum(axis=(1, 2, 3)); union = ((pred + mask) > 0).sum(axis=(1, 2, 3))
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def run(L, x_train, m_train, x_test, m_test, epochs=30, batch=16, device=None, log=print):
    """Train the U-Net on (x_train, m_train), answer masks for both sets. Returns
    (train predictions, test predictions, seconds)."""
    to = (lambda t: t.to(device)) if device else (lambda t: t)
    scope = getattr(L, "scope", None)
    model = to(build_unet(L))
    opt = L.optim.Adam(model.parameters(), lr=1e-3)
    crit = L.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    n = len(x_train)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for s in range(0, n - batch + 1, batch):
            idx = order[s:s + batch]
            def step():
                opt.zero_grad()
                loss = crit(model(to(L.tensor(x_train[idx]))), to(L.tensor(m_train[idx])))
                loss.backward()
                opt.step()
                return float(loss.item())
            if scope is not None:
                with scope():
                    total += step()
            else:
                total += step()
        log(f"epoch {epoch + 1}/{epochs} · loss {total / (n // batch):.4f} · {time.perf_counter() - t0:.0f} s")

    def answer(x):
        model.eval()
        out = []
        for s in range(0, len(x), 32):
            if scope is not None:
                with scope():
                    out.append((model(to(L.tensor(x[s:s + 32]))).numpy() > 0).astype(np.float32))
            else:
                with L.no_grad():
                    out.append((model(to(L.tensor(x[s:s + 32]))).cpu().numpy() > 0).astype(np.float32))
        return np.concatenate(out)
    return answer(x_train), answer(x_test), time.perf_counter() - t0


def evaluate(L, x, m, n_train=800, noise=0.2, epochs=30, device=None, log=print):
    """The whole measurement: train on corrupted masks, report held-out IoU against the
    clean masks and the queue's metrics against the corrupted rows."""
    rng = np.random.default_rng(0)
    x_tr, m_tr, x_te, m_te = x[:n_train], m[:n_train], x[n_train:], m[n_train:]
    given, wrong = corrupt(m_tr, noise, rng)
    p_tr, p_te, seconds = run(L, x_tr, given, x_te, m_te, epochs=epochs, device=device, log=log)
    score = 1 - iou(p_tr, given)
    q = queue_metrics(score, wrong)
    return {"n_train": int(n_train), "epochs": epochs, "seconds": round(seconds, 1),
            "test_iou": float(iou(p_te, m_te).mean()), "train_iou_vs_clean": float(iou(p_tr, m_tr).mean()),
            "queue": q}


def main(argv):
    if argv[:1] == ["prepare"]:
        prepare(); return 0
    import torch
    arg = lambda k, d: type(d)(next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{k}=")), d))  # noqa: E731
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    x, m = load()
    torch.manual_seed(0)
    results = {"dataset": "kvasir-seg 96px", "device": device, "runs": {}}
    for noise in ([arg("noise", 0.2)] if "--noise=" in " ".join(argv) else [0.0, 0.2]):
        r = evaluate(torch, x, m, n_train=arg("train", 800), noise=noise, epochs=arg("epochs", 30), device=device)
        results["runs"][f"noise_{noise:g}"] = r
        q = r["queue"]
        print(f"noise {noise:g} · test IoU {r['test_iou']:.3f} · {r['seconds']:.0f} s" + (
            f" · queue AUROC {q['auroc']:.3f} · p@20 {q['p@20']:.2f} · p@wrong {q['p@wrong']:.2f} · read {q['read_for_90pct']:.0%} for 90%" if noise else ""))
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
