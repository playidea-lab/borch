"""Does the review queue put the real mislabels first? Measured on open data.

    uv run --project ~/git/borch --with timm python tests/review_eval.py [--n=5000] [--inject=0.1]

The field trainer's review order is `borch.suspects` — the share of a sample's five
nearest neighbours (cosine, on a frozen EfficientNet-B0's pre-logits) that disagree with
its label. This runs exactly that path natively (timm's `efficientnet_b0.ra_in1k`, the
model the catalogue converted) on CIFAR-10N, where fifty thousand training images carry
labels real people gave and the clean labels are known — so "was this label wrong" has an
answer for every row. Two label sets: `aggre_label` (about 9 % wrong) and `worse_label`
(about 40 %). `--inject` adds a controlled run: clean labels with a fraction flipped
uniformly, which is the shape of a typing error rather than a human's confusion.

What is reported, per label set: the noise rate (the random-order baseline), AUROC of the
score against "this label is wrong", precision at 20 / 100 / 500 and at the noise count,
and how much of the queue has to be read to catch half and nine tenths of the mislabels.
The numbers go to tests/review_eval.json and the site quotes them.

Data: ~/data/cifar10n — CIFAR-10_human.pt from github.com/UCSC-REAL/cifar-10-100n and the
CIFAR-10 python batches (see noez/scripts/bench_cifar10n.py for the two downloads).
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from borch._data import suspects  # noqa: E402

DATA = pathlib.Path.home() / "data" / "cifar10n"
OUT = pathlib.Path(__file__).resolve().parent / "review_eval.json"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def auroc(score, positive):
    """Rank-based AUROC — no sklearn here."""
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n_pos = int(positive.sum()); n_neg = len(score) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def queue_metrics(score, wrong):
    order = np.argsort(-score)
    hits = wrong[order]
    n_wrong = int(wrong.sum()); n = len(wrong)
    cum = np.cumsum(hits)
    at = lambda k: float(hits[:k].mean()) if k <= n else float("nan")     # noqa: E731
    read_for = lambda frac: int(np.searchsorted(cum, frac * n_wrong) + 1) / n if n_wrong else float("nan")  # noqa: E731
    return {
        "n": n, "wrong": n_wrong, "noise_rate": n_wrong / n,
        "auroc": auroc(score, wrong),
        "p@20": at(20), "p@100": at(100), "p@500": at(500), "p@wrong": at(n_wrong),
        "read_for_half": read_for(0.5), "read_for_90pct": read_for(0.9),
    }


def features(images, device, batch=64):
    """B0 pre-logits (1280-d) for uint8 HWC images, the product's own path."""
    import timm
    import torch
    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True).eval().to(device)
    out = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(images), batch):
            x = torch.from_numpy(images[s:s + batch]).to(device).float().div(255)
            x = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=224, mode="bilinear", align_corners=False)
            x = (x - torch.tensor(MEAN, device=device)[None, :, None, None]) / torch.tensor(STD, device=device)[None, :, None, None]
            out.append(model.forward_head(model.forward_features(x), pre_logits=True).float().cpu().numpy())
    print(f"features: {len(images)} images in {time.time() - t0:.0f} s on {device}", flush=True)
    return np.concatenate(out)


def main(argv):
    import torch
    from torchvision.datasets import CIFAR10
    n = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--n=")), "5000"))
    inject = float(next((a.split("=", 1)[1] for a in argv if a.startswith("--inject=")), "0"))
    k = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--k=")), "5"))
    human = torch.load(DATA / "CIFAR-10_human.pt", weights_only=False)
    train = CIFAR10(str(DATA), train=True, download=False)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(train), size=min(n, len(train)), replace=False))
    images = train.data[idx]                                  # uint8 (n, 32, 32, 3)
    clean = np.asarray(human["clean_label"])[idx]
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cache = DATA / f"feats_b0_prelogits_n{len(idx)}_seed0.npy"
    if cache.exists():
        feats = np.load(cache)
    else:
        feats = features(images, device)
        np.save(cache, feats)
    results = {"n": int(len(idx)), "k": k, "backbone": "efficientnet_b0.ra_in1k pre_logits 1280-d", "sets": {}}
    sets = {name: np.asarray(human[name])[idx] for name in ("aggre_label", "worse_label")}
    if inject:
        flipped = clean.copy()
        hit = rng.random(len(idx)) < inject
        flipped[hit] = (flipped[hit] + rng.integers(1, 10, hit.sum())) % 10
        sets[f"inject_{inject:g}"] = flipped
    for name, labels in sets.items():
        score = suspects(feats, labels, k=k)
        wrong = labels != clean
        m = queue_metrics(score, wrong)
        results["sets"][name] = m
        print(f"{name:14s} noise {m['noise_rate']:.3f} · AUROC {m['auroc']:.3f} · p@20 {m['p@20']:.2f} · p@100 {m['p@100']:.2f} · p@500 {m['p@500']:.2f} · p@wrong {m['p@wrong']:.2f} · read {m['read_for_half']:.0%} for half, {m['read_for_90pct']:.0%} for 90%")
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
