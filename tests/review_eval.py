"""Does the review queue put the real mislabels first? Measured on open data.

    uv run --project ~/git/borch --with timm python tests/review_eval.py [--dataset=cifar10n|cifar100n|food101n] [--n=5000] [--inject=0.1]

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


# ── the datasets: each answers (images uint8 HWC or a list of paths, clean labels, {set: labels}) ──

def load_cifar10n(n, rng):
    import torch
    from torchvision.datasets import CIFAR10
    human = torch.load(DATA / "CIFAR-10_human.pt", weights_only=False)
    train = CIFAR10(str(DATA), train=True, download=False)
    idx = np.sort(rng.choice(len(train), size=min(n, len(train)), replace=False))
    clean = np.asarray(human["clean_label"])[idx]
    return train.data[idx], clean, {name: np.asarray(human[name])[idx] for name in ("aggre_label", "worse_label")}, "cifar10n"


def load_cifar100n(n, rng):
    """CIFAR-100N: the same people, a hundred fine classes, about 40 % of labels wrong."""
    import torch
    from torchvision.datasets import CIFAR100
    root = pathlib.Path.home() / "data" / "cifar100n"
    human = torch.load(root / "CIFAR-100_human.pt", weights_only=False)
    train = CIFAR100(str(root), train=True, download=False)
    idx = np.sort(rng.choice(len(train), size=min(n, len(train)), replace=False))
    clean = np.asarray(human["clean_label"])[idx]
    return train.data[idx], clean, {"noisy_label": np.asarray(human["noisy_label"])[idx]}, "cifar100n"


def load_food101n(n, rng):
    """Food-101N's verified training subset: web images whose class label people checked.
    `verification_label` 1 = the label is right, 0 = wrong — so the wrong ones are the truth."""
    root = pathlib.Path.home() / "data" / "food101n" / "Food-101N_release"
    rows = [line.split("\t") for line in (root / "meta" / "verified_train.tsv").read_text().splitlines()[1:]]
    # Only the rows whose image the release actually carries — a few verified keys are missing from it.
    rows = [(r[0], int(r[1])) for r in rows if len(r) >= 2 and (root / "images" / r[0]).exists()]
    idx = np.sort(rng.choice(len(rows), size=min(n, len(rows)), replace=False))
    picked = [rows[i] for i in idx]
    classes = sorted({r[0].split("/")[0] for r in rows})
    given = np.array([classes.index(r[0].split("/")[0]) for r in picked])
    # A wrong label's clean class is not known; what is known is that it is wrong. The
    # judge needs only `wrong`, so the clean label is given as "the label, or -1".
    clean = np.where(np.array([r[1] for r in picked]) == 1, given, -1)
    paths = [str(root / "images" / r[0]) for r in picked]
    return paths, clean, {"web_label": given}, "food101n"


DATASETS = {"cifar10n": load_cifar10n, "cifar100n": load_cifar100n, "food101n": load_food101n}


def decode(paths, size=224, batch=256):
    """JPEG paths → uint8 HWC at `size`, in chunks the feature extractor takes."""
    from PIL import Image
    for s in range(0, len(paths), batch):
        yield np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size)), dtype=np.uint8) for p in paths[s:s + batch]])


def main(argv):
    import torch
    n = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--n=")), "5000"))
    injects = [float(v) for v in next((a.split("=", 1)[1] for a in argv if a.startswith("--inject=")), "0").split(",") if float(v) > 0]
    k = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--k=")), "5"))
    which = next((a.split("=", 1)[1] for a in argv if a.startswith("--dataset=")), "cifar10n")
    rng = np.random.default_rng(0)
    images, clean, sets, tag = DATASETS[which](n, rng)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cache = DATA.parent / f"feats_b0_prelogits_{tag}_n{len(clean)}_seed0.npy"
    if cache.exists():
        feats = np.load(cache)
    elif isinstance(images, np.ndarray):
        feats = features(images, device)
        np.save(cache, feats)
    else:
        feats = np.concatenate([features(chunk, device) for chunk in decode(images)])
        np.save(cache, feats)
    results = {"dataset": tag, "n": int(len(clean)), "k": k, "backbone": "efficientnet_b0.ra_in1k pre_logits 1280-d", "sets": {}}
    idx = np.arange(len(clean))
    # `--inject=0.1,0.2,0.4`: the clean labels with a fraction flipped uniformly — the
    # noise-rate curve, one set per rate. Only where every clean label is known.
    if injects and (clean >= 0).all():
        classes = int(clean.max()) + 1
        for inject in injects:
            flipped = clean.copy()
            hit = rng.random(len(idx)) < inject
            flipped[hit] = (flipped[hit] + rng.integers(1, classes, hit.sum())) % classes
            sets[f"inject_{inject:g}"] = flipped
    for name, labels in sets.items():
        score = suspects(feats, labels, k=k)
        wrong = labels != clean
        m = queue_metrics(score, wrong)
        results["sets"][name] = m
        print(f"{name:14s} noise {m['noise_rate']:.3f} · AUROC {m['auroc']:.3f} · p@20 {m['p@20']:.2f} · p@100 {m['p@100']:.2f} · p@500 {m['p@500']:.2f} · p@wrong {m['p@wrong']:.2f} · read {m['read_for_half']:.0%} for half, {m['read_for_90pct']:.0%} for 90%")
    # One file per dataset, and the sets of every dataset side by side in review_eval.json.
    per = OUT.with_name(f"review_eval_{tag}.json")
    per.write_text(json.dumps(results, indent=2), encoding="utf-8")
    combined = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if "sets" in combined and "datasets" not in combined:          # the first shape, cifar10n alone
        combined = {"datasets": {"cifar10n": combined}}
    combined.setdefault("datasets", {})[tag] = results
    OUT.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"→ {per}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
