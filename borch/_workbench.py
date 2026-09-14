"""One call carrying every setting, and the session that runs it.

    from borch_webgpu import workbench as wb

    s = wb.setup(files, epochs=12, backbone="imagenet-efficientnet-b0", val=0.2)
    s.fit()
    s.accuracy, s.order          # and s.fit(my_model) to lend the loop to your own

**Why a namespace and not `torch.setup`.** This is not a name torch has, and the binding
is imported as `torch` in every example on the site — so spelled at the top level it would
teach `torch.setup(...)`, which stops on real torch with `AttributeError`. That is the
defect this library exists to refuse, pointing the other way. It sits one level in, beside
`peft`, where the reader can see whose it is.

**Why it exists at all.** The recipe below — a folder of images in, a frozen backbone or a
small CNN, a review queue ordered by doubt, an ONNX file out — was written by hand three
times in page code, in two languages, before it was written once here. The last defect
that cost a day came from page code doing library work.

**What it does not do.** Lightning's Trainer earns its size on things a tab has not got:
several devices, another process, a run long enough to resume. A tab trains for seconds on
one device, so none of that is here. What is here is the configuration in one place —
which is also what makes the run reportable: `facts()` hands `torch.report` the settings it
could not otherwise know.
"""

import time

import numpy as _np

from ._data import ImageFiles, label_from_name, suspects as _suspects

# A name here is the optimizer's name in torch. Anything else is refused with the list,
# rather than falling back to Adam and printing a number somebody will trust.
_OPTIMIZERS = ("adam", "sgd")


def _small_cnn(nn, classes):
    """The from-scratch path: three conv blocks to a linear head. The workbench's own."""
    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, classes))


def _Frozen(nn, backbone, head):
    """backbone → pre-logits → head as one module, so the export is the whole thing."""

    class Frozen(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head

        def forward(self, x):
            return self.head(backbone.forward_head(backbone.forward_features(x), pre_logits=True))

    return Frozen()


class Session:
    """What `setup()` returns: the settings, and the run once `fit()` has been called.

    Every attribute that is a result is `None` until `fit()`, so reading one early says
    so by being absent rather than by being zero.
    """

    def __init__(self, torch, files, *, size=None, batch=16, epochs=12, lr=None,
                 optimizer=None, backbone=None, val=0.0, seed=0, k=None,
                 standardise=True, label=label_from_name):
        # **The CPU door is a different library, not a slower one.** `borch_cpu` has no
        # `nn`, no `optim` and no autograd: what it has is a frozen backbone's forward and
        # a linear head that fits itself. So the surface is asked what it can do rather
        # than what it is called, and a caller who wants the other path is told which door
        # they are at instead of meeting `module has no attribute 'nn'` inside the loop.
        eager = hasattr(torch, "nn") and hasattr(torch, "optim")
        cpu_door = hasattr(torch, "LinearHead")
        if optimizer is None:
            optimizer = "adam" if eager else "sgd"
        if lr is None:
            lr = 1e-3 if eager else 0.05
        if optimizer not in _OPTIMIZERS:
            raise ValueError(f"workbench: optimizer {optimizer!r} is not one of {list(_OPTIMIZERS)}")
        if cpu_door and not eager:
            if backbone is None:
                raise ValueError(
                    f"workbench: {torch.__name__} trains a linear head on a frozen backbone and "
                    "nothing else — there is no autograd on this door. Name a backbone, or use "
                    "borch_webgpu (a tab with an adapter) for a model of your own.")
            if optimizer != "sgd":
                raise ValueError(
                    f"workbench: {torch.__name__}'s head fits with SGD and momentum; "
                    f"optimizer={optimizer!r} is not on this door.")
        self._eager = eager
        self._cpu_head = None
        if not eager and not cpu_door:
            raise ValueError(
                f"workbench: {torch.__name__} offers neither nn/optim nor a LinearHead — "
                "there is nothing here to train with.")
        if not 0.0 <= val < 1.0:
            raise ValueError(f"workbench: val must be in [0, 1), not {val!r}")
        if epochs < 1 or batch < 1:
            raise ValueError("workbench: epochs and batch are at least 1")
        # **A backbone brings its own size, so asking for one here is a contradiction.**
        # The manifest says what its weights were trained on — 224 for most of the
        # registry, 448 for EfficientNet-B5 — and measured, feeding one a 64px square
        # without normalising costs 63 points of top-1 (tests/browser/preprocess_cost.py).
        # `size` is the from-scratch path's, where there is nothing to disagree with.
        if backbone is not None and size is not None:
            raise ValueError(
                f"workbench: backbone={backbone!r} prepares its own images the way its "
                f"weights were trained, and size={size} asks for something else. Leave "
                "size out with a backbone, or leave the backbone out to choose the size.")
        # **`load` is not the name to ask for.** The numpy core has `torch.load` and it
        # reads a checkpoint; the CPU door's is a backbone. `LinearHead` belongs to that
        # door alone, so it is the one that says which door this is.
        if backbone is not None and not (hasattr(torch, "hub") or hasattr(torch, "LinearHead")):
            # **Said here rather than at the first batch.** The numpy core has no hub, so a
            # backbone asked for there can never arrive; the surface is named so the reader
            # knows which door they are at.
            raise ValueError(
                f"workbench: backbone={backbone!r} needs a model hub and {torch.__name__} has none. "
                "Use borch_webgpu (a browser tab with an adapter), or leave backbone out for "
                "the small CNN, which trains here.")
        self.torch = torch
        # **A dataset already in hand is taken as it is.** A page that has decoded a folder
        # once should not decode it again to use this; `size` is then the set's, and a
        # different one asked for here is a contradiction rather than a resize.
        # Asked by what it offers, not by its class: `borch._data` reaches a session under
        # two module paths in some runs, and then `isinstance` says no to the very object
        # this branch is for. `suspects` reads its inputs the same way.
        if all(hasattr(files, name) for name in ("stack", "batches", "targets", "classes", "size")):
            if size is not None and size != files.size:
                raise ValueError(
                    f"workbench: the set was decoded at {files.size} px and size={size} was asked "
                    "for — make a new ImageFiles at that size, or leave size out.")
            self.data, size = files, files.size
        else:
            self.data = ImageFiles(files, size=64 if size is None else size, label=label)
            size = self.data.size
        self.config = {
            "size": int(size), "batch": int(batch), "epochs": int(epochs), "lr": float(lr),
            "optimizer": optimizer, "backbone": backbone, "val": float(val), "seed": int(seed),
            "standardise": bool(standardise),
            "images": len(self.data), "classes": list(self.data.classes),
        }
        self.k = k
        self.model = self.head = self._sample = None
        self.losses = self.features = self.predicted = None
        self.accuracy = self.measured_on = self.seconds = self.how = None
        # How many rows the accuracy was measured on. **A number without its denominator
        # is not a measurement**, and a board built from these has to say how finely it
        # can separate anything (see `resolution`).
        self.held_out = None

    # -- the split ---------------------------------------------------------------
    def _split(self):
        """`(train rows, scored rows)`. With `val=0` they are the same rows, and
        `measured_on` says so — a number measured on what it learned is worth having and
        worth labelling, not worth hiding."""
        n = len(self.data)
        if self.config["val"] <= 0:
            rows = _np.arange(n)
            return rows, rows
        order = _np.random.default_rng(self.config["seed"]).permutation(n)
        cut = max(1, int(round(n * self.config["val"])))
        return _np.sort(order[cut:]), _np.sort(order[:cut])

    # -- the run -----------------------------------------------------------------
    def fit(self, model=None):
        """Train, and leave the run on the session. Returns the session.

        `model` is yours if you pass one — the session lends the loop and nothing else.
        Absent, it builds the small CNN, or the frozen backbone's head when a `backbone`
        was named at setup.
        """
        torch, cfg = self.torch, self.config
        if self._eager:
            torch.manual_seed(cfg["seed"])
        classes = len(self.data.classes)
        started = time.perf_counter()
        train_rows, scored_rows = self._split()
        y_all = self.data.targets

        if not self._eager:
            if model is not None:
                raise ValueError(
                    f"workbench: {torch.__name__} cannot train a model of your own — there is "
                    "no autograd on this door. It fits a linear head on a frozen backbone.")
            self._fit_head_cpu(classes, train_rows, y_all)
        elif cfg["backbone"] is not None and model is None:
            self._fit_head(classes, train_rows, y_all)
        else:
            mine = model is not None
            self.model = model if mine else _small_cnn(torch.nn, classes)
            self._fit_model(train_rows, y_all, mine)

        self.seconds = time.perf_counter() - started
        self.predicted = self._predict()
        self.accuracy = float((self.predicted[scored_rows] == y_all[scored_rows]).mean())
        self.held_out = int(len(scored_rows))
        self.measured_on = "held-out" if cfg["val"] > 0 else "the training images"
        return self

    def _steps(self, rows):
        batch = self.config["batch"]
        return max(1, len(rows) // batch)

    def _optimizer(self, params):
        torch, cfg = self.torch, self.config
        maker = torch.optim.Adam if cfg["optimizer"] == "adam" else torch.optim.SGD
        return maker(params, lr=cfg["lr"])

    def _run_steps(self, step_fn, rows, per_epoch):
        """The loop, with the browser's two speedups used **only where they exist.**

        `compiled` records a step once and replays it, and `scope` returns a batch's
        buffers to the pool. Neither is on the numpy core and neither is in torch, so the
        same loop has to run without them rather than be written twice.
        """
        torch, cfg = self.torch, self.config
        compiled = getattr(torch, "compiled", None)
        run = compiled(step_fn) if compiled else step_fn
        scope = getattr(torch, "scope", None)
        losses = []
        try:
            for _epoch in range(cfg["epochs"]):
                last = None
                for s in range(per_epoch):
                    if scope:
                        with scope():
                            last = float(run(*self._batch(rows, s)).item())
                    else:
                        last = float(run(*self._batch(rows, s)).item())
                losses.append(last)
        finally:
            if compiled and hasattr(run, "dispose"):
                run.dispose()
        self.losses = losses

    def _batch(self, rows, s):
        raise NotImplementedError                                # set by the two paths

    # -- path one: a small CNN (or yours) on the images ---------------------------
    def _fit_model(self, train_rows, y_all, mine):
        torch, cfg = self.torch, self.config
        xs = torch.tensor(self.data.stack()[train_rows])
        ys = torch.tensor(y_all[train_rows])
        opt = self._optimizer(self.model.parameters())
        crit = torch.nn.CrossEntropyLoss()
        batch = cfg["batch"]
        model = self.model

        def step(xb, yb):
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            return loss

        self._batch = lambda rows, s: (xs[s * batch:(s + 1) * batch], ys[s * batch:(s + 1) * batch])
        self._run_steps(step, train_rows, self._steps(train_rows))
        # **Which model trained, read off what was passed.** This asked `self.model is not
        # None`, which `fit()` has just filled in either way, so it said "your model" for
        # the small CNN too — caught by the workbench page, whose line names the model.
        self.how = (f"{'your model' if mine else 'small CNN'} · "
                    f"{cfg['epochs']} epochs × {self._steps(train_rows)} steps")

    def _standardise(self, train_rows):
        """Centre and scale the cached features, **from the training rows alone.**

        Measured across the registry's backbones, the features' spread runs 0.18 to 3.68 —
        twentyfold (`tests/browser/backbone_sweep.py`). One learning rate over all of them
        ranks which model happens to suit that rate, so a comparison that does not do this
        is measuring the rate.

        The statistics come from the rows the head trains on. Taking them from everything
        lets the held-out rows inform the scaling that the model is then judged under —
        a small leak, and the kind that makes a number look better than it is.
        """
        train = self.features[train_rows]
        mean = train.mean(axis=0, keepdims=True)
        # A dimension that never varies carries nothing; dividing by its zero would carry
        # an infinity into every row instead.
        spread = train.std(axis=0, keepdims=True)
        spread = _np.where(spread > 1e-6, spread, 1.0)
        self.features = ((self.features - mean) / spread).astype(_np.float32)

    # -- path two: a frozen backbone, and only the head learns -------------------
    def _prepared(self, transform, batch):
        """The photographs, prepared the way this backbone's weights were trained.

        **From the originals, not from what this set decoded.** `ImageFiles` hands back a
        square at its own size; the manifest asks for its own resize and centre crop, and
        running that over an already-squashed square is two resizes of which neither is
        the one that was measured.
        """
        for start in range(0, len(self.data), batch):
            idx = _np.arange(start, min(start + batch, len(self.data)))
            yield _np.stack([transform(self.data.raw(int(i))) for i in idx]), idx

    def _fit_head(self, classes, train_rows, y_all):
        torch, cfg = self.torch, self.config
        net = torch.hub.load(cfg["backbone"])
        transform = getattr(net, "transform", None)
        if transform is None:
            raise RuntimeError(
                f"workbench: {cfg['backbone']} arrived without the preparation its weights "
                "were trained under — `hub.load` attaches it as `.transform`, and an older "
                "wheel does not. Feeding it something else is worth up to sixty points.")
        self.config["size"] = int(net.manifest["preprocess"]["inputSize"][1])
        chunks = []
        with torch.no_grad():
            for xb, _idx in self._prepared(transform, cfg["batch"]):
                chunks.append(net.forward_head(net.forward_features(torch.tensor(xb)),
                                               pre_logits=True).numpy())
        self.features = _np.concatenate(chunks)
        if cfg["standardise"]:
            self._standardise(train_rows)
        # One prepared photograph, kept for the export: the composed net takes the
        # manifest's size, not this set's.
        self._sample = next(iter(self._prepared(transform, 1)))[0]
        self.head = torch.nn.Linear(net.num_features, classes)
        ft = torch.tensor(self.features[train_rows])
        ys = torch.tensor(y_all[train_rows])
        opt = self._optimizer(self.head.parameters())
        crit = torch.nn.CrossEntropyLoss()
        head = self.head

        def step(f, t):
            opt.zero_grad()
            loss = crit(head(f), t)
            loss.backward()
            opt.step()
            return loss

        # Full batch: the features never change, so there is one shape and one recording.
        self._batch = lambda rows, s: (ft, ys)
        self._run_steps(step, train_rows, 1)
        # **The exported thing is the whole net, not the head.** The head alone is weights
        # against features nobody outside this tab can produce; a file that cannot be run
        # elsewhere is the wrong answer this library refuses. The page composed these two
        # by hand to export them — that is the eight lines this is.
        self.model = _Frozen(torch.nn, net, head)
        self.model.eval()
        self.how = f"{cfg['backbone']} frozen · head {cfg['epochs']} steps"

    # -- the CPU door: a frozen forward and a head that fits itself ---------------
    def _fit_head_cpu(self, classes, train_rows, y_all):
        torch, cfg = self.torch, self.config
        net = torch.load(cfg["backbone"], features=True)
        from ._preprocess import transform_for                # noqa: PLC0415

        transform = transform_for((net.manifest or {}).get("preprocess"))
        self.config["size"] = int(net.input_size[1])
        chunks = [net.features(xb) for xb, _idx in self._prepared(transform, cfg["batch"])]
        self.features = _np.concatenate(chunks)
        if cfg["standardise"]:
            self._standardise(train_rows)
        # `epochs` is steps here: the head sees every cached feature at once, so an epoch
        # and a step are the same thing — as they are on the other door's frozen path.
        # **`head` is the thing that learned, whichever door this is.** On the other one it
        # is an `nn.Linear`; here it is a `LinearHead` that fits itself. A caller that
        # hands the head on — the workbench page exports its weights — should not have to
        # ask which kind it got.
        self._cpu_head = self.head = torch.LinearHead(net.num_features, classes, lr=cfg["lr"], momentum=0.9)
        every = self._cpu_head.fit(self.features[train_rows], y_all[train_rows], steps=cfg["epochs"])
        marks = _np.linspace(0, len(every) - 1, min(len(every), 6)).astype(int)
        self.losses = [float(every[i]) for i in marks]
        self.how = f"{cfg['backbone']} frozen, on the CPU · head {cfg['epochs']} steps"

    # -- what came out -----------------------------------------------------------
    def _predict(self):
        torch = self.torch
        if self._cpu_head is not None:
            return self._cpu_head.predict(self.features).argmax(1)
        with torch.no_grad():
            if self.head is not None:
                logits = self.head(torch.tensor(self.features)).numpy()
            else:
                logits = self.model(torch.tensor(self.data.stack())).numpy()
                self.features = logits
        return logits.argmax(1)

    @property
    def suspects(self):
        """How much each given label is doubted, in [0, 1] — the review queue's score."""
        if self.features is None:
            raise RuntimeError("workbench: call fit() before reading suspects")
        k = self.k if self.k else (5 if len(self.data) < 2000 else 20)
        return _suspects(self.features, self.data.targets, k=k)

    @property
    def order(self):
        """The rows of `data`, most doubted first."""
        return _np.argsort(-self.suspects)

    def onnx(self):
        """The trained model as ONNX bytes. Needs a surface with an exporter."""
        if self._cpu_head is not None:
            raise RuntimeError(
                f"workbench: {self.torch.__name__} has the head's weights and no graph to "
                "export — read them with `state_dict()`, or train on a door with an exporter.")
        if self.model is None:
            raise RuntimeError("workbench: call fit() before exporting")
        exporter = getattr(self.torch, "onnx", None)
        if exporter is None:
            raise RuntimeError(
                f"workbench: {self.torch.__name__} has no ONNX exporter; borch_webgpu does.")
        # The frozen net takes the manifest's size; the from-scratch one takes this set's.
        sample = self._sample if self._sample is not None else self.data.stack()[:1]
        return exporter.export(self.model, self.torch.tensor(sample))

    def facts(self):
        """The settings and the result, flat — what `torch.report(**facts)` could not know."""
        got = {f"cfg_{k}": v for k, v in self.config.items() if k != "classes"}
        got["cfg_classes"] = ",".join(self.config["classes"])
        if self.accuracy is not None:
            got.update(how=self.how, accuracy=round(self.accuracy, 4),
                       measured_on=self.measured_on, seconds=round(self.seconds, 2),
                       loss_first=self.losses[0], loss_last=self.losses[-1])
        return got


def setup(torch, files, **config):
    """Every setting in one call. Returns a `Session`; `fit()` runs it."""
    return Session(torch, files, **config)


def candidates(torch, budget_mb=60, task="image-classification"):
    """The registry's models this data could be handed to, smallest first.

    **The task is not the filter.** `cifar10-resnet18` is image-classification too, and
    its manifest asks for no resize because it was trained on 32px tiles that arrive at
    that size — handed a photograph it refuses, which is right and is not a candidate
    (measured, `tests/browser/backbone_sweep.py`). What makes a model able to take an
    arbitrary picture is that its manifest carries a resize.

    The newest version of each name, because the registry lists every version.
    """
    newest = {}
    for row in torch.hub.list():
        newest[row.get("name")] = row
    ceiling = budget_mb * 1_000_000
    out = [row for row in newest.values()
           if row.get("task") == task and row.get("bytes", 0) <= ceiling]
    return sorted(out, key=lambda r: r.get("bytes", 0))


# 1.96 standard errors — the ordinary ninety-five per cent.
_CONFIDENT = 1.96


def interval(accuracy, held_out):
    """How far an accuracy measured on `held_out` rows could be from the truth, in points.

    A share measured on a sample carries `sqrt(p(1-p)/n)` of standard error, and at a
    hundred rows that is eight and a half points at ninety-five per cent. The workbench's
    own board over CIFAR-10 read 0.720, 0.670 and 0.760 on a hundred held-out rows — nine
    points apart, which is **less than the width of any one of them.**
    """
    if held_out <= 0:
        return 1.0
    p = min(max(float(accuracy), 0.0), 1.0)
    return _CONFIDENT * ((p * (1.0 - p) / held_out) ** 0.5)


def resolution(held_out, around=0.75):
    """The smallest difference two models have to show before it is more than the split.

    Two independent estimates, so the difference carries `sqrt(2)` times one interval:
    seventeen points at fifty rows, twelve at a hundred, five at five hundred. **A board
    whose spread is under this is a board of one answer**, and saying otherwise ranks the
    rows the split happened to hand out.
    """
    return (2.0 ** 0.5) * interval(around, held_out)


def compare(torch, files, *, budget_mb=60, val=0.2, seed=0, stop_at=None, **config):
    """Train the same head on each backbone under the budget; return the rows, best first.

    **Every candidate sees the same photographs and the same split**, because `val` and
    `seed` are fixed here rather than left to each call — a leaderboard where the models
    saw different held-out rows is a leaderboard of the splits.

    The features are standardised per model from the training rows (see `_standardise`);
    without it the ranking follows whichever backbone's feature scale happens to suit the
    learning rate.

    `stop_at` is what `pick` passes: the first candidate to reach it ends the run, so the
    rest are never fetched. Rows come back in the order they were tried — smallest first —
    with the failures kept, because a model that refuses says something about itself.
    """
    if val <= 0:
        raise ValueError("workbench.compare: a ranking needs held-out rows — val must be above 0")
    rows = []
    for row in candidates(torch, budget_mb=budget_mb):
        got = {"name": row["name"], "mb": round(row.get("bytes", 0) / 1e6, 1)}
        try:
            s = Session(torch, files, backbone=row["name"], val=val, seed=seed, **config).fit()
            got.update(accuracy=s.accuracy, measured_on=s.measured_on, held_out=s.held_out,
                       interval=round(interval(s.accuracy, s.held_out), 4),
                       seconds=round(s.seconds, 1), features=int(s.features.shape[1]),
                       size=s.config["size"])
        except Exception as e:                                   # noqa: BLE001
            # One refusal is not the end of a comparison — `cifar10-resnet18` refuses
            # every photograph, and knowing that is part of the answer.
            got["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rows.append(got)
        if stop_at is not None and got.get("accuracy", -1.0) >= stop_at:
            break
    return _mark_ties(rows)


def _mark_ties(rows):
    """Say which rows the evidence cannot separate from the best one.

    **A leaderboard is a claim, and a short one cannot support it.** Every row that sits
    within the resolution of the leader carries `ties_with_best`, so a reader ordering by
    accuracy can see that the order is not the finding. The smallest of a tie is the one
    worth taking, which is what `pick` does by trying them in that order.
    """
    scored = [r for r in rows if "accuracy" in r]
    if not scored:
        return rows
    best = max(r["accuracy"] for r in scored)
    for r in scored:
        gap = resolution(r.get("held_out", 0))
        r["ties_with_best"] = bool(best - r["accuracy"] < gap)
        r["resolution"] = round(gap, 4)
    return rows


def pick(torch, files, *, at_least=0.9, budget_mb=60, val=0.2, seed=0, **config):
    """The smallest backbone under the budget that reaches `at_least` on the held-out rows.

    **The question a tab actually asks.** A leaderboard is a table to read; this is the
    answer to "what should I use", and it fetches less: the candidates are tried smallest
    first and the first one to clear the bar ends it, so the larger weights are never
    downloaded.

    Returns `(name, rows)` — the rows are every candidate tried, so the reader can see
    what it cost. `name` is None when nothing cleared the bar.
    """
    rows = compare(torch, files, budget_mb=budget_mb, val=val, seed=seed,
                   stop_at=at_least, **config)
    cleared = [r for r in rows if r.get("accuracy", -1.0) >= at_least]
    return (cleared[-1]["name"] if cleared else None), rows


def say(rows):
    """The board as lines a person reads, with what the evidence can and cannot say."""
    out = []
    scored = [r for r in rows if "accuracy" in r]
    for r in rows:
        if "error" in r:
            out.append(f"  {r['name']:32s} {r['mb']:6.1f} MB   refused: {r['error'][:60]}")
            continue
        tie = "  = best" if r.get("ties_with_best") else ""
        out.append(f"  {r['name']:32s} {r['mb']:6.1f} MB   {r['accuracy']:.3f}"
                   f" ± {r['interval'] * 100:.0f} points on {r['held_out']} held out{tie}")
    if scored:
        gap = resolution(scored[0].get("held_out", 0))
        tied = [r["name"] for r in scored if r.get("ties_with_best")]
        out.append(f"  {scored[0].get('held_out', 0)} held-out rows separate models"
                   f" {gap * 100:.0f} points apart and no closer")
        if len(tied) > 1:
            out.append(f"  this board does not rank {len(tied)} of them — take the smallest")
    return "\n".join(out)
