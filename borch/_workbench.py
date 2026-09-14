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
                 label=label_from_name):
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
            "images": len(self.data), "classes": list(self.data.classes),
        }
        self.k = k
        self.model = self.head = self._sample = None
        self.losses = self.features = self.predicted = None
        self.accuracy = self.measured_on = self.seconds = self.how = None

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
