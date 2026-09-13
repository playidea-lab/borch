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

    def __init__(self, torch, files, *, size=64, batch=16, epochs=12, lr=1e-3,
                 optimizer="adam", backbone=None, val=0.0, seed=0, k=None,
                 label=label_from_name):
        if optimizer not in _OPTIMIZERS:
            raise ValueError(f"workbench: optimizer {optimizer!r} is not one of {list(_OPTIMIZERS)}")
        if not 0.0 <= val < 1.0:
            raise ValueError(f"workbench: val must be in [0, 1), not {val!r}")
        if epochs < 1 or batch < 1:
            raise ValueError("workbench: epochs and batch are at least 1")
        if backbone is not None and not hasattr(torch, "hub"):
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
            if size != files.size and size != 64:
                raise ValueError(
                    f"workbench: the set was decoded at {files.size} px and size={size} was asked "
                    "for — make a new ImageFiles at that size, or leave size out.")
            self.data, size = files, files.size
        else:
            self.data = ImageFiles(files, size=size, label=label)
        self.config = {
            "size": int(size), "batch": int(batch), "epochs": int(epochs), "lr": float(lr),
            "optimizer": optimizer, "backbone": backbone, "val": float(val), "seed": int(seed),
            "images": len(self.data), "classes": list(self.data.classes),
        }
        self.k = k
        self.model = self.head = None
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
        torch.manual_seed(cfg["seed"])
        classes = len(self.data.classes)
        started = time.perf_counter()
        train_rows, scored_rows = self._split()
        y_all = self.data.targets

        if cfg["backbone"] is not None and model is None:
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
    def _fit_head(self, classes, train_rows, y_all):
        torch, cfg = self.torch, self.config
        net = torch.hub.load(cfg["backbone"])
        chunks = []
        with torch.no_grad():
            for xb, _idx in self.data.batches(cfg["batch"]):
                chunks.append(net.forward_head(net.forward_features(torch.tensor(xb)),
                                               pre_logits=True).numpy())
        self.features = _np.concatenate(chunks)
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

    # -- what came out -----------------------------------------------------------
    def _predict(self):
        torch = self.torch
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
        if self.model is None:
            raise RuntimeError("workbench: call fit() before exporting")
        exporter = getattr(self.torch, "onnx", None)
        if exporter is None:
            raise RuntimeError(
                f"workbench: {self.torch.__name__} has no ONNX exporter; borch_webgpu does.")
        return exporter.export(self.model, self.torch.tensor(self.data.stack()[:1]))

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
