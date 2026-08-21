"""The integration scenario — code written the way a tutorial writes it, run whole on both sides.

Unit comparison looks at one operation at a time. That alone cannot catch **what diverges
once several pieces are wired together.** Two things came out the first time this file ran.

- BatchNorm's backward was wrong (mean and variance were computed outside the graph). It had
  survived a long time because only the forward was being compared — training ran, the loss
  came down, and only the values differed.
- `p.data = ndarray` was being accepted. torch refuses it.

The same code runs with only `LIB` changed, and the numbers are compared **within
tolerance.** They must not be compared as strings — bit equality is an explicit non-goal of
this project, and Linux's BLAS and macOS's do differ at the sixth decimal (CI went red on
it).

    uv run --with numpy --with torch python tests/scenario.py real
    uv run --with numpy python tests/scenario.py nano
"""
import sys
import numpy as np

LIB = sys.argv[1]
if LIB == "nano":
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import borch as torch
    sys.modules["torch"] = torch
    nn, optim = torch.nn, torch.optim
    from_data = torch.utils.data
else:
    import torch
    nn, optim = torch.nn, torch.optim
    from_data = torch.utils.data

rng = np.random.default_rng(0)
out = {}

# ── ① a tabular classifier: Dataset, DataLoader, training, validation, save, load
X = rng.standard_normal((240, 6)).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

class Table(from_data.Dataset):
    def __init__(self, x, t): self.x, self.t = torch.tensor(x), torch.tensor(t)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.t[i]

train = from_data.DataLoader(Table(X[:200], y[:200]), batch_size=32, shuffle=False)
test  = from_data.DataLoader(Table(X[200:], y[200:]), batch_size=32, shuffle=False)

model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Dropout(0.0), nn.Linear(16, 2))
W = [rng.standard_normal(tuple(p.shape)).astype(np.float32) * 0.3 for p in model.parameters()]
for p, w in zip(model.parameters(), W): p.data = torch.tensor(w.copy())

crit, opt = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.01)
for _ in range(20):
    model.train()
    for xb, yb in train:
        opt.zero_grad(); crit(model(xb), yb.long()).backward(); opt.step()
model.eval()
with torch.no_grad():
    correct = sum(int((model(xb).argmax(dim=1) == yb.long()).sum().item()) for xb, yb in test)
out["① MLP accuracy"] = correct / 40
out["① final loss"] = float(crit(model(torch.tensor(X[:32])), torch.tensor(y[:32]).long()).item())

# ── ② one training batch of a CNN
img = rng.standard_normal((8, 1, 12, 12)).astype(np.float32)
lab = rng.integers(0, 3, 8).astype(np.int64)
cnn = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(4 * 6 * 6, 3))
W2 = [rng.standard_normal(tuple(p.shape)).astype(np.float32) * 0.2 for p in cnn.parameters()]
for p, w in zip(cnn.parameters(), W2): p.data = torch.tensor(w.copy())
o2 = optim.SGD(cnn.parameters(), lr=0.05, momentum=0.9)
for _ in range(5):
    o2.zero_grad(); nn.CrossEntropyLoss()(cnn(torch.tensor(img)), torch.tensor(lab)).backward(); o2.step()
cnn.eval()
out["② CNN loss"] = float(nn.CrossEntropyLoss()(cnn(torch.tensor(img)), torch.tensor(lab)).item())

# ── ③ sequences: LSTM + Linear
seq = rng.standard_normal((10, 4, 3)).astype(np.float32)
lstm, head = nn.LSTM(3, 5), nn.Linear(5, 1)
W3 = [rng.standard_normal(tuple(p.shape)).astype(np.float32) * 0.2 for p in list(lstm.parameters()) + list(head.parameters())]
for p, w in zip(list(lstm.parameters()) + list(head.parameters()), W3): p.data = torch.tensor(w.copy())
o3 = optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=0.01)
tgt = torch.tensor(rng.standard_normal((4, 1)).astype(np.float32))
for _ in range(10):
    o3.zero_grad()
    h = lstm(torch.tensor(seq))[0][-1]
    nn.MSELoss()(head(h), tgt).backward(); o3.step()
out["③ LSTM loss"] = float(nn.MSELoss()(head(lstm(torch.tensor(seq))[0][-1]), tgt).item())

# ── ④ transformer encoder with a causal mask
emb = nn.Embedding(20, 8)
layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
enc = nn.TransformerEncoder(layer, 2)
proj = nn.Linear(8, 20)
mods = [emb, enc, proj]
W4 = [rng.standard_normal(tuple(p.shape)).astype(np.float32) * 0.1 for m in mods for p in m.parameters()]
for p, w in zip([p for m in mods for p in m.parameters()], W4): p.data = torch.tensor(w.copy())
tokens = torch.tensor(rng.integers(0, 20, (2, 6)).astype(np.int64))
mask = nn.Transformer.generate_square_subsequent_mask(6)
enc.eval()
logits = proj(enc(emb(tokens), mask=mask))
out["④ transformer logit sum"] = float(logits.sum().item())

# ── ⑤ save and load
import tempfile, os
path = os.path.join(tempfile.mkdtemp(), "m.pt")
torch.save(model.state_dict(), path)
fresh = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Dropout(0.0), nn.Linear(16, 2))
fresh.load_state_dict(torch.load(path))
fresh.eval()
with torch.no_grad():
    out["⑤ output sum after loading"] = float(fresh(torch.tensor(X[:8])).sum().item())

for k, v in out.items(): print(f"{k}\t{v!r}")
