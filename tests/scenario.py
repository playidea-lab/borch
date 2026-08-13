"""통합 시나리오 — 튜토리얼처럼 쓴 코드를 양쪽에서 통째로 돌린다.

단위 대조는 연산 하나씩만 본다. 그것만으로는 **여러 조각이 엮였을 때 갈리는 것**을 못 잡는다.
실제로 이 파일이 처음 돌았을 때 둘이 나왔다.

- BatchNorm 의 역방향이 틀렸다(평균·분산을 그래프 밖에서 계산했다). 순방향만 대조하고
  있어서 오래 남아 있었다 — 학습은 돌아가고 손실도 내려가는데 값만 달랐다
- `p.data = ndarray` 를 받아주고 있었다. torch 는 거부한다

`LIB` 만 바꿔 같은 코드를 돌리고, 나온 숫자를 **허용 오차 안에서** 비교한다.
문자열로 견주면 안 된다 — 비트 동등은 이 프로젝트의 명시적 비목표이고, 실제로
리눅스의 BLAS 와 맥의 BLAS 가 소수점 여섯 자리에서 다르다(CI 가 그걸로 빨갛게 떴다).

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
    import browsertorch as torch
    sys.modules["torch"] = torch
    nn, optim = torch.nn, torch.optim
    from_data = torch.utils.data
else:
    import torch
    nn, optim = torch.nn, torch.optim
    from_data = torch.utils.data

rng = np.random.default_rng(0)
out = {}

# ── ① 표 데이터 분류기: Dataset·DataLoader·학습·검증·저장·불러오기
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
out["① MLP 정확도"] = correct / 40
out["① 최종 손실"] = float(crit(model(torch.tensor(X[:32])), torch.tensor(y[:32]).long()).item())

# ── ② CNN 한 배치 학습
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
out["② CNN 손실"] = float(nn.CrossEntropyLoss()(cnn(torch.tensor(img)), torch.tensor(lab)).item())

# ── ③ 시퀀스: LSTM + Linear
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
out["③ LSTM 손실"] = float(nn.MSELoss()(head(lstm(torch.tensor(seq))[0][-1]), tgt).item())

# ── ④ 트랜스포머 인코더 + 인과 마스크
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
out["④ 트랜스포머 로짓합"] = float(logits.sum().item())

# ── ⑤ 저장·불러오기
import tempfile, os
path = os.path.join(tempfile.mkdtemp(), "m.pt")
torch.save(model.state_dict(), path)
fresh = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Dropout(0.0), nn.Linear(16, 2))
fresh.load_state_dict(torch.load(path))
fresh.eval()
with torch.no_grad():
    out["⑤ 불러온 뒤 출력합"] = float(fresh(torch.tensor(X[:8])).sum().item())

for k, v in out.items(): print(f"{k}\t{v!r}")
