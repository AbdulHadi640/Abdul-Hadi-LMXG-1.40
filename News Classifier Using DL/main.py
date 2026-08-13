import re, random
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# ---------- config ----------
FILE = "corrected_news.csv"
SEED, BATCH, MIN_FREQ = 42, 32, 2
EMB, HID, LAYERS, DROP = 100, 128, 2, 0.3
EPOCHS, PATIENCE = 20, 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["Business", "Energy", "Health", "Markets", "Politics", "Technology"]
L2I = {x:i for i,x in enumerate(LABELS)}
I2L = {i:x for x,i in L2I.items()}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- text ----------
def norm(s):
    return re.sub(r"\s+", " ", str(s).lower().strip())

def tok(s):
    s = re.sub(r"[^a-z0-9\s]", " ", str(s).lower())
    return [w for w in re.sub(r"\s+", " ", s).strip().split() if w != "s"]

# ---------- data ----------
df = pd.read_csv(FILE)[["Title", "Final_Category"]].dropna()
df = df[df.Final_Category.isin(L2I)].copy()
df["norm"] = df.Title.map(norm)
df = df.drop_duplicates("norm").reset_index(drop=True)

train, tmp = train_test_split(df, test_size=.2, random_state=SEED, stratify=df.Final_Category)
val, test = train_test_split(tmp, test_size=.5, random_state=SEED, stratify=tmp.Final_Category)

for d in (train, val, test):
    d["tokens"] = d.Title.map(tok)

train = train[train.tokens.map(len) > 0].copy()
val   = val[val.tokens.map(len) > 0].copy()
test  = test[test.tokens.map(len) > 0].copy()

# ---------- vocab / ids ----------
cnt = Counter(w for row in train.tokens for w in row)
vocab = {"<PAD>":0, "<UNK>":1}
for w,c in cnt.items():
    if c >= MIN_FREQ:
        vocab[w] = len(vocab)

def to_ids(xs):
    return [vocab.get(x, 1) for x in xs]

for d in (train, val, test):
    d["ids"] = d.tokens.map(to_ids)
    d["y"] = d.Final_Category.map(L2I)

# ---------- pretrained GloVe ----------
print("Loading GloVe...")
glove = api.load("glove-wiki-gigaword-100")
W = np.random.normal(0, .05, (len(vocab), EMB)).astype("float32")
W[0] = 0
found = 0
for w,i in vocab.items():
    if w in glove:
        W[i] = glove[w]
        found += 1
print(f"GloVe coverage: {found}/{len(vocab)}")
W = torch.tensor(W)
del glove

# ---------- dataset / padding ----------
class NewsDS(Dataset):
    def __init__(self, d):
        self.x, self.y = d.ids.tolist(), d.y.tolist()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])

def collate(batch):
    x,y = zip(*batch)
    lengths = torch.tensor([len(s) for s in x])
    return pad_sequence(x, batch_first=True, padding_value=0), lengths, torch.stack(y)

def loader(d, shuffle=False):
    return DataLoader(NewsDS(d), batch_size=BATCH, shuffle=shuffle, collate_fn=collate)

tr, va, te = loader(train, True), loader(val), loader(test)

# ---------- Embedding -> 2-layer BiLSTM -> Dropout -> Linear ----------
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(W, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(EMB, HID, num_layers=LAYERS, batch_first=True,
                            bidirectional=True, dropout=DROP)
        self.drop = nn.Dropout(DROP)
        self.fc = nn.Linear(HID*2, len(LABELS))

    def forward(self, x, lengths):
        x = self.emb(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(x)
        return self.fc(self.drop(torch.cat((h[-2], h[-1]), 1)))

model = BiLSTM().to(DEVICE)

# Slight label smoothing + AdamW to reduce overfitting
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

def evaluate(dl):
    model.eval(); total=0; ys=[]; ps=[]
    with torch.no_grad():
        for x,l,y in dl:
            x,y = x.to(DEVICE), y.to(DEVICE)
            z = model(x,l)
            total += loss_fn(z,y).item()
            ys += y.cpu().tolist()
            ps += z.argmax(1).cpu().tolist()
    return total/len(dl), accuracy_score(ys,ps), f1_score(ys,ps,average="macro"), ys, ps

# ---------- train on validation only ----------
best_f1, best_state, wait, best_epoch = -1, None, 0, 0

for ep in range(1, EPOCHS+1):
    model.train(); total=0
    for x,l,y in tr:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(x,l), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()

    vl, va_acc, va_f1, _, _ = evaluate(va)
    print(f"{ep:02d} | train {total/len(tr):.4f} | val {vl:.4f} | acc {va_acc*100:.2f}% | F1 {va_f1*100:.2f}%")

    if va_f1 > best_f1:
        best_f1, best_epoch, wait = va_f1, ep, 0
        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            break

model.load_state_dict(best_state)
model.to(DEVICE)

# ---------- final test once ----------
tl, ta, tf1, y, p = evaluate(te)
print(f"\nBest val epoch: {best_epoch}")
print(f"Test Accuracy: {ta*100:.2f}%")
print(f"Test Macro F1: {tf1*100:.2f}%")
print(classification_report(y, p, target_names=LABELS, digits=4))

torch.save({"model":model.state_dict(), "vocab":vocab, "labels":L2I}, "bilstm_short.pt")