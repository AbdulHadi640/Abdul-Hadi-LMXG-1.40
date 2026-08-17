import re, random
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

FILE = "corrected_news.csv"
SEED, BATCH, MIN_FREQ = 42, 32, 2
EMB, HID, LAYERS, DROP = 100, 128, 2, 0.3
EPOCHS, PATIENCE = 20, 4
LR, WD = 5e-4, 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["Business", "Energy", "Health", "Markets", "Politics", "Technology"]
L2I = {x:i for i,x in enumerate(LABELS)}

def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

set_seed()

def norm(s):
    return re.sub(r"\s+", " ", str(s).lower().strip())

def tok(s):
    s = re.sub(r"[^a-z0-9\s]", " ", str(s).lower())
    return [w for w in re.sub(r"\s+", " ", s).strip().split() if w != "s"]

df = pd.read_csv(FILE)[["Title", "Final_Category"]].dropna()
df = df[df.Final_Category.isin(L2I)].copy()
df["norm"] = df.Title.map(norm)
df = df.drop_duplicates("norm").reset_index(drop=True)

train, tmp = train_test_split(df, test_size=.2, random_state=SEED, stratify=df.Final_Category)
val, test = train_test_split(tmp, test_size=.5, random_state=SEED, stratify=tmp.Final_Category)

for d in (train, val, test):
    d["tokens"] = d.Title.map(tok)

train = train[train.tokens.map(len) > 0].copy()
val = val[val.tokens.map(len) > 0].copy()
test = test[test.tokens.map(len) > 0].copy()

cnt = Counter(w for row in train.tokens for w in row)
vocab = {"<PAD>":0, "<UNK>":1}
for w, c in cnt.items():
    if c >= MIN_FREQ:
        vocab[w] = len(vocab)

def to_ids(xs):
    return [vocab.get(x, 1) for x in xs]

for d in (train, val, test):
    d["ids"] = d.tokens.map(to_ids)
    d["y"] = d.Final_Category.map(L2I)

print("Loading GloVe...")
glove = api.load("glove-wiki-gigaword-100")
W = np.random.normal(0, .05, (len(vocab), EMB)).astype("float32")
W[0] = 0
found = 0
for w, i in vocab.items():
    if w in glove:
        W[i] = glove[w]
        found += 1
print(f"GloVe coverage: {found}/{len(vocab)}")
W = torch.tensor(W)
del glove

class NewsDS(Dataset):
    def __init__(self, d):
        self.x, self.y = d.ids.tolist(), d.y.tolist()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])

def collate(batch):
    x, y = zip(*batch)
    lengths = torch.tensor([len(s) for s in x])
    x = pad_sequence(x, batch_first=True, padding_value=0)
    return x, lengths, torch.stack(y)

def make_loader(d, shuffle=False):
    return DataLoader(NewsDS(d), batch_size=BATCH, shuffle=shuffle, collate_fn=collate)

tr, va, te = make_loader(train, True), make_loader(val), make_loader(test)

def pretrained_embedding():
    return nn.Embedding.from_pretrained(W.clone(), freeze=False, padding_idx=0)

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = pretrained_embedding()
        self.fc1 = nn.Linear(EMB, HID)
        self.drop = nn.Dropout(DROP)
        self.fc2 = nn.Linear(HID, len(LABELS))
    def forward(self, x, lengths):
        e = self.emb(x)
        mask = (x != 0).unsqueeze(-1)
        pooled = (e * mask).sum(1) / lengths.to(x.device).unsqueeze(1)
        return self.fc2(self.drop(torch.relu(self.fc1(pooled))))

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = pretrained_embedding()
        self.rnn = nn.RNN(EMB, HID, num_layers=LAYERS, batch_first=True, dropout=DROP)
        self.drop = nn.Dropout(DROP)
        self.fc = nn.Linear(HID, len(LABELS))
    def forward(self, x, lengths):
        e = self.emb(x)
        p = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(p)
        return self.fc(self.drop(h[-1]))

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = pretrained_embedding()
        self.lstm = nn.LSTM(EMB, HID, num_layers=LAYERS, batch_first=True, dropout=DROP)
        self.drop = nn.Dropout(DROP)
        self.fc = nn.Linear(HID, len(LABELS))
    def forward(self, x, lengths):
        e = self.emb(x)
        p = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(p)
        return self.fc(self.drop(h[-1]))

class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = pretrained_embedding()
        self.lstm = nn.LSTM(EMB, HID, num_layers=LAYERS, batch_first=True, bidirectional=True, dropout=DROP)
        self.drop = nn.Dropout(DROP)
        self.fc = nn.Linear(HID * 2, len(LABELS))
    def forward(self, x, lengths):
        e = self.emb(x)
        p = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(p)
        return self.fc(self.drop(torch.cat((h[-2], h[-1]), 1)))

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = pretrained_embedding()
        self.gru = nn.GRU(EMB, HID, num_layers=LAYERS, batch_first=True, dropout=DROP)
        self.drop = nn.Dropout(DROP)
        self.fc = nn.Linear(HID, len(LABELS))
    def forward(self, x, lengths):
        e = self.emb(x)
        p = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(p)
        return self.fc(self.drop(h[-1]))

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

def evaluate(model, dl):
    model.eval()
    total, ys, ps = 0, [], []
    with torch.no_grad():
        for x, l, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            z = model(x, l)
            total += loss_fn(z, y).item()
            ys += y.cpu().tolist()
            ps += z.argmax(1).cpu().tolist()
    return total/len(dl), accuracy_score(ys, ps), f1_score(ys, ps, average="macro"), f1_score(ys, ps, average="weighted")

def train_model(name, model):
    set_seed()
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_f1, best_state, best_epoch, wait = -1, None, 0, 0

    print(f"\n{'='*60}\n{name}\n{'='*60}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0
        for x, l, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x, l), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        vl, va_acc, va_f1, _ = evaluate(model, va)
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
    _, test_acc, test_f1, weighted_f1 = evaluate(model, te)

    torch.save({"model":model.state_dict(), "vocab":vocab, "labels":L2I}, f"{name.lower()}_news.pt")

    return {
        "Model": name,
        "Best Epoch": best_epoch,
        "Val Macro F1": best_f1 * 100,
        "Test Accuracy": test_acc * 100,
        "Test Macro F1": test_f1 * 100,
        "Weighted F1": weighted_f1 * 100
    }

MODELS = [
    ("ANN", ANN),
    ("RNN", RNNModel),
    ("LSTM", LSTMModel),
    ("BiLSTM", BiLSTMModel),
    ("GRU", GRUModel),
]

results = []
for name, ModelClass in MODELS:
    results.append(train_model(name, ModelClass()))

results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)

print("\n" + "="*75)
print("FINAL MODEL COMPARISON")
print("="*75)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

best = results_df.iloc[0]
print(f"\nBest Model: {best['Model']} | Accuracy: {best['Test Accuracy']:.2f}% | Macro F1: {best['Test Macro F1']:.2f}%")

results_df.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")