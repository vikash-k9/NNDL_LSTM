"""
Next Word Prediction using LSTM (PyTorch version, works on Python 3.14)
Run:  python train_torch.py
Produces: next_word_lstm.pt, vocab.pickle
"""
import pickle
from collections import Counter

import nltk
import torch
import torch.nn as nn
from nltk.corpus import gutenberg
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import NextWordLSTM, predict_top_k, tokenize

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---------- 1. Data Collection ----------
nltk.download("gutenberg", quiet=True)
text = gutenberg.raw("shakespeare-hamlet.txt").lower()
print(f"Characters in corpus: {len(text):,}")

# ---------- 2. Data Preprocessing ----------
counts = Counter(tokenize(text))
word2idx = {w: i + 1 for i, (w, _) in enumerate(counts.most_common())}  # 0 = padding
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx) + 1
print(f"Vocabulary size: {vocab_size:,}")

sequences = []
for line in text.split("\n"):
    ids = [word2idx[w] for w in tokenize(line)]
    for i in range(1, len(ids)):
        sequences.append(ids[: i + 1])

max_len = max(len(s) for s in sequences)
padded = [[0] * (max_len - len(s)) + s for s in sequences]  # pre-padding
data = torch.tensor(padded)
X, y = data[:, :-1], data[:, -1]
print(f"Total sequences: {len(X):,} | Max length: {max_len}")

dataset = TensorDataset(X, y)
n_test = int(0.2 * len(dataset))
train_set, test_set = random_split(
    dataset, [len(dataset) - n_test, n_test], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)

# ---------- 3. Model Building ----------
model = NextWordLSTM(vocab_size).to(device)
print(model)
print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
criterion = nn.CrossEntropyLoss()  # includes softmax
optimizer = torch.optim.Adam(model.parameters())


def run_eval(loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += criterion(out, yb).item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)
    return total_loss / n, correct / n


# ---------- 4. Model Training (with early stopping on validation loss) ----------
best_val, best_state, patience, bad = float("inf"), None, 3, 0
for epoch in range(1, 51):
    model.train()
    run_loss, correct, n = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * len(xb)
        correct += (out.argmax(1) == yb).sum().item()
        n += len(xb)
    val_loss, val_acc = run_eval(test_loader)
    print(f"Epoch {epoch:02d} | loss {run_loss/n:.4f} acc {correct/n:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

    if val_loss < best_val:
        best_val, bad = val_loss, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping: validation loss stopped improving.")
            break

model.load_state_dict(best_state)

# ---------- 5. Model Evaluation ----------
loss, acc = run_eval(test_loader)
print(f"\nTest loss: {loss:.4f} | Test accuracy: {acc:.4f}")

print("\n--- Example predictions ---")
for s in ["to be or not to", "the king is", "my lord", "what a piece of work is", "good night sweet"]:
    top = predict_top_k(model, word2idx, idx2word, max_len, s, k=1, device=device)
    print(f"'{s}'  ->  {top[0][0] if top else '?'}")

# ---------- Save for the Streamlit app ----------
torch.save(model.state_dict(), "next_word_lstm.pt")
with open("vocab.pickle", "wb") as f:
    pickle.dump({"word2idx": word2idx, "idx2word": idx2word, "max_len": max_len}, f)
print("\nSaved: next_word_lstm.pt, vocab.pickle")
