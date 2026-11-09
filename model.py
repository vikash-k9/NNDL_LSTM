import re
import torch
import torch.nn as nn


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class NextWordLSTM(nn.Module):
    """Embedding -> LSTM -> Dropout -> LSTM -> Dense (softmax is applied in the loss / at prediction)."""

    def __init__(self, vocab_size, emb_dim=100, h1=150, h2=100):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(emb_dim, h1, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.fc = nn.Linear(h2, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm1(x)
        x = self.drop(x)
        x, _ = self.lstm2(x)
        return self.fc(x[:, -1, :])  # sequences are left-padded, so the last step is a real word


def predict_top_k(model, word2idx, idx2word, max_len, text, k=3, device="cpu"):
    ids = [word2idx[w] for w in tokenize(text) if w in word2idx][-(max_len - 1):]
    if not ids:
        return []
    ids = [0] * (max_len - 1 - len(ids)) + ids
    x = torch.tensor([ids], device=device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1)[0]
    p, i = probs.topk(k)
    return [(idx2word.get(j.item(), "?"), q.item()) for q, j in zip(p, i)]
