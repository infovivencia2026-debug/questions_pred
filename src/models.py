import numpy as np, torch, torch.nn as nn


class MarkovBaseline:
    """Clusters-only baseline: first-order transition matrix with add-one smoothing."""
    def __init__(self, k):
        self.k = k; self.T = np.ones((k, k))

    def fit(self, X, y):
        for h, t in zip(X, y):
            self.T[h[-1], t] += 1
        self.T /= self.T.sum(1, keepdims=True)
        return self

    def predict_proba(self, X):
        return np.stack([self.T[h[-1]] for h in X])


class GRUModel(nn.Module):
    def __init__(self, k, hidden, seq_len):
        super().__init__()
        self.emb = nn.Embedding(k + 1, hidden, padding_idx=k)   # index k = PAD
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, k)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.out(h[:, -1])


class TransformerModel(nn.Module):
    def __init__(self, k, hidden, seq_len, heads=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(k + 1, hidden, padding_idx=k)
        self.pos = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(hidden, k)

    def forward(self, x):
        L = x.size(1)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        h = self.enc(self.emb(x) + self.pos[:, :L], mask=mask)
        return self.out(h[:, -1])


class HybridModel(nn.Module):
    """Cluster sequence (GRU) + mean content embedding of the author's history."""
    def __init__(self, k, hidden, seq_len, emb_dim):
        super().__init__()
        self.gru = GRUModel(k, hidden, seq_len)
        self.gru.out = nn.Identity()
        self.proj = nn.Linear(emb_dim, hidden)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, x, content):
        return self.out(torch.cat([self.gru(x), self.proj(content)], 1))
