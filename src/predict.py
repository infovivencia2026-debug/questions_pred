"""Predict the next topic for an author by name with the trained GRU.
usage: python src/predict.py "Author Name"
"""
import sys, numpy as np, pandas as pd, torch
from utils import load_config
from dataset import load_sequences
from models import GRUModel

if __name__ == "__main__":
    cfg = load_config(); k, L = cfg["n_clusters"], cfg["seq_len"]
    seqs = load_sequences()
    kw = pd.read_csv("data/processed/cluster_keywords.csv").set_index("cluster")
    df = pd.read_csv("data/processed/papers.csv")
    name = sys.argv[1] if len(sys.argv) > 1 else max(seqs, key=lambda a: len(seqs[a]))
    if name not in seqs:
        sys.exit(f"author not found; examples: {list(seqs)[:5]}")
    hist = [c for c, _, _ in seqs[name]][-L:]
    m = GRUModel(k, cfg["hidden"], L)
    m.load_state_dict(torch.load("models/gru.pt", map_location="cpu")); m.eval()
    x = torch.tensor([[k] * (L - len(hist)) + hist])
    p = torch.softmax(m(x), 1)[0].detach().numpy()
    print(f"History for {name}:")
    for c, y, i in seqs[name][-L:]:
        print(f"  {y} [cluster {c}] {df.title[i][:80]}")
    print("\nPredicted next topics:")
    for c in np.argsort(-p)[:5]:
        print(f"  cluster {c} p={p[c]:.2f}: {kw.keywords[c]}")
        for t in df[(df.cluster == c) & (df.year >= cfg['test_from_year'])].title.head(2):
            print(f"      e.g. {t[:90]}")
