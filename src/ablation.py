"""Feature ablation (sequence vs content) and cold-start study (<= 3 prior papers).
Writes results/ablation.json and results/ablation.md"""
import json, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from utils import load_config, set_seed, device
from dataset import build_examples, load_sequences
from models import HybridModel
from metrics import evaluate
from train import train_nn, predict, HybridDS, content_feats

if __name__ == "__main__":
    cfg = load_config(); set_seed(cfg["seed"]); dev = device()
    k, L = cfg["n_clusters"], cfg["seq_len"]
    seqs = load_sequences(); emb = np.load("data/processed/embeddings.npy")
    sp = {s: build_examples(seqs, L, s, cfg) for s in ["train", "val", "test"]}
    C = {s: content_feats(m, seqs, emb, L) for s, (_, _, m) in sp.items()}
    out = {}
    variants = {"full": (False, False), "no_sequence": (True, False), "no_content": (False, True)}
    for name, (drop_seq, drop_content) in variants.items():
        set_seed(cfg["seed"]); loaders = {}
        for s, (X, y, _) in sp.items():
            X_ = [[k] * len(h) for h in X] if drop_seq else X
            C_ = torch.zeros_like(C[s]) if drop_content else C[s]
            loaders[s] = DataLoader(HybridDS(X_, y, L, k, C_), batch_size=cfg["batch_size"], shuffle=(s == "train"))
        print(f"\n== ablation: {name}")
        m = train_nn(HybridModel(k, cfg["hidden"], L, emb.shape[1]).to(dev), loaders["train"], loaders["val"], cfg, dev, True)
        P = predict(m, loaders["test"], dev, True); yte = np.array(sp["test"][1])
        out[name] = evaluate(P, yte, cfg["top_k"])
        if name == "full":
            cold = np.array([t <= 3 for _, t in sp["test"][2]])
            if cold.any():
                out["full_cold_start(<=3 prior)"] = evaluate(P[cold], yte[cold], cfg["top_k"])
            if (~cold).any():
                out["full_established(>3 prior)"] = evaluate(P[~cold], yte[~cold], cfg["top_k"])
    json.dump(out, open("results/ablation.json", "w"), indent=2)
    t = pd.DataFrame(out).T.round(3)
    with open("results/ablation.md", "w") as f:
        f.write(t.to_markdown())
    print("\n", t.to_string())
