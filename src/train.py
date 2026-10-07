"""Step 3: train and compare Majority, Markov (clusters only), GRU, Transformer,
Hybrid (clusters + content). Writes results/results.json and results/results.md"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_config, set_seed, device
from dataset import build_examples, SeqDataset, load_sequences
from models import MarkovBaseline, GRUModel, TransformerModel, HybridModel
from metrics import evaluate


def content_feats(meta, seqs, emb, seq_len):
    out = []
    for author, t in meta:
        idx = [p for _, _, p in seqs[author][max(0, t - seq_len):t]]
        out.append(emb[idx].mean(0))
    return torch.tensor(np.stack(out), dtype=torch.float32)


class HybridDS(SeqDataset):
    def __init__(self, X, y, seq_len, pad, C):
        super().__init__(X, y, seq_len, pad); self.C = C

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x, y, self.C[i]


@torch.no_grad()
def predict(model, loader, dev, hybrid=False):
    model.eval(); P = []
    for batch in loader:
        batch = [b.to(dev) for b in batch]
        logits = model(batch[0], batch[2]) if hybrid else model(batch[0])
        P.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(P)


def train_nn(model, tr, va, cfg, dev, hybrid=False):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(); best, best_state = -1, None
    yva = np.array(va.dataset.y)
    for ep in range(cfg["epochs"]):
        model.train()
        for batch in tr:
            batch = [b.to(dev) for b in batch]
            logits = model(batch[0], batch[2]) if hybrid else model(batch[0])
            loss = lossf(logits, batch[1])
            opt.zero_grad(); loss.backward(); opt.step()
        acc = (predict(model, va, dev, hybrid).argmax(1) == yva).mean()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep+1:02d} loss={loss.item():.3f} val top1={acc:.3f}")
    model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    cfg = load_config(); set_seed(cfg["seed"]); dev = device()
    k, L = cfg["n_clusters"], cfg["seq_len"]
    seqs = load_sequences(); emb = np.load("data/processed/embeddings.npy")
    splits = {s: build_examples(seqs, L, s, cfg) for s in ["train", "val", "test"]}
    for s, (X, y, _) in splits.items():
        print(f"{s}: {len(y)} examples")
    Xtr, ytr, mtr = splits["train"]; Xva, yva, mva = splits["val"]; Xte, yte, mte = splits["test"]
    results = {}

    maj = np.bincount(ytr, minlength=k) / len(ytr)
    results["Majority"] = evaluate(np.tile(maj, (len(yte), 1)), yte, cfg["top_k"])
    results["Markov (clusters only)"] = evaluate(MarkovBaseline(k).fit(Xtr, ytr).predict_proba(Xte), yte, cfg["top_k"])

    def mk(X, y, shuffle=False):
        return DataLoader(SeqDataset(X, y, L, k), batch_size=cfg["batch_size"], shuffle=shuffle)

    for name, M in [("GRU", GRUModel), ("Transformer", TransformerModel)]:
        print(f"\n== {name}"); set_seed(cfg["seed"])
        m = train_nn(M(k, cfg["hidden"], L).to(dev), mk(Xtr, ytr, True), mk(Xva, yva), cfg, dev)
        results[name] = evaluate(predict(m, mk(Xte, yte), dev), yte, cfg["top_k"])
        torch.save(m.state_dict(), f"models/{name.lower()}.pt")

    print("\n== Hybrid (GRU + content embeddings)"); set_seed(cfg["seed"])
    C = {s: content_feats(m_, seqs, emb, L) for s, (_, _, m_) in splits.items()}

    def hl(X, y, C_, shuffle=False):
        return DataLoader(HybridDS(X, y, L, k, C_), batch_size=cfg["batch_size"], shuffle=shuffle)

    m = train_nn(HybridModel(k, cfg["hidden"], L, emb.shape[1]).to(dev),
                 hl(Xtr, ytr, C["train"], True), hl(Xva, yva, C["val"]), cfg, dev, hybrid=True)
    results["Hybrid (clusters + content)"] = evaluate(predict(m, hl(Xte, yte, C["test"]), dev, True), yte, cfg["top_k"])
    torch.save(m.state_dict(), "models/hybrid.pt")

    json.dump(results, open("results/results.json", "w"), indent=2)
    table = pd.DataFrame(results).T.round(3)
    with open("results/results.md", "w") as f:
        f.write(table.to_markdown())
    print("\n", table.to_string())
