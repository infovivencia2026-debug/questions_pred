"""Step 2: embed title+abstract, cluster into topics (fit ONLY on training-period papers
to avoid temporal leakage), and build per-author topic sequences.
Outputs: data/processed/embeddings.npy, papers.csv (with cluster), sequences.pkl, cluster_keywords.csv"""
import pickle, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from utils import load_config, set_seed, device

if __name__ == "__main__":
    cfg = load_config(); set_seed(cfg["seed"])
    df = pd.read_csv("data/raw/papers.csv")
    texts = (df.title + ". " + df.abstract).tolist()

    model = SentenceTransformer(cfg["embedding_model"], device=device())
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save("data/processed/embeddings.npy", emb)

    train_mask = (df.year < cfg["train_before_year"]).values
    k = cfg["n_clusters"]
    if cfg["clustering"] == "kmeans":
        clu = KMeans(k, n_init=10, random_state=cfg["seed"]).fit(emb[train_mask])
        df["cluster"] = clu.predict(emb)
    else:
        agg = AgglomerativeClustering(k).fit(emb[train_mask])
        centers = np.stack([emb[train_mask][agg.labels_ == c].mean(0) for c in range(k)])
        df["cluster"] = (emb @ centers.T).argmax(1)

    # Human-readable keywords per cluster
    tf = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    X = tf.fit_transform(texts); vocab = np.array(tf.get_feature_names_out())
    kw = []
    for c in range(k):
        m = (df.cluster == c).values
        top = np.asarray(X[m].mean(0)).ravel().argsort()[::-1][:8]
        kw.append(dict(cluster=c, size=int(m.sum()), keywords=", ".join(vocab[top])))
    pd.DataFrame(kw).to_csv("data/processed/cluster_keywords.csv", index=False)

    # Per-author chronological sequences of (cluster, year, paper_index)
    seqs = defaultdict(list)
    for i, r in df.sort_values("date").iterrows():
        for a in str(r.authors).split("|"):
            seqs[a].append((int(r.cluster), int(r.year), int(i)))
    seqs = {a: s for a, s in seqs.items() if len(s) >= cfg["min_papers_per_author"]}
    df.to_csv("data/processed/papers.csv", index=False)
    with open("data/processed/sequences.pkl", "wb") as f:
        pickle.dump(seqs, f)
    print(f"{len(df)} papers, {k} clusters, {len(seqs)} authors with >= {cfg['min_papers_per_author']} papers")
    print(pd.DataFrame(kw).to_string(index=False))
