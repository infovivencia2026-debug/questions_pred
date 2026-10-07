"""Emerging-topic detection: cluster growth per year, plus DBSCAN discovery of dense groups
among the newest papers. Writes results/emerging_topics.csv and results/emerging_topics.png"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from utils import load_config

if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_csv("data/processed/papers.csv")
    kw = pd.read_csv("data/processed/cluster_keywords.csv").set_index("cluster")
    emb = np.load("data/processed/embeddings.npy")
    share = pd.crosstab(df.year, df.cluster, normalize="index")
    years = sorted(share.index)
    if len(years) >= 2:
        prev, now = share.loc[years[-2]], share.loc[years[-1]]
        growth = (now - prev) / (prev + 1e-6)
        res = pd.DataFrame(dict(growth=growth.round(2), share_now=now.round(3),
                                keywords=kw.keywords.reindex(growth.index)))
        res = res.sort_values("growth", ascending=False)
        res.to_csv("results/emerging_topics.csv")
        print(f"Fastest-growing topics ({years[-2]} -> {years[-1]}):\n", res.head(10).to_string())
        share[res.index[:6]].plot(figsize=(9, 5), marker="o")
        plt.ylabel("share of papers"); plt.title("Top emerging clusters"); plt.tight_layout()
        plt.savefig("results/emerging_topics.png")
    new = df[df.year >= cfg["test_from_year"]]
    if len(new):
        lab = DBSCAN(eps=0.25, min_samples=5, metric="cosine").fit_predict(emb[new.index])
        found = sorted([(l, int((lab == l).sum()), new.title.values[lab == l][:3].tolist())
                        for l in set(lab) if l != -1], key=lambda x: -x[1])
        print(f"\nDBSCAN found {len(found)} dense groups among {len(new)} recent papers")
        for l, n, titles in found[:10]:
            print(f"  group {l} ({n} papers): {titles[0][:90]}")
