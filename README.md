# Predicting a Researcher's Next Topic with Clustering + Neural Networks

Pipeline: arXiv papers -> SPECTER embeddings -> K-Means topic clusters -> per-author cluster
sequences -> Markov / GRU / Transformer / Hybrid models -> leak-free temporal evaluation.

## Run
```
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU build
python run_all.py                  # or run the steps in src/ one by one
python src/predict.py "Author Name"
```

## Steps
| script | what it does | output |
|---|---|---|
| src/collect_data.py | downloads papers via the arXiv API (categories in config.yaml) | data/raw/papers.csv |
| src/embed_cluster.py | embeds, clusters (fit on training years only), builds author sequences | data/processed/* |
| src/train.py | Majority, Markov, GRU, Transformer, Hybrid; top-k acc, MRR, macro-F1 | results/results.md |
| src/ablation.py | drops sequence / content features; cold-start vs established authors | results/ablation.md |
| src/emerging_topics.py | cluster growth per year + DBSCAN discovery of new dense groups | results/emerging_topics.* |
| src/predict.py | predicts the next topic for one author and lists example papers | console |

## Evaluation protocol
Split by the year of the *target* paper (train < 2024, val = 2024, test >= 2025, see config.yaml).
Clusters are fitted only on training-period papers so no future information leaks.
