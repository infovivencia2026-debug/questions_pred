"""Temporal, leak-free examples: predict the cluster of the paper published in a given year
from the previous `seq_len` papers of that author. Split by the TARGET paper's year."""
import pickle, torch
from torch.utils.data import Dataset


def build_examples(seqs, seq_len, split, cfg):
    X, y, meta = [], [], []
    for author, s in seqs.items():
        clusters = [c for c, _, _ in s]
        for t in range(1, len(s)):
            year = s[t][1]
            if split == "train" and not year < cfg["train_before_year"]:
                continue
            if split == "val" and year != cfg["val_year"]:
                continue
            if split == "test" and not year >= cfg["test_from_year"]:
                continue
            hist = clusters[max(0, t - seq_len):t]
            X.append(hist); y.append(clusters[t]); meta.append((author, t))
    return X, y, meta


class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len, pad):
        self.X = [[pad] * (seq_len - len(h)) + h for h in X]
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])


def load_sequences(path="data/processed/sequences.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
