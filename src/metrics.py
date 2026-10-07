import numpy as np
from sklearn.metrics import f1_score


def evaluate(probs, y, ks=(1, 3, 5)):
    y = np.asarray(y); order = np.argsort(-probs, 1)
    res = {f"top{k}_acc": float((order[:, :k] == y[:, None]).any(1).mean()) for k in ks}
    ranks = np.array([np.where(order[i] == y[i])[0][0] + 1 for i in range(len(y))])
    res["MRR"] = float((1 / ranks).mean())
    res["macro_F1"] = float(f1_score(y, order[:, 0], average="macro"))
    return res
