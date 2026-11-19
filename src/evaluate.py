"""Metrics and the long/short backtest."""
import numpy as np
from sklearn.metrics import roc_auc_score

from .config import BOOTSTRAP_DRAWS, SEED


def accuracy(y, p):
    return float(((p >= 0.5).astype(int) == y).mean())


def auc(y, p):
    if len(np.unique(p)) < 2 or len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def long_short(p, outcome, frac=1 / 3):
    """Average outcome of the top-ranked third minus the bottom-ranked third."""
    order = np.argsort(p, kind="stable")
    k = max(int(round(len(p) * frac)), 1)
    return float(outcome[order[-k:]].mean() - outcome[order[:k]].mean())


def rule_spread(signal, outcome):
    """Long when the rule says up, short when it says down."""
    up, down = outcome[signal == 1], outcome[signal == 0]
    if len(up) == 0 or len(down) == 0:
        return float("nan")
    return float(up.mean() - down.mean())


def bootstrap_ci(stat, n, *arrays, level=0.90):
    """Percentile interval from resampling calls with replacement."""
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n, n)
        try:
            draws.append(stat(*[a[idx] for a in arrays]))
        except ValueError:
            continue
    draws = np.array([d for d in draws if not np.isnan(d)])
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
