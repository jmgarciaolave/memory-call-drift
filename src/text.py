"""Turn transcripts into features.

Two representations are built here, both without any pre-trained model:
  1. a simple word-count tone score, and
  2. LSA document vectors (TF-IDF followed by a truncated SVD).

Both are expressed relative to the company's own average, so the model sees
"how this call differs from this company's usual call" rather than "which
company is speaking".
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import SEED

# A short finance tone list we wrote ourselves. It is deliberately small and
# readable; it is not the Loughran-McDonald dictionary. Words that cut both ways
# in memory ("inventory", "demand", production "cuts") are left out.
POSITIVE = """strong stronger strongest strength record growth grow growing grew improve improved
improving improvement increase increased increasing higher exceed exceeded exceeding
robust solid healthy favorable favourable accelerate accelerating accelerated expand
expanding expansion momentum outperform outperformed tight tightness shortage
recovery recover recovering rebound upside profitable profitability gain gains
opportunity opportunities confident confidence optimistic pleased excellent
leadership ramp ramping""".split()

NEGATIVE = """weak weaker weakest weakness decline declined declining decrease decreased decreasing
lower fell fall falling drop dropped soft softer softness slow slower slowing slowdown
oversupply excess glut headwind headwinds pressure pressures challenging challenge
challenges difficult uncertainty uncertain loss losses impairment writedown
underutilization correction cautious caution deteriorate deteriorated
deteriorating disappoint disappointing disappointed delay delayed delays shortfall
miss missed unfavorable unfavourable downturn recession""".split()

_WORD = re.compile(r"[a-z]{3,}")


def load_transcripts(folder, call_ids):
    """Return {call_id: text}. Files are named YYYY-MM-DD__Producer__CALLID.txt."""
    paths = {os.path.basename(p)[:-4].split("__")[-1]: p
             for p in glob.glob(os.path.join(folder, "*", "*.txt"))}
    missing = [c for c in call_ids if c not in paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} transcripts not found, e.g. {missing[:3]}")
    out = {}
    for c in call_ids:
        with open(paths[c], encoding="utf-8", errors="ignore") as fh:
            out[c] = fh.read()
    return out


def tone_scores(texts):
    """Share of tone words that are positive minus share that are negative."""
    pos, neg = set(POSITIVE), set(NEGATIVE)
    rows = []
    for text in texts:
        words = _WORD.findall(text.lower())
        p = sum(w in pos for w in words)
        n = sum(w in neg for w in words)
        rows.append({"n_words": len(words),
                     "tone_net": (p - n) / max(p + n, 1),
                     "tone_density": (p + n) / max(len(words), 1)})
    return pd.DataFrame(rows)


def add_tone_features(df, texts):
    """Tone level and its change versus the same producer's previous call."""
    tone = tone_scores(texts)
    df = df.copy()
    df["tone_net"] = tone["tone_net"].to_numpy()
    df["n_words"] = tone["n_words"].to_numpy()
    df["tone_chg"] = (df.sort_values("call_date").groupby("producer")["tone_net"].diff())
    df["tone_chg"] = df["tone_chg"].fillna(0.0)
    return df


def _demean_by_group(matrix, groups, fit_mask):
    """Subtract each producer's average over the fitting rows."""
    out = matrix.copy()
    overall = matrix[fit_mask].mean(axis=0)
    for g in np.unique(groups):
        rows = groups == g
        ref = rows & fit_mask
        centre = matrix[ref].mean(axis=0) if ref.sum() >= 5 else overall
        out[rows] = matrix[rows] - centre
    return out


class LsaVectors:
    """TF-IDF + SVD fitted on the fitting rows only, then centred by producer."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform_all(self, texts, producers, fit_mask):
        texts = np.asarray(texts, dtype=object)
        producers = np.asarray(producers)
        self.tfidf_ = TfidfVectorizer(lowercase=True, stop_words="english",
                                      token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
                                      min_df=5, max_df=0.8, sublinear_tf=True,
                                      max_features=20000)
        self.tfidf_.fit(texts[fit_mask])
        counts = self.tfidf_.transform(texts)
        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=SEED)
        self.svd_.fit(counts[fit_mask])
        vectors = self.svd_.transform(counts)
        vectors = _demean_by_group(vectors, producers, fit_mask)
        scale = vectors[fit_mask].std(axis=0)
        scale[scale == 0] = 1.0
        return vectors / scale
