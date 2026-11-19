"""Model families and the fit/predict helpers used by run_all.py.

Every helper takes a boolean `fit` mask (rows to learn from) and returns a
predicted probability for every row, so callers can score any split.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from .config import MLP_HIDDEN, MLP_SEEDS, SEED
from .data import NumericPrep
from .text import LsaVectors


def numeric_matrix(df, fit):
    return NumericPrep().fit(df[fit]).transform(df)


def fit_logit(x, y, fit, C):
    model = LogisticRegression(C=C, max_iter=2000)
    model.fit(x[fit], y[fit])
    return model.predict_proba(x)[:, 1], model


def fit_gbm(x, y, fit, n_estimators, max_depth):
    model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=SEED)
    model.fit(x[fit], y[fit])
    return model.predict_proba(x)[:, 1], model


def _mlp_ensemble(x_fit, y_fit, x_all, alpha):
    """Small one-hidden-layer network, averaged over several seeds."""
    probs = np.zeros(len(x_all))
    for s in range(MLP_SEEDS):
        net = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, alpha=alpha,
                            max_iter=3000, random_state=SEED + s)
        net.fit(x_fit, y_fit)
        probs += net.predict_proba(x_all)[:, 1]
    return probs / MLP_SEEDS


def text_network(vectors, y, fit, alpha):
    """Text-only model: network on document vectors. Returns a probability per call."""
    return _mlp_ensemble(vectors[fit], y[fit], vectors, alpha)


def lsa_vectors(texts, producers, fit, n_components):
    return LsaVectors(n_components).fit_transform_all(texts, producers, fit)


def text_score_out_of_fold(vectors, y, fit, alpha, folds=5):
    """Text score to feed the combined model.

    Rows used for fitting get an out-of-fold score (so the combined model does
    not see a text score that has already memorised their label); all other
    rows are scored by a network trained on every fitting row.
    """
    score = text_network(vectors, y, fit, alpha)
    idx = np.where(fit)[0]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, te in cv.split(idx, y[idx]):
        score[idx[te]] = _mlp_ensemble(vectors[idx[tr]], y[idx[tr]], vectors[idx[te]], alpha)
    return score
