"""Two follow-up analyses run after the fixed-split result.

1. Walk-forward test: retrain every year on all earlier calls and predict that
   year's calls. Settings are the ones chosen in the main run; nothing is
   re-tuned. This gives ~170 out-of-sample calls instead of 54.
2. Does our tone measure line up with the day-one move and with the later
   drift? Rank correlations over the full sample.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import evaluate as ev
from . import models as m

FIRST_YEAR = 2015          # leaves five years (about 70 calls) for the first fit


FB_NET = "Text: neural network on FinBERT embeddings"
FB_TONE = "Numbers + FinBERT tone"


def walk_forward(df, texts, y, outcome, settings, tone_matrix, finbert=None):
    """`finbert`, if given, is a dict with `vectors(fit)` (PCA of the FinBERT
    embeddings, fitted on `fit`) and `tone(fit)` (standardised FinBERT tone)."""
    years = df["call_date"].dt.year.to_numpy()
    producers = df["producer"].to_numpy()
    names = ["Numbers: logistic regression", "Numbers: gradient-boosted trees",
             "Text: neural network on call vectors", "Numbers + word-count tone",
             "Numbers + text network"]
    if finbert is not None:
        names += [FB_NET, FB_TONE]
    pred = {n: np.full(len(df), np.nan) for n in names}
    k, alpha = settings[names[2]]["lsa_components"], settings[names[2]]["alpha"]

    for year in range(FIRST_YEAR, years.max() + 1):
        fit, now = years < year, years == year
        if now.sum() == 0:
            continue
        x = m.numeric_matrix(df, fit)
        pred[names[0]][now] = m.fit_logit(x, y, fit, settings[names[0]]["C"])[0][now]
        pred[names[1]][now] = m.fit_gbm(x, y, fit, **settings[names[1]])[0][now]
        vec = m.lsa_vectors(texts, producers, fit, k)
        pred[names[2]][now] = m.text_network(vec, y, fit, alpha)[now]
        xt = np.column_stack([x, tone_matrix(fit)])
        pred[names[3]][now] = m.fit_logit(xt, y, fit, settings[names[3]]["C"])[0][now]
        s = m.text_score_out_of_fold(vec, y, fit, alpha)
        s = (s - s[fit].mean()) / max(s[fit].std(), 1e-9)
        pred[names[4]][now] = m.fit_logit(np.column_stack([x, s]), y, fit,
                                          settings[names[4]]["C"])[0][now]
        if finbert is not None:
            fv = finbert["vectors"](fit)
            pred[FB_NET][now] = m.text_network(fv, y, fit, settings[FB_NET]["alpha"])[now]
            xf = np.column_stack([x, finbert["tone"](fit)])
            pred[FB_TONE][now] = m.fit_logit(xf, y, fit, settings[FB_TONE]["C"])[0][now]

    scored = years >= FIRST_YEAR
    ys, os_ = y[scored], outcome[scored]
    rule = df["rule_follow_day1"].to_numpy().astype(float)[scored]
    rows = [{"model": "Benchmark: always the majority class", "n_calls": int(scored.sum()),
             "auc": 0.5, "auc_lo": np.nan, "auc_hi": np.nan,
             "accuracy": float(1 - ys.mean()), "spread": float(-os_.mean()),
             "spread_lo": np.nan, "spread_hi": np.nan},
            {"model": "Benchmark: follow the day-one move", "n_calls": int(scored.sum()),
             "auc": ev.auc(ys, rule), "auc_lo": np.nan, "auc_hi": np.nan,
             "accuracy": ev.accuracy(ys, rule), "spread": ev.rule_spread(rule, os_),
             "spread_lo": np.nan, "spread_hi": np.nan}]
    for n in names:
        p = pred[n][scored]
        lo, hi = ev.bootstrap_ci(ev.auc, len(ys), ys, p)
        slo, shi = ev.bootstrap_ci(ev.long_short, len(ys), p, os_)
        rows.append({"model": n, "n_calls": int(scored.sum()), "auc": ev.auc(ys, p),
                     "auc_lo": lo, "auc_hi": hi, "accuracy": ev.accuracy(ys, p),
                     "spread": ev.long_short(p, os_), "spread_lo": slo, "spread_hi": shi})
    return pd.DataFrame(rows)


def tone_checks(df, tone_std):
    """Rank correlation of tone with the day-one move and with the later drift."""
    rows = []
    for label, series in [("tone level (vs company average)", tone_std),
                          ("tone change vs previous call", df["tone_chg"].to_numpy())]:
        for target, col in [("day-one move vs SOX", "rel_fwd_1d"),
                            ("days 2-20 return vs SOX", "rel_drift_2_20d")]:
            rho, p = spearmanr(series, df[col].to_numpy())
            rows.append({"text_measure": label, "compared_with": target,
                         "spearman_rho": rho, "p_value": p, "n_calls": len(df)})
    return pd.DataFrame(rows)


def sign_by_period(df):
    """Rank correlation of each input with the outcome, before and after 2023."""
    from .config import NUMERIC_FEATURES, OUTCOME
    early, late = df["split"] != "test", df["split"] == "test"
    rows = []
    for f in NUMERIC_FEATURES + ["tone_net", "tone_chg"]:
        r = {"feature": f}
        for name, mask in [("2010_2022", early), ("2023_2026", late)]:
            sub = df.loc[mask, [f, OUTCOME]].dropna()
            r[f"rho_{name}"], r[f"p_{name}"] = spearmanr(sub[f], sub[OUTCOME])
        rows.append(r)
    return pd.DataFrame(rows)
