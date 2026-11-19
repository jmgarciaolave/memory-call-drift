"""Placebo test: how much does AUC move when the text input is pure noise?

    python scripts/placebo.py --finbert data/text_features.csv

The bootstrap intervals in results/ resample the predictions of one fitted
model; they do not capture the variation that comes from fitting the model
itself. On 132-180 training calls that variation is large. Here we rerun the
two FinBERT models of run_all.py many times with the FinBERT input replaced by
random numbers of the same shape, keeping everything else (numbers, settings,
splits, walk-forward years) identical. The spread of those AUCs is what chance
alone produces; the real FinBERT result is then placed in that distribution.

Added after the main results had been seen. Writes results/placebo.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_all import standardise  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import models as m  # noqa: E402
from src.config import TARGET  # noqa: E402
from src.data import load_events  # noqa: E402
from src.diagnostics import FIRST_YEAR  # noqa: E402

NAMES = {"tone": "Numbers + FinBERT tone", "net": "Text: neural network on FinBERT embeddings"}


def run(df, y, x_numbers, tone, pcs, settings, producers):
    """Fixed-split test AUC and walk-forward AUC for the two FinBERT models."""
    years = df["call_date"].dt.year.to_numpy()
    train, val, test = ((df["split"] == s).to_numpy() for s in ("train", "validation", "test"))
    fit2 = train | val
    out = {}
    # numbers + tone (logistic regression, C fixed at the value chosen in run_all.py)
    c = settings[NAMES["tone"]]["C"]
    p = m.fit_logit(np.column_stack([x_numbers(fit2), standardise(tone, producers, fit2)]),
                    y, fit2, c)[0]
    out["tone_test"] = ev.auc(y[test], p[test])
    # network on 10 components (alpha fixed at the value chosen in run_all.py)
    a = settings[NAMES["net"]]["alpha"]
    out["net_test"] = ev.auc(y[test], m.text_network(pcs(fit2), y, fit2, a)[test])
    wf_tone, wf_net = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for year in range(FIRST_YEAR, years.max() + 1):
        fit, now = years < year, years == year
        if not now.any():
            continue
        xt = np.column_stack([x_numbers(fit), standardise(tone, producers, fit)])
        wf_tone[now] = m.fit_logit(xt, y, fit, c)[0][now]
        wf_net[now] = m.text_network(pcs(fit), y, fit, a)[now]
    sc = years >= FIRST_YEAR
    out["tone_wf"] = ev.auc(y[sc], wf_tone[sc])
    out["net_wf"] = ev.auc(y[sc], wf_net[sc])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events.csv")
    ap.add_argument("--finbert", default="data/text_features.csv")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from sklearn.decomposition import PCA
    df = load_events(args.events)
    y = df[TARGET].to_numpy()
    producers = df["producer"].to_numpy()
    res = pd.read_csv(args.results).set_index("model")
    settings = {n: json.loads(res.loc[n, "settings"]) for n in NAMES.values()}
    fb = pd.read_csv(args.finbert).set_index("call_id").loc[df["primary_call_id"]]
    emb = fb.filter(like="emb_").to_numpy()

    def x_numbers(fit):
        return m.numeric_matrix(df, fit)

    def pcs_from(matrix):
        def f(fit, k=10):
            z = PCA(n_components=k, random_state=0).fit(matrix[fit]).transform(matrix)
            return np.column_stack([standardise(z[:, j], producers, fit) for j in range(k)])
        return f

    real = run(df, y, x_numbers, fb["tone_net"].to_numpy(), pcs_from(emb), settings, producers)
    rng = np.random.default_rng(2026)
    draws = []
    for i in range(args.draws):
        noise_emb = rng.normal(size=emb.shape)
        noise_tone = rng.normal(size=len(df))
        draws.append(run(df, y, x_numbers, noise_tone, pcs_from(noise_emb), settings, producers))
        print(f"{i + 1}/{args.draws} placebo draws", flush=True)
    draws = pd.DataFrame(draws)

    rows = []
    for key, label in [("tone_test", "Numbers + FinBERT tone, test AUC"),
                       ("tone_wf", "Numbers + FinBERT tone, walk-forward AUC"),
                       ("net_test", "Network on FinBERT embeddings, test AUC"),
                       ("net_wf", "Network on FinBERT embeddings, walk-forward AUC")]:
        d = draws[key]
        rows.append({"measure": label, "real_finbert": real[key],
                     "placebo_mean": d.mean(), "placebo_p05": d.quantile(0.05),
                     "placebo_p95": d.quantile(0.95),
                     "share_placebo_at_or_above_real": float((d >= real[key]).mean()),
                     "draws": len(d)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, "placebo.csv"), index=False, float_format="%.4f")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
