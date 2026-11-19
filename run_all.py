"""Run the whole analysis: features, models, evaluation, figures.

    python run_all.py --events data/events.csv --transcripts data/transcripts_text

Add --finbert data/text_features.csv to include the FinBERT models (see
scripts/score_finbert.py). Outputs go to results/.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src import evaluate as ev
from src import models as m
from src.config import (GBM_GRID, LOGIT_C, LSA_COMPONENTS, MLP_ALPHA, NUMERIC_FEATURES,
                        OUTCOME, TARGET)
from src.data import load_events
from src.text import add_tone_features, load_transcripts


def standardise(col, groups, fit):
    """Centre on each producer's fitting-row mean, scale by the fitting-row spread."""
    out = col.astype(float).copy()
    overall = col[fit].mean()
    for g in np.unique(groups):
        rows = groups == g
        ref = rows & fit
        out[rows] = col[rows] - (col[ref].mean() if ref.sum() >= 5 else overall)
    sd = out[fit].std()
    return out / (sd if sd > 0 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events.csv")
    ap.add_argument("--transcripts", default="data/transcripts_text")
    ap.add_argument("--finbert", default=None, help="optional text_features.csv from FinBERT")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    df = load_events(args.events)
    texts_by_id = load_transcripts(args.transcripts, df["primary_call_id"].tolist())
    texts = [texts_by_id[c] for c in df["primary_call_id"]]
    df = add_tone_features(df, texts)

    y = df[TARGET].to_numpy()
    outcome = df[OUTCOME].to_numpy()
    producers = df["producer"].to_numpy()
    train = (df["split"] == "train").to_numpy()
    val = (df["split"] == "validation").to_numpy()
    test = (df["split"] == "test").to_numpy()
    fit1, fit2 = train, train | val          # stage 1: choose settings; stage 2: final fit

    val_pred, test_pred, chosen = {}, {}, {}

    def register(name, stage1, stage2, settings):
        val_pred[name], test_pred[name], chosen[name] = stage1[val], stage2[test], settings

    # ---- benchmarks ---------------------------------------------------------
    register("Benchmark: always the majority class",
             np.full(len(df), y[fit1].mean()), np.full(len(df), y[fit2].mean()), {})
    rule = df["rule_follow_day1"].to_numpy().astype(float)
    register("Benchmark: follow the day-one move", rule, rule, {})

    # ---- numbers only -------------------------------------------------------
    x1, x2 = m.numeric_matrix(df, fit1), m.numeric_matrix(df, fit2)

    best = max(LOGIT_C, key=lambda c: ev.auc(y[val], m.fit_logit(x1, y, fit1, c)[0][val]))
    p2, logit_final = m.fit_logit(x2, y, fit2, best)
    register("Numbers: logistic regression", m.fit_logit(x1, y, fit1, best)[0], p2, {"C": best})

    best = max(GBM_GRID, key=lambda g: ev.auc(y[val], m.fit_gbm(x1, y, fit1, **g)[0][val]))
    register("Numbers: gradient-boosted trees",
             m.fit_gbm(x1, y, fit1, **best)[0], m.fit_gbm(x2, y, fit2, **best)[0], best)

    # ---- text only: our network on LSA call vectors --------------------------
    grid = [(k, a) for k in LSA_COMPONENTS for a in MLP_ALPHA]
    vec1 = {k: m.lsa_vectors(texts, producers, fit1, k) for k in LSA_COMPONENTS}
    scores = {(k, a): m.text_network(vec1[k], y, fit1, a) for k, a in grid}
    k, a = max(grid, key=lambda g: ev.auc(y[val], scores[g][val]))
    vec2 = m.lsa_vectors(texts, producers, fit2, k)
    register("Text: neural network on call vectors",
             scores[(k, a)], m.text_network(vec2, y, fit2, a), {"lsa_components": k, "alpha": a})

    # ---- numbers + simple tone ----------------------------------------------
    def with_tone(x, fit):
        tone = standardise(df["tone_net"].to_numpy(), producers, fit)
        chg = df["tone_chg"].to_numpy() / max(df["tone_chg"].to_numpy()[fit].std(), 1e-9)
        return np.column_stack([x, tone, chg])

    xt1, xt2 = with_tone(x1, fit1), with_tone(x2, fit2)
    best = max(LOGIT_C, key=lambda c: ev.auc(y[val], m.fit_logit(xt1, y, fit1, c)[0][val]))
    p2, tone_final = m.fit_logit(xt2, y, fit2, best)
    register("Numbers + word-count tone", m.fit_logit(xt1, y, fit1, best)[0], p2, {"C": best})

    # ---- numbers + text network (late fusion) --------------------------------
    s1 = m.text_score_out_of_fold(vec1[k], y, fit1, a)
    s2 = m.text_score_out_of_fold(vec2, y, fit2, a)
    z = lambda s, fit: (s - s[fit].mean()) / max(s[fit].std(), 1e-9)
    xc1, xc2 = np.column_stack([x1, z(s1, fit1)]), np.column_stack([x2, z(s2, fit2)])
    best = max(LOGIT_C, key=lambda c: ev.auc(y[val], m.fit_logit(xc1, y, fit1, c)[0][val]))
    register("Numbers + text network", m.fit_logit(xc1, y, fit1, best)[0],
             m.fit_logit(xc2, y, fit2, best)[0], {"C": best})

    # ---- optional: FinBERT features -----------------------------------------
    finbert_wf = None                      # handed to the walk-forward test below
    if args.finbert and not os.path.exists(args.finbert):
        raise FileNotFoundError(f"--finbert file not found: {args.finbert}")
    if args.finbert and os.path.exists(args.finbert):
        from sklearn.decomposition import PCA
        fb = pd.read_csv(args.finbert).set_index("call_id").loc[df["primary_call_id"]]
        emb = fb.filter(like="emb_").to_numpy()

        def fb_vectors(fit, k=10):
            pcs = PCA(n_components=k, random_state=0).fit(emb[fit]).transform(emb)
            pcs = np.column_stack([standardise(pcs[:, j], producers, fit) for j in range(k)])
            return pcs

        f1, f2 = fb_vectors(fit1), fb_vectors(fit2)
        a_fb = max(MLP_ALPHA, key=lambda al: ev.auc(y[val], m.text_network(f1, y, fit1, al)[val]))
        register("Text: neural network on FinBERT embeddings",
                 m.text_network(f1, y, fit1, a_fb), m.text_network(f2, y, fit2, a_fb), {"alpha": a_fb})
        ft = fb["tone_net"].to_numpy()
        xf1 = np.column_stack([x1, standardise(ft, producers, fit1)])
        xf2 = np.column_stack([x2, standardise(ft, producers, fit2)])
        best = max(LOGIT_C, key=lambda c: ev.auc(y[val], m.fit_logit(xf1, y, fit1, c)[0][val]))
        register("Numbers + FinBERT tone", m.fit_logit(xf1, y, fit1, best)[0],
                 m.fit_logit(xf2, y, fit2, best)[0], {"C": best})
        finbert_wf = {"vectors": fb_vectors,
                      "tone": lambda fit: standardise(ft, producers, fit)[:, None]}

    # ---- evaluation ----------------------------------------------------------
    rows = []
    yt, ot = y[test], outcome[test]
    for name in test_pred:
        pv, pt = val_pred[name], test_pred[name]
        is_rule = name.startswith("Benchmark: follow")
        is_major = name.startswith("Benchmark: always")
        if is_major:                       # "always underperform" = short every call
            spread, ci = float(-ot.mean()), ev.bootstrap_ci(lambda o: -o.mean(), len(ot), ot)
            v_spread = float(-outcome[val].mean())
        elif is_rule:
            spread = ev.rule_spread(pt, ot)
            ci = ev.bootstrap_ci(ev.rule_spread, len(ot), pt, ot)
            v_spread = ev.rule_spread(pv, outcome[val])
        else:
            spread = ev.long_short(pt, ot)
            ci = ev.bootstrap_ci(ev.long_short, len(ot), pt, ot)
            v_spread = ev.long_short(pv, outcome[val])
        auc_ci = (np.nan, np.nan) if is_major else ev.bootstrap_ci(ev.auc, len(yt), yt, pt)
        rows.append({"model": name, "settings": json.dumps(chosen[name]),
                     "val_auc": ev.auc(y[val], pv), "val_accuracy": ev.accuracy(y[val], pv),
                     "val_spread": v_spread,
                     "test_auc": ev.auc(yt, pt), "test_auc_lo": auc_ci[0], "test_auc_hi": auc_ci[1],
                     "test_accuracy": ev.accuracy(yt, pt),
                     "test_spread": spread, "test_spread_lo": ci[0], "test_spread_hi": ci[1]})
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(args.out, "results.csv"), index=False, float_format="%.6f")

    # coefficients of the two interpretable models (final fit, standardised inputs)
    coefs = pd.DataFrame({"feature": NUMERIC_FEATURES + ["tone_net", "tone_chg"],
                          "numbers_only": list(logit_final.coef_[0]) + [np.nan, np.nan],
                          "numbers_plus_tone": tone_final.coef_[0]})
    coefs.to_csv(os.path.join(args.out, "coefficients.csv"), index=False, float_format="%.6f")

    # robustness: the same test predictions without Samsung
    keep = (df["producer"] != "Samsung").to_numpy()[test]
    robust = [{"model": n, "test_auc_ex_samsung": ev.auc(yt[keep], p[keep]),
               "test_accuracy_ex_samsung": ev.accuracy(yt[keep], p[keep])}
              for n, p in test_pred.items()]
    pd.DataFrame(robust).to_csv(os.path.join(args.out, "robustness_ex_samsung.csv"),
                                index=False, float_format="%.6f")

    # ---- follow-up analyses (see src/diagnostics.py) -----------------------------
    from src import diagnostics as dg

    def tone_matrix(fit):
        tone = standardise(df["tone_net"].to_numpy(), producers, fit)
        chg = df["tone_chg"].to_numpy() / max(df["tone_chg"].to_numpy()[fit].std(), 1e-9)
        return np.column_stack([tone, chg])

    wf = dg.walk_forward(df, texts, y, outcome, chosen, tone_matrix, finbert=finbert_wf)
    wf.to_csv(os.path.join(args.out, "walk_forward.csv"), index=False, float_format="%.6f")
    everyone = np.ones(len(df), dtype=bool)
    tone_std = standardise(df["tone_net"].to_numpy(), producers, everyone)
    dg.tone_checks(df, tone_std).to_csv(os.path.join(args.out, "tone_checks.csv"),
                                        index=False, float_format="%.6f")
    signs = dg.sign_by_period(df)
    signs.to_csv(os.path.join(args.out, "sign_by_period.csv"), index=False, float_format="%.6f")

    summary = {"n_calls": int(len(df)), "n_train": int(train.sum()), "n_val": int(val.sum()),
               "n_test": int(test.sum()),
               "share_up": {s: float(y[msk].mean()) for s, msk in
                            [("train", train), ("validation", val), ("test", test)]},
               "calls_by_producer": df["producer"].value_counts().to_dict(),
               "first_call": str(df["call_date"].min().date()),
               "last_call": str(df["call_date"].max().date()),
               "median_words": int(df["n_words"].median()),
               "mean_outcome": {s: float(outcome[msk].mean()) for s, msk in
                                [("train", train), ("validation", val), ("test", test)]}}
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # per-call test predictions: handy for checking, but derived from licensed data
    preds = df.loc[test, ["event_id", "call_date", "producer", TARGET, OUTCOME]].copy()
    for n, p in test_pred.items():
        preds[n] = p
    preds.to_csv(os.path.join(args.out, "test_predictions.csv"), index=False, float_format="%.6f")

    from src.plots import make_figures
    make_figures(df, results, wf, signs, os.path.join(args.out, "figures"))
    print(results[["model", "val_auc", "test_auc", "test_accuracy", "test_spread"]].to_string(index=False))


if __name__ == "__main__":
    main()
