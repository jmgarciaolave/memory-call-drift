"""Follow-up analyses added after the main run.

    python scripts/extra_analyses.py --finbert data/text_features.csv

1. FinBERT checks: does FinBERT tone line up with the day-one move and with
   the days 2-20 drift? Tone level (relative to the company's average) and tone
   change (versus the same company's previous call); whole call, prepared
   remarks and Q&A; all companies, US-listed and Korea-listed.
1b. When does the market use the call? US-listed calls grouped into thirds by
   tone change: average day-one move and days 2-20 return vs SOX, with a 90%
   bootstrap interval for the top-minus-bottom gap.
2. Section tone in the drift model: numbers + FinBERT tone of the remarks and
   of the Q&A, same protocol as run_all.py (setting chosen on validation,
   fixed test scored once, walk-forward from 2015 with the setting fixed).
3. Exploratory: predict the day-one move itself (stock beats SOX on day one)
   from information available before day one: pre-call numbers and the call
   text. Not tradable with our data: the target uses closing prices, and for
   the Korean names the call happens during day one.

These analyses were designed after the main results had been seen. They are
reported as such, and nothing in them was tuned on the test period.
Writes results/extra_*.csv (aggregates only, safe to commit).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_all import standardise  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import models as m  # noqa: E402
from src.config import GBM_GRID, LOGIT_C, MLP_ALPHA, NUMERIC_FEATURES, OUTCOME, TARGET  # noqa: E402
from src.data import load_events  # noqa: E402
from src.diagnostics import FIRST_YEAR  # noqa: E402

# Day-one inputs are not available before day one, so the day-one model drops them.
PRE_CALL_FEATURES = [f for f in NUMERIC_FEATURES if f not in ("rel_fwd_1d", "sox_fwd_1d")]
DAY1_TARGET = "target_day1_up"
# Report after the US close, so day one is the next session. The Korean names report during
# Korean trading hours, after preliminary results have already been published.
US_LISTED = ("Micron", "Sandisk_Lineage")


def prep(df, cols, fit):
    """Median-impute, standardise and clip on the fitting rows (as src.data.NumericPrep)."""
    x = df[cols]
    med = x[fit].median()
    filled = x.fillna(med)
    mu, sd = filled[fit].mean(), filled[fit].std().replace(0, 1.0)
    return ((filled - mu) / sd).clip(-3, 3).to_numpy()


def fb_pcs(emb, producers, fit, k=10):
    from sklearn.decomposition import PCA
    pcs = PCA(n_components=k, random_state=0).fit(emb[fit]).transform(emb)
    return np.column_stack([standardise(pcs[:, j], producers, fit) for j in range(k)])


def evaluate_family(name, build, y, outcome, splits, years, model="logit"):
    """Choose the setting on validation, score the fixed test once, then walk forward."""
    train, val, test = splits
    grid = {"logit": LOGIT_C, "mlp": MLP_ALPHA, "gbm": GBM_GRID}[model]

    def predict(fit, setting):
        x = build(fit)
        if model == "logit":
            return m.fit_logit(x, y, fit, setting)[0]
        if model == "gbm":
            return m.fit_gbm(x, y, fit, **setting)[0]
        return m.text_network(x, y, fit, setting)

    best = max(grid, key=lambda s: ev.auc(y[val], predict(train, s)[val]))
    p_test = predict(train | val, best)[test]
    wf = np.full(len(y), np.nan)
    for year in range(FIRST_YEAR, years.max() + 1):
        fit, now = years < year, years == year
        if now.any():
            wf[now] = predict(fit, best)[now]
    scored = years >= FIRST_YEAR
    yt, ot, yw, ow = y[test], outcome[test], y[scored], outcome[scored]
    lo, hi = ev.bootstrap_ci(ev.auc, len(yt), yt, p_test)
    wlo, whi = ev.bootstrap_ci(ev.auc, len(yw), yw, wf[scored])
    slo, shi = ev.bootstrap_ci(ev.long_short, len(yw), wf[scored], ow)
    return {"model": name, "setting": best,
            "val_auc": ev.auc(y[val], predict(train, best)[val]),
            "test_auc": ev.auc(yt, p_test), "test_auc_lo": lo, "test_auc_hi": hi,
            "test_accuracy": ev.accuracy(yt, p_test),
            "wf_n": int(scored.sum()), "wf_auc": ev.auc(yw, wf[scored]), "wf_auc_lo": wlo,
            "wf_auc_hi": whi, "wf_accuracy": ev.accuracy(yw, wf[scored]),
            "wf_spread": ev.long_short(wf[scored], ow), "wf_spread_lo": slo, "wf_spread_hi": shi}


def majority_row(name, y, splits, years):
    train, val, test = splits
    scored = years >= FIRST_YEAR
    return {"model": name, "setting": None, "val_auc": 0.5, "test_auc": 0.5,
            "test_accuracy": float((y[test] == int(y[train | val].mean() >= 0.5)).mean()),
            "wf_n": int(scored.sum()), "wf_auc": 0.5,
            "wf_accuracy": float(max(y[scored].mean(), 1 - y[scored].mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events.csv")
    ap.add_argument("--finbert", default="data/text_features.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--placebo-draws", type=int, default=20)
    args = ap.parse_args()

    df = load_events(args.events)
    fb = pd.read_csv(args.finbert).set_index("call_id").loc[df["primary_call_id"]]
    producers = df["producer"].to_numpy()
    years = df["call_date"].dt.year.to_numpy()
    splits = tuple((df["split"] == s).to_numpy() for s in ("train", "validation", "test"))
    everyone = np.ones(len(df), dtype=bool)
    emb = fb.filter(like="emb_").to_numpy()

    # Tone change versus the same company's previous call (past information only;
    # df is sorted by call date, so diff() looks back one call within a company).
    for col in ("tone_net", "tone_net_remarks", "tone_net_qa"):
        fb[col + "_chg"] = pd.Series(fb[col].to_numpy(), index=df.index).groupby(df["producer"]).diff().to_numpy()
    us = df["producer"].isin(US_LISTED).to_numpy()
    groups = {"all companies": everyone, "US-listed": us, "Korea-listed": ~us}

    # ---- 1. FinBERT checks --------------------------------------------------
    checks = []
    measures = [("tone level", "whole call", standardise(fb["tone_net"].to_numpy(), producers, everyone)),
                ("tone level", "prepared remarks", standardise(fb["tone_net_remarks"].fillna(fb["tone_net_remarks"].median()).to_numpy(), producers, everyone)),
                ("tone level", "Q&A", standardise(fb["tone_net_qa"].fillna(fb["tone_net_qa"].median()).to_numpy(), producers, everyone)),
                ("tone change vs previous call", "whole call", fb["tone_net_chg"].to_numpy()),
                ("tone change vs previous call", "Q&A", fb["tone_net_qa_chg"].to_numpy())]
    for kind, section, tone in measures:
        for target, out_col in [("day-one move vs SOX", "rel_fwd_1d"), ("days 2-20 return vs SOX", OUTCOME)]:
            for group, mask in groups.items():
                ok = mask & ~np.isnan(tone)
                rho, p = spearmanr(tone[ok], df[out_col].to_numpy()[ok])
                checks.append({"measure": f"FinBERT {kind}, {section}", "compared_with": target,
                               "companies": group, "spearman_rho": rho, "p_value": p, "n_calls": int(ok.sum())})
    checks = pd.DataFrame(checks)
    checks.to_csv(os.path.join(args.out, "extra_finbert_checks.csv"), index=False, float_format="%.6f")

    # ---- 1b. when does the market use the call? --------------------------------
    sub = df.loc[us].assign(tone_chg=fb["tone_net_chg"].to_numpy()[us]).dropna(subset=["tone_chg"])
    sub["third"] = pd.qcut(sub["tone_chg"], 3, labels=["tone fell most", "middle", "tone rose most"])
    thirds = sub.groupby("third", observed=True).agg(calls=("event_id", "size"),
                                                     day_one_vs_sox=("rel_fwd_1d", "mean"),
                                                     days_2_20_vs_sox=(OUTCOME, "mean")).reset_index()
    thirds.to_csv(os.path.join(args.out, "extra_day1_tone_change.csv"), index=False, float_format="%.6f")

    # gap between the top and bottom thirds, with a 90% bootstrap interval (calls resampled)
    rng = np.random.default_rng(0)
    top = sub.loc[sub["third"] == "tone rose most"]
    bot = sub.loc[sub["third"] == "tone fell most"]
    gaps = []
    for label, col in [("day-one move vs SOX", "rel_fwd_1d"), ("days 2-20 return vs SOX", OUTCOME)]:
        a, b = top[col].to_numpy(), bot[col].to_numpy()
        draws = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)]
        lo, hi = np.quantile(draws, [0.05, 0.95])
        gaps.append({"outcome": label, "top_minus_bottom": a.mean() - b.mean(), "gap_90_lo": lo, "gap_90_hi": hi})
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(os.path.join(args.out, "extra_day1_tone_change_gap.csv"), index=False, float_format="%.6f")

    split_info = fb["qa_split_method"].value_counts().rename_axis("method").reset_index(name="calls")
    split_info.to_csv(os.path.join(args.out, "extra_qa_split_methods.csv"), index=False)

    # ---- 2. section tone in the drift model ---------------------------------
    y = df[TARGET].to_numpy()
    outcome = df[OUTCOME].to_numpy()

    def tone_col(col, fit):
        v = fb[col].to_numpy().astype(float)
        v = np.where(np.isnan(v), np.nanmedian(v[fit]), v)
        return standardise(v, producers, fit)

    def numbers(fit):
        return prep(df, NUMERIC_FEATURES, fit)

    drift = [
        majority_row("Benchmark: always the majority class", y, splits, years),
        evaluate_family("Numbers only (reference)", numbers, y, outcome, splits, years),
        evaluate_family("Numbers + FinBERT tone of remarks and Q&A",
                        lambda f: np.column_stack([numbers(f), tone_col("tone_net_remarks", f),
                                                   tone_col("tone_net_qa", f)]),
                        y, outcome, splits, years),
        evaluate_family("Numbers + FinBERT tone of Q&A only",
                        lambda f: np.column_stack([numbers(f), tone_col("tone_net_qa", f)]),
                        y, outcome, splits, years),
    ]
    pd.DataFrame(drift).to_csv(os.path.join(args.out, "extra_drift_section_tone.csv"),
                               index=False, float_format="%.6f")

    # ---- 2b. early fusion (Lecture 15): numbers + FinBERT components -------
    def fused(matrix):
        return lambda f: np.column_stack([numbers(f), fb_pcs(matrix, producers, f)])

    fusion = [majority_row("Benchmark: always the majority class", y, splits, years)]
    for label, model in [("logistic regression", "logit"), ("gradient-boosted trees", "gbm")]:
        row = evaluate_family(f"Early fusion, {label}: numbers + 10 FinBERT components",
                              fused(emb), y, outcome, splits, years, model=model)
        # placebo: the same family with random vectors in place of FinBERT
        rng = np.random.default_rng(7)
        noise = [evaluate_family("placebo", fused(rng.normal(size=emb.shape)), y, outcome,
                                 splits, years, model=model) for _ in range(args.placebo_draws)]
        for key in ("test_auc", "wf_auc"):
            vals = np.array([n[key] for n in noise])
            row[f"placebo_{key}_p05"] = float(np.quantile(vals, 0.05))
            row[f"placebo_{key}_p95"] = float(np.quantile(vals, 0.95))
            row[f"placebo_{key}_share_at_or_above"] = float((vals >= row[key]).mean())
        row["placebo_draws"] = args.placebo_draws
        fusion.append(row)
        print(f"early fusion ({label}) done", flush=True)
    pd.DataFrame(fusion).to_csv(os.path.join(args.out, "extra_early_fusion.csv"),
                                index=False, float_format="%.6f")

    # ---- 3. exploratory: the day-one move -----------------------------------
    y1 = (df["rel_fwd_1d"] > 0).astype(int).to_numpy()
    o1 = df["rel_fwd_1d"].to_numpy()

    def pre_numbers(fit):
        return prep(df, PRE_CALL_FEATURES, fit)

    def section_tones(fit):
        """Tone level of the whole call, remarks and Q&A, plus the change in whole-call
        and Q&A tone versus the previous call (first call of a company: change = 0)."""
        chg = [np.nan_to_num(fb[c].to_numpy()) / max(np.nanstd(fb[c].to_numpy()[fit]), 1e-9)
               for c in ("tone_net_chg", "tone_net_qa_chg")]
        return np.column_stack([tone_col("tone_net", fit), tone_col("tone_net_remarks", fit),
                                tone_col("tone_net_qa", fit)] + chg)

    day1 = [
        majority_row("Benchmark: always the majority class", y1, splits, years),
        evaluate_family("Pre-call numbers", pre_numbers, y1, o1, splits, years),
        evaluate_family("FinBERT tone (level and change)", section_tones, y1, o1, splits, years),
        evaluate_family("Pre-call numbers + FinBERT tone",
                        lambda f: np.column_stack([pre_numbers(f), section_tones(f)]),
                        y1, o1, splits, years),
        evaluate_family("Text: neural network on FinBERT embeddings",
                        lambda f: fb_pcs(emb, producers, f), y1, o1, splits, years, model="mlp"),
    ]
    day1 = pd.DataFrame(day1)
    day1.insert(1, "share_up_all", float(y1.mean()))
    day1.to_csv(os.path.join(args.out, "extra_day1_move.csv"), index=False, float_format="%.6f")

    pd.set_option("display.width", 200)
    print(checks.to_string(index=False))
    print("\nUS-listed calls by tone change:"); print(thirds.to_string(index=False)); print(gaps.to_string(index=False))
    print(split_info.to_string(index=False))
    cols = ["model", "setting", "val_auc", "test_auc", "test_auc_lo", "test_auc_hi", "wf_auc",
            "wf_auc_lo", "wf_auc_hi", "wf_spread"]
    print("\nDrift (days 2-20):"); print(pd.DataFrame(drift)[cols].to_string(index=False))
    print("\nEarly fusion:"); print(pd.DataFrame(fusion).drop(columns=["wf_n"]).to_string(index=False))
    print("\nDay-one move (exploratory):"); print(day1[cols].to_string(index=False))


if __name__ == "__main__":
    main()
