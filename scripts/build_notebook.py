"""Write notebooks/memory_call_drift.ipynb, a narrated walk-through of the whole project.

    python scripts/build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# Should we pay for an AI reader of earnings calls?

**MIT 15.071 The AI Edge, Project 1 (predictive AI).** Team: Arthur Luetkemeyer, Jonathan Severns,
Jose Manuel Garcia Olave.

**Client and decision.** A fund's semiconductor team is deciding whether to build, or buy from a
vendor, an AI signal that reads earnings calls of memory-chip makers (Micron, SK Hynix, Samsung,
Western Digital, Sandisk). The signal would predict, at the close of the first trading day after a
call, whether the stock beats the SOX semiconductor index over trading days 2 to 20.

**Answer in one paragraph.** It does not work reliably. Across 234 calls (2010 to April 2026), no
model built on the reported numbers, on the call text, or on FinBERT (a language model trained on
finance) beats the rule "always bet the stock lags SOX". FinBERT looked good on 2023-2026, but it
was below chance before 2023 and random inputs match its score 2 times in 20. Why is nothing
left? The call does carry information, but the market prices it on day one: for US-listed names,
calls whose tone fell most versus the previous call lagged the sector by 3.9% on day one, and
nothing is left after the first close (a follow-up correlation, not a tested model). The value for
the client is a decision: do not build or buy this signal, and use our test (walk-forward plus
placebo) on any signal a vendor offers. If the fund studies anything next, it should be day one.

**How to read this notebook.** Sections follow the four questions of the project brief: the problem
(1), what AI can do (2-3), does it work (4-8), does it matter (9-10).

**How to run.** Licensed Bloomberg data must be in `data/` (see `docs/data.md`). The main pipeline
runs here in about eight minutes. Two slow steps are read from `results/` and can be regenerated
with the commands shown where they are used: FinBERT scoring (about one hour on a laptop CPU) and
the placebo test (about fifteen minutes).

**Use of AI.** Claude (Anthropic) was used heavily for code, documents and slides; the team reviewed
the code and results and is responsible for both."""))

C.append(code("""import os, sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = os.path.abspath("..")
sys.path.insert(0, ROOT); os.chdir(ROOT)
from src.data import load_events
from src.config import NUMERIC_FEATURES, TARGET, OUTCOME
pd.set_option("display.width", 160)

# round half up (0.625 -> 0.63), as in the README and the slides; Python's default rounds half to even
def r2(x): return np.floor(np.asarray(x, dtype=float) * 100 + 0.5 + 1e-9) / 100
def pct(x): return f"{np.floor(float(x) * 1000 + 0.5 + 1e-9) / 10:.1f}%"
def auc_ci(r, lo="auc_lo", hi="auc_hi"): return "" if pd.isna(r[lo]) else f"{r2(r[lo]):.2f}-{r2(r[hi]):.2f}" """))

C.append(md("""## 1. The problem and the data

One row per earnings call. We predict at the close of the first trading day, when the transcript
and the first price reaction are public. The target is 1 if the stock beats SOX from the day-one
close to the day-20 close. Inputs known at that moment: ten numbers (the day-one move against SOX,
our stand-in for the earnings surprise since point-in-time consensus was not available, the SOX
move that day, the run-up into the call, margins, revenue growth, inventory days, capex and their
changes) and the transcript."""))
C.append(code("""df = load_events("data/events.csv")
split_order = ["train", "validation", "test"]
overview = df.groupby("split").agg(calls=("event_id", "size"), first=("call_date", "min"), last=("call_date", "max"),
                                   stock_beat_SOX=(TARGET, "mean")).loc[split_order]
overview["first"], overview["last"] = overview["first"].dt.date, overview["last"].dt.date
overview["stock_beat_SOX"] = overview["stock_beat_SOX"].map(pct)
print(f"{len(df)} calls; by company: {df.producer.value_counts().to_dict()}")
overview"""))
C.append(md("""The stock beat SOX after 42-44% of calls in every period. **The bar for any signal is the
majority rule, "always bet it lags", which is right 56-58% of the time.**"""))

C.append(md("""## 2. What AI can do: the models

| | Model | Inputs |
|---|---|---|
| 1 | Always predict the majority class (benchmark) | none |
| 2 | Follow the day-one move (benchmark) | day-one move |
| 3-4 | Logistic regression, gradient-boosted trees (Lectures 3-6) | the ten numbers |
| 5 | Small neural network on TF-IDF call vectors (Lectures 8-10) | text |
| 6 | Logistic regression | numbers + our word-count tone |
| 7 | Logistic regression (late fusion) | numbers + out-of-fold score of model 5 |
| 8 | Small neural network on 10 principal components of FinBERT vectors (Lectures 12, 15) | text |
| 9 | Logistic regression | numbers + FinBERT tone |

**FinBERT** (`ProsusAI/finbert`) reads each transcript in overlapping 512-token windows (median 25
per call) and returns, per window, the probability that the text is positive, negative or neutral,
and a 768-number vector. We average over the call. It is used as published, not fine-tuned: 132
training calls are too few. Text inputs are centred on each company's own average, so the models
see how a call differs from that company's usual call, not which company is speaking.

**Selection.** Train 2010-2019, choose settings on 2020-2022 (validation), refit on both, score
2023-2026 (test) once. The FinBERT features come from `scripts/score_finbert.py` (about one hour;
`data/text_features.csv`, not committed)."""))
C.append(code("""fb = pd.read_csv("data/text_features.csv")
print(f"FinBERT features: {len(fb)} calls, {fb.n_chunks.median():.0f} windows per call (median), "
      f"label order checked, probabilities sum to {(fb.tone_pos + fb.tone_neg + fb.tone_neu).round(6).unique()}")
print(f"Mean FinBERT net tone: prepared remarks {fb.tone_net_remarks.mean():+.2f}, Q&A {fb.tone_net_qa.mean():+.2f}")"""))

C.append(md("""## 3. Running the pipeline

`run_all.py` builds the features, fits every model, scores the test period once, runs the
walk-forward test and writes `results/`. It is called here unchanged, so the notebook and the
repository give the same numbers."""))
C.append(code("""import run_all
sys.argv = ["run_all.py", "--events", "data/events.csv", "--transcripts", "data/transcripts_text",
            "--finbert", "data/text_features.csv"]
run_all.main()"""))

C.append(md("""## 4. Does it work? The test period, 2023-2026

AUC asks whether a model ranks winners above losers (0.50 = coin flip). Accuracy is read against
the majority rule."""))
C.append(code("""res = pd.read_csv("results/results.csv")
t = pd.DataFrame({"Model": res.model,
                  "Validation AUC": r2(res.val_auc),
                  "Test AUC": r2(res.test_auc),
                  "Test AUC 90% interval": res.apply(lambda r: auc_ci(r, "test_auc_lo", "test_auc_hi"), axis=1),
                  "Test accuracy": res.test_accuracy.map(pct)}).set_index("Model")
t"""))
C.append(md("""Only the FinBERT network beats the benchmark on the 54 test calls (accuracy 63% against 56%).
The models on numbers looked mildly useful in validation and then ranked the test period below
chance. 54 calls is a thin test, so the next section widens it."""))

C.append(md("""## 5. Does it hold? Walk-forward, 2015-2026

Retrain every year on all earlier calls and predict that year, with the settings chosen above: 176
out-of-sample calls. Added after the test result was seen, and reported as a robustness check."""))
C.append(code("""wf = pd.read_csv("results/walk_forward.csv")
w = pd.DataFrame({"Model": wf.model, "AUC": r2(wf.auc),
                  "AUC 90% interval": wf.apply(auc_ci, axis=1),
                  "% called right": wf.accuracy.map(pct),
                  "Long/short per call (pts)": np.floor(wf.spread * 1000 + 0.5 + 1e-9) / 10}).set_index("Model")
w"""))
C.append(code("""fig, ax = plt.subplots(figsize=(9, 4.8))
order = wf.sort_values("accuracy")
colors = ["#C2541F" if m.startswith("Benchmark: always") else "#2A6FC9" if "FinBERT network" in m else "#9DB9E8"
          for m in order.model]
ax.barh(order.model.str.replace("Benchmark: ", ""), order.accuracy * 100, color=colors)
bench = wf.loc[wf.model.str.startswith("Benchmark: always"), "accuracy"].iloc[0] * 100
ax.axvline(bench, color="#C2541F", ls="--", lw=1.5)
ax.set_xlim(40, 62); ax.set_xlabel("% of calls called right, walk-forward 2015-2026 (176 calls)")
ax.set_title("Nothing beats 'always bet it lags' (dashed line)", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.show()"""))
C.append(md("""No model clears the benchmark. The models near 58% get there by mostly predicting that the
stock lags, which the benchmark already does."""))

C.append(md("""## 6. The one promising result, under stress

FinBERT's test edge is checked two ways. First, the walk-forward AUC of the FinBERT network by
period. Second, a placebo test (`scripts/placebo.py`, about fifteen minutes): the same network is
fed random vectors instead of FinBERT, 20 times, everything else fixed. The bootstrap intervals
above hold the fitted model fixed and are too narrow; the placebo shows how far AUC moves by chance."""))
C.append(code("""from src import models as m, evaluate as ev
from run_all import standardise
from sklearn.decomposition import PCA
y, years, prod = df[TARGET].to_numpy(), df.call_date.dt.year.to_numpy(), df.producer.to_numpy()
emb = fb.set_index("call_id").loc[df.primary_call_id].filter(like="emb_").to_numpy()
alpha = json.loads(res.set_index("model").loc["Text: neural network on FinBERT embeddings", "settings"])["alpha"]
pred = np.full(len(df), np.nan)
for yr in range(2015, years.max() + 1):
    fit, now = years < yr, years == yr
    pcs = PCA(n_components=10, random_state=0).fit(emb[fit]).transform(emb)
    pcs = np.column_stack([standardise(pcs[:, j], prod, fit) for j in range(10)])
    pred[now] = m.text_network(pcs, y, fit, alpha)[now]
periods = {"2015-2019": (years >= 2015) & (years <= 2019), "2020-2022": (years >= 2020) & (years <= 2022),
           "2023-2026": years >= 2023}
by_period = pd.DataFrame({"Calls": [int(k.sum()) for k in periods.values()],
                          "FinBERT network, walk-forward AUC": [float(r2(ev.auc(y[k], pred[k]))) for k in periods.values()]},
                         index=periods.keys())
display(by_period)
plc = pd.read_csv("results/placebo.csv")
pd.DataFrame({"Measure": plc.measure, "Real FinBERT": r2(plc.real_finbert),
              "Random inputs, middle 90%": plc.apply(lambda r: f"{r2(r.placebo_p05):.2f}-{r2(r.placebo_p95):.2f}", axis=1),
              "Random draws at or above real": (plc.share_placebo_at_or_above_real * plc.draws).astype(int).astype(str) + " of " + plc.draws.astype(int).astype(str)}
             ).set_index("Measure")"""))
C.append(md("""Below chance before 2023, above chance only in the AI-boom years, and matched by random inputs
2 times in 20 on the test period. **The edge is consistent with luck; we would not pay for it yet.**"""))

C.append(md("""## 7. Why signals break: the 2023 regime shift

Rank correlation of each input with the days 2-20 outcome, before and after 2023."""))
C.append(code("""sig = pd.read_csv("results/sign_by_period.csv")
s = sig.assign(**{"2010-2022": r2(sig.rho_2010_2022), "2023-2026": r2(sig.rho_2023_2026)})
s[["feature", "2010-2022", "2023-2026"]].set_index("feature").sort_values("2010-2022")"""))
C.append(md("""Before 2023 a rising gross margin went with weaker returns (-0.15, p = 0.05): good news was a
reason to sell, as in a normal memory cycle. From 2023 the sign flipped (+0.34, p = 0.02) as the
stocks traded on AI demand. A model trained on the old cycle learned to fade strength, and
strength kept going. With twelve inputs tested, some flips could be chance, but the direction fits
what happened. Any signal trained mostly on pre-2023 data carries this risk."""))

C.append(md("""## 8. Follow-up analyses (added after the main results)

Produced by `scripts/extra_analyses.py`. The models below are chosen on validation and tested on
the fixed test period and walk-forward, like the main models. The day-one tables that follow are
correlations, not tested models."""))
C.append(code("""ext = []
for f, label in [("results/extra_drift_section_tone.csv", "Section tone (drift)"),
                 ("results/extra_early_fusion.csv", "Early fusion (drift)"),
                 ("results/extra_day1_move.csv", "Day-one move (exploratory)")]:
    e = pd.read_csv(f)
    ext.append(pd.DataFrame({"Analysis": label, "Model": e.model, "Test AUC": r2(e.test_auc),
                             "Walk-forward AUC": r2(e.wf_auc)}))
pd.concat(ext).set_index(["Analysis", "Model"])"""))
C.append(md("""- **Prepared remarks vs Q&A.** Management is far more positive in prepared remarks (FinBERT net
  tone +0.40) than in the Q&A (+0.20), but neither section's tone predicts the drift.
- **Early fusion** (numbers + FinBERT vectors in one model, Lecture 15) does not help either, and
  it sits inside the range random vectors produce.
- **The day-one move.** Models pooled over all 234 calls predict the direction of the first day's
  move barely better than chance (walk-forward AUC 0.48-0.52). The next cell shows why: the link
  is concentrated in the US-listed names."""))

C.append(md("""### When does the market use the call?

The intuition that an earnings call moves the stock is right, but on day one. The information is
in the *change* of FinBERT tone versus the same company's previous call (known at call time),
not in its level. US-listed companies (Micron, Western Digital / Sandisk) report after the close,
so day one is the first session to react; the Korean names report during their trading hours,
after publishing preliminary results."""))
C.append(code("""chk = pd.read_csv("results/extra_finbert_checks.csv")
chk = chk[chk.measure == "FinBERT tone change vs previous call, whole call"]
corr = chk.assign(value=chk.apply(lambda r: f"{r2(r.spearman_rho):+.2f} ({'p < 0.01' if r.p_value < 0.01 else f'p = {r.p_value:.2f}'}, n = {r.n_calls})", axis=1)) \\
          .pivot(index="compared_with", columns="companies", values="value")[["all companies", "US-listed", "Korea-listed"]]
display(corr.rename_axis("Rank correlation of tone change with"))
th = pd.read_csv("results/extra_day1_tone_change.csv")
gap = pd.read_csv("results/extra_day1_tone_change_gap.csv")
thirds = pd.DataFrame({"Calls": th.calls.values, "Day-one move vs SOX": th.day_one_vs_sox.map(pct).values,
                       "Days 2-20 vs SOX": th.days_2_20_vs_sox.map(pct).values}, index=th.third.values)
thirds.loc["Top minus bottom (90% interval)"] = ["",
    *[f"{g.top_minus_bottom * 100:+.1f} pts ({g.gap_90_lo * 100:+.1f} to {g.gap_90_hi * 100:+.1f})" for g in gap.itertuples()]]
thirds.rename_axis("US-listed calls, by tone change")"""))
C.append(md("""A call whose tone falls versus the last one is followed by a day-one drop of about four points
against the sector; after the first close the same measure says nothing about the next four
weeks. **The market prices the call within a day**, which is why a signal that starts at the
day-one close finds nothing. Cautions: found after the main results and among many comparisons;
closing prices only (a signal scored overnight could trade at the next open, which we cannot test
without opening prices)."""))

C.append(md("""## 9. Does it matter? What the answer is worth to the client

A null is a decision. Three sources of value:
1. **Avoided cost:** no need to build or license an earnings-call signal for this use.
2. **Avoided loss:** the numbers strategy that looked best on 2010-2022 data, long the top third and
   short the bottom third, lost 4.8 points per call in the walk-forward (90% interval -8.5 to -1.1).
   On an illustrative $10 million long/short pair that is about $480,000 per call traded
   (assumption, not in the data; transaction costs ignored).
3. **A reusable test:** walk-forward plus placebo is a due-diligence check for any vendor signal.

It also says where to look next, if anywhere: the call's information is priced on day one, so a
follow-up study would test the next open for the US-listed names."""))
C.append(code("""best = w.loc["Numbers: logistic regression"]
print(f"Numbers strategy: {best['Long/short per call (pts)']:+.1f} points per call over 176 calls "
      f"-> on a $10m pair about ${abs(best['Long/short per call (pts)']) / 100 * 10e6:,.0f} per call")
day1 = w.loc["Benchmark: follow the day-one move", "% called right"]
print(f"Following the first day's move: {day1} right over 176 calls")"""))

C.append(md("""## 10. Recommendations and limitations

**Recommendations to the client**
1. Do not build or buy this signal (days 2 to 20): nothing beat the simple rule over 176 calls,
   and the best-looking numbers strategy lost 4.8 points per call.
2. Make walk-forward plus placebo the bar for any vendor signal, with results after 2023.
3. Keep the FinBERT reader running on new calls, unchanged: a near-free live test of whether its
   2023-2026 edge is a new regime or chance.
4. Next study, if any: the day-one window. Test FinBERT tone change at the next open for the
   US-listed names (found after the main results, not yet tested). Then, if useful: point-in-time
   consensus, all semiconductor names, and scoring guidance passages rather than whole-call averages.

**Limitations.** 234 calls from five companies, so only a strong signal could be detected. No
consensus estimates. FinBERT is used off the shelf and averaged over the call. Price returns in US
dollars against a US index mix currency and time-zone effects into the Korean names. Samsung is a
conglomerate (results excluding it are in `results/robustness_ex_samsung.csv` and tell the same
story). The walk-forward, the FinBERT stress tests and the follow-ups were added after the first
results were seen. Full discussion: `docs/methodology.md`."""))

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
os_path = "notebooks/memory_call_drift.ipynb"
import os
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, os_path)
print("wrote", os_path)
