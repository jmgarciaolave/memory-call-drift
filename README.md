# Should we pay for an AI reader of earnings calls?

MIT 15.071 The AI Edge, Fall 2026, Project 1 (predictive AI).
Team: Arthur Luetkemeyer, Jonathan Severns, Jose Manuel Garcia Olave. For: a data scientist
checking or extending the work.

**Client and decision.** A fund's semiconductor team is deciding whether to build, or license, an
AI signal that reads the earnings calls of memory-chip makers and predicts, at the close of the
first trading day, whether the stock beats the SOX index over trading days 2 to 20.

**Result: no reliable signal.** Across 234 calls (Micron, SK Hynix, Samsung, Western Digital /
Sandisk; 2010 to April 2026), no model built on the numbers, the call text or FinBERT beats
"always bet the stock lags SOX" (58.5% right over 176 walk-forward calls). FinBERT looked good on
2023-2026 (test AUC 0.63) but was below chance before 2023, and random inputs match its score 2
times in 20. Why is nothing left? The call does carry information, but the market prices it on
day one: for the US-listed names, calls whose FinBERT tone fell most versus the previous call
lagged SOX by 3.9% on day one, against +0.9% for those whose tone rose most, and nothing is left
after the first close (a follow-up correlation, not a tested model). For the client this is a
decision: do not build or buy this signal, and use walk-forward plus placebo as the bar for any
vendor signal. If the fund studies anything next, it should be the day-one window.

**Start here:** `notebooks/memory_call_drift.ipynb` walks through the whole project, runs the
pipeline and shows every result. `docs/methodology.md` has the full discussion and
`docs/slides.pdf` is the client pitch.

**Task and design.** One row per call. Inputs known at the day-one close: ten numbers (day-one
move vs SOX as the surprise proxy, run-up, margins, growth, inventory days, capex) and the
transcript. Split by date: train to 2019 (132), validation 2020-2022 (48), test 2023 on (54),
plus a walk-forward test from 2015 (176 calls, retrain yearly, settings fixed).

**Models.** Benchmarks (majority class, follow day one); logistic regression and boosted trees on
the numbers; a small network on TF-IDF call vectors; FinBERT (`ProsusAI/finbert`, 512-token
windows, tone and 768-d vectors averaged per call, not fine-tuned) feeding a small network;
numbers + tone or text score (late fusion); numbers + FinBERT vectors (early fusion, follow-up).

| Model | Test AUC (54) | Walk-forward AUC (176) | Walk-forward % right |
|---|---|---|---|
| Always bet it lags | 0.50 | 0.50 | 58.5% |
| Follow the day-one move | 0.48 | 0.46 | 46.6% |
| Numbers: logistic regression | 0.32 | 0.46 | 58.0% |
| Text: TF-IDF network | 0.54 | 0.50 | 58.0% |
| Text: FinBERT network | 0.63 | 0.48 | 51.7% |
| Numbers + FinBERT tone | 0.34 | 0.47 | 57.4% |

**Reproduce** (Python 3.11+; `torch` and `transformers` only for FinBERT):

```bash
pip install -r requirements.txt
python scripts/make_synthetic_sample.py      # no licensed data needed: smoke test only
python run_all.py --events data/sample/events.csv --transcripts data/sample/transcripts_text --out results_sample
# with the licensed data in data/ (see docs/data.md):
python scripts/score_finbert.py              # ~1 h on CPU, resumable
python run_all.py --finbert data/text_features.csv   # ~8 min; writes results/
python scripts/extra_analyses.py             # follow-ups: tone change and day one, section tone, early fusion
python scripts/placebo.py                    # placebo test for the FinBERT models (~15 min)
python scripts/build_notebook.py             # rebuilds the walk-through notebook
```

**Layout.** `run_all.py` (entry point) · `src/` (features, models, evaluation, walk-forward,
figures) · `scripts/` (FinBERT, follow-ups, placebo, notebook builder, synthetic sample) ·
`notebooks/` · `docs/` · `results/` (aggregate tables and figures only).

**Data.** Bloomberg prices, fundamentals and transcripts are licensed and not in this repository,
nor are the FinBERT features or per-call predictions derived from them.

**Use of AI.** Claude (Anthropic) was used heavily for code, documents and slides. The team
reviewed the code and results and is responsible for both.
