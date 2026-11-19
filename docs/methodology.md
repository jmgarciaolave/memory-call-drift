# Methodology and findings

## 1. Problem

**Client and decision.** A fund's semiconductor team is deciding whether to build, or license
from a vendor, an AI signal that reads the earnings calls of memory-chip makers. Such a signal
would be used after each call: the first day's move is over before most funds can act in size, so
the useful question is what happens next. Over the following four weeks, will the stock beat or
lag the sector? Today that call rests on sell-side notes and the portfolio manager's own read of
the transcript.

We asked whether a model reading the same call, together with the reported numbers, does better
than simple rules. If it does, the signal is worth building or buying; if it does not, the fund
saves that cost and avoids trading on it.

## 2. Prediction task

We predict at the close of the first trading day after the call. The target is whether the stock
beats SOX from that close to the close of day 20. Measuring against SOX strips out sector moves;
starting at the day-one close means the transcript and the initial reaction are both public
before the clock starts.

We do not have consensus estimates, so we cannot compute a beat or miss. The day-one move
relative to SOX takes its place: it is the market's own summary of the surprise across revenue,
margins and guidance.

## 3. Data

234 usable calls from five companies, 2010 to April 2026. The stock beat SOX over days 2 to 20
in 42% of training calls, 44% of validation calls and 44% of test calls. So "always predict
underperformance" is right 56% to 58% of the time, and that is the bar for accuracy.
`docs/data.md` has the details.

## 4. Features

**Numbers (10).** Day-one move vs SOX, SOX move on day one, 20-day run-up, gross margin,
operating margin, year-on-year revenue growth, inventory days, capex to sales, and the change in
gross margin and inventory days since the previous call. Missing values are filled with the
median, then each input is standardised and clipped at three standard deviations. All of this
uses the fitting rows only.

**Text.** Three representations. The first two use no pre-trained model:

- *Tone score.* Positive words minus negative words over their sum, from a short list we wrote
  for memory calls (`src/text.py`). We use the level relative to the company's average and the
  change since the previous call.
- *Call vectors.* TF-IDF over the transcript, reduced to 10 or 25 dimensions by truncated SVD,
  then centred on the company's average vector. Centring matters: without it the vectors mostly
  encode which company is speaking.

- *FinBERT.* `ProsusAI/finbert` is a BERT model further trained on financial text to label
  sentences as positive, negative or neutral. It reads at most 512 tokens, so
  `scripts/score_finbert.py` cuts each transcript into overlapping 512-token windows (64 tokens of
  overlap; 12 to 33 windows per call, median 25). For each window we keep the three probabilities
  and the 768-dimensional [CLS] vector of the last layer, and average them over the call. Net
  tone is positive minus negative probability. Each window is also tagged as prepared remarks or
  Q&A from the transcript's "Questions And Answers" / "Q&A" heading (230 calls; the other 4 use
  the operator's first call for questions). The model is used as published: it is not fine-tuned
  on our labels, which would be unwise with 132 training calls.

Checks on the FinBERT output: 234 rows, probabilities sum to one, the label order is read from
the model's own configuration (0 positive, 1 negative, 2 neutral), and net tone is positively,
though weakly, related to the day-one move (rank correlation +0.05; our word list gave +0.12).

## 5. Models and selection

Nine models in the main run (listed in the notebook, section 2), plus two early-fusion
models added as a follow-up. Grids are deliberately small (three values of the
regularisation strength, three tree configurations, four text-network configurations, two for
the FinBERT network) because 132 training calls cannot support a wide search.

The two FinBERT models mirror models 5 and 6. Model 8 is the same small network as model 5, fed
with the first ten principal components of the FinBERT call vectors (principal components fitted
on the fitting rows, then centred on each company's average). Model 9 adds FinBERT net tone,
centred by company, to the numbers. For each family we pick the setting with the
best validation AUC, refit on train plus validation, and score the test period once.

The combined model (7) is a late fusion. The text network produces a score for each call, and
that score enters a logistic regression next to the numbers. For the rows the regression is
fitted on, the text score is out-of-fold, so the regression never sees a score that has already
memorised the label.

## 6. Evaluation

- **AUC**: does the model rank winners above losers? 0.50 is no skill.
- **Accuracy** at a 0.5 threshold, read against the majority-class benchmark.
- **Long/short spread**: go long the third of calls the model likes most and short the third it
  likes least; report the difference in average days 2-20 return vs SOX.
- **Intervals**: 90% percentile bootstrap over calls, 2,000 draws.

## 7. Results

### Fixed split

| Model | Validation AUC | Test AUC (90% interval) | Test accuracy |
|---|---|---|---|
| Always the majority class | 0.50 | 0.50 | 55.6% |
| Follow the day-one move | 0.52 | 0.48 (0.36-0.59) | 48.1% |
| Numbers: logistic regression | 0.56 | 0.32 (0.20-0.44) | 48.1% |
| Numbers: gradient-boosted trees | 0.54 | 0.39 (0.26-0.52) | 44.4% |
| Text: neural network | 0.39 | 0.54 (0.41-0.67) | 55.6% |
| Numbers + word-count tone | 0.54 | 0.35 (0.22-0.47) | 48.1% |
| Numbers + text network | 0.54 | 0.32 (0.20-0.45) | 48.1% |
| Text: neural network on FinBERT | 0.59 | 0.63 (0.49-0.75) | 63.0% |
| Numbers + FinBERT tone | 0.56 | 0.34 (0.22-0.46) | 51.9% |

Before FinBERT, nothing beat the majority benchmark on accuracy. The numbers models looked mildly useful in
validation (AUC 0.54 to 0.56) and then ranked calls *backwards* in the test period: the
logistic regression's interval lies entirely below 0.50. The text network did the opposite, poor
in validation and slightly above 0.50 in test, which is what noise looks like. Adding text to
the numbers changes nothing.

The FinBERT network is the one model that is above 0.50 in both validation (0.59) and test
(0.63), with 63% test accuracy against 56% for the benchmark and a long/short spread of +7.1
points per call (90% interval -2.6 to +16.4). Its test interval still touches 0.50. Excluding
Samsung, its test AUC is 0.60. Adding FinBERT tone to the numbers changes nothing.

### Walk-forward

A single 54-call test period, all of it inside the AI memory boom, is a narrow basis for a
verdict. After seeing the result above we added a second test: for each year from 2015, fit on
every earlier call and predict that year's calls, keeping the settings chosen above. This was
not planned in advance and the settings were picked on 2020-2022 data, so read it as a
robustness check.

| Model | AUC (90% interval) | Accuracy | Long/short spread per call (90% interval) |
|---|---|---|---|
| Always the majority class | 0.50 | 58.5% | +0.5 pts (short every call) |
| Follow the day-one move | 0.46 | 46.6% | -2.0 pts |
| Numbers: logistic regression | 0.46 (0.38-0.54) | 58.0% | -4.8 pts (-8.5 to -1.1) |
| Numbers: gradient-boosted trees | 0.46 (0.38-0.53) | 51.1% | -1.8 pts (-4.3 to +0.6) |
| Text: neural network | 0.50 (0.42-0.57) | 58.0% | +0.5 pts (-2.0 to +4.0) |
| Numbers + word-count tone | 0.47 (0.39-0.54) | 55.7% | -4.4 pts (-8.3 to -0.8) |
| Numbers + text network | 0.46 (0.38-0.54) | 57.4% | -5.1 pts (-8.7 to -1.2) |
| Text: neural network on FinBERT | 0.48 (0.41-0.55) | 51.7% | +0.7 pts (-2.2 to +3.0) |
| Numbers + FinBERT tone | 0.47 (0.39-0.54) | 57.4% | -4.4 pts (-7.8 to -0.3) |

Same picture over 176 calls. No model has an AUC above 0.50, none beats the majority benchmark,
and the models built on numbers would have lost money. The FinBERT models were added to the
walk-forward with the settings chosen in the main run, as for the others.

### Why the numbers models fail

The relationships are not stable. Across 2010-2022, a rising gross margin went with *weaker*
returns over the next four weeks (rank correlation -0.15, p = 0.05), the pattern you would
expect in a cyclical industry where good news is a reason to sell. In 2023-2026 the sign
flipped (+0.34, p = 0.02). Several other inputs flip sign too (`results/sign_by_period.csv`,
figure 4). With twelve inputs tested, one or two differences of this size could be chance, but
the direction fits what happened: from 2023 the stocks traded on AI demand rather than on the
usual cycle. A model trained on the cycle learned to fade strength, and strength kept going.

### Is the text measuring anything?

Weakly. Calls with a more positive tone than the company's norm had a slightly better day-one
move (rank correlation +0.12, p = 0.06), and tone has no reliable relationship with the
following four weeks (-0.09, p = 0.15). That is consistent with the market reading the call on
day one. Our tone list is crude, so this is suggestive at best. The level of FinBERT tone is
barely related to either (+0.05 and -0.03). What it misses is the change: see "When does the
market use the call?" below.

### Does the FinBERT result survive?

No. Split by period, the FinBERT network's walk-forward AUC is:

| Period | Calls | AUC |
|---|---|---|
| 2015-2019 | 74 | 0.42 |
| 2020-2022 | 48 | 0.43 |
| 2023-2026 | 54 | 0.60 |

The model reads the call usefully only in the AI-boom years, and is below chance before. Year by
year, AUC ranges from 0.22 to 0.80 on 7 to 17 calls, and adding one more year of training data
can move it by 0.15. Two readings fit: the market started to price memory calls differently from
2023 (the same break seen in the numbers, above), or the test-period edge is chance on 54 calls.
We cannot tell them apart with this sample, so we do not change the recommendation.

### How much of this is chance? A placebo test

The 90% intervals above resample the predictions of a single fitted model. They do not include
the variation that comes from fitting the model on 130 to 180 calls, which is large. While
testing the code we fed the FinBERT network random numbers instead of FinBERT vectors: it scored
a walk-forward AUC of 0.41 with an interval of 0.34 to 0.49, apparently "significantly" worse
than chance. So these intervals are too narrow, and AUCs of 0.32 to 0.41 should not be read as
evidence that a model ranks calls backwards.

`scripts/placebo.py` makes this explicit: it reruns the two FinBERT models many times with the
FinBERT input replaced by noise, everything else fixed (20 draws; `results/placebo.csv`).

| Measure | Real FinBERT | Placebo: middle 90% | Placebo draws at or above real |
|---|---|---|---|
| Network on FinBERT, test AUC | 0.63 | 0.37 to 0.64 | 2 of 20 |
| Network on FinBERT, walk-forward AUC | 0.48 | 0.40 to 0.53 | 10 of 20 |
| Numbers + FinBERT tone, test AUC | 0.34 | 0.32 to 0.35 | 9 of 20 |
| Numbers + FinBERT tone, walk-forward AUC | 0.47 | 0.46 to 0.48 | 11 of 20 |

Random vectors fed to the same network reach the real test AUC of 0.63 2 times in 20. The real
FinBERT network sits at the edge of that range, not beyond it, and over the walk-forward it is
exactly where noise lands. FinBERT tone adds nothing to the numbers model that noise does not.
With 20 draws these shares are rough, but the reading is clear: the test-period edge is
consistent with chance. The same placebo also shows that the numbers model's low test AUC (0.32
to 0.35 with any extra input) is a property of the numbers model on this period, not of the
text.

### Samsung

Excluding Samsung from the test period does not change the conclusion
(`results/robustness_ex_samsung.csv`).

### Early fusion (follow-up analysis)

Lecture 15's recipe for combining modalities is early fusion: concatenate the embeddings with the
tabular inputs and fit one model. The main run only fuses FinBERT *tone* with the numbers, so we
added numbers + the 10 FinBERT components, with logistic regression and with boosted trees,
settings chosen on validation as before (`results/extra_early_fusion.csv`).

| Model | Validation AUC | Test AUC (90% interval) | Walk-forward AUC (90% interval) | Long/short per call |
|---|---|---|---|---|
| Early fusion, logistic regression | 0.56 | 0.34 (0.22-0.47) | 0.46 (0.39-0.54) | -0.9 pts |
| Early fusion, boosted trees | 0.56 | 0.41 (0.29-0.54) | 0.42 (0.35-0.50) | -4.9 pts |

Each was also run 20 times with random vectors in place of FinBERT. The real results sit inside
the placebo range (logistic walk-forward: placebo 0.45 to 0.50; trees: 0.42 to 0.51, and 18 of 20
random draws scored at least as well). Adding the FinBERT vectors to the numbers does not help,
and the fused models inherit the numbers' failure after 2023.

### Prepared remarks vs Q&A, and the day-one move (follow-up analyses)

Both were designed after the main results were known, and are reported as follow-ups.

- **Sections.** Management is markedly more positive in prepared remarks (mean FinBERT net tone
  +0.40) than in Q&A (+0.20). Neither section's tone is related to the days 2-20 drift (rank
  correlations -0.08 and +0.02), and adding them to the numbers leaves walk-forward AUC at 0.46
  to 0.47 (`results/extra_drift_section_tone.csv`).
- **The day-one move (exploratory).** Models built on all 234 calls (pre-call numbers,
  FinBERT tone level and change, FinBERT vectors) predict the direction of the day-one move
  barely better than chance (walk-forward AUC 0.48 to 0.52, `results/extra_day1_move.csv`). The
  association below is concentrated in the US-listed names, and pooling with the Korean names
  dilutes it.

### When does the market use the call? (follow-up analysis)

The intuition that an earnings call moves the stock is right, but on day one. What carries the
information is not the level of FinBERT tone but its change versus the same company's previous
call (`results/extra_finbert_checks.csv`, `results/extra_day1_tone_change*.csv`):

| Rank correlation of FinBERT tone change with | All (230) | US-listed (101) | Korea-listed (129) |
|---|---|---|---|
| Day-one move vs SOX | +0.17 (p < 0.01) | +0.30 (p < 0.01) | +0.08 (p = 0.39) |
| Days 2-20 return vs SOX | +0.03 (p = 0.68) | -0.06 (p = 0.57) | +0.10 (p = 0.28) |

US-listed calls in thirds by tone change:

| Third | Calls | Day-one move vs SOX | Days 2-20 vs SOX |
|---|---|---|---|
| Tone fell most | 34 | -3.9% | +0.1% |
| Middle | 33 | +1.4% | +4.3% |
| Tone rose most | 34 | +0.9% | -1.6% |
| Top minus bottom (90% interval) | | +4.8 pts (+2.4 to +7.3) | -1.6 pts (-4.5 to +1.2) |

A call whose tone falls versus the last one is followed by a day-one drop of about four points
against the sector; after the first close, the same measure says nothing about the next four
weeks. The market prices the call within a day, which is why a signal that starts at the
day-one close finds nothing. Two cautions: this was found after the main results and among many
comparisons, and it uses closing prices. US companies report after the close, so a signal scored
overnight could in principle trade at the next open; we do not have open prices to test that.
The Korean names show no such pattern; a likely reason is that they publish preliminary results
before the call, so the surprise is already priced when the call starts.

## 8. What it means for the client

A well-tested null is a decision. For a fund weighing an earnings-call signal it is worth money
in three ways:

1. **Avoided cost.** No model beat the simple rule over 176 calls, so there is no case for
   building or licensing such a signal for this use.
2. **Avoided loss.** The numbers strategy that looked best on 2010-2022 data, long the top third
   and short the bottom third, lost 4.8 points per call in the walk-forward test (90% interval
   -8.5 to -1.1). On an illustrative $10 million long/short pair that is about $480,000 per call
   traded (the pair size is an assumption; transaction costs are ignored).
3. **A reusable test.** Walk-forward plus a placebo comparison is a due-diligence check the fund
   can apply to any signal a vendor offers, before committing money. Ask for results after 2023,
   given the regime shift.

It also explains why nothing is left. The call does carry information: for the US-listed names,
a fall in FinBERT tone versus the previous call went with a day-one drop of 3.9% against the
sector. The market prices it within a day, so by the day-one close, when the signal would start,
the news is already in the price. If the fund studies anything next, it should be the day-one
window (a follow-up correlation, not yet a tested model).

What follows for the portfolio manager:

- **Do not trade post-earnings drift in memory names on these signals.** In the walk-forward
  test the numbers model cost 4.8 points per long/short pair per call.
- **Do not chase or fade the first day mechanically.** Following the day-one move was right 47%
  of the time over 176 calls.
- **The base rate is the one stable fact.** These stocks lagged SOX after 57% of calls. The
  average gap is small and it reversed in 2023-2026 (+2.8 points on average), so it is not a
  trade either.
- **Watch the FinBERT reader, do not trade it yet.** It is the only model that read 2023-2026
  calls better than chance, but it failed on 2015-2022. Running it on each new call, without
  changes, is a cheap way to find out whether it has found something real.
- **A model like this should be used as a check, not a signal.** Its main value here was showing
  that a rule which looked reasonable on 2010-2022 data would have failed in the next regime.

## 9. Limitations

- 234 calls, five companies, one industry. Intervals are wide: we can rule out a strong signal,
  not a weak one.
- No consensus estimates, so no true surprise measure.
- FinBERT is used off the shelf. It was trained on financial news and analyst reports, not on
  earnings calls, and we do not fine-tune it. Averaging 25 windows into one vector also blurs
  any single passage that matters (a guidance cut, a pricing comment).
- The bootstrap intervals understate uncertainty, because they hold the fitted model fixed (see
  the placebo test in section 7).
- The walk-forward test, the FinBERT models, the placebo test and the follow-up analyses were all
  added after the first fixed-split result was known.
- Price returns in US dollars against a US index; for the Korean names this mixes in currency
  moves and a time-zone gap.
- Daily closes only. Anything that happens inside day one is invisible to us.

## 10. What we would do next

1. Test the day-one window for the US-listed names: score each call overnight and trade at the
   next open, using intraday or opening prices. This is where the call's information is.
2. Keep scoring new calls with the FinBERT network, unchanged, as a live out-of-sample test of
   whether its 2023-2026 edge is a new regime or chance. A few years of calls would settle it.
3. Get point-in-time consensus so the surprise can be measured directly.
4. Widen the sample to all semiconductor companies (roughly ten times the calls) and let the
   model learn what is specific to memory from an indicator.
5. Target specific passages instead of whole-call averages: guidance, pricing and inventory
   comments, and the analysts' questions, scored sentence by sentence.
