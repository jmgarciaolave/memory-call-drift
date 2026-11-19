"""Write a made-up events.csv and transcripts so the pipeline can be run
without the licensed data. The numbers are random: results on this sample
mean nothing, it only proves the code runs end to end.

    python scripts/make_synthetic_sample.py
    python run_all.py --events data/sample/events.csv \
        --transcripts data/sample/transcripts_text --out results_sample
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
OUT = "data/sample"
PRODUCERS = {"Micron": "MU US Equity", "SK_Hynix": "000660 KS Equity",
             "Samsung": "005930 KS Equity", "Sandisk_Lineage": "WDC US Equity"}
WORDS = ("demand supply pricing bit growth inventory customers capacity wafer margin strong weak "
         "record decline improve pressure server mobile data center contract spot quarter guidance "
         "shipments cost node transition capex utilization shortage oversupply recovery headwinds "
         "growth momentum uncertainty healthy soft robust challenging").split()

# pad the vocabulary with nonsense tokens so that documents differ from each other
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
VOCAB = WORDS + ["".join(rng.choice(LETTERS, size=6)) for _ in range(600)]

rows = []
for producer, ticker in PRODUCERS.items():
    for k, date in enumerate(pd.date_range("2010-02-01", "2026-04-30", freq="QS-FEB")):
        date = date + pd.Timedelta(days=int(rng.integers(0, 25)))
        call_id = f"SYN{producer[:2].upper()}{k:04d}"
        r1, s1 = rng.normal(0, 0.05), rng.normal(0, 0.015)
        drift, sdrift = rng.normal(0, 0.11), rng.normal(0.01, 0.06)   # days 2-20, independent of day one
        r20, s20 = (1 + r1) * (1 + drift) - 1, (1 + s1) * (1 + sdrift) - 1
        rev = float(np.exp(rng.normal(8.5, 0.5)))
        gm = float(np.clip(rng.normal(0.35, 0.15), -0.2, 0.7))
        rows.append({
            "event_id": f"{producer}_{date.date()}", "call_date": date.date(), "company": producer,
            "producer": producer, "ticker": ticker, "primary_call_id": call_id, "n_transcripts": 1,
            "all_call_ids": call_id, "period_end": (date - pd.Timedelta(days=27)).date(),
            "rev_usd": rev, "gp_usd": rev * gm, "ebit_usd": rev * (gm - 0.15), "ni_usd": rev * (gm - 0.2),
            "inv_usd": rev * 0.5, "capex_usd": -rev * 0.25, "cogs_usd": rev * (1 - gm),
            "gross_margin": gm, "op_margin": gm - 0.15, "rev_growth_yoy": rng.normal(0.1, 0.4),
            "inventory_days": rng.normal(95, 25), "capex_to_sales": abs(rng.normal(0.25, 0.1)),
            "base_date": date.date(), "ret_pre_20d": rng.normal(0.02, 0.1),
            "ret_fwd_1d": r1, "ret_fwd_5d": rng.normal(0, 0.07), "ret_fwd_20d": r20,
            "sox_fwd_1d": s1, "sox_fwd_5d": rng.normal(0, 0.03), "sox_fwd_20d": s20,
            "rel_fwd_1d": r1 - s1, "rel_fwd_5d": rng.normal(0, 0.06), "rel_fwd_20d": r20 - s20,
            "ret_drift_2_20d": drift, "sox_drift_2_20d": sdrift, "rel_drift_2_20d": drift - sdrift,
            "target_drift_up": int(drift > sdrift), "rule_follow_day1": int(r1 > s1),
            "split": "train" if date.year <= 2019 else "validation" if date.year <= 2022 else "test",
            "flag": "", "use": 1})
        folder = os.path.join(OUT, "transcripts_text", producer)
        os.makedirs(folder, exist_ok=True)
        pool = rng.choice(VOCAB, size=150, replace=False)
        text = " ".join(rng.choice(pool, size=int(rng.integers(400, 800))))
        with open(os.path.join(folder, f"{date.date()}__{producer}__{call_id}.txt"), "w") as fh:
            fh.write(text)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "events.csv"), index=False, float_format="%.6f")
print(f"wrote {len(rows)} synthetic calls to {OUT}/")
