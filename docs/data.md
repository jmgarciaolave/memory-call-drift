# Data

Nothing in `data/` is committed except the synthetic sample. To run on the real data you need
two things in place:

```
data/events.csv              one row per earnings call
data/transcripts_text/       <Producer>/<YYYY-MM-DD>__<Producer>__<CALLID>.txt
```

## Sources

- **Prices.** Daily closes in US dollars for Micron, SK Hynix, Samsung, Western Digital, Sandisk
  and the SOX index, December 2009 to November 2026 (Bloomberg, pulled 16 November 2026).
  Returns are price returns; dividends are ignored, which is immaterial over 20 days.
- **Fundamentals.** Quarterly revenue, gross profit, operating profit, net income, inventory,
  capex and cost of goods sold in US dollars (Bloomberg).
- **Transcripts.** Bloomberg earnings-call transcripts, converted from PDF to text and cleaned in
  an earlier project. Where a call has more than one transcript version we use the longest.

## How events.csv is built

The table is assembled in the workbook `Memory Calls Data v05.xlsx` (not committed) and exported.

1. Start from the list of calls we hold transcripts for. Calls dated within five days of each
   other are treated as one event.
2. Match each call to the latest fiscal quarter that ended before it.
3. Set the base date. US companies report after the close, so the base is the close on the call
   date and day one is the next session. Samsung and SK Hynix report during Korean trading
   hours, so the base is the previous close and day one is the call date.
4. Compute returns from the base date for the stock and for SOX, and take the difference.

## Columns used by the models

| Column | Meaning |
|---|---|
| `event_id`, `call_date`, `producer`, `ticker` | identifiers |
| `primary_call_id` | transcript ID to read for this call |
| `rel_fwd_1d` | stock return minus SOX return on day one |
| `sox_fwd_1d` | SOX return on day one |
| `ret_pre_20d` | stock return over the 20 sessions before the base date |
| `gross_margin`, `op_margin` | gross and operating profit over revenue, latest reported quarter |
| `rev_growth_yoy` | revenue vs the same quarter a year earlier |
| `inventory_days` | inventory / cost of goods sold x 91 |
| `capex_to_sales` | capex / revenue |
| `rel_drift_2_20d` | stock return minus SOX return from the day-one close to day 20 |
| `target_drift_up` | 1 if `rel_drift_2_20d` > 0 |
| `rule_follow_day1` | 1 if `rel_fwd_1d` > 0 |
| `split` | train (to 2019), validation (2020-2022), test (2023 on) |
| `use` | 1 if the row is modelled |

Two more inputs, `gross_margin_chg` and `inventory_days_chg`, are computed in `src/data.py` as
the change from the same company's previous call.

The file also carries `ret_fwd_5d`, `ret_fwd_20d` and the other forward returns. They are listed
in `LOOKAHEAD` in `src/config.py` and are never model inputs.

## Known problems

- **SanDisk before 2016 is missing.** We hold 22 transcripts but did not pull prices for the
  delisted ticker.
- **Kioxia is excluded.** It listed in December 2024 and we have no transcripts.
- **No consensus estimates.** We could not get point-in-time consensus revenue out of Bloomberg
  in a usable form, so the day-one move stands in for the surprise.
- **SK Hynix gross profit** exceeds revenue in Bloomberg's data for the last two quarters. The
  margin is set to missing on those rows.
- **Two Samsung calls in September 2013** fall in a quarter that already has a call. They have
  `use = 0`.
- **Missing fundamentals** (11 calls lack year-on-year growth, 5 lack capex) are filled with the
  training-period median.
- **Samsung** is a conglomerate. Its share price reflects phones, displays and foundry as well
  as memory.
