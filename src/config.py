"""Settings shared by every step. Change things here, not in the scripts."""

SEED = 42

TARGET = "target_drift_up"        # 1 if the stock beat SOX over trading days 2-20
OUTCOME = "rel_drift_2_20d"       # the same thing as a return, used in the backtest

# Everything in this list is known at the close of day one after the call.
NUMERIC_FEATURES = [
    "rel_fwd_1d",          # day-one move vs SOX: our stand-in for the earnings surprise
    "sox_fwd_1d",          # what the sector did that day
    "ret_pre_20d",         # run-up into the call
    "gross_margin",
    "op_margin",
    "rev_growth_yoy",
    "inventory_days",
    "capex_to_sales",
    "gross_margin_chg",    # vs the company's previous call
    "inventory_days_chg",
]

# Columns that only exist after the prediction date. Never model inputs.
LOOKAHEAD = [
    "ret_fwd_5d", "ret_fwd_20d", "sox_fwd_5d", "sox_fwd_20d", "rel_fwd_5d",
    "rel_fwd_20d", "ret_drift_2_20d", "sox_drift_2_20d", "rel_drift_2_20d",
    "target_drift_up",
]

# Small grids on purpose: 132 training calls do not support a big search.
LOGIT_C = [0.01, 0.1, 1.0]
GBM_GRID = [
    {"n_estimators": 50, "max_depth": 1},
    {"n_estimators": 100, "max_depth": 1},
    {"n_estimators": 50, "max_depth": 2},
]
LSA_COMPONENTS = [10, 25]
MLP_ALPHA = [1.0, 10.0]
MLP_HIDDEN = (16,)
MLP_SEEDS = 10             # the text network is averaged over this many seeds

BOOTSTRAP_DRAWS = 2000
