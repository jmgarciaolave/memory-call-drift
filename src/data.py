"""Load events.csv and build the numeric features."""
import numpy as np
import pandas as pd

from .config import NUMERIC_FEATURES, TARGET


def load_events(path):
    df = pd.read_csv(path, parse_dates=["call_date", "period_end", "base_date"])
    df = df[df["use"] == 1].copy()
    df = df.sort_values(["producer", "call_date"]).reset_index(drop=True)

    # Bloomberg shows SK Hynix gross profit above revenue in two recent quarters.
    # That cannot be right, so we treat the margin as missing there.
    bad = df["flag"].fillna("").str.contains("gross profit")
    df.loc[bad, "gross_margin"] = np.nan

    # Changes versus the same company's previous call (past information only).
    by_ticker = df.groupby("ticker")
    df["gross_margin_chg"] = by_ticker["gross_margin"].diff()
    df["inventory_days_chg"] = by_ticker["inventory_days"].diff()

    df[TARGET] = df[TARGET].astype(int)
    return df.sort_values("call_date").reset_index(drop=True)


class NumericPrep:
    """Median-impute, standardise and clip, using the fitting rows only."""

    def fit(self, df):
        x = df[NUMERIC_FEATURES]
        self.median_ = x.median()
        filled = x.fillna(self.median_)
        self.mean_ = filled.mean()
        self.std_ = filled.std().replace(0, 1.0)
        return self

    def transform(self, df):
        x = df[NUMERIC_FEATURES].fillna(self.median_)
        z = (x - self.mean_) / self.std_
        return z.clip(-3, 3).to_numpy()
